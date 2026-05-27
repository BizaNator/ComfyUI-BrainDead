"""
BD_FaceSocketInfill — one-shot face socket creator for 2D animation flipbook textures.

Detects eye / brow / lip / nose regions via MediaPipe and fills the selected
regions with a flat colour or the surrounding skin tone, producing a base
texture with empty "sockets" that separate flipbook animation layers can be
composited into.

feature_expand is normalised to 512 px and auto-scales with image resolution:
  6 px at  512 × 512  →  6 px effective
  6 px at 1024 × 1024 → 12 px effective
  6 px at 2048 × 2048 → 24 px effective

fill_mode:
  flat      — fill_r / fill_g / fill_b (default white, useful for UV textures)
  surround  — Gaussian-blur the frame and composite through the socket mask,
              so the fill colour matches the surrounding skin tone
  inpaint   — OpenCV Telea inpaint (good quality, slower)

Wire pattern:
  LoadImage → BD_FaceSocketInfill → socket_image  (UV / texture export)
                                  → alpha_image   (RGBA, sockets transparent)
                                  → socket_mask   (for further compositing)
                                  → left_eye / right_eye / … (animation align)
"""

from __future__ import annotations

import os
import numpy as np
import torch
from comfy_api.latest import io

_MODEL_PATH = "/srv/AI_Stuff/models/mediapipe/face_landmarker.task"

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mpt
    from mediapipe.tasks.python import vision as _mpv
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as _FLC
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    _FLC = None


# ── Landmark index sets ───────────────────────────────────────────────────────

_NOSE_INDICES = sorted({
    168, 6, 197, 195, 5, 4, 1, 2,
    19, 94, 141, 370,
    51, 45, 131, 134, 102, 48, 115, 49,
    281, 275, 360, 363, 331, 278, 344, 279,
    98, 327,
})

_MP_IDX: dict[str, list[int]] = {}


def _init_mp_idx() -> None:
    global _MP_IDX
    if _MP_IDX or not HAS_MEDIAPIPE:
        return

    def _verts(connections) -> list[int]:
        s = set()
        for c in connections:
            s.add(c.start)
            s.add(c.end)
        return sorted(s)

    _MP_IDX = {
        'left_eye':   _verts(_FLC.FACE_LANDMARKS_LEFT_EYE),
        'right_eye':  _verts(_FLC.FACE_LANDMARKS_RIGHT_EYE),
        'left_brow':  _verts(_FLC.FACE_LANDMARKS_LEFT_EYEBROW),
        'right_brow': _verts(_FLC.FACE_LANDMARKS_RIGHT_EYEBROW),
        'lips':       _verts(_FLC.FACE_LANDMARKS_LIPS),
    }


# ── Mask rasterisation helpers ────────────────────────────────────────────────

def _pts(indices: list[int], lm, H: int, W: int) -> np.ndarray:
    return np.array([[int(lm[i].x * W), int(lm[i].y * H)] for i in indices], dtype=np.int32)


def _ellipse_k(r: int) -> np.ndarray:
    r = max(1, r)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _fill_convex(pts: np.ndarray, H: int, W: int, expand: int = 0) -> np.ndarray:
    out = np.zeros((H, W), dtype=np.uint8)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(out, hull, 255)
        if expand > 0:
            out = cv2.dilate(out, _ellipse_k(expand))
        elif expand < 0:
            out = cv2.erode(out, _ellipse_k(-expand))
    return out


def _union(*masks: np.ndarray) -> np.ndarray:
    out = masks[0].copy()
    for m in masks[1:]:
        np.maximum(out, m, out=out)
    return out


def _blank(H: int, W: int) -> np.ndarray:
    return np.zeros((H, W), dtype=np.uint8)


# ── Per-frame mask extraction ─────────────────────────────────────────────────

_MASK_KEYS = [
    'left_eye', 'right_eye', 'eyes',
    'left_brow', 'right_brow', 'brows',
    'lips', 'nose',
]


def _process_frame(
    frame_rgb: np.ndarray,
    landmarker,
    effective_expand: int,
) -> tuple[dict[str, np.ndarray], str]:
    """Return raw binary uint8 masks at the frame's native resolution."""
    H, W = frame_rgb.shape[:2]
    blank = _blank(H, W)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result   = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return {k: blank for k in _MASK_KEYS}, "no face detected"

    lm = result.face_landmarks[0]

    left_eye  = _fill_convex(_pts(_MP_IDX['left_eye'],  lm, H, W), H, W, expand=effective_expand)
    right_eye = _fill_convex(_pts(_MP_IDX['right_eye'], lm, H, W), H, W, expand=effective_expand)
    eyes      = _union(left_eye, right_eye)

    left_brow  = _fill_convex(_pts(_MP_IDX['left_brow'],  lm, H, W), H, W, expand=effective_expand)
    right_brow = _fill_convex(_pts(_MP_IDX['right_brow'], lm, H, W), H, W, expand=effective_expand)
    brows      = _union(left_brow, right_brow)

    lips = _fill_convex(_pts(_MP_IDX['lips'], lm, H, W), H, W, expand=effective_expand)
    nose = _fill_convex(_pts(_NOSE_INDICES,   lm, H, W), H, W)

    return {
        'left_eye': left_eye, 'right_eye': right_eye, 'eyes': eyes,
        'left_brow': left_brow, 'right_brow': right_brow, 'brows': brows,
        'lips': lips, 'nose': nose,
    }, "ok"


# ── Fill helpers ──────────────────────────────────────────────────────────────

def _feather_mask(mask_u8: np.ndarray, feather: int) -> np.ndarray:
    """Gaussian-blur a binary uint8 mask → float32 [0,1] with soft edges."""
    if feather <= 0:
        return mask_u8.astype(np.float32) / 255.0
    ksize = feather * 2 + 1
    blurred = cv2.GaussianBlur(mask_u8.astype(np.float32), (ksize, ksize), feather * 0.5)
    return (blurred / 255.0).clip(0.0, 1.0)


def _surround_fill(frame_u8: np.ndarray, socket_u8: np.ndarray, radius: int) -> np.ndarray:
    """
    Return a (H,W,3) float32 fill image by blurring the frame with a large
    Gaussian so socket colours become the weighted average of surrounding skin.
    Radius ≈ 2-3× the feature_expand value gives good results.
    """
    ksize = max(3, radius * 2 + 1)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(frame_u8.astype(np.float32), (ksize, ksize), radius * 0.5)
    return blurred / 255.0


def _inpaint_fill(
    frame_u8: np.ndarray,
    socket_u8: np.ndarray,
    effective_expand: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    OpenCV Telea inpaint.

    Returns (fill_np, inpaint_mask_u8):
      fill_np         — (H,W,3) float32, inpainted result
      inpaint_mask_u8 — expanded uint8 mask actually fed to cv2.inpaint

    The inpaint mask is dilated by an extra margin so the mask boundary
    sits on clean skin, not on hair fringe — that's what causes traces.
    Callers should use inpaint_mask_u8 (not the original socket_u8) when
    feathering/blending so the soft edge also sits on clean skin.
    """
    # Extra dilation: push the boundary off hair pixels onto clean skin.
    # ~0.5× effective_expand gives a visible safety margin without being huge.
    fringe_margin = max(4, effective_expand // 2)
    inpaint_mask  = cv2.dilate(socket_u8, _ellipse_k(fringe_margin))

    # Inpaint radius must reach across any remaining visible feature detail.
    # Scale with effective_expand so it stays proportional at any resolution.
    inpaint_radius = max(12, effective_expand)

    inpainted = cv2.inpaint(frame_u8, inpaint_mask, inpaintRadius=inpaint_radius,
                             flags=cv2.INPAINT_TELEA)
    return inpainted.astype(np.float32) / 255.0, inpaint_mask


# ── Node ─────────────────────────────────────────────────────────────────────

_FILL_MODES = ["flat", "surround", "inpaint"]


class BD_FaceSocketInfill(io.ComfyNode):
    """
    One-shot face socket creator for 2D animation flipbook textures.

    Detects face features via MediaPipe, fills selected regions with a flat
    colour or surrounding skin tone, and outputs the socket image, RGBA
    version with transparent sockets, combined socket mask, and individual
    feature masks for animation overlay alignment.

    feature_expand is normalised to 512 px — it auto-scales with image
    resolution, so the same value looks consistent across resolutions.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_FaceSocketInfill",
            display_name="BD Face Socket Infill",
            category="🧠BrainDead/Segmentation",
            description=(
                "Detects eye / brow / lip / nose regions via MediaPipe and fills "
                "selected regions to produce a base texture with empty sockets for "
                "2D animation flipbook compositing.\n\n"
                "feature_expand is 512px-normalised and auto-scales with resolution.\n\n"
                "fill_mode:\n"
                "  flat     — fill_r/g/b (white by default, good for UV textures)\n"
                "  surround — Gaussian-blur surrounding skin into the socket\n"
                "  inpaint  — OpenCV Telea inpaint (best quality, slower)\n\n"
                "feather blurs the socket mask edges for smooth compositing."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Input image or batch. Each frame is processed independently.",
                ),
                io.Float.Input(
                    "detection_confidence", default=0.5, min=0.1, max=1.0, step=0.05,
                    optional=True,
                    tooltip="MediaPipe face detection confidence threshold.",
                ),
                io.Boolean.Input(
                    "eyes", default=True, optional=True,
                    tooltip="Include left + right eye regions in the socket.",
                ),
                io.Boolean.Input(
                    "brows", default=True, optional=True,
                    tooltip="Include left + right eyebrow regions in the socket.",
                ),
                io.Boolean.Input(
                    "lips", default=True, optional=True,
                    tooltip="Include lip region in the socket.",
                ),
                io.Boolean.Input(
                    "nose", default=False, optional=True,
                    tooltip="Include nose region in the socket.",
                ),
                io.Int.Input(
                    "feature_expand", default=6, min=0, max=60, step=1, optional=True,
                    tooltip="Socket region expansion in 512px-normalised pixels. "
                            "Auto-scales with image resolution — 6 at 512px becomes "
                            "12 at 1024px, 24 at 2048px. Covers lashes and lip border.",
                ),
                io.Int.Input(
                    "feather", default=3, min=0, max=30, step=1, optional=True,
                    tooltip="Gaussian blur radius applied to socket mask edges. "
                            "0 = hard binary edge. 3-8 gives natural blending.",
                ),
                io.Combo.Input(
                    "fill_mode",
                    options=_FILL_MODES,
                    default="flat", optional=True,
                    tooltip="flat: fill_r/g/b colour (UV textures). "
                            "surround: Gaussian-blurred surrounding skin. "
                            "inpaint: OpenCV Telea inpaint (slow, best quality).",
                ),
                io.Int.Input(
                    "fill_r", default=255, min=0, max=255, step=1, optional=True,
                    tooltip="Fill colour red channel (0-255). Used when fill_mode=flat.",
                ),
                io.Int.Input(
                    "fill_g", default=255, min=0, max=255, step=1, optional=True,
                    tooltip="Fill colour green channel.",
                ),
                io.Int.Input(
                    "fill_b", default=255, min=0, max=255, step=1, optional=True,
                    tooltip="Fill colour blue channel.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="socket_image",
                    tooltip="RGB image with socket regions replaced by the fill (flat colour, "
                            "surrounding skin, or inpainted). Good for UV texture export.",
                ),
                io.Image.Output(
                    display_name="alpha_image",
                    tooltip="RGBA image — socket regions fully transparent (alpha=0). "
                            "Non-socket pixels keep full opacity. Drop directly into a "
                            "game engine or compositing tool.",
                ),
                io.Mask.Output(
                    display_name="socket_mask",
                    tooltip="Feathered union of all enabled feature masks. "
                            "White = socket region. Soft edges when feather > 0.",
                ),
                io.Mask.Output(display_name="left_eye",   tooltip="Left eye convex hull (binary)."),
                io.Mask.Output(display_name="right_eye",  tooltip="Right eye convex hull (binary)."),
                io.Mask.Output(display_name="eyes",       tooltip="Both eyes combined (binary)."),
                io.Mask.Output(display_name="left_brow",  tooltip="Left eyebrow mask (binary)."),
                io.Mask.Output(display_name="right_brow", tooltip="Right eyebrow mask (binary)."),
                io.Mask.Output(display_name="brows",      tooltip="Both brows combined (binary)."),
                io.Mask.Output(display_name="lips",       tooltip="Full lip area mask (binary)."),
                io.Mask.Output(display_name="nose",       tooltip="Nose region mask (binary)."),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        detection_confidence: float = 0.5,
        eyes: bool = True,
        brows: bool = True,
        lips: bool = True,
        nose: bool = False,
        feature_expand: int = 6,
        feather: int = 3,
        fill_mode: str = "flat",
        fill_r: int = 255,
        fill_g: int = 255,
        fill_b: int = 255,
    ) -> io.NodeOutput:

        _blank1     = torch.zeros((1, 1, 1),    dtype=torch.float32)
        _blank_img  = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        _blank_rgba = torch.zeros((1, 1, 1, 4), dtype=torch.float32)
        _n_mask     = len(_MASK_KEYS)  # 8

        def _fail(msg: str) -> io.NodeOutput:
            return io.NodeOutput(
                _blank_img, _blank_rgba, _blank1,
                *([_blank1] * _n_mask),
                msg,
            )

        if not HAS_MEDIAPIPE or not HAS_CV2:
            missing = [p for p, h in [("mediapipe", HAS_MEDIAPIPE), ("opencv-python", HAS_CV2)] if not h]
            return _fail(f"BD_FaceSocketInfill: missing packages — {', '.join(missing)}")

        if not os.path.exists(_MODEL_PATH):
            return _fail(f"BD_FaceSocketInfill: model not found at {_MODEL_PATH}")

        _init_mp_idx()

        if image.ndim == 3:
            image = image.unsqueeze(0)
        B, H, W, C = image.shape

        # Scale expand relative to 512px baseline
        effective_expand = max(1, round(feature_expand * max(H, W) / 512))

        flat_rgb = np.array([fill_r / 255.0, fill_g / 255.0, fill_b / 255.0], dtype=np.float32)

        batches: dict[str, list[torch.Tensor]] = {k: [] for k in _MASK_KEYS + ['socket_soft']}
        socket_images: list[torch.Tensor] = []
        alpha_images:  list[torch.Tensor] = []
        statuses:      list[str] = []

        base_opts = _mpt.BaseOptions(model_asset_path=_MODEL_PATH)
        opts = _mpv.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=_mpv.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=detection_confidence,
            min_face_presence_confidence=detection_confidence,
            min_tracking_confidence=detection_confidence,
        )

        with _mpv.FaceLandmarker.create_from_options(opts) as landmarker:
            for b in range(B):
                frame = image[b].cpu().numpy()            # (H, W, C) float32
                frame_u8 = (frame * 255.0).clip(0, 255).astype(np.uint8)
                if C == 4:
                    frame_u8 = frame_u8[..., :3]
                frame_rgb_u8 = frame_u8.copy()

                masks, status = _process_frame(frame_rgb_u8, landmarker, effective_expand)
                statuses.append(status)

                # ── Socket mask (binary uint8) from enabled features ─────────
                active_np = []
                if eyes:
                    active_np.append(masks['eyes'])
                if brows:
                    active_np.append(masks['brows'])
                if lips:
                    active_np.append(masks['lips'])
                if nose:
                    active_np.append(masks['nose'])
                socket_u8 = _union(*active_np) if active_np else _blank(H, W)

                # ── Individual binary mask tensors ───────────────────────────
                for k in _MASK_KEYS:
                    batches[k].append(
                        torch.from_numpy(masks[k].astype(np.float32) / 255.0)
                    )

                # ── Fill image + determine which mask to feather/blend with ──
                blend_mask_u8 = socket_u8   # default: feather the original socket

                if fill_mode == "surround":
                    radius  = max(15, effective_expand * 4)
                    fill_np = _surround_fill(frame_rgb_u8, socket_u8, radius)
                elif fill_mode == "inpaint":
                    fill_np, blend_mask_u8 = _inpaint_fill(
                        frame_rgb_u8, socket_u8, effective_expand
                    )
                    # blend_mask_u8 is the dilated inpaint mask — its boundary
                    # is on clean skin, so the feathered edge won't blend back
                    # any original hair pixels
                else:  # flat
                    fill_np = np.full((H, W, 3), flat_rgb, dtype=np.float32)

                # ── Feathered mask (from the blend_mask for this mode) ───────
                socket_soft = _feather_mask(blend_mask_u8, feather)   # (H,W) [0,1]
                batches['socket_soft'].append(torch.from_numpy(socket_soft))

                # ── Blend: filled = frame * (1-alpha) + fill * alpha ─────────
                frame_f  = frame[..., :3].astype(np.float32)           # (H,W,3)
                alpha_2d = socket_soft[:, :, np.newaxis]                # (H,W,1)
                blended   = frame_f * (1.0 - alpha_2d) + fill_np * alpha_2d
                blended   = blended.clip(0.0, 1.0)

                socket_images.append(torch.from_numpy(blended))        # (H,W,3)

                # ── RGBA: blended RGB + (1-socket_soft) as alpha ─────────────
                alpha_ch = (1.0 - socket_soft)[:, :, np.newaxis]
                rgba     = np.concatenate([blended, alpha_ch], axis=-1)
                alpha_images.append(torch.from_numpy(rgba.astype(np.float32)))  # (H,W,4)

        socket_image = torch.stack(socket_images, dim=0)   # (B, H, W, 3)
        alpha_image  = torch.stack(alpha_images,  dim=0)   # (B, H, W, 4)
        out = {k: torch.stack(batches[k], dim=0) for k in _MASK_KEYS + ['socket_soft']}

        detected  = sum(1 for s in statuses if s == "ok")
        status_str = f"BD_FaceSocketInfill: {detected}/{B} detected | expand={effective_expand}px feather={feather} fill={fill_mode}"
        if detected < B:
            failed = [i for i, s in enumerate(statuses) if s != "ok"]
            status_str += f" | missed frames: {failed}"
        print(f"[BD_FaceSocketInfill] {status_str}", flush=True)

        return io.NodeOutput(
            socket_image, alpha_image, out['socket_soft'],
            out['left_eye'], out['right_eye'], out['eyes'],
            out['left_brow'], out['right_brow'], out['brows'],
            out['lips'], out['nose'],
            status_str,
        )


FACE_SOCKET_INFILL_V3_NODES    = [BD_FaceSocketInfill]
FACE_SOCKET_INFILL_NODES       = {"BD_FaceSocketInfill": BD_FaceSocketInfill}
FACE_SOCKET_INFILL_DISPLAY_NAMES = {"BD_FaceSocketInfill": "BD Face Socket Infill"}
