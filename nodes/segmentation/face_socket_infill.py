"""
BD_FaceSocketInfill — one-shot face socket creator for 2D animation flipbook textures.

Detects eye / brow / lip / nose regions via MediaPipe and fills the selected
regions with a flat colour or the surrounding skin tone, producing a base
texture with empty "sockets" that separate flipbook animation layers can be
composited into.

feature_expand / feather are the master (default) values used for all zones.
Per-zone overrides (eyes_expand, brows_expand, lips_expand, nose_expand and
their *_feather counterparts) take priority when set to any value ≥ 0.
Set them to -1 (or leave unwired) to fall back to the master values.

feature_expand is normalised to 512 px and auto-scales with image resolution:
  6 px at  512 × 512  →   6 px effective
  6 px at 1024 × 1024 →  12 px effective
  6 px at 1536 × 1536 →  18 px effective

fill_mode:
  flat     — fill_r / fill_g / fill_b (default white, good for UV textures)
  surround — Gaussian-blur frame and composite so fill matches surrounding skin
  inpaint  — OpenCV Telea inpaint per zone (best quality, slower)
             Each zone is inpainted independently so brows can't bleed into eyes.
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


def _feather_mask(mask_u8: np.ndarray, feather: int) -> np.ndarray:
    """Binary uint8 mask → float32 [0,1] with Gaussian-softened edges."""
    if feather <= 0:
        return mask_u8.astype(np.float32) / 255.0
    ksize = feather * 2 + 1
    blurred = cv2.GaussianBlur(mask_u8.astype(np.float32), (ksize, ksize), feather * 0.5)
    return (blurred / 255.0).clip(0.0, 1.0)


# ── Per-frame mask extraction ─────────────────────────────────────────────────

_MASK_KEYS = [
    'left_eye', 'right_eye', 'eyes',
    'left_brow', 'right_brow', 'brows',
    'lips', 'nose',
]

# zone keys referenced in zone_expand / zone_feather dicts
_ZONES = ('eyes', 'brows', 'lips', 'nose')


def _process_frame(
    frame_rgb: np.ndarray,
    landmarker,
    zone_expand: dict[str, int],
) -> tuple[dict[str, np.ndarray], str]:
    """
    Return per-feature binary uint8 masks at the frame's native resolution.
    Each zone uses its own expansion radius from zone_expand.
    """
    H, W = frame_rgb.shape[:2]
    blank = _blank(H, W)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result   = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return {k: blank for k in _MASK_KEYS}, "no face detected"

    lm = result.face_landmarks[0]
    ex = zone_expand   # shorthand

    left_eye  = _fill_convex(_pts(_MP_IDX['left_eye'],  lm, H, W), H, W, expand=ex['eyes'])
    right_eye = _fill_convex(_pts(_MP_IDX['right_eye'], lm, H, W), H, W, expand=ex['eyes'])
    eyes      = _union(left_eye, right_eye)

    left_brow  = _fill_convex(_pts(_MP_IDX['left_brow'],  lm, H, W), H, W, expand=ex['brows'])
    right_brow = _fill_convex(_pts(_MP_IDX['right_brow'], lm, H, W), H, W, expand=ex['brows'])
    brows      = _union(left_brow, right_brow)

    lips = _fill_convex(_pts(_MP_IDX['lips'], lm, H, W), H, W, expand=ex['lips'])
    nose = _fill_convex(_pts(_NOSE_INDICES,   lm, H, W), H, W, expand=ex['nose'])

    return {
        'left_eye': left_eye, 'right_eye': right_eye, 'eyes': eyes,
        'left_brow': left_brow, 'right_brow': right_brow, 'brows': brows,
        'lips': lips, 'nose': nose,
    }, "ok"


# ── Fill helpers ──────────────────────────────────────────────────────────────

def _surround_fill(frame_u8: np.ndarray, radius: int) -> np.ndarray:
    """Large Gaussian blur so fill colour matches surrounding skin."""
    ksize = radius * 2 + 1
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(frame_u8.astype(np.float32), (ksize, ksize), radius * 0.5) / 255.0


def _inpaint_zone(
    frame_u8: np.ndarray,
    zone_mask_u8: np.ndarray,
    effective_expand: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inpaint a single zone.  Returns (inpainted_frame_u8, expanded_mask_u8).

    The mask is dilated before inpainting so its boundary sits on clean skin,
    not on hair fringe — that's what causes trace artifacts.
    """
    fringe = max(4, effective_expand // 2)
    inpaint_mask = cv2.dilate(zone_mask_u8, _ellipse_k(fringe))
    radius       = max(12, effective_expand)
    inpainted    = cv2.inpaint(frame_u8, inpaint_mask, radius, cv2.INPAINT_TELEA)
    return inpainted, inpaint_mask


# ── Node ─────────────────────────────────────────────────────────────────────

_FILL_MODES = ["flat", "surround", "inpaint"]


class BD_FaceSocketInfill(io.ComfyNode):
    """
    One-shot face socket creator for 2D animation flipbook textures.

    Detects face features via MediaPipe, fills selected regions with a flat
    colour, surrounding skin tone, or per-zone OpenCV inpaint.  Outputs:
    - socket_image (RGB)  — sockets filled
    - alpha_image  (RGBA) — sockets transparent
    - socket_mask         — feathered union of all active zones
    - per-feature binary masks for animation layer alignment

    Per-zone expand / feather overrides (-1 = use master value).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_FaceSocketInfill",
            display_name="BD Face Socket Infill",
            category="🧠BrainDead/Segmentation",
            description=(
                "Detects eye/brow/lip/nose regions via MediaPipe and fills them "
                "to produce a base texture with empty sockets for 2D animation.\n\n"
                "feature_expand is 512px-normalised and auto-scales with resolution.\n"
                "Per-zone expand/feather overrides: set ≥ 0 to override, -1 = use master.\n\n"
                "fill_mode:\n"
                "  flat     — fill_r/g/b (white by default)\n"
                "  surround — Gaussian-blurred surrounding skin\n"
                "  inpaint  — OpenCV Telea per zone (cleanest, slower)\n\n"
                "inpaint dilates each zone's mask before inpainting so the boundary\n"
                "sits on clean skin, not hair fringe — eliminates trace artifacts."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Input image or batch. Each frame processed independently.",
                ),
                io.Float.Input(
                    "detection_confidence", default=0.5, min=0.1, max=1.0, step=0.05,
                    optional=True,
                    tooltip="MediaPipe face detection confidence threshold.",
                ),

                # ── Zone toggles ─────────────────────────────────────────────
                io.Boolean.Input("eyes",  default=True,  optional=True,
                                 tooltip="Include left + right eye regions."),
                io.Boolean.Input("brows", default=True,  optional=True,
                                 tooltip="Include left + right eyebrow regions."),
                io.Boolean.Input("lips",  default=True,  optional=True,
                                 tooltip="Include lip region."),
                io.Boolean.Input("nose",  default=False, optional=True,
                                 tooltip="Include nose region."),

                # ── Master expand / feather ───────────────────────────────────
                io.Int.Input(
                    "feature_expand", default=6, min=0, max=60, step=1, optional=True,
                    tooltip="Master socket expansion in 512px-normalised pixels. "
                            "Auto-scales: 6 at 512px → 12 at 1024px → 18 at 1536px. "
                            "Per-zone overrides take priority when set ≥ 0.",
                ),
                io.Int.Input(
                    "feather", default=3, min=0, max=30, step=1, optional=True,
                    tooltip="Master Gaussian blur radius for socket mask edges. "
                            "0 = hard edge. Per-zone overrides take priority when ≥ 0.",
                ),

                # ── Per-zone expand overrides (-1 = use feature_expand) ───────
                io.Int.Input("eyes_expand",  default=-1, min=-1, max=60, step=1, optional=True,
                             tooltip="Eye expand override (−1 = use feature_expand)."),
                io.Int.Input("brows_expand", default=-1, min=-1, max=60, step=1, optional=True,
                             tooltip="Brow expand override. Raise this when brow hairs leave a trace."),
                io.Int.Input("lips_expand",  default=-1, min=-1, max=60, step=1, optional=True,
                             tooltip="Lip expand override (−1 = use feature_expand)."),
                io.Int.Input("nose_expand",  default=-1, min=-1, max=60, step=1, optional=True,
                             tooltip="Nose expand override (−1 = use feature_expand)."),

                # ── Per-zone feather overrides (-1 = use feather) ────────────
                io.Int.Input("eyes_feather",  default=-1, min=-1, max=30, step=1, optional=True,
                             tooltip="Eye feather override (−1 = use feather)."),
                io.Int.Input("brows_feather", default=-1, min=-1, max=30, step=1, optional=True,
                             tooltip="Brow feather override. Higher = softer edge on brow socket."),
                io.Int.Input("lips_feather",  default=-1, min=-1, max=30, step=1, optional=True,
                             tooltip="Lip feather override (−1 = use feather)."),
                io.Int.Input("nose_feather",  default=-1, min=-1, max=30, step=1, optional=True,
                             tooltip="Nose feather override (−1 = use feather)."),

                # ── Fill ─────────────────────────────────────────────────────
                io.Combo.Input(
                    "fill_mode", options=_FILL_MODES, default="flat", optional=True,
                    tooltip="flat: fill_r/g/b. surround: skin-tone blur. "
                            "inpaint: per-zone Telea (cleanest).",
                ),
                io.Int.Input("fill_r", default=255, min=0, max=255, step=1, optional=True,
                             tooltip="Fill red (0-255). Used when fill_mode=flat."),
                io.Int.Input("fill_g", default=255, min=0, max=255, step=1, optional=True,
                             tooltip="Fill green (0-255)."),
                io.Int.Input("fill_b", default=255, min=0, max=255, step=1, optional=True,
                             tooltip="Fill blue (0-255)."),
            ],
            outputs=[
                io.Image.Output(
                    display_name="socket_image",
                    tooltip="RGB image with socket regions replaced by the fill.",
                ),
                io.Image.Output(
                    display_name="alpha_image",
                    tooltip="RGBA — socket regions fully transparent (alpha=0).",
                ),
                io.Mask.Output(
                    display_name="socket_mask",
                    tooltip="Feathered union of all active zone masks.",
                ),
                io.Mask.Output(display_name="left_eye"),
                io.Mask.Output(display_name="right_eye"),
                io.Mask.Output(display_name="eyes"),
                io.Mask.Output(display_name="left_brow"),
                io.Mask.Output(display_name="right_brow"),
                io.Mask.Output(display_name="brows"),
                io.Mask.Output(display_name="lips"),
                io.Mask.Output(display_name="nose"),
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
        eyes_expand: int = -1,
        brows_expand: int = -1,
        lips_expand: int = -1,
        nose_expand: int = -1,
        eyes_feather: int = -1,
        brows_feather: int = -1,
        lips_feather: int = -1,
        nose_feather: int = -1,
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
            return io.NodeOutput(_blank_img, _blank_rgba, _blank1,
                                 *([_blank1] * _n_mask), msg)

        if not HAS_MEDIAPIPE or not HAS_CV2:
            missing = [p for p, h in [("mediapipe", HAS_MEDIAPIPE), ("opencv-python", HAS_CV2)] if not h]
            return _fail(f"BD_FaceSocketInfill: missing — {', '.join(missing)}")

        if not os.path.exists(_MODEL_PATH):
            return _fail(f"BD_FaceSocketInfill: model not found at {_MODEL_PATH}")

        _init_mp_idx()

        if image.ndim == 3:
            image = image.unsqueeze(0)
        B, H, W, C = image.shape

        # ── Scale master expand to image resolution ───────────────────────────
        scale = max(H, W) / 512.0

        def _eff_expand(override: int) -> int:
            base = override if override >= 0 else feature_expand
            return max(1, round(base * scale))

        def _eff_feather(override: int) -> int:
            return override if override >= 0 else feather

        eff_expand = {
            'eyes':  _eff_expand(eyes_expand),
            'brows': _eff_expand(brows_expand),
            'lips':  _eff_expand(lips_expand),
            'nose':  _eff_expand(nose_expand),
        }
        eff_feather = {
            'eyes':  _eff_feather(eyes_feather),
            'brows': _eff_feather(brows_feather),
            'lips':  _eff_feather(lips_feather),
            'nose':  _eff_feather(nose_feather),
        }

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
                frame    = image[b].cpu().numpy()
                frame_u8 = (frame * 255.0).clip(0, 255).astype(np.uint8)
                if C == 4:
                    frame_u8 = frame_u8[..., :3]
                frame_rgb_u8 = frame_u8.copy()

                masks, status = _process_frame(frame_rgb_u8, landmarker, eff_expand)
                statuses.append(status)

                # ── Individual binary masks ───────────────────────────────────
                for k in _MASK_KEYS:
                    batches[k].append(
                        torch.from_numpy(masks[k].astype(np.float32) / 255.0)
                    )

                # ── Active zones: zone mask + its blend mask + feathered soft ─
                # For inpaint: we sequentially inpaint each zone independently
                # so brows can't contaminate eyes and vice versa.
                # blend_mask per zone may be the inpaint-expanded mask (wider
                # than the feature mask) so the feathered edge sits on clean skin.

                zone_info = [
                    # (enabled, zone_key, feature_mask)
                    (eyes,  'eyes',  masks['eyes']),
                    (brows, 'brows', masks['brows']),
                    (lips,  'lips',  masks['lips']),
                    (nose,  'nose',  masks['nose']),
                ]

                combined_soft = np.zeros((H, W), dtype=np.float32)

                if fill_mode == "inpaint":
                    # Inpaint each active zone into a running result buffer
                    result_u8 = frame_rgb_u8.copy()
                    for enabled, zone, zone_mask in zone_info:
                        if not enabled or not zone_mask.any():
                            continue
                        result_u8, exp_mask = _inpaint_zone(result_u8, zone_mask,
                                                             eff_expand[zone])
                        zone_soft = _feather_mask(exp_mask, eff_feather[zone])
                        np.maximum(combined_soft, zone_soft, out=combined_soft)
                    fill_np = result_u8.astype(np.float32) / 255.0

                elif fill_mode == "surround":
                    # One big blur, per-zone feather
                    max_expand = max(eff_expand.values())
                    fill_np = _surround_fill(frame_rgb_u8, max(15, max_expand * 4))
                    for enabled, zone, zone_mask in zone_info:
                        if not enabled or not zone_mask.any():
                            continue
                        zone_soft = _feather_mask(zone_mask, eff_feather[zone])
                        np.maximum(combined_soft, zone_soft, out=combined_soft)

                else:  # flat
                    fill_np = np.full((H, W, 3), flat_rgb, dtype=np.float32)
                    for enabled, zone, zone_mask in zone_info:
                        if not enabled or not zone_mask.any():
                            continue
                        zone_soft = _feather_mask(zone_mask, eff_feather[zone])
                        np.maximum(combined_soft, zone_soft, out=combined_soft)

                batches['socket_soft'].append(torch.from_numpy(combined_soft))

                # ── Blend ─────────────────────────────────────────────────────
                frame_f  = frame[..., :3].astype(np.float32)
                alpha_2d = combined_soft[:, :, np.newaxis]
                blended  = (frame_f * (1.0 - alpha_2d) + fill_np * alpha_2d).clip(0.0, 1.0)
                socket_images.append(torch.from_numpy(blended))

                alpha_ch = (1.0 - combined_soft)[:, :, np.newaxis]
                alpha_images.append(torch.from_numpy(
                    np.concatenate([blended, alpha_ch], axis=-1).astype(np.float32)
                ))

        socket_image = torch.stack(socket_images, dim=0)
        alpha_image  = torch.stack(alpha_images,  dim=0)
        out = {k: torch.stack(batches[k], dim=0) for k in _MASK_KEYS + ['socket_soft']}

        detected = sum(1 for s in statuses if s == "ok")
        ex_str   = " ".join(f"{z}={eff_expand[z]}" for z in _ZONES)
        ft_str   = " ".join(f"{z}={eff_feather[z]}" for z in _ZONES)
        status_str = (
            f"BD_FaceSocketInfill: {detected}/{B} detected | {fill_mode} | "
            f"expand({ex_str}) feather({ft_str})"
        )
        if detected < B:
            failed = [i for i, s in enumerate(statuses) if s != "ok"]
            status_str += f" | missed: {failed}"
        print(f"[BD_FaceSocketInfill] {status_str}", flush=True)

        return io.NodeOutput(
            socket_image, alpha_image, out['socket_soft'],
            out['left_eye'], out['right_eye'], out['eyes'],
            out['left_brow'], out['right_brow'], out['brows'],
            out['lips'], out['nose'],
            status_str,
        )


FACE_SOCKET_INFILL_V3_NODES      = [BD_FaceSocketInfill]
FACE_SOCKET_INFILL_NODES         = {"BD_FaceSocketInfill": BD_FaceSocketInfill}
FACE_SOCKET_INFILL_DISPLAY_NAMES = {"BD_FaceSocketInfill": "BD Face Socket Infill"}
