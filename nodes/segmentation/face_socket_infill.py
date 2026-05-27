"""
BD_FaceSocketInfill — one-shot face socket creator for 2D animation flipbook textures.

Detects eye / brow / lip / nose regions via MediaPipe and fills the selected
regions with a flat colour, producing a base texture with empty "sockets" that
separate flipbook animation layers can be composited into.

Wire pattern:
  LoadImage → BD_FaceSocketInfill → socket_image  (UV/texture export)
                                  → alpha_image   (RGBA, sockets transparent)
                                  → socket_mask   (for inpaint / composite)
                                  → left_eye / right_eye / … (per-feature)

Outputs
-------
socket_image  IMAGE (RGB)   original image, socket regions filled with fill_r/g/b
alpha_image   IMAGE (RGBA)  same as socket_image but socket regions have alpha = 0
socket_mask   MASK          union of all enabled feature masks
left_eye      MASK
right_eye     MASK
eyes          MASK          union of left + right eye
left_brow     MASK
right_brow    MASK
brows         MASK          union of left + right brow
lips          MASK
nose          MASK
status        STRING
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


def _to_tensor(m: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(m.astype(np.float32) / 255.0)


# ── Per-frame processing ──────────────────────────────────────────────────────

_MASK_KEYS = ['left_eye', 'right_eye', 'eyes', 'left_brow', 'right_brow', 'brows', 'lips', 'nose']


def _process_frame(
    frame_rgb: np.ndarray,
    landmarker,
    feature_expand: int,
) -> tuple[dict[str, np.ndarray], str]:
    H, W = frame_rgb.shape[:2]
    blank = _blank(H, W)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return {k: blank for k in _MASK_KEYS}, "no face detected"

    lm = result.face_landmarks[0]

    left_eye  = _fill_convex(_pts(_MP_IDX['left_eye'],  lm, H, W), H, W, expand=feature_expand)
    right_eye = _fill_convex(_pts(_MP_IDX['right_eye'], lm, H, W), H, W, expand=feature_expand)
    eyes      = _union(left_eye, right_eye)

    left_brow  = _fill_convex(_pts(_MP_IDX['left_brow'],  lm, H, W), H, W, expand=feature_expand)
    right_brow = _fill_convex(_pts(_MP_IDX['right_brow'], lm, H, W), H, W, expand=feature_expand)
    brows      = _union(left_brow, right_brow)

    lips = _fill_convex(_pts(_MP_IDX['lips'], lm, H, W), H, W, expand=feature_expand)
    nose = _fill_convex(_pts(_NOSE_INDICES,   lm, H, W), H, W)

    return {
        'left_eye': left_eye, 'right_eye': right_eye, 'eyes': eyes,
        'left_brow': left_brow, 'right_brow': right_brow, 'brows': brows,
        'lips': lips, 'nose': nose,
    }, "ok"


# ── Node ─────────────────────────────────────────────────────────────────────

class BD_FaceSocketInfill(io.ComfyNode):
    """
    One-shot face socket creator for 2D animation flipbook textures.

    Detects face features via MediaPipe and fills selected regions with a flat
    colour.  Outputs the socket image (RGB), an RGBA version with transparent
    sockets, the combined socket mask, and individual feature masks for
    animation overlay alignment.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_FaceSocketInfill",
            display_name="BD Face Socket Infill",
            category="🧠BrainDead/Segmentation",
            description=(
                "Detects eye / brow / lip / nose regions via MediaPipe and fills the "
                "selected regions with a flat colour, producing a base texture with "
                "empty sockets for 2D animation flipbook compositing.\n\n"
                "socket_image — RGB, sockets filled with fill_r/g/b\n"
                "alpha_image  — RGBA, sockets alpha=0 (transparent)\n"
                "socket_mask  — union of all enabled regions\n"
                "Per-feature masks for animation layer alignment."
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
                io.Boolean.Input(
                    "eyes", default=True, optional=True,
                    tooltip="Include eye regions (left + right) in socket mask and infill.",
                ),
                io.Boolean.Input(
                    "brows", default=True, optional=True,
                    tooltip="Include eyebrow regions in socket mask and infill.",
                ),
                io.Boolean.Input(
                    "lips", default=True, optional=True,
                    tooltip="Include lip region in socket mask and infill.",
                ),
                io.Boolean.Input(
                    "nose", default=False, optional=True,
                    tooltip="Include nose region in socket mask and infill.",
                ),
                io.Int.Input(
                    "feature_expand", default=4, min=0, max=30, step=1, optional=True,
                    tooltip="Pixels to expand each feature mask outward before filling. "
                            "3–6 px covers lashes and lip border cleanly.",
                ),
                io.Int.Input(
                    "fill_r", default=255, min=0, max=255, step=1, optional=True,
                    tooltip="Fill colour red channel (0–255). Default 255 = white.",
                ),
                io.Int.Input(
                    "fill_g", default=255, min=0, max=255, step=1, optional=True,
                    tooltip="Fill colour green channel (0–255).",
                ),
                io.Int.Input(
                    "fill_b", default=255, min=0, max=255, step=1, optional=True,
                    tooltip="Fill colour blue channel (0–255).",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="socket_image",
                    tooltip="Original image with socket regions replaced by the fill colour (RGB).",
                ),
                io.Image.Output(
                    display_name="alpha_image",
                    tooltip="RGBA image — socket regions are fully transparent (alpha=0). "
                            "Non-socket pixels keep full opacity.",
                ),
                io.Mask.Output(
                    display_name="socket_mask",
                    tooltip="Union of all enabled feature masks. White = socket region.",
                ),
                io.Mask.Output(display_name="left_eye",   tooltip="Left eye convex hull mask."),
                io.Mask.Output(display_name="right_eye",  tooltip="Right eye convex hull mask."),
                io.Mask.Output(display_name="eyes",       tooltip="Both eyes combined."),
                io.Mask.Output(display_name="left_brow",  tooltip="Left eyebrow mask."),
                io.Mask.Output(display_name="right_brow", tooltip="Right eyebrow mask."),
                io.Mask.Output(display_name="brows",      tooltip="Both brows combined."),
                io.Mask.Output(display_name="lips",       tooltip="Full lip area mask."),
                io.Mask.Output(display_name="nose",       tooltip="Nose region mask."),
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
        feature_expand: int = 4,
        fill_r: int = 255,
        fill_g: int = 255,
        fill_b: int = 255,
    ) -> io.NodeOutput:

        _blank1 = torch.zeros((1, 1, 1), dtype=torch.float32)
        _blank_img  = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        _blank_rgba = torch.zeros((1, 1, 1, 4), dtype=torch.float32)
        # socket_image, alpha_image, socket_mask, 8 feature masks, status
        _n_mask = len(_MASK_KEYS)

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

        fill_rgb = torch.tensor([fill_r / 255.0, fill_g / 255.0, fill_b / 255.0], dtype=torch.float32)

        batches: dict[str, list[torch.Tensor]] = {k: [] for k in _MASK_KEYS + ['socket']}
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
                frame = image[b].cpu().numpy()
                frame_u8 = (frame * 255.0).clip(0, 255).astype(np.uint8)
                if C == 4:
                    frame_u8 = frame_u8[..., :3]

                masks, status = _process_frame(frame_u8, landmarker, feature_expand)
                statuses.append(status)

                # Socket mask = union of enabled features
                active_np = []
                if eyes:
                    active_np.append(masks['eyes'])
                if brows:
                    active_np.append(masks['brows'])
                if lips:
                    active_np.append(masks['lips'])
                if nose:
                    active_np.append(masks['nose'])
                socket_np = _union(*active_np) if active_np else _blank(H, W)
                masks['socket'] = socket_np

                for k in _MASK_KEYS + ['socket']:
                    batches[k].append(_to_tensor(masks[k]))

                # Colour infill
                frame_rgb = image[b, :, :, :3]             # (H, W, 3)
                socket_t  = _to_tensor(socket_np)           # (H, W)
                mask3     = socket_t.unsqueeze(-1)           # (H, W, 1)
                filled    = frame_rgb * (1.0 - mask3) + fill_rgb * mask3   # (H, W, 3)
                socket_images.append(filled)

                # RGBA — socket alpha = 0
                alpha_ch = (1.0 - socket_t).unsqueeze(-1)   # (H, W, 1)
                alpha_images.append(torch.cat([filled, alpha_ch], dim=-1))  # (H, W, 4)

        socket_image = torch.stack(socket_images, dim=0)   # (B, H, W, 3)
        alpha_image  = torch.stack(alpha_images,  dim=0)   # (B, H, W, 4)
        out = {k: torch.stack(batches[k], dim=0) for k in _MASK_KEYS + ['socket']}

        detected = sum(1 for s in statuses if s == "ok")
        status_str = f"BD_FaceSocketInfill: {detected}/{B} detected"
        if detected < B:
            failed = [i for i, s in enumerate(statuses) if s != "ok"]
            status_str += f" — missed frames: {failed}"
        print(f"[BD_FaceSocketInfill] {status_str}", flush=True)

        return io.NodeOutput(
            socket_image, alpha_image, out['socket'],
            out['left_eye'], out['right_eye'], out['eyes'],
            out['left_brow'], out['right_brow'], out['brows'],
            out['lips'], out['nose'],
            status_str,
        )


FACE_SOCKET_INFILL_V3_NODES = [BD_FaceSocketInfill]
FACE_SOCKET_INFILL_NODES = {"BD_FaceSocketInfill": BD_FaceSocketInfill}
FACE_SOCKET_INFILL_DISPLAY_NAMES = {"BD_FaceSocketInfill": "BD Face Socket Infill"}
