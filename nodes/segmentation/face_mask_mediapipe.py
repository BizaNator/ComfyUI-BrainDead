"""
BD_MediaPipeFaceMask — landmark-precise face region masks via MediaPipe Face Mesh.

Uses the MediaPipe Tasks API (mediapipe >= 0.10). Requires the face_landmarker.task
model file, searched in ComfyUI model directories under mediapipe/face_landmarker.task.

Deterministic, CPU-only, ~5 ms per frame. No SAM3 prompts, no sampling noise.
All masks returned as ComfyUI MASK tensors (B, H, W) in [0, 1].

Regions
-------
face_oval   Complete face silhouette following the jawline/hairline oval.
            Does NOT include neck, shoulders, or background.

skin        face_oval minus (eyes + brows + lips [+ nose if subtract_nose]).
            Primary "paintable skin" mask for the GLSL skin shader.

left_eye /  Individual eye contour masks. MediaPipe convention: "left" = subject's
right_eye   left eye, which appears on the RIGHT side of a front-facing image.

left_iris / Iris-only masks — the coloured disc of each eye.  Derived from MediaPipe's
right_iris  4-point iris landmark ring (indices 469-477).  Requires the 478-point model.
            Falls back to blank if iris landmarks are absent.

eyes        Union of left_eye + right_eye.

left_brow / Individual eyebrow masks.
right_brow

brows       Union of both brows.

lips        Full lip area (upper + lower, inner + outer contour).

nose        Nose tip, bridge, and nostril rims (custom landmark set — no named
            nose region in MediaPipe).

left_ear /  Preauricular / temporal region — lateral face oval slice near
right_ear   landmarks 234 (left) and 454 (right), expanded outward. Covers
            what is visible in front/3-quarter views. Not the full ear pinna.

ears        Union of both ears.

forehead    Upper ~40% of face_oval minus brow band.

hair        Region above the face oval extending to the image top edge.
"""

from __future__ import annotations

import os
import numpy as np
import torch
from comfy_api.latest import io

import folder_paths as _folder_paths


def _find_mediapipe_model() -> str:
    """Find face_landmarker.task in ComfyUI model directories, or return default path."""
    filename = "face_landmarker.task"
    for base_dir in _folder_paths.get_folder_paths("models"):
        candidate = os.path.join(base_dir, "mediapipe", filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(_folder_paths.models_dir, "mediapipe", filename)


_MODEL_PATH = _find_mediapipe_model()

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


# ── All mask generation imported from shared module ───────────────────────────
from .face_mp_shared import (
    _conn_verts, _OVAL_PATH, _LEFT_EAR_OVAL, _RIGHT_EAR_OVAL, _NOSE_INDICES,
    _MP_IDX, _init_mp_idx,
    _pts, _ellipse_k, _blank, _fill_convex, _fill_poly_ordered,
    _union, _subtract, _hair_mask, _forehead_mask, _ear_mask, _bbox_from_mask,
    _KEYS, _BBOX_FEATURES,
    _masks_from_landmarks,
)


def _to_tensor(m: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(m.astype(np.float32) / 255.0)


# ── Per-frame processor (wraps shared _masks_from_landmarks + MP detection) ───

def _process_frame(
    frame_rgb: np.ndarray,
    landmarker,
    face_expand: int,
    feature_expand: int,
    iris_expand: int,
    subtract_nose: bool,
    ear_expand: int,
    hair_expand: int,
    head_mask_np: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], str]:
    import mediapipe as mp
    H, W = frame_rgb.shape[:2]
    empty = {k: _blank(H, W) for k in _KEYS}

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return empty, "no face detected"

    lm = result.face_landmarks[0]
    masks = _masks_from_landmarks(
        lm, H, W,
        face_expand=face_expand,
        feature_expand=feature_expand,
        iris_expand=iris_expand,
        subtract_nose=subtract_nose,
        ear_expand=ear_expand,
        hair_expand=hair_expand,
        head_mask_np=head_mask_np,
    )
    return masks, "ok"


# ── Node ─────────────────────────────────────────────────────────────────────

class BD_MediaPipeFaceMask(io.ComfyNode):
    """Landmark-precise face region masks using MediaPipe Face Landmarker (CPU, ~5 ms/frame)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_MediaPipeFaceMask",
            display_name="BD MP Face Mask",
            category="🧠BrainDead/Segmentation",
            description=(
                "Extracts per-region face masks via MediaPipe Face Landmarker landmarks. "
                "Deterministic, CPU-only, ~5 ms per frame. No SAM3 prompts needed. "
                "Outputs face_oval, skin (face minus eyes/brows/lips), "
                "individual feature masks (eye, brow, lips, nose, ear, forehead, hair)."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Input image or batch. Each frame processed independently.",
                ),
                io.Mask.Input(
                    "head_mask", optional=True,
                    tooltip="Optional external head silhouette (e.g. from SAM3). When wired, this mask "
                            "is used as the base for computing 'skin' instead of MediaPipe's face_oval. "
                            "MediaPipe still runs for landmark detection (eyes/brows/lips/etc.) — it "
                            "just doesn't determine the head boundary. Use this when MediaPipe's oval "
                            "misses bald heads, full chin edges, or unusual head shapes.",
                ),
                io.Float.Input(
                    "detection_confidence", default=0.5, min=0.1, max=1.0, step=0.05,
                    optional=True,
                    tooltip="Face detection confidence threshold.",
                ),
                io.Int.Input(
                    "face_expand", default=0, min=-30, max=80, step=1, optional=True,
                    tooltip="Pixels to expand (+) or contract (-) the face oval boundary. "
                            "+4 helps catch jaw-edge hair strands.",
                ),
                io.Int.Input(
                    "feature_expand", default=4, min=0, max=40, step=1, optional=True,
                    tooltip="Pixels to expand eye/brow/lip regions before subtracting from skin. "
                            "Covers lashes and lip border. 3–6 px is typical.",
                ),
                io.Boolean.Input(
                    "subtract_nose", default=False, optional=True,
                    tooltip="When True, nose region is also excluded from skin mask.",
                ),
                io.Int.Input(
                    "iris_expand", default=4, min=0, max=20, step=1, optional=True,
                    tooltip="Pixels to expand iris landmark ring outward to fill the coloured disc. "
                            "Iris ring is 4 points; 3-6 px gives a natural circle. 0 = tight hull only.",
                ),
                io.Int.Input(
                    "ear_expand", default=25, min=0, max=80, step=1, optional=True,
                    tooltip="Pixels to expand ear (preauricular) region beyond the face oval edge.",
                ),
                io.Int.Input(
                    "hair_expand", default=20, min=0, max=80, step=1, optional=True,
                    tooltip="Pixels to expand hair region downward into the hairline transition zone.",
                ),
                io.Combo.Input(
                    "bbox_feature",
                    options=_BBOX_FEATURES,
                    default="none", optional=True,
                    tooltip="Which feature to emit as a bounding box. 'none' skips bbox computation. "
                            "bbox is from frame 0 (or first detected frame). "
                            "Wire bbox to SAM3_Detect bboxes for box-prompted segmentation.",
                ),
                io.Int.Input(
                    "bbox_frame", default=0, min=0, max=63, step=1, optional=True,
                    tooltip="Which frame index to extract the bbox from (default 0).",
                ),
            ],
            outputs=[
                io.Mask.Output(display_name="face_oval",
                               tooltip="Complete face silhouette (jaw to hairline). No neck, no background."),
                io.Mask.Output(display_name="skin",
                               tooltip="head_mask (if wired) or face_oval, minus eyes + brows + lips "
                                       "(and nose if subtract_nose). Primary skin-painting region."),
                io.Mask.Output(display_name="left_eye",
                               tooltip="Left eye (subject's left = image right for front-facing)."),
                io.Mask.Output(display_name="right_eye",
                               tooltip="Right eye (subject's right = image left for front-facing)."),
                io.Mask.Output(display_name="eyes",
                               tooltip="Both eyes combined."),
                io.Mask.Output(display_name="left_brow",
                               tooltip="Left eyebrow."),
                io.Mask.Output(display_name="right_brow",
                               tooltip="Right eyebrow."),
                io.Mask.Output(display_name="brows",
                               tooltip="Both brows combined."),
                io.Mask.Output(display_name="left_iris",
                               tooltip="Left iris disc (coloured part of eye). Requires 478-point model."),
                io.Mask.Output(display_name="right_iris",
                               tooltip="Right iris disc."),
                io.Mask.Output(display_name="irises",
                               tooltip="Both irises combined."),
                io.Mask.Output(display_name="lips",
                               tooltip="Full lip area (upper + lower, inner + outer contour)."),
                io.Mask.Output(display_name="nose",
                               tooltip="Nose — tip, bridge, and nostril rims."),
                io.Mask.Output(display_name="left_ear",
                               tooltip="Left preauricular region (near lm 234), expanded by ear_expand."),
                io.Mask.Output(display_name="right_ear",
                               tooltip="Right preauricular region (near lm 454)."),
                io.Mask.Output(display_name="ears",
                               tooltip="Both ears combined."),
                io.Mask.Output(display_name="forehead",
                               tooltip="Upper ~40% of face_oval minus brow band."),
                io.Mask.Output(display_name="hair",
                               tooltip="Region above face oval, to image top edge."),
                io.String.Output(display_name="status"),
                io.String.Output(
                    display_name="bbox_json",
                    tooltip='Tight bbox of bbox_feature from bbox_frame as JSON string {"x","y","width","height"}. '
                            '"{}\"when bbox_feature=\'none\' or face not detected. '
                            "Note: BOUNDING_BOX typed output is omitted — V3 dispatcher routes the dict to the "
                            "wrong downstream parameter. Use bbox_json + a JSON parse node if you need a bbox socket.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        detection_confidence: float = 0.5,
        face_expand: int = 0,
        feature_expand: int = 4,
        iris_expand: int = 4,
        subtract_nose: bool = False,
        ear_expand: int = 25,
        hair_expand: int = 20,
        bbox_feature: str = "none",
        bbox_frame: int = 0,
    ) -> io.NodeOutput:

        n_out = len(_KEYS)
        _blank1 = torch.zeros((1, 1, 1), dtype=torch.float32)
        _empty_bbox_json: str = "{}"

        if not HAS_MEDIAPIPE or not HAS_CV2:
            missing = []
            if not HAS_MEDIAPIPE:
                missing.append("mediapipe")
            if not HAS_CV2:
                missing.append("opencv-python")
            return io.NodeOutput(
                *([_blank1] * n_out),
                f"BD_MediaPipeFaceMask: missing packages — pip install {' '.join(missing)}",
                _empty_bbox_json,
            )

        if not os.path.exists(_MODEL_PATH):
            return io.NodeOutput(
                *([_blank1] * n_out),
                f"BD_MediaPipeFaceMask: model not found at {_MODEL_PATH}",
                _empty_bbox_json,
            )

        _init_mp_idx()

        if image.ndim == 3:
            image = image.unsqueeze(0)
        B, H, W, C = image.shape

        # Normalise head_mask to (B, H, W) uint8 numpy if provided
        hm_batch: list[np.ndarray | None] = [None] * B
        if head_mask is not None:
            hm = head_mask
            if hm.ndim == 2:
                hm = hm.unsqueeze(0)           # (H,W) → (1,H,W)
            if hm.ndim == 3 and hm.shape[0] == 1 and B > 1:
                hm = hm.expand(B, -1, -1)     # broadcast single mask to all frames
            hm_np = (hm.cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
            for b in range(min(B, hm_np.shape[0])):
                # Resize to match image frame if needed
                frame_hm = hm_np[b]
                if frame_hm.shape[0] != H or frame_hm.shape[1] != W:
                    frame_hm = cv2.resize(frame_hm, (W, H), interpolation=cv2.INTER_LINEAR)
                hm_batch[b] = frame_hm

        region_batches: dict[str, list[torch.Tensor]] = {k: [] for k in _KEYS}
        statuses: list[str] = []

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
                # MediaPipe Tasks API expects RGB uint8
                frame_rgb = frame_u8.copy()

                masks, status = _process_frame(
                    frame_rgb, landmarker,
                    face_expand=face_expand,
                    feature_expand=feature_expand,
                    iris_expand=iris_expand,
                    subtract_nose=subtract_nose,
                    ear_expand=ear_expand,
                    hair_expand=hair_expand,
                    head_mask_np=hm_batch[b],
                )
                for k in _KEYS:
                    region_batches[k].append(_to_tensor(masks[k]))
                statuses.append(status)

        outputs = [torch.stack(region_batches[k], dim=0) for k in _KEYS]

        detected = sum(1 for s in statuses if s == "ok")
        failed = [i for i, s in enumerate(statuses) if s != "ok"]
        base_note = " [skin base: external head_mask]" if head_mask is not None else ""
        status_str = (
            f"BD_MediaPipeFaceMask: {detected}/{B} faces detected{base_note}"
            + (f" — failed frames: {failed}" if failed else "")
        )
        print(f"[BD_MediaPipeFaceMask] {status_str}", flush=True)

        # Bbox JSON for selected feature from bbox_frame
        import json as _json
        bbox_json_out: str = "{}"
        if bbox_feature != "none" and bbox_feature in _KEYS:
            fi = min(bbox_frame, B - 1)
            feat_mask_np = (region_batches[bbox_feature][fi].numpy() * 255).astype(np.uint8)
            result = _bbox_from_mask(feat_mask_np)
            if result is not None:
                bbox_json_out = _json.dumps(result)

        return io.NodeOutput(*outputs, status_str, bbox_json_out)


FACE_MASK_MP_V3_NODES = [BD_MediaPipeFaceMask]
FACE_MASK_MP_NODES = {"BD_MediaPipeFaceMask": BD_MediaPipeFaceMask}
FACE_MASK_MP_DISPLAY_NAMES = {"BD_MediaPipeFaceMask": "BD MP Face Mask"}
