"""
BD_FaceFeatureCoords — extract pixel coords for selected face feature(s) from LANDMARKS_BATCH.

Converts MediaPipe landmark positions into SAM3_Detect-ready outputs:
  - positive_coords: JSON [{x,y},...] — wire to SAM3_Detect positive_coords
  - bbox: BOUNDING_BOX {x,y,width,height} — wire to SAM3_Detect bboxes

Supported feature names (single or comma-separated):
  Individual:  left_eye  right_eye  left_brow  right_brow
               left_iris  right_iris  lips  nose  face_oval  contours
  Groups:      eyes (= left_eye + right_eye)
               brows / eyebrows (= left_brow + right_brow)
               irises (= left_iris + right_iris)
               all_features (= eyes + brows + irises + lips + nose)
               all (= all_features + face_oval + contours)

Wire pattern per feature:
  BD_FaceLandmarks → BD_FaceFeatureCoords (features="left_eye")
                   → positive_coords → SAM3_Detect → precise eye mask
                   → bbox           → SAM3_Detect bboxes (optional box constraint)

  Use two BD_FaceFeatureCoords nodes with "left_eye" / "right_eye" to get
  independent masks from two SAM3_Detect nodes when you need them separated.
"""

from __future__ import annotations

import json
import numpy as np

from comfy_api.latest import io
from .types import LandmarksBatchInput


# ── MediaPipe indices, lazily initialised ─────────────────────────────────────

_MP_IDX: dict[str, list[int]] = {}

# Nose: custom set (no named MediaPipe connection group)
_NOSE_INDICES: list[int] = sorted({
    168, 6, 197, 195, 5, 4, 1, 2,
    19, 94, 141, 370,
    51, 45, 131, 134, 102, 48, 115, 49,
    281, 275, 360, 363, 331, 278, 344, 279,
    98, 327,
})

# Hardcoded fallbacks — matches MediaPipe 0.10 478-point model
_FALLBACK_IDX: dict[str, list[int]] = {
    "left_eye":   [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "right_eye":  [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    "left_brow":  [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
    "right_brow": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
    "left_iris":  [474, 475, 476, 477],
    "right_iris": [469, 470, 471, 472],
    "lips": [
        0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95,
        146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314,
        317, 318, 321, 324, 375, 402, 405, 409, 415,
    ],
    "face_oval": [
        10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152,
        162, 172, 176, 234, 251, 284, 288, 297, 323, 332, 338, 356, 361, 365,
        377, 378, 379, 389, 397, 400, 454,
    ],
    "contours": [
        0, 7, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61,
        63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105,
        107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153,
        154, 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181,
        185, 191, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284,
        285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317,
        318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374,
        375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390,
        397, 398, 400, 402, 405, 409, 415, 454, 466,
    ],
    "nose": _NOSE_INDICES,
}

# Group aliases expand to one or more base feature names
_ALIASES: dict[str, list[str]] = {
    "eyes":        ["left_eye", "right_eye"],
    "brows":       ["left_brow", "right_brow"],
    "eyebrows":    ["left_brow", "right_brow"],
    "irises":      ["left_iris", "right_iris"],
    "all_features": ["left_eye", "right_eye", "left_brow", "right_brow",
                     "left_iris", "right_iris", "lips", "nose"],
    "all":         ["left_eye", "right_eye", "left_brow", "right_brow",
                    "left_iris", "right_iris", "lips", "nose",
                    "face_oval", "contours"],
}

_BASE_FEATURES = [
    "left_eye", "right_eye", "left_brow", "right_brow",
    "left_iris", "right_iris", "lips", "nose", "face_oval", "contours",
]


def _init_mp_idx() -> None:
    global _MP_IDX
    if _MP_IDX:
        return
    try:
        from mediapipe.tasks.python.vision.face_landmarker import (
            FaceLandmarksConnections as FLC,
        )

        def _verts(connections) -> list[int]:
            s = set()
            for c in connections:
                s.add(c.start)
                s.add(c.end)
            return sorted(s)

        _MP_IDX = {
            "left_eye":   _verts(FLC.FACE_LANDMARKS_LEFT_EYE),
            "right_eye":  _verts(FLC.FACE_LANDMARKS_RIGHT_EYE),
            "left_brow":  _verts(FLC.FACE_LANDMARKS_LEFT_EYEBROW),
            "right_brow": _verts(FLC.FACE_LANDMARKS_RIGHT_EYEBROW),
            "left_iris":  _verts(FLC.FACE_LANDMARKS_LEFT_IRIS),
            "right_iris": _verts(FLC.FACE_LANDMARKS_RIGHT_IRIS),
            "lips":       _verts(FLC.FACE_LANDMARKS_LIPS),
            "face_oval":  _verts(FLC.FACE_LANDMARKS_FACE_OVAL),
            "contours":   _verts(FLC.FACE_LANDMARKS_CONTOURS),
            "nose":       _NOSE_INDICES,
        }
    except Exception:
        _MP_IDX = dict(_FALLBACK_IDX)


def _resolve_features(features_str: str) -> tuple[list[int], list[str]]:
    """Parse a comma-separated feature string, expand aliases, return (indices, resolved_names)."""
    _init_mp_idx()
    tokens = [t.strip().lower() for t in features_str.split(",") if t.strip()]
    seen: set[str] = set()
    resolved: list[str] = []
    for tok in tokens:
        if tok in _ALIASES:
            for base in _ALIASES[tok]:
                if base not in seen:
                    seen.add(base)
                    resolved.append(base)
        elif tok in _MP_IDX:
            if tok not in seen:
                seen.add(tok)
                resolved.append(tok)
        # unknown tokens are silently skipped (status will show 0 pts)

    combined: set[int] = set()
    for name in resolved:
        combined.update(_MP_IDX.get(name, []))
    return sorted(combined), resolved


# ── Node ─────────────────────────────────────────────────────────────────────

class BD_FaceFeatureCoords(io.ComfyNode):
    """
    Extract pixel-space point coords for selected face feature(s) from LANDMARKS_BATCH.

    Outputs SAM3_Detect-ready positive_coords (STRING JSON) and a tight
    BOUNDING_BOX covering all selected features.  Select multiple features
    with a comma-separated list ("eyes, lips") to segment their union in one
    SAM3_Detect call.  Use two nodes with "left_eye" / "right_eye" when you
    need independent masks.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_FaceFeatureCoords",
            display_name="BD Face Feature Coords",
            category="🧠BrainDead/FaceWrap",
            description=(
                "Extract MediaPipe landmark coordinates for the selected face feature(s) "
                "and emit them as SAM3_Detect-ready JSON.\n\n"
                "positive_coords → SAM3_Detect positive_coords (STRING)\n"
                "bbox            → SAM3_Detect bboxes (BOUNDING_BOX)\n\n"
                "Features: left_eye  right_eye  eyes  left_brow  right_brow  brows\n"
                "          left_iris  right_iris  irises  lips  nose\n"
                "          face_oval  contours  all_features  all\n\n"
                "Comma-separate to combine: 'eyes, lips' → one SAM3 pass covering both."
            ),
            inputs=[
                LandmarksBatchInput(
                    "landmarks_batch",
                    tooltip="LANDMARKS_BATCH from BD_FaceLandmarks.",
                ),
                io.String.Input(
                    "features",
                    default="left_eye",
                    tooltip=(
                        "Feature(s) to extract. Single name or comma-separated list.\n"
                        "Individual: left_eye  right_eye  left_brow  right_brow\n"
                        "            left_iris  right_iris  lips  nose  face_oval  contours\n"
                        "Groups:     eyes (left+right eye)  brows (left+right brow)\n"
                        "            irises (left+right iris)  all_features  all"
                    ),
                ),
                io.Int.Input(
                    "frame_index", default=0, min=0, max=63, step=1, optional=True,
                    tooltip="Which frame in the batch to extract. 0 for single-image pipelines.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="positive_coords",
                    tooltip='JSON [{x,y},...] of all landmark points for the selected feature(s). '
                            'Wire to SAM3_Detect positive_coords.',
                ),
                io.BoundingBox.Output(
                    display_name="bbox",
                    tooltip='Tight bounding box {x,y,width,height} covering all selected points. '
                            'Wire to SAM3_Detect bboxes.',
                ),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        landmarks_batch: dict,
        features: str = "left_eye",
        frame_index: int = 0,
    ) -> io.NodeOutput:

        empty_bbox = {"x": 0.0, "y": 0.0, "width": 512.0, "height": 512.0}

        views = (landmarks_batch or {}).get("views", [])
        if not views:
            return io.NodeOutput("[]", empty_bbox, "BD_FaceFeatureCoords: no views in landmarks_batch")

        idx = min(frame_index, len(views) - 1)
        view = views[idx]

        if not view.get("detected", False):
            return io.NodeOutput(
                "[]", empty_bbox,
                f"BD_FaceFeatureCoords: frame {idx} — no face detected",
            )

        lm2d: np.ndarray = view["landmarks_2d"]   # (478, 2) float pixel coords
        H, W = view["image_size"]

        indices, resolved_names = _resolve_features(features)

        if not indices:
            return io.NodeOutput(
                "[]", empty_bbox,
                f"BD_FaceFeatureCoords: no valid features in '{features}'",
            )

        pts_arr = lm2d[indices]   # (N, 2)
        coords_json = json.dumps([
            {"x": int(round(float(p[0]))), "y": int(round(float(p[1])))}
            for p in pts_arr
        ])

        xs = pts_arr[:, 0]
        ys = pts_arr[:, 1]
        x1 = float(max(0.0, float(xs.min())))
        y1 = float(max(0.0, float(ys.min())))
        x2 = float(min(float(W), float(xs.max())))
        y2 = float(min(float(H), float(ys.max())))
        bbox = {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}

        status = (
            f"BD_FaceFeatureCoords: frame={idx} features=[{', '.join(resolved_names)}] "
            f"pts={len(indices)} bbox=({x1:.0f},{y1:.0f},{x2-x1:.0f}×{y2-y1:.0f})"
        )
        print(f"[BD_FaceFeatureCoords] {status}", flush=True)

        return io.NodeOutput(coords_json, bbox, status)


FACEWRAP_FEATURE_COORDS_V3_NODES = [BD_FaceFeatureCoords]
FACEWRAP_FEATURE_COORDS_NODES = {"BD_FaceFeatureCoords": BD_FaceFeatureCoords}
FACEWRAP_FEATURE_COORDS_DISPLAY_NAMES = {"BD_FaceFeatureCoords": "BD Face Feature Coords"}
