"""
BD_FaceFeatureCoords — extract pixel coords for selected face features from LANDMARKS_BATCH.

Toggle individual features on/off. The combined point cloud of all enabled
features is emitted as SAM3_Detect-ready JSON.

Outputs:
  positive_coords  STRING  JSON [{x,y},...] — wire to SAM3_Detect positive_coords
  bbox_json        STRING  JSON {x,y,width,height} — informational tight bbox
  status           STRING

Wire pattern:
  BD_FaceLandmarks → BD_FaceFeatureCoords → positive_coords → SAM3_Detect → mask

  For independent left/right masks: use two nodes with only the relevant
  eye/iris toggle enabled on each.
"""

from __future__ import annotations

import json
import numpy as np

from comfy_api.latest import io
from .types import LandmarksBatchInput


# ── MediaPipe indices, lazily initialised ─────────────────────────────────────

_MP_IDX: dict[str, list[int]] = {}

_NOSE_INDICES: list[int] = sorted({
    168, 6, 197, 195, 5, 4, 1, 2,
    19, 94, 141, 370,
    51, 45, 131, 134, 102, 48, 115, 49,
    281, 275, 360, 363, 331, 278, 344, 279,
    98, 327,
})

_FALLBACK_IDX: dict[str, list[int]] = {
    "left_eye":   [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246],
    "right_eye":  [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398],
    "left_brow":  [46,53,52,65,55,70,63,105,66,107],
    "right_brow": [276,283,282,295,285,300,293,334,296,336],
    "left_iris":  [474,475,476,477],
    "right_iris": [469,470,471,472],
    "lips": [
        0,13,14,17,37,39,40,61,78,80,81,82,84,87,88,91,95,
        146,178,181,185,191,267,269,270,291,308,310,311,312,314,
        317,318,321,324,375,402,405,409,415,
    ],
    "face_oval": [
        10,21,54,58,67,93,103,109,127,132,136,148,149,150,152,
        162,172,176,234,251,284,288,297,323,332,338,356,361,365,
        377,378,379,389,397,400,454,
    ],
    "contours": [
        0,7,10,13,14,17,21,33,37,39,40,46,52,53,54,55,58,61,
        63,65,66,67,70,78,80,81,82,84,87,88,91,93,95,103,105,
        107,109,127,132,133,136,144,145,146,148,149,150,152,153,
        154,155,157,158,159,160,161,162,163,172,173,176,178,181,
        185,191,234,246,249,251,263,267,269,270,276,282,283,284,
        285,288,291,293,295,296,297,300,308,310,311,312,314,317,
        318,321,323,324,332,334,336,338,356,361,362,365,373,374,
        375,377,378,379,380,381,382,384,385,386,387,388,389,390,
        397,398,400,402,405,409,415,454,466,
    ],
    "nose": _NOSE_INDICES,
}

_ALL_FEATURES = [
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

        def _verts(conns) -> list[int]:
            s = set()
            for c in conns:
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


# ── Node ─────────────────────────────────────────────────────────────────────

class BD_FaceFeatureCoords(io.ComfyNode):
    """
    Extract pixel-space landmark coords for selected face features from LANDMARKS_BATCH.

    Toggle individual features on/off.  All enabled features are merged into a
    single positive_coords JSON string for one SAM3_Detect call.

    Use two nodes (e.g. left_eye only / right_eye only) to produce independent
    masks from two separate SAM3_Detect calls.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_FaceFeatureCoords",
            display_name="BD Face Feature Coords",
            category="🧠BrainDead/FaceWrap",
            description=(
                "Extract MediaPipe landmark coords for selected face features and emit "
                "them as SAM3_Detect-ready JSON.\n\n"
                "Toggle individual features. All enabled features are merged into one "
                "positive_coords payload for a single SAM3_Detect call.\n\n"
                "positive_coords → SAM3_Detect positive_coords (STRING)\n"
                "bbox_json       → informational bounding box (STRING JSON)"
            ),
            inputs=[
                LandmarksBatchInput(
                    "landmarks_batch",
                    tooltip="LANDMARKS_BATCH from BD_FaceLandmarks.",
                ),
                io.Int.Input(
                    "frame_index", default=0, min=0, max=63, step=1, optional=True,
                    tooltip="Which frame in the batch to extract. 0 for single-image pipelines.",
                ),
                io.Boolean.Input("left_eye",   default=False, optional=True),
                io.Boolean.Input("right_eye",  default=False, optional=True),
                io.Boolean.Input("left_brow",  default=False, optional=True),
                io.Boolean.Input("right_brow", default=False, optional=True),
                io.Boolean.Input("left_iris",  default=False, optional=True),
                io.Boolean.Input("right_iris", default=False, optional=True),
                io.Boolean.Input("lips",       default=False, optional=True),
                io.Boolean.Input("nose",       default=False, optional=True),
                io.Boolean.Input("face_oval",  default=False, optional=True),
                io.Boolean.Input("contours",   default=False, optional=True),
            ],
            outputs=[
                io.String.Output(
                    display_name="positive_coords",
                    tooltip="JSON [{x,y},...] of all enabled feature landmarks. "
                            "Wire to SAM3_Detect positive_coords.",
                ),
                io.String.Output(
                    display_name="bbox_json",
                    tooltip="Tight bounding box as JSON {x,y,width,height} covering all enabled features.",
                ),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        landmarks_batch: dict,
        frame_index: int = 0,
        left_eye: bool = False,
        right_eye: bool = False,
        left_brow: bool = False,
        right_brow: bool = False,
        left_iris: bool = False,
        right_iris: bool = False,
        lips: bool = False,
        nose: bool = False,
        face_oval: bool = False,
        contours: bool = False,
    ) -> io.NodeOutput:

        enabled = {
            "left_eye": left_eye, "right_eye": right_eye,
            "left_brow": left_brow, "right_brow": right_brow,
            "left_iris": left_iris, "right_iris": right_iris,
            "lips": lips, "nose": nose,
            "face_oval": face_oval, "contours": contours,
        }
        selected = [f for f in _ALL_FEATURES if enabled.get(f, False)]

        if not selected:
            return io.NodeOutput(
                "[]", "{}",
                "BD_FaceFeatureCoords: no features selected",
            )

        views = (landmarks_batch or {}).get("views", [])
        if not views:
            return io.NodeOutput("[]", "{}", "BD_FaceFeatureCoords: no views in landmarks_batch")

        idx = min(frame_index, len(views) - 1)
        view = views[idx]

        if not view.get("detected", False):
            return io.NodeOutput(
                "[]", "{}",
                f"BD_FaceFeatureCoords: frame {idx} — no face detected",
            )

        lm2d: np.ndarray = view["landmarks_2d"]   # (478, 2) float pixel coords
        H, W = view["image_size"]

        _init_mp_idx()

        combined: set[int] = set()
        for feat in selected:
            combined.update(_MP_IDX.get(feat, []))
        indices = sorted(combined)

        pts_arr = lm2d[indices]   # (N, 2)
        coords_json = json.dumps([
            {"x": int(round(float(p[0]))), "y": int(round(float(p[1])))}
            for p in pts_arr
        ])

        xs, ys = pts_arr[:, 0], pts_arr[:, 1]
        x1 = float(max(0.0, float(xs.min())))
        y1 = float(max(0.0, float(ys.min())))
        x2 = float(min(float(W), float(xs.max())))
        y2 = float(min(float(H), float(ys.max())))
        bbox_json = json.dumps({"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1})

        status = (
            f"BD_FaceFeatureCoords: frame={idx} features=[{', '.join(selected)}] "
            f"pts={len(indices)} bbox=({x1:.0f},{y1:.0f} {x2-x1:.0f}×{y2-y1:.0f})"
        )
        print(f"[BD_FaceFeatureCoords] {status}", flush=True)

        return io.NodeOutput(coords_json, bbox_json, status)


FACEWRAP_FEATURE_COORDS_V3_NODES = [BD_FaceFeatureCoords]
FACEWRAP_FEATURE_COORDS_NODES = {"BD_FaceFeatureCoords": BD_FaceFeatureCoords}
FACEWRAP_FEATURE_COORDS_DISPLAY_NAMES = {"BD_FaceFeatureCoords": "BD Face Feature Coords"}
