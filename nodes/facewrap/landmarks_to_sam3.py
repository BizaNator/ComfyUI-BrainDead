"""
BD_FaceLandmarksToSAM3 — extract per-feature pixel coords from a LANDMARKS_BATCH
and format them as JSON strings for SAM3_Detect's positive_coords / negative_coords.

BD_FaceLandmarks already runs MediaPipe and carries all 478 landmark pixel coords
in landmarks_2d (N, 2). This node indexes into that array using the standard
MediaPipe feature index sets and emits one JSON string per feature region.

Wire pattern:
  BD_FaceLandmarks → BD_FaceLandmarksToSAM3 → left_eye_coords → SAM3_Detect (positive_coords)
                                             → left_eye_bbox  → SAM3_Detect (bboxes)

Why use coords vs text prompts with BD_SAM3MultiPrompt:
  Point prompts are deterministic — SAM3 segments the exact region you clicked.
  Text prompts require IoU matching to link the output back to a feature.
  With positive_coords + bbox, SAM3 segments the feature in one pass with no ambiguity.
"""

from __future__ import annotations

import json
import numpy as np
import torch

from comfy_api.latest import io
from .types import LandmarksBatchInput


# ── MediaPipe landmark indices per feature ──────────────────────────────────
# These are initialised lazily from the MediaPipe API (same constants that
# BD_FaceMaskMediaPipe uses internally). Hard-coded fallback never fires in
# practice since MediaPipe is always available when BD_FaceLandmarks is running.

_MP_IDX: dict[str, list[int]] = {}

# Nose: custom set (no named connection in MediaPipe FaceLandmarksConnections)
_NOSE_INDICES = sorted({
    168, 6, 197, 195, 5, 4, 1, 2,
    19, 94, 141, 370,
    51, 45, 131, 134, 102, 48, 115, 49,
    281, 275, 360, 363, 331, 278, 344, 279,
    98, 327,
})

# Face oval — ordered path matching BD_FaceMaskMediaPipe
_OVAL_PATH = [
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
    148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
    127, 162, 21, 54, 103, 67, 109,
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
            "lips":       _verts(FLC.FACE_LANDMARKS_LIPS),
        }
    except Exception:
        # Fallback known indices (MediaPipe 0.10 / FaceMesh 478-point model)
        _MP_IDX = {
            "left_eye":   [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246],
            "right_eye":  [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398],
            "left_brow":  [46,53,52,65,55,70,63,105,66,107],
            "right_brow": [276,283,282,295,285,300,293,334,296,336],
            "lips":       [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,
                           84,181,91,146,76,184,74,73,72,11,302,303,304,408,415,310,
                           311,312,13,82,81,80,191,78,95,88,178,87,14,317,402,318,324,308],
        }


def _pts_from_batch(lm2d: np.ndarray, indices: list[int]) -> list[dict]:
    """Extract pixel coords for the given landmark indices as SAM3 coord dicts."""
    pts = lm2d[indices]   # (N, 2) float — already in pixel space
    return [{"x": int(round(float(p[0]))), "y": int(round(float(p[1])))} for p in pts]


def _bbox_from_pts(pts: list[dict], H: int, W: int) -> dict:
    """Compute tight bounding box from a list of {x, y} pixel dicts."""
    if not pts:
        return {"x": 0.0, "y": 0.0, "width": float(W), "height": float(H)}
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    x1, y1 = max(0, min(xs)), max(0, min(ys))
    x2, y2 = min(W, max(xs)), min(H, max(ys))
    return {"x": float(x1), "y": float(y1), "width": float(x2 - x1), "height": float(y2 - y1)}


# ── Node ─────────────────────────────────────────────────────────────────────

class BD_FaceLandmarksToSAM3(io.ComfyNode):
    """
    Convert a LANDMARKS_BATCH (from BD_FaceLandmarks) into per-feature JSON
    coordinate strings for SAM3_Detect's positive_coords and bboxes inputs.

    Eliminates the need for text prompts + IoU matching: SAM3 segments exactly
    the region pointed to by the MediaPipe landmarks, one precise mask per feature.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_FaceLandmarksToSAM3",
            display_name="BD Face Landmarks → SAM3",
            category="🧠BrainDead/FaceWrap",
            description=(
                "Convert BD_FaceLandmarks output into per-feature JSON coordinate strings "
                "for SAM3_Detect's positive_coords and bboxes inputs.\n\n"
                "Wire positive_coords directly to SAM3_Detect — no text prompts, no IoU matching. "
                "SAM3 segments exactly the region the MediaPipe landmarks point to.\n\n"
                "Typical use: run one SAM3_Detect per feature (left_eye, right_eye, lips, etc.) "
                "wiring the matching *_coords output as positive_coords and *_bbox as bboxes."
            ),
            inputs=[
                LandmarksBatchInput(
                    "landmarks_batch",
                    tooltip="LANDMARKS_BATCH from BD_FaceLandmarks.",
                ),
                io.Int.Input(
                    "frame_index", default=0, min=0, max=63, step=1, optional=True,
                    tooltip="Which frame in the batch to extract coords for. "
                            "For single-image pipelines this is always 0.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="left_eye_coords",
                    tooltip="Left eye landmark polygon as JSON [{x,y},...] — wire to SAM3_Detect positive_coords.",
                ),
                io.String.Output(display_name="right_eye_coords",  tooltip="Right eye landmark polygon."),
                io.String.Output(display_name="left_brow_coords",  tooltip="Left eyebrow landmark polygon."),
                io.String.Output(display_name="right_brow_coords", tooltip="Right eyebrow landmark polygon."),
                io.String.Output(display_name="lips_coords",       tooltip="Lips landmark polygon."),
                io.String.Output(display_name="nose_coords",       tooltip="Nose landmark polygon."),
                io.String.Output(display_name="face_oval_coords",  tooltip="Full face oval landmark polygon."),
                io.String.Output(
                    display_name="left_eye_bbox",
                    tooltip="Left eye tight bounding box as JSON {x,y,width,height} — wire to SAM3_Detect bboxes.",
                ),
                io.String.Output(display_name="right_eye_bbox",  tooltip="Right eye bbox JSON."),
                io.String.Output(display_name="left_brow_bbox",  tooltip="Left brow bbox JSON."),
                io.String.Output(display_name="right_brow_bbox", tooltip="Right brow bbox JSON."),
                io.String.Output(display_name="lips_bbox",       tooltip="Lips bbox JSON."),
                io.String.Output(display_name="nose_bbox",       tooltip="Nose bbox JSON."),
                io.String.Output(display_name="face_oval_bbox",  tooltip="Face oval bbox JSON."),
                io.String.Output(
                    display_name="feature_coords",
                    tooltip='All features as a JSON dict {"left_eye": [{x,y},...], "right_eye": [...], ...}. '
                            "Use when you need all coords in one wire rather than per-feature strings.",
                ),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        landmarks_batch: dict,
        frame_index: int = 0,
    ) -> io.NodeOutput:

        empty_coords = "[]"
        empty_bbox = "{}"
        n_coord_outputs = 7
        n_bbox_outputs = 7

        def _empty_return(reason: str):
            return io.NodeOutput(
                *([empty_coords] * n_coord_outputs),
                *([empty_bbox] * n_bbox_outputs),
                "{}",  # feature_coords
                reason,
            )

        views = (landmarks_batch or {}).get("views", [])
        if not views:
            return _empty_return("BD_FaceLandmarksToSAM3: no views in landmarks_batch")

        idx = min(frame_index, len(views) - 1)
        view = views[idx]

        if not view.get("detected", False):
            return _empty_return(
                f"BD_FaceLandmarksToSAM3: frame {idx} — no face detected (view={view.get('view_hint','?')})"
            )

        lm2d: np.ndarray = view["landmarks_2d"]   # (478, 2) float pixel coords
        H, W = view["image_size"]

        _init_mp_idx()

        feature_index_map = {
            "left_eye":   _MP_IDX.get("left_eye", []),
            "right_eye":  _MP_IDX.get("right_eye", []),
            "left_brow":  _MP_IDX.get("left_brow", []),
            "right_brow": _MP_IDX.get("right_brow", []),
            "lips":       _MP_IDX.get("lips", []),
            "nose":       _NOSE_INDICES,
            "face_oval":  _OVAL_PATH,
        }

        coords: dict[str, list[dict]] = {}
        bboxes: dict[str, dict] = {}
        for feat, indices in feature_index_map.items():
            pts = _pts_from_batch(lm2d, indices)
            coords[feat] = pts
            bboxes[feat] = _bbox_from_pts(pts, H, W)

        feat_order = ["left_eye", "right_eye", "left_brow", "right_brow", "lips", "nose", "face_oval"]

        coord_jsons = [json.dumps(coords[f]) for f in feat_order]
        bbox_jsons  = [json.dumps(bboxes[f]) for f in feat_order]
        all_coords_json = json.dumps({f: coords[f] for f in feat_order})

        n_pts = {f: len(coords[f]) for f in feat_order}
        status = (
            f"BD_FaceLandmarksToSAM3: frame={idx} detected=True H={H} W={W} | "
            + " ".join(f"{f}={n_pts[f]}pts" for f in feat_order)
        )
        print(f"[BD_FaceLandmarksToSAM3] {status}", flush=True)

        return io.NodeOutput(
            *coord_jsons,   # 7 coord strings
            *bbox_jsons,    # 7 bbox strings
            all_coords_json,
            status,
        )


FACEWRAP_LANDMARKS_SAM3_V3_NODES = [BD_FaceLandmarksToSAM3]
FACEWRAP_LANDMARKS_SAM3_NODES = {"BD_FaceLandmarksToSAM3": BD_FaceLandmarksToSAM3}
FACEWRAP_LANDMARKS_SAM3_DISPLAY_NAMES = {"BD_FaceLandmarksToSAM3": "BD Face Landmarks → SAM3"}
