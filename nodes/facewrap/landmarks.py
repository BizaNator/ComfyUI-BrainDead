"""
BD_FaceLandmarks — MediaPipe FaceLandmarker (Tasks API) on an IMAGE batch.

First node in the face-wrap pipeline. Takes a batch of head photos and
emits a LANDMARKS_BATCH carrying 478-point FaceMesh landmarks per view,
the per-view 4x4 facial transformation matrix (head pose), and a coarse
view-hint (front / left / right / rear) inferred from that matrix's yaw.

Rear / failed-detection views are passed through with `detected=False`
so downstream nodes can still see them in the batch (the texture-bake
step uses them with a pose derived from camera baselines).

Model: MediaPipe FaceLandmarker task bundle (~3.6MB), auto-downloaded
       to /srv/AI_Stuff/models/mediapipe/face_landmarker.task on first use.
"""

import os
from pathlib import Path

import numpy as np
import torch

from comfy_api.latest import io

from .types import LandmarksBatchOutput


DEFAULT_MODEL_PATH = "/srv/AI_Stuff/models/mediapipe/face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _resolve_model_path(override: str | None = None) -> str:
    """Return path to face_landmarker.task; auto-download to default if missing."""
    if override and override.strip():
        p = Path(os.path.expanduser(os.path.expandvars(override.strip())))
        if not p.exists():
            raise FileNotFoundError(f"FaceLandmarker model not found: {p}")
        return str(p)

    target = Path(DEFAULT_MODEL_PATH)
    if target.exists():
        return str(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[BD FaceLandmarks] Downloading FaceLandmarker model to {target}", flush=True)
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, str(target))
    return str(target)


def _yaw_from_transform(tform4x4: np.ndarray) -> float:
    """Extract yaw (signed degrees) from MediaPipe's facial_transformation_matrix.

    MediaPipe returns a 4x4 column-major rigid transform of the canonical
    face model in camera space. The rotation submatrix's [0,2] entry is
    sin(yaw) under YXZ Euler ordering (yaw=Y rotation).
    """
    R = tform4x4[:3, :3]
    # Yaw = atan2(R[0,2], R[2,2])  (Y-rotation under YXZ)
    yaw = float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))
    return yaw


def _classify_view(yaw_deg: float) -> str:
    """Map yaw angle to coarse view label."""
    a = abs(yaw_deg)
    if a < 25.0:
        return "front"
    if a > 155.0:
        return "rear"
    # Positive yaw (head turned to subject's right): camera sees subject's left side.
    return "left" if yaw_deg > 0 else "right"


def _draw_overlay(rgb: np.ndarray, lm_2d: np.ndarray, view_hint: str,
                  detected: bool) -> np.ndarray:
    """Draw landmark dots + view label onto an RGB float image."""
    import cv2

    h, w = rgb.shape[:2]
    out = (rgb * 255.0).astype(np.uint8).copy()

    if detected:
        for x, y in lm_2d.astype(np.int32):
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(out, (int(x), int(y)), 1, (0, 255, 0), -1)
        color = (255, 255, 0)
        label = f"view: {view_hint}"
    else:
        color = (255, 128, 0)
        label = f"view: {view_hint} (no face detected)"

    cv2.putText(out, label, (10, max(20, h // 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out.astype(np.float32) / 255.0


class BD_FaceLandmarks(io.ComfyNode):
    """Detect MediaPipe FaceLandmarker landmarks on a batch of head photos."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_FaceLandmarks",
            display_name="BD Face Landmarks",
            category="🧠BrainDead/FaceWrap",
            description=(
                "MediaPipe FaceLandmarker (Tasks API) on a batch of head photos.\n\n"
                "Outputs:\n"
                "- LANDMARKS_BATCH consumed by BD_FlameFit. Each view carries\n"
                "  478 2D+3D landmarks, a 4x4 head-pose transform, and a\n"
                "  coarse view-hint (front / left / right / rear).\n"
                "- Debug overlay image with landmarks + view label per view.\n\n"
                "Failed detections (rear / extreme profile) stay in the batch\n"
                "with detected=False so view ordering is preserved.\n\n"
                "Model bundle (~3.6MB) auto-downloads to\n"
                "/srv/AI_Stuff/models/mediapipe/face_landmarker.task on first use."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Batch of head photos. For the 4-view face-wrap "
                            "stack front/left/right/rear in that order.",
                ),
                io.Float.Input(
                    "min_detection_confidence",
                    default=0.4,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Lower = more profile views accepted. 0.4 is a "
                            "decent profile floor.",
                ),
                io.Float.Input(
                    "min_presence_confidence",
                    default=0.4,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="MediaPipe landmark-presence threshold.",
                ),
                io.String.Input(
                    "model_path",
                    default="",
                    optional=True,
                    tooltip=f"Override path to face_landmarker.task. Empty = "
                            f"use {DEFAULT_MODEL_PATH} (auto-download if missing).",
                ),
            ],
            outputs=[
                LandmarksBatchOutput(display_name="landmarks_batch"),
                io.Image.Output(display_name="overlay"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        min_detection_confidence: float = 0.4,
        min_presence_confidence: float = 0.4,
        model_path: str = "",
    ) -> io.NodeOutput:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                FaceLandmarker, FaceLandmarkerOptions, RunningMode,
            )
        except ImportError as e:
            empty = {"views": [], "model": "face_landmarker_v1", "n_landmarks": 478}
            return io.NodeOutput(empty, images, f"ERROR: mediapipe not available ({e})")

        if images is None or images.ndim != 4:
            empty = {"views": [], "model": "face_landmarker_v1", "n_landmarks": 478}
            return io.NodeOutput(empty, images, "ERROR: expected (B,H,W,3) image batch")

        try:
            resolved_model = _resolve_model_path(model_path)
        except Exception as e:
            empty = {"views": [], "model": "face_landmarker_v1", "n_landmarks": 478}
            return io.NodeOutput(empty, images, f"ERROR: model resolution failed: {e}")

        batch = images.detach().cpu().numpy()
        b, h, w, _ = batch.shape
        n_landmarks = 478

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=resolved_model),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )

        views = []
        overlays = []

        with FaceLandmarker.create_from_options(options) as landmarker:
            for i in range(b):
                rgb = np.clip(batch[i], 0.0, 1.0)
                rgb_u8 = (rgb * 255.0).astype(np.uint8)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_u8)
                result = landmarker.detect(mp_image)

                if not result.face_landmarks:
                    views.append({
                        "landmarks_2d": np.zeros((n_landmarks, 2), dtype=np.float32),
                        "landmarks_3d": np.zeros((n_landmarks, 3), dtype=np.float32),
                        "transform_4x4": np.eye(4, dtype=np.float32),
                        "detected": False,
                        "view_hint": "rear",
                        "image_size": (h, w),
                        "yaw_estimate": 0.0,
                    })
                    overlays.append(_draw_overlay(rgb, np.zeros((0, 2)), "rear", False))
                    continue

                lm = result.face_landmarks[0]
                lm_3d = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
                # NormalizedLandmark x,y in [0,1] image-space; z roughly in image-width units
                lm_2d_px = np.stack([lm_3d[:, 0] * w, lm_3d[:, 1] * h], axis=-1)

                tform = np.asarray(result.facial_transformation_matrixes[0],
                                   dtype=np.float32) if result.facial_transformation_matrixes \
                        else np.eye(4, dtype=np.float32)
                yaw_deg = _yaw_from_transform(tform)
                view_hint = _classify_view(yaw_deg)

                views.append({
                    "landmarks_2d": lm_2d_px,
                    "landmarks_3d": lm_3d,
                    "transform_4x4": tform,
                    "detected": True,
                    "view_hint": view_hint,
                    "image_size": (h, w),
                    "yaw_estimate": yaw_deg,
                })
                overlays.append(_draw_overlay(rgb, lm_2d_px, view_hint, True))

        overlay_tensor = torch.from_numpy(np.stack(overlays, axis=0)).float()

        n_detected = sum(1 for v in views if v["detected"])
        hints = ", ".join(v["view_hint"] for v in views)
        yaws = ", ".join(f"{v['yaw_estimate']:.0f}" for v in views)
        status = f"{n_detected}/{b} detected | views: [{hints}] | yaws°: [{yaws}]"

        landmarks_batch = {
            "views": views,
            "model": "face_landmarker_v1",
            "n_landmarks": n_landmarks,
            "model_path": resolved_model,
        }
        return io.NodeOutput(landmarks_batch, overlay_tensor, status)


FACEWRAP_LANDMARKS_V3_NODES = [BD_FaceLandmarks]

FACEWRAP_LANDMARKS_NODES = {
    "BD_FaceLandmarks": BD_FaceLandmarks,
}

FACEWRAP_LANDMARKS_DISPLAY_NAMES = {
    "BD_FaceLandmarks": "BD Face Landmarks",
}
