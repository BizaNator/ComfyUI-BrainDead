"""
BD_MediaPipeFaceExport — passthrough landmark export node.

Inserts anywhere in the pipeline on the full-color, full-resolution image
(before greyscale/albedo/channel-pack steps that break detection). Runs
MediaPipe FaceLandmarker on image[0], writes a landmark JSON + RGBA zone
mask PNG + source image to disk, then passes the image tensor through unchanged.

Mask PNG uses the SAME mask generation as BD MP Face Mask (shared module),
not crude convex hulls — masks are pixel-accurate and match BD MP Face Mask output.

RGBA mask PNG (same pixel dimensions as input):
    R = lips mask (shared library, matches BD MP Face Mask)
    G = brows mask
    B = eyes mask
    A = face_oval mask (traced polygon, not convex hull)

No-face: writes JSON with landmark_count=0, skips PNG, logs warning.

File placement:
    output_dir  = Characters/<char>/images/mp  (or any target directory)
    filename_stem = <char>_head_<ver>_mp  (angle appended automatically)
    → <stem>_<angle>.json + <stem>_<angle>_mask.png + <stem>_<angle>_image.png
"""

from __future__ import annotations

import os
import json
import numpy as np
import torch
from comfy_api.latest import io

import folder_paths as _folder_paths


from .face_mp_shared import find_mediapipe_model as _find_mediapipe_model

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
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

# Shared mask generation + robust detection — same code as BD MP Face Mask
from .face_mp_shared import (
    _init_mp_idx, _masks_from_landmarks,
    detect_landmarks_robust,
    _NOSE_INDICES,
)

# ── Region index lists for JSON (constant for all characters — MediaPipe topology) ─

_LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]
_LIPS_INNER = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191]
_LEFT_EYE   = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
_RIGHT_EYE  = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
_LEFT_BROW  = [70,63,105,66,107,55,65,52,53,46]
_RIGHT_BROW = [300,293,334,296,336,285,295,282,283,276]
_NOSE_TIP   = [4]
_CHIN       = [152]


# ── Node ──────────────────────────────────────────────────────────────────────

class BD_MediaPipeFaceExport(io.ComfyNode):
    """
    Passthrough node that exports MediaPipe landmark JSON + RGBA zone mask PNG.

    Insert BEFORE any albedo/greyscale/channel-pack steps that would break
    MediaPipe detection. The image is passed through unchanged — zero effect
    on downstream nodes.

    Outputs feed the face plate UV calibration pipeline:
        DCC/Characters/Head/calibrate_faceplate_uv.py (Eros P4)

    context_id mode (recommended):
        Wire a BD_SaveContext context_id — the node resolves output_dir and
        filename_stem automatically from the context template. angle becomes
        the suffix (e.g. _mp_front, _mp_side_left). output_dir and
        filename_stem inputs are ignored when context_id is set.

    Manual mode:
        Set output_dir to any writable directory, e.g.:
            Characters/<char>/images/mp
        Set filename_stem to:
            <char>_head_<ver>_mp_<angle>
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_MediaPipeFaceExport",
            display_name="BD MP Face Export",
            category="🧠BrainDead/Segmentation",
            description=(
                "Passthrough node: runs MediaPipe on the input image, writes landmark "
                "JSON + RGBA zone mask PNG to disk, passes image through unchanged. "
                "Insert before greyscale/albedo/channel-pack steps. "
                "Feeds calibrate_faceplate_uv.py in the face plate UV pipeline. "
                "Wire context_id for automatic path resolution."
            ),
            inputs=[
                io.Image.Input("image",
                               tooltip="Full-color full-resolution image. Only image[0] is processed; "
                                       "all batch items pass through unchanged."),
                io.Combo.Input("angle",
                               options=["front", "side_left", "side_right"],
                               default="front",
                               tooltip="Camera angle. Stored in JSON and used as path suffix in context mode "
                                       "(e.g. _mp_front → <stem>_mp_front.json)."),
                io.String.Input("context_id", default="",
                                optional=True,
                                tooltip="BD_SaveContext context_id. When set, output_dir and filename_stem "
                                        "are resolved from the context template. angle becomes the suffix."),
                io.String.Input("output_dir",
                                default="",
                                optional=True,
                                tooltip="Fallback absolute path when no context is registered. "
                                        "Ignored when context_id resolves."),
                io.String.Input("filename_stem",
                                default="",
                                optional=True,
                                tooltip="Fallback filename stem (no extension) when no context is registered. "
                                        "angle is appended automatically: <stem>_mp_front.json"),
                io.String.Input("_model_path_deprecated",
                                default=_MODEL_PATH,
                                optional=True,
                                tooltip="Deprecated — model path is now fixed internally. This input is kept for "
                                        "workflow backward-compatibility only and is ignored."),
                io.Float.Input("detection_confidence",
                               default=0.3, min=0.1, max=1.0, step=0.05,
                               optional=True,
                               tooltip="Minimum face detection confidence. 0.3 works for stylized renders."),
                io.Float.Input("min_face_span",
                               default=0.35, min=0.0, max=1.0, step=0.05,
                               optional=True,
                               tooltip="Sanity guard: minimum plausible face span (fraction of frame, larger of "
                                       "x/y). MediaPipe occasionally returns a degenerate tiny detection on a "
                                       "frame-filling head; below this the node retries with a fresh landmarker "
                                       "and padded copies before accepting. Set 0 to disable the guard."),
            ],
            outputs=[
                io.Image.Output(display_name="image",
                                tooltip="Passthrough — identical to input image."),
                io.Int.Output(display_name="landmark_count",
                              tooltip="478 if face detected, 0 if not."),
                io.String.Output(display_name="json_path",
                                 tooltip="Absolute path of the written JSON file."),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        angle: str = "front",
        context_id: str = "",
        output_dir: str = "",
        filename_stem: str = "",
        _model_path_deprecated: str = "",  # kept for workflow compat — value ignored
        detection_confidence: float = 0.3,
        min_face_span: float = 0.35,
    ) -> io.NodeOutput:

        # Passthrough is always valid regardless of detection
        def _pass(count: int, json_p: str) -> io.NodeOutput:
            return io.NodeOutput(image, count, json_p)

        if not HAS_MEDIAPIPE or not HAS_CV2:
            missing = [p for p, h in [("mediapipe", HAS_MEDIAPIPE), ("opencv-python", HAS_CV2)] if not h]
            print(f"[BD MP FaceExport] WARNING: missing {missing} — passthrough only, no export")
            return _pass(0, "")

        if not os.path.exists(_MODEL_PATH):
            print(f"[BD MP FaceExport] WARNING: model not found: {_MODEL_PATH} — passthrough only")
            return _pass(0, "")

        # ── Context resolution (same pattern as BD_SaveBatch) ─────────────────
        from ..cache.save_context import resolve_context_path, get_context, auto_pick_context

        effective_cid = (context_id or "").strip()
        if not effective_cid:
            effective_cid = auto_pick_context() or ""

        json_path = mask_path = ""
        if effective_cid and get_context(effective_cid) is not None:
            try:
                suffix = f"_mp_{angle}"
                full_path, _ = resolve_context_path(effective_cid, suffix, "json")
                json_path = full_path
                mask_path = full_path.replace(".json", "_mask.png")
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                print(f"[BD MP FaceExport] context '{effective_cid}' → {full_path}")
            except Exception as e:
                print(f"[BD MP FaceExport] WARNING: context resolve failed: {e}")

        if not json_path:
            # Manual fallback: output_dir + filename_stem
            out_dir = (output_dir or "").strip()
            if not out_dir:
                print(f"[BD MP FaceExport] WARNING: no context registered and no output_dir set — passthrough only")
                return _pass(0, "")
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                print(f"[BD MP FaceExport] WARNING: cannot create output_dir {out_dir!r}: {e}")
                return _pass(0, "")
            stem = (filename_stem or "face_mp").strip()
            if angle not in stem:
                stem = f"{stem}_{angle}"
            json_path = os.path.join(out_dir, f"{stem}.json")
            mask_path = os.path.join(out_dir, f"{stem}_mask.png")

        # image[0] → uint8 RGB numpy
        if image.ndim == 3:
            image = image.unsqueeze(0)
        frame = image[0].detach().cpu().float().numpy()
        H, W = frame.shape[:2]
        np_img = (frame[..., :3] * 255.0).clip(0, 255).astype(np.uint8)

        # Run MediaPipe with the tiny-detection guard (retry + padded fallback).
        # Coords come back normalized to THIS image regardless of which attempt won.
        _init_mp_idx()  # populate shared landmark index dicts

        try:
            lm, det_meta = detect_landmarks_robust(
                np_img, _MODEL_PATH,
                min_conf=detection_confidence,
                min_span=min_face_span,
            )
        except Exception as e:
            print(f"[BD MP FaceExport] WARNING: MediaPipe error: {e}")
            return _pass(0, "")

        if lm is None:
            # No face — write minimal JSON + image (no mask PNG)
            img_path = json_path.replace(".json", "_image.png")
            try:
                from PIL import Image as _PIL
                _PIL.fromarray(np_img, mode='RGB').save(img_path, format='PNG')
            except ImportError:
                cv2.imwrite(img_path, np_img[:, :, ::-1])
            no_face_data = {
                "schema_version": 1,
                "angle": angle,
                "image_size": [W, H],
                "landmark_count": 0,
                "landmarks": [],
                "face_bbox": None,
                "regions": _region_index_map(),
                "image_file": os.path.basename(img_path),
                "detection": det_meta,
            }
            with open(json_path, 'w') as f:
                json.dump(no_face_data, f, indent=2)
            print(f"[BD MP FaceExport] WARNING: no face detected — wrote {json_path}, image saved")
            return _pass(0, json_path)

        if det_meta.get("quality") == "degraded":
            # Every attempt stayed below min_face_span — the result is suspect.
            # Write it anyway (so the run isn't lost) but make the failure LOUD.
            print(f"[BD MP FaceExport] ERROR: degenerate detection — face span "
                  f"{det_meta.get('span')} < min_face_span {min_face_span} after "
                  f"{det_meta.get('attempts')} attempts. Mask will be UNDERSIZED. "
                  f"Check that a full-color, full-resolution face is wired in. "
                  f"({json_path})")
        else:
            print(f"[BD MP FaceExport] detection ok: span={det_meta.get('span')} "
                  f"via {det_meta.get('method')} ({det_meta.get('attempts')} attempt(s))")

        n_lm = len(lm)
        # Landmarks: normalized [x, y], Z omitted
        landmarks_xy = [[round(p.x, 6), round(p.y, 6)] for p in lm]

        # Face bounding box from all detected landmarks
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        face_bbox = {
            "x_min": round(min(xs), 6),
            "y_min": round(min(ys), 6),
            "x_max": round(max(xs), 6),
            "y_max": round(max(ys), 6),
        }

        # Write JSON
        export_data = {
            "schema_version": 1,
            "angle": angle,
            "image_size": [W, H],
            "landmark_count": n_lm,
            "landmarks": landmarks_xy,
            "face_bbox": face_bbox,
            "regions": _region_index_map(),
            "detection": det_meta,
        }
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        # Write RGBA zone mask PNG using shared mask generation (matches BD MP Face Mask)
        _write_mask_png_shared(mask_path, lm, H, W)

        # Write source image — Blender and other tools need this to cross-reference
        # landmark coordinates (which are normalised to this exact image's pixel space).
        img_path = json_path.replace(".json", "_image.png")
        try:
            from PIL import Image as _PIL
            _PIL.fromarray(np_img, mode='RGB').save(img_path, format='PNG')
        except ImportError:
            bgr = np_img[:, :, ::-1]
            cv2.imwrite(img_path, bgr)

        # Embed image filename in JSON for cross-reference
        export_data["image_file"] = os.path.basename(img_path)
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"[BD MP FaceExport] {n_lm} landmarks → {json_path}")
        print(f"[BD MP FaceExport] zone mask  → {mask_path}")
        print(f"[BD MP FaceExport] image      → {img_path}")
        return _pass(n_lm, json_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _region_index_map() -> dict:
    return {
        "lips_outer":  _LIPS_OUTER,
        "lips_inner":  _LIPS_INNER,
        "left_eye":    _LEFT_EYE,
        "right_eye":   _RIGHT_EYE,
        "left_brow":   _LEFT_BROW,
        "right_brow":  _RIGHT_BROW,
        "nose_tip":    _NOSE_TIP,
        "chin":        _CHIN,
    }


def _write_mask_png_shared(path: str, lm, H: int, W: int) -> None:
    """Write RGBA zone mask PNG using the same mask generation as BD MP Face Mask.

    Uses _masks_from_landmarks from the shared module — identical quality and
    region coverage to BD MP Face Mask (not crude convex hulls).

    Channel mapping (per spec):
        R = lips,      G = brows,      B = eyes,      A = face_oval
    """
    masks = _masks_from_landmarks(lm, H, W,
                                  face_expand=0, feature_expand=0,
                                  iris_expand=0, ear_expand=0, hair_expand=0)

    rgba = np.stack([
        masks['lips'],       # R
        masks['brows'],      # G
        masks['eyes'],       # B
        masks['face_oval'],  # A
    ], axis=-1)  # (H, W, 4) uint8

    try:
        from PIL import Image as _PIL
        _PIL.fromarray(rgba, mode='RGBA').save(path, format='PNG')
    except ImportError:
        bgra = rgba[:, :, [2, 1, 0, 3]]
        cv2.imwrite(path, bgra)


# ── Registration ──────────────────────────────────────────────────────────────

FACE_MP_EXPORT_V3_NODES = [BD_MediaPipeFaceExport]

FACE_MP_EXPORT_NODES = {
    "BD_MediaPipeFaceExport": BD_MediaPipeFaceExport,
}

FACE_MP_EXPORT_DISPLAY_NAMES = {
    "BD_MediaPipeFaceExport": "BD MP Face Export",
}
