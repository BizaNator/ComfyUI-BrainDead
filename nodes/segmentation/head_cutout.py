"""
BD_HeadCutout — clean head-only cutout (no neck), the same way every time.

Extracted from BD_MediaPipeSAM3FaceSegment's head-silhouette path so you can get JUST the
head cut + head mask without the 30-output face-segment node. The cut is angle-agnostic —
it works on front-facing AND side/profile heads with NO angle selector, because the
silhouette comes from text-grounded SAM3 (head_prompts ∪ minus exclude_prompts) plus an
in-house SegFormer neck removal, neither of which depends on a fixed pose. MediaPipe is used
only for the connected-component seed (and to report the detected orientation).

Typical use: wire `head_mask` onto any downstream output to strip the neck Qwen likes to add,
or use `head_cutout` directly as the isolated head on transparent / black / white.
"""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .face_mp_shared import (
    find_mediapipe_model, detect_landmarks_robust, _masks_from_landmarks, _blank, _init_mp_idx,
    HAS_MEDIAPIPE,
)
from .face_mp_sam3 import _fill_holes, _refine_feature_mask

_MODEL_PATH = find_mediapipe_model()


def _orientation(lm, W: int) -> str:
    """Rough front/side classification from landmark symmetry — informational only.
    nose tip (1) vs the left (234) / right (454) face edges."""
    try:
        nx = lm[1].x * W
        lx, rx = lm[234].x * W, lm[454].x * W
        span = max(1.0, abs(rx - lx))
        bias = ((nx - lx) - (rx - nx)) / span      # 0 = centred, + = turned right, − = left
        if bias > 0.30:
            return "side_right"
        if bias < -0.30:
            return "side_left"
        return "front"
    except Exception:
        return "front"


class BD_HeadCutout(io.ComfyNode):
    """Clean head-only cutout + head mask (no neck). Front or side, no angle selector."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_HeadCutout",
            display_name="BD Head Cutout",
            category="🧠BrainDead/Segmentation",
            description=(
                "Isolate JUST the head (no neck), the same way for every image — front-facing or "
                "side/profile, no angle selector. SAM3 segments head_prompts and subtracts "
                "exclude_prompts (neck / clothing); an in-house SegFormer pass then removes the neck "
                "chin-safely (bridges + drops the disconnected neck via connected-components). "
                "Outputs the head cutout (on transparent / black / white) and the head MASK to apply "
                "onto any downstream output (e.g. strip the neck Qwen adds)."
            ),
            inputs=[
                io.Image.Input("image"),
                io.String.Input("head_prompts", default="head\nface\nhair\near", multiline=True, optional=True,
                                tooltip="SAM3 positives — one per line. Their union = the head."),
                io.String.Input("exclude_prompts", default="neck\nshirt\nclothing\nshoulder",
                                multiline=True, optional=True,
                                tooltip="SAM3 negatives — subtracted from the head (peels off neck / clothing / body)."),
                io.Boolean.Input("neck_cut", default=True, optional=True,
                                 tooltip="Remove the neck via the SegFormer face parser (chin-safe: it never cuts a "
                                         "flat line into the chin — the chin stays as part of the head component). "
                                         "Off = keep whatever exclude_prompts left."),
                io.Combo.Input("cutout_bg", options=["transparent", "black", "white"], default="transparent",
                               optional=True,
                               tooltip="Background for the head_cutout image: transparent = RGBA; black/white = baked RGB."),
                io.Combo.Input("edge_refine", options=["off", "guided", "matting", "vitmatte"], default="off",
                               optional=True,
                               tooltip="Snap the head edge to the image: guided (fast), matting (soft hair), "
                                       "vitmatte (best, downloads a model). off = the raw SAM3 + close silhouette."),
                io.Int.Input("refine_radius", default=8, min=1, max=64, step=1, optional=True),
                io.Float.Input("refine_eps", default=1e-4, min=1e-6, max=1e-1, step=1e-5, optional=True),
                io.Float.Input("refine_threshold", default=0.5, min=0.0, max=1.0, step=0.05, optional=True),
                io.Combo.Input("vitmatte_model", options=["small", "base"], default="small", optional=True),
                io.Float.Input("mask_threshold", default=0.5, min=0.0, max=1.0, step=0.05, optional=True,
                               tooltip="SAM3 text-segmentation threshold."),
            ],
            outputs=[
                io.Image.Output(display_name="head_cutout"),
                io.Mask.Output(display_name="head_mask"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, image, head_prompts="head\nface\nhair\near",
                exclude_prompts="neck\nshirt\nclothing\nshoulder", neck_cut=True,
                cutout_bg="transparent", edge_refine="off", refine_radius=8, refine_eps=1e-4,
                refine_threshold=0.5, vitmatte_model="small", mask_threshold=0.5) -> io.NodeOutput:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        B, H, W, C = image.shape
        rgb01 = image[0, ..., :3].detach().cpu().float().numpy()
        np_img = (rgb01 * 255.0).clip(0, 255).astype(np.uint8)

        def _compose(mask_u8):
            a = (mask_u8 > 0).astype(np.float32)[..., None]
            if cutout_bg == "transparent":
                out = np.concatenate([rgb01, a], axis=-1)
            else:
                bgv = 0.0 if cutout_bg == "black" else 1.0
                out = rgb01 * a + bgv * (1.0 - a)
            return torch.from_numpy(np.ascontiguousarray(out.astype(np.float32))).unsqueeze(0)

        def _bail(msg):
            ch = 4 if cutout_bg == "transparent" else 3
            blank = torch.zeros((1, H, W, ch), dtype=torch.float32)
            return io.NodeOutput(blank, torch.zeros((1, H, W), dtype=torch.float32), msg)

        if not HAS_MEDIAPIPE or not HAS_CV2:
            return _bail("missing mediapipe/opencv")

        # ── MediaPipe (face_oval seed + orientation) ──
        _init_mp_idx()
        try:
            lm, det = detect_landmarks_robust(np_img, _MODEL_PATH, min_conf=0.3, min_span=0.2)
        except Exception as e:
            return _bail(f"MediaPipe error: {e}")
        if lm is None:
            return _bail("no face detected")
        orient = _orientation(lm, W)
        mp_masks = _masks_from_landmarks(lm, H, W, face_expand=0, feature_expand=0,
                                         iris_expand=0, ear_expand=25, hair_expand=0,
                                         tight_features=True)
        face_oval = mp_masks["face_oval"]

        # ── SAM3 text-grounded head silhouette (head_prompts ∪ minus exclude_prompts) ──
        from . import bd_sam3
        _sam = {}

        def _text_mask(prompt: str) -> np.ndarray:
            if "clip" not in _sam:
                _sam["model"], _sam["clip"] = bd_sam3.load_sam3(need_clip=True)
            t = bd_sam3.segment_text(_sam["model"], _sam["clip"], image, prompt, threshold=mask_threshold)
            m = (t[0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            return m

        def _union(lines):
            acc = _blank(H, W)
            for p in [s.strip() for s in (lines or "").split("\n") if s.strip()]:
                acc = np.maximum(acc, _text_mask(p))
            return acc

        pos = _union(head_prompts)
        neg = _union(exclude_prompts)
        head_sil = (pos.astype(bool) & ~neg.astype(bool)).astype(np.uint8) * 255
        if not head_sil.any():
            return _bail("empty head silhouette")

        # ── chin-safe neck removal (SegFormer Neck/Clothing/Necklace) ──
        cut_info = ""
        if neck_cut:
            parts = []
            nc = None
            try:
                from . import bd_face_parse
                pred = bd_face_parse.parse(image)
                nc = bd_face_parse.class_mask(pred, ["Neck", "Clothing", "Necklace"])
                if nc.shape != head_sil.shape:
                    nc = cv2.resize(nc, (W, H), interpolation=cv2.INTER_NEAREST)
            except Exception as e:
                print(f"[BD_HeadCutout] face-parser unavailable ({e})", flush=True)
            if nc is not None and (nc > 0).any():
                area = float((nc > 0).mean())
                band = nc.copy()
                if area > 0.005:
                    bw = max(15, int(0.05 * W)) | 1
                    band = cv2.dilate(band, cv2.getStructuringElement(cv2.MORPH_RECT, (bw, 3)))
                head_sil[band > 0] = 0
                parts.append(f"faceparse{100.0 * area:.1f}%")
                nlbl, lab, stats, _ = cv2.connectedComponentsWithStats((head_sil > 0).astype(np.uint8), 8)
                if nlbl > 1:
                    ys, xs = np.where(face_oval > 0)
                    seed = 0
                    if len(xs):
                        cy, cx = int(ys.mean()), int(xs.mean())
                        if 0 <= cy < H and 0 <= cx < W and head_sil[cy, cx] > 0:
                            seed = int(lab[cy, cx])
                    if seed == 0:
                        seed = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    head_sil = (lab == seed).astype(np.uint8) * 255
                    parts.append("head-cc")
            else:
                parts.append("no-neck-detected")
            cut_info = " | neck:" + "+".join(parts) if parts else ""

        # ── close + fill + optional edge refine ──
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        head_sil = cv2.morphologyEx(head_sil, cv2.MORPH_CLOSE, k)
        head_sil = _fill_holes(head_sil)
        if edge_refine != "off":
            try:
                head_sil = _refine_feature_mask(head_sil, np_img, edge_refine,
                                                int(refine_radius), float(refine_eps),
                                                float(refine_threshold), vitmatte_model)
            except Exception as e:
                print(f"[BD_HeadCutout] edge_refine '{edge_refine}' failed ({e})", flush=True)

        coverage = 100.0 * (head_sil > 0).mean()
        status = (f"head {coverage:.1f}% | orientation={orient} | bg={cutout_bg} "
                  f"| edge_refine={edge_refine}{cut_info}")
        print(f"[BD_HeadCutout] {status}", flush=True)
        return io.NodeOutput(_compose(head_sil),
                             torch.from_numpy((head_sil.astype(np.float32) / 255.0)).unsqueeze(0),
                             status)


HEAD_CUTOUT_V3_NODES = [BD_HeadCutout]
HEAD_CUTOUT_NODES = {"BD_HeadCutout": BD_HeadCutout}
HEAD_CUTOUT_DISPLAY_NAMES = {"BD_HeadCutout": "BD Head Cutout"}
