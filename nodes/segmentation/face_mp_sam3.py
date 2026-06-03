"""
BD_MediaPipeSAM3FaceSegment — MediaPipe-guided SAM3 face feature segmentation.

Runs MediaPipe ONCE to localize each face feature, then prompts SAM3 per feature
with the MediaPipe bbox + positive points (the feature's landmarks) + negative
points (sibling features) → pixel-accurate masks in a single node.

Why this beats text-prompt + IoU matching (BD_SAM3MultiPrompt + BD MP Face Refine):
  • No prompt ambiguity / no IoU matching / no exclusion list — every SAM3 call is
    spatially seeded at the exact MediaPipe location.
  • Solves the brow-offset problem: a positive point ON the painted brow makes SAM3
    grow to the WHOLE eyebrow object, even when MediaPipe's (human-trained) landmark
    extent sits low/inside the stylized art. A fixed band/envelope cannot do this.
  • Sibling negatives (eye points when segmenting brow, etc.) stop adjacent features
    from merging.

Model: wire a comfy-core SAM3 MODEL (the same one SAM3_Detect uses — load sam3.pt
with a standard model loader). This node uses model.model.diffusion_model.forward_segment
(the SAM decoder box+point path). comfyui-rmbg's Sam3Processor is NOT used (it has no
point-prompt path).

Output reconciliation = "trust SAM3, light bleed-guard": each SAM3 mask is intersected
with the feature's MediaPipe zone dilated by `bleed_guard` px. Large bleed_guard ⇒ SAM3
shape dominates (correct for offset features like brows); small ⇒ stays near MediaPipe.

MediaPipe still owns the landmark JSON for Blender UV lineup — this node produces the
pixel masks (RGBA + individual). Wire the RGBA/masks into your existing save nodes.

Single responsibility: this node SEGMENTS. Saving to disk / context is left to the
save nodes (BD_SaveBatch / BD MP Save Face Data).
"""

from __future__ import annotations

import os
import numpy as np
import torch
from comfy_api.latest import io

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import mediapipe as mp  # noqa: F401 — gate only
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

import comfy.model_management
import comfy.utils
import folder_paths as _folder_paths

from .face_mp_shared import (
    _init_mp_idx, detect_landmarks_robust, _masks_from_landmarks,
    _MP_IDX, _OUTER_LIP_IDX, _NOSE_INDICES, _subtract, _union, _blank,
    find_mediapipe_model as _find_mediapipe_model,
)


_MODEL_PATH = _find_mediapipe_model()
_SAM3_SIZE = 1008  # SAM3 works in a 1008×1008 preprocessed space
# Standalone VitMatte: load straight from the HF repo id — transformers auto-downloads
# to the HF cache (HF_HOME) if absent. No dependency on any other custom-node pack's
# model dir, so this node stands alone.
_VITMATTE_REPOS = {
    "small": "hustvl/vitmatte-small-composition-1k",
    "base":  "hustvl/vitmatte-base-composition-1k",
}
_VITMATTE = {}  # cache by variant: {variant: (model, proc, device)}


def _load_vitmatte(variant: str = "small"):
    """Lazy-load + cache VitMatte (transformers), auto-downloading from the HF hub on
    first use. Returns (model, proc, device); (None, None, None) on failure."""
    if variant in _VITMATTE:
        return _VITMATTE[variant]
    try:
        import torch as _t
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor
        repo = _VITMATTE_REPOS.get(variant, _VITMATTE_REPOS["small"])
        dev = "cuda" if _t.cuda.is_available() else "cpu"
        print(f"[BD MP SAM3] loading VitMatte '{variant}' ({repo}) — downloads to HF cache if missing", flush=True)
        proc = VitMatteImageProcessor.from_pretrained(repo)
        model = VitMatteForImageMatting.from_pretrained(repo).to(dev).eval()
        _VITMATTE[variant] = (model, proc, dev)
    except Exception as e:
        print(f"[BD MP SAM3] VitMatte load failed ({e})", flush=True)
        _VITMATTE[variant] = (None, None, None)
    return _VITMATTE[variant]


# ── Feature → (positive landmark source, sibling-negative sources) ──────────────
# Sources resolve to landmark-index lists at runtime: 'IDX:<key>' → _MP_IDX[key],
# 'LIPS' → _OUTER_LIP_IDX, 'NOSE' → _NOSE_INDICES.
_FEATURE_SPECS = {
    "left_brow":  ("IDX:left_brow",  ["IDX:left_eye"]),
    "right_brow": ("IDX:right_brow", ["IDX:right_eye"]),
    "left_eye":   ("IDX:left_eye",   ["IDX:left_brow"]),
    "right_eye":  ("IDX:right_eye",  ["IDX:right_brow"]),
    "lips":       ("LIPS",           ["NOSE"]),
    "nose":       ("NOSE",           ["LIPS", "IDX:left_eye", "IDX:right_eye"]),
}


def _resolve_idx(source: str) -> list[int]:
    if source == "LIPS":
        return list(_OUTER_LIP_IDX)
    if source == "NOSE":
        return list(_NOSE_INDICES)
    if source.startswith("IDX:"):
        return list(_MP_IDX.get(source[4:], []))
    return []


def _subsample(idx_list: list[int], k: int) -> list[int]:
    if len(idx_list) <= k:
        return idx_list
    step = len(idx_list) / float(k)
    return [idx_list[int(i * step)] for i in range(k)]


def _norm_pts(idx_list: list[int], lm) -> list[tuple[float, float]]:
    return [(float(lm[i].x), float(lm[i].y)) for i in idx_list if 0 <= i < len(lm)]


def _centroid(idx_list: list[int], lm) -> tuple[float, float] | None:
    pts = _norm_pts(idx_list, lm)
    if not pts:
        return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _norm_bbox(idx_list: list[int], lm, pad: float) -> tuple[float, float, float, float]:
    pts = _norm_pts(idx_list, lm)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0 = max(0.0, min(xs) - pad); y0 = max(0.0, min(ys) - pad)
    x1 = min(1.0, max(xs) + pad); y1 = min(1.0, max(ys) + pad)
    return (x0, y0, x1, y1)


def _fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    """Fill interior holes (e.g. the open mouth inside the lip ring) via border floodfill."""
    h, w = mask_u8.shape
    ff = mask_u8.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)  # flood background from a corner
    holes = cv2.bitwise_not(ff)                                          # interior pockets the flood couldn't reach
    return cv2.bitwise_or(mask_u8, holes)


def _refine_feature_mask(mask_u8: np.ndarray, rgb_u8: np.ndarray, mode: str,
                         radius: int, eps: float, threshold: float,
                         vitmatte_variant: str = "small") -> np.ndarray:
    """Snap the mask boundary to image color/edge transitions.

    mode='guided'  : cv2.ximgproc.guidedFilter — the RGB render guides an edge-aware
                     smoothing of the mask alpha, so the boundary follows real edges.
    mode='matting' : PyMatting closed-form alpha matting from an auto-trimap
                     (eroded mask = sure-fg, outside dilated mask = sure-bg, band =
                     unknown). Higher quality on soft/hair edges (eyebrows).

    Runs on the feature's ROI crop (bbox + margin) for speed; pastes back. Returns a
    re-binarized mask snapped to the refined alpha.
    """
    if mode == "off" or mask_u8.max() == 0:
        return mask_u8
    ys, xs = np.where(mask_u8 > 0)
    pad = max(radius * 3, 24)
    H, W = mask_u8.shape
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    m = mask_u8[y0:y1, x0:x1]
    g = np.ascontiguousarray(rgb_u8[y0:y1, x0:x1, :3])

    if mode == "guided":
        if not hasattr(cv2, "ximgproc"):
            print("[BD MP SAM3] guided refine unavailable (no cv2.ximgproc) — skipping", flush=True)
            return mask_u8
        guide = g.astype(np.float32) / 255.0
        src = (m.astype(np.float32) / 255.0)
        alpha = cv2.ximgproc.guidedFilter(guide, src, max(1, radius), float(eps))
        ref = (alpha > threshold).astype(np.uint8) * 255
    elif mode == "matting":
        try:
            from pymatting import estimate_alpha_cf
        except Exception as e:
            print(f"[BD MP SAM3] matting unavailable ({e}) — skipping", flush=True)
            return mask_u8
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        fg = cv2.erode(m, k)
        bg = cv2.dilate(m, k)
        trimap = np.full(m.shape, 0.5, dtype=np.float64)
        trimap[fg > 0] = 1.0
        trimap[bg == 0] = 0.0
        try:
            alpha = estimate_alpha_cf(g.astype(np.float64) / 255.0, trimap)
        except Exception as e:
            print(f"[BD MP SAM3] matting failed ({e}) — skipping", flush=True)
            return mask_u8
        ref = (alpha > threshold).astype(np.uint8) * 255
    elif mode == "vitmatte":
        import torch as _t
        from PIL import Image as _PIL
        model, proc, dev = _load_vitmatte(vitmatte_variant)
        if model is None:
            return mask_u8
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        fg = cv2.erode(m, k); bg = cv2.dilate(m, k)
        tri = np.full(m.shape, 128, np.uint8)   # unknown band
        tri[fg > 0] = 255                        # sure foreground
        tri[bg == 0] = 0                         # sure background
        try:
            inputs = proc(images=_PIL.fromarray(g), trimaps=_PIL.fromarray(tri), return_tensors="pt")
            inputs = {kk: vv.to(dev) for kk, vv in inputs.items()}
            with _t.no_grad():
                alpha = model(**inputs).alphas[0, 0].float().cpu().numpy()
            alpha = alpha[:m.shape[0], :m.shape[1]]   # processor pads to /32 — crop back
        except Exception as e:
            print(f"[BD MP SAM3] vitmatte failed ({e}) — skipping", flush=True)
            return mask_u8
        ref = (alpha > threshold).astype(np.uint8) * 255
    else:
        return mask_u8

    out = mask_u8.copy()
    out[y0:y1, x0:x1] = ref
    return out


def _clean_feature_mask(mask_u8: np.ndarray, pos_px: list[tuple[float, float]],
                        smooth_px: int = 3, fill: bool = True) -> np.ndarray:
    """Remove SAM3 noise: keep only the connected component(s) a positive seed point
    lands in (drops stray non-contiguous chunks around the feature), smooth the edge,
    and optionally fill interior holes (solid lips).
    """
    m = (mask_u8 > 0).astype(np.uint8)
    if m.sum() == 0:
        return mask_u8
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask_u8
    keep = set()
    H, W = lbl.shape
    for (px, py) in pos_px:
        xi, yi = int(round(px)), int(round(py))
        if 0 <= yi < H and 0 <= xi < W and lbl[yi, xi] > 0:
            keep.add(int(lbl[yi, xi]))
    if not keep:  # no seed landed on a component — fall back to the largest
        keep = {1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))}
    out = (np.isin(lbl, list(keep))).astype(np.uint8) * 255
    if smooth_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * smooth_px + 1, 2 * smooth_px + 1))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)   # seal small gaps / jagged edge
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)    # shave specks
    if fill:
        out = _fill_holes(out)
    return out


class BD_MediaPipeSAM3FaceSegment(io.ComfyNode):
    """
    MediaPipe-guided SAM3 face feature segmentation.

    One pass: MediaPipe localizes each feature → SAM3 segments it with a box + point
    prompt seeded from the landmarks → pixel-accurate masks. Outputs an RGBA zone map
    (R=lips, G=brows, B=eyes, A=face_oval) plus individual feature masks compatible
    with BD MP Face Mask / Face Refine wiring.

    Wire a comfy-core SAM3 MODEL (the SAM3_Detect model). MediaPipe still owns the
    Blender landmark JSON (run BD MP Face Export for that).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_MediaPipeSAM3FaceSegment",
            display_name="BD MP SAM3 Face Segment",
            category="🧠BrainDead/Segmentation",
            description=(
                "MediaPipe-guided SAM3 face segmentation: localizes each feature with "
                "MediaPipe, then prompts SAM3 per feature (box + positive landmark points "
                "+ sibling negatives) for pixel-accurate masks. Solves the stylized-brow "
                "offset that landmark-only bands can't. Outputs RGBA zone map + individual "
                "masks. Wire a comfy-core SAM3 MODEL (same as SAM3 Detect)."
            ),
            inputs=[
                io.Model.Input("model", optional=True,
                               tooltip="Comfy-core SAM3 model (override). Leave UNWIRED to auto-load + "
                                       "auto-download the official SAM3 checkpoint in-house (bd_sam3) — no setup."),
                io.Image.Input("image", tooltip="Full-color full-resolution face image. Only image[0] is used."),
                io.Combo.Input("angle", options=["front", "side_left", "side_right"], default="front",
                               optional=True, tooltip="Stored for downstream/context use only."),
                io.Boolean.Input("do_brows", default=True, optional=True,
                                 tooltip="Segment eyebrows with SAM3 (recommended — fixes the offset)."),
                io.Boolean.Input("do_eyes", default=True, optional=True, tooltip="Segment eyes with SAM3."),
                io.Boolean.Input("do_lips", default=True, optional=True, tooltip="Segment lips with SAM3."),
                io.Boolean.Input("do_nose", default=True, optional=True, tooltip="Segment nose with SAM3."),
                io.Boolean.Input("do_ears", default=False, optional=True,
                                 tooltip="Segment ears with TEXT-grounded SAM3 ('ear', split L/R at the face centre) "
                                         "instead of the weak MediaPipe approximation (the face mesh has no outer-ear "
                                         "points). Off = MediaPipe ears. Loads the SAM3 text encoder in-house."),
                io.Float.Input("detection_confidence", default=0.3, min=0.1, max=1.0, step=0.05, optional=True,
                               tooltip="MediaPipe face detection confidence. 0.3 works for stylized renders."),
                io.Float.Input("min_face_span", default=0.35, min=0.0, max=1.0, step=0.05, optional=True,
                               tooltip="Tiny-detection guard (see BD MP Face Export). 0 disables."),
                io.Float.Input("mask_threshold", default=0.5, min=0.05, max=0.95, step=0.05, optional=True,
                               tooltip="SAM3 mask probability cutoff."),
                io.Int.Input("refine_iterations", default=1, min=0, max=5, optional=True,
                             tooltip="SAM decoder refinement passes. NOTE: extra passes tend to SHRINK masks on "
                                     "stylized/non-photo renders — 1 (no mask-refine) gives the fullest feature; "
                                     "raise only if SAM3 over-segments."),
                io.Int.Input("bleed_guard", default=48, min=0, max=200, step=2, optional=True,
                             tooltip="GENERAL bleed-guard: dilate the MediaPipe feature zone by this many px (at "
                                     "native res), then clip SAM3's mask to it. Used for NOSE + as the fallback. "
                                     "Brows/eyes/lips have their own (brow_/eye_/lips_bleed_guard). "
                                     "0 = clip exactly to MediaPipe."),
                io.Int.Input("brow_bleed_guard", default=44, min=0, max=200, step=2, optional=True,
                             tooltip="Bleed-guard for BROWS only. Brows need a LARGE guard (~40-45): their MediaPipe "
                                     "landmarks sit inside/below the painted brow, so SAM3 must grow out to fill it. "
                                     "Raise if brows read thin; this won't loosen eyes/lips (they're separate)."),
                io.Int.Input("lips_bleed_guard", default=12, min=0, max=200, step=2, optional=True,
                             tooltip="Bleed-guard for LIPS only — kept separate because brows need a LARGE guard "
                                     "(40-45) to fill while lips need a SMALL one or they overflow onto the face. "
                                     "Lower = tighter lips (12 hugs the lip contour; 0 = clip exactly to MediaPipe)."),
                io.Int.Input("eye_bleed_guard", default=10, min=0, max=200, step=2, optional=True,
                             tooltip="Bleed-guard for EYES only. SAM3 grows to the whole eye (lid/lashes/sclera); a "
                                     "small guard clips it back toward the eyelid aperture so the mask hugs the "
                                     "eyeball — like BD Face Infill's eroded eyelid hull. Lower = tighter "
                                     "(0 = clip exactly to the MediaPipe eye contour)."),
                io.Boolean.Input("cleanup", default=True, optional=True,
                                 tooltip="Clean SAM3 noise: keep only the connected component(s) the positive "
                                         "landmark seeds land in (drops stray chunks around eyes/lips)."),
                io.Boolean.Input("fill_holes", default=True, optional=True,
                                 tooltip="Fill interior holes on NON-lip features (eyes/nose/etc.) so each is solid. "
                                         "Lips are controlled separately by lips_mode."),
                io.Combo.Input("lips_mode", options=["mouth", "lips_only"], default="mouth", optional=True,
                               tooltip="Lips-specific:\n"
                                       "  mouth     — fill the whole mouth area (lips + teeth + tongue) into the "
                                       "lips mask (default; what the pipeline wants).\n"
                                       "  lips_only — lip flesh only; color-aware edge_refine excludes teeth/tongue.\n"
                                       "Overrides fill_holes for the lips feature."),
                io.Int.Input("edge_smooth", default=3, min=0, max=15, step=1, optional=True,
                             tooltip="Morphological close+open radius (px @native) to smooth jagged SAM3 edges "
                                     "during cleanup. 0 = no smoothing."),
                io.Combo.Input("edge_refine", options=["off", "guided", "matting", "vitmatte"], default="off",
                               optional=True,
                               tooltip="Snap the mask edge to image color/edges after cleanup.\n"
                                       "  off      — no refinement\n"
                                       "  guided   — guided filter (fast, edge-aware; good general snap)\n"
                                       "  matting  — PyMatting closed-form alpha matting (CPU, no model)\n"
                                       "  vitmatte — VitMatte deep matting model (best on soft/hair edges; "
                                       "GPU; loads hustvl vitmatte-small). All run on the feature ROI crop."),
                io.Int.Input("refine_radius", default=8, min=1, max=40, step=1, optional=True,
                             tooltip="Guided-filter radius / matting trimap band width (px @native). Larger = "
                                     "looks further for an edge / wider uncertain band."),
                io.Float.Input("refine_eps", default=1e-4, min=1e-6, max=1e-1, step=1e-4, optional=True,
                               tooltip="Guided-filter edge sensitivity (smaller = sharper, hugs edges harder). "
                                       "Ignored by matting."),
                io.Float.Input("refine_threshold", default=0.5, min=0.05, max=0.95, step=0.05, optional=True,
                               tooltip="Binarize the refined alpha at this level."),
                io.Combo.Input("vitmatte_model", options=["small", "base"], default="small", optional=True,
                               tooltip="VitMatte variant for edge_refine='vitmatte'. Auto-downloaded from the HF "
                                       "hub (hustvl/vitmatte-{small,base}-composition-1k) on first use — standalone, "
                                       "no other node pack required. base = higher quality, larger."),
                io.Mask.Input("silhouette_mask", optional=True,
                              tooltip="Optional head silhouette (white=head). When wired, ALL outputs are clipped "
                                      "to it and it becomes head_mask; masked_skin = skin within it. Overrides "
                                      "remove_background. (Or let remove_background compute it in-house.)"),
                io.Mask.Input("head_mask", optional=True,
                              tooltip="Optional inner head/face-plate mask used as the skin base (skin = head_mask "
                                      "− eyes − brows − lips). Falls back to face_oval. Echoed to the head_mask output."),
                io.Boolean.Input("remove_background", default=False, optional=True,
                                 tooltip="Compute a clean HEAD silhouette IN-HOUSE (text-grounded SAM3, no extra "
                                         "nodes): segment head_prompts as positive, exclude_prompts (neck/shirt/etc.) "
                                         "as negative → head minus neck/clothing/background. Clips every output to it, "
                                         "becomes head_mask, and is emitted as the `silhouette` output. Skipped if "
                                         "silhouette_mask is wired. Auto-downloads the SAM3 checkpoint if needed."),
                io.String.Input("head_prompts", default="head\nface\nhair\near", multiline=True, optional=True,
                                tooltip="remove_background positives — one per line. Union = the head."),
                io.String.Input("exclude_prompts", default="neck\nshirt\nclothing\nshoulder", multiline=True, optional=True,
                                tooltip="remove_background negatives — one per line. Their union is SUBTRACTED from the "
                                        "head (peels off neck / clothing / body)."),
                io.Boolean.Input("neck_cut", default=True, optional=True,
                                 tooltip="With remove_background: remove the neck. PRIMARY = in-house SegFormer face "
                                         "parser (1038lab/segformer_face, auto-downloaded) — subtracts its real "
                                         "Neck/Clothing/Necklace classes (per-pixel, reliable). FALLBACK if it can't "
                                         "load = cut below the MediaPipe jawline contour. Off = keep the neck."),
                io.Combo.Input("cutout_bg", options=["transparent", "black", "white"], default="transparent", optional=True,
                               tooltip="Background for the head_cutout / head_cutout_clean image outputs: transparent "
                                       "(RGBA alpha), or composited on black / white."),
                io.Boolean.Input("drop_eyes", default=True, optional=True,
                                 tooltip="head_cutout_clean: remove the eyes (punch-out — fill is left to BD MP Face Infill)."),
                io.Boolean.Input("drop_brows", default=True, optional=True,
                                 tooltip="head_cutout_clean: remove the brows."),
                io.Boolean.Input("drop_lips", default=True, optional=True,
                                 tooltip="head_cutout_clean: remove the lips/mouth."),
                io.Boolean.Input("drop_nose", default=False, optional=True,
                                 tooltip="head_cutout_clean: remove the nose (off by default — nose is usually kept skin)."),
            ],
            outputs=[
                io.Image.Output(display_name="rgba",
                                tooltip="RGBA zone map: R=lips, G=brows, B=eyes, A=face_oval."),
                io.Mask.Output(display_name="face_oval"),
                io.Mask.Output(display_name="skin"),
                io.Mask.Output(display_name="left_eye"),
                io.Mask.Output(display_name="right_eye"),
                io.Mask.Output(display_name="eyes"),
                io.Mask.Output(display_name="left_brow"),
                io.Mask.Output(display_name="right_brow"),
                io.Mask.Output(display_name="brows"),
                io.Mask.Output(display_name="left_iris"),
                io.Mask.Output(display_name="right_iris"),
                io.Mask.Output(display_name="irises"),
                io.Mask.Output(display_name="lips"),
                io.Mask.Output(display_name="nose"),
                io.Mask.Output(display_name="left_ear"),
                io.Mask.Output(display_name="right_ear"),
                io.Mask.Output(display_name="ears"),
                io.Mask.Output(display_name="forehead"),
                io.Mask.Output(display_name="hair"),
                io.Mask.Output(display_name="head_mask"),
                io.Mask.Output(display_name="masked_skin"),
                io.Mask.Output(display_name="silhouette",
                               tooltip="The head silhouette used for clipping (head − neck/clothing/bg). When "
                                       "remove_background is on it's the in-house text-SAM3 head cutout; else the "
                                       "wired silhouette_mask, else face_oval. Wire this anywhere you need 'head "
                                       "but not neck' — replaces a separate background-removal + silhouette chain."),
                io.Image.Output(display_name="debug_overlay",
                                tooltip="Render with feature masks tinted (lips=R, brows=G, eyes=B, nose=Y)."),
                io.String.Output(display_name="status"),
                io.Image.Output(display_name="head_cutout",
                                tooltip="The input image with the head silhouette applied, on cutout_bg "
                                        "(transparent RGBA / black / white) — head only, background+neck removed."),
                io.Image.Output(display_name="head_cutout_clean",
                                tooltip="Same as head_cutout but with the drop_* features (eyes/brows/lips/nose) "
                                        "punched out (holes empty / bg-filled). Wire into BD MP Face Infill to fill "
                                        "them, or use as the skin plate."),
            ],
        )

    @classmethod
    def execute(cls, image, model=None, angle="front", do_brows=True, do_eyes=True, do_lips=True,
                do_nose=True, do_ears=False, detection_confidence=0.3, min_face_span=0.35, mask_threshold=0.5,
                refine_iterations=1, bleed_guard=48, brow_bleed_guard=44, lips_bleed_guard=12, eye_bleed_guard=10,
                remove_background=False, head_prompts="head\nface\nhair\near", exclude_prompts="neck\nshirt\nclothing\nshoulder",
                neck_cut=True, cutout_bg="transparent", drop_eyes=True, drop_brows=True,
                drop_lips=True, drop_nose=False, cleanup=True, fill_holes=True, lips_mode="mouth",
                edge_smooth=3, edge_refine="off", refine_radius=8, refine_eps=1e-4, refine_threshold=0.5,
                vitmatte_model="small", silhouette_mask=None, head_mask=None) -> io.NodeOutput:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        B, H, W, C = image.shape

        def _m(np_u8):
            return torch.from_numpy(np_u8.astype(np.float32) / 255.0)

        def _img(np_u8_or_f):
            arr = np_u8_or_f.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            return torch.from_numpy(arr).unsqueeze(0)

        def _mask_in(t):
            """MASK tensor → np uint8 (H,W), resized to (H,W) if needed."""
            if t is None:
                return None
            a = t.detach().cpu().float().numpy()
            if a.ndim == 3:
                a = a[0]
            u = (a > 0.5).astype(np.uint8) * 255
            if u.shape != (H, W):
                u = cv2.resize(u, (W, H), interpolation=cv2.INTER_NEAREST)
            return u

        def _bail(status):
            z = _blank(H, W)
            rgba = torch.from_numpy(np.zeros((H, W, 4), np.float32)).unsqueeze(0)
            dbg = torch.from_numpy(np.zeros((H, W, 3), np.float32)).unsqueeze(0)
            blank_rgba = torch.from_numpy(np.zeros((H, W, 4), np.float32)).unsqueeze(0)
            return io.NodeOutput(rgba, *([_m(z)] * 21), dbg, status, blank_rgba, blank_rgba)

        if not HAS_MEDIAPIPE or not HAS_CV2:
            return _bail("missing mediapipe/opencv — no segmentation")

        # ── MediaPipe detection (with tiny-detection guard) ───────────────────
        _init_mp_idx()
        frame_np = image[0].detach().cpu().float().numpy()
        np_img = (frame_np[..., :3] * 255.0).clip(0, 255).astype(np.uint8)
        try:
            lm, det = detect_landmarks_robust(np_img, _MODEL_PATH,
                                              min_conf=detection_confidence, min_span=min_face_span)
        except Exception as e:
            return _bail(f"MediaPipe error: {e}")
        if lm is None:
            return _bail("no face detected")

        # MediaPipe zones (face_oval + per-feature) for bleed-guard + face_oval/skin/irises.
        # ear_expand>0 so the ear masks aren't empty (the ear-oval landmarks are ~collinear
        # face-edge points; at expand 0 their convex hull collapses to nothing).
        mp_masks = _masks_from_landmarks(lm, H, W, face_expand=0, feature_expand=0,
                                         iris_expand=0, ear_expand=25, hair_expand=0,
                                         tight_features=True)

        # ── SAM3 setup (mirrors comfy_extras/nodes_sam3.SAM3_Detect) ──────────
        if model is None:   # no MODEL wired — auto-load + auto-download in-house (no setup)
            from . import bd_sam3
            model, _ = bd_sam3.load_sam3(need_clip=False)
        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        sam3 = model.model.diffusion_model
        frame = comfy.utils.common_upscale(
            image[0:1, ..., :3].movedim(-1, 1), _SAM3_SIZE, _SAM3_SIZE, "bilinear", crop="disabled"
        ).to(device=device, dtype=dtype)

        scale = max(H, W) / 1536.0
        guard_px = max(0, int(round(bleed_guard * scale)))   # general guard (status display)

        def _feature_guard(out_key: str):
            """Per-feature bleed-guard. Brows need a LARGE guard (their MediaPipe landmarks
            sit inside the painted brow, so SAM3 must grow out to fill); lips and eyes need
            a SMALL one (else SAM3 grows past the lip/eyelid aperture onto the face/skin).
            Returns (guard_px, dilate_kernel|None)."""
            if out_key == "lips":
                g = lips_bleed_guard
            elif out_key in ("left_eye", "right_eye"):
                g = eye_bleed_guard
            elif out_key in ("left_brow", "right_brow"):
                g = brow_bleed_guard
            else:
                g = bleed_guard   # nose + general fallback
            gpx = max(0, int(round(g * scale)))
            k = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gpx + 1, 2 * gpx + 1))
                 if gpx > 0 and HAS_CV2 else None)
            return gpx, k

        def _sam3_feature(out_key: str) -> np.ndarray:
            pos_src, neg_srcs = _FEATURE_SPECS[out_key]
            pos_pts = _norm_pts(_subsample(_resolve_idx(pos_src), 8), lm)
            if not pos_pts:
                return _blank(H, W)
            # Per-feature hole-fill: lips use lips_mode (mouth=fill, lips_only=no fill);
            # all other features use the general fill_holes toggle.
            feat_fill = (lips_mode == "mouth") if out_key == "lips" else fill_holes
            neg_pts = []
            for ns in neg_srcs:
                c = _centroid(_resolve_idx(ns), lm)
                if c is not None:
                    neg_pts.append(c)
            box = _norm_bbox(_resolve_idx(pos_src), lm, pad=0.01)

            coords = ([[x * _SAM3_SIZE, y * _SAM3_SIZE] for (x, y) in pos_pts]
                      + [[x * _SAM3_SIZE, y * _SAM3_SIZE] for (x, y) in neg_pts])
            labels = [1] * len(pos_pts) + [0] * len(neg_pts)
            point_inputs = {
                "point_coords": torch.tensor([coords], dtype=dtype, device=device),
                "point_labels": torch.tensor([labels], dtype=torch.int32, device=device),
            }
            x0, y0, x1, y1 = box
            box_inputs = torch.tensor([[[x0 * _SAM3_SIZE, y0 * _SAM3_SIZE],
                                        [x1 * _SAM3_SIZE, y1 * _SAM3_SIZE]]], dtype=dtype, device=device)
            def _logit_to_mask(logit):
                mm = torch.nn.functional.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
                return (torch.sigmoid(mm[0, 0]) > mask_threshold).detach().cpu().numpy().astype(np.uint8) * 255

            # The SAM-decoder mask_inputs loop sharpens brows/eyes but destabilizes lips
            # (shrinks toward the lip-flesh and can vanish). Lips always use a single pass.
            feat_iters = 1 if out_key == "lips" else refine_iterations
            try:
                base = sam3.forward_segment(frame, point_inputs=point_inputs, box_inputs=box_inputs)
                ml = base
                for _ in range(max(0, feat_iters - 1)):
                    ml = sam3.forward_segment(frame, mask_inputs=ml)
            except Exception as e:
                print(f"[BD MP SAM3] WARNING: forward_segment failed for {out_key}: {e}", flush=True)
                return _blank(H, W)
            sam = _logit_to_mask(ml)
            # Collapse-guard: if the mask_inputs loop lost most of the base detection on
            # stylized art, revert to the base (single-pass) mask.
            if feat_iters > 1:
                base_m = _logit_to_mask(base)
                if (sam > 0).sum() < 0.3 * max(1, int((base_m > 0).sum())):
                    print(f"[BD MP SAM3] {out_key}: refine_iterations collapsed the mask — reverting to base", flush=True)
                    sam = base_m

            # Light bleed-guard: clip SAM3 to the MediaPipe zone dilated by guard_px.
            # Large guard ⇒ SAM3's shape dominates (correct for offset features like brows).
            feat_guard_px, feat_kernel = _feature_guard(out_key)
            mp_zone = mp_masks.get(out_key)
            if mp_zone is not None and feat_kernel is not None:
                sam = np.minimum(sam, cv2.dilate(mp_zone, feat_kernel))
            elif mp_zone is not None and feat_guard_px == 0:
                sam = np.minimum(sam, mp_zone)

            # Cleanup: drop stray non-contiguous chunks (keep seeded component) + fill holes.
            if cleanup:
                pos_px = [(x * W, y * H) for (x, y) in pos_pts]
                # NB: fill=False here. Hole-fill must happen AFTER edge_refine, never before —
                # refining a pre-filled solid blob bulges its outer edge onto the skin. We snap
                # the tight outer lip edge first, then fill the enclosed interior.
                sam = _clean_feature_mask(sam, pos_px,
                                          smooth_px=max(0, int(round(edge_smooth * scale))),
                                          fill=False)
            # Edge-snap refinement (guided filter / alpha matting) to follow image color/edges.
            if edge_refine != "off":
                sam = _refine_feature_mask(sam, np_img, edge_refine,
                                           radius=max(1, int(round(refine_radius * scale))),
                                           eps=refine_eps, threshold=refine_threshold,
                                           vitmatte_variant=vitmatte_model)
            # Re-fill AFTER refinement: color-aware refine (vitmatte/matting/guided) excludes
            # teeth/tongue/mouth-interior (different colour from the lip flesh), which would
            # leave lips-only. With fill_holes on (default) we re-close those interior holes so
            # the WHOLE mouth area is masked. lips_only / fill off → keep the lip-flesh result.
            if feat_fill:
                sam = _fill_holes(sam)
            return sam

        # ── Text-grounded SAM3 helper (shared by remove_background + do_ears) ──
        # Lazy-loads SAM3's text encoder in-house (bd_sam3) and segments a word prompt.
        _txt = {}

        def _text_mask(prompt: str) -> np.ndarray:
            if "clip" not in _txt:
                from . import bd_sam3
                _txt["model"], _txt["clip"] = bd_sam3.load_sam3(need_clip=True)
            from . import bd_sam3
            t = bd_sam3.segment_text(_txt["model"], _txt["clip"], image, prompt, threshold=mask_threshold)
            m = (t[0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            return m

        # ── Per-feature segmentation ──────────────────────────────────────────
        left_eye  = _sam3_feature("left_eye")  if do_eyes  else _blank(H, W)
        right_eye = _sam3_feature("right_eye") if do_eyes  else _blank(H, W)
        left_brow = _sam3_feature("left_brow") if do_brows else _blank(H, W)
        right_brow= _sam3_feature("right_brow")if do_brows else _blank(H, W)
        lips      = _sam3_feature("lips")      if do_lips  else _blank(H, W)
        nose      = _sam3_feature("nose")      if do_nose  else _blank(H, W)

        # Fall back to the MediaPipe zone for any disabled feature (so skin subtraction
        # and the RGBA map stay sensible).
        if not do_eyes:
            left_eye, right_eye = mp_masks["left_eye"], mp_masks["right_eye"]
        if not do_brows:
            left_brow, right_brow = mp_masks["left_brow"], mp_masks["right_brow"]
        if not do_lips:
            lips = mp_masks["lips"]
        if not do_nose:
            nose = mp_masks["nose"]

        eyes  = _union(left_eye, right_eye)
        brows = _union(left_brow, right_brow)
        face_oval = mp_masks["face_oval"]
        # MediaPipe-only regions (not SAM3-segmented) — kept for a uniform region set
        # matching BD MP Face Mask / Save Face Data.
        left_iris  = mp_masks["left_iris"]
        right_iris = mp_masks["right_iris"]
        irises     = mp_masks["irises"]
        left_ear   = mp_masks["left_ear"]
        right_ear  = mp_masks["right_ear"]
        ears       = mp_masks["ears"]
        forehead   = mp_masks["forehead"]
        hair       = mp_masks["hair"]

        # do_ears: replace the (weak) MediaPipe ear approximation with a text-grounded SAM3
        # "ear" segmentation, split into left/right at the face-oval centre. (The 478-pt face
        # mesh has no outer-ear points, so SAM3 is the only way to get the actual ear.)
        if do_ears:
            try:
                ear_all = _text_mask("ear")
                cols = np.any(face_oval > 0, axis=0)
                if cols.any():
                    xs = np.where(cols)[0]
                    face_cx = int((xs[0] + xs[-1]) / 2)
                else:
                    face_cx = W // 2
                lclip = _blank(H, W); lclip[:, :face_cx] = 255
                rclip = _blank(H, W); rclip[:, face_cx:] = 255
                left_ear = np.minimum(ear_all, lclip)
                right_ear = np.minimum(ear_all, rclip)
                ears = _union(left_ear, right_ear)
                print(f"[BD MP SAM3] do_ears: SAM3 ear {100.0 * (ears > 0).mean():.2f}%", flush=True)
            except Exception as e:
                print(f"[BD MP SAM3] do_ears failed ({e}) — keeping MediaPipe ears", flush=True)

        # head_mask: explicit input > silhouette > face_oval. Used as the skin base.
        sil = _mask_in(silhouette_mask)

        # In-house background/neck removal: text-grounded SAM3 head silhouette (no extra
        # nodes). head_prompts (positive) ∪ minus exclude_prompts (neck/clothing/bg).
        if sil is None and remove_background:
            try:
                pos_lines = [p.strip() for p in (head_prompts or "").split("\n") if p.strip()]
                neg_lines = [p.strip() for p in (exclude_prompts or "").split("\n") if p.strip()]

                def _text_union(lines):
                    acc = _blank(H, W)
                    for p in lines:
                        acc = np.maximum(acc, _text_mask(p))
                    return acc

                pos = _text_union(pos_lines)
                neg = _text_union(neg_lines)
                head_sil = (pos.astype(bool) & ~neg.astype(bool)).astype(np.uint8) * 255

                # Neck removal — STACKED for robustness (face-parse alone under-detects the
                # neck on stylized heads, leaving the bulk below its thin band):
                #   1. in-house SegFormer face-parser (1038lab/segformer_face): subtract the
                #      real Neck/Clothing/Necklace classes (catches neck/collar above the jaw).
                #   2. jawline cut: below the MediaPipe jaw contour, INTERPOLATED across the
                #      full width (np.interp) so the neck is cut in side columns too, not only
                #      under the face — a contour, not a flat line.
                #   3. keep the head's connected component: drops the now-disconnected neck/
                #      shoulder remnant under the cut (the "clean up the rest under it" step).
                cut_info = ""
                if neck_cut:
                    parts = []
                    nc_sub = nc_loc = None
                    try:
                        from . import bd_face_parse
                        pred = bd_face_parse.parse(image)
                        nc_sub = bd_face_parse.class_mask(pred, ["Neck", "Clothing", "Necklace"])
                        nc_loc = bd_face_parse.class_mask(pred, ["Neck"])           # boundary locator
                        if nc_sub.shape != head_sil.shape:
                            nc_sub = cv2.resize(nc_sub, (W, H), interpolation=cv2.INTER_NEAREST)
                            nc_loc = cv2.resize(nc_loc, (W, H), interpolation=cv2.INTER_NEAREST)
                    except Exception as e:
                        print(f"[BD MP SAM3] face-parser unavailable ({e})", flush=True)
                    if nc_sub is not None and (nc_sub > 0).any():
                        head_sil[nc_sub > 0] = 0                                   # remove detected neck/clothing
                        parts.append(f"faceparse{100.0 * (nc_sub > 0).mean():.0f}%")
                        # The detected NECK band marks the jaw/neck boundary. Cut everything below
                        # its TOP edge (per column, interpolated across the width) — removes the
                        # rest of the neck the model mislabelled as skin, WITHOUT MediaPipe's
                        # too-high chin (which ate the jaw/ear). Skin above the band = kept.
                        loc = nc_loc if (nc_loc is not None and (nc_loc > 0).any()) else nc_sub
                        lb = loc > 0
                        lcols = np.where(np.any(lb, axis=0))[0]
                        if len(lcols):
                            neck_top = np.argmax(lb, axis=0).astype(np.float64)    # first neck row per col
                            top_full = np.interp(np.arange(W), lcols, neck_top[lcols])
                            head_sil[np.arange(H)[:, None] >= top_full[None, :]] = 0
                            parts.append("below-neck")
                        # keep the head connected component — drop the disconnected neck/shoulder remnant
                        nlbl, lab, stats, _ = cv2.connectedComponentsWithStats((head_sil > 0).astype(np.uint8), 8)
                        if nlbl > 1:
                            ys2, xs2 = np.where(mp_masks["face_oval"] > 0)
                            seed = 0
                            if len(xs2):
                                cy, cx = int(ys2.mean()), int(xs2.mean())
                                if 0 <= cy < H and 0 <= cx < W and head_sil[cy, cx] > 0:
                                    seed = int(lab[cy, cx])
                            if seed == 0:
                                seed = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                            head_sil = (lab == seed).astype(np.uint8) * 255
                            parts.append("head-cc")
                    else:
                        parts.append("no-neck-detected")   # leave the silhouette (don't risk eating the jaw)
                    cut_info = " | neck:" + "+".join(parts) if parts else ""

                if head_sil.any():
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                    head_sil = cv2.morphologyEx(head_sil, cv2.MORPH_CLOSE, k)
                    head_sil = _fill_holes(head_sil)
                    sil = head_sil
                    print(f"[BD MP SAM3] remove_background: head silhouette {100.0 * (sil > 0).mean():.1f}% "
                          f"(+{len(pos_lines)} -{len(neg_lines)} prompts){cut_info}", flush=True)
                else:
                    print("[BD MP SAM3] remove_background: empty head silhouette — skipping", flush=True)
            except Exception as e:
                print(f"[BD MP SAM3] remove_background failed ({e}) — skipping", flush=True)

        hm_in = _mask_in(head_mask)
        head_out = hm_in if hm_in is not None else (sil if sil is not None else face_oval)
        skin = _subtract(head_out, eyes, brows, lips)

        # Optional silhouette clip on every output.
        all_keys = ["face_oval", "skin", "left_eye", "right_eye", "eyes", "left_brow", "right_brow",
                    "brows", "left_iris", "right_iris", "irises", "lips", "nose", "left_ear",
                    "right_ear", "ears", "forehead", "hair"]
        local = dict(face_oval=face_oval, skin=skin, left_eye=left_eye, right_eye=right_eye, eyes=eyes,
                     left_brow=left_brow, right_brow=right_brow, brows=brows, left_iris=left_iris,
                     right_iris=right_iris, irises=irises, lips=lips, nose=nose, left_ear=left_ear,
                     right_ear=right_ear, ears=ears, forehead=forehead, hair=hair)
        if sil is not None:
            for k in all_keys:
                local[k] = np.minimum(local[k], sil)
            head_out = sil
        masked_skin = local["skin"]
        # silhouette output: the head cutout used for clipping (head − neck/clothing/bg when
        # remove_background; else the wired silhouette; else the face oval).
        silhouette_out = sil if sil is not None else face_oval

        rgba_np = np.stack([local["lips"], local["brows"], local["eyes"], local["face_oval"]],
                           axis=-1).astype(np.float32) / 255.0
        rgba = torch.from_numpy(rgba_np).unsqueeze(0)

        # Debug overlay: preview every region. Dim everything OUTSIDE the head silhouette
        # (so the kept head / removed neck+bg is obvious), then tint each feature — incl.
        # ears, irises, hair, forehead — over the render.
        ov = np_img.astype(np.float32).copy()
        outside = silhouette_out == 0
        ov[outside] *= 0.30
        for msk, col in [(local["hair"], (160, 90, 0)), (local["forehead"], (90, 90, 90)),
                         (local["ears"], (0, 220, 220)), (local["irises"], (255, 0, 255)),
                         (local["nose"], (255, 255, 0)), (local["eyes"], (0, 80, 255)),
                         (local["brows"], (0, 255, 0)), (local["lips"], (255, 0, 0))]:
            sel = msk > 0
            for ch in range(3):
                ov[..., ch][sel] = 0.45 * ov[..., ch][sel] + 0.55 * col[ch]
        debug_overlay = _img(ov.clip(0, 255))

        # head_cutout(_clean): input image masked to the head, on cutout_bg. _clean also
        # punches out the drop_* features (fill is left to BD MP Face Infill).
        _rgb01 = np_img.astype(np.float32) / 255.0

        def _compose(mask_u8):
            a = (mask_u8 > 0).astype(np.float32)[..., None]
            if cutout_bg == "transparent":
                rgba = np.concatenate([_rgb01, a], axis=-1)            # straight alpha
            else:
                bgv = 0.0 if cutout_bg == "black" else 1.0
                comp = _rgb01 * a + bgv * (1.0 - a)
                rgba = np.concatenate([comp, np.ones_like(a)], axis=-1)
            return torch.from_numpy(rgba).unsqueeze(0)

        head_cutout = _compose(silhouette_out)
        clean = (silhouette_out > 0)
        if drop_eyes:
            clean &= ~(local["eyes"] > 0)
        if drop_brows:
            clean &= ~(local["brows"] > 0)
        if drop_lips:
            clean &= ~(local["lips"] > 0)
        if drop_nose:
            clean &= ~(local["nose"] > 0)
        head_cutout_clean = _compose(clean.astype(np.uint8) * 255)

        feats = [k for k, on in
                 [("brows", do_brows), ("eyes", do_eyes), ("lips", do_lips), ("nose", do_nose)] if on]
        lips_guard_px = max(0, int(round(lips_bleed_guard * scale)))
        eye_guard_px = max(0, int(round(eye_bleed_guard * scale)))
        status = (f"SAM3-guided: {', '.join(feats) or 'none'} | det {det.get('quality')} "
                  f"span={det.get('span')} | bleed_guard={guard_px}px (lips={lips_guard_px}px eyes={eye_guard_px}px)"
                  f"{' | silhouette-clipped' if sil is not None else ''}"
                  f"{' | refine=' + edge_refine if edge_refine != 'off' else ''}")
        print(f"[BD MP SAM3 Face Segment] {status}", flush=True)

        return io.NodeOutput(
            rgba, _m(local["face_oval"]), _m(local["skin"]), _m(local["left_eye"]),
            _m(local["right_eye"]), _m(local["eyes"]), _m(local["left_brow"]),
            _m(local["right_brow"]), _m(local["brows"]), _m(local["left_iris"]),
            _m(local["right_iris"]), _m(local["irises"]), _m(local["lips"]), _m(local["nose"]),
            _m(local["left_ear"]), _m(local["right_ear"]), _m(local["ears"]),
            _m(local["forehead"]), _m(local["hair"]), _m(head_out), _m(masked_skin),
            _m(silhouette_out), debug_overlay, status, head_cutout, head_cutout_clean,
        )


# ── Registration ────────────────────────────────────────────────────────────────

FACE_MP_SAM3_V3_NODES = [BD_MediaPipeSAM3FaceSegment]
FACE_MP_SAM3_NODES = {"BD_MediaPipeSAM3FaceSegment": BD_MediaPipeSAM3FaceSegment}
FACE_MP_SAM3_DISPLAY_NAMES = {"BD_MediaPipeSAM3FaceSegment": "BD MP SAM3 Face Segment"}
