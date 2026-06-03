"""
BD_MouthParts — separate an isolated mouth render into lips / teeth / tongue and
pack them into the Unreal viseme-atlas RGBA contract (R=lips, G=teeth, B=tongue,
A=POM depth).

Built for the PinkF1 lip-viseme atlas (ARTS-1): the source mouths are low-poly
renders on black — magenta/purple lips, white teeth, salmon/rose tongue — a clean
colour-separation problem. This node does that separation in ONE pass with no model,
replacing the SAM3×3 (lips/teeth/tongue) → FillMaskHoles×3 → MaskResolver → PackChannels
chain in LipViseme-Atlas-Packed.json.

Two engines (CLAUDE.md "tools stand alone" = no dependency on another node pack, NOT
"no models" — we may load any model we download ourselves):

  • engine='color' (default, no model) — pure HSV classification tuned on the seven
    PinkF1 visemes:
      teeth  = bright + desaturated (white)                  S<teeth_s_max, V>teeth_v_min
      tongue = warm hue, region-grown from a confident salmon core, gated to the eroded
               mouth INTERIOR so warm lip-corner highlights don't read as tongue
      lips   = saturated remainder (S floor drops the dark mouth cavity)

  • engine='sam3' (wire a comfy-core SAM3 MODEL) — the colour split becomes the PRIOR:
    each part's colour mask seeds SAM3 (bbox + interior positive points + the other two
    parts' centroids as negatives), SAM3 grows the true object boundary, then the result
    is clipped to the mouth foreground, component-cleaned (seeded), edge-refined and
    filled. This fixes what colour alone can't — e.g. the lit front of the tongue that
    reads as saturated magenta and leaks into lips. Same SAM3 path as BD MP SAM3 Face
    Segment; built in-house (no rmbg/other pack).

`edge_refine` (optional, both engines) snaps each mask edge to image colour/edges via the
shared guided/matting/vitmatte helpers — those load on demand and degrade gracefully.

A (POM depth) comes from upstream (Lotus2 depth → NormLuma → CenterLuma): wire it into
`pom`. With nothing wired, A = the mouth foreground (lips∪teeth∪tongue) so rgba_packed
is still usable.

Single responsibility: this node SEPARATES + PACKS. Saving to disk and depth/POM
derivation live in their own nodes.
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

import comfy.model_management
import comfy.utils

# Shared edge/cleanup helpers (single source — the same code BD MP SAM3 uses;
# built in-house, no other node pack required).
from .face_mp_sam3 import _fill_holes, _refine_feature_mask, _clean_feature_mask

_SAM3_SIZE = 1008  # SAM3 works in a 1008×1008 preprocessed space (matches SAM3 Detect)


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    """Keep the single largest 8-connected component of a boolean/uint8 mask."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    return lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def _seed_points(mask_u8: np.ndarray, k: int = 6):
    """Interior positive seeds + bbox from a colour-prior mask. Points are the most-
    interior pixels (distance-transform peak band) so SAM3 grows from inside the part,
    not its noisy edge. Returns (list[(x,y) px], (x0,y0,x1,y1) px) — ([], None) if empty.
    """
    m = (mask_u8 > 0).astype(np.uint8)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return [], None
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    thr = 0.5 * float(dt.max())
    iy, ix = np.where(dt >= max(thr, 1.0))
    if len(ix) == 0:
        iy, ix = ys, xs
    sel = np.linspace(0, len(ix) - 1, min(k, len(ix))).astype(int)
    pts = [(int(ix[i]), int(iy[i])) for i in sel]
    return pts, box


def _centroid_px(mask_u8: np.ndarray):
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def _sam3_segment(sam3, frame, H, W, device, dtype, pos_px, neg_px, box_px,
                  mask_threshold: float, iters: int) -> np.ndarray:
    """One SAM3 forward_segment seeded by colour-prior box + positive/negative points.
    Coords map pixel-space → the 1008² SAM3 space (x/W·SIZE, y/H·SIZE)."""
    sx, sy = _SAM3_SIZE / float(W), _SAM3_SIZE / float(H)
    coords = ([[x * sx, y * sy] for (x, y) in pos_px]
              + [[x * sx, y * sy] for (x, y) in neg_px])
    labels = [1] * len(pos_px) + [0] * len(neg_px)
    point_inputs = {
        "point_coords": torch.tensor([coords], dtype=dtype, device=device),
        "point_labels": torch.tensor([labels], dtype=torch.int32, device=device),
    }
    x0, y0, x1, y1 = box_px
    box_inputs = torch.tensor([[[x0 * sx, y0 * sy], [x1 * sx, y1 * sy]]], dtype=dtype, device=device)

    def _to_mask(logit):
        mm = torch.nn.functional.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        return (torch.sigmoid(mm[0, 0]) > mask_threshold).detach().cpu().numpy().astype(np.uint8) * 255

    base = sam3.forward_segment(frame, point_inputs=point_inputs, box_inputs=box_inputs)
    ml = base
    for _ in range(max(0, iters - 1)):
        ml = sam3.forward_segment(frame, mask_inputs=ml)
    sam = _to_mask(ml)
    # Collapse-guard: extra mask_inputs passes can over-shrink small parts on stylized art.
    if iters > 1:
        base_m = _to_mask(base)
        if (sam > 0).sum() < 0.3 * max(1, int((base_m > 0).sum())):
            sam = base_m
    return sam


def _split_mouth(bgr: np.ndarray, *, bg_v_min: int, teeth_s_max: int, teeth_v_min: int,
                 tongue_h_lo: int, tongue_h_hi: int, tongue_s_min: int, tongue_s_max: int,
                 tongue_v_min: int, lips_s_min: int, interior_frac: float,
                 edge_smooth: int):
    """HSV classification of an isolated mouth render → (lips, teeth, tongue) bool masks.

    Tuned on the seven PinkF1 visemes. See module docstring for the rules.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.float32) * 2.0      # OpenCV packs hue 0-179 → scale to 0-360°
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)

    fg = V > bg_v_min                               # drop the near-black background
    teeth = fg & (S < teeth_s_max) & (V > teeth_v_min)

    # Tongue: warm hue (wraps red), region-grown from a confident salmon core that must
    # sit in the eroded mouth interior — this is what stops a warm lip-corner highlight
    # from being claimed as tongue.
    side = max(bgr.shape[:2])
    k = max(9, int(interior_frac * side)) | 1       # force odd
    interior = cv2.erode(fg.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    warm = fg & ~teeth & (((H >= tongue_h_lo) & (H <= tongue_h_hi)) | (H <= 6)) & (V > bg_v_min)
    core = warm & interior & (S >= tongue_s_min) & (S <= tongue_s_max) & (V > tongue_v_min)

    tongue = np.zeros_like(fg)
    if core.sum() > 50:
        cc = _largest_cc(core)
        ys, xs = np.where(cc)
        pad = int(0.15 * max(ys.max() - ys.min(), xs.max() - xs.min()))
        roi = np.zeros_like(fg)
        roi[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad] = True
        tongue = warm & roi
        if edge_smooth > 0:
            kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge_smooth + 1, 2 * edge_smooth + 1))
            tongue = cv2.morphologyEx(tongue.astype(np.uint8), cv2.MORPH_CLOSE, kk).astype(bool)
        tongue = _largest_cc(tongue)

    # Lips: saturated remainder. The S floor drops the dark, desaturated mouth cavity so
    # the interior gap is NOT labelled lips.
    lips = fg & ~teeth & ~tongue & (S >= lips_s_min)
    if edge_smooth > 0:
        kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge_smooth + 1, 2 * edge_smooth + 1))
        lips = cv2.morphologyEx(lips.astype(np.uint8), cv2.MORPH_CLOSE, kk).astype(bool)

    return lips, teeth, tongue


class BD_MouthParts(io.ComfyNode):
    """
    Separate an isolated mouth render into lips / teeth / tongue and pack them into the
    Unreal viseme-atlas RGBA contract (R=lips, G=teeth, B=tongue, A=POM).

    Standalone HSV colour engine — no model required. Tuned on the PinkF1 visemes; the
    thresholds are exposed so other mouth palettes can be dialled in. Optional
    `edge_refine` snaps the per-part edges to the image. Wire a depth/POM map into `pom`
    for the A channel (else A = the mouth foreground).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_MouthParts",
            display_name="BD MP Mouth Parts",
            category="🧠BrainDead/Segmentation",
            description=(
                "Separate an isolated mouth render into lips / teeth / tongue by colour "
                "(no model) and pack them RGBA for the Unreal viseme atlas: R=lips, "
                "G=teeth, B=tongue, A=POM (wire depth into `pom`). Replaces the "
                "SAM3×3 → FillHoles → PackChannels chain. Tuned on the PinkF1 visemes; "
                "thresholds exposed for other palettes."
            ),
            inputs=[
                io.Image.Input("image", tooltip="Isolated mouth render (background near-black). Only image[0] is used."),
                io.Combo.Input("engine", options=["color", "sam3"], default="color", optional=True,
                               tooltip="color — HSV classification, no model (default).\n"
                                       "sam3  — colour split SEEDS SAM3 (box + interior + sibling-negative points) "
                                       "for object-accurate boundaries. Requires `model`; falls back to color if unwired."),
                io.Model.Input("model", optional=True,
                               tooltip="Comfy-core SAM3 model (load sam3.pt — the same MODEL SAM3 Detect / BD MP "
                                       "SAM3 Face Segment use). Only used when engine='sam3'."),
                io.Image.Input("pom", optional=True,
                               tooltip="Optional POM / depth map (e.g. Lotus2 depth → NormLuma → CenterLuma) → packed "
                                       "into the A channel. Its luminance is used. If unwired, A = mouth foreground "
                                       "(lips∪teeth∪tongue)."),
                io.Int.Input("bg_v_min", default=25, min=0, max=128, step=1, optional=True,
                             tooltip="Background cutoff: pixels with HSV value ≤ this are background (black). Raise if "
                                     "the render has a lifted/grey background."),
                io.Int.Input("teeth_s_max", default=60, min=0, max=255, step=1, optional=True,
                             tooltip="Teeth = saturation BELOW this (white/grey is desaturated). Raise to grab "
                                     "tinted teeth; lower if pale lip facets leak into teeth."),
                io.Int.Input("teeth_v_min", default=140, min=0, max=255, step=1, optional=True,
                             tooltip="Teeth = value (brightness) ABOVE this. Raise to require brighter teeth."),
                io.Int.Input("tongue_h_lo", default=325, min=0, max=360, step=1, optional=True,
                             tooltip="Tongue warm-hue band low edge (°). The tongue is warmer (closer to red, "
                                     "~338°) than the magenta lips (~310°); the |H≤6° wrap is always included."),
                io.Int.Input("tongue_h_hi", default=358, min=0, max=360, step=1, optional=True,
                             tooltip="Tongue warm-hue band high edge (°)."),
                io.Int.Input("tongue_s_min", default=80, min=0, max=255, step=1, optional=True,
                             tooltip="Tongue core min saturation (excludes near-grey)."),
                io.Int.Input("tongue_s_max", default=170, min=0, max=255, step=1, optional=True,
                             tooltip="Tongue core MAX saturation — the tongue is LESS saturated than the lips, so "
                                     "this ceiling separates the salmon tongue from saturated magenta lips."),
                io.Int.Input("tongue_v_min", default=120, min=0, max=255, step=1, optional=True,
                             tooltip="Tongue core min value (brightness)."),
                io.Int.Input("lips_s_min", default=85, min=0, max=255, step=1, optional=True,
                             tooltip="Lips min saturation. Acts as the cavity guard: the dark, desaturated mouth "
                                     "interior falls below this so it is NOT labelled lips. Lower to include "
                                     "darker lip facets; raise if cavity bleeds in."),
                io.Float.Input("interior_frac", default=0.06, min=0.0, max=0.25, step=0.005, optional=True,
                               tooltip="Mouth-interior erosion (fraction of the longer side). The tongue core must "
                                       "lie inside the foreground eroded by this much — kills warm lip-corner "
                                       "highlights that would otherwise grow into a false tongue. 0 disables the gate."),
                io.Int.Input("sam3_iters", default=1, min=0, max=5, optional=True,
                             tooltip="engine='sam3' only: SAM decoder refinement passes. 1 = single pass (fullest "
                                     "part — extra passes tend to shrink small parts on stylized renders); a "
                                     "collapse-guard reverts any part the loop over-shrinks."),
                io.Int.Input("bleed_guard", default=24, min=0, max=200, step=2, optional=True,
                             tooltip="engine='sam3' only: dilate the colour mouth foreground by this many px, then "
                                     "clip each SAM3 part to it — stops SAM3 from escaping onto skin/specular. "
                                     "0 = clip exactly to the colour foreground."),
                io.Int.Input("edge_smooth", default=3, min=0, max=15, step=1, optional=True,
                             tooltip="Morphological close radius (px) to seal jagged lip/tongue edges. 0 = none."),
                io.Boolean.Input("fill_holes", default=True, optional=True,
                                 tooltip="Fill interior holes so each part is solid (e.g. specular gaps in the "
                                         "tongue, gaps between teeth in the lips ring)."),
                io.Combo.Input("edge_refine", options=["off", "guided", "matting", "vitmatte"], default="off",
                               optional=True,
                               tooltip="Snap each part's edge to the image colour/edges (shared with BD MP SAM3):\n"
                                       "  off      — colour masks as-is (fast)\n"
                                       "  guided   — guided filter (fast, edge-aware)\n"
                                       "  matting  — PyMatting closed-form (CPU, no model)\n"
                                       "  vitmatte — VitMatte deep matting (GPU, auto-downloads). Each degrades to "
                                       "'off' if its backend is missing."),
                io.Int.Input("refine_radius", default=6, min=1, max=40, step=1, optional=True,
                             tooltip="Guided-filter radius / matting trimap band width (px)."),
                io.Float.Input("refine_eps", default=1e-4, min=1e-6, max=1e-1, step=1e-4, optional=True,
                               tooltip="Guided-filter edge sensitivity (smaller hugs edges harder). Ignored by matting."),
                io.Float.Input("refine_threshold", default=0.5, min=0.05, max=0.95, step=0.05, optional=True,
                               tooltip="Binarize the refined alpha at this level."),
                io.Combo.Input("vitmatte_model", options=["small", "base"], default="small", optional=True,
                               tooltip="VitMatte variant for edge_refine='vitmatte' (auto-downloaded from the HF hub)."),
            ],
            outputs=[
                io.Mask.Output(display_name="lips"),
                io.Mask.Output(display_name="teeth"),
                io.Mask.Output(display_name="tongue"),
                io.Image.Output(display_name="rgb_packed",
                                tooltip="R=lips, G=teeth, B=tongue (no alpha). The PackChannels equivalent."),
                io.Image.Output(display_name="rgba_packed",
                                tooltip="R=lips, G=teeth, B=tongue, A=POM (from `pom`, else mouth foreground). "
                                        "The viseme-atlas cell contract."),
                io.Image.Output(display_name="debug_overlay",
                                tooltip="Render tinted by part (lips=R, teeth=G, tongue=B) for QC."),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, image, engine="color", model=None, pom=None, bg_v_min=25, teeth_s_max=60,
                teeth_v_min=140, tongue_h_lo=325, tongue_h_hi=358, tongue_s_min=80, tongue_s_max=170,
                tongue_v_min=120, lips_s_min=85, interior_frac=0.06, sam3_iters=1, bleed_guard=24,
                edge_smooth=3, fill_holes=True, edge_refine="off", refine_radius=6, refine_eps=1e-4,
                refine_threshold=0.5, vitmatte_model="small") -> io.NodeOutput:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        B, H, W, C = image.shape

        def _m(np_u8):
            return torch.from_numpy((np_u8 > 0).astype(np.float32))

        def _img(arr_f):
            return torch.from_numpy(arr_f.astype(np.float32)).unsqueeze(0)

        def _bail(status):
            z = np.zeros((H, W), np.uint8)
            rgb = np.zeros((H, W, 3), np.float32)
            rgba = np.zeros((H, W, 4), np.float32)
            return io.NodeOutput(_m(z), _m(z), _m(z), _img(rgb), _img(rgba), _img(rgb), status)

        if not HAS_CV2:
            return _bail("opencv missing — no separation")

        frame = image[0].detach().cpu().float().numpy()
        rgb_u8 = (frame[..., :3] * 255.0).clip(0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)

        # Colour split — the standalone result AND the SAM3 prior.
        lips, teeth, tongue = _split_mouth(
            bgr, bg_v_min=bg_v_min, teeth_s_max=teeth_s_max, teeth_v_min=teeth_v_min,
            tongue_h_lo=tongue_h_lo, tongue_h_hi=tongue_h_hi, tongue_s_min=tongue_s_min,
            tongue_s_max=tongue_s_max, tongue_v_min=tongue_v_min, lips_s_min=lips_s_min,
            interior_frac=interior_frac, edge_smooth=edge_smooth,
        )
        color_u8 = {"lips": lips.astype(np.uint8) * 255,
                    "teeth": teeth.astype(np.uint8) * 255,
                    "tongue": tongue.astype(np.uint8) * 255}

        used_engine = engine
        if engine == "sam3" and model is None:
            print("[BD MP Mouth Parts] engine='sam3' but no model wired — falling back to color", flush=True)
            used_engine = "color"

        if used_engine == "sam3":
            # Colour masks SEED SAM3: per part, box + interior positives + the other two
            # parts' centroids as negatives → SAM3 grows the true object, clipped to the
            # mouth foreground and component-cleaned. Built in-house (no other pack).
            comfy.model_management.load_model_gpu(model)
            device = comfy.model_management.get_torch_device()
            dtype = model.model.get_dtype()
            sam3 = model.model.diffusion_model
            sam_frame = comfy.utils.common_upscale(
                image[0:1, ..., :3].movedim(-1, 1), _SAM3_SIZE, _SAM3_SIZE, "bilinear", crop="disabled"
            ).to(device=device, dtype=dtype)

            fg_union = ((color_u8["lips"] | color_u8["teeth"] | color_u8["tongue"]) > 0).astype(np.uint8) * 255
            guard = None
            if bleed_guard > 0:
                guard = cv2.dilate(fg_union, cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * bleed_guard + 1, 2 * bleed_guard + 1)))
            else:
                guard = fg_union

            u8 = {}
            for key in ("lips", "teeth", "tongue"):
                cm = color_u8[key]
                if cm.max() == 0:
                    u8[key] = cm
                    continue
                pos_px, box = _seed_points(cm, k=6)
                neg_px = [c for c in (_centroid_px(color_u8[o]) for o in ("lips", "teeth", "tongue") if o != key)
                          if c is not None]
                try:
                    sam = _sam3_segment(sam3, sam_frame, H, W, device, dtype,
                                        pos_px, neg_px, box, mask_threshold=0.5, iters=sam3_iters)
                except Exception as e:
                    print(f"[BD MP Mouth Parts] SAM3 failed for {key} ({e}) — using colour mask", flush=True)
                    u8[key] = cm
                    continue
                sam = np.minimum(sam, guard)                       # keep inside the mouth
                sam = _clean_feature_mask(sam, pos_px, smooth_px=edge_smooth, fill=False)
                u8[key] = sam if sam.max() > 0 else cm             # never lose a part SAM3 dropped
        else:
            u8 = dict(color_u8)

        # Optional edge snap, then optional hole-fill (fill AFTER refine — pre-filling
        # bulges the refined edge, same lesson as BD MP SAM3).
        for k in ("lips", "teeth", "tongue"):
            if edge_refine != "off" and u8[k].max() > 0:
                u8[k] = _refine_feature_mask(u8[k], rgb_u8, edge_refine, refine_radius,
                                             refine_eps, refine_threshold, vitmatte_model)
            if fill_holes and u8[k].max() > 0:
                u8[k] = _fill_holes(u8[k])

        # Keep the parts mutually exclusive after fill: teeth/tongue win their pixels back
        # from the lips ring (lips is the saturated remainder, so it can overlap a filled
        # interior). Priority: teeth > tongue > lips.
        teeth_b = u8["teeth"] > 0
        tongue_b = (u8["tongue"] > 0) & ~teeth_b
        lips_b = (u8["lips"] > 0) & ~teeth_b & ~tongue_b
        u8 = {"lips": lips_b.astype(np.uint8) * 255,
              "teeth": teeth_b.astype(np.uint8) * 255,
              "tongue": tongue_b.astype(np.uint8) * 255}

        # Packed maps: R=lips, G=teeth, B=tongue.
        rgb_packed = np.stack([u8["lips"], u8["teeth"], u8["tongue"]], axis=-1).astype(np.float32) / 255.0

        # A channel: provided POM/depth luminance, else the mouth foreground.
        if pom is not None:
            p = pom[0].detach().cpu().float().numpy() if pom.ndim == 4 else pom.detach().cpu().float().numpy()
            if p.ndim == 3:
                alpha = p[..., :3].mean(axis=-1) if p.shape[-1] >= 3 else p[..., 0]
            else:
                alpha = p
            if alpha.shape != (H, W):
                alpha = cv2.resize(alpha, (W, H), interpolation=cv2.INTER_LINEAR)
            alpha = alpha.astype(np.float32)
            if alpha.max() > 1.0:
                alpha /= 255.0
            a_src = "pom"
        else:
            alpha = ((u8["lips"] | u8["teeth"] | u8["tongue"]) > 0).astype(np.float32)
            a_src = "foreground"
        rgba_packed = np.concatenate([rgb_packed, alpha[..., None]], axis=-1)

        # Debug overlay: tint the render by part.
        ov = rgb_u8.astype(np.float32).copy()
        tint = {"lips": (255, 40, 40), "teeth": (40, 255, 40), "tongue": (40, 40, 255)}
        for k, col in tint.items():
            m = u8[k] > 0
            ov[m] = 0.5 * ov[m] + 0.5 * np.array(col, np.float32)
        debug_overlay = ov.clip(0, 255) / 255.0

        def pct(k):
            return f"{(u8[k] > 0).mean() * 100:.1f}%"
        status = (f"mouth parts [{used_engine}]: lips={pct('lips')} teeth={pct('teeth')} tongue={pct('tongue')} | "
                  f"A={a_src}" + (f" | refine={edge_refine}" if edge_refine != "off" else ""))
        print(f"[BD MP Mouth Parts] {status}", flush=True)

        return io.NodeOutput(
            _m(u8["lips"]), _m(u8["teeth"]), _m(u8["tongue"]),
            _img(rgb_packed), _img(rgba_packed), _img(debug_overlay), status,
        )


# ── Registration ────────────────────────────────────────────────────────────────

MOUTH_PARTS_V3_NODES = [BD_MouthParts]
MOUTH_PARTS_NODES = {"BD_MouthParts": BD_MouthParts}
MOUTH_PARTS_DISPLAY_NAMES = {"BD_MouthParts": "BD MP Mouth Parts"}
