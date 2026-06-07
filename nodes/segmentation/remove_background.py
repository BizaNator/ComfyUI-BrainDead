"""
BD_RemoveBackground — self-contained one-node background removal.

Pipeline:
  1. SAM3 text-grounded segmentation (per positive prompt, union combined)
  2. Optional negative prompts subtracted
  3. Morphological hole-fill
  4. Optional alpha matting edge refinement (closed-form via pymatting)
  5. Optional Gaussian edge blur
  6. Crop to content + optional square pad
  7. Outputs: rgba, mask, white-bg composite, black-bg composite, crop_box
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io
from . import bd_sam3
from . import matting  # shared matting/edge/decontam/sticker lib


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_prompts(text: str) -> list[str]:
    return [l.strip() for l in (text or "").strip().splitlines() if l.strip()]


def _union_masks(masks: list[torch.Tensor]) -> torch.Tensor:
    """Union a list of (1,H,W) masks → (1,H,W)."""
    if not masks:
        return None
    out = masks[0].clone()
    for m in masks[1:]:
        out = torch.maximum(out, m)
    return out


def _fill_holes(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Morphological closing + binary fill for a (1,H,W) mask."""
    if radius <= 0:
        return mask
    import cv2
    from scipy.ndimage import binary_fill_holes
    arr = mask[0].cpu().numpy().astype(np.float32)
    binary = (arr >= 0.5).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    filled = binary_fill_holes(closed > 127).astype(np.float32)
    return torch.from_numpy(filled).unsqueeze(0)


def _edge_blur(mask: torch.Tensor, radius: float) -> torch.Tensor:
    """Gaussian blur the alpha mask for soft edges."""
    if radius <= 0:
        return mask
    import cv2
    arr = mask[0].cpu().numpy().astype(np.float32)
    ksize = max(3, int(radius * 2 + 1) | 1)
    blurred = cv2.GaussianBlur(arr, (ksize, ksize), radius * 0.5)
    return torch.from_numpy(blurred).unsqueeze(0)


def _crop_to_content(image: torch.Tensor, alpha: torch.Tensor,
                     padding: int, output_size: int,
                     output_size_mode: str) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Crop image+alpha to the bounding box of the alpha mask, then optionally pad/resize."""
    H, W = alpha.shape[-2], alpha.shape[-1]
    m = (alpha[0] > 0.01)
    ys = m.any(dim=1).nonzero(as_tuple=True)[0]
    xs = m.any(dim=0).nonzero(as_tuple=True)[0]

    if ys.numel() == 0 or xs.numel() == 0:
        crop_box = f"0,0,{W},{H}"
        return image, alpha, crop_box

    y0 = max(0, int(ys[0]) - padding)
    y1 = min(H, int(ys[-1]) + 1 + padding)
    x0 = max(0, int(xs[0]) - padding)
    x1 = min(W, int(xs[-1]) + 1 + padding)
    crop_box = f"{x0},{y0},{x1},{y1}"

    img_crop = image[:, y0:y1, x0:x1, :]
    alpha_crop = alpha[:, :, y0:y1, x0:x1] if alpha.ndim == 4 else alpha[:, y0:y1, x0:x1]

    if output_size <= 0 or output_size_mode == "none":
        return img_crop, alpha_crop, crop_box

    ch, cw = img_crop.shape[1], img_crop.shape[2]
    if output_size_mode == "resize":
        img_out = F.interpolate(
            img_crop.permute(0, 3, 1, 2).float(),
            size=(output_size, output_size), mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1)
        alpha_4d = alpha_crop if alpha_crop.ndim == 4 else alpha_crop.unsqueeze(1)
        alpha_out = F.interpolate(alpha_4d.float(), size=(output_size, output_size),
                                  mode="bilinear", align_corners=False).squeeze(1)
    else:  # pad
        side = max(ch, cw, output_size)
        py = (side - ch) // 2
        px = (side - cw) // 2
        img_out = F.pad(
            img_crop.permute(0, 3, 1, 2),
            (px, side - cw - px, py, side - ch - py),
        ).permute(0, 2, 3, 1)
        alpha_4d = alpha_crop if alpha_crop.ndim == 4 else alpha_crop.unsqueeze(1)
        alpha_out = F.pad(
            alpha_4d,
            (px, side - cw - px, py, side - ch - py),
        ).squeeze(1)

    return img_out, alpha_out, crop_box


def _compose_rgba(image: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Compose RGB + alpha → RGBA (B,H,W,4)."""
    rgb = image[..., :3]
    a = alpha[0].unsqueeze(0).unsqueeze(-1).expand_as(rgb[:, :, :, :1])
    return torch.cat([rgb, a], dim=-1)


def _compose_on_bg(image: torch.Tensor, alpha: torch.Tensor,
                   bg: float) -> torch.Tensor:
    """Alpha-composite over flat bg colour → RGB (B,H,W,3)."""
    rgb = image[..., :3].float()
    a = alpha[0].unsqueeze(0).unsqueeze(-1)
    return (rgb * a + bg * (1.0 - a)).clamp(0, 1)


# ── node ─────────────────────────────────────────────────────────────────────

class BD_RemoveBackground(io.ComfyNode):
    """
    One-node background removal. SAM3 text-segments the subject, optionally
    refines edges with closed-form alpha matting (pymatting), then outputs
    an RGBA image with clean transparency plus white/black composites.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_RemoveBackground",
            display_name="BD Remove Background",
            category="🧠BrainDead/Segmentation",
            description=(
                "Self-contained background removal. SAM3 text-grounded segmentation → "
                "hole fill → optional pymatting edge refinement → RGBA output. "
                "Positive prompts select the subject; negative prompts are subtracted. "
                "Matting mode 'closed_form' uses pymatting for soft, hair-accurate edges."
            ),
            inputs=[
                io.Image.Input("image"),
                io.String.Input(
                    "prompts", default="subject\nperson\nforeground",
                    multiline=True,
                    tooltip="One SAM3 text prompt per line. Their masks are union-combined. "
                            "Typical: 'person', 'character', 'product', 'foreground object'.",
                ),
                io.String.Input(
                    "negative_prompts", default="", multiline=True, optional=True,
                    tooltip="One prompt per line. SAM3 runs each → unioned → subtracted from positive mask. "
                            "Useful to punch out backgrounds that bleed into the positive mask.",
                ),
                io.Mask.Input(
                    "external_mask", optional=True,
                    tooltip="Optional mask from any source (ATR parser, SAM3, MediaPipe, manual, etc.). "
                            "Behaviour is controlled by mask_mode.",
                ),
                io.Combo.Input(
                    "mask_mode",
                    options=["constrain", "subtract", "use_directly"],
                    default="constrain",
                    optional=True,
                    tooltip="How to use external_mask:\n"
                            "  constrain     — intersect SAM3 result with external_mask "
                            "(prevents bleed outside rough silhouette)\n"
                            "  subtract      — subtract external_mask from SAM3 result "
                            "(punch out an already-masked region)\n"
                            "  use_directly  — skip SAM3 entirely, use external_mask as the alpha "
                            "(prompts ignored). Useful for hole-fill + matting on a pre-made mask.",
                ),
                io.Combo.Input(
                    "matting_mode",
                    options=["none", "closed_form"],
                    default="closed_form",
                    tooltip="Edge refinement after SAM3:\n"
                            "  none         — use SAM3 mask directly (fast, hard edges)\n"
                            "  closed_form  — pymatting closed-form alpha matting around the "
                            "mask boundary for soft, hair-accurate transparency.",
                ),
                io.Int.Input(
                    "matting_erode", default=8, min=0, max=64, optional=True,
                    tooltip="Pixels to erode inward to define the definite-foreground trimap band. "
                            "Larger = wider unknown band = more matting area.",
                ),
                io.Int.Input(
                    "matting_dilate", default=8, min=0, max=64, optional=True,
                    tooltip="Pixels to dilate outward to define the definite-background trimap band.",
                ),
                io.Int.Input(
                    "fill_holes_radius", default=4, min=0, max=64, optional=True,
                    tooltip="Morphological closing radius (pixels) applied before matting to seal "
                            "interior gaps from SAM3. 0 = skip.",
                ),
                io.Float.Input(
                    "edge_blur", default=0.0, min=0.0, max=16.0, step=0.5, optional=True,
                    tooltip="Gaussian blur radius applied to the final alpha for soft feathering. "
                            "0 = no blur. Use after 'none' matting for quick soft edges.",
                ),
                io.Boolean.Input(
                    "decontaminate", default=True, optional=True,
                    tooltip="Estimate the true foreground colour (pymatting) to remove background "
                            "colour spill at edges — fixes the faint white outline you see on the "
                            "black-bg composite. Slight cost; runs on the cropped region.",
                ),
                io.Int.Input(
                    "edge_shrink", default=0, min=0, max=32, optional=True,
                    tooltip="Erode the alpha inward by N px to cut a thin fringe halo. "
                            "Use 1–2 if a hard edge ring remains after decontaminate.",
                ),
                io.Combo.Input(
                    "edge_refine", options=["none", "guided", "vitmatte"], default="none", optional=True,
                    tooltip="Edge-aware SOFT matte refine, run on the BOUNDARY ROI only (confident "
                            "interior/exterior are locked, so it refines the edge — it won't fade the "
                            "whole image):\n"
                            "  guided   — cv2 guided filter (fast, RGB-guided)\n"
                            "  vitmatte — VitMatte deep matting (cleanest on hair/soft edges; GPU; "
                            "auto-downloads hustvl/vitmatte on first use).",
                ),
                io.Combo.Input(
                    "vitmatte_model", options=["small", "base"], default="small", optional=True,
                    tooltip="VitMatte variant for edge_refine='vitmatte'. Auto-downloaded from HF.",
                ),
                io.Float.Input(
                    "sharpen", default=0.0, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="Crisp the matte edge (smoothstep). 0 = leave soft, 1 = tight/clean cutout. "
                            "Use to reduce a noisy/blurry boundary.",
                ),
                io.Float.Input(
                    "bg_clean", default=0.0, min=0.0, max=0.5, step=0.01, optional=True,
                    tooltip="Zero any alpha below this value to kill faint background ghosting / noise "
                            "(e.g. 0.05). 0 = off.",
                ),
                io.Boolean.Input(
                    "key_gaps", default=False, optional=True,
                    tooltip="Punch out background-coloured ISLANDS left inside the mask — the bits of "
                            "original background showing through gaps (between legs, fingers, handles). "
                            "Samples the bg colour from the removed area and removes only small matching "
                            "islands, so large same-coloured SUBJECT areas (white clothing) are kept.",
                ),
                io.Float.Input(
                    "key_tolerance", default=0.10, min=0.01, max=0.4, step=0.01, optional=True,
                    tooltip="key_gaps colour-match tolerance (LAB). Higher catches more (risks subject); "
                            "lower is safer. ~0.08–0.15 for a clean white/solid background.",
                ),
                io.Float.Input(
                    "key_max_area", default=0.04, min=0.0, max=0.5, step=0.01, optional=True,
                    tooltip="key_gaps only removes bg-coloured islands smaller than this fraction of the "
                            "frame (protects large subject areas). ~0.03–0.06 for small gaps.",
                ),
                io.Int.Input(
                    "sticker_outline", default=0, min=0, max=128, optional=True,
                    tooltip="Die-cut STICKER mode: add a coloured trim border of N px around the "
                            "subject (0 = off). The `sticker` output is RGBA with the subject + this "
                            "outline, transparent outside.",
                ),
                io.String.Input(
                    "sticker_color", default="#ffffff", optional=True,
                    tooltip="Trim colour for the sticker outline (hex, e.g. #ffffff white, #f00078 pink).",
                ),
                io.Boolean.Input(
                    "crop_to_content", default=True, optional=True,
                    tooltip="Crop the output to the bounding box of the mask (plus padding). "
                            "Removes excess transparent border.",
                ),
                io.Int.Input(
                    "crop_padding", default=16, min=0, max=512, optional=True,
                    tooltip="Extra pixels added around the crop bounding box.",
                ),
                io.Int.Input(
                    "output_size", default=0, min=0, max=8192, optional=True,
                    tooltip="0 = native resolution. >0 = target size applied via output_size_mode.",
                ),
                io.Combo.Input(
                    "output_size_mode",
                    options=["none", "pad_square", "resize_square"],
                    default="none",
                    optional=True,
                    tooltip="How to apply output_size:\n"
                            "  none          — ignore output_size\n"
                            "  pad_square    — pad shortest side to make it output_size × output_size\n"
                            "  resize_square — resize to output_size × output_size (may distort)",
                ),
                io.Float.Input(
                    "mask_threshold", default=0.5, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="SAM3 confidence threshold. Lower = keep more area. Higher = tighter mask.",
                ),
                io.Boolean.Input(
                    "unload_model", default=False, optional=True,
                    tooltip="Unload SAM3 from VRAM after this node finishes.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="rgba",
                                tooltip="RGBA image with alpha = subject mask."),
                io.Mask.Output(display_name="mask",
                               tooltip="Subject alpha mask (H,W) [0,1]."),
                io.Image.Output(display_name="rgb_white_bg",
                                tooltip="Subject composited over pure white — good for 3D pipelines."),
                io.Image.Output(display_name="rgb_black_bg",
                                tooltip="Subject composited over pure black."),
                io.String.Output(display_name="crop_box",
                                 tooltip="'x0,y0,x1,y1' bounding box of the cropped region in the original image."),
                io.Image.Output(display_name="sticker",
                                tooltip="RGBA die-cut sticker: subject + coloured trim outline "
                                        "(sticker_outline/sticker_color), transparent outside. "
                                        "When sticker_outline=0 this is the plain RGBA subject."),
            ],
        )

    @classmethod
    def execute(cls, image, prompts,
                negative_prompts="",
                external_mask=None,
                mask_mode="constrain",
                matting_mode="closed_form",
                matting_erode=8, matting_dilate=8,
                fill_holes_radius=4,
                edge_blur=0.0,
                decontaminate=True,
                edge_shrink=0,
                edge_refine="none",
                vitmatte_model="small",
                sharpen=0.0,
                bg_clean=0.0,
                key_gaps=False,
                key_tolerance=0.10,
                key_max_area=0.04,
                sticker_outline=0,
                sticker_color="#ffffff",
                crop_to_content=True,
                crop_padding=16,
                output_size=0,
                output_size_mode="none",
                mask_threshold=0.5,
                unload_model=False) -> io.NodeOutput:

        H, W = image.shape[1], image.shape[2]

        # ── Normalise external mask ────────────────────────────────────────
        # Accept a mask if one is meaningfully provided, otherwise ignore it — so
        # templates can wire LoadImage's MASK output unconditionally. LoadImage
        # emits a tiny 64×64 placeholder (all-zeros) when the image has no alpha;
        # using that in `constrain` mode would crash (size mismatch) or blank the
        # output, so we drop empty/full placeholders and resize any real mask.
        ext = None
        if external_mask is not None:
            e = external_mask.detach().cpu().float()
            if e.ndim == 2:
                e = e.unsqueeze(0)
            e = e[:1]  # (1,h,w) — first mask from any batch
            emin, emax = float(e.min()), float(e.max())
            if emax <= 1e-6 or emin >= 1.0 - 1e-6:
                # all-zeros (no subject) or all-ones (no constraint) placeholder
                print("[BD RemoveBackground] external_mask is an empty/full placeholder "
                      "(e.g. LoadImage with no alpha) — ignoring it, using SAM3.", flush=True)
            else:
                if e.shape[-2:] != (H, W):
                    e = F.interpolate(e.unsqueeze(0), size=(H, W),
                                      mode="bilinear", align_corners=False).squeeze(0)
                    print(f"[BD RemoveBackground] resized external_mask to {H}×{W}.", flush=True)
                ext = e

        # ── Skip SAM3 if use_directly ──────────────────────────────────────
        if mask_mode == "use_directly" and ext is not None:
            combined = ext
            print("[BD RemoveBackground] Using external_mask directly (SAM3 skipped)", flush=True)
        else:
            pos_prompts = _parse_prompts(prompts)
            neg_prompts = _parse_prompts(negative_prompts or "")
            if not pos_prompts:
                pos_prompts = ["foreground"]

            # ── Load SAM3 ──────────────────────────────────────────────────
            sam3_model, sam3_clip = bd_sam3.load_sam3(need_clip=True)
            print(f"[BD RemoveBackground] SAM3 loaded. Pos: {pos_prompts}", flush=True)

            try:
                pos_masks = []
                for p in pos_prompts:
                    m = bd_sam3.segment_text(
                        sam3_model, sam3_clip, image,
                        prompt=p, threshold=float(mask_threshold),
                    )
                    pos_masks.append(m)

                combined = _union_masks(pos_masks)
                if combined is None:
                    combined = torch.zeros((1, H, W), dtype=torch.float32)

                # Subtract negative prompts
                if neg_prompts:
                    neg_masks = [
                        bd_sam3.segment_text(sam3_model, sam3_clip, image,
                                             prompt=p, threshold=float(mask_threshold))
                        for p in neg_prompts
                    ]
                    neg_union = _union_masks(neg_masks)
                    if neg_union is not None:
                        combined = (combined - neg_union).clamp(0.0, 1.0)

                # Apply external_mask per mask_mode
                if ext is not None:
                    if mask_mode == "constrain":
                        combined = torch.minimum(combined, ext)
                    elif mask_mode == "subtract":
                        combined = (combined - ext).clamp(0.0, 1.0)

            finally:
                if unload_model:
                    import comfy.model_management as mm
                    mm.soft_empty_cache()

        # ── Hole fill ─────────────────────────────────────────────────────
        combined = _fill_holes(combined, int(fill_holes_radius))

        # ── Alpha matting (shared lib) ────────────────────────────────────
        if matting_mode == "closed_form":
            img_np = image[0, :, :, :3].detach().cpu().float().numpy()
            mask_np = combined[0].detach().cpu().numpy()
            refined = matting.closed_form_alpha(
                img_np, mask_np, int(matting_erode), int(matting_dilate))
            combined = torch.from_numpy(refined).unsqueeze(0)

        # ── Edge-aware refine (boundary-ROI only; confident regions locked) ─
        if edge_refine == "guided":
            combined = matting.guided_refine(combined, image)
        elif edge_refine == "vitmatte":
            combined = matting.vitmatte_refine(combined, image, variant=vitmatte_model)

        # ── Key out bg-coloured gaps inside the mask (between legs/fingers) ─
        if key_gaps:
            combined = matting.key_background(
                combined, image,
                tolerance=float(key_tolerance), max_area_frac=float(key_max_area))

        # ── Fringe shrink + crisp/clean the matte ─────────────────────────
        if int(edge_shrink) > 0:
            combined = matting.erode_alpha(combined, int(edge_shrink))
        combined = matting.clean_alpha(combined, sharpen=float(sharpen), floor=float(bg_clean))

        # ── Edge blur ─────────────────────────────────────────────────────
        combined = _edge_blur(combined, float(edge_blur))

        # ── Crop to content ───────────────────────────────────────────────
        alpha_bhw = combined.unsqueeze(0)  # (1,1,H,W) for crop
        img_out = image
        if crop_to_content:
            img_out, alpha_bhw, crop_box = _crop_to_content(
                image, combined, int(crop_padding),
                int(output_size), output_size_mode,
            )
            combined = alpha_bhw
        else:
            crop_box = f"0,0,{image.shape[2]},{image.shape[1]}"
            if output_size > 0 and output_size_mode != "none":
                img_out, alpha_bhw, crop_box = _crop_to_content(
                    image, combined, 0, int(output_size), output_size_mode,
                )
                combined = alpha_bhw

        # ── Decontaminate (remove bg colour spill at edges) ───────────────
        if decontaminate:
            img_out = matting.decontaminate(img_out, combined)

        # ── Compose outputs ───────────────────────────────────────────────
        rgba      = _compose_rgba(img_out, combined)
        white_bg  = _compose_on_bg(img_out, combined, 1.0)
        black_bg  = _compose_on_bg(img_out, combined, 0.0)
        sticker   = matting.make_sticker(img_out, combined, int(sticker_outline), sticker_color)

        print(
            f"[BD RemoveBackground] Done. Output {rgba.shape[1]}×{rgba.shape[2]} "
            f"matting={matting_mode} decontaminate={decontaminate} "
            f"edge_refine={edge_refine} shrink={edge_shrink} "
            f"sticker={sticker_outline} crop={crop_box}",
            flush=True,
        )

        return io.NodeOutput(rgba, combined, white_bg, black_bg, crop_box, sticker)


REMOVE_BACKGROUND_V3_NODES = [BD_RemoveBackground]
REMOVE_BACKGROUND_NODES = {"BD_RemoveBackground": BD_RemoveBackground}
REMOVE_BACKGROUND_DISPLAY_NAMES = {"BD_RemoveBackground": "BD Remove Background"}
