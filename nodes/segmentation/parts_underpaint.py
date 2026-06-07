"""
BD_PartsUnderpaint — reveal what's beneath each removed part.

Companion to BD_PartsBatchEdit: instead of rebuilding isolated part renders,
this node inpaints the SOURCE IMAGE where each part was removed, revealing
what is naturally visible underneath — skin under clothing, car interior
under panels, the wall behind a shelf, etc.

Two modes:
  per_part_sequential — one Qwen pass per part, processed outer-to-inner.
    Each pass's result feeds the next, so successive layers are progressively
    stripped away. Yields per-step intermediate frames.
  all_parts_combined — union all part masks, single Qwen inpaint pass.
    Faster but harder for Qwen to fill very large masked areas coherently.

Output base_image → wire into BD_PartsExport[base_image] so the rebuilt
part renders from BD_PartsBatchEdit composite onto this clean base instead
of the original occluded source image.
"""

import re
import math

import numpy as np
import torch
from PIL import Image

import comfy.utils
import comfy.model_management
import node_helpers
from nodes import common_ksampler

from comfy_api.latest import io
import comfy.samplers as _samplers

from .parts_types import PARTS_BUNDLE, ensure_bundle, empty_bundle, frame_size as _frame_size

# ── re-use shared encoding / decoding helpers from BD_PartsBatchEdit ──────────
from .parts_batch_edit import (
    _LLAMA_TEMPLATE,
    _encode_qwen_edit_plus,
    _encode_negative,
    _latent_spatial_dims,
    _upscale_latent_spatial,
    _tonemap_reinhard_latent,
    _decode_to_rgb,
    _extend_bbox,
    _resize_rgb_to,
    _dilate_alpha,
    _parse_skip,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_order(order_str: str, tag2pinfo: dict, skip_set: set) -> list[str]:
    """Return ordered tag list for sequential removal."""
    if order_str.strip():
        ordered = [t.strip() for t in re.split(r"[,\n]", order_str) if t.strip()]
        return [t for t in ordered if t in tag2pinfo and t not in skip_set]
    return [
        t for t, info in tag2pinfo.items()
        if t not in skip_set and isinstance(info, dict) and info.get("img") is not None
    ]


def _build_part_mask_in_source(info: dict, src_h: int, src_w: int
                                ) -> tuple[np.ndarray, list[int]]:
    """
    Build a (src_h, src_w) uint8 mask for a part in source-image coordinates.
    Returns (full_mask, [x1, y1, x2, y2]).
    """
    x1, y1, x2, y2 = [int(v) for v in info["xyxy"]]
    x1 = max(0, min(x1, src_w))
    x2 = max(0, min(x2, src_w))
    y1 = max(0, min(y1, src_h))
    y2 = max(0, min(y2, src_h))

    _img = info.get("img")
    arr = np.asarray(_img if _img is not None else np.zeros((1, 1, 4), dtype=np.uint8))
    alpha = arr[:, :, 3] if arr.ndim == 3 and arr.shape[2] == 4 else np.full(
        arr.shape[:2], 255, dtype=np.uint8
    )

    bbox_w, bbox_h = x2 - x1, y2 - y1
    full_mask = np.zeros((src_h, src_w), dtype=np.uint8)
    if bbox_w > 0 and bbox_h > 0:
        if alpha.shape[:2] != (bbox_h, bbox_w):
            alpha = np.asarray(
                Image.fromarray(alpha, mode="L").resize((bbox_w, bbox_h), Image.NEAREST)
            )
        full_mask[y1:y2, x1:x2] = alpha

    return full_mask, [x1, y1, x2, y2]


def _prefill_region(rgb_np: np.ndarray, mask_u8: np.ndarray, mode: str) -> np.ndarray:
    """
    Fill mask > 127 pixels in rgb_np with surrounding colors.
    rgb_np: (H, W, 3) uint8   mask_u8: (H, W) uint8
    Returns (H, W, 3) uint8.
    """
    result = rgb_np.copy()
    m = mask_u8 > 127
    if not m.any():
        return result

    if mode == "color_spread":
        try:
            import cv2
            result = cv2.inpaint(rgb_np, m.astype(np.uint8), inpaintRadius=7,
                                 flags=cv2.INPAINT_TELEA)
        except Exception:
            result[m] = 128
    elif mode == "nearest_neighbor":
        try:
            from scipy.ndimage import distance_transform_edt
            outside = ~m
            if outside.any():
                _, idx = distance_transform_edt(outside, return_indices=True)
                result[m] = rgb_np[idx[0][m], idx[1][m]]
        except ImportError:
            result[m] = 128
    elif mode == "neutral_gray":
        result[m] = 128
    elif mode == "white":
        result[m] = 255
    elif mode == "black":
        result[m] = 0
    # else "none": leave as-is

    return result


def _run_underpaint_pass(
    *,
    model, clip, vae,
    current_img_np: np.ndarray,   # (src_h, src_w, 3) uint8, current accumulation
    full_mask: np.ndarray,         # (src_h, src_w) uint8
    xyxy: list,
    prompt: str,
    neg_cond,
    context_extend_factor: float,
    prefill_mode: str,
    latent_upscale_factor: float,
    latent_upscale_method: str,
    model_sampling_shift: float,
    cfg_norm_strength: float,
    tonemap_reinhard_multiplier: float,
    tp: int,
    seed: int, steps: int, cfg: float,
    sampler_name: str, scheduler: str, denoise: float,
    mask_dilate_pixels: int,
    mask_blend_pixels: int,
) -> np.ndarray:
    """
    Inpaint the masked region in current_img_np and return the updated image.
    Only pixels inside the mask are changed; surrounding context is preserved.
    """
    src_h, src_w = current_img_np.shape[:2]

    # Extended context crop
    if context_extend_factor > 1.0:
        ext_xyxy, _ = _extend_bbox(xyxy, float(context_extend_factor), src_h, src_w)
    else:
        ext_xyxy = list(xyxy)
    ex1, ey1, ex2, ey2 = ext_xyxy

    crop_h = ey2 - ey1
    crop_w = ex2 - ex1
    if crop_h <= 0 or crop_w <= 0:
        return current_img_np

    crop_rgb = current_img_np[ey1:ey2, ex1:ex2].copy()
    crop_mask = full_mask[ey1:ey2, ex1:ex2].copy()

    # Pre-fill the masked region so Qwen sees plausible surrounding color
    prefilled = _prefill_region(crop_rgb, crop_mask, prefill_mode)

    # Reference tensor for Qwen
    ref_t = torch.from_numpy(prefilled.astype(np.float32) / 255.0).unsqueeze(0)

    # Encode
    pos_cond, ref_latent, (h_lat, w_lat) = _encode_qwen_edit_plus(
        clip, vae, prompt, ref_t, target_pixels=tp,
    )

    # Build noise mask at latent spatial dims (the KSampler inpaint region)
    lat_h, lat_w = _latent_spatial_dims(ref_latent)

    inpaint_mask = crop_mask.copy()
    if mask_dilate_pixels > 0:
        inpaint_mask = _dilate_alpha(inpaint_mask, int(mask_dilate_pixels))

    noise_mask_np = np.asarray(
        Image.fromarray(inpaint_mask, mode="L").resize((lat_w, lat_h), Image.NEAREST)
    ).astype(np.float32) / 255.0
    noise_mask_t = torch.from_numpy(noise_mask_np).unsqueeze(0)

    # Optional latent upscale (high-res-fix trick — more detail per crop)
    if latent_upscale_factor > 1.0:
        new_lat_h = int(round(lat_h * latent_upscale_factor))
        new_lat_w = int(round(lat_w * latent_upscale_factor))
        upscaled = _upscale_latent_spatial(ref_latent, new_lat_h, new_lat_w, latent_upscale_method)
        noise_mask_t = torch.nn.functional.interpolate(
            noise_mask_t.unsqueeze(0).float(),
            size=(new_lat_h, new_lat_w), mode="nearest",
        ).squeeze(0)
    else:
        upscaled = ref_latent

    start_latent = {
        "samples": upscaled.to(dtype=torch.float32),
        "noise_mask": noise_mask_t,
    }
    pos_cond = node_helpers.conditioning_set_values(
        pos_cond, {"reference_latents": [upscaled]}, append=True,
    )

    # KSampler — only regenerates the masked region
    (out_latent,) = common_ksampler(
        model, int(seed), int(steps), float(cfg),
        sampler_name, scheduler,
        pos_cond, neg_cond, start_latent, denoise=float(denoise),
    )

    # Tonemap + decode
    if tonemap_reinhard_multiplier > 0.0:
        out_latent = out_latent.copy()
        out_latent["samples"] = _tonemap_reinhard_latent(
            out_latent["samples"], float(tonemap_reinhard_multiplier),
        )
    rgb_out = _decode_to_rgb(vae, out_latent)

    # Resize decoded result back to crop dims
    rgb_out_resized = _resize_rgb_to(rgb_out, crop_h, crop_w)
    rgb_np = (rgb_out_resized[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    # Blend result back into current image using soft mask so edges are clean
    mask_f = crop_mask.astype(np.float32) / 255.0
    if mask_blend_pixels > 0:
        import cv2
        ksize = 2 * int(mask_blend_pixels) + 1
        mask_f = cv2.GaussianBlur(mask_f, (ksize, ksize), mask_blend_pixels * 0.5)
    mask_f = mask_f[..., None]  # (crop_h, crop_w, 1)

    blended = (crop_rgb.astype(np.float32) * (1.0 - mask_f)
               + rgb_np.astype(np.float32) * mask_f).clip(0, 255).astype(np.uint8)

    result = current_img_np.copy()
    result[ey1:ey2, ex1:ex2] = blended
    return result


# ── node ─────────────────────────────────────────────────────────────────────

class BD_PartsUnderpaint(io.ComfyNode):
    """
    Reveal what's beneath each removed part by inpainting the source image.
    Companion to BD_PartsBatchEdit — wire base_image output to BD_PartsExport
    so rebuilt part renders composite onto a clean base surface.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_PartsUnderpaint",
            display_name="BD Parts Underpaint (Qwen)",
            category="🧠BrainDead/Segmentation",
            description=(
                "Inpaint the source image where each part was removed, revealing what is "
                "naturally visible underneath — skin under clothing, car interior under panels, "
                "background behind an object, etc.\n\n"
                "Companion to BD_PartsBatchEdit: use both together —\n"
                "  BD_PartsBatchEdit  → rebuilt part renders (each part redrawn cleanly)\n"
                "  BD_PartsUnderpaint → base_image (source with all parts removed)\n"
                "Wire base_image into BD_PartsExport[base_image] so the rebuilt parts "
                "composite onto the clean bare surface.\n\n"
                "Two modes:\n"
                "  per_part_sequential — one Qwen pass per part in remove_order. Each "
                "pass result feeds the next, progressively stripping layers. Outputs "
                "intermediate frames as image_batch.\n"
                "  all_parts_combined — union all part masks, single Qwen pass. Faster "
                "but harder for Qwen to fill very large masked regions coherently."
            ),
            inputs=[
                io.Custom(PARTS_BUNDLE).Input("parts"),
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Image.Input(
                    "source_image",
                    tooltip="Original full-resolution source image (before any parts were removed). "
                            "Required — this is what gets inpainted.",
                ),
                io.Combo.Input(
                    "mode",
                    options=["per_part_sequential", "all_parts_combined"],
                    default="per_part_sequential",
                    tooltip="per_part_sequential — one Qwen inpaint call per part, processed in "
                            "remove_order (outer layers first). Each pass feeds the next so successive "
                            "layers reveal beneath each other. Gives the most detailed per-layer result "
                            "but takes N × Qwen inference time.\n\n"
                            "all_parts_combined — union all part masks into one, single Qwen call. "
                            "Faster. Best when parts don't overlap much. May struggle to inpaint very "
                            "large masked areas coherently.",
                ),
                io.String.Input(
                    "remove_order", default="", optional=True, multiline=True,
                    tooltip="Comma or newline separated list of tags specifying which parts to remove "
                            "and in what order (outer-to-inner). Missing = use bundle's dict order.\n"
                            "Example for a clothed character: jacket, shirt, pants\n"
                            "Tags not in this list that exist in the bundle are silently skipped "
                            "ONLY if they aren't in the bundle. Unknown tags are ignored.\n"
                            "Leave empty to process all parts in bundle order.",
                ),
                io.String.Input(
                    "prompt_template", default=(
                        "Remove the {tag} from this image. "
                        "Fill the region with whatever would naturally be visible underneath "
                        "or behind the {tag} in this scene — skin, clothing, background surface, "
                        "or interior. Preserve all surrounding pixels exactly."
                    ),
                    multiline=True,
                    tooltip="Prompt for per_part_sequential mode. {tag} is replaced with the part's "
                            "label. Adjust for your domain:\n"
                            "  Characters: 'Remove the {tag} and show the garment or bare skin underneath'\n"
                            "  Vehicles: 'Remove the {tag} and show the interior/chassis beneath'\n"
                            "  Products: 'Remove the {tag} and reveal the underlying surface'",
                ),
                io.String.Input(
                    "all_parts_prompt", default=(
                        "Remove all the highlighted parts from this image and reveal the complete "
                        "underlying surface — bare skin, undergarments, interior, or background — "
                        "as it would naturally appear with all parts absent."
                    ),
                    multiline=True, optional=True,
                    tooltip="Prompt used in all_parts_combined mode (single Qwen call, all masks unioned). "
                            "No {tag} substitution — describe the desired outcome directly.",
                ),
                io.Combo.Input(
                    "prefill_mode",
                    options=["color_spread", "nearest_neighbor", "neutral_gray", "white", "black", "none"],
                    default="color_spread",
                    tooltip="How to fill the masked region BEFORE Qwen sees it:\n"
                            "  color_spread (DEFAULT) — cv2 Telea inpaint: fills from surrounding "
                            "boundary colors. Gives Qwen a plausible color gradient to refine, which "
                            "typically produces the most natural 'underneath' texture.\n"
                            "  nearest_neighbor — Voronoi spread from nearest outside pixel. Sharper "
                            "color boundaries. Good for geometric shapes.\n"
                            "  neutral_gray — 50% gray fill. Safe baseline.\n"
                            "  white / black — flat fills. Black required for Qwen Inpaint LoRA "
                            "(prompt must start with 'Inpaint the black areas.').\n"
                            "  none — pass the raw crop as-is (Qwen sees existing occluders).",
                ),
                io.Float.Input(
                    "context_extend_factor", default=1.5, min=1.0, max=3.0, step=0.1, optional=True,
                    tooltip="How much surrounding context to give Qwen around each part's bbox. "
                            "1.0 = crop exactly to part bbox. 1.5 = 50% larger crop so Qwen sees "
                            "more neighboring skin/surface for better extrapolation. 2.0 = double. "
                            "Higher = better context but smaller part gets larger canvas proportion.",
                ),
                io.String.Input(
                    "skip_tags", multiline=True, default="", optional=True,
                    tooltip="Tags to skip entirely (csv or newline). Parts not in this list are still "
                            "processed. Useful to skip hair, background, or accessory parts that shouldn't "
                            "be removed before the underpaint.",
                ),
                io.Int.Input(
                    "mask_dilate_pixels", default=4, min=0, max=64, optional=True,
                    tooltip="Dilate the noise_mask (the Qwen inpaint region) by N pixels beyond the "
                            "exact part boundary. Gives Qwen a buffer zone so boundary pixels blend "
                            "cleanly with the surrounding context. 4-8 recommended.",
                ),
                io.Int.Input(
                    "mask_blend_pixels", default=4, min=0, max=32, optional=True,
                    tooltip="Soft-edge feather when pasting the Qwen result back into the source. "
                            "Prevents hard seams at the part boundary. 4-8 recommended.",
                ),
                io.Float.Input(
                    "latent_upscale_factor", default=1.25, min=1.0, max=2.0, step=0.05, optional=True,
                    tooltip="Upscale the encoded latent by this factor before KSampler (high-res-fix). "
                            "1.25 is a good default with the Lightning LoRA stack. 1.0 = no upscale.",
                ),
                io.Combo.Input(
                    "latent_upscale_method",
                    options=["nearest-exact", "bilinear", "area", "bicubic", "bislerp"],
                    default="nearest-exact",
                    tooltip="Interpolation for latent upscale.",
                ),
                io.Float.Input(
                    "model_sampling_shift", default=3.0, min=0.0, max=20.0, step=0.1, optional=True,
                    tooltip="ModelSamplingAuraFlow shift. Required for Lightning LoRA + latent upscale. "
                            "3.0 from the standard recipe. 0.0 = disabled.",
                ),
                io.Float.Input(
                    "cfg_norm_strength", default=0.85, min=0.0, max=2.0, step=0.05, optional=True,
                    tooltip="CFGNorm post-cfg function. 0.85 from recipe. 0.0 = disabled.",
                ),
                io.Float.Input(
                    "tonemap_reinhard_multiplier", default=2.0, min=0.0, max=10.0, step=0.1, optional=True,
                    tooltip="Reinhard tonemap on KSampler output before VAE decode. 2.0 from recipe. 0.0 = off.",
                ),
                io.Combo.Input(
                    "target_pixels",
                    options=["512x512", "768x768", "1024x1024", "1280x1280", "1536x1536", "custom"],
                    default="1024x1024",
                    tooltip="Working resolution for each Qwen inpaint pass.",
                ),
                io.Int.Input(
                    "target_pixels_custom", default=1048576, min=65536, max=4194304, optional=True,
                    tooltip="Custom pixel budget (used when target_pixels='custom').",
                ),
                io.String.Input(
                    "negative_prompt", multiline=True, default="", optional=True,
                    tooltip="Negative prompt (no effect at cfg=1.0).",
                ),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Int.Input("steps", default=4, min=1, max=100,
                             tooltip="4 for Lightning LoRA, 8-20 for non-Lightning."),
                io.Float.Input("cfg", default=1.0, min=0.0, max=20.0, step=0.1),
                io.Combo.Input("sampler_name", options=_samplers.KSampler.SAMPLERS, default="euler"),
                io.Combo.Input("scheduler", options=_samplers.KSampler.SCHEDULERS, default="simple"),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.05),
            ],
            outputs=[
                io.Custom(PARTS_BUNDLE).Output(display_name="parts",
                                               tooltip="Passthrough — unchanged PARTS_BUNDLE."),
                io.Image.Output(display_name="base_image",
                                tooltip="Source image with all processed parts inpainted away. "
                                        "Wire into BD_PartsExport[base_image] so rebuilt part renders "
                                        "composite onto this clean surface."),
                io.Image.Output(display_name="image_batch",
                                tooltip="(N, H, W, 3) batch of per-step intermediate frames for "
                                        "sequential mode. Frame 0 = after removing first tag, "
                                        "frame N-1 = base_image. Single frame for all_parts_combined."),
                io.String.Output(display_name="summary"),
            ],
        )

    @classmethod
    def execute(cls, parts, model, clip, vae, source_image,
                mode="per_part_sequential",
                remove_order="",
                prompt_template=(
                    "Remove the {tag} from this image. "
                    "Fill the region with whatever would naturally be visible underneath "
                    "or behind the {tag} in this scene — skin, clothing, background surface, "
                    "or interior. Preserve all surrounding pixels exactly."
                ),
                all_parts_prompt=(
                    "Remove all the highlighted parts from this image and reveal the complete "
                    "underlying surface — bare skin, undergarments, interior, or background — "
                    "as it would naturally appear with all parts absent."
                ),
                prefill_mode="color_spread",
                context_extend_factor=1.5,
                skip_tags="",
                mask_dilate_pixels=4,
                mask_blend_pixels=4,
                latent_upscale_factor=1.25,
                latent_upscale_method="nearest-exact",
                model_sampling_shift=3.0,
                cfg_norm_strength=0.85,
                tonemap_reinhard_multiplier=2.0,
                target_pixels="1024x1024",
                target_pixels_custom=1048576,
                negative_prompt="",
                seed=0, steps=4, cfg=1.0,
                sampler_name="euler", scheduler="simple", denoise=1.0,
                ) -> io.NodeOutput:
        import time as _time

        if parts is None:
            empty = empty_bundle()
            blank = torch.zeros(1, 64, 64, 3)
            return io.NodeOutput(empty, blank, blank, "skipped — no parts input")
        ensure_bundle(parts, source="BD_PartsUnderpaint.parts")

        tag2pinfo = parts["tag2pinfo"]
        skip_set = _parse_skip(skip_tags)

        # ── Convert source_image to uint8 numpy ───────────────────────────────
        src_t = source_image.detach().cpu().float()
        if src_t.dim() == 4:
            src_t = src_t[0]
        src_np = (src_t.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        if src_np.shape[-1] == 4:
            src_np = src_np[..., :3]
        src_h, src_w = src_np.shape[:2]

        # ── Apply model patches (same recipe as BD_PartsBatchEdit) ───────────
        patches = []
        if model_sampling_shift > 0.0:
            try:
                from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
                model = ModelSamplingAuraFlow().patch_aura(model, float(model_sampling_shift))[0]
                patches.append(f"shift={model_sampling_shift}")
            except Exception as e:
                print(f"[BD PartsUnderpaint] WARNING: ModelSamplingAuraFlow failed: {e}", flush=True)
        if cfg_norm_strength > 0.0:
            try:
                from comfy_extras.nodes_cfg import CFGNorm
                model = CFGNorm.execute(model, float(cfg_norm_strength)).args[0]
                patches.append(f"cfg_norm={cfg_norm_strength}")
            except Exception as e:
                print(f"[BD PartsUnderpaint] WARNING: CFGNorm failed: {e}", flush=True)
        if patches:
            print(f"[BD PartsUnderpaint] Model patches: {' '.join(patches)}", flush=True)

        # ── Resolve target pixel budget ───────────────────────────────────────
        if target_pixels == "custom":
            tp = int(target_pixels_custom)
        else:
            n = int(target_pixels.split("x")[0])
            tp = n * n

        # Negative conditioning (computed once)
        neg_cond = _encode_negative(clip, negative_prompt or "")

        # ── Shared kwargs for _run_underpaint_pass ────────────────────────────
        pass_kwargs = dict(
            model=model, clip=clip, vae=vae,
            neg_cond=neg_cond,
            context_extend_factor=float(context_extend_factor),
            prefill_mode=prefill_mode,
            latent_upscale_factor=float(latent_upscale_factor),
            latent_upscale_method=latent_upscale_method,
            model_sampling_shift=float(model_sampling_shift),
            cfg_norm_strength=float(cfg_norm_strength),
            tonemap_reinhard_multiplier=float(tonemap_reinhard_multiplier),
            tp=tp,
            seed=int(seed), steps=int(steps), cfg=float(cfg),
            sampler_name=sampler_name, scheduler=scheduler, denoise=float(denoise),
            mask_dilate_pixels=int(mask_dilate_pixels),
            mask_blend_pixels=int(mask_blend_pixels),
        )

        run_start = _time.time()
        intermediates = []
        processed_tags = []

        if mode == "per_part_sequential":
            tags = _parse_order(remove_order, tag2pinfo, skip_set)
            print(f"[BD PartsUnderpaint] Sequential mode: {len(tags)} parts — {tags}", flush=True)
            current_np = src_np.copy()

            for i, tag in enumerate(tags):
                info = tag2pinfo.get(tag)
                if not isinstance(info, dict) or info.get("img") is None:
                    print(f"[BD PartsUnderpaint] [{i+1}/{len(tags)}] '{tag}' — no img, skipping",
                          flush=True)
                    continue

                full_mask, xyxy = _build_part_mask_in_source(info, src_h, src_w)
                if not (full_mask > 127).any():
                    print(f"[BD PartsUnderpaint] [{i+1}/{len(tags)}] '{tag}' — empty mask, skipping",
                          flush=True)
                    continue

                prompt = prompt_template.replace("{tag}", tag).replace("%tag%", tag)
                t0 = _time.time()
                current_np = _run_underpaint_pass(
                    current_img_np=current_np, full_mask=full_mask, xyxy=xyxy,
                    prompt=prompt, **pass_kwargs,
                )
                dt = _time.time() - t0
                intermediates.append(current_np.copy())
                processed_tags.append(tag)
                mask_pct = 100.0 * (full_mask > 127).sum() / max(full_mask.size, 1)
                print(f"[BD PartsUnderpaint] [{i+1}/{len(tags)}] '{tag}' "
                      f"mask={mask_pct:.1f}% inpainted in {dt:.1f}s", flush=True)

        else:  # all_parts_combined
            tags = _parse_order(remove_order, tag2pinfo, skip_set)
            print(f"[BD PartsUnderpaint] Combined mode: unioning {len(tags)} masks", flush=True)
            union_mask = np.zeros((src_h, src_w), dtype=np.uint8)
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = src_w, src_h, 0, 0

            for tag in tags:
                info = tag2pinfo.get(tag)
                if not isinstance(info, dict) or info.get("img") is None:
                    continue
                full_mask, xyxy = _build_part_mask_in_source(info, src_h, src_w)
                union_mask = np.maximum(union_mask, full_mask)
                x1, y1, x2, y2 = xyxy
                bbox_x1 = min(bbox_x1, x1); bbox_y1 = min(bbox_y1, y1)
                bbox_x2 = max(bbox_x2, x2); bbox_y2 = max(bbox_y2, y2)
                processed_tags.append(tag)

            if (union_mask > 127).any() and processed_tags:
                xyxy_union = [bbox_x1, bbox_y1, bbox_x2, bbox_y2]
                t0 = _time.time()
                current_np = _run_underpaint_pass(
                    current_img_np=src_np.copy(), full_mask=union_mask,
                    xyxy=xyxy_union, prompt=all_parts_prompt or prompt_template,
                    **pass_kwargs,
                )
                dt = _time.time() - t0
                mask_pct = 100.0 * (union_mask > 127).sum() / max(union_mask.size, 1)
                print(f"[BD PartsUnderpaint] Combined inpaint of {len(processed_tags)} parts "
                      f"({mask_pct:.1f}% masked) in {dt:.1f}s", flush=True)
                intermediates.append(current_np.copy())
            else:
                print("[BD PartsUnderpaint] No valid masks found for combined mode", flush=True)
                current_np = src_np.copy()

        run_dt = _time.time() - run_start

        # ── Build output tensors ──────────────────────────────────────────────
        def np_to_tensor(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)

        if intermediates:
            base_t = np_to_tensor(intermediates[-1])  # final state
            # Pad all intermediates to the same H, W for batch
            max_h = max(a.shape[0] for a in intermediates)
            max_w = max(a.shape[1] for a in intermediates)
            frames = []
            for arr in intermediates:
                h, w = arr.shape[:2]
                if h == max_h and w == max_w:
                    frames.append(arr)
                else:
                    canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                    canvas[:h, :w] = arr
                    frames.append(canvas)
            batch_t = torch.from_numpy(
                np.stack(frames, axis=0).astype(np.float32) / 255.0
            )
        else:
            base_t = np_to_tensor(src_np)
            batch_t = base_t.clone()

        summary = (
            f"BD_PartsUnderpaint: {len(processed_tags)} part(s) inpainted in {run_dt:.1f}s\n"
            f"mode={mode} prefill={prefill_mode} context={context_extend_factor}x\n"
            f"Processed: {processed_tags}"
        )
        print(f"[BD PartsUnderpaint] {summary}", flush=True)
        return io.NodeOutput(parts, base_t, batch_t, summary)


PARTS_UNDERPAINT_V3_NODES = [BD_PartsUnderpaint]
PARTS_UNDERPAINT_NODES = {"BD_PartsUnderpaint": BD_PartsUnderpaint}
PARTS_UNDERPAINT_DISPLAY_NAMES = {"BD_PartsUnderpaint": "BD Parts Underpaint (Qwen)"}
