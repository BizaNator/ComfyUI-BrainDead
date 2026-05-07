"""
BD_PartsBatchEdit — Qwen Image Edit 2509 internal-loop edit over a PARTS_BUNDLE.

Single-execution node. Iterates every kept part inside one execute() call:
  for each part:
    - composite RGBA onto neutral background → RGB
    - build prompt via interpolation: "rebuild this {tag} in same style"
    - encode with Qwen-Edit-Plus encoder logic (VL tokens + reference_latent)
    - common_ksampler with the model + Lightning LoRA (4 steps, cfg=1.0)
    - VAE decode → RGB output
    - reapply original alpha → new RGBA
    - mutate tag2pinfo[tag]["img"]

Default sampler/cfg/steps target the Qwen Image Edit 2509 + 4-step Lightning
LoRA stack. Override per workflow if using a different stack.

NOTE: the encoder logic here is a faithful adaptation of comfy_extras/nodes_qwen.py
::TextEncodeQwenImageEditPlus.execute() — kept inline so we don't need to
re-construct a node-execution context per part.
"""

import math
import re

import numpy as np
import torch
from PIL import Image

import comfy.utils
import comfy.model_management
import node_helpers
from nodes import common_ksampler

from comfy_api.latest import io
import comfy.samplers as _samplers

from .parts_types import PARTS_BUNDLE, ensure_bundle


_LLAMA_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image "
    "(color, shape, size, texture, objects, background), then explain how the "
    "user's text instruction should alter or modify the image. Generate a new "
    "image that meets the user's requirements while maintaining consistency "
    "with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


def _rgba_to_rgb_on_neutral(rgba: np.ndarray, bg_value: float = 0.5) -> torch.Tensor:
    """Composite RGBA (HxWx4 uint8) onto neutral gray, return (1, H, W, 3) float [0,1]."""
    arr = rgba.astype(np.float32) / 255.0
    if arr.shape[-1] == 3:
        rgb = arr
    else:
        rgb_part = arr[..., :3]
        a = arr[..., 3:4]
        bg = np.full_like(rgb_part, bg_value, dtype=np.float32)
        rgb = rgb_part * a + bg * (1.0 - a)
    return torch.from_numpy(rgb.astype(np.float32)).unsqueeze(0)


def _extend_bbox(xyxy, factor: float, src_h: int, src_w: int) -> tuple[list[int], list[int]]:
    """Extend bbox by `factor` while keeping the same center, clipped to source dims.
    Returns (extended_xyxy, original_xyxy_relative_to_extended_crop).
    """
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2, y1 + bh / 2
    new_w, new_h = bw * factor, bh * factor
    ex1 = max(0, int(round(cx - new_w / 2)))
    ey1 = max(0, int(round(cy - new_h / 2)))
    ex2 = min(src_w, int(round(cx + new_w / 2)))
    ey2 = min(src_h, int(round(cy + new_h / 2)))
    # Where the original part bbox sits within the extended crop's coords:
    ox1 = x1 - ex1
    oy1 = y1 - ey1
    ox2 = ox1 + bw
    oy2 = oy1 + bh
    return [ex1, ey1, ex2, ey2], [ox1, oy1, ox2, oy2]


def _crop_source_to_bbox(source_image: torch.Tensor, xyxy) -> torch.Tensor:
    """Crop full source IMAGE (B, H, W, C) to the part's bbox. Returns (1, h, w, 3)."""
    if source_image.dim() == 4:
        src = source_image[0]
    else:
        src = source_image
    src_np = src.detach().cpu().numpy().astype(np.float32)
    if src_np.shape[-1] == 4:
        src_np = src_np[..., :3]
    H, W = src_np.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    x1 = max(0, min(x1, W)); x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H)); y2 = max(0, min(y2, H))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = src_np[y1:y2, x1:x2, :]
    return torch.from_numpy(crop.astype(np.float32)).unsqueeze(0)


def _detect_enclosed_holes(part_alpha_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect enclosed holes within the part shape.

    Uses scipy.ndimage.binary_fill_holes — a hole is a non-part region completely
    surrounded by the part. Pants with a gun-shaped cutout INSIDE = enclosed hole.
    Background outside the part outline = NOT enclosed (connected to image edge).
    Leg gap in pants = NOT enclosed (connected to outside).

    Returns (holes_mask_bool, filled_alpha_uint8):
      holes_mask_bool: True where there's an enclosed hole to inpaint
      filled_alpha_uint8: part_alpha with internal holes filled (= the corrected
                         part shape, used as alpha_after_edit='fill_holes')
    """
    try:
        from scipy.ndimage import binary_fill_holes
    except ImportError:
        return np.zeros_like(part_alpha_2d, dtype=bool), part_alpha_2d
    binary = part_alpha_2d > 128
    filled = binary_fill_holes(binary)
    holes = filled & ~binary
    filled_uint8 = (filled.astype(np.uint8) * 255)
    return holes, filled_uint8


def _prefill_occluders(source_crop: torch.Tensor, part_alpha_2d: np.ndarray,
                       inpaint_mask_bool: np.ndarray,
                       mode: str = "average_color") -> torch.Tensor:
    """Pre-fill ONLY the inpaint region (enclosed holes) with part-color.

    Background OUTSIDE the part outline is NOT modified — stays as source pixels.
    Only enclosed holes within the part get the prefill treatment.

    source_crop: (1, H, W, 3) float [0,1]
    part_alpha_2d: (H, W) uint8 — sample colors from where this is > 128
    inpaint_mask_bool: (H, W) bool — True where to prefill (enclosed holes only)
    Returns (1, H, W, 3) float [0,1].
    """
    if mode == "none" or not inpaint_mask_bool.any():
        return source_crop

    src = source_crop[0].detach().cpu().numpy().astype(np.float32)  # (H, W, 3)
    H, W = src.shape[:2]
    if part_alpha_2d.shape != (H, W):
        part_alpha_2d = np.asarray(
            Image.fromarray(part_alpha_2d, mode="L").resize((W, H), Image.NEAREST)
        )
    if inpaint_mask_bool.shape != (H, W):
        # Resize bool mask via nearest
        inp_uint = (inpaint_mask_bool.astype(np.uint8) * 255)
        inpaint_mask_bool = np.asarray(
            Image.fromarray(inp_uint, mode="L").resize((W, H), Image.NEAREST)
        ) > 128

    part_mask_bool = part_alpha_2d > 128
    if not part_mask_bool.any():
        return source_crop  # no part pixels to sample from

    if mode == "average_color":
        avg = src[part_mask_bool].mean(axis=0)  # (3,)
        src[inpaint_mask_bool] = avg
    elif mode == "nearest_part_pixel":
        try:
            from scipy.ndimage import distance_transform_edt
            # We want each hole pixel to take the nearest PART pixel's color.
            # distance_transform_edt of (~part_mask) returns indices of nearest part pixel.
            _, indices = distance_transform_edt(
                ~part_mask_bool, return_distances=True, return_indices=True
            )
            src[inpaint_mask_bool] = src[indices[0], indices[1]][inpaint_mask_bool]
        except ImportError:
            avg = src[part_mask_bool].mean(axis=0)
            src[inpaint_mask_bool] = avg
    elif mode == "neutral_gray":
        src[inpaint_mask_bool] = 0.5
    elif mode == "white":
        src[inpaint_mask_bool] = 1.0
    elif mode == "black":
        src[inpaint_mask_bool] = 0.0
    return torch.from_numpy(src).unsqueeze(0)


def _dilate_alpha(alpha_2d: np.ndarray, pixels: int) -> np.ndarray:
    """Binary-style dilate by N pixels using a max-filter (no scipy dependency)."""
    if pixels <= 0:
        return alpha_2d
    from PIL import ImageFilter
    pil = Image.fromarray(alpha_2d, mode="L")
    pil = pil.filter(ImageFilter.MaxFilter(size=2 * pixels + 1))
    return np.asarray(pil)


def _encode_qwen_edit_plus(clip, vae, prompt: str, image_rgb: torch.Tensor,
                           image_rgb_2: torch.Tensor | None = None,
                           target_pixels: int = 1024 * 1024):
    """Inline replica of TextEncodeQwenImageEditPlus for 1-2 image inputs.

    image_rgb (image1): primary reference (the part to rebuild).
    image_rgb_2 (image2, optional): secondary reference (depth, etc.) for
        Qwen's cross-attention guidance.

    Returns (positive_conditioning, ref_latent (full tensor of image1),
             (h_lat, w_lat) pixel dims of image1).
    """
    images = [image_rgb] + ([image_rgb_2] if image_rgb_2 is not None else [])
    images_vl = []
    ref_latents = []
    image_prompt = ""
    primary_latent = None
    primary_h_lat = None
    primary_w_lat = None

    for i, img in enumerate(images):
        samples = img.movedim(-1, 1)  # (1, 3, H, W)

        # VL token sizing: ~384x384 area
        total_vl = int(384 * 384)
        scale_vl = math.sqrt(total_vl / (samples.shape[3] * samples.shape[2]))
        w_vl = round(samples.shape[3] * scale_vl)
        h_vl = round(samples.shape[2] * scale_vl)
        s_vl = comfy.utils.common_upscale(samples, w_vl, h_vl, "area", "disabled")
        images_vl.append(s_vl.movedim(1, -1))

        # Ref latent sizing: target_pixels area, multiples of 8
        scale_lat = math.sqrt(target_pixels / (samples.shape[3] * samples.shape[2]))
        w_lat = round(samples.shape[3] * scale_lat / 8.0) * 8
        h_lat = round(samples.shape[2] * scale_lat / 8.0) * 8
        s_lat = comfy.utils.common_upscale(samples, w_lat, h_lat, "area", "disabled")
        ref_image = s_lat.movedim(1, -1)[:, :, :, :3]
        ref_latent = vae.encode(ref_image)
        ref_latents.append(ref_latent)

        if i == 0:
            primary_latent = ref_latent
            primary_h_lat, primary_w_lat = h_lat, w_lat

        image_prompt += f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>"

    tokens = clip.tokenize(image_prompt + prompt, images=images_vl,
                           llama_template=_LLAMA_TEMPLATE)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    conditioning = node_helpers.conditioning_set_values(
        conditioning, {"reference_latents": ref_latents}, append=True,
    )
    return conditioning, primary_latent, (primary_h_lat, primary_w_lat)


def _encode_negative(clip, prompt: str = ""):
    tokens = clip.tokenize(prompt, llama_template=_LLAMA_TEMPLATE)
    return clip.encode_from_tokens_scheduled(tokens)


def _interpolate_template(template: str, tag: str) -> str:
    """Resolve {tag} placeholders. Falls back to literal template if no {tag}."""
    return template.replace("{tag}", tag).replace("%tag%", tag)


def _decode_to_rgb(vae, latent_dict) -> torch.Tensor:
    img = vae.decode(latent_dict["samples"])
    if img.dim() == 5:
        img = img.reshape(-1, *img.shape[-3:])
    return img  # (1, H, W, 3) in [0, 1] approx


def _resize_alpha(alpha_2d: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if alpha_2d.shape == (target_h, target_w):
        return alpha_2d
    pil = Image.fromarray(alpha_2d, mode="L").resize((target_w, target_h), Image.BILINEAR)
    return np.asarray(pil)


def _resize_rgb_to(rgb_t: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    if rgb_t.shape[1] == target_h and rgb_t.shape[2] == target_w:
        return rgb_t
    samples = rgb_t.movedim(-1, 1)
    s = comfy.utils.common_upscale(samples, target_w, target_h, "lanczos", "disabled")
    return s.movedim(1, -1)


def _parse_skip(skip_tags: str) -> set[str]:
    return {s.strip() for s in re.split(r"[,\n]", skip_tags or "") if s.strip()}


class BD_PartsBatchEdit(io.ComfyNode):
    """Run Qwen Image Edit per part in a PARTS_BUNDLE within a single execution."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_PartsBatchEdit",
            display_name="BD Parts Batch Edit (Qwen)",
            category="🧠BrainDead/Segmentation",
            description=(
                "Iterate every part in a PARTS_BUNDLE inside one execution and run "
                "Qwen Image Edit per part. Mutates tag2pinfo[tag]['img'] with the "
                "edited result. Designed for the Qwen Image Edit 2509 fp8 stack with "
                "the 4-step Lightning LoRA — defaults match that setup (steps=4, "
                "cfg=1.0, sampler=euler, scheduler=simple).\n\n"
                "Each part's prompt is built by interpolating {tag} into prompt_template, "
                "e.g. 'rebuild this {tag} in the same style'. The part's RGBA is "
                "composited onto neutral gray for inference, then the original alpha "
                "is reapplied so edges stay clean.\n\n"
                "Skips parts whose tag matches skip_tags (csv/newline). For tags with "
                "tiny crops (below min_pixels) the edit is skipped and original kept."
            ),
            inputs=[
                io.Custom(PARTS_BUNDLE).Input("parts"),
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.String.Input(
                    "prompt_template", multiline=True,
                    default=("Recreate the complete {tag} in the same low-poly game style, "
                             "sharp clean edges, fill any occluded or missing regions, "
                             "no holes or transparent gaps inside the {tag}, "
                             "preserve original colors"),
                    tooltip="Per-part prompt. {tag} (or %tag%) is replaced with the part's tag. "
                            "Default is tuned for low-poly stylized characters with occluded items "
                            "(e.g. pants under a holster, shirt under a jacket). Adjust for other styles.",
                ),
                io.Image.Input(
                    "source_image", optional=True,
                    tooltip="Original source IMAGE (whole frame). Required when inpaint_mode='source_crop'. "
                            "Lets Qwen see the FULL visual context (e.g. pants WITH the gun on top) so it "
                            "can properly extract + complete the part underneath.",
                ),
                io.Combo.Input(
                    "inpaint_mode",
                    options=["true_inpaint", "source_crop", "masked_part"],
                    default="true_inpaint",
                    tooltip="true_inpaint (RECOMMENDED): latent-space inpaint with prefill. Source crop "
                            "is pre-filled (occluders replaced with part-color via prefill_mode), then "
                            "encoded as starting latent + reference. Inpaint mask says regenerate the "
                            "occluded regions. Qwen has NO information about the gun/holster/girl — only "
                            "sees the part with uniform patches → fills patches with detailed part. "
                            "Requires source_image wired.\n\n"
                            "source_crop: feed Qwen the unmasked source crop as reference; sample from "
                            "empty noise. Qwen sees occluders + tries to render the part — usually "
                            "reproduces source faithfully (occluders included). Use only for isolated parts.\n\n"
                            "masked_part: feed only the segmented part (composited on neutral gray) as "
                            "reference; sample from empty noise. Reproduces input including holes.",
                ),
                io.Combo.Input(
                    "prefill_mode",
                    options=["white", "nearest_part_pixel", "average_color",
                             "neutral_gray", "black", "none"],
                    default="white",
                    tooltip="How to fill HOLE pixels (enclosed cutouts inside the part) in the source "
                            "crop BEFORE Qwen sees it.\n"
                            "  white (DEFAULT): solid white. Empirically the best trigger for vanilla "
                            "Qwen Image Edit 2509 to recognize 'fill this region'. Matches BD_MaskFlatten "
                            "+ invert_mask + white_bg pattern.\n"
                            "  nearest_part_pixel: Voronoi color spread (preserves shading gradients).\n"
                            "  average_color: mean color of part pixels.\n"
                            "  neutral_gray: solid 50% gray.\n"
                            "  black: REQUIRED for Qwen Image Edit Inpaint LoRA (prompt must start "
                            "with 'Inpaint the black areas.'). Hard edges only.\n"
                            "  none: pass source as-is (Qwen sees occluders — usually wrong).",
                ),
                io.Boolean.Input(
                    "use_part_depth_as_image2", default=False, optional=True,
                    tooltip="If True AND the parts bundle has per-part depth (wire depth_image into "
                            "BD_PartsBuilder upstream), pass each part's pre-cropped depth to Qwen as "
                            "image2 (secondary reference) for cross-attention guidance. Note: Qwen "
                            "Image Edit isn't trained on depth as a structural input — this is a "
                            "weak signal. For real depth conditioning, use a depth ControlNet.",
                ),
                io.Float.Input(
                    "context_extend_factor", default=1.0, min=1.0, max=3.0, step=0.1, optional=True,
                    tooltip="CropAndStitch-style context expansion. Multiplier applied to each part's "
                            "bbox before cropping the source for Qwen. 1.0 = use part bbox as-is "
                            "(default). 1.5 = crop 50% bigger so Qwen sees more surrounding visual "
                            "context (skin tones, lighting, neighboring items). After Qwen output, "
                            "we crop back to the original part bbox region only — the extended pixels "
                            "are discarded but their context already informed the generation.",
                ),
                io.Int.Input(
                    "mask_expand_pixels", default=0, min=0, max=128, optional=True,
                    tooltip="CropAndStitch-style buffer zone. Dilate the noise_mask outward by N "
                            "pixels so existing part pixels at the boundary get partially regenerated "
                            "alongside the holes. Helps the model blend the regen smoothly with the "
                            "preserved area. 4-8 is a good starting range. 0 = mask exactly matches "
                            "the detected holes (sharp boundary).",
                ),
                io.Int.Input(
                    "mask_blend_pixels", default=4, min=0, max=64, optional=True,
                    tooltip="Soft-edge alpha feathering on the FINAL output. Blurs the alpha by N "
                            "pixels at edges so cutouts blend cleanly with whatever they're composited "
                            "over. 0 = sharp PNG cutouts. 4-8 = soft anti-aliased edges.",
                ),
                io.Combo.Input(
                    "target_pixels",
                    options=["512x512", "768x768", "1024x1024", "1280x1280", "1536x1536", "custom"],
                    default="1024x1024",
                    tooltip="Working resolution for Qwen's VAE encode. Each part's bbox is scaled to "
                            "approximately this total area (preserving aspect, dims rounded to multiples of 8). "
                            "1024² = ~1MP, default for fp8 stability. 1536² = ~2.36MP, Qwen's max native "
                            "resolution (best quality, ~1.4× slower). Output is resized back to the part's "
                            "original crop size after inference.",
                ),
                io.Int.Input(
                    "target_pixels_custom", default=1048576, min=65536, max=4194304, optional=True,
                    tooltip="Used when target_pixels='custom'. Total pixel budget (e.g. 1048576 = 1024²).",
                ),
                io.String.Input(
                    "negative_prompt", multiline=True, default="", optional=True,
                    tooltip="Negative prompt (used only when cfg > 1.0). Empty string fine for cfg=1.0.",
                ),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                             tooltip="Same seed used per part. Set to 0 for randomness; "
                                     "fix for reproducible runs."),
                io.Int.Input("steps", default=4, min=1, max=100,
                             tooltip="Sampling steps. 4 for Lightning LoRA. 8-20 for non-Lightning."),
                io.Float.Input("cfg", default=1.0, min=0.0, max=20.0, step=0.1,
                               tooltip="Classifier-free guidance scale. 1.0 for Lightning (no negative used)."),
                io.Combo.Input("sampler_name",
                               options=_samplers.KSampler.SAMPLERS,
                               default="euler",
                               tooltip="Lightning LoRA was trained with euler — keep matching."),
                io.Combo.Input("scheduler",
                               options=_samplers.KSampler.SCHEDULERS,
                               default="simple",
                               tooltip="Lightning LoRA expects simple scheduler."),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="1.0 = full denoise (typical for edit). Lower = preserve more of the input latent shape."),
                io.String.Input("skip_tags", multiline=True, default="", optional=True,
                                tooltip="Tags to skip entirely (csv or newline). Original images preserved."),
                io.Int.Input("min_pixels", default=64, min=1, max=10_000_000, optional=True,
                             tooltip="Skip parts whose RGBA crop is smaller than sqrt(N)x sqrt(N)."),
                io.Combo.Input(
                    "alpha_after_edit",
                    options=["fill_holes", "original_part", "original_dilated", "bbox_full"],
                    default="fill_holes",
                    tooltip="How to compute the part's alpha AFTER the edit:\n"
                            "  fill_holes (DEFAULT): part shape with internal enclosed holes filled "
                            "(detected via flood-fill). Pants outline preserved, gun-shaped hole inside "
                            "becomes opaque. Background outside part stays transparent. Use this for "
                            "everything — it produces clean part-shaped cutouts.\n"
                            "  original_part: cut exactly to SAM3's original mask. Holes stay transparent.\n"
                            "  original_dilated: original_part + N px dilation (edge feathering only).\n"
                            "  bbox_full: entire crop opaque. Use only for debugging/inspection.",
                ),
                io.Int.Input("mask_dilate_pixels", default=2, min=0, max=64, optional=True,
                             tooltip="Dilation amount when alpha_after_edit='original_dilated' or "
                                     "'inpaint_filled' (applied as edge-feather to soften the cut)."),
            ],
            outputs=[
                io.Custom(PARTS_BUNDLE).Output(display_name="parts"),
                io.String.Output(display_name="summary"),
                io.Image.Output(display_name="image_batch"),
            ],
        )

    @classmethod
    def execute(cls, parts, model, clip, vae,
                prompt_template="rebuild this {tag} in the same style",
                source_image=None, inpaint_mode="true_inpaint",
                prefill_mode="white",
                use_part_depth_as_image2=False,
                context_extend_factor=1.0,
                mask_expand_pixels=0, mask_blend_pixels=4,
                target_pixels="1024x1024", target_pixels_custom=1048576,
                negative_prompt="", seed=0, steps=4, cfg=1.0,
                sampler_name="euler", scheduler="simple", denoise=1.0,
                skip_tags="", min_pixels=64,
                alpha_after_edit="fill_holes",
                mask_dilate_pixels=2) -> io.NodeOutput:
        import time as _time
        ensure_bundle(parts, source="BD_PartsBatchEdit.parts")

        skip_set = _parse_skip(skip_tags)
        tag2pinfo = parts["tag2pinfo"]

        # Resolve target pixel budget
        if target_pixels == "custom":
            tp = int(target_pixels_custom)
        else:
            n = int(target_pixels.split("x")[0])
            tp = n * n

        # Resolve effective inpaint mode
        effective_mode = inpaint_mode
        if effective_mode in ("true_inpaint", "source_crop") and source_image is None:
            effective_mode = "masked_part"
            print(f"[BD PartsBatchEdit] inpaint_mode={inpaint_mode} requires source_image; "
                  f"falling back to masked_part", flush=True)

        # Negative conditioning (computed once, reused per part)
        neg_cond = _encode_negative(clip, negative_prompt or "")

        edited = []
        skipped_pix = []
        skipped_tag = []
        total_parts = sum(1 for tag, info in tag2pinfo.items()
                          if isinstance(info, dict) and info.get("img") is not None
                          and tag not in skip_set)
        part_idx = 0
        run_start = _time.time()

        for tag, info in tag2pinfo.items():
            if not isinstance(info, dict):
                continue
            if tag in skip_set:
                skipped_tag.append(tag)
                continue
            img = info.get("img")
            if img is None:
                continue
            arr = np.asarray(img)
            if arr.ndim != 3 or arr.shape[2] not in (3, 4):
                continue
            h, w = arr.shape[:2]
            if h * w < min_pixels:
                skipped_pix.append((tag, h * w))
                continue

            # Stash original alpha for reapply
            if arr.shape[2] == 4:
                orig_alpha = arr[..., 3].copy()
            else:
                orig_alpha = np.full((h, w), 255, dtype=np.uint8)

            # Compute extended bbox if context_extend_factor > 1.0.
            # Working alpha for extended crop is the original alpha placed at the
            # right offset within the larger crop, with surrounding zeros.
            xyxy_orig = info["xyxy"]
            if context_extend_factor > 1.0 and source_image is not None:
                src_h_full = (source_image.shape[1] if source_image.dim() == 4
                              else source_image.shape[0])
                src_w_full = (source_image.shape[2] if source_image.dim() == 4
                              else source_image.shape[1])
                ext_xyxy, orig_in_ext = _extend_bbox(
                    xyxy_orig, float(context_extend_factor), src_h_full, src_w_full,
                )
                ex_w = ext_xyxy[2] - ext_xyxy[0]
                ex_h = ext_xyxy[3] - ext_xyxy[1]
                # Build extended alpha: zeros except where original part lives
                work_alpha = np.zeros((ex_h, ex_w), dtype=np.uint8)
                ox1, oy1, ox2, oy2 = orig_in_ext
                work_alpha[oy1:oy2, ox1:ox2] = orig_alpha
            else:
                ext_xyxy = xyxy_orig
                orig_in_ext = [0, 0, w, h]
                work_alpha = orig_alpha

            # Detect ENCLOSED HOLES inside the part (gun-shaped cutout inside pants
            # outline, etc.). Background outside the part is NOT a hole.
            holes_bool, filled_alpha = _detect_enclosed_holes(work_alpha)
            has_holes = holes_bool.any()

            # Choose Qwen reference image (cropped to extended bbox)
            if effective_mode in ("true_inpaint", "source_crop"):
                ref_t = _crop_source_to_bbox(source_image, ext_xyxy)
                if ref_t is None:
                    # Bbox out of source bounds — fall back
                    ref_t = _rgba_to_rgb_on_neutral(arr, bg_value=0.5)
                # For true_inpaint: pre-fill ONLY enclosed holes (not whole non-part)
                if effective_mode == "true_inpaint" and prefill_mode != "none" and has_holes:
                    ref_t = _prefill_occluders(ref_t, work_alpha, holes_bool, mode=prefill_mode)
            else:
                ref_t = _rgba_to_rgb_on_neutral(arr, bg_value=0.5)

            # Build per-part prompt
            prompt = _interpolate_template(prompt_template, tag)

            # Optional secondary depth reference (image2) — read from parts dict
            # (PartsBuilder stores per-part depth under info['depth'] when wired).
            depth_ref_t = None
            if use_part_depth_as_image2:
                depth_arr = info.get("depth")
                if depth_arr is not None:
                    depth_np = np.asarray(depth_arr).astype(np.float32) / 255.0
                    if depth_np.ndim == 2:
                        depth_np = np.stack([depth_np] * 3, axis=-1)  # gray → RGB for Qwen
                    depth_ref_t = torch.from_numpy(depth_np).unsqueeze(0)

            # Encode positive (with reference_latent attached) at target resolution
            pos_cond, ref_latent, (h_lat, w_lat) = _encode_qwen_edit_plus(
                clip, vae, prompt, ref_t, image_rgb_2=depth_ref_t, target_pixels=tp,
            )

            # Build the starting latent.
            # - true_inpaint: samples = source latent, noise_mask = inverse of part alpha
            #   (= holes/occlusions). KSampler regenerates only those pixels.
            # - source_crop / masked_part: empty noise, full denoise — Qwen reproduces
            #   the whole crop as text-conditioned synthesis.
            if effective_mode == "true_inpaint":
                # Inpaint mask = enclosed holes (+ optional buffer dilation).
                # If no holes detected, fall back to "regen everything inside the
                # part outline" so the part still gets a quality refresh per user
                # request (no skip — every part goes through KSampler).
                if has_holes:
                    inpaint_seed_2d = holes_bool.astype(np.uint8) * 255
                else:
                    # No enclosed holes — regen the whole filled-part shape so Qwen
                    # at least cleans/refines the whole region at full target res.
                    inpaint_seed_2d = filled_alpha
                if mask_expand_pixels > 0:
                    inpaint_seed_2d = _dilate_alpha(inpaint_seed_2d, int(mask_expand_pixels))
                pil_mask = Image.fromarray(inpaint_seed_2d, mode="L").resize(
                    (w_lat, h_lat), Image.NEAREST,
                )
                noise_mask_2d = np.asarray(pil_mask).astype(np.float32) / 255.0
                noise_mask_t = torch.from_numpy(noise_mask_2d).unsqueeze(0)
                start_latent = {
                    "samples": ref_latent.to(dtype=torch.float32),
                    "noise_mask": noise_mask_t,
                }
            else:
                start_latent = {"samples": torch.zeros(
                    ref_latent.shape, dtype=torch.float32,
                    device=comfy.model_management.intermediate_device(),
                )}

            # Always sample — every part goes through KSampler at full target
            # resolution for the quality lift, even if no enclosed holes exist.
            t0 = _time.time()
            (out_latent,) = common_ksampler(
                model, int(seed), int(steps), float(cfg),
                sampler_name, scheduler,
                pos_cond, neg_cond, start_latent, denoise=float(denoise),
            )
            dt = _time.time() - t0

            # Decode — keep at WORKING resolution (don't resize back to crop dims).
            # PartsCompose / PartsExport scale at paint time, so high-res pixels survive.
            rgb_out = _decode_to_rgb(vae, out_latent)
            out_h, out_w = rgb_out.shape[1], rgb_out.shape[2]

            # If we extended the bbox, crop the output back to ONLY the original
            # part region. Map original_in_ext (in source pixels) → output pixels.
            if context_extend_factor > 1.0:
                ex_w = ext_xyxy[2] - ext_xyxy[0]
                ex_h = ext_xyxy[3] - ext_xyxy[1]
                sx = out_w / max(ex_w, 1)
                sy = out_h / max(ex_h, 1)
                ox1, oy1, ox2, oy2 = orig_in_ext
                cy1 = int(round(oy1 * sy))
                cy2 = int(round(oy2 * sy))
                cx1 = int(round(ox1 * sx))
                cx2 = int(round(ox2 * sx))
                rgb_out = rgb_out[:, cy1:cy2, cx1:cx2, :]
                out_h, out_w = rgb_out.shape[1], rgb_out.shape[2]
                # Update work_alpha and filled_alpha to original-only region for
                # the alpha computation below.
                work_alpha = orig_alpha
                _, filled_alpha = _detect_enclosed_holes(orig_alpha)
            rgb_np = (rgb_out[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            # Build alpha — always sourced from ORIGINAL part dims, since by this
            # point rgb_out has been cropped back to the original part region.
            if alpha_after_edit == "fill_holes":
                _, filled_orig = _detect_enclosed_holes(orig_alpha)
                src_alpha = filled_orig
            elif alpha_after_edit == "bbox_full":
                src_alpha = np.full(orig_alpha.shape, 255, dtype=np.uint8)
            elif alpha_after_edit == "original_dilated":
                src_alpha = (_dilate_alpha(orig_alpha, int(mask_dilate_pixels))
                             if mask_dilate_pixels > 0 else orig_alpha)
            else:  # original_part
                src_alpha = orig_alpha

            alpha = np.asarray(
                Image.fromarray(src_alpha, mode="L").resize((out_w, out_h), Image.BILINEAR)
            )
            # Optional soft-edge blend for cleaner cutouts when composited
            if mask_blend_pixels > 0:
                from PIL import ImageFilter
                pil_a = Image.fromarray(alpha, mode="L").filter(
                    ImageFilter.GaussianBlur(radius=float(mask_blend_pixels))
                )
                alpha = np.asarray(pil_a)
            rgba_new = np.concatenate([rgb_np, alpha[..., None]], axis=-1)

            # Mutate the bundle (img now at working resolution; xyxy unchanged)
            info["img"] = rgba_new
            edited.append((tag, rgba_new))
            part_idx += 1
            holes_pct = 100.0 * holes_bool.sum() / max(work_alpha.size, 1)
            depth_str = " +depth_ref" if depth_ref_t is not None else ""
            extras = []
            if context_extend_factor > 1.0:
                extras.append(f"context={context_extend_factor:.1f}x")
            if effective_mode == "true_inpaint":
                extras.append(f"prefill={prefill_mode}")
            if mask_expand_pixels > 0:
                extras.append(f"expand={mask_expand_pixels}px")
            if mask_blend_pixels > 0:
                extras.append(f"blend={mask_blend_pixels}px")
            extras_str = " " + " ".join(extras) if extras else ""
            print(f"[BD PartsBatchEdit] [{part_idx}/{total_parts}] '{tag}' "
                  f"crop={w}x{h} → working={w_lat}x{h_lat} holes={holes_pct:.1f}% "
                  f"mode={effective_mode}{depth_str}{extras_str} "
                  f"alpha={alpha_after_edit} ksampler={dt:.1f}s",
                  flush=True)

        run_dt = _time.time() - run_start

        # Build batch IMAGE output of all edited crops (padded to common max size)
        if edited:
            max_h = max(rgba.shape[0] for _, rgba in edited)
            max_w = max(rgba.shape[1] for _, rgba in edited)
            batch_imgs = []
            for _, rgba in edited:
                h, w = rgba.shape[:2]
                # Center-pad to (max_h, max_w) on transparent black
                pad_top = (max_h - h) // 2
                pad_left = (max_w - w) // 2
                canvas = np.zeros((max_h, max_w, 4), dtype=np.uint8)
                canvas[pad_top:pad_top + h, pad_left:pad_left + w] = rgba
                batch_imgs.append(canvas)
            batch_arr = np.stack(batch_imgs, axis=0).astype(np.float32) / 255.0
            batch_t = torch.from_numpy(batch_arr)  # (N, H, W, 4)
        else:
            batch_t = torch.zeros((1, 1, 1, 4), dtype=torch.float32)

        edited_tags = [t for t, _ in edited]
        summary = (
            f"BD_PartsBatchEdit: edited {len(edited_tags)} of {total_parts} parts in {run_dt:.1f}s | "
            f"mode={effective_mode} target={target_pixels} alpha={alpha_after_edit} | "
            f"skipped tags: {skipped_tag} | "
            f"skipped <{min_pixels}px: {[s[0] for s in skipped_pix]}\n"
            f"Edited: {edited_tags}"
        )
        print(f"[BD PartsBatchEdit] {summary}", flush=True)
        return io.NodeOutput(parts, summary, batch_t)


PARTS_BATCH_EDIT_V3_NODES = [BD_PartsBatchEdit]
PARTS_BATCH_EDIT_NODES = {"BD_PartsBatchEdit": BD_PartsBatchEdit}
PARTS_BATCH_EDIT_DISPLAY_NAMES = {"BD_PartsBatchEdit": "BD Parts Batch Edit (Qwen)"}
