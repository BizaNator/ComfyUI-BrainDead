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


def _alpha_from_white_bg(rgb_np: np.ndarray, white_threshold: float = 0.85,
                         saturation_threshold: float = 0.10,
                         soft_edge_px: int = 2,
                         corner_sample: bool = True) -> np.ndarray:
    """Auto-derive alpha from "background-colored" pixels in Qwen's output.

    Two-stage detection:
    1. Hard threshold: pixels with luminance > white_threshold AND low saturation
       → flagged as background (treat as if white).
    2. Corner sampling (when enabled): sample mean color of the four corners
       (assumed to be background), then mark pixels close to that color as
       background too. Catches off-white / slightly-tinted backgrounds.
    Pixels matching either stage → alpha 0. Everything else → alpha 255.
    """
    rgb_f = rgb_np.astype(np.float32) / 255.0
    H, W = rgb_f.shape[:2]
    lum = rgb_f.mean(axis=-1)
    saturation = rgb_f.max(axis=-1) - rgb_f.min(axis=-1)
    is_white = (lum > white_threshold) & (saturation < saturation_threshold)

    if corner_sample and H > 8 and W > 8:
        # Sample 8x8 patches at each of the four corners
        patch = 8
        corners = np.concatenate([
            rgb_f[:patch, :patch].reshape(-1, 3),
            rgb_f[:patch, -patch:].reshape(-1, 3),
            rgb_f[-patch:, :patch].reshape(-1, 3),
            rgb_f[-patch:, -patch:].reshape(-1, 3),
        ], axis=0)
        bg_color = np.median(corners, axis=0)  # robust against the rare corner-content case
        # If sampled bg is bright + low-sat, treat as background reference and
        # also flag pixels close to it as transparent.
        bg_lum = bg_color.mean()
        bg_sat = bg_color.max() - bg_color.min()
        if bg_lum > 0.75 and bg_sat < 0.15:
            diff = np.abs(rgb_f - bg_color[None, None, :]).max(axis=-1)
            is_bg = diff < 0.10
            is_white = is_white | is_bg

    alpha = np.where(is_white, 0, 255).astype(np.uint8)
    if soft_edge_px > 0:
        from PIL import ImageFilter
        pil = Image.fromarray(alpha, mode="L").filter(
            ImageFilter.GaussianBlur(radius=float(soft_edge_px))
        )
        alpha = np.asarray(pil)
    return alpha


def _crop_rgba_to_content(rgba: np.ndarray, alpha_threshold: int = 32,
                          padding_px: int = 0) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop RGBA to the bbox of opaque content (alpha > threshold).
    Returns (cropped_rgba, (x1, y1, x2, y2) within the original).
    If no opaque content found, returns the original unchanged.
    """
    if rgba.shape[2] < 4:
        return rgba, (0, 0, rgba.shape[1], rgba.shape[0])
    alpha = rgba[..., 3]
    rows = np.any(alpha > alpha_threshold, axis=1)
    cols = np.any(alpha > alpha_threshold, axis=0)
    if not rows.any() or not cols.any():
        return rgba, (0, 0, rgba.shape[1], rgba.shape[0])
    y1, y2 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    x1, x2 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    H, W = rgba.shape[:2]
    y1 = max(0, y1 - padding_px); x1 = max(0, x1 - padding_px)
    y2 = min(H, y2 + padding_px); x2 = min(W, x2 + padding_px)
    return rgba[y1:y2, x1:x2], (x1, y1, x2, y2)


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

    target_long = int(round(math.sqrt(float(target_pixels))))
    for i, img in enumerate(images):
        samples = img.movedim(-1, 1)  # (1, 3, H, W)

        # VL token sizing: ~384x384 area (this is for the embedding only — keep area sizing)
        total_vl = int(384 * 384)
        scale_vl = math.sqrt(total_vl / (samples.shape[3] * samples.shape[2]))
        w_vl = round(samples.shape[3] * scale_vl)
        h_vl = round(samples.shape[2] * scale_vl)
        s_vl = comfy.utils.common_upscale(samples, w_vl, h_vl, "area", "disabled")
        images_vl.append(s_vl.movedim(1, -1))

        # Ref latent sizing: longest edge = target_long (BOTH cap and floor — small
        # parts get upscaled, oversized parts get downscaled). Every part gets edited
        # at Qwen's full target resolution for best detail. After the latent_upscale
        # step the output ends up at target_long * latent_upscale.
        in_h, in_w = samples.shape[2], samples.shape[3]
        longest = max(in_h, in_w)
        scale_lat = target_long / max(longest, 1)
        w_lat = max(64, round(in_w * scale_lat / 8.0) * 8)
        h_lat = max(64, round(in_h * scale_lat / 8.0) * 8)
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


def _encode_text_only_with_ref(clip, vae, prompt: str, image_rgb: torch.Tensor,
                                target_pixels: int = 1024 * 1024,
                                min_short_edge: int = 256):
    """Plain CLIPTextEncode + VAE encode + ReferenceLatent pattern.

    NO VL image tokens. Matches the user's manual flatten_redraw recipe.

    Uses LONGEST-EDGE sizing (not area sizing) so thin parts like beards
    don't get squished into 1-px-tall strips. target_pixels is interpreted
    as the target longest edge (sqrt of the area param).
    Ensures shortest edge >= min_short_edge to keep latents non-degenerate.

    Returns (conditioning, ref_latent, (h_lat_pixels, w_lat_pixels)).
    """
    samples = image_rgb.movedim(-1, 1)  # (1, 3, H, W)
    in_h, in_w = samples.shape[2], samples.shape[3]
    target_long = int(round(math.sqrt(float(target_pixels))))

    # 1) scale so longest edge = target_long
    longest = max(in_h, in_w)
    scale = target_long / max(longest, 1)
    h0 = int(round(in_h * scale))
    w0 = int(round(in_w * scale))

    # 2) ensure shortest edge >= min_short_edge (don't let thin parts collapse)
    shortest = min(h0, w0)
    if shortest < min_short_edge:
        scale_up = min_short_edge / max(shortest, 1)
        h0 = int(round(h0 * scale_up))
        w0 = int(round(w0 * scale_up))

    # 3) round to multiples of 8 for VAE
    w_lat = max(64, round(w0 / 8.0) * 8)
    h_lat = max(64, round(h0 / 8.0) * 8)

    s_lat = comfy.utils.common_upscale(samples, w_lat, h_lat, "area", "disabled")
    ref_image = s_lat.movedim(1, -1)[:, :, :, :3]
    ref_latent = vae.encode(ref_image)

    # Plain text tokenize, no images param → no VL embeddings
    tokens = clip.tokenize(prompt)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    conditioning = node_helpers.conditioning_set_values(
        conditioning, {"reference_latents": [ref_latent]}, append=True,
    )
    return conditioning, ref_latent, (h_lat, w_lat)


def _interpolate_template(template: str, tag: str) -> str:
    """Resolve {tag} placeholders. Falls back to literal template if no {tag}."""
    return template.replace("{tag}", tag).replace("%tag%", tag)


def _latent_spatial_dims(latent: torch.Tensor) -> tuple[int, int]:
    """Return (H, W) from a latent tensor, handling both 4D and Qwen's 5D shape.

    Qwen Image latents are (B, C, layers+1, H, W) — 5D. Standard SD/Flux are
    (B, C, H, W) — 4D. Use this helper instead of indexing shape[2/3] directly.
    """
    if latent.dim() == 5:
        return int(latent.shape[3]), int(latent.shape[4])
    return int(latent.shape[2]), int(latent.shape[3])


def _upscale_latent_spatial(latent: torch.Tensor, new_h: int, new_w: int,
                            method: str) -> torch.Tensor:
    """Upscale only the spatial dims (H, W) of a latent. Preserves Qwen's
    5D layer/temporal dim by collapsing it into batch for the upscale and
    reshaping back. For 4D latents, just calls common_upscale."""
    if latent.dim() == 5:
        B, C, T, H, W = latent.shape
        # (B, C, T, H, W) → (B*T, C, H, W) → upscale → (B*T, C, new_h, new_w) → (B, C, T, new_h, new_w)
        flat = latent.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        up = comfy.utils.common_upscale(flat, new_w, new_h, method, "disabled")
        return up.reshape(B, T, C, new_h, new_w).permute(0, 2, 1, 3, 4)
    return comfy.utils.common_upscale(latent, new_w, new_h, method, "disabled")


def _tonemap_reinhard_latent(latent: torch.Tensor, multiplier: float) -> torch.Tensor:
    """Reinhard tonemap on a latent — copied from comfy_extras/nodes_latent.py.
    Compresses the latent value range to prevent overshoot at upscaled dims."""
    eps = 1e-10
    latent_vector_magnitude = (torch.linalg.vector_norm(latent, dim=1) + eps)[:, None]
    normalized_latent = latent / latent_vector_magnitude
    dims = list(range(1, latent_vector_magnitude.ndim))
    mean = torch.mean(latent_vector_magnitude, dim=dims, keepdim=True)
    std = torch.std(latent_vector_magnitude, dim=dims, keepdim=True)
    top = (std * 5 + mean) * multiplier
    latent_vector_magnitude = latent_vector_magnitude * (1.0 / top)
    new_magnitude = latent_vector_magnitude / (latent_vector_magnitude + 1.0)
    new_magnitude = new_magnitude * top
    return normalized_latent * new_magnitude


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
                    options=["flatten_redraw", "true_inpaint"],
                    default="flatten_redraw",
                    tooltip="flatten_redraw (DEFAULT): flatten part RGBA on white bg → Qwen Edit Plus "
                            "encoder (positive + negative both with image1) → VAE encode → upscale "
                            "latent by latent_upscale_factor → APPEND upscaled to ref_latents → KSampler "
                            "→ Reinhard tonemap → decode. Generates a clean object render. Default alpha "
                            "(fill_holes) cuts to original part outline so the composite assembles "
                            "without white backgrounds. Internal patches: shift=3.0 + cfg_norm=0.85 + "
                            "tonemap=2.0 — pair with 4-step Lightning LoRA on the input model.\n\n"
                            "true_inpaint: latent-space inpaint with prefill. Source crop pre-filled, "
                            "noise_mask = enclosed holes. Qwen regenerates only the holes, preserves "
                            "the rest. Best for completing obstructed parts (pants under holster). "
                            "Requires source_image wired.",
                ),
                io.Float.Input(
                    "latent_upscale_factor", default=1.25, min=1.0, max=2.0, step=0.05, optional=True,
                    tooltip="Used by flatten_redraw mode. After encoding the part to latent, upscale "
                            "by this factor before sampling. Forces the model to denoise/refine "
                            "(high-res-fix trick). 1.25 is a good default; 1.0 = no upscale.",
                ),
                io.Combo.Input(
                    "latent_upscale_method",
                    options=["nearest-exact", "bilinear", "area", "bicubic", "bislerp"],
                    default="nearest-exact",
                    tooltip="Interpolation for latent upscale. nearest-exact preserves encoded values; "
                            "bilinear/bicubic interpolate values that VAE decodes as noise stripes.",
                ),
                io.Float.Input(
                    "model_sampling_shift", default=3.0, min=0.0, max=20.0, step=0.1, optional=True,
                    tooltip="Apply ModelSamplingAuraFlow shift internally. REQUIRED for flatten_redraw "
                            "+ latent upscale + Lightning LoRA combo — without it Lightning's noise "
                            "schedule doesn't match the upscaled latent and you get noise stripes "
                            "instead of clean output. 3.0 is what the user's manual recipe uses. "
                            "0.0 = disabled (use only if you've already wired ModelSamplingAuraFlow "
                            "externally on the model input).",
                ),
                io.Float.Input(
                    "cfg_norm_strength", default=0.85, min=0.0, max=2.0, step=0.05, optional=True,
                    tooltip="Apply CFGNorm post-cfg function internally. Pairs with the shift patch "
                            "to keep guidance stable on Lightning + upscale. 0.85 from user recipe. "
                            "0.0 = disabled (use only if already wired externally).",
                ),
                io.Float.Input(
                    "tonemap_reinhard_multiplier", default=2.0, min=0.0, max=10.0, step=0.1,
                    optional=True,
                    tooltip="Reinhard tonemap on the KSampler output latent (POST-sampling, BEFORE "
                            "VAE decode). Compresses latent value range to prevent overshoot when "
                            "running Lightning at upscaled dims. 2.0 from user recipe. 0.0 = disabled.",
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
                io.Float.Input(
                    "context_extend_factor", default=1.0, min=1.0, max=3.0, step=0.1, optional=True,
                    tooltip="(true_inpaint) CropAndStitch-style context expansion. Multiplier on the "
                            "part bbox before cropping the source for Qwen. 1.0 = use part bbox as-is. "
                            "1.5 = crop 50% bigger so Qwen sees more surrounding visual context.",
                ),
                io.Float.Input(
                    "flatten_pad_factor", default=1.25, min=1.0, max=2.5, step=0.05, optional=True,
                    tooltip="(flatten_redraw) White-padding around the part image BEFORE encoding. "
                            "Qwen tends to render content edge-to-edge; padding gives it breathing "
                            "room so the rendered object isn't cropped at the canvas boundary. 1.25 "
                            "= 25% extra white on all sides. 1.0 = no padding (Qwen will fill the "
                            "frame). The canvas dim grows by this factor before encoder fits to "
                            "target_pixels, so net detail per part is similar.",
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
                    options=["auto_from_white_bg", "fill_holes", "original_part",
                             "original_dilated", "bbox_full"],
                    default="auto_from_white_bg",
                    tooltip="How to compute the part's alpha AFTER the edit:\n"
                            "  auto_from_white_bg (DEFAULT): detect alpha from white pixels in Qwen's "
                            "output → white = transparent. The full generated shape is preserved "
                            "(no SAM3-mask cropping that would re-introduce holes). Cropped to "
                            "non-white content bbox so the rendered object fits its original part bbox "
                            "when composited (instead of being lost in white space). Pants obstructed "
                            "by shirt come out complete with no shirt-shaped chunks missing.\n"
                            "  fill_holes: part shape with internal enclosed holes flood-filled — "
                            "cuts generated content to original SAM3 outline + internal fills.\n"
                            "  original_part: cut exactly to SAM3's mask. Holes stay transparent.\n"
                            "  original_dilated: original_part + N px dilation (edge feathering).\n"
                            "  bbox_full: entire crop opaque. Debugging only.",
                ),
                io.Float.Input(
                    "white_threshold", default=0.85, min=0.5, max=1.0, step=0.01, optional=True,
                    tooltip="Used by alpha_after_edit=auto_from_white_bg. Pixels with luminance > "
                            "this value AND low saturation are treated as white background → alpha 0. "
                            "Also auto-samples the 4 corner patches as a bg reference and flags "
                            "pixels close to that color as transparent (catches off-white tints). "
                            "Lower = more aggressive.",
                ),
                io.Boolean.Input(
                    "auto_crop_to_content", default=False, optional=True,
                    tooltip="When alpha_after_edit=auto_from_white_bg: crop the output RGBA tight "
                            "to the bbox of opaque (non-white) content. Off by default — when Qwen "
                            "renders content touching the canvas edges, cropping looks like the "
                            "object is cut off. Leave OFF to preserve the full rendered output; "
                            "composite still resizes to fit the part's bbox correctly via the alpha.",
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
                source_image=None, inpaint_mode="flatten_redraw",
                latent_upscale_factor=1.25, latent_upscale_method="nearest-exact",
                model_sampling_shift=3.0, cfg_norm_strength=0.85,
                tonemap_reinhard_multiplier=2.0,
                prefill_mode="white",
                context_extend_factor=1.0,
                flatten_pad_factor=1.25,
                mask_expand_pixels=0, mask_blend_pixels=4,
                target_pixels="1024x1024", target_pixels_custom=1048576,
                negative_prompt="", seed=0, steps=4, cfg=1.0,
                sampler_name="euler", scheduler="simple", denoise=1.0,
                skip_tags="", min_pixels=64,
                alpha_after_edit="auto_from_white_bg",
                white_threshold=0.85,
                auto_crop_to_content=False,
                mask_dilate_pixels=2) -> io.NodeOutput:
        import time as _time
        ensure_bundle(parts, source="BD_PartsBatchEdit.parts")

        skip_set = _parse_skip(skip_tags)
        tag2pinfo = parts["tag2pinfo"]

        # Apply model patches once (matches user's manual recipe for Lightning + latent upscale).
        # ModelSamplingAuraFlow shift adjusts the noise schedule so Lightning denoises properly
        # at non-standard latent shapes. CFGNorm keeps guidance stable. Both are no-op if 0.0.
        patches_applied = []
        if model_sampling_shift > 0.0:
            try:
                from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
                model = ModelSamplingAuraFlow().patch_aura(model, float(model_sampling_shift))[0]
                patches_applied.append(f"shift={model_sampling_shift}")
            except Exception as e:
                print(f"[BD PartsBatchEdit] WARNING: ModelSamplingAuraFlow patch failed: {e}", flush=True)
        if cfg_norm_strength > 0.0:
            try:
                # CFGNorm v3 ComfyNode — call its execute classmethod
                from comfy_extras.nodes_cfg import CFGNorm
                model = CFGNorm.execute(model, float(cfg_norm_strength)).args[0]
                patches_applied.append(f"cfg_norm={cfg_norm_strength}")
            except Exception as e:
                print(f"[BD PartsBatchEdit] WARNING: CFGNorm patch failed: {e}", flush=True)
        if patches_applied:
            print(f"[BD PartsBatchEdit] Internal model patches: {' '.join(patches_applied)}", flush=True)

        # Resolve target pixel budget
        if target_pixels == "custom":
            tp = int(target_pixels_custom)
        else:
            n = int(target_pixels.split("x")[0])
            tp = n * n

        # Resolve effective inpaint mode
        effective_mode = inpaint_mode
        if effective_mode in ("true_inpaint", "source_crop") and source_image is None:
            effective_mode = "flatten_redraw"
            print(f"[BD PartsBatchEdit] inpaint_mode={inpaint_mode} requires source_image; "
                  f"falling back to flatten_redraw", flush=True)

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
            # Wrap in try/except so per-part bbox/alpha failures fall back to no
            # extension instead of killing the whole run.
            xyxy_orig = info["xyxy"]
            ext_failed = False
            if context_extend_factor > 1.0 and source_image is not None:
                try:
                    src_h_full = (source_image.shape[1] if source_image.dim() == 4
                                  else source_image.shape[0])
                    src_w_full = (source_image.shape[2] if source_image.dim() == 4
                                  else source_image.shape[1])
                    ext_xyxy, orig_in_ext = _extend_bbox(
                        xyxy_orig, float(context_extend_factor), src_h_full, src_w_full,
                    )
                    ex_w = ext_xyxy[2] - ext_xyxy[0]
                    ex_h = ext_xyxy[3] - ext_xyxy[1]
                    if ex_w <= 0 or ex_h <= 0:
                        raise ValueError(f"extended bbox has non-positive dim: {ext_xyxy}")
                    work_alpha = np.zeros((ex_h, ex_w), dtype=np.uint8)
                    ox1, oy1, ox2, oy2 = orig_in_ext
                    # Defensive clamp of slice indices to valid alpha-target dims
                    ox1 = max(0, min(ox1, ex_w)); ox2 = max(0, min(ox2, ex_w))
                    oy1 = max(0, min(oy1, ex_h)); oy2 = max(0, min(oy2, ex_h))
                    if (oy2 - oy1) != orig_alpha.shape[0] or (ox2 - ox1) != orig_alpha.shape[1]:
                        # Resize original alpha to fit clamped target slot
                        orig_resized = np.asarray(
                            Image.fromarray(orig_alpha, mode="L").resize(
                                (ox2 - ox1, oy2 - oy1), Image.NEAREST,
                            )
                        )
                        work_alpha[oy1:oy2, ox1:ox2] = orig_resized
                    else:
                        work_alpha[oy1:oy2, ox1:ox2] = orig_alpha
                    orig_in_ext = [ox1, oy1, ox2, oy2]
                except Exception as e:
                    print(f"[BD PartsBatchEdit] WARNING: context_extend_factor={context_extend_factor} "
                          f"failed for '{tag}' ({e}); falling back to no extension.", flush=True)
                    ext_failed = True

            if context_extend_factor <= 1.0 or source_image is None or ext_failed:
                ext_xyxy = xyxy_orig
                orig_in_ext = [0, 0, w, h]
                work_alpha = orig_alpha

            # Detect ENCLOSED HOLES inside the part (gun-shaped cutout inside pants
            # outline, etc.). Background outside the part is NOT a hole.
            holes_bool, filled_alpha = _detect_enclosed_holes(work_alpha)
            has_holes = holes_bool.any()

            # Choose Qwen reference image
            if effective_mode == "flatten_redraw":
                # Flatten the part RGBA on WHITE bg — clean isolated object on white,
                # no source context, no occluders. Matches user's manual recipe.
                ref_t = _rgba_to_rgb_on_neutral(arr, bg_value=1.0)
                # Pad with white on all sides so Qwen has breathing room and doesn't
                # render edge-to-edge (which makes the object look cropped).
                if flatten_pad_factor > 1.0:
                    _, ph, pw, _ = ref_t.shape
                    new_h = int(round(ph * flatten_pad_factor))
                    new_w = int(round(pw * flatten_pad_factor))
                    pad_t = (new_h - ph) // 2
                    pad_l = (new_w - pw) // 2
                    canvas = torch.ones((1, new_h, new_w, 3), dtype=ref_t.dtype)
                    canvas[:, pad_t:pad_t + ph, pad_l:pad_l + pw, :] = ref_t
                    ref_t = canvas
            elif effective_mode in ("true_inpaint", "source_crop"):
                ref_t = _crop_source_to_bbox(source_image, ext_xyxy)
                if ref_t is None:
                    ref_t = _rgba_to_rgb_on_neutral(arr, bg_value=0.5)
                # For true_inpaint: pre-fill ONLY enclosed holes (not whole non-part)
                if effective_mode == "true_inpaint" and prefill_mode != "none" and has_holes:
                    ref_t = _prefill_occluders(ref_t, work_alpha, holes_bool, mode=prefill_mode)
            else:
                # masked_part: composite RGBA on neutral gray
                ref_t = _rgba_to_rgb_on_neutral(arr, bg_value=0.5)

            # Build per-part prompt
            prompt = _interpolate_template(prompt_template, tag)

            # Encode: both modes use Qwen Edit Plus (with VL image tokens).
            # flatten_redraw additionally encodes a NEGATIVE conditioning the same
            # way (image1 + empty prompt) — matches user's manual recipe.
            pos_cond, ref_latent, (h_lat, w_lat) = _encode_qwen_edit_plus(
                clip, vae, prompt, ref_t, target_pixels=tp,
            )
            neg_cond_local = neg_cond
            if effective_mode == "flatten_redraw":
                neg_cond_local, _, _ = _encode_qwen_edit_plus(
                    clip, vae, "", ref_t, target_pixels=tp,
                )

            # Build the starting latent.
            # - true_inpaint: samples = source latent, noise_mask = inverse of part alpha
            #   (= holes/occlusions). KSampler regenerates only those pixels.
            # - source_crop / masked_part: empty noise, full denoise — Qwen reproduces
            #   the whole crop as text-conditioned synthesis.
            if effective_mode == "flatten_redraw":
                # User's manual workflow exactly:
                # - VAEEncode → LatentUpscaleBy(1.25, nearest-exact) → KSampler.latent_image
                # - ReferenceLatent APPENDS the upscaled latent to positive conditioning
                #   (so positive has 2 reference_latents: base from encoder + upscaled)
                # - Negative also has the encoder's image-derived ref_latent (no append)
                if latent_upscale_factor > 1.0:
                    cur_h, cur_w = _latent_spatial_dims(ref_latent)
                    new_lat_h = int(round(cur_h * latent_upscale_factor))
                    new_lat_w = int(round(cur_w * latent_upscale_factor))
                    upscaled = _upscale_latent_spatial(
                        ref_latent, new_lat_h, new_lat_w, latent_upscale_method,
                    )
                    h_lat = new_lat_h * 8
                    w_lat = new_lat_w * 8
                else:
                    upscaled = ref_latent
                start_latent = {"samples": upscaled.to(dtype=torch.float32)}
                # APPEND (not replace) — matches the ReferenceLatent node behavior
                pos_cond = node_helpers.conditioning_set_values(
                    pos_cond, {"reference_latents": [upscaled]}, append=True,
                )
            elif effective_mode == "true_inpaint":
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
                pos_cond, neg_cond_local, start_latent, denoise=float(denoise),
            )
            dt = _time.time() - t0

            # POST-KSampler tonemap (matches user's manual recipe — applied after
            # sampling, before VAE decode). Compresses latent value range; critical
            # for clean decode at upscaled dims with Lightning.
            if effective_mode == "flatten_redraw" and tonemap_reinhard_multiplier > 0.0:
                out_latent = out_latent.copy()
                out_latent["samples"] = _tonemap_reinhard_latent(
                    out_latent["samples"], float(tonemap_reinhard_multiplier),
                )

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

            # Build alpha. auto_from_white_bg derives directly from the Qwen output;
            # other modes scale the original mask up to the output dims.
            # NOTE: auto_from_white_bg + true_inpaint produces an inverted alpha
            # (true_inpaint output is the full source crop, not an object on white).
            # Don't auto-redirect — let the user choose explicitly per their use case.
            effective_alpha_mode = alpha_after_edit

            if effective_alpha_mode == "auto_from_white_bg":
                alpha = _alpha_from_white_bg(
                    rgb_np,
                    white_threshold=float(white_threshold),
                    soft_edge_px=int(mask_blend_pixels),
                )
            else:
                if effective_alpha_mode == "fill_holes":
                    _, filled_orig = _detect_enclosed_holes(orig_alpha)
                    src_alpha = filled_orig
                elif effective_alpha_mode == "bbox_full":
                    src_alpha = np.full(orig_alpha.shape, 255, dtype=np.uint8)
                elif effective_alpha_mode == "original_dilated":
                    src_alpha = (_dilate_alpha(orig_alpha, int(mask_dilate_pixels))
                                 if mask_dilate_pixels > 0 else orig_alpha)
                else:  # original_part
                    src_alpha = orig_alpha
                alpha = np.asarray(
                    Image.fromarray(src_alpha, mode="L").resize((out_w, out_h), Image.BILINEAR)
                )
            # Optional soft-edge blend for cleaner cutouts when composited.
            # auto_from_white_bg already feathered internally — skip double blur.
            if mask_blend_pixels > 0 and effective_alpha_mode != "auto_from_white_bg":
                from PIL import ImageFilter
                pil_a = Image.fromarray(alpha, mode="L").filter(
                    ImageFilter.GaussianBlur(radius=float(mask_blend_pixels))
                )
                alpha = np.asarray(pil_a)
            rgba_new = np.concatenate([rgb_np, alpha[..., None]], axis=-1)

            # Optional content-bbox crop. Off by default — when Qwen renders content
            # touching the canvas edges, the crop looks like the object is cut off.
            if effective_alpha_mode == "auto_from_white_bg" and auto_crop_to_content:
                rgba_new, _content_bbox = _crop_rgba_to_content(
                    rgba_new, alpha_threshold=32, padding_px=4,
                )

            # Stash the ORIGINAL SAM3 alpha (pre-edit) so PartsExport can save it
            # as a separate mask layer/file for users who want to re-apply the
            # original visibility cut. Resize to match the rebuilt img dims.
            if rgba_new.shape[:2] != orig_alpha.shape:
                orig_resized = np.asarray(
                    Image.fromarray(orig_alpha, mode="L").resize(
                        (rgba_new.shape[1], rgba_new.shape[0]), Image.BILINEAR,
                    )
                )
            else:
                orig_resized = orig_alpha
            info["original_alpha"] = orig_resized

            # Mutate the bundle (img now at working resolution; xyxy unchanged)
            info["img"] = rgba_new
            edited.append((tag, rgba_new))
            part_idx += 1
            holes_pct = 100.0 * holes_bool.sum() / max(work_alpha.size, 1)
            extras = []
            if effective_mode == "flatten_redraw":
                if latent_upscale_factor > 1.0:
                    extras.append(f"latent_up={latent_upscale_factor:.2f}x")
                if tonemap_reinhard_multiplier > 0.0:
                    extras.append(f"tonemap={tonemap_reinhard_multiplier:.1f}")
            if context_extend_factor > 1.0 and not ext_failed:
                extras.append(f"context={context_extend_factor:.1f}x")
            if effective_mode == "true_inpaint":
                extras.append(f"prefill={prefill_mode}")
            if mask_expand_pixels > 0:
                extras.append(f"expand={mask_expand_pixels}px")
            if mask_blend_pixels > 0:
                extras.append(f"blend={mask_blend_pixels}px")
            extras_str = " " + " ".join(extras) if extras else ""
            alpha_str = effective_alpha_mode if effective_alpha_mode == alpha_after_edit \
                        else f"{alpha_after_edit}→{effective_alpha_mode}"
            print(f"[BD PartsBatchEdit] [{part_idx}/{total_parts}] '{tag}' "
                  f"crop={w}x{h} → working={w_lat}x{h_lat} holes={holes_pct:.1f}% "
                  f"mode={effective_mode}{extras_str} "
                  f"alpha={alpha_str} ksampler={dt:.1f}s",
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
