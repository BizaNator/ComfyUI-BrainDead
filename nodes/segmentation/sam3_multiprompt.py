"""
SAM3 Multi-Prompt — run text-grounded SAM3 once per prompt and combine the masks.

Replaces a chain of SAM3Segment nodes for multi-region segmentation tasks like
"global skin mask" (skin / face / arm / leg / hand / foot / neck) where you'd
otherwise wire 7-12 SAM3 nodes manually and combine with mask-OR nodes.

Wraps comfyui-rmbg's SAM3Segment class — same model, same accuracy, same VRAM.
The only thing that changes is UI and one-shot batched execution.
"""

import sys

import numpy as np
import torch

from comfy_api.latest import io


_RMBG_PATH = "/opt/comfyui/dev/custom_nodes/comfyui-rmbg"


def _import_sam3_segment():
    """Lazy import — folder_paths only available when ComfyUI is running."""
    if _RMBG_PATH not in sys.path:
        sys.path.insert(0, _RMBG_PATH)
    from AILab_SAM3Segment import SAM3Segment
    return SAM3Segment


def _skin_tone_likelihood(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel HSV-based skin tone likelihood (0..1).

    Ported from the GLSL Mask Resolver shader's `skinToneLikelihood` function.
    rgb: (H, W, 3) float in [0, 1]
    returns: (H, W) float in [0, 1]
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    val = cmax
    sat = np.where(cmax > 1e-3, delta / np.maximum(cmax, 1e-3), 0.0)

    hue = np.zeros_like(r)
    mask_r = (cmax == r) & (delta > 1e-3)
    mask_g = (cmax == g) & (delta > 1e-3) & (~mask_r)
    mask_b = (cmax == b) & (delta > 1e-3) & (~mask_r) & (~mask_g)
    safe_delta = np.maximum(delta, 1e-6)
    hue[mask_r] = ((g[mask_r] - b[mask_r]) / safe_delta[mask_r])
    hue[mask_g] = (2.0 + (b[mask_g] - r[mask_g]) / safe_delta[mask_g])
    hue[mask_b] = (4.0 + (r[mask_b] - g[mask_b]) / safe_delta[mask_b])
    hue = np.mod(hue / 6.0, 1.0)

    def _smoothstep(edge0, edge1, x):
        t = np.clip((x - edge0) / (edge1 - edge0 + 1e-9), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    hue_score = _smoothstep(0.14, 0.10, hue) + _smoothstep(0.92, 0.96, hue)
    hue_score = np.clip(hue_score, 0.0, 1.0)
    sat_score = _smoothstep(0.05, 0.15, sat) * _smoothstep(0.85, 0.65, sat)
    val_score = _smoothstep(0.12, 0.25, val) * _smoothstep(0.98, 0.88, val)
    rgb_order = ((b <= g) & (g <= r)).astype(np.float32) * 0.3
    return np.clip(hue_score * sat_score * val_score + rgb_order, 0.0, 1.0)


def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation by `radius` pixels using max pooling."""
    if radius <= 0:
        return mask
    if mask.ndim == 2:
        m = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        m = mask.unsqueeze(1)
    else:
        m = mask
    k = max(1, 2 * radius + 1)
    pad = radius
    dilated = torch.nn.functional.max_pool2d(m.float(), kernel_size=k, stride=1, padding=pad)
    if mask.ndim == 2:
        return dilated.squeeze(0).squeeze(0)
    if mask.ndim == 3:
        return dilated.squeeze(1)
    return dilated


def _apply_skin_color_filter(mask: torch.Tensor, image: torch.Tensor,
                             mode: str = "off",
                             threshold: float = 0.3,
                             strength: float = 0.7,
                             silhouette_mask: torch.Tensor | None = None,
                             include_dilate_radius: int = 64) -> torch.Tensor:
    """Color-aware mask refinement using HSV-based skin tone likelihood.

    mode:
      "off"
      "exclude"             — mask * (floor + (1-floor)*likelihood). floor = 1 - strength.
      "exclude_hard"        — mask zeroed where likelihood < threshold.
      "include"             — ADD skin-toned pixels to mask, constrained to silhouette OR a dilation
                              of the existing mask (so we don't grab background).
      "exclude_and_include" — apply exclude THEN include.
    """
    if mode == "off":
        return mask
    img_np = image.detach().cpu().numpy()
    if img_np.ndim == 4:
        img_np = img_np[0]
    likelihood = _skin_tone_likelihood(img_np[..., :3])
    likelihood_t = torch.from_numpy(likelihood).to(mask.device).to(mask.dtype)
    if mask.ndim == 3 and likelihood_t.ndim == 2:
        likelihood_t = likelihood_t.unsqueeze(0)

    floor = max(0.0, 1.0 - float(strength))

    def _candidate_zone():
        if silhouette_mask is not None:
            sil_mean = float(silhouette_mask.float().mean().item())
            sil_max = float(silhouette_mask.float().max().item())
            if sil_mean >= 0.005 and sil_max >= 0.1:
                sil = silhouette_mask.to(mask.device).to(mask.dtype)
                if sil.ndim == 2:
                    sil = sil.unsqueeze(0)
                if sil.shape[-2:] != mask.shape[-2:]:
                    sil = torch.nn.functional.interpolate(
                        sil.unsqueeze(0).float(), size=mask.shape[-2:], mode="nearest"
                    ).squeeze(0)
                return (sil > 0.5).to(mask.dtype)
        return _dilate_mask((mask > 0.05).to(mask.dtype), include_dilate_radius)

    if mode == "exclude":
        return mask * (floor + (1.0 - floor) * likelihood_t)
    if mode == "exclude_hard":
        return mask * (likelihood_t >= threshold).to(mask.dtype)
    if mode == "include":
        zone = _candidate_zone()
        skin_in_zone = ((likelihood_t >= threshold).to(mask.dtype) * zone)
        return torch.maximum(mask, skin_in_zone)
    if mode == "exclude_and_include":
        excluded = mask * (floor + (1.0 - floor) * likelihood_t)
        zone = _candidate_zone()
        included = ((likelihood_t >= threshold).to(mask.dtype) * zone)
        return torch.maximum(excluded, included)
    return mask


def _combine_masks(masks: list[torch.Tensor], mode: str) -> torch.Tensor:
    if not masks:
        raise ValueError("No masks to combine")
    if len(masks) == 1 or mode == "first_only":
        return masks[0]
    if mode == "union":
        out = masks[0]
        for m in masks[1:]:
            out = torch.maximum(out, m)
        return out
    if mode == "intersection":
        out = masks[0]
        for m in masks[1:]:
            out = torch.minimum(out, m)
        return out
    if mode == "subtract_first":
        out = masks[0].clone()
        for m in masks[1:]:
            out = torch.clamp(out - m, 0.0, 1.0)
        return out
    raise ValueError(f"Unknown combine_mode: {mode}")


class BD_SAM3MultiPrompt(io.ComfyNode):
    """Run SAM3 once per prompt line, combine results in one node."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_SAM3MultiPrompt",
            display_name="BD SAM3 Multi-Prompt",
            category="🧠BrainDead/Segmentation",
            description=(
                "Run SAM3 once per prompt line (one per row) and combine the masks. "
                "Replaces a chain of SAM3Segment nodes for multi-region segmentation. "
                "Wraps comfyui-rmbg's SAM3Segment — same model, same VRAM, batched execution."
            ),
            inputs=[
                io.Image.Input("image"),
                io.String.Input(
                    "prompts", multiline=True,
                    default="skin\nface\nneck\narm\nleg\nhand\nfoot",
                    tooltip="One positive prompt per line. SAM3 runs once per non-empty line.",
                ),
                io.String.Input(
                    "negative_prompts", multiline=True, default="", optional=True,
                    tooltip="One negative prompt per line. SAM3 runs each, then their union is SUBTRACTED from the combined positive mask. "
                            "Use to explicitly remove regions, e.g. 'clothing\\nglove\\nshoe' to peel non-skin off a body mask.",
                ),
                io.Combo.Input(
                    "combine_mode",
                    options=["union", "intersection", "subtract_first", "first_only"],
                    default="union", optional=True,
                    tooltip="union = OR all positive masks (typical for 'all skin'). "
                            "intersection = AND all (regions present in every prompt). "
                            "subtract_first = first mask MINUS all others (built-in negation, alt to negative_prompts). "
                            "first_only = ignore combine, just return prompt 1.",
                ),
                io.Combo.Input(
                    "skin_color_filter",
                    options=["off", "exclude", "exclude_hard", "include", "exclude_and_include"],
                    default="off", optional=True,
                    tooltip=(
                        "Color-aware mask refinement using HSV skin-tone detection.\n"
                        "  off — no filtering.\n"
                        "  exclude — REMOVE non-skin-toned pixels from the SAM3 mask (suppresses bleed).\n"
                        "  exclude_hard — same but binary threshold (use color_strength=1).\n"
                        "  include — REPLACE the SAM3 mask with skin-toned pixels (good when SAM3 missed obvious skin).\n"
                        "  exclude_and_include — keep SAM3's mask AND add skin pixels SAM3 missed (best of both)."
                    ),
                ),
                io.Float.Input(
                    "color_strength", default=0.7, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="How aggressive 'exclude' is. 0 = no effect, 1 = fully suppress non-skin pixels (old behavior). "
                            "Try 0.5 for stylized art that doesn't match strict HSV skin ranges.",
                ),
                io.Float.Input(
                    "skin_color_threshold", default=0.3, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="Threshold for binary modes (exclude_hard, include). Pixels with skin-tone likelihood "
                            "above this count as skin. Lower for stylized art (try 0.15-0.25).",
                ),
                io.Mask.Input(
                    "silhouette_mask", optional=True,
                    tooltip="Optional character outline mask. When provided AND enforce_silhouette=True, "
                            "the final combined_mask is multiplied by this silhouette as the LAST step — "
                            "any SAM3 false positives outside the character (e.g. detecting 'skin' in the "
                            "background) are zeroed. Also constrains the color filter's 'include' zone to "
                            "within the silhouette.",
                ),
                io.Boolean.Input(
                    "enforce_silhouette", default=True, optional=True,
                    tooltip="When True (and silhouette_mask is provided AND non-blank): EVERY SAM3 result "
                            "(positive and negative) is multiplied by the silhouette immediately after inference, "
                            "BEFORE union/subtract. So per_prompt_masks, combined_mask, and masked_image all "
                            "reflect 'what SAM3 found INSIDE the character only' — background noise is removed "
                            "everywhere, not just the final output. Set False to see raw SAM3 output including background hallucinations.",
                ),
                io.Int.Input(
                    "include_dilate_radius", default=64, min=0, max=512, step=8, optional=True,
                    tooltip="When 'include' has no silhouette_mask: how far (in pixels) to dilate the SAM3 mask "
                            "to define the 'candidate zone' for adding skin pixels. Higher = pulls in more "
                            "distant skin (good for filling chest/torso when SAM3 only got arms/legs).",
                ),
                io.Combo.Input(
                    "masked_image_bg",
                    options=["white", "black", "transparent", "checker"],
                    default="white", optional=True,
                    tooltip="Background composited under the mask in masked_image. "
                            "masked_image is literally combined_mask used as alpha to blend image over this color "
                            "(transparent = RGBA output with mask as alpha channel). "
                            "Continuous mask values produce continuous alpha blending — no threshold trickery.",
                ),
                io.Boolean.Input(
                    "invert_negatives_in_per_prompt", default=True, optional=True,
                    tooltip="When True, negative prompts in per_prompt_masks are inverted before batching: "
                            "white = pixels KEPT after subtraction, black = pixels REMOVED. Makes it visually "
                            "obvious which areas each negative is excluding when previewed as a batch. "
                            "Set False to get raw 'white-where-detected' for both positives and negatives.",
                ),
                io.Float.Input(
                    "confidence_threshold", default=0.5, min=0.0, max=1.0, step=0.05, optional=True,
                ),
                io.Combo.Input("device", options=["Auto", "CPU", "GPU"], default="Auto", optional=True),
                io.Int.Input("mask_blur", default=0, min=0, max=64, step=1, optional=True),
                io.Int.Input("mask_offset", default=0, min=-64, max=64, step=1, optional=True),
                io.Boolean.Input(
                    "unload_model", default=False, optional=True,
                    tooltip="Unload SAM3 from VRAM after this node finishes. Useful when SAM3 won't be used again this Run.",
                ),
                io.Boolean.Input(
                    "invert_combined", default=False, optional=True,
                    tooltip="Invert the final combined mask. Useful for 'everything except skin' workflows.",
                ),
            ],
            outputs=[
                io.Mask.Output(display_name="combined_mask"),
                io.Image.Output(display_name="masked_image"),
                io.Mask.Output(display_name="per_prompt_masks"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, image, prompts, negative_prompts="", combine_mode="union",
                skin_color_filter="off", color_strength=0.7, skin_color_threshold=0.3,
                silhouette_mask=None, enforce_silhouette=True,
                include_dilate_radius=64,
                masked_image_bg="white",
                invert_negatives_in_per_prompt=True,
                confidence_threshold=0.5, device="Auto",
                mask_blur=0, mask_offset=0, unload_model=False,
                invert_combined=False) -> io.NodeOutput:
        positive_list = [p.strip() for p in prompts.strip().split("\n") if p.strip()]
        negative_list = [p.strip() for p in (negative_prompts or "").strip().split("\n") if p.strip()]
        if not positive_list:
            raise ValueError("BD_SAM3MultiPrompt: no non-empty positive prompts provided")

        SAM3Segment = _import_sam3_segment()
        sam3 = SAM3Segment()

        coverage_lines = []

        # Pre-validate silhouette ONCE so we don't re-check per prompt
        sil_active = None
        if silhouette_mask is not None and enforce_silhouette:
            sil_mean = float(silhouette_mask.float().mean().item())
            sil_max = float(silhouette_mask.float().max().item())
            if sil_mean >= 0.005 and sil_max >= 0.1:
                sil_active = silhouette_mask.float()
                if sil_active.ndim == 2:
                    sil_active = sil_active.unsqueeze(0)
                coverage_lines.append(
                    f"  [silhouette] applying to every SAM3 result (sil_mean={sil_mean:.3f})"
                )
            else:
                coverage_lines.append(
                    f"  [silhouette] BLANK — ignored (mean={sil_mean:.4f}, max={sil_max:.3f})"
                )

        def _run_sam3(prompt: str, last_in_chain: bool):
            unload_now = unload_model if last_in_chain else False
            _, mask, _ = sam3.segment(
                image=image,
                prompt=prompt,
                sam3_model="sam3",
                device=device,
                confidence_threshold=confidence_threshold,
                mask_blur=mask_blur,
                mask_offset=mask_offset,
                invert_output=False,
                unload_model=unload_now,
                background="Alpha",
                background_color="#000000",
            )
            if mask.ndim == 4:
                mask = mask.squeeze(0)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if sil_active is not None:
                sil = sil_active.to(mask.device).to(mask.dtype)
                if sil.shape[-2:] != mask.shape[-2:]:
                    sil = torch.nn.functional.interpolate(
                        sil.unsqueeze(0), size=mask.shape[-2:], mode="nearest"
                    ).squeeze(0)
                mask = mask * sil
            return mask

        per_prompt = []
        total_calls = len(positive_list) + len(negative_list)
        call_idx = 0

        for i, prompt in enumerate(positive_list):
            call_idx += 1
            mask = _run_sam3(prompt, last_in_chain=(call_idx == total_calls))
            per_prompt.append(mask)
            cov = 100.0 * mask.float().mean().item()
            coverage_lines.append(f"  [+{i + 1}/{len(positive_list)}] '{prompt}' → {cov:.2f}%")

        combined = _combine_masks(per_prompt, combine_mode)

        negative_masks_for_preview = []
        for j, neg_prompt in enumerate(negative_list):
            call_idx += 1
            neg_mask = _run_sam3(neg_prompt, last_in_chain=(call_idx == total_calls))
            cov_before = 100.0 * combined.float().mean().item()
            combined = torch.clamp(combined - neg_mask, 0.0, 1.0)
            cov_after = 100.0 * combined.float().mean().item()
            coverage_lines.append(
                f"  [-{j + 1}/{len(negative_list)}] '{neg_prompt}' subtract → {cov_before:.2f}% → {cov_after:.2f}%"
            )
            preview_mask = (1.0 - neg_mask) if invert_negatives_in_per_prompt else neg_mask
            negative_masks_for_preview.append(preview_mask)

        if skin_color_filter != "off":
            cov_before = 100.0 * combined.float().mean().item()
            combined = _apply_skin_color_filter(
                combined, image, mode=skin_color_filter,
                threshold=skin_color_threshold, strength=color_strength,
                silhouette_mask=silhouette_mask,
                include_dilate_radius=include_dilate_radius,
            )
            cov_after = 100.0 * combined.float().mean().item()
            zone_str = "silhouette" if silhouette_mask is not None else f"dilate={include_dilate_radius}px"
            coverage_lines.append(
                f"  [color {skin_color_filter} strength={color_strength:.2f} th={skin_color_threshold:.2f} "
                f"zone={zone_str}] {cov_before:.2f}% → {cov_after:.2f}%"
            )

        # NOTE: silhouette already applied to every SAM3 mask inside _run_sam3,
        # so per_prompt_masks, combined, and downstream are all silhouette-clean.
        # No final combined-only clip needed.

        if invert_combined:
            combined = 1.0 - combined

        all_preview = per_prompt + negative_masks_for_preview
        per_prompt_batch = torch.cat(all_preview, dim=0) if len(all_preview) > 1 else all_preview[0]

        img = image if image.ndim == 4 else image.unsqueeze(0)
        b, h, w, c = img.shape
        if combined.shape[-2:] != (h, w):
            combined_resized = torch.nn.functional.interpolate(
                combined.unsqueeze(1), size=(h, w), mode="nearest"
            ).squeeze(1)
        else:
            combined_resized = combined

        # masked_image is the literal alpha composite of combined_mask:
        # output = image * mask + bg * (1 - mask). No threshold, no reinterpretation —
        # whatever combined_mask's continuous values are, you see them as alpha here.
        alpha = combined_resized[..., None].to(img.dtype)
        if masked_image_bg == "transparent":
            masked_image = torch.cat([img[..., :3], combined_resized[..., None]], dim=-1)
        else:
            if masked_image_bg == "white":
                bg = torch.ones_like(img)
            elif masked_image_bg == "black":
                bg = torch.zeros_like(img)
            elif masked_image_bg == "checker":
                check = 64
                yy = torch.arange(h, device=img.device).view(1, h, 1, 1)
                xx = torch.arange(w, device=img.device).view(1, 1, w, 1)
                pattern = (((yy // check) + (xx // check)) % 2).to(img.dtype)
                bg = (pattern * 0.4 + 0.5).expand(b, h, w, 3)
            else:
                bg = torch.ones_like(img)
            masked_image = img * alpha + bg * (1.0 - alpha)

        cov_combined = 100.0 * combined.float().mean().item()
        status = (
            f"SAM3 calls: {len(positive_list)} positive + {len(negative_list)} negative; "
            f"combine={combine_mode} color_filter={skin_color_filter} "
            f"(invert={invert_combined}) → final coverage={cov_combined:.2f}%\n"
            + "\n".join(coverage_lines)
        )
        print(f"[BD SAM3 Multi-Prompt] {status}", flush=True)

        return io.NodeOutput(combined, masked_image, per_prompt_batch, status)


SAM3_MULTIPROMPT_V3_NODES = [BD_SAM3MultiPrompt]
SAM3_MULTIPROMPT_NODES = {"BD_SAM3MultiPrompt": BD_SAM3MultiPrompt}
SAM3_MULTIPROMPT_DISPLAY_NAMES = {"BD_SAM3MultiPrompt": "BD SAM3 Multi-Prompt"}
