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
from .mask_resolver import _rgb_to_lab, _adaptive_skin_likelihood


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


def _compute_skin_likelihood(image_np: np.ndarray, color_mode: str,
                             adaptive_ref_binary: np.ndarray | None,
                             adaptive_tolerance: float, adaptive_min_samples: int) -> tuple[np.ndarray, dict]:
    """Pick between fixed_hsv, adaptive_lab, or both. Returns (likelihood (H,W), debug dict)."""
    debug = {"used_mode": color_mode, "adaptive": None}
    if color_mode == "fixed_hsv" or adaptive_ref_binary is None:
        return _skin_tone_likelihood(image_np[..., :3]), debug
    if color_mode == "adaptive_lab":
        likelihood, adaptive_dbg = _adaptive_skin_likelihood(
            image_np[..., :3], adaptive_ref_binary,
            tolerance=adaptive_tolerance, min_samples=adaptive_min_samples,
        )
        debug["adaptive"] = adaptive_dbg
        if adaptive_dbg.get("skipped"):
            debug["used_mode"] = "fixed_hsv (adaptive skipped — too few samples)"
            return _skin_tone_likelihood(image_np[..., :3]), debug
        return likelihood, debug
    if color_mode == "both":
        fixed = _skin_tone_likelihood(image_np[..., :3])
        adapt, adaptive_dbg = _adaptive_skin_likelihood(
            image_np[..., :3], adaptive_ref_binary,
            tolerance=adaptive_tolerance, min_samples=adaptive_min_samples,
        )
        debug["adaptive"] = adaptive_dbg
        if adaptive_dbg.get("skipped"):
            debug["used_mode"] = "fixed_hsv (adaptive skipped)"
            return fixed, debug
        return fixed * adapt, debug
    return _skin_tone_likelihood(image_np[..., :3]), debug


def _apply_skin_color_filter(mask: torch.Tensor, image: torch.Tensor,
                             mode: str = "off",
                             threshold: float = 0.3,
                             strength: float = 0.7,
                             silhouette_mask: torch.Tensor | None = None,
                             include_dilate_radius: int = 64,
                             negative_exclude_zone: torch.Tensor | None = None,
                             color_mode: str = "fixed_hsv",
                             adaptive_ref_binary: np.ndarray | None = None,
                             adaptive_tolerance: float = 25.0,
                             adaptive_min_samples: int = 50) -> tuple[torch.Tensor, dict]:
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
        return mask, {"used_mode": "off"}
    img_np = image.detach().cpu().numpy()
    if img_np.ndim == 4:
        img_np = img_np[0]
    likelihood, debug = _compute_skin_likelihood(
        img_np, color_mode, adaptive_ref_binary, adaptive_tolerance, adaptive_min_samples,
    )
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
                zone = (sil > 0.5).to(mask.dtype)
            else:
                zone = _dilate_mask((mask > 0.05).to(mask.dtype), include_dilate_radius)
        else:
            zone = _dilate_mask((mask > 0.05).to(mask.dtype), include_dilate_radius)
        if negative_exclude_zone is not None:
            ne = negative_exclude_zone.to(mask.device).to(mask.dtype)
            if ne.ndim == 2:
                ne = ne.unsqueeze(0)
            if ne.shape[-2:] != mask.shape[-2:]:
                ne = torch.nn.functional.interpolate(
                    ne.unsqueeze(0).float(), size=mask.shape[-2:], mode="nearest"
                ).squeeze(0)
            zone = zone * (1.0 - (ne > 0.5).to(mask.dtype))
        return zone

    if mode == "exclude":
        return mask * (floor + (1.0 - floor) * likelihood_t), debug
    if mode == "exclude_hard":
        return mask * (likelihood_t >= threshold).to(mask.dtype), debug
    if mode == "include":
        zone = _candidate_zone()
        skin_in_zone = ((likelihood_t >= threshold).to(mask.dtype) * zone)
        return torch.maximum(mask, skin_in_zone), debug
    if mode == "exclude_and_include":
        excluded = mask * (floor + (1.0 - floor) * likelihood_t)
        zone = _candidate_zone()
        included = ((likelihood_t >= threshold).to(mask.dtype) * zone)
        return torch.maximum(excluded, included), debug
    if mode == "remove_matching":
        return mask * (1.0 - likelihood_t * float(strength)), debug
    return mask, debug


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
                    options=["union", "intersection", "subtract_first", "first_only", "vote", "weighted_vote"],
                    default="union", optional=True,
                    tooltip=(
                        "How to combine positive prompts (and how negatives are applied):\n"
                        "  union — OR all positives, then SUBTRACT negatives cumulatively. Single strong positive overrides many weak negatives.\n"
                        "  intersection — AND all positives. Then subtract negatives.\n"
                        "  subtract_first — first positive MINUS all others (then subtract negatives too).\n"
                        "  first_only — return positive #1, ignore the rest.\n"
                        "  vote — MAJORITY VOTE. Each positive detection = +1, each negative = -1. "
                        "Pixel kept if net votes >= vote_threshold. Use when you have many prompts and want "
                        "agreement (e.g. 1 positive vs 6 negatives → removed even if positive was strong).\n"
                        "  weighted_vote — like vote but each detection contributes its CONTINUOUS value, "
                        "not binary. Soft detections weigh proportionally."
                    ),
                ),
                io.Float.Input(
                    "vote_threshold", default=1.0, min=-20.0, max=20.0, step=0.5, optional=True,
                    tooltip="Vote-mode threshold. Pixel kept if (positive_votes - negative_votes) >= this. "
                            "Default 1 = need at least one MORE positive than negative. "
                            "Lower (0, -1) = positives win ties / get a head start. "
                            "Higher (2+) = need stronger positive consensus.",
                ),
                io.Float.Input(
                    "vote_pos_min", default=0.5, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="In 'vote' mode, a positive prompt's mask must have a pixel value >= this to count as a vote. "
                            "Higher = only confident positives count.",
                ),
                io.Float.Input(
                    "vote_neg_min", default=0.5, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="In 'vote' mode, a negative prompt's mask must have a pixel value >= this to count as a vote. "
                            "Higher = only confident negatives count.",
                ),
                io.Combo.Input(
                    "color_filter",
                    options=["off", "exclude", "exclude_hard", "include", "exclude_and_include", "remove_matching"],
                    default="off", optional=True,
                    tooltip=(
                        "Color-aware mask refinement. Works for ANY positive-prompt category — the reference "
                        "color is sampled from your positives (or color_reference_mask if provided), so "
                        "red jacket → red ref, pale skin → pale ref, dark skin → dark ref, etc.\n"
                        "  off — no filtering.\n"
                        "  exclude — KEEP pixels matching the reference, suppress non-matching (cleans bleed in mask).\n"
                        "  exclude_hard — same but binary threshold (use color_strength=1).\n"
                        "  include — ADD pixels matching reference into the mask (catches what SAM3 missed).\n"
                        "  exclude_and_include — combine: keep matching AND add matching pixels SAM3 missed.\n"
                        "  remove_matching — REMOVE pixels matching reference from the mask. Use case: pass "
                        "color_reference_mask=ear_region to subtract ear-colored pixels from a skin detection."
                    ),
                ),
                io.Float.Input(
                    "color_strength", default=0.7, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="How aggressive 'exclude' is. 0 = no effect, 1 = fully suppress non-matching pixels. "
                            "Try 0.5 for stylized art that doesn't match the reference range strictly.",
                ),
                io.Float.Input(
                    "color_threshold", default=0.3, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="Threshold for binary color modes (exclude_hard, include). Pixels with color-match "
                            "likelihood above this count as 'matches reference'. Lower for stylized art (try 0.15-0.25).",
                ),
                io.Combo.Input(
                    "color_mode", options=["adaptive_lab", "fixed_hsv", "both"],
                    default="adaptive_lab", optional=True,
                    tooltip="How the color filter scores match likelihood:\n"
                            "  adaptive_lab — GENERIC, works for ANY positive-prompt category. Samples reference "
                            "color from the current combined mask (or color_reference_mask if provided), then scores "
                            "every image pixel by CIE LAB ΔE distance. Self-tunes per character/object.\n"
                            "  fixed_hsv — SKIN-SPECIFIC. Hard-coded HSV ranges from the legacy GLSL shader. Only "
                            "use this for skin detection on photo-trained characters; fails on very pale/dark skin "
                            "and on non-skin categories entirely.\n"
                            "  both — multiply the two likelihoods (strictest, skin-only).\n"
                            "Falls back to fixed_hsv if not enough confident pixels for adaptive sampling.",
                ),
                io.Mask.Input(
                    "color_reference_mask", optional=True,
                    tooltip="Optional explicit color reference mask. When provided, adaptive_lab samples reference "
                            "color from this region (useful when you want to lock the reference to a specific area, "
                            "e.g. a face mask for skin, a swatch region for clothing). When NOT provided, samples "
                            "from the current SAM3 combined mask — which is normally what you want, since the "
                            "positives ALREADY found the category you're refining.",
                ),
                io.Float.Input(
                    "adaptive_tolerance", default=25.0, min=2.0, max=100.0, step=1.0, optional=True,
                    tooltip="LAB ΔE distance at which adaptive likelihood = 0.5. ΔE 2.3 = just-noticeable, "
                            "25 = clearly different but related (typical skin variation), 50+ = very loose. "
                            "Tighten for strict skin matching, loosen for shadowed/rim-lit skin.",
                ),
                io.Float.Input(
                    "adaptive_sample_threshold", default=0.5, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="Only mask pixels above this value are used as reference samples for adaptive sampling. "
                            "Higher = use only the most confident pixels (better reference, fewer samples).",
                ),
                io.Int.Input(
                    "adaptive_min_samples", default=50, min=10, max=10000, step=10, optional=True,
                    tooltip="Minimum number of confident skin pixels needed for adaptive mode. "
                            "If fewer, falls back to fixed_hsv.",
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
                    tooltip="Invert the FINAL combined mask. Global flip — turns 'detected' into 'not detected'.\n\n"
                            "WARNING: if you wire combined_mask into BD_PartsBuilder.combined_mask "
                            "downstream, leaving this ON will INVERT the silhouette and zero every "
                            "real per-class mask. Symptom: only one full-image leftover part survives. "
                            "Keep this OFF for Parts pipeline workflows.",
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
                vote_threshold=1.0, vote_pos_min=0.5, vote_neg_min=0.5,
                color_filter="off", color_strength=0.7, color_threshold=0.3,
                color_mode="adaptive_lab", color_reference_mask=None,
                adaptive_tolerance=25.0, adaptive_sample_threshold=0.5, adaptive_min_samples=50,
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
            # CRITICAL: SAM3 may return multiple instances when its detector finds
            # >1 region matching the prompt (e.g. "earrings" → left + right). Collapse
            # via pixel-wise max so we always emit exactly ONE mask per prompt — keeps
            # label↔mask alignment intact for BD_PartsRefine / BD_PartsBuilder.
            if mask.ndim == 3 and mask.shape[0] > 1:
                mask = mask.amax(dim=0, keepdim=True)
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
            extra = ""
            if combine_mode in ("vote", "weighted_vote"):
                vote_px = int((mask.float() >= vote_pos_min).sum().item())
                extra = f"  → {vote_px} px voted" if vote_px > 0 else "  → NO VOTES (empty after filter)"
            coverage_lines.append(f"  [+{i + 1}/{len(positive_list)}] '{prompt}' → {cov:.2f}%{extra}")

        negative_masks_for_preview = []
        negative_raw_masks = []
        for j, neg_prompt in enumerate(negative_list):
            call_idx += 1
            neg_mask = _run_sam3(neg_prompt, last_in_chain=(call_idx == total_calls))
            negative_raw_masks.append(neg_mask)
            cov = 100.0 * neg_mask.float().mean().item()
            extra = ""
            if combine_mode in ("vote", "weighted_vote"):
                vote_px = int((neg_mask.float() >= vote_neg_min).sum().item())
                extra = f"  → {vote_px} px voted" if vote_px > 0 else "  → NO VOTES (empty after filter)"
            coverage_lines.append(f"  [-{j + 1}/{len(negative_list)}] '{neg_prompt}' detected → {cov:.2f}%{extra}")
            preview_mask = (1.0 - neg_mask) if invert_negatives_in_per_prompt else neg_mask
            negative_masks_for_preview.append(preview_mask)

        if combine_mode in ("vote", "weighted_vote"):
            if combine_mode == "vote":
                pos_votes = sum((m.float() >= vote_pos_min).float() for m in per_prompt)
                neg_votes = (sum((m.float() >= vote_neg_min).float() for m in negative_raw_masks)
                             if negative_raw_masks else torch.zeros_like(pos_votes))
                vote_kind = "binary"
            else:  # weighted_vote
                pos_votes = sum(m.float() for m in per_prompt)
                neg_votes = (sum(m.float() for m in negative_raw_masks)
                             if negative_raw_masks else torch.zeros_like(pos_votes))
                vote_kind = "weighted"
            net = pos_votes - neg_votes
            combined = (net >= vote_threshold).float()

            pos_contribs = sum(1 for m in per_prompt if (m.float() >= vote_pos_min).any())
            neg_contribs = sum(1 for m in negative_raw_masks if (m.float() >= vote_neg_min).any())
            coverage_lines.append(
                f"  [vote {vote_kind}] {pos_contribs}/{len(per_prompt)} positives + "
                f"{neg_contribs}/{len(negative_raw_masks)} negatives contributed votes "
                f"(empty masks counted nothing). "
                f"pos_max={pos_votes.max().item():.1f}, neg_max={neg_votes.max().item():.1f}, "
                f"net range=[{net.min().item():.1f}, {net.max().item():.1f}], "
                f"threshold={vote_threshold:.1f} → kept {100.0 * combined.mean().item():.2f}%"
            )
        else:
            combined = _combine_masks(per_prompt, combine_mode)
            for j, neg_mask in enumerate(negative_raw_masks):
                cov_before = 100.0 * combined.float().mean().item()
                combined = torch.clamp(combined - neg_mask, 0.0, 1.0)
                cov_after = 100.0 * combined.float().mean().item()
                coverage_lines.append(
                    f"  [subtract -{j + 1}] {cov_before:.2f}% → {cov_after:.2f}%"
                )

        if color_filter != "off":
            neg_exclude = None
            if negative_raw_masks:
                neg_union = negative_raw_masks[0].float().clone()
                for nm in negative_raw_masks[1:]:
                    neg_union = torch.maximum(neg_union, nm.float())
                neg_exclude = neg_union

            adaptive_ref_binary = None
            ref_source = "none"
            if color_mode in ("adaptive_lab", "both"):
                if color_reference_mask is not None:
                    ref = color_reference_mask.float()
                    if ref.ndim == 4:
                        ref = ref.squeeze(0)
                    if ref.ndim == 3:
                        ref = ref.squeeze(0)
                    adaptive_ref_binary = (ref >= adaptive_sample_threshold).cpu().numpy()
                    ref_source = "explicit color_reference_mask"
                else:
                    ref = combined.float()
                    if ref.ndim == 3:
                        ref = ref.squeeze(0)
                    adaptive_ref_binary = (ref >= adaptive_sample_threshold).cpu().numpy()
                    ref_source = "current combined mask"

            cov_before = 100.0 * combined.float().mean().item()
            combined, color_debug = _apply_skin_color_filter(
                combined, image, mode=color_filter,
                threshold=color_threshold, strength=color_strength,
                silhouette_mask=silhouette_mask,
                include_dilate_radius=include_dilate_radius,
                negative_exclude_zone=neg_exclude,
                color_mode=color_mode,
                adaptive_ref_binary=adaptive_ref_binary,
                adaptive_tolerance=adaptive_tolerance,
                adaptive_min_samples=adaptive_min_samples,
            )
            cov_after = 100.0 * combined.float().mean().item()
            zone_str = "silhouette" if silhouette_mask is not None else f"dilate={include_dilate_radius}px"
            neg_str = f", neg_exclude={int((neg_exclude > 0.5).sum().item())}px" if neg_exclude is not None else ""
            adaptive_str = ""
            if color_debug.get("adaptive") and not color_debug["adaptive"].get("skipped"):
                ref_rgb = color_debug["adaptive"].get("ref_rgb")
                n = color_debug["adaptive"].get("sample_count")
                adaptive_str = f", adaptive_ref RGB={tuple(ref_rgb)} from {n} samples ({ref_source})"
            coverage_lines.append(
                f"  [color {color_filter} mode={color_debug.get('used_mode')} strength={color_strength:.2f} "
                f"th={color_threshold:.2f} zone={zone_str}{neg_str}{adaptive_str}] "
                f"{cov_before:.2f}% → {cov_after:.2f}%"
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
            f"combine={combine_mode} color_filter={color_filter} "
            f"(invert={invert_combined}) → final coverage={cov_combined:.2f}%\n"
            + "\n".join(coverage_lines)
        )
        print(f"[BD SAM3 Multi-Prompt] {status}", flush=True)

        return io.NodeOutput(combined, masked_image, per_prompt_batch, status)


SAM3_MULTIPROMPT_V3_NODES = [BD_SAM3MultiPrompt]
SAM3_MULTIPROMPT_NODES = {"BD_SAM3MultiPrompt": BD_SAM3MultiPrompt}
SAM3_MULTIPROMPT_DISPLAY_NAMES = {"BD_SAM3MultiPrompt": "BD SAM3 Multi-Prompt"}
