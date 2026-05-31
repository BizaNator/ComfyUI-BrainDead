"""
BD_CenterMedianLuma — shift an image so its luminance median lands on a target value.

Pairs with the GLSL skin-tinting shader (V-curve). The V-curve has a single midpoint
(u_float0) where alpha=0 → 100% tint shows. For the tint to dominate the skin area,
the input image's MEDIAN luma must sit at that midpoint. Source images vary per
character (different lighting / albedo / stylization), so a static u_float0 only fits
one image. This node makes the median exactly equal to target_center for every image,
letting you set u_float0 once and forget it.

Difference from BD_NormalizeLuma:
- NormalizeLuma is min/max remapping — it stretches the histogram to fit target bounds
  but preserves the histogram SHAPE, so the median ends up wherever the input's
  internal distribution placed it.
- CenterMedianLuma is a pure ADDITIVE shift — it moves the histogram bodily so the
  median lands on target, preserving spread. Spread changes are NormalizeLuma's job.

Common pipeline:
    BD_ImageToGreyscale → BD_NormalizeLuma → BD_CenterMedianLuma → u_image0
                          ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^
                          (set spread)       (set position)
"""

import numpy as np
import torch

from comfy_api.latest import io

from ...utils.luma import LUMA_STANDARDS, LUMA_TOOLTIP, get_luma_weights


def _mask_to_bhw(m):
    if m is None:
        return None
    if m.ndim == 4:
        return m.squeeze(0) if m.shape[0] == 1 else m[..., 0]
    if m.ndim == 2:
        return m.unsqueeze(0)
    return m


class BD_CenterMedianLuma(io.ComfyNode):
    """Additive shift so the masked-luma median equals target_center. Pairs with V-curve shaders."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CenterMedianLuma",
            display_name="BD Center Median Luma",
            category="🧠BrainDead/Segmentation",
            description=(
                "Compute the luma MEDIAN of the (optionally masked) input pixels, then "
                "ADD a constant offset so that median lands exactly on target_center. "
                "Preserves histogram shape (spread/contrast unchanged), only shifts position.\n\n"
                "Why: V-curve skin shaders need the input's median to sit AT the midpoint "
                "(u_float0) for tint to dominate. BD_NormalizeLuma controls spread; this "
                "node controls POSITION. Use both: normalize first (set spread), then center "
                "(set median position), then feed u_image0."
            ),
            inputs=[
                io.Image.Input("image",
                               tooltip="Source image (single or batch). Each frame is centered independently."),
                io.Combo.Input("luma_standard", options=LUMA_STANDARDS, default="bt709",
                               tooltip=LUMA_TOOLTIP),
                io.Float.Input("target_center", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="Desired median luma after centering. Set to your V-curve "
                                       "midpoint (u_float0). Default 0.5 keeps things symmetric."),
                io.Combo.Input("statistic", options=["median", "mean"], default="median",
                               tooltip="median = robust to outliers (cheek highlights, deep shadows). "
                                       "mean = sensitive to outliers but gives a smoother shift. "
                                       "Median is the right default for V-curve centering."),
                io.Boolean.Input("calc_from_mask", default=True,
                                 tooltip="Controls which pixels are used to MEASURE the median.\n\n"
                                         "ON (default): median is computed from mask pixels only "
                                         "(where mask > 0.5). Recommended — background doesn't bias.\n"
                                         "OFF: median is computed from the whole image. Use with "
                                         "exclude_transparent to skip fully-transparent background "
                                         "pixels (e.g. when the source already has a body-shape alpha).\n\n"
                                         "Independent from apply_to_mask_only — you can measure from "
                                         "the mask but apply the shift to the whole image, or vice versa.\n"
                                         "Has no effect when no mask is wired."),
                io.Boolean.Input("apply_to_mask_only", default=True,
                                 tooltip="Controls WHERE the computed shift is APPLIED.\n\n"
                                         "ON (default): only pixels WITHIN the mask are shifted; "
                                         "pixels outside pass through unchanged.\n"
                                         "OFF: the shift is applied to the whole image.\n\n"
                                         "Independent from calc_from_mask — common combinations:\n"
                                         "  calc_from_mask=ON,  apply=mask   → measure+apply in mask only\n"
                                         "  calc_from_mask=ON,  apply=whole  → measure mask, shift entire image\n"
                                         "  calc_from_mask=OFF, apply=mask   → measure whole image, apply in mask\n"
                                         "  calc_from_mask=OFF, apply=whole  → whole-image both (mask irrelevant)\n"
                                         "Has no effect when no mask is wired."),
                io.Boolean.Input("exclude_transparent", default=True,
                                 tooltip="When calc_from_mask is OFF (measuring from whole image), "
                                         "exclude pixels where the source alpha < 0.5 from the median "
                                         "calculation. Prevents a fully-transparent background from "
                                         "biasing the median when the source already has a body/head alpha.\n"
                                         "Ignored when calc_from_mask is ON, or when the image has no "
                                         "alpha channel."),
                io.Boolean.Input("preserve_alpha", default=True,
                                 tooltip="ON: alpha channel passes through unchanged. OFF: alpha is "
                                         "also shifted (rarely what you want)."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Optional region mask (e.g. skin mask). Used for calculation "
                                      "and/or application depending on calc_from_mask / apply_to_mask_only.\n"
                                      "STRONGLY RECOMMENDED — without a mask, background pixels bias "
                                      "the median (use exclude_transparent as a fallback)."),
            ],
            outputs=[
                io.Image.Output(display_name="image",
                                tooltip="Centered image (RGB shifted by the same constant)."),
                io.Float.Output(display_name="measured_median",
                                tooltip="The actual masked-luma median found in the input. "
                                        "Useful for debugging or wiring to another node."),
                io.Float.Output(display_name="shift_applied",
                                tooltip="The constant added to RGB. = target_center - measured_median. "
                                        "Negative if input was too bright, positive if too dark."),
            ],
        )

    @classmethod
    def execute(cls, image, luma_standard="bt709", target_center=0.5,
                statistic="median", calc_from_mask=True, apply_to_mask_only=True,
                exclude_transparent=True, preserve_alpha=True,
                mask=None) -> io.NodeOutput:

        img = image if image.ndim == 4 else image.unsqueeze(0)
        img = img.float()
        b, h, w, c = img.shape

        weights = get_luma_weights(luma_standard).to(img.device).to(img.dtype)
        luma = (img[..., :3] * weights).sum(dim=-1)  # (B, H, W)

        # Process optional ROI mask
        roi = _mask_to_bhw(mask)
        if roi is not None:
            roi = roi.to(luma.device).float()
            if roi.shape[0] != b:
                roi = roi.expand(b, -1, -1) if roi.shape[0] == 1 else roi[:b]
            if roi.shape[-2:] != luma.shape[-2:]:
                roi = torch.nn.functional.interpolate(
                    roi.unsqueeze(1), size=luma.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(1)

        out_imgs = []
        measured = []
        shifts = []

        for i in range(b):
            luma_i = luma[i]

            # ── Pixel selection for median calculation ────────────────────────
            if roi is not None and calc_from_mask:
                # Measure from mask pixels only (recommended — no BG bias)
                valid = luma_i[roi[i] > 0.5]
                if valid.numel() == 0:
                    valid = luma_i.flatten()
            elif exclude_transparent and c == 4:
                # Measure from whole image but exclude transparent BG pixels
                alpha_i = img[i, ..., 3]
                valid = luma_i[alpha_i > 0.5]
                if valid.numel() == 0:
                    valid = luma_i.flatten()
            else:
                # Measure from all pixels
                valid = luma_i.flatten()

            if statistic == "mean":
                stat = float(valid.mean().item())
            else:
                stat = float(valid.median().item())

            shift = target_center - stat
            img_i = img[i]

            # Apply the additive shift to RGB only (preserve alpha if present)
            if c == 4 and preserve_alpha:
                rgb_shifted = (img_i[..., :3] + shift).clamp(0.0, 1.0)
                shifted = torch.cat([rgb_shifted, img_i[..., 3:4]], dim=-1)
            else:
                shifted = (img_i + shift).clamp(0.0, 1.0)

            # If mask is wired AND apply_to_mask_only is on, blend back to original
            # outside the mask region (soft mask values give natural feather).
            if apply_to_mask_only and roi is not None:
                roi_i = roi[i].clamp(0.0, 1.0)
                roi_w = roi_i.unsqueeze(-1)  # (H, W, 1)
                shifted = shifted * roi_w + img[i] * (1.0 - roi_w)

            out_imgs.append(shifted)
            measured.append(stat)
            shifts.append(shift)

        out = torch.stack(out_imgs, dim=0)
        avg_median = float(np.mean(measured)) if measured else 0.0
        avg_shift = float(np.mean(shifts)) if shifts else 0.0

        if mask is not None:
            calc_scope = "mask" if calc_from_mask else "whole_image"
            apply_scope = "mask_only" if apply_to_mask_only else "whole_image"
        else:
            xp_note = "+excl_transparent" if (exclude_transparent and c == 4) else ""
            calc_scope = f"whole_image{xp_note}"
            apply_scope = "whole_image"
        print(f"[BD_CenterMedianLuma] luma={luma_standard}, stat={statistic}, "
              f"measured={avg_median:.4f} → target={target_center:.4f}, "
              f"shift={avg_shift:+.4f}, "
              f"calc={calc_scope}, apply={apply_scope}")

        return io.NodeOutput(out, avg_median, avg_shift)


CENTER_MEDIAN_LUMA_V3_NODES = [BD_CenterMedianLuma]
CENTER_MEDIAN_LUMA_NODES = {"BD_CenterMedianLuma": BD_CenterMedianLuma}
CENTER_MEDIAN_LUMA_DISPLAY_NAMES = {"BD_CenterMedianLuma": "BD Center Median Luma"}
