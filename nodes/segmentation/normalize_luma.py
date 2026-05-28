"""
BD_NormalizeLuma — auto-rescale an image's luminance range to fit a target range.

Common use cases:
- Source is too bright (lots of saturation): compress range to e.g. [0.0, 0.85] so
  there's actually headroom for percentile-based highlight extraction downstream.
- Source is too dark: lift to e.g. [0.1, 0.95] so the V-curve has real values to work with.
- Pair with BD_LuminanceMask to make the percentile calculation meaningful — after
  normalization, "top 2%" actually selects ~2% of pixels because the histogram is
  spread instead of collapsed at the extremes.

Uses np.percentile (not absolute min/max) so a few outlier pixels can't define
the range. clip_percent_high=1.0 means "treat the top 1% of pixels as the
source-max" — robust to specular spikes / bad pixels.
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


class BD_NormalizeLuma(io.ComfyNode):
    """Rescale image luma to fit a target range — auto-levels with optional mask awareness."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_NormalizeLuma",
            display_name="BD Normalize Luma",
            category="🧠BrainDead/Segmentation",
            description=(
                "Auto-rescale image luma to a target range. Uses percentile clipping so "
                "outlier pixels (e.g. saturated highlights, dead-black background) don't "
                "define the source range. Optional mask restricts the percentile calc to "
                "a region (e.g. skin only). Useful as a preprocess before BD_LuminanceMask "
                "when the source is too bright/dark and the percentile collapses."
            ),
            inputs=[
                io.Image.Input("image",
                               tooltip="Source image to normalize."),
                io.Combo.Input("luma_standard", options=LUMA_STANDARDS, default="bt709",
                               tooltip=LUMA_TOOLTIP),
                io.Float.Input("target_max", default=0.95, min=0.0, max=1.0, step=0.01,
                               tooltip="Brightest output value. Set lower (e.g. 0.85) to compress "
                                       "an over-bright image so there's headroom."),
                io.Float.Input("target_min", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Darkest output value. Set higher (e.g. 0.1) to lift shadows "
                                       "in an under-exposed image."),
                io.Float.Input("clip_percent_high", default=1.0, min=0.0, max=50.0, step=0.5,
                               tooltip="Treat the top X% of valid pixels as the source-max. "
                                       "Higher values ignore more outliers. 1.0 = robust default. "
                                       "0 = use absolute max (sensitive to single-pixel spikes)."),
                io.Float.Input("clip_percent_low", default=1.0, min=0.0, max=50.0, step=0.5,
                               tooltip="Treat the bottom X% of valid pixels as the source-min. "
                                       "Higher values ignore more outliers."),
                io.Boolean.Input("preserve_color", default=True,
                                 tooltip="ON: rescale RGB proportionally (keeps color, just darkens/brightens). "
                                         "OFF: convert to greyscale using the normalized luma."),
                io.Boolean.Input("proportional_scale", default=False,
                                 tooltip="OFF (default): RANGE FIT — remaps source [min, max] to target [min, max]. "
                                         "Stretches/compresses both ends, lifts or drops the black floor. "
                                         "ON: PROPORTIONAL — single multiplier (target_max / src_max), then HARD "
                                         "CLAMP at target_max. No pixel can exceed target_max. Brightest non-outlier "
                                         "pixel hits exactly target_max, outlier pixels (top clip_percent_high%) are "
                                         "clamped down to target_max. Everything below scales by the same factor, "
                                         "input=0 stays at 0. target_min is IGNORED. "
                                         "Use when you want to dim the whole image without lifting shadows — "
                                         "e.g. shift brightest part down to mid-grey (target_max=0.5) — "
                                         "even saturated pixels are guaranteed ≤ 0.5."),
                io.Boolean.Input("apply_to_mask_only", default=True,
                                 tooltip="ON (default): only pixels WITHIN the mask are normalized — "
                                         "pixels outside the mask pass through UNCHANGED from the source. "
                                         "OFF: normalization is applied to the whole image, even outside "
                                         "the mask area (mask still controls the percentile calc either way). "
                                         "If no mask is wired, this setting has no effect."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Optional region mask. Percentile calc only considers pixels "
                                      "where mask > 0.5. By default (apply_to_mask_only=ON), the "
                                      "normalization is also only applied within this region."),
                io.Float.Input("luma_apply_max", default=1.0, min=0.0, max=1.0, step=0.01,
                               optional=True,
                               tooltip="Tone ceiling — normalization fades out above this luminance. "
                                       "1.0 = apply to all tones (default, existing behaviour). "
                                       "0.5 = only pixels at or below mid-grey receive the full "
                                       "normalized result; brighter pixels fade back to original. "
                                       "Use this to lift shadows without touching highlights.\n\n"
                                       "The blend is computed on the ORIGINAL luma so it doesn't "
                                       "chase the output — a highlight stays a highlight."),
                io.Float.Input("luma_apply_feather", default=0.15, min=0.0, max=1.0, step=0.01,
                               optional=True,
                               tooltip="Width of the soft transition above luma_apply_max. "
                                       "0 = hard cutoff. 0.15 = blend ramps from full→none over "
                                       "0.15 luma units above luma_apply_max (recommended). "
                                       "Ignored when luma_apply_max = 1.0."),
            ],
            outputs=[
                io.Image.Output(display_name="image",
                                tooltip="Normalized image (RGB preserved if preserve_color=on)."),
                io.Float.Output(display_name="found_min",
                                tooltip="The source luma value that maps to target_min."),
                io.Float.Output(display_name="found_max",
                                tooltip="The source luma value that maps to target_max."),
            ],
        )

    @classmethod
    def execute(cls, image, luma_standard="bt709",
                target_max=0.95, target_min=0.0,
                clip_percent_high=1.0, clip_percent_low=1.0,
                preserve_color=True, proportional_scale=False,
                apply_to_mask_only=True, mask=None,
                luma_apply_max=1.0, luma_apply_feather=0.15) -> io.NodeOutput:

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
        found_mins = []
        found_maxs = []

        for i in range(b):
            luma_i = luma[i]
            if roi is not None:
                valid = luma_i[roi[i] > 0.5]
                if valid.numel() == 0:
                    valid = luma_i.flatten()
            else:
                valid = luma_i.flatten()

            valid_np = valid.cpu().numpy()

            if clip_percent_low > 0.0:
                src_min = float(np.percentile(valid_np, clip_percent_low))
            else:
                src_min = float(valid_np.min())

            if clip_percent_high > 0.0:
                src_max = float(np.percentile(valid_np, 100.0 - clip_percent_high))
            else:
                src_max = float(valid_np.max())

            if proportional_scale:
                # Single multiplier: src_max → target_max, everything else scales by same factor.
                # Brightest pixel becomes target_max exactly, 0 stays at 0.
                scale = target_max / max(src_max, 1e-6)
                # src_min and target_min are not used in this mode
            else:
                # Range fit: src [min, max] → target [min, max] (current default)
                src_range = max(src_max - src_min, 1e-6)
                tgt_range = target_max - target_min
                scale = tgt_range / src_range

            if preserve_color:
                img_i = img[i]
                if proportional_scale:
                    rescaled = img_i * scale
                else:
                    rescaled = (img_i - src_min) * scale + target_min
                # Keep alpha channel unchanged if present
                if c == 4:
                    rescaled = torch.cat([rescaled[..., :3], img_i[..., 3:4]], dim=-1)
            else:
                # Greyscale output based on normalized luma
                if proportional_scale:
                    norm_luma = luma_i * scale
                else:
                    norm_luma = (luma_i - src_min) * scale + target_min
                norm_luma_3 = norm_luma.unsqueeze(-1).expand(-1, -1, 3)
                if c == 4:
                    rescaled = torch.cat([norm_luma_3, img[i, ..., 3:4]], dim=-1)
                else:
                    rescaled = norm_luma_3

            # In proportional mode: hard-clamp RGB to target_max so outlier pixels
            # (above the percentile src_max) can't exceed the ceiling. Without this,
            # pixels in the top clip_percent_high% stay brighter than target_max
            # because they were excluded from the percentile calculation.
            if proportional_scale:
                if c == 4:
                    rgb_clamped = rescaled[..., :3].clamp(max=target_max)
                    rescaled = torch.cat([rgb_clamped, rescaled[..., 3:4]], dim=-1)
                else:
                    rescaled = rescaled.clamp(max=target_max)

            rescaled = rescaled.clamp(0.0, 1.0)

            # Tone-range blend: fade normalization out above luma_apply_max.
            # Weight is computed from ORIGINAL luma so highlights stay anchored
            # to their source values regardless of what the normalization does.
            if luma_apply_max < 1.0:
                fade_end = luma_apply_max + max(luma_apply_feather, 1e-4)
                # tone_w = 1 below luma_apply_max, ramps to 0 at fade_end
                tone_w = 1.0 - ((luma_i - luma_apply_max) / (fade_end - luma_apply_max)).clamp(0.0, 1.0)
                tone_w = tone_w.unsqueeze(-1)   # (H, W, 1)
                rescaled = rescaled * tone_w + img[i] * (1.0 - tone_w)
                rescaled = rescaled.clamp(0.0, 1.0)

            # When mask is wired AND apply_to_mask_only is on, blend back to the
            # original outside the mask region (soft mask values give natural feather).
            if apply_to_mask_only and roi is not None:
                roi_i = roi[i].clamp(0.0, 1.0)              # (H, W)
                roi_w = roi_i.unsqueeze(-1)                  # (H, W, 1) for broadcasting
                rescaled = rescaled * roi_w + img[i] * (1.0 - roi_w)

            out_imgs.append(rescaled)
            found_mins.append(src_min)
            found_maxs.append(src_max)

        out = torch.stack(out_imgs, dim=0)
        avg_min = float(np.mean(found_mins)) if found_mins else 0.0
        avg_max = float(np.mean(found_maxs)) if found_maxs else 1.0

        scope = "mask only" if (apply_to_mask_only and mask is not None) else "whole image"
        if proportional_scale:
            mode_desc = f"PROPORTIONAL scale={target_max / max(avg_max, 1e-6):.4f}"
        else:
            mode_desc = f"RANGE FIT [{target_min:.3f}, {target_max:.3f}]"
        tone_desc = (f" tone_range=[0,{luma_apply_max:.2f}]+feather{luma_apply_feather:.2f}"
                     if luma_apply_max < 1.0 else "")
        print(f"[BD_NormalizeLuma] luma={luma_standard}, "
              f"source range: [{avg_min:.4f}, {avg_max:.4f}] "
              f"→ {mode_desc}{tone_desc}, "
              f"clip=[{clip_percent_low:.1f}%, {clip_percent_high:.1f}%], "
              f"preserve_color={'yes' if preserve_color else 'no'}, "
              f"mask_active={'yes' if mask is not None else 'no'}, "
              f"applied_to={scope}")

        return io.NodeOutput(out, avg_min, avg_max)


NORMALIZE_LUMA_V3_NODES = [BD_NormalizeLuma]
NORMALIZE_LUMA_NODES = {"BD_NormalizeLuma": BD_NormalizeLuma}
NORMALIZE_LUMA_DISPLAY_NAMES = {"BD_NormalizeLuma": "BD Normalize Luma"}
