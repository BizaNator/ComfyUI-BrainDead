"""
BD_LumaStats — measure luminance statistics of an image (optionally within a mask).

Drop inline after BD_NormalizeLuma / BD_CenterMedianLuma to verify the grey-source
pipeline is hitting its targets before feeding u_image0 or u_image2.

Key output: recommended_outer_band  (paste directly into u_float2)
    = (max_luma - min_luma) / 2 × 0.90
  Ensures the extreme ~10% of pixels exceed the outer band and reach V-curve alpha=1.0,
  giving the shader visible shadows and highlights rather than a flat tinted wash.

If normalization is working correctly:
  min_luma  ≈  BD_NormalizeLuma  low_point
  max_luma  ≈  BD_NormalizeLuma  high_point
  median    ≈  BD_CenterMedianLuma  target_center
"""

import numpy as np
import torch
from comfy_api.latest import io

from ...utils.luma import LUMA_STANDARDS, LUMA_TOOLTIP, get_luma_weights


class BD_LumaStats(io.ComfyNode):
    """Inline luma diagnostic: min, max, median, mean, recommended_outer_band."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_LumaStats",
            display_name="BD Luma Stats",
            category="🧠BrainDead/Segmentation",
            description=(
                "Measure luminance statistics of an image within an optional mask. "
                "Image passes through unchanged.\n\n"
                "Place inline in the grey-source pipeline to verify BD_NormalizeLuma "
                "+ BD_CenterMedianLuma are hitting their targets:\n"
                "  min_luma  → should ≈ NormalizeLuma low_point\n"
                "  max_luma  → should ≈ NormalizeLuma high_point\n"
                "  median    → should ≈ CenterMedianLuma target_center\n\n"
                "recommended_outer_band = (max−min)/2 × 0.90\n"
                "Paste this directly into u_float2 in BD_GLSLBatch."
            ),
            inputs=[
                io.Image.Input("image",
                               tooltip="Source image. Passed through to the image output unchanged."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Optional region mask. When connected, stats are computed "
                                      "only from pixels where mask > 0.5.\n\n"
                                      "Strongly recommended: use your skin mask so background "
                                      "and clothing don't bias the measurements."),
                io.Combo.Input("luma_standard", options=LUMA_STANDARDS, default="bt709",
                               tooltip=LUMA_TOOLTIP),
            ],
            outputs=[
                io.Image.Output("image",
                                tooltip="Pass-through — image is unchanged."),
                io.Float.Output("min_luma",
                                tooltip="Darkest pixel luma in the masked region. "
                                        "Should ≈ BD_NormalizeLuma low_point (e.g. 0.10)."),
                io.Float.Output("max_luma",
                                tooltip="Brightest pixel luma in the masked region. "
                                        "Should ≈ BD_NormalizeLuma high_point (e.g. 0.90)."),
                io.Float.Output("median_luma",
                                tooltip="Median luma. Should ≈ BD_CenterMedianLuma target_center (e.g. 0.50)."),
                io.Float.Output("mean_luma",
                                tooltip="Mean luma. Compare with median — large gap means a skewed distribution."),
                io.Float.Output("recommended_outer_band",
                                tooltip="(max−min)/2 × 0.90. Paste into u_float2 in BD_GLSLBatch.\n\n"
                                        "The ×0.90 factor ensures the extreme ~10% of pixels exceed "
                                        "the outer band and reach V-curve alpha=1.0, giving the "
                                        "shader visible shadow/highlight depth."),
            ],
        )

    @classmethod
    def execute(cls, image, mask=None, luma_standard="bt709") -> io.NodeOutput:
        img = image if image.ndim == 4 else image.unsqueeze(0)
        img = img.float()
        B, H, W, C = img.shape

        weights = get_luma_weights(luma_standard).to(img.device).to(img.dtype)
        luma = (img[..., :3] * weights).sum(dim=-1)  # (B, H, W)

        # Normalise mask to (B, H, W)
        roi = None
        if mask is not None:
            m = mask if mask.ndim == 3 else (mask.unsqueeze(0) if mask.ndim == 2 else mask[..., 0])
            m = m.float()
            if m.shape[0] != B:
                m = m.expand(B, -1, -1) if m.shape[0] == 1 else m[:B]
            if m.shape[-2:] != (H, W):
                m = torch.nn.functional.interpolate(
                    m.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
                ).squeeze(1)
            roi = m

        # Gather valid pixels across all frames in the batch
        all_vals: list[torch.Tensor] = []
        for b in range(B):
            luma_b = luma[b]
            if roi is not None:
                vals = luma_b[roi[b] > 0.5]
                if vals.numel() == 0:
                    vals = luma_b.flatten()
            else:
                vals = luma_b.flatten()
            all_vals.append(vals)

        pixels = torch.cat(all_vals)

        min_luma    = float(pixels.min().item())
        max_luma    = float(pixels.max().item())
        median_luma = float(pixels.median().item())
        mean_luma   = float(pixels.mean().item())
        luma_range  = max_luma - min_luma
        recommended_outer_band = luma_range / 2.0 * 0.90

        scope = "masked" if mask is not None else "full_image"
        print(
            f"[BD_LumaStats] luma={luma_standard} scope={scope} "
            f"min={min_luma:.4f} max={max_luma:.4f} range={luma_range:.4f} "
            f"median={median_luma:.4f} mean={mean_luma:.4f} "
            f"recommended_outer_band={recommended_outer_band:.4f}"
        )

        return io.NodeOutput(image, min_luma, max_luma, median_luma, mean_luma, recommended_outer_band)


LUMA_STATS_V3_NODES      = [BD_LumaStats]
LUMA_STATS_NODES         = {"BD_LumaStats": BD_LumaStats}
LUMA_STATS_DISPLAY_NAMES = {"BD_LumaStats": "BD Luma Stats"}
