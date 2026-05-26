"""
BD_LuminanceMask — extract top X% / bottom Y% luminance masks from an image.

Uses np.percentile so thresholds adapt to the actual image content (a dark
image's "top 10%" is still its brightest pixels even if absolute values are low).
Optionally restricts the percentile computation to a region of interest via
mask_within, so background pixels don't skew the threshold.

Use case: feed a mannequin render in, get a highlight mask of natural lit spots
and a dark mask of natural shadow zones. Pack these into G and B of u_image3
for the skin shader's overlay channels.
"""

import numpy as np
import torch

from comfy_api.latest import io

from ...utils.luma import LUMA_STANDARDS, LUMA_TOOLTIP, get_luma_weights


def _to_bhwc(t: torch.Tensor) -> torch.Tensor:
    """Coerce a tensor to (B, H, W, C) layout."""
    if t.ndim == 3:
        return t.unsqueeze(0)
    return t


def _mask_to_bhw(m: torch.Tensor | None) -> torch.Tensor | None:
    if m is None:
        return None
    if m.ndim == 4:
        return m.squeeze(0) if m.shape[0] == 1 else m[..., 0]
    if m.ndim == 2:
        return m.unsqueeze(0)
    return m


def _smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    """GLSL smoothstep: 0 below edge0, 1 above edge1, hermite interpolation between."""
    t = ((x - edge0) / max(edge1 - edge0, 1e-6)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _gaussian_blur_2d(mask_bhw: torch.Tensor, radius_px: float) -> torch.Tensor:
    """Separable gaussian blur on (B, H, W) tensor. radius_px=0 returns unchanged."""
    if radius_px <= 0.0:
        return mask_bhw
    radius = max(1, int(round(radius_px)))
    sigma = max(radius_px / 2.0, 0.5)
    coords = torch.arange(-radius, radius + 1, dtype=mask_bhw.dtype, device=mask_bhw.device)
    g = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kh = g.view(1, 1, 1, -1)  # horizontal
    kv = g.view(1, 1, -1, 1)  # vertical
    m = mask_bhw.unsqueeze(1)  # B, 1, H, W
    m = torch.nn.functional.conv2d(m, kh, padding=(0, radius))
    m = torch.nn.functional.conv2d(m, kv, padding=(radius, 0))
    return m.squeeze(1).clamp(0.0, 1.0)


class BD_LuminanceMask(io.ComfyNode):
    """Extract top/bottom luminance percentile masks from an image."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_LuminanceMask",
            display_name="BD Luminance Mask",
            category="🧠BrainDead/Segmentation",
            description=(
                "Extract top X% brightest and bottom Y% darkest pixels as separate masks. "
                "Uses np.percentile so thresholds adapt to image content (dark images still "
                "produce a useful highlight mask from their brightest regions). "
                "Optional mask_within restricts the percentile computation to a region — "
                "e.g. pass the skin mask so clothes/background don't skew the percentiles."
            ),
            inputs=[
                io.Image.Input("image",
                               tooltip="Source image. Luminance computed via the selected weighting."),
                io.Combo.Input("luma_standard", options=LUMA_STANDARDS, default="bt709",
                               tooltip=LUMA_TOOLTIP),
                io.Float.Input("top_percent", default=10.0, min=0.0, max=50.0, step=0.05,
                               tooltip="Extract pixels in the top X% of luminance. Set to 0 to disable highlight output. "
                                       "If your image has lots of pure-white pixels (reflections, background), the "
                                       "percentile will collapse to 1.0 — use exclude_above to clip those out first."),
                io.Float.Input("bottom_percent", default=8.0, min=0.0, max=50.0, step=0.05,
                               tooltip="Extract pixels in the bottom Y% of luminance. Set to 0 to disable dark output. "
                                       "If your image has lots of pure-black pixels (background, lines), use exclude_below."),
                io.Float.Input("exclude_above", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Pixels with luminance ABOVE this are excluded from BOTH the percentile "
                                       "calculation AND the output mask. Use to ignore pure-white reflections/background "
                                       "(e.g. 0.95 excludes pixels brighter than 95%). Default 1.0 = no exclusion."),
                io.Float.Input("exclude_below", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Pixels with luminance BELOW this are excluded from BOTH the percentile "
                                       "calculation AND the output mask. Use to ignore pure-black background/lines "
                                       "(e.g. 0.05 excludes pixels darker than 5%). Default 0.0 = no exclusion."),
                io.Float.Input("top_soft_edge", default=0.03, min=0.0, max=0.3, step=0.005,
                               tooltip="Soft edge BELOW the highlight threshold — pixels at or above thr_hi are "
                                       "fully selected (=1), pixels in [thr_hi - top_soft_edge, thr_hi] ramp 0→1. "
                                       "0 = hard binary edge. Bigger = wider luma-range feather (smooths threshold)."),
                io.Float.Input("bottom_soft_edge", default=0.03, min=0.0, max=0.3, step=0.005,
                               tooltip="Soft edge ABOVE the dark threshold — pixels at or below thr_lo are "
                                       "fully selected (=1), pixels in [thr_lo, thr_lo + bottom_soft_edge] ramp 1→0. "
                                       "0 = hard binary edge. Bigger = wider luma-range feather."),
                io.Float.Input("smooth_pixels", default=0.0, min=0.0, max=200.0, step=1.0,
                               tooltip="SPATIAL gaussian blur radius in pixels applied to BOTH output masks. "
                                       "This spreads the mask spatially (softens chunk edges) but doesn't change "
                                       "the value relationships. Typical: 4-16 subtle, 32+ heavy. 0 = no blur."),
                io.Boolean.Input("gradient_mode", default=False,
                                 tooltip="When ON: mask values become a SMOOTH RAMP based on how far each pixel's "
                                         "luminance is from the threshold toward the extreme. Brightest pixel = white, "
                                         "threshold pixel = black, smooth gradient between (no binary cutoff). "
                                         "Ignores soft_edge. This is the 'fade from white to black' look — different from "
                                         "smooth_pixels (which is spatial blur)."),
                io.Float.Input("gradient_curve", default=1.0, min=0.2, max=4.0, step=0.05,
                               tooltip="Shape of the gradient ramp (only when gradient_mode is ON). "
                                       "1.0 = linear. <1 (e.g. 0.5 = sqrt) ramps quickly toward white. "
                                       ">1 (e.g. 2.0) keeps most pixels dark, only brightest reach white."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Optional region mask. When provided, percentile calculation only considers "
                                      "pixels where this mask > 0.5 (typically the skin mask). "
                                      "Output masks are also restricted to this region (zero outside)."),
            ],
            outputs=[
                io.Image.Output(display_name="highlight_mask",
                                tooltip="White (1) where pixels are in the top X% of luminance, black (0) elsewhere."),
                io.Image.Output(display_name="dark_mask",
                                tooltip="White (1) where pixels are in the bottom Y% of luminance, black (0) elsewhere."),
                io.Float.Output(display_name="threshold_hi",
                                tooltip="The computed luminance threshold above which pixels become highlight_mask=1."),
                io.Float.Output(display_name="threshold_lo",
                                tooltip="The computed luminance threshold below which pixels become dark_mask=1."),
            ],
        )

    @classmethod
    def execute(cls, image, luma_standard="bt709",
                top_percent=10.0, bottom_percent=8.0,
                exclude_above=1.0, exclude_below=0.0,
                top_soft_edge=0.03, bottom_soft_edge=0.03,
                smooth_pixels=0.0, gradient_mode=False, gradient_curve=1.0,
                mask=None) -> io.NodeOutput:

        img = _to_bhwc(image).float()
        b, h, w, c = img.shape

        # Luma per pixel: (B, H, W)
        weights = get_luma_weights(luma_standard).to(img.device).to(img.dtype)
        luma = (img[..., :3] * weights).sum(dim=-1)

        # Region of interest for percentile computation AND for restricting outputs
        roi = _mask_to_bhw(mask)
        if roi is not None:
            roi = roi.to(luma.device).float()
            if roi.shape[0] != b:
                roi = roi.expand(b, -1, -1) if roi.shape[0] == 1 else roi[:b]
            if roi.shape[-2:] != luma.shape[-2:]:
                roi = torch.nn.functional.interpolate(
                    roi.unsqueeze(1), size=luma.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(1)

        # Compute percentiles per-batch (treat each image independently)
        hi_masks = []
        dark_masks = []
        thresholds_hi = []
        thresholds_lo = []

        for i in range(b):
            luma_i = luma[i]

            # Build the "valid" pool for percentile: mask region (if provided) AND
            # within [exclude_below, exclude_above] luminance range.
            if roi is not None:
                roi_keep = roi[i] > 0.5
                exclude_keep = (luma_i >= exclude_below) & (luma_i <= exclude_above)
                keep = roi_keep & exclude_keep
                valid = luma_i[keep]
                if valid.numel() == 0:
                    valid = luma_i[roi_keep] if roi_keep.any() else luma_i.flatten()
            else:
                exclude_keep = (luma_i >= exclude_below) & (luma_i <= exclude_above)
                valid = luma_i[exclude_keep]
                if valid.numel() == 0:
                    valid = luma_i.flatten()

            valid_np = valid.cpu().numpy()

            if top_percent > 0.0:
                thr_hi = float(np.percentile(valid_np, 100.0 - top_percent))
            else:
                thr_hi = 1.01  # above max, mask is all black

            if bottom_percent > 0.0:
                thr_lo = float(np.percentile(valid_np, bottom_percent))
            else:
                thr_lo = -0.01  # below min, mask is all black

            # ── Highlight mask ──
            if top_percent <= 0.0:
                hi_mask = torch.zeros_like(luma_i)
            elif gradient_mode:
                # Ramp from threshold to the effective max luma (exclude_above).
                # Pixels at thr_hi → 0, pixels at exclude_above (or 1.0) → 1.
                # Pixels above exclude_above are clamped to the max (still = 1).
                hi_extreme = exclude_above if exclude_above < 1.0 else 1.0
                hi_range   = max(hi_extreme - thr_hi, 1e-6)
                luma_clamped_hi = luma_i.clamp(max=hi_extreme)
                hi_t = ((luma_clamped_hi - thr_hi) / hi_range).clamp(0.0, 1.0)
                hi_mask = hi_t.pow(gradient_curve)
            elif top_soft_edge > 0.0:
                hi_mask = _smoothstep(thr_hi - top_soft_edge, thr_hi, luma_i)
            else:
                hi_mask = (luma_i >= thr_hi).float()

            # ── Dark mask ──
            if bottom_percent <= 0.0:
                dark_mask = torch.zeros_like(luma_i)
            elif gradient_mode:
                # Ramp from threshold down to exclude_below (or 0.0).
                # Pixels at thr_lo → 0, pixels at exclude_below → 1.
                # Pixels below exclude_below are clamped to the min (still = 1).
                dark_extreme = exclude_below if exclude_below > 0.0 else 0.0
                dark_range   = max(thr_lo - dark_extreme, 1e-6)
                luma_clamped_lo = luma_i.clamp(min=dark_extreme)
                dark_t = ((thr_lo - luma_clamped_lo) / dark_range).clamp(0.0, 1.0)
                dark_mask = dark_t.pow(gradient_curve)
            elif bottom_soft_edge > 0.0:
                dark_mask = 1.0 - _smoothstep(thr_lo, thr_lo + bottom_soft_edge, luma_i)
            else:
                dark_mask = (luma_i <= thr_lo).float()

            # Restrict outputs to mask region (zero outside)
            if roi is not None:
                hi_mask   = hi_mask   * roi[i]
                dark_mask = dark_mask * roi[i]

            # Restrict output to exclude_above/exclude_below range — only in BINARY mode.
            # In gradient mode the clamp() already handles bounds correctly (saturated
            # pixels reach the gradient's max value instead of being zeroed).
            if not gradient_mode:
                exclude_keep_f = ((luma_i >= exclude_below) & (luma_i <= exclude_above)).float()
                hi_mask   = hi_mask   * exclude_keep_f
                dark_mask = dark_mask * exclude_keep_f

            hi_masks.append(hi_mask)
            dark_masks.append(dark_mask)
            thresholds_hi.append(thr_hi)
            thresholds_lo.append(thr_lo)

        hi_batch = torch.stack(hi_masks, dim=0)
        dark_batch = torch.stack(dark_masks, dim=0)

        # Spatial gaussian blur for smooth fading (fixes "broken-up chunks")
        if smooth_pixels > 0.0:
            hi_batch   = _gaussian_blur_2d(hi_batch,   smooth_pixels)
            dark_batch = _gaussian_blur_2d(dark_batch, smooth_pixels)

        # Output as IMAGE (B, H, W, 3) — replicate the mask across RGB
        hi_out = hi_batch.unsqueeze(-1).expand(-1, -1, -1, 3).clamp(0.0, 1.0)
        dark_out = dark_batch.unsqueeze(-1).expand(-1, -1, -1, 3).clamp(0.0, 1.0)

        # Return average thresholds (most users have a single image in the batch)
        avg_thr_hi = float(np.mean(thresholds_hi)) if thresholds_hi else 1.0
        avg_thr_lo = float(np.mean(thresholds_lo)) if thresholds_lo else 0.0

        # Diagnostic — show the actual histogram so user can see WHY a percentile
        # collapsed. If 80% of valid pixels are >= 0.99, "top 2%" cannot be smaller
        # than the saturated bucket — every tied value at the threshold gets picked up.
        first = luma[0]
        if roi is not None:
            valid_diag = first[roi[0] > 0.5]
            if valid_diag.numel() == 0:
                valid_diag = first.flatten()
        else:
            valid_diag = first.flatten()
        exclude_keep_diag = (valid_diag >= exclude_below) & (valid_diag <= exclude_above)
        valid_in_range = valid_diag[exclude_keep_diag]
        n_valid = valid_in_range.numel()

        if n_valid > 0:
            sat = float((valid_in_range >= 0.99).float().mean() * 100)
            near_sat = float((valid_in_range >= 0.95).float().mean() * 100)
            near_blk = float((valid_in_range <= 0.01).float().mean() * 100)
        else:
            sat = near_sat = near_blk = 0.0

        print(f"[BD_LuminanceMask] luma={luma_standard}, "
              f"top_percent={top_percent}% → thr_hi={avg_thr_hi:.4f}, "
              f"bottom_percent={bottom_percent}% → thr_lo={avg_thr_lo:.4f}, "
              f"exclude=[{exclude_below:.3f}, {exclude_above:.3f}], "
              f"smooth_px={smooth_pixels:.1f}, "
              f"gradient={'on (curve=' + str(gradient_curve) + ')' if gradient_mode else 'off'}, "
              f"mask_active={'yes' if mask is not None else 'no'}")
        print(f"[BD_LuminanceMask] valid pixels in eligible range: {n_valid} "
              f"(saturated ≥0.99: {sat:.1f}%, ≥0.95: {near_sat:.1f}%, ≤0.01: {near_blk:.1f}%)")
        if sat > top_percent and exclude_above >= 1.0 and top_percent > 0.0:
            print(f"[BD_LuminanceMask] ⚠ {sat:.1f}% of eligible pixels are saturated (≥0.99) "
                  f"but you asked for top {top_percent}% — percentile collapses to 1.0 because "
                  f"too many pixels are tied at max. Lower exclude_above below 0.99 to fix.")

        return io.NodeOutput(hi_out, dark_out, avg_thr_hi, avg_thr_lo)


LUMINANCE_MASK_V3_NODES = [BD_LuminanceMask]
LUMINANCE_MASK_NODES = {"BD_LuminanceMask": BD_LuminanceMask}
LUMINANCE_MASK_DISPLAY_NAMES = {"BD_LuminanceMask": "BD Luminance Mask"}
