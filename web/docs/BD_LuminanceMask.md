# BD Luminance Mask

Extract adaptive top% / bottom% luminance masks from any image, with optional mask scoping.

## Overview

Pulls out the brightest X% and darkest Y% of an image as separate greyscale masks. Uses `np.percentile` so thresholds adapt to image content — a dark mannequin render still produces a useful "top 10%" highlight extraction relative to its own range. Designed as the companion that feeds the G (highlight) and B (dark) channels of the skin shader's `u_image3` overlay pack via `BD_PackChannels`.

## Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | IMAGE | — | Source to extract from |
| `luma_standard` | COMBO | `bt709` | RGB → luma weighting. `bt709` (default, modern sRGB), `bt601` (legacy NTSC, matches Photoshop Desaturate), `average` ((R+G+B)/3, not perceptual). |
| `top_percent` | FLOAT | 10.0 | Brightest X% of pixels become the highlight mask. 0 = highlight output disabled. Step 0.05 so you can dial in tiny percentages. |
| `bottom_percent` | FLOAT | 8.0 | Darkest Y% become the dark mask. 0 = dark output disabled. |
| `exclude_above` | FLOAT | 1.0 | Pixels brighter than this are clipped from BOTH the percentile calc AND the binary-mode output. Use ~0.95 to ignore pure-white reflections that would otherwise dominate the percentile. |
| `exclude_below` | FLOAT | 0.0 | Pixels darker than this are clipped from both percentile and output. Use ~0.05 to ignore pure-black background/line pixels. |
| `top_soft_edge` | FLOAT | 0.03 | Soft edge BELOW the highlight threshold (binary mode only). Pixels at/above threshold = 1, pixels in `[thr - edge, thr]` ramp 0→1. 0 = hard binary. |
| `bottom_soft_edge` | FLOAT | 0.03 | Symmetric for dark mask. |
| `smooth_pixels` | FLOAT | 0.0 | SPATIAL gaussian blur radius (px) applied to both output masks. Softens edges but doesn't change value relationships. Typical 4–32. |
| `gradient_mode` | BOOL | false | When ON: mask values become a smooth LUMA-AWARE ramp. Brightest pixel = white, threshold = black, smooth gradient between. Ignores `soft_edge`. Use when you want "fade from white to black based on luminance" (not just binary cutoff + blur). |
| `gradient_curve` | FLOAT | 1.0 | Shape of the gradient ramp (gradient_mode only). 1.0 = linear. <1 (e.g. 0.5 = sqrt) ramps fast to white. >1 keeps most pixels dark, only brightest reach white. |
| `mask` | MASK | — | Optional region mask. Percentile calc only considers pixels where mask > 0.5. Output is also restricted to the mask region. |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `highlight_mask` | IMAGE | Greyscale 3-channel mask. White where pixels are in the top X% of luminance, black elsewhere. |
| `dark_mask` | IMAGE | Greyscale 3-channel mask. White where pixels are in the bottom Y% of luminance, black elsewhere. |
| `threshold_hi` | FLOAT | The actual luma threshold computed for the highlight mask. |
| `threshold_lo` | FLOAT | The actual luma threshold computed for the dark mask. |

## Diagnostic Output

Every execution prints a status line to the ComfyUI console:

```
[BD_LuminanceMask] top_percent=2.0% → thr_hi=0.7234, bottom_percent=8.0% → thr_lo=0.1245,
                   exclude=[0.050, 0.950], smooth_px=0.0, gradient=off, mask_active=yes
[BD_LuminanceMask] valid pixels in eligible range: 245,760 (saturated ≥0.99: 2.3%, ≥0.95: 4.1%, ≤0.01: 0.4%)
```

If the source has too many tied saturated pixels, a warning is printed suggesting you lower `exclude_above`:

```
[BD_LuminanceMask] ⚠ 87.3% of eligible pixels are saturated (≥0.99) but you asked for top 2.0% —
                   percentile collapses to 1.0 because too many pixels are tied at max.
                   Lower exclude_above below 0.99 to fix.
```

## Common Use: Skin Highlight + Dark Overlay Extraction

```
[mannequin render] ──┬─→ BD_LuminanceMask ─→ highlight_mask ─┐
                     │  (top_percent=2)                       │
                     │  (mask=skin_mask)                      ├─→ BD_PackChannels (G=hi, B=dark)
                     ↓                       dark_mask ───────┘            ↓
                  BD_LuminanceMask                                       u_image3
                  (bottom_percent=8)                                   for skin shader
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `thr_hi = 1.0000` and mask shows huge bright area | Source has too many saturated pixels (white background, reflections). Percentile can't subdivide tied values. | Set `exclude_above = 0.95` or lower |
| Mask doesn't change between different `top_percent` values | Same — percentile collapsed at tied value | Same fix |
| "Broken-up white chunks" instead of smooth fade | Binary threshold with adjacent pixels jittering above/below cutoff | Enable `gradient_mode` for luma-aware ramp, or use `smooth_pixels` for spatial blur |
| Highlight mask returns all black | Default `top_soft_edge=0` would discard everything if percentile=1.0 (no pixels strictly >1) | Use `gradient_mode=on`, or lower `exclude_above` so percentile is < 1 |
| Mask outside skin region is non-zero | No `mask` input wired | Wire your skin mask into the `mask` slot |

## Notes

- All output masks are 3-channel IMAGE (greyscale replicated to RGB) so they feed cleanly into any downstream IMAGE input.
- The `mask` input both **scopes the percentile calc** AND **restricts the output** to that region.
- `exclude_above`/`exclude_below` are the primary defense against saturated/dead-black pixels skewing the percentile.
- Step is 0.05 on `top_percent`/`bottom_percent` so you can dial sub-1% values precisely.
