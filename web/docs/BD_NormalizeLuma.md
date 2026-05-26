# BD Normalize Luma

Auto-rescale image luminance to fit a target range, with mask-aware percentile clipping and an optional proportional-scale mode.

## Overview

Adjusts the luminance range of an image so downstream nodes (like `BD_LuminanceMask`) have something meaningful to work with. Two modes:

- **Range fit (default)**: remaps source `[min, max]` to target `[min, max]`. Stretches or compresses both ends. Lifts black floor, drops white ceiling.
- **Proportional scale**: single multiplier `target_max / src_max`, then a hard clamp at `target_max`. Brightest non-outlier pixel hits exactly `target_max`, outlier pixels (top `clip_percent_high%`) are clamped down to `target_max`. Everything below scales by the same factor (black stays black). Guarantees no output pixel exceeds `target_max`.

Uses `np.percentile` for min/max (not absolute min/max) so a few outlier pixels (specular spikes, dead black) can't define the source range.

## Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | IMAGE | — | Source to normalize. |
| `luma_standard` | COMBO | `bt709` | RGB → luma weighting used for percentile computation. `bt709` (default, modern sRGB), `bt601` (legacy NTSC), `average` ((R+G+B)/3). |
| `target_max` | FLOAT | 0.95 | Brightest output value. Set lower (e.g. 0.85) to compress over-bright. In proportional mode, this is the exact value the brightest pixel maps to. |
| `target_min` | FLOAT | 0.0 | Darkest output value. Set higher (e.g. 0.1) to lift shadows. **Ignored in proportional mode.** |
| `clip_percent_high` | FLOAT | 1.0 | Treat the top X% of valid pixels as the source-max. 1.0 = robust default. 0 = use absolute max (vulnerable to single-pixel spikes). |
| `clip_percent_low` | FLOAT | 1.0 | Treat the bottom X% as source-min. |
| `preserve_color` | BOOL | true | ON: rescale RGB proportionally (keeps hue, just darkens/brightens). OFF: output greyscale based on normalized luma. |
| `proportional_scale` | BOOL | false | OFF (default): range fit mode. ON: proportional scale — single multiplier, `target_min` is ignored, blacks stay at 0. |
| `apply_to_mask_only` | BOOL | true | ON: only pixels within the mask are normalized (outside pass through unchanged). OFF: normalize the whole image. No effect if no mask is wired. |
| `mask` | MASK | — | Optional region mask. Percentile calc only considers pixels where mask > 0.5. When `apply_to_mask_only=true`, output is also restricted. |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | Normalized image. |
| `found_min` | FLOAT | The source luma value detected as min (after `clip_percent_low` filtering). |
| `found_max` | FLOAT | The source luma value detected as max (after `clip_percent_high` filtering). |

## Mode Comparison

For source pixels `[0.1, 0.8]` with target `[0.0, 0.5]`:

| Pixel | Range Fit (proportional_scale=OFF) | Proportional (proportional_scale=ON, target_max=0.5) |
|-------|-------------------------------------|------------------------------------------------------|
| 0.8 (src_max) | 0.5 | 0.5 (scale = 0.625) |
| 0.5 | 0.286 | 0.313 |
| 0.3 | 0.143 | 0.188 |
| 0.1 (src_min) | 0.0 (lifted to target_min) | 0.063 |
| 0.0 | clamp to 0 | **0.0 (preserved)** |

## Diagnostic Output

```
[BD_NormalizeLuma] source range: [0.0234, 0.9987] → RANGE FIT [0.000, 0.850],
                   clip=[1.0%, 1.0%], preserve_color=yes, mask_active=yes, applied_to=mask only
```

In proportional mode:
```
[BD_NormalizeLuma] source range: [0.0234, 0.9987] → PROPORTIONAL scale=0.5006,
                   clip=[1.0%, 1.0%], preserve_color=yes, mask_active=yes, applied_to=mask only
```

## Common Use: Fix Saturation Before BD_LuminanceMask

When `BD_LuminanceMask` reports `thr_hi = 1.0000` because too many pixels are tied at white, insert this node first:

```
[source]  ──→ BD_NormalizeLuma  ──→ BD_LuminanceMask  ──→ highlight_mask
              (target_max=0.85)     (now percentile actually works)
              (mask=skin_mask) ──┐
                                 │
[skin_mask] ───────────────────┴─→ (mask, same input)
```

After normalization: brightest skin pixel = 0.85, darkest = 0.0, spread across the range. Percentile now distinguishes pixels properly — "top 2%" is actually ~2% of pixels.

## Common Use: Shift Brightest Down to Mid-Grey

To dim the whole image proportionally without lifting the floor:

```
proportional_scale = ON
target_max = 0.5           ← mid-grey ≈ 127/255
clip_percent_high = 1.0    ← ignore top 1% specular outliers
preserve_color = ON
```

Result: brightest non-outlier skin pixel → 0.5, 0.7 → 0.35, 0.3 → 0.15, 0.0 → 0.0. **Outlier pixels (top 1%) are hard-clamped to 0.5**, so no pixel in the output exceeds mid-grey.

**Don't forget to check `apply_to_mask_only` and the mask scope.** If `apply_to_mask_only=ON` and the mask only covers skin, pixels OUTSIDE the mask (clothes, background, eyes) pass through UNCHANGED — they can still be brighter than mid-grey because they were never touched. To enforce the ceiling globally, either:
- Turn `apply_to_mask_only=OFF`, or
- Don't wire a mask

## Common Use: Lift Underexposed Shadows

For too-dark sources:

```
proportional_scale = OFF (range fit)
target_max = 0.95
target_min = 0.10        (lift the floor)
clip_percent_low = 1.0   (ignore deepest 1% as outliers)
```

Result: darkest valid skin pixel → 0.10, brightest → 0.95, spread across.

## Notes

- `preserve_color` rescales each RGB channel by the same factor — saturation can clip if push is large.
- `apply_to_mask_only` uses the mask value directly for blending. Soft mask edges give natural feather between modified and original areas.
- Alpha channel (if present in input) is always preserved unchanged.
- The two found_* float outputs are useful for routing into downstream nodes that need to know the actual values used.
