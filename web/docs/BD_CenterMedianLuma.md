# BD Center Median Luma

Additive luma-shift so the (optionally masked) median of an image lands exactly on a target value. Designed for V-curve-driven shaders that have a fixed midpoint uniform but receive inputs with drifting medians per character.

## What it does

```
shift = target_center - median(luma(image[mask]))
output = clamp(image + shift, 0, 1)
```

A single constant offset is added to every pixel's RGB. The luma median of the masked region becomes exactly `target_center`. **Spread, contrast, and histogram shape are preserved** — only position changes.

## Why it exists — the V-curve drift problem

The skin-tinting GLSL shader (and any V-curve-based effect) uses a midpoint uniform (e.g. `u_float0 = 0.5`) where alpha = 0 → 100% tint shows. For the tint to actually dominate the skin area, the input image's luma median must sit at that midpoint. Per-character source images have different lighting / albedo / stylization, so the median drifts:

| Character | Source median (post-normalize) | Drift from 0.5 |
|---|---|---|
| Linda (overlit photo) | 0.67 | +0.17 |
| Don Juan (balanced) | 0.50 | 0.00 |
| Hypothetical underlit char | 0.32 | −0.18 |

Without centering, you'd need a different `u_float0` per character. With this node, the median always lands at `target_center` and you set `u_float0` once.

## Difference from BD_NormalizeLuma

| Operation | Controls | Preserves |
|---|---|---|
| **BD_NormalizeLuma** (min/max remap) | The SPREAD (dynamic range) | The histogram SHAPE inside the new bounds — median ends up wherever input distribution placed it |
| **BD_CenterMedianLuma** (additive shift) | The POSITION (median location) | The SPREAD, the contrast, the histogram shape — bodily shift only |

They're complementary. Standard pipeline uses both in sequence:

```
input
  → BD_NormalizeLuma          # sets spread (e.g. min=0.10, max=0.85)
  → BD_CenterMedianLuma       # sets median position (e.g. 0.50)
  → V-curve shader input
```

Normalize doesn't center; center doesn't normalize.

## Inputs

| Input | Type | Default | Purpose |
|---|---|---|---|
| `image` | IMAGE | — | Source (single or batch — each frame centered independently). |
| `luma_standard` | COMBO | `bt709` | RGB→luma weighting. `bt709` (sRGB/modern, default), `bt601` (legacy NTSC), `average` (simple R+G+B/3). |
| `target_center` | FLOAT | 0.5 | Desired median after centering. **Match your V-curve midpoint uniform.** |
| `statistic` | COMBO | `median` | `median` (robust to outliers — recommended). `mean` (smoother but biased by highlights/shadows). |
| `apply_to_mask_only` | BOOL | True | ON: only shift pixels inside the mask; outside pixels pass through unchanged. OFF: shift whole frame (mask still drives the median calc). No effect if no mask wired. |
| `preserve_alpha` | BOOL | True | Keep alpha channel unchanged. Almost always ON. |
| `mask` | MASK | optional | **Strongly recommended.** Region to compute median over (e.g. skin mask). Without it, background pixels bias the median (a white BG pushes shift wrong direction). |

## Outputs

| Output | Type | Purpose |
|---|---|---|
| `image` | IMAGE | Centered image. RGB shifted by the same constant; alpha untouched (if preserve_alpha=ON). |
| `measured_median` | FLOAT | The pre-shift median of the masked pixels. Useful for diagnostics, wiring elsewhere, or as a value to set a uniform with dynamically. |
| `shift_applied` | FLOAT | `target_center − measured_median`. Negative if input was too bright, positive if too dark. |

## Luma standards

`luma_standard` selects how RGB → luminance is computed for the median calculation. Standardized across all BD nodes via `utils/luma.py`:

| Standard | Weights (R, G, B) | When to use |
|---|---|---|
| **bt709** (default) | (0.2126, 0.7152, 0.0722) | Modern sRGB / HD video / Unity / Unreal / AI pipelines. **Use this unless you have a reason not to.** Matches the GLSL skin shader's internal luma. |
| **bt601** | (0.2990, 0.5870, 0.1140) | Legacy NTSC. Matches Photoshop "Desaturate" and older tools. Use if you're trying to bit-match a legacy pipeline output. |
| **average** | (0.333, 0.333, 0.333) | Simple R+G+B/3. Not perceptual — included for matching naive averaging tools. |

For the V-curve shader, **always use bt709** — the shader's internal luminance dot uses bt709, and any mismatch creates a drift between what this node thinks is "median" and what the shader sees.

## Pipeline patterns

### Pattern A — Standard V-curve preprocessing (most common)

```
photo (color)
  → BD_ImageToGreyscale    (luma_standard=bt709, mode=luminance)
  → BD_NormalizeLuma       (target_min=0.10, target_max=0.85, mask=skin)
  → BD_CenterMedianLuma    (target_center=0.5, mask=skin)
  → u_image0 of skin shader
```

Set shader `u_float0 = 0.5` and it works for every character.

### Pattern B — Centering only (skip explicit normalize)

If your input is already a controlled greyscale render (no wild dynamic range issues), you can skip normalize and let center do the only work needed:

```
controlled greyscale render
  → BD_CenterMedianLuma    (target_center=0.5)
  → u_image0
```

The spread will be whatever the source has — fine if it's already reasonable.

### Pattern C — Centering a multi-frame batch (per-tone variants)

`BD_CenterMedianLuma` iterates per-batch-frame (`for i in range(B)`), computing each frame's median independently. So if you batch 4 differently-shaded greyscale frames for the multi-tone iteration:

```
[light_grey, medium_grey, dark_grey, zombie_grey]   (B=4)
  → BD_CenterMedianLuma    (target_center=0.5)
  → batched u_imageN
```

Each frame is centered independently — the zombie frame's typically lower median gets shifted +0.20 while a normal frame might only shift +0.02.

### Pattern D — Use measured_median to set a uniform dynamically

If you don't want to shift the image but instead want the shader to track the input's median, wire `measured_median` directly to a shader uniform (via floats or vary_floats override):

```
input → BD_CenterMedianLuma (or BD_LuminanceMask for median only)
        → measured_median (FLOAT)
        → BD_GLSLBatch.floats  (as 'u_float0=…')
```

Less common but useful when the image content must stay un-shifted (e.g. you need exact source RGB for downstream nodes).

## Debugging

If centering "doesn't seem to work":

1. **Check the console log:**
   ```
   [BD_CenterMedianLuma] luma=bt709, stat=median, measured=0.6683 → target=0.5000, shift=-0.1683, mask_active=yes, applied_to=mask only
   ```
   If `shift` is large but visible output looks unchanged, the node IS running — you may be measuring whole-body pixels instead of skin-only and the non-skin pixels are pulling the apparent median in a different direction.

2. **Wire the skin mask.** Without `mask`, background pixels skew the median. A white background pushes the measured median up → centering then darkens the image too much. A black background does the opposite.

3. **Confirm `apply_to_mask_only` matches intent.** If the background is bright/dark and you turned this OFF, the background also gets shifted (often not what you want).

4. **Output median is at target only WITHIN THE MASKED REGION** (when apply_to_mask_only=ON). Whole-frame median will differ because non-skin pixels are passed through unchanged.

## Pairs With

- **BD_NormalizeLuma** — set spread first, then center.
- **BD_ImageToGreyscale** — collapse to single channel before centering for cleanest luma math.
- **BD_GLSLBatch** + skin shader — the V-curve consumer that needs centered inputs.
- **BD_LuminanceMask** — if you want to compute median without shifting, use BD_LuminanceMask's percentile output instead.
