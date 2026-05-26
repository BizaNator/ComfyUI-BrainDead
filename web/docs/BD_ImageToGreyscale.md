# BD Image To Greyscale

Convert an RGB/RGBA image to greyscale using a selectable conversion mode.

## Overview

Single node that handles all common ways of reducing a color image to a single brightness value per pixel. Supports both perceptually accurate luminance weights (BT.709 modern, BT.601 legacy) and simpler modes like channel average, max channel, or pure single-channel passthrough.

## Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | IMAGE | — | Source image (RGB or RGBA — alpha is dropped). |
| `mode` | COMBO | `luminance` | Conversion mode (see table below). |
| `mask_mode` | COMBO | `apply_within` | Controls how the optional `mask` input affects output. `apply_within`: greyscale only inside the mask, original color outside. `cutout`: legacy multiply behavior — masked area shows grey, outside is zeroed black. |
| `mask` | MASK | — | Optional mask. Soft edges give natural feather. See `mask_mode` for behavior. |

## Modes

| Mode | Formula | When to use |
|------|---------|-------------|
| `luminance` (default) | 0.2126·R + 0.7152·G + 0.0722·B (BT.709) | Modern sRGB / games / AI pipelines. **Matches Unity/Unreal internal luma.** Perceptually accurate for modern displays. |
| `luminance_bt601` | 0.299·R + 0.587·G + 0.114·B (BT.601 / NTSC) | Legacy SD video weights. Matches Photoshop's "Desaturate" and older print tools. Use when matching outputs from legacy pipelines. |
| `average` | (R + G + B) / 3 | Simple, not perceptual. Useful when matching naive averaging tools. |
| `max_channel` | max(R, G, B) | "Brightness" in the colorimetric sense (HSV value). Picks brightest channel per pixel. |
| `red` | R only | Pass through a single channel as greyscale. |
| `green` | G only | Same. |
| `blue` | B only | Same. |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | 3-channel greyscale (R=G=B=luma, alpha=1.0). Plug into SaveImage, GLSL uniforms, etc. |
| `mask` | MASK | Single-channel mask version. Plug into mask-aware downstream nodes. |

## Common Use Cases

### Source for `BD_LuminanceMask`

```
[mannequin render] → BD_ImageToGreyscale (mode=luminance) → BD_LuminanceMask (mask=skin_mask)
```

Reduces a color render to a single luminance channel before percentile extraction. (`BD_LuminanceMask` already does luma internally, but this lets you preview/inspect the greyscale step.)

### Channel extraction for shader inputs

```
[normal map] → BD_ImageToGreyscale (mode=red)   → height map (X component)
            └→ BD_ImageToGreyscale (mode=green) → height map (Y component)
            └→ BD_ImageToGreyscale (mode=blue)  → height map (Z component)
```

### Match a legacy Photoshop "Desaturate" output

```
[source]  → BD_ImageToGreyscale (mode=luminance_bt601) → matches Photoshop
```

## Mask Mode Behaviors

When the optional `mask` input is wired:

| Mode | Inside mask (m=1) | Outside mask (m=0) | When to use |
|------|-------------------|--------------------|-------------|
| `apply_within` (default) | Shows greyscale conversion | Shows **original color** (untouched) | Greyscale only a region (e.g. skin) while keeping the rest in color |
| `cutout` | Shows greyscale conversion | Shows **black** (zeroed out) | Legacy/isolation use — discard everything outside the mask |

Soft mask values (0 < m < 1) blend naturally between the two states in both modes — `apply_within` blends from greyscale to original color, `cutout` blends from greyscale to black.

## Notes

- For the BD pipeline, prefer `luminance` (BT.709) — it matches everything else in the BrainDead skin shader / Unity stack.
- `luminance_bt601` exists specifically for matching legacy outputs; don't use it for new work.
- The `mask` output is single-channel — the IMAGE output replicates the same value across R, G, B.
- **The mask input was previously called `alpha_mask` and only did `cutout` behavior.** Renamed to `mask` with the new `apply_within` default that matches typical "process only this region" intent.
