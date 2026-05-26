# BD Color Extract

Extract pixels matching a target color (with tolerance) from an image as a greyscale mask.

## Overview

Designed for **painted-color extraction**. If you paint shadows in blue (or any specific color) onto a base image, this node pulls out all the painted pixels as a mask. Output feeds cleanly into `BD_PackChannels` for downstream use (e.g. feeding the GLSL skin shader's `u_image3` overlay channels).

Two match modes:
- **`hue`** (default) — matches by color identity in HSV space. Brightness-independent: "any blue" whether dark or light, saturated or muted, contributes. Best for painted color extraction.
- **`rgb_distance`** — strict Euclidean distance in RGB space. Use when you want only pixels very close to a specific RGB triplet.

## Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | IMAGE | — | Source image to extract color from. |
| `target_r` | FLOAT | 0.0 | Target color RED component (0-1). Default 0 (for blue). |
| `target_g` | FLOAT | 0.0 | Target color GREEN component. Default 0 (for blue). |
| `target_b` | FLOAT | 1.0 | Target color BLUE component. Default 1 (for blue). |
| `match_mode` | COMBO | `hue` | Matching algorithm. `hue` = color identity (brightness-independent), `rgb_distance` = exact RGB triplet match. |
| `tolerance` | FLOAT | 0.15 | How loose the match is. In hue mode: hue-unit tolerance (0.15 ≈ ±54°). In rgb_distance: RGB-space distance (0.15 ≈ tight match). Lower = stricter. |
| `min_saturation` | FLOAT | 0.10 | HUE MODE ONLY. Pixels with saturation below this are ignored. Prevents greys/whites from accidentally matching (their hue calc is unreliable). Set 0 to allow desaturated matches. |
| `soft_edge` | FLOAT | 0.05 | Smoothstep falloff around the tolerance boundary. 0 = hard binary cutoff. Bigger = softer edge. |
| `gradient_mode` | BOOL | false | When ON: output is a smooth gradient based on HOW CLOSE each pixel is to the target. Pixel at target = 1.0, fades with distance. Ignores `soft_edge`. Best for "how much of this color tint" extraction. |
| `invert` | BOOL | false | Invert the mask (1 - output). Extracts everything EXCEPT the target color. |
| `mask` | MASK | — | Optional region mask. Output is zero outside this region (restricts to e.g. skin only). |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `color_mask` | IMAGE | Greyscale 3-channel image. White where pixels match the target color, black elsewhere. |
| `mask` | MASK | Single-channel MASK version for nodes that expect MASK type. |

## Mode Comparison: Extracting "Blue Shadows"

Say your source has white skin with blue-painted shadow areas of varying brightness:
- pixel A: pure white (1, 1, 1) — base skin
- pixel B: pure blue (0, 0, 1) — saturated shadow paint
- pixel C: dark blue (0, 0, 0.3) — dark shadow paint
- pixel D: blue-tinted grey (0.4, 0.4, 0.6) — semi-transparent paint over skin
- pixel E: pure red (1, 0, 0) — wrong color

| Mode | A (white) | B (sat blue) | C (dark blue) | D (blue-tint) | E (red) |
|------|-----------|--------------|---------------|---------------|---------|
| **hue** (default) | 0 (low sat) | **1.0** (perfect match) | **1.0** (perfect hue) | ~0.3 (matches hue but low sat) | 0 (wrong hue) |
| **rgb_distance** (tolerance=0.15) | 0 (far) | **1.0** (exact match) | 0 (different RGB) | 0 (different RGB) | 0 (far) |

For painted shadow extraction, **hue mode is the right choice** — it catches all the blue-tinted pixels regardless of brightness.

## Common Use: Blue Shadow Paint → Skin Shader Overlay

```
[base + blue painted shadows]
        ↓
   BD_ColorExtract
        target = (0, 0, 1)  ← blue
        match_mode = hue
        mask = skin_mask    ← restrict to skin area only
        ↓
   color_mask  ────────→  B input of BD_PackChannels (dark overlay)
                              ↓
                          u_image3 ────→ skin GLSL shader
```

Pair with another `BD_ColorExtract` for other painted colors:
- Red-painted highlights → G input (highlight overlay)
- Yellow-painted reflections → another channel
- Etc.

## Common Use: Extract Specific Tint

To grab all pixels of a specific color (say, the lavender used for clothing):

```
target = (0.7, 0.6, 0.85)
match_mode = hue
tolerance = 0.08              ← strict hue match
min_saturation = 0.15         ← must be saturated, not grey
soft_edge = 0.02
```

## Common Use: Extract Skin-Tone Pixels (Inverse)

To get a mask of NON-skin pixels (background, clothes):

```
target = (0.85, 0.65, 0.55)   ← typical skin tone
match_mode = hue
tolerance = 0.10
invert = ON                   ← invert to get non-skin
```

## Diagnostic Output

Every execution prints to console:

```
[BD_ColorExtract] target_rgb=(0.00,0.00,1.00), mode=hue, tolerance=0.150, min_sat=0.10,
                  gradient=off, invert=no, mask_active=yes, matched pixels (>0.5): 12.3%
```

`matched pixels (>0.5)` tells you what fraction of the image actually got marked — useful for tuning tolerance.

## Notes

- HSV hue is circular: red at 0°, yellow at 60°, green at 120°, cyan at 180°, blue at 240°, magenta at 300°, back to red at 360°. `tolerance=0.15` in hue units ≈ 54°, so a tolerance of 0.15 around blue (~0.667) catches everything from cyan-blue (~0.55) to violet (~0.78).
- For monochrome targets (pure white/black/grey), hue is meaningless — use `rgb_distance` mode instead.
- Use `min_saturation` to filter out desaturated pixels in hue mode (greys have unreliable hue values).
- `gradient_mode` is great for "how much of this color is here" — produces a soft mask weighted by closeness.
- Soft edge values are relative to the tolerance scale (hue units in hue mode, RGB-space in rgb_distance mode).
