# BD Unpack Channels

Split an RGBA image into its 4 individual channels.

## Overview

Inverse of `BD_PackChannels`. Takes one packed image and outputs the R, G, B, A channels separately, each in both IMAGE (greyscale 3-channel) and MASK (single-channel) form so they connect cleanly to any downstream node. Also provides a `debug_preview` matching the layout of `BD_PackChannels` for visual verification.

Common use: pull apart the `u_image3` RGBA pack used by the skin shader to verify which channel has which data, or to feed individual channels of a packed asset into other nodes.

## Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | IMAGE | — | Source RGB or RGBA image to unpack. |
| `alpha_default` | FLOAT | 1.0 | Value used for the alpha output when the input image is RGB (no real alpha channel). 1.0 = fully opaque. |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `red_image` | IMAGE | R channel as greyscale 3-channel image (R replicated to RGB). |
| `green_image` | IMAGE | G channel as greyscale 3-channel image. |
| `blue_image` | IMAGE | B channel as greyscale 3-channel image. |
| `alpha_image` | IMAGE | A channel as greyscale 3-channel image. |
| `red_mask` | MASK | R channel as single-channel mask. |
| `green_mask` | MASK | G channel as single-channel mask. |
| `blue_mask` | MASK | B channel as single-channel mask. |
| `alpha_mask` | MASK | A channel as single-channel mask. |
| `debug_preview` | IMAGE | 2×2 grid of all four channels tinted in their own colors (same layout as `BD_PackChannels`). |

## Debug Preview Layout

```
┌──────────┬──────────┐
│   RED    │  GREEN   │
│ (R chan) │ (G chan) │
├──────────┼──────────┤
│   BLUE   │  WHITE   │
│ (B chan) │ (A chan) │
└──────────┴──────────┘
```

R as red, G as green, B as blue, A as white. Matches `BD_PackChannels` so visual inspection of a round-trip (pack → unpack) shows identical layouts.

## IMAGE vs MASK Outputs

Both formats are provided for downstream flexibility:

| Use case | Pick |
|----------|------|
| Plug into a node expecting IMAGE (most preview, save, image-processing nodes) | `{channel}_image` |
| Plug into a node expecting MASK (compositing, mask-aware nodes) | `{channel}_mask` |
| Plug into `BD_PackChannels.{channel}_mask` to re-route a channel | `{channel}_mask` |
| Plug into `BD_PackChannels.{channel}_image` | `{channel}_image` |

## Diagnostic Output

```
[BD_UnpackChannels] mean values: R=0.523, G=0.000, B=0.039, A=1.000, input_channels=4
```

The mean values match what `BD_PackChannels` reported when the pack was built — useful for diagnosing whether data survived through intermediate processing or whether a channel was zeroed somewhere.

## Common Use: Verify Skin Shader Pack

After `BD_PackChannels` builds the `u_image3` pack, wire it through `BD_UnpackChannels` → `debug_preview` to confirm each channel has what you expected:

```
[shadow + hi + dark + lines]  →  BD_PackChannels  →  u_image3 to shader
                                         ↓
                                BD_UnpackChannels  →  debug_preview → preview node
```

Or unpack the pack to check individual channels in isolation:

```
[u_image3 source]  →  BD_UnpackChannels  ┬→  red_mask    (verify shadow data)
                                          ├→  green_mask  (verify highlight data)
                                          ├→  blue_mask   (verify dark data)
                                          └→  alpha_mask  (verify line data)
```

## Notes

- When the input is RGB (3 channels), the alpha output is filled with `alpha_default`.
- All IMAGE outputs are 3-channel greyscale (channel value replicated across RGB) so they pass through standard image-processing pipelines.
- All MASK outputs are single-channel, the native form for single-value data.
- No data loss — this is a pure split operation. Mean values match input.
