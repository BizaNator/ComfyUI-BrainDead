# BD Pack Channels

Combine up to 4 source images/masks into the RGBA channels of one output image.

## Overview

Inverse of channel splitting. Common use cases:

- **Game engine packed textures**: roughness in R, metalness in G, AO in B, height in A
- **Skin shader overlay pack**: shadow map in R, highlight mask in G, dark mask in B, line mask in A (u_image3 for the skin shader)
- **Normal map repackaging**: separate XYZ channels into RGB
- **Combining greyscale outputs**: depth + normal + segmentation into one image

Each channel slot accepts either an IMAGE or a MASK input (or both — MASK wins since it's already single-channel and unambiguous). When an IMAGE is provided, its luminance is computed (`0.299 R + 0.587 G + 0.114 B`). When neither is provided, the channel is filled with the `default_value` for that channel.

## Inputs

Each of the four channels (R, G, B, A) has the same set of inputs:

| Per-Channel Input | Type | Default | Description |
|-------------------|------|---------|-------------|
| `{channel}_image` | IMAGE | — | Source image — luminance becomes the channel value. |
| `{channel}_mask` | MASK | — | Source mask — used directly. Wins over the image if both provided. |
| `{channel}_default` | FLOAT | 0.0 (1.0 for alpha) | Fill value when neither image nor mask is wired. |
| `{channel}_invert` | BOOL | false | Invert the resolved channel before packing (`1 - x`). |

Additional inputs:

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `output_alpha` | BOOL | false | If true, output is 4-channel RGBA. Wiring any alpha source auto-enables this. |
| `width_override` | INT | 0 | Override output width. 0 = use first non-None source's width. |
| `height_override` | INT | 0 | Override output height. 0 = use first non-None source's height. All channels are resized to this dimension. |
| `luma_standard` | COMBO | `bt709` | When an IMAGE input is provided (not MASK), this controls the RGB → luma weighting used to reduce it to a single channel. `bt709` (modern sRGB default), `bt601` (legacy NTSC), `average`. Does not affect MASK inputs (already single-channel). |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | The packed image (RGB or RGBA depending on `output_alpha`). |
| `alpha` | MASK | The resolved alpha channel as a single-channel mask. |
| `debug_preview` | IMAGE | 2×2 grid showing each channel tinted in its own color (R top-left as red, G top-right as green, B bottom-left as blue, A bottom-right as white). Useful for verifying you wired what you think you wired. |

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

Each channel renders as solid color on black — intensity matches the channel value. Lets you tell at a glance which channels have data and where. A pixel that's only in the R channel will show bright red top-left, black elsewhere.

## Diagnostic Output

Every execution prints mean values per channel:

```
[BD_PackChannels] mean values: R=0.523, G=0.000, B=0.039, A=1.000, output_alpha=YES
```

Compare with `BD_UnpackChannels` output — values round-trip identically when there's no compression in between.

## Image vs Mask Input Conventions

| Source type wired | Behavior |
|-------------------|----------|
| `{channel}_image` only | Luminance of the image becomes the channel value |
| `{channel}_mask` only | Mask value used directly (single-channel) |
| Both | **Mask wins** (unambiguous, no luma computation needed) |
| Neither | Filled with `{channel}_default` |

## Common Use: Skin Shader Overlay Pack

```
[3D shadow render]               → R ──┐
[BD_LuminanceMask highlight]     → G ──┤  BD_PackChannels
[BD_LuminanceMask dark]          → B ──┤  (output_alpha = true)
[line drawing]                   → A ──┘            ↓
                                                  u_image3
                                              for skin shader
```

Set `output_alpha = true` to get the 4-channel pack. Use `debug_preview` to verify each channel landed where you expected before feeding into the shader.

## Notes

- Auto-enables `output_alpha` when any alpha source is wired.
- `invert` lets you flip a channel without needing a separate invert node upstream (handy when your source has the wrong convention).
- All channels are resized to match (uses bilinear interpolation when sources differ in resolution).
- `width_override` / `height_override` force a specific output size — useful when packing into a fixed-size shader texture.

## Pairs With

- **BD_ChannelMerge** — modify individual channels of an existing pack downstream without rebuilding from scratch. Use BD_PackChannels to create the initial pack, BD_ChannelMerge to update it as more data arrives later in the graph.
- **BD_UnpackChannels** — split a packed image back into individual channels for inspection or routing.
- **BD_GLSLBatch** — the primary consumer of packed textures (u_image0–4).
- **BD_DepthToShadowMap** — produces a shadow map suitable for packing into the R channel of u_image3.
