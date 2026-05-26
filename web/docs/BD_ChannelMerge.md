# BD Channel Merge

Blend or composite a source image into specific RGBA channels of a base image, using Photoshop-style blend modes. The source image's own alpha channel can act as a soft compositing mask so that background transparency is automatically ignored.

Designed for progressively building up texture packs in a ComfyUI graph: pass a packed image through several BD_ChannelMerge nodes, each one modifying a different channel, without having to re-pack from scratch.

## What it does

Takes a **base** image (the existing pack) and a **source** image to merge into it. Writes the blended result into only the specified target channels — all other channels pass through unchanged.

```
blend_mask = strength × (source.alpha if use_source_alpha) × (external mask if wired)
blended    = blend_mode(base_channel, source_value)
output     = base_channel × (1 − blend_mask) + blended × blend_mask
```

## Why a separate node from BD_PackChannels

| Node | Role |
|---|---|
| **BD_PackChannels** | Constructor — 4 sources → 1 packed image (build from scratch) |
| **BD_ChannelMerge** | Modifier — existing packed image + source → updated pack (in-place edit) |

Use BD_PackChannels to create the initial pack. Use BD_ChannelMerge to update individual channels as more data becomes available downstream in the graph.

## The alpha mask (key feature)

When `use_source_alpha=True` (default), the source image's alpha channel is used as a soft blend mask:

- **Source alpha = 1.0** (opaque) → full blend applies to base channel
- **Source alpha = 0.0** (transparent) → base channel is left completely unchanged

This lets you composite a character render (which has a transparent background) onto a single channel of a packed texture without the background corrupting the data. The alpha boundary acts as a cut-out, exactly like Photoshop's "paste into" with layer masking.

The final effective blend amount per pixel is:
```
effective = strength × source_alpha × external_mask
```
All three multiply together, so any of them alone can gate the blend.

## Target channels

| Setting | What gets written | Source value |
|---|---|---|
| `R` / `G` / `B` / `A` | One channel | Scalar extracted from source (see `source_from`) |
| `RG` / `RB` / `GB` | Two channels | Source R→first target, source G→second target |
| `RGB` | Three channels | Source R, G, B → base R, G, B |
| `RGBA` | All four | Source R, G, B, A → base R, G, B, A |

For multi-channel targets, `source_from` is ignored — source channels map 1:1.

## Source extraction (`source_from`)

Controls how a multi-channel source image is collapsed to a scalar when targeting a single channel:

| Option | Formula |
|---|---|
| `luma_bt709` (default) | `0.2126R + 0.7152G + 0.0722B` — perceptual greyscale |
| `channel_R/G/B/A` | Pick one channel directly |
| `average` | `(R + G + B) / 3` — simple mean |
| `max_rgb` | `max(R, G, B)` — nearest to perceived brightness |

Always use `luma_bt709` when the source and target workflow both use BT.709 luma (e.g. the skin shader pipeline) to avoid drift.

## Blend modes

| Mode | Formula | Effect |
|---|---|---|
| `replace` | `src` | Overwrite — standard paste |
| `add` | `clamp(base + src, 0, 1)` | Brightens; black src = no change |
| `multiply` | `base × src` | Darkens proportionally; white src = no change |
| `screen` | `1 − (1−base)(1−src)` | Brightens; black src = no change |
| `overlay` | multiply/screen split at 0.5 | Increases contrast both ways |
| `darken` | `min(base, src)` | Keeps darker pixel |
| `lighten` | `max(base, src)` | Keeps lighter pixel |
| `subtract` | `clamp(base − src, 0, 1)` | Darkens by source value |
| `difference` | `|base − src|` | Shows deviation from source |
| `soft_light` | Pegtop formula | Gentle S-curve contrast; grey src = no change |

## Inputs

| Input | Type | Default | Purpose |
|---|---|---|---|
| `base` | IMAGE | — | Existing image to modify. RGB or RGBA; untargeted channels pass through unchanged. |
| `source` | IMAGE | — | Image to merge from. Auto-resized to match base dimensions. |
| `target_channels` | COMBO | `R` | Which channels of base to write to. |
| `source_from` | COMBO | `luma_bt709` | How to extract a scalar from source for single-channel targets. |
| `blend_mode` | COMBO | `replace` | Photoshop-style compositing operation. |
| `strength` | FLOAT | `1.0` | Overall blend opacity 0–1. Multiplied with other masks. |
| `use_source_alpha` | BOOL | `True` | Use source's alpha as soft mask (transparent areas don't affect base). |
| `invert_alpha_mask` | BOOL | `False` | Invert the alpha mask (blend where source IS transparent). |
| `mask` | MASK | optional | Additional external blend mask. Multiplied with strength and source alpha. |
| `luma_standard` | COMBO | `bt709` | Luma weighting for the `luma_bt709` source_from path. |

## Outputs

| Output | Type | Purpose |
|---|---|---|
| `image` | IMAGE | Modified base. Same resolution as base. RGBA if base was RGBA or target includes A; otherwise matches base channel count. |

## Alpha channel handling

- **Base is RGB (3ch), target doesn't include A** → output is RGB (3ch)
- **Base is RGB (3ch), target includes A** → alpha channel is initialized to 1.0 (fully opaque), then written; output is RGBA (4ch)
- **Base is RGBA (4ch), any target** → alpha is preserved unless A is in target_channels

## Pipeline patterns

### Pattern A — Build a pack progressively

```
shadow_map  → BD_ChannelMerge  (target=R, mode=replace)  → partial_pack
              ↑
ILM_map     → BD_ChannelMerge  (target=G, mode=replace)  → partial_pack_2
              ↑
roughness   → BD_ChannelMerge  (target=B, mode=replace)  → final_pack
              ↑
BD_PackChannels (seed pack with defaults)
```

Equivalent to BD_PackChannels but lets you insert processing nodes between each channel assignment.

### Pattern B — Character composite onto a channel (the main use case)

```
character_photo (RGBA, transparent BG)
  → BD_ImageToGreyscale (luma)
  → BD_NormalizeLuma
  → BD_CenterMedianLuma
  → BD_ChannelMerge (
        base = existing_pack,
        target_channels = R,
        source_from = luma_bt709,
        blend_mode = replace,
        use_source_alpha = True    ← ignores transparent BG
    )
  → u_image0 of skin shader
```

The transparent background of the character photo never corrupts the R channel data because `use_source_alpha=True` masks it out at exactly the alpha boundary.

### Pattern C — Non-destructive shadow overlay

```
existing_pack
  → BD_ChannelMerge (
        source = shadow_map,
        target_channels = R,
        blend_mode = multiply,         ← darkens existing R by shadow
        strength = 0.6,
        use_source_alpha = False       ← shadow map has no meaningful alpha
    )
  → final_pack
```

`multiply` preserves existing data: where shadow_map = 1.0 (fully lit), base R is unchanged. Where shadow_map = 0.5, base R darkens by half.

### Pattern D — Depth AO baked into alpha channel

```
BD_DepthToShadowMap → cavity_map
  → BD_ChannelMerge (
        base = diffuse_rgb,         ← 3-ch input
        target_channels = A,        ← auto-expands to RGBA output
        blend_mode = replace,
        use_source_alpha = False
    )
  → 4ch RGBA output with AO baked in alpha
```

### Pattern E — External mask gates a channel write

```
BD_ChannelMerge (
    base = current_pack,
    source = highlight_map,
    target_channels = G,
    blend_mode = lighten,
    strength = 0.8,
    mask = skin_only_mask,           ← external mask: only affects skin area
    use_source_alpha = False
)
```

## Common issues

| Symptom | Likely cause | Fix |
|---|---|---|
| Entire image changes (including BG) | `use_source_alpha=False` with full-frame source | Enable `use_source_alpha=True` if source has alpha |
| Target channel untouched | `strength=0` or mask is all-black | Check strength > 0; verify mask |
| Wrong luma match vs shader | `source_from` mismatch | Always use `luma_bt709` when paired with skin shader |
| 3-ch output when expecting 4-ch | A not in `target_channels` + 3-ch base | Include A in target, or start with a 4-ch base |
| Output too dark with `multiply` | Source has dark pixels in non-subject area | Enable `use_source_alpha` or provide an external mask |

## Pairs With

- **BD_PackChannels** — create the initial pack; BD_ChannelMerge modifies it downstream
- **BD_UnpackChannels** — extract individual channels for inspection or routing
- **BD_GLSLBatch** — the final consumer of packed texture inputs (u_image0–4)
- **BD_DepthToShadowMap** — produce a shadow map to merge into the R channel (u_image3)
- **BD_CenterMedianLuma** — normalize source before merging to avoid V-curve drift
- **BD_SaveBatch** — save the resulting pack after merges are done
