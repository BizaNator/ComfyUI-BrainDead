# BD Parts Builder 2 (Qwen 2.1)

Decompose a character headshot into asset layers using Qwen Image 2.1 edit prompts only (no SAM), with every layer in the source's own frame.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | The headshot (e.g. HeadMasters MASTER 1). Its RGB goes to every edit exactly as given. |
| `model` | MODEL | Qwen Image 2.1, wired through `QwenImage21Cache`. |
| `clip` | CLIP | `qwen3vl_8b`, loaded with CLIPLoader type `qwen_image`. |
| `vae` | VAE | The Qwen Image 2.1 VAE (decodes RGBA). |
| `presence` | STRING (multiline) | The vision model's JSON answer (`AILab_QwenVL` running `BD_PartsVocabulary`'s `vlm_prompt`), or a typed list such as `hair, eyes, mouth, hat`. Empty = nothing is extracted. |
| `vocabulary` | STRING (multiline) | From `BD_PartsVocabulary`: one line per item (key, ask, noun, accessory). |
| `head_mask` | MASK (optional) | Where the head is (1 = head), e.g. the image's alpha. It only marks the region; the RGB is not re-composited. Not wired: every non-white pixel counts as head. |
| `depth_image` | IMAGE (optional) | `BD_Lotus2Predict` `map` or `raw_linear` - **brighter = nearer**. Only breaks paint-order ties; an item that owns no visible pixels sorts farthest. |
| `seed` | INT | Seed for every edit. Default 20260931. |
| `steps` | INT | Sampling steps. Default 40 (the Qwen Image 2.1 model card value for edits). |
| `cfg` | FLOAT | CFG. Default 1.0. |
| `sampler_name` | COMBO | Default `euler`. |
| `scheduler` | COMBO | Default `simple`. |
| `make_complete` | BOOL | Also make the COMPLETE layers. Default on. |
| `make_base` | BOOL | Also make the head without items. Default on. |
| `colour_retry` | BOOL | One retry that names the measured colour ("the white sunglasses") for an accessory whose edit does not match the source. Default on. |
| `base_attempts` | INT | Head without items: seeds tried (seed, seed+1, ...) until it registers onto the source. Default 3. |
| `fidelity_min` | FLOAT | Share of an edit's pixels that must match the source (within 40 levels) for the edit to own pixels. Default 0.5. |
| `explained_max` | FLOAT | A pixel is explained by an edit when their colours are within this many levels. Default 60. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `parts` | PARTS_BUNDLE | VISIBLE layers: the source's own pixels, one layer per item, in paint order. |
| `complete_parts` | PARTS_BUNDLE | COMPLETE layers: each whole item, split into `<item>_back` (behind the head) and `<item>` (in front). |
| `base_image` | IMAGE | The head without items (voted accessories removed). The source itself when no accessory was voted or `make_base` is off. |
| `layers_preview` | IMAGE | Batch of all visible layers, then all complete layers (RGBA, full frame). |
| `report` | STRING | JSON: vote, present items, paint order and its evidence, per-item fidelity and pixel counts, retries, prompts. |

## How it works

1. **Vote.** `presence` is parsed against `vocabulary`. Only voted items are requested, because
   2.1 draws any item it is asked for, present or not.
2. **Two edits per item.** A VISIBLE edit ("extract only the X exactly where it is") and a
   COMPLETE edit ("decompose the head into layers, output only the X layer"). Every edit keeps
   the source's size (resolution 0), so all layers share one frame and nothing needs placing back.
3. **Resolve visible layers from the source** (below).
4. **Colour retry** for accessories that did not match.
5. **Paint order** read off the source, then the **back/front split** of each complete layer.
6. **Head without items**: one more edit removing the voted accessories.

Edit count: 2 per voted item, +1 for the base when an accessory is voted, +1 per colour retry.

## Visible = where, not pixels

2.1's visible edit is trusted for WHERE an item is, not for its pixels: it restyles (white
sunglasses came back grey) and draws hidden things (brows under sunglasses). So each head pixel
goes to the edit that covers it and matches the source colour best, and the layer holds the
**source's own pixels** there. Stacked over the base, the visible layers give the source back.

An item whose edit matches less than `fidelity_min` of the source is treated as hidden or
restyled: it owns no pixels, gets no visible layer, and keeps only its complete layer.

## Complete layers and the back/front split

A complete layer is the whole object, hidden parts included (the full hat, the hair under it).
Its pixels that sit over bare head skin in the source must be behind the head, like a hat's back
brim, so each complete layer is split: `<item>_back` goes under the head plate, `<item>` over it.

## Paint order read off the source

Where two complete layers overlap, the one the source actually shows there is in front. Depth
alone put brows in front of hats; the source never does. `depth_image` only breaks ties and
cycles. The order is written into `depth_median` in both bundles (backs before fronts in
`complete_parts`), so `BD_PartsExport` stacks the PSD correctly.

## Colour retry

Only for accessories: the retry names the colour measured on the source ("the white
sunglasses"). A complete layer whose colour is more than 45 levels off the source is retried the
same way. Hair, brows, eyes and mouth are never retried: next to an accessory their measured
colour is the accessory's, which gave "white eyebrows".

## Export

- **A PSD that rebuilds the source:** wire `parts` into `BD_PartsExport` `parts` and
  `base_image` into its `base_image`. The base goes at the bottom and the visible layers go over it
  in paint order. Hide an accessory layer and the head underneath shows.
- **The engine stack:** wire `complete_parts` into `BD_PartsBuilder2Assemble`, then its
  `engine_parts` into a second `BD_PartsExport`.
- Wire `parts` and `complete_parts` into `BD_PartsBuilder2Plates` too; the frame proof uses them.

## Usage

- Needs ComfyUI core with `TextEncodeQwenImage21` (6bfaacc6 or newer).
- An edit takes about 15 s at 1024 (40 steps) on the studio card. A head with 6 voted items is
  about 16 edits, before the plates.
- `base_image` is a product, not a plate input. `BD_PartsBuilder2Plates` makes its bald head
  from the source, because a third generation from the source turns skin to foil.
- The report's `alpha_repaired_px` counts pixels an edit placed on the head outside the source
  alpha (a white item the matte had cut). Those pixels are treated as head.
