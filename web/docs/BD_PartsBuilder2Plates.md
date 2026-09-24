# BD Parts Builder 2 Plates (Qwen 2.1)

Make the featureless head plates (bald, mouthless/browless, eyeless) from the source with Qwen Image 2.1, and prove that each one stays in the source's frame.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | The same source as `BD_PartsBuilder2`. |
| `model` | MODEL | Qwen Image 2.1 through `QwenImage21Cache`. |
| `clip` | CLIP | `qwen3vl_8b`, CLIPLoader type `qwen_image`. |
| `vae` | VAE | Qwen Image 2.1 VAE (RGBA). |
| `parts` | PARTS_BUNDLE (optional) | `BD_PartsBuilder2` `parts`: where each item is, for the frame proof. |
| `complete_parts` | PARTS_BUNDLE (optional) | `BD_PartsBuilder2` `complete_parts`, for the frame proof. |
| `head_mask` | MASK (optional) | Head alpha. Not wired: non-white pixels. |
| `bald_prompt` | STRING (multiline) | Bald edit of the source. The default removes hair, head coverings, glasses, headphones, earrings, beard and moustache, and keeps the brows. |
| `mouthless_prompt` | STRING (multiline) | Edit of the bald: removes lips, mouth interior and brows, keeps the eyes open. |
| `eyeless_prompt` | STRING (multiline) | Edit of the bald: removes brows, eyes (closed smooth lids) and mouth. |
| `seed` | INT | Seed for the first try of each stage. A failed try retries with seed+1. Default 20260933 (the seed the eight proving heads' plates were made with). |
| `steps` | INT | Default 40. |
| `cfg` | FLOAT | Default 1.0. |
| `sampler_name` | COMBO | Default `euler`. |
| `scheduler` | COMBO | Default `simple`. |
| `attempts_per_stage` | INT | Tries per stage (seed, seed+1, ...) before the chain stops. Default 2. |
| `work_size` | COMBO | `1024` or `2048`. Plate resolution. 2048 is Qwen Image 2.1's maximum and takes about 4x the time. |
| `reframe_below` | FLOAT | Re-frame when the bald head fills less than this share of the frame. 0 = never. Default 0.90. |
| `reframe_fill` | FLOAT | Share of the frame the bald head fills after re-framing. Default 0.94. |
| `max_scale_err` | FLOAT | Largest allowed scale change per axis. Default 0.015 (1.5%). |
| `max_offset_px` | FLOAT | Largest allowed centre move, in 1024-px units. Default 3.0. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `bald` | IMAGE | Bald head, cut by the matte onto white. |
| `mouthless` | IMAGE | Mouthless/browless plate, eyes kept. |
| `eyeless` | IMAGE | Eyeless plate: no brows, eyes or mouth. The engine base. |
| `matte` | MASK | The plate matte: the bald edit's own alpha, faint noise under 5% zeroed. |
| `plate_source` | IMAGE | The source in the plate frame. |
| `plate_head` | MASK | The source's head mask in the plate frame. |
| `plate_frame` | STRING | JSON: `box_source_px` [x0, y0, side], `size`, `source_size`, `scale`, and the mapping `plate_px = (source_px - box_xy) * size / box_side`. |
| `report` | STRING | JSON: per stage the seed used, the fit to its input and to the source, `pass`, `pass_by`; overall `ok`; the prompts. |

## The chain

```
source --bald edit--> bald (RGBA: its alpha becomes the matte)
bald   --one edit---> mouthless        bald --one edit---> eyeless
```

The rules below each come from a failed run on the proving heads.

- **Every stage is cut onto white.** 2.1 amplifies noise in its input: background noise grew
  from 0.9 to 5.4 to 48.8 over three stages, and skin turned to "crinkled foil". So each result
  is cut by the matte onto pure white before anything reads it.
- **At most two generations from the source.** The bald is made from the source, and
  mouthless and eyeless are each ONE edit of the bald. Eyeless is never made from mouthless, and
  the bald is never made from `BD_PartsBuilder2`'s head without items. A third generation turns
  skin to foil.
- **Jaw-keeping mouth wording.** The default says the mouth area becomes plain skin "while the
  chin, the jaw and the cheeks stay exactly where they are". The earlier "closed mouth is one
  smooth surface" shut an open laugh's jaw and failed the frame proof by 4.3%. The jaw wording holds on open
  and closed mouths alike.
- **No "head only, no neck" wording.** It makes 2.1 recompose the image, so heads move or shrink.
  Keep that out of edited prompts too.

## Frame proof

Each stage must land on the same pixels as its input AND the source. The node registers the
result onto both (coarse-to-fine affine ECC on edges) over the head, minus a margin around what
the stage removes. Bare cheeks alone false-passed moved heads, so the region must contain
features. Pass = scale within `max_scale_err` and centre within `max_offset_px`.

**Chain proof.** Removing a laughing mouth relaxes the cheeks, so a plate can share too little
with the source for a direct fit (ECC < 0.7). If the plate fits the bald, the proof is taken
through the chain instead: the bald's fit to the source combined with the plate's fit to the bald,
under the same limits. The report's `pass_by` says `direct` or `chain`.

A stage that fails retries with seed+1, up to `attempts_per_stage` tries. If every try fails, the
**chain stops**: nothing is made from a rejected plate. The failed stage's last try is output for
inspection, the stages after it come out white, and both the report and `plate_frame` say
`"ok": false` with the `failed_stage`. `BD_PartsBuilder2Assemble` then builds nothing.
Wire `parts` and `complete_parts`, because without them the proof also measures the regions the
stage changes on purpose.

## Plate frame

A hat sets the source's framing, so once the head is bald it can be small:
one proving head's bald head filled 57% of the frame, too few pixels to texture. When the bald head
fills less than `reframe_below` of the frame, the node crops the source around the bald head so
it fills `reframe_fill`, resizes the crop to `work_size`, and re-runs the whole chain there.
`plate_frame` records the crop. `plate_source` and `plate_head` are the source in that frame,
ready for `BD_PartsBuilder2Assemble`. Without a crop and at the source size, the frame is `identity`.

`work_size` 2048 runs every edit at 2048 (Qwen Image 2.1's maximum) instead of 1024. A smaller source or crop is resized up to 2048 first.

## Wiring

Wire `eyeless`, `matte`, `plate_frame`, `plate_source` and `plate_head` into
`BD_PartsBuilder2Assemble`. `mouthless` (eyes kept) is for uses that keep the eyes, such as Tripo.
