# BD MP Mouth Parts

Separate an **isolated mouth render** into **lips / teeth / tongue** and pack them into the Unreal viseme-atlas RGBA contract — **R=lips, G=teeth, B=tongue, A=POM depth** — in a single node.

Built for the PinkF1 lip-viseme atlas (ARTS-1). The source mouths are low-poly renders on black: magenta/purple lips, white teeth, salmon/rose tongue. This node replaces the `SAM3×3 (lips/teeth/tongue) → FillMaskHoles×3 → MaskResolver → PackChannels` chain in `LipViseme-Atlas-Packed.json` with one pass.

## Two engines

> "Tools stand alone" = no dependency on **another node pack** — NOT "no models". The colour engine needs nothing; the SAM3 engine loads a model we manage ourselves.

**`engine='color'` (default, no model)** — a tuned HSV classifier:
- **teeth** = bright + desaturated (white) — `S < teeth_s_max`, `V > teeth_v_min`
- **tongue** = warm hue (closer to red than the magenta lips), **region-grown** from a confident salmon core **gated to the eroded mouth interior** — so a warm lip-corner highlight can't grow into a false tongue
- **lips** = the saturated remainder — the `lips_s_min` floor drops the dark mouth cavity so the interior gap is **not** mislabelled lips

Thresholds tuned on the seven PinkF1 visemes (Aa, i, Kk, O, sil, th, U), exposed for other palettes.

**`engine='sam3'` (wire a comfy-core SAM3 `MODEL`)** — the colour split becomes the **prior** and SAM3 refines it:
- **teeth & tongue** get SAM3 seeded by their own colour bbox + interior positive points + the other parts' centroids as negatives. A **greediness guard** reverts a part to its colour mask if SAM3 over- or under-grows it (`[0.25–3.0]×` the colour area). Because the parts are nested in a tiny area, teeth typically over-grows and reverts to the (already clean) colour teeth; the real win is the **tongue**, where SAM3 captures the brightly-lit front that colour misreads as magenta lips.
- **lips** stays the colour ring (a ring has no usable single SAM3 box); mutual-exclusivity carves teeth/tongue back out.

**No wiring needed:** leave `model` unwired and the node **auto-loads + auto-downloads** the official SAM3 checkpoint (`Comfy-Org/sam3.1`) in-house via `bd_sam3` on first use. Wire a comfy-core SAM3 `MODEL` only if you want to override it.

## Inputs

| Input | Description |
|-------|-------------|
| `image` | Isolated mouth render (background near-black). Only `image[0]` is used. |
| `engine` | `color` (HSV, no model) or `sam3` (colour-seeded SAM3 refine; needs `model`, falls back to color if unwired). |
| `model` | comfy-core SAM3 `MODEL` (only used by `engine='sam3'`). |
| `pom` | Optional POM / depth map (e.g. **Lotus2 depth → NormLuma → CenterLuma**) → packed into the **A** channel (its luminance). If unwired, `A` = the mouth foreground (lips∪teeth∪tongue). |
| `bg_v_min` | Background cutoff — HSV value ≤ this is background (black). Raise for a lifted/grey background. |
| `teeth_s_max` / `teeth_v_min` | Teeth = saturation **below** `teeth_s_max` and value **above** `teeth_v_min` (white/grey, bright). |
| `tongue_h_lo` / `tongue_h_hi` | Tongue warm-hue band (degrees). Tongue ≈ 338°, lips ≈ 310°. The `H ≤ 6°` red wrap is always included. |
| `tongue_s_min` / `tongue_s_max` | Tongue core saturation window. The **max** is the key separator — the tongue is *less* saturated than the magenta lips. |
| `tongue_v_min` | Tongue core min value. |
| `lips_s_min` | Lips min saturation — the **cavity guard**. Lower to include darker lip facets; raise if the dark interior bleeds into lips. |
| `interior_frac` | Mouth-interior erosion (fraction of the longer side). The tongue core must lie inside the foreground eroded by this much — kills warm corner highlights. `0` disables the gate. |
| `sam3_iters` | *(sam3)* SAM decoder refinement passes (1 = single, fullest; a collapse-guard reverts over-shrunk parts). |
| `bleed_guard` | *(sam3)* Dilate the colour mouth-foreground by N px and clip each SAM3 part to it — stops SAM3 escaping onto skin. |
| `edge_smooth` | Morphological close radius (px) to seal jagged lip/tongue edges. |
| `fill_holes` | Fill interior holes so each part is solid (matches the atlas `FillMaskHoles` step). |
| `despeckle` | Drop connected components smaller than this fraction of the image (removes specular glints on the tongue that read as teeth and punch holes in it). Keep small so separate real teeth survive. `0` disables. |
| `edge_refine` | Snap each part's edge to the image: `off` / `guided` (cv2 guidedFilter) / `matting` (PyMatting, CPU) / `vitmatte` (deep, GPU, auto-downloads). Shared with **BD MP SAM3**; each degrades to `off` if its backend is missing. |
| `refine_radius` / `refine_eps` / `refine_threshold` | Edge-refine tuning (see BD MP SAM3). |
| `vitmatte_model` | `small` / `base` for `edge_refine='vitmatte'`. |

## Outputs

| Output | Description |
|--------|-------------|
| `lips`, `teeth`, `tongue` | Individual part masks (MASK), mutually exclusive (priority **teeth > tongue > lips**). |
| `rgb_packed` | **R=lips, G=teeth, B=tongue** (no alpha) — the PackChannels equivalent. |
| `rgba_packed` | **R=lips, G=teeth, B=tongue, A=POM** (from `pom`, else mouth foreground) — the viseme-atlas cell contract. |
| `debug_overlay` | Render tinted by part (lips=R, teeth=G, tongue=B) for QC. |
| `status` | Run summary (per-part coverage, A source, refine mode). |

## Where it fits

Drop-in for the colour-split step of the lip-viseme atlas:

```
LoadImage → BiRefNetRMBG → PreResize 1024 → BD MP Mouth Parts → CropToMask → 512×512 cell
                                              (A ← Lotus2 depth → NormLuma → CenterLuma)
```

`rgba_packed` is the packed cell; tile cells into the 2048×1024 (4×2) atlas. The lip/teeth/tongue separation also feeds any downstream that needs the parts individually (e.g. per-part tinting).

## Recommended settings

Defaults are tuned for the PinkF1 renders. Start with `engine=color`, `edge_refine=guided`, `fill_holes=on`, `despeckle=0.0008`. For the cleanest tongue (its weak spot under colour), switch to `engine=sam3` and wire the SAM3 model — teeth/lips are protected by the greediness guard, so SAM3 only helps. If a render's tongue reads thin, widen `tongue_s_max` / lower `tongue_v_min`; if the cavity bleeds into lips, raise `lips_s_min`; if a warm lip corner becomes a false tongue, raise `interior_frac`; if real teeth get despeckled away, lower `despeckle`.

Single responsibility: this node **separates + packs**. Depth/POM derivation (Lotus2) and saving to disk live in their own nodes.
