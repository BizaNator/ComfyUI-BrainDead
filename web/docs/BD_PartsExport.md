# BD Parts Export

Save a PARTS_BUNDLE to disk as individual PNGs, depth maps, SAM3 masks, layered PSD, and composite images — all in one output node.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `parts` | PARTS_BUNDLE | Bundle to export. Passed through unchanged. |
| `context_id` | STRING (optional) | Wire `BD_SaveContext` output. Auto-picks if exactly one context is registered. |
| `save_pngs` | BOOL | Save per-tag RGBA PNGs. |
| `save_depth` | BOOL | Save per-tag depth PNGs. |
| `save_masks` | BOOL | Save original SAM3 masks (pre-edit alpha, stashed by `BD_PartsBatchEdit`). |
| `save_masked_pngs` | BOOL | Save masked variants: RGB = rebuilt/edited content, alpha = original SAM3 mask. |
| `save_composite` | BOOL | Save depth-sorted RGBA composite PNG + separate alpha PNG. |
| `save_psd` | BOOL | Save layered PSD with one layer per part, back-to-front by `depth_median`. RAW compression, Photoshop-compatible. |
| `composite_size` | INT | Scale canvas for composite and PSD outputs. 0 = use `frame_size`. |
| `base_image` | IMAGE (optional) | Painted under all parts in both the PSD and composite outputs. |
| `background_image` | IMAGE (optional) | Bottom-most layer in PSD only (below `base_image`). |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `parts` | PARTS_BUNDLE | Passthrough — unchanged bundle for further processing. |
| `output_dir` | STRING | Resolved output directory path. |
| `summary` | STRING | Files saved, sizes, any errors. |
| `composite_image` | IMAGE | Depth-sorted composite using current (post-edit) alpha. |
| `composite_image_masked` | IMAGE | Depth-sorted composite using original SAM3 alpha. |
| `parts_image_batch` | IMAGE | All part images as a batch, for downstream preview or batch save. |

## PSD layer structure

```
[background_image layer]       ← bottom (optional)
[base_image layer]             ← above background (optional)
[part layers, back-to-front by depth_median]
```

Each part layer is sized to the full PSD canvas. Layers are named by `tag`. Optional mask layers (visibility off, same scale) are inserted when `save_masks=ON`.

## Usage

- `context_id` from `BD_SaveContext` controls the output directory and filename template. Without it, files save to `output/BrainDead_Cache/parts_<timestamp>/`.
- Two composite variants are always computed when `save_composite=ON`: `composite_image` uses the edited alpha (post-`BD_PartsBatchEdit`), `composite_image_masked` uses the original SAM3 visibility mask (pre-edit).
- `save_psd=True` is the recommended archival format — it preserves each part as an independently positioned layer, matching game-engine UV-atlas workflows where each part is placed manually in Photoshop.
- Wire `base_image` to paint the character silhouette or a flat-color background under all parts in the PSD and composite, without affecting per-tag PNG exports.
