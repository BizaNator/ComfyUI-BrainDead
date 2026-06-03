# BD Parts Builder

Crop and composite each segmented part into its own RGBA image, producing a PARTS_BUNDLE for downstream editing, compositing, and export nodes.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Source character image. |
| `sapiens2_labels` | SAPIENS2_LABELS (optional) | Sapiens2 parse output. When wired, generates one part per Sapiens2 class. Takes priority over `masks` + `mask_labels`. |
| `masks` | MASK (optional) | Batch of masks — one mask per part. Aligned with `mask_labels`. |
| `mask_labels` | STRING (optional) | Newline-separated labels, one per mask in `masks`. |
| `combined_mask` | MASK (optional) | Silhouette clip. All per-part masks are intersected with this. Do NOT wire an inverted mask. |
| `depth_image` | IMAGE (optional) | Depth map. Computes `depth_median` per part for depth-sorted compositing in `BD_PartsCompose`. |
| `selection_mode` | COMBO | `all` (include every detected part), `clothing`, `bodyparts`, `specific` (Sapiens2 presets). |
| `specific_parts` | STRING (optional) | Comma-separated part names when `selection_mode=specific`. |
| `background` | COMBO | Part image background: `alpha` (transparent RGBA, best for T3D pipeline), `white`, `black`, or hex color (best for 2D pipeline). |
| `square_pad` | COMBO | `pad_to_square` (uniform aspect ratio, required for image batches) or `none` (tight crop). |
| `min_area` | INT | Skip parts whose mask has fewer than this many pixels. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image_batch` | IMAGE | Batch of all part images, one per active part, padded to same dimensions. |
| `label_list` | STRING | Newline-separated tag names in bundle order. |
| `bbox_list` | STRING | Newline-separated bounding box strings `x1,y1,x2,y2` per part. |
| `parts` | PARTS_BUNDLE | Bundle containing all part images, bounding boxes, tags, and depth data. |

## PARTS_BUNDLE structure

```python
{
  "tag2pinfo": {
    "shirt": {
      "img": np.ndarray,       # RGBA uint8 crop
      "xyxy": [x1, y1, x2, y2],
      "tag": "shirt",
      "depth_median": float,   # larger = farther from camera
      "depth": np.ndarray      # optional cropped depth uint8
    },
    ...
  },
  "frame_size": (H, W)          # source image dimensions
}
```

## Usage

- Wire `masks` + `mask_labels` from `BD_PartsRefine` (after dedup) or `BD_SAM3MultiPrompt.per_prompt_masks` directly.
- `background=alpha` produces RGBA PNGs that Trellis3D / Pixal3D can read correctly. Use `white` or `black` when feeding into 2D Qwen-Edit pipelines where a white/black bg signals the background to the model.
- `depth_image` from a Lotus-2 or DepthAnything node enables depth-sorted compositing in `BD_PartsCompose`. Without it, parts composite in label order.
- `square_pad=pad_to_square` is required when `image_batch` is fed into nodes that expect uniform batch shapes (Qwen Edit, ControlNet, etc.).
