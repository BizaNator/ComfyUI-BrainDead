# BD Parts Refine

IoU-based deduplication of overlapping segmentation masks — merges near-duplicate prompts (e.g. "shoe", "sneaker", "left shoe") into canonical entries, with optional part count cap.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `masks` | MASK | Batch of masks — one per part, aligned with `labels`. |
| `labels` | STRING (multiline) | Newline-separated labels, one per mask. Must use newlines, not commas. |
| `iou_threshold` | FLOAT | IoU above this value merges two masks into one cluster. Default 0.7. |
| `merge_strategy` | COMBO | `union` (pixel-wise max of all masks in cluster) or `keep_largest` (most pixels wins, others discarded). |
| `max_parts` | INT | Cap total output parts by area — keeps the largest N. 0 = no cap. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `refined_masks` | MASK | Deduplicated mask batch. One mask per canonical cluster. |
| `refined_labels` | STRING | Newline-separated canonical labels (one per output mask). |
| `debug_overlay` | IMAGE | Colorized visualization of final clusters. |
| `summary` | STRING | Merge decisions: which labels were merged, which canonical tag was chosen, how many merged. |

## Canonical tag selection

Within each cluster, the canonical tag is the shortest label string. This is a heuristic for "most general" — "shoe" is preferred over "left shoe" or "sneaker shoe".

## Usage

- Always use `labels` with newline separators (not CSV). The mask batch index must align exactly with the label line index.
- Feed `refined_masks` and `refined_labels` directly into `BD_PartsBuilder.masks` and `BD_PartsBuilder.mask_labels`.
- `iou_threshold=0.7` is a good default. Lower it (e.g. 0.5) for looser merging when prompts produce significantly different crops of the same object. Raise it (e.g. 0.85) to preserve near-duplicate instances as separate parts.
- `merge_strategy=keep_largest` is faster and avoids mask bleed. Use `union` when you want full coverage across all instances (e.g. a shoe seen from two different SAM3 prompts that each captured different parts of it).
