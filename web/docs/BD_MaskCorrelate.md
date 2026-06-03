# BD Mask Correlate

IoU-based matching of SAM3 candidate masks to up to 8 named target slots, with per-slot mode overrides and post-match subtraction.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Source image (for `masked_image` and `debug_overlay`). |
| `candidate_masks` | MASK | Batch of SAM3 masks to match against targets. |
| `silhouette_mask` | MASK (optional) | Outer boundary — used for `combined_mask_invert` and `masked_image` clipping. |
| `target_1` … `target_8` | MASK (optional) | Reference hulls (e.g. MediaPipe feature masks) for each named slot. |
| `labels` | STRING (multiline) | One label per line, aligned with `target_1..target_N`. Default: `left_brow\nright_brow\nleft_eye\nright_eye\nlips`. |
| `priorities` | STRING | Comma-separated priority values aligned with slots. Higher = first pick in exclusive matching. |
| `mode` | COMBO | Global default combine mode: `intersect`, `replace`, `union`, `weighted_blend`. |
| `slot_modes` | STRING (multiline) | Per-slot overrides: `label_or_index: mode` one per line. |
| `subtract_slots` | STRING (multiline) | Post-match subtraction: `target_label: src_label1, src_label2` one per line. |
| `combined_mask_exclude` | STRING | Comma-separated slot labels to exclude from `combined_mask` union. |
| `invert_masked_image` | BOOL | Show everything EXCEPT matched regions (clamped to `silhouette_mask`). |
| `combined_mask_invert` | BOOL | Output `silhouette_mask - combined_mask` (head-minus-features). |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `refined_1` … `refined_8` | MASK | Matched and processed mask for each slot. Empty mask if no candidate matched. |
| `debug_overlay` | IMAGE | Colorized visualization of all matched slots. |
| `match_info` | STRING | Per-slot match results: candidate index, IoU score, mode applied. |
| `masked_image` | IMAGE | `image` with matched regions shown (or inverted when `invert_masked_image=ON`). |
| `combined_mask` | MASK | Union of all active slot masks (excluding `combined_mask_exclude` slots). |

## Combine modes

| Mode | Behavior |
|------|----------|
| `intersect` | `target ∩ candidate` — restricts the SAM3 result to the reference hull. |
| `replace` | Use the SAM3 candidate directly (ignores target hull). |
| `union` | `target ∪ candidate` — expands to cover both. |
| `weighted_blend` | Blend proportionally by IoU score. |

## Usage

- `labels` must be newline-separated to align with `target_1..target_N`. The labels are used by `slot_modes`, `subtract_slots`, and `combined_mask_exclude`.
- Use `subtract_slots` to remove eye masks from brow masks: `left_brow: left_eye` prevents brow masks from overlapping the eye socket.
- `combined_mask_invert` with `silhouette_mask` gives a "head minus features" mask — useful as a skin zone when you want everything except eyes/brows/lips.
- `mode=intersect` (default) is the safest setting for the face pipeline: it clips SAM3 to the MediaPipe hull, preventing bleed into adjacent features.
