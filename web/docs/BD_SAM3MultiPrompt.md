# BD SAM3 Multi-Prompt

Run SAM3 segmentation with multiple positive/negative prompts in one node, then combine results with vote, union, intersection, or weighted modes — replacing a 12-node SAM3Segment chain.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Input image to segment. |
| `sam3_model` | MODEL | SAM3 model (wire from a SAM3ModelLoader or compatible loader). |
| `positive_prompts` | STRING (multiline) | One prompt per line. Each line produces one SAM3 mask. |
| `negative_prompts` | STRING (multiline) | One prompt per line. Used as negative guidance per iteration. |
| `combine_mode` | COMBO | How to merge all positive masks: `union`, `intersection`, `subtract_first`, `first_only`, `vote`, `weighted_vote`. |
| `vote_threshold` | FLOAT | Pixel kept if `(pos_votes - neg_votes) >= threshold`. Default 1.0 (majority). |
| `color_filter` | COMBO | Post-SAM color filtering: `off`, `exclude`, `exclude_hard`, `include`, `exclude_and_include`, `remove_matching`. |
| `color_mode` | COMBO | Color matching algorithm: `adaptive_lab` (self-tuning LAB ΔE from mask), `fixed_hsv` (skin-specific HSV range), `both`. |
| `color_reference_mask` | MASK (optional) | Override adaptive sampling region for LAB color matching. |
| `silhouette_mask` | MASK (optional) | Outer boundary. When `enforce_silhouette=ON`, every SAM3 result is clipped to this mask. |
| `enforce_silhouette` | BOOL | Clip EVERY SAM3 candidate (not just the final combined result) to `silhouette_mask`. |
| `silhouette_composite_enable` | BOOL | Compute `silhouette_mask - combined_mask` as `silhouette_composite`. |
| `invert_negatives_in_per_prompt` | BOOL | Show negative prompt masks as inverted in `per_prompt_masks` output. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `combined_mask` | MASK | Final combined mask after `combine_mode` and color filtering. |
| `masked_image` | IMAGE | `image` composited with `combined_mask`. |
| `per_prompt_masks` | MASK | Batch of all individual prompt masks, aligned with the prompt list. Feed into `BD_PartsBuilder`. |
| `status` | STRING | Per-prompt match results and combination summary. |
| `silhouette_composite` | MASK | `silhouette_mask - combined_mask` (residual region), only when `silhouette_composite_enable=ON`. |

## Combine modes

| Mode | Behavior |
|------|----------|
| `union` | Pixel-wise max of all positive masks. |
| `intersection` | Pixel-wise min — only pixels found in every prompt. |
| `subtract_first` | First prompt mask minus the union of remaining masks. |
| `first_only` | Returns only the first positive mask result. |
| `vote` | Each positive prompt votes +1, each negative votes -1. Pixel kept if sum ≥ `vote_threshold`. |
| `weighted_vote` | Like `vote` but each prompt's weight equals its pixel area (larger detections count more). |

## Usage

- Use `vote` or `weighted_vote` for multi-instance objects (e.g. "shoe", "sneaker", "left shoe") — overlapping detections reinforce each other without double-counting.
- Wire `per_prompt_masks` directly into `BD_PartsRefine` or `BD_PartsBuilder.masks`. Labels are aligned with the positive prompt line order.
- `adaptive_lab` color filtering is self-tuning: it samples from the combined mask region and rejects candidate masks whose dominant color is too different (ΔE threshold). Works for skin, clothing, and hair — not just skin-specific HSV.
- `enforce_silhouette` is the most reliable way to prevent SAM3 from grabbing background or adjacent objects — wire the character silhouette and everything outside is eliminated before combination.
