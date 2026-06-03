# BD MP Face Refine

Refine MediaPipe face feature masks using IoU-matched SAM3 candidates, then derive a pixel-accurate skin mask as the face oval minus all refined features.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Source image (for debug overlay and `masked_image` output). |
| `sam3_masks` | MASK | Batch of SAM3 candidate masks to match against MediaPipe hulls. |
| `face_oval` | MASK | MediaPipe face oval (from `BD MP Face Mask` or `BD MP Save/Load Face Data`). |
| `left_eye` | MASK | MediaPipe left eye hull. |
| `right_eye` | MASK | MediaPipe right eye hull. |
| `left_brow` | MASK | MediaPipe left brow hull. |
| `right_brow` | MASK | MediaPipe right brow hull. |
| `lips` | MASK | MediaPipe lip hull. |
| `nose` | MASK | MediaPipe nose hull. |
| `target_expand` | FLOAT | Dilate MediaPipe hulls before IoU matching only. Does not affect the final refined masks. |
| `clip_expand` | FLOAT | Maximum allowed bleed outside the raw MediaPipe hull in the refined mask. |
| `feature_expand` | FLOAT | Dilate refined feature masks before subtracting from skin (edge buffer). |
| `max_target_fill` | FLOAT | Reject SAM3 candidates whose area exceeds this fraction of the image (default 0.95). Prevents SAM3 "not found" full-white masks from matching. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `skin` | MASK | Pixel-accurate skin: `face_oval` minus all refined features. |
| `left_eye` | MASK | Refined left eye mask. |
| `right_eye` | MASK | Refined right eye mask. |
| `left_brow` | MASK | Refined left brow mask. |
| `right_brow` | MASK | Refined right brow mask. |
| `lips` | MASK | Refined lip mask. |
| `nose` | MASK | Refined nose mask. |
| `debug_overlay` | IMAGE | Colored visualization of all refined feature zones. |
| `match_info` | STRING | Per-feature match results: candidate index, IoU score, decision. |
| `masked_image` | IMAGE | `image` composited with the skin mask. |
| `masked_skin` | IMAGE | `image` cropped to skin pixels only. |
| `head_mask` | MASK | Full head mask (same as `face_oval` input). |

## Matching priority

Higher-priority features get first pick of SAM3 candidates in exclusive mode:

1. `lips` (5)
2. `left_eye`, `right_eye` (4)
3. `left_brow`, `right_brow` (3)
4. `nose` (2)

## Usage

- Feed `sam3_masks` from `BD_SAM3MultiPrompt`'s `per_prompt_masks` output — run SAM3 with prompts targeting each face feature, then let this node do the IoU matching.
- `target_expand` helps when MediaPipe hulls are tight and SAM3 masks include a small margin; expand to improve IoU hit rate without blowing out the final mask.
- `clip_expand` limits how far the SAM3 result is allowed to bleed past the raw hull. Keep it small (2–5px) to prevent hair or background leaking into the skin mask.
- Check `match_info` output to diagnose why a feature didn't match — the IoU score and reject reason are printed per feature.
