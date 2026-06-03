# BD Mask Resolver

Priority-based mask separation for skin, clothing, and accessories — Python port of the GLSL Mask Resolver shader, with neighbor-vote gap fill and adaptive LAB color scoring.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Source image for color scoring. |
| `skin_mask` | MASK | Raw skin candidate mask. |
| `clothes_mask` | MASK | Raw clothing candidate mask. |
| `accessories_mask` | MASK | Raw accessories candidate mask. |
| `skin_priority` | FLOAT | Score weight for skin (default 1.0). |
| `clothes_priority` | FLOAT | Score weight for clothing (default 2.0). |
| `accessories_priority` | FLOAT | Score weight for accessories (default 3.0). |
| `skin_color_conf` | FLOAT | Boost skin score for pixels matching skin LAB color (adaptive sampling from skin_mask). |
| `cloth_color_conf` | FLOAT | Boost clothing score for pixels matching clothing LAB color. |
| `claim_threshold` | FLOAT | Pixels with max-category score below this are residual (eligible for gap fill). |
| `overlap_mode` | COMBO | `priority` (winner-takes-all at each pixel) or `soft_blend` (proportional blend by score). |
| `gap_fill` | BOOL | Enable neighbor-vote gap fill for unclaimed (residual) pixels. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `clean_skin` | MASK | Skin pixels after conflict resolution. |
| `clean_clothes` | MASK | Clothing pixels after conflict resolution. |
| `clean_accessories` | MASK | Accessories pixels after conflict resolution. |
| `residual` | MASK | Pixels below `claim_threshold` — unclaimed by any category. |
| `debug_overlay` | IMAGE | Colorized visualization: skin=red, clothes=green, accessories=blue, residual=grey. |
| `status` | STRING | Resolution summary and gap-fill stats. |

## Usage

- Default priorities (accessories=3 > clothes=2 > skin=1) match the typical character pipeline: hard-edge items (jewelry, belts) take precedence over soft garments, which take precedence over skin.
- Wire outputs from `BD_SAM3MultiPrompt` or `BD_MP_FaceRefine` as inputs — resolver handles the overlap arbitration you'd otherwise need per-node logic for.
- `gap_fill` uses `scipy.ndimage.uniform_filter` (neighbor voting) to assign residual pixels to the nearest winning category. Disable when you want to inspect unclaimed areas explicitly.
- `soft_blend` mode preserves partial coverage at boundaries — useful when the target is a composited render rather than a hard segmentation map.
