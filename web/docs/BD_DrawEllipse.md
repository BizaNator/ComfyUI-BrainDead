# BD Draw Ellipse

Draw a flat-color ellipse or circle with feathered edges, composited onto an image with optional packed-channel injection.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `mask` | MASK (optional) | Use this mask's bounding box as the ellipse bounds instead of manual geometry. Takes priority over `mask_from_packed`. |
| `mask_from_packed` | COMBO | Extract a mask from a channel of `packed_image`: `none`, `R`, `G`, `B`, `A`. Only used when `mask` is not wired. |
| `image` | IMAGE (optional) | Background to composite the ellipse onto. Canvas size defaults to this image's dimensions. |
| `packed_image` | IMAGE (optional) | Source for `mask_from_packed` extraction, and/or the destination for `pack_channel` injection. |
| `pack_channel` | COMBO | Write the output mask into this channel of `packed_image`: `none`, `R`, `G`, `B`, `A`. |
| `x_center` | FLOAT | Horizontal center, normalized 0–1. |
| `y_center` | FLOAT | Vertical center, normalized 0–1. |
| `radius_x` | FLOAT | Horizontal radius as a fraction of canvas width. |
| `radius_y` | FLOAT | Vertical radius as a fraction of canvas height. Set equal to `radius_x` for a circle. |
| `feather` | INT | Edge feather in pixels. Positive = outward (interior stays 1.0), negative = inward, 0 = hard edge. |
| `fill_r` | INT | Red fill component, 0–255. |
| `fill_g` | INT | Green fill component, 0–255. |
| `fill_b` | INT | Blue fill component, 0–255. |
| `opacity` | FLOAT | Composite opacity, 0.0–1.0. |
| `canvas_width` | INT (optional) | Override canvas width in pixels. |
| `canvas_height` | INT (optional) | Override canvas height in pixels. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | RGB composited result. |
| `mask` | MASK | Ellipse mask (1.0 inside, feathered at boundary). |
| `packed_out` | IMAGE | `packed_image` with mask injected into `pack_channel`, or passthrough if `pack_channel=none`. |

## Shape priority

1. `mask` (wired) — bounding ellipse of mask's non-zero region
2. `mask_from_packed` — extract channel from `packed_image`, use bounding ellipse
3. Manual geometry — `x_center`, `y_center`, `radius_x`, `radius_y`

## Usage

- Identical wiring pattern to `BD_DrawRect` — the two nodes are interchangeable when all you need to change is the primitive shape.
- Use `radius_x != radius_y` for eye-socket ovals or lip ellipses; set them equal for perfectly circular coverage.
- `feather > 0` expands the soft zone outside the ellipse boundary — the solid interior (where mask = 1.0) is unaffected. Negative feather bites into the interior.
