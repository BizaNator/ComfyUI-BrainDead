# BD Draw Rect

Draw a flat-color rectangle (optionally rounded) with feathered edges, composited onto an image with optional packed-channel injection.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `mask` | MASK (optional) | Use this mask's bounding box as the rectangle shape instead of manual geometry. Takes priority over `mask_from_packed`. |
| `mask_from_packed` | COMBO | Extract a mask from a channel of `packed_image`: `none`, `R`, `G`, `B`, `A`. Only used when `mask` is not wired. |
| `image` | IMAGE (optional) | Background to composite the rectangle onto. Canvas size defaults to this image's dimensions. |
| `packed_image` | IMAGE (optional) | Source for `mask_from_packed` extraction, and/or the destination for `pack_channel` injection. |
| `pack_channel` | COMBO | Write the output mask into this channel of `packed_image`: `none`, `R`, `G`, `B`, `A`. |
| `x_center` | FLOAT | Horizontal center of the rectangle, normalized 0–1 (0=left, 1=right). |
| `y_center` | FLOAT | Vertical center of the rectangle, normalized 0–1 (0=top, 1=bottom). |
| `rect_width` | FLOAT | Width of the rectangle as a fraction of canvas width, 0–1. |
| `rect_height` | FLOAT | Height of the rectangle as a fraction of canvas height, 0–1. |
| `corner_radius` | INT | Corner rounding radius in pixels. 0 = sharp corners. |
| `feather` | INT | Edge feather in pixels. Positive = outward (interior stays 1.0), negative = inward, 0 = hard edge. |
| `fill_r` | INT | Red fill component, 0–255. |
| `fill_g` | INT | Green fill component, 0–255. |
| `fill_b` | INT | Blue fill component, 0–255. |
| `opacity` | FLOAT | Composite opacity, 0.0–1.0. |
| `canvas_width` | INT (optional) | Override canvas width in pixels. Defaults to `image` width or 512. |
| `canvas_height` | INT (optional) | Override canvas height in pixels. Defaults to `image` height or 512. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | RGB composited result. |
| `mask` | MASK | Rectangle mask (1.0 inside, feathered at edges). |
| `packed_out` | IMAGE | `packed_image` with the mask injected into `pack_channel`, or passthrough if `pack_channel=none`. |

## Shape priority

1. `mask` (wired) — bounding box of non-zero pixels defines the rectangle
2. `mask_from_packed` — extract a channel from `packed_image` and use its bounding box
3. Manual geometry — `x_center`, `y_center`, `rect_width`, `rect_height`, `corner_radius`

## Usage

- Use `mask_from_packed` + `pack_channel` to read a mask from one channel of a packed image, draw a rect over it, and write the result back into another channel in one node.
- Set `feather` positive for a smooth fade-out at the rectangle boundary. Negative feather shrinks the solid region and blurs inward (useful for soft glows or inner shadows).
- Wire `image` to composite directly onto a character layer; leave it unwired to produce a mask-only output on a black canvas.
