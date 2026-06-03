# BD Mask Flatten

Flatten an RGBA or RGB+mask image onto a background, with optional Voronoi edge padding and channel routing modes.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Source image (RGB or RGBA). |
| `mask` | MASK (optional) | Alpha mask. When wired, overrides the image's own alpha. If unwired and image is RGBA, the image alpha is used. If unwired and image is RGB, alpha defaults to 1.0 (fully opaque). |
| `background_image` | IMAGE (optional) | Background image. When wired, overrides `background_color`. |
| `background_color` | STRING | Hex color string (e.g. `#ffffff`) or preset: `white`, `black`, `transparent`, `checker`. Default `#ffffff`. |
| `flatten_mode` | COMBO | `alpha_composite` (standard RGBA blend, default), `grayscale` (luma output), `image_to_red`, `image_to_green`, `image_to_blue` (write image luma into a single channel). |
| `edge_pad_pixels` | INT | Voronoi nearest-neighbor fill applied BEFORE flattening. Bleeds opaque pixels into transparent border zones to prevent UV edge halos in game engines. |
| `mask_threshold` | FLOAT | Binarize alpha at this level before composite. 0.0 = no thresholding. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Flattened RGB result. Always RGB — no alpha channel in output. |
| `alpha_used` | MASK | The alpha mask that was used for compositing (after threshold). |

## Usage

- `edge_pad_pixels > 0` is the primary reason to use this node for game-asset prep: it fills transparent border pixels with the nearest opaque color before flattening, eliminating the dark/bright UV-seam halos that appear when a game engine samples beyond the UV island boundary.
- `flatten_mode=image_to_red/green/blue` routes the image luma into a single channel of a black output — use this to pack a greyscale map (roughness, AO) into one channel of a composite texture before a channel merge node.
- `mask` overrides image alpha, so you can apply a refined segmentation mask from `BD_FaceSkinRefine` or `BD_MaskCorrelate` without re-compositing the image manually.
- Output is always RGB. Wire into `BD_PackChannels` or a SaveImage node directly.
