# BD Luma Stats

Compute luma statistics for an image and output key values for routing into shader uniforms or normalization nodes.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Source image. Passed through unchanged. |
| `mask` | MASK (optional) | Restrict stats to masked pixels only (mask > 0.5). |
| `luma_standard` | COMBO | RGB → luma weighting: `bt709` (default, modern sRGB), `bt601` (legacy NTSC), `average` ((R+G+B)/3). |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Passthrough — identical to input, for chaining. |
| `min_luma` | FLOAT | Minimum luma in the (masked) region. |
| `max_luma` | FLOAT | Maximum luma in the (masked) region. |
| `median_luma` | FLOAT | Median luma. |
| `mean_luma` | FLOAT | Mean luma. |
| `recommended_outer_band` | FLOAT | `(max - min) / 2 × 0.90` — suggested value for the `u_float2` outer-band uniform in `skin_shader.glsl`. |

## Usage

- `recommended_outer_band` is pre-calculated for `BD_CenterMedianLuma`'s `outer_band` or the GLSL skin shader's `u_float2`. Wire it directly — no manual calculation needed.
- Wire `mask=skin_mask` so statistics are computed only over skin pixels, excluding background and clothing from skewing the range.
- `bt709` is the correct standard for any image destined for the skin shader pipeline. Use `bt601` only when matching legacy content.
- The image passthrough makes it safe to insert this node inline without breaking the graph.
