# BD Mask Color Fill

Fill up to four mask regions with solid colors on a composited canvas, with optional background image and alpha control.

## Inputs

**Slot inputs (repeated for slots 1–4):**

| Name | Type | Description |
|------|------|-------------|
| `mask_N` | MASK (optional) | Mask for slot N. When unwired, slot is skipped. |
| `r_N` / `g_N` / `b_N` | INT | Fill color for slot N (0–255). Defaults: slot1=red, slot2=green, slot3=blue, slot4=yellow. |
| `expand_N` | INT | Morphological expand (pixels) before compositing. |
| `feather_N` | INT | Edge feather (pixels). Positive = outward, negative = inward, 0 = hard. |

**Background inputs:**

| Name | Type | Description |
|------|------|-------------|
| `background` | IMAGE (optional) | Background image. When wired, overrides `bg_r/g/b`. |
| `bg_r` / `bg_g` / `bg_b` | INT | Solid background color (0–255). Used when `background` is not wired. |
| `bg_alpha` | FLOAT | Background opacity (0.0 = transparent, 1.0 = opaque). Default 0.0 (transparent background). |
| `bg_mask` | MASK (optional) | Restrict background to masked pixels. ComfyUI convention: 1.0 = keep / opaque, 0.0 = transparent. |
| `bg_alpha_from_image` | BOOL | Derive background alpha from `background` image's own alpha channel. |
| `bg_alpha_invert` | BOOL | Invert the background alpha after derivation. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | RGB composite of all filled regions on background. |
| `rgba` | IMAGE | RGBA version — alpha = union mask of all active slots. |
| `mask` | MASK | Union of all active slot masks (after expand + feather). |
| `status` | STRING | Summary of which slots were active and fill colors used. |

## Usage

- Leave `bg_alpha=0.0` (default) to get a transparent background and route `rgba` into downstream compositing.
- Use all four slots simultaneously to colorize a full segmentation map (skin, clothes, accessories, hair) in one node.
- `expand_N` and `feather_N` are per-slot — apply different softness to different feature types (e.g. soft brows, hard-edge shoes).
- Wire `bg_mask` with 1.0 = visible regions to restrict the background to a silhouette without affecting the slot fills.
