# BD MP Face Infill

MediaPipe landmark-based face socket creator: fills eye, brow, lip, and nose zones with flat color, surrounding blur, or Telea inpaint — for 2D animation flipbook texture production.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Primary image. Detection + color sampling source. |
| `image1` | IMAGE (optional) | Fill target. When wired, sockets are painted into this image instead of `image`. |
| `face_data_path` | STRING (optional) | Path to a `.mpface.npz` or companion `.mpface.json`. When supplied, bypasses MediaPipe detection and uses saved landmarks directly. |
| `fill_mode` | COMBO | `flat` (solid fill_r/g/b color), `surround` (NS inpaint + `surround_style`), `inpaint` (per-zone Telea; brows processed before eyes). |
| `surround_style` | COMBO | `fill_mode=surround` only. `diffuse` (default): NS inpaint + internal Gaussian smoothing — soft gradient. `solid`: one flat color averaged from each zone's local surround ring, no internal blur — edges are feathered, not blurred. Use `solid` if `diffuse` reads as translucent/mixed. |
| `solid_bevel` | BOOL | `surround_style=solid` only. Adds a fake pseudo-3D bevel (distance-from-edge height field → fake normal → Lambert shade) on top of the flat color instead of a perfectly uniform fill — reads as a raised/embossed plate. Pairs well with `lip_mode=trapezoid`. |
| `solid_bevel_width` | INT | Bevel ramp width in px at 1536px (auto-scales). Distance inward from the edge before shading plateaus flat. |
| `solid_bevel_strength` | FLOAT | 0 = flat color (same as `solid_bevel=False`). 1 = full lit/shadow contrast. |
| `solid_bevel_zones` | STRING | Comma-separated zones that get the bevel + lift shadow (`lips`, `eyes`, `brows`, `nose`). Unlisted zones get a plain flat fill. Default `lips`. |
| `solid_lift_shadow` | FLOAT | Soft contact shadow on the skin just below the plate so it reads as lifted off the face. 0 = none. Default 0.3. |
| `eye_mode` | COMBO | `iris` (eyelid hull eroded by `eye_inset`), `eyelid` (raw eyelid hull). |
| `lip_mode` | COMBO | `organic` (outer contour + `lip_band` + `expand_lips`), `contour` (exact landmark polygon), `plane` (rotated rectangle along 61→291 mouth axis), `trapezoid` (4-point isosceles trapezoid along the same axis — see `lip_trapezoid_taper`). |
| `lip_trapezoid_taper` | FLOAT | `lip_mode=trapezoid` only. Philtrum-edge width ÷ chin-edge width. 1.0 = rectangle. <1.0 = narrower toward the philtrum (classic trapezoid). Pair with `fill_mode=surround` + `surround_style=solid` for a constant, deterministic flat-color lip cover plate instead of an inconsistent image-edit pass. |
| `expand_eyes` | FLOAT | Eye zone expand, 1536px-normalized. |
| `expand_brows` | FLOAT | Brow zone expand, 1536px-normalized. |
| `expand_lips` | FLOAT | Lip zone expand, 1536px-normalized. |
| `expand_nose` | FLOAT | Nose zone expand, 1536px-normalized. |
| `lip_band` | FLOAT | Extra height added to organic lip region. |
| `eye_inset` | FLOAT | Erode eyelid hull inward (iris mode only). |
| `feather` | INT | Master feather applied to all zones (positive = outward). |
| `eyes_feather` | INT | Per-zone override. -1 = use master `feather`. |
| `brows_feather` | INT | Per-zone override. -1 = use master `feather`. |
| `lips_feather` | INT | Per-zone override. -1 = use master `feather`. |
| `nose_feather` | INT | Per-zone override. -1 = use master `feather`. |
| `fill_r` | INT | Flat fill color R (0–255). |
| `fill_g` | INT | Flat fill color G (0–255). |
| `fill_b` | INT | Flat fill color B (0–255). |
| `fill_from_guide` | BOOL | When ON (with `image1` wired), sample fill colors from `image` (guide) instead of `image1`. |
| `oval_subtract_sockets` | BOOL | When ON, subtract active socket zones from the `face_oval` output. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `socket_image` | IMAGE | Filled result — all active zones painted. |
| `alpha_image` | IMAGE | RGBA version of `socket_image`. |
| `socket_mask` | MASK | Union of all filled zones. |
| `left_eye` | MASK | Left eye zone mask. |
| `right_eye` | MASK | Right eye zone mask. |
| `eyes` | MASK | Both eyes combined. |
| `left_brow` | MASK | Left brow zone mask. |
| `right_brow` | MASK | Right brow zone mask. |
| `brows` | MASK | Both brows combined. |
| `lips` | MASK | Lip zone mask. |
| `nose` | MASK | Nose zone mask. |
| `face_oval` | MASK | Full face oval (optionally minus sockets). |
| `lip_plane` | MASK | Rotated rectangle along the 61→291 mouth axis, always emitted regardless of `lip_mode`. |
| `status` | STRING | Detection result or error message. |

## Usage

- **Expand values are 1536px-normalized**: the node scales them by `max(H, W) / 1536.0`. Values tuned at 1536px stay consistent at any input resolution.
- **Two-image mode**: wire `image` as the reference character (for detection), wire `image1` as the version you want to paint. Useful when `image` is the raw render and `image1` is a cleaned or composited version.
- **lip_plane** is always available — use it as a pre-crop guide before running Qwen Image Edit on a lip region, regardless of which `lip_mode` you select for the actual fill.
- **fill_mode=inpaint** processes brows before eyes so brow-fill pixels can inform the eye inpaint; use for seamless skin reconstruction rather than hard socket fills.
- Wire `face_data_path` from `BD MP Save Face Data` to avoid re-running MediaPipe on every execution when the character image is fixed.
- **Mannequin prep for stylised low-poly 3D heads** (brows removed, eyes flattened in, mouth lifted as a plate the image-to-3D generator will build as a plane): `fill_mode=surround`, `surround_style=solid`, `eyes`/`brows` on, `lip_mode=trapezoid` (box) or `hull` (contoured plate), `solid_bevel=True`. Leave `solid_bevel_zones=lips` so eyes/brows stay flat, and `solid_lift_shadow` at 0.3 for the contact shadow that makes the plate read as raised off the face. Raise `solid_bevel_strength` if the generator needs a stronger cue; push `lip_trapezoid_taper` toward 1.0 for less taper.
- `surround_style=solid` paints its fill out to the raw mask **dilated by the positive feather**, so the feather ramp has real fill colour under it. (A hard fill that stopped at the raw edge left the outward ramp blending original-against-original — i.e. no visible feather.)
