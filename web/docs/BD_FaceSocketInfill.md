# BD MP Face Infill

MediaPipe landmark-based face socket creator: fills eye, brow, lip, and nose zones with flat color, surrounding blur, or Telea inpaint — for 2D animation flipbook texture production.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Primary image. Detection + color sampling source. |
| `image1` | IMAGE (optional) | Fill target. When wired, sockets are painted into this image instead of `image`. |
| `face_data_path` | STRING (optional) | Path to a `.mpface.npz` or companion `.mpface.json`. When supplied, bypasses MediaPipe detection and uses saved landmarks directly. |
| `fill_mode` | COMBO | `flat` (solid fill_r/g/b color), `surround` (Gaussian blur of surrounding pixels), `inpaint` (per-zone Telea; brows processed before eyes). |
| `eye_mode` | COMBO | `iris` (eyelid hull eroded by `eye_inset`), `eyelid` (raw eyelid hull). |
| `lip_mode` | COMBO | `organic` (outer contour + `lip_band` + `expand_lips`), `contour` (exact landmark polygon), `plane` (rotated rectangle along 61→291 mouth axis). |
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
