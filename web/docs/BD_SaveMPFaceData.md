# BD MP Save Face Data

Saves all MediaPipe face region masks — plus optional `head_mask`, `masked_skin`, and pre-resolution image — to a `.mpface.npz` + `.mpface.json` pair on disk.

**Purpose:** Once an image is processed (eyes removed, mouth closed, delighted, albedo prepared), MediaPipe can no longer detect the face reliably. Save the masks from the clean image first, then reload them at any later stage with **BD MP Load Face Data**.

## File Format

| File | Contents |
|------|----------|
| `{name}.mpface.npz` | All mask arrays as `uint8 (H, W)` — readable from Python/NumPy/Blender |
| `{name}.mpface.json` | Image dimensions, saved region list, per-region bboxes with `{x, y, width, height, cx, cy}` |

The JSON is readable directly from Blender Python:
```python
import numpy as np, json
data = np.load("face_01.mpface.npz")
meta = json.load(open("face_01.mpface.json"))
skin = data['skin']                      # uint8 (H, W)
eye_bbox = meta['bboxes']['left_eye']    # {x, y, width, height, cx, cy}
```

## Inputs

| Input | Description |
|-------|-------------|
| `context_id` | Wire from `BD_SaveContext` for automatic path resolution. When set, `output_dir` and `name` are ignored. |
| `face_oval` … `hair` | 18 region mask outputs from **BD MP Face Mask** (all optional — any subset can be wired). |
| `head_mask` | External head silhouette from SAM3 or **BD MP Face Refine**. |
| `masked_skin` | Refined skin mask from **BD MP Face Refine** (`skin` output). |
| `image` | Pre-resolution image for texture generation / Blender UV projection. Embedded in the NPZ as `image` (uint8 RGB). |
| `output_dir` | Subdirectory under `ComfyUI output/` (ignored when `context_id` resolves). |
| `name` | Base filename (ignored when `context_id` resolves). |
| `auto_increment` | Append `_001`, `_002`, … to avoid overwriting. |

## Outputs

| Output | Description |
|--------|-------------|
| `npz_path` | Full path to the saved `.mpface.npz` — wire to `BD MP Load Face Data` or `BD MP Face Infill`. |
| `json_path` | Full path to the companion `.mpface.json`. |
| `status` | Summary of what was saved. |

## Wiring BD MP Face Refine Masks

Refined SAM3-accurate masks from **BD MP Face Refine** wire directly to the matching inputs and replace the raw MediaPipe masks in the save:

```
BD MP Face Mask ──────────────────────┐
                                       ▼
BD MP Face Refine ──skin──────────→ BD MP Save Face Data
                 └─head_mask──────→
```

## Context ID Pattern

```
BD_SaveContext (context_id="facemaker") ──[wire]──→ BD MP Save Face Data context_id
```

Wire the `context_id` string output from `BD_SaveContext` to create the execution dependency. Typed-only (not wired) may fail if SaveContext hasn't executed yet.
