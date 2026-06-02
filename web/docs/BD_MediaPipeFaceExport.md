# BD MP Face Export

Passthrough node — runs MediaPipe on `image[0]`, writes three reference files to disk, then passes the full image batch through unchanged. Zero effect on downstream nodes.

**Insert before** any albedo/greyscale/channel-pack/delight steps. MediaPipe needs the full-color, full-resolution image.

Feeds the face plate UV calibration pipeline: `DCC/Characters/Head/calibrate_faceplate_uv.py` (Eros P4).

## Files Written

Three files are written per run, using the same stem:

| File | Contents |
|------|----------|
| `{stem}_mp_{angle}.json` | Landmark JSON (schema below) |
| `{stem}_mp_{angle}_mask.png` | RGBA zone mask PNG |
| `{stem}_mp_{angle}_image.png` | Source image RGB — landmark coordinates are normalized to this pixel grid |

### Landmark JSON (`schema_version: 1`)

```json
{
  "schema_version": 1,
  "angle": "front",
  "image_size": [6144, 6144],
  "landmark_count": 478,
  "landmarks": [[0.5035, 0.4178], ...],
  "face_bbox": {"x_min": 0.18, "y_min": 0.07, "x_max": 0.82, "y_max": 0.93},
  "regions": {
    "lips_outer": [61, 146, 91, ...],
    "lips_inner": [78, 95, 88, ...],
    "left_eye": [33, 7, 163, ...],
    "right_eye": [362, 382, 381, ...],
    "left_brow": [70, 63, 105, ...],
    "right_brow": [300, 293, 334, ...],
    "nose_tip": [4],
    "chin": [152]
  },
  "image_file": "ernest_chavez_head_v3SR_mp_front_image.png"
}
```

Coordinates: normalized `[0.0, 1.0]`, origin top-left, x right, y down. Z omitted. `image_file` cross-references the saved source image — load both from Blender to map landmarks to pixels: `lm[i][0] * W, lm[i][1] * H`.

### RGBA Mask PNG

Same pixel dimensions as input. Values: 0 = outside, 255 = inside. Uses the same mask generation as **BD MP Face Mask** (shared library) — pixel-accurate, not crude convex hulls.

| Channel | Region |
|---------|--------|
| R | Lips |
| G | Brows |
| B | Eyes |
| A | Face oval |

Channels map directly to `FacePlate_Zone` vertex color painting in Blender.

### No-Face Handling

If detection fails: writes JSON with `landmark_count: 0` and no mask PNG. Source image is still saved. Node logs a warning and passes image through.

## Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `image` | — | Full-color, full-resolution image. Only `image[0]` is processed; all batch items pass through unchanged. |
| `angle` | `front` | Camera angle: `front`, `side_left`, `side_right`. Stored in JSON and used as the path suffix. |
| `context_id` | — | Wire from `BD_SaveContext` for automatic path resolution. `output_dir` and `filename_stem` are ignored when context resolves. Auto-picks if empty and only one context is registered. |
| `output_dir` | — | Fallback absolute path when no context is registered. NAS B: → `/mnt/tank/Studio/Brains/` on this server. |
| `filename_stem` | — | Fallback stem when no context is registered. Angle is appended automatically. |
| `detection_confidence` | 0.3 | Lower values catch stylized renders. |

## Outputs

| Output | Description |
|--------|-------------|
| `image` | Passthrough — identical to input. |
| `landmark_count` | 478 if detected, 0 if not. |
| `json_path` | Absolute path of the written JSON file. |

## Context ID + File Placement

```
BD_SaveContext (context_id="facemaker") ──[wire]──→ BD MP Face Export context_id
```

With a `BD_SaveContext` template like `FaceMaker/%character%/images/mp/%character%_head_v%version%%suffix%`, the export resolves to:
```
FaceMaker/ernest_chavez/images/mp/ernest_chavez_head_v3SR_mp_front.json
FaceMaker/ernest_chavez/images/mp/ernest_chavez_head_v3SR_mp_front_mask.png
FaceMaker/ernest_chavez/images/mp/ernest_chavez_head_v3SR_mp_front_image.png
```

**Wire the context_id output** (don't just type the same string) to ensure `BD_SaveContext` executes before this node.

### Manual NAS Placement Convention

```
output_dir  = /mnt/tank/Studio/Brains/Characters/<char>/images/mp
filename_stem = <char>_head_<ver>_mp
```

For side_right: `has_right_image=True` in `calibrate_faceplate_uv.calibrate_from_json()`. If absent, the UV calibration mirrors side_left — acceptable for crowd/NPC characters.

## Pairs With

- **BD MP Save Face Data** — save the full 18-region mask set alongside the export
- **BD MP Face Infill** — use saved face data to infill on modified images
- `calibrate_faceplate_uv.py` — Blender script that reads the JSON to calibrate face plate UV
