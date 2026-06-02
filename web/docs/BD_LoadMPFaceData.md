# BD MP Load Face Data

Reloads face region masks saved by **BD MP Save Face Data** — outputs the same 18 region masks as **BD MP Face Mask**, plus `head_mask`, `masked_skin`, and the embedded pre-resolution image.

**No MediaPipe required at load time.** Use this after image modification steps (eyes removed, mouth closed, delighted, albedo prepped) that would break live detection.

## Inputs

| Input | Description |
|-------|-------------|
| `file_path` | Path to `.mpface.npz` (from `BD MP Save Face Data` `npz_path` output, or typed manually). Also accepts the companion `.mpface.json` path — the node finds the NPZ automatically. Leave empty to auto-locate via `context_id`. |
| `context_id` | Wire from `BD_SaveContext` to auto-resolve the path from the context template. If `file_path` is empty and `context_id` is also empty, tries `auto_pick_context()` (works when exactly one context is registered). |
| `frame_index` | Which saved frame to load (0 for single-frame saves). |

## Outputs

Outputs match **BD MP Face Mask** exactly — wire to the same downstream nodes:

| Output | Type |
|--------|------|
| `face_oval` | MASK |
| `skin` | MASK |
| `left_eye` / `right_eye` / `eyes` | MASK |
| `left_brow` / `right_brow` / `brows` | MASK |
| `left_iris` / `right_iris` / `irises` | MASK |
| `lips` | MASK |
| `nose` | MASK |
| `left_ear` / `right_ear` / `ears` | MASK |
| `forehead` | MASK |
| `hair` | MASK |
| `head_mask` | MASK (blank if not saved) |
| `masked_skin` | MASK (blank if not saved) |
| `image` | IMAGE (black if not saved) |
| `status` | STRING |

Missing regions produce blank masks — the node never errors due to missing data.

## Typical Workflow

```
Clean render ──→ BD MP Face Mask ──→ BD MP Save Face Data
                                              │ npz_path
                                              ▼
Modified render ──────────────────→ BD MP Load Face Data
                                              │
                                         face_oval, skin, eyes, lips…
                                              │
                                              ▼
                                    BD MP Face Infill (or BD_GLSLBatch etc.)
```
