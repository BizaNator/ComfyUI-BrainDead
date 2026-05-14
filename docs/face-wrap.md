# BD Face Wrap — multi-view UV texture pipeline

Take 4 head photos (front / left / right / rear) and produce a single
UV-unwrapped face/head texture suitable for a 3D rig. Optionally retarget
the texture to a non-FLAME UV (CC5, Metahuman) and use Qwen Image Edit to
fill gaps.

## Pipeline shape

```
front.png ─┐
left.png  ─┼─→ BD_FaceLandmarks (batch)  ─→  LANDMARKS_BATCH
right.png ─┤
rear.png  ─┘

LANDMARKS_BATCH ─→ BD_FaceFit  ─→  FACE_FIT (mesh + per-view 2D/3D verts)
                   mesh_source: mediapipe_canonical | ict_facekit

images + FACE_FIT ─→ BD_FaceTextureBake (all views, nvdiffrast)
                      └─→ uv_textures   (N × UV partials)
                      └─→ confidences   (N × view-cosine masks)

uv_textures + confidences ─→ BD_UVConfidenceBlend
                              └─→ uv_texture  (composite)
                              └─→ filled_mask (filled vs gap)

uv_texture + filled_mask ─→ [Qwen Image Edit inpaint] ─→ qwen_output
                                          ↓
        uv_texture + filled_mask + qwen_output ─→ BD_FaceWrapComposite
                              └─→ uv_texture (face preserved, gaps filled)

(optional retarget to a different rig:)
final texture ─→ BD_UVTransfer (donor UV → CC5/Metahuman/your-mesh UV)
```

## Node-by-node contract

### `BD_FaceLandmarks`
- **Inputs:** `IMAGE` (1..N batch), `model` combo (`face_mesh_full` / `face_mesh_lite`).
- **Outputs:** `LANDMARKS_BATCH` (custom type — list of `{landmarks_2d: (468,2),
  landmarks_3d: (468,3), view_hint: str, image_size: (h,w)}`), `IMAGE`
  (debug overlay with landmark dots).
- **Implementation:** MediaPipe `FaceMesh(refine_landmarks=True)`. Already
  installed (0.10.31). No model download.
- **Notes:** `view_hint` is auto-inferred from yaw angle of the canonical
  pose MediaPipe returns: front (|yaw|<25°), left (yaw>25°), right (yaw<-25°),
  rear (no face detected → user must label). User can override per image.

### `BD_FaceFit`
- **Inputs:** `LANDMARKS_BATCH`, `mesh_source` combo
  (`mediapipe_canonical` | `ict_facekit`), `mesh_obj_path` STRING
  (optional override; empty = bundled asset for the source).
- **Outputs:** `FACE_FIT` (see `nodes/facewrap/types.py`).
- **mesh_source = `mediapipe_canonical`** (default): MediaPipe canonical
  face mesh — 468 verts, 898 faces, per-vertex UVs (Apache-2.0, bundled).
  The 468 vertices correspond 1:1 with FaceLandmarker landmarks 0..467,
  so the fit is **pure assembly** — no optimization. MediaPipe already
  produces subject-specific 2D + 3D per-vertex positions; the bake only
  needs each vertex's 2D projection in the photo, which the landmarks
  ARE. Face-only: **no ears, scalp, or back of head.**
- **mesh_source = `ict_facekit`**: ICT-FaceKit head-skin mesh — 14,062
  verts, 28,068 tris (MIT, bundled; preprocessed by
  `tools/preprocess_ict.py` from ICT's `generic_neutral_mesh.obj` —
  extracts the `M_Face` + `M_BackHead` skin, triangulates, packs the
  two UDIM tiles into one [0,1] atlas). **Full head incl. ears + scalp
  + back.** Per view, the ICT neutral head is Procrustes-fitted
  (Umeyama similarity: scale + rotation + translation) to MediaPipe's
  68 iBUG landmarks. The fit poses the **neutral** ICT shape — not
  per-subject shape-accurate, so the baked face is somewhat distorted,
  but it gives the bake somewhere to land ear / scalp / rear pixels.
  Fit-quality upgrade tracked: landmark-exact TPS warp or per-subject
  identity-coeff fit.
- **Rear view / no detection:** flagged `detected=False`; per-view fields
  zero-filled. `BD_FaceTextureBake` skips undetected views.
- **Future swap-in:** `BD_FlameFit` can produce the same FACE_FIT type
  using FLAME 2023 (requires MPI auth).

### `BD_FaceTextureBake`
- **Inputs:** `IMAGE` (photo batch), `FACE_FIT`, `view_index` INT
  (default `-1` = bake ALL detected views; `>= 0` = bake just that view),
  `texture_size` INT (default 2048), `min_confidence` FLOAT (optional).
- **Outputs:** `IMAGE` (`uv_textures` — an N-length batch of partial UV
  textures), `MASK` (`confidences` — N-length batch of per-texel
  view-cosine masks in [0,1]).
- **Batched by default:** the node bakes every detected view in one call
  and outputs batches that wire straight into `BD_UVConfidenceBlend` —
  no per-view nodes or merge nodes needed. `view_index >= 0` still works
  for debugging a single view (output is a 1-length batch). Undetected
  views (rear / failed detection) are skipped — they carry no usable
  landmarks.
- **Implementation:** Reuses the exact UV-rasterization recipe from
  `ovoxel_texture_bake.py:191-220` — `dr.rasterize` UVs in NDC to get
  per-texel (face_id, barycentric). With v1's MediaPipe canonical fit we
  skip the 3D-camera-projection step entirely: each vertex already has
  a known 2D position in the source photo (`verts_2d` from the FACE_FIT),
  so we just `dr.interpolate(verts_2d, rast, faces)` to get per-texel
  source-image coords, then bilinear-sample. Confidence = per-face normal
  in 3D-landmark space dotted with view direction, clamp to [0,1]. One
  `RasterizeCudaContext` is created and reused across all baked views.
- **Failure mode:** if the interpolated source coord lands outside the
  photo bounds, confidence = 0 for that texel.

### `BD_UVConfidenceBlend`
- **Inputs:** `uv_textures` IMAGE batch (N partial textures), `confidences`
  MASK batch (N view-cosine maps), `seam_dilate` INT (default 4),
  `fill_threshold` FLOAT (optional), `confidence_gamma` FLOAT (default 3).
- **Outputs:** `uv_texture` IMAGE (composite), `confidence` MASK (combined),
  `filled_mask` MASK (binary — the Qwen inpaint region).
- **Implementation:** per-pixel confidence-weighted average. `confidence_gamma`
  raises the per-view weights to a power so the highest-confidence view
  dominates each texel instead of averaging in slightly-misaligned weaker
  views (cuts ghosting/blur — default 3, was 1). The fill decision uses the
  raw pre-gamma confidence so gamma can't push texels below `fill_threshold`.
  `seam_dilate` grows the filled region by N px via **non-blurring edge
  extend** (each new pixel copies a filled neighbour, no averaging) to
  prevent black-bleed at UV seams under bilinear sampling.

### `BD_FaceWrapComposite` (Qwen form guarantee)
- **Inputs:** `original_texture` IMAGE (the pre-Qwen blend output),
  `filled_mask` MASK (the blend's filled_mask), `qwen_output` IMAGE
  (Qwen Image Edit inpaint result), `feather` INT (default 2).
- **Outputs:** `uv_texture` IMAGE — `filled*original + (1-filled)*qwen`.
- **Why:** the Qwen-fill step must only invent the *gap* regions; the
  baked face is real photo data and must not drift. Even a hard latent
  noise mask can shift the "preserved" region via VAE round-trips / mask
  bleed. This node makes the guarantee explicit — the baked face pixels
  come back byte-identical. `feather` softens the preserved/filled seam.
  Resolution drift (Qwen output at a different size) is auto-resized.

### `BD_UVTransfer` (donor UV → CC5/Metahuman/your-mesh, OPTIONAL)
- **Inputs:** `source_texture` IMAGE (texture in the donor UV),
  `correspondence_path` STRING (path to a pre-built `.npz`),
  `output_size` INT, `source_mask` MASK (optional).
- **Outputs:** `uv_texture` IMAGE (texture in target UV), `filled_mask` MASK.
- **Implementation:** for each target-UV texel: rasterize target mesh in
  target UV → (target_face_id, target_bary) → `dr.interpolate` the
  per-target-vertex source-UV lookup → bilinear-sample the source texture.
  The correspondence `.npz` is built ONCE per target rig with
  `tools/build_correspondence.py` (BVH closest-point match donor→target).

### Qwen-fill wiring (between Blend and Transfer)

Run Qwen **on the blend output, in the donor-mesh UV** — before
`BD_UVTransfer`, not after. The donor UV (canonical / ICT) is consistent
and well-formed; a target rig's UV (low-poly, hard-edged) is much harder
for Qwen to match.

```
BD_UVConfidenceBlend
  ├─ uv_texture ──→ VAE Encode ──→ latent ──────────────┐
  ├─ filled_mask ─→ InvertMask ──→ gap_mask ──→ SetLatentNoiseMask
  └─ uv_texture ──→ (also feed as the reference image) ─┤
                                                        ↓
        TextEncodeQwenImageEditPlus (prompt: "seamlessly complete the
        facial skin texture, consistent skin tone + lighting, preserve
        all existing detail, add no new features")
                                                        ↓
                          KSampler → VAE Decode → qwen_output
                                                        ↓
        BD_FaceWrapComposite(original=uv_texture, filled_mask, qwen_output)
                                                        ↓
                          BD_UVTransfer → target rig UV
```

Three layers protect "form": the hard noise mask (Qwen only denoises the
gap), `BD_FaceWrapComposite` (baked pixels pasted back byte-identical),
and feeding `uv_texture` as Qwen's reference (gap-fill matches skin tone).

## Custom types

Two opaque types defined in `nodes/facewrap/types.py`. The "FLAME_FIT" name
was generalized to "FACE_FIT" so the same type works regardless of which
parametric / canonical mesh actually generated it (MediaPipe canonical
today, FLAME / ICT-FaceKit tomorrow).

```python
LANDMARKS_BATCH = io.Custom("BD_LANDMARKS_BATCH")
FACE_FIT        = io.Custom("BD_FACE_FIT")
```

Each carries a Python dict with the fields above. No validation node yet —
contract enforced by the consuming node's `execute()`.

## File layout

New subfolder: `nodes/facewrap/`. Mirrors the segmentation/blender layout.

```
nodes/facewrap/
├── __init__.py            # registers FACEWRAP_V3_NODES + V1 dicts
├── types.py               # LANDMARKS_BATCH, FACE_FIT custom-type helpers
├── landmarks.py           # BD_FaceLandmarks
├── face_fit.py            # BD_FaceFit (mediapipe_canonical | ict_facekit)
├── texture_bake.py        # BD_FaceTextureBake
├── confidence_blend.py    # BD_UVConfidenceBlend
├── qwen_composite.py      # BD_FaceWrapComposite
└── uv_transfer.py         # BD_UVTransfer

lib/facewrap/
├── canonical_face_model.obj  # MediaPipe canonical (Apache-2.0, bundled)
├── ict/
│   ├── ict_head_skin.obj     # ICT-FaceKit head skin (MIT, preprocessed)
│   └── ict_landmarks_68.json # 68 iBUG landmark vertex indices
└── NOTICE.md                 # Third-party attribution

tools/
├── build_correspondence.py   # one-time per-rig: donor→target UV warp map
└── preprocess_ict.py         # one-time: ICT neutral mesh → bundled head skin
```

Add `FACEWRAP_V3_NODES` to the top-level `__init__.py` alongside the other
domain bundles.

## External assets

| Asset | Path | Source | Size | Required for |
|-------|------|--------|------|--------------|
| MediaPipe FaceLandmarker model | `/srv/AI_Stuff/models/mediapipe/face_landmarker.task` | auto-downloaded from `storage.googleapis.com/mediapipe-models/face_landmarker/...` | ~3.6 MB | `BD_FaceLandmarks` |
| MediaPipe canonical face mesh | `lib/facewrap/canonical_face_model.obj` (bundled in repo) | github.com/google-ai-edge/mediapipe `mediapipe/modules/face_geometry/data/` — Apache-2.0 | ~45 KB | `BD_FaceFit`, `BD_FaceTextureBake` |
| Canonical ↔ CC5 correspondence | `/srv/AI_Stuff/models/facewrap/correspondences/cc5.npz` | built once with `tools/build_correspondence.py` | ~1 MB | `BD_UVTransfer` |
| Canonical ↔ Metahuman correspondence | `/srv/AI_Stuff/models/facewrap/correspondences/metahuman.npz` | built once with same util | ~1 MB | `BD_UVTransfer` |

**Future (when FLAME credentials are available):**
- FLAME 2023 Open release at https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2023Open.zip
- Requires MPI account registration — the URL serves an auth page to logged-out clients.
- Once obtained, a `BD_FlameFit` node can produce the same `FACE_FIT`
  custom type using FLAME's parametric shape/expression basis and full
  head topology (including scalp/neck — wider coverage than the canonical
  face-only mesh).

## MVP ordering

Land in this order — each layer is testable on its own:

1. `BD_FaceLandmarks` — proves the MediaPipe install + the per-view batch
   shape. ✅ landed.
2. `BD_FaceFit` — loads the canonical .obj, assembles LANDMARKS_BATCH into
   FACE_FIT. Pure assembly in v1, no optimization.
3. `BD_FaceTextureBake` — uses fit + image; first node that actually
   produces a partial UV texture.
4. `BD_UVConfidenceBlend` — pure-torch, no new deps, easy to land.
5. `BD_UVTransfer` — only after the canonical pipeline is solid AND a
   correspondence has been built for one target rig.

## Existing infra we leverage

- `BlenderNodeMixin` for any auxiliary mesh massaging (e.g. exporting
  finalized textured FLAME mesh as GLB).
- `nvdiffrast` UV-rasterization recipe from `ovoxel_texture_bake.py` —
  identical pattern, just sampling a 2D image through a camera instead of
  a 3D voxel grid.
- `cumesh.cuBVH` for the correspondence-builder utility (closest-point
  search FLAME → target mesh).
- `io.Custom("…")` pattern for opaque types — same as `TRIMESH`,
  `VOXELGRID`, `PARTS_BUNDLE`.

## Out of scope (intentional)

- DECA / EMOCA / MICA — MediaPipe's landmarks already encode subject
  shape; the canonical-mesh fit gets us 90% of the quality for texture
  purposes (we don't need fine geometry; we need each vertex's 2D
  projection in the source photo, which MediaPipe already gives us).
- Per-rig retraining of a ControlNet — Qwen-fill against the canonical
  UV (or whichever target UV) is enough for v1.
- Hair / scalp generation — the canonical face mesh covers face area only.
  Scalp + rear coverage is the Qwen-fill step's job. A FLAME-based future
  node would extend down to the upper neck.
- Animation rig binding — pure texture pipeline; rig handoff is the user's
  problem downstream.
