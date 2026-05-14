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

LANDMARKS_BATCH ─→ BD_FlameFit  ─→  FLAME_FIT (verts + cameras, per view)

FLAME_FIT + images ─→ BD_FlameTextureBake (per view, nvdiffrast)
                       └─→ TEXTURE_BATCH    (4 × UV partials)
                       └─→ CONFIDENCE_BATCH (4 × view-cosine masks)

TEXTURE_BATCH + CONFIDENCE_BATCH ─→ BD_UVConfidenceBlend
                                     └─→ IMAGE   (FLAME-UV composite)
                                     └─→ MASK    (filled vs gap)

IMAGE + MASK ─→ [existing Qwen Image Edit nodes] ─→ finalized FLAME texture

(optional retarget to a different rig:)
finalized FLAME texture ─→ BD_UVTransfer (FLAME→CC5/Metahuman) ─→ retargeted texture
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
- **Inputs:** `LANDMARKS_BATCH`, `model_path` STRING (default
  `lib/facewrap/canonical_face_model.obj` shipped with the pack).
- **Outputs:** `FACE_FIT` (see `nodes/facewrap/types.py`).
- **Model (v1):** MediaPipe canonical face mesh — 468 vertices, 898 triangle
  faces, per-vertex UVs. Apache-2.0 licensed, bundled in `lib/facewrap/`.
  The 468 vertex indices correspond 1:1 with FaceLandmarker landmarks
  0..467; the extra 10 iris/pupil landmarks (468..477) are NOT mesh
  vertices and are dropped at fit time.
- **Implementation:** NO optimization in v1 — MediaPipe already produces
  subject-specific 2D + 3D per-vertex positions, and the 4x4 transform
  gives per-view pose. We just assemble the LANDMARKS_BATCH into a
  shape-and-pose-bearing FACE_FIT for the bake node to consume.
- **Why this is enough for textures:** the bake step only needs to know
  where each vertex projects in the source photo. MediaPipe's 2D landmarks
  ARE that projection. No shape basis, no L-BFGS — Delaunay-warp the
  photo into UV space using 468 control points.
- **Rear view / no detection:** flagged `detected=False`; per-view fields
  zero-filled. The bake node skips them or uses a pose synthesized from
  camera baselines.
- **Future swap-in:** `BD_FlameFit` can later produce the same FACE_FIT
  custom type using FLAME 2023 (requires MPI auth) or ICT-FaceKit (open).

### `BD_FaceTextureBake`
- **Inputs:** `IMAGE`, `FACE_FIT`, `view_index` INT (which view to bake
  through), `texture_size` INT (default 2048).
- **Outputs:** `IMAGE` (partial UV texture), `MASK` (visibility/confidence —
  per-texel cosine of triangle normal vs view direction, clamped to [0,1]).
- **Implementation:** Reuses the exact UV-rasterization recipe from
  `ovoxel_texture_bake.py:191-220` — `dr.rasterize` UVs in NDC to get
  per-texel (face_id, barycentric). With v1's MediaPipe canonical fit we
  skip the 3D-camera-projection step entirely: each vertex already has
  a known 2D position in the source photo (`verts_2d` from the FACE_FIT),
  so we just `dr.interpolate(verts_2d, rast, faces)` to get per-texel
  source-image coords, then bilinear-sample. Confidence = per-face normal
  in 3D-landmark space dotted with view direction, clamp to [0,1].
- **Failure mode:** if the interpolated source coord lands outside the
  photo bounds, confidence = 0 for that texel.

### `BD_UVConfidenceBlend`
- **Inputs:** `IMAGE` batch (N partial textures), `MASK` batch (N confidence
  maps), `blend_mode` combo (`linear` / `multiband`), `seam_dilate` INT
  (default 4).
- **Outputs:** `IMAGE` (composite), `MASK` (binary "filled" — Qwen inpaint mask).
- **Implementation:** linear = per-pixel weighted average by confidence;
  multiband = Laplacian pyramid blend at confidence boundaries (better seam
  hiding). Edge-dilate the output by `seam_dilate` to prevent black-bleed
  at UV seams when sampled.

### `BD_UVTransfer` (FLAME → CC5/Metahuman, OPTIONAL)
- **Inputs:** `IMAGE` (source texture in source UV), `TRIMESH` source mesh,
  `TRIMESH` target mesh, `correspondence_path` STRING (path to a pre-built
  `.npz` with vertex-pair indices), `output_size` INT.
- **Outputs:** `IMAGE` (texture in target UV), `MASK` (target-UV alpha).
- **Implementation:** for each target-UV texel: rasterize target mesh in
  target UV → get (target_face_id, target_bary). Map target-face vertices
  via correspondence → source-face vertices → source UV → sample source
  texture. nvdiffrast rasterize + bilinear interpolate. Pre-built
  correspondence is built ONCE per rig with a small utility (closest-point
  match between FLAME and the target mesh, hand-tweakable in Blender).

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
├── face_fit.py            # BD_FaceFit (MediaPipe canonical)
├── texture_bake.py        # BD_FaceTextureBake
├── confidence_blend.py    # BD_UVConfidenceBlend
└── uv_transfer.py         # BD_UVTransfer

lib/facewrap/
├── canonical_face_model.obj  # MediaPipe canonical (Apache-2.0, bundled)
└── NOTICE.md                 # Third-party attribution
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
