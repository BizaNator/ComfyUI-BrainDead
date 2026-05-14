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

### `BD_FlameFit`
- **Inputs:** `LANDMARKS_BATCH`, `flame_model_path` STRING (default
  `/srv/AI_Stuff/models/flame/FLAME2023.npz`), `shape_coeffs` INT (default 100),
  `exp_coeffs` INT (default 50), `iterations` INT (default 200).
- **Outputs:** `FLAME_FIT` (custom type — `{verts: (V,3) tensor, faces: (F,3)
  tensor, uvs: (V_uv,2) tensor, face_uv_idx: (F,3) tensor, cameras:
  [{intrinsics, R, t, view_hint} per input image]}`).
- **Implementation:** PyTorch L-BFGS on 4 views *jointly* — single shared
  shape/expression, per-view pose + camera. Loss = MediaPipe→FLAME landmark
  index map reprojection error + symmetric shape prior. No DECA, no CNN
  inference. ~200 lines of fit code + the 99-point MediaPipe→FLAME index
  map from FLAME's standard release.
- **Rear view:** if MediaPipe finds no face, drop that view from the fit
  but keep its image+rough-pose for the bake step (pose inferred from camera
  baseline + symmetry).

### `BD_FlameTextureBake`
- **Inputs:** `IMAGE`, `FLAME_FIT`, `view_index` INT (which camera to bake
  through), `texture_size` INT (default 2048).
- **Outputs:** `IMAGE` (partial UV texture), `MASK` (visibility/confidence —
  per-texel cosine of triangle normal vs view direction, clamped to [0,1]).
- **Implementation:** Reuses the exact UV-rasterization recipe from
  `ovoxel_texture_bake.py:191-220` — `dr.rasterize` UVs in NDC to get
  per-texel (face_id, barycentric), `dr.interpolate` to get 3D position +
  normal, project 3D → 2D via the fit's camera matrix, sample the photo
  with bilinear interpolation. Multiply alpha by cos(normal·view_dir).
- **Failure mode:** if the camera projection puts a texel outside the
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

Two new entries in `nodes/mesh/types.py` (or a new
`nodes/facewrap/types.py` if we go subfolder route):

```python
# Opaque types — full schema lives in the node file
LANDMARKS_BATCH = io.Custom("BD_LANDMARKS_BATCH")
FLAME_FIT       = io.Custom("BD_FLAME_FIT")
```

Each carries a Python dict with the fields above. No validation node yet —
contract enforced by the consuming node's `execute()`.

## File layout

New subfolder: `nodes/facewrap/`. Mirrors the segmentation/blender layout.

```
nodes/facewrap/
├── __init__.py            # registers FACEWRAP_V3_NODES + V1 dicts
├── types.py               # LANDMARKS_BATCH, FLAME_FIT custom-type helpers
├── landmarks.py           # BD_FaceLandmarks
├── flame_fit.py           # BD_FlameFit
├── texture_bake.py        # BD_FlameTextureBake
├── confidence_blend.py    # BD_UVConfidenceBlend
└── uv_transfer.py         # BD_UVTransfer
```

Add `FACEWRAP_V3_NODES` to the top-level `__init__.py` alongside the other
domain bundles.

## External assets

| Asset | Path | Source | Size | Required for |
|-------|------|--------|------|--------------|
| FLAME 2023 model | `/srv/AI_Stuff/models/flame/FLAME2023.npz` | https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2023Open.zip | ~150MB | `BD_FlameFit`, `BD_FlameTextureBake` |
| FLAME UV layout | bundled with FLAME 2023 release | — | — | `BD_FlameTextureBake` |
| MediaPipe → FLAME landmark map | static `.npy` shipped in `lib/facewrap/landmark_map.npy` | published in FLAME docs / DECA repo | <1KB | `BD_FlameFit` |
| FLAME ↔ CC5 correspondence | `/srv/AI_Stuff/models/flame/correspondences/cc5.npz` | built once with `tools/build_correspondence.py` | ~1MB | `BD_UVTransfer` |
| FLAME ↔ Metahuman correspondence | `/srv/AI_Stuff/models/flame/correspondences/metahuman.npz` | built once with same util | ~1MB | `BD_UVTransfer` |

License note: FLAME 2023 Open release is the carved-out version that drops
the research-only restriction. Verify `LICENSE.txt` inside the zip matches
intended use before bundling.

## MVP ordering

Land in this order — each layer is testable on its own:

1. `BD_FaceLandmarks` — proves the MediaPipe install + the per-view batch
   shape. No big deps.
2. `BD_FlameFit` — needs FLAME .npz + the landmark map. Tests the optimizer
   alone (no rendering).
3. `BD_FlameTextureBake` — uses fit + image; first node that actually
   produces a partial UV texture.
4. `BD_UVConfidenceBlend` — pure-numpy/torch, no new deps, easy to land.
5. `BD_UVTransfer` — only after the FLAME-side pipeline is solid AND a
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

- DECA / EMOCA / MICA — landmark-only fit gets us 90% of the quality for
  texture purposes (we don't need DECA's fine geometry; we need the canonical
  FLAME mesh in the right pose).
- Per-rig retraining of a ControlNet — Qwen-fill against a fixed FLAME UV
  is enough for v1.
- Hair / scalp generation — FLAME 2023 covers down to upper neck;
  hair is a separate problem and the Qwen-fill step covers minor scalp gaps.
- Animation rig binding — pure texture pipeline; rig handoff is the user's
  problem downstream.
