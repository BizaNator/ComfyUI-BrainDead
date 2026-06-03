# BD Pixal3D

Two-node pipeline for image-to-3D generation using Pixal3D (TencentARC/Pixal3D, Trellis2-based): preprocess estimates camera FOV, then generate produces a TRIMESH + TRELLIS2_VOXELGRID.

---

## BD Pixal3D Preprocess

Prepare an image for Pixal3D: apply mask, crop to subject, composite on background, and estimate camera FOV via MoGe-2 or manual input.

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Source image to preprocess. |
| `mask` | MASK (optional) | Object mask. When unwired, uses full image rectangle. |
| `background` | COMBO | Compositing background color: `black` (Pixal3D default), `gray`. |
| `fov_mode` | COMBO | Camera FOV estimation: `auto_moge` (MoGe-2 monocular depth, recommended) or `manual`. |
| `manual_fov` | FLOAT | Camera angle X in radians (0.05–2.0). Only used when `fov_mode=manual`. Try 0.2 if MoGe gives distorted results. |
| `mesh_scale` | FLOAT | Mesh scale factor for camera distance calculation (0.1–5.0). Default 1.0. |
| `extend_pixel` | INT | Expand/shrink FOV pixel range (-64 to 64). 0 = standard. |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `pixal3d_input` | PIXAL3D_INPUT | Preprocessed image + camera parameters bundle. Wire to `BD Pixal3D Image to 3D`. |
| `preprocessed_image` | IMAGE | Preview of the 512×512 image Pixal3D will receive. |

### Preprocessing steps

1. Apply `mask` to produce RGBA (if wired)
2. Crop tight to subject bounding box with 5% margin, expand to square
3. Composite on `background` color
4. Resize to 512×512
5. Estimate `camera_angle_x` via MoGe-2 or `manual_fov`

---

## BD Pixal3D Image to 3D

Generate a 3D mesh and voxelgrid from a preprocessed Pixal3D image.

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `pixal3d_input` | PIXAL3D_INPUT | From `BD Pixal3D Preprocess`. |
| `seed` | INT | Random seed for reproducibility. |
| `pipeline_type` | COMBO | `1024_cascade` (standard, recommended) or `1536_cascade` (higher geometry + texture resolution). |
| `max_tokens` | INT | Max sparse structure tokens (16384–65536). Lower = less VRAM. Default 49152. |
| `ss_guidance_strength` | FLOAT | Sparse structure CFG guidance scale. Default 7.5. |
| `ss_guidance_rescale` | FLOAT | Sparse structure guidance rescale. Default 0.7. |
| `ss_sampling_steps` | INT | Sparse structure sampling steps. Default 25. |
| `ss_rescale_t` | FLOAT | Sparse structure timestep rescale. Default 5.0. |
| `shape_guidance_strength` | FLOAT | Shape latent CFG guidance scale. Default 7.5. |
| `shape_guidance_rescale` | FLOAT | Shape latent guidance rescale. Default 0.5. |
| `shape_sampling_steps` | INT | Shape latent sampling steps. Default 25. |
| `shape_rescale_t` | FLOAT | Shape latent timestep rescale. Default 3.0. |
| `tex_guidance_strength` | FLOAT | Texture latent CFG guidance scale. Default 1.0. |
| `tex_guidance_rescale` | FLOAT | Texture latent guidance rescale. Default 0.0. |
| `tex_sampling_steps` | INT | Texture latent sampling steps. Default 25. |
| `tex_rescale_t` | FLOAT | Texture latent timestep rescale. Default 3.0. |
| `model_path` | STRING | HuggingFace repo or local path. Default `TencentARC/Pixal3D`. |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `mesh` | TRIMESH | Untextured Z-up geometry. Feed into `BD_CuMeshSimplify` or `BD_BlenderDecimate`. |
| `voxelgrid` | TRELLIS2_VOXELGRID | Full PBR voxel data. Feed into `BD_OVoxelBake` or `BD_OVoxelTextureBake`. |

---

## Usage

- Models (~10GB) download automatically on first run to `$HF_HOME` (`/srv/AI_Stuff/models/huggingface/`).
- `auto_moge` FOV estimation is recommended for most images. Use `manual` with `manual_fov=0.2` when MoGe produces distorted geometry (common for very stylized art or pure front-facing character sheets).
- The `voxelgrid` output is in the same format as `BD_Trellis2ShapeToTexturedMesh` — you can route it into `BD_OVoxelBake` for a complete sharp-edge-preserving PBR bake without any additional conversion.

## Recommended pipeline

```
[Source image + mask]
  ↓
BD Pixal3D Preprocess
  (fov_mode=auto_moge, background=black)
  ↓ pixal3d_input
BD Pixal3D Image to 3D
  (pipeline_type=1024_cascade)
  ↓ mesh              ↓ voxelgrid
BD CuMesh Simplify    BD OVoxelBake
  (target=20000)        → albedo, normal, roughness, metallic
  ↓ mesh
BD Blender Planar Normals
  → .glb for Unreal
```
