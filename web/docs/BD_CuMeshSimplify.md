# BD CuMesh Simplify

GPU-accelerated mesh simplification using CuMesh, with optional hole-fill, remesh, and multi-stage progressive decimation for better topology.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `mesh` | TRIMESH | Input mesh. Automatically detected as face-split if verts_per_face > 2.9 (vertices are merged first). |
| `target_faces` | INT | Target face count after simplification. |
| `fill_holes` | BOOL | Fill open boundaries before simplification. |
| `remesh` | BOOL | Isotropic remesh before simplification (requires `cumesh.remeshing`). |
| `remesh_target_len` | FLOAT | Target edge length for remeshing. |
| `multi_stage` | BOOL | Progressive simplification: 3× target → clean → 1× target. Produces better topology than single-pass. |
| `clean_duplicates` | BOOL | Remove duplicate vertices before simplification. |
| `clean_non_manifold` | BOOL | Remove non-manifold geometry before simplification. |
| `clean_small_components` | BOOL | Remove disconnected components smaller than a threshold. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mesh` | TRIMESH | Simplified mesh in Z-up coordinate system. |
| `status` | STRING | Face count before/after, VRAM used, any warnings. |

## Coordinate convention

CuMesh uses Y-up. The node converts automatically:

- Input: Z-up → Y-up swap before processing
- Output: Y-up → Z-up swap before returning

The TRIMESH output is always in ComfyUI's Z-up convention.

## Simplification pipeline

1. Face-split detection → vertex merge (if needed)
2. `fill_holes` (optional)
3. `remesh` (optional)
4. Clean: duplicates, non-manifold, small components
5. Simplify: multi-stage (3× → clean → 1×) or single-pass
6. Final cleanup

## Usage

- `multi_stage=ON` gives significantly better edge distribution for game-asset topology. Use it whenever face count is dropping more than 50%.
- For Trellis2 meshes: run `fill_holes=ON` + `clean_non_manifold=ON` first — Trellis2 outputs frequently have small holes and non-manifold edges that confuse CuMesh's simplifier.
- Wire into `BD_BlenderPlanarNormals` after simplification to assign flat panel normals before export to Unreal.

---

# BD CuMesh Quad Remesh

GPU-accelerated quad remeshing, returning native quad topology as OBJ and a triangulated TRIMESH.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `mesh` | TRIMESH | Input triangle mesh. |
| `target_faces` | INT | Target quad face count. |
| `fill_holes` | BOOL | Fill open boundaries before remeshing. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mesh` | TRIMESH | Triangulated version of the quad remesh (for downstream BD mesh nodes). |
| `quad_obj_path` | STRING | Path to the saved OBJ file containing native quad topology. |
| `status` | STRING | Quad/triangle counts, output path. |

## Usage

- `quad_obj_path` can be loaded directly in Blender or a DCC tool to get the true quad mesh.
- The `mesh` TRIMESH output is the triangulated version for continued processing through `BD_UVUnwrap`, `BD_OVoxelTextureBake`, etc.
- Requires `cumesh.remeshing` — the node checks for this at load time and will report an error if unavailable.
