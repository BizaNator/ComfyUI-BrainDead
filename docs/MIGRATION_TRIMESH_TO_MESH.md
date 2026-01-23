# Migration Plan: TRIMESH (Custom) to io.Mesh (Built-in)

## Status: INVESTIGATION REQUIRED

## Current State

BrainDead uses a custom `TRIMESH` type (`io.Custom("TRIMESH")`) that wraps `trimesh.Trimesh` objects. This type carries:
- Vertices, faces, normals
- Vertex colors (RGBA)
- UV coordinates
- TextureVisuals / material (PBR)
- Metadata dict (edge marks, cache keys, etc.)
- Export methods (GLB, OBJ, PLY, STL, etc.)

## Built-in io.Mesh Analysis

The ComfyUI built-in `io.Mesh` type (`comfy_api.latest._util.geometry_types.MESH`) is:

```python
class MESH:
    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor):
        self.vertices = vertices
        self.faces = faces
```

**It only stores vertices and faces as torch.Tensors.** No vertex colors, UVs, materials, normals, metadata, or export capabilities.

## Gap Analysis

| Feature | TRIMESH (current) | io.Mesh (built-in) | Gap |
|---------|-------------------|--------------------|----|
| Vertices | numpy array | torch.Tensor | Conversion needed |
| Faces | numpy array | torch.Tensor | Conversion needed |
| Vertex colors | RGBA uint8/float | NOT SUPPORTED | BLOCKER |
| UV coordinates | per-vertex float | NOT SUPPORTED | BLOCKER |
| Normals | per-vertex/face | NOT SUPPORTED | |
| Material/PBR | TextureVisuals | NOT SUPPORTED | BLOCKER |
| Metadata | arbitrary dict | NOT SUPPORTED | |
| Export (GLB/OBJ) | built-in | NOT SUPPORTED | |
| Scene graphs | trimesh.Scene | NOT SUPPORTED | |

## Conclusion

**io.Mesh is NOT a viable replacement for TRIMESH** in its current form. The BrainDead pipeline relies heavily on vertex colors, UVs, materials, and export functionality that io.Mesh does not provide.

## Options

### Option A: Keep TRIMESH (Recommended Short-term)

Keep the custom TRIMESH type. It provides everything needed and is used consistently across:
- BrainDead (35 files, 243 occurrences)
- ComfyUI-TRELLIS2 (3 files)
- ComfyUI-GeometryPack (40+ files)
- ComfyUI-MeshSegmenter (12+ files)
- ComfyUI-UniRig (5 files)
- ComfyUI-Hunyuan3d-2-1 (1 file)
- comfyui-hunyuan3d-part (5 files)

**Pros:** No work needed, already functional, shared across ecosystem
**Cons:** Not using official type, could diverge from ComfyUI core

### Option B: Propose io.Mesh Extension Upstream

Open a PR/issue to ComfyUI core proposing to extend `MESH` class:

```python
class MESH:
    def __init__(self, vertices, faces, vertex_colors=None, uvs=None,
                 normals=None, material=None, metadata=None):
        self.vertices = vertices      # torch.Tensor (N, 3)
        self.faces = faces            # torch.Tensor (F, 3)
        self.vertex_colors = vertex_colors  # torch.Tensor (N, 4) RGBA float
        self.uvs = uvs                # torch.Tensor (N, 2) or None
        self.normals = normals        # torch.Tensor (N, 3) or None
        self.material = material      # dict with PBR textures
        self.metadata = metadata      # dict
```

**Pros:** Official type, future-proof, community alignment
**Cons:** Requires ComfyUI core team buy-in, may take time, torch.Tensor vs numpy friction

### Option C: Adapter Pattern (Bridge)

Create converter nodes `TRIMESH <-> MESH` and accept both types:

```python
# Accept both types in inputs
io.Custom("TRIMESH|MESH").Input("mesh")  # If ComfyUI supports union types

# Or use type coercion in execute()
if isinstance(mesh, MESH):
    mesh = trimesh.Trimesh(
        vertices=mesh.vertices.cpu().numpy(),
        faces=mesh.faces.cpu().numpy()
    )
```

**Pros:** Backward compatible, works with both ecosystems
**Cons:** Conversion overhead, lost data (colors/UVs) from io.Mesh inputs

### Option D: Wrap trimesh in io.Mesh (Long-term)

If io.Mesh gets extended upstream, migrate all nodes gradually:

1. Update `types.py` to use `io.Mesh.Input` / `io.Mesh.Output`
2. Convert trimesh.Trimesh to MESH at output boundaries
3. Convert MESH to trimesh.Trimesh at input boundaries
4. Eventually remove the conversion layer once io.Mesh is rich enough

## Affected Files (BrainDead Only)

### Core Type Definition
- `nodes/mesh/types.py` — TrimeshInput/TrimeshOutput definitions

### Mesh Nodes (all use TrimeshInput/TrimeshOutput)
- `nodes/mesh/cache.py` (12 occurrences)
- `nodes/mesh/sampling.py` (9)
- `nodes/mesh/transfer.py` (16)
- `nodes/mesh/processing.py` (10)
- `nodes/mesh/export.py` (6)
- `nodes/mesh/simplify.py` (6)
- `nodes/mesh/unwrap.py` (6)
- `nodes/mesh/bake.py` (6)
- `nodes/mesh/export_bundle.py` (7)
- `nodes/mesh/grouping.py` (3)
- `nodes/mesh/color_field.py` (6)
- `nodes/mesh/fix_normals.py` (6)
- `nodes/mesh/bundle.py` (8)
- `nodes/mesh/inspector.py` (5)
- `nodes/mesh/ovoxel_bake.py` (1)
- `nodes/mesh/ovoxel_texture_bake.py` (4)
- `nodes/mesh/ovoxel_convert.py` (1)
- `nodes/mesh/edge_utils.py` (2)

### TRELLIS2 Integration
- `nodes/trellis2/shape.py` (6)
- `nodes/trellis2/cache.py` (19)
- `nodes/trellis2/texture.py` (8)

### Blender Nodes
- `nodes/blender/remesh.py` (7)
- `nodes/blender/repair.py` (7)
- `nodes/blender/decimate.py` (7)
- `nodes/blender/decimate_full.py` (6)
- `nodes/blender/transfer.py` (8)
- `nodes/blender/export_mesh.py` (4)
- `nodes/blender/base.py` (4)
- `nodes/blender/addon_nodes.py` (25)

### Cache System
- `nodes/cache/file_ops.py` (7)

## External Dependencies (Other Packs Using TRIMESH)

These packs also use the TRIMESH type and would need coordinated migration:

- **ComfyUI-TRELLIS2** — Outputs TRIMESH from inference nodes
- **ComfyUI-GeometryPack** — 40+ files, all mesh I/O uses TRIMESH
- **ComfyUI-MeshSegmenter** — 12+ files for mesh segmentation
- **ComfyUI-UniRig** — 5 files for rigging pipeline
- **ComfyUI-Hunyuan3d-2-1** — 1 file for 3D generation
- **comfyui-hunyuan3d-part** — 5 files for part segmentation

## Recommended Action

1. **Now:** Keep TRIMESH. It works, is battle-tested, and io.Mesh is insufficient.
2. **Track:** Monitor ComfyUI core for io.Mesh extensions (check changelogs/PRs).
3. **Propose:** Open an issue/PR on ComfyUI suggesting io.Mesh should support vertex colors and UVs at minimum.
4. **Future:** If io.Mesh gets extended, create a phased migration plan starting with types.py adapter layer.

## Estimated Scope

- **35 files** in BrainDead alone
- **243 total occurrences** of TRIMESH references
- **6+ external packs** that would need coordination
- **Migration risk:** High (type mismatches break node connections in existing workflows)
