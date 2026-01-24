# BD Load OVoxel

Load a VOXELGRID from exported .vxz + .mesh.npz files.

## Overview

Reconstructs a full VOXELGRID dict from files saved by BD Export OVoxel. The loaded voxelgrid can be used with any downstream baking node (BD OVoxel Bake, BD OVoxel Texture Bake, BD Sample Voxelgrid Colors) without needing to regenerate from TRELLIS2.

## Inputs

| Input | Type | Description |
|-------|------|-------------|
| `vxz_path` | STRING | Path to the .vxz file (absolute, or relative to ComfyUI output/) |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `voxelgrid` | VOXELGRID | Reconstructed VOXELGRID dict with coords, attrs, mesh, and layout |
| `status` | STRING | Summary of loaded data (voxel count, mesh size, channels) |

## File Requirements

Both files must exist in the same directory with matching base names:

```
mesh_export/
  voxelgrid.vxz        <- Compressed voxel data
  voxelgrid.mesh.npz   <- Original mesh + metadata
```

Only the `.vxz` path needs to be specified; the `.mesh.npz` sidecar is found automatically.

## Reconstructed VOXELGRID

The output VOXELGRID dict contains:

| Key | Type | Description |
|-----|------|-------------|
| `coords` | numpy float32 (N,3) | Voxel grid coordinates |
| `attrs` | numpy float32 (N,6) | PBR attributes in [-1, 1] |
| `voxel_size` | float | Voxel size for coordinate scaling |
| `layout` | dict | Channel layout (slice objects for base_color, metallic, roughness, alpha) |
| `original_vertices` | torch.Tensor | High-poly mesh vertices (Y-up) |
| `original_faces` | torch.Tensor | High-poly mesh faces |

## Recommended Workflow

```
[BD Load OVoxel] -> voxelgrid -> [BD OVoxel Bake]
                                       |
                                  mesh, diffuse, normal, ...
                                       |
                              [BD Pack Bundle] -> [BD Blender Export Mesh]
```

Or with custom decimation:

```
[BD Load OVoxel] -> voxelgrid -> [BD OVoxel Texture Bake]
                         |                    ^
                         |                    |
                         +-> mesh (from cache/decimate) -> [BD UV Unwrap] -+
```

## Tips

- Supports both absolute paths and paths relative to ComfyUI's output directory
- The loaded voxelgrid is identical to the original (within uint8 quantization precision)
- Use this to iterate on mesh topology/UVs without expensive TRELLIS2 regeneration
- Combine with BD Cache Bundle to build a fast iteration workflow
