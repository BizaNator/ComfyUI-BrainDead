# BD OVoxel Bake

All-in-one PBR texture baking using Microsoft's o_voxel reference implementation.

## Overview

Takes the TRELLIS2 voxelgrid directly and produces a textured mesh with individual PBR texture maps. Handles mesh simplification, UV unwrapping, BVH projection, trilinear texture sampling, and tangent-space normal map baking in one integrated step.

## Pipeline (Internal)

1. **Multi-stage CuMesh simplification** with topology repair
2. **CuMesh UV unwrapping** with vertex normal computation
3. **BVH projection** to original mesh surface for accurate sampling
4. **Trilinear interpolation** from sparse voxel tensor (grid_sample_3d)
5. **UV seam inpainting** (TELEA algorithm)
6. **Tangent-space normal map** from high-poly → low-poly face normal transfer
7. **Coordinate system conversion** for GLB output

## Inputs

| Input | Type | Description |
|-------|------|-------------|
| `voxelgrid` | VOXELGRID | TRELLIS2 voxelgrid with PBR attributes and original mesh data |
| `decimation_target` | INT | Target face count for mesh simplification (default: 50000) |
| `texture_size` | INT | Output texture resolution (default: 2048) |
| `remesh` | BOOLEAN | Enable dual-contouring remesh for better topology (slower) |
| `remesh_band` | FLOAT | Remesh narrow band width (only if remesh=True) |
| `remesh_project` | FLOAT | Project back to original surface (0=none, 1=full) |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `mesh` | TRIMESH | Simplified mesh with PBR material and UVs |
| `diffuse` | IMAGE | Base color texture (RGB) |
| `normal` | IMAGE | Tangent-space normal map |
| `metallic` | IMAGE | Metallic texture (grayscale broadcast to RGB) |
| `roughness` | IMAGE | Roughness texture (grayscale broadcast to RGB) |
| `alpha` | IMAGE | Opacity texture (grayscale broadcast to RGB) |
| `status` | STRING | Summary of bake results |

## Normal Map

The normal map is computed as a proper tangent-space transfer from the original high-poly mesh to the simplified output:

- For each UV texel, BVH projects to the original high-poly surface
- Computes the high-poly face normal at the hit point
- Computes the low-poly tangent frame from UV gradients (Gram-Schmidt orthogonalized)
- Transforms the high-poly normal into tangent space

This captures geometric detail lost during decimation.

## Recommended Workflow

```
[TRELLIS2 Texture] → voxelgrid → [BD OVoxel Bake]
                                       ├── mesh → [BD Pack Bundle] → [BD Cache Bundle]
                                       ├── diffuse → preview
                                       └── normal → preview
```

## Tips

- For more control over decimation (hard edges, color preservation), use **BD OVoxel Texture Bake** instead with your own decimation + UV unwrap pipeline
- The `remesh` option gives cleaner topology but is significantly slower
- Output mesh includes `doubleSided=True` PBR material for correct rendering of any remaining flipped faces
