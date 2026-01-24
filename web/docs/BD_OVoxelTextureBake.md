# BD OVoxel Texture Bake

Bake PBR textures onto a pre-processed mesh using o_voxel's trilinear sampling.

## Overview

The "bake-only" node. Takes a mesh that already has UVs (from your own decimation + UV unwrap pipeline) and bakes PBR textures using BVH projection + trilinear grid_sample_3d from the voxelgrid. This gives you full control over mesh topology and edge preservation while using o_voxel's high-quality texture baking.

## Pipeline

1. **UV-space rasterization** of your mesh (nvdiffrast)
2. **BVH projection** to original high-res mesh surface
3. **Trilinear sampling** from sparse voxel tensor at projected positions
4. **Channel extraction** (base_color, metallic, roughness, alpha)
5. **Tangent-space normal map** from high-poly → low-poly face normal transfer
6. **UV seam inpainting** (TELEA algorithm, configurable radius)
7. **PBR material assembly** with normalTexture attached

## Inputs

| Input | Type | Description |
|-------|------|-------------|
| `mesh` | TRIMESH | Pre-processed mesh WITH UVs (from your own simplify + unwrap pipeline) |
| `voxelgrid` | VOXELGRID | TRELLIS2 voxelgrid with PBR attributes and original mesh data |
| `texture_size` | INT | Output texture resolution (default: 2048) |
| `inpaint_radius` | INT | Inpainting radius for UV seam filling (default: 3) |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `mesh` | TRIMESH | Input mesh with PBR material + UVs + normalTexture applied |
| `diffuse` | IMAGE | Base color texture (RGB) |
| `normal` | IMAGE | Tangent-space normal map |
| `metallic` | IMAGE | Metallic texture (grayscale broadcast to RGB) |
| `roughness` | IMAGE | Roughness texture (grayscale broadcast to RGB) |
| `alpha` | IMAGE | Opacity texture (grayscale broadcast to RGB) |
| `status` | STRING | UV coverage and bake results |

## Normal Map

Proper tangent-space normal map computed from the BVH projection data:

- Uses `face_id` from BVH to get the high-poly surface normal at each texel
- Computes low-poly tangent frame from UV gradients (Gram-Schmidt)
- Transforms high-poly normal into tangent space
- Result: flat blue (0.5, 0.5, 1.0) where surfaces agree, colored where detail was lost

## Recommended Workflow

```
[BD Cache Bundle] → [BD Unpack Bundle]
                         ├── mesh → [BD Blender Decimate] → [BD UV Unwrap] → [BD OVoxel Texture Bake]
                         └── voxelgrid ─────────────────────────────────────────────────┘
                                                                                   ├── mesh → [BD Pack Bundle]
                                                                                   ├── diffuse → preview
                                                                                   └── normal → preview
```

## Coordinate Space

- Input mesh: Z-up (ComfyUI TRIMESH convention)
- Voxelgrid original mesh: Y-up (internal TRELLIS2 convention)
- The node handles the conversion internally (Z-up → Y-up for BVH, then back)

## Tips

- Mesh MUST have UVs before connecting — use BD UV Unwrap first
- Higher `inpaint_radius` fills larger UV seam gaps but may blur edges
- UV coverage percentage in status tells you how much of the texture is used
- For best results, ensure your UV unwrap has good island packing (less wasted space = more texture detail)
