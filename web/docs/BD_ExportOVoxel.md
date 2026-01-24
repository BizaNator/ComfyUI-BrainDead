# BD Export OVoxel

Export VOXELGRID to o_voxel's native .vxz compressed format.

## Overview

Saves a TRELLIS2 VOXELGRID to disk using Microsoft's VXZ format (Sparse Voxel Octree + compression). This allows caching expensive texture generations for later re-baking with different decimation/UV settings without regenerating.

## Output Files

Two files are produced in the output directory:

| File | Format | Contents |
|------|--------|----------|
| `<name>.vxz` | VXZ (SVO + compression) | Voxel coordinates + PBR attributes (uint8) |
| `<name>.mesh.npz` | NumPy compressed | Original high-poly mesh vertices/faces + metadata |

Both files are required to fully reconstruct the VOXELGRID.

## Inputs

| Input | Type | Description |
|-------|------|-------------|
| `voxelgrid` | VOXELGRID | VOXELGRID from TRELLIS2 texture generation |
| `output_dir` | STRING | Output directory name relative to ComfyUI output/ (default: "mesh_export") |
| `filename` | STRING | Base filename without extension (default: "voxelgrid") |
| `compression` | COMBO | Compression algorithm: zstd, lzma, or deflate |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `vxz_path` | STRING | Full path to the saved .vxz file |
| `status` | STRING | Summary of export (voxel count, file size, compression ratio) |

## Compression Options

| Algorithm | Speed | Ratio | Notes |
|-----------|-------|-------|-------|
| **zstd** | Fast | Good | Recommended for most use cases, supports multi-threading |
| **lzma** | Slow | Best | Smallest files, good for archival |
| **deflate** | Medium | Medium | Standard zlib, widely compatible |

## VXZ Format Details

The VXZ format stores voxel data as a Sparse Voxel Octree (SVO):

1. Coordinates are encoded as Morton codes in the SVO structure
2. PBR attributes are quantized from float [-1,1] to uint8 [0,255]
3. Attributes are stored per-channel: base_color(3), metallic(1), roughness(1), alpha(1)
4. Chunks (256^3 blocks) are compressed independently for parallel I/O

## Quantization

Attributes undergo lossy quantization during export:
- Float [-1, 1] -> uint8 [0, 255] (256 levels)
- This is sufficient for PBR textures (equivalent to 8-bit color depth)
- Reconstruction error: max 1/255 per channel (~0.4%)

## Recommended Workflow

```
[BD TRELLIS.2 Texture] -> voxelgrid -> [BD Export OVoxel]
                                             |
                                        (saved to disk)
                                             |
                               [BD Load OVoxel] -> voxelgrid
                                                       |
                          [BD OVoxel Bake] or [BD OVoxel Texture Bake]
```

## Tips

- Use the same `output_dir` as BD Blender Export Mesh to keep mesh + voxelgrid together
- zstd compression is ~10-50x faster than lzma with only slightly larger files
- Typical file sizes: 50k-200k voxels = 5-20 MB (zstd), 3-15 MB (lzma)
- The mesh sidecar contains the original ~14M vertex mesh needed for BVH projection
