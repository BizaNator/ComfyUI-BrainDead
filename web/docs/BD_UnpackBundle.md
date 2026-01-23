# BD Unpack Bundle

Unpack a MESH_BUNDLE into its individual components for re-processing.

## Overview

Extracts all contents from a MESH_BUNDLE (created by BD Pack Bundle or BD Cache Bundle) as separate outputs. This allows you to re-process the cached high-poly mesh through decimation, UV unwrapping, and texture baking without re-running expensive generation steps.

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `mesh` | TRIMESH | The mesh geometry with material/UVs intact |
| `color_field` | COLOR_FIELD | Voxelgrid color data for spatial reapplication |
| `diffuse` | IMAGE | Albedo/base color texture |
| `normal` | IMAGE | Normal map texture |
| `metallic` | IMAGE | Metallic texture |
| `roughness` | IMAGE | Roughness texture |
| `alpha` | IMAGE | Opacity/alpha texture |
| `name` | STRING | Bundle name (for export filenames) |
| `status` | STRING | Summary of unpacked contents |

## Recommended Workflow

```
[BD Cache Bundle] --> [BD Unpack Bundle]
                          +-- mesh ---------> [BD Blender Decimate] --> [BD UV Unwrap] --> [BD Pack Bundle] --> [BD Export]
                          +-- color_field --> [BD Blender Decimate]
                          +-- textures ----> (preview / edit if needed)
```

## Tips

- **Re-processing pipeline**: Use this to grab the cached high-poly mesh and color_field, then pipe through decimation and re-texturing without re-running TRELLIS2 generation.
- **Color field**: Connect directly to BD Blender Decimate's `color_field` input for spatial color sampling on the decimated mesh.
- **Textures**: Single-channel textures (metallic, roughness, alpha) are broadcast to 3-channel RGB for IMAGE compatibility.
- **Material preservation**: The mesh retains its original PBR material and UVs from the bundle.
