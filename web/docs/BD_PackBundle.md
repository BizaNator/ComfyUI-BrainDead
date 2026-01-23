# BD Pack Bundle

Pack mesh, textures, and color data into a MESH_BUNDLE container.

## Overview

Collects all mesh asset data into a single MESH_BUNDLE for caching (BD Cache Bundle) and export (BD Blender Export). Textures can be provided explicitly or extracted automatically from the mesh's PBR material.

## Inputs

| Input | Type | Description |
|-------|------|-------------|
| `mesh` | TRIMESH | Mesh geometry (may include TextureVisuals with PBR material) |
| `color_field` | COLOR_FIELD | Optional voxelgrid color data for downstream use |
| `diffuse` | IMAGE | Optional albedo texture (extracted from mesh material if not provided) |
| `normal` | IMAGE | Optional normal map |
| `metallic` | IMAGE | Optional metallic texture |
| `roughness` | IMAGE | Optional roughness texture |
| `alpha` | IMAGE | Optional opacity texture |
| `name` | STRING | Bundle name (used for export filenames) |

## What Gets Bundled

- **Mesh**: Geometry with faces, vertices, and any attached material/UVs
- **Vertex colors**: Extracted from mesh visual data (if present)
- **Color field**: Raw voxelgrid color data for spatial reapplication
- **PBR textures**: Either explicitly provided or auto-extracted from mesh material
- **Name**: Used as the filename base during export

## Recommended Workflow

```
[TRELLIS2 Generate] --> [BD OVoxel Texture Bake] --> [BD Pack Bundle] --> [BD Cache Bundle]
                                                          ^
[BD Sample Voxelgrid Colors] --> color_field ------------+
```
