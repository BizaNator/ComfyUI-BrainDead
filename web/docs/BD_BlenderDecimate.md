# BD Blender Decimate

Full-featured stylized low-poly mesh decimation using Blender.

## Overview

Takes a high-poly mesh and reduces it to a target face count while preserving color boundaries and structural edges. Supports direct color field input for spatial color sampling on clean topology (no face splitting required).

## Pipeline Steps

1. **Color Field Sampling** - If `color_field` is connected, samples colors spatially onto vertices using KDTree (no face splitting needed)
2. **Pre-Cleanup** - Merge doubles, fix non-manifold geometry, remove loose vertices
3. **Planar Grouping** - Clusters faces by normal similarity, marks group boundaries as sharp/seam
4. **Color Edge Detection** - Detects color boundaries between adjacent faces, marks as sharp/seam
5. **Hole Filling** - Closes open boundary edges
6. **Internal Removal** - Removes hidden/interior faces
7. **Planar Decimate** - Merges coplanar faces using dissolve with `delimit` (preserves SHARP, SEAM, MATERIAL edges)
8. **Triangulate** - Converts n-gons to triangles for proper collapse behavior
9. **Collapse Decimate** - Reduces to target face count
10. **Sharp Edge Marking** - Marks edges above angle threshold as sharp
11. **Color Transfer** - Transfers colors back from reference mesh (face-based, no bleeding)
12. **Normal Fixing** - Recalculates normals to face outward

## Key Inputs

| Input | Description |
|-------|-------------|
| `mesh` | Input TRIMESH geometry |
| `color_field` | Optional COLOR_FIELD for spatial color sampling on clean topology |
| `target_faces` | Target face count after decimation |
| `planar_angle` | Angle threshold for planar dissolve (merges coplanar faces) |
| `sharp_angle` | Angle threshold for marking sharp edges after decimation |
| `detect_color_edges` | Enable color boundary detection (marks edges for preservation) |
| `color_edge_threshold` | Color difference threshold (lower = more edges detected) |
| `use_planar_grouping` | Enable normal-based planar group detection |
| `planar_group_angle` | Max angle between normals in same group (lower = more groups) |

## Recommended Workflow

```
[BD UnpackBundle]
    +-- mesh ----------> [BD Blender Decimate] --> [BD UV Unwrap] --> [BD Export]
    +-- color_field ---->        ^
```

Connect `color_field` directly instead of using BD_ApplyColorField first. This avoids face splitting and preserves clean topology for proper edge detection and decimation.

## Tips

- **Color field vs vertex colors**: When `color_field` is connected, colors are sampled spatially inside Blender on the actual imported vertex positions. This is more accurate than pre-computing vertex colors which can have index-order mismatches after GLB roundtrip.
- **Planar grouping**: Enable `use_planar_grouping` for meshes with clear flat-shaded planes (e.g., stylized characters, buildings). This preserves structural boundaries even without color data.
- **Edge preservation**: The planar decimate uses `delimit` to respect both color-detected edges AND planar group boundaries simultaneously.
- **Target faces**: The collapse step only runs if the mesh still exceeds `target_faces` after planar decimation.
