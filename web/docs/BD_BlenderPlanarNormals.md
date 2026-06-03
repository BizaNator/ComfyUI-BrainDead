# BD Blender Planar Normals

Detect connected face groups by angle threshold, assign each group's average normal as flat custom split normals, and mark sharp edges between groups — for the flat-panel stylized look in Unreal/Unity import.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `mesh` | TRIMESH | Input mesh in Z-up (ComfyUI convention). |
| `angle_threshold` | FLOAT | Maximum dihedral angle (degrees) between adjacent faces to consider them part of the same planar group. Lower values = more groups = more hard edges. |
| `output_dir` | STRING (optional) | Directory for sidecar GLB output. Defaults to ComfyUI output directory. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `mesh` | TRIMESH | Processed mesh for downstream BD mesh nodes (e.g. UV unwrap, OVoxel bake). |
| `glb_path` | STRING | Path to the sidecar GLB file with custom normals embedded, for direct import into Unreal/Blender. |
| `status` | STRING | Group count, sharp edge count, output paths. |

## Algorithm

1. BFS planar grouping: connected faces within `angle_threshold` are assigned to the same group.
2. Sharp edges are marked between faces in different groups.
3. Each face-corner within a group receives the group's average flat normal as a custom split normal.
4. Within a group, edges are marked smooth.

Two outputs are saved:

- **PLY** — for the TRIMESH pipeline (returned as `mesh`)
- **Sidecar GLB** — contains the custom split normals as exported by Blender, suitable for Unreal Engine import and Blender-to-game-engine workflows

## Usage

- `angle_threshold=30°` is a good starting point for stylized low-poly characters. Lower values (10–15°) produce more panel groups (sharper look). Higher values (45°+) merge more faces into fewer groups (softer look).
- The sidecar GLB (`glb_path`) is the primary deliverable for Unreal import — Unreal reads Blender custom split normals directly and preserves the flat-panel appearance.
- Wire upstream from `BD_UVUnwrap` or `BD_OVoxelTextureBake` — Blender Planar Normals should be the last geometry-modification step before UV and texture baking.
- Uses the bundled Blender at `lib/blender/blender-5.0.1-linux-x64/blender` via `BlenderNodeMixin.run_blender_script()`. No separate Blender installation required.
