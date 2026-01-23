# BD Mesh Inspector

Interactive Three.js-based 3D mesh viewer with PBR channel switching. Inspect mesh geometry, vertex colors, UV layout, normals, and individual PBR material channels without leaving ComfyUI.

## How It Works

BD Mesh Inspector renders your mesh in an embedded iframe using a bundled Three.js viewer (offline capable, no CDN dependency). The node exports the input mesh to a temporary GLB file and passes PBR channel data to the viewer via postMessage.

The viewer toolbar provides instant mode switching between 10 visualization modes - no re-execution required. You can orbit, pan, and zoom the mesh using standard mouse controls (left-drag rotate, right-drag pan, scroll zoom).

### Mesh Source Priority

The node accepts mesh data from three sources (in priority order):

1. **mesh** (TRIMESH) - Direct mesh input from any BD mesh node or TRELLIS2
2. **bundle** (MESH_BUNDLE) - From BD Pack Bundle or BD Cache Bundle, extracts mesh + all textures
3. **glb_path** (string) - Path to a GLB/glTF file on disk

If multiple sources are connected, higher-priority sources override lower ones.

### Bundle Integration

When a MESH_BUNDLE is connected, the node automatically extracts:
- The mesh geometry (if no direct mesh input is connected)
- Normal, alpha, diffuse textures (if no individual texture inputs override them)
- Metallic and roughness texture maps

Individual inputs always take priority over bundle data, allowing selective overrides.

## View Modes

| Mode | Visualization | Data Source |
|------|--------------|-------------|
| **Material** | Full PBR with vertex colors + lighting | GLB material |
| **Geometry** | Blue wireframe overlay | Mesh topology |
| **Colors** | Raw vertex colors (unlit) | Vertex color attribute |
| **UV** | UV coordinates as color (R=U, G=V) with checkerboard | UV attribute |
| **Normal** | Normal map texture or per-vertex normals as RGB | Normal map / geometry |
| **Metallic** | Texture map or per-vertex grayscale | Bundle texture / JSON array |
| **Roughness** | Texture map or per-vertex grayscale | Bundle texture / JSON array |
| **Alpha** | Alpha texture or vertex color alpha channel | Alpha map / vertex colors |
| **Emissive** | Emissive texture | Emissive map |
| **Diffuse** | Diffuse/albedo texture | Bundle texture / direct input |

Buttons for unavailable channels (no data provided) are grayed out automatically.

## Inputs

| Parameter | Data Type | Input Type | Default | Description |
|-----------|-----------|------------|---------|-------------|
| `mesh` | TRIMESH | Optional | - | Direct mesh input (from BD mesh nodes or TRELLIS2) |
| `bundle` | MESH_BUNDLE | Optional | - | Bundle from BD Pack Bundle or BD Cache Bundle |
| `glb_path` | STRING | Optional | `""` | Path to a GLB/glTF file to inspect directly |
| `initial_mode` | COMBO | Required | `full_material` | Initial view mode when the viewer loads |
| `metallic_json` | STRING | Optional | `""` | JSON array of per-vertex metallic values [0-1] |
| `roughness_json` | STRING | Optional | `""` | JSON array of per-vertex roughness values [0-1] |
| `normal_map` | IMAGE | Optional | - | Normal map texture |
| `emissive_map` | IMAGE | Optional | - | Emissive map texture |
| `alpha_map` | IMAGE | Optional | - | Alpha/opacity map texture |
| `diffuse_map` | IMAGE | Optional | - | Diffuse/albedo texture |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `status` | STRING | Summary: vertex/face count, source, available channels |

## Common Connections

```
BD_SampleVoxelgridPBR ──┬── mesh ─────────────► BD_MeshInspector
                         ├── metallic_json ────►
                         └── roughness_json ───►

BD_CacheBundle ──────────── bundle ────────────► BD_MeshInspector

BD_OVoxelTextureBake ───── mesh ──────────────► BD_MeshInspector
                         ├── diffuse ──────────►
                         └── normal ───────────►
```

## Technical Details

- **Renderer**: Three.js r160, bundled locally (712KB, no network required)
- **Export**: Mesh exported as GLB to ComfyUI output directory (preserves vertex colors + UVs)
- **Channel data**: Textures encoded as base64 PNG, per-vertex arrays as JSON
- **Widget**: 520x460 iframe with 4:3 aspect ratio
- **Controls**: OrbitControls (damped rotation, pan, zoom)
- **Dependencies**: No additional Python packages (uses existing trimesh, numpy, Pillow)
