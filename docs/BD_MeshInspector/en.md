# BD Mesh Inspector

Interactive Three.js-based 3D mesh viewer with PBR channel switching. Inspect mesh geometry, vertex colors, UV layout, normals, and individual PBR material channels without leaving ComfyUI.

## How It Works

BD Mesh Inspector renders your mesh in an embedded iframe using a bundled Three.js viewer (offline capable, no CDN dependency). The node exports the input mesh to a temporary GLB file and passes PBR channel data to the viewer via postMessage.

The viewer toolbar provides instant mode switching between 10 visualization modes - no re-execution required. You can orbit, pan, and zoom the mesh using standard mouse controls (left-drag rotate, right-drag pan, scroll zoom).

### Mesh Source Priority

The node accepts mesh data from three sources (in priority order):

1. **mesh** (TRIMESH) - Direct mesh input from any BD mesh node or TRELLIS2
2. **mesh_path** (string) - Path to a GLB/glTF/PLY/OBJ file on disk
3. **bundle** (MESH_BUNDLE) - From BD Pack Bundle or BD Cache Bundle, extracts mesh + all textures

If multiple sources are connected, higher-priority sources override lower ones for the **mesh geometry**.

### Combining mesh_path + bundle (Optimal Performance)

For the fastest inspection with full PBR channels, connect **both** `mesh_path` and `bundle`:

- **mesh_path** provides the GLB file to serve directly (no re-export needed)
- **bundle** provides all PBR data (vertex colors, metallic, roughness, normal, alpha, diffuse)

This combination is ideal when you have a GLB output from BD_BlenderExportMesh and want to inspect it with all the PBR channels from the original bundle. The status will show `Source: file+bundle`.

```
BD_BlenderExportMesh ──── glb_path ──────────► BD_MeshInspector
                                                      ▲
BD_CacheBundle ──────────── bundle ───────────────────┘
```

### Bundle Integration

When a MESH_BUNDLE is connected, the node automatically extracts:
- The mesh geometry (if no direct mesh input or mesh_path is provided)
- Vertex colors (from bundle's vertex_colors or sampled from color_field)
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
| `mesh_path` | STRING | Optional | `""` | Path to a mesh file (GLB, glTF, PLY, OBJ, FBX, STL, OFF) |
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

# Optimal: file + bundle for speed with full PBR
BD_BlenderExportMesh ───── glb_path ───────────► BD_MeshInspector
BD_CacheBundle ──────────── bundle ────────────►      (same node)
```

## Technical Details

- **Renderer**: Three.js r160, bundled locally (712KB, no network required)
- **Export**: Mesh exported as GLB to ComfyUI temp directory with hash-based deduplication
- **Direct serving**: GLB/glTF files in output/input/temp directories are served directly (no re-export)
- **Channel data**: Textures downsampled to 1024px max and encoded as JPEG (PNG for alpha maps)
- **Widget**: 520x460 iframe, resizable (viewer fills available height when node is resized)
- **Controls**: OrbitControls (damped rotation, pan, zoom)
- **Dependencies**: No additional Python packages (uses existing trimesh, numpy, Pillow)

## Performance Tips

1. **Use mesh_path + bundle together** for the fastest inspection with full PBR channels
2. **Hash-based caching**: The same mesh geometry reuses the same temp file (no duplicate writes)
3. **Texture downsampling**: All textures are downsampled to 1024px for the preview viewer (faster encoding, smaller transfer)
4. **JPEG encoding**: Non-alpha textures use JPEG (quality 80) instead of PNG for ~10x faster encoding
