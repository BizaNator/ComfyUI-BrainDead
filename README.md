# ComfyUI-BrainDead


A comprehensive collection of ComfyUI custom nodes for caching, character consistency, and prompt iteration.


<div align="center">

  **🧠 BrainDeadGuild**

  *Don't Be BrAIn Dead Alone*

  *Games | AI | Community*

  [![BrainDeadGuild](https://img.shields.io/badge/BrainDeadGuild-Community-purple)](https://BrainDeadGuild.com/Discord)
  [![BrainDead.TV](https://img.shields.io/badge/BrainDead.TV-Lore-red)](https://BrainDead.TV)

</div>

## 🎯 About BrainDead Nodes
AN interanlly used node pack for Biloxi Studios designed to help with character, story, 3d and audio generation for next generation UGC game and TV pipelines.

## Features

### Cache Nodes (`BrainDead/Cache`)
Smart caching with **lazy evaluation** - upstream nodes are completely SKIPPED when cache is valid.

| Node | Description |
|------|-------------|
| **BD Cache Image** | Cache IMAGE tensors as PNG files |
| **BD Cache Mask** | Cache MASK tensors as PNG files |
| **BD Cache Latent** | Cache LATENT tensors as safetensors |
| **BD Cache Audio** | Cache AUDIO data as WAV files |
| **BD Cache String** | Cache STRING data as TXT files |
| **BD Cache Any** | Cache any data type as pickle |
| **BD Cache Mesh** | Cache TRIMESH objects as PLY files |
| **BD Save File** | Save any data in native format, output file path |
| **BD Load Image** | Load image from STRING path |
| **BD Load Mesh** | Load 3D mesh from STRING path |
| **BD Load Audio** | Load audio from STRING path |
| **BD Load Text** | Load text from STRING path |
| **BD Clear Cache** | Clear cached files by pattern |

**How Caching Works:**
1. Place a cache node after an expensive operation
2. First run: generates data, saves to `output/BrainDead_Cache/`
3. Subsequent runs: loads from cache, **skips upstream generation entirely**
4. Change the `seed` to force regeneration

### Character Nodes (`BrainDead/Character`)
Advanced tools for maintaining character consistency with Qwen-Image models.

| Node | Description |
|------|-------------|
| **BD Qwen Character Edit** | Enhanced image editing with character preservation |
| **BD Qwen T2I Custom** | Text-to-image with custom system prompts |
| **BD Qwen Multi-Image** | Multi-reference image editing |
| **BD Qwen Identity Lock** | Strong identity preservation with weighted references |

**Features:**
- Customizable system prompts for character consistency
- Multi-image support (up to 3 reference images)
- Per-image weight and role control
- Identity-focused templates that prioritize facial features

### Mesh Nodes (`BrainDead/Mesh`)
3D mesh processing, color sampling, and simplification tools.

| Node | Description |
|------|-------------|
| **BD Cache Mesh** | Cache TRIMESH objects as PLY files |
| **BD Sample Voxelgrid Colors** | Sample vertex colors from TRELLIS2 voxelgrid |
| **BD Sample Voxelgrid PBR** | Sample full PBR attributes from voxelgrid |
| **BD Transfer Vertex Colors** | BVH-based vertex color transfer between meshes |
| **BD Transfer Colors Pymeshlab** | Transfer colors using pymeshlab |
| **BD Mesh Repair** | Repair mesh topology (holes, normals, duplicates) |
| **BD Smart Decimate** | Edge-preserving decimation with pymeshlab |
| **BD Export Mesh With Colors** | Export mesh with vertex colors to GLB/PLY/OBJ |
| **BD CuMesh Simplify** | GPU-accelerated mesh simplification with color preservation |
| **BD Planar Grouping** | Group faces by normal direction with boundary straightening |

**BD Planar Grouping Features:**
- **Normal-based grouping**: Clusters faces with similar normals into planar regions
- **Boundary straightening**: Projects vertices onto plane intersections for clean edges
- **Face-split mesh support**: Works with meshes where each face has unique vertices
- **Configurable thresholds**: Angle threshold and minimum group size
- **Visualization**: Optional color-coding by group

### Blender Nodes (`BrainDead/Blender`)
Advanced mesh processing using Blender's geometry tools (requires Blender 5.0+).

| Node | Description |
|------|-------------|
| **BD Blender Decimate V3** | Full-featured decimation with edge preservation |
| **BD Blender Edge Marking** | Detect and mark edges from colors/angles |
| **BD Blender Merge Planes** | Merge geometry within marked regions |
| **BD Blender Remesh** | Voxel/quad remeshing with Blender |
| **BD Blender Cleanup** | Advanced mesh cleanup and repair |
| **BD Blender Vertex Colors** | Vertex color operations (bake, transfer) |
| **BD Blender Normals** | Normal fixing and recalculation |

**Hard Edge Preservation Pipeline:**
```
[Input Mesh]
    ↓
[BD Planar Grouping] (straighten_boundaries=True)
    ↓ Clean geometric boundaries
[BD Blender Edge Marking] (FROM_COLORS_AND_ANGLE)
    ↓ Mark SHARP/SEAM edges
[BD Blender Merge Planes] (delimit_sharp=True)
    ↓ Dissolve while respecting marks
[Low-poly mesh with hard edges intact]
```

**BD Blender Edge Marking Operations:**
- `FROM_COLORS`: Mark edges where vertex colors differ
- `FROM_ANGLE`: Mark edges by dihedral angle threshold
- `FROM_COLORS_AND_ANGLE`: Combine both methods
- `CLEAR`: Remove existing edge marks

**BD Blender Merge Planes Features:**
- **Delimit options**: Respect SHARP, SEAM, MATERIAL, NORMAL edges
- **Dissolve angle**: Control coplanar face merging threshold
- **Region subdivision**: Proportional face density based on area
- **Output topology**: TRI, QUAD, or NGON output

### Prompt Nodes (`BrainDead/Prompt`)
Iterate through multiple prompts with automatic filename generation.

| Node | Description |
|------|-------------|
| **BD Prompt Iterator** | Basic prompt cycling with filename generation |
| **BD Prompt Iterator (Advanced)** | Templates, suffix lists, seed modes, ping-pong |
| **BD Prompt Iterator (Dynamic)** | Up to 20 connected prompt inputs |

**Modes:**
- `sequential`: Cycle through prompts in order
- `manual`: Select specific prompt by index
- `random`: Shuffle prompts randomly
- `single`: Always use first prompt

**Filename Modes:**
- `auto_index`: `base_000`, `base_001`, etc.
- `suffix_list`: `base_front`, `base_left`, etc.
- `template`: `{base}_{index:03d}_{suffix}`

## Installation

### ComfyUI Manager
Search for "BrainDead" in ComfyUI Manager and install.

### Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/BizaNator/ComfyUI-BrainDead
```

### Dependencies
Most dependencies are included with ComfyUI. Optional:
```bash
# For audio caching
pip install torchaudio

# For mesh caching
pip install trimesh
```

## Usage Examples

### Cache Expensive Generation
```
[Expensive Node] → [BD Cache Image] → [Next Node]
                     seed: 42
```
First run generates and caches. Subsequent runs skip generation.

### Character Sheet Workflow
```
[BD Prompt Iterator (Advanced)]
  prompts: "front view\nleft profile\nright profile\nback view"
  suffixes: "_front\n_left\n_right\n_back"
  base_filename: "character"
  ↓
[BD Qwen Character Edit]
  ↓
[BD Save File]
  name_prefix: "MyCharacter"
```

### Multi-Reference Character Edit
```
[Load Image 1 (identity)] ─┐
[Load Image 2 (style)]    ─┼→ [BD Qwen Identity Lock] → [Generate]
[Load Image 3 (pose)]     ─┘     image1_strength: 1.5
                                  image1_role: character
```

## Directory Structure

```
output/
├── BrainDead_Cache/     # Cache nodes save here (clearable)
│   ├── cached_image_abc123.png
│   ├── cached_mesh_def456.ply
│   └── MyProject/       # Subdirectories supported via name_prefix
│       └── step1_xyz789.png
└── saved_file.png       # BD Save File saves to main output/
```

## Node Categories

```
BrainDead/
├── Cache/       # Caching and file I/O nodes
├── Mesh/        # 3D mesh processing and color tools
├── Blender/     # Blender-based mesh operations
├── TRELLIS2/    # TRELLIS2-specific caching
├── Character/   # Qwen-Image character consistency
└── Prompt/      # Prompt iteration tools
```

## Tips

### Caching Best Practices
- Use descriptive `cache_name` values: `"trellis_base_mesh"`, `"character_face_gen"`
- Use `name_prefix` for project organization: `"Project1/Step1"`
- Connect workflow seed to cache `seed` for automatic invalidation
- Use `force_refresh` to regenerate without changing seed

### Character Consistency
- Always use the highest quality reference image for `image1`
- Set `image1_role` to `"character"` and highest strength
- Use secondary images only for style/pose reference
- The Identity Lock node is optimized for face preservation

### Prompt Iteration
- Use `suffix_list` mode for clean filenames: `char_front`, `char_left`
- Connect `seed` output to sampler for different seeds per prompt
- Use `workflow_id` to maintain separate iteration states


---

<div align="center">

## 🧠 BrainDeadGuild

**Professional AI Tools for Creative Production**

Created by **BizaNator**

[BrainDeadGuild.com](https://BrainDeadGuild.com) | [BrainDead.TV](https://BrainDead.TV) | [GitHub](https://github.com/BrainDeadGuild) | [Discord](https://braindeadguild.com/discord)

### Other BrainDead ComfyUI Nodes
- BD - Image Descriptor (Coming Soon)
- BD - Prompt Iterator Enhanced (Coming Soon)
- BD - Character Consistency Suite (Coming Soon)

*Building tools for the BrainDeadGuild community*

---

**A Biloxi Studios Inc. Production**

© 2024 Biloxi Studios Inc. - All Rights Reserved

</div>

## Support

- GitHub Issues: [Report bugs](https://github.com/BizaNator/ComfyUI-BrainDead/issues)
- Discord: [BrainDeadGuild](https://BrainDeadGuild.com/Discord)
