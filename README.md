# ComfyUI-BrainDead

A comprehensive collection of ComfyUI custom nodes for caching, character consistency, and prompt iteration.

**Created by BizaNator for [BrainDeadGuild.com](https://braindeadguild.com)**
*A Biloxi Studios Inc. Production*

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

## License

MIT License - see LICENSE file for details.

## Credits

- **Author**: BizaNator
- **Organization**: BrainDeadGuild.com
- **Studio**: Biloxi Studios Inc.

## Support

- GitHub Issues: [Report bugs](https://github.com/BizaNator/ComfyUI-BrainDead/issues)
- Discord: [BrainDeadGuild](https://discord.gg/braindeadguild)
