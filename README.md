# ComfyUI-BrainDead


A comprehensive collection of ComfyUI custom nodes for caching, character consistency, prompt iteration, **3D mesh / PBR baking, character segmentation pipelines, and game-asset preparation**.


<div align="center">

  **🧠 BrainDeadGuild**

  *Don't Be BrAIn Dead Alone*

  *Games | AI | Community*

  [![BrainDeadGuild](https://img.shields.io/badge/BrainDeadGuild-Community-purple)](https://BrainDeadGuild.com/discord)
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
| **BD Save File** | Save any data in native format, output file path. Optional `context_id` + `suffix` for template-based saves |
| **BD Bulk Save** | Save N inputs in ONE Run using a save context (16 typed slots + parallel labels) |
| **BD Save Context** | Define a path template + variables once, downstream save nodes auto-resolve |
| **BD Get Context Path** | Resolve a context's template into a STRING — wire into ComfyUI's built-in SaveImage `filename_prefix` or any third-party save node |
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

**Save Context System** — replace ad-hoc concat-string filename chains:
1. Add ONE `BD Save Context` node with a template like `%project%/%character%/%character%_%name%_v%version%%suffix%` and set vars (character, name, version, project, etc.)
2. Downstream `BD Save File` / `BD Bulk Save` nodes auto-pick the context (or specify `context_id` if multiple are registered)
3. Each save just sets its `suffix` (e.g. `_albedo`, `_skin_mask`, `_head`) — context resolves the rest
4. Per-save `custom_vars` (multiline `key=value`) layer on top of the context for one-off variations like `subfolder=PBR` or `materials=metal`
5. `BD Get Context Path` outputs a STRING path for ANY save node — works with ComfyUI's built-in SaveImage and third-party packs

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
3D mesh processing, color sampling, simplification, and PBR texture baking tools.

| Node | Description |
|------|-------------|
| **BD Cache Mesh** | Cache TRIMESH objects as PLY files |
| **BD Sample Voxelgrid Colors** | Sample vertex colors from TRELLIS2 voxelgrid |
| **BD Sample Voxelgrid PBR** | Sample full PBR attributes from voxelgrid |
| **BD Apply Color Field** | Apply COLOR_FIELD to any mesh (deferred color application) |
| **BD Transfer Vertex Colors** | BVH-based vertex color transfer between meshes |
| **BD Transfer Colors Pymeshlab** | Transfer colors using pymeshlab |
| **BD Mesh Repair** | Repair mesh topology (holes, normals, duplicates) |
| **BD Smart Decimate** | Edge-preserving decimation with pymeshlab |
| **BD Export Mesh With Colors** | Export mesh with vertex colors to GLB/PLY/OBJ |
| **BD CuMesh Simplify** | GPU-accelerated mesh simplification with color preservation |
| **BD UV Unwrap** | UV unwrap with xatlas (GPU) or Blender Smart UV |
| **BD Planar Grouping** | Group faces by normal direction with boundary straightening |
| **BD Combine Edge Metadata** | Combine edge metadata from multiple sources |
| **BD OVoxel Bake** | All-in-one PBR bake from voxelgrid (simplify + UV + bake) |
| **BD OVoxel Texture Bake** | Bake-only: takes pre-processed mesh + voxelgrid |
| **BD Mesh To OVoxel** | Convert textured mesh to VOXELGRID format |
| **BD Export OVoxel** | Export VOXELGRID to .vxz compressed format |
| **BD Load OVoxel** | Load VOXELGRID from .vxz + sidecar files |
| **BD Fix Normals** | Fix face orientation (Python-only, fast) |
| **BD Pack Bundle** | Pack mesh + textures + colors into MESH_BUNDLE |
| **BD Unpack Bundle** | Unpack MESH_BUNDLE into individual components |
| **BD Cache Bundle** | Cache MESH_BUNDLE for fast reload |
| **BD Mesh Inspector** | Inspect mesh properties (verts, faces, UVs, colors) |

**OVoxel Baking Pipeline:**
```
[TRELLIS2 Texture] → voxelgrid → [BD OVoxel Bake] → mesh + PBR textures
                                       OR
[TRELLIS2 Texture] → voxelgrid → [BD OVoxel Texture Bake]
                          ↑ mesh → [BD Blender Decimate] → [BD UV Unwrap] ↗
```

**VXZ Caching Pipeline:**
```
[TRELLIS2 Texture] → voxelgrid → [BD Export OVoxel] → saved .vxz + .mesh.npz
                                                            ↓
[BD Load OVoxel] → voxelgrid → [BD OVoxel Bake] → re-bake without regenerating
```

### Blender Nodes (`BrainDead/Blender`)
Advanced mesh processing using Blender's geometry tools (requires Blender 5.0+).

| Node | Description |
|------|-------------|
| **BD Blender Decimate** | Full-featured decimation with edge preservation |
| **BD Blender Edge Marking** | Detect and mark edges from colors/angles |
| **BD Blender Merge Planes** | Merge geometry within marked regions |
| **BD Blender Remesh** | Voxel/quad remeshing with Blender |
| **BD Blender Cleanup** | Advanced mesh cleanup and repair |
| **BD Blender Vertex Colors** | Vertex color operations (bake, transfer) |
| **BD Blender Normals** | Normal fixing and recalculation |
| **BD Blender Export Mesh** | Export MESH_BUNDLE as GLB with material + vertex colors |

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
Iterate through multiple prompts with automatic filename generation, plus per-Run iteration over typed inputs.

| Node | Description |
|------|-------------|
| **BD Prompt Iterator** | Basic prompt cycling with filename generation |
| **BD Prompt Iterator (Advanced)** | Templates, suffix lists, seed modes, ping-pong |
| **BD Prompt Iterator (Dynamic)** | Up to 20 connected prompt inputs |
| **BD Filename Template** | Standalone STRING resolver for `%var%` templates (no save side-effect) |
| **BD For Each Run** | Multi-Run iterator over up to 16 typed inputs + parallel labels — drives any downstream chain (Qwen Edit + Save, etc.) where downstream needs to actually re-execute per item |

**Iterator vs Bulk Save:** `BD Bulk Save` (in Cache) saves N files in ONE Run — use when downstream IS just save. `BD For Each Run` (here) emits one (data, label) per Run — use when downstream is more complex (e.g. per-iteration upscale before save).

**Modes:**
- `sequential`: Cycle through prompts in order
- `manual`: Select specific prompt by index
- `random`: Shuffle prompts randomly
- `single`: Always use first prompt

**Filename Modes:**
- `auto_index`: `base_000`, `base_001`, etc.
- `suffix_list`: `base_front`, `base_left`, etc.
- `template`: `{base}_{index:03d}_{suffix}`

### Segmentation Nodes (`BrainDead/Segmentation`)
Character segmentation, PBR map derivation, and game-texture asset prep.

**SeeThrough pipeline** — operates on `SEETHROUGH_PARTS` dict from `SeeThrough_PostProcess`:

| Node | Description |
|------|-------------|
| **BD SeeThrough Extract Skin Mask** | Slice a global skin MASK into per-part skin masks (intersect with each part's xyxy + alpha). Excluded-tag and clothing-alpha-subtract options. |
| **BD SeeThrough Get Part** | Pull one tag's IMAGE + body_mask + skin_mask + bbox + depth_median |
| **BD SeeThrough Set Part** | Replace one tag's image (mutates in place — required for queue-iteration accumulation) |
| **BD SeeThrough Composite** | Assemble parts back to a flat RGBA image (depth-sorted). `trigger=is_last` gate so heavy compose runs once. |
| **BD SeeThrough Part Iterator** | Cycle through tags per queue Run (sequential/manual/single + filter modes) |
| **BD SeeThrough List Tags** | Diagnostic: report all detected tags + per-tag stats |
| **BD SeeThrough Export PNGs** | Per-tag PNGs + composite + skin-composite saved to disk |

**SAM3 multi-prompt + mask cleanup:**

| Node | Description |
|------|-------------|
| **BD SAM3 Multi-Prompt** | One node replaces 12-chain SAM3Segment. Vote/weighted_vote modes (majority counting), positive + negative prompts, color filter (off/exclude/include/exclude_and_include/remove_matching), adaptive LAB color sampling (works for any category — skin, clothing, hair), silhouette enforcement |
| **BD Mask Resolver** | Python port of GLSL Mask Resolver shader. Priority-based separation (skin/clothes/accessories) with adaptive LAB color, neighbor-vote gap fill |
| **BD Human Parser Mask Clean** | Per-class morphological cleanup + min-area filter on parse maps |
| **BD Human Parser Mask Split** | Split a parse map into per-region MASK outputs |
| **BD Human Parser Named Mask** | Extract a single label as MASK |
| **BD Human Parser Preview** | Colorized RGB visualization of a parse map |

**Human parser backends** (deferred for stylized art — trained on photos):

| Node | Description |
|------|-------------|
| **BD Fashn Human Parser** | SegFormer-B4 from fashn-ai (NVIDIA license). 18 FASHN classes. |
| **BD ATR Human Parser** | mattmdjaga/segformer_b2_clothes (MIT). 18 ATR classes. |

**BrainPed bridge:**

| Node | Description |
|------|-------------|
| **BD Parts Remap 26 → 11 (BrainPed)** | Translate SeeThrough's 26-tag dict into BrainPed's 11-part rig vocabulary. Mirrors BrainPed's `LAYER_TO_PART_MAP`; reads `config/segments/{Part}.json` for per-part crop_bounds. L/R-ambiguous parts bbox-masked. |

**Asset prep / game textures:**

| Node | Description |
|------|-------------|
| **BD Mask Flatten** | Flatten RGB+alpha onto chosen background (white/black/colors/checker/custom hex/background_image). Channel-routing modes (image_to_red/green/blue/grayscale). `edge_pad_pixels` for Voronoi alpha bleed (game-engine UV edge fix). |
| **BD Pack Channels** | Combine 4 IMAGE/MASK sources into RGBA channels. Each slot accepts IMAGE (auto-luminance OR alpha-channel for the alpha slot) or MASK. Auto-enables RGBA when alpha source is wired. |
| **BD Derive PBR Maps** | Heuristic roughness/metallic/AO/normal from image + depth (+ optional normal_map / silhouette_mask / aux_shading_alpha / aux_detail_texture / metallic_zone_mask). Outputs each map separately + packed ORM/ARM. `albedo_treatment=edge_pad` (default) prevents transparent-PNG halo bleed. |

**Recommended PBR pipeline** using existing depth/normal estimators (DepthAnythingV2 / Lotus / MiDaS / Metric3D — all installed via comfyui_controlnet_aux):

```
Source character image
  ├─→ ImageToMask (alpha) → silhouette_mask
  ├─→ DepthAnythingV2Preprocessor → depth (IMAGE)
  ├─→ Metric3D-NormalMapPreprocessor → normal_map (optional)
  ├─→ Skin shader alpha → ImageToMask → aux_shading_alpha
  └─→ Manga lineart ControlNet → aux_detail_texture

ALL into BD Derive PBR Maps with metallic_zone_mask=accessories_mask
  → albedo, normal, roughness, metallic, ao, packed_orm, packed_arm

→ BD Bulk Save with labels=albedo,normal,roughness,metallic,ao,orm,arm
   → context auto-pick → 7 files saved in one Run
```

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

### Save Context Workflow (PBR Maps Example)

Replace concat-string chains with one context + N save nodes:

```
BD Save Context
  template: "%project%/%character%/%character%_%name%_v%version%%suffix%"
  character: "letti"  name: "topwear"  version: "03"  project: "biloxi"
                                ↓
                          context_id (auto-picks if only one registered)
                                ↓
BD Derive PBR Maps             BD Bulk Save
  ├─ albedo ────→ input_1        labels = "albedo
  ├─ normal ────→ input_2                  normal
  ├─ roughness ─→ input_3                  roughness
  ├─ metallic ──→ input_4                  metallic
  └─ ao ────────→ input_5                  ao"
                                custom_vars = "subfolder=PBR"
                                ↓
                          Saves all 5 in ONE Run:
                          biloxi/letti/PBR/letti_topwear_v03_albedo_001.png
                          biloxi/letti/PBR/letti_topwear_v03_normal_001.png
                          biloxi/letti/PBR/letti_topwear_v03_roughness_001.png
                          ...
```

**Per-save custom_vars** layer on top of the context's vars. Empty `%var%` segments collapse cleanly (`a/%b%/c` with `b=""` → `a/c`). Undefined vars stay as `%var%` literals to surface typos.

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

## Node Counts at a Glance

90+ nodes across 8 categories:

- **Cache** — caching, save/load, save-context system (~16)
- **Mesh** — 3D processing, color sampling, simplification, OVoxel PBR baking (~24)
- **Blender** — Blender-based geometry tools (~10)
- **TRELLIS2** — TRELLIS2-specific shape/texture nodes (~9)
- **Character** — Qwen-Image character consistency (~4)
- **Prompt** — prompt iteration, filename templates, ForEach iteration (~5)
- **Segmentation** — SeeThrough/SAM3/MaskResolver pipeline + asset prep + PBR derivation (~16)
- **PBR / asset prep** — MaskFlatten, PackChannels, DerivePBR (in Segmentation)

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

[BrainDeadGuild.com](https://BrainDeadGuild.com) | [BrainDead.TV](https://BrainDead.TV) | [GitHub](https://github.com/BrainDeadGuild) | [discord](https://braindeadguild.com/discord)

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
- discord: [BrainDeadGuild](https://BrainDeadGuild.com/discord)
