# ComfyUI-BrainDead

<div align="center">

<img src="docs/images/banner.jpg" alt="ComfyUI-BrainDead" width="100%">

A comprehensive collection of ComfyUI custom nodes for **3D mesh processing**, **PBR baking**, **character segmentation**, **image-to-3D generation**, **GLSL shaders**, **smart caching**, and **game-asset pipelines** — built for professional production at Biloxi Studios.

[![BrainDeadGuild](https://img.shields.io/badge/BrainDeadGuild-Community-purple)](https://BrainDeadGuild.com/discord)
[![BrainDead.TV](https://img.shields.io/badge/BrainDead.TV-Lore-red)](https://BrainDead.TV)
[![ComfyUI V3](https://img.shields.io/badge/ComfyUI-V3%20API-blue)](https://docs.comfy.org/custom-nodes/v3_migration)
[![~110 nodes](https://img.shields.io/badge/nodes-~110-green)](nodes/)

</div>

---

## Workflow Templates

Nine ready-to-use workflows ship in [`example_workflows/`](example_workflows/) and appear in ComfyUI under **Workflow → Browse Templates → ComfyUI-BrainDead**.

<table>
<tr>
<td align="center" width="33%">
<img src="docs/images/workflow_pixal3d.jpg" width="100%" alt="Pixal3D Image to 3D"><br>
<b>Pixal3D Image to 3D</b><br>
<sub>Image → MoGe FOV → 3-stage pipeline → mesh + voxelgrid → PBR bake</sub>
</td>
<td align="center" width="33%">
<img src="docs/images/workflow_cubepart.jpg" width="100%" alt="CubePart Part Decomposition"><br>
<b>CubePart Part Decomposition</b><br>
<sub>Mesh → open-vocab part names → TRIMESH per part → preview + export</sub>
</td>
<td align="center" width="33%">
<img src="docs/images/workflow_sam3.jpg" width="100%" alt="SAM3 Parts Segmentation"><br>
<b>SAM3 Parts Segmentation</b><br>
<sub>Image → SAM3 multi-prompt → Parts Refine → Batch Edit → PSD export</sub>
</td>
</tr>
<tr>
<td align="center" width="33%">
<img src="docs/images/workflow_lotus2.jpg" width="100%" alt="Lotus-2 Depth & Normal"><br>
<b>Lotus-2 Depth & Normal</b><br>
<sub>FLUX-based diffusion depth — far higher quality than feedforward estimators</sub>
</td>
<td align="center" width="33%">
<img src="docs/images/workflow_glsl.jpg" width="100%" alt="GLSL Skin Tinting"><br>
<b>GLSL Skin Tinting</b><br>
<sub>4-output GPU shader: ILM / SR+Parts / Unity / Unreal skin-tone pipeline</sub>
</td>
<td align="center" width="33%">
<img src="docs/images/workflow_ovoxel.jpg" width="100%" alt="OVoxel PBR Bake"><br>
<b>OVoxel PBR Bake</b><br>
<sub>TRELLIS2 voxelgrid → simplify → UV unwrap → albedo/normal/roughness/metallic</sub>
</td>
</tr>
<tr>
<td align="center" width="33%">
<img src="docs/images/workflow_trellis2.jpg" width="100%" alt="TRELLIS2 Shape to Texture"><br>
<b>TRELLIS2 Shape to Texture</b><br>
<sub>Voxelgrid → textured mesh with full PBR material bake via BD OVoxel Bake</sub>
</td>
<td align="center" width="33%">
<img src="docs/images/workflow_facewrap.jpg" width="100%" alt="FaceWrap Pipeline"><br>
<b>FaceWrap Pipeline</b><br>
<sub>MediaPipe landmarks → SAM3 guided face masks → socket infill → UV export</sub>
</td>
<td align="center" width="33%">
<img src="docs/images/workflow_character.jpg" width="100%" alt="Character Consistency"><br>
<b>Character Consistency</b><br>
<sub>Qwen-Image multi-view edit with identity lock, prompt iteration, save context</sub>
</td>
</tr>
</table>

---

## About

AN internally-used node pack for Biloxi Studios designed to help with character, story, 3D, and audio generation for next-generation UGC game and TV pipelines. All nodes use the **ComfyUI V3 API** and appear under the `🧠BrainDead` category in the node browser.

---

## Installation

### ComfyUI Manager
Search **"BrainDead"** in ComfyUI Manager and install.

### Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/BizaNator/ComfyUI-BrainDead
cd ComfyUI-BrainDead
pip install -r requirements.txt
```

### Dependencies
`requirements.txt` is installed automatically by ComfyUI Manager. It covers:

| Package | Purpose |
|---------|---------|
| `moge` (git) | MoGe-2 monocular geometry for Pixal3D FOV estimation |
| `utils3d` (git, pinned) | Required by MoGe `.pt` submodule — PyPI version is incomplete |
| `pipeline` (git, pinned) | Required by MoGe |
| `natten` | Neighborhood Attention for Pixal3D NAF upsampler |

> **Pixal3D model weights** (~10 GB) are downloaded automatically on first run via HuggingFace.
> `cube_part` is vendored under `nodes/cubepart/vendor/` — no separate install needed.

---

## Node Reference

### Cache Nodes (`🧠BrainDead/Cache`)
Smart caching with **lazy evaluation** — upstream nodes are completely **skipped** when cache is valid.

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
3. Subsequent runs: load from cache — **skips upstream generation entirely**
4. Change the `seed` to force regeneration

**Save Context System** — replace ad-hoc concat-string filename chains:
```
BD Save Context
  template: "%project%/%character%/%character%_%name%_v%version%%suffix%"
  character: "letti"  name: "topwear"  version: "03"  project: "biloxi"
                            ↓ context_id
BD Bulk Save  (labels = "albedo\nnormal\nroughness\nmetallic\nao")
  → biloxi/letti/PBR/letti_topwear_v03_albedo_001.png
  → biloxi/letti/PBR/letti_topwear_v03_normal_001.png  ...
```

---

### Character Nodes (`🧠BrainDead/Character`)
Advanced tools for maintaining character consistency with Qwen-Image models.

| Node | Description |
|------|-------------|
| **BD Qwen Character Edit** | Enhanced image editing with character preservation |
| **BD Qwen T2I Custom** | Text-to-image with custom system prompts |
| **BD Qwen Multi-Image** | Multi-reference image editing |
| **BD Qwen Identity Lock** | Strong identity preservation with weighted references |

---

### Mesh Nodes (`🧠BrainDead/Mesh`)
3D mesh processing, color sampling, simplification, and PBR texture baking.

| Node | Description |
|------|-------------|
| **BD Load Mesh** | Load GLB/GLTF/OBJ/PLY/STL/FBX as TRIMESH. File picker + upload button, or `file_path` override |
| **BD Cache Mesh** | Cache TRIMESH objects as PLY files |
| **BD Sample Voxelgrid Colors** | Sample vertex colors from TRELLIS2 voxelgrid |
| **BD Sample Voxelgrid PBR** | Sample full PBR attributes from voxelgrid |
| **BD Apply Color Field** | Apply COLOR_FIELD to any mesh |
| **BD Transfer Vertex Colors** | BVH-based vertex color transfer between meshes |
| **BD Transfer Colors Pymeshlab** | Transfer colors using pymeshlab |
| **BD Mesh Repair** | Repair mesh topology (holes, normals, duplicates) |
| **BD Smart Decimate** | Edge-preserving decimation with pymeshlab |
| **BD Export Mesh With Colors** | Export mesh with vertex colors to GLB/PLY/OBJ. SaveContext-aware |
| **BD Trimesh → MESH** | Convert TRIMESH → ComfyUI native MESH (for built-in 3D save nodes) |
| **BD MESH → Trimesh** | Convert native MESH → TRIMESH (pull Hunyuan3D output into BD pipeline) |
| **BD CuMesh Simplify** | GPU-accelerated mesh simplification with color preservation |
| **BD CuMesh Quad Remesh** | GPU-accelerated quad remeshing |
| **BD MeshLib Fill Holes** | Fill mesh holes and open boundaries via pymeshlab |
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
| **BD Mesh Inspector** | Inspect mesh properties + interactive three.js viewer |
| **BD Mesh Preview (Thumbnails)** | Render TRIMESH_LIST or single TRIMESH to a labeled contact-sheet IMAGE. GPU EGL render |
| **BD Preview 3D** | Interactive three.js viewer for a mesh or CubePart `parts` (auto color-coded) |

**OVoxel Baking Pipeline:**
```
[TRELLIS2 Texture] → voxelgrid → [BD OVoxel Bake] → mesh + PBR textures
                                       OR
[TRELLIS2 Texture] → voxelgrid → [BD OVoxel Texture Bake]
                          ↑ mesh → [BD Blender Decimate] → [BD UV Unwrap] ↗
```

---

### Blender Nodes (`🧠BrainDead/Blender`)
Advanced mesh processing using Blender's geometry tools (requires Blender 5.0+, bundled in `lib/`).

| Node | Description |
|------|-------------|
| **BD Blender Decimate** | Full-featured decimation with edge preservation |
| **BD Blender Quad Decimate** | Quad-aware decimation: optional pre-quad-remesh, hole fill, quad OBJ guide |
| **BD Blender Edge Marking** | Detect and mark edges from colors/angles |
| **BD Blender Merge Planes** | Merge geometry within marked regions |
| **BD Blender Remesh** | Voxel/quad remeshing with Blender |
| **BD Blender Planar Normals** | Flat-panel stylized look for Unreal/Unity import |
| **BD Blender Cleanup** | Advanced mesh cleanup and repair |
| **BD Blender Vertex Colors** | Vertex color operations (bake, transfer) |
| **BD Blender Normals** | Normal fixing and recalculation |
| **BD Blender Export Mesh** | Export MESH_BUNDLE as GLB with material + vertex colors |

**Hard Edge Preservation Pipeline:**
```
[Input Mesh]
    ↓
[BD Planar Grouping] (straighten_boundaries=True)
    ↓
[BD Blender Edge Marking] (FROM_COLORS_AND_ANGLE)
    ↓
[BD Blender Merge Planes] (delimit_sharp=True)
    ↓
[Low-poly mesh with hard edges intact]
```

---

### Pixal3D Nodes (`🧠BrainDead/Pixal3D`)
Image-to-3D generation using TencentARC/Pixal3D. Downloads ~10 GB model weights on first run.

| Node | Description |
|------|-------------|
| **BD Pixal3D Preprocess** | Prepare image for Pixal3D: apply mask, crop to subject, estimate camera FOV via MoGe-2 (auto) or manual radians. Outputs PIXAL3D_INPUT bundle + 512×512 preview |
| **BD Pixal3D Image to 3D** | Generate 3D from PIXAL3D_INPUT. Three-stage pipeline: sparse structure → shape latent → texture latent. Outputs TRIMESH + TRELLIS2_VOXELGRID for PBR bake |

**Pixal3D pipeline:**
```
[Source image + mask]
  ↓
BD Pixal3D Preprocess  (fov_mode=auto_moge)
  ↓ pixal3d_input
BD Pixal3D Image to 3D  (pipeline_type=1024_cascade)
  ↓ mesh              ↓ voxelgrid
BD CuMesh Simplify    BD OVoxelBake → albedo, normal, roughness, metallic
```

---

### CubePart Nodes (`🧠BrainDead/CubePart`)
Open-vocabulary, part-controllable 3D decomposition using Roblox **CubePart**.

| Node | Description |
|------|-------------|
| **BD CubePart Segment** | TRIMESH + up to 8 part names → `parts` (TRIMESH_LIST, one mesh per part) + `combined` (color-coded preview) + `part_names` |
| **BD CubePart Get Part** | `parts` (TRIMESH_LIST) + `index` → single TRIMESH + `name` |

**Pre-download models:**
```bash
huggingface-cli download Roblox/cubepart --local-dir /srv/AI_Stuff/models/cubepart
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir /srv/AI_Stuff/models/LLM/Qwen3-VL-4B-Instruct
```

> **License:** CubePart code is MIT; model weights are under the **CUBE3D RESEARCH-ONLY RAIL-MS** license — review before commercial use.

---

### Segmentation Nodes (`🧠BrainDead/Segmentation`)
Character segmentation, parts pipeline, PBR map derivation, face masks, and asset prep.

**Parts pipeline — single-execution character processing via `PARTS_BUNDLE`:**

| Node | Description |
|------|-------------|
| **BD Parts Builder** | Crop/composite each segmented part. Wire SAM3 `per_prompt_masks` + labels OR Sapiens2 `labels`. Outputs PARTS_BUNDLE + image_batch + label_list + bbox_list |
| **BD Parts Refine** | IoU-based dedup of overlapping prompts → canonical tags |
| **BD Parts Batch Edit (Qwen)** | Internal-loop Qwen Image Edit per part. Modes: `flatten_redraw` + `true_inpaint` |
| **BD Parts Compose** | Flatten bundle to single RGBA at chosen `output_size` |
| **BD Parts Export** | Save per-tag RGBA PNGs + depth PNGs + composite PNG + layered PSD |

**SAM3 multi-prompt:**

| Node | Description |
|------|-------------|
| **BD SAM3 Multi-Prompt** | One node replaces 12-chain SAM3Segment. **Standalone** — auto-downloads `Comfy-Org/sam3.1`. Vote/weighted_vote modes, positive + negative prompts, color filter, silhouette enforcement |
| **BD Mask Resolver** | Priority-based pixel-level separation (skin/clothes/accessories) |
| **BD Human Parser Mask Clean** | Per-class morphological cleanup + min-area filter |
| **BD Human Parser Mask Split** | Split parse map into per-region MASK outputs |
| **BD Human Parser Named Mask** | Extract a single label as MASK |
| **BD Human Parser Preview** | Colorized RGB visualization |

**MediaPipe face masks:**

| Node | Description |
|------|-------------|
| **BD MP Face Mask** | Landmark-precise face region masks (face_oval, skin, eyes, brows, lips, nose, irises, ears, forehead, hair) |
| **BD MP Face Export** | Write landmark JSON (478 pts) + reference RGBA mask for Blender face-plate UV pipeline |
| **BD MP Save / Load Face Data** | Persist all region masks + head_mask + image to `.mpface.npz` / `.json` |
| **BD MP Face Refine** | Refine MediaPipe masks with SAM3 batch (IoU match → intersect) |
| **BD MP SAM3 Face Segment** | MediaPipe-guided SAM3: localise each feature then SAM3-segment it. Pixel-accurate masks in one node |
| **BD MP Mouth Parts** | Separate mouth render → lips / teeth / tongue, packed RGBA for Unreal viseme atlas |
| **BD Face Socket Infill** | MediaPipe-based face socket creator for 2D animation flipbook textures |

**Luma / lighting tools:**

| Node | Description |
|------|-------------|
| **BD Luma Stats** | Compute luma statistics: min, max, median, mean, recommended_outer_band |
| **BD Center Median Luma** | Shift image luma so median lands at 0.5 |
| **BD Normalize Luma** | Auto-rescale luminance to target range. Mask-aware percentile clipping |
| **BD Depth To Shadow Map** | Depth → shadow/shading map via top-down N·L shading + cavity AO |

**Asset prep / game textures:**

| Node | Description |
|------|-------------|
| **BD Mask Flatten** | Flatten RGB+alpha onto chosen background. Channel-routing modes. `edge_pad_pixels` for UV edge fix |
| **BD Pack Channels** | Combine 4 IMAGE/MASK sources into RGBA channels |
| **BD Derive PBR Maps** | Heuristic roughness/metallic/AO/normal from image + depth. Outputs each map + packed ORM/ARM |
| **BD Draw Rect** | Flat-color rectangle with configurable rounded corners and edge feather |
| **BD Draw Ellipse** | Ellipse/circle with flat color, anisotropic radii and rotation |
| **BD Mask Correlate** | IoU-based matching of SAM3 candidate masks to up to 8 named target slots |
| **BD Mask Color Fill** | Fill up to four mask regions with solid colors on a composited canvas |

**Recommended character parts pipeline:**
```
[Source IMAGE]
   ├─→ [VLM, e.g. ComfyUI-QwenVL] → newline-separated part list
   ├─→ BD Lotus-2 Predict → depth_image
   └─→ BD SAM3 Multi-Prompt
         ↓ per_prompt_masks, combined_mask
       BD Parts Refine (iou_threshold=0.7)
         ↓
       BD Parts Builder (depth_image wired)
         ↓ parts (PARTS_BUNDLE)
       BD Parts Batch Edit (Qwen) — flatten_redraw
         ↓ parts (rebuilt)
       BD Parts Export — save_psd=true, composite_size=4096
         → per-tag PNGs + layered PSD
```

---

### Depth Nodes (`🧠BrainDead/Depth`)
SOTA monocular geometry prediction via diffusion.

| Node | Description |
|------|-------------|
| **BD Lotus-2 Model Loader** | Load FLUX.1-dev + Lotus-2 depth/normal LoRA + LCM bridge. Module-level cache for instant reuse |
| **BD Lotus-2 Predict** | Run Lotus-2 on an image. Outputs map (IMAGE), raw_linear, colorized_preview |

---

### Prompt Nodes (`🧠BrainDead/Prompt`)
Iterate through multiple prompts with automatic filename generation.

| Node | Description |
|------|-------------|
| **BD Prompt Iterator** | Basic prompt cycling with filename generation |
| **BD Prompt Iterator (Advanced)** | Templates, suffix lists, seed modes, ping-pong |
| **BD Prompt Iterator (Dynamic)** | Up to 20 connected prompt inputs |
| **BD Filename Template** | Standalone STRING resolver for `%var%` templates |
| **BD For Each Run** | Multi-Run iterator over up to 16 typed inputs + parallel labels |

---

## Usage Examples

### Cache Expensive Generation
```
[Expensive Node] → [BD Cache Image] → [Next Node]
                     seed: 42
```
First run generates and caches. Subsequent runs skip generation entirely.

### Save Context + Bulk PBR Save
```
BD Save Context
  template: "%project%/%character%/%character%_%name%_v%version%%suffix%"

BD Derive PBR Maps → BD Bulk Save
  labels = "albedo\nnormal\nroughness\nmetallic\nao"
  custom_vars = "subfolder=PBR"

→ biloxi/letti/PBR/letti_topwear_v03_albedo_001.png  (5 files, one Run)
```

### Character Consistency Workflow
```
[BD Prompt Iterator (Advanced)]
  prompts: "front view\nleft profile\nright profile\nback view"
  suffixes: "_front\n_left\n_right\n_back"
  ↓
[BD Qwen Character Edit]
  ↓
[BD Save File] (context_id → Save Context)
```

### Multi-Reference Identity Lock
```
[Load Image 1 (identity)] ─┐
[Load Image 2 (style)]    ─┼→ [BD Qwen Identity Lock] → [Generate]
[Load Image 3 (pose)]     ─┘     image1_strength: 1.5
                                  image1_role: character
```

---

## Node Counts at a Glance

~110 nodes across 10 categories:

| Category | Nodes |
|----------|-------|
| **Cache** — caching, save/load, save-context system | ~16 |
| **Mesh** — 3D processing, color sampling, OVoxel PBR baking | ~26 |
| **Blender** — Blender-based geometry tools | ~12 |
| **TRELLIS2** — TRELLIS2-specific shape/texture nodes | ~9 |
| **Pixal3D** — image-to-3D generation (Pixal3D + MoGe FOV) | ~2 |
| **CubePart** — open-vocabulary part decomposition (Roblox) | ~2 |
| **Character** — Qwen-Image character consistency | ~4 |
| **Prompt** — prompt iteration, filename templates, ForEach | ~5 |
| **Segmentation** — SAM3 + Parts + Face + Luma + PBR asset prep | ~25 |
| **Depth** — Lotus-2 (FLUX-based diffusion depth/normal) | ~2 |

---

## Directory Structure

```
ComfyUI-BrainDead/
├── nodes/
│   ├── mesh/          # 3D mesh processing
│   ├── blender/       # Blender-based operations
│   ├── cache/         # Caching and save nodes
│   ├── character/     # Qwen character consistency
│   ├── pixal3d/       # Pixal3D image-to-3D
│   ├── cubepart/      # Roblox CubePart (vendored)
│   ├── trellis2/      # TRELLIS2 shape/texture
│   └── segmentation/  # SAM3, Parts, Face, Luma, PBR
├── example_workflows/ # 9 ready-to-use workflow templates
├── docs/images/       # Banner + workflow thumbnails
├── tools/             # audit_workflows.py, etc.
├── lib/               # Bundled Blender 5.0
└── requirements.txt   # Auto-installed by ComfyUI Manager
```

```
output/
├── BrainDead_Cache/   # Cache nodes save here (clearable)
└── saved_file.png     # BD Save File saves to main output/
```

---

<div align="center">

## 🧠 BrainDeadGuild

**Professional AI Tools for Creative Production**

Created by **BizaNator**

[BrainDeadGuild.com](https://BrainDeadGuild.com) | [BrainDead.TV](https://BrainDead.TV) | [GitHub](https://github.com/BrainDeadGuild) | [Discord](https://braindeadguild.com/discord)

*Building tools for the BrainDeadGuild community*

---

**A Biloxi Studios Inc. Production**

© 2025 Biloxi Studios Inc. — All Rights Reserved

</div>

## Support

- GitHub Issues: [Report bugs](https://github.com/BizaNator/ComfyUI-BrainDead/issues)
- Discord: [BrainDeadGuild](https://BrainDeadGuild.com/discord)
