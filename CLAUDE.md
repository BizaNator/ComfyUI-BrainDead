# ComfyUI-BrainDead Development Rules

## Category Naming Convention

**IMPORTANT**: All BrainDead nodes MUST use the 🧠 emoji prefix in their category names.

```python
# CORRECT
category="🧠BrainDead/Mesh"
category="🧠BrainDead/Cache"
category="🧠BrainDead/Blender"
category="🧠BrainDead/Character"
category="🧠BrainDead/Prompt"
category="🧠BrainDead/TRELLIS2"

# WRONG - will show under different menu location
category="BrainDead/Mesh"
```

This ensures all BrainDead nodes appear together in the ComfyUI node browser under the "🧠BrainDead" category.

## V3 API Pattern

All nodes use the ComfyUI V3 API with `io.ComfyNode` base class:

```python
from comfy_api.latest import io

class BD_MyNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_MyNode",
            display_name="BD My Node",
            category="🧠BrainDead/Category",
            description="...",
            inputs=[...],
            outputs=[...],
        )

    @classmethod
    def execute(cls, ...) -> io.NodeOutput:
        return io.NodeOutput(...)
```

## Module Export Pattern

Each module (mesh, cache, blender, etc.) exports:
- `*_NODES` - V1 compatibility dict: `{"NodeName": NodeClass}`
- `*_DISPLAY_NAMES` - Display name dict: `{"NodeName": "Display Name"}`
- `*_V3_NODES` - V3 node list: `[NodeClass, ...]`

## Custom Types

For TRIMESH input/output, use the helpers from `nodes/mesh/types.py`:
```python
from .types import TrimeshInput, TrimeshOutput

inputs=[
    TrimeshInput("mesh"),
    TrimeshInput("mesh", optional=True),
]
outputs=[
    TrimeshOutput(display_name="mesh"),
]
```

For other custom types (like TRELLIS2_VOXELGRID), use `io.Custom("TYPE").Input()`:
```python
# CORRECT
io.Custom("TRELLIS2_VOXELGRID").Input("voxelgrid")
io.Custom("TRELLIS2_VOXELGRID").Input("voxelgrid", optional=True)

# WRONG - will silently fail to register node!
io.Custom.Input("voxelgrid", "TRELLIS2_VOXELGRID")
```

## Standard Output Node Pattern

All output nodes that save files to disk MUST use the standard `filename`/`name_prefix` pattern:

```python
inputs=[
    io.String.Input("filename", default="mesh_output", tooltip="Base name for output files"),
    io.String.Input("name_prefix", default="", optional=True,
                    tooltip="Prepended to filename. Supports subdirs (e.g., 'Project/Name')"),
    io.Boolean.Input("auto_increment", default=True, optional=True,
                    tooltip="Auto-increment filename to avoid overwriting"),
]
```

**Path Resolution Logic:**
```python
import folder_paths
from glob import glob

output_base = folder_paths.get_output_directory()

# Concatenate name_prefix + filename
full_name = f"{name_prefix}_{filename}" if name_prefix else filename

# Handle subdirectories in full_name
full_name = full_name.replace('\\', '/')
if '/' in full_name:
    parts = full_name.rsplit('/', 1)
    subdir, base_filename = parts
    output_dir = os.path.join(output_base, subdir)
else:
    output_dir = output_base
    base_filename = full_name

os.makedirs(output_dir, exist_ok=True)

# Auto-increment pattern
if auto_increment:
    pattern = os.path.join(output_dir, f"{base_filename}_*.{format}")
    existing = glob(pattern)
    # ... find max number, increment
    final_filename = f"{base_filename}_{next_num:03d}.{format}"
else:
    final_filename = f"{base_filename}.{format}"
```

**Example paths:**
- `filename="mesh"` → `output/mesh_001.glb`
- `filename="mesh", name_prefix="Project"` → `output/Project_mesh_001.glb`
- `filename="mesh", name_prefix="Project/Character"` → `output/Project/Character_mesh_001.glb`

This pattern is used by:
- `BD_ExportMeshWithColors` (export.py)

## Blender Integration

Nodes using Blender inherit from `BlenderNodeMixin`:
```python
from ..blender.base import BlenderNodeMixin

class BD_MyBlenderNode(BlenderNodeMixin, io.ComfyNode):
    # Use cls._run_blender_script(), cls._check_blender(), etc.
```

Bundled Blender location: `lib/blender/blender-5.0.1-linux-x64/blender`

## File Organization

```
nodes/
├── mesh/                   # 3D mesh processing
│   ├── types.py            # TrimeshInput/TrimeshOutput, MeshBundleInput/Output, etc.
│   ├── cache.py            # BD_CacheMesh
│   ├── sampling.py         # BD_SampleVoxelgridColors, BD_SampleVoxelgridPBR
│   ├── transfer.py         # BD_TransferVertexColors, etc.
│   ├── processing.py       # BD_MeshRepair, BD_SmartDecimate
│   ├── export.py           # BD_ExportMeshWithColors
│   ├── simplify.py         # BD_CuMeshSimplify
│   ├── unwrap.py           # BD_UVUnwrap
│   ├── grouping.py         # BD_PlanarGrouping
│   ├── edge_utils.py       # BD_CombineEdgeMetadata
│   ├── color_field.py      # BD_ApplyColorField
│   ├── ovoxel_bake.py      # BD_OVoxelBake (all-in-one PBR bake)
│   ├── ovoxel_texture_bake.py # BD_OVoxelTextureBake (bake-only)
│   ├── ovoxel_convert.py   # BD_MeshToOVoxel
│   ├── ovoxel_io.py        # BD_ExportOVoxel, BD_LoadOVoxel
│   ├── fix_normals.py      # BD_FixNormals
│   ├── bundle.py           # BD_PackBundle, BD_UnpackBundle, BD_CacheBundle
│   └── inspector.py        # BD_MeshInspector
├── blender/                # Blender-based operations
│   ├── base.py             # BlenderNodeMixin
│   ├── decimate.py         # BD_BlenderDecimate
│   ├── export_mesh.py      # BD_BlenderExportMesh
│   └── addon_nodes.py      # Edge marking, merge planes, remesh, etc.
├── cache/                  # Caching nodes
├── character/              # Qwen character nodes
├── prompt/                 # Prompt iteration
└── trellis2/               # TRELLIS2 specific
    ├── shape.py            # BD_Trellis2GenerateShape
    ├── texture.py          # BD_Trellis2ShapeToTexturedMesh, BD_Trellis2Retexture
    └── utils/helpers.py    # fix_normals_outward, etc.
```

## Workflow Template Styling Convention

**ALL `example_workflows/*.json` templates MUST follow a consistent node-title style.**
Templates are the first thing users see — inconsistent titling looks unprofessional and
makes the step order ambiguous. When you regenerate or hand-edit a template, preserve
these rules exactly (do not drop titles when rebuilding `widgets_values`):

1. **Every pipeline node gets a circled-number prefix** marking its step order, following
   the visual flow left→right / top→bottom:
   `①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳` (Unicode `chr(0x2460 + step - 1)`).
   - Format: `"① Load Image"`, `"② BD Remove Background (SAM3 + matting)"`.
   - Add a short parenthetical when the same node type appears twice
     (e.g. `"③ Lotus-2 Model Loader (depth)"` vs `"⑤ Lotus-2 Model Loader (normal)"`).

2. **Pure plumbing/adapter nodes** (e.g. `MaskToImage`, format converters) may use a
   plain descriptive title with **no number** (e.g. `"Mask → Image"`) so the numbering
   tracks the meaningful pipeline steps, not glue.

3. **The About note** is a `MarkdownNote` titled:
   `"ℹ️ About — 🧠 BrainDead <Pack Name>"`
   The 🧠 brain emoji (the BrainDead brand mark, matching the `🧠BrainDead` category) is
   **required** in the About title. Its body starts with `## <Pack Name>` and lists the
   outputs + tips.

4. **Filenames:** `BD-<snake_case_name>.json` + matching `BD-<snake_case_name>.jpg`
   thumbnail (`.jpg` only — `.png` is not served). No spaces.

5. When you only change widget values or wiring, **keep the existing `title` fields**.
   The template builder must carry titles through; a rebuild that emits untitled nodes is
   a regression.

## Template → API → Pipeline Sync (automation — keep these in lockstep)

ComfyUI-BrainDead templates flow through **four artifacts**. The UI-graph `.json` is the **single
source of truth**; everything else is *generated* from it. When you change a template, regenerate
the whole chain — never hand-edit the downstream artifacts.

```
tools/build_<name>.py        # builder: fetches live /object_info, emits the UI graph
        │  python3 tools/build_<name>.py
        ▼
example_workflows/BD-<name>.json        # ① UI graph (source of truth; styled per the convention above)
        │  python3 tools/export_api.py example_workflows/BD-<name>.json
        ▼
api/BD-<name>.api.json                  # ② frozen API/prompt export (node_id → class_type/inputs)
        │     NOTE: lives in api/, NOT example_workflows/ — ComfyUI scans example_workflows/ for
        │     UI templates and would try to load an API json as a graph → empty-canvas error.
        │  cp → studio Workflows share (COB naming)
        ▼
/mnt/tank/Studio/Brains/Workflows/COB_<cat>_<Name>_v<NN>_API.json   # ③ registered studio executable
        │  tools/regen_all_thumbnails.py   (add a CONFIGS + GALLERY_SLUG entry first)
        ▼
example_workflows/BD-<name>.jpg         # ④ branded card thumbnail (jpg only)
```

### The builders (`tools/`)
- `build_<name>.py` — emit a template's UI graph. **They query the LIVE server's `/object_info`**,
  so the server must be running with the current nodes loaded before you build. They handle widget
  alignment incl. the hidden `control_after_generate` companion after `seed`/`noise_seed`
  (see [[feedback_append_node_inputs_bottom]] / `feedback_comfyui_v3_widget_order`). Validate widget
  counts vs object_info after building.
- `export_api.py <template.json>` — regenerate `<name>.api.json` via `run_workflow.py`'s
  `workflow_to_api` (also object_info-driven). Run this after EVERY template change.
- `make_thumbnail.py` / `regen_all_thumbnails.py` — the **only** way to make a thumbnail. Add a
  `CONFIGS` entry (title/subtitle/bullets/chips, optional `background` screenshot under
  `example_workflows/screenshots/`) + a `GALLERY_SLUG` entry, then run `regen_all_thumbnails.py`.
  NEVER drop a raw screenshot in as the `.jpg` — it skips the branded card (wordmark, footer,
  BizaNator logo). The card can use a screenshot as a faded `background`.

### Workflow launchers (use these — don't write a `run_<x>.py` per workflow)
- `run_workflow.py` (`/opt/comfyui/`) — runs any workflow JSON as-is → outputs (no param injection).
- **`run_bd.py`** — generic launcher: `--workflow <name|path>` + `--image` (uploads, sets LoadImage) +
  repeatable `--set "NodeType.input=value"` (or `id.input=value`) + collects saved `/history` images →
  `--output-dir` / character folder. Works for ANY template that has Save nodes.
- `run_unreal_fbx.py --image --name --part [--decimation] [--detail-strength]` — specialized per-part
  Trellis→FBX dispatcher; submits the **COB** export, copies fbx+maps+thumbnail into
  `Characters/<name>/models/<part>/unreal/`. Source of truth stays in this repo.
- **`run_part_to_3d.py`** — SAM3→Trellis chain: `--image --prompts "tank top" --name --part` →
  `BD-isolate_part` (SAM3 isolate) → `run_unreal_fbx.py`.
- Studio convention: `COB_<category>_<name>_v<NN>_API.json` in `/mnt/tank/Studio/Brains/Workflows/`,
  run via `/opt/comfyui/run_workflow.py`. Studio skill: `/mnt/tank/Studio/Brains/Skills/Pipeline/`.

### Deploy rule (CRITICAL)
Deploy dev→stable with **`git pull --rebase` ONLY**. Never `cp`, and **do not use
`regen_all_thumbnails.py --deploy`** — both write to the stable tree and then block the next
`git pull` with "untracked/modified files." Workflow: commit + `git push origin main` in dev →
`git pull --rebase origin main` on stable. See `feedback_braindead_deploy_pull_only` in memory.

### Regeneration checklist (run after any template/builder change)
1. (server up) `python3 tools/build_<name>.py` → rebuild the UI `.json`
2. `python3 tools/export_api.py example_workflows/BD-<name>.json` → refresh the `.api.json`
3. `cp api/BD-<name>.api.json <studio Workflows>/COB_<...>_API.json` (if registered)
4. `python3 tools/regen_all_thumbnails.py` (NO `--deploy`) → refresh the `.jpg`
5. commit + push (dev) → `git pull --rebase` (stable) → restart `comfyui-stable` if nodes changed
