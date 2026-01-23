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
├── mesh/           # 3D mesh processing
│   ├── types.py    # TrimeshInput/TrimeshOutput
│   ├── cache.py    # BD_CacheMesh
│   ├── sampling.py # BD_SampleVoxelgridColors, BD_SampleVoxelgridPBR
│   ├── transfer.py # BD_TransferVertexColors, etc.
│   ├── processing.py # BD_MeshRepair, BD_SmartDecimate
│   ├── export.py   # BD_ExportMeshWithColors
│   ├── simplify.py # BD_CuMeshSimplify
│   ├── unwrap.py   # BD_UVUnwrap
│   ├── ovoxel_bake.py # BD_OVoxelBake
│   └── bundle.py   # BD_PackBundle, BD_UnpackBundle, BD_CacheBundle
├── blender/        # Blender-based operations
├── cache/          # Caching nodes
├── character/      # Qwen character nodes
├── prompt/         # Prompt iteration
└── trellis2/       # TRELLIS2 specific
```
