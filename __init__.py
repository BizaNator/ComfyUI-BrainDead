"""
ComfyUI-BrainDead
Created by BizaNator for BrainDeadGuild.com
A Biloxi Studios Inc. Production

A comprehensive collection of ComfyUI custom nodes for:
- Cache: Smart caching with lazy evaluation to skip expensive generation
- Mesh: 3D mesh processing, color sampling, and export
- Blender: High-quality mesh operations using Blender headlessly
- TRELLIS2: TRELLIS2-specific caching and conditioning tools
- Character: Qwen-Image character consistency tools
- Prompt: Prompt iteration for batch processing

https://github.com/BizaNator/ComfyUI-BrainDead
"""

__version__ = "2.0.0"

from comfy_api.latest import io, ComfyExtension

# =============================================================================
# V3 Node Imports
# =============================================================================

from .nodes.cache import CACHE_V3_NODES
from .nodes.mesh import MESH_V3_NODES
from .nodes.blender import BLENDER_V3_NODES
from .nodes.trellis2 import TRELLIS2_V3_NODES
from .nodes.character import CHARACTER_V3_NODES
from .nodes.prompt import PROMPT_V3_NODES

# =============================================================================
# V3 Extension Entry Point
# =============================================================================

class BrainDeadExtension(ComfyExtension):
    """ComfyUI-BrainDead V3 Extension with cache, mesh, blender, TRELLIS2, character, and prompt nodes."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """Return all BrainDead nodes for V3 registration."""
        return [
            *CACHE_V3_NODES,
            *MESH_V3_NODES,
            *BLENDER_V3_NODES,
            *TRELLIS2_V3_NODES,
            *CHARACTER_V3_NODES,
            *PROMPT_V3_NODES,
        ]


async def comfy_entrypoint() -> BrainDeadExtension:
    """V3 entry point for ComfyUI extension system."""
    return BrainDeadExtension()


# =============================================================================
# V1 Backward Compatibility
# =============================================================================

# Import V1 node mappings for backward compatibility
from .nodes.cache import NODE_CLASS_MAPPINGS as CACHE_NODES, NODE_DISPLAY_NAME_MAPPINGS as CACHE_DISPLAY
from .nodes.mesh import NODE_CLASS_MAPPINGS as MESH_NODES, NODE_DISPLAY_NAME_MAPPINGS as MESH_DISPLAY
from .nodes.blender import BLENDER_NODES, BLENDER_DISPLAY_NAMES
from .nodes.trellis2 import NODE_CLASS_MAPPINGS as TRELLIS2_NODES, NODE_DISPLAY_NAME_MAPPINGS as TRELLIS2_DISPLAY
from .nodes.character import NODE_CLASS_MAPPINGS as CHARACTER_NODES, NODE_DISPLAY_NAME_MAPPINGS as CHARACTER_DISPLAY
from .nodes.prompt import NODE_CLASS_MAPPINGS as PROMPT_NODES, NODE_DISPLAY_NAME_MAPPINGS as PROMPT_DISPLAY

# Aggregate all node mappings for V1 registration
NODE_CLASS_MAPPINGS = {
    **CACHE_NODES,
    **MESH_NODES,
    **BLENDER_NODES,
    **TRELLIS2_NODES,
    **CHARACTER_NODES,
    **PROMPT_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **CACHE_DISPLAY,
    **MESH_DISPLAY,
    **BLENDER_DISPLAY_NAMES,
    **TRELLIS2_DISPLAY,
    **CHARACTER_DISPLAY,
    **PROMPT_DISPLAY,
}

__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
    'comfy_entrypoint',
    'BrainDeadExtension',
]

# =============================================================================
# Startup Message
# =============================================================================

print("=" * 60)
print(f"ComfyUI-BrainDead v{__version__} (V3 API)")
print("Created by BizaNator for BrainDeadGuild.com")
print(f"Loaded {len(NODE_CLASS_MAPPINGS)} custom nodes:")
print(f"  - Cache: {len(CACHE_NODES)} nodes")
print(f"  - Mesh: {len(MESH_NODES)} nodes")
print(f"  - Blender: {len(BLENDER_NODES)} nodes")
print(f"  - TRELLIS2: {len(TRELLIS2_NODES)} nodes")
print(f"  - Character: {len(CHARACTER_NODES)} nodes")
print(f"  - Prompt: {len(PROMPT_NODES)} nodes")
print("=" * 60)
