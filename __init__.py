"""
ComfyUI-BrainDead
Created by BizaNator for BrainDeadGuild.com
A Biloxi Studios Inc. Production

A comprehensive collection of ComfyUI custom nodes for:
- Cache: Smart caching with lazy evaluation to skip expensive generation
- Character: Qwen-Image character consistency tools
- Prompt: Prompt iteration for batch processing

https://github.com/BizaNator/ComfyUI-BrainDead
"""

__version__ = "1.0.0"

# Import node mappings from each submodule
from .nodes.cache import NODE_CLASS_MAPPINGS as CACHE_NODES, NODE_DISPLAY_NAME_MAPPINGS as CACHE_DISPLAY
from .nodes.character import NODE_CLASS_MAPPINGS as CHARACTER_NODES, NODE_DISPLAY_NAME_MAPPINGS as CHARACTER_DISPLAY
from .nodes.prompt import NODE_CLASS_MAPPINGS as PROMPT_NODES, NODE_DISPLAY_NAME_MAPPINGS as PROMPT_DISPLAY

# Aggregate all node mappings
NODE_CLASS_MAPPINGS = {
    **CACHE_NODES,
    **CHARACTER_NODES,
    **PROMPT_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **CACHE_DISPLAY,
    **CHARACTER_DISPLAY,
    **PROMPT_DISPLAY,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Startup message
print("=" * 60)
print("ComfyUI-BrainDead v1.0.0")
print("Created by BizaNator for BrainDeadGuild.com")
print(f"Loaded {len(NODE_CLASS_MAPPINGS)} custom nodes:")
print(f"  - Cache: {len(CACHE_NODES)} nodes")
print(f"  - Character: {len(CHARACTER_NODES)} nodes")
print(f"  - Prompt: {len(PROMPT_NODES)} nodes")
print("=" * 60)
