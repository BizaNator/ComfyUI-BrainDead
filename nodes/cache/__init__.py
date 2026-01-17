"""
BrainDead Cache Nodes for ComfyUI
Created by BizaNator for BrainDeadGuild.com
A Biloxi Studios Inc. Production

Universal checkpoint/cache nodes that work with ANY data type.
Generate once, reuse forever - skip expensive regeneration on subsequent runs.

Features:
- Lazy evaluation: upstream nodes are SKIPPED when cache is valid
- Seed-based cache keys: change seed to force regeneration
- Type-specific serialization: PNG for images, WAV for audio, PLY for meshes
"""

from .base import BaseCacheNode
from .core import (
    CACHE_CORE_NODES,
    CACHE_CORE_DISPLAY_NAMES,
)
from .file_ops import (
    FILE_OPS_NODES,
    FILE_OPS_DISPLAY_NAMES,
)
from .workflow import (
    WORKFLOW_NODES,
    WORKFLOW_DISPLAY_NAMES,
)

# Combine all cache nodes
NODE_CLASS_MAPPINGS = {
    **CACHE_CORE_NODES,
    **FILE_OPS_NODES,
    **WORKFLOW_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **CACHE_CORE_DISPLAY_NAMES,
    **FILE_OPS_DISPLAY_NAMES,
    **WORKFLOW_DISPLAY_NAMES,
}

# Export lists for external use
CACHE_NODES = list(NODE_CLASS_MAPPINGS.values())

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "CACHE_NODES",
    "BaseCacheNode",
]
