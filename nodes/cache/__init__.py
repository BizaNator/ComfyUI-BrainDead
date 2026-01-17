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
    CACHE_CORE_V3_NODES,
)
from .file_ops import (
    FILE_OPS_NODES,
    FILE_OPS_DISPLAY_NAMES,
    FILE_OPS_V3_NODES,
)
from .workflow import (
    WORKFLOW_NODES,
    WORKFLOW_DISPLAY_NAMES,
    WORKFLOW_V3_NODES,
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

# V3 node list for extension
CACHE_V3_NODES = [
    *CACHE_CORE_V3_NODES,
    *FILE_OPS_V3_NODES,
    *WORKFLOW_V3_NODES,
]

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "CACHE_NODES",
    "CACHE_V3_NODES",
    "BaseCacheNode",
]
