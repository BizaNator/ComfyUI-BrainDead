"""
BrainDead TRELLIS2 nodes for ComfyUI.

CATEGORY: BrainDead/TRELLIS2

Inference Nodes (non-isolated, run in main process):
- BD_Trellis2GetConditioning - DinoV3 feature extraction (includes model config)
- BD_Trellis2ImageToShape - Shape generation
- BD_Trellis2ShapeToTexturedMesh - Texture generation (with dual conditioning!)

Cache Nodes:
- BD_CacheTrellis2Conditioning - Cache conditioning to skip preprocessing
- BD_CacheTrellis2Shape - Cache shape + mesh (saves 30-60s per run)
- BD_CacheTrellis2Texture - Cache texture outputs

Info Nodes:
- BD_Trellis2DualConditioning - Info node for dual conditioning workflow
"""

from .cache import (
    TRELLIS2_CACHE_NODES,
    TRELLIS2_CACHE_DISPLAY_NAMES,
    TRELLIS2_CACHE_V3_NODES,
)
from .info import (
    TRELLIS2_INFO_NODES,
    TRELLIS2_INFO_DISPLAY_NAMES,
    TRELLIS2_INFO_V3_NODES,
)
from .conditioning import (
    TRELLIS2_CONDITIONING_NODES,
    TRELLIS2_CONDITIONING_DISPLAY_NAMES,
    TRELLIS2_CONDITIONING_V3_NODES,
)
from .shape import (
    TRELLIS2_SHAPE_NODES,
    TRELLIS2_SHAPE_DISPLAY_NAMES,
    TRELLIS2_SHAPE_V3_NODES,
)
from .texture import (
    TRELLIS2_TEXTURE_NODES,
    TRELLIS2_TEXTURE_DISPLAY_NAMES,
    TRELLIS2_TEXTURE_V3_NODES,
)

# Combine all TRELLIS2 nodes
NODE_CLASS_MAPPINGS = {
    **TRELLIS2_CONDITIONING_NODES,
    **TRELLIS2_SHAPE_NODES,
    **TRELLIS2_TEXTURE_NODES,
    **TRELLIS2_CACHE_NODES,
    **TRELLIS2_INFO_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **TRELLIS2_CONDITIONING_DISPLAY_NAMES,
    **TRELLIS2_SHAPE_DISPLAY_NAMES,
    **TRELLIS2_TEXTURE_DISPLAY_NAMES,
    **TRELLIS2_CACHE_DISPLAY_NAMES,
    **TRELLIS2_INFO_DISPLAY_NAMES,
}

# Export lists for external use
TRELLIS2_NODES = list(NODE_CLASS_MAPPINGS.values())

# V3 node list for extension
TRELLIS2_V3_NODES = [
    *TRELLIS2_CONDITIONING_V3_NODES,
    *TRELLIS2_SHAPE_V3_NODES,
    *TRELLIS2_TEXTURE_V3_NODES,
    *TRELLIS2_CACHE_V3_NODES,
    *TRELLIS2_INFO_V3_NODES,
]

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "TRELLIS2_NODES",
    "TRELLIS2_V3_NODES",
]
