"""
BrainDead GLSL nodes — shader-driven processing.

CATEGORY: 🧠BrainDead/GLSL
"""

from .glsl_batch import (
    GLSL_BATCH_NODES,
    GLSL_BATCH_DISPLAY_NAMES,
    GLSL_BATCH_V3_NODES,
)


GLSL_V3_NODES = [
    *GLSL_BATCH_V3_NODES,
]

NODE_CLASS_MAPPINGS = {
    **GLSL_BATCH_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **GLSL_BATCH_DISPLAY_NAMES,
}
