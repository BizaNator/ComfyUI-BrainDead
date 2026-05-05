"""
BrainDead depth nodes — high-quality monocular depth + normal estimation.

Backends:
- Lotus-2 (jingheya/Lotus-2): FLUX.1-dev based, 2-stage (core_predictor + detail_sharpener)
  with LCM bridge module. Supports depth and normal.
"""

from .lotus2 import (
    LOTUS2_NODES,
    LOTUS2_DISPLAY_NAMES,
    LOTUS2_V3_NODES,
)

DEPTH_V3_NODES = [
    *LOTUS2_V3_NODES,
]

NODE_CLASS_MAPPINGS = {
    **LOTUS2_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **LOTUS2_DISPLAY_NAMES,
}
