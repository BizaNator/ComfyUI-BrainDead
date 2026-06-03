"""
BrainDead Pixal3D nodes for ComfyUI.

CATEGORY: 🧠BrainDead/Pixal3D

Nodes:
- BD_Pixal3DPreprocess  - Image + mask → PIXAL3D_INPUT (preprocessing + camera estimation)
- BD_Pixal3DImageTo3D   - PIXAL3D_INPUT → TRIMESH + GLB file (generation + PBR bake)

Workflow:
  image → BD_Pixal3DPreprocess → BD_Pixal3DImageTo3D
              (+ optional mask)         → mesh (TRIMESH, feeds BD mesh nodes)
                                        → glb_path
"""

from .preprocess import (
    PIXAL3D_PREPROCESS_NODES,
    PIXAL3D_PREPROCESS_DISPLAY_NAMES,
    PIXAL3D_PREPROCESS_V3_NODES,
)
from .generate import (
    PIXAL3D_GENERATE_NODES,
    PIXAL3D_GENERATE_DISPLAY_NAMES,
    PIXAL3D_GENERATE_V3_NODES,
)

NODE_CLASS_MAPPINGS = {
    **PIXAL3D_PREPROCESS_NODES,
    **PIXAL3D_GENERATE_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **PIXAL3D_PREPROCESS_DISPLAY_NAMES,
    **PIXAL3D_GENERATE_DISPLAY_NAMES,
}

PIXAL3D_NODES = list(NODE_CLASS_MAPPINGS.values())

PIXAL3D_V3_NODES = [
    *PIXAL3D_PREPROCESS_V3_NODES,
    *PIXAL3D_GENERATE_V3_NODES,
]

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "PIXAL3D_NODES",
    "PIXAL3D_V3_NODES",
]
