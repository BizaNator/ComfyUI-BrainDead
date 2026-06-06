"""
BrainDead CubePart nodes for ComfyUI.

CATEGORY: 🧠BrainDead/CubePart

Roblox CubePart open-vocabulary, part-controllable 3D decomposition.
Vendors cube_part (nodes/cubepart/vendor) so the nodes are self-contained.

Nodes:
- BD_CubePartSegment  - TRIMESH + up to 8 part names -> TRIMESH_LIST + combined TRIMESH + names
- BD_CubePartGetPart  - TRIMESH_LIST + index -> single TRIMESH (+ name)

Workflow:
  mesh (Pixal3D / Trellis2 / .glb) → BD_CubePartSegment → parts (TRIMESH_LIST)
                                                        → combined (colored TRIMESH preview)
                                     BD_CubePartGetPart(parts, i) → TRIMESH → CuMesh/Blender/export
"""

from .segment import (
    CUBEPART_SEGMENT_NODES,
    CUBEPART_SEGMENT_DISPLAY_NAMES,
    CUBEPART_SEGMENT_V3_NODES,
)
from .getpart import (
    CUBEPART_GETPART_NODES,
    CUBEPART_GETPART_DISPLAY_NAMES,
    CUBEPART_GETPART_V3_NODES,
)

NODE_CLASS_MAPPINGS = {
    **CUBEPART_SEGMENT_NODES,
    **CUBEPART_GETPART_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **CUBEPART_SEGMENT_DISPLAY_NAMES,
    **CUBEPART_GETPART_DISPLAY_NAMES,
}

CUBEPART_NODES = list(NODE_CLASS_MAPPINGS.values())

CUBEPART_V3_NODES = [
    *CUBEPART_SEGMENT_V3_NODES,
    *CUBEPART_GETPART_V3_NODES,
]

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "CUBEPART_NODES",
    "CUBEPART_V3_NODES",
]
