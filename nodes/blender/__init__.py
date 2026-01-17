"""
Blender-based mesh processing nodes for ComfyUI.

Provides high-quality mesh operations using Blender headlessly:
- BD_BlenderDecimate: Edge-preserving mesh decimation
- BD_BlenderRemesh: Voxel/quad remeshing
- BD_BlenderRepair: Advanced mesh repair
- BD_BlenderTransferColors: BVH-based vertex color transfer
"""

from .decimate import (
    BD_BlenderDecimate,
    DECIMATE_V3_NODES,
    DECIMATE_NODES,
    DECIMATE_DISPLAY_NAMES,
)
from .remesh import (
    BD_BlenderRemesh,
    REMESH_V3_NODES,
    REMESH_NODES,
    REMESH_DISPLAY_NAMES,
)
from .repair import (
    BD_BlenderRepair,
    REPAIR_V3_NODES,
    REPAIR_NODES,
    REPAIR_DISPLAY_NAMES,
)
from .transfer import (
    BD_BlenderTransferColors,
    TRANSFER_V3_NODES,
    TRANSFER_NODES,
    TRANSFER_DISPLAY_NAMES,
)

# V3 node list for extension
BLENDER_V3_NODES = [
    *DECIMATE_V3_NODES,
    *REMESH_V3_NODES,
    *REPAIR_V3_NODES,
    *TRANSFER_V3_NODES,
]

# V1 compatibility - NODE_CLASS_MAPPINGS dict
BLENDER_NODES = {
    **DECIMATE_NODES,
    **REMESH_NODES,
    **REPAIR_NODES,
    **TRANSFER_NODES,
}

BLENDER_DISPLAY_NAMES = {
    **DECIMATE_DISPLAY_NAMES,
    **REMESH_DISPLAY_NAMES,
    **REPAIR_DISPLAY_NAMES,
    **TRANSFER_DISPLAY_NAMES,
}

__all__ = [
    # V3 nodes
    "BLENDER_V3_NODES",
    # Individual classes
    "BD_BlenderDecimate",
    "BD_BlenderRemesh",
    "BD_BlenderRepair",
    "BD_BlenderTransferColors",
    # V1 compatibility
    "BLENDER_NODES",
    "BLENDER_DISPLAY_NAMES",
]
