"""
Blender-based mesh processing nodes for ComfyUI.

Provides high-quality mesh operations using Blender headlessly:
- BD_BlenderDecimate: Basic decimation
- BD_BlenderDecimateV2: Full-featured decimation (Decimate_v1.py port)
- BD_BlenderDecimateV3: Full decimation using BrainDeadBlender addon
- BD_BlenderRemesh: Voxel/quad remeshing (native + addon)
- BD_BlenderCleanup: Smart mesh cleanup using addon
- BD_BlenderEdgeMarking: Edge marking from colors/angles
- BD_BlenderVertexColors: Vertex color operations
- BD_BlenderNormals: Normal fixing and verification
- BD_BlenderRepair: Advanced mesh repair
- BD_BlenderTransferColors: BVH-based vertex color transfer
"""

from .decimate import (
    BD_BlenderDecimate,
    DECIMATE_V3_NODES,
    DECIMATE_NODES,
    DECIMATE_DISPLAY_NAMES,
)
from .decimate_full import (
    BD_BlenderDecimateV2,
    DECIMATE_FULL_V3_NODES,
    DECIMATE_FULL_NODES,
    DECIMATE_FULL_DISPLAY_NAMES,
)
from .remesh import (
    BD_BlenderRemesh as BD_BlenderRemeshBasic,
    REMESH_V3_NODES as REMESH_BASIC_V3_NODES,
    REMESH_NODES as REMESH_BASIC_NODES,
    REMESH_DISPLAY_NAMES as REMESH_BASIC_DISPLAY_NAMES,
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
from .addon_nodes import (
    BD_BlenderDecimateV3,
    BD_BlenderRemesh,
    BD_BlenderCleanup,
    BD_BlenderEdgeMarking,
    BD_BlenderVertexColors,
    BD_BlenderNormals,
    ADDON_V3_NODES,
    ADDON_NODES,
    ADDON_DISPLAY_NAMES,
)
from .export_mesh import (
    BD_BlenderExportMesh,
    EXPORT_MESH_V3_NODES,
    EXPORT_MESH_NODES,
    EXPORT_MESH_DISPLAY_NAMES,
)

# V3 node list for extension
BLENDER_V3_NODES = [
    *DECIMATE_V3_NODES,
    *DECIMATE_FULL_V3_NODES,
    *REMESH_BASIC_V3_NODES,
    *REPAIR_V3_NODES,
    *TRANSFER_V3_NODES,
    *ADDON_V3_NODES,
    *EXPORT_MESH_V3_NODES,
]

# V1 compatibility - NODE_CLASS_MAPPINGS dict
BLENDER_NODES = {
    **DECIMATE_NODES,
    **DECIMATE_FULL_NODES,
    **REMESH_BASIC_NODES,
    **REPAIR_NODES,
    **TRANSFER_NODES,
    **ADDON_NODES,
    **EXPORT_MESH_NODES,
}

BLENDER_DISPLAY_NAMES = {
    **DECIMATE_DISPLAY_NAMES,
    **DECIMATE_FULL_DISPLAY_NAMES,
    **REMESH_BASIC_DISPLAY_NAMES,
    **REPAIR_DISPLAY_NAMES,
    **TRANSFER_DISPLAY_NAMES,
    **ADDON_DISPLAY_NAMES,
    **EXPORT_MESH_DISPLAY_NAMES,
}

__all__ = [
    # V3 nodes
    "BLENDER_V3_NODES",
    # Individual classes - Decimate
    "BD_BlenderDecimate",
    "BD_BlenderDecimateV2",
    "BD_BlenderDecimateV3",
    # Individual classes - Addon-based
    "BD_BlenderRemesh",
    "BD_BlenderCleanup",
    "BD_BlenderEdgeMarking",
    "BD_BlenderVertexColors",
    "BD_BlenderNormals",
    # Individual classes - Export
    "BD_BlenderExportMesh",
    # Individual classes - Legacy
    "BD_BlenderRepair",
    "BD_BlenderTransferColors",
    # V1 compatibility
    "BLENDER_NODES",
    "BLENDER_DISPLAY_NAMES",
]
