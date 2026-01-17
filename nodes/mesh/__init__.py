"""
BrainDead Mesh nodes for ComfyUI.

CATEGORY: BrainDead/Mesh

Nodes:
- BD_CacheMesh - Cache TRIMESH objects
- BD_SampleVoxelgridColors - Sample colors from TRELLIS2 voxelgrid
- BD_SampleVoxelgridPBR - Sample full PBR attributes from voxelgrid
- BD_TransferPointcloudColors - Transfer from pointcloud (deprecated)
- BD_TransferColorsPymeshlab - Transfer using pymeshlab
- BD_TransferVertexColors - BVH-based vertex color transfer
- BD_MeshRepair - Repair mesh topology
- BD_SmartDecimate - Edge-preserving decimation
- BD_ExportMeshWithColors - Export mesh with vertex colors
- BD_CuMeshSimplify - GPU-accelerated simplification with color preservation
"""

# Import TRIMESH type helpers for re-export
from .types import TrimeshInput, TrimeshOutput

from .cache import (
    MESH_CACHE_NODES,
    MESH_CACHE_DISPLAY_NAMES,
    MESH_CACHE_V3_NODES,
)
from .sampling import (
    MESH_SAMPLING_NODES,
    MESH_SAMPLING_DISPLAY_NAMES,
    MESH_SAMPLING_V3_NODES,
)
from .transfer import (
    MESH_TRANSFER_NODES,
    MESH_TRANSFER_DISPLAY_NAMES,
    MESH_TRANSFER_V3_NODES,
)
from .processing import (
    MESH_PROCESSING_NODES,
    MESH_PROCESSING_DISPLAY_NAMES,
    MESH_PROCESSING_V3_NODES,
)
from .export import (
    MESH_EXPORT_NODES,
    MESH_EXPORT_DISPLAY_NAMES,
    MESH_EXPORT_V3_NODES,
)
from .simplify import (
    SIMPLIFY_NODES,
    SIMPLIFY_DISPLAY_NAMES,
    SIMPLIFY_V3_NODES,
)

# Combine all mesh nodes
NODE_CLASS_MAPPINGS = {
    **MESH_CACHE_NODES,
    **MESH_SAMPLING_NODES,
    **MESH_TRANSFER_NODES,
    **MESH_PROCESSING_NODES,
    **MESH_EXPORT_NODES,
    **SIMPLIFY_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **MESH_CACHE_DISPLAY_NAMES,
    **MESH_SAMPLING_DISPLAY_NAMES,
    **MESH_TRANSFER_DISPLAY_NAMES,
    **MESH_PROCESSING_DISPLAY_NAMES,
    **MESH_EXPORT_DISPLAY_NAMES,
    **SIMPLIFY_DISPLAY_NAMES,
}

# Export lists for external use
MESH_NODES = list(NODE_CLASS_MAPPINGS.values())

# V3 node list for extension
MESH_V3_NODES = [
    *MESH_CACHE_V3_NODES,
    *MESH_SAMPLING_V3_NODES,
    *MESH_TRANSFER_V3_NODES,
    *MESH_PROCESSING_V3_NODES,
    *MESH_EXPORT_V3_NODES,
    *SIMPLIFY_V3_NODES,
]

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "MESH_NODES",
    "MESH_V3_NODES",
]
