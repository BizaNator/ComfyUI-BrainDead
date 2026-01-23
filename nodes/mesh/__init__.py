"""
BrainDead Mesh nodes for ComfyUI.

CATEGORY: BrainDead/Mesh

Nodes:
- BD_CacheMesh - Cache TRIMESH objects
- BD_SampleVoxelgridColors - Sample colors from TRELLIS2 voxelgrid (outputs COLOR_FIELD)
- BD_SampleVoxelgridPBR - Sample full PBR attributes from voxelgrid
- BD_ApplyColorField - Apply COLOR_FIELD to any mesh (deferred color application)
- BD_TransferPointcloudColors - Transfer from pointcloud (deprecated)
- BD_TransferColorsPymeshlab - Transfer using pymeshlab
- BD_TransferVertexColors - BVH-based vertex color transfer
- BD_MeshRepair - Repair mesh topology
- BD_SmartDecimate - Edge-preserving decimation
- BD_ExportMeshWithColors - Export mesh with vertex colors
- BD_CuMeshSimplify - GPU-accelerated simplification with color preservation
- BD_UVUnwrap - UV unwrap with xatlas (GPU) or Blender Smart UV
- BD_PlanarGrouping - Structure-aware segmentation into planar regions
- BD_CombineEdgeMetadata - Combine edge metadata from multiple sources
- BD_OVoxelBake - Bake PBR textures using Microsoft's o_voxel reference implementation
- BD_OVoxelTextureBake - Bake-only: takes pre-processed mesh + voxelgrid for PBR baking
- BD_MeshToOVoxel - Convert textured mesh to VOXELGRID format
- BD_FixNormals - Fix face orientation (Python-only, fast)
- BD_PackBundle - Pack mesh + textures + colors into MESH_BUNDLE
- BD_CacheBundle - Cache MESH_BUNDLE for fast reload
"""

# Import type helpers for re-export
from .types import (
    TrimeshInput, TrimeshOutput,
    EdgeMetadataInput, EdgeMetadataOutput,
    ColorFieldInput, ColorFieldOutput,
    MeshBundleInput, MeshBundleOutput,
)

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
from .unwrap import (
    UNWRAP_NODES,
    UNWRAP_DISPLAY_NAMES,
    UNWRAP_V3_NODES,
)
from .grouping import (
    GROUPING_NODES,
    GROUPING_DISPLAY_NAMES,
    GROUPING_V3_NODES,
)
from .edge_utils import (
    EDGE_UTILS_NODES,
    EDGE_UTILS_DISPLAY_NAMES,
    EDGE_UTILS_V3_NODES,
)
from .color_field import (
    COLOR_FIELD_NODES,
    COLOR_FIELD_DISPLAY_NAMES,
    COLOR_FIELD_V3_NODES,
)
from .ovoxel_bake import (
    OVOXEL_BAKE_NODES,
    OVOXEL_BAKE_DISPLAY_NAMES,
    OVOXEL_BAKE_V3_NODES,
)
from .ovoxel_texture_bake import (
    OVOXEL_TEXTURE_BAKE_NODES,
    OVOXEL_TEXTURE_BAKE_DISPLAY_NAMES,
    OVOXEL_TEXTURE_BAKE_V3_NODES,
)
from .ovoxel_convert import (
    OVOXEL_CONVERT_NODES,
    OVOXEL_CONVERT_DISPLAY_NAMES,
    OVOXEL_CONVERT_V3_NODES,
)
from .fix_normals import (
    FIX_NORMALS_NODES,
    FIX_NORMALS_DISPLAY_NAMES,
    FIX_NORMALS_V3_NODES,
)
from .bundle import (
    BUNDLE_NODES,
    BUNDLE_DISPLAY_NAMES,
    BUNDLE_V3_NODES,
)
from .inspector import (
    MESH_INSPECTOR_NODES,
    MESH_INSPECTOR_DISPLAY_NAMES,
    MESH_INSPECTOR_V3_NODES,
)

# Combine all mesh nodes
NODE_CLASS_MAPPINGS = {
    **MESH_CACHE_NODES,
    **MESH_SAMPLING_NODES,
    **MESH_TRANSFER_NODES,
    **MESH_PROCESSING_NODES,
    **MESH_EXPORT_NODES,
    **SIMPLIFY_NODES,
    **UNWRAP_NODES,
    **GROUPING_NODES,
    **EDGE_UTILS_NODES,
    **COLOR_FIELD_NODES,
    **OVOXEL_BAKE_NODES,
    **OVOXEL_TEXTURE_BAKE_NODES,
    **OVOXEL_CONVERT_NODES,
    **FIX_NORMALS_NODES,
    **BUNDLE_NODES,
    **MESH_INSPECTOR_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **MESH_CACHE_DISPLAY_NAMES,
    **MESH_SAMPLING_DISPLAY_NAMES,
    **MESH_TRANSFER_DISPLAY_NAMES,
    **MESH_PROCESSING_DISPLAY_NAMES,
    **MESH_EXPORT_DISPLAY_NAMES,
    **SIMPLIFY_DISPLAY_NAMES,
    **UNWRAP_DISPLAY_NAMES,
    **GROUPING_DISPLAY_NAMES,
    **EDGE_UTILS_DISPLAY_NAMES,
    **COLOR_FIELD_DISPLAY_NAMES,
    **OVOXEL_BAKE_DISPLAY_NAMES,
    **OVOXEL_TEXTURE_BAKE_DISPLAY_NAMES,
    **OVOXEL_CONVERT_DISPLAY_NAMES,
    **FIX_NORMALS_DISPLAY_NAMES,
    **BUNDLE_DISPLAY_NAMES,
    **MESH_INSPECTOR_DISPLAY_NAMES,
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
    *UNWRAP_V3_NODES,
    *GROUPING_V3_NODES,
    *EDGE_UTILS_V3_NODES,
    *COLOR_FIELD_V3_NODES,
    *OVOXEL_BAKE_V3_NODES,
    *OVOXEL_TEXTURE_BAKE_V3_NODES,
    *OVOXEL_CONVERT_V3_NODES,
    *FIX_NORMALS_V3_NODES,
    *BUNDLE_V3_NODES,
    *MESH_INSPECTOR_V3_NODES,
]

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "MESH_NODES",
    "MESH_V3_NODES",
]
