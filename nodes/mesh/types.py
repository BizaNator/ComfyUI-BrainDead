"""
Custom types for BrainDead mesh nodes.

Provides TRIMESH type wrapper to match TRELLIS2 output.
Provides EDGE_METADATA type for passing edge mark data between nodes.
"""

from comfy_api.latest import io


# TRIMESH type wrapper - use io.Custom("TRIMESH") for input/output
# This matches TRELLIS2's "TRIMESH" type
def TrimeshInput(name: str, **kwargs):
    """Create a TRIMESH input (matches TRELLIS2 output type)."""
    return io.Custom("TRIMESH").Input(name, **kwargs)


def TrimeshOutput(display_name: str = "mesh"):
    """Create a TRIMESH output (matches TRELLIS2 input type)."""
    return io.Custom("TRIMESH").Output(display_name=display_name)


# EDGE_METADATA type for passing boundary/sharp edge data between nodes
# Structure: {
#   'boundary_edges': [[v1, v2], ...],  # Vertex index pairs
#   'num_groups': int,                   # Number of planar groups (if from PlanarGrouping)
#   'source': str,                       # 'planar_grouping', 'edge_marking', etc.
#   'angle_threshold': float,            # Threshold used for detection
# }
def EdgeMetadataInput(name: str = "edge_metadata", **kwargs):
    """Create an EDGE_METADATA input for receiving edge mark data."""
    kwargs.setdefault('optional', True)  # Usually optional - nodes can work without it
    return io.Custom("EDGE_METADATA").Input(name, **kwargs)


def EdgeMetadataOutput(display_name: str = "edge_metadata"):
    """Create an EDGE_METADATA output for passing edge mark data."""
    return io.Custom("EDGE_METADATA").Output(display_name=display_name)


# COLOR_FIELD type for deferred color application
# Stores voxelgrid color data that can be applied to any mesh at any pipeline stage.
# Structure: {
#   'positions': np.ndarray,      # (N, 3) voxel world positions
#   'colors': np.ndarray,         # (N, 3) or (N, 4) RGB/RGBA float [0-1]
#   'attributes': {               # Optional PBR attributes
#       'metallic': np.ndarray,   # (N,) float [0-1]
#       'roughness': np.ndarray,  # (N,) float [0-1]
#       'alpha': np.ndarray,      # (N,) float [0-1]
#   },
#   'voxel_size': float,          # Original voxel resolution
#   'num_voxels': int,            # Total voxel count
#   'bounds': ((min_x, min_y, min_z), (max_x, max_y, max_z)),
#   'source': str,                # 'trellis2_voxelgrid', etc.
# }
def ColorFieldInput(name: str = "color_field", **kwargs):
    """Create a COLOR_FIELD input for receiving voxelgrid color data."""
    kwargs.setdefault('optional', True)
    return io.Custom("COLOR_FIELD").Input(name, **kwargs)


def ColorFieldOutput(display_name: str = "color_field"):
    """Create a COLOR_FIELD output for passing voxelgrid color data."""
    return io.Custom("COLOR_FIELD").Output(display_name=display_name)


# MESH_BUNDLE type for carrying mesh + textures + vertex colors as a unit.
# Allows caching and exporting complete mesh assets through the pipeline.
# Structure: {
#   'mesh': trimesh.Trimesh,              # Mesh (may have TextureVisuals with PBR)
#   'color_field': dict | None,           # COLOR_FIELD data (optional)
#   'vertex_colors': np.ndarray | None,   # (N, 4) uint8 vertex colors for COLOR_0
#   'diffuse': np.ndarray | None,         # (H, W, 3) uint8 albedo texture
#   'normal': np.ndarray | None,          # (H, W, 3) uint8 normal map
#   'metallic': np.ndarray | None,        # (H, W, 1|3) uint8 metallic
#   'roughness': np.ndarray | None,       # (H, W, 1|3) uint8 roughness
#   'alpha': np.ndarray | None,           # (H, W, 1|3) uint8 alpha/opacity
#   'name': str,                          # Bundle name (for export filenames)
# }
def MeshBundleInput(name: str = "bundle", **kwargs):
    """Create a MESH_BUNDLE input."""
    return io.Custom("MESH_BUNDLE").Input(name, **kwargs)


def MeshBundleOutput(display_name: str = "bundle"):
    """Create a MESH_BUNDLE output."""
    return io.Custom("MESH_BUNDLE").Output(display_name=display_name)
