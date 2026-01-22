"""
BD_PlanarGrouping - Structure-aware mesh segmentation into planar regions.

Groups connected faces by normal direction and marks boundary edges.
Essential for preserving flat-shaded planes during decimation.
"""

import numpy as np
from comfy_api.latest import io
from .types import TrimeshInput, TrimeshOutput, EdgeMetadataOutput
from .planar_grouping import planar_group_mesh, apply_grouping_to_trimesh


class BD_PlanarGrouping(io.ComfyNode):
    """
    Segment mesh into planar groups and mark boundary edges.

    Groups connected faces that share similar normal directions,
    then marks edges at group boundaries for preservation during decimation.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_PlanarGrouping",
            display_name="BD Planar Grouping",
            category="🧠BrainDead/Mesh",
            description="""Structure-aware mesh segmentation into planar regions.

Groups connected faces by normal direction (angle threshold).
Marks boundary edges between groups as SHARP for decimation preservation.

BOUNDARY STRAIGHTENING (Option 2):
- Enable 'straighten_boundaries' to project vertices onto fitted planes
- Boundary edges become straight lines where planes intersect
- Interior vertices are flattened onto their group's plane
- Creates clean, geometric boundaries instead of jagged mesh edges

Use BEFORE decimation to preserve flat-shaded planes:
1. BD_PlanarGrouping → marks group boundaries (+ optional straightening)
2. BD_BlenderDecimateV3 → preserves marked edges with delimit

Parameters:
- angle_threshold: Max angle between normals in same group (lower = more groups)
- min_group_size: Merge small groups into neighbors (reduces noise)
- straighten_boundaries: Project vertices to make boundaries straight lines
- color_by_group: Apply distinct colors to visualize groups""",
            inputs=[
                TrimeshInput("mesh"),
                io.Float.Input(
                    "angle_threshold",
                    default=15.0,
                    min=1.0,
                    max=90.0,
                    step=1.0,
                    tooltip="Maximum angle (degrees) between face normals in the same planar group. Lower = more groups, stricter planes.",
                ),
                io.Int.Input(
                    "min_group_size",
                    default=10,
                    min=1,
                    max=1000,
                    step=1,
                    tooltip="Minimum faces per group. Smaller groups get merged into nearest neighbor.",
                ),
                io.Boolean.Input(
                    "straighten_boundaries",
                    default=False,
                    tooltip="Straighten boundary edges by projecting vertices onto plane intersections. Creates clean geometric boundaries.",
                ),
                io.Float.Input(
                    "straighten_weight",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    tooltip="How much to move vertices toward straightened positions (0=none, 1=fully straighten).",
                ),
                io.Boolean.Input(
                    "project_interior",
                    default=True,
                    tooltip="Also flatten interior vertices onto their group's fitted plane.",
                ),
                io.Boolean.Input(
                    "color_by_group",
                    default=True,
                    tooltip="Apply distinct colors to each planar group for visualization.",
                ),
                io.Boolean.Input(
                    "mark_boundary_colors",
                    default=True,
                    tooltip="Mark boundary edges with contrasting vertex colors (for BD_BlenderEdgeMarking).",
                ),
            ],
            outputs=[
                TrimeshOutput(display_name="mesh"),
                EdgeMetadataOutput(display_name="edge_metadata"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        angle_threshold: float = 15.0,
        min_group_size: int = 10,
        straighten_boundaries: bool = False,
        straighten_weight: float = 1.0,
        project_interior: bool = True,
        color_by_group: bool = True,
        mark_boundary_colors: bool = True,
    ) -> io.NodeOutput:
        import trimesh

        if mesh is None:
            return io.NodeOutput(None, None, "ERROR: No input mesh")

        try:
            vertices = np.array(mesh.vertices)
            faces = np.array(mesh.faces)

            # Run planar grouping algorithm
            result = planar_group_mesh(
                vertices, faces,
                angle_threshold=angle_threshold,
                min_group_size=min_group_size,
                straighten_boundaries_enabled=straighten_boundaries,
                straighten_weight=straighten_weight,
                project_interior=project_interior,
            )

            group_labels = result['group_labels']
            num_groups = result['num_groups']
            boundary_edges = result['boundary_edges']
            stats = result['stats']

            # Create output mesh (copy to avoid modifying input)
            output_mesh = mesh.copy()

            # Apply modified vertices if boundary straightening was enabled
            if 'modified_vertices' in result:
                output_mesh.vertices = result['modified_vertices']
                print(f"[PlanarGrouping] Applied straightened vertices to mesh")

            # Apply group colors if requested
            if color_by_group:
                output_mesh = apply_grouping_to_trimesh(
                    output_mesh, result,
                    mark_sharp=True,
                    mark_seam=False,
                )

            # Mark boundary edges with contrasting colors
            if mark_boundary_colors and len(boundary_edges) > 0:
                # Get or create vertex colors
                if not hasattr(output_mesh.visual, 'vertex_colors') or output_mesh.visual.vertex_colors is None:
                    # Initialize from face colors if we have them
                    if hasattr(output_mesh.visual, 'face_colors') and output_mesh.visual.face_colors is not None:
                        # Convert face colors to vertex colors
                        vertex_colors = np.zeros((len(vertices), 4), dtype=np.uint8)
                        for face_idx, face in enumerate(faces):
                            for vert_idx in face:
                                vertex_colors[vert_idx] = output_mesh.visual.face_colors[face_idx]
                        output_mesh.visual.vertex_colors = vertex_colors
                    else:
                        output_mesh.visual.vertex_colors = np.full((len(vertices), 4), [128, 128, 128, 255], dtype=np.uint8)

                # Mark boundary edge vertices with bright color
                boundary_verts = set()
                for v1, v2 in boundary_edges:
                    boundary_verts.add(v1)
                    boundary_verts.add(v2)

                # Apply contrasting color to boundary vertices
                for vert_idx in boundary_verts:
                    # Make boundary vertices bright white/yellow for edge detection
                    output_mesh.visual.vertex_colors[vert_idx] = [255, 255, 0, 255]

                print(f"[PlanarGrouping] Marked {len(boundary_verts)} boundary vertices with contrasting color")

            # Store metadata for downstream nodes
            output_mesh.metadata['planar_grouping'] = {
                'group_labels': group_labels.tolist(),
                'num_groups': num_groups,
                'boundary_edges': boundary_edges,
                'angle_threshold': angle_threshold,
                'min_group_size': min_group_size,
            }

            # Build status string
            status_lines = [
                f"Planar Grouping Complete",
                f"",
                f"Input: {len(faces):,} faces",
                f"Groups found: {num_groups}",
                f"Boundary edges: {len(boundary_edges):,}",
                f"",
                f"Group sizes:",
                f"  Min: {min(stats['group_sizes'])}",
                f"  Max: {max(stats['group_sizes'])}",
                f"  Avg: {stats['avg_group_size']:.0f}",
                f"",
                f"Boundary ratio: {stats['boundary_edge_ratio']*100:.1f}%",
            ]

            if straighten_boundaries:
                status_lines.append(f"")
                status_lines.append(f"BOUNDARIES STRAIGHTENED (weight={straighten_weight})")
                if project_interior:
                    status_lines.append(f"Interior vertices flattened to planes")

            if color_by_group:
                status_lines.append(f"")
                status_lines.append(f"Groups colored for visualization")

            if mark_boundary_colors:
                status_lines.append(f"Boundary vertices marked yellow")

            status = "\n".join(status_lines)

            # Build edge metadata for explicit passthrough
            # Convert to native Python types for JSON serialization
            # Include positions for position-based deduplication in CombineEdgeMetadata
            edge_positions = []
            for v1, v2 in boundary_edges:
                p1 = vertices[v1].tolist()
                p2 = vertices[v2].tolist()
                edge_positions.append([p1, p2])

            edge_metadata = {
                'boundary_edges': [[int(v1), int(v2)] for v1, v2 in boundary_edges],
                'boundary_edge_positions': edge_positions,  # For position-based deduplication
                'num_groups': int(num_groups),
                'source': 'planar_grouping',
                'angle_threshold': float(angle_threshold),
            }

            return io.NodeOutput(output_mesh, edge_metadata, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return io.NodeOutput(mesh, None, f"ERROR: {e}")


# V3 API exports
GROUPING_V3_NODES = [BD_PlanarGrouping]

# Legacy exports for compatibility
GROUPING_NODES = {
    "BD_PlanarGrouping": BD_PlanarGrouping,
}

GROUPING_DISPLAY_NAMES = {
    "BD_PlanarGrouping": "BD Planar Grouping",
}
