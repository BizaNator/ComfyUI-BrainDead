"""
BD_CombineEdgeMetadata - Combine edge metadata from multiple sources.

Unions boundary edges from PlanarGrouping, EdgeMarking, or other sources
into a single deduplicated edge_metadata for downstream nodes like MergePlanes.

Uses POSITION-BASED deduplication to handle face-split meshes where the same
geometric edge may have different vertex indices across sources.
"""

import numpy as np
from comfy_api.latest import io
from .types import TrimeshInput, EdgeMetadataInput, EdgeMetadataOutput


def _quantize_position(pos, precision: int = 5) -> tuple:
    """Quantize a 3D position to a hashable tuple with given decimal precision."""
    scale = 10 ** precision
    return (int(round(pos[0] * scale)),
            int(round(pos[1] * scale)),
            int(round(pos[2] * scale)))


def _make_edge_key_by_position(p1, p2, precision: int = 5) -> tuple:
    """Create a hashable edge key from two vertex positions, order-independent."""
    q1 = _quantize_position(p1, precision)
    q2 = _quantize_position(p2, precision)
    return tuple(sorted([q1, q2]))


def _get_edge_positions_from_metadata(edge_metadata: dict, mesh_vertices=None) -> list:
    """
    Extract edge positions from edge_metadata.

    If edge_metadata contains 'boundary_edge_positions', use those directly.
    Otherwise, use 'boundary_edges' indices with mesh_vertices to compute positions.

    Returns list of ((x1,y1,z1), (x2,y2,z2)) tuples.
    """
    # Check for pre-computed positions first
    if 'boundary_edge_positions' in edge_metadata:
        positions = edge_metadata['boundary_edge_positions']
        return [(tuple(p1), tuple(p2)) for p1, p2 in positions]

    # Fall back to computing from indices
    edges = edge_metadata.get('boundary_edges', [])
    if not edges or mesh_vertices is None:
        return []

    # Get vertices array
    if hasattr(mesh_vertices, 'vertices'):
        verts = np.array(mesh_vertices.vertices)
    else:
        verts = np.array(mesh_vertices)

    positions = []
    for edge in edges:
        v1_idx, v2_idx = int(edge[0]), int(edge[1])
        # Bounds check
        if v1_idx < len(verts) and v2_idx < len(verts):
            p1 = tuple(verts[v1_idx])
            p2 = tuple(verts[v2_idx])
            positions.append((p1, p2))

    return positions


class BD_CombineEdgeMetadata(io.ComfyNode):
    """
    Combine edge metadata from multiple sources into unified output.

    Uses POSITION-BASED deduplication to handle face-split meshes correctly.
    Same geometric edge from different sources will be deduplicated even if
    they have different vertex indices.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CombineEdgeMetadata",
            display_name="BD Combine Edge Metadata",
            category="🧠BrainDead/Mesh",
            description="""Combine edge metadata from multiple detection sources.

Use this to union edges from:
- BD_PlanarGrouping (structure-based boundaries)
- BD_BlenderEdgeMarking (color/angle-based edges)
- Any other edge detection node

POSITION-BASED DEDUPLICATION:
This node uses vertex POSITIONS (not indices) for deduplication.
This correctly handles face-split meshes where the same geometric edge
may have different vertex indices in different sources.

Connect 'reference_mesh' to enable index remapping for downstream nodes.
Without it, output will use indices from the first source containing each edge.

Example workflow:
[Mesh] ─┬─→ [PlanarGrouping] ──→ edge_metadata_1 ─┐
        │                                          │
        └─→ [EdgeMarking] ─────→ edge_metadata_2 ─┼─→ [CombineEdgeMetadata] → [MergePlanes]
                                                   │
[Final Mesh] ─────────────────→ reference_mesh ───┘""",
            inputs=[
                TrimeshInput(
                    "reference_mesh",
                    tooltip="Reference mesh for position-based deduplication and index remapping. Should be the mesh that will be passed to downstream nodes.",
                ),
                EdgeMetadataInput(
                    "edge_metadata_1",
                    optional=False,
                    tooltip="First edge metadata source (required)",
                ),
                EdgeMetadataInput(
                    "edge_metadata_2",
                    optional=True,
                    tooltip="Second edge metadata source (optional)",
                ),
                EdgeMetadataInput(
                    "edge_metadata_3",
                    optional=True,
                    tooltip="Third edge metadata source (optional)",
                ),
                EdgeMetadataInput(
                    "edge_metadata_4",
                    optional=True,
                    tooltip="Fourth edge metadata source (optional)",
                ),
                io.Int.Input(
                    "position_precision",
                    default=5,
                    min=3,
                    max=8,
                    tooltip="Decimal precision for position comparison (5 = 0.00001 units). Lower = more aggressive merging.",
                ),
            ],
            outputs=[
                EdgeMetadataOutput(display_name="edge_metadata"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        reference_mesh=None,
        edge_metadata_1: dict = None,
        edge_metadata_2: dict = None,
        edge_metadata_3: dict = None,
        edge_metadata_4: dict = None,
        position_precision: int = 5,
    ) -> io.NodeOutput:
        if edge_metadata_1 is None:
            return io.NodeOutput(None, "ERROR: edge_metadata_1 is required")

        # Collect all metadata sources
        sources = []
        if edge_metadata_1:
            sources.append(edge_metadata_1)
        if edge_metadata_2:
            sources.append(edge_metadata_2)
        if edge_metadata_3:
            sources.append(edge_metadata_3)
        if edge_metadata_4:
            sources.append(edge_metadata_4)

        if not sources:
            return io.NodeOutput(None, "ERROR: No edge metadata provided")

        # Get reference mesh vertices for position lookup and remapping
        ref_vertices = None
        if reference_mesh is not None:
            ref_vertices = np.array(reference_mesh.vertices)
            print(f"[BD CombineEdgeMetadata] Using reference mesh: {len(ref_vertices)} vertices")

        # Build position-to-edge mapping
        # Key: position-based edge key, Value: (positions, source_name, original_indices)
        edge_position_map = {}
        source_names = []
        source_counts = []

        for meta in sources:
            source_name = meta.get('source', 'unknown')
            source_names.append(source_name)

            # Get edge positions from this source
            edge_positions = _get_edge_positions_from_metadata(meta, reference_mesh)
            edges = meta.get('boundary_edges', [])

            count_before = len(edge_position_map)

            for i, (p1, p2) in enumerate(edge_positions):
                # Create position-based key for deduplication
                pos_key = _make_edge_key_by_position(p1, p2, position_precision)

                if pos_key not in edge_position_map:
                    # Store first occurrence
                    original_indices = edges[i] if i < len(edges) else [0, 0]
                    edge_position_map[pos_key] = {
                        'positions': (p1, p2),
                        'source': source_name,
                        'original_indices': original_indices,
                    }

            new_edges = len(edge_position_map) - count_before
            source_counts.append((source_name, len(edge_positions), new_edges))
            print(f"[BD CombineEdgeMetadata] {source_name}: {len(edge_positions)} edges, {new_edges} new")

        # Remap to reference mesh indices if available
        combined_edges = []
        combined_positions = []
        remapped_count = 0
        failed_remap = 0

        if ref_vertices is not None:
            # Build KDTree for fast vertex lookup
            from scipy.spatial import cKDTree
            tree = cKDTree(ref_vertices)

            for pos_key, edge_data in edge_position_map.items():
                p1, p2 = edge_data['positions']

                # Find nearest vertices in reference mesh
                _, idx1 = tree.query(p1)
                _, idx2 = tree.query(p2)

                # Verify the match is close enough
                dist1 = np.linalg.norm(ref_vertices[idx1] - np.array(p1))
                dist2 = np.linalg.norm(ref_vertices[idx2] - np.array(p2))
                threshold = 10 ** (-position_precision + 1)  # Allow 10x precision tolerance

                if dist1 < threshold and dist2 < threshold:
                    combined_edges.append([int(idx1), int(idx2)])
                    combined_positions.append([list(p1), list(p2)])
                    remapped_count += 1
                else:
                    # Use original indices as fallback
                    combined_edges.append([int(edge_data['original_indices'][0]),
                                          int(edge_data['original_indices'][1])])
                    combined_positions.append([list(p1), list(p2)])
                    failed_remap += 1

            print(f"[BD CombineEdgeMetadata] Remapped {remapped_count} edges to reference mesh, {failed_remap} fallback")
        else:
            # No reference mesh - use original indices and positions
            for pos_key, edge_data in edge_position_map.items():
                p1, p2 = edge_data['positions']
                combined_edges.append([int(edge_data['original_indices'][0]),
                                      int(edge_data['original_indices'][1])])
                combined_positions.append([list(p1), list(p2)])

        # Build combined metadata
        combined_metadata = {
            'boundary_edges': combined_edges,
            'boundary_edge_positions': combined_positions,  # Include positions for downstream
            'num_groups': sum(m.get('num_groups', 0) for m in sources),
            'source': 'combined:' + '+'.join(source_names),
            'sources': source_names,
            'position_precision': position_precision,
        }

        # Build status
        status_lines = [
            f"Combined {len(sources)} edge metadata sources (position-based):",
            "",
        ]

        total_input = 0
        for source_name, total, unique in source_counts:
            status_lines.append(f"  {source_name}: {total} edges ({unique} new)")
            total_input += total

        status_lines.append("")
        status_lines.append(f"Total input edges: {total_input}")
        status_lines.append(f"Deduplicated output: {len(combined_edges)} edges")

        duplicates = total_input - len(combined_edges)
        if duplicates > 0:
            status_lines.append(f"Duplicates merged: {duplicates}")

        if ref_vertices is not None:
            status_lines.append("")
            status_lines.append(f"Remapped to reference mesh: {remapped_count}")
            if failed_remap > 0:
                status_lines.append(f"Remap fallbacks: {failed_remap}")

        status = "\n".join(status_lines)

        print(f"[BD CombineEdgeMetadata] Combined {len(sources)} sources → {len(combined_edges)} unique edges")

        return io.NodeOutput(combined_metadata, status)


# V3 API exports
EDGE_UTILS_V3_NODES = [BD_CombineEdgeMetadata]

# Legacy exports for compatibility
EDGE_UTILS_NODES = {
    "BD_CombineEdgeMetadata": BD_CombineEdgeMetadata,
}

EDGE_UTILS_DISPLAY_NAMES = {
    "BD_CombineEdgeMetadata": "BD Combine Edge Metadata",
}
