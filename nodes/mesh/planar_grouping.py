"""
Planar Grouping - Structure-aware mesh segmentation into planar regions.

Groups connected faces by normal direction, preserving flat-shaded planes
while identifying hard edges at group boundaries.

Pure NumPy implementation - works in both ComfyUI and Blender.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from collections import deque


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute normalized face normals.

    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices

    Returns:
        (M, 3) normalized face normals
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Cross product of two edges
    normals = np.cross(v1 - v0, v2 - v0)

    # Normalize (handle degenerate faces)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-10)  # Avoid division by zero
    normals = normals / lengths

    return normals


def _quantize_position(pos: np.ndarray, precision: int = 6) -> Tuple:
    """Quantize a 3D position to a hashable tuple with given decimal precision."""
    scale = 10 ** precision
    return (int(round(pos[0] * scale)),
            int(round(pos[1] * scale)),
            int(round(pos[2] * scale)))


def _make_edge_key_by_position(p1: np.ndarray, p2: np.ndarray, precision: int = 6) -> Tuple:
    """Create a hashable edge key from two vertex positions, order-independent."""
    q1 = _quantize_position(p1, precision)
    q2 = _quantize_position(p2, precision)
    return tuple(sorted([q1, q2]))


def build_face_adjacency(
    faces: np.ndarray,
    vertices: np.ndarray = None,
    precision: int = 6,
) -> Dict[int, List[int]]:
    """
    Build face adjacency graph - which faces share an edge.

    Handles both normal meshes (shared vertices) and face-split meshes
    (where each face has unique vertices). For face-split meshes, edges
    are matched by vertex positions instead of indices.

    Args:
        faces: (M, 3) face indices
        vertices: (N, 3) vertex positions (required for face-split mesh detection)
        precision: Decimal precision for position comparison (default 6 = 0.000001)

    Returns:
        Dict mapping face_idx -> list of adjacent face indices
    """
    num_faces = len(faces)
    num_verts = int(faces.max()) + 1 if len(faces) > 0 else 0

    # Detect face-split mesh: exactly 3 unique verts per face
    is_face_split = (vertices is not None and
                     len(vertices) == num_faces * 3 and
                     num_verts == num_faces * 3)

    if is_face_split and vertices is not None:
        print(f"[PlanarGrouping] Face-split mesh detected ({num_verts} verts = {num_faces} faces × 3)")
        print(f"[PlanarGrouping] Using position-based adjacency (precision={precision})")

        # Build edge -> faces mapping using vertex positions
        edge_to_faces = {}

        for face_idx, face in enumerate(faces):
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]

            # Each face has 3 edges - use position-based keys
            edges = [
                _make_edge_key_by_position(v0, v1, precision),
                _make_edge_key_by_position(v1, v2, precision),
                _make_edge_key_by_position(v2, v0, precision),
            ]
            for edge in edges:
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(face_idx)
    else:
        # Normal mesh: use vertex indices
        edge_to_faces = {}

        for face_idx, face in enumerate(faces):
            # Each face has 3 edges
            edges = [
                tuple(sorted([face[0], face[1]])),
                tuple(sorted([face[1], face[2]])),
                tuple(sorted([face[2], face[0]])),
            ]
            for edge in edges:
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(face_idx)

    # Build adjacency from shared edges
    adjacency = {i: [] for i in range(num_faces)}

    for edge, face_list in edge_to_faces.items():
        if len(face_list) == 2:
            f1, f2 = face_list
            adjacency[f1].append(f2)
            adjacency[f2].append(f1)

    return adjacency


def angle_between_normals(n1: np.ndarray, n2: np.ndarray) -> float:
    """
    Compute angle between two normal vectors in degrees.

    Args:
        n1, n2: Normal vectors (can be single or batch)

    Returns:
        Angle in degrees
    """
    dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def cluster_faces_by_normal(
    faces: np.ndarray,
    normals: np.ndarray,
    adjacency: Dict[int, List[int]],
    angle_threshold: float = 15.0,
    min_group_size: int = 1,
) -> Tuple[np.ndarray, int]:
    """
    Cluster connected faces by normal direction using flood-fill.

    Faces are grouped together if:
    1. They share an edge (are adjacent)
    2. Their normals differ by less than angle_threshold

    Args:
        faces: (M, 3) face indices
        normals: (M, 3) face normals
        adjacency: Face adjacency dict from build_face_adjacency()
        angle_threshold: Maximum angle (degrees) between normals in same group
        min_group_size: Minimum faces per group (smaller groups get merged)

    Returns:
        Tuple of (group_labels, num_groups)
        group_labels: (M,) array of group IDs for each face
    """
    num_faces = len(faces)
    group_labels = np.full(num_faces, -1, dtype=np.int32)
    current_group = 0

    # Flood-fill clustering
    for start_face in range(num_faces):
        if group_labels[start_face] >= 0:
            continue  # Already assigned

        # BFS from this face
        queue = deque([start_face])
        group_faces = []

        while queue:
            face_idx = queue.popleft()

            if group_labels[face_idx] >= 0:
                continue

            group_labels[face_idx] = current_group
            group_faces.append(face_idx)

            # Check neighbors
            for neighbor_idx in adjacency[face_idx]:
                if group_labels[neighbor_idx] >= 0:
                    continue

                # Check if normals are similar
                angle = angle_between_normals(
                    normals[face_idx],
                    normals[neighbor_idx]
                )

                if angle <= angle_threshold:
                    queue.append(neighbor_idx)

        current_group += 1

    num_groups = current_group

    # Handle small groups - merge into nearest group iteratively
    # Run multiple passes until no small groups remain or no progress is made
    if min_group_size > 1:
        max_iterations = 50  # Prevent infinite loops
        for iteration in range(max_iterations):
            group_sizes = np.bincount(group_labels, minlength=num_groups)
            small_groups = np.where(group_sizes < min_group_size)[0]

            if len(small_groups) == 0:
                break  # All groups are large enough

            merged_any = False

            for small_group in small_groups:
                if group_sizes[small_group] == 0:
                    continue  # Already emptied

                small_faces = np.where(group_labels == small_group)[0]

                for face_idx in small_faces:
                    # Find best neighboring group - prefer large groups, but accept any
                    best_group = -1
                    best_angle = float('inf')
                    best_is_large = False

                    for neighbor_idx in adjacency[face_idx]:
                        neighbor_group = group_labels[neighbor_idx]
                        if neighbor_group == small_group:
                            continue

                        neighbor_is_large = group_sizes[neighbor_group] >= min_group_size

                        angle = angle_between_normals(
                            normals[face_idx],
                            normals[neighbor_idx]
                        )

                        # Prefer large groups; if same size category, prefer better angle
                        if neighbor_is_large and not best_is_large:
                            # Large group beats any small group
                            best_angle = angle
                            best_group = neighbor_group
                            best_is_large = True
                        elif neighbor_is_large == best_is_large and angle < best_angle:
                            # Same category, better angle
                            best_angle = angle
                            best_group = neighbor_group
                            best_is_large = neighbor_is_large

                    if best_group >= 0:
                        old_group = group_labels[face_idx]
                        group_labels[face_idx] = best_group
                        group_sizes[old_group] -= 1
                        group_sizes[best_group] += 1
                        merged_any = True

            if not merged_any:
                break  # No progress, stop iterating

        if iteration > 0:
            print(f"[PlanarGrouping] Small group merging: {iteration + 1} iterations")

    # Renumber groups to be contiguous
    unique_groups = np.unique(group_labels)
    remap = {old: new for new, old in enumerate(unique_groups)}
    group_labels = np.array([remap[g] for g in group_labels], dtype=np.int32)
    num_groups = len(unique_groups)

    return group_labels, num_groups


def find_group_boundary_edges(
    faces: np.ndarray,
    group_labels: np.ndarray,
    vertices: np.ndarray = None,
    precision: int = 6,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Find edges that lie on boundaries between different planar groups.

    Handles both normal meshes and face-split meshes.

    Args:
        faces: (M, 3) face indices
        group_labels: (M,) group ID for each face
        vertices: (N, 3) vertex positions (for face-split mesh handling)
        precision: Decimal precision for position comparison

    Returns:
        Tuple of (boundary_edges, edge_face_pairs)
        boundary_edges: List of (v1, v2) vertex index pairs
        edge_face_pairs: (K, 2) array of (face1, face2) for each boundary edge
    """
    num_faces = len(faces)
    num_verts = int(faces.max()) + 1 if len(faces) > 0 else 0

    # Detect face-split mesh
    is_face_split = (vertices is not None and
                     len(vertices) == num_faces * 3 and
                     num_verts == num_faces * 3)

    # Build edge -> (faces, vertex_indices) mapping
    edge_to_data = {}

    for face_idx, face in enumerate(faces):
        if is_face_split and vertices is not None:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            edges_data = [
                (_make_edge_key_by_position(v0, v1, precision), (face[0], face[1])),
                (_make_edge_key_by_position(v1, v2, precision), (face[1], face[2])),
                (_make_edge_key_by_position(v2, v0, precision), (face[2], face[0])),
            ]
        else:
            edges_data = [
                (tuple(sorted([face[0], face[1]])), (face[0], face[1])),
                (tuple(sorted([face[1], face[2]])), (face[1], face[2])),
                (tuple(sorted([face[2], face[0]])), (face[2], face[0])),
            ]

        for edge_key, vert_pair in edges_data:
            if edge_key not in edge_to_data:
                edge_to_data[edge_key] = []
            edge_to_data[edge_key].append((face_idx, vert_pair))

    # Find boundary edges (different groups on each side)
    boundary_edges = []
    edge_face_pairs = []

    for edge_key, face_data in edge_to_data.items():
        if len(face_data) != 2:
            continue  # Skip boundary/non-manifold edges

        (f1, vp1), (f2, vp2) = face_data
        if group_labels[f1] != group_labels[f2]:
            # Use vertex indices from first face's edge
            boundary_edges.append(vp1)
            edge_face_pairs.append((f1, f2))

    return boundary_edges, np.array(edge_face_pairs) if edge_face_pairs else np.array([]).reshape(0, 2)


def compute_group_planes(
    vertices: np.ndarray,
    faces: np.ndarray,
    group_labels: np.ndarray,
    num_groups: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Fit a plane to each group using SVD (least squares).

    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices
        group_labels: (M,) group ID for each face
        num_groups: Number of groups

    Returns:
        List of (centroid, normal) tuples for each group's fitted plane
    """
    planes = []

    for group_id in range(num_groups):
        group_faces = faces[group_labels == group_id]

        if len(group_faces) == 0:
            planes.append((np.zeros(3), np.array([0, 0, 1])))
            continue

        # Get all vertices in this group
        group_vert_indices = np.unique(group_faces.flatten())
        group_verts = vertices[group_vert_indices]

        # Centroid
        centroid = group_verts.mean(axis=0)

        # SVD to find best-fit plane
        centered = group_verts - centroid
        _, _, Vt = np.linalg.svd(centered)
        normal = Vt[-1]  # Last row = smallest singular value = plane normal

        # Ensure consistent normal direction (pointing "up" on average)
        if normal[2] < 0:
            normal = -normal

        planes.append((centroid, normal))

    return planes


def straighten_boundaries(
    vertices: np.ndarray,
    faces: np.ndarray,
    group_labels: np.ndarray,
    group_planes: List[Tuple[np.ndarray, np.ndarray]],
    group_sizes: np.ndarray,
    project_interior: bool = True,
    boundary_weight: float = 1.0,
    max_displacement: float = 0.0,
    min_group_size_for_straighten: int = 10,
) -> np.ndarray:
    """
    Straighten boundary edges by projecting vertices onto plane intersections.

    For each vertex:
    - 1 group: optionally project onto that group's fitted plane
    - 2 groups: project onto the intersection line of those two planes
    - 3+ groups: keep at intersection point of multiple planes (corner)

    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices
        group_labels: (M,) group ID for each face
        group_planes: List of (centroid, normal) for each group
        group_sizes: (num_groups,) face count per group
        project_interior: If True, project interior vertices onto their plane
        boundary_weight: How much to move boundary vertices (0-1, 1=fully straighten)
        max_displacement: Maximum distance a vertex can move (0=auto based on mesh size)
        min_group_size_for_straighten: Only straighten boundaries between groups >= this size

    Returns:
        (N, 3) modified vertex positions with straightened boundaries
    """
    new_vertices = vertices.copy()
    num_verts = len(vertices)

    # Auto-calculate max displacement if not set (use 5% of mesh bounding box diagonal)
    if max_displacement <= 0:
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)
        max_displacement = bbox_diagonal * 0.05
        print(f"[PlanarGrouping] Auto max_displacement: {max_displacement:.4f} (5% of bbox diagonal)")

    # Step 1: Build vertex -> groups mapping
    vertex_groups = [set() for _ in range(num_verts)]

    for face_idx, face in enumerate(faces):
        group_id = group_labels[face_idx]
        for vert_idx in face:
            vertex_groups[vert_idx].add(group_id)

    # Statistics
    single_group_count = 0
    two_group_count = 0
    multi_group_count = 0
    skipped_small_group = 0
    clamped_count = 0

    # Step 2: Process each vertex based on how many groups it touches
    for vert_idx in range(num_verts):
        groups = list(vertex_groups[vert_idx])
        pos = vertices[vert_idx]

        if len(groups) == 0:
            continue  # Isolated vertex, skip

        elif len(groups) == 1:
            # Interior vertex - optionally project to plane
            single_group_count += 1
            if project_interior:
                group_id = groups[0]
                # Skip small groups - their planes are unreliable
                if group_sizes[group_id] < min_group_size_for_straighten:
                    skipped_small_group += 1
                    continue

                centroid, normal = group_planes[group_id]
                # Project point onto plane: p' = p - ((p - c) · n) * n
                dist = np.dot(pos - centroid, normal)
                displacement = -dist * normal

                # Clamp displacement
                disp_mag = np.linalg.norm(displacement)
                if disp_mag > max_displacement:
                    displacement = displacement * (max_displacement / disp_mag)
                    clamped_count += 1

                new_vertices[vert_idx] = pos + displacement

        elif len(groups) == 2:
            # Boundary vertex between two groups - project to intersection line
            two_group_count += 1

            # Skip if either group is too small
            if (group_sizes[groups[0]] < min_group_size_for_straighten or
                group_sizes[groups[1]] < min_group_size_for_straighten):
                skipped_small_group += 1
                continue

            c1, n1 = group_planes[groups[0]]
            c2, n2 = group_planes[groups[1]]

            # Line direction = cross product of plane normals
            line_dir = np.cross(n1, n2)
            line_dir_norm = np.linalg.norm(line_dir)

            if line_dir_norm < 1e-6:
                # Planes are nearly parallel - just project to average plane
                avg_normal = (n1 + n2) / 2
                norm_len = np.linalg.norm(avg_normal)
                if norm_len < 1e-10:
                    continue
                avg_normal /= norm_len
                avg_centroid = (c1 + c2) / 2
                dist = np.dot(pos - avg_centroid, avg_normal)
                displacement = -dist * avg_normal
            else:
                line_dir = line_dir / line_dir_norm

                d1 = np.dot(c1, n1)
                d2 = np.dot(c2, n2)

                # Solve for point on line closest to current vertex
                A = np.array([n1, n2])
                b = np.array([d1, d2])
                A_full = np.vstack([A, line_dir])
                b_full = np.append(b, np.dot(pos, line_dir))

                try:
                    line_point, residuals, rank, s = np.linalg.lstsq(A_full, b_full, rcond=None)

                    # Check if solution is valid (rank should be 3 for unique solution)
                    if rank < 3:
                        continue

                    displacement = boundary_weight * (line_point - pos)
                except:
                    continue  # Skip on error

            # Clamp displacement
            disp_mag = np.linalg.norm(displacement)
            if disp_mag > max_displacement:
                displacement = displacement * (max_displacement / disp_mag)
                clamped_count += 1

            new_vertices[vert_idx] = pos + displacement

        else:
            # Multi-group junction (corner vertex) - these are tricky, skip for now
            # Corner vertices where 3+ groups meet often produce bad results
            multi_group_count += 1

            # Only process if ALL groups are large enough
            all_large = all(group_sizes[g] >= min_group_size_for_straighten for g in groups)
            if not all_large:
                skipped_small_group += 1
                continue

            # Build system of plane equations and solve for intersection
            planes_n = np.array([group_planes[g][1] for g in groups])
            planes_d = np.array([np.dot(group_planes[g][0], group_planes[g][1]) for g in groups])

            try:
                # Least squares solution for over-determined system
                corner_point, residuals, rank, s = np.linalg.lstsq(planes_n, planes_d, rcond=None)

                # Check solution quality - high residual means planes don't intersect well
                if len(residuals) > 0 and residuals[0] > 0.1:
                    continue

                displacement = boundary_weight * (corner_point - pos)

                # Clamp displacement
                disp_mag = np.linalg.norm(displacement)
                if disp_mag > max_displacement:
                    displacement = displacement * (max_displacement / disp_mag)
                    clamped_count += 1

                new_vertices[vert_idx] = pos + displacement
            except:
                pass  # Keep original position

    print(f"[PlanarGrouping] Boundary straightening: {single_group_count} interior, "
          f"{two_group_count} edge, {multi_group_count} corner vertices")
    print(f"[PlanarGrouping] Skipped {skipped_small_group} vertices (small groups), "
          f"clamped {clamped_count} displacements")

    return new_vertices


def planar_group_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    angle_threshold: float = 15.0,
    min_group_size: int = 10,
    refine_with_plane_fit: bool = False,
    plane_distance_threshold: float = 0.01,
    straighten_boundaries_enabled: bool = False,
    straighten_weight: float = 1.0,
    project_interior: bool = True,
) -> Dict:
    """
    Main function: Group mesh faces into planar regions.

    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices
        angle_threshold: Max angle (degrees) between normals in same group
        min_group_size: Minimum faces per group
        refine_with_plane_fit: If True, refine groups using plane fitting
        plane_distance_threshold: Max distance from fitted plane (for refinement)
        straighten_boundaries_enabled: If True, straighten boundary edges
        straighten_weight: How much to move vertices (0-1, 1=fully straighten)
        project_interior: If True, project interior vertices onto their plane

    Returns:
        Dict with:
            - 'group_labels': (M,) group ID for each face
            - 'num_groups': Number of planar groups
            - 'boundary_edges': List of (v1, v2) boundary edge vertex pairs
            - 'edge_face_pairs': (K, 2) face pairs for each boundary edge
            - 'group_planes': List of (centroid, normal) for each group
            - 'stats': Statistics dict
            - 'modified_vertices': (N, 3) if boundary straightening enabled
    """
    print(f"[PlanarGrouping] Input: {len(vertices)} vertices, {len(faces)} faces")
    print(f"[PlanarGrouping] Parameters: angle_threshold={angle_threshold}°, min_group_size={min_group_size}")

    # Step 1: Compute face normals
    normals = compute_face_normals(vertices, faces)
    print(f"[PlanarGrouping] Computed {len(normals)} face normals")

    # Step 2: Build adjacency (pass vertices for face-split mesh handling)
    print(f"[PlanarGrouping] Building face adjacency graph...")
    adjacency = build_face_adjacency(faces, vertices)

    # Count adjacency stats
    adj_counts = [len(adj) for adj in adjacency.values()]
    print(f"[PlanarGrouping] Adjacency: avg={np.mean(adj_counts):.1f}, max={max(adj_counts)}")

    # Step 3: Cluster faces
    print(f"[PlanarGrouping] Clustering faces by normal direction...")
    group_labels, num_groups = cluster_faces_by_normal(
        faces, normals, adjacency,
        angle_threshold=angle_threshold,
        min_group_size=min_group_size,
    )
    print(f"[PlanarGrouping] Found {num_groups} planar groups")

    # Group size statistics
    group_sizes = np.bincount(group_labels, minlength=num_groups)
    print(f"[PlanarGrouping] Group sizes: min={group_sizes.min()}, max={group_sizes.max()}, "
          f"median={np.median(group_sizes):.0f}")

    # Step 4: Find boundary edges (pass vertices for face-split handling)
    print(f"[PlanarGrouping] Finding group boundary edges...")
    boundary_edges, edge_face_pairs = find_group_boundary_edges(faces, group_labels, vertices)
    print(f"[PlanarGrouping] Found {len(boundary_edges)} boundary edges")

    # Step 5: Compute fitted planes for each group
    print(f"[PlanarGrouping] Fitting planes to groups...")
    group_planes = compute_group_planes(vertices, faces, group_labels, num_groups)

    # Step 6: Optional refinement using plane fitting
    if refine_with_plane_fit and plane_distance_threshold > 0:
        print(f"[PlanarGrouping] Refining groups with plane fitting (threshold={plane_distance_threshold})...")
        # TODO: Implement refinement pass
        # For each face, check if it's closer to a neighbor group's plane
        pass

    # Step 7: Optional boundary straightening
    modified_vertices = None
    if straighten_boundaries_enabled:
        print(f"[PlanarGrouping] Straightening boundaries (weight={straighten_weight}, project_interior={project_interior})...")
        modified_vertices = straighten_boundaries(
            vertices, faces, group_labels, group_planes, group_sizes,
            project_interior=project_interior,
            boundary_weight=straighten_weight,
            max_displacement=0.0,  # Auto-calculate based on mesh size
            min_group_size_for_straighten=min_group_size,  # Use same threshold as grouping
        )

    # Compute statistics
    stats = {
        'num_faces': len(faces),
        'num_groups': num_groups,
        'num_boundary_edges': len(boundary_edges),
        'group_sizes': group_sizes.tolist(),
        'avg_group_size': float(np.mean(group_sizes)),
        'boundary_edge_ratio': len(boundary_edges) / max(len(faces) * 1.5, 1),  # ~1.5 edges per face
        'boundaries_straightened': straighten_boundaries_enabled,
    }

    print(f"[PlanarGrouping] Complete: {num_groups} groups, {len(boundary_edges)} boundary edges")

    result = {
        'group_labels': group_labels,
        'num_groups': num_groups,
        'boundary_edges': boundary_edges,
        'edge_face_pairs': edge_face_pairs,
        'group_planes': group_planes,
        'face_normals': normals,
        'stats': stats,
    }

    if modified_vertices is not None:
        result['modified_vertices'] = modified_vertices

    return result


def apply_grouping_to_trimesh(mesh, grouping_result: Dict, mark_sharp: bool = True, mark_seam: bool = False):
    """
    Apply planar grouping results to a trimesh object.

    Creates face colors based on group membership and optionally
    stores boundary edge information in metadata.

    Args:
        mesh: trimesh.Trimesh object
        grouping_result: Result dict from planar_group_mesh()
        mark_sharp: Store boundary edges for sharp marking
        mark_seam: Store boundary edges for seam marking

    Returns:
        Modified mesh with group colors and metadata
    """
    import trimesh

    group_labels = grouping_result['group_labels']
    num_groups = grouping_result['num_groups']
    boundary_edges = grouping_result['boundary_edges']

    # Generate distinct colors for each group
    np.random.seed(42)  # Reproducible colors
    group_colors = np.random.randint(50, 255, size=(num_groups, 3), dtype=np.uint8)

    # Create face colors
    face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    face_colors[:, :3] = group_colors[group_labels]
    face_colors[:, 3] = 255  # Alpha

    # Apply to mesh
    mesh.visual.face_colors = face_colors

    # Store in metadata
    mesh.metadata['planar_grouping'] = {
        'group_labels': group_labels,
        'num_groups': num_groups,
        'boundary_edges': boundary_edges,
        'mark_sharp': mark_sharp,
        'mark_seam': mark_seam,
        'stats': grouping_result['stats'],
    }

    return mesh


# ============================================================================
# Blender-compatible functions (work with bmesh)
# ============================================================================

def planar_group_bmesh(bm, angle_threshold: float = 15.0, min_group_size: int = 10) -> Dict:
    """
    Apply planar grouping to a Blender bmesh.

    Args:
        bm: bmesh.types.BMesh object
        angle_threshold: Max angle between normals in same group
        min_group_size: Minimum faces per group

    Returns:
        Dict with grouping results including edge references
    """
    import numpy as np

    # Ensure lookup tables
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    # Extract vertices and faces
    vertices = np.array([v.co[:] for v in bm.verts])
    faces = np.array([[v.index for v in f.verts] for f in bm.faces])

    # Run grouping algorithm
    result = planar_group_mesh(
        vertices, faces,
        angle_threshold=angle_threshold,
        min_group_size=min_group_size,
    )

    # Map boundary edges back to bmesh edges
    boundary_bm_edges = []
    edge_lookup = {tuple(sorted([e.verts[0].index, e.verts[1].index])): e for e in bm.edges}

    for v1, v2 in result['boundary_edges']:
        edge_key = tuple(sorted([v1, v2]))
        if edge_key in edge_lookup:
            boundary_bm_edges.append(edge_lookup[edge_key])

    result['boundary_bm_edges'] = boundary_bm_edges

    return result


def mark_planar_group_edges(
    bm,
    grouping_result: Dict,
    mark_sharp: bool = True,
    mark_seam: bool = False,
    mark_crease: bool = False,
    crease_value: float = 1.0,
) -> int:
    """
    Mark boundary edges between planar groups in a bmesh.

    Args:
        bm: bmesh.types.BMesh object
        grouping_result: Result from planar_group_bmesh()
        mark_sharp: Mark edges as sharp (affects shading)
        mark_seam: Mark edges as seams (affects UV unwrapping)
        mark_crease: Mark edges with crease weight (affects subdivision)
        crease_value: Crease weight to apply (0-1)

    Returns:
        Number of edges marked
    """
    boundary_edges = grouping_result.get('boundary_bm_edges', [])

    if not boundary_edges:
        print("[PlanarGrouping] No boundary edges to mark")
        return 0

    # Get crease layer if needed
    crease_layer = None
    if mark_crease:
        crease_layer = bm.edges.layers.float.get('crease_edge')
        if crease_layer is None:
            crease_layer = bm.edges.layers.float.new('crease_edge')

    marked = 0
    for edge in boundary_edges:
        if mark_sharp:
            edge.smooth = False
        if mark_seam:
            edge.seam = True
        if mark_crease and crease_layer:
            edge[crease_layer] = crease_value
        marked += 1

    print(f"[PlanarGrouping] Marked {marked} boundary edges "
          f"(sharp={mark_sharp}, seam={mark_seam}, crease={mark_crease})")

    return marked
