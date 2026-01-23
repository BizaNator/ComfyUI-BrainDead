"""
BD_FixNormals - Fix face orientation and normal direction.

Lightweight Python-only node (no Blender required).
For topology-based fixing, use BD_BlenderNormals instead.

Methods:
- centroid: Connected-component centroid heuristic (fast, works for convex shapes)
- trimesh: trimesh.repair.fix_normals (merge-based)
- both: centroid + trimesh combined
- double_sided_only: Just set material flag, don't modify geometry
"""

import time

import numpy as np

from comfy_api.latest import io

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

from .types import TrimeshInput, TrimeshOutput


def _fix_normals_centroid(vertices, faces):
    """
    Connected-component centroid heuristic for outward-facing normals.

    For each connected component, checks if face normals point toward or
    away from the mesh center. Flips components where normals point inward.

    Works well for convex/simple geometry. May fail for complex concave shapes.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n_faces = len(faces)
    if n_faces == 0:
        return faces, 0

    faces = faces.copy()

    # Compute face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    face_normals = face_normals / norms
    face_centers = (v0 + v1 + v2) / 3.0

    # Build face adjacency from shared edges
    edge_to_faces = {}
    for fi in range(n_faces):
        for ei in range(3):
            e = tuple(sorted([faces[fi, ei], faces[fi, (ei + 1) % 3]]))
            if e in edge_to_faces:
                edge_to_faces[e].append(fi)
            else:
                edge_to_faces[e] = [fi]

    rows, cols = [], []
    for edge_faces in edge_to_faces.values():
        if len(edge_faces) == 2:
            rows.extend([edge_faces[0], edge_faces[1]])
            cols.extend([edge_faces[1], edge_faces[0]])

    mesh_center = vertices.mean(axis=0)

    if not rows:
        # No adjacency - global heuristic
        to_outside = face_centers - mesh_center
        dots = np.sum(face_normals * to_outside, axis=1)
        if dots.mean() < 0:
            faces = faces[:, ::-1]
            return faces, n_faces
        return faces, 0

    data = np.ones(len(rows))
    graph = csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces))
    n_comp, labels = connected_components(graph, directed=False)

    total_flipped = 0
    for c in range(n_comp):
        comp_mask = labels == c
        comp_centers = face_centers[comp_mask]
        comp_normals = face_normals[comp_mask]
        to_outside = comp_centers - mesh_center
        dots = np.sum(comp_normals * to_outside, axis=1)
        if dots.mean() < 0:
            idx = np.where(comp_mask)[0]
            faces[idx] = faces[idx][:, ::-1]
            total_flipped += len(idx)

    return faces, total_flipped


def _fix_normals_trimesh(mesh):
    """Use trimesh's built-in fix_normals (merge-based adjacency)."""
    # Create a copy to merge without affecting original
    merged = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        process=False,
    )
    # Merge vertices to build proper adjacency
    merged.merge_vertices()
    pre_fix = merged.faces.copy()

    # Run trimesh fix
    trimesh.repair.fix_normals(merged)

    # Find which faces were flipped
    flipped_mask = np.any(pre_fix != merged.faces, axis=1)
    n_flipped = flipped_mask.sum()

    if n_flipped > 0:
        # Apply flips to original mesh faces
        # Map merged face indices back to original
        # Since merge may change face count, use direct approach on original
        new_faces = mesh.faces.copy()
        # For faces that trimesh flipped in merged, flip in original
        # This is approximate - if merge changed topology significantly,
        # fall back to centroid method
        if len(merged.faces) == len(mesh.faces):
            new_faces[flipped_mask] = new_faces[flipped_mask][:, ::-1]
        else:
            # Face count changed from merge - use centroid as fallback
            new_faces, n_flipped = _fix_normals_centroid(
                np.array(mesh.vertices, dtype=np.float32),
                np.array(mesh.faces),
            )
        return new_faces, n_flipped

    return mesh.faces.copy(), 0


def _set_double_sided(mesh):
    """Set doubleSided flag on mesh material if present."""
    if (hasattr(mesh, 'visual')
            and hasattr(mesh.visual, 'material')
            and mesh.visual.material is not None):
        mat = mesh.visual.material
        if hasattr(mat, 'doubleSided'):
            mat.doubleSided = True
        # Also store in extras for glTF export
        if hasattr(mat, 'extras') and mat.extras is None:
            mat.extras = {}
        if hasattr(mat, 'extras') and isinstance(getattr(mat, 'extras', None), dict):
            mat.extras['doubleSided'] = True
        return True
    return False


class BD_FixNormals(io.ComfyNode):
    """
    Fix face normals to point outward. Python-only (no Blender required).

    For heavy-duty topology-based fixing, use BD_BlenderNormals.
    This node is fast and preserves materials/UVs.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_FixNormals",
            display_name="BD Fix Normals",
            category="🧠BrainDead/Mesh",
            description="""Fix face normals to point outward (Python-only, fast).

Methods:
- centroid: Component-based centroid heuristic (fast, good for convex)
- trimesh: trimesh merge-based fix_normals
- both: Run centroid then trimesh
- none: Don't fix geometry (use with double_sided only)

Options:
- double_sided: Set PBR material doubleSided flag (masks remaining flips)

For topology-based fixing via Blender, use BD_BlenderNormals instead.""",
            inputs=[
                TrimeshInput("mesh"),
                io.Combo.Input(
                    "method",
                    options=["centroid", "both", "trimesh", "none"],
                    default="centroid",
                    tooltip="centroid=fast heuristic | trimesh=merge-based | both=combined | none=skip",
                ),
                io.Boolean.Input(
                    "double_sided",
                    default=False,
                    optional=True,
                    tooltip="Set material doubleSided flag (renders both sides, masks flips)",
                ),
            ],
            outputs=[
                TrimeshOutput(display_name="mesh"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        method: str = "centroid",
        double_sided: bool = False,
    ) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "ERROR: trimesh not installed")

        if mesh is None:
            return io.NodeOutput(None, "ERROR: No input mesh")

        start_time = time.time()
        n_faces = len(mesh.faces) if hasattr(mesh, 'faces') and mesh.faces is not None else 0
        total_flipped = 0

        if method != "none" and n_faces > 0:
            vertices = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces)

            if method == "centroid":
                faces, total_flipped = _fix_normals_centroid(vertices, faces)
            elif method == "trimesh":
                faces, total_flipped = _fix_normals_trimesh(mesh)
            elif method == "both":
                # Centroid first, then trimesh for any remaining
                faces, flipped1 = _fix_normals_centroid(vertices, faces)
                # Apply centroid fix to mesh for trimesh pass
                temp_mesh = trimesh.Trimesh(
                    vertices=vertices, faces=faces, process=False
                )
                faces2, flipped2 = _fix_normals_trimesh(temp_mesh)
                faces = faces2
                total_flipped = flipped1 + flipped2

            # Apply fixed faces to output mesh (preserve everything else)
            new_mesh = trimesh.Trimesh(
                vertices=mesh.vertices.copy(),
                faces=faces,
                process=False,
            )
            # Preserve visual (material/UVs/vertex colors)
            if hasattr(mesh, 'visual') and mesh.visual is not None:
                new_mesh.visual = mesh.visual
            # Preserve metadata
            if mesh.metadata:
                new_mesh.metadata.update(mesh.metadata)
            # Preserve vertex normals if present
            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                try:
                    new_mesh.vertex_normals = mesh.vertex_normals.copy()
                except Exception:
                    pass
        else:
            new_mesh = mesh

        # Set doubleSided if requested
        ds_applied = False
        if double_sided:
            ds_applied = _set_double_sided(new_mesh)

        elapsed = time.time() - start_time
        flip_pct = (100 * total_flipped / n_faces) if n_faces > 0 else 0

        parts = []
        if method != "none":
            parts.append(f"{method}: flipped {total_flipped:,}/{n_faces:,} ({flip_pct:.1f}%)")
        if double_sided:
            parts.append("doubleSided" + ("=set" if ds_applied else "=no_material"))
        parts.append(f"{elapsed:.1f}s")

        status = " | ".join(parts)
        print(f"[BD FixNormals] {status}")

        return io.NodeOutput(new_mesh, status)


# V3 node list
FIX_NORMALS_V3_NODES = [BD_FixNormals]

# V1 compatibility
FIX_NORMALS_NODES = {
    "BD_FixNormals": BD_FixNormals,
}

FIX_NORMALS_DISPLAY_NAMES = {
    "BD_FixNormals": "BD Fix Normals",
}
