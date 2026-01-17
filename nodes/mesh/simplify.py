"""
BD_CuMeshSimplify - GPU-accelerated mesh simplification with color preservation.

Uses CuMesh (from TRELLIS2) for fast simplification, then transfers
vertex colors from the original mesh using BVH nearest-neighbor lookup.
"""

import gc
import numpy as np

from comfy_api.latest import io

from .types import TrimeshInput, TrimeshOutput

# Check for required dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cumesh
    HAS_CUMESH = True
except ImportError:
    HAS_CUMESH = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


class BD_CuMeshSimplify(io.ComfyNode):
    """
    GPU-accelerated mesh simplification with vertex color preservation.

    Uses CuMesh for fast simplification, then transfers colors from
    original mesh via BVH lookup. Also marks hard edges based on angle.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CuMeshSimplify",
            display_name="BD CuMesh Simplify",
            category="🧠BrainDead/Mesh",
            description="""GPU-accelerated mesh simplification with color preservation.

Uses CuMesh (TRELLIS2) for fast GPU simplification.
Preserves vertex colors via BVH nearest-neighbor transfer.
Marks hard edges based on angle threshold.

Much faster than Blender for large meshes.""",
            inputs=[
                TrimeshInput("mesh"),
                io.Int.Input(
                    "target_faces",
                    default=5000,
                    min=100,
                    max=1000000,
                    tooltip="Target number of faces after simplification",
                ),
                io.Float.Input(
                    "sharp_angle",
                    default=7.0,
                    min=0.0,
                    max=90.0,
                    step=0.5,
                    tooltip="Angle threshold for marking edges as sharp (degrees). 7° works well for stylized meshes.",
                ),
                io.Boolean.Input(
                    "fill_holes",
                    default=True,
                    tooltip="Fill small holes before simplification",
                ),
                io.Float.Input(
                    "fill_holes_perimeter",
                    default=0.03,
                    min=0.001,
                    max=0.5,
                    step=0.001,
                    tooltip="Maximum hole perimeter to fill (mesh units)",
                ),
                io.Boolean.Input(
                    "preserve_colors",
                    default=True,
                    tooltip="Transfer vertex colors from original mesh after simplification",
                ),
                io.Boolean.Input(
                    "face_colors",
                    default=True,
                    tooltip="Use face-based colors (no bleeding) vs vertex-based (smoother)",
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
        target_faces: int = 5000,
        sharp_angle: float = 7.0,
        fill_holes: bool = True,
        fill_holes_perimeter: float = 0.03,
        preserve_colors: bool = True,
        face_colors: bool = True,
    ) -> io.NodeOutput:
        # Check dependencies
        if not HAS_TORCH:
            return io.NodeOutput(mesh, "ERROR: torch not installed")
        if not HAS_CUMESH:
            return io.NodeOutput(mesh, "ERROR: cumesh not installed (need TRELLIS2)")
        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "ERROR: trimesh not installed")

        if mesh is None:
            return io.NodeOutput(None, "ERROR: No input mesh")

        orig_verts = len(mesh.vertices)
        orig_faces = len(mesh.faces)
        print(f"[BD CuMesh] Input: {orig_verts:,} verts, {orig_faces:,} faces -> target {target_faces:,}")

        # Check for vertex colors in original mesh
        has_colors = False
        orig_colors = None
        if preserve_colors:
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                has_colors = True
                orig_colors = mesh.visual.vertex_colors.copy()
                print(f"[BD CuMesh] Found vertex colors: {orig_colors.shape}")

        try:
            # Convert to torch tensors
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
            faces = torch.tensor(mesh.faces, dtype=torch.int32).cuda()

            # Store original for color transfer
            orig_vertices = vertices.clone()
            orig_faces_tensor = faces.clone()

            # CuMesh expects Y-up, ComfyUI uses Z-up
            # Convert Z-up to Y-up: swap Y and Z, negate new Y
            vertices_yup = vertices.clone()
            vertices_yup[:, 1], vertices_yup[:, 2] = -vertices[:, 2].clone(), vertices[:, 1].clone()

            # Initialize CuMesh
            cu = cumesh.CuMesh()
            cu.init(vertices_yup, faces)
            print(f"[BD CuMesh] Initialized: {cu.num_vertices} verts, {cu.num_faces} faces")

            # Fill holes
            if fill_holes:
                cu.fill_holes(max_hole_perimeter=fill_holes_perimeter)
                print(f"[BD CuMesh] After fill holes: {cu.num_vertices} verts, {cu.num_faces} faces")

            # Remove degenerate faces
            cu.remove_degenerate_faces()

            # Unify face orientations before simplify
            cu.unify_face_orientations()
            print("[BD CuMesh] Unified face orientations")

            # Simplify
            if cu.num_faces > target_faces:
                print(f"[BD CuMesh] Simplifying to {target_faces} faces...")
                cu.simplify(target_faces, verbose=True)
                print(f"[BD CuMesh] After simplify: {cu.num_vertices} verts, {cu.num_faces} faces")

            # Unify orientations again (simplify can break it)
            cu.unify_face_orientations()

            # Remove unreferenced vertices
            cu.remove_unreferenced_vertices()

            # Read result
            out_verts, out_faces = cu.read()
            out_verts_np = out_verts.cpu().numpy()
            out_faces_np = out_faces.cpu().numpy()

            # Convert back to Z-up: swap Y and Z, negate new Z
            out_verts_np[:, 1], out_verts_np[:, 2] = out_verts_np[:, 2].copy(), -out_verts_np[:, 1].copy()

            # Build result mesh
            result = trimesh.Trimesh(
                vertices=out_verts_np,
                faces=out_faces_np,
                process=False
            )

            # Transfer colors from original mesh
            if has_colors and orig_colors is not None:
                print("[BD CuMesh] Transferring colors from original mesh...")
                result = cls._transfer_colors(
                    result, mesh, orig_colors, face_colors
                )

            # Mark sharp edges based on angle
            if sharp_angle > 0:
                sharp_count = cls._mark_sharp_edges(result, sharp_angle)
                print(f"[BD CuMesh] Marked {sharp_count} sharp edges (>{sharp_angle}°)")

            # Stats
            new_verts = len(result.vertices)
            new_faces = len(result.faces)
            reduction = (1 - new_faces / orig_faces) * 100 if orig_faces > 0 else 0

            status = f"Simplified: {orig_faces:,} -> {new_faces:,} faces ({reduction:.1f}% reduction)"
            if has_colors:
                status += " | colors preserved"
            if sharp_angle > 0:
                status += f" | sharp edges marked"

            # Cleanup GPU
            del vertices, faces, vertices_yup, out_verts, out_faces, cu
            del orig_vertices, orig_faces_tensor
            gc.collect()
            torch.cuda.empty_cache()

            return io.NodeOutput(result, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return io.NodeOutput(mesh, f"ERROR: {e}")

    @classmethod
    def _transfer_colors(cls, target_mesh, source_mesh, source_colors, face_based=True):
        """Transfer vertex colors from source to target mesh using nearest neighbor."""
        from scipy.spatial import cKDTree

        print(f"[BD CuMesh] Building KDTree from {len(source_mesh.vertices)} source vertices...")

        if face_based:
            # Face-based: compute face centers and colors, then transfer per-face
            source_face_centers = source_mesh.triangles_center
            source_face_colors = source_colors[source_mesh.faces].mean(axis=1)

            tree = cKDTree(source_face_centers)

            target_face_centers = target_mesh.triangles_center
            _, indices = tree.query(target_face_centers)

            # Assign face colors to vertices (split vertices for no bleeding)
            new_verts = []
            new_faces = []
            new_colors = []

            for i, face in enumerate(target_mesh.faces):
                face_color = source_face_colors[indices[i]]
                base_idx = len(new_verts)

                for j, vert_idx in enumerate(face):
                    new_verts.append(target_mesh.vertices[vert_idx])
                    new_colors.append(face_color)

                new_faces.append([base_idx, base_idx + 1, base_idx + 2])

            new_verts = np.array(new_verts)
            new_faces = np.array(new_faces)
            new_colors = np.array(new_colors, dtype=np.uint8)

            result = trimesh.Trimesh(
                vertices=new_verts,
                faces=new_faces,
                vertex_colors=new_colors,
                process=False
            )
            print(f"[BD CuMesh] Face-based transfer: {len(new_faces)} faces, {len(new_verts)} verts")

        else:
            # Vertex-based: direct nearest neighbor lookup
            tree = cKDTree(source_mesh.vertices)
            _, indices = tree.query(target_mesh.vertices)

            new_colors = source_colors[indices]

            result = trimesh.Trimesh(
                vertices=target_mesh.vertices,
                faces=target_mesh.faces,
                vertex_colors=new_colors,
                process=False
            )
            print(f"[BD CuMesh] Vertex-based transfer: {len(target_mesh.faces)} faces")

        return result

    @classmethod
    def _mark_sharp_edges(cls, mesh, angle_threshold_deg):
        """Mark edges as sharp where adjacent face normals differ by more than threshold."""
        import numpy as np

        angle_threshold_rad = np.radians(angle_threshold_deg)

        # Get face normals
        face_normals = mesh.face_normals

        # Get face adjacency (which faces share each edge)
        # trimesh stores this as face_adjacency: (n, 2) array of adjacent face indices
        if not hasattr(mesh, 'face_adjacency') or mesh.face_adjacency is None:
            print("[BD CuMesh] Computing face adjacency...")
            mesh._cache.clear()  # Force recompute

        face_adj = mesh.face_adjacency
        if face_adj is None or len(face_adj) == 0:
            return 0

        # Compute angle between adjacent face normals
        n1 = face_normals[face_adj[:, 0]]
        n2 = face_normals[face_adj[:, 1]]

        # Dot product gives cos(angle)
        dots = np.einsum('ij,ij->i', n1, n2)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)

        # Mark edges where angle exceeds threshold
        sharp_mask = angles > angle_threshold_rad
        sharp_count = np.sum(sharp_mask)

        # Store sharp edges in mesh metadata
        if sharp_count > 0:
            sharp_edges = mesh.face_adjacency_edges[sharp_mask]
            mesh.metadata['sharp_edges'] = sharp_edges
            mesh.metadata['sharp_angle'] = angle_threshold_deg

        return sharp_count


# V3 node list
SIMPLIFY_V3_NODES = [BD_CuMeshSimplify]

# V1 compatibility
SIMPLIFY_NODES = {
    "BD_CuMeshSimplify": BD_CuMeshSimplify,
}

SIMPLIFY_DISPLAY_NAMES = {
    "BD_CuMeshSimplify": "BD CuMesh Simplify",
}
