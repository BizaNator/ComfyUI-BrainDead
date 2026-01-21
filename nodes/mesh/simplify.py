"""
BD_CuMeshSimplify - GPU-accelerated mesh simplification with color preservation.

Uses CuMesh for fast simplification with optional dual-contouring remesh
and sharp edge preservation (CuMesh PR #19).

Features:
- GPU-accelerated mesh simplification
- Optional dual-contouring remesh for cleaner topology
- Sharp edge preservation during remeshing (angle-based detection)
- Mesh cleaning (fill holes, remove duplicates, repair non-manifold)
- Vertex color transfer from original mesh via BVH lookup
- Sharp edge marking based on geometry AND color boundaries
"""

import gc
import time
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
    # Check for remeshing module with sharp edge support
    HAS_REMESH = hasattr(cumesh, 'remeshing')
    if HAS_REMESH:
        from cumesh import _C
        HAS_SHARP_DC = hasattr(_C, 'simple_dual_contour_sharp')
    else:
        HAS_SHARP_DC = False
except ImportError:
    HAS_CUMESH = False
    HAS_REMESH = False
    HAS_SHARP_DC = False

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
            description="""GPU-accelerated mesh simplification with color preservation and optional remeshing.

Features:
- CuMesh GPU simplification (much faster than Blender)
- Optional dual-contouring remesh for cleaner topology
- Sharp edge preservation during remeshing (uses CuMesh PR #19)
- Mesh cleaning: fill holes, remove duplicates, repair non-manifold
- Vertex color transfer from original mesh
- Sharp edge marking based on geometry AND color boundaries

Typical workflow for TRELLIS2 output:
1. Enable remesh + preserve_sharp_edges for low-poly stylized look
2. Enable mesh cleaning to fix topology issues
3. Target 5k-50k faces depending on use case""",
            inputs=[
                TrimeshInput("mesh"),
                io.Int.Input(
                    "target_faces",
                    default=5000,
                    min=100,
                    max=1000000,
                    tooltip="Target number of faces after simplification",
                ),
                # === REMESH OPTIONS ===
                io.Boolean.Input(
                    "remesh",
                    default=False,
                    tooltip="Apply dual-contouring remesh before simplification for cleaner topology",
                ),
                io.Int.Input(
                    "remesh_resolution",
                    default=512,
                    min=128,
                    max=2048,
                    step=64,
                    tooltip="Voxel grid resolution for remeshing. Higher = more detail but slower.",
                ),
                io.Float.Input(
                    "remesh_band",
                    default=1.0,
                    min=0.1,
                    max=5.0,
                    step=0.1,
                    tooltip="Narrow band width in voxel units for remeshing",
                ),
                io.Boolean.Input(
                    "preserve_sharp_edges_remesh",
                    default=True,
                    tooltip="Preserve sharp edges during remeshing (requires CuMesh PR #19)",
                ),
                io.Float.Input(
                    "remesh_sharp_angle",
                    default=30.0,
                    min=10.0,
                    max=90.0,
                    step=5.0,
                    tooltip="Angle threshold (degrees) for sharp edge detection during remeshing",
                ),
                io.Float.Input(
                    "project_back",
                    default=0.9,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    tooltip="How much to project remeshed vertices back to original surface (0=none, 1=full)",
                ),
                # === MESH CLEANING OPTIONS ===
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
                    "remove_duplicate_faces",
                    default=True,
                    tooltip="Remove duplicate/degenerate faces",
                ),
                io.Boolean.Input(
                    "repair_non_manifold",
                    default=True,
                    tooltip="Repair non-manifold edges",
                ),
                io.Boolean.Input(
                    "remove_small_components",
                    default=True,
                    tooltip="Remove small disconnected mesh components",
                ),
                io.Float.Input(
                    "small_component_threshold",
                    default=1e-5,
                    min=0.0,
                    max=0.1,
                    step=1e-6,
                    tooltip="Volume threshold for removing small components (as fraction of total)",
                ),
                # === OUTPUT OPTIONS ===
                io.Float.Input(
                    "sharp_angle",
                    default=7.0,
                    min=0.0,
                    max=90.0,
                    step=0.5,
                    tooltip="Angle threshold for marking edges as sharp (degrees). 7° works well for stylized meshes.",
                ),
                io.Float.Input(
                    "color_edge_threshold",
                    default=0.1,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Color difference threshold for marking edges as sharp (0-1). 0.1 = ~10% color change marks edge. 0 = disabled.",
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
        # Remesh options
        remesh: bool = False,
        remesh_resolution: int = 512,
        remesh_band: float = 1.0,
        preserve_sharp_edges_remesh: bool = True,
        remesh_sharp_angle: float = 30.0,
        project_back: float = 0.9,
        # Mesh cleaning options
        fill_holes: bool = True,
        fill_holes_perimeter: float = 0.03,
        remove_duplicate_faces: bool = True,
        repair_non_manifold: bool = True,
        remove_small_components: bool = True,
        small_component_threshold: float = 1e-5,
        # Output options
        sharp_angle: float = 7.0,
        color_edge_threshold: float = 0.1,
        preserve_colors: bool = True,
        face_colors: bool = True,
    ) -> io.NodeOutput:
        # Check dependencies
        if not HAS_TORCH:
            return io.NodeOutput(mesh, "ERROR: torch not installed")
        if not HAS_CUMESH:
            return io.NodeOutput(mesh, "ERROR: cumesh not installed")
        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "ERROR: trimesh not installed")

        if mesh is None:
            return io.NodeOutput(None, "ERROR: No input mesh")

        start_time = time.time()
        orig_verts = len(mesh.vertices)
        orig_faces = len(mesh.faces)
        print(f"[BD CuMesh] Input: {orig_verts:,} verts, {orig_faces:,} faces -> target {target_faces:,}")

        # Check remesh capability
        if remesh and not HAS_REMESH:
            print("[BD CuMesh] WARNING: remesh requested but cumesh.remeshing not available")
            remesh = False
        if remesh and preserve_sharp_edges_remesh and not HAS_SHARP_DC:
            print("[BD CuMesh] WARNING: preserve_sharp_edges_remesh requested but simple_dual_contour_sharp not available")
            print("[BD CuMesh] Will use standard dual contouring (no edge preservation)")
            preserve_sharp_edges_remesh = False

        # Check for vertex colors in original mesh
        has_colors = False
        orig_colors = None
        if preserve_colors:
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                has_colors = True
                orig_colors = mesh.visual.vertex_colors.copy()
                print(f"[BD CuMesh] Found vertex colors: {orig_colors.shape}")

        try:
            # Check if mesh is "face-split" (no shared vertices)
            verts_per_face = orig_verts / orig_faces if orig_faces > 0 else 0
            is_face_split = verts_per_face > 2.9

            if is_face_split:
                print(f"[BD CuMesh] Detected face-split mesh ({verts_per_face:.2f} verts/face)")
                print("[BD CuMesh] Merging duplicate vertices for proper simplification...")

                work_mesh = trimesh.Trimesh(
                    vertices=mesh.vertices.copy(),
                    faces=mesh.faces.copy(),
                    process=False
                )
                work_mesh.merge_vertices()
                merged_verts = len(work_mesh.vertices)
                merged_faces = len(work_mesh.faces)
                print(f"[BD CuMesh] After merge: {merged_verts:,} verts, {merged_faces:,} faces")

                vertices = torch.tensor(work_mesh.vertices, dtype=torch.float32).cuda()
                faces = torch.tensor(work_mesh.faces, dtype=torch.int32).cuda()
            else:
                vertices = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
                faces = torch.tensor(mesh.faces, dtype=torch.int32).cuda()

            # CuMesh expects Y-up, ComfyUI uses Z-up
            # Convert Z-up to Y-up: swap Y and Z, negate new Y
            vertices_yup = vertices.clone()
            vertices_yup[:, 1], vertices_yup[:, 2] = -vertices[:, 2].clone(), vertices[:, 1].clone()

            # Initialize CuMesh
            cu = cumesh.CuMesh()
            cu.init(vertices_yup, faces)
            print(f"[BD CuMesh] Initialized: {cu.num_vertices} verts, {cu.num_faces} faces")

            # === PHASE 1: INITIAL MESH CLEANING ===
            if fill_holes:
                pre_fill_faces = cu.num_faces
                try:
                    cu.fill_holes(max_hole_perimeter=fill_holes_perimeter)
                    post_fill_faces = cu.num_faces
                    if post_fill_faces > pre_fill_faces * 2:
                        print(f"[BD CuMesh] WARNING: fill_holes exploded faces ({pre_fill_faces} -> {post_fill_faces}), reinitializing...")
                        cu = cumesh.CuMesh()
                        cu.init(vertices_yup, faces)
                    else:
                        print(f"[BD CuMesh] After fill holes: {cu.num_vertices} verts, {cu.num_faces} faces")
                except Exception as e:
                    print(f"[BD CuMesh] Warning: fill_holes failed: {e}")

            # === PHASE 2: REMESH (OPTIONAL) ===
            did_remesh = False
            if remesh and HAS_REMESH:
                print(f"[BD CuMesh] Remeshing (resolution={remesh_resolution}, band={remesh_band}, sharp_edges={preserve_sharp_edges_remesh})...")
                try:
                    # Get current mesh state for remeshing
                    curr_verts, curr_faces = cu.read()

                    # Build BVH for remesh
                    bvh = cumesh.cuBVH(curr_verts, curr_faces)

                    # Calculate remesh parameters based on mesh bounds
                    # Use standard AABB for normalized meshes
                    aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], device='cuda', dtype=torch.float32)
                    center = aabb.mean(dim=0)
                    scale = (aabb[1] - aabb[0]).max().item()

                    # Dual contour remesh with optional sharp edge preservation
                    remesh_start = time.time()
                    new_verts, new_faces = cumesh.remeshing.remesh_narrow_band_dc(
                        curr_verts, curr_faces,
                        center=center,
                        scale=(remesh_resolution + 3 * remesh_band) / remesh_resolution * scale,
                        resolution=remesh_resolution,
                        band=remesh_band,
                        project_back=project_back,
                        verbose=True,
                        bvh=bvh,
                        preserve_sharp_edges=preserve_sharp_edges_remesh,
                        sharp_angle_threshold=remesh_sharp_angle,
                    )
                    remesh_time = time.time() - remesh_start

                    # Reinitialize CuMesh with remeshed geometry
                    cu.init(new_verts, new_faces)
                    did_remesh = True

                    print(f"[BD CuMesh] After remesh: {cu.num_vertices} verts, {cu.num_faces} faces ({remesh_time:.1f}s)")

                    # Clean up BVH
                    del bvh, curr_verts, curr_faces, new_verts, new_faces
                    torch.cuda.empty_cache()

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[BD CuMesh] WARNING: Remesh failed: {e}")
                    print("[BD CuMesh] Continuing with original mesh...")

            # === PHASE 3: MESH CLEANING ===
            if remove_duplicate_faces:
                try:
                    cu.remove_duplicate_faces()
                    cu.remove_degenerate_faces()
                except Exception as e:
                    print(f"[BD CuMesh] Warning: remove_duplicate_faces failed: {e}")

            if repair_non_manifold:
                try:
                    cu.repair_non_manifold_edges()
                except Exception as e:
                    print(f"[BD CuMesh] Warning: repair_non_manifold_edges failed: {e}")

            if remove_small_components:
                try:
                    cu.remove_small_connected_components(small_component_threshold)
                except Exception as e:
                    print(f"[BD CuMesh] Warning: remove_small_connected_components failed: {e}")

            # Fill holes again after cleaning (remesh can create new holes)
            if fill_holes and did_remesh:
                try:
                    cu.fill_holes(max_hole_perimeter=fill_holes_perimeter)
                except Exception:
                    pass

            # Unify face orientations before simplify
            if cu.num_faces > 0:
                try:
                    cu.unify_face_orientations()
                    print("[BD CuMesh] Unified face orientations")
                except Exception as e:
                    print(f"[BD CuMesh] Warning: unify_face_orientations failed (continuing anyway): {e}")

            # === PHASE 4: SIMPLIFY ===
            if cu.num_faces > target_faces:
                print(f"[BD CuMesh] Simplifying to {target_faces} faces...")
                try:
                    cu.simplify(target_faces, verbose=True)
                    print(f"[BD CuMesh] After simplify: {cu.num_vertices} verts, {cu.num_faces} faces")
                except Exception as e:
                    print(f"[BD CuMesh] ERROR: simplify crashed: {e}")
                    return io.NodeOutput(mesh, f"ERROR: CuMesh simplify failed - {e}")

            # Check for bad results
            if cu.num_faces < target_faces * 0.1:
                print(f"[BD CuMesh] WARNING: Simplify reduced to only {cu.num_faces} faces (expected ~{target_faces})")

            if cu.num_faces == 0:
                print("[BD CuMesh] ERROR: Simplify removed all faces! Returning original mesh.")
                return io.NodeOutput(mesh, "ERROR: Simplify removed all faces - mesh may have issues")

            # === PHASE 5: FINAL CLEANUP ===
            if cu.num_faces > 0:
                try:
                    cu.unify_face_orientations()
                except Exception:
                    pass

            try:
                cu.remove_unreferenced_vertices()
            except Exception:
                pass

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

            # Mark sharp edges based on angle AND color boundaries
            if sharp_angle > 0 or color_edge_threshold > 0:
                sharp_count, angle_count, color_count = cls._mark_sharp_edges(
                    result, sharp_angle, color_edge_threshold
                )
                print(f"[BD CuMesh] Marked {sharp_count} sharp edges ({angle_count} by angle >{sharp_angle}°, {color_count} by color >{color_edge_threshold:.0%})")

            # Stats
            total_time = time.time() - start_time
            new_verts = len(result.vertices)
            new_faces = len(result.faces)
            reduction = (1 - new_faces / orig_faces) * 100 if orig_faces > 0 else 0

            status = f"{orig_faces:,} -> {new_faces:,} faces ({reduction:.1f}% reduction)"
            if did_remesh:
                status += " | remeshed"
                if preserve_sharp_edges_remesh:
                    status += " (sharp edges)"
            if has_colors:
                status += " | colors"
            status += f" | {total_time:.1f}s"

            print(f"[BD CuMesh] Complete: {status}")

            # Cleanup GPU
            del vertices, faces, vertices_yup, out_verts, out_faces, cu
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
    def _mark_sharp_edges(cls, mesh, angle_threshold_deg, color_threshold=0.1):
        """
        Mark edges as sharp based on geometry AND color boundaries.

        An edge is marked sharp if:
        - Adjacent face normals differ by more than angle_threshold_deg, OR
        - Adjacent face colors differ by more than color_threshold (0-1 normalized)

        This allows flat areas with same color to merge into single planes,
        while preserving both geometric creases and color boundaries.

        Returns: (total_sharp, angle_sharp, color_sharp) counts
        """
        import numpy as np

        angle_threshold_rad = np.radians(angle_threshold_deg) if angle_threshold_deg > 0 else np.pi

        # Get face adjacency (which faces share each edge)
        if not hasattr(mesh, 'face_adjacency') or mesh.face_adjacency is None:
            print("[BD CuMesh] Computing face adjacency...")
            mesh._cache.clear()  # Force recompute

        face_adj = mesh.face_adjacency
        if face_adj is None or len(face_adj) == 0:
            return 0, 0, 0

        # === ANGLE-BASED SHARP EDGES ===
        angle_sharp_mask = np.zeros(len(face_adj), dtype=bool)
        if angle_threshold_deg > 0:
            face_normals = mesh.face_normals
            n1 = face_normals[face_adj[:, 0]]
            n2 = face_normals[face_adj[:, 1]]

            # Dot product gives cos(angle)
            dots = np.einsum('ij,ij->i', n1, n2)
            dots = np.clip(dots, -1.0, 1.0)
            angles = np.arccos(dots)

            angle_sharp_mask = angles > angle_threshold_rad

        # === COLOR-BASED SHARP EDGES ===
        color_sharp_mask = np.zeros(len(face_adj), dtype=bool)
        if color_threshold > 0:
            # Get face colors (average of vertex colors for each face)
            has_colors = (
                hasattr(mesh.visual, 'vertex_colors') and
                mesh.visual.vertex_colors is not None
            )

            if has_colors:
                vertex_colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0  # RGB only, normalized

                # Compute face colors as average of vertex colors
                face_colors = vertex_colors[mesh.faces].mean(axis=1)  # (n_faces, 3)

                # Get colors of adjacent faces
                c1 = face_colors[face_adj[:, 0]]  # (n_edges, 3)
                c2 = face_colors[face_adj[:, 1]]  # (n_edges, 3)

                # Compute color difference (Euclidean distance in RGB space, normalized to 0-1)
                # Max possible distance is sqrt(3) ≈ 1.73, so normalize by that
                color_diff = np.linalg.norm(c1 - c2, axis=1) / np.sqrt(3)

                color_sharp_mask = color_diff > color_threshold

                print(f"[BD CuMesh] Color edge detection: max diff={color_diff.max():.3f}, mean={color_diff.mean():.3f}")

        # === COMBINE: Sharp if angle OR color boundary ===
        sharp_mask = angle_sharp_mask | color_sharp_mask

        angle_count = np.sum(angle_sharp_mask)
        color_only_count = np.sum(color_sharp_mask & ~angle_sharp_mask)  # Color edges not already angle edges
        sharp_count = np.sum(sharp_mask)

        # Store sharp edges in mesh metadata
        if sharp_count > 0:
            sharp_edges = mesh.face_adjacency_edges[sharp_mask]
            mesh.metadata['sharp_edges'] = sharp_edges
            mesh.metadata['sharp_angle'] = angle_threshold_deg
            mesh.metadata['sharp_color_threshold'] = color_threshold

            # Also store separate masks for debugging/analysis
            mesh.metadata['angle_sharp_edges'] = mesh.face_adjacency_edges[angle_sharp_mask] if angle_count > 0 else np.array([])
            mesh.metadata['color_sharp_edges'] = mesh.face_adjacency_edges[color_sharp_mask] if np.sum(color_sharp_mask) > 0 else np.array([])

        return sharp_count, angle_count, color_only_count


# V3 node list
SIMPLIFY_V3_NODES = [BD_CuMeshSimplify]

# V1 compatibility
SIMPLIFY_NODES = {
    "BD_CuMeshSimplify": BD_CuMeshSimplify,
}

SIMPLIFY_DISPLAY_NAMES = {
    "BD_CuMeshSimplify": "BD CuMesh Simplify",
}
