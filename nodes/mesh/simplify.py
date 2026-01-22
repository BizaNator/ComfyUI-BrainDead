"""
BD_CuMeshSimplify - GPU-accelerated mesh simplification using CuMesh.

All operations are CUDA-accelerated via CuMesh:
- Mesh simplification (edge collapse)
- Dual-contouring remesh with sharp edge preservation
- Fill holes, remove duplicates, repair non-manifold edges
- Remove small disconnected components

Note: CuMesh operates on geometry only - vertex colors are NOT preserved.
Use BD_BakeTextures or similar for color/texture transfer after simplification.
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
    GPU-accelerated mesh simplification using CuMesh (CUDA).

    All operations are CUDA-accelerated. Geometry only - no color preservation.
    Use BD_BakeTextures afterward for texture/color transfer.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CuMeshSimplify",
            display_name="BD CuMesh Simplify",
            category="🧠BrainDead/Mesh",
            description="""GPU-accelerated mesh simplification using CuMesh (CUDA).

All operations are CUDA-accelerated:
- target_faces → cu.simplify() edge collapse
- remesh → cumesh.remeshing.remesh_narrow_band_dc()
- preserve_sharp_edges_remesh → sharp edge preservation during remesh
- fill_holes → cu.fill_holes()
- remove_duplicate_faces → cu.remove_duplicate_faces() + cu.remove_degenerate_faces()
- repair_non_manifold → cu.repair_non_manifold_edges()
- remove_small_components → cu.remove_small_connected_components()

Typical workflow for TRELLIS2 output:
1. Enable remesh + preserve_sharp_edges for cleaner topology
2. Enable mesh cleaning to fix topology issues
3. Target 5k-50k faces depending on use case
4. Use BD_BakeTextures afterward for texture transfer

Note: CuMesh operates on geometry only - vertex colors are NOT preserved.""",
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
                    max=8192,
                    step=64,
                    tooltip="Voxel grid resolution for remeshing. Higher = more detail but slower. VRAM usage scales with resolution^3.",
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

                    # Calculate remesh parameters from ACTUAL mesh bounds
                    mesh_min = curr_verts.min(dim=0).values
                    mesh_max = curr_verts.max(dim=0).values
                    center = (mesh_min + mesh_max) / 2
                    mesh_extent = (mesh_max - mesh_min).max().item()

                    # Add small padding to avoid clipping
                    scale = mesh_extent * 1.05

                    print(f"[BD CuMesh] Mesh bounds: min={mesh_min.cpu().numpy()}, max={mesh_max.cpu().numpy()}, extent={mesh_extent:.4f}")

                    # Warn if remesh resolution seems too high for mesh detail
                    # Estimate: ~6 faces per voxel cell on surface is reasonable
                    estimated_surface_voxels = curr_faces.shape[0] / 6
                    estimated_resolution = int(estimated_surface_voxels ** (1/2))  # Rough estimate
                    if remesh_resolution > estimated_resolution * 2:
                        print(f"[BD CuMesh] WARNING: remesh_resolution={remesh_resolution} may be too high")
                        print(f"[BD CuMesh]   Mesh has ~{curr_faces.shape[0]} faces, estimated detail supports ~{estimated_resolution} resolution")
                        print(f"[BD CuMesh]   Consider using resolution <= {min(remesh_resolution, estimated_resolution * 2)} to avoid artifacts")

                    # Dual contour remesh with optional sharp edge preservation
                    # Scale needs to account for band width expansion
                    effective_scale = (remesh_resolution + 3 * remesh_band) / remesh_resolution * scale

                    remesh_start = time.time()
                    new_verts, new_faces = cumesh.remeshing.remesh_narrow_band_dc(
                        curr_verts, curr_faces,
                        center=center,
                        scale=effective_scale,
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


# V3 node list
SIMPLIFY_V3_NODES = [BD_CuMeshSimplify]

# V1 compatibility
SIMPLIFY_NODES = {
    "BD_CuMeshSimplify": BD_CuMeshSimplify,
}

SIMPLIFY_DISPLAY_NAMES = {
    "BD_CuMeshSimplify": "BD CuMesh Simplify",
}
