"""
BD_CuMeshSimplify - GPU-accelerated mesh simplification using CuMesh.

All operations are CUDA-accelerated via CuMesh:
- Mesh simplification (edge collapse)
- Dual-contouring remesh with sharp edge preservation
- Fill holes, remove duplicates, repair non-manifold edges
- Remove small disconnected components

Note: CuMesh operates on geometry only - vertex colors are NOT preserved.
Use BD_BlenderDecimate with color_field for edge-preserving decimation with colors.
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
    Use BD_BlenderDecimate with color_field afterward for color transfer.
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
4. Use BD_BlenderDecimate with color_field for texture/color transfer

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
                # === MULTI-STAGE SIMPLIFICATION ===
                io.Boolean.Input(
                    "multi_stage",
                    default=True,
                    tooltip="Progressive simplification: 3x target → clean → 1x target → clean. Better topology than single pass.",
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
        # Multi-stage
        multi_stage: bool = True,
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
                    # 10% margin keeps surface safely inside the voxel domain
                    effective_scale = scale * 1.1

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
                if multi_stage and cu.num_faces > target_faces * 3:
                    # Multi-stage progressive simplification (from o_voxel reference):
                    # Stage 1: 3x target → clean
                    # Stage 2: 1x target → clean
                    intermediate_target = target_faces * 3
                    print(f"[BD CuMesh] Multi-stage: {cu.num_faces} → {intermediate_target} → {target_faces} faces")
                    try:
                        # Stage 1: coarse reduction
                        cu.simplify(intermediate_target, verbose=True)
                        print(f"[BD CuMesh] Stage 1 result: {cu.num_vertices} verts, {cu.num_faces} faces")

                        # Intermediate cleanup
                        try:
                            cu.remove_duplicate_faces()
                            cu.remove_degenerate_faces()
                            cu.repair_non_manifold_edges()
                            cu.remove_small_connected_components(small_component_threshold)
                            cu.fill_holes(max_hole_perimeter=fill_holes_perimeter)
                            cu.unify_face_orientations()
                        except Exception:
                            pass  # Non-critical cleanup

                        # Stage 2: fine reduction to target
                        if cu.num_faces > target_faces:
                            cu.simplify(target_faces, verbose=True)
                            print(f"[BD CuMesh] Stage 2 result: {cu.num_vertices} verts, {cu.num_faces} faces")

                    except Exception as e:
                        print(f"[BD CuMesh] ERROR: multi-stage simplify crashed: {e}")
                        return io.NodeOutput(mesh, f"ERROR: CuMesh simplify failed - {e}")
                else:
                    # Single-pass simplification
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


def _save_quad_obj(vertices_np: np.ndarray, quad_faces_np: np.ndarray, path: str) -> None:
    """Write an OBJ file preserving quad faces (4-vertex polygons)."""
    with open(path, 'w') as f:
        f.write("# BD CuMesh Quad Remesh\n")
        for v in vertices_np:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for q in quad_faces_np:
            # OBJ is 1-indexed
            f.write(f"f {q[0]+1} {q[1]+1} {q[2]+1} {q[3]+1}\n")


class BD_CuMeshQuadRemesh(io.ComfyNode):
    """
    GPU-accelerated quad remesh via dual contouring.

    Returns the native quad topology from the DC algorithm before triangle
    splitting. Outputs a triangulated TRIMESH for pipeline use plus saves
    a quad OBJ file for Blender/Maya/animation workflows.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CuMeshQuadRemesh",
            display_name="BD CuMesh Quad Remesh",
            category="🧠BrainDead/Mesh",
            description="""GPU dual-contouring remesh that preserves quad topology.

Outputs:
- mesh: triangulated TRIMESH for pipeline use (quads split to tris)
- quad_obj_path: path to OBJ file with true quad faces for Blender/animation

Quad OBJ is the primary deliverable for rigging and subdivision workflows.
The TRIMESH output is a triangulated copy for downstream ComfyUI nodes.""",
            inputs=[
                TrimeshInput("mesh"),
                io.Int.Input(
                    "resolution",
                    default=512,
                    min=128,
                    max=2048,
                    step=64,
                    tooltip="Voxel grid resolution. Higher = more quads, more detail.",
                ),
                io.Float.Input(
                    "band",
                    default=1.0,
                    min=0.5,
                    max=4.0,
                    step=0.5,
                    tooltip="Narrow band width in voxel units around the surface.",
                ),
                io.Float.Input(
                    "project_back",
                    default=0.9,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    tooltip="How much to snap quad vertices back to the original surface.",
                ),
                io.Boolean.Input(
                    "preserve_sharp_edges",
                    default=True,
                    tooltip="Use sharp-edge-aware dual contouring kernel.",
                ),
                io.Float.Input(
                    "sharp_angle",
                    default=30.0,
                    min=10.0,
                    max=90.0,
                    step=5.0,
                    tooltip="Dihedral angle threshold (degrees) for sharp edge detection.",
                ),
                io.String.Input(
                    "filename",
                    default="quad_mesh",
                    tooltip="Base filename for the quad OBJ output.",
                ),
                io.String.Input(
                    "name_prefix",
                    default="",
                    optional=True,
                    tooltip="Optional subdirectory prefix (e.g. 'Project/Head').",
                ),
            ],
            outputs=[
                TrimeshOutput(display_name="mesh"),
                io.String.Output(display_name="quad_obj_path"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        resolution: int = 512,
        band: float = 1.0,
        project_back: float = 0.9,
        preserve_sharp_edges: bool = True,
        sharp_angle: float = 30.0,
        filename: str = "quad_mesh",
        name_prefix: str = "",
    ) -> io.NodeOutput:
        import folder_paths
        from glob import glob

        if not HAS_TORCH:
            return io.NodeOutput(mesh, "", "ERROR: torch not installed")
        if not HAS_CUMESH:
            return io.NodeOutput(mesh, "", "ERROR: cumesh not installed")
        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "", "ERROR: trimesh not installed")
        if mesh is None:
            return io.NodeOutput(None, "", "ERROR: No input mesh")

        start_time = time.time()
        orig_verts = len(mesh.vertices)
        orig_faces = len(mesh.faces)
        print(f"[BD QuadRemesh] Input: {orig_verts:,} verts, {orig_faces:,} faces @ res={resolution}")

        try:
            # Merge split vertices if needed
            verts_per_face = orig_verts / orig_faces if orig_faces > 0 else 0
            if verts_per_face > 2.9:
                work = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
                work.merge_vertices()
            else:
                work = mesh

            vertices = torch.tensor(work.vertices, dtype=torch.float32).cuda()
            faces = torch.tensor(work.faces, dtype=torch.int32).cuda()

            # Z-up → Y-up for CuMesh
            v_yup = vertices.clone()
            v_yup[:, 1], v_yup[:, 2] = -vertices[:, 2].clone(), vertices[:, 1].clone()

            # Compute mesh bounds for scale/center
            mesh_min = v_yup.min(dim=0).values
            mesh_max = v_yup.max(dim=0).values
            center = (mesh_min + mesh_max) / 2
            scale = (mesh_max - mesh_min).max().item() * 1.1

            bvh = cumesh.cuBVH(v_yup, faces)

            new_verts, quad_faces = cumesh.remeshing.remesh_narrow_band_dc(
                v_yup, faces,
                center=center,
                scale=scale,
                resolution=resolution,
                band=band,
                project_back=project_back,
                verbose=True,
                bvh=bvh,
                preserve_sharp_edges=preserve_sharp_edges,
                sharp_angle_threshold=sharp_angle,
                return_quads=True,
            )

            del bvh, v_yup, vertices, faces
            torch.cuda.empty_cache()

            verts_np = new_verts.cpu().numpy().astype(np.float32)
            quads_np = quad_faces.cpu().numpy()

            # Y-up → Z-up for output
            verts_np[:, 1], verts_np[:, 2] = verts_np[:, 2].copy(), -verts_np[:, 1].copy()

            # Triangulate quads for TRIMESH output (split each quad into 2 tris)
            tri_faces = np.concatenate([
                quads_np[:, [0, 1, 2]],
                quads_np[:, [0, 2, 3]],
            ], axis=0)
            result = trimesh.Trimesh(vertices=verts_np, faces=tri_faces, process=False)

            # Resolve output path
            output_base = folder_paths.get_output_directory()
            full_name = f"{name_prefix}_{filename}" if name_prefix else filename
            full_name = full_name.replace('\\', '/')
            if '/' in full_name:
                subdir, base = full_name.rsplit('/', 1)
                output_dir = os.path.join(output_base, subdir)
            else:
                output_dir = output_base
                base = full_name
            os.makedirs(output_dir, exist_ok=True)

            pattern = os.path.join(output_dir, f"{base}_*.obj")
            existing = glob(pattern)
            nums = []
            for p in existing:
                try:
                    nums.append(int(os.path.splitext(os.path.basename(p))[0].rsplit('_', 1)[1]))
                except (ValueError, IndexError):
                    pass
            next_num = max(nums, default=0) + 1
            obj_path = os.path.join(output_dir, f"{base}_{next_num:03d}.obj")

            _save_quad_obj(verts_np, quads_np, obj_path)

            elapsed = time.time() - start_time
            status = (f"{orig_faces:,} tris → {len(quads_np):,} quads ({len(tri_faces):,} tris) "
                      f"@ res={resolution} | {elapsed:.1f}s | {obj_path}")
            print(f"[BD QuadRemesh] {status}")

            gc.collect()
            torch.cuda.empty_cache()

            return io.NodeOutput(result, obj_path, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            return io.NodeOutput(mesh, "", f"ERROR: {e}")


# V3 node list
SIMPLIFY_V3_NODES = [BD_CuMeshSimplify, BD_CuMeshQuadRemesh]

# V1 compatibility
SIMPLIFY_NODES = {
    "BD_CuMeshSimplify": BD_CuMeshSimplify,
    "BD_CuMeshQuadRemesh": BD_CuMeshQuadRemesh,
}

SIMPLIFY_DISPLAY_NAMES = {
    "BD_CuMeshSimplify": "BD CuMesh Simplify",
    "BD_CuMeshQuadRemesh": "BD CuMesh Quad Remesh",
}
