"""
V3 API Mesh processing nodes for repair and decimation.

BD_MeshRepair - Repair mesh topology using PyMeshLab
BD_SmartDecimate - Edge-preserving mesh decimation
"""

import os
import time

from comfy_api.latest import io

# Check for optional trimesh support
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

# Import custom TRIMESH type (matches TRELLIS2)
from .types import TrimeshInput, TrimeshOutput


class BD_MeshRepair(io.ComfyNode):
    """
    Repair mesh topology using PyMeshLab.

    Fixes common mesh issues like holes, duplicate vertices/faces,
    degenerate triangles, and inconsistent normals.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_MeshRepair",
            display_name="BD Mesh Repair",
            category="🧠BrainDead/Mesh",
            description="Repair mesh topology using PyMeshLab. Fixes holes, duplicates, degenerates, and normals.",
            inputs=[
                TrimeshInput("mesh"),
                io.Boolean.Input("remove_duplicates", default=True, optional=True, tooltip="Merge duplicate/close vertices"),
                io.Boolean.Input("remove_degenerate", default=True, optional=True, tooltip="Remove zero-area faces and duplicates"),
                io.Boolean.Input("close_holes", default=True, optional=True, tooltip="Close holes in mesh"),
                io.Int.Input("max_hole_edges", default=100, min=3, max=1000, step=10, optional=True, tooltip="Max edges for hole closing"),
                io.Boolean.Input("repair_normals", default=True, optional=True, tooltip="Fix face orientation and recompute normals"),
                io.Float.Input("merge_threshold", default=0.0001, min=0.00001, max=0.01, step=0.00001, optional=True, tooltip="Distance threshold for vertex merging"),
            ],
            outputs=[
                TrimeshOutput(display_name="mesh"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, mesh, remove_duplicates: bool = True, remove_degenerate: bool = True,
                close_holes: bool = True, max_hole_edges: int = 100, repair_normals: bool = True,
                merge_threshold: float = 0.0001) -> io.NodeOutput:
        import numpy as np
        import tempfile

        try:
            import pymeshlab
        except ImportError:
            return io.NodeOutput(mesh, "ERROR: pymeshlab not installed")

        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "ERROR: trimesh not installed")

        if mesh is None:
            return io.NodeOutput(None, "ERROR: mesh is None")

        start_time = time.time()
        changes = []

        # Get initial stats
        initial_verts = len(mesh.vertices)
        initial_faces = len(mesh.faces) if hasattr(mesh, 'faces') and mesh.faces is not None else 0

        print(f"[BD Mesh Repair] Input: {initial_verts} vertices, {initial_faces} faces")

        # Save mesh to temp file
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            temp_path = tmp.name
            mesh.export(temp_path, file_type='ply')

        try:
            # Load into PyMeshLab
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(temp_path)

            if remove_duplicates:
                before = ms.current_mesh().vertex_number()
                ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(merge_threshold * 100))
                after = ms.current_mesh().vertex_number()
                if before != after:
                    changes.append(f"merged {before - after} duplicate verts")
                    print(f"[BD Mesh Repair] Merged {before - after} duplicate vertices")

            if remove_degenerate:
                before_f = ms.current_mesh().face_number()
                ms.meshing_remove_duplicate_faces()
                ms.meshing_remove_null_faces()
                after_f = ms.current_mesh().face_number()
                if before_f != after_f:
                    changes.append(f"removed {before_f - after_f} degenerate faces")
                    print(f"[BD Mesh Repair] Removed {before_f - after_f} degenerate faces")

            # Repair non-manifold geometry BEFORE closing holes (required for hole closing)
            try:
                before_v = ms.current_mesh().vertex_number()
                before_f = ms.current_mesh().face_number()
                ms.meshing_repair_non_manifold_edges()
                ms.meshing_repair_non_manifold_vertices()
                after_v = ms.current_mesh().vertex_number()
                after_f = ms.current_mesh().face_number()
                if before_v != after_v or before_f != after_f:
                    changes.append(f"fixed non-manifold ({before_v - after_v} verts, {before_f - after_f} faces)")
                    print(f"[BD Mesh Repair] Fixed non-manifold: {before_v - after_v} verts, {before_f - after_f} faces removed")
                else:
                    print(f"[BD Mesh Repair] Non-manifold repair: no changes needed")
            except Exception as e:
                print(f"[BD Mesh Repair] Warning: non-manifold repair failed: {e}")

            if close_holes:
                before_f = ms.current_mesh().face_number()
                try:
                    ms.meshing_close_holes(maxholesize=max_hole_edges)
                    after_f = ms.current_mesh().face_number()
                    if after_f != before_f:
                        changes.append(f"closed holes (+{after_f - before_f} faces)")
                        print(f"[BD Mesh Repair] Closed holes, added {after_f - before_f} faces")
                except Exception as e:
                    print(f"[BD Mesh Repair] Warning: hole closing failed: {e}")

            if repair_normals:
                try:
                    ms.meshing_re_orient_faces_coherently()
                    ms.compute_normal_per_vertex()
                    changes.append("normals repaired")
                    print(f"[BD Mesh Repair] Reoriented faces and recomputed normals")
                except Exception as e:
                    print(f"[BD Mesh Repair] Warning: normal repair failed: {e}")

            # Export repaired mesh
            repaired_path = temp_path.replace(".ply", "_repaired.ply")
            ms.save_current_mesh(repaired_path)

            # Load back into trimesh
            repaired_mesh = trimesh.load(repaired_path, process=False)

            # Clean up temp files
            os.unlink(temp_path)
            os.unlink(repaired_path)

            # Final stats
            final_verts = len(repaired_mesh.vertices)
            final_faces = len(repaired_mesh.faces) if hasattr(repaired_mesh, 'faces') else 0
            total_time = time.time() - start_time

            if not changes:
                changes.append("no changes needed")

            status = f"Repaired: {initial_verts}->{final_verts} verts, {initial_faces}->{final_faces} faces | {', '.join(changes)} | {total_time:.1f}s"
            print(f"[BD Mesh Repair] {status}")

            return io.NodeOutput(repaired_mesh, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return io.NodeOutput(mesh, f"ERROR: {e}")


class BD_SmartDecimate(io.ComfyNode):
    """
    Edge-preserving mesh decimation using PyMeshLab.

    Decimates using quadric edge collapse while respecting boundary constraints.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_SmartDecimate",
            display_name="BD Smart Decimate",
            category="🧠BrainDead/Mesh",
            description="Edge-preserving mesh decimation using PyMeshLab quadric edge collapse.",
            inputs=[
                TrimeshInput("mesh"),
                io.Int.Input("target_faces", default=50000, min=100, max=10000000, step=1000),
                io.Boolean.Input("preserve_boundary", default=True, optional=True, tooltip="Preserve mesh boundary edges"),
                io.Boolean.Input("preserve_topology", default=True, optional=True, tooltip="Preserve mesh topology"),
                io.Float.Input("quality_threshold", default=0.3, min=0.0, max=1.0, step=0.1, optional=True, tooltip="Quality threshold for edge collapse"),
                io.Boolean.Input("planar_quadric", default=True, optional=True, tooltip="Use planar simplification for flat regions"),
            ],
            outputs=[
                TrimeshOutput(display_name="mesh"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, mesh, target_faces: int, preserve_boundary: bool = True,
                preserve_topology: bool = True, quality_threshold: float = 0.3,
                planar_quadric: bool = True) -> io.NodeOutput:
        import numpy as np
        import tempfile

        try:
            import pymeshlab
        except ImportError:
            return io.NodeOutput(mesh, "ERROR: pymeshlab not installed")

        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "ERROR: trimesh not installed")

        if mesh is None:
            return io.NodeOutput(None, "ERROR: mesh is None")

        start_time = time.time()

        # Get initial stats
        initial_faces = len(mesh.faces) if hasattr(mesh, 'faces') and mesh.faces is not None else 0
        initial_verts = len(mesh.vertices)

        if initial_faces <= target_faces:
            return io.NodeOutput(mesh, f"Mesh already has {initial_faces} faces (<= target {target_faces})")

        print(f"[BD Smart Decimate] Input: {initial_verts} vertices, {initial_faces} faces")
        print(f"[BD Smart Decimate] Target: {target_faces} faces ({100*target_faces/initial_faces:.1f}%)")

        # Check for vertex colors
        has_colors = hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None

        # Save mesh to temp file
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            temp_path = tmp.name
            mesh.export(temp_path, file_type='ply')

        try:
            # Load into PyMeshLab
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(temp_path)

            # Apply quadric edge collapse decimation
            print(f"[BD Smart Decimate] Running quadric edge collapse decimation...")

            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=target_faces,
                qualitythr=quality_threshold,
                preserveboundary=preserve_boundary,
                preservetopology=preserve_topology,
                planarquadric=planar_quadric,
                autoclean=True
            )

            # Export decimated mesh
            decimated_path = temp_path.replace(".ply", "_decimated.ply")
            ms.save_current_mesh(decimated_path)

            # Load back into trimesh
            decimated_mesh = trimesh.load(decimated_path, process=False)

            # Clean up temp files
            os.unlink(temp_path)
            os.unlink(decimated_path)

            # Final stats
            final_verts = len(decimated_mesh.vertices)
            final_faces = len(decimated_mesh.faces) if hasattr(decimated_mesh, 'faces') else 0
            total_time = time.time() - start_time
            reduction = 100 * (1 - final_faces / initial_faces)

            status = f"Decimated: {initial_faces}->{final_faces} faces ({reduction:.1f}% reduction), {initial_verts}->{final_verts} verts | {total_time:.1f}s"
            if has_colors:
                status += " | NOTE: Vertex colors lost - use BD_TransferVertexColors to restore"

            print(f"[BD Smart Decimate] {status}")

            return io.NodeOutput(decimated_mesh, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return io.NodeOutput(mesh, f"ERROR: {e}")


# V3 node list for extension
MESH_PROCESSING_V3_NODES = [BD_MeshRepair, BD_SmartDecimate]

# V1 compatibility - NODE_CLASS_MAPPINGS dict
MESH_PROCESSING_NODES = {
    "BD_MeshRepair": BD_MeshRepair,
    "BD_SmartDecimate": BD_SmartDecimate,
}

MESH_PROCESSING_DISPLAY_NAMES = {
    "BD_MeshRepair": "BD Mesh Repair",
    "BD_SmartDecimate": "BD Smart Decimate",
}
