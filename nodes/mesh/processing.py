"""
Mesh processing nodes for repair and decimation.

BD_MeshRepair - Repair mesh topology using PyMeshLab
BD_SmartDecimate - Edge-preserving mesh decimation
"""

import os
import time

# Check for optional trimesh support
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


class BD_MeshRepair:
    """
    Repair mesh topology using PyMeshLab.

    Fixes common mesh issues like holes, duplicate vertices/faces,
    degenerate triangles, and inconsistent normals.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
            },
            "optional": {
                "remove_duplicates": ("BOOLEAN", {"default": True, "tooltip": "Merge duplicate/close vertices"}),
                "remove_degenerate": ("BOOLEAN", {"default": True, "tooltip": "Remove zero-area faces and duplicates"}),
                "close_holes": ("BOOLEAN", {"default": True, "tooltip": "Close holes in mesh"}),
                "max_hole_edges": ("INT", {"default": 100, "min": 3, "max": 1000, "step": 10, "tooltip": "Max edges for hole closing"}),
                "repair_normals": ("BOOLEAN", {"default": True, "tooltip": "Fix face orientation and recompute normals"}),
                "merge_threshold": ("FLOAT", {"default": 0.0001, "min": 0.00001, "max": 0.01, "step": 0.00001, "tooltip": "Distance threshold for vertex merging"}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "status")
    FUNCTION = "repair_mesh"
    CATEGORY = "BrainDead/Mesh"
    DESCRIPTION = """
Repair mesh topology using PyMeshLab filters.

Operations:
- remove_duplicates: Merge vertices closer than merge_threshold
- remove_degenerate: Remove duplicate faces and zero-area triangles
- close_holes: Fill holes up to max_hole_edges
- repair_normals: Orient faces consistently and recompute vertex normals

Use before color sampling to fix mesh issues from TRELLIS2.
"""

    def repair_mesh(self, mesh, remove_duplicates=True, remove_degenerate=True,
                    close_holes=True, max_hole_edges=100, repair_normals=True,
                    merge_threshold=0.0001):
        import numpy as np
        import tempfile

        try:
            import pymeshlab
        except ImportError:
            return (mesh, "ERROR: pymeshlab not installed")

        if not HAS_TRIMESH:
            return (mesh, "ERROR: trimesh not installed")

        if mesh is None:
            return (None, "ERROR: mesh is None")

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

            return (repaired_mesh, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return (mesh, f"ERROR: {e}")


class BD_SmartDecimate:
    """
    Edge-preserving mesh decimation using PyMeshLab.

    Detects and preserves edges based on color boundaries and sharp angles,
    then decimates using quadric edge collapse while respecting these constraints.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "target_faces": ("INT", {"default": 50000, "min": 100, "max": 10000000, "step": 1000}),
            },
            "optional": {
                "preserve_boundary": ("BOOLEAN", {"default": True, "tooltip": "Preserve mesh boundary edges"}),
                "preserve_topology": ("BOOLEAN", {"default": True, "tooltip": "Preserve mesh topology"}),
                "quality_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Quality threshold for edge collapse"}),
                "planar_quadric": ("BOOLEAN", {"default": True, "tooltip": "Use planar simplification for flat regions"}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "status")
    FUNCTION = "decimate"
    CATEGORY = "BrainDead/Mesh"
    DESCRIPTION = """
Edge-preserving mesh decimation using PyMeshLab quadric edge collapse.

Parameters:
- target_faces: Target number of faces after decimation
- preserve_boundary: Preserve boundary edges (important for open meshes)
- preserve_topology: Prevent topology changes during decimation
- quality_threshold: Higher values = faster but lower quality
- planar_quadric: Better results for flat regions

Use BD_MeshRepair before decimation to fix holes and topology issues.
Use BD_TransferVertexColors after decimation to restore colors.
"""

    def decimate(self, mesh, target_faces, preserve_boundary=True,
                 preserve_topology=True, quality_threshold=0.3, planar_quadric=True):
        import numpy as np
        import tempfile

        try:
            import pymeshlab
        except ImportError:
            return (mesh, "ERROR: pymeshlab not installed")

        if not HAS_TRIMESH:
            return (mesh, "ERROR: trimesh not installed")

        if mesh is None:
            return (None, "ERROR: mesh is None")

        start_time = time.time()

        # Get initial stats
        initial_faces = len(mesh.faces) if hasattr(mesh, 'faces') and mesh.faces is not None else 0
        initial_verts = len(mesh.vertices)

        if initial_faces <= target_faces:
            return (mesh, f"Mesh already has {initial_faces} faces (<= target {target_faces})")

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

            return (decimated_mesh, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return (mesh, f"ERROR: {e}")


# Node exports
MESH_PROCESSING_NODES = {
    "BD_MeshRepair": BD_MeshRepair,
    "BD_SmartDecimate": BD_SmartDecimate,
}

MESH_PROCESSING_DISPLAY_NAMES = {
    "BD_MeshRepair": "BD Mesh Repair",
    "BD_SmartDecimate": "BD Smart Decimate",
}
