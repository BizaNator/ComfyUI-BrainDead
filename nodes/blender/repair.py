"""
BD_BlenderRepair - Advanced mesh repair using Blender.

Fixes common mesh issues: holes, duplicate vertices, flipped normals.
"""

import os
import tempfile

from comfy_api.latest import io

from .base import BlenderNodeMixin, REPAIR_SCRIPT, HAS_TRIMESH

if HAS_TRIMESH:
    import trimesh


class BD_BlenderRepair(BlenderNodeMixin, io.ComfyNode):
    """
    Repair mesh issues using Blender's mesh editing tools.

    Features:
    - Fill holes in the mesh
    - Remove duplicate/overlapping vertices
    - Recalculate consistent normals
    - Optional manifold repair for 3D printing
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_BlenderRepair",
            display_name="BD Blender Repair",
            category="🧠BrainDead/Blender",
            description="Advanced mesh repair using Blender. Fixes holes, duplicates, and normals.",
            inputs=[
                io.Mesh.Input("mesh"),
                io.Boolean.Input(
                    "fill_holes",
                    default=True,
                    tooltip="Fill holes in the mesh surface",
                ),
                io.Boolean.Input(
                    "remove_doubles",
                    default=True,
                    tooltip="Merge duplicate/overlapping vertices",
                ),
                io.Float.Input(
                    "merge_distance",
                    default=0.0001,
                    min=0.00001,
                    max=0.1,
                    step=0.0001,
                    tooltip="Distance threshold for merging vertices",
                ),
                io.Boolean.Input(
                    "recalc_normals",
                    default=True,
                    tooltip="Recalculate normals to be consistent (outward facing)",
                ),
                io.Boolean.Input(
                    "make_manifold",
                    default=False,
                    tooltip="Aggressive repair for 3D printing (may change topology)",
                ),
                io.Int.Input(
                    "timeout",
                    default=300,
                    min=30,
                    max=1800,
                    optional=True,
                    tooltip="Maximum processing time in seconds",
                ),
            ],
            outputs=[
                io.Mesh.Output(display_name="mesh"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        fill_holes: bool,
        remove_doubles: bool,
        merge_distance: float,
        recalc_normals: bool,
        make_manifold: bool,
        timeout: int = 300,
    ) -> io.NodeOutput:
        # Check dependencies
        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "ERROR: trimesh not installed")

        available, msg = cls._check_blender()
        if not available:
            return io.NodeOutput(mesh, f"ERROR: {msg}")

        if mesh is None:
            return io.NodeOutput(None, "ERROR: No input mesh")

        # Get original stats
        orig_verts = len(mesh.vertices) if hasattr(mesh, 'vertices') else 0
        orig_faces = len(mesh.faces) if hasattr(mesh, 'faces') else 0

        # Save input mesh to temp file
        input_path = None
        output_path = None
        try:
            input_path = cls._mesh_to_temp_file(mesh, suffix='.ply')
            fd, output_path = tempfile.mkstemp(suffix='.ply')
            os.close(fd)

            # Run Blender repair
            success, message = cls._run_blender_script(
                REPAIR_SCRIPT,
                input_path,
                output_path,
                extra_args={
                    'fill_holes': fill_holes,
                    'remove_doubles': remove_doubles,
                    'merge_distance': merge_distance,
                    'recalc_normals': recalc_normals,
                    'make_manifold': make_manifold,
                },
                timeout=timeout,
            )

            if not success:
                return io.NodeOutput(mesh, f"ERROR: {message}")

            # Load result
            result_mesh = cls._load_mesh_from_file(output_path)

            # Stats
            new_verts = len(result_mesh.vertices)
            new_faces = len(result_mesh.faces)

            # Build status message
            repairs = []
            if fill_holes:
                repairs.append("holes filled")
            if remove_doubles:
                verts_removed = orig_verts - new_verts
                if verts_removed > 0:
                    repairs.append(f"{verts_removed} duplicates removed")
            if recalc_normals:
                repairs.append("normals fixed")
            if make_manifold:
                repairs.append("manifold")

            status = f"Repaired: {', '.join(repairs) if repairs else 'no changes'}"
            status += f" | {new_verts:,} verts, {new_faces:,} faces"

            return io.NodeOutput(result_mesh, status)

        except Exception as e:
            return io.NodeOutput(mesh, f"ERROR: {e}")

        finally:
            # Cleanup temp files
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if output_path and os.path.exists(output_path):
                os.remove(output_path)


# V3 node list
REPAIR_V3_NODES = [BD_BlenderRepair]

# V1 compatibility
REPAIR_NODES = {
    "BD_BlenderRepair": BD_BlenderRepair,
}

REPAIR_DISPLAY_NAMES = {
    "BD_BlenderRepair": "BD Blender Repair",
}
