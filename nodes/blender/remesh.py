"""
BD_BlenderRemesh - Mesh remeshing using Blender's remesh modifier.

Provides voxel-based and quad-based remeshing options.
"""

import os
import tempfile

from comfy_api.latest import io

# Import custom TRIMESH type (matches TRELLIS2)
from ..mesh.types import TrimeshInput, TrimeshOutput

from .base import BlenderNodeMixin, REMESH_SCRIPT, HAS_TRIMESH

if HAS_TRIMESH:
    import trimesh


class BD_BlenderRemesh(BlenderNodeMixin, io.ComfyNode):
    """
    Remesh a mesh using Blender's remesh modifier.

    Features:
    - Voxel remeshing for uniform quad topology
    - Quad/Sharp remeshing for edge-preserving topology
    - Optional smoothing pass
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_BlenderRemesh",
            display_name="BD Blender Remesh",
            category="🧠BrainDead/Blender",
            description="Remesh using Blender. Creates clean, uniform topology from any mesh.",
            inputs=[
                TrimeshInput("mesh"),
                io.Combo.Input(
                    "mode",
                    options=["VOXEL", "QUAD", "SHARP"],
                    default="VOXEL",
                    tooltip="VOXEL=uniform, QUAD=quad-based, SHARP=edge-preserving",
                ),
                io.Float.Input(
                    "voxel_size",
                    default=0.01,
                    min=0.0001,
                    max=1.0,
                    step=0.001,
                    tooltip="Voxel size for VOXEL mode (smaller = more detail)",
                ),
                io.Int.Input(
                    "octree_depth",
                    default=6,
                    min=1,
                    max=12,
                    tooltip="Octree depth for QUAD/SHARP modes (higher = more detail)",
                ),
                io.Int.Input(
                    "smooth_iterations",
                    default=0,
                    min=0,
                    max=50,
                    tooltip="Optional smoothing passes after remesh",
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
                TrimeshOutput(display_name="mesh"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        mode: str,
        voxel_size: float,
        octree_depth: int,
        smooth_iterations: int,
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

            # Run Blender remesh
            success, message = cls._run_blender_script(
                REMESH_SCRIPT,
                input_path,
                output_path,
                extra_args={
                    'mode': mode,
                    'voxel_size': voxel_size,
                    'octree_depth': octree_depth,
                    'smooth_iterations': smooth_iterations,
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

            status = f"Remeshed ({mode}): {orig_faces:,} → {new_faces:,} faces"
            if smooth_iterations > 0:
                status += f" + {smooth_iterations} smooth"
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
REMESH_V3_NODES = [BD_BlenderRemesh]

# V1 compatibility
REMESH_NODES = {
    "BD_BlenderRemesh": BD_BlenderRemesh,
}

REMESH_DISPLAY_NAMES = {
    "BD_BlenderRemesh": "BD Blender Remesh",
}
