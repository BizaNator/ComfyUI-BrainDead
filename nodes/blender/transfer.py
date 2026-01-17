"""
BD_BlenderTransferColors - Transfer vertex colors between meshes using Blender.

Uses Blender's data transfer modifier with BVH acceleration for accurate color transfer.
"""

import os
import tempfile

from comfy_api.latest import io

# Import custom TRIMESH type (matches TRELLIS2)
from ..mesh.types import TrimeshInput, TrimeshOutput

from .base import BlenderNodeMixin, TRANSFER_COLORS_SCRIPT, HAS_TRIMESH

if HAS_TRIMESH:
    import trimesh


class BD_BlenderTransferColors(BlenderNodeMixin, io.ComfyNode):
    """
    Transfer vertex colors from source mesh to target mesh using Blender.

    Features:
    - BVH-accelerated nearest-neighbor color lookup
    - Supports meshes with different topology
    - Distance-based falloff for better edge handling
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_BlenderTransferColors",
            display_name="BD Blender Transfer Colors",
            category="🧠BrainDead/Blender",
            description="Transfer vertex colors from source to target mesh using Blender's data transfer.",
            inputs=[
                TrimeshInput("source_mesh", tooltip="Source mesh with vertex colors"),
                TrimeshInput("target_mesh", tooltip="Target mesh to receive colors"),
                io.Float.Input(
                    "max_distance",
                    default=0.1,
                    min=0.001,
                    max=10.0,
                    step=0.01,
                    tooltip="Maximum distance for color transfer (larger = more coverage)",
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
        source_mesh,
        target_mesh,
        max_distance: float,
        timeout: int = 300,
    ) -> io.NodeOutput:
        # Check dependencies
        if not HAS_TRIMESH:
            return io.NodeOutput(target_mesh, "ERROR: trimesh not installed")

        available, msg = cls._check_blender()
        if not available:
            return io.NodeOutput(target_mesh, f"ERROR: {msg}")

        if source_mesh is None:
            return io.NodeOutput(target_mesh, "ERROR: No source mesh")
        if target_mesh is None:
            return io.NodeOutput(None, "ERROR: No target mesh")

        # Get stats
        source_verts = len(source_mesh.vertices) if hasattr(source_mesh, 'vertices') else 0
        target_verts = len(target_mesh.vertices) if hasattr(target_mesh, 'vertices') else 0

        # Save input meshes to temp files
        source_path = None
        target_path = None
        output_path = None
        try:
            source_path = cls._mesh_to_temp_file(source_mesh, suffix='.ply')
            target_path = cls._mesh_to_temp_file(target_mesh, suffix='.ply')
            fd, output_path = tempfile.mkstemp(suffix='.ply')
            os.close(fd)

            # Run Blender color transfer
            success, message, log_lines = cls._run_blender_script(
                TRANSFER_COLORS_SCRIPT,
                source_path,
                output_path,
                extra_args={
                    'target_path': target_path,
                    'max_distance': max_distance,
                },
                timeout=timeout,
            )

            if not success:
                return io.NodeOutput(target_mesh, f"ERROR: {message}")

            # Load result
            result_mesh = cls._load_mesh_from_file(output_path)

            # Check if colors were transferred
            has_colors = (
                hasattr(result_mesh.visual, 'vertex_colors') and
                result_mesh.visual.vertex_colors is not None
            )

            status = f"Colors transferred: {source_verts:,} → {target_verts:,} verts"
            if not has_colors:
                status += " (warning: colors may not have transferred)"

            return io.NodeOutput(result_mesh, status)

        except Exception as e:
            return io.NodeOutput(target_mesh, f"ERROR: {e}")

        finally:
            # Cleanup temp files
            for path in [source_path, target_path, output_path]:
                if path and os.path.exists(path):
                    os.remove(path)


# V3 node list
TRANSFER_V3_NODES = [BD_BlenderTransferColors]

# V1 compatibility
TRANSFER_NODES = {
    "BD_BlenderTransferColors": BD_BlenderTransferColors,
}

TRANSFER_DISPLAY_NAMES = {
    "BD_BlenderTransferColors": "BD Blender Transfer Colors",
}
