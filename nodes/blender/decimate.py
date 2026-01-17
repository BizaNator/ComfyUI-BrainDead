"""
BD_BlenderDecimate - Mesh decimation using Blender's decimate modifier.

Provides high-quality edge-preserving mesh simplification.
"""

import os
import tempfile

from comfy_api.latest import io

from .base import BlenderNodeMixin, DECIMATE_SCRIPT, HAS_TRIMESH

if HAS_TRIMESH:
    import trimesh


class BD_BlenderDecimate(BlenderNodeMixin, io.ComfyNode):
    """
    Decimate (simplify) a mesh using Blender's decimate modifier.

    Features:
    - Edge-preserving collapse decimation
    - Symmetry-aware decimation option
    - Preserves vertex colors where possible
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_BlenderDecimate",
            display_name="BD Blender Decimate",
            category="🧠BrainDead/Blender",
            description="High-quality mesh decimation using Blender. Reduces polygon count while preserving shape.",
            inputs=[
                io.Mesh.Input("mesh"),
                io.Float.Input(
                    "ratio",
                    default=0.5,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    tooltip="Target ratio (0.5 = reduce to 50% of faces)",
                ),
                io.Boolean.Input(
                    "use_collapse",
                    default=True,
                    tooltip="Use collapse decimation (best quality) vs unsubdivide",
                ),
                io.Boolean.Input(
                    "use_symmetry",
                    default=False,
                    tooltip="Preserve symmetry during decimation",
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
        ratio: float,
        use_collapse: bool,
        use_symmetry: bool,
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

            # Run Blender decimate
            success, message = cls._run_blender_script(
                DECIMATE_SCRIPT,
                input_path,
                output_path,
                extra_args={
                    'ratio': ratio,
                    'use_collapse': use_collapse,
                    'use_symmetry': use_symmetry,
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
            reduction = (1 - new_faces / orig_faces) * 100 if orig_faces > 0 else 0

            status = f"Decimated: {orig_faces:,} → {new_faces:,} faces ({reduction:.1f}% reduction)"
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
DECIMATE_V3_NODES = [BD_BlenderDecimate]

# V1 compatibility
DECIMATE_NODES = {
    "BD_BlenderDecimate": BD_BlenderDecimate,
}

DECIMATE_DISPLAY_NAMES = {
    "BD_BlenderDecimate": "BD Blender Decimate",
}
