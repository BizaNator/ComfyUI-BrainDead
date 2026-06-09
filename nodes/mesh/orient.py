"""
BD_OrientMesh - Rotate a mesh by X/Y/Z degrees, non-destructively.

Single responsibility: fix orientation of a finished mesh. Applied AFTER texturing /
vertex-color sampling, so UVs, the baked PBR material, and COLOR_0 vertex colors all ride
along unchanged (a rotation moves vertices + normals only). Use this to bring a mesh into the
Z-up / forward convention your engine expects.

Pixal3D note: Pixal3D's mesh comes out lying down / facing up — set **rotate_x = 180** to stand
it upright facing forward (Blender -Y), matching the Trellis2 convention. (Trellis2 already
emits the correct orientation, so it needs no rotation.)
"""

import numpy as np

from comfy_api.latest import io

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

from .types import TrimeshInput, TrimeshOutput


class BD_OrientMesh(io.ComfyNode):
    """Rotate a mesh by X/Y/Z degrees, preserving UV + material + vertex colors."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_OrientMesh",
            display_name="BD Orient Mesh",
            category="🧠BrainDead/Mesh",
            description="Rotate a finished mesh by X/Y/Z degrees (applied in that order) about its "
                        "centroid. Non-destructive: UVs, baked PBR material, and COLOR_0 vertex "
                        "colors are preserved (only vertices/normals move). For Pixal3D → Unreal, "
                        "set rotate_x=180 to stand the character upright facing forward.",
            inputs=[
                TrimeshInput("mesh"),
                io.Float.Input("rotate_x", default=0.0, min=-360.0, max=360.0, step=1.0,
                               tooltip="Degrees about X. Pixal3D→Unreal: 180."),
                io.Float.Input("rotate_y", default=0.0, min=-360.0, max=360.0, step=1.0,
                               tooltip="Degrees about Y."),
                io.Float.Input("rotate_z", default=0.0, min=-360.0, max=360.0, step=1.0,
                               tooltip="Degrees about Z."),
                io.Boolean.Input("about_centroid", default=True, optional=True,
                                 tooltip="Rotate about the mesh centroid (True) or the world origin (False)."),
            ],
            outputs=[
                TrimeshOutput(display_name="mesh"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, mesh, rotate_x=0.0, rotate_y=0.0, rotate_z=0.0,
                about_centroid=True) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "ERROR: trimesh not installed")
        if mesh is None:
            return io.NodeOutput(None, "ERROR: mesh is None")

        if rotate_x == 0.0 and rotate_y == 0.0 and rotate_z == 0.0:
            return io.NodeOutput(mesh, "BD Orient Mesh: no rotation (all axes 0)")

        center = mesh.centroid if about_centroid else np.zeros(3)
        R = trimesh.transformations
        # X then Y then Z, all about `center`.
        M = np.eye(4)
        for deg, axis in ((rotate_x, [1, 0, 0]), (rotate_y, [0, 1, 0]), (rotate_z, [0, 0, 1])):
            if deg != 0.0:
                M = R.rotation_matrix(np.deg2rad(deg), axis, point=center) @ M

        # Copy so we don't mutate the upstream mesh; apply_transform moves verts + normals
        # and leaves visual (UV/material) and vertex_attributes (COLOR_0) intact.
        out = mesh.copy()
        out.apply_transform(M)

        status = (f"Oriented: X={rotate_x:g}° Y={rotate_y:g}° Z={rotate_z:g}° "
                  f"about {'centroid' if about_centroid else 'origin'}; UV/material/COLOR_0 kept")
        print(f"[BD Orient Mesh] {status}")
        return io.NodeOutput(out, status)


ORIENT_V3_NODES = [BD_OrientMesh]
ORIENT_NODES = {"BD_OrientMesh": BD_OrientMesh}
ORIENT_DISPLAY_NAMES = {"BD_OrientMesh": "BD Orient Mesh"}
