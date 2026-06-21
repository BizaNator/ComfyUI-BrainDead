"""
BD_Preview3D — native ComfyUI 3D viewer for FBX, GLB, OBJ files.

Uses ComfyUI's built-in Preview3D / three.js infrastructure.
No external dependencies beyond ComfyUI itself.
"""

import os
from pathlib import Path

import folder_paths
from comfy_api.latest import io, UI


class BD_Preview3D(io.ComfyNode):
    """Display any 3D file (FBX, GLB, OBJ) in ComfyUI's native interactive 3D viewer.

    Accepts FBX paths from BD_AutoRigUEFN or BD_AnimRetarget, or any GLB/OBJ/STL.
    The viewer supports animated FBX — use the timeline in the node panel to scrub.
    No UniRig or third-party dependency required.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_Preview3D",
            display_name="BD Preview 3D",
            category="🧠BrainDead/Mesh",
            description=(
                "Display a 3D model file (FBX, GLB, OBJ, STL) in ComfyUI's native 3D viewer. "
                "Wire in FBX paths from BD_AutoRigUEFN or BD_AnimRetarget. "
                "Supports skeletal animation — scrub the timeline in the viewer."
            ),
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "model_path",
                    tooltip=(
                        "Path to a 3D file. Accepts output from BD_AutoRigUEFN, "
                        "BD_AnimRetarget, or any absolute path to FBX/GLB/OBJ/STL."
                    ),
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, model_path: str) -> io.NodeOutput:
        model_path = str(Path(model_path).resolve())
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"BD_Preview3D: file not found: {model_path}")

        output_dir = os.path.realpath(folder_paths.get_output_directory())
        input_dir  = os.path.realpath(folder_paths.get_input_directory())

        # Convert to path that ComfyUI's /view endpoint can serve.
        # Preview3D expects either an absolute path (ComfyUI resolves it via /view)
        # or a path relative to output/input.
        real = os.path.realpath(model_path)
        if real.startswith(output_dir + os.sep) or real == output_dir:
            display_path = os.path.relpath(real, output_dir)
        elif real.startswith(input_dir + os.sep) or real == input_dir:
            display_path = os.path.relpath(real, input_dir)
        else:
            # Outside managed directories — pass absolute path;
            # ComfyUI Preview3D handles absolute paths via /view?type=absolute.
            display_path = real

        # Normalize to forward slashes (ComfyUI convention)
        display_path = display_path.replace("\\", "/")

        return io.NodeOutput(ui=UI.PreviewUI3D(display_path, None))


PREVIEW_3D_V3_NODES      = [BD_Preview3D]
PREVIEW_3D_NODES         = {"BD_Preview3D": BD_Preview3D}
PREVIEW_3D_DISPLAY_NAMES = {"BD_Preview3D": "BD Preview 3D"}
