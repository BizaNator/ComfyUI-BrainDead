"""
BD_CubePartGetPart - pull one part mesh out of a CubePart TRIMESH_LIST.

Lets a single part flow into the rest of the BD mesh pipeline (CuMesh simplify,
Blender decimate, export) as a plain TRIMESH.
"""
from comfy_api.latest import io


class BD_CubePartGetPart(io.ComfyNode):
    """
    Select one part from a BD_CubePartSegment `parts` list by index.

    Outputs the chosen part as a plain TRIMESH plus its name. The index is
    clamped to the valid range (logged) so it never crashes on out-of-range.
    If `part_names` (the newline string from BD_CubePartSegment) is wired, the
    matching name is returned; otherwise a `part_NN` placeholder is used.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CubePartGetPart",
            display_name="BD CubePart Get Part",
            category="🧠BrainDead/CubePart",
            description="Select one part mesh from a CubePart parts list by index.",
            inputs=[
                io.Custom("TRIMESH_LIST").Input(
                    "parts", tooltip="parts output from BD CubePart Segment."),
                io.Int.Input(
                    "index", default=0, min=0, max=4096,
                    tooltip="Which part to extract (0-based)."),
                io.String.Input(
                    "part_names", default="", optional=True, multiline=True,
                    tooltip="Optional part_names string from BD CubePart Segment, "
                            "used to resolve the output name."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.String.Output(display_name="name"),
            ],
        )

    @classmethod
    def execute(cls, parts, index: int = 0, part_names: str = "") -> io.NodeOutput:
        if not parts:
            raise ValueError("BD CubePart Get Part: `parts` list is empty.")

        n = len(parts)
        idx = index
        if idx < 0 or idx >= n:
            clamped = max(0, min(idx, n - 1))
            print(f"[BD CubePart] Get Part: index {idx} out of range [0,{n - 1}], "
                  f"using {clamped}.")
            idx = clamped

        names = [s.strip() for s in part_names.split("\n") if s.strip()] if part_names else []
        name = names[idx] if idx < len(names) else f"part_{idx:02d}"

        return io.NodeOutput(parts[idx], name)


CUBEPART_GETPART_V3_NODES = [BD_CubePartGetPart]

CUBEPART_GETPART_NODES = {
    "BD_CubePartGetPart": BD_CubePartGetPart,
}

CUBEPART_GETPART_DISPLAY_NAMES = {
    "BD_CubePartGetPart": "BD CubePart Get Part",
}
