"""
BD_AnimRetarget — retarget SMPL-H animation (HunyuanMotion) onto a UEFN-rigged character.

Pipeline position:
    HYMotionExportFBX ──────────────────────┐
                                            ▼
    BD_AutoRigUEFN ──► BD_AnimRetarget ──► animated character FBX

The SMPL-H skeleton (Pelvis, L_Hip, Spine1...) is mapped to the UEFN skeleton
(pelvis, thigh_l, spine_01...) via Blender Copy Rotation constraints + NLA bake.
"""

import os
from pathlib import Path

from comfy_api.latest import io

from ..blender.base import BlenderNodeMixin

_PACK_ROOT      = Path(__file__).resolve().parent.parent.parent
_BLENDER_SCRIPT = _PACK_ROOT / "lib" / "blender" / "anim_retarget.py"


class BD_AnimRetarget(io.ComfyNode, BlenderNodeMixin):
    """Retarget HunyuanMotion animation onto a UEFN-rigged character.

    Takes the animated SMPL-H FBX from HYMotionExportFBX and the character
    FBX from BD_AutoRigUEFN, maps the SMPL-H skeleton onto the UEFN skeleton
    using Blender Copy Rotation constraints, bakes the result, and exports
    an animated character FBX ready for UEFN/Fortnite.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_AnimRetarget",
            display_name="BD Anim Retarget",
            category="🧠BrainDead/AutoRig",
            description=(
                "Retarget a HunyuanMotion animation (SMPL-H skeleton) onto a "
                "UEFN-rigged character (from BD_AutoRigUEFN). Outputs an animated "
                "FBX with the character mesh driven by the transferred motion."
            ),
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "motion_fbx",
                    tooltip=(
                        "Animated SMPL-H FBX — output of HYMotionExportFBX. "
                        "Contains the skeleton animation data."
                    ),
                ),
                io.String.Input(
                    "character_fbx",
                    tooltip=(
                        "UEFN-rigged character FBX — output of BD_AutoRigUEFN. "
                        "The mesh and skeleton that will be animated."
                    ),
                ),
                io.String.Input(
                    "output_name",
                    default="",
                    optional=True,
                    tooltip="Custom output filename (no extension). Auto-generated if empty.",
                ),
                io.Int.Input(
                    "fps",
                    default=30,
                    min=1,
                    max=120,
                    tooltip="Frame rate of the source animation (must match HunyuanMotion generation fps).",
                ),
            ],
            outputs=[
                io.String.Output(display_name="animated_fbx"),
            ],
        )

    @classmethod
    def execute(
        cls,
        motion_fbx: str,
        character_fbx: str,
        output_name: str = "",
        fps: int = 30,
    ) -> io.NodeOutput:
        import folder_paths

        motion_fbx    = str(Path(motion_fbx).resolve())
        character_fbx = str(Path(character_fbx).resolve())

        if not os.path.exists(motion_fbx):
            raise FileNotFoundError(f"BD_AnimRetarget: motion_fbx not found: {motion_fbx}")
        if not os.path.exists(character_fbx):
            raise FileNotFoundError(f"BD_AnimRetarget: character_fbx not found: {character_fbx}")
        if not _BLENDER_SCRIPT.exists():
            raise FileNotFoundError(f"BD_AnimRetarget: script missing: {_BLENDER_SCRIPT}")

        ok, err = cls._check_blender()
        if not ok:
            raise RuntimeError(f"BD_AnimRetarget: Blender unavailable — {err}")

        # Derive output name from character FBX if not given
        if not output_name:
            base = Path(character_fbx).stem
            output_name = f"{base}_animated"

        output_dir = os.path.join(folder_paths.get_output_directory(), "autorig")
        os.makedirs(output_dir, exist_ok=True)

        # Auto-increment to avoid overwriting
        idx = 1
        while True:
            candidate = os.path.join(output_dir, f"{output_name}_{idx:03d}.fbx")
            if not os.path.exists(candidate):
                break
            idx += 1
        output_fbx = candidate

        script = _BLENDER_SCRIPT.read_text(encoding="utf-8")
        ok, msg, lines = cls._run_blender_script(
            script=script,
            input_path=motion_fbx,
            output_path=output_fbx,
            extra_args={
                "CHAR_FBX": character_fbx,
                "FPS":      str(fps),
            },
            timeout=900,
        )
        if not ok:
            tail = "\n".join(lines[-30:]) if lines else msg
            raise RuntimeError(f"BD_AnimRetarget Blender failed:\n{tail}")

        if not os.path.exists(output_fbx):
            raise RuntimeError(f"BD_AnimRetarget: output FBX not created: {output_fbx}")

        return io.NodeOutput(output_fbx)


ANIM_RETARGET_V3_NODES      = [BD_AnimRetarget]
ANIM_RETARGET_NODES         = {"BD_AnimRetarget": BD_AnimRetarget}
ANIM_RETARGET_DISPLAY_NAMES = {"BD_AnimRetarget": "BD Anim Retarget"}
