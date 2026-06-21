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
from glob import glob
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
                io.Int.Input(
                    "fps",
                    default=30,
                    min=1,
                    max=120,
                    tooltip="Frame rate of the source animation (must match HunyuanMotion generation fps).",
                ),
                io.String.Input(
                    "filename",
                    default="anim_retarget",
                    tooltip="Base filename without extension.",
                    optional=True,
                ),
                io.String.Input(
                    "name_prefix",
                    default="",
                    tooltip="Prepended to filename. Supports subdirs (e.g. 'autorig/mychar').",
                    optional=True,
                ),
                io.Boolean.Input(
                    "auto_increment",
                    default=True,
                    tooltip="Auto-increment filename to avoid overwriting.",
                    optional=True,
                ),
                io.String.Input(
                    "context_id",
                    default="",
                    tooltip=(
                        "BD_SaveContext id for template-based naming. Empty + one registered "
                        "context = auto-pick. When resolved, filename/name_prefix pass through "
                        "as %filename%/%name_prefix%."
                    ),
                    optional=True,
                ),
                io.String.Input(
                    "suffix",
                    default="",
                    tooltip="Per-save suffix → %suffix% in the context template.",
                    optional=True,
                ),
                io.String.Input(
                    "context_custom_vars",
                    multiline=True,
                    default="",
                    tooltip="Extra key=value template vars (one per line). Only used when context_id resolves.",
                    optional=True,
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
        fps: int = 30,
        filename: str = "anim_retarget",
        name_prefix: str = "",
        auto_increment: bool = True,
        context_id: str = "",
        suffix: str = "",
        context_custom_vars: str = "",
    ) -> io.NodeOutput:
        import folder_paths

        base_output_dir = folder_paths.get_output_directory()

        def resolve_input_path(p: str) -> str:
            resolved = str(Path(p).resolve())
            if not os.path.exists(resolved):
                # HYMotionExportFBX returns paths relative to COMFY_OUTPUT_DIR
                via_output = str(Path(base_output_dir) / p)
                if os.path.exists(via_output):
                    return via_output
            return resolved

        motion_fbx    = resolve_input_path(motion_fbx)
        character_fbx = resolve_input_path(character_fbx)

        if not os.path.exists(motion_fbx):
            raise FileNotFoundError(f"BD_AnimRetarget: motion_fbx not found: {motion_fbx}")
        if not os.path.exists(character_fbx):
            raise FileNotFoundError(f"BD_AnimRetarget: character_fbx not found: {character_fbx}")
        if not _BLENDER_SCRIPT.exists():
            raise FileNotFoundError(f"BD_AnimRetarget: script missing: {_BLENDER_SCRIPT}")

        ok, err = cls._check_blender()
        if not ok:
            raise RuntimeError(f"BD_AnimRetarget: Blender unavailable — {err}")

        from ..cache.save_context import resolve_context_path, get_context, auto_pick_context

        effective_ctx_id = context_id.strip() if context_id else ""
        if not effective_ctx_id:
            picked = auto_pick_context()
            if picked is not None:
                effective_ctx_id = picked
        use_context = bool(effective_ctx_id) and get_context(effective_ctx_id) is not None

        if use_context:
            file_path, _rel = resolve_context_path(
                effective_ctx_id, suffix, "fbx",
                node_filename=filename, node_name_prefix=name_prefix,
                node_custom_vars=context_custom_vars,
            )
            output_dir = os.path.dirname(file_path)
            os.makedirs(output_dir, exist_ok=True)
        else:
            full_name = f"{name_prefix}_{filename}" if name_prefix else filename
            full_name = full_name.replace("\\", "/")
            if "/" in full_name:
                subdir, base_filename = full_name.rsplit("/", 1)
                output_dir = os.path.join(base_output_dir, subdir)
            else:
                output_dir = base_output_dir
                base_filename = full_name

            os.makedirs(output_dir, exist_ok=True)

            if auto_increment:
                pattern = os.path.join(output_dir, f"{base_filename}_*.fbx")
                existing = glob(pattern)
                if existing:
                    numbers = []
                    for f in existing:
                        try:
                            numbers.append(int(os.path.basename(f).replace(".fbx", "").split("_")[-1]))
                        except Exception:
                            pass
                    next_num = max(numbers) + 1 if numbers else 1
                else:
                    next_num = 1
                file_path = os.path.join(output_dir, f"{base_filename}_{next_num:03d}.fbx")
            else:
                file_path = os.path.join(output_dir, f"{base_filename}.fbx")

        script = _BLENDER_SCRIPT.read_text(encoding="utf-8")
        ok, msg, lines = cls._run_blender_script(
            script=script,
            input_path=motion_fbx,
            output_path=file_path,
            extra_args={
                "CHAR_FBX": character_fbx,
                "FPS":      str(fps),
            },
            timeout=900,
        )
        if not ok:
            tail = "\n".join(lines[-30:]) if lines else msg
            raise RuntimeError(f"BD_AnimRetarget Blender failed:\n{tail}")

        if not os.path.exists(file_path):
            raise RuntimeError(f"BD_AnimRetarget: output FBX not created: {file_path}")

        return io.NodeOutput(file_path)


ANIM_RETARGET_V3_NODES      = [BD_AnimRetarget]
ANIM_RETARGET_NODES         = {"BD_AnimRetarget": BD_AnimRetarget}
ANIM_RETARGET_DISPLAY_NAMES = {"BD_AnimRetarget": "BD Anim Retarget"}
