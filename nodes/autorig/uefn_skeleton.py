"""
BD_AutoRigUEFN — Mixamo FBX → full UEFN skeleton via Blender weight transfer.

Step 2 of the BrainDead autorig pipeline:

    Mesh  →  BD_AutoRigMIA  →  BD_AutoRigUEFN  →  UEFN-rigged FBX
                   (step 1)         (this node)

What this node does (headless Blender):
  1. Import the SKM_UEFN_Mannequin reference (bundled in lib/assets/uefn/)
     into a "Source" collection.
  2. Import the input character FBX (Mixamo-rigged, from BD_AutoRigMIA)
     into a "Target" collection.
  3. Scale-match the character to the mannequin height.
  4. Align the character armature to the mannequin (hip translate + yaw).
  5. Bake the character mesh geometry (apply armature deform in REST pose).
  6. Clear vertex groups; create empty UEFN groups; transfer skin weights
     from the mannequin via Blender's Data Transfer modifier.
  7. Bind the character mesh to the UEFN armature.
  8. Export the result as FBX.

The output FBX contains the full UEFN/Fortnite skeleton with genuine skin
weights interpolated from the mannequin reference, ready for import into UEFN.

Accepts both Mixamo-named (Hips/LeftShoulder) and already-UEFN-renamed
(pelvis/clavicle_l) input FBX; the headless script auto-detects bone names.
"""

import os
from glob import glob
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from ..blender.base import BlenderNodeMixin

# Bundled reference and headless script, both in the pack root
_PACK_ROOT = Path(__file__).resolve().parent.parent.parent
_UEFN_REF_FBX   = _PACK_ROOT / "lib" / "assets" / "uefn" / "skm_uefn_mannequin.fbx"
_BLENDER_SCRIPT  = _PACK_ROOT / "lib" / "blender" / "uefn_skeletonize.py"


class BD_AutoRigUEFN(io.ComfyNode, BlenderNodeMixin):
    """Convert a Mixamo-rigged FBX to the full UEFN skeleton.

    Uses the bundled SKM_UEFN_Mannequin.fbx as the weight-donor reference.
    Runs a headless Blender process that scale-matches, aligns, bakes,
    transfers skin weights via Data Transfer, and exports the result.

    Input: output of BD_AutoRigMIA (Mixamo or UEFN-named bones accepted).
    Output: FBX with the UEFN/Fortnite armature, ready for UEFN import.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_AutoRigUEFN",
            display_name="BD AutoRig → UEFN Skeleton",
            category="🧠BrainDead/AutoRig",
            description=(
                "Step 2: Convert a Mixamo-rigged FBX (from BD AutoRig MIA) to a full "
                "UEFN/Fortnite skeleton. Uses the bundled SKM_UEFN_Mannequin as a "
                "weight-transfer donor — scales and aligns the character, bakes the mesh "
                "to REST pose, transfers UEFN skin weights via Data Transfer modifier, and "
                "binds it to the UEFN armature. Output is importable directly into UEFN."
            ),
            inputs=[
                io.String.Input(
                    "input_fbx",
                    tooltip=(
                        "Path to a Mixamo-rigged FBX — the output of BD AutoRig MIA. "
                        "Both Mixamo bone names (Hips, LeftShoulder) and already-renamed "
                        "UEFN names (pelvis, clavicle_l) are accepted."
                    ),
                ),
                io.String.Input(
                    "filename",
                    default="uefn_rig",
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
                io.String.Output(display_name="fbx_path"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        input_fbx: str,
        filename: str = "uefn_rig",
        name_prefix: str = "",
        auto_increment: bool = True,
        context_id: str = "",
        suffix: str = "",
        context_custom_vars: str = "",
    ) -> io.NodeOutput:
        input_fbx = str(Path(input_fbx).resolve())
        if not os.path.exists(input_fbx):
            raise FileNotFoundError(f"BD_AutoRigUEFN: input_fbx not found: {input_fbx}")

        if not _UEFN_REF_FBX.exists():
            raise FileNotFoundError(
                f"BD_AutoRigUEFN: bundled UEFN reference not found: {_UEFN_REF_FBX}\n"
                "The pack may be incomplete — re-install ComfyUI-BrainDead."
            )

        if not _BLENDER_SCRIPT.exists():
            raise FileNotFoundError(
                f"BD_AutoRigUEFN: headless Blender script not found: {_BLENDER_SCRIPT}"
            )

        base_output_dir = folder_paths.get_output_directory()

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

        ok, blender_path_or_err = cls._check_blender()
        if not ok:
            raise RuntimeError(f"BD_AutoRigUEFN: Blender not available — {blender_path_or_err}")

        script_text = _BLENDER_SCRIPT.read_text(encoding="utf-8")

        ok, msg, lines = cls._run_blender_script(
            script=script_text,
            input_path=input_fbx,
            output_path=file_path,
            extra_args={"UEFN_MANNY_FBX": str(_UEFN_REF_FBX)},
            timeout=600,
        )
        if not ok:
            tail = "\n".join(lines[-30:]) if lines else msg
            raise RuntimeError(f"BD_AutoRigUEFN failed:\n{tail}")

        return io.NodeOutput(file_path)


UEFN_SKEL_V3_NODES       = [BD_AutoRigUEFN]
UEFN_SKEL_NODES           = {"BD_AutoRigUEFN": BD_AutoRigUEFN}
UEFN_SKEL_DISPLAY_NAMES   = {"BD_AutoRigUEFN": "BD AutoRig → UEFN Skeleton"}
