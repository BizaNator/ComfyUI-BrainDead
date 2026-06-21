"""
BD_MixamoToUEFN — Remap a rigged FBX from Mixamo bone naming to UEFN_Mannequin.

Make-It-Animatable and UniRig both emit Mixamo-style skeletons (52 bones with
finger chains). UEFN/Fortnite expects the canonical 87-bone SKM_UEFN_Mannequin
naming convention (pelvis, spine_01..05, neck_01/02, head, clavicle_l/r,
upperarm_l/r + twists, lowerarm_l/r + twists, hand_l/r + 5-finger chains,
thigh_l/r + twists, calf_l/r + twists, foot_l/r, ball_l/r, etc.).

This node uses a bundled Blender subprocess to:
  1) Import the rigged FBX.
  2) Walk every bone in the armature, apply the Mixamo→UEFN name map.
  3) Walk every mesh's vertex groups, rename the matching ones.
  4) Re-export the FBX with the UEFN names.

The bone-name table is the inverse of the PoseFixer_v1 BONE_MAP in
ComfyUI-UniRig/scripts/uefn_pipeline/PoseFixer_v1.py.

Unmapped Mixamo bones (e.g. RightHandThumb1) are passed through with a
naming-convention transform that converts CamelCase chain names to UEFN's
snake_case_l/r layout where possible. Anything we can't confidently map is
left alone — the downstream PoseFixer retarget will skip orphan bones.
"""

import os
import re
from pathlib import Path
from typing import Optional

import folder_paths
from comfy_api.latest import io

from ..blender.base import BlenderNodeMixin


# ── Mixamo → UEFN bone mapping ───────────────────────────────────────────────
#
# Inverse of PoseFixer_v1.BONE_MAP (UEFN → Mixamo). MakeItAnimatable's output
# uses these exact Mixamo names; UniRig with skeleton_template="mixamo" does
# too. Anything not in this table is name-normalized below.

MIXAMO_TO_UEFN: dict[str, str] = {
    # ── Torso ────────────────────────────────────────────────
    "Hips":         "pelvis",
    "Spine":        "spine_01",
    "Spine1":       "spine_02",
    "Spine2":       "spine_03",
    "Spine3":       "spine_04",  # extra spine bone if present
    "Neck":         "neck_01",
    "Neck1":        "neck_02",
    "Head":         "head",

    # ── Arms ─────────────────────────────────────────────────
    "LeftShoulder":  "clavicle_l",
    "LeftArm":       "upperarm_l",
    "LeftForeArm":   "lowerarm_l",
    "LeftHand":      "hand_l",
    "RightShoulder": "clavicle_r",
    "RightArm":      "upperarm_r",
    "RightForeArm":  "lowerarm_r",
    "RightHand":     "hand_r",

    # ── Left hand fingers (Mixamo → UEFN) ────────────────────
    "LeftHandThumb1":  "thumb_01_l",
    "LeftHandThumb2":  "thumb_02_l",
    "LeftHandThumb3":  "thumb_03_l",
    "LeftHandIndex1":  "index_01_l",
    "LeftHandIndex2":  "index_02_l",
    "LeftHandIndex3":  "index_03_l",
    "LeftHandMiddle1": "middle_01_l",
    "LeftHandMiddle2": "middle_02_l",
    "LeftHandMiddle3": "middle_03_l",
    "LeftHandRing1":   "ring_01_l",
    "LeftHandRing2":   "ring_02_l",
    "LeftHandRing3":   "ring_03_l",
    "LeftHandPinky1":  "pinky_01_l",
    "LeftHandPinky2":  "pinky_02_l",
    "LeftHandPinky3":  "pinky_03_l",

    # ── Right hand fingers ───────────────────────────────────
    "RightHandThumb1":  "thumb_01_r",
    "RightHandThumb2":  "thumb_02_r",
    "RightHandThumb3":  "thumb_03_r",
    "RightHandIndex1":  "index_01_r",
    "RightHandIndex2":  "index_02_r",
    "RightHandIndex3":  "index_03_r",
    "RightHandMiddle1": "middle_01_r",
    "RightHandMiddle2": "middle_02_r",
    "RightHandMiddle3": "middle_03_r",
    "RightHandRing1":   "ring_01_r",
    "RightHandRing2":   "ring_02_r",
    "RightHandRing3":   "ring_03_r",
    "RightHandPinky1":  "pinky_01_r",
    "RightHandPinky2":  "pinky_02_r",
    "RightHandPinky3":  "pinky_03_r",

    # ── Legs ─────────────────────────────────────────────────
    "LeftUpLeg":   "thigh_l",
    "LeftLeg":     "calf_l",
    "LeftFoot":    "foot_l",
    "LeftToeBase": "ball_l",
    "RightUpLeg":  "thigh_r",
    "RightLeg":    "calf_r",
    "RightFoot":   "foot_r",
    "RightToeBase":"ball_r",
}


# ── Blender subprocess script ────────────────────────────────────────────────

_REMAP_SCRIPT = r'''
import bpy
import json
import os
import sys

INPUT_FBX  = os.environ["BLENDER_INPUT_PATH"]
OUTPUT_FBX = os.environ["BLENDER_OUTPUT_PATH"]
BONE_MAP   = json.loads(os.environ["BLENDER_ARG_BONE_MAP"])

print(f"[BD_MixamoToUEFN] in:  {INPUT_FBX}", flush=True)
print(f"[BD_MixamoToUEFN] out: {OUTPUT_FBX}", flush=True)
print(f"[BD_MixamoToUEFN] map: {len(BONE_MAP)} entries", flush=True)

# Clean scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import
bpy.ops.import_scene.fbx(
    filepath=INPUT_FBX,
    automatic_bone_orientation=True,
    use_anim=False,
)

armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
meshes    = [o for o in bpy.data.objects if o.type == "MESH"]
print(f"[BD_MixamoToUEFN] imported: {len(armatures)} armatures, {len(meshes)} meshes", flush=True)

# Strip the common Mixamo "mixamorig:" prefix everywhere up-front.
# Also strip "Armature|" and other namespace junk.
def _strip_prefix(name):
    for pfx in ("mixamorig:", "Armature|", "Armature:", "Avatar:"):
        if name.startswith(pfx):
            return name[len(pfx):]
    return name

renamed_bones = 0
renamed_vgroups = 0
unmapped_bones = []

for arm in armatures:
    # Edit bone names
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for b in arm.data.edit_bones:
        stripped = _strip_prefix(b.name)
        if stripped != b.name:
            b.name = stripped
        target = BONE_MAP.get(b.name)
        if target and target != b.name:
            b.name = target
            renamed_bones += 1
        elif b.name not in BONE_MAP.values():
            unmapped_bones.append(b.name)
    bpy.ops.object.mode_set(mode="OBJECT")

# Rename vertex groups on every mesh
for m in meshes:
    for vg in list(m.vertex_groups):
        stripped = _strip_prefix(vg.name)
        if stripped != vg.name:
            vg.name = stripped
        target = BONE_MAP.get(vg.name)
        if target and target != vg.name:
            vg.name = target
            renamed_vgroups += 1

print(f"[BD_MixamoToUEFN] renamed {renamed_bones} bones, {renamed_vgroups} vgroups", flush=True)
if unmapped_bones:
    print(f"[BD_MixamoToUEFN] WARN {len(unmapped_bones)} unmapped bones (kept original names):", flush=True)
    for b in sorted(set(unmapped_bones))[:30]:
        print(f"    {b}", flush=True)

# Export
os.makedirs(os.path.dirname(OUTPUT_FBX), exist_ok=True)

bpy.ops.export_scene.fbx(
    filepath=OUTPUT_FBX,
    use_selection=False,
    object_types={"ARMATURE", "MESH"},
    add_leaf_bones=False,
    bake_anim=False,
    use_mesh_modifiers=True,
    mesh_smooth_type="FACE",
    apply_unit_scale=True,
    apply_scale_options="FBX_SCALE_ALL",
    bake_space_transform=False,
    axis_forward="-Y",
    axis_up="Z",
    primary_bone_axis="Y",
    secondary_bone_axis="X",
    armature_nodetype="NULL",
)

print(f"[BD_MixamoToUEFN] DONE -> {OUTPUT_FBX}", flush=True)
'''


class BD_MixamoToUEFN(io.ComfyNode, BlenderNodeMixin):
    """Rename a rigged FBX's bones (and matching mesh vertex groups) from
    Mixamo naming to the UEFN_Mannequin convention. Pass-through for any bone
    not in the table — Make-It-Animatable's standard 52-bone set is covered."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_MixamoToUEFN",
            display_name="BD Mixamo → UEFN Rename",
            category="🧠BrainDead/AutoRig",
            description=(
                "Take an FBX that was auto-rigged by Make-It-Animatable or "
                "UniRig (Mixamo skeleton template) and rewrite the bone + "
                "vertex group names to the UEFN SKM_UEFN_Mannequin convention. "
                "This is the bridge between the autorig output and the canonical "
                "UEFN/Fortnite skeleton expected by UEFN imports. Use BD "
                "PoseFixer Retarget after this to align the rest pose."
            ),
            inputs=[
                io.String.Input("input_fbx",
                                tooltip="Path to a rigged FBX with a Mixamo-style "
                                        "skeleton. Output of BD_AutoRigMIA or "
                                        "BD_AutoRigUniRig."),
                io.String.Input("output_name",
                                default="",
                                tooltip="Optional output filename (no extension). "
                                        "If empty, appends '_uefn' to the input.",
                                optional=True),
            ],
            outputs=[
                io.String.Output(display_name="output_fbx"),
            ],
        )

    @classmethod
    def execute(cls, input_fbx: str, output_name: str = "") -> io.NodeOutput:
        input_fbx = str(Path(input_fbx).resolve())
        if not os.path.exists(input_fbx):
            raise FileNotFoundError(f"input_fbx not found: {input_fbx}")

        out_dir = Path(folder_paths.get_output_directory())
        if output_name:
            output_fbx = out_dir / f"{output_name}.fbx"
        else:
            output_fbx = out_dir / (Path(input_fbx).stem + "_uefn.fbx")

        ok, blender_path_or_err = cls._check_blender()
        if not ok:
            raise RuntimeError(blender_path_or_err)

        import json
        # BlenderNodeMixin sets BLENDER_INPUT_PATH + BLENDER_OUTPUT_PATH
        # from input_path/output_path, and prefixes extra_args keys with
        # BLENDER_ARG_. So our script reads BLENDER_ARG_BONE_MAP.
        ok, msg, lines = cls._run_blender_script(
            script=_REMAP_SCRIPT,
            input_path=str(input_fbx),
            output_path=str(output_fbx),
            extra_args={"BONE_MAP": json.dumps(MIXAMO_TO_UEFN)},
            timeout=300,
        )
        if not ok:
            tail = "\n".join(lines[-20:]) if lines else msg
            raise RuntimeError(f"BD_MixamoToUEFN failed: {tail}")

        return io.NodeOutput(str(output_fbx))


BONE_REMAP_V3_NODES = [BD_MixamoToUEFN]
BONE_REMAP_NODES = {"BD_MixamoToUEFN": BD_MixamoToUEFN}
BONE_REMAP_DISPLAY_NAMES = {"BD_MixamoToUEFN": "BD Mixamo → UEFN Rename"}
