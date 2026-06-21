"""
Headless Blender: Retarget SMPL-H animation to a UEFN-rigged character.

Source: HunyuanMotion FBX (SMPL-H bones: Pelvis, L_Hip, Spine1, ...)
Target: Character FBX with UEFN skeleton (bones: pelvis, thigh_l, spine_01, ...)

Uses Copy Rotation constraints + NLA bake to transfer motion.

Environment variables:
  BLENDER_INPUT_PATH   — motion FBX (from HYMotionExportFBX, SMPL-H animated)
  BLENDER_OUTPUT_PATH  — output path for animated character FBX
  BLENDER_ARG_CHAR_FBX — UEFN-rigged character FBX (from BD_AutoRigUEFN)
  BLENDER_ARG_FPS      — frame rate (default 30)
"""

import bpy
import os

MOTION_FBX = os.environ["BLENDER_INPUT_PATH"]
OUTPUT_FBX = os.environ["BLENDER_OUTPUT_PATH"]
CHAR_FBX   = os.environ["BLENDER_ARG_CHAR_FBX"]
FPS        = int(os.environ.get("BLENDER_ARG_FPS", "30"))


def log(msg):
    print(msg, flush=True)


# SMPL-H bone → UEFN bone mapping (confirmed against both skeletons)
SMPLH_TO_UEFN = {
    "Pelvis":     "pelvis",
    "L_Hip":      "thigh_l",      "R_Hip":      "thigh_r",
    "Spine1":     "spine_01",
    "L_Knee":     "calf_l",       "R_Knee":     "calf_r",
    "Spine2":     "spine_02",
    "L_Ankle":    "foot_l",       "R_Ankle":    "foot_r",
    "Spine3":     "spine_03",
    "L_Foot":     "ball_l",       "R_Foot":     "ball_r",
    "Neck":       "neck_01",
    "L_Collar":   "clavicle_l",   "R_Collar":   "clavicle_r",
    "Head":       "head",
    "L_Shoulder": "upperarm_l",   "R_Shoulder": "upperarm_r",
    "L_Elbow":    "lowerarm_l",   "R_Elbow":    "lowerarm_r",
    "L_Wrist":    "hand_l",       "R_Wrist":    "hand_r",
    # Fingers L
    "L_Index1":   "index_01_l",   "L_Index2":   "index_02_l",   "L_Index3":   "index_03_l",
    "L_Middle1":  "middle_01_l",  "L_Middle2":  "middle_02_l",  "L_Middle3":  "middle_03_l",
    "L_Pinky1":   "pinky_01_l",   "L_Pinky2":   "pinky_02_l",   "L_Pinky3":   "pinky_03_l",
    "L_Ring1":    "ring_01_l",    "L_Ring2":    "ring_02_l",    "L_Ring3":    "ring_03_l",
    "L_Thumb1":   "thumb_01_l",   "L_Thumb2":   "thumb_02_l",   "L_Thumb3":   "thumb_03_l",
    # Fingers R
    "R_Index1":   "index_01_r",   "R_Index2":   "index_02_r",   "R_Index3":   "index_03_r",
    "R_Middle1":  "middle_01_r",  "R_Middle2":  "middle_02_r",  "R_Middle3":  "middle_03_r",
    "R_Pinky1":   "pinky_01_r",   "R_Pinky2":   "pinky_02_r",   "R_Pinky3":   "pinky_03_r",
    "R_Ring1":    "ring_01_r",    "R_Ring2":    "ring_02_r",    "R_Ring3":    "ring_03_r",
    "R_Thumb1":   "thumb_01_r",   "R_Thumb2":   "thumb_02_r",   "R_Thumb3":   "thumb_03_r",
}

# Lowercase fallback names (some HY-Motion FBX exports use these instead of SMPL-H standard)
SMPLH_LOWERCASE = {
    "Pelvis": "pelvis", "L_Hip": "left_hip", "R_Hip": "right_hip",
    "Spine1": "spine1", "L_Knee": "left_knee", "R_Knee": "right_knee",
    "Spine2": "spine2", "L_Ankle": "left_ankle", "R_Ankle": "right_ankle",
    "Spine3": "spine3", "L_Foot": "left_foot", "R_Foot": "right_foot",
    "Neck": "neck", "L_Collar": "left_collar", "R_Collar": "right_collar",
    "Head": "head", "L_Shoulder": "left_shoulder", "R_Shoulder": "right_shoulder",
    "L_Elbow": "left_elbow", "R_Elbow": "right_elbow",
    "L_Wrist": "left_wrist", "R_Wrist": "right_wrist",
}


def resolve_src_bone(smplh_name, src_names):
    """Find actual bone name in source FBX, trying direct then lowercase fallback."""
    if smplh_name in src_names:
        return smplh_name
    lc = SMPLH_LOWERCASE.get(smplh_name)
    if lc and lc in src_names:
        return lc
    return None


# --- Setup scene ---
log("[AnimRetarget] Clearing scene")
bpy.ops.wm.read_factory_settings(use_empty=True)

# --- Import source (SMPL-H animation) ---
log(f"[AnimRetarget] Importing motion FBX: {MOTION_FBX}")
bpy.ops.import_scene.fbx(filepath=MOTION_FBX, automatic_bone_orientation=True, use_anim=True)
bpy.context.view_layer.update()

src_arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
if not src_arms:
    raise RuntimeError("[AnimRetarget] No armature found in motion FBX")
src_arm = src_arms[0]

src_bone_names = {b.name for b in src_arm.data.bones}
log(f"[AnimRetarget] Source armature '{src_arm.name}': {len(src_bone_names)} bones")

# Determine animation frame range
scene = bpy.context.scene
scene.render.fps = FPS
frame_start, frame_end = 1, 120
if src_arm.animation_data and src_arm.animation_data.action:
    action = src_arm.animation_data.action
    frame_start = int(action.frame_range[0])
    frame_end   = int(action.frame_range[1])
scene.frame_start = frame_start
scene.frame_end   = frame_end
log(f"[AnimRetarget] Animation range: {frame_start}–{frame_end} ({frame_end - frame_start + 1} frames @ {FPS} fps)")

# --- Import target (UEFN character) ---
log(f"[AnimRetarget] Importing character FBX: {CHAR_FBX}")
pre_objects = set(bpy.data.objects)
bpy.ops.import_scene.fbx(filepath=CHAR_FBX, automatic_bone_orientation=True, use_anim=False)
bpy.context.view_layer.update()

new_objects = list(set(bpy.data.objects) - pre_objects)
tgt_arm = next((o for o in new_objects if o.type == "ARMATURE"), None)
if tgt_arm is None:
    raise RuntimeError("[AnimRetarget] No armature found in character FBX")

log(f"[AnimRetarget] Target armature '{tgt_arm.name}': {len(tgt_arm.data.bones)} bones")

# --- Add Copy Rotation (and root Copy Location) constraints ---
bpy.context.view_layer.objects.active = tgt_arm
bpy.ops.object.mode_set(mode="POSE")

mapped_count = 0
skipped_src  = []
skipped_tgt  = []

for smplh_name, uefn_name in SMPLH_TO_UEFN.items():
    src_bone_name = resolve_src_bone(smplh_name, src_bone_names)
    if src_bone_name is None:
        skipped_src.append(smplh_name)
        continue

    tgt_pbone = tgt_arm.pose.bones.get(uefn_name)
    if tgt_pbone is None:
        skipped_tgt.append(uefn_name)
        continue

    crc = tgt_pbone.constraints.new("COPY_ROTATION")
    crc.name     = "AnimRetarget"
    crc.target       = src_arm
    crc.subtarget    = src_bone_name
    crc.mix_mode     = "REPLACE"
    crc.target_space = "WORLD"
    crc.owner_space  = "LOCAL"

    if uefn_name == "pelvis":
        cl = tgt_pbone.constraints.new("COPY_LOCATION")
        cl.name         = "AnimRetarget_Loc"
        cl.target       = src_arm
        cl.subtarget    = src_bone_name
        cl.target_space = "WORLD"
        cl.owner_space  = "WORLD"

    mapped_count += 1

log(f"[AnimRetarget] Mapped {mapped_count} bone pairs")
if skipped_src:
    log(f"[AnimRetarget] Src bones not found (using direct names): {skipped_src[:8]}")
if skipped_tgt:
    log(f"[AnimRetarget] Tgt bones not in UEFN armature: {skipped_tgt[:8]}")

# --- Bake animation ---
log(f"[AnimRetarget] Baking animation frames {frame_start}–{frame_end} ...")
bpy.ops.pose.select_all(action="SELECT")
bpy.ops.nla.bake(
    frame_start=frame_start,
    frame_end=frame_end,
    only_selected=False,
    visual_keying=True,
    clear_constraints=True,
    bake_types={"POSE"},
)
bpy.ops.object.mode_set(mode="OBJECT")
log("[AnimRetarget] Bake complete")

# --- Select only character objects for export ---
bpy.ops.object.select_all(action="DESELECT")
for o in new_objects:
    o.select_set(True)
bpy.context.view_layer.objects.active = tgt_arm

# --- Export animated FBX ---
os.makedirs(os.path.dirname(OUTPUT_FBX) or ".", exist_ok=True)
log(f"[AnimRetarget] Exporting animated character FBX: {OUTPUT_FBX}")
bpy.ops.export_scene.fbx(
    filepath=OUTPUT_FBX,
    use_selection=True,
    bake_anim=True,
    bake_anim_use_all_bones=True,
    bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=False,
    bake_anim_force_startend_keying=True,
    add_leaf_bones=False,
    axis_forward="-Y",
    axis_up="Z",
    apply_scale_options="FBX_SCALE_ALL",
)
log("[AnimRetarget] Done.")
