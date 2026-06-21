"""
Headless Blender script: Mixamo-rigged FBX → full UEFN skeleton.

Adapted from BrainDeadBlender/scripts/uefn_pipeline/Pipeline_v31.py for
blender --background subprocess use. Accepts both Mixamo-named and
UEFN-renamed FBX (auto-detects hip bone).

Environment variables:
  BLENDER_INPUT_PATH          — input FBX (from BD_AutoRigMIA)
  BLENDER_OUTPUT_PATH         — output FBX path
  BLENDER_ARG_UEFN_MANNY_FBX — path to bundled SKM_UEFN_Mannequin.fbx
"""

import bpy
import math
import os
import sys
from mathutils import Matrix, Vector

INPUT_FBX    = os.environ["BLENDER_INPUT_PATH"]
OUTPUT_FBX   = os.environ["BLENDER_OUTPUT_PATH"]
UEFN_REF_FBX = os.environ["BLENDER_ARG_UEFN_MANNY_FBX"]


def log(msg):
    print(msg, flush=True)


# ── Bone-name helpers (handle Mixamo + UEFN naming) ──────────────────────────

def _bone_names(arm_obj):
    return {b.name for b in arm_obj.data.bones}

def _find_bone(arm_obj, *candidates):
    names = _bone_names(arm_obj)
    for c in candidates:
        if c in names:
            return c
    # Fall back to the root bone (no parent)
    for b in arm_obj.data.bones:
        if not b.parent:
            return b.name
    return next(iter(names), None)

def _hip_bone(arm_obj):
    return _find_bone(arm_obj, "Hips", "pelvis", "mixamorig:Hips", "Hip", "Root")

def _shoulder_pair(arm_obj):
    names = _bone_names(arm_obj)
    # UEFN
    if "clavicle_l" in names and "clavicle_r" in names:
        return "clavicle_l", "clavicle_r"
    # Mixamo stripped
    if "LeftShoulder" in names and "RightShoulder" in names:
        return "LeftShoulder", "RightShoulder"
    # Mixamo with prefix
    if "mixamorig:LeftShoulder" in names:
        return "mixamorig:LeftShoulder", "mixamorig:RightShoulder"
    return None, None

def _thigh_pair(arm_obj):
    names = _bone_names(arm_obj)
    if "thigh_l" in names and "thigh_r" in names:
        return "thigh_l", "thigh_r"
    if "LeftUpLeg" in names and "RightUpLeg" in names:
        return "LeftUpLeg", "RightUpLeg"
    if "mixamorig:LeftUpLeg" in names:
        return "mixamorig:LeftUpLeg", "mixamorig:RightUpLeg"
    return None, None


# ── Scene helpers ─────────────────────────────────────────────────────────────

def ensure_object_mode():
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def depsgraph_update():
    bpy.context.view_layer.update()


def collection_for(name):
    col = bpy.data.collections.get(name)
    if not col:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def link_to_collection(obj, col):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)


def import_fbx_to_collection(fbx_path, col_name):
    """Import an FBX and move all new objects into col_name."""
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.fbx(
        filepath=fbx_path,
        automatic_bone_orientation=True,
        use_anim=False,
    )
    depsgraph_update()

    col = collection_for(col_name)
    new_objs = [bpy.data.objects[k] for k in bpy.data.objects.keys() if k not in before]
    for obj in new_objs:
        link_to_collection(obj, col)

    arms  = [o for o in new_objs if o.type == "ARMATURE"]
    meshes = [o for o in new_objs if o.type == "MESH"]
    log(f"[{col_name}] imported {len(arms)} armature(s), {len(meshes)} mesh(es)")
    return arms, meshes


def find_armature(col):
    arms = [o for o in col.all_objects if o.type == "ARMATURE"]
    if not arms:
        raise RuntimeError(f"No armature in '{col.name}'")
    return arms[0]


def find_mesh(col, arm_obj):
    meshes = [o for o in col.all_objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh in '{col.name}'")
    # Prefer the mesh driven by arm_obj
    for m in meshes:
        for mod in m.modifiers:
            if mod.type == "ARMATURE" and mod.object == arm_obj:
                return m
    return max(meshes, key=lambda o: len(o.data.vertices))


# ── Measurement ──────────────────────────────────────────────────────────────

def z_height(mesh_obj):
    arm_mod = next((m for m in mesh_obj.modifiers if m.type == "ARMATURE"), None)
    prev = None
    if arm_mod:
        prev = arm_mod.show_viewport
        arm_mod.show_viewport = False
        depsgraph_update()
    dg  = bpy.context.evaluated_depsgraph_get()
    me  = mesh_obj.evaluated_get(dg).to_mesh()
    mw  = mesh_obj.evaluated_get(dg).matrix_world
    zs  = sorted((mw @ v.co).z for v in me.vertices)
    n   = len(zs)
    height = zs[int(0.99 * (n - 1))] - zs[int(0.01 * (n - 1))]
    mesh_obj.evaluated_get(dg).to_mesh_clear()
    if arm_mod and prev is not None:
        arm_mod.show_viewport = prev
        depsgraph_update()
    return height


# ── Alignment ────────────────────────────────────────────────────────────────

def bone_head_world(arm_obj, bone_name):
    b = arm_obj.data.bones.get(bone_name)
    return (arm_obj.matrix_world @ b.head_local) if b else None


def forward_from_pair(arm_obj, left_bone, right_bone):
    L = bone_head_world(arm_obj, left_bone)
    R = bone_head_world(arm_obj, right_bone)
    if L is None or R is None:
        return None
    right = (R - L); right.z = 0
    if right.length < 1e-6:
        return None
    right.normalize()
    fwd = Vector((0, 0, 1)).cross(right)
    return fwd.normalized() if fwd.length > 1e-6 else None


def align_to_source(src_arm, tgt_arm):
    # Translate hip to hip
    src_hip = _hip_bone(src_arm)
    tgt_hip = _hip_bone(tgt_arm)
    log(f"[Align] src_hip={src_hip}  tgt_hip={tgt_hip}")
    sp = bone_head_world(src_arm, src_hip)
    tp = bone_head_world(tgt_arm, tgt_hip)
    if sp is None or tp is None:
        raise RuntimeError("Could not find hip bone for alignment")
    tgt_arm.location += sp - tp
    depsgraph_update()

    # Yaw: try shoulders, fall back to thighs
    sl, sr = _shoulder_pair(src_arm)
    tl, tr = _shoulder_pair(tgt_arm)
    src_fwd = forward_from_pair(src_arm, sl, sr) if sl else None
    tgt_fwd = forward_from_pair(tgt_arm, tl, tr) if tl else None

    if src_fwd is None or tgt_fwd is None:
        sl, sr = _thigh_pair(src_arm)
        tl, tr = _thigh_pair(tgt_arm)
        src_fwd = forward_from_pair(src_arm, sl, sr) if sl else None
        tgt_fwd = forward_from_pair(tgt_arm, tl, tr) if tl else None

    if src_fwd and tgt_fwd:
        a = Vector((tgt_fwd.x, tgt_fwd.y, 0)).normalized()
        b = Vector((src_fwd.x, src_fwd.y, 0)).normalized()
        yaw = math.atan2(a.cross(b).z, max(-1.0, min(1.0, a.dot(b))))
        tgt_arm.rotation_mode = "XYZ"
        tgt_arm.rotation_euler.z += yaw
        depsgraph_update()
        log(f"[Align] Yaw {math.degrees(yaw):.1f}°")
    else:
        log("[Align] Yaw skipped — no landmark pair found")


# ── Weight transfer ───────────────────────────────────────────────────────────

def create_vgroups(mesh_obj, arm_obj):
    existing = {vg.name for vg in mesh_obj.vertex_groups}
    n = 0
    for bone in arm_obj.data.bones:
        if bone.name not in existing:
            mesh_obj.vertex_groups.new(name=bone.name)
            n += 1
    log(f"[VGroups] Created {n} vertex groups")


def transfer_weights(src_mesh, tgt_mesh):
    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    tgt_mesh.select_set(True)
    bpy.context.view_layer.objects.active = tgt_mesh

    mod = tgt_mesh.modifiers.new("DT_Weights", "DATA_TRANSFER")
    mod.object = src_mesh
    mod.use_vert_data = True
    mod.data_types_verts = {"VGROUP_WEIGHTS"}
    avail = [e.identifier for e in mod.bl_rna.properties["vert_mapping"].enum_items]
    mod.vert_mapping = next((v for v in ("POLYINTERP_NEAREST", "NEAREST") if v in avail), avail[0])
    mod.mix_mode = "REPLACE"
    mod.mix_factor = 1.0
    bpy.ops.object.modifier_apply(modifier=mod.name)
    log("[Weights] Transfer complete")


# ── Bind ─────────────────────────────────────────────────────────────────────

def bind_to_armature(mesh_obj, arm_obj):
    ensure_object_mode()
    arm_obj.data.pose_position = "REST"
    depsgraph_update()

    for m in list(mesh_obj.modifiers):
        if m.type == "ARMATURE":
            mesh_obj.modifiers.remove(m)

    loc, rot, _scale = arm_obj.matrix_world.decompose()
    inv = (Matrix.Translation(loc) @ rot.to_matrix().to_4x4()).inverted()
    for v in mesh_obj.data.vertices:
        v.co = inv @ v.co
    mesh_obj.data.update()

    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type="ARMATURE", keep_transform=False)
    depsgraph_update()

    for m in mesh_obj.modifiers:
        if m.type == "ARMATURE":
            m.use_vertex_groups = True
            m.use_bone_envelopes = False
    log("[Bind] Mesh bound to UEFN armature")


# ── Main ──────────────────────────────────────────────────────────────────────

log(f"[BD_UEFNSkeleton] Input FBX:    {INPUT_FBX}")
log(f"[BD_UEFNSkeleton] UEFN ref FBX: {UEFN_REF_FBX}")
log(f"[BD_UEFNSkeleton] Output FBX:   {OUTPUT_FBX}")

# Clear default scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# ── 1. Import reference (UEFN mannequin) ─────────────────────────────────────
log("[BD_UEFNSkeleton] Importing UEFN reference...")
src_arms, src_meshes = import_fbx_to_collection(UEFN_REF_FBX, "Source")
src_arm  = src_arms[0]
src_mesh = max(src_meshes, key=lambda o: len(o.data.vertices)) if src_meshes else None
if src_mesh is None:
    raise RuntimeError("UEFN reference FBX has no mesh — cannot transfer weights")

# Put UEFN armature in REST so measurements are clean
src_arm.data.pose_position = "REST"
depsgraph_update()
log(f"[BD_UEFNSkeleton] Source: {src_arm.name} / {src_mesh.name} "
    f"({len(src_mesh.data.vertices)} verts, "
    f"{len(src_arm.data.bones)} bones)")

# ── 2. Import target (our character) ─────────────────────────────────────────
log("[BD_UEFNSkeleton] Importing character FBX...")
tgt_arms, tgt_meshes = import_fbx_to_collection(INPUT_FBX, "Target")
tgt_arm  = tgt_arms[0]
tgt_mesh = find_mesh(collection_for("Target"), tgt_arm)
log(f"[BD_UEFNSkeleton] Target: {tgt_arm.name} / {tgt_mesh.name} "
    f"({len(tgt_mesh.data.vertices)} verts, "
    f"{len(tgt_arm.data.bones)} bones)")
log(f"[BD_UEFNSkeleton] Target bone sample: "
    f"{list(_bone_names(tgt_arm))[:5]}")

# ── Step 0: Set target to REST + clear pose ───────────────────────────────────
log("[Step 0] Target → REST + clear pose")
tgt_arm.data.pose_position = "REST"
depsgraph_update()
bpy.ops.object.select_all(action="DESELECT")
tgt_arm.select_set(True)
bpy.context.view_layer.objects.active = tgt_arm
bpy.ops.object.mode_set(mode="POSE")
bpy.ops.pose.select_all(action="SELECT")
bpy.ops.pose.transforms_clear()
bpy.ops.object.mode_set(mode="OBJECT")
depsgraph_update()

if tgt_mesh.parent == tgt_arm:
    bpy.ops.object.select_all(action="DESELECT")
    tgt_mesh.select_set(True)
    bpy.context.view_layer.objects.active = tgt_mesh
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    depsgraph_update()

# ── Step 1: Scale match ───────────────────────────────────────────────────────
log("[Step 1] Scale matching")
src_h = z_height(src_mesh)
tgt_h = z_height(tgt_mesh)
if tgt_h < 1e-6:
    raise RuntimeError("Target mesh height near zero")
scale_factor = src_h / tgt_h
log(f"[Step 1] src_h={src_h:.4f}  tgt_h={tgt_h:.4f}  scale={scale_factor:.4f}")
tgt_arm.scale   *= scale_factor
tgt_mesh.scale  *= scale_factor
depsgraph_update()
bpy.ops.object.select_all(action="DESELECT")
for o in (tgt_arm, tgt_mesh):
    o.select_set(True)
bpy.context.view_layer.objects.active = tgt_arm
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
depsgraph_update()

# ── Step 2: Object-level alignment ───────────────────────────────────────────
log("[Step 2] Aligning to UEFN mannequin")
align_to_source(src_arm, tgt_arm)

# ── Step 3: Bake geometry (apply armature deform) ────────────────────────────
log("[Step 3] Baking target mesh geometry")
ensure_object_mode()
bpy.ops.object.select_all(action="DESELECT")
tgt_mesh.select_set(True)
bpy.context.view_layer.objects.active = tgt_mesh
for m in tgt_mesh.modifiers:
    if m.type == "ARMATURE" and m.object == tgt_arm:
        bpy.ops.object.modifier_apply(modifier=m.name)
        break

# ── Step 4: Clean ─────────────────────────────────────────────────────────────
log("[Step 4] Cleaning target")
for m in list(tgt_mesh.modifiers):
    if m.type == "ARMATURE":
        tgt_mesh.modifiers.remove(m)
tgt_mesh.vertex_groups.clear()
bpy.ops.object.select_all(action="DESELECT")
tgt_mesh.select_set(True)
bpy.context.view_layer.objects.active = tgt_mesh
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
depsgraph_update()

# ── Step 5: Create vgroups + weight transfer ──────────────────────────────────
log("[Step 5] Creating UEFN vertex groups and transferring weights")
create_vgroups(tgt_mesh, src_arm)
transfer_weights(src_mesh, tgt_mesh)

# ── Step 6: Bind to UEFN armature ────────────────────────────────────────────
log("[Step 6] Binding to UEFN armature")
bind_to_armature(tgt_mesh, src_arm)

# ── Step 7: Export ───────────────────────────────────────────────────────────
log(f"[Step 7] Exporting to {OUTPUT_FBX}")
os.makedirs(os.path.dirname(OUTPUT_FBX) or ".", exist_ok=True)

bpy.ops.object.select_all(action="DESELECT")
src_arm.select_set(True)
tgt_mesh.select_set(True)
bpy.context.view_layer.objects.active = src_arm

bpy.ops.export_scene.fbx(
    filepath=OUTPUT_FBX,
    check_existing=False,
    use_selection=True,
    object_types={"ARMATURE", "MESH"},
    use_mesh_modifiers=True,
    global_scale=1.0,
    apply_unit_scale=True,
    apply_scale_options="FBX_SCALE_ALL",
    axis_forward="-Y",
    axis_up="Z",
    add_leaf_bones=False,
    primary_bone_axis="Y",
    secondary_bone_axis="X",
    armature_nodetype="NULL",
    mesh_smooth_type="FACE",
    bake_anim=False,
    path_mode="COPY",
    embed_textures=False,
)

log(f"[BD_UEFNSkeleton] DONE → {OUTPUT_FBX}")
