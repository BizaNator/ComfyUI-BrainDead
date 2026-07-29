"""
Blender script to export MIA (Make-It-Animatable) rigged mesh to FBX.
Takes MIA inference output and a Mixamo template, creates rigged character FBX.

Usage: blender --background --python mia_export.py --
    --input_path <json> --output_path <fbx> --template_path <template_fbx>
    [--remove_fingers] [--reset_to_rest] [--apply_predicted]

Pose/joint semantics (ARTS-37 follow-up — OPT-IN, default unchanged):
  DEFAULT (no --apply_predicted): legacy behavior — the fixed T-pose Mixamo
    template armature is exported verbatim; predicted joints/pose are loaded
    but NOT applied. --reset_to_rest alone remains a no-op (as it has always
    been in this pack) so existing callers are byte-compatible.
  --apply_predicted: the template bones are moved to MIA's predicted joint
    positions (predicted REST skeleton — character-specific proportions
    instead of the fixed template).
  --apply_predicted + --reset_to_rest (and pose data present): additionally
    FK-poses the skeleton by MIA's predicted per-bone rest->current pose
    transforms (ortho6d local rotations, global-axis convention mirroring
    mia/utils.py::pose_local_to_global), so the exported skeleton lands in
    the INPUT MESH'S OWN POSE (e.g. A-pose). This is the route for rigging
    non-T-posed inputs without a T->A march downstream.
"""

import bpy
import sys
import os
import json
import numpy as np
from mathutils import Vector, Matrix, Quaternion
import math

# Parse arguments after '--'
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

# Parse named arguments
input_path = None
output_path = None
template_path = None
remove_fingers = False
reset_to_rest = False
apply_predicted = False

i = 0
while i < len(argv):
    if argv[i] == "--input_path" and i + 1 < len(argv):
        input_path = argv[i + 1]
        i += 2
    elif argv[i] == "--output_path" and i + 1 < len(argv):
        output_path = argv[i + 1]
        i += 2
    elif argv[i] == "--template_path" and i + 1 < len(argv):
        template_path = argv[i + 1]
        i += 2
    elif argv[i] == "--remove_fingers":
        remove_fingers = True
        i += 1
    elif argv[i] == "--reset_to_rest":
        reset_to_rest = True
        i += 1
    elif argv[i] == "--apply_predicted":
        apply_predicted = True
        i += 1
    else:
        i += 1

if not input_path or not output_path or not template_path:
    print("Usage: blender --background --python mia_export.py --")
    print("    --input_path <json> --output_path <fbx> --template_path <template_fbx>")
    print("    [--remove_fingers] [--reset_to_rest] [--apply_predicted]")
    sys.exit(1)

print(f"[MIA Export] Input: {input_path}")
print(f"[MIA Export] Output: {output_path}")
print(f"[MIA Export] Template: {template_path}")
print(f"[MIA Export] Remove fingers: {remove_fingers}")
print(f"[MIA Export] Reset to rest: {reset_to_rest}")
print(f"[MIA Export] Apply predicted: {apply_predicted}")

# Load JSON metadata
try:
    with open(input_path, 'r') as f:
        meta = json.load(f)

    # Get mesh path (GLB file saved by mia_inference.py)
    mesh_path = meta["mesh_path"]

    # Load bone weights
    bw_path = meta["bw_path"]
    bw_shape = meta["bw_shape"]
    skin_weights = np.fromfile(bw_path, dtype=np.float32).reshape(bw_shape)

    # Load joints
    joints_path = meta["joints_path"]
    joints_shape = meta["joints_shape"]
    joints = np.fromfile(joints_path, dtype=np.float32).reshape(joints_shape)

    # Load joint tails if available
    joints_tail = None
    if "joints_tail_path" in meta:
        joints_tail_path = meta["joints_tail_path"]
        joints_tail_shape = meta["joints_tail_shape"]
        joints_tail = np.fromfile(joints_tail_path, dtype=np.float32).reshape(joints_tail_shape)

    # Bone index mapping
    bones_idx_dict = meta["bones_idx_dict"]

    # Predicted rest->current per-bone pose transforms (only present when the
    # inference side was asked to serialize them; ortho6d local rotations,
    # shape (K, 6), row order = bones_idx_dict index order)
    pose = None
    if "pose_path" in meta:
        pose = np.fromfile(meta["pose_path"], dtype=np.float32).reshape(
            meta["pose_shape"])

    print(f"[MIA Export] Mesh path: {mesh_path}")
    print(f"[MIA Export] Skin weights shape: {skin_weights.shape}")
    print(f"[MIA Export] Joints: {len(joints)}")
    print(f"[MIA Export] Pose: {'present ' + str(pose.shape) if pose is not None else 'absent'}")
    print(f"[MIA Export] Bones: {list(bones_idx_dict.keys())}")

except Exception as e:
    print(f"[MIA Export] Failed to load input data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── predicted-skeleton helpers (opt-in --apply_predicted path) ─────────────
# All math runs in MIA's normalized Y-up frame (the frame of mesh.glb,
# joints.bin and pose.bin). Conversion to Blender's Z-up mirrors the glTF
# importer: (x, y, z) -> (x, -z, y).

def _mia_to_blender(p):
    return Vector((float(p[0]), float(-p[2]), float(p[1])))


def _ortho6d_to_matrix_np(o6):
    """numpy mirror of mia/utils.py::ortho6d_to_matrix (columns x, y, z)."""
    x = o6[..., 0:3]
    y_raw = o6[..., 3:6]
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x)
    return np.stack((x, y, z), axis=-1)


def _rot_about_point_np(R, c):
    """4x4 rotation of R about point c (mia/utils.py::get_rotation_about_point)."""
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = c - R @ c
    return M


def _fk_predicted_pose(rest_joints, parents, Rloc, order):
    """numpy mirror of mia/utils.py::pose_local_to_global
    (relative_to_source=False). Rotation axes are global (rest-world);
    rotation origin is each joint's posed position. parents: list with -1
    for roots. order: joint indices in parents-before-children order.
    Returns (G, posed): per-bone 4x4 global transforms and posed joint
    positions, both in the MIA frame."""
    K = rest_joints.shape[0]
    G = [None] * K
    posed = [None] * K
    for i in order:
        p = parents[i]
        if p is None or p < 0 or G[p] is None:
            M = _rot_about_point_np(Rloc[i], rest_joints[i])
            G[i] = M
            posed[i] = rest_joints[i].copy()
        else:
            posed[i] = (G[p] @ np.append(rest_joints[i], 1.0))[:3]
            M = _rot_about_point_np(Rloc[i], posed[i])
            G[i] = M @ G[p]
    return G, posed


def _bone_lookup(arm_data, name):
    """Find an edit/edit-mode bone by dict name with colon-variant fallbacks."""
    b = arm_data.edit_bones.get(name)
    if b is not None:
        return b
    b = arm_data.edit_bones.get(name.replace("mixamorig:", "mixamorig"))
    if b is not None:
        return b
    short = name.split(":")[-1]
    for cand in arm_data.edit_bones:
        if cand.name.split(":")[-1] == short:
            return cand
    return None


def _apply_predicted_skeleton(armature_obj, joints, joints_tail,
                              bones_idx_dict, pose6d):
    """Move the template armature onto MIA's predicted skeleton.

    joints: (K, 3) predicted REST joint positions (MIA frame).
    joints_tail: (K, 3) predicted tail positions or None.
    pose6d: (K, 6) ortho6d local rest->current rotations, or None. When
        given, the skeleton is FK-posed into the input mesh's own pose.
    Bone rolls/hierarchy are preserved from the template.
    """
    inv = armature_obj.matrix_world.inverted()
    K = len(bones_idx_dict)
    idx_of = dict(bones_idx_dict)

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')
    arm_data = armature_obj.data

    bone_of = {}
    parent_of = {}
    skipped = []
    for name, idx in idx_of.items():
        eb = _bone_lookup(arm_data, name)
        if eb is None:
            skipped.append(name)
            continue
        bone_of[idx] = eb
        p_idx = -1
        if eb.parent is not None:
            # parent may itself be outside the dict (extra root bones)
            for p_name, p_i in idx_of.items():
                pb = _bone_lookup(arm_data, p_name)
                if pb is not None and pb.name == eb.parent.name:
                    p_idx = p_i
                    break
        parent_of[idx] = p_idx
    if skipped:
        print(f"[MIA Export] WARNING: {len(skipped)} dict bones not found in "
              f"template armature: {skipped[:6]}{'...' if len(skipped) > 6 else ''}")

    def _depth(i):
        d, p = 0, parent_of.get(i, -1)
        while p is not None and p >= 0:
            d += 1
            p = parent_of.get(p, -1)
        return d

    order = sorted(bone_of.keys(), key=_depth)

    if joints.shape[0] < K:
        print(f"[MIA Export] WARNING: joints rows ({joints.shape[0]}) < dict "
              f"size ({K}); bones with out-of-range indices are left untouched")
        order = [i for i in order if i < joints.shape[0]]
    KJ = joints.shape[0]

    G = posed = None
    if pose6d is not None:
        Rloc = _ortho6d_to_matrix_np(pose6d)
        parents = [parent_of.get(i, -1) if i < KJ else -1 for i in range(KJ)]
        order_full = sorted(range(KJ), key=_depth)
        G_all, posed_all = _fk_predicted_pose(joints, parents, Rloc, order_full)
        G = {i: G_all[i] for i in order}
        posed = {i: posed_all[i] for i in order}

    # child map for tail fallback
    child_of = {}
    for i in order:
        p = parent_of.get(i, -1)
        if p is not None and p >= 0 and p not in child_of:
            child_of[p] = i

    moved = 0
    for i in order:
        eb = bone_of[i]
        head_w = posed[i] if posed is not None else joints[i]
        if joints_tail is not None:
            if posed is not None:
                tail_w = (G[i][:3, :3] @ (joints_tail[i] - joints[i])) + posed[i]
            else:
                tail_w = joints_tail[i]
        elif child_of.get(i) is not None:
            c = child_of[i]
            tail_w = posed[c] if posed is not None else joints[c]
        else:
            # leaf: keep template direction/length relative to its old head
            tail_w = None
        h_b = inv @ _mia_to_blender(head_w)
        eb.head = h_b
        if tail_w is not None:
            t_b = inv @ _mia_to_blender(tail_w)
            if (t_b - h_b).length < 1e-6:
                t_b = h_b + Vector((0, 0, 1e-4))
            eb.tail = t_b
        moved += 1

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[MIA Export] apply_predicted: moved {moved} bones "
          f"({'FK-posed to predicted pose' if posed is not None else 'predicted rest skeleton'})")


# Clean default scene
def clean_bpy():
    for c in bpy.data.actions:
        bpy.data.actions.remove(c)
    for c in bpy.data.armatures:
        bpy.data.armatures.remove(c)
    for c in bpy.data.cameras:
        bpy.data.cameras.remove(c)
    for c in bpy.data.collections:
        bpy.data.collections.remove(c)
    for c in bpy.data.images:
        bpy.data.images.remove(c)
    for c in bpy.data.materials:
        bpy.data.materials.remove(c)
    for c in bpy.data.meshes:
        bpy.data.meshes.remove(c)
    for c in bpy.data.objects:
        bpy.data.objects.remove(c)
    for c in bpy.data.textures:
        bpy.data.textures.remove(c)

clean_bpy()

# Import template FBX to get the armature structure
print(f"[MIA Export] Importing template: {template_path}")
try:
    bpy.ops.import_scene.fbx(filepath=template_path)
except Exception as e:
    print(f"[MIA Export] Failed to import template: {e}")
    sys.exit(1)

# Find armature in imported objects
armature_obj = None
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE':
        armature_obj = obj
        break

if not armature_obj:
    print("[MIA Export] No armature found in template!")
    sys.exit(1)

print(f"[MIA Export] Found armature: {armature_obj.name}")

# Remove template mesh(es), keep only armature
template_meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
for mesh_obj in template_meshes:
    bpy.data.objects.remove(mesh_obj, do_unlink=True)

# Import mesh using Blender's native GLB/GLTF importer
print(f"[MIA Export] Importing mesh from: {mesh_path}")
try:
    bpy.ops.import_scene.gltf(filepath=mesh_path)

    # Find the imported mesh object
    mesh_obj = None
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            mesh_obj = obj
            break

    if not mesh_obj:
        raise RuntimeError("No mesh found after GLB import")

    print(f"[MIA Export] Loaded mesh: {len(mesh_obj.data.vertices)} vertices, {len(mesh_obj.data.polygons)} faces")

except Exception as e:
    print(f"[MIA Export] Failed to import mesh: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Rename mesh object
mesh_obj.name = 'MIA_Character'

# ── opt-in: apply MIA's predicted skeleton / pose ──────────────────────────
# DEFAULT (no --apply_predicted): legacy behavior — the fixed T-pose template
# armature is exported verbatim; predicted joints/pose remain unused.
if apply_predicted:
    if reset_to_rest and pose is not None:
        print("[MIA Export] apply_predicted + reset_to_rest: FK-posing "
              "skeleton to MIA's predicted pose (input mesh's own pose)")
        _apply_predicted_skeleton(armature_obj, joints, joints_tail,
                                  bones_idx_dict, pose)
    else:
        if reset_to_rest and pose is None:
            print("[MIA Export] WARNING: reset_to_rest requested but no pose "
                  "data in input — applying predicted rest joints only")
        print("[MIA Export] apply_predicted: skeleton from MIA's predicted "
              "rest joints")
        _apply_predicted_skeleton(armature_obj, joints, joints_tail,
                                  bones_idx_dict, None)
elif reset_to_rest:
    print("[MIA Export] NOTE: --reset_to_rest without --apply_predicted is a "
          "legacy no-op (template T-pose skeleton exported verbatim)")

# Parent mesh to armature
mesh_obj.parent = armature_obj
mesh_obj.parent_type = 'ARMATURE'

# Add armature modifier
armature_mod = mesh_obj.modifiers.new(name='Armature', type='ARMATURE')
armature_mod.object = armature_obj
armature_mod.use_bone_envelopes = False
armature_mod.use_vertex_groups = True

# Create vertex groups and assign weights
print("[MIA Export] Assigning skin weights...")
bone_names = list(bones_idx_dict.keys())
num_vertices = len(mesh_obj.data.vertices)

# Create vertex groups for each bone
for bone_name in bone_names:
    if bone_name not in mesh_obj.vertex_groups:
        mesh_obj.vertex_groups.new(name=bone_name)

# Assign weights
for v_idx in range(num_vertices):
    for bone_name, bone_idx in bones_idx_dict.items():
        if bone_idx < skin_weights.shape[1]:
            weight = float(skin_weights[v_idx, bone_idx])
            if weight > 0.001:  # Skip very small weights
                vg = mesh_obj.vertex_groups.get(bone_name)
                if vg:
                    vg.add([v_idx], weight, 'REPLACE')

print(f"[MIA Export] Assigned weights to {len(bone_names)} bones")

# Optionally remove finger bones
if remove_fingers:
    print("[MIA Export] Removing finger bones...")
    finger_keywords = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky', 'Hand']
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')
    armature = armature_obj.data
    bones_to_remove = []
    for bone in armature.edit_bones:
        if any(kw in bone.name for kw in finger_keywords):
            bones_to_remove.append(bone.name)
    for bone_name in bones_to_remove:
        bone = armature.edit_bones.get(bone_name)
        if bone:
            armature.edit_bones.remove(bone)
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[MIA Export] Removed {len(bones_to_remove)} finger bones")

# Export to FBX
print(f"[MIA Export] Exporting to: {output_path}")
os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

try:
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        check_existing=False,
        add_leaf_bones=False,
        path_mode='COPY',
        embed_textures=True,
    )
    print(f"[MIA Export] ✓ Export complete: {output_path}")
except Exception as e:
    print(f"[MIA Export] Export failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[MIA Export] Done!")
