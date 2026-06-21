"""
Blender script to export skeleton data to FBX file.
This script takes skeleton data from pickle and creates a Blender armature, then exports to FBX.
Usage: blender --background --python blender_export_fbx.py -- <input_pkl> <output_fbx> [options]
"""

import bpy
import sys
import os
import pickle
import numpy as np
from mathutils import Vector, Matrix, Quaternion
from collections import defaultdict
import math
import tempfile
import base64
import struct
import zlib

# Get arguments after '--'
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

if len(argv) < 2:
    print("Usage: blender --background --python blender_export_fbx.py -- <input_pkl> <output_fbx> [options]")
    sys.exit(1)

input_pkl = argv[0]
output_fbx = argv[1]

# Parse optional parameters
extrude_size = 0.03
add_root = False
use_extrude_bone = True
use_connect_unique_child = True
extrude_from_parent = True

for arg in argv[2:]:
    if arg.startswith("--extrude_size="):
        extrude_size = float(arg.split("=")[1])
    elif arg == "--add_root":
        add_root = True
    elif arg == "--no_extrude_bone":
        use_extrude_bone = False
    elif arg == "--no_connect_unique_child":
        use_connect_unique_child = False
    elif arg == "--no_extrude_from_parent":
        extrude_from_parent = False

print(f"[Blender FBX Export] Input: {input_pkl}")
print(f"[Blender FBX Export] Output: {output_fbx}")

# Load skeleton data from pickle
try:
    with open(input_pkl, 'rb') as f:
        data = pickle.load(f)  # No numpy imports needed - only plain Python types!

    # Now convert plain Python lists to numpy arrays using Blender's own numpy
    joints = np.array(data['joints'], dtype=np.float32)
    parents = data['parents']  # Plain list, keep as-is
    names = data['names']  # Plain list, keep as-is

    # Optional data - convert to numpy if present and not None
    vertices = np.array(data['vertices'], dtype=np.float32) if data.get('vertices') is not None else None
    faces = np.array(data['faces'], dtype=np.int32) if data.get('faces') is not None else None
    skin = np.array(data['skin'], dtype=np.float32) if data.get('skin') is not None else None
    tails = np.array(data['tails'], dtype=np.float32) if data.get('tails') is not None else None

    # UV data - convert to numpy if present
    uv_coords = np.array(data['uv_coords'], dtype=np.float32) if data.get('uv_coords') is not None and len(data.get('uv_coords', [])) > 0 else None
    uv_faces = np.array(data['uv_faces'], dtype=np.int32) if data.get('uv_faces') is not None and len(data.get('uv_faces', [])) > 0 else None

    # Texture data - keep as string (base64 encoded)
    texture_data_base64 = data.get('texture_data_base64', "")
    texture_format = data.get('texture_format', "PNG")
    material_name_from_data = data.get('material_name', "Material")

    if texture_data_base64 and len(texture_data_base64) > 0:
        print(f"[Blender FBX Export] Found texture data: {texture_format} ({len(texture_data_base64) // 1024}KB base64)")
    else:
        print(f"[Blender FBX Export] No texture data found")

    print(f"[Blender FBX Export] Loaded skeleton with {len(joints)} joints")
    print(f"[Blender FBX Export] Joints shape: {joints.shape}")
    print(f"[Blender FBX Export] Tails: {'None' if tails is None else f'shape {tails.shape}'}")
    if vertices is not None:
        print(f"[Blender FBX Export] Found mesh with {len(vertices)} vertices")
    if skin is not None:
        print(f"[Blender FBX Export] Skin weights shape: {skin.shape}")

    # DEBUG: Show bounds of what Blender is receiving
    print(f"[Blender FBX Export] DEBUG - Joints bounds: {joints.min(axis=0)} to {joints.max(axis=0)}")
    if tails is not None:
        print(f"[Blender FBX Export] DEBUG - Tails bounds: {tails.min(axis=0)} to {tails.max(axis=0)}")
    if vertices is not None:
        print(f"[Blender FBX Export] DEBUG - Mesh vertices bounds: {vertices.min(axis=0)} to {vertices.max(axis=0)}")

        # Debug: Y position of vertex with highest Z value
        max_z_idx = vertices[:, 2].argmax()
        max_z_vertex = vertices[max_z_idx]
        print(f"[Blender FBX Export] DEBUG - Mesh vertex with max Z: position={max_z_vertex}, Z={max_z_vertex[2]:.6f}, Y={max_z_vertex[1]:.6f}")

    max_z_joint_idx = joints[:, 2].argmax()
    max_z_joint = joints[max_z_joint_idx]
    print(f"[Blender FBX Export] DEBUG - Skeleton joint with max Z: position={max_z_joint}, Z={max_z_joint[2]:.6f}, Y={max_z_joint[1]:.6f}")
except Exception as e:
    print(f"[Blender FBX Export] Failed to load pickle: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

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

# === T-POSE CONVERSION FOR SMPL SKELETONS ===
# Check if this is an SMPL skeleton and convert to T-pose if needed
# IMPORTANT: This must happen BEFORE mesh creation so vertices are transformed
SMPL_JOINT_NAMES_CHECK = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee',
                    'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
                    'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
                    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist']

is_smpl_skeleton = len(names) == 22 and all(n in SMPL_JOINT_NAMES_CHECK for n in names)

if is_smpl_skeleton:
    print("[Blender FBX Export] Detected SMPL skeleton, checking if T-pose conversion needed...")

    # Get arm joint indices
    l_shoulder_idx = names.index('L_Shoulder')
    l_elbow_idx = names.index('L_Elbow')
    l_wrist_idx = names.index('L_Wrist')
    r_shoulder_idx = names.index('R_Shoulder')
    r_elbow_idx = names.index('R_Elbow')
    r_wrist_idx = names.index('R_Wrist')

    l_shoulder = joints[l_shoulder_idx]
    l_elbow = joints[l_elbow_idx]
    l_wrist = joints[l_wrist_idx]
    r_shoulder = joints[r_shoulder_idx]
    r_elbow = joints[r_elbow_idx]
    r_wrist = joints[r_wrist_idx]

    # Detect mesh orientation - which axis is lateral (left-right)?
    shoulder_diff = r_shoulder - l_shoulder
    print(f"[Blender FBX Export] Shoulder diff (R-L): {shoulder_diff}")

    if abs(shoulder_diff[0]) > abs(shoulder_diff[2]):
        lateral_axis = 'X'
        l_tpose_dir = np.array([1.0, 0.0, 0.0])
        r_tpose_dir = np.array([-1.0, 0.0, 0.0])
    else:
        lateral_axis = 'Z'
        # Left shoulder has smaller Z means left points toward -Z
        if l_shoulder[2] < r_shoulder[2]:
            l_tpose_dir = np.array([0.0, 0.0, -1.0])
            r_tpose_dir = np.array([0.0, 0.0, 1.0])
        else:
            l_tpose_dir = np.array([0.0, 0.0, 1.0])
            r_tpose_dir = np.array([0.0, 0.0, -1.0])

    print(f"[Blender FBX Export] Lateral axis: {lateral_axis}, L_tpose_dir: {l_tpose_dir}, R_tpose_dir: {r_tpose_dir}")

    # Check if already T-posed (arm Y component near zero)
    l_arm_vec = l_wrist - l_shoulder
    l_arm_vec_norm = l_arm_vec / (np.linalg.norm(l_arm_vec) + 1e-8)

    if abs(l_arm_vec_norm[1]) < 0.1:
        print("[Blender FBX Export] Arms already horizontal (T-pose), skipping conversion")
    else:
        print(f"[Blender FBX Export] Arms not horizontal (Y component: {l_arm_vec_norm[1]:.3f}), converting to T-pose...")

        # Compute arm lengths
        l_upper_len = np.linalg.norm(l_elbow - l_shoulder)
        l_lower_len = np.linalg.norm(l_wrist - l_elbow)
        r_upper_len = np.linalg.norm(r_elbow - r_shoulder)
        r_lower_len = np.linalg.norm(r_wrist - r_elbow)

        # Compute new T-pose joint positions
        new_l_elbow = l_shoulder + l_tpose_dir * l_upper_len
        new_l_wrist = new_l_elbow + l_tpose_dir * l_lower_len
        new_r_elbow = r_shoulder + r_tpose_dir * r_upper_len
        new_r_wrist = new_r_elbow + r_tpose_dir * r_lower_len

        # Compute rotations using mathutils
        l_arm_vec_v = Vector(l_arm_vec).normalized()
        new_l_arm_vec = Vector(new_l_wrist - l_shoulder).normalized()
        l_rotation = l_arm_vec_v.rotation_difference(new_l_arm_vec)

        r_arm_vec = r_wrist - r_shoulder
        r_arm_vec_v = Vector(r_arm_vec).normalized()
        new_r_arm_vec = Vector(new_r_wrist - r_shoulder).normalized()
        r_rotation = r_arm_vec_v.rotation_difference(new_r_arm_vec)

        print(f"[Blender FBX Export] Left arm rotation: {math.degrees(l_rotation.angle):.1f}°")
        print(f"[Blender FBX Export] Right arm rotation: {math.degrees(r_rotation.angle):.1f}°")

        # Transform mesh vertices if skin weights available
        if vertices is not None and skin is not None:
            print(f"[Blender FBX Export] Transforming {len(vertices)} mesh vertices...")

            left_arm_bones = {'L_Shoulder', 'L_Elbow', 'L_Wrist'}
            right_arm_bones = {'R_Shoulder', 'R_Elbow', 'R_Wrist'}

            # Get bone indices for weight lookup
            left_bone_indices = [names.index(b) for b in left_arm_bones if b in names]
            right_bone_indices = [names.index(b) for b in right_arm_bones if b in names]

            transformed_count = 0
            for v_idx in range(len(vertices)):
                # Sum weights for left and right arm bones
                left_weight = sum(skin[v_idx, idx] for idx in left_bone_indices)
                right_weight = sum(skin[v_idx, idx] for idx in right_bone_indices)

                if left_weight < 0.001 and right_weight < 0.001:
                    continue

                displacement = np.zeros(3)

                if left_weight > 0.001:
                    # Rotate around shoulder pivot
                    rel_pos = vertices[v_idx] - l_shoulder
                    rotated = np.array(l_rotation @ Vector(rel_pos))
                    displacement += (rotated - rel_pos) * left_weight

                if right_weight > 0.001:
                    # Rotate around shoulder pivot
                    rel_pos = vertices[v_idx] - r_shoulder
                    rotated = np.array(r_rotation @ Vector(rel_pos))
                    displacement += (rotated - rel_pos) * right_weight

                vertices[v_idx] += displacement
                transformed_count += 1

            print(f"[Blender FBX Export] Transformed {transformed_count} vertices")

        # Update joint positions
        joints[l_elbow_idx] = new_l_elbow
        joints[l_wrist_idx] = new_l_wrist
        joints[r_elbow_idx] = new_r_elbow
        joints[r_wrist_idx] = new_r_wrist

        # Update tails if provided
        if tails is not None:
            # Shoulder tails point to elbow
            tails[l_shoulder_idx] = new_l_elbow
            tails[r_shoulder_idx] = new_r_elbow
            # Elbow tails point to wrist
            tails[l_elbow_idx] = new_l_wrist
            tails[r_elbow_idx] = new_r_wrist
            # Wrist tails extend in T-pose direction
            wrist_tail_len = np.linalg.norm(tails[l_wrist_idx] - joints[l_wrist_idx]) if tails is not None else 0.05
            tails[l_wrist_idx] = new_l_wrist + l_tpose_dir * wrist_tail_len
            tails[r_wrist_idx] = new_r_wrist + r_tpose_dir * wrist_tail_len

        print("[Blender FBX Export] ✓ T-pose conversion complete")

        # Debug: show new bounds
        if vertices is not None:
            print(f"[Blender FBX Export] T-posed mesh bounds: {vertices.min(axis=0)} to {vertices.max(axis=0)}")

# === MIXAMO NORMALIZATION ===
# Normalize Mixamo skeletons to Mixamo-compatible rest pose for animation
is_mixamo_skeleton = any(n.startswith('mixamorig:') for n in names)

if is_mixamo_skeleton and vertices is not None and skin is not None:
    print("[Blender FBX Export] Detected Mixamo skeleton, normalizing to Mixamo rest pose...")

    # Get key bone indices
    hips_idx = None
    head_idx = None
    l_arm_idx = None
    r_arm_idx = None
    l_forearm_idx = None
    r_forearm_idx = None
    l_hand_idx = None
    r_hand_idx = None

    for i, name in enumerate(names):
        if name == 'mixamorig:Hips':
            hips_idx = i
        elif name == 'mixamorig:Head':
            head_idx = i
        elif name == 'mixamorig:LeftArm':
            l_arm_idx = i
        elif name == 'mixamorig:RightArm':
            r_arm_idx = i
        elif name == 'mixamorig:LeftForeArm':
            l_forearm_idx = i
        elif name == 'mixamorig:RightForeArm':
            r_forearm_idx = i
        elif name == 'mixamorig:LeftHand':
            l_hand_idx = i
        elif name == 'mixamorig:RightHand':
            r_hand_idx = i

    # === STEP 0: ORIENT MODEL TO MIXAMO STANDARD ===
    # Mixamo expects: face toward -Y, arms along X, Z up
    # We need to detect current orientation and rotate if needed

    if l_arm_idx is not None and r_arm_idx is not None and hips_idx is not None:
        l_shoulder = joints[l_arm_idx]
        r_shoulder = joints[r_arm_idx]
        hips = joints[hips_idx]

        # Get head position for up direction (fallback to highest joint if no head)
        if head_idx is not None:
            head = joints[head_idx]
        else:
            head = joints[np.argmax(joints[:, 2])]  # Highest Z joint

        # Compute current coordinate system
        # Lateral: from left shoulder to right shoulder
        lateral_vec = r_shoulder - l_shoulder
        lateral_vec = lateral_vec / (np.linalg.norm(lateral_vec) + 1e-8)

        # Up: from hips to head
        up_vec = head - hips
        up_vec = up_vec / (np.linalg.norm(up_vec) + 1e-8)

        # Forward: cross product of up and lateral (right-hand rule)
        forward_vec = np.cross(up_vec, lateral_vec)
        forward_vec = forward_vec / (np.linalg.norm(forward_vec) + 1e-8)

        print(f"[Blender FBX Export] Current orientation:")
        print(f"[Blender FBX Export]   Lateral (L→R): {lateral_vec}")
        print(f"[Blender FBX Export]   Up (Hips→Head): {up_vec}")
        print(f"[Blender FBX Export]   Forward: {forward_vec}")

        # Mixamo target orientation:
        # Lateral = +X (left shoulder at -X, right at +X... wait, that's opposite)
        # Actually in Mixamo: LeftArm is at positive X, RightArm at negative X
        # So lateral vector (R-L) should point toward -X
        target_lateral = np.array([-1.0, 0.0, 0.0])
        target_up = np.array([0.0, 0.0, 1.0])
        target_forward = np.array([0.0, -1.0, 0.0])  # Face toward -Y

        # Check if we need to rotate around Z to align lateral with X axis
        # Project lateral onto XY plane
        lateral_xy = np.array([lateral_vec[0], lateral_vec[1], 0])
        lateral_xy_len = np.linalg.norm(lateral_xy)

        if lateral_xy_len > 0.1:  # Lateral has significant XY component
            lateral_xy = lateral_xy / lateral_xy_len

            # Angle to rotate around Z to align lateral with target
            # Target lateral in XY is [-1, 0]
            target_lateral_xy = np.array([-1.0, 0.0])

            # Compute angle between current and target lateral
            dot = lateral_xy[0] * target_lateral_xy[0] + lateral_xy[1] * target_lateral_xy[1]
            cross_z = lateral_xy[0] * target_lateral_xy[1] - lateral_xy[1] * target_lateral_xy[0]
            z_rotation_angle = math.atan2(cross_z, dot)

            if abs(z_rotation_angle) > 0.05:  # More than ~3 degrees
                print(f"[Blender FBX Export] Rotating {math.degrees(z_rotation_angle):.1f}° around Z to align with Mixamo orientation")

                # Create rotation matrix around Z
                cos_a = math.cos(z_rotation_angle)
                sin_a = math.sin(z_rotation_angle)

                def rotate_z(points):
                    """Rotate points around Z axis"""
                    rotated = np.zeros_like(points)
                    rotated[..., 0] = cos_a * points[..., 0] - sin_a * points[..., 1]
                    rotated[..., 1] = sin_a * points[..., 0] + cos_a * points[..., 1]
                    rotated[..., 2] = points[..., 2]
                    return rotated

                # Rotate everything
                vertices = rotate_z(vertices)
                joints = rotate_z(joints)
                if tails is not None:
                    tails = rotate_z(tails)

                # Update shoulder positions after rotation
                l_shoulder = joints[l_arm_idx]
                r_shoulder = joints[r_arm_idx]

                print(f"[Blender FBX Export] ✓ Model rotated to Mixamo orientation")
            else:
                print(f"[Blender FBX Export] Model already aligned with Mixamo orientation (within 3°)")
        else:
            print(f"[Blender FBX Export] Warning: Lateral vector has no XY component, skipping orientation fix")

    # 1. T-POSE CONVERSION (if arms are not horizontal)
    # After orientation fix, arms should extend along X axis
    # Mixamo: LeftArm at +X, RightArm at -X
    if l_arm_idx is not None and r_arm_idx is not None and l_hand_idx is not None and r_hand_idx is not None:
        l_shoulder = joints[l_arm_idx]
        r_shoulder = joints[r_arm_idx]
        l_hand = joints[l_hand_idx]
        r_hand = joints[r_hand_idx]

        # T-pose directions (Mixamo standard: left arm +X, right arm -X)
        l_tpose_dir = np.array([1.0, 0.0, 0.0])
        r_tpose_dir = np.array([-1.0, 0.0, 0.0])

        # Check if arms need rotation
        l_arm_vec = l_hand - l_shoulder
        l_arm_vec_norm = l_arm_vec / (np.linalg.norm(l_arm_vec) + 1e-8)

        # Arms should be mostly along X axis for T-pose
        l_arm_x_component = abs(l_arm_vec_norm[0])
        needs_tpose = l_arm_x_component < 0.9  # Arms not yet horizontal along X

        if needs_tpose:
            print(f"[Blender FBX Export] Arms not in T-pose (X component: {l_arm_x_component:.3f}), converting...")

            # Get forearm positions if available
            l_elbow = joints[l_forearm_idx] if l_forearm_idx else None
            r_elbow = joints[r_forearm_idx] if r_forearm_idx else None

            # Compute arm segment lengths
            if l_elbow is not None:
                l_upper_len = np.linalg.norm(l_elbow - l_shoulder)
                l_lower_len = np.linalg.norm(l_hand - l_elbow)
            else:
                l_upper_len = np.linalg.norm(l_hand - l_shoulder) / 2
                l_lower_len = l_upper_len

            if r_elbow is not None:
                r_upper_len = np.linalg.norm(r_elbow - r_shoulder)
                r_lower_len = np.linalg.norm(r_hand - r_elbow)
            else:
                r_upper_len = np.linalg.norm(r_hand - r_shoulder) / 2
                r_lower_len = r_upper_len

            # Compute new T-pose positions
            new_l_elbow = l_shoulder + l_tpose_dir * l_upper_len if l_elbow is not None else None
            new_l_hand = (new_l_elbow if new_l_elbow is not None else l_shoulder) + l_tpose_dir * l_lower_len
            new_r_elbow = r_shoulder + r_tpose_dir * r_upper_len if r_elbow is not None else None
            new_r_hand = (new_r_elbow if new_r_elbow is not None else r_shoulder) + r_tpose_dir * r_lower_len

            # Compute rotations for vertex transformation
            # IMPORTANT: Use axis-angle rotation around the cross product axis to avoid twist
            l_arm_vec_v = Vector(l_arm_vec).normalized()
            new_l_arm_vec = Vector(new_l_hand - l_shoulder).normalized()

            # Compute rotation axis (perpendicular to both vectors)
            l_rot_axis = l_arm_vec_v.cross(new_l_arm_vec)
            if l_rot_axis.length > 0.0001:
                l_rot_axis.normalize()
                # Compute rotation angle
                l_rot_angle = math.acos(max(-1, min(1, l_arm_vec_v.dot(new_l_arm_vec))))
                l_rotation = Quaternion(l_rot_axis, l_rot_angle)
            else:
                l_rotation = Quaternion()  # Identity

            r_arm_vec = r_hand - r_shoulder
            r_arm_vec_v = Vector(r_arm_vec).normalized()
            new_r_arm_vec = Vector(new_r_hand - r_shoulder).normalized()

            r_rot_axis = r_arm_vec_v.cross(new_r_arm_vec)
            if r_rot_axis.length > 0.0001:
                r_rot_axis.normalize()
                r_rot_angle = math.acos(max(-1, min(1, r_arm_vec_v.dot(new_r_arm_vec))))
                r_rotation = Quaternion(r_rot_axis, r_rot_angle)
            else:
                r_rotation = Quaternion()  # Identity

            print(f"[Blender FBX Export] Left arm rotation: {math.degrees(l_rot_angle if l_rot_axis.length > 0.0001 else 0):.1f} deg around {l_rot_axis[:]}")
            print(f"[Blender FBX Export] Right arm rotation: {math.degrees(r_rot_angle if r_rot_axis.length > 0.0001 else 0):.1f} deg around {r_rot_axis[:]}")

            # Get bone indices for weight lookup
            left_arm_bones = {'mixamorig:LeftArm', 'mixamorig:LeftForeArm', 'mixamorig:LeftHand'}
            right_arm_bones = {'mixamorig:RightArm', 'mixamorig:RightForeArm', 'mixamorig:RightHand'}

            # Add finger bones if present
            for name in names:
                if name.startswith('mixamorig:LeftHand'):
                    left_arm_bones.add(name)
                elif name.startswith('mixamorig:RightHand'):
                    right_arm_bones.add(name)

            left_bone_indices = [names.index(b) for b in left_arm_bones if b in names]
            right_bone_indices = [names.index(b) for b in right_arm_bones if b in names]

            # Transform vertices using skin weights
            transformed_count = 0
            for v_idx in range(len(vertices)):
                left_weight = sum(skin[v_idx, idx] for idx in left_bone_indices if idx < skin.shape[1])
                right_weight = sum(skin[v_idx, idx] for idx in right_bone_indices if idx < skin.shape[1])

                if left_weight < 0.001 and right_weight < 0.001:
                    continue

                displacement = np.zeros(3)

                if left_weight > 0.001:
                    rel_pos = vertices[v_idx] - l_shoulder
                    rotated = np.array(l_rotation @ Vector(rel_pos))
                    displacement += (rotated - rel_pos) * left_weight

                if right_weight > 0.001:
                    rel_pos = vertices[v_idx] - r_shoulder
                    rotated = np.array(r_rotation @ Vector(rel_pos))
                    displacement += (rotated - rel_pos) * right_weight

                vertices[v_idx] += displacement
                transformed_count += 1

            print(f"[Blender FBX Export] Transformed {transformed_count} vertices for T-pose")

            # Update joint positions
            if l_forearm_idx is not None and new_l_elbow is not None:
                joints[l_forearm_idx] = new_l_elbow
            joints[l_hand_idx] = new_l_hand

            if r_forearm_idx is not None and new_r_elbow is not None:
                joints[r_forearm_idx] = new_r_elbow
            joints[r_hand_idx] = new_r_hand

            # Update tails
            if tails is not None:
                if l_forearm_idx is not None and new_l_elbow is not None:
                    tails[l_arm_idx] = new_l_elbow
                    tails[l_forearm_idx] = new_l_hand
                else:
                    tails[l_arm_idx] = new_l_hand

                if r_forearm_idx is not None and new_r_elbow is not None:
                    tails[r_arm_idx] = new_r_elbow
                    tails[r_forearm_idx] = new_r_hand
                else:
                    tails[r_arm_idx] = new_r_hand

                # Update hand tails
                hand_tail_len = 0.05
                tails[l_hand_idx] = new_l_hand + l_tpose_dir * hand_tail_len
                tails[r_hand_idx] = new_r_hand + r_tpose_dir * hand_tail_len

            print("[Blender FBX Export] ✓ Mixamo T-pose conversion complete")
        else:
            print("[Blender FBX Export] Arms already horizontal, skipping T-pose conversion")

    # 2. SCALE TO HUMAN SIZE
    # Current mesh is normalized to [-1, 1], scale to realistic human height
    current_height = vertices[:, 2].max() - vertices[:, 2].min()
    target_height = 1.7  # meters (average human height)

    if current_height > 0.01:  # Avoid division by zero
        scale_factor = target_height / current_height
        print(f"[Blender FBX Export] Scaling from {current_height:.3f} to {target_height:.1f}m (factor: {scale_factor:.2f}x)")

        vertices *= scale_factor
        joints *= scale_factor
        if tails is not None:
            tails *= scale_factor
    else:
        scale_factor = 1.0
        print(f"[Blender FBX Export] Warning: mesh height too small ({current_height:.6f}), skipping scale")

    # 3. POSITION FEET AT GROUND LEVEL (Z=0)
    # This ensures the mesh stands on the ground like official Mixamo
    mesh_min_z = vertices[:, 2].min()
    z_offset = -mesh_min_z  # Move so feet are at Z=0

    print(f"[Blender FBX Export] Moving feet from Z={mesh_min_z:.3f} to Z=0.0 (offset: {z_offset:.3f})")

    vertices[:, 2] += z_offset
    joints[:, 2] += z_offset
    if tails is not None:
        tails[:, 2] += z_offset

    if hips_idx is not None:
        print(f"[Blender FBX Export] Hips now at Z={joints[hips_idx, 2]:.3f}")

    print(f"[Blender FBX Export] ✓ Mixamo normalization complete")
    print(f"[Blender FBX Export]   Final mesh bounds: {vertices.min(axis=0)} to {vertices.max(axis=0)}")
    if hips_idx is not None:
        print(f"[Blender FBX Export]   Hips position: {joints[hips_idx]}")

    # 4. CONVERT TO MIXAMO LOCAL SPACE
    # Official Mixamo FBX stores coordinates in Y-up format (local space)
    # with 90° X rotation converting to Z-up (world space)
    # We need to: 1) Convert Z-up to Y-up, 2) Scale by 100x
    print(f"[Blender FBX Export] Converting to Mixamo local space (Y-up, 100x scale)...")

    # Convert from Z-up to Y-up: (x, y, z) → (x, z, -y)
    # This is the inverse of what 90° X rotation does
    def convert_to_yup(coords):
        result = np.zeros_like(coords)
        result[..., 0] = coords[..., 0]      # X stays X
        result[..., 1] = coords[..., 2]      # Z becomes Y
        result[..., 2] = -coords[..., 1]     # -Y becomes Z
        return result

    vertices = convert_to_yup(vertices) * 100.0
    joints = convert_to_yup(joints) * 100.0
    if tails is not None:
        tails = convert_to_yup(tails) * 100.0

    print(f"[Blender FBX Export]   Converted mesh bounds: {vertices.min(axis=0)} to {vertices.max(axis=0)}")

# Make collection
collection = bpy.data.collections.new('new_collection')
bpy.context.scene.collection.children.link(collection)

# Make mesh if vertices provided
if vertices is not None:
    mesh = bpy.data.meshes.new('mesh')
    if faces is None:
        faces = []
    mesh.from_pydata(vertices.tolist(), [], faces.tolist() if faces is not None else [])
    mesh.update()

    # Add UV coordinates if available
    if uv_coords is not None and uv_faces is not None and len(uv_coords) > 0:
        print(f"[Blender FBX Export] Adding UV coordinates: {len(uv_coords)} UVs for {len(uv_faces)} faces")
        # Create UV layer
        uv_layer = mesh.uv_layers.new(name='UVMap')

        # Apply UV coordinates per face loop
        for face_idx, poly in enumerate(mesh.polygons):
            if face_idx < len(uv_faces):
                for loop_offset, loop_idx in enumerate(poly.loop_indices):
                    uv_idx = uv_faces[face_idx][loop_offset]
                    if uv_idx < len(uv_coords):
                        uv_layer.data[loop_idx].uv = uv_coords[uv_idx]

        print(f"[Blender FBX Export] ✓ UV coordinates applied")
    else:
        print(f"[Blender FBX Export] No UV coordinates available")

    # Make object from mesh
    obj = bpy.data.objects.new('character', mesh)

    # Add object to scene collection
    collection.objects.link(obj)

    # Create and apply textured material if texture data is available
    if texture_data_base64 and len(texture_data_base64) > 0:
        print(f"[Blender FBX Export] Creating textured material...")
        try:
            # Decode base64 PNG to raw bytes
            png_data = base64.b64decode(texture_data_base64)

            # Save PNG to temporary file (no PIL needed - already PNG format)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp.write(png_data)
                tmp_texture_path = tmp.name

            print(f"[Blender FBX Export] Saved texture to temp file: {tmp_texture_path}")
            print(f"[Blender FBX Export] PNG data size: {len(png_data)} bytes")

            # Create material with nodes
            mat = bpy.data.materials.new(name=material_name_from_data)
            mat.use_nodes = True
            mat.blend_method = 'OPAQUE'  # Prevent transparency
            mat.shadow_method = 'OPAQUE'

            # Clear default nodes
            mat.node_tree.nodes.clear()

            # Create nodes
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            # Create image texture node
            img_node = nodes.new(type='ShaderNodeTexImage')
            img_node.location = (-300, 300)

            # Create principled BSDF node
            bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_node.location = (0, 300)

            # Set material to be non-metallic and fairly rough (more diffuse/matte)
            # This prevents the shiny/translucent look when only diffuse texture is available
            bsdf_node.inputs['Metallic'].default_value = 0.0  # Non-metallic
            bsdf_node.inputs['Roughness'].default_value = 0.8  # Fairly rough/matte
            bsdf_node.inputs['Specular IOR Level'].default_value = 0.3  # Reduced specular
            bsdf_node.inputs['Alpha'].default_value = 1.0  # Fully opaque

            # Create output node
            output_node = nodes.new(type='ShaderNodeOutputMaterial')
            output_node.location = (300, 300)

            # Link nodes: Image -> BSDF -> Output
            links.new(img_node.outputs['Color'], bsdf_node.inputs['Base Color'])
            links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

            # Load image into Blender (Blender can load PNG natively)
            blender_image = bpy.data.images.load(tmp_texture_path)
            img_node.image = blender_image

            print(f"[Blender FBX Export] Loaded texture: {blender_image.size[0]}x{blender_image.size[1]}")

            # Pack image into blend file (so it's embedded in FBX)
            blender_image.pack()

            print(f"[Blender FBX Export] Packed texture image into blend file")

            # Assign material to mesh
            obj.data.materials.append(mat)

            # Assign material to all faces
            for poly in obj.data.polygons:
                poly.material_index = 0

            print(f"[Blender FBX Export] ✓ Textured material applied: {material_name_from_data}")

            # Clean up temp file
            try:
                os.remove(tmp_texture_path)
            except:
                pass

        except Exception as tex_err:
            print(f"[Blender FBX Export] Warning: Could not apply texture: {tex_err}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[Blender FBX Export] No texture to apply")

# Create armature
print("[Blender FBX Export] Creating armature...")
try:
    bpy.ops.object.armature_add(enter_editmode=True)
    armature = bpy.data.armatures.get('Armature')
    edit_bones = armature.edit_bones

    J = joints.shape[0]
    if tails is None:
        print(f"[Blender FBX Export] Tails not provided, auto-generating...")
        tails = joints.copy()
        tails[:, 2] += extrude_size
        print(f"[Blender FBX Export] Generated tails shape: {tails.shape}")

    connects = [False for _ in range(J)]
    children = defaultdict(list)
    for i in range(1, J):
        if parents[i] is not None and parents[i] != -1:
            children[parents[i]].append(i)

    if tails is not None:
        if use_extrude_bone:
            for i in range(J):
                if len(children[i]) != 1 and extrude_from_parent and i != 0:
                    if parents[i] is not None and parents[i] != -1:
                        pjoint = joints[parents[i]]
                        joint = joints[i]
                        d = joint - pjoint
                        if np.linalg.norm(d) < 0.000001:
                            d = np.array([0., 0., 1.])  # in case son.head == parent.head
                        else:
                            d = d / np.linalg.norm(d)
                        tails[i] = joint + d * extrude_size
        if use_connect_unique_child:
            for i in range(J):
                if len(children[i]) == 1:
                    child = children[i][0]
                    tails[i] = joints[child]
                if parents[i] is not None and parents[i] != -1 and len(children[parents[i]]) == 1:
                    connects[i] = True

    # Create root bone
    if add_root:
        bone_root = edit_bones.get('Bone')
        bone_root.name = 'Root'
        bone_root.tail = Vector((joints[0, 0], joints[0, 1], joints[0, 2]))
    else:
        bone_root = edit_bones.get('Bone')
        bone_root.name = names[0]
        bone_root.head = Vector((joints[0, 0], joints[0, 1], joints[0, 2]))
        bone_root.tail = Vector((joints[0, 0], joints[0, 1], joints[0, 2] + extrude_size))

    # Create bones
    def extrude_bone(edit_bones, name, parent_name, head, tail, connect):
        bone = edit_bones.new(name)
        bone.head = Vector((head[0], head[1], head[2]))
        bone.tail = Vector((tail[0], tail[1], tail[2]))
        bone.name = name
        parent_bone = edit_bones.get(parent_name)
        bone.parent = parent_bone
        bone.use_connect = connect
        assert not np.isnan(head).any(), f"nan found in head of bone {name}"
        assert not np.isnan(tail).any(), f"nan found in tail of bone {name}"

    for i in range(J):
        if add_root is False and i == 0:
            continue
        edit_bones = armature.edit_bones
        pname = 'Root' if parents[i] is None or parents[i] == -1 else names[parents[i]]
        extrude_bone(edit_bones, names[i], pname, joints[i], tails[i], connects[i])

    # Update bone positions
    for i in range(J):
        bone = edit_bones.get(names[i])
        bone.head = Vector((joints[i, 0], joints[i, 1], joints[i, 2]))
        bone.tail = Vector((tails[i, 0], tails[i, 1], tails[i, 2]))

    # === FIX BONE ORIENTATIONS FOR MIXAMO COMPATIBILITY ===
    # Mixamo animations expect specific bone Y-axis directions (bone direction = head→tail)
    # After Y-up conversion: +Y is "up", so spine/head bones should point +Y
    # Arms point outward along ±X, legs point down along -Y
    if is_mixamo_skeleton:
        print("[Blender FBX Export] Fixing bone orientations for Mixamo compatibility...")

        # Standard bone lengths for each bone type (used when repositioning tails)
        DEFAULT_BONE_LENGTH = 5.0  # In 100x scaled Mixamo space

        # Bone directions in Y-up space (after convert_to_yup transformation)
        # +Y = up, +X = left, -X = right, +Z = forward, -Z = back
        MIXAMO_BONE_DIRECTIONS = {
            # Spine chain - all point UP (+Y)
            'mixamorig:Hips': Vector((0, 1, 0)),
            'mixamorig:Spine': Vector((0, 1, 0)),
            'mixamorig:Spine1': Vector((0, 1, 0)),
            'mixamorig:Spine2': Vector((0, 1, 0)),
            'mixamorig:Neck': Vector((0, 1, 0)),
            'mixamorig:Head': Vector((0, 1, 0)),
            # Left leg - points DOWN (-Y)
            'mixamorig:LeftUpLeg': Vector((0, -1, 0)),
            'mixamorig:LeftLeg': Vector((0, -1, 0)),
            'mixamorig:LeftFoot': Vector((0, -0.632, 0.775)).normalized(),  # Points down-forward (matches Mixamo)
            'mixamorig:LeftToeBase': Vector((0, -0.632, 0.775)).normalized(),
            # Right leg - points DOWN (-Y)
            'mixamorig:RightUpLeg': Vector((0, -1, 0)),
            'mixamorig:RightLeg': Vector((0, -1, 0)),
            'mixamorig:RightFoot': Vector((0, -0.632, 0.775)).normalized(),  # Points down-forward (matches Mixamo)
            'mixamorig:RightToeBase': Vector((0, -0.632, 0.775)).normalized(),
            # Left arm - points LEFT (+X)
            'mixamorig:LeftShoulder': Vector((1, 0, 0)),
            'mixamorig:LeftArm': Vector((1, 0, 0)),
            'mixamorig:LeftForeArm': Vector((1, 0, 0)),
            'mixamorig:LeftHand': Vector((1, 0, 0)),
            # Right arm - points RIGHT (-X)
            'mixamorig:RightShoulder': Vector((-1, 0, 0)),
            'mixamorig:RightArm': Vector((-1, 0, 0)),
            'mixamorig:RightForeArm': Vector((-1, 0, 0)),
            'mixamorig:RightHand': Vector((-1, 0, 0)),
        }

        # Build a dict of bone name → joint index for quick lookup
        bone_to_idx = {name: i for i, name in enumerate(names)}

        orientation_fixes = 0
        for bone_name, target_direction in MIXAMO_BONE_DIRECTIONS.items():
            bone = edit_bones.get(bone_name)
            if not bone:
                continue

            # Get current bone direction
            current_direction = (bone.tail - bone.head).normalized()

            # Check if direction matches target (dot product close to 1)
            dot = current_direction.dot(target_direction)

            if dot < 0.95:  # More than ~18° off
                # Need to fix orientation
                bone_length = (bone.tail - bone.head).length
                if bone_length < 0.1:
                    bone_length = DEFAULT_BONE_LENGTH

                # Set new tail position in correct direction
                new_tail = bone.head + target_direction * bone_length
                bone.tail = new_tail

                angle_diff = math.degrees(math.acos(max(-1, min(1, dot))))
                print(f"[Blender FBX Export]   Fixed {bone_name}: {angle_diff:.1f}° correction")
                orientation_fixes += 1

        print(f"[Blender FBX Export] ✓ Fixed {orientation_fixes} bone orientations")

        # === FIX BONE ROLLS FOR MIXAMO COMPATIBILITY ===
        # Bone roll determines X and Z axes (rotation around Y-axis)
        # Mixamo expects specific roll values for correct animation
        print("[Blender FBX Export] Fixing bone rolls for Mixamo compatibility...")

        # Define target Z-axis for each bone (used with align_roll)
        # In Y-up space after conversion
        MIXAMO_BONE_ROLLS = {
            # Arms: Z-axis points DOWN (-Y in Y-up space)
            'mixamorig:LeftShoulder': Vector((0, -1, 0)),
            'mixamorig:LeftArm': Vector((0, -1, 0)),
            'mixamorig:LeftForeArm': Vector((0, -1, 0)),
            'mixamorig:LeftHand': Vector((0, -1, 0)),
            'mixamorig:RightShoulder': Vector((0, -1, 0)),
            'mixamorig:RightArm': Vector((0, -1, 0)),
            'mixamorig:RightForeArm': Vector((0, -1, 0)),
            'mixamorig:RightHand': Vector((0, -1, 0)),
            # Legs: Z-axis points FORWARD (+Z in Y-up space)
            'mixamorig:LeftUpLeg': Vector((0, 0, 1)),
            'mixamorig:LeftLeg': Vector((0, 0, 1)),
            'mixamorig:LeftFoot': Vector((0, 1, 0)),  # Foot Z points up
            'mixamorig:LeftToeBase': Vector((0, 1, 0)),
            'mixamorig:RightUpLeg': Vector((0, 0, 1)),
            'mixamorig:RightLeg': Vector((0, 0, 1)),
            'mixamorig:RightFoot': Vector((0, 1, 0)),
            'mixamorig:RightToeBase': Vector((0, 1, 0)),
            # Spine: Z-axis points BACK (+Z in Y-up space)
            'mixamorig:Hips': Vector((0, 0, 1)),
            'mixamorig:Spine': Vector((0, 0, 1)),
            'mixamorig:Spine1': Vector((0, 0, 1)),
            'mixamorig:Spine2': Vector((0, 0, 1)),
            'mixamorig:Neck': Vector((0, 0, 1)),
            'mixamorig:Head': Vector((0, 0, 1)),
        }

        roll_fixes = 0
        for bone_name, target_z in MIXAMO_BONE_ROLLS.items():
            bone = edit_bones.get(bone_name)
            if bone:
                bone.align_roll(target_z)
                roll_fixes += 1

        print(f"[Blender FBX Export] ✓ Fixed {roll_fixes} bone rolls")

    # Set bone roll for SMPL compatibility
    # SMPL motion expects local X = bone direction, but Blender has local Y = bone direction
    # We set roll so that the bone's local coordinate system matches what SMPL expects
    # after the basis transformation: SMPL_local = Basis^T @ Blender_local @ Basis
    # The goal is to set roll so that rotations can be applied with a consistent basis
    # Note: is_smpl_skeleton was already computed earlier for T-pose conversion

    if is_smpl_skeleton:
        print("[Blender FBX Export] Setting bone rolls for SMPL compatibility...")
        for i in range(J):
            bone = edit_bones.get(names[i])
            if bone:
                # Get bone direction
                direction = (bone.tail - bone.head).normalized()
                dx, dy, dz = direction.x, direction.y, direction.z

                # Set roll based on bone direction to get consistent local axes
                # For SMPL, we want the local coordinate system such that:
                # - Y axis = bone direction (Blender default)
                # - X and Z axes are consistent for the basis transformation

                # Use Blender's align_roll to set the Z-axis direction
                # Then the X axis is determined by cross product

                if abs(dx) > 0.9:  # Arms (bone along ±X)
                    # For arms, we want local Z to point up (world +Y in SMPL coords)
                    bone.align_roll(Vector((0, 1, 0)))  # Z up
                elif abs(dy) > 0.9:  # Spine/Legs (bone along ±Y)
                    # For vertical bones, we want local Z to point back (world +Z in SMPL coords)
                    bone.align_roll(Vector((0, 0, 1)))  # Z back
                elif abs(dz) > 0.9:  # Feet (bone along ±Z)
                    # For feet, we want local Z to point up (world +Y)
                    bone.align_roll(Vector((0, 1, 0)))  # Z up
                else:
                    # Default: use world up as reference
                    bone.align_roll(Vector((0, 1, 0)))

        print("[Blender FBX Export] ✓ Bone rolls set for SMPL compatibility")

    # Add skinning weights if vertices and skin provided
    if vertices is not None and skin is not None:
        print("[Blender FBX Export] Adding skinning weights...")
        # Must set to object mode to enable parent_set
        bpy.ops.object.mode_set(mode='OBJECT')
        objects = bpy.data.objects
        for o in bpy.context.selected_objects:
            o.select_set(False)
        ob = objects['character']
        arm = bpy.data.objects['Armature']
        ob.select_set(True)
        arm.select_set(True)
        bpy.ops.object.parent_set(type='ARMATURE_NAME')

        vis = []
        for x in ob.vertex_groups:
            vis.append(x.name)

        # Sparsify
        argsorted = np.argsort(-skin, axis=1)
        vertex_group_reweight = skin[np.arange(skin.shape[0])[..., None], argsorted]

        group_per_vertex = data.get('group_per_vertex', -1)
        if group_per_vertex == -1:
            group_per_vertex = vertex_group_reweight.shape[-1]

        do_not_normalize = data.get('do_not_normalize', False)
        if not do_not_normalize:
            vertex_group_reweight = vertex_group_reweight / vertex_group_reweight[..., :group_per_vertex].sum(axis=1)[..., None]

        for v, w in enumerate(skin):
            for ii in range(group_per_vertex):
                i = argsorted[v, ii]
                if i >= J:
                    continue
                n = names[i]
                if n not in vis:
                    continue
                ob.vertex_groups[n].add([v], vertex_group_reweight[v, ii], 'REPLACE')

    print("[Blender FBX Export] ✓ Armature created successfully")

    # Apply Mixamo-standard object transforms for animation compatibility
    # Official Mixamo FBX has: 90° X rotation, 0.01 scale on armature object only
    # The mesh inherits the transform through armature parenting
    if is_mixamo_skeleton:
        print("[Blender FBX Export] Applying Mixamo-standard object transforms...")
        bpy.ops.object.mode_set(mode='OBJECT')

        arm_obj = bpy.data.objects.get('Armature')

        if arm_obj:
            arm_obj.rotation_euler = (math.radians(90), 0, 0)
            arm_obj.scale = (0.01, 0.01, 0.01)
            print(f"[Blender FBX Export]   Armature: rotation=90° X, scale=0.01")

        bpy.context.view_layer.update()
        print("[Blender FBX Export] ✓ Mixamo object transforms applied")

except Exception as e:
    print(f"[Blender FBX Export] Armature creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Export to FBX
print("[Blender FBX Export] Exporting to FBX...")
os.makedirs(os.path.dirname(output_fbx) if os.path.dirname(output_fbx) else '.', exist_ok=True)

try:
    # Export with embedded textures
    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
        check_existing=False,
        add_leaf_bones=False,
        path_mode='COPY',  # Copy textures alongside FBX
        embed_textures=True,  # Embed textures in FBX file
    )
    print(f"[Blender FBX Export] Saved to: {output_fbx}")
    if texture_data_base64 and len(texture_data_base64) > 0:
        print(f"[Blender FBX Export] ✓ Textures embedded in FBX")
    print("[Blender FBX Export] Done!")
except Exception as e:
    print(f"[Blender FBX Export] Export failed: {e}")
    sys.exit(1)
