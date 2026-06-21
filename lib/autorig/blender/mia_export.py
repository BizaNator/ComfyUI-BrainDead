"""
Blender script to export MIA (Make-It-Animatable) rigged mesh to FBX.
Takes MIA inference output and a Mixamo template, creates rigged character FBX.

Usage: blender --background --python mia_export.py --
    --input_path <json> --output_path <fbx> --template_path <template_fbx>
    [--remove_fingers] [--reset_to_rest]
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
    else:
        i += 1

if not input_path or not output_path or not template_path:
    print("Usage: blender --background --python mia_export.py --")
    print("    --input_path <json> --output_path <fbx> --template_path <template_fbx>")
    print("    [--remove_fingers] [--reset_to_rest]")
    sys.exit(1)

print(f"[MIA Export] Input: {input_path}")
print(f"[MIA Export] Output: {output_path}")
print(f"[MIA Export] Template: {template_path}")
print(f"[MIA Export] Remove fingers: {remove_fingers}")
print(f"[MIA Export] Reset to rest: {reset_to_rest}")

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

    print(f"[MIA Export] Mesh path: {mesh_path}")
    print(f"[MIA Export] Skin weights shape: {skin_weights.shape}")
    print(f"[MIA Export] Joints: {len(joints)}")
    print(f"[MIA Export] Bones: {list(bones_idx_dict.keys())}")

except Exception as e:
    print(f"[MIA Export] Failed to load input data: {e}")
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
