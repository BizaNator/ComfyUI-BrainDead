"""
BrainDeadBlender Addon Nodes - ComfyUI nodes using BrainDeadBlender addon operators.

These nodes leverage the BrainDeadBlender Blender addon for advanced mesh operations:
- Decimation (Planar, Collapse, Full Pipeline)
- Remeshing (Sharp, Voxel, Quad)
- Cleanup (Smart cleanup, fill holes, remove internal, fix manifold)
- Edge Marking (from colors, from angle, sharp/crease conversion)
- Vertex Colors (solidify, smooth, transfer, convert domain)
- Normals (fix, verify)
"""

import os
import tempfile

import numpy as np
from comfy_api.latest import io

from ..mesh.types import TrimeshInput, TrimeshOutput, EdgeMetadataInput, EdgeMetadataOutput
from .base import BlenderNodeMixin, HAS_TRIMESH

if HAS_TRIMESH:
    import trimesh


# ============================================================================
# BLENDER SCRIPT TEMPLATE - ADDON SETUP
# ============================================================================
ADDON_SETUP_SCRIPT = '''
import bpy
import bmesh
import math
import sys
import os

def log(msg):
    """Print with immediate flush for real-time output."""
    print(msg)
    sys.stdout.flush()

def get_env_bool(name, default=True):
    val = os.environ.get(name, str(default))
    return val.lower() in ('true', '1', 'yes')

def get_env_float(name, default=0.0):
    return float(os.environ.get(name, str(default)))

def get_env_int(name, default=0):
    return int(os.environ.get(name, str(default)))

def get_env_str(name, default=''):
    return os.environ.get(name, default)

# Paths
INPUT_PATH = os.environ['BLENDER_INPUT_PATH']
OUTPUT_PATH = os.environ['BLENDER_OUTPUT_PATH']

# Enable BrainDeadBlender addon
log("[BD Addon] Enabling BrainDeadBlender addon...")

# Find and add addon path to sys.path
blender_dir = os.path.dirname(bpy.app.binary_path)
addon_paths = [
    os.path.join(blender_dir, '5.0', 'scripts', 'addons'),
    os.path.join(blender_dir, 'scripts', 'addons'),
]
for addon_path in addon_paths:
    if os.path.isdir(addon_path) and addon_path not in sys.path:
        sys.path.insert(0, addon_path)
        log(f"[BD Addon] Added addon path: {addon_path}")

# Also check for braindead_blender directly
for addon_path in addon_paths:
    bd_path = os.path.join(addon_path, 'braindead_blender')
    if os.path.isdir(bd_path):
        log(f"[BD Addon] Found braindead_blender at: {bd_path}")
        break
else:
    log("[BD Addon] WARNING: braindead_blender folder not found in addon paths")

try:
    bpy.ops.preferences.addon_enable(module='braindead_blender')
    log("[BD Addon] BrainDeadBlender addon enabled")
except Exception as e:
    log(f"[BD Addon] WARNING: Could not enable addon: {e}")
    log("[BD Addon] Continuing without addon - using fallback methods")

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh
ext = os.path.splitext(INPUT_PATH)[1].lower()
log(f"[BD Addon] Importing {ext} from {INPUT_PATH}")

if ext == '.ply':
    bpy.ops.wm.ply_import(filepath=INPUT_PATH)
elif ext == '.obj':
    bpy.ops.wm.obj_import(filepath=INPUT_PATH)
elif ext in ['.glb', '.gltf']:
    bpy.ops.import_scene.gltf(filepath=INPUT_PATH)
elif ext == '.stl':
    bpy.ops.wm.stl_import(filepath=INPUT_PATH)
else:
    raise ValueError(f"Unsupported format: {ext}")

# Get imported object
obj = bpy.context.active_object
if obj is None or obj.type != 'MESH':
    mesh_objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not mesh_objects:
        raise ValueError("No mesh objects found after import!")
    obj = mesh_objects[0]

bpy.context.view_layer.objects.active = obj
obj.select_set(True)
log(f"[BD Addon] Loaded mesh: {len(obj.data.vertices)} verts, {len(obj.data.polygons)} faces")

# Log color attributes (Blender 4.0+ API)
if hasattr(obj.data, 'color_attributes') and len(obj.data.color_attributes) > 0:
    log(f"[BD Addon] Color attributes found: {len(obj.data.color_attributes)}")
    for attr in obj.data.color_attributes:
        log(f"[BD Addon]   - {attr.name}: domain={attr.domain}, data_type={attr.data_type}")
else:
    log("[BD Addon] No color_attributes found on mesh")

# Also check legacy vertex_colors
if hasattr(obj.data, 'vertex_colors') and len(obj.data.vertex_colors) > 0:
    log(f"[BD Addon] Legacy vertex_colors found: {len(obj.data.vertex_colors)}")
    for vc in obj.data.vertex_colors:
        log(f"[BD Addon]   - {vc.name}")
else:
    log("[BD Addon] No legacy vertex_colors found")

def export_result():
    """Export the result mesh with vertex colors preserved."""
    ext_out = os.path.splitext(OUTPUT_PATH)[1].lower()

    # Log color attributes before export
    if hasattr(obj.data, 'color_attributes') and len(obj.data.color_attributes) > 0:
        log(f"[BD Addon] Pre-export color_attributes: {len(obj.data.color_attributes)}")
        for attr in obj.data.color_attributes:
            log(f"[BD Addon]   - {attr.name}: domain={attr.domain}, data_type={attr.data_type}")
    else:
        log("[BD Addon] WARNING: No color_attributes before export!")

    log(f"[BD Addon] Exporting to {ext_out}...")
    if ext_out == '.ply':
        bpy.ops.wm.ply_export(filepath=OUTPUT_PATH, export_colors='SRGB', ascii_format=True)
    elif ext_out == '.obj':
        bpy.ops.wm.obj_export(filepath=OUTPUT_PATH)
    elif ext_out in ['.glb', '.gltf']:
        # GLTF exporter parameters vary by Blender version
        export_kwargs = {
            'filepath': OUTPUT_PATH,
            'export_format': 'GLB',
            'export_attributes': True,
            'export_yup': True,
        }

        # Blender 5.0 uses export_vertex_color (not export_colors!)
        # Values: 'NONE', 'MATERIAL', 'ACTIVE', 'ALL' (Blender 5.0+)
        export_attempts = [
            # Blender 5.0+: export ALL vertex color layers
            {'export_vertex_color': 'ACTIVE'},
            # Try 'MATERIAL' which exports colors used by materials
            {'export_vertex_color': 'MATERIAL'},
            # No extra params (default behavior)
            {},
        ]

        exported = False
        for attempt_kwargs in export_attempts:
            try:
                bpy.ops.export_scene.gltf(**export_kwargs, **attempt_kwargs)
                log(f"[BD Addon] Exported GLB with params: {attempt_kwargs or 'defaults'}")
                exported = True
                break
            except TypeError as e:
                log(f"[BD Addon] Export attempt failed ({attempt_kwargs}): {e}")
                continue

        if not exported:
            raise RuntimeError("All GLB export attempts failed")
    elif ext_out == '.stl':
        bpy.ops.wm.stl_export(filepath=OUTPUT_PATH)
    log(f"[BD Addon] Saved to {OUTPUT_PATH}")
'''


# ============================================================================
# BD_BlenderDecimateV3 - Full Decimation Pipeline using Addon
# ============================================================================
DECIMATE_V3_SCRIPT = ADDON_SETUP_SCRIPT + '''
import numpy as np
from collections import deque

# Read settings
TARGET_FACES = get_env_int('BLENDER_ARG_TARGET_FACES', 5000)
PLANAR_ANGLE = get_env_float('BLENDER_ARG_PLANAR_ANGLE', 7.0)
SHARP_ANGLE = get_env_float('BLENDER_ARG_SHARP_ANGLE', 14.0)
PRESERVE_BOUNDARIES = get_env_bool('BLENDER_ARG_PRESERVE_BOUNDARIES', True)

# Planar grouping (structure-aware)
USE_PLANAR_GROUPING = get_env_bool('BLENDER_ARG_USE_PLANAR_GROUPING', False)
PLANAR_GROUP_ANGLE = get_env_float('BLENDER_ARG_PLANAR_GROUP_ANGLE', 15.0)
PLANAR_GROUP_MIN_SIZE = get_env_int('BLENDER_ARG_PLANAR_GROUP_MIN_SIZE', 10)

# Color edge detection
DETECT_COLOR_EDGES = get_env_bool('BLENDER_ARG_DETECT_COLOR_EDGES', True)
COLOR_EDGE_THRESHOLD = get_env_float('BLENDER_ARG_COLOR_EDGE_THRESHOLD', 0.15)
EDGE_MARK_TYPE = get_env_str('BLENDER_ARG_EDGE_MARK_TYPE', 'BOTH')

# Cleanup options
PRE_CLEANUP = get_env_bool('BLENDER_ARG_PRE_CLEANUP', True)
FILL_HOLES = get_env_bool('BLENDER_ARG_FILL_HOLES', True)
REMOVE_INTERNAL = get_env_bool('BLENDER_ARG_REMOVE_INTERNAL', False)
FIX_NORMALS = get_env_bool('BLENDER_ARG_FIX_NORMALS', True)

# Mode
USE_FULL_PIPELINE = get_env_bool('BLENDER_ARG_USE_FULL_PIPELINE', True)

# ============================================================================
# PLANAR GROUPING ALGORITHM (embedded for Blender subprocess)
# ============================================================================
def compute_face_normals_np(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-10)
    return normals / lengths

def quantize_pos(pos, precision=6):
    scale = 10 ** precision
    return (int(round(pos[0] * scale)), int(round(pos[1] * scale)), int(round(pos[2] * scale)))

def make_edge_key_pos(v1, v2, precision=6):
    q1 = quantize_pos(v1, precision)
    q2 = quantize_pos(v2, precision)
    return tuple(sorted([q1, q2]))

def build_face_adjacency_np(faces, vertices=None, precision=6):
    num_faces = len(faces)
    num_verts = int(faces.max()) + 1 if len(faces) > 0 else 0
    is_face_split = (vertices is not None and len(vertices) == num_faces * 3 and num_verts == num_faces * 3)

    edge_to_faces = {}
    if is_face_split and vertices is not None:
        log(f"[Planar Group] Face-split mesh detected, using position-based adjacency")
        for face_idx, face in enumerate(faces):
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            edges = [
                make_edge_key_pos(v0, v1, precision),
                make_edge_key_pos(v1, v2, precision),
                make_edge_key_pos(v2, v0, precision),
            ]
            for edge in edges:
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(face_idx)
    else:
        for face_idx, face in enumerate(faces):
            edges = [
                tuple(sorted([face[0], face[1]])),
                tuple(sorted([face[1], face[2]])),
                tuple(sorted([face[2], face[0]])),
            ]
            for edge in edges:
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(face_idx)

    adjacency = {i: [] for i in range(num_faces)}
    for edge, face_list in edge_to_faces.items():
        if len(face_list) == 2:
            adjacency[face_list[0]].append(face_list[1])
            adjacency[face_list[1]].append(face_list[0])
    return adjacency

def cluster_faces_np(faces, normals, adjacency, angle_threshold, min_group_size):
    num_faces = len(faces)
    group_labels = np.full(num_faces, -1, dtype=np.int32)
    current_group = 0
    for start_face in range(num_faces):
        if group_labels[start_face] >= 0:
            continue
        queue = deque([start_face])
        while queue:
            face_idx = queue.popleft()
            if group_labels[face_idx] >= 0:
                continue
            group_labels[face_idx] = current_group
            for neighbor_idx in adjacency[face_idx]:
                if group_labels[neighbor_idx] >= 0:
                    continue
                dot = np.clip(np.dot(normals[face_idx], normals[neighbor_idx]), -1.0, 1.0)
                angle = np.degrees(np.arccos(dot))
                if angle <= angle_threshold:
                    queue.append(neighbor_idx)
        current_group += 1
    # Renumber groups
    unique = np.unique(group_labels)
    remap = {old: new for new, old in enumerate(unique)}
    return np.array([remap[g] for g in group_labels], dtype=np.int32), len(unique)

def find_boundary_edges_np(faces, group_labels, vertices=None, precision=6):
    num_faces = len(faces)
    num_verts = int(faces.max()) + 1 if len(faces) > 0 else 0
    is_face_split = (vertices is not None and len(vertices) == num_faces * 3 and num_verts == num_faces * 3)

    edge_to_data = {}
    for face_idx, face in enumerate(faces):
        if is_face_split and vertices is not None:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            edges_data = [
                (make_edge_key_pos(v0, v1, precision), (face[0], face[1])),
                (make_edge_key_pos(v1, v2, precision), (face[1], face[2])),
                (make_edge_key_pos(v2, v0, precision), (face[2], face[0])),
            ]
        else:
            edges_data = [
                (tuple(sorted([face[0], face[1]])), (face[0], face[1])),
                (tuple(sorted([face[1], face[2]])), (face[1], face[2])),
                (tuple(sorted([face[2], face[0]])), (face[2], face[0])),
            ]
        for edge_key, vert_pair in edges_data:
            if edge_key not in edge_to_data:
                edge_to_data[edge_key] = []
            edge_to_data[edge_key].append((face_idx, vert_pair))

    boundary = []
    for edge_key, face_data in edge_to_data.items():
        if len(face_data) == 2:
            (f1, vp1), (f2, vp2) = face_data
            if group_labels[f1] != group_labels[f2]:
                boundary.append(vp1)
    return boundary

def apply_planar_grouping(obj, angle_threshold, min_group_size):
    """Apply planar grouping and mark boundary edges."""
    mesh = obj.data
    vertices = np.array([v.co[:] for v in mesh.vertices])
    faces = np.array([[v for v in p.vertices] for p in mesh.polygons])

    log(f"[Planar Group] Input: {len(faces)} faces, angle={angle_threshold}°")

    normals = compute_face_normals_np(vertices, faces)
    adjacency = build_face_adjacency_np(faces, vertices)
    group_labels, num_groups = cluster_faces_np(faces, normals, adjacency, angle_threshold, min_group_size)
    boundary_edges = find_boundary_edges_np(faces, group_labels, vertices)

    log(f"[Planar Group] Found {num_groups} groups, {len(boundary_edges)} boundary edges")

    # Mark boundary edges
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh)
    bm.edges.ensure_lookup_table()

    # Get crease layer
    crease_layer = bm.edges.layers.float.get('crease_edge')
    if crease_layer is None:
        crease_layer = bm.edges.layers.float.new('crease_edge')

    # Build edge lookup
    edge_lookup = {}
    for edge in bm.edges:
        key = tuple(sorted([edge.verts[0].index, edge.verts[1].index]))
        edge_lookup[key] = edge

    marked = 0
    for v1, v2 in boundary_edges:
        key = tuple(sorted([v1, v2]))
        if key in edge_lookup:
            edge = edge_lookup[key]
            edge.smooth = False  # Mark as sharp
            edge[crease_layer] = 1.0  # Mark with crease
            marked += 1

    bmesh.update_edit_mesh(mesh)
    bpy.ops.object.mode_set(mode='OBJECT')

    log(f"[Planar Group] Marked {marked} boundary edges as sharp/crease")
    return num_groups, marked

original_faces = len(obj.data.polygons)
log(f"[BD Decimate V3] Starting - Target: {TARGET_FACES} faces")
log(f"[BD Decimate V3] Settings: planar={PLANAR_ANGLE}deg, sharp={SHARP_ANGLE}deg")

# Configure addon settings
scene = bpy.context.scene

# Check if addon settings exist
has_addon = hasattr(scene, 'bd_decimate')

if has_addon:
    log("[BD Decimate V3] Using BrainDeadBlender addon operators")

    # Set decimate settings
    scene.bd_decimate.target_faces = TARGET_FACES
    scene.bd_decimate.planar_angle = PLANAR_ANGLE
    scene.bd_decimate.sharp_angle = SHARP_ANGLE
    scene.bd_decimate.preserve_boundaries = PRESERVE_BOUNDARIES

    # Set edge marking settings
    if hasattr(scene, 'bd_edges'):
        scene.bd_edges.edge_color_threshold = COLOR_EDGE_THRESHOLD
        scene.bd_edges.edge_mark_type = EDGE_MARK_TYPE

    # Set cleanup settings
    if hasattr(scene, 'bd_cleanup'):
        scene.bd_cleanup.fill_holes_max_sides = 100

    # Pre-cleanup
    if PRE_CLEANUP:
        log("[BD Decimate V3] Running pre-cleanup...")
        try:
            bpy.ops.braindead.merge_vertices()
            bpy.ops.braindead.fix_manifold()
        except:
            log("[BD Decimate V3] Pre-cleanup skipped (operator not found)")

    # Planar grouping (structure-aware edge marking)
    if USE_PLANAR_GROUPING:
        log(f"[BD Decimate V3] Running planar grouping (angle={PLANAR_GROUP_ANGLE}°, min_size={PLANAR_GROUP_MIN_SIZE})...")
        try:
            num_groups, edges_marked = apply_planar_grouping(obj, PLANAR_GROUP_ANGLE, PLANAR_GROUP_MIN_SIZE)
            log(f"[BD Decimate V3] Planar grouping: {num_groups} groups, {edges_marked} boundary edges marked")
        except Exception as e:
            log(f"[BD Decimate V3] Planar grouping failed: {e}")

    # Detect color edges
    if DETECT_COLOR_EDGES:
        log("[BD Decimate V3] Detecting color edges...")
        try:
            bpy.ops.braindead.mark_edges_from_colors()
        except:
            log("[BD Decimate V3] Color edge detection skipped (operator not found)")

    # Fill holes
    if FILL_HOLES:
        log("[BD Decimate V3] Filling holes...")
        try:
            bpy.ops.braindead.fill_holes()
        except:
            log("[BD Decimate V3] Fill holes skipped (operator not found)")

    # Remove internal geometry
    if REMOVE_INTERNAL:
        log("[BD Decimate V3] Removing internal geometry...")
        try:
            bpy.ops.braindead.remove_internal()
        except:
            log("[BD Decimate V3] Remove internal skipped (operator not found)")

    # Run full decimate pipeline or individual steps
    if USE_FULL_PIPELINE:
        log("[BD Decimate V3] Running full decimate pipeline...")
        try:
            bpy.ops.braindead.decimate_full()
        except Exception as e:
            log(f"[BD Decimate V3] Full pipeline failed: {e}, using individual steps")
            USE_FULL_PIPELINE = False

    if not USE_FULL_PIPELINE:
        # Planar decimate
        if PLANAR_ANGLE > 0:
            log("[BD Decimate V3] Running planar decimate...")
            try:
                bpy.ops.braindead.planar_decimate()
            except:
                pass

        # Collapse decimate
        log("[BD Decimate V3] Running collapse decimate...")
        try:
            bpy.ops.braindead.collapse_decimate()
        except:
            pass

        # Mark sharp edges
        if SHARP_ANGLE > 0:
            log("[BD Decimate V3] Marking sharp edges...")
            try:
                bpy.ops.braindead.mark_sharp_edges()
            except:
                pass

    # Fix normals
    if FIX_NORMALS:
        log("[BD Decimate V3] Fixing normals...")
        try:
            bpy.ops.braindead.fix_normals()
        except:
            pass

else:
    log("[BD Decimate V3] Addon not available, using Blender native operators")

    # Planar grouping (structure-aware edge marking) - uses embedded algorithm
    if USE_PLANAR_GROUPING:
        log(f"[BD Decimate V3] Running planar grouping (angle={PLANAR_GROUP_ANGLE}°, min_size={PLANAR_GROUP_MIN_SIZE})...")
        try:
            num_groups, edges_marked = apply_planar_grouping(obj, PLANAR_GROUP_ANGLE, PLANAR_GROUP_MIN_SIZE)
            log(f"[BD Decimate V3] Planar grouping: {num_groups} groups, {edges_marked} boundary edges marked")
        except Exception as e:
            log(f"[BD Decimate V3] Planar grouping failed: {e}")

    # Fallback to native Blender decimate
    if PLANAR_ANGLE > 0:
        log("[BD Decimate V3] Planar decimate (native)...")
        mod = obj.modifiers.new(name="Planar", type='DECIMATE')
        mod.decimate_type = 'DISSOLVE'
        mod.angle_limit = math.radians(PLANAR_ANGLE)
        mod.use_dissolve_boundaries = not PRESERVE_BOUNDARIES
        bpy.ops.object.modifier_apply(modifier=mod.name)

    current = len(obj.data.polygons)
    if current > TARGET_FACES:
        log(f"[BD Decimate V3] Collapse decimate (native): {current} -> {TARGET_FACES}...")
        ratio = TARGET_FACES / current
        mod = obj.modifiers.new(name="Collapse", type='DECIMATE')
        mod.decimate_type = 'COLLAPSE'
        mod.ratio = ratio
        bpy.ops.object.modifier_apply(modifier=mod.name)

    # Fix normals
    if FIX_NORMALS:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')

final_faces = len(obj.data.polygons)
reduction = (1 - final_faces / original_faces) * 100 if original_faces > 0 else 0
log(f"[BD Decimate V3] Complete: {original_faces} -> {final_faces} faces ({reduction:.1f}% reduction)")

# Convert Z-up to Y-up for PLY export (GLTF import converts Y-up to Z-up)
log("[BD Decimate V3] Converting Z-up to Y-up for output...")

# Ensure we're in object mode and object is properly selected
try:
    bpy.ops.object.mode_set(mode='OBJECT')
except:
    pass

bpy.ops.object.select_all(action='DESELECT')
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

# Rotate -90 degrees around X axis (Z-up → Y-up)
try:
    bpy.ops.transform.rotate(value=math.radians(-90), orient_axis='X', orient_type='GLOBAL')
    # Apply the rotation to mesh data
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    log("[BD Decimate V3] Applied -90° X rotation (Z-up → Y-up)")
except Exception as e:
    log(f"[BD Decimate V3] Warning: Transform apply failed: {e}")
    # Fallback: apply rotation directly to mesh data
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    rot_matrix = mathutils.Matrix.Rotation(math.radians(-90), 4, 'X')
    bmesh.ops.transform(bm, matrix=rot_matrix, verts=bm.verts)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    log("[BD Decimate V3] Applied rotation via bmesh fallback")

export_result()
'''


class BD_BlenderDecimateV3(BlenderNodeMixin, io.ComfyNode):
    """
    Full-featured stylized decimation using BrainDeadBlender addon.

    Uses addon operators for:
    - Color edge detection and marking
    - Planar decimation with edge preservation
    - Collapse decimation
    - Sharp edge marking
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_BlenderDecimateV3",
            display_name="BD Blender Decimate V3",
            category="🧠BrainDead/Blender",
            description="""Full stylized decimation using BrainDeadBlender addon.

Features:
- Planar grouping (structure-aware boundary detection from normals)
- Color edge detection (marks boundaries as sharp/seam)
- Planar decimation (merges coplanar faces)
- Collapse decimation (reaches target face count)
- Sharp edge marking by angle
- Automatic cleanup and normal fixing

Use planar grouping to preserve flat-shaded planes during decimation.""",
            inputs=[
                TrimeshInput("mesh"),
                io.Int.Input(
                    "target_faces",
                    default=5000,
                    min=100,
                    max=500000,
                    tooltip="Target face count",
                ),
                io.Float.Input(
                    "planar_angle",
                    default=7.0,
                    min=0.0,
                    max=45.0,
                    step=0.5,
                    tooltip="Planar decimation angle threshold (degrees)",
                ),
                io.Float.Input(
                    "sharp_angle",
                    default=14.0,
                    min=0.0,
                    max=90.0,
                    step=1.0,
                    tooltip="Sharp edge marking angle threshold (degrees)",
                ),
                io.Boolean.Input(
                    "detect_color_edges",
                    default=True,
                    tooltip="Detect and preserve color boundaries",
                ),
                io.Float.Input(
                    "color_edge_threshold",
                    default=0.15,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    tooltip="Color difference threshold for edge detection",
                ),
                io.Combo.Input(
                    "edge_mark_type",
                    options=["BOTH", "SHARP", "CREASE"],
                    default="BOTH",
                    tooltip="How to mark detected edges",
                ),
                io.Boolean.Input(
                    "preserve_boundaries",
                    default=True,
                    tooltip="Preserve mesh boundary edges",
                ),
                io.Boolean.Input(
                    "pre_cleanup",
                    default=True,
                    tooltip="Run cleanup before decimation",
                ),
                io.Boolean.Input(
                    "fill_holes",
                    default=True,
                    tooltip="Fill holes before decimation",
                ),
                io.Boolean.Input(
                    "remove_internal",
                    default=False,
                    tooltip="Remove internal/hidden geometry",
                ),
                io.Boolean.Input(
                    "fix_normals",
                    default=True,
                    tooltip="Fix normals after decimation",
                ),
                io.Boolean.Input(
                    "use_full_pipeline",
                    default=True,
                    tooltip="Use addon's full pipeline operator (recommended)",
                ),
                io.Boolean.Input(
                    "use_planar_grouping",
                    default=False,
                    tooltip="Run planar grouping BEFORE decimation to mark structure-aware boundaries",
                ),
                io.Float.Input(
                    "planar_group_angle",
                    default=15.0,
                    min=1.0,
                    max=90.0,
                    step=1.0,
                    tooltip="Max angle between face normals in same planar group (lower = more groups)",
                ),
                io.Int.Input(
                    "planar_group_min_size",
                    default=10,
                    min=1,
                    max=1000,
                    step=1,
                    tooltip="Minimum faces per planar group (smaller groups merged into neighbors)",
                ),
                io.Int.Input(
                    "timeout",
                    default=600,
                    min=60,
                    max=3600,
                    optional=True,
                    tooltip="Maximum processing time in seconds",
                ),
            ],
            outputs=[
                TrimeshOutput(display_name="mesh"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        target_faces: int = 5000,
        planar_angle: float = 7.0,
        sharp_angle: float = 14.0,
        detect_color_edges: bool = True,
        color_edge_threshold: float = 0.15,
        edge_mark_type: str = "BOTH",
        preserve_boundaries: bool = True,
        pre_cleanup: bool = True,
        fill_holes: bool = True,
        remove_internal: bool = False,
        fix_normals: bool = True,
        use_full_pipeline: bool = True,
        use_planar_grouping: bool = False,
        planar_group_angle: float = 15.0,
        planar_group_min_size: int = 10,
        timeout: int = 600,
    ) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "ERROR: trimesh not installed")

        available, msg = cls._check_blender()
        if not available:
            return io.NodeOutput(mesh, f"ERROR: {msg}")

        if mesh is None:
            return io.NodeOutput(None, "ERROR: No input mesh")

        orig_faces = len(mesh.faces) if hasattr(mesh, 'faces') else 0

        input_path = None
        output_path = None
        try:
            input_path = cls._mesh_to_temp_file(mesh, suffix='.glb')
            fd, output_path = tempfile.mkstemp(suffix='.ply')
            os.close(fd)

            extra_args = {
                'target_faces': target_faces,
                'planar_angle': planar_angle,
                'sharp_angle': sharp_angle,
                'detect_color_edges': detect_color_edges,
                'color_edge_threshold': color_edge_threshold,
                'edge_mark_type': edge_mark_type,
                'preserve_boundaries': preserve_boundaries,
                'pre_cleanup': pre_cleanup,
                'fill_holes': fill_holes,
                'remove_internal': remove_internal,
                'fix_normals': fix_normals,
                'use_full_pipeline': use_full_pipeline,
                'use_planar_grouping': use_planar_grouping,
                'planar_group_angle': planar_group_angle,
                'planar_group_min_size': planar_group_min_size,
            }

            success, message, log_lines = cls._run_blender_script(
                DECIMATE_V3_SCRIPT,
                input_path,
                output_path,
                extra_args=extra_args,
                timeout=timeout,
            )

            if not success:
                return io.NodeOutput(mesh, f"ERROR: {message}")

            result_mesh = cls._load_mesh_from_file(output_path)
            new_faces = len(result_mesh.faces)
            reduction = (1 - new_faces / orig_faces) * 100 if orig_faces > 0 else 0

            status = f"V3: {orig_faces:,} -> {new_faces:,} faces ({reduction:.1f}% reduction)"
            return io.NodeOutput(result_mesh, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return io.NodeOutput(mesh, f"ERROR: {e}")

        finally:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if output_path and os.path.exists(output_path):
                os.remove(output_path)


# ============================================================================
# BD_BlenderRemesh - Remeshing using Addon
# ============================================================================
REMESH_ADDON_SCRIPT = ADDON_SETUP_SCRIPT + '''
# Read settings
MODE = get_env_str('BLENDER_ARG_MODE', 'VOXEL')
OCTREE_DEPTH = get_env_int('BLENDER_ARG_OCTREE_DEPTH', 8)
SHARPNESS = get_env_float('BLENDER_ARG_SHARPNESS', 1.0)
VOXEL_SIZE = get_env_float('BLENDER_ARG_VOXEL_SIZE', 0.0)
TARGET_POLYS = get_env_int('BLENDER_ARG_TARGET_POLYS', 100000)

original_faces = len(obj.data.polygons)
log(f"[BD Remesh] Starting - Mode: {MODE}")

# Check if addon is available
has_addon = hasattr(bpy.context.scene, 'bd_remesh')

if has_addon:
    log("[BD Remesh] Using BrainDeadBlender addon")
    scene = bpy.context.scene
    scene.bd_remesh.mode = MODE
    scene.bd_remesh.octree_depth = OCTREE_DEPTH
    scene.bd_remesh.sharpness = SHARPNESS
    scene.bd_remesh.voxel_size = VOXEL_SIZE
    scene.bd_remesh.target_polys = TARGET_POLYS

    try:
        bpy.ops.braindead.remesh()
    except Exception as e:
        log(f"[BD Remesh] Addon operator failed: {e}, using native")
        has_addon = False

if not has_addon:
    log("[BD Remesh] Using native Blender remesh")
    mod = obj.modifiers.new(name="Remesh", type='REMESH')

    if MODE == 'VOXEL' or MODE == 'VOXEL_HIGH':
        mod.mode = 'VOXEL'
        if VOXEL_SIZE > 0:
            mod.voxel_size = VOXEL_SIZE
        else:
            # Auto-calculate from target polys
            bounds = obj.dimensions
            volume = bounds.x * bounds.y * bounds.z
            mod.voxel_size = (volume / TARGET_POLYS) ** (1/3) * 2
    elif MODE == 'SHARP':
        mod.mode = 'SHARP'
        mod.octree_depth = OCTREE_DEPTH
        mod.sharpness = SHARPNESS
    elif MODE == 'QUAD':
        mod.mode = 'QUAD'
        mod.octree_depth = OCTREE_DEPTH
    else:
        mod.mode = 'BLOCKS'
        mod.octree_depth = OCTREE_DEPTH

    bpy.ops.object.modifier_apply(modifier=mod.name)

final_faces = len(obj.data.polygons)
log(f"[BD Remesh] Complete: {original_faces} -> {final_faces} faces")

# Convert Z-up to Y-up for PLY export (GLTF import converts Y-up to Z-up)
log("[BD Remesh] Converting Z-up to Y-up for output...")
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.transform.rotate(value=math.radians(-90), orient_axis='X', orient_type='GLOBAL')
bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

export_result()
'''


class BD_BlenderRemesh(BlenderNodeMixin, io.ComfyNode):
    """
    Remesh using BrainDeadBlender addon or Blender native.

    Modes:
    - SHARP: Edge-preserving remesh (good for hard surface)
    - VOXEL: Voxel-based remesh (good for organic)
    - VOXEL_HIGH: High-quality voxel remesh
    - QUAD: Quadrilateral remesh
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_BlenderRemesh",
            display_name="BD Blender Remesh",
            category="🧠BrainDead/Blender",
            description="""Remesh using BrainDeadBlender addon.

Modes:
- SHARP: Edge-preserving (hard surface)
- VOXEL: Voxel-based (organic shapes)
- VOXEL_HIGH: High-quality voxel
- QUAD: Quadrilateral output""",
            inputs=[
                TrimeshInput("mesh"),
                io.Combo.Input(
                    "mode",
                    options=["VOXEL", "VOXEL_HIGH", "SHARP", "QUAD"],
                    default="VOXEL",
                    tooltip="Remesh algorithm",
                ),
                io.Int.Input(
                    "octree_depth",
                    default=8,
                    min=4,
                    max=12,
                    tooltip="Detail level for SHARP/QUAD modes (higher = more detail)",
                ),
                io.Float.Input(
                    "sharpness",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    tooltip="Edge sharpness for SHARP mode",
                ),
                io.Float.Input(
                    "voxel_size",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip="Voxel size (0 = auto from target_polys)",
                ),
                io.Int.Input(
                    "target_polys",
                    default=100000,
                    min=1000,
                    max=10000000,
                    tooltip="Target polygon count (for auto voxel size)",
                ),
                io.Int.Input(
                    "timeout",
                    default=300,
                    min=60,
                    max=3600,
                    optional=True,
                    tooltip="Maximum processing time",
                ),
            ],
            outputs=[
                TrimeshOutput(display_name="mesh"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        mode: str = "VOXEL",
        octree_depth: int = 8,
        sharpness: float = 1.0,
        voxel_size: float = 0.0,
        target_polys: int = 100000,
        timeout: int = 300,
    ) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "ERROR: trimesh not installed")

        available, msg = cls._check_blender()
        if not available:
            return io.NodeOutput(mesh, f"ERROR: {msg}")

        if mesh is None:
            return io.NodeOutput(None, "ERROR: No input mesh")

        orig_faces = len(mesh.faces) if hasattr(mesh, 'faces') else 0

        input_path = None
        output_path = None
        try:
            input_path = cls._mesh_to_temp_file(mesh, suffix='.glb')
            fd, output_path = tempfile.mkstemp(suffix='.ply')
            os.close(fd)

            extra_args = {
                'mode': mode,
                'octree_depth': octree_depth,
                'sharpness': sharpness,
                'voxel_size': voxel_size,
                'target_polys': target_polys,
            }

            success, message, log_lines = cls._run_blender_script(
                REMESH_ADDON_SCRIPT,
                input_path,
                output_path,
                extra_args=extra_args,
                timeout=timeout,
            )

            if not success:
                return io.NodeOutput(mesh, f"ERROR: {message}")

            result_mesh = cls._load_mesh_from_file(output_path)
            new_faces = len(result_mesh.faces)

            status = f"Remeshed ({mode}): {orig_faces:,} -> {new_faces:,} faces"
            return io.NodeOutput(result_mesh, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return io.NodeOutput(mesh, f"ERROR: {e}")

        finally:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if output_path and os.path.exists(output_path):
                os.remove(output_path)


# ============================================================================
# BD_BlenderCleanup - Smart Mesh Cleanup using Addon
# ============================================================================
CLEANUP_ADDON_SCRIPT = ADDON_SETUP_SCRIPT + '''
# Read settings
REMOVE_LOOSE = get_env_bool('BLENDER_ARG_REMOVE_LOOSE', True)
REMOVE_ISLANDS = get_env_bool('BLENDER_ARG_REMOVE_ISLANDS', True)
ISLAND_THRESHOLD = get_env_int('BLENDER_ARG_ISLAND_THRESHOLD', 10)
MERGE_VERTICES = get_env_bool('BLENDER_ARG_MERGE_VERTICES', True)
MERGE_DISTANCE = get_env_float('BLENDER_ARG_MERGE_DISTANCE', 0.0001)
FILL_HOLES = get_env_bool('BLENDER_ARG_FILL_HOLES', True)
FILL_HOLES_MAX = get_env_int('BLENDER_ARG_FILL_HOLES_MAX', 100)
FIX_NORMALS = get_env_bool('BLENDER_ARG_FIX_NORMALS', True)
FIX_MANIFOLD = get_env_bool('BLENDER_ARG_FIX_MANIFOLD', True)
TRIANGULATE_NGONS = get_env_bool('BLENDER_ARG_TRIANGULATE_NGONS', False)
DISSOLVE_EMBEDDED = get_env_bool('BLENDER_ARG_DISSOLVE_EMBEDDED', False)
USE_SMART_CLEANUP = get_env_bool('BLENDER_ARG_USE_SMART_CLEANUP', True)

original_verts = len(obj.data.vertices)
original_faces = len(obj.data.polygons)
log(f"[BD Cleanup] Starting - {original_verts} verts, {original_faces} faces")

# Check if addon is available
has_addon = hasattr(bpy.context.scene, 'bd_cleanup')

if has_addon and USE_SMART_CLEANUP:
    log("[BD Cleanup] Using BrainDeadBlender smart cleanup")
    scene = bpy.context.scene

    # Configure smart cleanup
    scene.bd_cleanup.smart_remove_loose = REMOVE_LOOSE
    scene.bd_cleanup.smart_remove_islands = REMOVE_ISLANDS
    scene.bd_cleanup.island_threshold = ISLAND_THRESHOLD
    scene.bd_cleanup.smart_merge_vertices = MERGE_VERTICES
    scene.bd_cleanup.merge_distance = MERGE_DISTANCE
    scene.bd_cleanup.smart_fill_holes = FILL_HOLES
    scene.bd_cleanup.fill_holes_max_sides = FILL_HOLES_MAX
    scene.bd_cleanup.smart_fix_normals = FIX_NORMALS
    scene.bd_cleanup.smart_triangulate_ngons = TRIANGULATE_NGONS
    scene.bd_cleanup.smart_dissolve_embedded = DISSOLVE_EMBEDDED

    try:
        bpy.ops.braindead.smart_cleanup()
    except Exception as e:
        log(f"[BD Cleanup] Smart cleanup failed: {e}, using individual ops")
        USE_SMART_CLEANUP = False

if not has_addon or not USE_SMART_CLEANUP:
    log("[BD Cleanup] Using individual cleanup operations")

    # Enter edit mode for cleanup
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    # Merge vertices
    if MERGE_VERTICES:
        log("[BD Cleanup] Merging vertices...")
        bpy.ops.mesh.remove_doubles(threshold=MERGE_DISTANCE)

    # Remove loose geometry
    if REMOVE_LOOSE:
        log("[BD Cleanup] Removing loose geometry...")
        bpy.ops.mesh.select_all(action='DESELECT')
        try:
            bpy.ops.mesh.select_loose()
            bpy.ops.mesh.delete(type='VERT')
        except:
            pass

    # Fill holes
    if FILL_HOLES:
        log("[BD Cleanup] Filling holes...")
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_non_manifold(extend=False, use_wire=False, use_boundary=True,
                                         use_multi_face=False, use_non_contiguous=False)
        try:
            bpy.ops.mesh.fill_holes(sides=FILL_HOLES_MAX)
        except:
            pass

    # Fix normals
    if FIX_NORMALS:
        log("[BD Cleanup] Fixing normals...")
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)

    # Triangulate n-gons
    if TRIANGULATE_NGONS:
        log("[BD Cleanup] Triangulating n-gons...")
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')

    bpy.ops.object.mode_set(mode='OBJECT')

# Fix manifold issues
if FIX_MANIFOLD:
    log("[BD Cleanup] Fixing manifold issues...")
    if has_addon:
        try:
            bpy.ops.braindead.fix_manifold()
        except:
            pass
    else:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        try:
            bpy.ops.mesh.dissolve_degenerate(threshold=0.0001)
        except:
            pass
        bpy.ops.object.mode_set(mode='OBJECT')

final_verts = len(obj.data.vertices)
final_faces = len(obj.data.polygons)
log(f"[BD Cleanup] Complete: {original_verts} -> {final_verts} verts, {original_faces} -> {final_faces} faces")

# Convert Z-up to Y-up for PLY export (GLTF import converts Y-up to Z-up)
log("[BD Cleanup] Converting Z-up to Y-up for output...")
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.transform.rotate(value=math.radians(-90), orient_axis='X', orient_type='GLOBAL')
bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

export_result()
'''


class BD_BlenderCleanup(BlenderNodeMixin, io.ComfyNode):
    """
    Smart mesh cleanup using BrainDeadBlender addon.

    Operations:
    - Remove loose geometry
    - Remove small islands
    - Merge vertices (remove doubles)
    - Fill holes
    - Fix normals
    - Fix manifold issues
    - Triangulate n-gons
    - Dissolve embedded faces
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_BlenderCleanup",
            display_name="BD Blender Cleanup",
            category="🧠BrainDead/Blender",
            description="""Smart mesh cleanup using BrainDeadBlender addon.

Configurable cleanup operations:
- Remove loose vertices/edges/faces
- Remove small disconnected islands
- Merge close vertices
- Fill holes
- Fix normals
- Fix manifold geometry
- Triangulate n-gons""",
            inputs=[
                TrimeshInput("mesh"),
                io.Boolean.Input(
                    "remove_loose",
                    default=True,
                    tooltip="Remove loose vertices, edges, faces",
                ),
                io.Boolean.Input(
                    "remove_islands",
                    default=True,
                    tooltip="Remove small disconnected mesh islands",
                ),
                io.Int.Input(
                    "island_threshold",
                    default=10,
                    min=1,
                    max=1000,
                    tooltip="Minimum faces to keep an island",
                ),
                io.Boolean.Input(
                    "merge_vertices",
                    default=True,
                    tooltip="Merge close vertices (remove doubles)",
                ),
                io.Float.Input(
                    "merge_distance",
                    default=0.0001,
                    min=0.0,
                    max=0.1,
                    step=0.0001,
                    tooltip="Distance threshold for merging vertices",
                ),
                io.Boolean.Input(
                    "fill_holes",
                    default=True,
                    tooltip="Fill holes (open boundaries)",
                ),
                io.Int.Input(
                    "fill_holes_max",
                    default=100,
                    min=3,
                    max=1000,
                    tooltip="Maximum sides for hole filling",
                ),
                io.Boolean.Input(
                    "fix_normals",
                    default=True,
                    tooltip="Fix face normals to point outward",
                ),
                io.Boolean.Input(
                    "fix_manifold",
                    default=True,
                    tooltip="Fix non-manifold geometry",
                ),
                io.Boolean.Input(
                    "triangulate_ngons",
                    default=False,
                    tooltip="Triangulate faces with >4 vertices",
                ),
                io.Boolean.Input(
                    "dissolve_embedded",
                    default=False,
                    tooltip="Dissolve small faces embedded in larger regions",
                ),
                io.Boolean.Input(
                    "use_smart_cleanup",
                    default=True,
                    tooltip="Use addon's smart cleanup (recommended)",
                ),
                io.Int.Input(
                    "timeout",
                    default=300,
                    min=60,
                    max=3600,
                    optional=True,
                    tooltip="Maximum processing time",
                ),
            ],
            outputs=[
                TrimeshOutput(display_name="mesh"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        remove_loose: bool = True,
        remove_islands: bool = True,
        island_threshold: int = 10,
        merge_vertices: bool = True,
        merge_distance: float = 0.0001,
        fill_holes: bool = True,
        fill_holes_max: int = 100,
        fix_normals: bool = True,
        fix_manifold: bool = True,
        triangulate_ngons: bool = False,
        dissolve_embedded: bool = False,
        use_smart_cleanup: bool = True,
        timeout: int = 300,
    ) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "ERROR: trimesh not installed")

        available, msg = cls._check_blender()
        if not available:
            return io.NodeOutput(mesh, f"ERROR: {msg}")

        if mesh is None:
            return io.NodeOutput(None, "ERROR: No input mesh")

        orig_verts = len(mesh.vertices) if hasattr(mesh, 'vertices') else 0
        orig_faces = len(mesh.faces) if hasattr(mesh, 'faces') else 0

        input_path = None
        output_path = None
        try:
            input_path = cls._mesh_to_temp_file(mesh, suffix='.glb')
            fd, output_path = tempfile.mkstemp(suffix='.ply')
            os.close(fd)

            extra_args = {
                'remove_loose': remove_loose,
                'remove_islands': remove_islands,
                'island_threshold': island_threshold,
                'merge_vertices': merge_vertices,
                'merge_distance': merge_distance,
                'fill_holes': fill_holes,
                'fill_holes_max': fill_holes_max,
                'fix_normals': fix_normals,
                'fix_manifold': fix_manifold,
                'triangulate_ngons': triangulate_ngons,
                'dissolve_embedded': dissolve_embedded,
                'use_smart_cleanup': use_smart_cleanup,
            }

            success, message, log_lines = cls._run_blender_script(
                CLEANUP_ADDON_SCRIPT,
                input_path,
                output_path,
                extra_args=extra_args,
                timeout=timeout,
            )

            if not success:
                return io.NodeOutput(mesh, f"ERROR: {message}")

            result_mesh = cls._load_mesh_from_file(output_path)
            new_verts = len(result_mesh.vertices)
            new_faces = len(result_mesh.faces)

            status = f"Cleanup: {orig_verts:,} -> {new_verts:,} verts, {orig_faces:,} -> {new_faces:,} faces"
            return io.NodeOutput(result_mesh, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return io.NodeOutput(mesh, f"ERROR: {e}")

        finally:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if output_path and os.path.exists(output_path):
                os.remove(output_path)


# ============================================================================
# BD_BlenderEdgeMarking - Edge Marking using Addon
# ============================================================================
EDGE_MARKING_SCRIPT = ADDON_SETUP_SCRIPT + '''
# Read settings
OPERATION = get_env_str('BLENDER_ARG_OPERATION', 'FROM_COLORS')
COLOR_THRESHOLD = get_env_float('BLENDER_ARG_COLOR_THRESHOLD', 0.15)
ANGLE_THRESHOLD = get_env_float('BLENDER_ARG_ANGLE_THRESHOLD', 30.0)
MARK_MODE = get_env_str('BLENDER_ARG_MARK_MODE', 'ADD')
MARK_TYPE = get_env_str('BLENDER_ARG_MARK_TYPE', 'BOTH')
CREASE_VALUE = get_env_float('BLENDER_ARG_CREASE_VALUE', 1.0)
CLEAR_SHARP = get_env_bool('BLENDER_ARG_CLEAR_SHARP', False)
CLEAR_CREASE = get_env_bool('BLENDER_ARG_CLEAR_CREASE', False)

log(f"[BD Edge Marking] Operation: {OPERATION}")
log(f"[BD Edge Marking] Color threshold: {COLOR_THRESHOLD}, Angle threshold: {ANGLE_THRESHOLD}deg")

import bmesh
from mathutils import Vector

# Statistics tracking
stats = {
    'total_edges': 0,
    'boundary_edges': 0,
    'color_edges_found': 0,
    'color_edges_marked': 0,
    'angle_edges_found': 0,
    'angle_edges_marked': 0,
    'sharp_before': 0,
    'sharp_after': 0,
    'crease_before': 0,
    'crease_after': 0,
}

def get_crease_layer(bm):
    """Get or create crease layer (Blender 4.0+ API)."""
    # Blender 4.0+ moved crease from dedicated layer to float layer
    crease_layer = bm.edges.layers.float.get('crease_edge')
    if crease_layer is None:
        crease_layer = bm.edges.layers.float.new('crease_edge')
    return crease_layer

def count_edge_marks(bm):
    """Count current sharp and crease edges."""
    sharp = 0
    crease = 0
    crease_layer = get_crease_layer(bm)
    for edge in bm.edges:
        if not edge.smooth:
            sharp += 1
        if edge[crease_layer] > 0.01:
            crease += 1
    return sharp, crease

def analyze_color_edges(bm, threshold):
    """Analyze edges where vertex colors change significantly."""
    # Blender 4.0+ uses Color Attributes (float_color layers)
    # Try multiple approaches to find color data
    color_layer = None
    color_domain = 'LOOP'  # LOOP = per-corner, VERT = per-vertex

    # Method 1: Try bmesh loop color layers (old API)
    if bm.loops.layers.color.active:
        color_layer = bm.loops.layers.color.active
        color_domain = 'LOOP'
        log(f"[BD Edge Marking] Found loop color layer: {color_layer.name}")
    elif len(bm.loops.layers.color) > 0:
        color_layer = list(bm.loops.layers.color.values())[0]
        color_domain = 'LOOP'
        log(f"[BD Edge Marking] Found loop color layer (first): {color_layer.name}")

    # Method 2: Try bmesh float_color layers (Blender 4.0+ Color Attributes)
    if not color_layer:
        if hasattr(bm.loops.layers, 'float_color'):
            if bm.loops.layers.float_color.active:
                color_layer = bm.loops.layers.float_color.active
                color_domain = 'LOOP'
                log(f"[BD Edge Marking] Found loop float_color layer: {color_layer.name}")
            elif len(bm.loops.layers.float_color) > 0:
                color_layer = list(bm.loops.layers.float_color.values())[0]
                color_domain = 'LOOP'
                log(f"[BD Edge Marking] Found loop float_color layer (first): {color_layer.name}")

    # Method 3: Try vertex float_color layers
    if not color_layer:
        if hasattr(bm.verts.layers, 'float_color'):
            if bm.verts.layers.float_color.active:
                color_layer = bm.verts.layers.float_color.active
                color_domain = 'VERT'
                log(f"[BD Edge Marking] Found vert float_color layer: {color_layer.name}")
            elif len(bm.verts.layers.float_color) > 0:
                color_layer = list(bm.verts.layers.float_color.values())[0]
                color_domain = 'VERT'
                log(f"[BD Edge Marking] Found vert float_color layer (first): {color_layer.name}")

    if not color_layer:
        # List what layers ARE available for debugging
        log("[BD Edge Marking] No vertex color layer found!")
        # Blender 5.0 BMLayerAccess doesn't have .keys() - list available layer types
        log(f"[BD Edge Marking] Loop layers: color={len(bm.loops.layers.color)}")
        if hasattr(bm.loops.layers, 'float_color'):
            log(f"[BD Edge Marking] Loop float_color: {len(bm.loops.layers.float_color)}")
        if hasattr(bm.verts.layers, 'float_color'):
            log(f"[BD Edge Marking] Vert float_color: {len(bm.verts.layers.float_color)}")
        return 0, []

    # Calculate face colors based on domain
    face_colors = {}
    for face in bm.faces:
        color_sum = Vector((0, 0, 0))
        count = 0
        if color_domain == 'LOOP':
            for loop in face.loops:
                col = loop[color_layer]
                color_sum += Vector((col[0], col[1], col[2]))
                count += 1
        else:  # VERT domain
            for vert in face.verts:
                col = vert[color_layer]
                color_sum += Vector((col[0], col[1], col[2]))
                count += 1
        if count > 0:
            face_colors[face.index] = color_sum / count

    log(f"[BD Edge Marking] Calculated colors for {len(face_colors)} faces")

    # Find edges with color difference
    color_edges = []
    for edge in bm.edges:
        if len(edge.link_faces) != 2:
            continue
        f1, f2 = edge.link_faces[0], edge.link_faces[1]
        c1 = face_colors.get(f1.index)
        c2 = face_colors.get(f2.index)
        if c1 is None or c2 is None:
            continue
        diff = (c1 - c2).length / 1.732  # Normalize by max RGB distance
        if diff > threshold:
            color_edges.append((edge, diff))

    return len(color_edges), color_edges

def analyze_angle_edges(bm, threshold_deg):
    """Analyze edges where face angle exceeds threshold."""
    angle_rad = math.radians(threshold_deg)
    angle_edges = []

    for edge in bm.edges:
        if len(edge.link_faces) != 2:
            continue
        n1 = edge.link_faces[0].normal
        n2 = edge.link_faces[1].normal
        if n1.length < 0.0001 or n2.length < 0.0001:
            continue
        try:
            angle = n1.angle(n2)
            if angle > angle_rad:
                angle_edges.append((edge, math.degrees(angle)))
        except:
            pass

    return len(angle_edges), angle_edges

# Enter edit mode for analysis and marking
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(obj.data)
bm.edges.ensure_lookup_table()
bm.faces.ensure_lookup_table()

stats['total_edges'] = len(bm.edges)
stats['boundary_edges'] = sum(1 for e in bm.edges if len(e.link_faces) != 2)

log(f"[BD Edge Marking] Total edges: {stats['total_edges']}, Boundary edges: {stats['boundary_edges']}")

# Log color attributes before any operations
def log_color_attributes(label):
    """Log current color attribute state."""
    if hasattr(obj.data, 'color_attributes') and len(obj.data.color_attributes) > 0:
        log(f"[BD Edge Marking] {label} color_attributes: {len(obj.data.color_attributes)}")
        for attr in obj.data.color_attributes:
            log(f"[BD Edge Marking]   - {attr.name}: domain={attr.domain}, data_type={attr.data_type}")
    else:
        log(f"[BD Edge Marking] {label}: NO color_attributes!")

bmesh.update_edit_mesh(obj.data)
bpy.ops.object.mode_set(mode='OBJECT')
log_color_attributes("Before processing")
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(obj.data)
bm.edges.ensure_lookup_table()
bm.faces.ensure_lookup_table()

# Check if this is a face-split mesh (all/most edges are boundaries)
boundary_ratio = stats['boundary_edges'] / max(stats['total_edges'], 1)
if boundary_ratio > 0.9:
    log(f"[BD Edge Marking] Face-split mesh detected ({boundary_ratio*100:.0f}% boundary edges)")
    log("[BD Edge Marking] Merging vertices by distance to enable edge detection...")
    log("[BD Edge Marking] NOTE: CORNER domain colors should be preserved during merge")

    # Return to object mode for merge operation
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Merge vertices by distance (weld)
    # This SHOULD preserve CORNER domain colors as they're per-loop, not per-vertex
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.0001)  # Very small threshold to only merge coincident verts
    bpy.ops.mesh.select_all(action='DESELECT')

    # Check colors after merge
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')
    log_color_attributes("After vertex merge")
    bpy.ops.object.mode_set(mode='EDIT')

    # Rebuild bmesh
    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    new_boundary = sum(1 for e in bm.edges if len(e.link_faces) != 2)
    log(f"[BD Edge Marking] After merge: {len(bm.edges)} edges, {new_boundary} boundary edges")
    stats['total_edges'] = len(bm.edges)
    stats['boundary_edges'] = new_boundary

stats['sharp_before'], stats['crease_before'] = count_edge_marks(bm)
log(f"[BD Edge Marking] Before - Sharp: {stats['sharp_before']}, Crease: {stats['crease_before']}")

# Perform operation
if OPERATION == 'FROM_COLORS':
    stats['color_edges_found'], color_edges = analyze_color_edges(bm, COLOR_THRESHOLD)
    log(f"[BD Edge Marking] Color edges found above threshold {COLOR_THRESHOLD}: {stats['color_edges_found']}")

    # Mark the edges
    crease_layer = get_crease_layer(bm)
    for edge, diff in color_edges:
        if MARK_TYPE in ['SHARP', 'BOTH']:
            edge.smooth = False
        if MARK_TYPE in ['CREASE', 'BOTH']:
            edge[crease_layer] = CREASE_VALUE
        stats['color_edges_marked'] += 1

    log(f"[BD Edge Marking] Marked {stats['color_edges_marked']} edges from colors")

elif OPERATION == 'FROM_ANGLE':
    stats['angle_edges_found'], angle_edges = analyze_angle_edges(bm, ANGLE_THRESHOLD)
    log(f"[BD Edge Marking] Angle edges found above {ANGLE_THRESHOLD}deg: {stats['angle_edges_found']}")

    # Mark the edges
    crease_layer = get_crease_layer(bm)
    for edge, angle in angle_edges:
        if MARK_TYPE in ['SHARP', 'BOTH']:
            edge.smooth = False
        if MARK_TYPE in ['CREASE', 'BOTH']:
            edge[crease_layer] = CREASE_VALUE
        stats['angle_edges_marked'] += 1

    log(f"[BD Edge Marking] Marked {stats['angle_edges_marked']} edges from angles")

elif OPERATION == 'FROM_COLORS_AND_ANGLE':
    # Combined mode: detect and mark both color and angle edges
    stats['color_edges_found'], color_edges = analyze_color_edges(bm, COLOR_THRESHOLD)
    stats['angle_edges_found'], angle_edges = analyze_angle_edges(bm, ANGLE_THRESHOLD)

    log(f"[BD Edge Marking] Color edges found above threshold {COLOR_THRESHOLD}: {stats['color_edges_found']}")
    log(f"[BD Edge Marking] Angle edges found above {ANGLE_THRESHOLD}deg: {stats['angle_edges_found']}")

    # Collect all edges to mark (avoid duplicates)
    edges_to_mark = set()
    for edge, _ in color_edges:
        edges_to_mark.add(edge)
        stats['color_edges_marked'] += 1
    for edge, _ in angle_edges:
        if edge not in edges_to_mark:
            stats['angle_edges_marked'] += 1
        edges_to_mark.add(edge)

    # Mark all collected edges
    crease_layer = get_crease_layer(bm)
    for edge in edges_to_mark:
        if MARK_TYPE in ['SHARP', 'BOTH']:
            edge.smooth = False
        if MARK_TYPE in ['CREASE', 'BOTH']:
            edge[crease_layer] = CREASE_VALUE

    total_marked = len(edges_to_mark)
    log(f"[BD Edge Marking] Marked {total_marked} total edges ({stats['color_edges_marked']} color + {stats['angle_edges_marked']} angle, some may overlap)")

elif OPERATION == 'CLEAR':
    crease_layer = get_crease_layer(bm)
    cleared_sharp = 0
    cleared_crease = 0
    for edge in bm.edges:
        if CLEAR_SHARP and not edge.smooth:
            edge.smooth = True
            cleared_sharp += 1
        if CLEAR_CREASE and edge[crease_layer] > 0.01:
            edge[crease_layer] = 0.0
            cleared_crease += 1
    log(f"[BD Edge Marking] Cleared {cleared_sharp} sharp, {cleared_crease} crease edges")

elif OPERATION == 'SHARP_TO_CREASE':
    crease_layer = get_crease_layer(bm)
    converted = 0
    for edge in bm.edges:
        if not edge.smooth:
            edge[crease_layer] = CREASE_VALUE
            converted += 1
            if CLEAR_SHARP:
                edge.smooth = True
    log(f"[BD Edge Marking] Converted {converted} sharp edges to crease")

elif OPERATION == 'CREASE_TO_SHARP':
    crease_layer = get_crease_layer(bm)
    converted = 0
    for edge in bm.edges:
        if edge[crease_layer] > 0.5:
            edge.smooth = False
            converted += 1
            if CLEAR_CREASE:
                edge[crease_layer] = 0.0
    log(f"[BD Edge Marking] Converted {converted} crease edges to sharp")

# Count final state
stats['sharp_after'], stats['crease_after'] = count_edge_marks(bm)

# Collect all marked edges for metadata output
marked_edge_pairs = []
for edge in bm.edges:
    if not edge.smooth:  # Sharp edge
        v1_idx = edge.verts[0].index
        v2_idx = edge.verts[1].index
        marked_edge_pairs.append(f"{v1_idx},{v2_idx}")

# Output marked edges for parsing (limit to 50000 to avoid huge logs)
if len(marked_edge_pairs) <= 50000:
    log(f"[MARKED_EDGES] {';'.join(marked_edge_pairs)}")
else:
    log(f"[MARKED_EDGES] {';'.join(marked_edge_pairs[:50000])}")
    log(f"[BD Edge Marking] WARNING: Truncated edge output ({len(marked_edge_pairs)} edges, showing first 50000)")

bmesh.update_edit_mesh(obj.data)
bpy.ops.object.mode_set(mode='OBJECT')

# Print summary
log(f"[BD Edge Marking] After - Sharp: {stats['sharp_after']}, Crease: {stats['crease_after']}")
log(f"[BD Edge Marking] Delta - Sharp: +{stats['sharp_after'] - stats['sharp_before']}, Crease: +{stats['crease_after'] - stats['crease_before']}")

# Output stats as structured data for parsing
log(f"[STATS] total_edges={stats['total_edges']}")
log(f"[STATS] color_edges_found={stats['color_edges_found']}")
log(f"[STATS] color_edges_marked={stats['color_edges_marked']}")
log(f"[STATS] angle_edges_found={stats['angle_edges_found']}")
log(f"[STATS] angle_edges_marked={stats['angle_edges_marked']}")
log(f"[STATS] sharp_before={stats['sharp_before']}")
log(f"[STATS] sharp_after={stats['sharp_after']}")
log(f"[STATS] crease_before={stats['crease_before']}")
log(f"[STATS] crease_after={stats['crease_after']}")

# Check colors before export
log_color_attributes("Before export")

# CRITICAL: Set color attribute as active for export (Blender 5.0+)
# Without this, GLTF exporter won't include vertex colors!
if hasattr(obj.data, 'color_attributes') and len(obj.data.color_attributes) > 0:
    color_attr = obj.data.color_attributes[0]
    obj.data.color_attributes.active_color = color_attr
    # Also set render color index if available
    if hasattr(obj.data.color_attributes, 'render_color_index'):
        idx = list(obj.data.color_attributes).index(color_attr)
        obj.data.color_attributes.render_color_index = idx
    log(f"[BD Edge Marking] Set '{color_attr.name}' as active color for export")

export_result()
'''


class BD_BlenderEdgeMarking(BlenderNodeMixin, io.ComfyNode):
    """
    Edge marking operations using BrainDeadBlender addon.

    Operations:
    - FROM_COLORS: Mark edges where vertex colors change
    - FROM_ANGLE: Mark edges by face angle
    - CLEAR: Clear edge marks
    - SHARP_TO_CREASE: Convert sharp edges to crease
    - CREASE_TO_SHARP: Convert crease to sharp edges
    """

    @classmethod
    def _generate_edge_preview(cls, vertices, edges, thickness: float = 0.002):
        """
        Generate a preview mesh showing marked edges as visible tubes.

        Args:
            vertices: Mesh vertices array
            edges: List of [v1, v2] edge pairs
            thickness: Tube radius

        Returns:
            trimesh.Trimesh with tube geometry along edges, or None on failure
        """
        import numpy as np
        import trimesh
        from trimesh.creation import cylinder

        if len(edges) == 0:
            return None

        verts = np.array(vertices)
        meshes = []

        # Limit edges for performance (preview shouldn't be too heavy)
        max_edges = 5000
        if len(edges) > max_edges:
            print(f"[BD EdgeMarking] Preview limited to {max_edges} of {len(edges)} edges")
            edges = edges[:max_edges]

        for v1, v2 in edges:
            if v1 >= len(verts) or v2 >= len(verts):
                continue

            p1 = verts[v1]
            p2 = verts[v2]

            # Skip degenerate edges
            length = np.linalg.norm(p2 - p1)
            if length < 1e-8:
                continue

            try:
                # Create cylinder along edge
                cyl = cylinder(radius=thickness, height=length, sections=6)

                # Compute transformation to align cylinder with edge
                direction = (p2 - p1) / length
                midpoint = (p1 + p2) / 2

                # Create rotation matrix to align Z-axis with edge direction
                z_axis = np.array([0, 0, 1])
                if np.abs(np.dot(direction, z_axis)) > 0.999:
                    # Edge is nearly vertical, use simple translation
                    rotation = np.eye(3) if direction[2] > 0 else np.diag([1, 1, -1])
                else:
                    # Compute rotation axis and angle
                    axis = np.cross(z_axis, direction)
                    axis = axis / np.linalg.norm(axis)
                    angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))

                    # Rodrigues rotation formula
                    K = np.array([
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]
                    ])
                    rotation = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

                # Apply transformation
                transform = np.eye(4)
                transform[:3, :3] = rotation
                transform[:3, 3] = midpoint
                cyl.apply_transform(transform)

                meshes.append(cyl)

            except Exception:
                continue  # Skip problematic edges

        if not meshes:
            return None

        # Combine all cylinders into one mesh
        combined = trimesh.util.concatenate(meshes)

        # Apply bright color for visibility (magenta)
        combined.visual.vertex_colors = np.full((len(combined.vertices), 4), [255, 0, 255, 255], dtype=np.uint8)

        return combined

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_BlenderEdgeMarking",
            display_name="BD Blender Edge Marking",
            category="🧠BrainDead/Blender",
            description="""Edge marking using BrainDeadBlender addon.

⚠️ COLOR DETECTION requires FACE/CORNER domain vertex colors!
Use BD_SampleVoxelgridColors with 'face' mode (CORNER domain).
POINT domain colors (smooth/sharp modes) share colors at vertices,
making edge detection impossible.

Operations:
- FROM_COLORS_AND_ANGLE: Mark edges by color change AND face angle (combined)
- FROM_COLORS: Mark edges where colors change
- FROM_ANGLE: Mark edges by face angle
- CLEAR: Clear sharp/crease marks
- SHARP_TO_CREASE: Convert sharp to crease
- CREASE_TO_SHARP: Convert crease to sharp

Returns statistics including pre-existing edge marks.""",
            inputs=[
                TrimeshInput("mesh"),
                io.Combo.Input(
                    "operation",
                    options=["FROM_COLORS_AND_ANGLE", "FROM_COLORS", "FROM_ANGLE", "CLEAR", "SHARP_TO_CREASE", "CREASE_TO_SHARP"],
                    default="FROM_COLORS_AND_ANGLE",
                    tooltip="Edge marking operation (combined mode detects both color and angle edges)",
                ),
                io.Float.Input(
                    "color_threshold",
                    default=0.15,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    tooltip="Color difference threshold (for FROM_COLORS)",
                ),
                io.Float.Input(
                    "angle_threshold",
                    default=30.0,
                    min=0.0,
                    max=180.0,
                    step=1.0,
                    tooltip="Angle threshold in degrees (for FROM_ANGLE)",
                ),
                io.Combo.Input(
                    "mark_mode",
                    options=["ADD", "REPLACE"],
                    default="ADD",
                    tooltip="ADD to existing marks or REPLACE all",
                ),
                io.Combo.Input(
                    "mark_type",
                    options=["BOTH", "SHARP", "CREASE"],
                    default="BOTH",
                    tooltip="Type of edge mark to apply",
                ),
                io.Float.Input(
                    "crease_value",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    tooltip="Crease strength (for CREASE marking)",
                ),
                io.Boolean.Input(
                    "clear_sharp",
                    default=False,
                    tooltip="Clear sharp marks (for CLEAR/conversion ops)",
                ),
                io.Boolean.Input(
                    "clear_crease",
                    default=False,
                    tooltip="Clear crease values (for CLEAR/conversion ops)",
                ),
                io.Int.Input(
                    "timeout",
                    default=120,
                    min=30,
                    max=600,
                    optional=True,
                    tooltip="Maximum processing time",
                ),
                io.Boolean.Input(
                    "generate_preview",
                    default=True,
                    tooltip="Generate edge preview mesh (visible wireframe of marked edges)",
                ),
                io.Float.Input(
                    "preview_thickness",
                    default=0.002,
                    min=0.0005,
                    max=0.02,
                    step=0.0005,
                    tooltip="Thickness of preview edge tubes (mesh units)",
                ),
            ],
            outputs=[
                TrimeshOutput(display_name="mesh"),
                TrimeshOutput(display_name="edge_preview"),
                EdgeMetadataOutput(display_name="edge_metadata"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        operation: str = "FROM_COLORS",
        color_threshold: float = 0.15,
        angle_threshold: float = 30.0,
        mark_mode: str = "ADD",
        mark_type: str = "BOTH",
        crease_value: float = 1.0,
        clear_sharp: bool = False,
        clear_crease: bool = False,
        timeout: int = 120,
        generate_preview: bool = True,
        preview_thickness: float = 0.002,
    ) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, None, None, "ERROR: trimesh not installed")

        available, msg = cls._check_blender()
        if not available:
            return io.NodeOutput(mesh, None, None, f"ERROR: {msg}")

        if mesh is None:
            return io.NodeOutput(None, None, None, "ERROR: No input mesh")

        input_path = None
        output_path = None
        try:
            # Use PLY format - has native vertex color support that trimesh handles reliably
            # GLB in Blender 5.0+ exports colors as material, not vertex attributes
            input_path = cls._mesh_to_temp_file(mesh, suffix='.ply')
            fd, output_path = tempfile.mkstemp(suffix='.ply')
            os.close(fd)

            extra_args = {
                'operation': operation,
                'color_threshold': color_threshold,
                'angle_threshold': angle_threshold,
                'mark_mode': mark_mode,
                'mark_type': mark_type,
                'crease_value': crease_value,
                'clear_sharp': clear_sharp,
                'clear_crease': clear_crease,
            }

            success, message, log_lines = cls._run_blender_script(
                EDGE_MARKING_SCRIPT,
                input_path,
                output_path,
                extra_args=extra_args,
                timeout=timeout,
            )

            if not success:
                return io.NodeOutput(mesh, f"ERROR: {message}")

            result_mesh = cls._load_mesh_from_file(output_path)

            # Parse stats from log lines
            stats = {}
            for line in log_lines:
                if line.startswith('[STATS]'):
                    parts = line.replace('[STATS] ', '').split('=')
                    if len(parts) == 2:
                        stats[parts[0]] = int(parts[1])

            # Get pre-existing edge counts
            sharp_before = stats.get('sharp_before', 0)
            sharp_after = stats.get('sharp_after', 0)
            crease_before = stats.get('crease_before', 0)
            crease_after = stats.get('crease_after', 0)

            # Build stats string - PRE-EXISTING MARKS FIRST (user requested)
            stats_lines = []

            # Show pre-existing marks prominently at the top
            if sharp_before > 0 or crease_before > 0:
                stats_lines.append(f"PRE-EXISTING: {sharp_before} sharp, {crease_before} crease edges")
            else:
                stats_lines.append("PRE-EXISTING: None (clean mesh)")

            stats_lines.append(f"Total edges: {stats.get('total_edges', 0)}")

            if operation == 'FROM_COLORS_AND_ANGLE':
                stats_lines.append(f"Color edges found: {stats.get('color_edges_found', 0)}")
                stats_lines.append(f"Color edges marked: {stats.get('color_edges_marked', 0)}")
                stats_lines.append(f"Angle edges found: {stats.get('angle_edges_found', 0)}")
                stats_lines.append(f"Angle edges marked: {stats.get('angle_edges_marked', 0)}")
            elif operation == 'FROM_COLORS':
                stats_lines.append(f"Color edges found: {stats.get('color_edges_found', 0)}")
                stats_lines.append(f"Color edges marked: {stats.get('color_edges_marked', 0)}")
            elif operation == 'FROM_ANGLE':
                stats_lines.append(f"Angle edges found: {stats.get('angle_edges_found', 0)}")
                stats_lines.append(f"Angle edges marked: {stats.get('angle_edges_marked', 0)}")

            stats_lines.append(f"Sharp: {sharp_before} -> {sharp_after} (+{sharp_after - sharp_before})")
            stats_lines.append(f"Crease: {crease_before} -> {crease_after} (+{crease_after - crease_before})")

            stats_str = "\n".join(stats_lines)

            # Build combined status
            if operation == 'FROM_COLORS_AND_ANGLE':
                color_marked = stats.get('color_edges_marked', 0)
                angle_marked = stats.get('angle_edges_marked', 0)
                status_line = f"Marked {color_marked} color + {angle_marked} angle edges (thresholds: color={color_threshold}, angle={angle_threshold}°)"
            elif operation == 'FROM_COLORS':
                status_line = f"Marked {stats.get('color_edges_marked', 0)} color edges (threshold: {color_threshold})"
            elif operation == 'FROM_ANGLE':
                status_line = f"Marked {stats.get('angle_edges_marked', 0)} angle edges (threshold: {angle_threshold}°)"
            else:
                status_line = f"Edge marking: {operation} complete"

            # Combine stats and status
            combined_status = f"{status_line}\n\n{stats_str}"

            # Parse marked edges from log output for edge_metadata
            marked_edges = []
            for line in log_lines:
                if line.startswith('[MARKED_EDGES]'):
                    # Format: [MARKED_EDGES] v1,v2;v1,v2;...
                    edges_str = line.replace('[MARKED_EDGES] ', '').strip()
                    if edges_str:
                        for edge_pair in edges_str.split(';'):
                            if ',' in edge_pair:
                                v1, v2 = edge_pair.split(',')
                                marked_edges.append([int(v1), int(v2)])

            # Compute edge positions from result mesh for position-based deduplication
            edge_positions = []
            if hasattr(result_mesh, 'vertices') and len(marked_edges) > 0:
                verts = np.array(result_mesh.vertices)
                for v1, v2 in marked_edges:
                    if v1 < len(verts) and v2 < len(verts):
                        p1 = verts[v1].tolist()
                        p2 = verts[v2].tolist()
                        edge_positions.append([p1, p2])

            # Build edge_metadata for downstream nodes
            edge_metadata = {
                'boundary_edges': marked_edges,
                'boundary_edge_positions': edge_positions,  # For position-based deduplication
                'num_groups': 0,  # Not applicable for edge marking
                'source': 'edge_marking',
                'operation': operation,
                'color_threshold': color_threshold if 'COLOR' in operation else None,
                'angle_threshold': angle_threshold if 'ANGLE' in operation else None,
            }

            print(f"[BD EdgeMarking] Outputting edge_metadata: {len(marked_edges)} edges ({len(edge_positions)} with positions)")

            # Clear stale planar_grouping metadata so downstream nodes don't pick it up
            # EdgeMarking supersedes PlanarGrouping - its edge_metadata should be used instead
            if hasattr(result_mesh, 'metadata') and result_mesh.metadata:
                if 'planar_grouping' in result_mesh.metadata:
                    del result_mesh.metadata['planar_grouping']
                    print(f"[BD EdgeMarking] Cleared stale planar_grouping metadata")
                # Store our edge marking data in mesh metadata as fallback
                result_mesh.metadata['edge_marking'] = edge_metadata

            # Generate edge preview mesh (visible tubes along marked edges)
            edge_preview = None
            if generate_preview and len(marked_edges) > 0 and result_mesh is not None and hasattr(result_mesh, 'vertices') and result_mesh.vertices is not None:
                try:
                    edge_preview = cls._generate_edge_preview(
                        result_mesh.vertices, marked_edges, preview_thickness
                    )
                    if edge_preview is not None:
                        print(f"[BD EdgeMarking] Generated edge preview: {len(edge_preview.faces)} faces")
                except Exception as preview_error:
                    print(f"[BD EdgeMarking] Warning: Failed to generate preview: {preview_error}")

            return io.NodeOutput(result_mesh, edge_preview, edge_metadata, combined_status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return io.NodeOutput(mesh, None, None, f"ERROR: {e}")

        finally:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if output_path and os.path.exists(output_path):
                os.remove(output_path)


# ============================================================================
# BD_BlenderVertexColors - Vertex Color Operations using Addon
# ============================================================================
VERTEX_COLORS_SCRIPT = ADDON_SETUP_SCRIPT + '''
# Read settings
OPERATION = get_env_str('BLENDER_ARG_OPERATION', 'SOLIDIFY')
SOLIDIFY_METHOD = get_env_str('BLENDER_ARG_SOLIDIFY_METHOD', 'DOMINANT')
SMOOTH_ITERATIONS = get_env_int('BLENDER_ARG_SMOOTH_ITERATIONS', 1)
TARGET_DOMAIN = get_env_str('BLENDER_ARG_TARGET_DOMAIN', 'CORNER')
OUTPUT_NAME = get_env_str('BLENDER_ARG_OUTPUT_NAME', 'Col')
APPLY_FLAT_SHADING = get_env_bool('BLENDER_ARG_APPLY_FLAT_SHADING', True)
CONVERT_DOMAIN_AFTER = get_env_bool('BLENDER_ARG_CONVERT_DOMAIN_AFTER', False)

log(f"[BD Vertex Colors] Operation: {OPERATION}, target_domain: {TARGET_DOMAIN}, convert_after: {CONVERT_DOMAIN_AFTER}")

# Check for vertex colors
has_colors = bool(obj.data.vertex_colors) or bool(obj.data.color_attributes)
if not has_colors:
    log("[BD Vertex Colors] WARNING: Mesh has no vertex colors")

# Check if addon is available
has_addon = hasattr(bpy.context.scene, 'bd_colors')

if has_addon:
    log("[BD Vertex Colors] Using BrainDeadBlender addon")
    scene = bpy.context.scene

    if hasattr(scene, 'bd_colors'):
        scene.bd_colors.solidify_method = SOLIDIFY_METHOD
        scene.bd_colors.smooth_iterations = SMOOTH_ITERATIONS
        scene.bd_colors.output_name = OUTPUT_NAME

    try:
        if OPERATION == 'SOLIDIFY':
            bpy.ops.braindead.solidify_colors()
            if APPLY_FLAT_SHADING:
                bpy.ops.braindead.apply_flat_shading()
        elif OPERATION == 'SMOOTH':
            bpy.ops.braindead.smooth_colors()
        elif OPERATION == 'CONVERT_DOMAIN':
            bpy.ops.braindead.convert_color_domain(target_domain=TARGET_DOMAIN)
        elif OPERATION == 'FINALIZE':
            bpy.ops.braindead.finalize_colors()
        elif OPERATION == 'CREATE_MATERIAL':
            bpy.ops.braindead.create_color_material()
        log(f"[BD Vertex Colors] {OPERATION} complete")

        # Optional: convert domain after operation (for SOLIDIFY, SMOOTH, etc.)
        if CONVERT_DOMAIN_AFTER and OPERATION not in ['CONVERT_DOMAIN']:
            log(f"[BD Vertex Colors] Converting domain to {TARGET_DOMAIN}...")
            bpy.ops.braindead.convert_color_domain(target_domain=TARGET_DOMAIN)
            log(f"[BD Vertex Colors] Domain conversion complete")
    except Exception as e:
        log(f"[BD Vertex Colors] Addon operation failed: {e}")

log(f"[BD Vertex Colors] Complete")

# Convert Z-up to Y-up for PLY export (GLTF import converts Y-up to Z-up)
log("[BD Vertex Colors] Converting Z-up to Y-up for output...")
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.transform.rotate(value=math.radians(-90), orient_axis='X', orient_type='GLOBAL')
bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

export_result()
'''


class BD_BlenderVertexColors(BlenderNodeMixin, io.ComfyNode):
    """
    Vertex color operations using BrainDeadBlender addon.

    Operations:
    - SOLIDIFY: Convert to solid face colors (no blending)
    - SMOOTH: Smooth/blend vertex colors
    - CONVERT_DOMAIN: Convert between CORNER and POINT domains
    - FINALIZE: Prepare colors for export
    - CREATE_MATERIAL: Create viewport material
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_BlenderVertexColors",
            display_name="BD Blender Vertex Colors",
            category="🧠BrainDead/Blender",
            description="""Vertex color operations using BrainDeadBlender addon.

Operations:
- SOLIDIFY: Solid face colors (no blending)
- SMOOTH: Blend vertex colors
- CONVERT_DOMAIN: CORNER <-> POINT
- FINALIZE: Prepare for export
- CREATE_MATERIAL: Viewport material""",
            inputs=[
                TrimeshInput("mesh"),
                io.Combo.Input(
                    "operation",
                    options=["SOLIDIFY", "SMOOTH", "CONVERT_DOMAIN", "FINALIZE", "CREATE_MATERIAL"],
                    default="SOLIDIFY",
                    tooltip="Vertex color operation",
                ),
                io.Combo.Input(
                    "solidify_method",
                    options=["DOMINANT", "AVERAGE", "FIRST"],
                    default="DOMINANT",
                    tooltip="Method for solidifying colors (SOLIDIFY op)",
                ),
                io.Int.Input(
                    "smooth_iterations",
                    default=1,
                    min=1,
                    max=100,
                    tooltip="Smoothing iterations (SMOOTH op)",
                ),
                io.Combo.Input(
                    "target_domain",
                    options=["CORNER", "POINT"],
                    default="CORNER",
                    tooltip="Target domain (CONVERT_DOMAIN op)",
                ),
                io.String.Input(
                    "output_name",
                    default="Col",
                    tooltip="Output color attribute name",
                ),
                io.Boolean.Input(
                    "apply_flat_shading",
                    default=True,
                    tooltip="Apply flat shading (for SOLIDIFY)",
                ),
                io.Boolean.Input(
                    "convert_domain_after",
                    default=False,
                    tooltip="Convert to target_domain after operation (use with SOLIDIFY, SMOOTH)",
                ),
                io.Int.Input(
                    "timeout",
                    default=120,
                    min=30,
                    max=600,
                    optional=True,
                    tooltip="Maximum processing time",
                ),
            ],
            outputs=[
                TrimeshOutput(display_name="mesh"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        operation: str = "SOLIDIFY",
        solidify_method: str = "DOMINANT",
        smooth_iterations: int = 1,
        target_domain: str = "CORNER",
        output_name: str = "Col",
        apply_flat_shading: bool = True,
        convert_domain_after: bool = False,
        timeout: int = 120,
    ) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "ERROR: trimesh not installed")

        available, msg = cls._check_blender()
        if not available:
            return io.NodeOutput(mesh, f"ERROR: {msg}")

        if mesh is None:
            return io.NodeOutput(None, "ERROR: No input mesh")

        input_path = None
        output_path = None
        try:
            input_path = cls._mesh_to_temp_file(mesh, suffix='.glb')
            fd, output_path = tempfile.mkstemp(suffix='.ply')
            os.close(fd)

            extra_args = {
                'operation': operation,
                'solidify_method': solidify_method,
                'smooth_iterations': smooth_iterations,
                'target_domain': target_domain,
                'output_name': output_name,
                'apply_flat_shading': apply_flat_shading,
                'convert_domain_after': convert_domain_after,
            }

            success, message, log_lines = cls._run_blender_script(
                VERTEX_COLORS_SCRIPT,
                input_path,
                output_path,
                extra_args=extra_args,
                timeout=timeout,
            )

            if not success:
                return io.NodeOutput(mesh, f"ERROR: {message}")

            result_mesh = cls._load_mesh_from_file(output_path)

            status = f"Vertex colors: {operation} complete"
            return io.NodeOutput(result_mesh, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return io.NodeOutput(mesh, f"ERROR: {e}")

        finally:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if output_path and os.path.exists(output_path):
                os.remove(output_path)


# ============================================================================
# BD_BlenderNormals - Normal Operations using Addon
# ============================================================================
NORMALS_SCRIPT = ADDON_SETUP_SCRIPT + '''
# Read settings
OPERATION = get_env_str('BLENDER_ARG_OPERATION', 'FIX')
METHOD = get_env_str('BLENDER_ARG_METHOD', 'BOTH')
DOUBLE_SIDED = get_env_str('BLENDER_ARG_DOUBLE_SIDED', 'False') == 'True'

log(f"[BD Normals] Operation: {OPERATION}, Method: {METHOD}, DoubleSided: {DOUBLE_SIDED}")

# Check if addon is available
has_addon = hasattr(bpy.context.scene, 'bd_normals')

if has_addon:
    log("[BD Normals] Using BrainDeadBlender addon")
    scene = bpy.context.scene

    if hasattr(scene, 'bd_normals'):
        scene.bd_normals.method = METHOD

    try:
        if OPERATION == 'FIX':
            bpy.ops.braindead.fix_normals()
        elif OPERATION == 'VERIFY':
            bpy.ops.braindead.verify_normals()
        log(f"[BD Normals] {OPERATION} complete")
    except Exception as e:
        log(f"[BD Normals] Addon operation failed: {e}, using native")
        has_addon = False

if not has_addon:
    log("[BD Normals] Using native Blender normal operations")
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    log("[BD Normals] Normals made consistent (native)")

# Set double-sided material flag if requested
if DOUBLE_SIDED:
    for mat in obj.data.materials:
        if mat and mat.use_backface_culling:
            mat.use_backface_culling = False
    log("[BD Normals] Set materials to double-sided")

# Export as GLB (preserves materials, UVs, vertex colors; handles Y-up natively)
export_result()
'''


class BD_BlenderNormals(BlenderNodeMixin, io.ComfyNode):
    """
    Normal operations using BrainDeadBlender addon.

    Operations:
    - FIX: Fix normals to point outward
    - VERIFY: Check normal orientation
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_BlenderNormals",
            display_name="BD Blender Normals",
            category="🧠BrainDead/Blender",
            description="""Normal operations using BrainDeadBlender addon.

Operations:
- FIX: Fix normals to point outward (Blender topology-based, most reliable)
- VERIFY: Check normal orientation

Methods:
- BLENDER: Topology-based recalculate outside
- DIRECTION: Center-based heuristic
- BOTH: Combined approach

Preserves PBR materials, UVs, and vertex colors.""",
            inputs=[
                TrimeshInput("mesh"),
                io.Combo.Input(
                    "operation",
                    options=["FIX", "VERIFY"],
                    default="FIX",
                    tooltip="Normal operation",
                ),
                io.Combo.Input(
                    "method",
                    options=["BOTH", "BLENDER", "DIRECTION"],
                    default="BOTH",
                    tooltip="Fix method",
                ),
                io.Boolean.Input(
                    "double_sided",
                    default=False,
                    optional=True,
                    tooltip="Set material to double-sided (renders both face sides, masks remaining flips)",
                ),
                io.Int.Input(
                    "timeout",
                    default=120,
                    min=30,
                    max=600,
                    optional=True,
                    tooltip="Maximum processing time",
                ),
            ],
            outputs=[
                TrimeshOutput(display_name="mesh"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        operation: str = "FIX",
        method: str = "BOTH",
        double_sided: bool = False,
        timeout: int = 120,
    ) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "ERROR: trimesh not installed")

        available, msg = cls._check_blender()
        if not available:
            return io.NodeOutput(mesh, f"ERROR: {msg}")

        if mesh is None:
            return io.NodeOutput(None, "ERROR: No input mesh")

        input_path = None
        output_path = None
        try:
            input_path = cls._mesh_to_temp_file(mesh, suffix='.glb')
            fd, output_path = tempfile.mkstemp(suffix='.glb')
            os.close(fd)

            extra_args = {
                'operation': operation,
                'method': method,
                'double_sided': double_sided,
            }

            success, message, log_lines = cls._run_blender_script(
                NORMALS_SCRIPT,
                input_path,
                output_path,
                extra_args=extra_args,
                timeout=timeout,
            )

            if not success:
                return io.NodeOutput(mesh, f"ERROR: {message}")

            result_mesh = cls._load_mesh_from_file(output_path)

            ds_info = " +doubleSided" if double_sided else ""
            status = f"Normals: {operation} ({method}){ds_info} | {len(result_mesh.vertices):,} verts"
            return io.NodeOutput(result_mesh, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return io.NodeOutput(mesh, f"ERROR: {e}")

        finally:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if output_path and os.path.exists(output_path):
                os.remove(output_path)


# ============================================================================
# BD_BlenderMergePlanes - Merge geometry within marked regions
# ============================================================================
MERGE_PLANES_SCRIPT = ADDON_SETUP_SCRIPT + '''
import math
import numpy as np
import json

# Read settings
DELIMIT_SHARP = get_env_bool('BLENDER_ARG_DELIMIT_SHARP', True)
DELIMIT_SEAM = get_env_bool('BLENDER_ARG_DELIMIT_SEAM', False)
DELIMIT_MATERIAL = get_env_bool('BLENDER_ARG_DELIMIT_MATERIAL', False)
DELIMIT_NORMAL = get_env_bool('BLENDER_ARG_DELIMIT_NORMAL', True)
DISSOLVE_ANGLE = get_env_float('BLENDER_ARG_DISSOLVE_ANGLE', 5.0)
MIN_FACES_PER_REGION = get_env_int('BLENDER_ARG_MIN_FACES_PER_REGION', 1)
FACES_PER_AREA_PERCENT = get_env_float('BLENDER_ARG_FACES_PER_AREA_PERCENT', 1.0)
MAX_FACES_PER_REGION = get_env_int('BLENDER_ARG_MAX_FACES_PER_REGION', 100)
OUTPUT_TOPOLOGY = get_env_str('BLENDER_ARG_OUTPUT_TOPOLOGY', 'TRI')
FLATTEN_REGIONS = get_env_bool('BLENDER_ARG_FLATTEN_REGIONS', True)
MIN_SUBDIVIDE_AREA_PERCENT = get_env_float('BLENDER_ARG_MIN_SUBDIVIDE_AREA_PERCENT', 3.0)
METADATA_PATH = os.environ.get('BLENDER_METADATA_PATH', '')

original_faces = len(obj.data.polygons)
original_verts = len(obj.data.vertices)

# Count existing sharp/seam edges
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(obj.data)
bm.edges.ensure_lookup_table()

sharp_count = sum(1 for e in bm.edges if not e.smooth)
seam_count = sum(1 for e in bm.edges if e.seam)
bpy.ops.object.mode_set(mode='OBJECT')

log(f"[BD Merge Planes] Starting - {original_faces} faces, {original_verts} verts")
log(f"[BD Merge Planes] Pre-existing marks: {sharp_count} sharp, {seam_count} seam edges")

# ============================================================================
# READ METADATA AND MARK BOUNDARY EDGES FROM PLANAR GROUPING
# ============================================================================
metadata_edges_marked = 0
if METADATA_PATH and os.path.exists(METADATA_PATH):
    log(f"[BD Merge Planes] Reading metadata from: {METADATA_PATH}")
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)

        if 'planar_grouping' in metadata:
            pg = metadata['planar_grouping']
            boundary_edges = pg.get('boundary_edges', [])
            num_groups = pg.get('num_groups', 0)

            log(f"[BD Merge Planes] Planar grouping: {num_groups} groups, {len(boundary_edges)} boundary edges")

            if boundary_edges:
                # Build edge lookup by vertex indices
                mesh = obj.data
                bpy.ops.object.mode_set(mode='EDIT')
                bm = bmesh.from_edit_mesh(mesh)
                bm.edges.ensure_lookup_table()
                bm.verts.ensure_lookup_table()

                # Create edge lookup: (min_vert, max_vert) -> edge
                edge_lookup = {}
                for edge in bm.edges:
                    v1, v2 = edge.verts[0].index, edge.verts[1].index
                    key = (min(v1, v2), max(v1, v2))
                    edge_lookup[key] = edge

                # Also build position-based lookup for face-split meshes
                # Quantize to 6 decimal places
                def quantize_pos(co, precision=6):
                    scale = 10 ** precision
                    return (int(round(co.x * scale)), int(round(co.y * scale)), int(round(co.z * scale)))

                pos_edge_lookup = {}
                for edge in bm.edges:
                    p1 = quantize_pos(edge.verts[0].co)
                    p2 = quantize_pos(edge.verts[1].co)
                    key = tuple(sorted([p1, p2]))
                    pos_edge_lookup[key] = edge

                # Mark boundary edges as sharp
                for edge_data in boundary_edges:
                    v1, v2 = edge_data[0], edge_data[1]
                    key = (min(v1, v2), max(v1, v2))

                    edge = edge_lookup.get(key)
                    if edge:
                        edge.smooth = False  # Mark as sharp
                        edge.seam = True     # Also mark as seam
                        metadata_edges_marked += 1

                bmesh.update_edit_mesh(mesh)
                bpy.ops.object.mode_set(mode='OBJECT')

                log(f"[BD Merge Planes] Marked {metadata_edges_marked} edges from planar grouping metadata")
                sharp_count += metadata_edges_marked
                seam_count += metadata_edges_marked
    except Exception as e:
        log(f"[BD Merge Planes] WARNING: Failed to read metadata: {e}")
else:
    log(f"[BD Merge Planes] No metadata file provided")

log(f"[BD Merge Planes] Delimit: sharp={DELIMIT_SHARP}, seam={DELIMIT_SEAM}, material={DELIMIT_MATERIAL}, normal={DELIMIT_NORMAL}")

# Build delimit set
delimit_set = set()
if DELIMIT_SHARP:
    delimit_set.add('SHARP')
if DELIMIT_SEAM:
    delimit_set.add('SEAM')
if DELIMIT_MATERIAL:
    delimit_set.add('MATERIAL')
if DELIMIT_NORMAL:
    delimit_set.add('NORMAL')

if not delimit_set:
    log("[BD Merge Planes] WARNING: No delimit options set, all edges can dissolve!")

# Pre-cleanup to avoid dissolve issues
log("[BD Merge Planes] Pre-cleanup: removing doubles, fixing normals...")
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=0.0001)
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')

pre_cleanup_faces = len(obj.data.polygons)
log(f"[BD Merge Planes] After pre-cleanup: {pre_cleanup_faces} faces")

# Step 1: Use Decimate Modifier (DISSOLVE mode) - more stable than dissolve_limited operator
log(f"[BD Merge Planes] Running Planar Decimate (angle={DISSOLVE_ANGLE}°, delimit={delimit_set})")

# Apply decimate modifier with DISSOLVE mode
mod = obj.modifiers.new(name="MergePlanes", type='DECIMATE')
mod.decimate_type = 'DISSOLVE'
mod.angle_limit = math.radians(DISSOLVE_ANGLE)
mod.use_dissolve_boundaries = False  # Don't dissolve mesh boundaries
mod.delimit = delimit_set

# Apply the modifier
bpy.context.view_layer.objects.active = obj
bpy.ops.object.modifier_apply(modifier=mod.name)

dissolved_faces = len(obj.data.polygons)
log(f"[BD Merge Planes] After dissolve: {dissolved_faces} faces (merged {pre_cleanup_faces - dissolved_faces})")

# Post-cleanup: fix any issues created by dissolve
log("[BD Merge Planes] Post-cleanup: fixing potential issues...")
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')

# Remove duplicate/overlapping faces (faces with same vertices)
bm = bmesh.from_edit_mesh(obj.data)
bm.faces.ensure_lookup_table()

# Build face signatures and find duplicates
face_signatures = {}
duplicate_faces = []
for face in bm.faces:
    # Signature = sorted tuple of vertex indices
    sig = tuple(sorted([v.index for v in face.verts]))
    if sig in face_signatures:
        duplicate_faces.append(face)
    else:
        face_signatures[sig] = face

# Delete duplicate faces
if duplicate_faces:
    log(f"[BD Merge Planes] Removing {len(duplicate_faces)} duplicate/overlapping faces")
    for face in duplicate_faces:
        bm.faces.remove(face)
    bmesh.update_edit_mesh(obj.data)

# Also check for degenerate faces (zero area)
bm = bmesh.from_edit_mesh(obj.data)
bm.faces.ensure_lookup_table()
degenerate_faces = [f for f in bm.faces if f.calc_area() < 1e-8]
if degenerate_faces:
    log(f"[BD Merge Planes] Removing {len(degenerate_faces)} degenerate (zero-area) faces")
    for face in degenerate_faces:
        bm.faces.remove(face)
    bmesh.update_edit_mesh(obj.data)

bpy.ops.mesh.select_all(action='SELECT')

# Fill any small holes that might have been created
try:
    bpy.ops.mesh.fill_holes(sides=4)
except:
    pass
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')

post_cleanup_faces = len(obj.data.polygons)
if post_cleanup_faces != dissolved_faces:
    log(f"[BD Merge Planes] Post-cleanup filled {post_cleanup_faces - dissolved_faces} holes")

# Step 2: Optional subdivision of large regions
# Only subdivide if faces_per_area_percent > 0 AND face is larger than threshold
if FACES_PER_AREA_PERCENT > 0 and dissolved_faces > 0:
    mesh = obj.data
    total_area = sum(p.area for p in mesh.polygons)

    if total_area > 0:
        # Find faces that are large enough to warrant subdivision
        # Only subdivide faces > X% of total area (configurable)
        min_area_threshold = (MIN_SUBDIVIDE_AREA_PERCENT / 100.0) * total_area

        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(mesh)
        bm.faces.ensure_lookup_table()

        large_faces = []
        for face in bm.faces:
            face_area = face.calc_area()
            if face_area > min_area_threshold:
                area_percent = (face_area / total_area) * 100
                target_faces = int(area_percent * FACES_PER_AREA_PERCENT)
                target_faces = max(MIN_FACES_PER_REGION, min(target_faces, MAX_FACES_PER_REGION))
                if target_faces > 1:
                    large_faces.append((face.index, face_area, target_faces))

        bmesh.update_edit_mesh(mesh)

        if large_faces:
            log(f"[BD Merge Planes] Found {len(large_faces)} large regions to subdivide")

            # Sort by area descending
            large_faces.sort(key=lambda x: x[1], reverse=True)

            # Use poke_faces for cleaner fan subdivision (center point + edges to corners)
            subdivided_count = 0
            for face_idx, face_area, target in large_faces[:20]:  # Limit to 20 largest
                bpy.ops.mesh.select_all(action='DESELECT')
                bm = bmesh.from_edit_mesh(obj.data)
                bm.faces.ensure_lookup_table()

                if face_idx < len(bm.faces):
                    bm.faces[face_idx].select = True
                    bmesh.update_edit_mesh(obj.data)

                    # Poke creates center vertex + fan triangles (cleaner than grid)
                    # For more density, poke then subdivide edges once
                    if target <= 4:
                        bpy.ops.mesh.poke()  # Creates n triangles from n-gon
                    else:
                        bpy.ops.mesh.poke()
                        bpy.ops.mesh.subdivide(number_cuts=1)  # One level of subdivision

                    subdivided_count += 1

            bpy.ops.object.mode_set(mode='OBJECT')
            log(f"[BD Merge Planes] Subdivided {subdivided_count} large regions (poke method)")
        else:
            bpy.ops.object.mode_set(mode='OBJECT')
            log(f"[BD Merge Planes] No regions large enough to subdivide (threshold: {MIN_SUBDIVIDE_AREA_PERCENT}% area)")
else:
    log(f"[BD Merge Planes] Subdivision disabled (faces_per_area_percent=0)")

# Step 3: Flatten regions (ensure coplanar within each bounded region)
if FLATTEN_REGIONS:
    log("[BD Merge Planes] Flattening regions...")
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    # Use smooth with factor 0 and flatten option
    try:
        # Select linked flat faces and average their normals
        bpy.ops.mesh.faces_shade_flat()
    except:
        pass
    bpy.ops.object.mode_set(mode='OBJECT')

# Step 4: Convert to output topology
current_faces = len(obj.data.polygons)
if OUTPUT_TOPOLOGY == 'TRI':
    log("[BD Merge Planes] Triangulating...")
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.object.mode_set(mode='OBJECT')
elif OUTPUT_TOPOLOGY == 'QUAD':
    log("[BD Merge Planes] Converting to quads...")
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.tris_convert_to_quads(
        face_threshold=math.radians(40),
        shape_threshold=math.radians(40)
    )
    bpy.ops.object.mode_set(mode='OBJECT')
else:
    log("[BD Merge Planes] Keeping ngons")

final_faces = len(obj.data.polygons)
final_verts = len(obj.data.vertices)

# Count final sharp/seam edges
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(obj.data)
bm.edges.ensure_lookup_table()
final_sharp = sum(1 for e in bm.edges if not e.smooth)
final_seam = sum(1 for e in bm.edges if e.seam)
final_edges = len(bm.edges)
bpy.ops.object.mode_set(mode='OBJECT')

log(f"[BD Merge Planes] Complete: {original_faces} -> {final_faces} faces")
log(f"[BD Merge Planes] Vertices: {original_verts} -> {final_verts}")
log(f"[BD Merge Planes] Final edges: {final_edges} ({final_sharp} sharp, {final_seam} seam)")

# Output stats for parsing
log(f"[STATS] original_faces={original_faces}")
log(f"[STATS] original_verts={original_verts}")
log(f"[STATS] pre_cleanup_faces={pre_cleanup_faces}")
log(f"[STATS] dissolved_faces={dissolved_faces}")
log(f"[STATS] final_faces={final_faces}")
log(f"[STATS] final_verts={final_verts}")
log(f"[STATS] final_edges={final_edges}")
log(f"[STATS] sharp_before={sharp_count}")
log(f"[STATS] sharp_after={final_sharp}")
log(f"[STATS] seam_before={seam_count}")
log(f"[STATS] seam_after={final_seam}")
log(f"[STATS] metadata_edges_marked={metadata_edges_marked}")
log(f"[STATS] faces_merged={original_faces - dissolved_faces}")

# Convert Z-up to Y-up for PLY export
log("[BD Merge Planes] Converting Z-up to Y-up for output...")
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.transform.rotate(value=math.radians(-90), orient_axis='X', orient_type='GLOBAL')
bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

export_result()
'''


class BD_BlenderMergePlanes(BlenderNodeMixin, io.ComfyNode):
    """
    Merge geometry within marked planar regions.

    Dissolves faces while respecting SHARP/SEAM edge marks,
    then optionally subdivides large regions for deformation flexibility.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_BlenderMergePlanes",
            display_name="BD Blender Merge Planes",
            category="🧠BrainDead/Blender",
            description="""Merge geometry within planar regions.

Uses Limited Dissolve to merge coplanar faces while preserving hard edges.
Then subdivides large regions proportionally for deformation flexibility.

KEY: delimit_normal=True (default) auto-detects planar boundaries!
No need for explicit SHARP marks - Blender detects them from face normals.

Workflow:
1. [Optional] BD_PlanarGrouping with straighten_boundaries for cleaner edges
2. BD_BlenderMergePlanes (delimit_normal=True) → auto-preserves plane boundaries
3. Result: clean low-poly with hard edges intact

Parameters:
- delimit_normal: Auto-detect planes from face normals (RECOMMENDED)
- delimit_sharp/seam: Respect explicit edge marks if present
- dissolve_angle: Max angle for faces to merge
- output_topology: NGON (fastest), TRI (compatible), QUAD (cleanest)""",
            inputs=[
                TrimeshInput("mesh"),
                EdgeMetadataInput(
                    "edge_metadata",
                    tooltip="Edge metadata from BD_PlanarGrouping or BD_BlenderEdgeMarking. If connected, edges will be marked before dissolve.",
                ),
                io.Boolean.Input(
                    "delimit_sharp",
                    default=True,
                    tooltip="Respect sharp edge marks (won't dissolve across)",
                ),
                io.Boolean.Input(
                    "delimit_seam",
                    default=False,
                    tooltip="Respect seam edge marks",
                ),
                io.Boolean.Input(
                    "delimit_material",
                    default=False,
                    tooltip="Respect material boundaries",
                ),
                io.Boolean.Input(
                    "delimit_normal",
                    default=True,
                    tooltip="Respect normal direction changes (auto-detect planes). Enable this to preserve planar group boundaries without needing explicit SHARP marks.",
                ),
                io.Float.Input(
                    "dissolve_angle",
                    default=5.0,
                    min=0.1,
                    max=45.0,
                    step=0.5,
                    tooltip="Max angle between faces to allow merging (degrees)",
                ),
                io.Int.Input(
                    "min_faces_per_region",
                    default=1,
                    min=1,
                    max=100,
                    tooltip="Minimum faces to keep per region",
                ),
                io.Float.Input(
                    "faces_per_area_percent",
                    default=0.0,
                    min=0.0,
                    max=50.0,
                    step=0.5,
                    tooltip="Target faces per 1% of surface area (0=no subdivision, keeps clean NGONs)",
                ),
                io.Int.Input(
                    "max_faces_per_region",
                    default=100,
                    min=1,
                    max=1000,
                    tooltip="Maximum faces per region (prevents over-subdivision)",
                ),
                io.Combo.Input(
                    "output_topology",
                    options=["TRI", "QUAD", "NGON"],
                    default="TRI",
                    tooltip="Output face type: TRI (compatible), QUAD (clean), NGON (minimal)",
                ),
                io.Boolean.Input(
                    "flatten_regions",
                    default=True,
                    tooltip="Ensure faces stay coplanar within each region",
                ),
                io.Float.Input(
                    "min_subdivide_area_percent",
                    default=3.0,
                    min=0.1,
                    max=50.0,
                    step=0.5,
                    tooltip="Minimum face area (% of total) to consider for subdivision. Lower = more faces subdivided.",
                ),
                io.Int.Input(
                    "timeout",
                    default=300,
                    min=60,
                    max=1800,
                    optional=True,
                    tooltip="Maximum processing time in seconds",
                ),
            ],
            outputs=[
                TrimeshOutput(display_name="mesh"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        edge_metadata: dict = None,
        delimit_sharp: bool = True,
        delimit_seam: bool = False,
        delimit_material: bool = False,
        delimit_normal: bool = True,
        dissolve_angle: float = 5.0,
        min_faces_per_region: int = 1,
        faces_per_area_percent: float = 0.0,
        max_faces_per_region: int = 100,
        output_topology: str = "TRI",
        flatten_regions: bool = True,
        min_subdivide_area_percent: float = 3.0,
        timeout: int = 300,
    ) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "ERROR: trimesh not installed")

        available, msg = cls._check_blender()
        if not available:
            return io.NodeOutput(mesh, f"ERROR: {msg}")

        if mesh is None:
            return io.NodeOutput(None, "ERROR: No input mesh")

        orig_faces = len(mesh.faces) if hasattr(mesh, 'faces') else 0

        import json

        input_path = None
        output_path = None
        metadata_path = None
        try:
            # Export mesh (without automatic metadata - we'll handle it manually)
            input_path = cls._mesh_to_temp_file(mesh, suffix='.glb', export_metadata=False)
            fd, output_path = tempfile.mkstemp(suffix='.ply')
            os.close(fd)

            # Handle edge metadata - prefer explicit input over mesh.metadata
            if edge_metadata is not None:
                # Use explicitly passed edge_metadata
                fd, metadata_path = tempfile.mkstemp(suffix='.json')
                os.close(fd)
                with open(metadata_path, 'w') as f:
                    json.dump({'planar_grouping': edge_metadata}, f)
                print(f"[BD MergePlanes] Using explicit edge_metadata: {len(edge_metadata.get('boundary_edges', []))} edges from {edge_metadata.get('source', 'unknown')}")
            elif hasattr(mesh, 'metadata') and mesh.metadata:
                # Fall back to mesh.metadata (backward compatibility)
                # Check for edge_marking first (from BD_BlenderEdgeMarking), then planar_grouping
                source_data = None
                source_name = None

                if 'edge_marking' in mesh.metadata:
                    source_data = mesh.metadata['edge_marking']
                    source_name = 'edge_marking'
                elif 'planar_grouping' in mesh.metadata:
                    source_data = mesh.metadata['planar_grouping']
                    source_name = 'planar_grouping'

                if source_data:
                    boundary_edges = source_data.get('boundary_edges', [])
                    boundary_edges_native = [[int(v1), int(v2)] for v1, v2 in boundary_edges]
                    metadata = {
                        'planar_grouping': {
                            'boundary_edges': boundary_edges_native,
                            'num_groups': int(source_data.get('num_groups', 0)),
                            'angle_threshold': float(source_data.get('angle_threshold', 15.0) if source_data.get('angle_threshold') else 15.0),
                        }
                    }
                    fd, metadata_path = tempfile.mkstemp(suffix='.json')
                    os.close(fd)
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f)
                    print(f"[BD MergePlanes] Using mesh.metadata['{source_name}']: {len(boundary_edges_native)} boundary edges")

            extra_args = {
                'delimit_sharp': delimit_sharp,
                'delimit_seam': delimit_seam,
                'delimit_material': delimit_material,
                'delimit_normal': delimit_normal,
                'dissolve_angle': dissolve_angle,
                'min_faces_per_region': min_faces_per_region,
                'faces_per_area_percent': faces_per_area_percent,
                'max_faces_per_region': max_faces_per_region,
                'output_topology': output_topology,
                'flatten_regions': flatten_regions,
                'min_subdivide_area_percent': min_subdivide_area_percent,
            }

            success, message, log_lines = cls._run_blender_script(
                MERGE_PLANES_SCRIPT,
                input_path,
                output_path,
                extra_args=extra_args,
                timeout=timeout,
                metadata_path=metadata_path,
            )

            if not success:
                return io.NodeOutput(mesh, f"ERROR: {message}")

            result_mesh = cls._load_mesh_from_file(output_path)
            new_faces = len(result_mesh.faces)

            # Parse stats from log lines
            stats = {}
            for line in log_lines:
                if line.startswith('[STATS]'):
                    parts = line.replace('[STATS] ', '').split('=')
                    if len(parts) == 2:
                        try:
                            stats[parts[0]] = int(parts[1])
                        except ValueError:
                            stats[parts[0]] = parts[1]

            # Build detailed status
            status_lines = []

            # Metadata edges from planar grouping
            metadata_edges = stats.get('metadata_edges_marked', 0)
            if metadata_edges > 0:
                status_lines.append(f"METADATA: {metadata_edges} boundary edges marked from planar grouping")

            # Pre-existing marks
            sharp_before = stats.get('sharp_before', 0)
            seam_before = stats.get('seam_before', 0)
            if sharp_before > 0 or seam_before > 0:
                status_lines.append(f"PRE-EXISTING: {sharp_before} sharp, {seam_before} seam")
            elif metadata_edges == 0:
                status_lines.append("PRE-EXISTING: None (use BD_PlanarGrouping to add boundary edges)")

            # Merge results
            faces_merged = stats.get('faces_merged', orig_faces - new_faces)
            status_lines.append(f"Faces: {orig_faces:,} → {new_faces:,} ({faces_merged:,} merged)")

            # Vertices
            orig_verts = stats.get('original_verts', 0)
            final_verts = stats.get('final_verts', 0)
            if orig_verts and final_verts:
                status_lines.append(f"Verts: {orig_verts:,} → {final_verts:,}")

            # Final edge marks
            sharp_after = stats.get('sharp_after', 0)
            seam_after = stats.get('seam_after', 0)
            status_lines.append(f"Final marks: {sharp_after} sharp, {seam_after} seam")

            # Output topology
            status_lines.append(f"Output: {output_topology}")

            status = "\n".join(status_lines)
            return io.NodeOutput(result_mesh, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return io.NodeOutput(mesh, f"ERROR: {e}")

        finally:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if output_path and os.path.exists(output_path):
                os.remove(output_path)
            if metadata_path and os.path.exists(metadata_path):
                os.remove(metadata_path)


# ============================================================================
# EXPORTS
# ============================================================================

# V3 node list
ADDON_V3_NODES = [
    BD_BlenderRemesh,
    BD_BlenderCleanup,
    BD_BlenderEdgeMarking,
    BD_BlenderVertexColors,
    BD_BlenderNormals,
    BD_BlenderMergePlanes,
]

# V1 compatibility
ADDON_NODES = {
    "BD_BlenderRemesh": BD_BlenderRemesh,
    "BD_BlenderCleanup": BD_BlenderCleanup,
    "BD_BlenderEdgeMarking": BD_BlenderEdgeMarking,
    "BD_BlenderVertexColors": BD_BlenderVertexColors,
    "BD_BlenderNormals": BD_BlenderNormals,
    "BD_BlenderMergePlanes": BD_BlenderMergePlanes,
}

ADDON_DISPLAY_NAMES = {
    "BD_BlenderRemesh": "BD Blender Remesh",
    "BD_BlenderCleanup": "BD Blender Cleanup",
    "BD_BlenderEdgeMarking": "BD Blender Edge Marking",
    "BD_BlenderVertexColors": "BD Blender Vertex Colors",
    "BD_BlenderNormals": "BD Blender Normals",
    "BD_BlenderMergePlanes": "BD Blender Merge Planes",
}
