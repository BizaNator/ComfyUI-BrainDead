"""
BD_BlenderDecimateV2 - Full-featured stylized low-poly mesh decimation using Blender.

Complete port of Decimate_v1.py with all features:
- Pre-cleanup (fix non-manifold geometry)
- Color edge detection (marks color boundaries as sharp/seams - CRITICAL for edge preservation)
- Hole filling
- Internal geometry removal (raycast or simple)
- Planar decimation with edge preservation (respects sharp/seam marks)
- Collapse decimation
- Face-based vertex color transfer (no bleeding)
- Sharp edge marking by angle
- Normal fixing
"""

import os
import tempfile

from comfy_api.latest import io

# Import custom TRIMESH type (matches TRELLIS2)
from ..mesh.types import TrimeshInput, TrimeshOutput

from .base import BlenderNodeMixin, HAS_TRIMESH

if HAS_TRIMESH:
    import trimesh


# ============================================================================
# FULL DECIMATE BLENDER SCRIPT
# Complete port of Decimate_v1.py with environment variable configuration
# ============================================================================
FULL_DECIMATE_SCRIPT = '''
import bpy
import bmesh
import math
import sys
import os
import time
from mathutils import Vector
from mathutils.bvhtree import BVHTree

# ============================================================================
# CONFIGURATION - Read from environment variables
# ============================================================================
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

# Pre-cleanup settings
PRE_CLEANUP = get_env_bool('BLENDER_ARG_PRE_CLEANUP', True)
FIX_NON_MANIFOLD = get_env_bool('BLENDER_ARG_FIX_NON_MANIFOLD', True)
MERGE_DISTANCE = get_env_float('BLENDER_ARG_MERGE_DISTANCE', 0.0001)

# Color edge detection (CRITICAL for edge preservation)
DETECT_COLOR_EDGES = get_env_bool('BLENDER_ARG_DETECT_COLOR_EDGES', True)
COLOR_EDGE_THRESHOLD = get_env_float('BLENDER_ARG_COLOR_EDGE_THRESHOLD', 0.15)
MARK_COLOR_EDGES_SHARP = get_env_bool('BLENDER_ARG_MARK_COLOR_EDGES_SHARP', True)
MARK_COLOR_EDGES_SEAM = get_env_bool('BLENDER_ARG_MARK_COLOR_EDGES_SEAM', True)

# Hole filling
FILL_HOLES = get_env_bool('BLENDER_ARG_FILL_HOLES', True)
FILL_HOLES_MAX_SIDES = get_env_int('BLENDER_ARG_FILL_HOLES_MAX_SIDES', 100)

# Internal geometry removal
REMOVE_INTERNAL = get_env_bool('BLENDER_ARG_REMOVE_INTERNAL', False)
INTERNAL_REMOVAL_METHOD = get_env_str('BLENDER_ARG_INTERNAL_REMOVAL_METHOD', 'SIMPLE')

# Decimation settings
TARGET_FACES = get_env_int('BLENDER_ARG_TARGET_FACES', 5000)
PLANAR_ANGLE = get_env_float('BLENDER_ARG_PLANAR_ANGLE', 7.0)
PRESERVE_BOUNDARIES = get_env_bool('BLENDER_ARG_PRESERVE_BOUNDARIES', True)
PRESERVE_SEAMS = get_env_bool('BLENDER_ARG_PRESERVE_SEAMS', True)

# Edge marking
SHARP_ANGLE = get_env_float('BLENDER_ARG_SHARP_ANGLE', 14.0)

# Normal fixing
FIX_NORMALS = get_env_bool('BLENDER_ARG_FIX_NORMALS', True)

# Color preservation
PRESERVE_COLORS = get_env_bool('BLENDER_ARG_PRESERVE_COLORS', True)

# Debug output
DEBUG_PATH = get_env_str('BLENDER_ARG_DEBUG_PATH', '')

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def log(msg):
    """Print with immediate flush for real-time output."""
    print(msg)
    sys.stdout.flush()

def ensure_object_mode():
    """Ensure we're in object mode."""
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

def get_face_count(obj):
    """Get face count from mesh."""
    return len(obj.data.polygons)

def get_vertex_count(obj):
    """Get vertex count from mesh."""
    return len(obj.data.vertices)

# ============================================================================
# PRE-CLEANUP FUNCTIONS
# ============================================================================
def pre_cleanup_mesh(obj):
    """Pre-cleanup mesh BEFORE decimation. Fixes non-manifold geometry."""
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='VERT')

    initial_verts = len(obj.data.vertices)

    # Merge doubles first
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=MERGE_DISTANCE)

    merged = initial_verts - len(obj.data.vertices)
    if merged > 0:
        log(f"[Pre-Cleanup] Merged {merged} duplicate vertices")

    # Recalculate normals
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)

    if FIX_NON_MANIFOLD:
        # Select and delete loose vertices
        bpy.ops.mesh.select_all(action='DESELECT')
        try:
            bpy.ops.mesh.select_loose()
            bm = bmesh.from_edit_mesh(obj.data)
            loose_count = sum(1 for v in bm.verts if v.select)
            if loose_count > 0:
                bpy.ops.mesh.delete(type='VERT')
                log(f"[Pre-Cleanup] Removed {loose_count} loose vertices")
        except:
            pass

        # Remove degenerate faces
        bpy.ops.mesh.select_all(action='SELECT')
        try:
            bpy.ops.mesh.dissolve_degenerate(threshold=0.0001)
        except:
            pass

    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    log(f"[Pre-Cleanup] Done: {get_vertex_count(obj)} verts, {get_face_count(obj)} faces")


# ============================================================================
# COLOR EDGE DETECTION (CRITICAL FOR EDGE PRESERVATION)
# ============================================================================
def color_difference(c1, c2):
    """Calculate color difference (0-1 range)."""
    if c1 is None or c2 is None:
        return 0
    return (c1 - c2).length / 1.732  # Normalize by max RGB distance

def detect_color_edges_from_vertex_colors(obj):
    """
    Detect edges where vertex colors change significantly.
    Mark them as sharp and/or seams for preservation during decimation.

    THIS IS CRITICAL - it tells the planar decimate to preserve these edges.
    """
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    # Find color layer
    color_layer = bm.loops.layers.color.active
    if not color_layer:
        for layer in bm.loops.layers.color.values():
            color_layer = layer
            break

    if not color_layer:
        log("[Color Edges] WARNING: No vertex color layer found")
        bpy.ops.object.mode_set(mode='OBJECT')
        return 0

    log(f"[Color Edges] Analyzing {len(bm.faces)} faces for color boundaries...")

    # Calculate average color for each face
    face_colors = {}
    for face in bm.faces:
        color_sum = Vector((0, 0, 0))
        count = 0
        for loop in face.loops:
            col = loop[color_layer]
            color_sum += Vector((col[0], col[1], col[2]))
            count += 1
        if count > 0:
            face_colors[face.index] = color_sum / count

    # Find edges where adjacent faces have different colors
    marked_count = 0
    for edge in bm.edges:
        if len(edge.link_faces) != 2:
            continue

        f1, f2 = edge.link_faces[0], edge.link_faces[1]
        c1 = face_colors.get(f1.index)
        c2 = face_colors.get(f2.index)

        diff = color_difference(c1, c2)
        if diff > COLOR_EDGE_THRESHOLD:
            if MARK_COLOR_EDGES_SHARP:
                edge.smooth = False
            if MARK_COLOR_EDGES_SEAM:
                edge.seam = True
            marked_count += 1

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    log(f"[Color Edges] Marked {marked_count} color boundary edges (threshold: {COLOR_EDGE_THRESHOLD})")
    return marked_count


# ============================================================================
# HOLE FILLING
# ============================================================================
def fill_holes_simple(obj):
    """Fill holes in mesh (open boundaries)."""
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bpy.ops.mesh.select_mode(type='EDGE')
    bpy.ops.mesh.select_all(action='DESELECT')

    # Select boundary edges (hole edges)
    bpy.ops.mesh.select_non_manifold(extend=False, use_wire=False, use_boundary=True,
                                     use_multi_face=False, use_non_contiguous=False)

    # Fill holes
    try:
        bpy.ops.mesh.fill_holes(sides=FILL_HOLES_MAX_SIDES)
        log(f"[Fill Holes] Filled holes")
    except:
        log(f"[Fill Holes] No holes to fill or fill failed")

    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')


# ============================================================================
# INTERNAL GEOMETRY REMOVAL
# ============================================================================
def remove_internal_simple(obj):
    """Remove internal/hidden faces using Blender's select_interior_faces."""
    ensure_object_mode()

    initial_faces = get_face_count(obj)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bpy.ops.mesh.select_mode(type='FACE')
    bpy.ops.mesh.select_all(action='DESELECT')

    try:
        bpy.ops.mesh.select_interior_faces()
        bpy.ops.mesh.delete(type='FACE')
    except:
        pass

    bpy.ops.object.mode_set(mode='OBJECT')

    removed = initial_faces - get_face_count(obj)
    log(f"[Internal Removal] Removed {removed} interior faces")
    return removed


def remove_internal_raycast(obj):
    """Remove internal faces using ray casting from multiple directions."""
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh_eval = obj_eval.data

    bm = bmesh.new()
    bm.from_mesh(mesh_eval)
    bm.faces.ensure_lookup_table()

    bvh = BVHTree.FromBMesh(bm)

    # Calculate mesh bounds
    min_co = Vector((float('inf'), float('inf'), float('inf')))
    max_co = Vector((float('-inf'), float('-inf'), float('-inf')))

    for v in bm.verts:
        for i in range(3):
            min_co[i] = min(min_co[i], v.co[i])
            max_co[i] = max(max_co[i], v.co[i])

    padding = (max_co - min_co).length * 0.1
    min_co -= Vector((padding, padding, padding))
    max_co += Vector((padding, padding, padding))

    # Ray origins from 6 cardinal directions
    ray_origins = [
        Vector((min_co.x, (min_co.y + max_co.y) / 2, (min_co.z + max_co.z) / 2)),
        Vector((max_co.x, (min_co.y + max_co.y) / 2, (min_co.z + max_co.z) / 2)),
        Vector(((min_co.x + max_co.x) / 2, min_co.y, (min_co.z + max_co.z) / 2)),
        Vector(((min_co.x + max_co.x) / 2, max_co.y, (min_co.z + max_co.z) / 2)),
        Vector(((min_co.x + max_co.x) / 2, (min_co.y + max_co.y) / 2, min_co.z)),
        Vector(((min_co.x + max_co.x) / 2, (min_co.y + max_co.y) / 2, max_co.z)),
    ]

    # Track visible faces
    visible_faces = set()
    RAY_OFFSET = 0.001

    for face in bm.faces:
        face_center = face.calc_center_median()
        face_normal = face.normal

        for origin in ray_origins:
            direction = (face_center - origin).normalized()
            hit, hit_normal, hit_index, hit_dist = bvh.ray_cast(origin, direction)

            if hit is not None and hit_index == face.index:
                visible_faces.add(face.index)
                break

            # Check from face center outward
            offset_origin = face_center + face_normal * RAY_OFFSET
            hit, hit_normal, hit_index, hit_dist = bvh.ray_cast(offset_origin, face_normal)

            if hit is None:
                visible_faces.add(face.index)
                break

    bm.free()

    # Remove internal faces
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    faces_to_delete = [f for f in bm.faces if f.index not in visible_faces]
    internal_count = len(faces_to_delete)

    if faces_to_delete:
        bmesh.ops.delete(bm, geom=faces_to_delete, context='FACES')

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    log(f"[Internal Removal] Removed {internal_count} internal faces (raycast)")
    return internal_count


# ============================================================================
# DECIMATION
# ============================================================================
def apply_planar_decimate(obj):
    """
    Apply planar decimation to merge coplanar faces.

    CRITICAL: Uses delimit options to preserve:
    - SEAM edges (color boundaries marked earlier)
    - SHARP edges (angle boundaries)
    - MATERIAL boundaries
    """
    ensure_object_mode()

    initial_faces = get_face_count(obj)
    angle_rad = math.radians(PLANAR_ANGLE)

    mod = obj.modifiers.new(name="Planar_Decimate", type='DECIMATE')
    mod.decimate_type = 'DISSOLVE'
    mod.angle_limit = angle_rad
    mod.use_dissolve_boundaries = not PRESERVE_BOUNDARIES

    # CRITICAL: Delimit options preserve important edges
    delimit_set = {'NORMAL'}
    if PRESERVE_SEAMS or DETECT_COLOR_EDGES:
        delimit_set.add('SEAM')
    if DETECT_COLOR_EDGES and MARK_COLOR_EDGES_SHARP:
        delimit_set.add('SHARP')
    delimit_set.add('MATERIAL')

    mod.delimit = delimit_set
    log(f"[Planar] Delimit: {delimit_set}")

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)

    final_faces = get_face_count(obj)
    reduction = ((initial_faces - final_faces) / initial_faces * 100) if initial_faces > 0 else 0
    log(f"[Planar] {initial_faces} -> {final_faces} faces ({reduction:.1f}% reduction)")
    return final_faces


def triangulate_ngons(obj):
    """Triangulate n-gons (5+ sided faces) for proper collapse behavior."""
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    ngons = [f for f in bm.faces if len(f.verts) > 4]

    if ngons:
        bmesh.ops.triangulate(bm, faces=ngons, quad_method='BEAUTY', ngon_method='BEAUTY')
        log(f"[Triangulate] Triangulated {len(ngons)} n-gons")

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')


def apply_collapse_decimate(obj, target_faces):
    """Apply collapse decimation to reach target face count."""
    ensure_object_mode()

    current_faces = get_face_count(obj)
    if current_faces <= target_faces:
        log(f"[Collapse] Already at {current_faces} faces (target: {target_faces}), skipping.")
        return current_faces

    ratio = target_faces / current_faces

    mod = obj.modifiers.new(name="Collapse_Decimate", type='DECIMATE')
    mod.decimate_type = 'COLLAPSE'
    mod.ratio = ratio
    mod.use_collapse_triangulate = False  # CRITICAL: Don't fragment geometry

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)

    final_faces = get_face_count(obj)
    log(f"[Collapse] {current_faces} -> {final_faces} faces (target was {target_faces})")
    return final_faces


# ============================================================================
# SHARP EDGE MARKING
# ============================================================================
def mark_sharp_edges(obj):
    """Mark edges as sharp based on face angle threshold."""
    ensure_object_mode()

    angle_rad = math.radians(SHARP_ANGLE)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    sharp_count = 0

    for edge in bm.edges:
        if len(edge.link_faces) != 2:
            continue

        face1, face2 = edge.link_faces[0], edge.link_faces[1]
        n1 = face1.normal
        n2 = face2.normal

        if n1.length < 0.0001 or n2.length < 0.0001:
            continue

        try:
            angle = n1.angle(n2)
        except ValueError:
            continue

        if angle > angle_rad:
            edge.smooth = False
            sharp_count += 1
        else:
            edge.smooth = True

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    log(f"[Sharp Edges] Marked {sharp_count} sharp edges (threshold: {SHARP_ANGLE}deg)")
    return sharp_count


# ============================================================================
# COLOR PRESERVATION (FACE-BASED, NO BLEEDING)
# ============================================================================
def create_color_reference_copy(obj):
    """Create a hidden copy for color transfer after decimation."""
    ensure_object_mode()

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.duplicate()

    ref_obj = bpy.context.active_object
    ref_obj.name = "ColorReference"
    ref_obj.hide_set(True)

    # Switch back to original
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    ref_obj.select_set(False)

    log(f"[Color Ref] Created reference copy with {get_face_count(ref_obj)} faces")
    return ref_obj


def transfer_colors_from_reference(obj, ref_obj):
    """
    Transfer vertex colors from reference mesh using BVH face lookup.
    Face-based transfer = NO color bleeding at vertices.
    """
    ensure_object_mode()

    ref_mesh = ref_obj.data

    # Get source face colors from reference
    source_face_colors = {}

    # Check for vertex_colors (older API)
    if ref_mesh.vertex_colors:
        color_layer = ref_mesh.vertex_colors.active
        for poly in ref_mesh.polygons:
            colors = []
            for loop_idx in poly.loop_indices:
                c = color_layer.data[loop_idx].color
                colors.append((c[0], c[1], c[2], c[3]))
            if colors:
                avg = [sum(x)/len(colors) for x in zip(*colors)]
                source_face_colors[poly.index] = tuple(avg)

    # Check for color_attributes (newer API)
    elif ref_mesh.color_attributes:
        attr = ref_mesh.color_attributes.active_color
        if attr and attr.domain == 'CORNER':
            for poly in ref_mesh.polygons:
                colors = []
                for loop_idx in poly.loop_indices:
                    c = attr.data[loop_idx].color
                    colors.append((c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0))
                if colors:
                    avg = [sum(x)/len(colors) for x in zip(*colors)]
                    source_face_colors[poly.index] = tuple(avg)
        elif attr and attr.domain == 'POINT':
            for poly in ref_mesh.polygons:
                colors = []
                for vert_idx in poly.vertices:
                    c = attr.data[vert_idx].color
                    colors.append((c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0))
                if colors:
                    avg = [sum(x)/len(colors) for x in zip(*colors)]
                    source_face_colors[poly.index] = tuple(avg)

    if not source_face_colors:
        log("[Color Transfer] WARNING: No source colors found")
        return

    log(f"[Color Transfer] Extracted {len(source_face_colors)} face colors from reference")

    # Build BVH from reference
    vertices = [v.co.copy() for v in ref_mesh.vertices]
    polygons = [tuple(p.vertices) for p in ref_mesh.polygons]
    bvh = BVHTree.FromPolygons(vertices, polygons)

    # Transfer to target
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    target_bm = bmesh.from_edit_mesh(obj.data)
    target_bm.faces.ensure_lookup_table()

    color_layer = target_bm.loops.layers.color.get("Col")
    if not color_layer:
        color_layer = target_bm.loops.layers.color.new("Col")

    total_faces = len(target_bm.faces)
    log_interval = max(1, total_faces // 10)

    for i, face in enumerate(target_bm.faces):
        if i % log_interval == 0:
            log(f"[Color Transfer] {i}/{total_faces} ({100*i//total_faces}%)")

        face_center = face.calc_center_median()
        location, normal, face_idx, distance = bvh.find_nearest(face_center)

        if face_idx is not None and face_idx in source_face_colors:
            color = source_face_colors[face_idx]
        else:
            color = (1.0, 0.0, 1.0, 1.0)  # Magenta for missing

        # Apply SAME color to ALL loops (no bleeding)
        for loop in face.loops:
            loop[color_layer] = color

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')
    log(f"[Color Transfer] Transferred colors to {total_faces} faces (face-based, no bleed)")


# ============================================================================
# NORMAL FIXING
# ============================================================================
def fix_normals(obj):
    """Fix face normals to point outward."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    log("[Fix Normals] Normals recalculated")


# ============================================================================
# MAIN PIPELINE
# ============================================================================
log("[BD Decimate V2] Starting full pipeline...")
log(f"[BD Decimate V2] Settings:")
log(f"  Pre-cleanup: {PRE_CLEANUP}")
log(f"  Detect color edges: {DETECT_COLOR_EDGES} (threshold: {COLOR_EDGE_THRESHOLD})")
log(f"  Fill holes: {FILL_HOLES}")
log(f"  Remove internal: {REMOVE_INTERNAL} ({INTERNAL_REMOVAL_METHOD})")
log(f"  Target faces: {TARGET_FACES}")
log(f"  Planar angle: {PLANAR_ANGLE}deg")
log(f"  Sharp angle: {SHARP_ANGLE}deg")
log(f"  Preserve colors: {PRESERVE_COLORS}")
log(f"  Fix normals: {FIX_NORMALS}")

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh
ext = os.path.splitext(INPUT_PATH)[1].lower()
log(f"[BD Decimate V2] Importing {ext} from {INPUT_PATH}")

# Check file exists and has content
if not os.path.exists(INPUT_PATH):
    raise ValueError(f"Input file not found: {INPUT_PATH}")
file_size = os.path.getsize(INPUT_PATH)
log(f"[BD Decimate V2] Input file size: {file_size} bytes")
if file_size == 0:
    raise ValueError(f"Input file is empty: {INPUT_PATH}")

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

# Get imported object - GLTF can create hierarchies, need to find mesh
log(f"[BD Decimate V2] Scene objects after import: {[o.name for o in bpy.context.scene.objects]}")

obj = bpy.context.active_object
if obj is None or obj.type != 'MESH':
    # Find first mesh object
    mesh_objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    log(f"[BD Decimate V2] Found {len(mesh_objects)} mesh objects")
    if not mesh_objects:
        raise ValueError("No mesh objects found after import!")
    obj = mesh_objects[0]

bpy.context.view_layer.objects.active = obj
obj.select_set(True)
log(f"[BD Decimate V2] Selected object: {obj.name}")

original_faces = get_face_count(obj)
original_verts = get_vertex_count(obj)
log(f"[BD Decimate V2] Input: {original_verts} verts, {original_faces} faces")

# Check for colors
has_colors = False
if obj.data.vertex_colors:
    has_colors = True
    log(f"[BD Decimate V2] Found vertex_colors: {len(obj.data.vertex_colors)} layers")
if obj.data.color_attributes:
    has_colors = True
    log(f"[BD Decimate V2] Found color_attributes: {len(obj.data.color_attributes)} layers")
    for attr in obj.data.color_attributes:
        log(f"  - {attr.name}: domain={attr.domain}, type={attr.data_type}")

# Create color reference for later transfer
ref_obj = None
if PRESERVE_COLORS and has_colors:
    ref_obj = create_color_reference_copy(obj)

# STEP 1: Pre-cleanup
if PRE_CLEANUP:
    log("\\n[STEP 1] Pre-cleanup...")
    pre_cleanup_mesh(obj)

# STEP 2: Detect color edges (CRITICAL for edge preservation)
if DETECT_COLOR_EDGES and has_colors:
    log("\\n[STEP 2] Detecting color edges...")
    detect_color_edges_from_vertex_colors(obj)

# STEP 3: Fill holes
if FILL_HOLES:
    log("\\n[STEP 3] Filling holes...")
    fill_holes_simple(obj)

# STEP 4: Remove internal geometry
if REMOVE_INTERNAL:
    log("\\n[STEP 4] Removing internal geometry...")
    if INTERNAL_REMOVAL_METHOD == 'RAYCAST':
        remove_internal_raycast(obj)
    else:
        remove_internal_simple(obj)

# STEP 5: Planar decimate (merges coplanar faces, preserves marked edges)
if PLANAR_ANGLE > 0:
    log("\\n[STEP 5] Planar decimate...")
    apply_planar_decimate(obj)

# STEP 6: Triangulate n-gons before collapse
log("\\n[STEP 6] Triangulating n-gons...")
triangulate_ngons(obj)

# STEP 7: Collapse decimate
if TARGET_FACES > 0:
    log("\\n[STEP 7] Collapse decimate...")
    apply_collapse_decimate(obj, TARGET_FACES)

# STEP 8: Mark sharp edges by angle
if SHARP_ANGLE > 0:
    log("\\n[STEP 8] Marking sharp edges...")
    mark_sharp_edges(obj)

# STEP 9: Transfer colors back from reference
if ref_obj and PRESERVE_COLORS:
    log("\\n[STEP 9] Transferring colors from reference...")
    transfer_colors_from_reference(obj, ref_obj)
    # Clean up reference
    bpy.data.objects.remove(ref_obj, do_unlink=True)

# STEP 10: Fix normals
if FIX_NORMALS:
    log("\\n[STEP 10] Fixing normals...")
    fix_normals(obj)

# Final stats
final_faces = get_face_count(obj)
final_verts = get_vertex_count(obj)
reduction = (1 - final_faces / original_faces) * 100 if original_faces > 0 else 0
log(f"\\n[BD Decimate V2] COMPLETE: {original_faces} -> {final_faces} faces ({reduction:.1f}% reduction)")
log(f"[BD Decimate V2] Final: {final_verts} verts, {final_faces} faces")

# Debug output
if DEBUG_PATH:
    log(f"[BD Decimate V2] Saving debug copy to: {DEBUG_PATH}")
    try:
        bpy.ops.export_scene.gltf(
            filepath=DEBUG_PATH,
            export_format='GLB',
            export_attributes=True,  # Include vertex colors/attributes
            export_yup=True,  # GLTF standard Y-up (trimesh expects this)
        )
        log(f"[BD Decimate V2] Debug copy saved")
    except Exception as e:
        log(f"[BD Decimate V2] Warning: Could not save debug: {e}")

# Export result
ext_out = os.path.splitext(OUTPUT_PATH)[1].lower()
log(f"[BD Decimate V2] Exporting to {ext_out}...")
if ext_out == '.ply':
    bpy.ops.wm.ply_export(filepath=OUTPUT_PATH, export_colors='SRGB', ascii_format=False)
elif ext_out == '.obj':
    bpy.ops.wm.obj_export(filepath=OUTPUT_PATH)
elif ext_out in ['.glb', '.gltf']:
    bpy.ops.export_scene.gltf(
        filepath=OUTPUT_PATH,
        export_format='GLB',
        export_attributes=True,  # Include vertex colors/attributes
        export_yup=True,  # GLTF standard Y-up (trimesh expects this)
    )
elif ext_out == '.stl':
    bpy.ops.wm.stl_export(filepath=OUTPUT_PATH)

log(f"[BD Decimate V2] Saved to {OUTPUT_PATH}")
'''


# ============================================================================
# COMFYUI NODE
# ============================================================================
class BD_BlenderDecimateV2(BlenderNodeMixin, io.ComfyNode):
    """
    Full-featured stylized low-poly decimation using Blender.

    Complete port of Decimate_v1.py with all features:
    - Pre-cleanup (fix non-manifold geometry)
    - Color edge detection (marks color boundaries as sharp/seams - CRITICAL)
    - Hole filling
    - Internal geometry removal
    - Planar decimation with edge preservation
    - Collapse decimation
    - Face-based color transfer (no bleeding)
    - Sharp edge marking
    - Normal fixing
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_BlenderDecimateV2",
            display_name="BD Blender Decimate V2 (Full)",
            category="BrainDead/Blender",
            description="""Full-featured stylized low-poly decimation.

CRITICAL FEATURES:
- Color Edge Detection: Marks color boundaries as sharp/seam edges
- Planar Decimate with Delimit: Preserves marked edges during decimation
- Face-Based Color Transfer: NO color bleeding at vertices

Complete pipeline from Decimate_v1.py script.""",
            inputs=[
                TrimeshInput("mesh"),

                # Target
                io.Int.Input(
                    "target_faces",
                    default=5000,
                    min=100,
                    max=500000,
                    tooltip="Target face count after decimation",
                ),

                # Pre-cleanup
                io.Boolean.Input(
                    "pre_cleanup",
                    default=True,
                    tooltip="Fix non-manifold geometry before decimation",
                ),

                # Color edge detection (CRITICAL)
                io.Boolean.Input(
                    "detect_color_edges",
                    default=True,
                    tooltip="CRITICAL: Detect color boundaries and mark as sharp/seam edges for preservation",
                ),
                io.Float.Input(
                    "color_edge_threshold",
                    default=0.15,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    tooltip="Color difference threshold for edge detection (lower = more edges)",
                ),

                # Hole filling
                io.Boolean.Input(
                    "fill_holes",
                    default=True,
                    tooltip="Fill holes in mesh before decimation",
                ),

                # Internal geometry removal
                io.Boolean.Input(
                    "remove_internal",
                    default=False,
                    tooltip="Remove internal/hidden faces (jacketing cleanup)",
                ),
                io.Combo.Input(
                    "internal_method",
                    options=["SIMPLE", "RAYCAST"],
                    default="SIMPLE",
                    tooltip="SIMPLE=fast, RAYCAST=more accurate but slower",
                ),

                # Decimation settings
                io.Float.Input(
                    "planar_angle",
                    default=7.0,
                    min=0.0,
                    max=45.0,
                    step=0.5,
                    tooltip="Angle threshold for planar decimation (merge coplanar faces)",
                ),
                io.Float.Input(
                    "sharp_angle",
                    default=14.0,
                    min=0.0,
                    max=90.0,
                    step=1.0,
                    tooltip="Angle threshold for marking sharp edges after decimation",
                ),

                # Preservation options
                io.Boolean.Input(
                    "preserve_boundaries",
                    default=True,
                    tooltip="Preserve mesh boundaries during planar decimation",
                ),
                io.Boolean.Input(
                    "preserve_colors",
                    default=True,
                    tooltip="Transfer vertex colors with face-based lookup (no bleeding)",
                ),
                io.Boolean.Input(
                    "fix_normals",
                    default=True,
                    tooltip="Recalculate normals to face outward",
                ),

                # Timeout and debug
                io.Int.Input(
                    "timeout",
                    default=600,
                    min=60,
                    max=3600,
                    optional=True,
                    tooltip="Maximum processing time in seconds",
                ),
                io.String.Input(
                    "debug_path",
                    default="",
                    optional=True,
                    tooltip="Optional: Full path to save debug GLB (e.g., /tmp/decimate_debug.glb)",
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
        target_faces: int,
        pre_cleanup: bool = True,
        detect_color_edges: bool = True,
        color_edge_threshold: float = 0.15,
        fill_holes: bool = True,
        remove_internal: bool = False,
        internal_method: str = "SIMPLE",
        planar_angle: float = 7.0,
        sharp_angle: float = 14.0,
        preserve_boundaries: bool = True,
        preserve_colors: bool = True,
        fix_normals: bool = True,
        timeout: int = 600,
        debug_path: str = "",
    ) -> io.NodeOutput:
        # Check dependencies
        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "ERROR: trimesh not installed")

        available, msg = cls._check_blender()
        if not available:
            return io.NodeOutput(mesh, f"ERROR: {msg}")

        if mesh is None:
            return io.NodeOutput(None, "ERROR: No input mesh")

        # Get original stats
        orig_verts = len(mesh.vertices) if hasattr(mesh, 'vertices') else 0
        orig_faces = len(mesh.faces) if hasattr(mesh, 'faces') else 0

        # Save input mesh to temp file
        input_path = None
        output_path = None
        try:
            input_path = cls._mesh_to_temp_file(mesh, suffix='.glb')
            fd, output_path = tempfile.mkstemp(suffix='.glb')
            os.close(fd)

            print(f"[BD Decimate V2] Input mesh: {orig_verts:,} verts, {orig_faces:,} faces")
            print(f"[BD Decimate V2] Target: {target_faces} faces")
            print(f"[BD Decimate V2] Color edge detection: {detect_color_edges} (threshold: {color_edge_threshold})")

            # Build extra args
            extra_args = {
                'target_faces': target_faces,
                'pre_cleanup': pre_cleanup,
                'fix_non_manifold': pre_cleanup,
                'merge_distance': 0.0001,
                'detect_color_edges': detect_color_edges,
                'color_edge_threshold': color_edge_threshold,
                'mark_color_edges_sharp': True,
                'mark_color_edges_seam': True,
                'fill_holes': fill_holes,
                'fill_holes_max_sides': 100,
                'remove_internal': remove_internal,
                'internal_removal_method': internal_method,
                'planar_angle': planar_angle,
                'sharp_angle': sharp_angle,
                'preserve_boundaries': preserve_boundaries,
                'preserve_seams': True,
                'preserve_colors': preserve_colors,
                'fix_normals': fix_normals,
            }

            if debug_path:
                extra_args['debug_path'] = debug_path
                print(f"[BD Decimate V2] Debug output: {debug_path}")

            # Run Blender
            success, message, log_lines = cls._run_blender_script(
                FULL_DECIMATE_SCRIPT,
                input_path,
                output_path,
                extra_args=extra_args,
                timeout=timeout,
            )

            if not success:
                error_context = '\n'.join(log_lines[-10:]) if log_lines else ''
                print(f"[BD Decimate V2] FAILED: {message}")
                if error_context:
                    print(f"[BD Decimate V2] Log tail:\n{error_context}")
                return io.NodeOutput(mesh, f"ERROR: {message}")

            # Load result
            result_mesh = cls._load_mesh_from_file(output_path)

            # Stats
            new_verts = len(result_mesh.vertices)
            new_faces = len(result_mesh.faces)
            reduction = (1 - new_faces / orig_faces) * 100 if orig_faces > 0 else 0

            status = f"V2 Decimated: {orig_faces:,} -> {new_faces:,} faces ({reduction:.1f}% reduction)"
            if detect_color_edges:
                status += " | color edges detected"
            if preserve_colors:
                status += " | colors preserved"

            return io.NodeOutput(result_mesh, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return io.NodeOutput(mesh, f"ERROR: {e}")

        finally:
            # Cleanup temp files
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if output_path and os.path.exists(output_path):
                os.remove(output_path)


# V3 node list
DECIMATE_FULL_V3_NODES = [BD_BlenderDecimateV2]

# V1 compatibility
DECIMATE_FULL_NODES = {
    "BD_BlenderDecimateV2": BD_BlenderDecimateV2,
}

DECIMATE_FULL_DISPLAY_NAMES = {
    "BD_BlenderDecimateV2": "BD Blender Decimate V2 (Full)",
}
