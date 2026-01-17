"""
BD_BlenderDecimate - Stylized low-poly mesh decimation using Blender.

Based on Decimate_v1.py with key features:
- Planar decimation (merge coplanar faces for flat-shaded look)
- Collapse decimation with edge preservation
- Hole filling option
- Internal geometry removal option
- Face-based vertex color preservation (no color bleeding)
"""

import os
import tempfile

from comfy_api.latest import io

# Import custom TRIMESH type (matches TRELLIS2)
from ..mesh.types import TrimeshInput, TrimeshOutput

from .base import BlenderNodeMixin, HAS_TRIMESH

if HAS_TRIMESH:
    import trimesh


# Advanced Blender decimate script with color preservation and multiple modes
ADVANCED_DECIMATE_SCRIPT = '''
import bpy
import bmesh
import os
import sys
from mathutils.bvhtree import BVHTree

def log(msg):
    """Print with immediate flush for real-time output."""
    print(msg)
    sys.stdout.flush()

log("[BD Decimate] Script starting...")

# Get paths from environment
input_path = os.environ['BLENDER_INPUT_PATH']
output_path = os.environ['BLENDER_OUTPUT_PATH']

# Get parameters
ratio = float(os.environ.get('BLENDER_ARG_RATIO', '0.5'))
target_faces = int(os.environ.get('BLENDER_ARG_TARGET_FACES', '0'))
planar_angle = float(os.environ.get('BLENDER_ARG_PLANAR_ANGLE', '5.0'))
fill_holes = os.environ.get('BLENDER_ARG_FILL_HOLES', 'True') == 'True'
remove_internal = os.environ.get('BLENDER_ARG_REMOVE_INTERNAL', 'False') == 'True'
preserve_boundaries = os.environ.get('BLENDER_ARG_PRESERVE_BOUNDARIES', 'True') == 'True'
preserve_colors = os.environ.get('BLENDER_ARG_PRESERVE_COLORS', 'True') == 'True'
use_planar = os.environ.get('BLENDER_ARG_USE_PLANAR', 'True') == 'True'

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh
ext = os.path.splitext(input_path)[1].lower()
if ext == '.ply':
    bpy.ops.wm.ply_import(filepath=input_path)
elif ext == '.obj':
    bpy.ops.wm.obj_import(filepath=input_path)
elif ext in ['.glb', '.gltf']:
    bpy.ops.import_scene.gltf(filepath=input_path)
elif ext == '.stl':
    bpy.ops.wm.stl_import(filepath=input_path)
else:
    raise ValueError(f"Unsupported format: {ext}")

# Get imported object
obj = bpy.context.active_object
if obj is None:
    obj = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

original_faces = len(obj.data.polygons)
log(f"[BD Decimate] Input: {original_faces} faces")

# Create reference copy for color transfer if we have colors
ref_obj = None
has_colors = obj.data.vertex_colors or obj.data.color_attributes
if preserve_colors and has_colors:
    # Duplicate for color reference
    bpy.ops.object.duplicate()
    ref_obj = bpy.context.active_object
    ref_obj.name = "ColorReference"
    ref_obj.hide_set(True)

    # Switch back to original
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    ref_obj.select_set(False)
    log(f"[BD Decimate] Created color reference copy")

# Step 1: Fill holes if requested
if fill_holes:
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold(extend=False)
    try:
        bpy.ops.mesh.fill_holes(sides=100)
        log(f"[BD Decimate] Filled holes")
    except:
        pass
    bpy.ops.object.mode_set(mode='OBJECT')

# Step 2: Remove internal geometry (hidden faces) if requested
if remove_internal:
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    try:
        bpy.ops.mesh.select_interior_faces()
        bpy.ops.mesh.delete(type='FACE')
        log(f"[BD Decimate] Removed internal faces")
    except:
        pass
    bpy.ops.object.mode_set(mode='OBJECT')

# Step 3: Planar decimation - merge coplanar faces
if use_planar and planar_angle > 0:
    modifier = obj.modifiers.new(name='PlanarDecimate', type='DECIMATE')
    modifier.decimate_type = 'DISSOLVE'
    modifier.angle_limit = planar_angle * 3.14159 / 180  # Convert to radians
    modifier.use_dissolve_boundaries = not preserve_boundaries
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    log(f"[BD Decimate] Planar dissolve at {planar_angle}° - now {len(obj.data.polygons)} faces")

# Step 4: Calculate target ratio based on target_faces if specified
current_faces = len(obj.data.polygons)
if target_faces > 0 and current_faces > target_faces:
    ratio = target_faces / current_faces
    ratio = max(0.01, min(1.0, ratio))
    log(f"[BD Decimate] Target {target_faces} faces, using ratio {ratio:.3f}")

# Step 5: Collapse decimation
if ratio < 1.0:
    modifier = obj.modifiers.new(name='CollapseDecimate', type='DECIMATE')
    modifier.decimate_type = 'COLLAPSE'
    modifier.ratio = ratio
    modifier.use_collapse_triangulate = True
    modifier.use_symmetry = False
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    log(f"[BD Decimate] Collapse decimate - now {len(obj.data.polygons)} faces")

# Step 6: Transfer colors back from reference (face-based, no bleeding)
if preserve_colors and ref_obj and has_colors:
    ref_mesh = ref_obj.data
    log(f"[BD Decimate] Extracting colors from {len(ref_mesh.polygons)} source faces...")

    # Get source face colors
    source_face_colors = {}
    if ref_mesh.vertex_colors:
        log(f"[BD Decimate] Using vertex_colors layer")
        color_layer = ref_mesh.vertex_colors.active
        for poly in ref_mesh.polygons:
            # Average color of face loops
            colors = []
            for loop_idx in poly.loop_indices:
                c = color_layer.data[loop_idx].color
                colors.append((c[0], c[1], c[2], c[3]))
            if colors:
                avg = [sum(x)/len(colors) for x in zip(*colors)]
                source_face_colors[poly.index] = tuple(avg)
    elif ref_mesh.color_attributes:
        attr = ref_mesh.color_attributes.active_color
        log(f"[BD Decimate] Using color_attributes layer: {attr.name if attr else 'None'} (domain: {attr.domain if attr else 'N/A'})")
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
            # Vertex-based colors - need to look up by vertex index
            log(f"[BD Decimate] Converting POINT domain colors to face colors")
            for poly in ref_mesh.polygons:
                colors = []
                for vert_idx in poly.vertices:
                    c = attr.data[vert_idx].color
                    colors.append((c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0))
                if colors:
                    avg = [sum(x)/len(colors) for x in zip(*colors)]
                    source_face_colors[poly.index] = tuple(avg)

    log(f"[BD Decimate] Extracted {len(source_face_colors)} face colors")

    if source_face_colors:
        log(f"[BD Decimate] Building BVH from {len(ref_mesh.polygons)} source faces...")
        # Build BVH from reference
        vertices = [v.co.copy() for v in ref_mesh.vertices]
        polygons = [tuple(p.vertices) for p in ref_mesh.polygons]
        bvh = BVHTree.FromPolygons(vertices, polygons)
        log(f"[BD Decimate] BVH built, starting color transfer...")

        # Prepare target for color transfer
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')

        target_bm = bmesh.from_edit_mesh(obj.data)
        target_bm.faces.ensure_lookup_table()

        # Create color layer
        color_layer = target_bm.loops.layers.color.get("Col")
        if not color_layer:
            color_layer = target_bm.loops.layers.color.new("Col")

        # Transfer colors face-by-face (NO BLEEDING)
        total_faces = len(target_bm.faces)
        log_interval = max(1, total_faces // 10)  # Log every 10%
        for i, face in enumerate(target_bm.faces):
            if i % log_interval == 0:
                log(f"[BD Decimate] Transferring colors: {i}/{total_faces} ({100*i//total_faces}%)")

            face_center = face.calc_center_median()
            location, normal, face_idx, distance = bvh.find_nearest(face_center)

            if face_idx is not None and face_idx in source_face_colors:
                color = source_face_colors[face_idx]
            else:
                color = (1.0, 0.0, 1.0, 1.0)  # Magenta for missing

            # Apply SAME color to ALL loops (no bleeding at vertices)
            for loop in face.loops:
                loop[color_layer] = color

        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')
        log(f"[BD Decimate] Transferred colors to {total_faces} faces (face-based, no bleed)")

    # Clean up reference
    bpy.data.objects.remove(ref_obj, do_unlink=True)

# Final stats
final_faces = len(obj.data.polygons)
reduction = (1 - final_faces / original_faces) * 100 if original_faces > 0 else 0
log(f"[BD Decimate] Final: {final_faces} faces ({reduction:.1f}% reduction)")

# Export result
ext_out = os.path.splitext(output_path)[1].lower()
if ext_out == '.ply':
    bpy.ops.wm.ply_export(filepath=output_path, export_colors='SRGB')
elif ext_out == '.obj':
    bpy.ops.wm.obj_export(filepath=output_path)
elif ext_out in ['.glb', '.gltf']:
    bpy.ops.export_scene.gltf(filepath=output_path)
elif ext_out == '.stl':
    bpy.ops.wm.stl_export(filepath=output_path)

log(f"[BD Decimate] Saved to {output_path}")
'''


class BD_BlenderDecimate(BlenderNodeMixin, io.ComfyNode):
    """
    Decimate (simplify) a mesh using Blender with stylized low-poly features.

    Based on Decimate_v1.py with key features:
    - Planar decimation for flat-shaded aesthetic
    - Face-based vertex colors (NO color bleeding)
    - Optional hole filling
    - Optional internal geometry removal
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_BlenderDecimate",
            display_name="BD Blender Decimate",
            category="🧠BrainDead/Blender",
            description="""Stylized low-poly decimation using Blender.

Features:
- Planar decimation: Merges coplanar faces for flat-shaded look
- Collapse decimation: Reduces to target face count
- Face-based color transfer: NO color bleeding at edges
- Hole filling: Close open edges before decimation
- Internal removal: Remove hidden faces (jacketing cleanup)

Best for: Stylized characters, props, game assets with vertex colors.""",
            inputs=[
                TrimeshInput("mesh"),
                io.Int.Input(
                    "target_faces",
                    default=5000,
                    min=100,
                    max=500000,
                    tooltip="Target face count (0 = use ratio instead)",
                ),
                io.Float.Input(
                    "ratio",
                    default=0.5,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip="Fallback ratio if target_faces is 0 (0.5 = 50% reduction)",
                ),
                io.Float.Input(
                    "planar_angle",
                    default=5.0,
                    min=0.0,
                    max=45.0,
                    step=0.5,
                    optional=True,
                    tooltip="Angle threshold for planar dissolve (degrees). Faces within this angle are merged. 0 = skip planar.",
                ),
                io.Boolean.Input(
                    "fill_holes",
                    default=True,
                    optional=True,
                    tooltip="Fill holes in mesh before decimation",
                ),
                io.Boolean.Input(
                    "remove_internal",
                    default=False,
                    optional=True,
                    tooltip="Remove internal/hidden faces (jacketing cleanup)",
                ),
                io.Boolean.Input(
                    "preserve_boundaries",
                    default=True,
                    optional=True,
                    tooltip="Preserve mesh boundaries during planar dissolve",
                ),
                io.Boolean.Input(
                    "preserve_colors",
                    default=True,
                    optional=True,
                    tooltip="Transfer vertex colors with face-based lookup (no bleeding)",
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
        target_faces: int,
        ratio: float = 0.5,
        planar_angle: float = 5.0,
        fill_holes: bool = True,
        remove_internal: bool = False,
        preserve_boundaries: bool = True,
        preserve_colors: bool = True,
        timeout: int = 300,
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
            input_path = cls._mesh_to_temp_file(mesh, suffix='.ply')
            fd, output_path = tempfile.mkstemp(suffix='.ply')
            os.close(fd)

            print(f"[BD Decimate] Input mesh: {orig_verts:,} verts, {orig_faces:,} faces")
            print(f"[BD Decimate] Target: {target_faces} faces, planar_angle={planar_angle}°")

            # Run Blender decimate
            success, message, log_lines = cls._run_blender_script(
                ADVANCED_DECIMATE_SCRIPT,
                input_path,
                output_path,
                extra_args={
                    'ratio': ratio,
                    'target_faces': target_faces,
                    'planar_angle': planar_angle,
                    'fill_holes': fill_holes,
                    'remove_internal': remove_internal,
                    'preserve_boundaries': preserve_boundaries,
                    'preserve_colors': preserve_colors,
                    'use_planar': planar_angle > 0,
                },
                timeout=timeout,
            )

            # Print Blender log summary
            if log_lines:
                bd_lines = [l for l in log_lines if l.startswith('[BD')]
                for line in bd_lines[-5:]:  # Last 5 status lines
                    print(line)

            if not success:
                # Include relevant log lines in error
                error_context = '\n'.join(log_lines[-10:]) if log_lines else ''
                print(f"[BD Decimate] FAILED: {message}")
                if error_context:
                    print(f"[BD Decimate] Log tail:\n{error_context}")
                return io.NodeOutput(mesh, f"ERROR: {message}")

            # Load result
            result_mesh = cls._load_mesh_from_file(output_path)

            # Stats
            new_verts = len(result_mesh.vertices)
            new_faces = len(result_mesh.faces)
            reduction = (1 - new_faces / orig_faces) * 100 if orig_faces > 0 else 0

            status = f"Decimated: {orig_faces:,} → {new_faces:,} faces ({reduction:.1f}% reduction)"
            if planar_angle > 0:
                status += f" | planar={planar_angle}°"
            if preserve_colors:
                status += " | colors=face-based"

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
DECIMATE_V3_NODES = [BD_BlenderDecimate]

# V1 compatibility
DECIMATE_NODES = {
    "BD_BlenderDecimate": BD_BlenderDecimate,
}

DECIMATE_DISPLAY_NAMES = {
    "BD_BlenderDecimate": "BD Blender Decimate",
}
