"""
BD_BlenderExportMesh - Export MESH_BUNDLE as GLB with material + vertex colors.

Uses Blender to combine PBR material and COLOR_0 vertex colors in one GLB file.
This is the only reliable way to get both attributes in a single export,
since trimesh can only have one visual type at a time.
"""

import os
import tempfile
import pickle

import numpy as np

from comfy_api.latest import io

from ..mesh.types import MeshBundleInput, TrimeshOutput

from .base import BlenderNodeMixin, HAS_TRIMESH

if HAS_TRIMESH:
    import trimesh


# Blender script for combined material + vertex color export
EXPORT_MESH_SCRIPT = '''
import bpy
import bmesh
import sys
import os
import struct

def log(msg):
    print(msg)
    sys.stdout.flush()

# Get paths
input_path = os.environ['BLENDER_INPUT_PATH']
output_path = os.environ['BLENDER_OUTPUT_PATH']
vcol_path = os.environ.get('BLENDER_ARG_VCOL_PATH', '')
solidify_mode = os.environ.get('BLENDER_ARG_SOLIDIFY_MODE', 'NONE')  # NONE, DOMINANT, AVERAGE

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import GLB (preserves PBR material + UVs)
log(f"[BD ExportMesh] Importing GLB: {input_path}")
bpy.ops.import_scene.gltf(filepath=input_path)

# Get mesh object
obj = bpy.context.active_object
if obj is None:
    mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if mesh_objs:
        obj = mesh_objs[0]
    else:
        raise RuntimeError("No mesh object found after import")

bpy.context.view_layer.objects.active = obj
obj.select_set(True)

log(f"[BD ExportMesh] Mesh: {len(obj.data.vertices)} verts, {len(obj.data.polygons)} faces")
log(f"[BD ExportMesh] Materials: {len(obj.data.materials)}")

# Load and apply vertex colors if provided
if vcol_path and os.path.exists(vcol_path):
    import numpy as np
    vcol_data = np.load(vcol_path)  # (N, 4) uint8

    log(f"[BD ExportMesh] Loading vertex colors: {vcol_data.shape}")

    # Create color attribute (CORNER domain for per-face-corner colors)
    if 'Col' in obj.data.color_attributes:
        obj.data.color_attributes.remove(obj.data.color_attributes['Col'])

    color_attr = obj.data.color_attributes.new(
        name='Col',
        type='BYTE_COLOR',
        domain='CORNER',  # Per loop/corner
    )

    # Apply colors - need to map vertex colors to loop colors
    n_verts = len(obj.data.vertices)
    n_loops = len(obj.data.loops)

    if len(vcol_data) == n_verts:
        # Per-vertex colors: expand to per-loop
        vcol_float = vcol_data.astype(np.float32) / 255.0
        loop_colors = np.zeros((n_loops, 4), dtype=np.float32)
        loop_verts = np.zeros(n_loops, dtype=np.int32)
        obj.data.loops.foreach_get('vertex_index', loop_verts)
        loop_colors = vcol_float[loop_verts]

        # Write to color attribute
        flat_colors = loop_colors.flatten()
        color_attr.data.foreach_set('color', flat_colors)
        log(f"[BD ExportMesh] Applied {n_verts} vertex colors to {n_loops} loops")

    elif len(vcol_data) == n_loops:
        # Already per-loop colors
        vcol_float = vcol_data.astype(np.float32) / 255.0
        flat_colors = vcol_float.flatten()
        color_attr.data.foreach_set('color', flat_colors)
        log(f"[BD ExportMesh] Applied {n_loops} per-loop colors directly")

    else:
        log(f"[BD ExportMesh] WARNING: vcol count ({len(vcol_data)}) != verts ({n_verts}) or loops ({n_loops})")

    # Solidify mode: make all loops in each face the same color
    if solidify_mode == 'DOMINANT':
        log("[BD ExportMesh] Solidifying colors: DOMINANT (most common per face)")
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        color_layer = bm.loops.layers.color.get('Col')
        if color_layer:
            for face in bm.faces:
                # Get all loop colors for this face
                face_colors = [tuple(loop[color_layer]) for loop in face.loops]
                # Find most common (dominant) color
                from collections import Counter
                dominant = Counter(face_colors).most_common(1)[0][0]
                # Apply dominant to all loops
                for loop in face.loops:
                    loop[color_layer] = dominant

            bm.to_mesh(obj.data)
        bm.free()
        log("[BD ExportMesh] Solidified to dominant corner color per face")

    elif solidify_mode == 'AVERAGE':
        log("[BD ExportMesh] Solidifying colors: AVERAGE per face")
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        color_layer = bm.loops.layers.color.get('Col')
        if color_layer:
            for face in bm.faces:
                avg = [0.0, 0.0, 0.0, 0.0]
                for loop in face.loops:
                    c = loop[color_layer]
                    avg[0] += c[0]; avg[1] += c[1]; avg[2] += c[2]; avg[3] += c[3]
                n = len(face.loops)
                avg = [v / n for v in avg]
                for loop in face.loops:
                    loop[color_layer] = avg

            bm.to_mesh(obj.data)
        bm.free()
        log("[BD ExportMesh] Solidified to average color per face")

    # Set as active render color
    obj.data.color_attributes.active_color = color_attr
    obj.data.color_attributes.render_color_index = obj.data.color_attributes.find('Col')

else:
    log("[BD ExportMesh] No vertex colors provided - exporting material only")

# Export as GLB with materials + vertex colors
log(f"[BD ExportMesh] Exporting GLB: {output_path}")

export_kwargs = {
    'filepath': output_path,
    'export_format': 'GLB',
    'export_attributes': True,
    'export_yup': True,
}

# Try Blender 5.0+ export params
export_attempts = [
    {'export_vertex_color': 'ACTIVE'},
    {'export_vertex_color': 'MATERIAL'},
    {},
]

exported = False
for attempt in export_attempts:
    try:
        bpy.ops.export_scene.gltf(**export_kwargs, **attempt)
        log(f"[BD ExportMesh] Export success with: {attempt or 'defaults'}")
        exported = True
        break
    except TypeError as e:
        continue

if not exported:
    raise RuntimeError("All GLB export attempts failed")

# Report file size
size_mb = os.path.getsize(output_path) / (1024 * 1024)
log(f"[BD ExportMesh] Output: {size_mb:.1f} MB")
'''


class BD_BlenderExportMesh(BlenderNodeMixin, io.ComfyNode):
    """
    Export MESH_BUNDLE as GLB with both PBR material and vertex colors.

    Uses Blender for the final GLB write, which is the only reliable way
    to include both material textures AND COLOR_0 vertex colors in one file.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_BlenderExportMesh",
            display_name="BD Blender Export Mesh",
            category="🧠BrainDead/Blender",
            description="""Export MESH_BUNDLE as GLB with material + vertex colors.

Uses Blender to combine PBR material (textures, UVs) and COLOR_0
vertex colors in one GLB file.

Solidify modes:
- NONE: Per-vertex colors as-is
- DOMINANT: Most common color per face (hard-body flat look)
- AVERAGE: Average color per face (smooth flat look)

Inputs a MESH_BUNDLE (from BD_PackBundle or BD_CacheBundle).
Also accepts a raw TRIMESH for simpler workflows.""",
            inputs=[
                MeshBundleInput("bundle", optional=True),
                io.String.Input(
                    "output_dir",
                    default="mesh_export",
                    tooltip="Output directory name (relative to ComfyUI output/)",
                ),
                io.Combo.Input(
                    "solidify_mode",
                    options=["NONE", "DOMINANT", "AVERAGE"],
                    default="DOMINANT",
                    tooltip="Vertex color solidify: DOMINANT=flat chunky look, AVERAGE=smooth per-face",
                ),
                io.Boolean.Input(
                    "export_textures",
                    default=True,
                    optional=True,
                    tooltip="Also export individual texture PNGs alongside GLB",
                ),
                io.Int.Input(
                    "timeout",
                    default=300,
                    min=60,
                    max=1800,
                    optional=True,
                    tooltip="Maximum Blender processing time in seconds",
                ),
            ],
            outputs=[
                io.String.Output(display_name="glb_path"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        bundle=None,
        output_dir: str = "mesh_export",
        solidify_mode: str = "DOMINANT",
        export_textures: bool = True,
        timeout: int = 300,
    ) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput("", "ERROR: trimesh not installed")

        available, msg = cls._check_blender()
        if not available:
            return io.NodeOutput("", f"ERROR: {msg}")

        if bundle is None:
            return io.NodeOutput("", "ERROR: No bundle input")

        # Extract bundle data
        mesh = bundle.get('mesh') if isinstance(bundle, dict) else bundle
        if mesh is None:
            return io.NodeOutput("", "ERROR: Bundle has no mesh")

        vertex_colors = bundle.get('vertex_colors') if isinstance(bundle, dict) else None
        name = bundle.get('name', 'mesh') if isinstance(bundle, dict) else 'mesh'

        # Set up output directory
        import folder_paths
        output_base = folder_paths.get_output_directory()
        out_dir = os.path.join(output_base, output_dir)
        os.makedirs(out_dir, exist_ok=True)

        glb_output = os.path.join(out_dir, f"{name}.glb")

        # Temp files
        input_path = None
        vcol_path = None
        blender_output = None

        try:
            # Save mesh as temp GLB (with material)
            input_path = cls._mesh_to_temp_file(mesh, suffix='.glb')

            # Save vertex colors as numpy file
            if vertex_colors is not None and len(vertex_colors) > 0:
                fd, vcol_path = tempfile.mkstemp(suffix='.npy')
                os.close(fd)
                np.save(vcol_path, vertex_colors)
            else:
                vcol_path = ''

            # Output goes directly to final location
            fd, blender_output = tempfile.mkstemp(suffix='.glb')
            os.close(fd)

            # Run Blender export script
            extra_args = {
                'vcol_path': vcol_path or '',
                'solidify_mode': solidify_mode,
            }

            success, message, log_lines = cls._run_blender_script(
                EXPORT_MESH_SCRIPT,
                input_path,
                blender_output,
                extra_args=extra_args,
                timeout=timeout,
            )

            if not success:
                return io.NodeOutput("", f"ERROR: {message}")

            # Move to final output location
            if os.path.exists(blender_output) and os.path.getsize(blender_output) > 100:
                import shutil
                shutil.move(blender_output, glb_output)
                blender_output = None  # Don't clean up - it's moved
            else:
                return io.NodeOutput("", "ERROR: Blender produced empty output")

            # Export individual textures if requested
            tex_count = 0
            if export_textures and isinstance(bundle, dict):
                from PIL import Image
                for tex_name in ['diffuse', 'normal', 'metallic', 'roughness', 'alpha']:
                    tex_data = bundle.get(tex_name)
                    if tex_data is not None:
                        tex_path = os.path.join(out_dir, f"{name}_{tex_name}.png")
                        Image.fromarray(tex_data).save(tex_path)
                        tex_count += 1

            # Status
            size_mb = os.path.getsize(glb_output) / (1024 * 1024)
            n_verts = len(mesh.vertices)
            parts = [f"{name}.glb: {size_mb:.1f}MB", f"{n_verts:,} verts"]
            if vertex_colors is not None:
                parts.append(f"+COLOR_0 ({solidify_mode})")
            if tex_count > 0:
                parts.append(f"+{tex_count} textures")

            status = " | ".join(parts)
            print(f"[BD BlenderExportMesh] {status}")
            return io.NodeOutput(glb_output, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return io.NodeOutput("", f"ERROR: {e}")

        finally:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if vcol_path and os.path.exists(vcol_path):
                os.remove(vcol_path)
            if blender_output and os.path.exists(blender_output):
                os.remove(blender_output)


# V3 node list
EXPORT_MESH_V3_NODES = [BD_BlenderExportMesh]

# V1 compatibility
EXPORT_MESH_NODES = {
    "BD_BlenderExportMesh": BD_BlenderExportMesh,
}

EXPORT_MESH_DISPLAY_NAMES = {
    "BD_BlenderExportMesh": "BD Blender Export Mesh",
}
