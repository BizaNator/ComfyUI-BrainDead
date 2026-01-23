"""
BD_BlenderExportMesh - Export MESH_BUNDLE as GLB with material + vertex colors.

Uses Blender to combine PBR material and COLOR_0 vertex colors in one GLB file.
This is the only reliable way to get both attributes in a single export,
since trimesh can only have one visual type at a time.
"""

import os
import tempfile

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
color_field_path = os.environ.get('BLENDER_ARG_COLOR_FIELD_PATH', '')
solidify_mode = os.environ.get('BLENDER_ARG_SOLIDIFY_MODE', 'NONE')  # NONE, DOMINANT, AVERAGE
flat_shading = os.environ.get('BLENDER_ARG_FLAT_SHADING', 'False') == 'True'

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import GLB (preserves PBR material + UVs)
log(f"[BD ExportMesh] Importing GLB: {input_path}")
bpy.ops.import_scene.gltf(filepath=input_path)

# Get mesh object - Blender 5.0+ may not set active_object after GLB import
obj = bpy.context.active_object
if obj is None or obj.type != 'MESH' or obj.data is None:
    mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH' and o.data is not None]
    if mesh_objs:
        # Pick the largest mesh if multiple
        obj = max(mesh_objs, key=lambda o: len(o.data.vertices))
    else:
        raise RuntimeError("No mesh object found after import")

bpy.context.view_layer.objects.active = obj
obj.select_set(True)

n_verts = len(obj.data.vertices)
n_loops = len(obj.data.loops)
n_faces = len(obj.data.polygons)
log(f"[BD ExportMesh] Mesh: {n_verts} verts, {n_faces} faces, {n_loops} loops")
log(f"[BD ExportMesh] Materials: {len(obj.data.materials)}")

# Determine color source: pre-computed vertex colors (.npy) or color_field (.npz)
import numpy as np
has_colors = False

if color_field_path and os.path.exists(color_field_path):
    # Color field mode: sample colors at actual Blender vertex positions
    # This avoids vertex-index mismatch from GLB roundtrip
    log(f"[BD ExportMesh] Loading color_field: {color_field_path}")
    cf_data = np.load(color_field_path)
    cf_positions = cf_data['positions']  # (M, 3) float32
    cf_colors = cf_data['colors']  # (M, 3 or 4) float32
    cf_voxel_size = float(cf_data['voxel_size'].flat[0])

    log(f"[BD ExportMesh] Color field: {len(cf_positions)} voxels, voxel_size={cf_voxel_size:.6f}")
    log(f"[BD ExportMesh] Color field bounds: pos=[{cf_positions.min():.4f}, {cf_positions.max():.4f}], col=[{cf_colors.min():.4f}, {cf_colors.max():.4f}]")

    # Get actual vertex positions from the imported mesh
    verts_flat = np.zeros(n_verts * 3, dtype=np.float32)
    obj.data.vertices.foreach_get('co', verts_flat)
    blender_verts = verts_flat.reshape(-1, 3)

    # Transform Blender vertex positions to voxel space
    # The GLB roundtrip already applies the Y/Z swap:
    #   trimesh (Z-up) → GLB (as-is) → Blender import (Y-up → Z-up swap)
    # So Blender verts are already in the same axis orientation as voxel space.
    # We only need the +0.5 offset to align with voxel centers.
    verts_voxel = blender_verts + 0.5

    log(f"[BD ExportMesh] Blender verts range: [{blender_verts.min():.4f}, {blender_verts.max():.4f}]")
    log(f"[BD ExportMesh] Voxel-space verts range: [{verts_voxel.min():.4f}, {verts_voxel.max():.4f}]")

    # Spatial query: find nearest color for each vertex
    max_dist = cf_voxel_size * 3.0
    try:
        from scipy.spatial import cKDTree
        log("[BD ExportMesh] Using scipy.cKDTree for color sampling")
        tree = cKDTree(cf_positions)
        distances, indices = tree.query(verts_voxel, k=1, workers=-1)
        sampled = cf_colors[indices].copy()
        far_verts_mask = distances > max_dist
    except ImportError:
        # Fallback: numpy grid-based lookup (works since color_field is on regular grid)
        log("[BD ExportMesh] scipy not available, using numpy grid lookup")
        scale = 1.0 / cf_voxel_size
        pos_grid = np.round(cf_positions * scale).astype(np.int64)
        q_grid = np.round(verts_voxel * scale).astype(np.int64)

        min_grid = pos_grid.min(axis=0)
        pos_grid -= min_grid
        q_grid -= min_grid
        # Clip query grid to valid range
        dims = pos_grid.max(axis=0) + 1
        q_grid = np.clip(q_grid, 0, dims - 1)

        # Create flat keys for binary search
        stride_y = int(dims[2])
        stride_x = int(dims[1]) * stride_y
        pos_flat = pos_grid[:,0] * stride_x + pos_grid[:,1] * stride_y + pos_grid[:,2]
        q_flat = q_grid[:,0] * stride_x + q_grid[:,1] * stride_y + q_grid[:,2]

        sort_order = np.argsort(pos_flat)
        sorted_flat = pos_flat[sort_order]

        found = np.searchsorted(sorted_flat, q_flat)
        found = np.clip(found, 0, len(sorted_flat) - 1)

        # Use the nearest sorted entry
        color_idx = sort_order[found]
        sampled = cf_colors[color_idx].copy()

        # Misses: check if query fell outside the occupied grid
        exact = sorted_flat[found] == q_flat
        far_verts_mask = ~exact
        log(f"[BD ExportMesh] Grid lookup: {exact.sum()} exact, {(~exact).sum()} approx")

    far_pct = 100 * far_verts_mask.sum() / n_verts
    if cf_colors.shape[1] >= 4:
        sampled[far_verts_mask] = [0.5, 0.5, 0.5, 1.0]
    else:
        sampled[far_verts_mask] = [0.5, 0.5, 0.5]

    # Add alpha channel if needed
    if sampled.shape[1] == 3:
        alpha = np.ones((len(sampled), 1), dtype=np.float32)
        sampled = np.hstack([sampled, alpha])

    log(f"[BD ExportMesh] Sampled {n_verts} colors: [{sampled[:,:3].min():.4f}, {sampled[:,:3].max():.4f}] ({far_pct:.1f}% beyond threshold)")

    # Create color attribute and apply
    vcol_float = sampled  # Already float32 in [0,1]
    has_colors = True

elif vcol_path and os.path.exists(vcol_path):
    # Pre-computed vertex colors mode (from BD_ApplyColorField)
    vcol_data = np.load(vcol_path)  # (N, 4) uint8
    log(f"[BD ExportMesh] Loading pre-computed vertex colors: {vcol_data.shape}")
    log(f"[BD ExportMesh] Color data range: min={vcol_data.min()}, max={vcol_data.max()}")

    # Convert uint8 [0,255] to float [0,1]
    vcol_float = vcol_data.astype(np.float32) / 255.0
    has_colors = True

if has_colors:
    # Determine domain based on solidify mode
    use_corner = solidify_mode in ('DOMINANT', 'AVERAGE')
    domain = 'CORNER' if use_corner else 'POINT'

    # Remove ALL existing color attributes to avoid conflicts
    while len(obj.data.color_attributes) > 0:
        obj.data.color_attributes.remove(obj.data.color_attributes[0])

    color_attr = obj.data.color_attributes.new(
        name='Color',
        type='FLOAT_COLOR',
        domain=domain,
    )

    log(f"[BD ExportMesh] Color domain: {domain} (solidify={solidify_mode}), type=FLOAT_COLOR")

    if domain == 'POINT':
        # Per-vertex colors
        if len(vcol_float) == n_verts:
            flat_colors = vcol_float.flatten()
            color_attr.data.foreach_set('color', flat_colors)
            log(f"[BD ExportMesh] Applied {n_verts} vertex colors (POINT domain)")
        elif len(vcol_float) == n_loops:
            # Per-loop data but POINT domain - average per vertex
            vert_colors = np.zeros((n_verts, 4), dtype=np.float32)
            vert_counts = np.zeros(n_verts, dtype=np.int32)
            loop_verts = np.zeros(n_loops, dtype=np.int32)
            obj.data.loops.foreach_get('vertex_index', loop_verts)
            for li in range(n_loops):
                vi = loop_verts[li]
                vert_colors[vi] += vcol_float[li]
                vert_counts[vi] += 1
            mask = vert_counts > 0
            vert_colors[mask] /= vert_counts[mask, None]
            flat_colors = vert_colors.flatten()
            color_attr.data.foreach_set('color', flat_colors)
            log(f"[BD ExportMesh] Averaged {n_loops} loop colors to {n_verts} vertices")
        else:
            log(f"[BD ExportMesh] WARNING: vcol count ({len(vcol_float)}) != verts ({n_verts}) or loops ({n_loops})")
    else:
        # CORNER domain: expand vertex colors to per-loop
        if len(vcol_float) == n_verts:
            loop_verts = np.zeros(n_loops, dtype=np.int32)
            obj.data.loops.foreach_get('vertex_index', loop_verts)
            loop_colors = vcol_float[loop_verts]
            flat_colors = loop_colors.flatten()
            color_attr.data.foreach_set('color', flat_colors)
            log(f"[BD ExportMesh] Applied {n_verts} vertex colors to {n_loops} loops (CORNER)")
        elif len(vcol_float) == n_loops:
            flat_colors = vcol_float.flatten()
            color_attr.data.foreach_set('color', flat_colors)
            log(f"[BD ExportMesh] Applied {n_loops} per-loop colors directly")
        else:
            log(f"[BD ExportMesh] WARNING: vcol count ({len(vcol_float)}) != verts ({n_verts}) or loops ({n_loops})")

    # Verify colors were set correctly (foreach_get requires full-size buffer)
    all_colors = np.zeros(len(color_attr.data) * 4, dtype=np.float32)
    color_attr.data.foreach_get('color', all_colors)
    all_reshaped = all_colors.reshape(-1, 4)
    log(f"[BD ExportMesh] Verify: RGB=[{all_reshaped[:,:3].min():.4f}, {all_reshaped[:,:3].max():.4f}], A=[{all_reshaped[:,3].min():.4f}, {all_reshaped[:,3].max():.4f}]")

    # Solidify: make all corners in each face the same color
    if solidify_mode == 'DOMINANT':
        log("[BD ExportMesh] Solidifying colors: DOMINANT (most common per face)")
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        color_layer = bm.loops.layers.float_color.get('Color')
        if color_layer is None:
            color_layer = bm.loops.layers.color.get('Color')
        if color_layer:
            from collections import Counter
            for face in bm.faces:
                face_colors = [tuple(round(c, 4) for c in loop[color_layer]) for loop in face.loops]
                dominant = Counter(face_colors).most_common(1)[0][0]
                for loop in face.loops:
                    loop[color_layer] = dominant
            bm.to_mesh(obj.data)
            log("[BD ExportMesh] Solidified to dominant corner color per face")
        else:
            log("[BD ExportMesh] WARNING: Could not find 'Color' layer in bmesh")
        bm.free()

    elif solidify_mode == 'AVERAGE':
        log("[BD ExportMesh] Solidifying colors: AVERAGE per face")
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        color_layer = bm.loops.layers.float_color.get('Color')
        if color_layer is None:
            color_layer = bm.loops.layers.color.get('Color')
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
            log("[BD ExportMesh] Solidified to average color per face")
        else:
            log("[BD ExportMesh] WARNING: Could not find 'Color' layer in bmesh")
        bm.free()

    # Set as active render color
    obj.data.color_attributes.active_color = color_attr
    obj.data.color_attributes.render_color_index = obj.data.color_attributes.find('Color')
    obj.data.update()

else:
    log("[BD ExportMesh] No vertex colors provided - exporting material only")

# Ensure normal map is connected in material
normal_path = os.environ.get('BLENDER_ARG_NORMAL_PATH', '')
if obj.data.materials:
    mat = obj.data.materials[0]
    if mat and mat.use_nodes:
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Find Principled BSDF
        principled = None
        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                principled = node
                break

        if principled:
            # Check if Normal input already connected
            normal_input = principled.inputs.get('Normal')
            if normal_input and not normal_input.is_linked:
                # Look for existing Image Texture with normal data in the node tree
                normal_tex_node = None
                for node in nodes:
                    if node.type == 'TEX_IMAGE' and node.image:
                        img_name = node.image.name.lower()
                        if 'normal' in img_name or 'nrm' in img_name or 'nor' in img_name:
                            normal_tex_node = node
                            break

                # If no existing normal texture found, load from path
                if normal_tex_node is None and normal_path and os.path.exists(normal_path):
                    log(f"[BD ExportMesh] Loading normal map: {normal_path}")
                    img = bpy.data.images.load(normal_path)
                    img.colorspace_settings.name = 'Non-Color'
                    normal_tex_node = nodes.new('ShaderNodeTexImage')
                    normal_tex_node.image = img
                    normal_tex_node.location = (principled.location.x - 600, principled.location.y - 300)

                    # Connect UV if available
                    uv_node = None
                    for node in nodes:
                        if node.type == 'UVMAP' or node.type == 'TEX_COORD':
                            uv_node = node
                            break

                if normal_tex_node:
                    # Create Normal Map node
                    normal_map_node = nodes.new('ShaderNodeNormalMap')
                    normal_map_node.location = (principled.location.x - 200, principled.location.y - 300)

                    # Connect: Image Texture → Normal Map → Principled BSDF
                    links.new(normal_tex_node.outputs['Color'], normal_map_node.inputs['Color'])
                    links.new(normal_map_node.outputs['Normal'], normal_input)
                    log("[BD ExportMesh] Connected normal map to Principled BSDF")
                else:
                    log("[BD ExportMesh] No normal map texture found to connect")
            else:
                log("[BD ExportMesh] Normal input already connected")
        else:
            log("[BD ExportMesh] No Principled BSDF found in material")
else:
    log("[BD ExportMesh] No materials on mesh")

# Update any Color Attribute nodes in material to reference our 'Color' attribute
if obj.data.materials:
    for mat in obj.data.materials:
        if mat and mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'ATTRIBUTE' and hasattr(node, 'attribute_name'):
                    if node.attribute_name != 'Color':
                        log(f"[BD ExportMesh] Updating Attribute node '{node.attribute_name}' -> 'Color'")
                        node.attribute_name = 'Color'
                # ShaderNodeVertexColor (legacy) or ShaderNodeAttribute
                if hasattr(node, 'layer_name') and node.type in ('VERTEX_COLOR',):
                    if node.layer_name != 'Color':
                        log(f"[BD ExportMesh] Updating VertexColor node '{node.layer_name}' -> 'Color'")
                        node.layer_name = 'Color'

# Apply flat shading if requested
if flat_shading:
    log("[BD ExportMesh] Applying flat shading")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    # Set flat shading on all faces
    for poly in obj.data.polygons:
        poly.use_smooth = False
    obj.data.update()
    log("[BD ExportMesh] Flat shading applied")

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
- NONE: Per-vertex colors, preserves mesh topology (shared vertices)
- DOMINANT: Per-face flat color (splits vertices - destroys topology!)
- AVERAGE: Per-face averaged color (splits vertices - destroys topology!)

Use NONE for edge detection, color masking, or further mesh processing.
Use DOMINANT/AVERAGE only for final stylized export where topology doesn't matter.

Inputs a MESH_BUNDLE (from BD_PackBundle or BD_CacheBundle).""",
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
                    default="NONE",
                    tooltip="NONE=per-vertex colors (preserves topology) | DOMINANT/AVERAGE=per-face colors (splits vertices)",
                ),
                io.Boolean.Input(
                    "flat_shading",
                    default=False,
                    optional=True,
                    tooltip="Apply flat shading in Blender (splits normals at edges for hard-edge look)",
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
        solidify_mode: str = "NONE",
        flat_shading: bool = False,
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
        color_field = bundle.get('color_field') if isinstance(bundle, dict) else None
        name = bundle.get('name', 'mesh') if isinstance(bundle, dict) else 'mesh'

        # Debug: show bundle contents
        if isinstance(bundle, dict):
            has_cf = color_field is not None
            has_vc = vertex_colors is not None
            has_norm = bundle.get('normal') is not None
            vc_info = f"shape={vertex_colors.shape}, max={vertex_colors.max()}" if has_vc else "None"
            print(f"[BD BlenderExportMesh] Bundle: mesh={len(mesh.vertices):,}v | vcol={vc_info} | color_field={'yes' if has_cf else 'no'} | normal={'yes' if has_norm else 'no'}")

        # Check if vertex_colors is effectively empty (all zeros/black)
        if vertex_colors is not None and isinstance(vertex_colors, np.ndarray):
            if vertex_colors.max() == 0:
                print(f"[BD BlenderExportMesh] vertex_colors is all zeros - treating as empty")
                vertex_colors = None

        # Set up output directory
        import folder_paths
        output_base = folder_paths.get_output_directory()
        out_dir = os.path.join(output_base, output_dir)
        os.makedirs(out_dir, exist_ok=True)

        glb_output = os.path.join(out_dir, f"{name}.glb")

        # Temp files
        input_path = None
        vcol_path = None
        color_field_path = None
        normal_path = None
        blender_output = None

        try:
            # Save mesh as temp GLB (with material)
            input_path = cls._mesh_to_temp_file(mesh, suffix='.glb')

            # Determine color source: pre-computed vertex_colors OR raw color_field
            if vertex_colors is not None and len(vertex_colors) > 0:
                # Pre-computed vertex colors (from BD_ApplyColorField path)
                fd, vcol_path = tempfile.mkstemp(suffix='.npy')
                os.close(fd)
                np.save(vcol_path, vertex_colors)
                print(f"[BD BlenderExportMesh] Saved {len(vertex_colors):,} pre-computed vertex colors (max={vertex_colors.max()})")
            elif color_field is not None:
                # Pass color_field to Blender for spatial sampling
                # This avoids vertex-index mismatch from GLB roundtrip
                try:
                    positions = np.asarray(color_field['positions'], dtype=np.float32)
                    colors = np.asarray(color_field['colors'], dtype=np.float32)
                    voxel_size = float(color_field.get('voxel_size', 1.0))

                    fd, color_field_path = tempfile.mkstemp(suffix='.npz')
                    os.close(fd)
                    np.savez(color_field_path,
                             positions=positions,
                             colors=colors,
                             voxel_size=np.array([voxel_size]))
                    print(f"[BD BlenderExportMesh] Saved color_field for Blender sampling: {len(positions):,} voxels, voxel_size={voxel_size:.6f}")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[BD BlenderExportMesh] Warning: color_field prep failed: {e}")
                    color_field_path = None
            else:
                vcol_path = ''
                print(f"[BD BlenderExportMesh] No vertex colors or color_field to apply")

            # Save normal texture as temp PNG if available in bundle
            if isinstance(bundle, dict) and bundle.get('normal') is not None:
                from PIL import Image
                normal_data = bundle['normal']
                fd, normal_path = tempfile.mkstemp(suffix='.png')
                os.close(fd)
                Image.fromarray(normal_data).save(normal_path)
                print(f"[BD BlenderExportMesh] Saved normal map: {normal_data.shape}")
            else:
                normal_path = ''

            # Output goes directly to final location
            fd, blender_output = tempfile.mkstemp(suffix='.glb')
            os.close(fd)

            # Run Blender export script
            extra_args = {
                'vcol_path': vcol_path or '',
                'color_field_path': color_field_path or '',
                'solidify_mode': solidify_mode,
                'flat_shading': 'True' if flat_shading else 'False',
                'normal_path': normal_path or '',
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
                # Fix permissions (tempfile creates with 0o600, we want group/world readable)
                os.chmod(glb_output, 0o666)
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
            elif color_field_path:
                parts.append(f"+COLOR_0 from color_field ({solidify_mode})")
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
            if color_field_path and os.path.exists(color_field_path):
                os.remove(color_field_path)
            if normal_path and os.path.exists(normal_path):
                os.remove(normal_path)
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
