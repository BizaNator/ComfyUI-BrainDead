"""
BD_UVUnwrap - UV unwrap mesh with xatlas (GPU) or Blender Smart UV.

Supports preserving color-marked sharp edges as UV seams.
"""

import os
import sys
import tempfile
import math

from comfy_api.latest import io

from ..blender.base import BlenderNodeMixin, HAS_TRIMESH

if HAS_TRIMESH:
    import trimesh
    import numpy as np

from .types import TrimeshInput, TrimeshOutput


# ============================================================================
# BLENDER UV UNWRAP SCRIPT
# ============================================================================
BLENDER_UV_SCRIPT = '''
import bpy
import bmesh
import os
import sys
from math import radians

def log(msg):
    """Print with immediate flush for real-time output."""
    print(msg)
    sys.stdout.flush()

def get_env_bool(name, default=True):
    val = os.environ.get(name, str(default))
    return val.lower() in ('true', '1', 'yes')

def get_env_float(name, default=0.0):
    return float(os.environ.get(name, str(default)))

# Configuration
INPUT_PATH = os.environ['BLENDER_INPUT_PATH']
OUTPUT_PATH = os.environ['BLENDER_OUTPUT_PATH']
SEAMS_FROM_SHARP = get_env_bool('BLENDER_ARG_SEAMS_FROM_SHARP', True)
ANGLE_LIMIT = get_env_float('BLENDER_ARG_ANGLE_LIMIT', 66.0)
ISLAND_MARGIN = get_env_float('BLENDER_ARG_ISLAND_MARGIN', 0.02)

log(f"[BD UV Unwrap] Starting...")
log(f"[BD UV Unwrap] Seams from sharp: {SEAMS_FROM_SHARP}")
log(f"[BD UV Unwrap] Angle limit: {ANGLE_LIMIT}°")
log(f"[BD UV Unwrap] Island margin: {ISLAND_MARGIN}")

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh
ext = os.path.splitext(INPUT_PATH)[1].lower()
log(f"[BD UV Unwrap] Importing {ext}...")

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

# Get mesh object
obj = bpy.context.active_object
if obj is None or obj.type != 'MESH':
    mesh_objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not mesh_objects:
        raise ValueError("No mesh objects found after import!")
    obj = mesh_objects[0]

bpy.context.view_layer.objects.active = obj
obj.select_set(True)

log(f"[BD UV Unwrap] Mesh: {len(obj.data.vertices)} verts, {len(obj.data.polygons)} faces")

# Check for vertex colors
has_colors = len(obj.data.color_attributes) > 0
if has_colors:
    log(f"[BD UV Unwrap] Vertex colors: {[c.name for c in obj.data.color_attributes]}")
else:
    log("[BD UV Unwrap] No vertex colors found in Blender")

# Enable BrainDead addon if available
addons_path = os.path.dirname(os.path.dirname(bpy.app.binary_path)) + '/5.0/scripts/addons'
if addons_path not in sys.path:
    sys.path.insert(0, addons_path)

HAS_BD_ADDON = False
try:
    import braindead_blender
    braindead_blender.register()
    HAS_BD_ADDON = True
    log("[BD UV Unwrap] BrainDead addon loaded")
except ImportError:
    log("[BD UV Unwrap] BrainDead addon not available - using built-in ops")

# Step 1: Mark UV seams from sharp edges
if SEAMS_FROM_SHARP:
    log("[BD UV Unwrap] Marking UV seams from sharp edges...")
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='EDGE')
    bpy.ops.mesh.select_all(action='DESELECT')

    # Select sharp edges
    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()

    sharp_count = 0
    for edge in bm.edges:
        if not edge.smooth:  # Sharp edge
            edge.select = True
            sharp_count += 1

    bmesh.update_edit_mesh(obj.data)
    log(f"[BD UV Unwrap] Found {sharp_count} sharp edges")

    # Mark selected edges as seams
    if sharp_count > 0:
        bpy.ops.mesh.mark_seam(clear=False)
        log(f"[BD UV Unwrap] Marked {sharp_count} edges as UV seams")

    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.object.mode_set(mode='OBJECT')

# Step 2: UV Unwrap with Smart UV Project
log("[BD UV Unwrap] Running Smart UV Project...")
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')

bpy.ops.uv.smart_project(
    angle_limit=radians(ANGLE_LIMIT),
    island_margin=ISLAND_MARGIN,
    area_weight=0.0,
    correct_aspect=True,
    scale_to_bounds=False,
)

bpy.ops.object.mode_set(mode='OBJECT')

# Check UV result
uv_layers = obj.data.uv_layers
if uv_layers:
    log(f"[BD UV Unwrap] UV layers: {len(uv_layers)}")
    for layer in uv_layers:
        log(f"  - {layer.name} (active: {layer.active})")
        # Debug: show UV bounds
        if layer.data:
            uvs = [d.uv for d in layer.data]
            if uvs:
                u_vals = [uv[0] for uv in uvs]
                v_vals = [uv[1] for uv in uvs]
                log(f"    UV bounds: U=[{min(u_vals):.3f}, {max(u_vals):.3f}] V=[{min(v_vals):.3f}, {max(v_vals):.3f}]")
                log(f"    UV count: {len(uvs)}")
else:
    log("[BD UV Unwrap] WARNING: No UV layers created!")

# CRITICAL: Add a material to the mesh so trimesh loads UVs properly
# Without a material, trimesh creates ColorVisuals and discards TEXCOORD_0 data
if not obj.data.materials:
    log("[BD UV Unwrap] Adding dummy material for GLB UV export...")
    mat = bpy.data.materials.new(name="UV_Material")
    mat.use_nodes = True
    # Set to use vertex colors as base color
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        # Create vertex color node
        vc_node = mat.node_tree.nodes.new('ShaderNodeVertexColor')
        vc_node.layer_name = 'Col'  # Default vertex color layer name
        mat.node_tree.links.new(vc_node.outputs['Color'], bsdf.inputs['Base Color'])
    obj.data.materials.append(mat)
    log("[BD UV Unwrap] Added material with vertex color input")

# Export - GLB handles coordinate conversion automatically
# GLB import converts Y-up to Z-up, export with export_yup=True converts back
ext_out = os.path.splitext(OUTPUT_PATH)[1].lower()
log(f"[BD UV Unwrap] Exporting to {ext_out}...")

if ext_out == '.ply':
    bpy.ops.wm.ply_export(filepath=OUTPUT_PATH, export_colors='SRGB', ascii_format=True)
elif ext_out == '.obj':
    bpy.ops.wm.obj_export(filepath=OUTPUT_PATH, export_uv=True, export_colors=True)
elif ext_out in ['.glb', '.gltf']:
    # Ensure UVs are exported - use explicit parameters
    bpy.ops.export_scene.gltf(
        filepath=OUTPUT_PATH,
        export_format='GLB',
        export_attributes=True,  # Vertex colors
        export_texcoords=True,   # UVs (TEXCOORD_0)
        export_normals=True,     # Normals
        export_materials='EXPORT',  # Export materials (needed for trimesh to load UVs!)
        export_yup=True,         # Convert Z-up (Blender) back to Y-up (GLTF standard)
    )
    log(f"[BD UV Unwrap] GLB exported with material + texcoords")

log(f"[BD UV Unwrap] Saved to {OUTPUT_PATH}")
log("[BD UV Unwrap] COMPLETE")
'''


class BD_UVUnwrap(BlenderNodeMixin, io.ComfyNode):
    """
    UV unwrap a mesh with choice of algorithm.

    Supports:
    - xatlas (GPU-accelerated via CuMesh) - fast, good for game assets
    - Blender Smart UV Project - quality unwrap, respects seams

    Sharp edges (from color detection in BD_BlenderDecimate) can
    automatically become UV seams for clean texture boundaries.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_UVUnwrap",
            display_name="BD UV Unwrap",
            category="🧠BrainDead/Mesh",
            description="""UV unwrap mesh with xatlas (GPU) or Blender Smart UV.

Sharp edges from color detection become UV seams for clean boundaries.

Methods:
- xatlas_gpu: Fast GPU-accelerated unwrap (CuMesh)
- blender_smart_uv: Quality unwrap with seam support""",
            inputs=[
                TrimeshInput("mesh"),
                io.Combo.Input(
                    "method",
                    options=["blender_smart_uv", "cumesh_gpu", "xatlas_gpu"],
                    default="blender_smart_uv",
                    tooltip="UV unwrap algorithm. cumesh_gpu uses CuMesh chart clustering + xatlas packing.",
                ),
                io.Boolean.Input(
                    "seams_from_sharp",
                    default=True,
                    tooltip="Auto-create UV seams from sharp edges (color boundaries) [Blender only]",
                ),
                io.Float.Input(
                    "angle_limit",
                    default=66.0,
                    min=1.0,
                    max=89.0,
                    step=1.0,
                    tooltip="Smart UV angle limit (Blender) / cone half-angle in degrees (CuMesh)",
                ),
                io.Float.Input(
                    "island_margin",
                    default=0.02,
                    min=0.0,
                    max=0.5,
                    step=0.01,
                    tooltip="UV island margin",
                ),
                io.Int.Input(
                    "chart_refine_iterations",
                    default=0,
                    min=0,
                    max=1000,
                    step=10,
                    optional=True,
                    tooltip="Chart clustering refinement iterations (CuMesh only). Higher = better charts, slower.",
                ),
                io.Int.Input(
                    "chart_global_iterations",
                    default=1,
                    min=1,
                    max=10,
                    step=1,
                    optional=True,
                    tooltip="Chart clustering global iterations (CuMesh only)",
                ),
                io.Float.Input(
                    "chart_smooth_strength",
                    default=1.0,
                    min=0.0,
                    max=5.0,
                    step=0.1,
                    optional=True,
                    tooltip="Chart boundary smoothing strength (CuMesh only)",
                ),
                io.Int.Input(
                    "timeout",
                    default=300,
                    min=60,
                    max=1800,
                    optional=True,
                    tooltip="Maximum processing time in seconds (Blender only)",
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
        method: str = "blender_smart_uv",
        seams_from_sharp: bool = True,
        angle_limit: float = 66.0,
        island_margin: float = 0.02,
        chart_refine_iterations: int = 0,
        chart_global_iterations: int = 1,
        chart_smooth_strength: float = 1.0,
        timeout: int = 300,
    ) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "ERROR: trimesh not installed")

        if mesh is None:
            return io.NodeOutput(None, "ERROR: No input mesh")

        orig_verts = len(mesh.vertices)
        orig_faces = len(mesh.faces)
        print(f"[BD UV Unwrap] Input: {orig_verts:,} verts, {orig_faces:,} faces")
        print(f"[BD UV Unwrap] Method: {method}")

        if method == "cumesh_gpu":
            node_output = cls._unwrap_cumesh(
                mesh, angle_limit, chart_refine_iterations,
                chart_global_iterations, chart_smooth_strength,
            )
            status = node_output.args[1] if len(node_output.args) > 1 else ""
            if isinstance(status, str) and status.startswith("ERROR:"):
                print("[BD UV Unwrap] CuMesh UV failed - falling back to Blender Smart UV...")
                return cls._unwrap_blender(mesh, seams_from_sharp, angle_limit, island_margin, timeout)
            return node_output
        elif method == "xatlas_gpu":
            node_output = cls._unwrap_xatlas(mesh, island_margin)
            status = node_output.args[1] if len(node_output.args) > 1 else ""
            if isinstance(status, str) and status.startswith("ERROR:"):
                if "manifold" in status.lower() or "invalid argument" in status.lower():
                    print("[BD UV Unwrap] xatlas failed (non-manifold mesh) - falling back to Blender Smart UV...")
                    return cls._unwrap_blender(mesh, seams_from_sharp, angle_limit, island_margin, timeout)
            return node_output
        else:
            return cls._unwrap_blender(mesh, seams_from_sharp, angle_limit, island_margin, timeout)

    @classmethod
    def _unwrap_xatlas(cls, mesh, island_margin: float) -> io.NodeOutput:
        """GPU-accelerated UV unwrap using CuMesh/xatlas."""
        try:
            import torch
            import cumesh as CuMesh
        except ImportError:
            return io.NodeOutput(mesh, "ERROR: cumesh not available - use blender_smart_uv")

        print("[BD UV Unwrap] Using xatlas GPU unwrap (cumesh)...")

        try:
            # Convert to tensors (int32 for faces, as CuMesh expects)
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
            faces = torch.tensor(mesh.faces, dtype=torch.int32).cuda()

            # Initialize CuMesh
            cumesh = CuMesh.CuMesh()
            cumesh.init(vertices, faces)

            # Run xatlas UV unwrap
            print("[BD UV Unwrap] Running xatlas unwrap...")
            out_vertices, out_faces, out_uvs, out_vmaps = cumesh.uv_unwrap(
                compute_charts_kwargs={
                    'threshold_cone_half_angle_rad': 1.57,  # ~90 degrees
                    'refine_iterations': 0,
                    'global_iterations': 1,
                    'smooth_strength': 1,
                },
                return_vmaps=True,
                verbose=True,
            )

            # Convert to numpy
            out_vertices_np = out_vertices.cpu().numpy()
            out_faces_np = out_faces.cpu().numpy()
            out_uvs_np = out_uvs.cpu().numpy()

            # Flip V coordinate for OpenGL compatibility
            out_uvs_np[:, 1] = 1 - out_uvs_np[:, 1]

            # Create new mesh with UVs
            # Note: UV unwrap may change topology (split vertices at seams)
            result = trimesh.Trimesh(
                vertices=out_vertices_np,
                faces=out_faces_np,
                process=False,
            )

            # Transfer vertex colors if present (using vmap to handle split vertices)
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                if mesh.visual.vertex_colors is not None:
                    orig_colors = mesh.visual.vertex_colors
                    # Map original colors to new vertices via vmap
                    vmaps_np = out_vmaps.cpu().numpy()
                    new_colors = orig_colors[vmaps_np]
                    result.visual.vertex_colors = new_colors

            # Attach UVs as TextureVisuals
            from trimesh.visual import TextureVisuals
            result.visual = TextureVisuals(uv=out_uvs_np)

            # Re-attach vertex colors after setting TextureVisuals
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                if mesh.visual.vertex_colors is not None:
                    orig_colors = mesh.visual.vertex_colors
                    vmaps_np = out_vmaps.cpu().numpy()
                    result.visual.vertex_colors = orig_colors[vmaps_np]

            status = f"UV unwrapped (xatlas): {len(result.vertices):,} verts, {len(out_uvs_np)} UVs"
            print(f"[BD UV Unwrap] {status}")

            # Cleanup
            del vertices, faces, cumesh, out_vertices, out_faces, out_uvs, out_vmaps
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            return io.NodeOutput(result, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return io.NodeOutput(mesh, f"ERROR: xatlas failed: {e}")

    @classmethod
    def _unwrap_cumesh(
        cls,
        mesh,
        angle_limit: float,
        refine_iterations: int,
        global_iterations: int,
        smooth_strength: float,
    ) -> io.NodeOutput:
        """GPU-accelerated UV unwrap using CuMesh chart clustering + xatlas packing."""
        try:
            import torch
            import cumesh as CuMesh
        except ImportError:
            return io.NodeOutput(mesh, "ERROR: cumesh not available - use blender_smart_uv")

        print(f"[BD UV Unwrap] Using CuMesh GPU unwrap (chart clustering)...")
        print(f"[BD UV Unwrap] Cone angle: {angle_limit}°, refine: {refine_iterations}, global: {global_iterations}, smooth: {smooth_strength}")

        try:
            # Convert to tensors
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
            faces = torch.tensor(mesh.faces, dtype=torch.int32).cuda()

            # Initialize CuMesh
            cu = CuMesh.CuMesh()
            cu.init(vertices, faces)

            # Compute vertex normals (needed for chart clustering)
            cu.compute_vertex_normals()

            # Run UV unwrap with chart clustering options
            print("[BD UV Unwrap] Running CuMesh chart clustering + xatlas pack...")
            out_vertices, out_faces, out_uvs, out_vmaps = cu.uv_unwrap(
                compute_charts_kwargs={
                    'threshold_cone_half_angle_rad': math.radians(angle_limit),
                    'refine_iterations': refine_iterations,
                    'global_iterations': global_iterations,
                    'smooth_strength': smooth_strength,
                    'area_penalty_weight': 0.1,
                    'perimeter_area_ratio_weight': 0.0001,
                },
                return_vmaps=True,
                verbose=True,
            )

            # Convert to numpy
            out_vertices_np = out_vertices.cpu().numpy()
            out_faces_np = out_faces.cpu().numpy()
            out_uvs_np = out_uvs.cpu().numpy()
            out_vmaps_np = out_vmaps.cpu().numpy()

            # Flip V coordinate for glTF convention
            out_uvs_np[:, 1] = 1 - out_uvs_np[:, 1]

            # Create result mesh with UVs
            result = trimesh.Trimesh(
                vertices=out_vertices_np,
                faces=out_faces_np,
                process=False,
            )

            # Transfer vertex colors if present
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                if mesh.visual.vertex_colors is not None:
                    orig_colors = mesh.visual.vertex_colors
                    new_colors = orig_colors[out_vmaps_np]
                    result.visual.vertex_colors = new_colors

            # Attach UVs
            from trimesh.visual import TextureVisuals
            result.visual = TextureVisuals(uv=out_uvs_np)

            # Transfer normals if present
            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                result.vertex_normals = np.array(mesh.vertex_normals)[out_vmaps_np]

            status = f"UV unwrapped (CuMesh): {len(result.vertices):,} verts, {len(out_uvs_np)} UVs | refine={refine_iterations}, global={global_iterations}"
            print(f"[BD UV Unwrap] {status}")

            # Cleanup
            del vertices, faces, cu, out_vertices, out_faces, out_uvs, out_vmaps
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            return io.NodeOutput(result, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return io.NodeOutput(mesh, f"ERROR: CuMesh UV failed: {e}")

    @classmethod
    def _unwrap_blender(
        cls,
        mesh,
        seams_from_sharp: bool,
        angle_limit: float,
        island_margin: float,
        timeout: int,
    ) -> io.NodeOutput:
        """UV unwrap using Blender Smart UV Project."""
        available, msg = cls._check_blender()
        if not available:
            return io.NodeOutput(mesh, f"ERROR: {msg}")

        input_path = None
        output_path = None

        # Store original vertex colors for re-transfer if needed
        orig_colors = None
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
            if mesh.visual.vertex_colors is not None:
                orig_colors = mesh.visual.vertex_colors.copy()
                print(f"[BD UV Unwrap] Stored {len(orig_colors)} vertex colors for re-transfer")

        try:
            # Use GLB for input/output - supports UVs natively
            input_path = cls._mesh_to_temp_file(mesh, suffix='.glb')
            fd, output_path = tempfile.mkstemp(suffix='.glb')
            os.close(fd)

            print(f"[BD UV Unwrap] Running Blender Smart UV Project...")

            extra_args = {
                'seams_from_sharp': seams_from_sharp,
                'angle_limit': angle_limit,
                'island_margin': island_margin,
            }

            success, message, log_lines = cls._run_blender_script(
                BLENDER_UV_SCRIPT,
                input_path,
                output_path,
                extra_args=extra_args,
                timeout=timeout,
            )

            if not success:
                error_context = '\n'.join(log_lines[-10:]) if log_lines else ''
                print(f"[BD UV Unwrap] FAILED: {message}")
                if error_context:
                    print(f"[BD UV Unwrap] Log tail:\n{error_context}")
                return io.NodeOutput(mesh, f"ERROR: {message}")

            # Load result (GLB preserves UVs)
            result_mesh = cls._load_mesh_from_file(output_path)

            # Check if UVs were created
            has_uvs = False
            if hasattr(result_mesh, 'visual') and hasattr(result_mesh.visual, 'uv'):
                if result_mesh.visual.uv is not None and len(result_mesh.visual.uv) > 0:
                    has_uvs = True

            # Check if vertex colors survived
            has_colors = False
            if hasattr(result_mesh, 'visual') and hasattr(result_mesh.visual, 'vertex_colors'):
                if result_mesh.visual.vertex_colors is not None:
                    has_colors = True

            # Re-transfer vertex colors from original if lost
            colors_transferred = False
            if not has_colors and orig_colors is not None:
                print("[BD UV Unwrap] Vertex colors lost - re-transferring from original...")
                # If vertex count matches, direct copy
                if len(result_mesh.vertices) == len(orig_colors):
                    result_mesh.visual.vertex_colors = orig_colors
                    colors_transferred = True
                    print(f"[BD UV Unwrap] Direct color transfer: {len(orig_colors)} colors")
                else:
                    # Vertex count changed (UV seams split vertices) - use nearest neighbor
                    try:
                        from scipy.spatial import cKDTree
                        tree = cKDTree(mesh.vertices)
                        _, indices = tree.query(result_mesh.vertices, k=1)
                        result_mesh.visual.vertex_colors = orig_colors[indices]
                        colors_transferred = True
                        print(f"[BD UV Unwrap] Nearest-neighbor color transfer: {len(result_mesh.vertices)} verts")
                    except ImportError:
                        print("[BD UV Unwrap] Warning: scipy not available for color transfer")

            status = f"UV unwrapped (Blender): {len(result_mesh.vertices):,} verts"
            if has_uvs:
                status += f" | {len(result_mesh.visual.uv)} UVs"
            if has_colors or colors_transferred:
                status += " | colors preserved" if has_colors else " | colors re-transferred"
            if seams_from_sharp:
                status += " | seams from sharp"

            print(f"[BD UV Unwrap] {status}")
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


# V3 node list
UNWRAP_V3_NODES = [BD_UVUnwrap]

# V1 compatibility
UNWRAP_NODES = {
    "BD_UVUnwrap": BD_UVUnwrap,
}

UNWRAP_DISPLAY_NAMES = {
    "BD_UVUnwrap": "BD UV Unwrap",
}
