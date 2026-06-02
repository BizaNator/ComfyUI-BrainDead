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

# Force smooth shading — PLY input has no stored normals so Blender computes smooth
# by default, but be explicit. Without this, any flat-shaded mesh would cause
# Blender's GLTF exporter to write one vertex per face-corner (3.0 boundary/face ratio).
bpy.ops.object.shade_smooth()

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

# Save UV sidecar BEFORE removing UV layers from the mesh
import numpy as np
uv_sidecar_path = OUTPUT_PATH + '.uvs.npy'
uv_layers = obj.data.uv_layers
if uv_layers and uv_layers.active:
    uv_data = uv_layers.active.data
    uvs_np = np.array([d.uv[:] for d in uv_data], dtype=np.float32)
    np.save(uv_sidecar_path, uvs_np)
    log(f"[BD UV Unwrap] Saved UV sidecar: {len(uvs_np)} UVs → {uv_sidecar_path}")
else:
    log("[BD UV Unwrap] WARNING: No UV data to save in sidecar")

# Remove UV layers from mesh before PLY export.
# UV data is stored as CORNER (per-loop) domain in Blender. Even though PLY
# doesn't carry UV coordinates, the CORNER-domain attribute forces Blender's
# PLY exporter to split vertices at UV seam boundaries (2,442 → 7,249).
# The UV is already saved to the sidecar, so remove it here.
for layer in list(obj.data.uv_layers):
    obj.data.uv_layers.remove(layer)
log("[BD UV Unwrap] UV layers removed before PLY export (UV in sidecar)")

# Clear custom split normals — these are stored per-loop and would also force
# vertex splitting in any format that serialises normals (GLB, FBX, etc.)
try:
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.customdata_custom_splitnormals_clear()
    bpy.ops.object.mode_set(mode='OBJECT')
    log("[BD UV Unwrap] Cleared custom split normals")
except Exception as e:
    log(f"[BD UV Unwrap] Note: {e}")
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass

# Export mesh — PLY carries geometry + vertex colors, no UV (UV is in sidecar)
ext_out = os.path.splitext(OUTPUT_PATH)[1].lower()
log(f"[BD UV Unwrap] Exporting to {ext_out}...")

if ext_out == '.ply':
    bpy.ops.wm.ply_export(filepath=OUTPUT_PATH, export_colors='SRGB', ascii_format=False)
elif ext_out == '.obj':
    bpy.ops.wm.obj_export(filepath=OUTPUT_PATH, export_uv=True, export_colors=True)
elif ext_out in ['.glb', '.gltf']:
    bpy.ops.export_scene.gltf(
        filepath=OUTPUT_PATH,
        export_format='GLB',
        export_attributes=True,
        export_texcoords=True,
        export_normals=True,
        export_materials='EXPORT',
        export_yup=True,
    )
    log(f"[BD UV Unwrap] GLB exported with texcoords + normals")

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

    @staticmethod
    def _loop_uv_to_per_vertex(uvs_loop: np.ndarray, faces: np.ndarray, n_verts: int) -> np.ndarray:
        """
        Convert per-loop (face-corner) UV array to per-vertex UV without splitting geometry.

        uvs_loop: (n_faces*3, 2) — one UV per face-corner, ordered by face then corner
        faces:    (n_faces, 3)   — vertex indices
        returns:  (n_verts, 2)   — per-vertex UV (first-wins at seam boundaries)
        """
        uvs_flat = uvs_loop.reshape(-1, 2)
        corners = faces.reshape(-1)          # (n_faces*3,) — vert index per loop
        _, first_idx = np.unique(corners, return_index=True)
        per_vert = np.zeros((n_verts, 2), dtype=np.float32)
        per_vert[corners[first_idx]] = uvs_flat[first_idx]
        return per_vert

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
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
            faces = torch.tensor(mesh.faces, dtype=torch.int32).cuda()

            cumesh_obj = CuMesh.CuMesh()
            cumesh_obj.init(vertices, faces)

            print("[BD UV Unwrap] Running xatlas unwrap...")
            out_vertices, out_faces, out_uvs, out_vmaps = cumesh_obj.uv_unwrap(
                compute_charts_kwargs={
                    'threshold_cone_half_angle_rad': 1.57,
                    'refine_iterations': 0,
                    'global_iterations': 1,
                    'smooth_strength': 1,
                },
                return_vmaps=True,
                verbose=True,
            )

            out_faces_np = out_faces.cpu().numpy()
            out_uvs_np = out_uvs.cpu().numpy()
            out_uvs_np[:, 1] = 1 - out_uvs_np[:, 1]  # Flip V for OpenGL

            # Face-corner UVs for the original face ordering (face count unchanged)
            face_uv = out_uvs_np[out_faces_np]  # (n_faces, 3, 2)

            # Return ORIGINAL mesh geometry — no vertex splitting
            result = trimesh.Trimesh(
                vertices=mesh.vertices.copy(),
                faces=mesh.faces.copy(),
                process=False,
            )

            # Extract vertex colors from incoming mesh — don't touch TextureVisuals.vertex_colors
            # (it's a computed rasterization property; its setter replaces visual with ColorVisuals)
            from trimesh.visual import TextureVisuals
            vc_to_transfer = None
            if hasattr(mesh, 'vertex_attributes') and 'COLOR_0' in mesh.vertex_attributes:
                vc_to_transfer = np.array(mesh.vertex_attributes['COLOR_0'])
            elif hasattr(mesh, 'visual') and mesh.visual is not None and not isinstance(mesh.visual, TextureVisuals):
                vc = mesh.visual.vertex_colors
                if vc is not None and len(vc) > 0:
                    vc_to_transfer = np.array(vc)

            # Per-vertex UV via first-wins (no splits)
            per_vert_uv = cls._loop_uv_to_per_vertex(
                face_uv.reshape(-1, 2), mesh.faces, len(mesh.vertices)
            )
            result.visual = TextureVisuals(uv=per_vert_uv)
            # Store colors in vertex_attributes — NOT visual.vertex_colors (that setter kills UV)
            if vc_to_transfer is not None:
                result.vertex_attributes['COLOR_0'] = vc_to_transfer

            # Store face-corner UV in metadata for downstream baking nodes
            result.metadata['face_uv'] = face_uv.astype(np.float32)

            status = (f"UV unwrapped (xatlas): {len(mesh.vertices):,} verts (no split) "
                      f"| {len(out_faces_np):,} faces | {len(per_vert_uv)} UVs")
            print(f"[BD UV Unwrap] {status}")

            del vertices, faces, cumesh_obj, out_vertices, out_faces, out_uvs, out_vmaps
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

            out_faces_np = out_faces.cpu().numpy()
            out_uvs_np = out_uvs.cpu().numpy()
            out_uvs_np[:, 1] = 1 - out_uvs_np[:, 1]  # Flip V for glTF

            # Face-corner UVs for the original face ordering
            face_uv = out_uvs_np[out_faces_np]  # (n_faces, 3, 2)

            # Return ORIGINAL mesh geometry — no vertex splitting
            result = trimesh.Trimesh(
                vertices=mesh.vertices.copy(),
                faces=mesh.faces.copy(),
                process=False,
            )

            # Extract vertex colors from incoming mesh — don't touch TextureVisuals.vertex_colors
            from trimesh.visual import TextureVisuals
            vc_to_transfer = None
            if hasattr(mesh, 'vertex_attributes') and 'COLOR_0' in mesh.vertex_attributes:
                vc_to_transfer = np.array(mesh.vertex_attributes['COLOR_0'])
            elif hasattr(mesh, 'visual') and mesh.visual is not None and not isinstance(mesh.visual, TextureVisuals):
                vc = mesh.visual.vertex_colors
                if vc is not None and len(vc) > 0:
                    vc_to_transfer = np.array(vc)

            # Per-vertex UV via first-wins (no splits)
            per_vert_uv = cls._loop_uv_to_per_vertex(
                face_uv.reshape(-1, 2), mesh.faces, len(mesh.vertices)
            )
            result.visual = TextureVisuals(uv=per_vert_uv)
            if vc_to_transfer is not None:
                result.vertex_attributes['COLOR_0'] = vc_to_transfer

            result.metadata['face_uv'] = face_uv.astype(np.float32)

            status = (f"UV unwrapped (CuMesh): {len(mesh.vertices):,} verts (no split) "
                      f"| {len(out_faces_np):,} faces | refine={refine_iterations}, global={global_iterations}")
            print(f"[BD UV Unwrap] {status}")

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

        # Store original vertex colors for re-transfer if needed.
        # Don't access TextureVisuals.vertex_colors — check vertex_attributes first.
        orig_colors = None
        from trimesh.visual import TextureVisuals as _TV
        if hasattr(mesh, 'vertex_attributes') and 'COLOR_0' in mesh.vertex_attributes:
            orig_colors = np.array(mesh.vertex_attributes['COLOR_0'])
            print(f"[BD UV Unwrap] Stored {len(orig_colors)} vertex colors (from vertex_attributes)")
        elif hasattr(mesh, 'visual') and mesh.visual is not None and not isinstance(mesh.visual, _TV):
            vc = mesh.visual.vertex_colors
            if vc is not None and len(vc) > 0:
                orig_colors = np.array(vc)
                print(f"[BD UV Unwrap] Stored {len(orig_colors)} vertex colors for re-transfer")

        try:
            # Both input and output are PLY — PLY has no stored normals so Blender
            # computes smooth normals by default, and PLY export writes geometry only
            # (no UV seam vertex splits). UVs travel separately as a numpy sidecar.
            input_path = cls._mesh_to_temp_file(mesh, suffix='.ply')
            fd, output_path = tempfile.mkstemp(suffix='.ply')
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

            # Load result — PLY preserves vertex count (no UV-seam vertex splits)
            result_mesh = cls._load_mesh_from_file(output_path)

            n_result_verts = len(result_mesh.vertices)
            n_result_faces = len(result_mesh.faces)

            # Load UV sidecar (per-loop, shape (n_faces*3, 2))
            uv_sidecar = output_path + '.uvs.npy'
            has_uvs = False
            uvs_loop = None
            if os.path.exists(uv_sidecar):
                try:
                    uvs_loop = np.load(uv_sidecar)
                    expected = n_result_faces * 3
                    if len(uvs_loop) == expected:
                        # Convert per-loop → per-vertex without splitting geometry
                        face_uv = uvs_loop.reshape(n_result_faces, 3, 2).astype(np.float32)
                        per_vert_uv = cls._loop_uv_to_per_vertex(
                            uvs_loop, result_mesh.faces, n_result_verts
                        )
                        from trimesh.visual import TextureVisuals
                        # Extract colors before replacing visual — don't access TextureVisuals.vertex_colors
                        existing_vc = None
                        if hasattr(result_mesh, 'vertex_attributes') and 'COLOR_0' in result_mesh.vertex_attributes:
                            existing_vc = np.array(result_mesh.vertex_attributes['COLOR_0'])
                        elif (hasattr(result_mesh.visual, 'vertex_colors') and
                              not isinstance(result_mesh.visual, TextureVisuals)):
                            vc = result_mesh.visual.vertex_colors
                            if vc is not None and len(vc) > 0:
                                existing_vc = np.array(vc)
                        result_mesh.visual = TextureVisuals(uv=per_vert_uv)
                        # Store colors in vertex_attributes — visual.vertex_colors setter kills UV
                        if existing_vc is not None:
                            result_mesh.vertex_attributes['COLOR_0'] = existing_vc
                        result_mesh.metadata['face_uv'] = face_uv
                        has_uvs = True
                        print(f"[BD UV Unwrap] UVs loaded: {len(uvs_loop)} loops → {n_result_verts} per-vertex (no split)")
                    else:
                        print(f"[BD UV Unwrap] UV sidecar size mismatch: {len(uvs_loop)} vs expected {expected}")
                except Exception as e:
                    print(f"[BD UV Unwrap] UV sidecar load failed: {e}")
                finally:
                    if os.path.exists(uv_sidecar):
                        os.remove(uv_sidecar)

            # Restore vertex colors if lost during PLY roundtrip.
            # After UV assignment, visual is TextureVisuals — use vertex_attributes for colors.
            has_colors = (
                (hasattr(result_mesh, 'vertex_attributes') and 'COLOR_0' in result_mesh.vertex_attributes) or
                (hasattr(result_mesh, 'visual') and result_mesh.visual is not None and
                 not isinstance(result_mesh.visual, TextureVisuals) and
                 hasattr(result_mesh.visual, 'vertex_colors') and result_mesh.visual.vertex_colors is not None)
            )
            colors_transferred = False
            if not has_colors and orig_colors is not None:
                if len(result_mesh.vertices) == len(orig_colors):
                    result_mesh.vertex_attributes['COLOR_0'] = orig_colors
                    colors_transferred = True
                else:
                    try:
                        from scipy.spatial import cKDTree
                        tree = cKDTree(mesh.vertices)
                        _, indices = tree.query(result_mesh.vertices, k=1)
                        result_mesh.vertex_attributes['COLOR_0'] = orig_colors[indices]
                        colors_transferred = True
                    except ImportError:
                        print("[BD UV Unwrap] Warning: scipy not available for color restore")

            status = (f"UV unwrapped (Blender): {n_result_verts:,} verts (no split) "
                      f"| {n_result_faces:,} faces")
            if has_uvs and uvs_loop is not None:
                status += f" | {len(uvs_loop)} UV loops"
            if has_colors or colors_transferred:
                status += " | colors ok"
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
