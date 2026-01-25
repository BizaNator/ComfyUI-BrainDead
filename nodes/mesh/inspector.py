"""
V3 API Mesh Inspector node for interactive 3D mesh inspection with PBR channel switching.

BD_MeshInspector - Three.js-based viewer with switchable visualization modes:
Full Material, Geometry, Vertex Colors, UV, Normal, Metallic, Roughness, Alpha, Emissive, Diffuse.
"""

import os
import hashlib
import json
import base64
import io as io_module
import numpy as np

from comfy_api.latest import io

from .types import TrimeshInput, MeshBundleInput

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import folder_paths
    COMFYUI_TEMP_FOLDER = folder_paths.get_temp_directory()
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
    COMFYUI_INPUT_FOLDER = folder_paths.get_input_directory()
except (ImportError, AttributeError):
    COMFYUI_TEMP_FOLDER = None
    COMFYUI_OUTPUT_FOLDER = None
    COMFYUI_INPUT_FOLDER = None


VIEW_MODES = [
    "full_material",
    "geometry",
    "vertex_colors",
    "uv",
    "normal",
    "metallic",
    "roughness",
    "alpha",
    "emissive",
    "diffuse",
]


INSPECTOR_MAX_TEX_SIZE = 1024  # Max texture resolution for preview viewer


def _encode_image_to_base64(image_tensor, use_jpeg=True) -> str:
    """Convert ComfyUI image tensor (B,H,W,C float 0-1) to base64 string.
    Downsamples to INSPECTOR_MAX_TEX_SIZE and uses JPEG for speed."""
    from PIL import Image

    img_np = (image_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    # Downsample for preview
    if max(img.size) > INSPECTOR_MAX_TEX_SIZE:
        img.thumbnail((INSPECTOR_MAX_TEX_SIZE, INSPECTOR_MAX_TEX_SIZE), Image.LANCZOS)
    buffer = io_module.BytesIO()
    if use_jpeg:
        img = img.convert('RGB')
        img.save(buffer, format='JPEG', quality=80)
    else:
        img.save(buffer, format='PNG', optimize=True)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _encode_numpy_to_base64(arr, use_jpeg=True) -> str:
    """Convert numpy uint8 array (H,W,C) to base64 string.
    Downsamples to INSPECTOR_MAX_TEX_SIZE and uses JPEG for speed."""
    from PIL import Image

    if arr is None:
        return ""
    if not isinstance(arr, np.ndarray):
        return ""
    # Handle single-channel images
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        # Keep alpha only if not using JPEG
        if use_jpeg:
            arr = arr[:, :, :3]
        # else keep RGBA for PNG
    img = Image.fromarray(arr.astype(np.uint8))
    # Downsample for preview
    if max(img.size) > INSPECTOR_MAX_TEX_SIZE:
        img.thumbnail((INSPECTOR_MAX_TEX_SIZE, INSPECTOR_MAX_TEX_SIZE), Image.LANCZOS)
    buffer = io_module.BytesIO()
    if use_jpeg:
        img = img.convert('RGB')
        img.save(buffer, format='JPEG', quality=80)
    else:
        img.save(buffer, format='PNG', optimize=True)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _sample_color_field(mesh, color_field) -> np.ndarray:
    """Sample a COLOR_FIELD onto mesh vertices using KD-tree nearest neighbor.

    Returns RGBA uint8 array (N, 4) or None on failure.
    """
    try:
        if mesh is None or color_field is None:
            return None

        positions = color_field.get('positions')
        colors = color_field.get('colors')
        voxel_size = color_field.get('voxel_size', 1.0)

        if positions is None or colors is None:
            return None

        # Ensure numpy
        if hasattr(positions, 'cpu'):
            positions = positions.cpu().numpy()
        positions = np.asarray(positions, dtype=np.float32)

        if hasattr(colors, 'cpu'):
            colors = colors.cpu().numpy()
        colors = np.asarray(colors, dtype=np.float32)

        # Add alpha if RGB only
        if colors.shape[1] == 3:
            colors = np.hstack([colors, np.ones((len(colors), 1), dtype=np.float32)])

        # Transform mesh vertices to voxel space (TRELLIS coordinate system)
        mesh_verts = np.array(mesh.vertices, dtype=np.float32)
        verts_transformed = mesh_verts.copy()
        verts_transformed[:, 1] = -mesh_verts[:, 2]
        verts_transformed[:, 2] = mesh_verts[:, 1]
        verts_in_voxel_space = verts_transformed + 0.5

        # KD-tree nearest neighbor sampling
        from scipy.spatial import cKDTree
        tree = cKDTree(positions)
        distances, indices = tree.query(verts_in_voxel_space, k=1, workers=-1)

        vertex_colors = colors[indices].copy()

        # Threshold: vertices too far get gray
        max_dist = voxel_size * 3.0
        far_verts = distances > max_dist
        vertex_colors[far_verts] = [0.5, 0.5, 0.5, 1.0]

        # Convert to uint8
        vertex_colors_uint8 = (vertex_colors * 255).clip(0, 255).astype(np.uint8)
        print(f"[BD Inspector] Sampled {len(vertex_colors_uint8)} colors from color_field ({len(positions)} voxels)")
        return vertex_colors_uint8

    except Exception as e:
        print(f"[BD Inspector] Color field sampling failed: {e}")
        return None


class BD_MeshInspector(io.ComfyNode):
    """
    Interactive 3D mesh inspector with PBR channel visualization.

    Displays mesh in a Three.js viewer with switchable view modes:
    Full Material, Geometry Only, Vertex Colors, UV, Normal, Metallic,
    Roughness, Alpha, Emissive, and Diffuse.

    Accepts either individual inputs or a MESH_BUNDLE (from BD Pack/Cache Bundle).
    Bundle data is used as fallback when individual inputs are not connected.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_MeshInspector",
            display_name="BD Mesh Inspector",
            category="\U0001f9e0BrainDead/Mesh",
            description="Interactive 3D mesh viewer with PBR channel switching. Accepts mesh directly or via MESH_BUNDLE. Inspect UVs, vertex colors, normals, metallic, roughness, alpha, emissive, and diffuse.",
            is_output_node=True,
            inputs=[
                TrimeshInput("mesh", optional=True),
                MeshBundleInput("bundle", optional=True),
                io.String.Input("mesh_path", default="", optional=True, multiline=False,
                                tooltip="Path to a mesh file (GLB, PLY, OBJ, FBX, STL, OFF, etc.)"),
                io.Combo.Input("initial_mode", options=VIEW_MODES, default="full_material",
                               tooltip="Initial view mode for the 3D viewer"),
                io.String.Input("metallic_json", default="", optional=True, multiline=False,
                                tooltip="JSON array of per-vertex metallic values (from BD Sample Voxelgrid PBR)"),
                io.String.Input("roughness_json", default="", optional=True, multiline=False,
                                tooltip="JSON array of per-vertex roughness values (from BD Sample Voxelgrid PBR)"),
                io.Image.Input("normal_map", optional=True,
                               tooltip="Normal map texture"),
                io.Image.Input("emissive_map", optional=True,
                               tooltip="Emissive map texture"),
                io.Image.Input("alpha_map", optional=True,
                               tooltip="Alpha/opacity map texture"),
                io.Image.Input("diffuse_map", optional=True,
                               tooltip="Diffuse/albedo texture"),
            ],
            outputs=[
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def _mesh_hash(cls, mesh) -> str:
        """Generate a short hash from mesh geometry for dedup filenames."""
        verts = mesh.vertices
        h = hashlib.md5()
        h.update(f"{len(verts)}_{len(mesh.faces) if mesh.faces is not None else 0}".encode())
        # Sample a few vertices for fast hashing
        if len(verts) > 0:
            h.update(verts[0].tobytes())
            h.update(verts[-1].tobytes())
            h.update(verts[len(verts) // 2].tobytes())
        return h.hexdigest()[:12]

    @classmethod
    def _resolve_servable_path(cls, filepath) -> tuple:
        """Check if a file is in a ComfyUI-servable directory.
        Returns (filename, view_type, subfolder) or None.
        """
        abs_path = os.path.abspath(filepath)
        for folder, view_type in [
            (COMFYUI_OUTPUT_FOLDER, "output"),
            (COMFYUI_INPUT_FOLDER, "input"),
            (COMFYUI_TEMP_FOLDER, "temp"),
        ]:
            if folder and abs_path.startswith(os.path.abspath(folder) + os.sep):
                rel = os.path.relpath(abs_path, folder)
                subfolder = os.path.dirname(rel)
                fname = os.path.basename(rel)
                return (fname, view_type, subfolder if subfolder != '.' else '')
        return None

    @classmethod
    def execute(cls, mesh=None, bundle=None, mesh_path="",
                initial_mode="full_material",
                metallic_json="", roughness_json="", normal_map=None,
                emissive_map=None, alpha_map=None, diffuse_map=None) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput("ERROR: trimesh not installed")

        # Extract PBR data from bundle (always, regardless of mesh source)
        bundle_vertex_colors = None
        if bundle is not None and isinstance(bundle, dict):
            # Extract vertex colors from bundle
            vc = bundle.get('vertex_colors')
            if vc is not None and isinstance(vc, np.ndarray) and len(vc) > 0 and vc.max() > 0:
                bundle_vertex_colors = vc
            # PBR textures from bundle (individual inputs override)
            if normal_map is None and bundle.get('normal') is not None:
                normal_map = bundle.get('normal')
            if alpha_map is None and bundle.get('alpha') is not None:
                alpha_map = bundle.get('alpha')
            if diffuse_map is None and bundle.get('diffuse') is not None:
                diffuse_map = bundle.get('diffuse')

        # Determine mesh source (priority: mesh > mesh_path > bundle.mesh)
        serve_existing = None
        source = "mesh"
        if mesh is not None:
            source = "mesh"
        elif mesh_path and mesh_path.strip():
            mesh_path = mesh_path.strip()
            if os.path.isfile(mesh_path):
                ext = os.path.splitext(mesh_path)[1].lower()
                if ext in ('.glb', '.gltf'):
                    serve_existing = cls._resolve_servable_path(mesh_path)
                try:
                    mesh = trimesh.load(mesh_path, force='mesh')
                    source = "file"
                    print(f"[BD Inspector] Loaded mesh from file: {mesh_path}")
                except Exception as e:
                    return io.NodeOutput(f"ERROR: Failed to load mesh - {e}")
            else:
                return io.NodeOutput(f"ERROR: Mesh file not found - {mesh_path}")
        elif bundle is not None and isinstance(bundle, dict) and bundle.get('mesh') is not None:
            mesh = bundle.get('mesh')
            source = "bundle"
        else:
            return io.NodeOutput("ERROR: No mesh (provide mesh, bundle, or mesh_path). If connected, check upstream node produced a valid mesh.")

        # Sample color_field onto mesh vertices if no vertex colors yet
        if bundle_vertex_colors is None and bundle is not None and isinstance(bundle, dict):
            if bundle.get('color_field') is not None:
                bundle_vertex_colors = _sample_color_field(mesh, bundle['color_field'])

        # Determine how to serve the mesh to the viewer
        subfolder = ""
        if serve_existing:
            filename, view_type, subfolder = serve_existing
        else:
            # Export to temp directory with content-hash filename (no duplicates)
            mesh_hash = cls._mesh_hash(mesh)
            filename = f"bd_inspector_{mesh_hash}.glb"
            view_type = "temp"

            temp_dir = COMFYUI_TEMP_FOLDER
            if not temp_dir:
                import tempfile
                temp_dir = tempfile.gettempdir()
            os.makedirs(temp_dir, exist_ok=True)

            filepath = os.path.join(temp_dir, filename)

            # Only export if file doesn't already exist (hash-based dedup)
            if not os.path.isfile(filepath):
                try:
                    mesh.export(filepath, file_type='glb')
                except Exception as e:
                    print(f"[BD Inspector] GLB export failed: {e}, trying OBJ fallback")
                    filename = filename.replace('.glb', '.obj')
                    filepath = filepath.replace('.glb', '.obj')
                    try:
                        mesh.export(filepath, file_type='obj')
                    except Exception as e2:
                        return io.NodeOutput(f"ERROR: Export failed - {e2}")

        # Truncate PBR arrays to 3 decimal places for size optimization
        metallic_out = ""
        if metallic_json and metallic_json.strip():
            try:
                arr = json.loads(metallic_json)
                metallic_out = json.dumps([round(v, 3) for v in arr])
            except (json.JSONDecodeError, TypeError):
                metallic_out = metallic_json

        roughness_out = ""
        if roughness_json and roughness_json.strip():
            try:
                arr = json.loads(roughness_json)
                roughness_out = json.dumps([round(v, 3) for v in arr])
            except (json.JSONDecodeError, TypeError):
                roughness_out = roughness_json

        # Encode texture maps to base64 (downsampled + JPEG for speed)
        # Handles both ComfyUI IMAGE tensors and numpy uint8 arrays (from bundle)
        def _encode_map(map_data, label, use_jpeg=True):
            if map_data is None:
                return ""
            try:
                if isinstance(map_data, np.ndarray):
                    return _encode_numpy_to_base64(map_data, use_jpeg=use_jpeg)
                else:
                    return _encode_image_to_base64(map_data, use_jpeg=use_jpeg)
            except Exception as e:
                print(f"[BD Inspector] {label} encoding failed: {e}")
                return ""

        normal_b64 = _encode_map(normal_map, "Normal map")
        emissive_b64 = _encode_map(emissive_map, "Emissive map")
        alpha_b64 = _encode_map(alpha_map, "Alpha map", use_jpeg=False)  # PNG for alpha
        diffuse_b64 = _encode_map(diffuse_map, "Diffuse map")

        # Bundle metallic/roughness textures (separate from per-vertex JSON)
        metallic_map_b64 = ""
        roughness_map_b64 = ""
        if bundle is not None and isinstance(bundle, dict):
            if not metallic_out and bundle.get('metallic') is not None:
                metallic_map_b64 = _encode_map(bundle.get('metallic'), "Metallic map (bundle)")
            if not roughness_out and bundle.get('roughness') is not None:
                roughness_map_b64 = _encode_map(bundle.get('roughness'), "Roughness map (bundle)")

        # Encode bundle vertex colors as base64 raw RGBA bytes
        vertex_colors_b64 = ""
        if bundle_vertex_colors is not None:
            try:
                vc = bundle_vertex_colors
                # Ensure RGBA uint8 format
                if vc.dtype != np.uint8:
                    vc = (vc * 255).clip(0, 255).astype(np.uint8)
                if vc.shape[-1] == 3:
                    # Add alpha channel
                    alpha = np.full((len(vc), 1), 255, dtype=np.uint8)
                    vc = np.hstack([vc, alpha])
                vertex_colors_b64 = base64.b64encode(vc.tobytes()).decode('utf-8')
                print(f"[BD Inspector] Encoded {len(vc)} vertex colors from bundle ({len(vertex_colors_b64)//1024}KB)")
            except Exception as e:
                print(f"[BD Inspector] Vertex color encoding failed: {e}")

        # Check if mesh has UVs
        has_uvs = (hasattr(mesh, 'visual')
                   and hasattr(mesh.visual, 'uv')
                   and mesh.visual.uv is not None
                   and len(mesh.visual.uv) > 0)

        # Indicate combined source when file + bundle PBR
        if source == "file" and bundle is not None:
            source = "file+bundle"

        # Build status string
        vert_count = len(mesh.vertices)
        face_count = len(mesh.faces) if hasattr(mesh, 'faces') and mesh.faces is not None else 0
        has_colors = (
            (hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors')
             and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) > 0)
            or bool(vertex_colors_b64)
        )
        channels = []
        if has_colors:
            channels.append("vertex_colors")
        if metallic_out or metallic_map_b64:
            channels.append("metallic")
        if roughness_out or roughness_map_b64:
            channels.append("roughness")
        if normal_b64:
            channels.append("normal")
        if emissive_b64:
            channels.append("emissive")
        if alpha_b64:
            channels.append("alpha")
        if diffuse_b64:
            channels.append("diffuse")
        if has_uvs:
            channels.append("uv")

        status = (
            f"{vert_count} verts, {face_count} faces | "
            f"Source: {source} | "
            f"Colors: {'yes' if has_colors else 'no'} | "
            f"UVs: {'yes' if has_uvs else 'no'} | "
            f"Channels: {', '.join(channels) if channels else 'none'}"
        )
        print(f"[BD Inspector] {status}")

        return io.NodeOutput(
            status,
            ui={
                "mesh_file": [filename],
                "view_type": [view_type],
                "subfolder": [subfolder],
                "initial_mode": [initial_mode],
                "metallic_json": [metallic_out],
                "roughness_json": [roughness_out],
                "metallic_map_b64": [metallic_map_b64],
                "roughness_map_b64": [roughness_map_b64],
                "normal_map_b64": [normal_b64],
                "emissive_map_b64": [emissive_b64],
                "alpha_map_b64": [alpha_b64],
                "diffuse_map_b64": [diffuse_b64],
                "vertex_colors_b64": [vertex_colors_b64],
                "has_uvs": [has_uvs],
                "has_colors": [has_colors],
                "vertex_count": [vert_count],
                "face_count": [face_count],
            }
        )


# V3 node list for extension
MESH_INSPECTOR_V3_NODES = [BD_MeshInspector]

# V1 compatibility
MESH_INSPECTOR_NODES = {
    "BD_MeshInspector": BD_MeshInspector,
}

MESH_INSPECTOR_DISPLAY_NAMES = {
    "BD_MeshInspector": "BD Mesh Inspector",
}
