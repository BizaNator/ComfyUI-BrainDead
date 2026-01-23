"""
V3 API Mesh Inspector node for interactive 3D mesh inspection with PBR channel switching.

BD_MeshInspector - Three.js-based viewer with switchable visualization modes:
Full Material, Geometry, Vertex Colors, UV, Normal, Metallic, Roughness, Alpha, Emissive, Diffuse.
"""

import os
import uuid
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
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except (ImportError, AttributeError):
    COMFYUI_OUTPUT_FOLDER = None


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


def _encode_image_to_base64(image_tensor) -> str:
    """Convert ComfyUI image tensor (B,H,W,C float 0-1) to base64 PNG string."""
    from PIL import Image

    img_np = (image_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    buffer = io_module.BytesIO()
    img.save(buffer, format='PNG', optimize=True)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _encode_numpy_to_base64(arr) -> str:
    """Convert numpy uint8 array (H,W,C) to base64 PNG string."""
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
        arr = arr[:, :, :3]
    img = Image.fromarray(arr.astype(np.uint8))
    buffer = io_module.BytesIO()
    img.save(buffer, format='PNG', optimize=True)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


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
    def execute(cls, mesh=None, bundle=None, mesh_path="",
                initial_mode="full_material",
                metallic_json="", roughness_json="", normal_map=None,
                emissive_map=None, alpha_map=None, diffuse_map=None) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput("ERROR: trimesh not installed")

        # Extract data from bundle if provided (individual inputs override)
        if bundle is not None and isinstance(bundle, dict):
            if mesh is None:
                mesh = bundle.get('mesh')
            if not metallic_json and bundle.get('metallic') is not None:
                # Bundle has metallic as texture, not per-vertex JSON
                pass  # Handled below as metallic_map
            if not roughness_json and bundle.get('roughness') is not None:
                # Bundle has roughness as texture, not per-vertex JSON
                pass  # Handled below as roughness_map
            if normal_map is None and bundle.get('normal') is not None:
                normal_map = bundle.get('normal')  # numpy array
            if alpha_map is None and bundle.get('alpha') is not None:
                alpha_map = bundle.get('alpha')  # numpy array
            if diffuse_map is None and bundle.get('diffuse') is not None:
                diffuse_map = bundle.get('diffuse')  # numpy array

        # Load from file path if no mesh from direct input or bundle
        if mesh is None and mesh_path and mesh_path.strip():
            mesh_path = mesh_path.strip()
            if os.path.isfile(mesh_path):
                try:
                    mesh = trimesh.load(mesh_path, force='mesh')
                    print(f"[BD Inspector] Loaded mesh from file: {mesh_path}")
                except Exception as e:
                    return io.NodeOutput(f"ERROR: Failed to load mesh - {e}")
            else:
                return io.NodeOutput(f"ERROR: Mesh file not found - {mesh_path}")

        if mesh is None:
            return io.NodeOutput("ERROR: No mesh (provide mesh, bundle, or mesh_path). If connected, check upstream node produced a valid mesh.")

        # Generate unique filename for this preview
        filename = f"bd_inspector_{uuid.uuid4().hex[:8]}.glb"

        # Determine output directory
        output_dir = COMFYUI_OUTPUT_FOLDER
        if not output_dir:
            import tempfile
            output_dir = tempfile.gettempdir()

        filepath = os.path.join(output_dir, filename)

        # Export mesh to GLB (preserves vertex colors and UVs)
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

        # Encode texture maps to base64
        # Handles both ComfyUI IMAGE tensors and numpy uint8 arrays (from bundle)
        def _encode_map(map_data, label):
            if map_data is None:
                return ""
            try:
                if isinstance(map_data, np.ndarray):
                    return _encode_numpy_to_base64(map_data)
                else:
                    return _encode_image_to_base64(map_data)
            except Exception as e:
                print(f"[BD Inspector] {label} encoding failed: {e}")
                return ""

        normal_b64 = _encode_map(normal_map, "Normal map")
        emissive_b64 = _encode_map(emissive_map, "Emissive map")
        alpha_b64 = _encode_map(alpha_map, "Alpha map")
        diffuse_b64 = _encode_map(diffuse_map, "Diffuse map")

        # Bundle metallic/roughness textures (separate from per-vertex JSON)
        metallic_map_b64 = ""
        roughness_map_b64 = ""
        if bundle is not None and isinstance(bundle, dict):
            if not metallic_out and bundle.get('metallic') is not None:
                metallic_map_b64 = _encode_map(bundle.get('metallic'), "Metallic map (bundle)")
            if not roughness_out and bundle.get('roughness') is not None:
                roughness_map_b64 = _encode_map(bundle.get('roughness'), "Roughness map (bundle)")

        # Check if mesh has UVs
        has_uvs = (hasattr(mesh, 'visual')
                   and hasattr(mesh.visual, 'uv')
                   and mesh.visual.uv is not None
                   and len(mesh.visual.uv) > 0)

        # Build status string
        vert_count = len(mesh.vertices)
        face_count = len(mesh.faces) if hasattr(mesh, 'faces') and mesh.faces is not None else 0
        has_colors = hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors')
        channels = []
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

        if bundle is not None and mesh is bundle.get('mesh'):
            source = "bundle"
        elif mesh_path and mesh_path.strip():
            source = "file"
        else:
            source = "mesh"
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
                "initial_mode": [initial_mode],
                "metallic_json": [metallic_out],
                "roughness_json": [roughness_out],
                "metallic_map_b64": [metallic_map_b64],
                "roughness_map_b64": [roughness_map_b64],
                "normal_map_b64": [normal_b64],
                "emissive_map_b64": [emissive_b64],
                "alpha_map_b64": [alpha_b64],
                "diffuse_map_b64": [diffuse_b64],
                "has_uvs": [has_uvs],
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
