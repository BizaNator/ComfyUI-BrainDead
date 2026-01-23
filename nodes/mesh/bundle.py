"""
BD_PackBundle - Pack mesh + textures + colors into a MESH_BUNDLE.
BD_CacheBundle - Cache a MESH_BUNDLE to skip expensive upstream computation.

MESH_BUNDLE is a container carrying:
- mesh (TRIMESH with material/UVs)
- color_field (COLOR_FIELD data)
- vertex_colors (COLOR_0 attribute data)
- PBR textures (diffuse, normal, metallic, roughness, alpha)
- name (for export filenames)
"""

import os
import pickle
import time

import numpy as np
import torch

from comfy_api.latest import io

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

from ...utils.shared import (
    get_cache_path,
    hash_from_seed,
    check_cache_exists,
)

from .types import (
    TrimeshInput,
    ColorFieldInput,
    MeshBundleInput, MeshBundleOutput,
)


def _image_tensor_to_numpy(tensor):
    """Convert ComfyUI IMAGE tensor [B,H,W,C] to numpy uint8 (H,W,C)."""
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        if tensor.dim() == 4:
            tensor = tensor[0]  # First batch
        arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    elif isinstance(tensor, np.ndarray):
        if tensor.dtype == np.float32 or tensor.dtype == np.float64:
            arr = (tensor * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = tensor
    else:
        return None
    return arr


def _extract_textures_from_mesh(mesh):
    """Extract PBR textures from mesh's TextureVisuals if present."""
    textures = {}
    if not (hasattr(mesh, 'visual')
            and hasattr(mesh.visual, 'material')
            and mesh.visual.material is not None):
        return textures

    mat = mesh.visual.material

    # Try to get baseColorTexture (diffuse/albedo)
    if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
        try:
            img = mat.baseColorTexture
            if hasattr(img, 'size'):  # PIL Image
                textures['diffuse'] = np.array(img.convert('RGB')).astype(np.uint8)
        except Exception:
            pass

    # normalTexture
    if hasattr(mat, 'normalTexture') and mat.normalTexture is not None:
        try:
            img = mat.normalTexture
            if hasattr(img, 'size'):
                textures['normal'] = np.array(img.convert('RGB')).astype(np.uint8)
        except Exception:
            pass

    # metallicRoughnessTexture (combined in glTF: R=0, G=roughness, B=metallic)
    if hasattr(mat, 'metallicRoughnessTexture') and mat.metallicRoughnessTexture is not None:
        try:
            img = mat.metallicRoughnessTexture
            if hasattr(img, 'size'):
                mr = np.array(img.convert('RGB')).astype(np.uint8)
                textures['roughness'] = mr[:, :, 1:2]  # Green channel
                textures['metallic'] = mr[:, :, 2:3]   # Blue channel
        except Exception:
            pass

    return textures


class BD_PackBundle(io.ComfyNode):
    """
    Pack mesh + textures + color data into a MESH_BUNDLE.

    Collects all mesh asset data into a single container for caching and export.
    Textures can be provided explicitly or extracted from the mesh's material.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_PackBundle",
            display_name="BD Pack Bundle",
            category="🧠BrainDead/Mesh",
            description="""Pack mesh + textures + colors into a MESH_BUNDLE container.

MESH_BUNDLE carries everything needed for export:
- Mesh geometry with PBR material and UVs
- Vertex colors (for COLOR_0 attribute / edge detection / stylized look)
- Color field data (for downstream reapplication)
- PBR textures (diffuse, normal, metallic, roughness, alpha)

Textures are extracted from mesh material if not provided explicitly.""",
            inputs=[
                TrimeshInput("mesh"),
                ColorFieldInput("color_field", optional=True),
                io.Image.Input("diffuse", optional=True),
                io.Image.Input("normal", optional=True),
                io.Image.Input("metallic", optional=True),
                io.Image.Input("roughness", optional=True),
                io.Image.Input("alpha", optional=True),
                io.String.Input(
                    "name",
                    default="mesh",
                    tooltip="Bundle name (used for export filenames)",
                ),
            ],
            outputs=[
                MeshBundleOutput(display_name="bundle"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        color_field=None,
        diffuse=None,
        normal=None,
        metallic=None,
        roughness=None,
        alpha=None,
        name: str = "mesh",
    ) -> io.NodeOutput:
        if mesh is None:
            return io.NodeOutput(None, "ERROR: No input mesh")

        # Extract vertex colors from metadata if present
        vertex_colors = None
        if hasattr(mesh, 'metadata') and mesh.metadata:
            vertex_colors = mesh.metadata.get('vertex_colors')

        # If no vertex_colors in metadata, try from ColorVisuals
        if vertex_colors is None and hasattr(mesh, 'visual'):
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                vc = mesh.visual.vertex_colors
                if isinstance(vc, np.ndarray) and len(vc) > 0:
                    vertex_colors = vc if vc.dtype == np.uint8 else (vc * 255).clip(0, 255).astype(np.uint8)

        # Convert explicit texture inputs
        tex_diffuse = _image_tensor_to_numpy(diffuse)
        tex_normal = _image_tensor_to_numpy(normal)
        tex_metallic = _image_tensor_to_numpy(metallic)
        tex_roughness = _image_tensor_to_numpy(roughness)
        tex_alpha = _image_tensor_to_numpy(alpha)

        # If no explicit textures, try extracting from mesh material
        if tex_diffuse is None:
            extracted = _extract_textures_from_mesh(mesh)
            if extracted:
                tex_diffuse = extracted.get('diffuse', tex_diffuse)
                tex_normal = extracted.get('normal', tex_normal)
                tex_metallic = extracted.get('metallic', tex_metallic)
                tex_roughness = extracted.get('roughness', tex_roughness)

        # Build bundle
        bundle = {
            'mesh': mesh,
            'color_field': color_field,
            'vertex_colors': vertex_colors,
            'diffuse': tex_diffuse,
            'normal': tex_normal,
            'metallic': tex_metallic,
            'roughness': tex_roughness,
            'alpha': tex_alpha,
            'name': name,
        }

        # Status
        n_verts = len(mesh.vertices) if hasattr(mesh, 'vertices') else 0
        n_faces = len(mesh.faces) if hasattr(mesh, 'faces') and mesh.faces is not None else 0
        parts = [f"{name}: {n_verts:,} verts, {n_faces:,} faces"]
        if vertex_colors is not None:
            parts.append(f"vcol({len(vertex_colors):,})")
        if color_field is not None:
            parts.append("color_field")
        tex_count = sum(1 for t in [tex_diffuse, tex_normal, tex_metallic, tex_roughness, tex_alpha] if t is not None)
        if tex_count > 0:
            tex_res = f"{tex_diffuse.shape[1]}x{tex_diffuse.shape[0]}" if tex_diffuse is not None else "?"
            parts.append(f"{tex_count} textures ({tex_res})")
        has_mat = hasattr(mesh.visual, 'material') and mesh.visual.material is not None
        if has_mat:
            parts.append("+material")

        status = " | ".join(parts)
        print(f"[BD PackBundle] {status}")
        return io.NodeOutput(bundle, status)


class BD_CacheBundle(io.ComfyNode):
    """
    Cache a MESH_BUNDLE to skip expensive upstream computation.

    Saves mesh as GLB, textures as PNG, metadata as pickle.
    Uses lazy evaluation to skip upstream when cache exists.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CacheBundle",
            display_name="BD Cache Bundle",
            category="🧠BrainDead/Mesh",
            description="""Cache MESH_BUNDLE to skip expensive generation.

Saves:
- Mesh as GLB (preserves material, UVs, vertex colors)
- Textures as individual PNGs (lossless)
- Color field + vertex colors + metadata as pickle

On cache hit, loads everything back as a complete MESH_BUNDLE.""",
            inputs=[
                MeshBundleInput("bundle", lazy=True),
                io.String.Input("cache_name", default="bundle"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Boolean.Input("force_refresh", default=False),
                io.String.Input("name_prefix", default="", optional=True),
            ],
            outputs=[
                MeshBundleOutput(display_name="bundle"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, cache_name: str, seed: int, force_refresh: bool,
                           bundle=None, name_prefix: str = "") -> str:
        if force_refresh:
            return f"force_{time.time()}"
        return f"{name_prefix}_{cache_name}_{seed}"

    @classmethod
    def check_lazy_status(cls, cache_name: str, seed: int, force_refresh: bool,
                          bundle=None, name_prefix: str = "") -> list[str]:
        if force_refresh:
            return ["bundle"]

        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        mesh_path = get_cache_path(full_name, cache_hash, "_mesh.glb")

        if check_cache_exists(mesh_path, min_size=100):
            print(f"[BD CacheBundle] Cache exists - SKIPPING upstream")
            return []
        print(f"[BD CacheBundle] No cache - will run upstream")
        return ["bundle"]

    @classmethod
    def execute(cls, bundle, cache_name: str, seed: int, force_refresh: bool,
                name_prefix: str = "") -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput(bundle, "ERROR: trimesh not installed")

        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        base_path = get_cache_path(full_name, cache_hash, "")  # No extension

        mesh_path = base_path + "_mesh.glb"
        meta_path = base_path + "_meta.pkl"
        tex_names = ['diffuse', 'normal', 'metallic', 'roughness', 'alpha']

        # Try loading from cache
        if check_cache_exists(mesh_path, min_size=100) and not force_refresh:
            try:
                return cls._load_cache(mesh_path, meta_path, base_path, tex_names)
            except Exception as e:
                print(f"[BD CacheBundle] Cache load failed: {e}")

        # No cache or load failed - save new cache
        if bundle is None:
            return io.NodeOutput(None, "ERROR: No bundle to cache")

        try:
            return cls._save_cache(bundle, mesh_path, meta_path, base_path, tex_names)
        except Exception as e:
            print(f"[BD CacheBundle] Save error: {e}")
            return io.NodeOutput(bundle, f"Cache save failed: {e}")

    @classmethod
    def _load_cache(cls, mesh_path, meta_path, base_path, tex_names):
        """Load bundle from cache files."""
        from PIL import Image

        # Load mesh
        mesh = trimesh.load(mesh_path, file_type='glb', force='mesh')

        # Load metadata (color_field, vertex_colors, name)
        color_field = None
        vertex_colors = None
        name = "mesh"
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            color_field = meta.get('color_field')
            vertex_colors = meta.get('vertex_colors')
            name = meta.get('name', 'mesh')
            # Restore mesh metadata if present
            mesh_meta = meta.get('mesh_metadata')
            if mesh_meta and isinstance(mesh_meta, dict):
                mesh.metadata.update(mesh_meta)

        # Load textures
        textures = {}
        for tex_name in tex_names:
            tex_path = base_path + f"_{tex_name}.png"
            if os.path.exists(tex_path):
                img = Image.open(tex_path)
                textures[tex_name] = np.array(img).astype(np.uint8)

        bundle = {
            'mesh': mesh,
            'color_field': color_field,
            'vertex_colors': vertex_colors,
            'diffuse': textures.get('diffuse'),
            'normal': textures.get('normal'),
            'metallic': textures.get('metallic'),
            'roughness': textures.get('roughness'),
            'alpha': textures.get('alpha'),
            'name': name,
        }

        n_verts = len(mesh.vertices)
        tex_count = sum(1 for v in textures.values() if v is not None)
        has_mat = hasattr(mesh.visual, 'material') and mesh.visual.material is not None
        size_mb = os.path.getsize(mesh_path) / (1024 * 1024)

        parts = [f"HIT: {name}", f"{n_verts:,} verts", f"{size_mb:.1f}MB"]
        if has_mat:
            parts.append("+material")
        if vertex_colors is not None:
            parts.append("+vcol")
        if tex_count > 0:
            parts.append(f"+{tex_count}tex")

        status = " | ".join(parts)
        print(f"[BD CacheBundle] {status}")
        return io.NodeOutput(bundle, status)

    @classmethod
    def _save_cache(cls, bundle, mesh_path, meta_path, base_path, tex_names):
        """Save bundle to cache files."""
        from PIL import Image

        mesh = bundle.get('mesh')
        if mesh is None:
            return io.NodeOutput(bundle, "ERROR: Bundle has no mesh")

        # Save mesh as GLB
        mesh.export(mesh_path, file_type='glb')

        # Save textures as PNG
        tex_count = 0
        for tex_name in tex_names:
            tex_data = bundle.get(tex_name)
            if tex_data is not None:
                tex_path = base_path + f"_{tex_name}.png"
                img = Image.fromarray(tex_data)
                img.save(tex_path)
                tex_count += 1

        # Save metadata
        meta = {
            'color_field': bundle.get('color_field'),
            'vertex_colors': bundle.get('vertex_colors'),
            'name': bundle.get('name', 'mesh'),
        }
        # Also save mesh.metadata if non-empty
        if mesh.metadata:
            serializable_meta = {}
            for key, val in mesh.metadata.items():
                if isinstance(val, (np.ndarray, str, int, float, bool, list, dict)):
                    serializable_meta[key] = val
            if serializable_meta:
                meta['mesh_metadata'] = serializable_meta

        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)

        size_mb = os.path.getsize(mesh_path) / (1024 * 1024)
        n_verts = len(mesh.vertices)
        parts = [f"SAVED: {bundle.get('name', 'mesh')}", f"{n_verts:,} verts", f"{size_mb:.1f}MB"]
        if tex_count > 0:
            parts.append(f"+{tex_count}tex")
        if bundle.get('vertex_colors') is not None:
            parts.append("+vcol")

        status = " | ".join(parts)
        print(f"[BD CacheBundle] {status}")
        return io.NodeOutput(bundle, status)


# V3 node list
BUNDLE_V3_NODES = [BD_PackBundle, BD_CacheBundle]

# V1 compatibility
BUNDLE_NODES = {
    "BD_PackBundle": BD_PackBundle,
    "BD_CacheBundle": BD_CacheBundle,
}

BUNDLE_DISPLAY_NAMES = {
    "BD_PackBundle": "BD Pack Bundle",
    "BD_CacheBundle": "BD Cache Bundle",
}
