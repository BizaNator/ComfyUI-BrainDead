"""
TRELLIS2 cache nodes for caching conditioning, shape, and texture outputs.

BD_CacheTrellis2Conditioning - Cache conditioning to skip image preprocessing
BD_CacheTrellis2Shape - Cache shape + mesh to skip expensive shape generation
BD_CacheTrellis2Texture - Cache texture outputs (trimesh, voxelgrid, pointcloud)
"""

import os
import time

from ...utils.shared import (
    LAZY_OPTIONS,
    get_cache_path,
    hash_from_seed,
    check_cache_exists,
    save_to_cache,
    PickleSerializer,
)

# Check for optional trimesh support
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


class BD_CacheTrellis2Conditioning:
    """Cache Trellis2 conditioning output to skip image preprocessing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("TRELLIS2_CONDITIONING", LAZY_OPTIONS),
                "cache_name": ("STRING", {"default": "trellis2_cond"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_refresh": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "name_prefix": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("TRELLIS2_CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "status")
    FUNCTION = "cache_conditioning"
    CATEGORY = "BrainDead/TRELLIS2"
    DESCRIPTION = """
Cache Trellis2 conditioning to skip image preprocessing.

Place AFTER Trellis2GetConditioning node.
"""

    @classmethod
    def IS_CHANGED(cls, cache_name, seed, force_refresh, conditioning=None, name_prefix=""):
        if force_refresh:
            return f"force_{time.time()}"
        return f"{name_prefix}_{cache_name}_{seed}"

    def check_lazy_status(self, cache_name, seed, force_refresh, conditioning=None, name_prefix=""):
        """Return [] to skip upstream, ["conditioning"] to evaluate upstream."""
        if force_refresh:
            print(f"[BD Trellis2 Conditioning] Force refresh - will run upstream")
            return ["conditioning"]

        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, ".pkl")

        if check_cache_exists(cache_path, min_size=100):
            print(f"[BD Trellis2 Conditioning] Cache exists - SKIPPING upstream execution")
            return []  # Empty list = don't need input, skip upstream
        print(f"[BD Trellis2 Conditioning] No cache found - will run upstream")
        return ["conditioning"]

    def cache_conditioning(self, conditioning, cache_name, seed, force_refresh, name_prefix=""):
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, ".pkl")

        if check_cache_exists(cache_path, min_size=100) and not force_refresh:
            try:
                cached_data = PickleSerializer.load(cache_path)
                if cached_data is not None:
                    return (cached_data, f"Cache HIT (main): {os.path.basename(cache_path)}")
            except:
                pass

        if conditioning is None:
            return (conditioning, "Input is None - cannot cache")

        print(f"[BD Trellis2 Conditioning] Saving new cache: {cache_path}")
        if save_to_cache(cache_path, conditioning, PickleSerializer):
            status = f"SAVED: {os.path.basename(cache_path)}"
            print(f"[BD Trellis2 Conditioning] Cache saved successfully")
        else:
            status = "Save failed"
            print(f"[BD Trellis2 Conditioning] Cache save failed")
        return (conditioning, status)


class BD_CacheTrellis2Shape:
    """
    Cache Trellis2 shape result AND mesh together.
    This is the KEY node - caches expensive shape generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shape_result": ("TRELLIS2_SHAPE_RESULT", LAZY_OPTIONS),
                "mesh": ("TRIMESH", LAZY_OPTIONS),
                "cache_name": ("STRING", {"default": "trellis2_shape"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_refresh": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "name_prefix": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("TRELLIS2_SHAPE_RESULT", "TRIMESH", "STRING")
    RETURN_NAMES = ("shape_result", "mesh", "status")
    FUNCTION = "cache_shape"
    CATEGORY = "BrainDead/TRELLIS2"
    DESCRIPTION = """
Cache Trellis2 shape result + mesh to skip expensive generation.

Place AFTER Trellis2ImageToShape node.
This is THE most important cache - saves ~30-60s per run!

Caches both shape_result (PKL) and mesh (PLY) together.
"""

    @classmethod
    def IS_CHANGED(cls, cache_name, seed, force_refresh, shape_result=None, mesh=None, name_prefix=""):
        if force_refresh:
            return f"force_{time.time()}"
        return f"{name_prefix}_{cache_name}_{seed}"

    def check_lazy_status(self, cache_name, seed, force_refresh, shape_result=None, mesh=None, name_prefix=""):
        """Return [] to skip upstream, ["shape_result", "mesh"] to evaluate upstream."""
        if force_refresh:
            print(f"[BD Trellis2 Shape] Force refresh - will run upstream")
            return ["shape_result", "mesh"]

        if not HAS_TRIMESH:
            return ["shape_result", "mesh"]

        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path_pkl = get_cache_path(full_name, cache_hash, ".pkl")
        cache_path_ply = get_cache_path(full_name, cache_hash, "_mesh.ply")

        if (check_cache_exists(cache_path_pkl, min_size=100) and
            check_cache_exists(cache_path_ply, min_size=100)):
            print(f"[BD Trellis2 Shape] Cache exists - SKIPPING upstream execution")
            return []  # Empty list = don't need inputs, skip upstream
        print(f"[BD Trellis2 Shape] No cache found - will run upstream")
        return ["shape_result", "mesh"]

    def cache_shape(self, shape_result, mesh, cache_name, seed, force_refresh, name_prefix=""):
        if not HAS_TRIMESH:
            return (shape_result, mesh, "ERROR: trimesh not installed")

        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path_pkl = get_cache_path(full_name, cache_hash, ".pkl")
        cache_path_ply = get_cache_path(full_name, cache_hash, "_mesh.ply")

        if (check_cache_exists(cache_path_pkl, min_size=100) and
            check_cache_exists(cache_path_ply, min_size=100) and not force_refresh):
            try:
                shape_data = PickleSerializer.load(cache_path_pkl)
                mesh_data = trimesh.load(cache_path_ply)
                if shape_data is not None and mesh_data is not None:
                    return (shape_data, mesh_data, f"Cache HIT (main): shape + mesh")
            except:
                pass

        if shape_result is None or mesh is None:
            return (shape_result, mesh, "Input is None - cannot cache")

        try:
            print(f"[BD Trellis2 Shape] Saving new cache...")
            # Save shape_result as pickle
            PickleSerializer.save(cache_path_pkl, shape_result)
            # Save mesh as PLY
            trimesh.exchange.export.export_mesh(mesh, cache_path_ply, file_type='ply')
            vert_count = len(mesh.vertices) if hasattr(mesh, 'vertices') else 'unknown'
            status = f"SAVED: shape + mesh ({vert_count} verts)"
            print(f"[BD Trellis2 Shape] Cache saved: {cache_path_pkl}")
        except Exception as e:
            status = f"Save failed: {e}"
            print(f"[BD Trellis2 Shape] Cache save failed: {e}")

        return (shape_result, mesh, status)


class BD_CacheTrellis2Texture:
    """
    Cache Trellis2 textured mesh output (trimesh + voxelgrid + pointcloud).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh_out": ("TRIMESH", LAZY_OPTIONS),
                "voxelgrid": ("TRELLIS2_VOXELGRID", LAZY_OPTIONS),
                "pbr_pointcloud": ("TRIMESH", LAZY_OPTIONS),
                "cache_name": ("STRING", {"default": "trellis2_texture"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_refresh": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "name_prefix": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "TRELLIS2_VOXELGRID", "TRIMESH", "STRING")
    RETURN_NAMES = ("trimesh", "voxelgrid", "pbr_pointcloud", "status")
    FUNCTION = "cache_texture"
    CATEGORY = "BrainDead/TRELLIS2"
    DESCRIPTION = """
Cache Trellis2 textured mesh outputs together.

Place AFTER Trellis2ShapeToTexturedMesh node.
Caches trimesh, voxelgrid, and pbr_pointcloud as single PKL.

Note: voxelgrid contains GPU tensors - may require GPU
to be available when loading from cache.
"""

    @classmethod
    def IS_CHANGED(cls, cache_name, seed, force_refresh, trimesh_out=None, voxelgrid=None, pbr_pointcloud=None, name_prefix=""):
        if force_refresh:
            return f"force_{time.time()}"
        return f"{name_prefix}_{cache_name}_{seed}"

    def check_lazy_status(self, cache_name, seed, force_refresh, trimesh_out=None, voxelgrid=None, pbr_pointcloud=None, name_prefix=""):
        """Return [] to skip upstream, ["trimesh_out", "voxelgrid", "pbr_pointcloud"] to evaluate upstream."""
        if force_refresh:
            print(f"[BD Trellis2 Texture] Force refresh - will run upstream")
            return ["trimesh_out", "voxelgrid", "pbr_pointcloud"]

        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, "_texture.pkl")

        if check_cache_exists(cache_path, min_size=100):
            print(f"[BD Trellis2 Texture] Cache exists - SKIPPING upstream execution")
            return []  # Empty list = don't need inputs, skip upstream
        print(f"[BD Trellis2 Texture] No cache found - will run upstream")
        return ["trimesh_out", "voxelgrid", "pbr_pointcloud"]

    def cache_texture(self, trimesh_out, voxelgrid, pbr_pointcloud, cache_name, seed, force_refresh, name_prefix=""):
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, "_texture.pkl")

        if check_cache_exists(cache_path, min_size=100) and not force_refresh:
            try:
                cached_data = PickleSerializer.load(cache_path)
                if cached_data and 'trimesh' in cached_data and 'voxelgrid' in cached_data:
                    return (cached_data['trimesh'], cached_data['voxelgrid'],
                           cached_data['pointcloud'], f"Cache HIT (main): texture data")
            except:
                pass

        if trimesh_out is None or voxelgrid is None:
            return (trimesh_out, voxelgrid, pbr_pointcloud, "Input is None - cannot cache")

        try:
            print(f"[BD Trellis2 Texture] Saving new cache...")
            cache_data = {
                'trimesh': trimesh_out,
                'voxelgrid': voxelgrid,
                'pointcloud': pbr_pointcloud
            }
            PickleSerializer.save(cache_path, cache_data)
            # Get voxelgrid info if available
            voxel_info = ""
            if isinstance(voxelgrid, dict):
                if 'coords' in voxelgrid:
                    voxel_info = f" ({len(voxelgrid['coords'])} voxels)"
            status = f"SAVED: texture data{voxel_info}"
            print(f"[BD Trellis2 Texture] Cache saved: {cache_path}")
        except Exception as e:
            status = f"Save failed: {e}"
            print(f"[BD Trellis2 Texture] Cache save failed: {e}")

        return (trimesh_out, voxelgrid, pbr_pointcloud, status)


# Node exports
TRELLIS2_CACHE_NODES = {
    "BD_CacheTrellis2Conditioning": BD_CacheTrellis2Conditioning,
    "BD_CacheTrellis2Shape": BD_CacheTrellis2Shape,
    "BD_CacheTrellis2Texture": BD_CacheTrellis2Texture,
}

TRELLIS2_CACHE_DISPLAY_NAMES = {
    "BD_CacheTrellis2Conditioning": "BD Cache Trellis2 Conditioning",
    "BD_CacheTrellis2Shape": "BD Cache Trellis2 Shape",
    "BD_CacheTrellis2Texture": "BD Cache Trellis2 Texture",
}
