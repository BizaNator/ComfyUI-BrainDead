"""
Mesh cache node for caching TRIMESH objects.

BD_CacheMesh - Cache TRIMESH objects to skip expensive mesh generation
"""

import os
import time

from ...utils.shared import (
    LAZY_OPTIONS,
    get_cache_path,
    hash_from_seed,
    check_cache_exists,
)

# Check for optional trimesh support
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


class BD_CacheMesh:
    """
    Cache TRIMESH objects to skip expensive mesh generation.
    Saves as PLY format for human-readable/editable files.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH", LAZY_OPTIONS),
                "cache_name": ("STRING", {"default": "cached_mesh"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_refresh": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "name_prefix": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "status")
    FUNCTION = "cache_mesh"
    CATEGORY = "BrainDead/Mesh"

    def check_lazy_status(self, cache_name, seed, force_refresh, mesh=None, name_prefix=""):
        """Return [] to skip upstream, ["mesh"] to evaluate upstream."""
        if force_refresh:
            print(f"[BD Cache Mesh] Force refresh - will run upstream")
            return ["mesh"]

        if not HAS_TRIMESH:
            return ["mesh"]

        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, ".ply")

        if check_cache_exists(cache_path, min_size=100):
            print(f"[BD Cache Mesh] Cache exists - SKIPPING upstream execution")
            return []  # Empty list = don't need input, skip upstream
        print(f"[BD Cache Mesh] No cache found - will run upstream")
        return ["mesh"]  # Need input = run upstream

    @classmethod
    def IS_CHANGED(cls, cache_name, seed, force_refresh, mesh=None, name_prefix=""):
        if force_refresh:
            return f"force_{time.time()}"
        return f"{name_prefix}_{cache_name}_{seed}"

    def cache_mesh(self, mesh, cache_name, seed, force_refresh, name_prefix=""):
        if not HAS_TRIMESH:
            return (mesh, "ERROR: trimesh not installed")

        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, ".ply")

        if check_cache_exists(cache_path, min_size=100) and not force_refresh:
            try:
                cached_mesh = trimesh.load(cache_path)
                if cached_mesh is not None and hasattr(cached_mesh, 'vertices'):
                    return (cached_mesh, f"Cache HIT (main): {os.path.basename(cache_path)}")
            except:
                pass

        if mesh is None:
            return (mesh, "Input is None - cannot cache")

        try:
            trimesh.exchange.export.export_mesh(mesh, cache_path, file_type='ply')
            status = f"SAVED: {os.path.basename(cache_path)}"
        except Exception as e:
            status = f"Save failed: {e}"
        return (mesh, status)


# Node exports
MESH_CACHE_NODES = {
    "BD_CacheMesh": BD_CacheMesh,
}

MESH_CACHE_DISPLAY_NAMES = {
    "BD_CacheMesh": "BD Cache Mesh",
}
