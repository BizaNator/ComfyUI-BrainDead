"""
V3 API Mesh cache node for caching TRIMESH objects.

BD_CacheMesh - Cache TRIMESH objects to skip expensive mesh generation
"""

import os
import time

from comfy_api.latest import io

from ...utils.shared import (
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

# Import custom TRIMESH type (matches TRELLIS2)
from .types import TrimeshInput, TrimeshOutput


class BD_CacheMesh(io.ComfyNode):
    """
    Cache TRIMESH objects to skip expensive mesh generation.
    Saves as PLY format for human-readable/editable files.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CacheMesh",
            display_name="BD Cache Mesh",
            category="🧠BrainDead/Mesh",
            description="Cache TRIMESH objects to skip expensive mesh generation. Saves as PLY format.",
            inputs=[
                TrimeshInput("mesh", lazy=True),
                io.String.Input("cache_name", default="cached_mesh"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Boolean.Input("force_refresh", default=False),
                io.String.Input("name_prefix", default="", optional=True),
            ],
            outputs=[
                TrimeshOutput(display_name="mesh"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, cache_name: str, seed: int, force_refresh: bool,
                           mesh=None, name_prefix: str = "") -> str:
        if force_refresh:
            return f"force_{time.time()}"
        return f"{name_prefix}_{cache_name}_{seed}"

    @classmethod
    def check_lazy_status(cls, cache_name: str, seed: int, force_refresh: bool,
                          mesh=None, name_prefix: str = "") -> list[str]:
        """Return [] to skip upstream, ['mesh'] to evaluate upstream."""
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
    def execute(cls, mesh, cache_name: str, seed: int, force_refresh: bool,
                name_prefix: str = "") -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "ERROR: trimesh not installed")

        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, ".ply")

        if check_cache_exists(cache_path, min_size=100) and not force_refresh:
            try:
                # Explicitly specify file_type='ply' to avoid auto-detection issues
                cached_mesh = trimesh.load(cache_path, file_type='ply', force='mesh')
                if cached_mesh is not None and hasattr(cached_mesh, 'vertices'):
                    return io.NodeOutput(cached_mesh, f"Cache HIT: {os.path.basename(cache_path)}")
            except Exception as e:
                print(f"[BD Cache Mesh] WARNING: Failed to load cache: {e}")
                pass

        if mesh is None:
            return io.NodeOutput(mesh, "Input is None - cannot cache")

        try:
            trimesh.exchange.export.export_mesh(mesh, cache_path, file_type='ply')
            status = f"SAVED: {os.path.basename(cache_path)}"
        except Exception as e:
            status = f"Save failed: {e}"
        return io.NodeOutput(mesh, status)


# V3 node list for extension
MESH_CACHE_V3_NODES = [BD_CacheMesh]

# V1 compatibility - NODE_CLASS_MAPPINGS dict
MESH_CACHE_NODES = {
    "BD_CacheMesh": BD_CacheMesh,
}

MESH_CACHE_DISPLAY_NAMES = {
    "BD_CacheMesh": "BD Cache Mesh",
}
