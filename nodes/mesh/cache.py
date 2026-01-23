"""
V3 API Mesh cache node for caching TRIMESH objects.

BD_CacheMesh - Cache TRIMESH objects to skip expensive mesh generation.

Uses GLB format to preserve PBR materials, UVs, and vertex colors.
Metadata (edge colors, planar grouping, etc.) stored in sidecar pickle.
"""

import os
import pickle
import time

import numpy as np

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


def _save_mesh_glb(mesh, glb_path):
    """Save mesh as GLB, preserving material + vertex colors."""
    mesh.export(glb_path, file_type='glb')


def _save_metadata(mesh, meta_path):
    """Save mesh metadata as pickle sidecar if non-empty."""
    if not mesh.metadata:
        # Remove stale sidecar if exists
        if os.path.exists(meta_path):
            os.remove(meta_path)
        return False

    # Only save serializable metadata entries
    serializable = {}
    for key, val in mesh.metadata.items():
        if isinstance(val, np.ndarray):
            serializable[key] = val
        elif isinstance(val, (str, int, float, bool, list, dict, tuple)):
            serializable[key] = val
        else:
            try:
                pickle.dumps(val)
                serializable[key] = val
            except (pickle.PicklingError, TypeError):
                print(f"[BD Cache Mesh] Skipping non-serializable metadata key: {key}")

    if serializable:
        with open(meta_path, 'wb') as f:
            pickle.dump(serializable, f)
        return True
    return False


def _load_metadata(mesh, meta_path):
    """Load metadata from pickle sidecar into mesh."""
    if not os.path.exists(meta_path):
        return
    try:
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        if isinstance(metadata, dict):
            mesh.metadata.update(metadata)
    except Exception as e:
        print(f"[BD Cache Mesh] Warning: failed to load metadata: {e}")


class BD_CacheMesh(io.ComfyNode):
    """
    Cache TRIMESH objects to skip expensive mesh generation.
    Uses GLB format to preserve PBR materials, UVs, and vertex colors.
    Metadata stored in sidecar pickle file.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CacheMesh",
            display_name="BD Cache Mesh",
            category="🧠BrainDead/Mesh",
            description="Cache TRIMESH objects to skip expensive mesh generation. Uses GLB to preserve materials, UVs, and vertex colors.",
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
        cache_path = get_cache_path(full_name, cache_hash, ".glb")

        if check_cache_exists(cache_path, min_size=100):
            print(f"[BD Cache Mesh] Cache exists - SKIPPING upstream execution")
            return []
        print(f"[BD Cache Mesh] No cache found - will run upstream")
        return ["mesh"]

    @classmethod
    def execute(cls, mesh, cache_name: str, seed: int, force_refresh: bool,
                name_prefix: str = "") -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "ERROR: trimesh not installed")

        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, ".glb")
        meta_path = cache_path.replace(".glb", ".meta.pkl")

        if check_cache_exists(cache_path, min_size=100) and not force_refresh:
            try:
                cached_mesh = trimesh.load(cache_path, file_type='glb', force='mesh')
                if cached_mesh is not None and hasattr(cached_mesh, 'vertices'):
                    _load_metadata(cached_mesh, meta_path)
                    n_verts = len(cached_mesh.vertices)
                    has_mat = hasattr(cached_mesh.visual, 'material') and cached_mesh.visual.material is not None
                    has_uv = hasattr(cached_mesh.visual, 'uv') and cached_mesh.visual.uv is not None
                    mat_info = " +material" if has_mat else ""
                    uv_info = " +UVs" if has_uv else ""
                    meta_info = f" +metadata({len(cached_mesh.metadata)})" if cached_mesh.metadata else ""
                    return io.NodeOutput(
                        cached_mesh,
                        f"Cache HIT: {n_verts:,} verts{mat_info}{uv_info}{meta_info}"
                    )
            except Exception as e:
                print(f"[BD Cache Mesh] WARNING: Failed to load GLB cache: {e}")
                # Try legacy PLY fallback
                ply_path = cache_path.replace(".glb", ".ply")
                if os.path.exists(ply_path):
                    try:
                        cached_mesh = trimesh.load(ply_path, file_type='ply', force='mesh')
                        if cached_mesh is not None and hasattr(cached_mesh, 'vertices'):
                            print(f"[BD Cache Mesh] Loaded legacy PLY cache")
                            return io.NodeOutput(cached_mesh, f"Cache HIT (legacy PLY): {os.path.basename(ply_path)}")
                    except Exception:
                        pass

        if mesh is None:
            return io.NodeOutput(mesh, "Input is None - cannot cache")

        try:
            _save_mesh_glb(mesh, cache_path)
            has_meta = _save_metadata(mesh, meta_path)

            size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            has_mat = hasattr(mesh.visual, 'material') and mesh.visual.material is not None
            has_uv = hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None
            mat_info = " +material" if has_mat else ""
            uv_info = " +UVs" if has_uv else ""
            meta_info = " +metadata" if has_meta else ""
            status = f"SAVED: {size_mb:.1f}MB{mat_info}{uv_info}{meta_info}"
        except Exception as e:
            status = f"Save failed: {e}"
            print(f"[BD Cache Mesh] ERROR: {e}")
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
