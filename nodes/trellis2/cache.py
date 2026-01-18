"""
V3 API TRELLIS2 cache nodes for caching conditioning, shape, and texture outputs.

BD_CacheTrellis2Conditioning - Cache conditioning to skip image preprocessing
BD_CacheTrellis2Shape - Cache shape + mesh to skip expensive shape generation
BD_CacheTrellis2Texture - Cache texture outputs (trimesh, voxelgrid, pointcloud)
"""

import os
import time

from comfy_api.latest import io

from ...utils.shared import (
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


class BD_CacheTrellis2Conditioning(io.ComfyNode):
    """Cache Trellis2 conditioning output to skip image preprocessing."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CacheTrellis2Conditioning",
            display_name="BD Cache Trellis2 Conditioning",
            category="🧠BrainDead/TRELLIS2",
            description="Cache Trellis2 conditioning to skip image preprocessing. Place AFTER Trellis2GetConditioning node.",
            inputs=[
                io.Custom("TRELLIS2_CONDITIONING").Input("conditioning", lazy=True),
                io.String.Input("cache_name", default="trellis2_cond"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Boolean.Input("force_refresh", default=False),
                io.String.Input("name_prefix", default="", optional=True),
            ],
            outputs=[
                io.Custom("TRELLIS2_CONDITIONING").Output(display_name="conditioning"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, cache_name: str, seed: int, force_refresh: bool,
                           name_prefix: str = "", **kwargs) -> str:
        if force_refresh:
            return f"force_{time.time()}"
        return f"{name_prefix}_{cache_name}_{seed}"

    @classmethod
    def check_lazy_status(cls, cache_name: str, seed: int, force_refresh: bool,
                          conditioning=None, name_prefix: str = "") -> list[str]:
        """Return [] to skip upstream, ['conditioning'] to evaluate upstream."""
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

    @classmethod
    def execute(cls, conditioning, cache_name: str, seed: int, force_refresh: bool,
                name_prefix: str = "") -> io.NodeOutput:
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, ".pkl")

        if check_cache_exists(cache_path, min_size=100) and not force_refresh:
            try:
                cached_data = PickleSerializer.load(cache_path)
                if cached_data is not None:
                    return io.NodeOutput(cached_data, f"Cache HIT: {os.path.basename(cache_path)}")
            except:
                pass

        if conditioning is None:
            return io.NodeOutput(conditioning, "Input is None - cannot cache")

        print(f"[BD Trellis2 Conditioning] Saving new cache: {cache_path}")
        if save_to_cache(cache_path, conditioning, PickleSerializer):
            status = f"SAVED: {os.path.basename(cache_path)}"
            print(f"[BD Trellis2 Conditioning] Cache saved successfully")
        else:
            status = "Save failed"
            print(f"[BD Trellis2 Conditioning] Cache save failed")
        return io.NodeOutput(conditioning, status)


class BD_CacheTrellis2Shape(io.ComfyNode):
    """
    Cache Trellis2 shape result AND mesh together.
    This is the KEY node - caches expensive shape generation.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CacheTrellis2Shape",
            display_name="BD Cache Trellis2 Shape",
            category="🧠BrainDead/TRELLIS2",
            description="Cache Trellis2 shape result + mesh to skip expensive generation. Place AFTER Trellis2ImageToShape node. Saves ~30-60s per run!",
            inputs=[
                io.Custom("TRELLIS2_SHAPE_RESULT").Input("shape_result", lazy=True),
                io.Mesh.Input("mesh", lazy=True),
                io.String.Input("cache_name", default="trellis2_shape"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Boolean.Input("force_refresh", default=False),
                io.String.Input("name_prefix", default="", optional=True),
            ],
            outputs=[
                io.Custom("TRELLIS2_SHAPE_RESULT").Output(display_name="shape_result"),
                io.Mesh.Output(display_name="mesh"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, cache_name: str, seed: int, force_refresh: bool,
                           shape_result=None, mesh=None, name_prefix: str = "") -> str:
        if force_refresh:
            return f"force_{time.time()}"
        return f"{name_prefix}_{cache_name}_{seed}"

    @classmethod
    def check_lazy_status(cls, cache_name: str, seed: int, force_refresh: bool,
                          shape_result=None, mesh=None, name_prefix: str = "") -> list[str]:
        """Return [] to skip upstream, ['shape_result', 'mesh'] to evaluate upstream."""
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

    @classmethod
    def execute(cls, shape_result, mesh, cache_name: str, seed: int, force_refresh: bool,
                name_prefix: str = "") -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput(shape_result, mesh, "ERROR: trimesh not installed")

        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path_pkl = get_cache_path(full_name, cache_hash, ".pkl")
        cache_path_ply = get_cache_path(full_name, cache_hash, "_mesh.ply")

        if (check_cache_exists(cache_path_pkl, min_size=100) and
            check_cache_exists(cache_path_ply, min_size=100) and not force_refresh):
            try:
                shape_data = PickleSerializer.load(cache_path_pkl)
                # Explicitly specify file_type='ply' to avoid auto-detection issues
                mesh_data = trimesh.load(cache_path_ply, file_type='ply', force='mesh')
                if shape_data is not None and mesh_data is not None:
                    return io.NodeOutput(shape_data, mesh_data, f"Cache HIT: shape + mesh")
            except Exception as e:
                print(f"[BD Trellis2 Shape] WARNING: Failed to load cache: {e}")
                pass

        if shape_result is None or mesh is None:
            return io.NodeOutput(shape_result, mesh, "Input is None - cannot cache")

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

        return io.NodeOutput(shape_result, mesh, status)


class BD_CacheTrellis2Texture(io.ComfyNode):
    """
    Cache Trellis2 textured mesh output (trimesh + voxelgrid + pointcloud).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CacheTrellis2Texture",
            display_name="BD Cache Trellis2 Texture",
            category="🧠BrainDead/TRELLIS2",
            description="Cache Trellis2 textured mesh outputs together. Place AFTER Trellis2ShapeToTexturedMesh node. Note: voxelgrid contains GPU tensors.",
            inputs=[
                io.Mesh.Input("trimesh_out", lazy=True),
                io.Custom("TRELLIS2_VOXELGRID").Input("voxelgrid", lazy=True),
                io.Mesh.Input("pbr_pointcloud", lazy=True),
                io.String.Input("cache_name", default="trellis2_texture"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Boolean.Input("force_refresh", default=False),
                io.String.Input("name_prefix", default="", optional=True),
            ],
            outputs=[
                io.Mesh.Output(display_name="trimesh"),
                io.Custom("TRELLIS2_VOXELGRID").Output(display_name="voxelgrid"),
                io.Mesh.Output(display_name="pbr_pointcloud"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, cache_name: str, seed: int, force_refresh: bool,
                           trimesh_out=None, voxelgrid=None, pbr_pointcloud=None,
                           name_prefix: str = "") -> str:
        if force_refresh:
            return f"force_{time.time()}"
        return f"{name_prefix}_{cache_name}_{seed}"

    @classmethod
    def check_lazy_status(cls, cache_name: str, seed: int, force_refresh: bool,
                          trimesh_out=None, voxelgrid=None, pbr_pointcloud=None,
                          name_prefix: str = "") -> list[str]:
        """Return [] to skip upstream, ['trimesh_out', 'voxelgrid', 'pbr_pointcloud'] to evaluate upstream."""
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

    @classmethod
    def execute(cls, trimesh_out, voxelgrid, pbr_pointcloud, cache_name: str, seed: int,
                force_refresh: bool, name_prefix: str = "") -> io.NodeOutput:
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, "_texture.pkl")

        if check_cache_exists(cache_path, min_size=100) and not force_refresh:
            try:
                cached_data = PickleSerializer.load(cache_path)
                if cached_data and 'trimesh' in cached_data and 'voxelgrid' in cached_data:
                    return io.NodeOutput(cached_data['trimesh'], cached_data['voxelgrid'],
                           cached_data['pointcloud'], f"Cache HIT: texture data")
            except:
                pass

        if trimesh_out is None or voxelgrid is None:
            return io.NodeOutput(trimesh_out, voxelgrid, pbr_pointcloud, "Input is None - cannot cache")

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

        return io.NodeOutput(trimesh_out, voxelgrid, pbr_pointcloud, status)


# V3 node list for extension
TRELLIS2_CACHE_V3_NODES = [
    BD_CacheTrellis2Conditioning,
    BD_CacheTrellis2Shape,
    BD_CacheTrellis2Texture,
]

# V1 compatibility - NODE_CLASS_MAPPINGS dict
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
