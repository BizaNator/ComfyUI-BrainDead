"""
V3 API TRELLIS2 texture generation node.

BD_Trellis2ShapeToTexturedMesh - Generate PBR textured mesh from shape.

Supports dual-input workflow:
- texture_model_config: Use different resolution for texture (e.g., 512 shape → 1536 texture)
- texture_conditioning: Use different image for texture (e.g., sketch → clean render)
"""

import gc
from fractions import Fraction
from typing import Dict, Any, Optional

import numpy as np
import torch
import trimesh

from comfy_api.latest import io

from .utils import HAS_TRELLIS2

# TRIMESH type for compatibility with BD mesh nodes
def TrimeshOutput(display_name: str = "mesh"):
    """Create a TRIMESH output (matches BD mesh nodes)."""
    return io.Custom("TRIMESH").Output(display_name=display_name)


def _dict_to_sparse_tensor(d: Dict[str, Any], device: torch.device):
    """
    Reconstruct a SparseTensor from a serialized dict.
    Must be called within environment where trellis2 is available.
    """
    from trellis2.modules.sparse import SparseTensor

    feats = d['feats'].to(device)
    coords = d['coords'].to(device)
    shape = torch.Size(d['shape']) if d['shape'] else None
    scale = tuple(Fraction(n, den) for n, den in d['scale'])

    return SparseTensor(feats=feats, coords=coords, shape=shape, scale=scale)


def _deserialize_from_storage(obj: Any, device: torch.device) -> Any:
    """
    Recursively reconstruct SparseTensor objects from serialized dicts.
    """
    if isinstance(obj, dict) and obj.get('_type') == 'SparseTensor':
        return _dict_to_sparse_tensor(obj, device)
    elif isinstance(obj, list):
        return [_deserialize_from_storage(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_deserialize_from_storage(x, device) for x in obj)
    elif isinstance(obj, dict):
        return {k: _deserialize_from_storage(v, device) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        return obj


class BD_Trellis2ShapeToTexturedMesh(io.ComfyNode):
    """
    Generate PBR textured mesh from shape using TRELLIS2.

    Supports dual-input workflow for advanced control:
    - texture_model_config: Use different resolution for texture
    - texture_conditioning: Use different image for texture

    Runs in main ComfyUI process (no subprocess isolation).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_Trellis2ShapeToTexturedMesh",
            display_name="BD TRELLIS.2 Shape to Textured Mesh",
            category="🧠BrainDead/TRELLIS2",
            description="""Generate PBR textured mesh from shape.

Takes shape_result from "BD Image to Shape" node and generates PBR materials:
- base_color (RGB)
- metallic
- roughness
- alpha

DUAL-INPUT WORKFLOW (Advanced):

1. Fast shape + detailed texture:
   - Get Conditioning at 512 → shape generation (fast)
   - Get Conditioning at 1536_cascade → texture_conditioning (detailed voxelgrid)

2. Different images for shape vs texture:
   - Get Conditioning with outline image → conditioning (for shape)
   - Get Conditioning with clean render → texture_conditioning (for texture)

Returns:
- trimesh: The 3D mesh with PBR vertex attributes
- voxelgrid: Sparse PBR voxel data for BD Bake Textures node
- pbr_pointcloud: Debug point cloud with all 6 PBR channels""",
            inputs=[
                io.Custom("TRELLIS2_CONDITIONING").Input("conditioning"),
                io.Custom("TRELLIS2_SHAPE_RESULT").Input("shape_result"),
                # DUAL INPUT - separate conditioning for texture (includes its own config)
                io.Custom("TRELLIS2_CONDITIONING").Input(
                    "texture_conditioning",
                    optional=True,
                    tooltip="Optional: Separate conditioning for texture (different resolution/image)"
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2**31 - 1,
                    tooltip="Random seed for texture variation"
                ),
                io.Float.Input(
                    "tex_guidance_strength",
                    default=7.5,
                    min=1.0,
                    max=20.0,
                    step=0.1,
                    tooltip="Texture CFG scale"
                ),
                io.Int.Input(
                    "tex_sampling_steps",
                    default=12,
                    min=1,
                    max=50,
                    tooltip="Texture sampling steps"
                ),
            ],
            outputs=[
                TrimeshOutput(display_name="trimesh"),
                io.Custom("TRELLIS2_VOXELGRID").Output(display_name="voxelgrid"),
                TrimeshOutput(display_name="pbr_pointcloud"),
            ],
        )

    @classmethod
    def execute(
        cls,
        conditioning: Dict[str, torch.Tensor],
        shape_result: Dict[str, Any],
        texture_conditioning: Optional[Dict[str, torch.Tensor]] = None,
        seed: int = 0,
        tex_guidance_strength: float = 7.5,
        tex_sampling_steps: int = 12,
    ) -> io.NodeOutput:
        if not HAS_TRELLIS2:
            raise ImportError("trellis2 not available. Ensure ComfyUI-TRELLIS2 is installed.")

        import cumesh as CuMesh
        from trellis2.representations.mesh import Mesh
        from .utils.model_manager import get_model_manager

        # Use texture_conditioning if provided, otherwise fall back to conditioning
        tex_cond = texture_conditioning if texture_conditioning is not None else conditioning

        # Extract model config from the conditioning we're using
        config = tex_cond.get('_config', {})
        model_name = config.get('model_name', 'microsoft/TRELLIS.2-4B')
        resolution = config.get('resolution', '1024_cascade')
        attn_backend = config.get('attn_backend', 'flash_attn')
        vram_mode = config.get('vram_mode', 'keep_loaded')

        print(f"[BD TRELLIS2] Running texture generation (seed={seed}, resolution={resolution})...")
        if texture_conditioning is not None:
            print(f"[BD TRELLIS2] Using separate texture conditioning")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move conditioning to device
        cond_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in tex_cond.items()
        }

        # Deserialize and move shape data to device
        shape_slat = _deserialize_from_storage(shape_result['shape_slat'], device)
        subs = _deserialize_from_storage(shape_result['subs'], device)
        resolution = shape_result['resolution']
        pipeline_type = shape_result['pipeline_type']

        # Reconstruct Mesh objects from saved data
        raw_vertices = shape_result['raw_mesh_vertices'].to(device)
        raw_faces = shape_result['raw_mesh_faces'].to(device)
        mesh = Mesh(vertices=raw_vertices, faces=raw_faces)
        mesh.fill_holes()
        meshes = [mesh]

        # Get model manager and texture pipeline
        manager = get_model_manager(
            model_name,
            resolution,
            attn_backend,
            vram_mode,
        )
        pipeline = manager.get_texture_pipeline(device)

        # Build sampler params
        sampler_params = {
            "tex_slat_sampler_params": {
                "steps": tex_sampling_steps,
                "guidance_strength": tex_guidance_strength,
            },
        }

        # Run texture generation
        torch.cuda.reset_peak_memory_stats()
        textured_meshes = pipeline.run_texture_with_subs(
            cond_on_device,
            shape_slat,
            subs,
            meshes,
            resolution,
            seed=seed,
            pipeline_type=pipeline_type,
            **sampler_params
        )
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[BD TRELLIS2] Texture generation peak VRAM: {peak_mem:.0f} MB")

        mesh = textured_meshes[0]
        mesh.simplify(16777216)

        # Get PBR layout from pipeline
        pbr_layout = pipeline.pbr_attr_layout

        # Convert mesh to outputs using CuMesh
        cumesh = CuMesh.CuMesh()
        cumesh.init(mesh.vertices, mesh.faces.int())
        cumesh.unify_face_orientations()
        unified_verts, unified_faces = cumesh.read()

        vertices = unified_verts.cpu().numpy().astype(np.float32)
        faces = unified_faces.cpu().numpy()
        del cumesh, unified_verts, unified_faces

        # Coordinate conversion (Y-up to Z-up)
        vertices[:, 1], vertices[:, 2] = vertices[:, 2].copy(), -vertices[:, 1].copy()

        # Get voxel grid data
        coords = mesh.coords.cpu().numpy().astype(np.float32)
        attrs = mesh.attrs.cpu().numpy()  # (L, 6) in [-1, 1]
        voxel_size = mesh.voxel_size

        # Build voxelgrid output
        voxel_grid = {
            'coords': coords,
            'attrs': attrs,
            'voxel_size': voxel_size,
            'layout': pbr_layout,
            'original_vertices': mesh.vertices.cpu(),
            'original_faces': mesh.faces.cpu(),
        }

        # Create output trimesh
        tri_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False
        )

        # Create debug point cloud
        point_positions = coords * voxel_size

        # Apply Y-up to Z-up conversion for point cloud
        point_positions[:, 1], point_positions[:, 2] = (
            point_positions[:, 2].copy(),
            -point_positions[:, 1].copy()
        )

        # Convert attrs from [-1, 1] to [0, 1]
        attrs_normalized = (attrs + 1.0) * 0.5

        # For trimesh.PointCloud colors, use base_color RGB + alpha
        base_color_slice = pbr_layout.get('base_color', slice(0, 3))
        alpha_slice = pbr_layout.get('alpha', slice(5, 6))

        colors_rgb = (attrs_normalized[:, base_color_slice] * 255).clip(0, 255).astype(np.uint8)
        colors_alpha = (attrs_normalized[:, alpha_slice] * 255).clip(0, 255).astype(np.uint8)
        colors_rgba = np.concatenate([colors_rgb, colors_alpha], axis=1)

        pointcloud = trimesh.PointCloud(
            vertices=point_positions,
            colors=colors_rgba
        )

        # Unload texture pipeline
        manager.unload_texture_pipeline()

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[BD TRELLIS2] Texture generated: {len(vertices):,} verts, {len(coords):,} voxels")

        return io.NodeOutput(tri_mesh, voxel_grid, pointcloud)


# V3 node list for extension
TRELLIS2_TEXTURE_V3_NODES = [
    BD_Trellis2ShapeToTexturedMesh,
]

# V1 compatibility
TRELLIS2_TEXTURE_NODES = {
    "BD_Trellis2ShapeToTexturedMesh": BD_Trellis2ShapeToTexturedMesh,
}

TRELLIS2_TEXTURE_DISPLAY_NAMES = {
    "BD_Trellis2ShapeToTexturedMesh": "BD TRELLIS.2 Shape to Textured Mesh",
}
