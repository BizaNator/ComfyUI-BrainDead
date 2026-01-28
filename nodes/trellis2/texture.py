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
                io.Boolean.Input(
                    "output_shape_voxelgrid",
                    default=False,
                    tooltip="When using dual conditioning, also output voxelgrid from shape conditioning (runs texture twice)"
                ),
            ],
            outputs=[
                TrimeshOutput(display_name="trimesh"),
                io.Custom("VOXELGRID").Output(display_name="voxelgrid"),
                TrimeshOutput(display_name="pbr_pointcloud"),
                io.Custom("VOXELGRID").Output(display_name="shape_voxelgrid"),
                TrimeshOutput(display_name="shape_pbr_pointcloud"),
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
        output_shape_voxelgrid: bool = False,
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
        if output_shape_voxelgrid and texture_conditioning is not None:
            print(f"[BD TRELLIS2] Will also generate shape voxelgrid (dual output)")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Helper to extract voxelgrid and pointcloud from a textured mesh
        def extract_voxelgrid_outputs(mesh_obj, pbr_layout):
            """Extract voxelgrid dict and debug pointcloud from textured mesh."""
            coords = mesh_obj.coords.cpu().numpy().astype(np.float32)
            attrs = mesh_obj.attrs.cpu().numpy()  # (L, 6) in [-1, 1]
            voxel_size = mesh_obj.voxel_size

            voxel_grid = {
                'coords': coords,
                'attrs': attrs,
                'voxel_size': voxel_size,
                'layout': pbr_layout,
                'original_vertices': mesh_obj.vertices.cpu(),
                'original_faces': mesh_obj.faces.cpu(),
            }

            # Create debug point cloud
            point_positions = coords * voxel_size
            point_positions[:, 1], point_positions[:, 2] = (
                point_positions[:, 2].copy(),
                -point_positions[:, 1].copy()
            )

            attrs_normalized = (attrs + 1.0) * 0.5
            base_color_slice = pbr_layout.get('base_color', slice(0, 3))
            alpha_slice = pbr_layout.get('alpha', slice(5, 6))
            colors_rgb = (attrs_normalized[:, base_color_slice] * 255).clip(0, 255).astype(np.uint8)
            colors_alpha = (attrs_normalized[:, alpha_slice] * 255).clip(0, 255).astype(np.uint8)
            colors_rgba = np.concatenate([colors_rgb, colors_alpha], axis=1)

            pointcloud = trimesh.PointCloud(vertices=point_positions, colors=colors_rgba)
            return voxel_grid, pointcloud

        # Helper to run texture generation with given conditioning
        def run_texture_gen(cond_dict, shape_slat, subs, raw_verts, raw_faces, resolution, pipeline_type, manager, device, label=""):
            cond_on_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in cond_dict.items()
            }

            mesh = Mesh(vertices=raw_verts.to(device), faces=raw_faces.to(device))
            mesh.fill_holes()
            meshes = [mesh]

            pipeline = manager.get_texture_pipeline(device)
            sampler_params = {
                "tex_slat_sampler_params": {
                    "steps": tex_sampling_steps,
                    "guidance_strength": tex_guidance_strength,
                },
            }

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
            print(f"[BD TRELLIS2] {label}Texture generation peak VRAM: {peak_mem:.0f} MB")

            mesh_out = textured_meshes[0]
            mesh_out.simplify(16777216)
            return mesh_out, pipeline.pbr_attr_layout

        # Deserialize and move shape data to device
        shape_slat = _deserialize_from_storage(shape_result['shape_slat'], device)
        subs = _deserialize_from_storage(shape_result['subs'], device)
        resolution = shape_result['resolution']
        pipeline_type = shape_result['pipeline_type']
        raw_vertices = shape_result['raw_mesh_vertices']
        raw_faces = shape_result['raw_mesh_faces']

        # Get model manager
        manager = get_model_manager(
            model_name,
            resolution,
            attn_backend,
            vram_mode,
        )

        # ====== PRIMARY TEXTURE GENERATION (texture_conditioning or conditioning) ======
        mesh, pbr_layout = run_texture_gen(
            tex_cond, shape_slat, subs, raw_vertices, raw_faces,
            resolution, pipeline_type, manager, device, label=""
        )

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

        # Extract primary voxelgrid outputs
        voxel_grid, pointcloud = extract_voxelgrid_outputs(mesh, pbr_layout)

        # Create output trimesh
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # ====== OPTIONAL: SHAPE VOXELGRID (from original conditioning) ======
        shape_voxel_grid = None
        shape_pointcloud = None

        if output_shape_voxelgrid:
            if texture_conditioning is not None:
                print(f"[BD TRELLIS2] Running shape conditioning texture generation...")
                # Run texture generation again with original conditioning
                shape_mesh, shape_pbr_layout = run_texture_gen(
                    conditioning, shape_slat, subs, raw_vertices, raw_faces,
                    resolution, pipeline_type, manager, device, label="[Shape] "
                )
                shape_voxel_grid, shape_pointcloud = extract_voxelgrid_outputs(shape_mesh, shape_pbr_layout)
                print(f"[BD TRELLIS2] Shape voxelgrid: {len(shape_voxel_grid['coords']):,} voxels")
            else:
                # No texture_conditioning - shape_voxelgrid would be same as voxelgrid
                # Just copy the main voxelgrid
                print(f"[BD TRELLIS2] No texture_conditioning provided - shape_voxelgrid = copy of voxelgrid")
                shape_voxel_grid = voxel_grid.copy()
                shape_pointcloud = pointcloud
        else:
            # Return empty placeholders to prevent downstream crashes
            print(f"[BD TRELLIS2] output_shape_voxelgrid=False - returning empty placeholders")
            shape_voxel_grid = {
                'coords': np.zeros((0, 3), dtype=np.float32),
                'attrs': np.zeros((0, 6), dtype=np.float32),
                'voxel_size': voxel_grid['voxel_size'],
                'layout': pbr_layout,
                'original_vertices': torch.zeros((0, 3)),
                'original_faces': torch.zeros((0, 3), dtype=torch.int32),
            }
            shape_pointcloud = trimesh.PointCloud(vertices=np.zeros((1, 3)), colors=np.array([[128, 128, 128, 255]], dtype=np.uint8))

        # Unload texture pipeline
        manager.unload_texture_pipeline()

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[BD TRELLIS2] Texture generated: {len(vertices):,} verts, {len(voxel_grid['coords']):,} voxels")

        return io.NodeOutput(tri_mesh, voxel_grid, pointcloud, shape_voxel_grid, shape_pointcloud)


class BD_Trellis2Retexture(io.ComfyNode):
    """
    Retexture an existing TRELLIS2 shape with a new reference image.

    Convenience node that combines conditioning extraction + texture generation
    in one step. Takes an existing shape_result and applies new texture based
    on a different reference image.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_Trellis2Retexture",
            display_name="BD TRELLIS.2 Retexture",
            category="🧠BrainDead/TRELLIS2",
            description="""Retexture an existing TRELLIS.2 shape with a new reference image.

Convenience node that combines conditioning + texture generation in one step.
Takes an existing shape_result and applies new PBR texture from a different image.

Use cases:
- Generate shape from one image, apply texture from another
- Iterate on textures without regenerating shape (faster iteration)
- Apply different art styles/colors to the same 3D shape
- Test multiple texture seeds on the same geometry

Equivalent to: Get Conditioning (new image) → Shape to Textured Mesh (texture_conditioning)
but simpler to wire up.""",
            inputs=[
                io.Custom("TRELLIS2_SHAPE_RESULT").Input("shape_result", tooltip="Shape from BD Image to Shape node"),
                io.Image.Input("image", tooltip="New reference image for retexturing"),
                io.Mask.Input("mask", tooltip="Foreground mask for the new image"),
                io.Combo.Input(
                    "resolution",
                    options=['512', '1024_cascade', '1536_cascade'],
                    default="1024_cascade",
                    tooltip="Texture resolution mode"
                ),
                io.Int.Input("seed", default=0, min=0, max=2**31 - 1, tooltip="Random seed for texture variation"),
                io.Float.Input("tex_guidance_strength", default=7.5, min=1.0, max=20.0, step=0.1, tooltip="Texture CFG scale"),
                io.Int.Input("tex_sampling_steps", default=12, min=1, max=50, tooltip="Texture sampling steps"),
                io.Combo.Input(
                    "background_color",
                    options=["black", "gray", "white"],
                    default="gray",
                    tooltip="Background color for image preprocessing"
                ),
            ],
            outputs=[
                TrimeshOutput(display_name="trimesh"),
                io.Custom("VOXELGRID").Output(display_name="voxelgrid"),
            ],
        )

    @classmethod
    def execute(
        cls,
        shape_result: Dict[str, Any],
        image: torch.Tensor,
        mask: torch.Tensor,
        resolution: str = "1024_cascade",
        seed: int = 0,
        tex_guidance_strength: float = 7.5,
        tex_sampling_steps: int = 12,
        background_color: str = "gray",
    ) -> io.NodeOutput:
        if not HAS_TRELLIS2:
            raise ImportError("trellis2 not available. Ensure ComfyUI-TRELLIS2 is installed.")

        import cumesh as CuMesh
        from PIL import Image
        from trellis2.representations.mesh import Mesh
        from .utils.model_manager import get_model_manager
        from .utils.helpers import smart_crop_square, Trellis2ModelConfig

        print(f"[BD TRELLIS2] Retexture (resolution={resolution}, seed={seed})...")

        model_config = Trellis2ModelConfig(
            model_name="microsoft/TRELLIS.2-4B",
            resolution=resolution,
            attn_backend="flash_attn",
            vram_mode="keep_loaded",
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ===== STEP 1: Extract conditioning from new image =====
        bg_colors = {"black": (0, 0, 0), "gray": (128, 128, 128), "white": (255, 255, 255)}
        bg_color = bg_colors.get(background_color, (128, 128, 128))

        manager = get_model_manager(
            model_config.model_name,
            model_config.resolution,
            model_config.attn_backend,
            model_config.vram_mode,
        )

        # Convert image to PIL
        if image.dim() == 4:
            img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            img_np = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # Process mask
        if mask.dim() == 3:
            mask_np = mask[0].cpu().numpy()
        else:
            mask_np = mask.cpu().numpy()

        if mask_np.shape[:2] != (pil_image.height, pil_image.width):
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((pil_image.width, pil_image.height), Image.LANCZOS)
            mask_np = np.array(mask_pil) / 255.0

        pil_image = pil_image.convert('RGB')
        alpha_np = (mask_np * 255).astype(np.uint8)
        rgba = np.dstack([np.array(pil_image), alpha_np])
        pil_image = Image.fromarray(rgba, 'RGBA')
        pil_image = smart_crop_square(pil_image, alpha_np, margin_ratio=0.1, background_color=bg_color)

        # Encode with DinoV3
        model = manager.get_dinov3(device)
        model.image_size = 512
        cond_512 = model([pil_image])
        model.image_size = 1024
        cond_1024 = model([pil_image])
        manager.unload_dinov3()

        neg_cond = torch.zeros_like(cond_512)
        tex_cond = {
            'cond_512': cond_512,
            'neg_cond': neg_cond,
            'cond_1024': cond_1024,
        }

        print(f"[BD TRELLIS2] New conditioning extracted, running texture generation...")

        # ===== STEP 2: Run texture generation =====
        shape_slat = _deserialize_from_storage(shape_result['shape_slat'], device)
        subs = _deserialize_from_storage(shape_result['subs'], device)
        res = shape_result['resolution']
        pipeline_type = shape_result['pipeline_type']
        raw_vertices = shape_result['raw_mesh_vertices']
        raw_faces = shape_result['raw_mesh_faces']

        # Move conditioning to device
        cond_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in tex_cond.items()
        }

        mesh_obj = Mesh(vertices=raw_vertices.to(device), faces=raw_faces.to(device))
        mesh_obj.fill_holes()

        pipeline = manager.get_texture_pipeline(device)
        sampler_params = {
            "tex_slat_sampler_params": {
                "steps": tex_sampling_steps,
                "guidance_strength": tex_guidance_strength,
            },
        }

        torch.manual_seed(seed)
        textured_meshes = pipeline.run_texture_with_subs(
            cond_on_device,
            shape_slat,
            subs,
            [mesh_obj],
            res,
            seed=seed,
            pipeline_type=pipeline_type,
            **sampler_params
        )

        mesh_out = textured_meshes[0]
        mesh_out.simplify(16777216)
        pbr_layout = pipeline.pbr_attr_layout

        manager.unload_texture_pipeline()

        # ===== STEP 3: Build outputs =====
        cumesh = CuMesh.CuMesh()
        cumesh.init(mesh_out.vertices, mesh_out.faces.int())
        cumesh.unify_face_orientations()
        unified_verts, unified_faces = cumesh.read()

        vertices = unified_verts.cpu().numpy().astype(np.float32)
        faces = unified_faces.cpu().numpy()
        del cumesh, unified_verts, unified_faces

        # Y-up to Z-up
        vertices[:, 1], vertices[:, 2] = vertices[:, 2].copy(), -vertices[:, 1].copy()

        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # Voxelgrid
        voxel_grid = {
            'coords': mesh_out.coords.cpu().numpy().astype(np.float32),
            'attrs': mesh_out.attrs.cpu().numpy(),
            'voxel_size': mesh_out.voxel_size,
            'layout': pbr_layout,
            'original_vertices': mesh_out.vertices.cpu(),
            'original_faces': mesh_out.faces.cpu(),
        }

        gc.collect()
        torch.cuda.empty_cache()

        print(f"[BD TRELLIS2] Retexture complete: {len(vertices):,} verts, {len(voxel_grid['coords']):,} voxels")

        return io.NodeOutput(tri_mesh, voxel_grid)


# V3 node list for extension
TRELLIS2_TEXTURE_V3_NODES = [
    BD_Trellis2ShapeToTexturedMesh,
    BD_Trellis2Retexture,
]

# V1 compatibility
TRELLIS2_TEXTURE_NODES = {
    "BD_Trellis2ShapeToTexturedMesh": BD_Trellis2ShapeToTexturedMesh,
    "BD_Trellis2Retexture": BD_Trellis2Retexture,
}

TRELLIS2_TEXTURE_DISPLAY_NAMES = {
    "BD_Trellis2ShapeToTexturedMesh": "BD TRELLIS.2 Shape to Textured Mesh",
    "BD_Trellis2Retexture": "BD TRELLIS.2 Retexture",
}
