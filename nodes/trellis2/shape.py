"""
V3 API TRELLIS2 shape generation node.

BD_Trellis2ImageToShape - Generate 3D shape from conditioning.
"""

import gc
from fractions import Fraction
from typing import Dict, Any

import numpy as np
import torch
import trimesh

from comfy_api.latest import io

from .utils import HAS_TRELLIS2

# TRIMESH type for compatibility with BD mesh nodes
def TrimeshOutput(display_name: str = "mesh"):
    """Create a TRIMESH output (matches BD mesh nodes)."""
    return io.Custom("TRIMESH").Output(display_name=display_name)


def _sparse_tensor_to_dict(st) -> Dict[str, Any]:
    """
    Convert a SparseTensor to a serializable dict.
    This allows SparseTensor to pass between nodes without import issues.
    """
    return {
        '_type': 'SparseTensor',
        'feats': st.feats.cpu(),
        'coords': st.coords.cpu(),
        'shape': tuple(st.shape) if st.shape else None,
        'scale': tuple((s.numerator, s.denominator) for s in st._scale),
    }


def _serialize_for_storage(obj: Any) -> Any:
    """
    Recursively convert SparseTensor objects to serializable dicts.
    """
    if hasattr(obj, 'feats') and hasattr(obj, 'coords') and hasattr(obj, '_scale'):
        return _sparse_tensor_to_dict(obj)
    elif isinstance(obj, list):
        return [_serialize_for_storage(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_serialize_for_storage(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: _serialize_for_storage(v) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        return obj.cpu()
    else:
        return obj


class BD_Trellis2ImageToShape(io.ComfyNode):
    """
    Generate 3D shape from conditioning using TRELLIS2.

    Runs in main ComfyUI process (no subprocess isolation).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_Trellis2ImageToShape",
            display_name="BD TRELLIS.2 Image to Shape",
            category="🧠BrainDead/TRELLIS2",
            description="""Generate 3D shape from image conditioning.

This node generates shape geometry (no texture/PBR).
Connect shape_result to "BD Shape to Textured Mesh" for PBR materials.

Parameters:
- conditioning: DinoV3 conditioning from "BD Get Conditioning" node (includes model config)
- seed: Random seed for reproducibility
- ss_*: Sparse structure sampling parameters
- shape_*: Shape latent sampling parameters
- max_tokens: Max tokens for cascade (lower = less VRAM)

Returns:
- shape_result: Shape data for texture generation
- mesh: Untextured mesh for preview/export (TRIMESH type)""",
            inputs=[
                io.Custom("TRELLIS2_CONDITIONING").Input("conditioning"),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2**31 - 1,
                    tooltip="Random seed for reproducible generation"
                ),
                io.Float.Input(
                    "ss_guidance_strength",
                    default=7.5,
                    min=1.0,
                    max=20.0,
                    step=0.1,
                    tooltip="Sparse structure CFG scale"
                ),
                io.Int.Input(
                    "ss_sampling_steps",
                    default=12,
                    min=1,
                    max=50,
                    tooltip="Sparse structure sampling steps"
                ),
                io.Float.Input(
                    "shape_guidance_strength",
                    default=7.5,
                    min=1.0,
                    max=20.0,
                    step=0.1,
                    tooltip="Shape CFG scale"
                ),
                io.Int.Input(
                    "shape_sampling_steps",
                    default=12,
                    min=1,
                    max=50,
                    tooltip="Shape sampling steps"
                ),
                io.Int.Input(
                    "max_tokens",
                    default=49152,
                    min=16384,
                    max=65536,
                    step=4096,
                    tooltip="Max tokens for cascade. Lower = less VRAM. Default 49152 (~9GB)"
                ),
            ],
            outputs=[
                io.Custom("TRELLIS2_SHAPE_RESULT").Output(display_name="shape_result"),
                TrimeshOutput(display_name="mesh"),
            ],
        )

    @classmethod
    def execute(
        cls,
        conditioning: Dict[str, torch.Tensor],
        seed: int = 0,
        ss_guidance_strength: float = 7.5,
        ss_sampling_steps: int = 12,
        shape_guidance_strength: float = 7.5,
        shape_sampling_steps: int = 12,
        max_tokens: int = 49152,
    ) -> io.NodeOutput:
        if not HAS_TRELLIS2:
            raise ImportError("trellis2 not available. Ensure ComfyUI-TRELLIS2 is installed.")

        import cumesh as CuMesh
        from .utils.model_manager import get_model_manager

        # Extract model config from conditioning
        config = conditioning.get('_config', {})
        model_name = config.get('model_name', 'microsoft/TRELLIS.2-4B')
        resolution = config.get('resolution', '1024_cascade')
        attn_backend = config.get('attn_backend', 'flash_attn')
        vram_mode = config.get('vram_mode', 'keep_loaded')

        print(f"[BD TRELLIS2] Running shape generation (seed={seed}, resolution={resolution})...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move conditioning to device
        cond_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in conditioning.items()
        }

        # Get model manager and shape pipeline
        manager = get_model_manager(
            model_name,
            resolution,
            attn_backend,
            vram_mode,
        )
        pipeline = manager.get_shape_pipeline(device)

        # Build sampler params
        sampler_params = {
            "sparse_structure_sampler_params": {
                "steps": ss_sampling_steps,
                "guidance_strength": ss_guidance_strength,
            },
            "shape_slat_sampler_params": {
                "steps": shape_sampling_steps,
                "guidance_strength": shape_guidance_strength,
            },
        }

        # Run shape generation
        torch.cuda.reset_peak_memory_stats()
        meshes, shape_slat, subs, res = pipeline.run_shape(
            cond_on_device,
            seed=seed,
            pipeline_type=resolution,
            max_num_tokens=max_tokens,
            **sampler_params
        )
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[BD TRELLIS2] Shape generation peak VRAM: {peak_mem:.0f} MB")

        mesh = meshes[0]
        mesh.fill_holes()

        # Save RAW mesh data for texture stage (before coordinate conversion)
        raw_mesh_vertices = mesh.vertices.cpu()
        raw_mesh_faces = mesh.faces.cpu()

        # Unify face orientations using CuMesh
        cumesh = CuMesh.CuMesh()
        cumesh.init(mesh.vertices, mesh.faces.int())
        cumesh.unify_face_orientations()
        unified_verts, unified_faces = cumesh.read()

        vertices = unified_verts.cpu().numpy().astype(np.float32)
        faces = unified_faces.cpu().numpy()
        del cumesh, unified_verts, unified_faces

        # Fix normals to point outward (unify only ensures consistent winding, not direction)
        from .utils.helpers import fix_normals_outward
        faces = fix_normals_outward(vertices, faces)

        # Coordinate system conversion (Y-up to Z-up) for output mesh
        vertices[:, 1], vertices[:, 2] = vertices[:, 2].copy(), -vertices[:, 1].copy()

        # Pack shape result - serialize SparseTensor objects for storage
        shape_result = {
            'shape_slat': _serialize_for_storage(shape_slat),
            'subs': _serialize_for_storage(subs),
            'mesh_vertices': vertices,
            'mesh_faces': faces,
            'resolution': res,
            'pipeline_type': resolution,
            'raw_mesh_vertices': raw_mesh_vertices,
            'raw_mesh_faces': raw_mesh_faces,
        }

        # Create trimesh for output
        tri_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False
        )

        # Unload shape pipeline
        manager.unload_shape_pipeline()

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[BD TRELLIS2] Shape generated: {len(vertices):,} verts, {len(faces):,} faces")

        return io.NodeOutput(shape_result, tri_mesh)


# V3 node list for extension
TRELLIS2_SHAPE_V3_NODES = [
    BD_Trellis2ImageToShape,
]

# V1 compatibility
TRELLIS2_SHAPE_NODES = {
    "BD_Trellis2ImageToShape": BD_Trellis2ImageToShape,
}

TRELLIS2_SHAPE_DISPLAY_NAMES = {
    "BD_Trellis2ImageToShape": "BD TRELLIS.2 Image to Shape",
}
