"""
BD_Pixal3DImageTo3D - Pixal3D image-to-3D generation node.
"""
import gc

import numpy as np
import torch
import trimesh

from comfy_api.latest import io

from .utils import HAS_PIXAL3D, get_pipeline


class BD_Pixal3DImageTo3D(io.ComfyNode):
    """
    Generate 3D mesh from a Pixal3D-preprocessed image.

    Requires a PIXAL3D_INPUT from BD Pixal3D Preprocess.
    Downloads TencentARC/Pixal3D model weights on first run (~10GB).

    Outputs:
    - mesh: Untextured TRIMESH in Z-up for downstream BD mesh nodes
    - voxelgrid: TRELLIS2_VOXELGRID for BD_OVoxelBake / BD_OVoxelTextureBake
      (same format as Trellis2 — enables CUMesh sharp-edge remesh + full PBR texture bake)
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_Pixal3DImageTo3D",
            display_name="BD Pixal3D Image to 3D",
            category="🧠BrainDead/Pixal3D",
            description="""Generate 3D from image using Pixal3D (Trellis2-based pixel-to-3D).

Downloads TencentARC/Pixal3D model weights (~10GB) on first run.
Model is cached at $HF_HOME (/srv/AI_Stuff/models/huggingface/).

Pipeline types:
- 1024_cascade: Standard quality (recommended)
- 1536_cascade: Higher geometry and texture resolution

Outputs:
- mesh (TRIMESH): Untextured Z-up geometry. Feed into BD_CuMeshSimplify or BD_BlenderDecimateV3.
- voxelgrid (TRELLIS2_VOXELGRID): Full PBR voxel data. Feed into BD_OVoxelBake for
  sharp-edge-preserving texture bake, or BD_OVoxelTextureBake for bake-only.""",
            inputs=[
                io.Custom("PIXAL3D_INPUT").Input("pixal3d_input"),
                io.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2**31 - 1,
                    tooltip="Random seed for reproducibility",
                ),
                io.Combo.Input(
                    "pipeline_type",
                    options=["1024_cascade", "1536_cascade"],
                    default="1024_cascade",
                    tooltip="Generation pipeline. 1536_cascade gives higher resolution.",
                ),
                io.Int.Input(
                    "max_tokens",
                    default=49152,
                    min=16384,
                    max=65536,
                    step=4096,
                    tooltip="Max sparse structure tokens. Lower = less VRAM.",
                ),
                # Sparse structure stage
                io.Float.Input(
                    "ss_guidance_strength",
                    default=7.5,
                    min=1.0,
                    max=20.0,
                    step=0.1,
                    tooltip="Sparse structure CFG guidance scale",
                ),
                io.Float.Input(
                    "ss_guidance_rescale",
                    default=0.7,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Sparse structure guidance rescale",
                ),
                io.Int.Input(
                    "ss_sampling_steps",
                    default=25,
                    min=1,
                    max=50,
                    tooltip="Sparse structure sampling steps",
                ),
                io.Float.Input(
                    "ss_rescale_t",
                    default=5.0,
                    min=0.0,
                    max=10.0,
                    step=0.1,
                    tooltip="Sparse structure timestep rescale",
                ),
                # Shape stage
                io.Float.Input(
                    "shape_guidance_strength",
                    default=7.5,
                    min=1.0,
                    max=20.0,
                    step=0.1,
                    tooltip="Shape latent CFG guidance scale",
                ),
                io.Float.Input(
                    "shape_guidance_rescale",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Shape latent guidance rescale",
                ),
                io.Int.Input(
                    "shape_sampling_steps",
                    default=25,
                    min=1,
                    max=50,
                    tooltip="Shape latent sampling steps",
                ),
                io.Float.Input(
                    "shape_rescale_t",
                    default=3.0,
                    min=0.0,
                    max=10.0,
                    step=0.1,
                    tooltip="Shape latent timestep rescale",
                ),
                # Texture stage
                io.Float.Input(
                    "tex_guidance_strength",
                    default=1.0,
                    min=0.0,
                    max=20.0,
                    step=0.1,
                    tooltip="Texture latent CFG guidance scale",
                ),
                io.Float.Input(
                    "tex_guidance_rescale",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Texture latent guidance rescale",
                ),
                io.Int.Input(
                    "tex_sampling_steps",
                    default=25,
                    min=1,
                    max=50,
                    tooltip="Texture latent sampling steps",
                ),
                io.Float.Input(
                    "tex_rescale_t",
                    default=3.0,
                    min=0.0,
                    max=10.0,
                    step=0.1,
                    tooltip="Texture latent timestep rescale",
                ),
                io.String.Input(
                    "model_path",
                    default="TencentARC/Pixal3D",
                    optional=True,
                    tooltip="HuggingFace repo or local path for Pixal3D model weights",
                ),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                # VOXELGRID (the canonical type used by BD_OVoxelBake / BD_OVoxelTextureBake /
                # Trellis2). Was mislabeled TRELLIS2_VOXELGRID, which blocked the link in the UI.
                io.Custom("VOXELGRID").Output(display_name="voxelgrid"),
            ],
        )

    @classmethod
    def execute(
        cls,
        pixal3d_input: dict,
        seed: int = 42,
        pipeline_type: str = "1024_cascade",
        max_tokens: int = 49152,
        ss_guidance_strength: float = 7.5,
        ss_guidance_rescale: float = 0.7,
        ss_sampling_steps: int = 25,
        ss_rescale_t: float = 5.0,
        shape_guidance_strength: float = 7.5,
        shape_guidance_rescale: float = 0.5,
        shape_sampling_steps: int = 25,
        shape_rescale_t: float = 3.0,
        tex_guidance_strength: float = 1.0,
        tex_guidance_rescale: float = 0.0,
        tex_sampling_steps: int = 25,
        tex_rescale_t: float = 3.0,
        model_path: str = "TencentARC/Pixal3D",
    ) -> io.NodeOutput:

        if not HAS_PIXAL3D:
            raise ImportError(
                "pixal3d package not found. "
                "Copy the pixal3d directory to the dev venv site-packages."
            )

        # Guard against seed/integer leaking into model_path (ComfyUI widget bug)
        model_path = str(model_path).strip()
        if not model_path or "/" not in model_path:
            print(f"[BD Pixal3D] WARNING: model_path '{model_path}' looks wrong, using default")
            model_path = "TencentARC/Pixal3D"

        image = pixal3d_input["image"]
        camera_params = pixal3d_input["camera_params"]

        pipeline = get_pipeline(model_path)

        ss_sampler_params = {
            "steps": ss_sampling_steps,
            "guidance_strength": ss_guidance_strength,
            "guidance_rescale": ss_guidance_rescale,
            "rescale_t": ss_rescale_t,
        }
        shape_sampler_params = {
            "steps": shape_sampling_steps,
            "guidance_strength": shape_guidance_strength,
            "guidance_rescale": shape_guidance_rescale,
            "rescale_t": shape_rescale_t,
        }
        tex_sampler_params = {
            "steps": tex_sampling_steps,
            "guidance_strength": tex_guidance_strength,
            "guidance_rescale": tex_guidance_rescale,
            "rescale_t": tex_rescale_t,
        }

        print(
            f"[BD Pixal3D] Generating 3D (seed={seed}, type={pipeline_type}, "
            f"max_tokens={max_tokens})..."
        )
        torch.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats()

        mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
            image,
            camera_params=camera_params,
            seed=seed,
            sparse_structure_sampler_params=ss_sampler_params,
            shape_slat_sampler_params=shape_sampler_params,
            tex_slat_sampler_params=tex_sampler_params,
            preprocess_image=False,
            return_latent=True,
            pipeline_type=pipeline_type,
            max_num_tokens=max_tokens,
        )

        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[BD Pixal3D] Generation peak VRAM: {peak_mem:.0f} MB")

        raw_mesh = mesh_list[0]

        # --- Untextured TRIMESH (Z-up) ---
        # Pixal3D's raw mesh is already Z-up (same convention as o_voxel/to_glb input).
        # Do NOT apply a Y-up→Z-up swap — the mesh is already in Z-up space.
        verts = raw_mesh.vertices.cpu().numpy().astype(np.float32)
        faces = raw_mesh.faces.cpu().numpy()
        tri_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        print(f"[BD Pixal3D] Geometry: {len(verts):,} verts, {len(faces):,} faces")

        # --- TRELLIS2_VOXELGRID ---
        # Same dict format as BD Trellis2 texture node. Enables routing through:
        #   BD_OVoxelBake       → sharp-edge remesh + full PBR texture bake
        #   BD_OVoxelTextureBake → bake-only with custom decimated mesh
        voxelgrid = {
            "coords": raw_mesh.coords.cpu().numpy().astype(np.float32),
            "attrs": raw_mesh.attrs.cpu().numpy(),
            "voxel_size": raw_mesh.voxel_size,
            "layout": pipeline.pbr_attr_layout,
            "original_vertices": raw_mesh.vertices.cpu(),
            "original_faces": raw_mesh.faces.cpu(),
        }
        print(
            f"[BD Pixal3D] Voxelgrid: {voxelgrid['coords'].shape[0]:,} voxels, "
            f"res={res}, voxel_size={raw_mesh.voxel_size:.5f}"
        )

        gc.collect()
        torch.cuda.empty_cache()

        return io.NodeOutput(tri_mesh, voxelgrid)


# V3 node list
PIXAL3D_GENERATE_V3_NODES = [BD_Pixal3DImageTo3D]

PIXAL3D_GENERATE_NODES = {
    "BD_Pixal3DImageTo3D": BD_Pixal3DImageTo3D,
}

PIXAL3D_GENERATE_DISPLAY_NAMES = {
    "BD_Pixal3DImageTo3D": "BD Pixal3D Image to 3D",
}
