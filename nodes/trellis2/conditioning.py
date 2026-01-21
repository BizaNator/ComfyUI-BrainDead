"""
V3 API TRELLIS2 conditioning node.

BD_Trellis2GetConditioning - All-in-one node for model config + DinoV3 conditioning.

Combines model loading config with DinoV3 feature extraction in a single node.
For dual workflow, use two of these nodes with different settings.
"""

import gc

import numpy as np
from PIL import Image
import torch

from comfy_api.latest import io

from .utils import HAS_TRELLIS2
from .utils.helpers import smart_crop_square, Trellis2ModelConfig


# Resolution modes
RESOLUTION_MODES = ['512', '1024_cascade', '1536_cascade']

# Attention backend options
ATTN_BACKENDS = ['flash_attn', 'xformers', 'sdpa']

# VRAM usage modes
VRAM_MODES = ['keep_loaded', 'cpu_offload']


class BD_Trellis2GetConditioning(io.ComfyNode):
    """
    All-in-one TRELLIS2 conditioning node with built-in model config.

    Combines model configuration + DinoV3 feature extraction.
    Outputs both model_config and conditioning for downstream nodes.

    For dual workflow (different settings for shape vs texture):
    - Use TWO of these nodes with different resolution settings
    - First node (512) → Image to Shape (model_config, conditioning)
    - Second node (1536_cascade) → Shape to Textured Mesh (texture_model_config, texture_conditioning)
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_Trellis2GetConditioning",
            display_name="BD TRELLIS.2 Get Conditioning",
            category="🧠BrainDead/TRELLIS2",
            description="""All-in-one TRELLIS2 conditioning with built-in model config.

Combines model configuration + DinoV3 feature extraction in one node.
Outputs both model_config AND conditioning for downstream use.

SINGLE WORKFLOW:
Just use one node, connect both outputs to Image to Shape and Shape to Textured Mesh.

DUAL WORKFLOW (recommended for quality):
1. First node: resolution=512 → fast shape generation
   - Connect model_config → Image to Shape (model_config)
   - Connect conditioning → Image to Shape (conditioning)

2. Second node: resolution=1536_cascade, different image if desired
   - Connect model_config → Shape to Textured Mesh (texture_model_config)
   - Connect conditioning → Shape to Textured Mesh (texture_conditioning)

This gives fast shape (~15s) with high-detail texture voxelgrid.

Resolution modes:
- 512: Fast (~15s), lower detail - good for shape
- 1024_cascade: Balanced (~30s) - good default
- 1536_cascade: Highest detail (~45s) - best for texture""",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                # Model config options (built-in)
                io.Combo.Input(
                    "resolution",
                    options=RESOLUTION_MODES,
                    default="1024_cascade",
                    tooltip="Model resolution. Use 512 for fast shape, 1536_cascade for detailed texture."
                ),
                io.Combo.Input(
                    "attn_backend",
                    options=ATTN_BACKENDS,
                    default="flash_attn",
                    tooltip="Attention implementation"
                ),
                io.Combo.Input(
                    "vram_mode",
                    options=VRAM_MODES,
                    default="keep_loaded",
                    tooltip="VRAM usage strategy"
                ),
                # Conditioning options
                io.Boolean.Input(
                    "include_1024",
                    default=True,
                    tooltip="Extract 1024px features (needed for cascade modes)"
                ),
                io.Combo.Input(
                    "background_color",
                    options=["black", "gray", "white"],
                    default="gray",
                    tooltip="Background color for preprocessing"
                ),
            ],
            outputs=[
                io.Custom("TRELLIS2_MODEL_CONFIG").Output(display_name="model_config"),
                io.Custom("TRELLIS2_CONDITIONING").Output(display_name="conditioning"),
                io.Image.Output(display_name="preprocessed_image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        resolution: str = "1024_cascade",
        attn_backend: str = "flash_attn",
        vram_mode: str = "keep_loaded",
        include_1024: bool = True,
        background_color: str = "gray",
    ) -> io.NodeOutput:
        if not HAS_TRELLIS2:
            raise ImportError("trellis2 not available. Ensure ComfyUI-TRELLIS2 is installed.")

        from .utils.model_manager import get_model_manager

        # Create model config
        model_config = Trellis2ModelConfig(
            model_name="microsoft/TRELLIS.2-4B",
            resolution=resolution,
            attn_backend=attn_backend,
            vram_mode=vram_mode,
        )

        print(f"[BD TRELLIS2] Conditioning (resolution={resolution}, include_1024={include_1024}, bg={background_color})...")

        # Background color mapping
        bg_colors = {
            "black": (0, 0, 0),
            "gray": (128, 128, 128),
            "white": (255, 255, 255),
        }
        bg_color = bg_colors.get(background_color, (128, 128, 128))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get model manager
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

        # Resize mask to match image if needed
        if mask_np.shape[:2] != (pil_image.height, pil_image.width):
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((pil_image.width, pil_image.height), Image.LANCZOS)
            mask_np = np.array(mask_pil) / 255.0

        # Apply mask as alpha channel
        pil_image = pil_image.convert('RGB')
        alpha_np = (mask_np * 255).astype(np.uint8)
        rgba = np.dstack([np.array(pil_image), alpha_np])
        pil_image = Image.fromarray(rgba, 'RGBA')

        # Smart crop to square with margin
        pil_image = smart_crop_square(pil_image, alpha_np, margin_ratio=0.1, background_color=bg_color)

        # Load DinoV3 and extract features
        model = manager.get_dinov3(device)

        # Get 512px conditioning
        model.image_size = 512
        cond_512 = model([pil_image])

        # Get 1024px conditioning if requested
        cond_1024 = None
        if include_1024:
            model.image_size = 1024
            cond_1024 = model([pil_image])

        # Unload DinoV3 to free VRAM
        manager.unload_dinov3()

        # Create negative conditioning (zeros)
        neg_cond = torch.zeros_like(cond_512)

        # Build conditioning dict
        conditioning = {
            'cond_512': cond_512.cpu(),
            'neg_cond': neg_cond.cpu(),
        }
        if cond_1024 is not None:
            conditioning['cond_1024'] = cond_1024.cpu()

        # Convert preprocessed image to tensor for output
        pil_rgb = pil_image.convert('RGB') if pil_image.mode != 'RGB' else pil_image
        preprocessed_np = np.array(pil_rgb).astype(np.float32) / 255.0
        preprocessed_tensor = torch.from_numpy(preprocessed_np).unsqueeze(0)

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[BD TRELLIS2] Conditioning ready (512: {cond_512.shape}, 1024: {cond_1024.shape if cond_1024 is not None else 'N/A'})")

        return io.NodeOutput(model_config, conditioning, preprocessed_tensor)


# V3 node list for extension
TRELLIS2_CONDITIONING_V3_NODES = [
    BD_Trellis2GetConditioning,
]

# V1 compatibility
TRELLIS2_CONDITIONING_NODES = {
    "BD_Trellis2GetConditioning": BD_Trellis2GetConditioning,
}

TRELLIS2_CONDITIONING_DISPLAY_NAMES = {
    "BD_Trellis2GetConditioning": "BD TRELLIS.2 Get Conditioning",
}
