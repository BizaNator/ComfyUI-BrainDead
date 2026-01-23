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
Model config is embedded in conditioning - no separate config needed!

SINGLE WORKFLOW:
One node → connect conditioning to both Image to Shape and Shape to Textured Mesh.

DUAL WORKFLOW (recommended for quality):
1. First node: resolution=512 → fast shape generation
   - Connect conditioning → Image to Shape

2. Second node: resolution=1536_cascade, different image if desired
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

        # Build conditioning dict with embedded model config
        conditioning = {
            'cond_512': cond_512.cpu(),
            'neg_cond': neg_cond.cpu(),
            # Embed model config so downstream nodes don't need separate input
            '_config': {
                'model_name': model_config.model_name,
                'resolution': model_config.resolution,
                'attn_backend': model_config.attn_backend,
                'vram_mode': model_config.vram_mode,
            }
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

        return io.NodeOutput(conditioning, preprocessed_tensor)


class BD_Trellis2MultiImageConditioning(io.ComfyNode):
    """
    Multi-image conditioning for TRELLIS2 multi-view generation.

    Takes a BATCH of images + masks and encodes each with DinoV3,
    then concatenates the feature tokens for multi-view conditioning.

    Enables the model to see multiple views of an object simultaneously.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_Trellis2MultiImageConditioning",
            display_name="BD TRELLIS.2 Multi-Image Conditioning",
            category="🧠BrainDead/TRELLIS2",
            description="""Multi-image conditioning for TRELLIS.2 multi-view generation.

Takes a BATCH of images + masks and encodes each with DinoV3,
then concatenates the feature tokens for multi-view conditioning.

This enables the model to see multiple views of an object and
generate more accurate 3D shapes.

Usage:
1. Batch multiple reference images (different angles) using a Batch node
2. Batch their corresponding masks
3. Connect output to BD Image to Shape or BD Shape to Textured Mesh

The model attends to all image features simultaneously,
similar to TRELLIS1's multi-image mode.

Resolution modes:
- 512: Fast, lower detail - good for shape
- 1024_cascade: Balanced
- 1536_cascade: Highest detail - best for texture""",
            inputs=[
                io.Image.Input("images", tooltip="Batched images (B > 1) - use a Batch node to combine multiple views"),
                io.Mask.Input("masks", tooltip="Batched masks corresponding to each image"),
                io.Combo.Input(
                    "resolution",
                    options=RESOLUTION_MODES,
                    default="1024_cascade",
                    tooltip="Model resolution mode"
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
                io.Custom("TRELLIS2_CONDITIONING").Output(display_name="conditioning"),
                io.Image.Output(display_name="preprocessed_images"),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        masks: torch.Tensor,
        resolution: str = "1024_cascade",
        attn_backend: str = "flash_attn",
        vram_mode: str = "keep_loaded",
        include_1024: bool = True,
        background_color: str = "gray",
    ) -> io.NodeOutput:
        if not HAS_TRELLIS2:
            raise ImportError("trellis2 not available. Ensure ComfyUI-TRELLIS2 is installed.")

        from .utils.model_manager import get_model_manager
        from .utils.helpers import Trellis2ModelConfig

        model_config = Trellis2ModelConfig(
            model_name="microsoft/TRELLIS.2-4B",
            resolution=resolution,
            attn_backend=attn_backend,
            vram_mode=vram_mode,
        )

        # Split batch into individual images
        if images.dim() == 4 and images.shape[0] > 1:
            num_images = images.shape[0]
        else:
            num_images = 1

        print(f"[BD TRELLIS2] Multi-image conditioning ({num_images} images, resolution={resolution})...")

        bg_colors = {
            "black": (0, 0, 0),
            "gray": (128, 128, 128),
            "white": (255, 255, 255),
        }
        bg_color = bg_colors.get(background_color, (128, 128, 128))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        manager = get_model_manager(
            model_config.model_name,
            model_config.resolution,
            model_config.attn_backend,
            model_config.vram_mode,
        )

        # Process each image into PIL RGBA
        pil_images = []
        for i in range(num_images):
            # Get image
            if images.dim() == 4:
                img_np = (images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            else:
                img_np = (images.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            # Get mask
            if masks.dim() == 3 and masks.shape[0] > i:
                mask_np = masks[i].cpu().numpy()
            elif masks.dim() == 2:
                mask_np = masks.cpu().numpy()
            else:
                mask_np = masks[0].cpu().numpy()

            # Resize mask to match image if needed
            if mask_np.shape[:2] != (pil_img.height, pil_img.width):
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((pil_img.width, pil_img.height), Image.LANCZOS)
                mask_np = np.array(mask_pil) / 255.0

            # Apply mask as alpha
            pil_img = pil_img.convert('RGB')
            alpha_np = (mask_np * 255).astype(np.uint8)
            rgba = np.dstack([np.array(pil_img), alpha_np])
            pil_img = Image.fromarray(rgba, 'RGBA')

            # Smart crop
            pil_img = smart_crop_square(pil_img, alpha_np, margin_ratio=0.1, background_color=bg_color)
            pil_images.append(pil_img)

        # Encode all images with DinoV3
        model = manager.get_dinov3(device)

        # Get 512px conditioning for each image
        model.image_size = 512
        cond_512_list = []
        for pil_img in pil_images:
            cond = model([pil_img])
            cond_512_list.append(cond)

        # Concatenate along token dimension
        cond_512 = torch.cat(cond_512_list, dim=1)  # (1, N*T, D)
        print(f"[BD TRELLIS2] Combined 512 conditioning: {cond_512.shape}")

        # Get 1024px conditioning if requested
        cond_1024 = None
        if include_1024:
            model.image_size = 1024
            cond_1024_list = []
            for pil_img in pil_images:
                cond = model([pil_img])
                cond_1024_list.append(cond)
            cond_1024 = torch.cat(cond_1024_list, dim=1)
            print(f"[BD TRELLIS2] Combined 1024 conditioning: {cond_1024.shape}")

        # Unload DinoV3
        manager.unload_dinov3()

        # Create negative conditioning
        neg_cond = torch.zeros_like(cond_512)

        conditioning = {
            'cond_512': cond_512.cpu(),
            'neg_cond': neg_cond.cpu(),
            '_config': {
                'model_name': model_config.model_name,
                'resolution': model_config.resolution,
                'attn_backend': model_config.attn_backend,
                'vram_mode': model_config.vram_mode,
            }
        }
        if cond_1024 is not None:
            conditioning['cond_1024'] = cond_1024.cpu()

        # Create preview grid
        preview_images = []
        for pil_img in pil_images:
            pil_rgb = pil_img.convert('RGB') if pil_img.mode != 'RGB' else pil_img
            preview_np = np.array(pil_rgb).astype(np.float32) / 255.0
            preview_images.append(torch.from_numpy(preview_np))

        # Stack into batch (resize to match first image)
        h, w = preview_images[0].shape[:2]
        batch_list = []
        for p in preview_images:
            if p.shape[0] != h or p.shape[1] != w:
                # Resize via PIL
                p_pil = Image.fromarray((p.numpy() * 255).astype(np.uint8))
                p_pil = p_pil.resize((w, h), Image.LANCZOS)
                p = torch.from_numpy(np.array(p_pil).astype(np.float32) / 255.0)
            batch_list.append(p)
        preprocessed_tensor = torch.stack(batch_list, dim=0)

        gc.collect()
        torch.cuda.empty_cache()

        print(f"[BD TRELLIS2] Multi-image conditioning ready ({num_images} images concatenated)")

        return io.NodeOutput(conditioning, preprocessed_tensor)


# V3 node list for extension
TRELLIS2_CONDITIONING_V3_NODES = [
    BD_Trellis2GetConditioning,
    BD_Trellis2MultiImageConditioning,
]

# V1 compatibility
TRELLIS2_CONDITIONING_NODES = {
    "BD_Trellis2GetConditioning": BD_Trellis2GetConditioning,
    "BD_Trellis2MultiImageConditioning": BD_Trellis2MultiImageConditioning,
}

TRELLIS2_CONDITIONING_DISPLAY_NAMES = {
    "BD_Trellis2GetConditioning": "BD TRELLIS.2 Get Conditioning",
    "BD_Trellis2MultiImageConditioning": "BD TRELLIS.2 Multi-Image Conditioning",
}
