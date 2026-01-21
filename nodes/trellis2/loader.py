"""
V3 API TRELLIS2 model loader node.

BD_LoadTrellis2Models - Create model configuration for TRELLIS2 inference.
"""

from comfy_api.latest import io

from .utils.helpers import Trellis2ModelConfig


# Resolution modes (matching original TRELLIS.2)
RESOLUTION_MODES = ['512', '1024_cascade', '1536_cascade']

# Attention backend options
ATTN_BACKENDS = ['flash_attn', 'xformers', 'sdpa']

# VRAM usage modes
VRAM_MODES = ['keep_loaded', 'cpu_offload']


class BD_LoadTrellis2Models(io.ComfyNode):
    """
    Create TRELLIS2 model configuration.

    This node creates a lightweight config object - actual model loading
    happens on-demand in the inference nodes.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_LoadTrellis2Models",
            display_name="BD Load TRELLIS.2 Models",
            category="🧠BrainDead/TRELLIS2",
            description="""Create TRELLIS2 model configuration for image-to-3D generation.

Resolution modes:
- 512: Fast shape generation (~15s), lower detail
- 1024_cascade: Best quality (~30s), uses 512→1024 cascade
- 1536_cascade: Highest resolution output (~45s)

Attention backend:
- flash_attn: FlashAttention (fastest, recommended)
- xformers: Memory-efficient attention
- sdpa: PyTorch native scaled_dot_product_attention

VRAM mode:
- keep_loaded: Keep all models in VRAM (fastest, ~12GB VRAM)
- cpu_offload: Offload unused models to CPU RAM (~3-4GB VRAM, ~15-25% slower)

For dual resolution workflow (fast shape + detailed texture):
- Add TWO loader nodes
- First: resolution=512 → Image to Shape
- Second: resolution=1536_cascade → Shape to Textured Mesh (texture_model_config)""",
            inputs=[
                io.Combo.Input(
                    "resolution",
                    options=RESOLUTION_MODES,
                    default="1024_cascade",
                    tooltip="Output resolution mode"
                ),
                io.Combo.Input(
                    "attn_backend",
                    options=ATTN_BACKENDS,
                    default="flash_attn",
                    tooltip="Attention implementation to use"
                ),
                io.Combo.Input(
                    "vram_mode",
                    options=VRAM_MODES,
                    default="keep_loaded",
                    tooltip="VRAM usage strategy"
                ),
            ],
            outputs=[
                io.Custom("TRELLIS2_MODEL_CONFIG").Output(display_name="model_config"),
            ],
        )

    @classmethod
    def execute(
        cls,
        resolution: str = "1024_cascade",
        attn_backend: str = "flash_attn",
        vram_mode: str = "keep_loaded",
    ) -> io.NodeOutput:
        # Create lightweight config object
        # Actual model loading happens in inference nodes
        config = Trellis2ModelConfig(
            model_name="microsoft/TRELLIS.2-4B",
            resolution=resolution,
            attn_backend=attn_backend,
            vram_mode=vram_mode,
        )

        print(f"[BD TRELLIS2] Created config: {config}")

        return io.NodeOutput(config)


# V3 node list for extension
TRELLIS2_LOADER_V3_NODES = [
    BD_LoadTrellis2Models,
]

# V1 compatibility
TRELLIS2_LOADER_NODES = {
    "BD_LoadTrellis2Models": BD_LoadTrellis2Models,
}

TRELLIS2_LOADER_DISPLAY_NAMES = {
    "BD_LoadTrellis2Models": "BD Load TRELLIS.2 Models",
}
