"""
V3 API cache nodes for common ComfyUI data types.

Each node caches a specific data type with lazy evaluation
to skip expensive upstream generation when cache is valid.

KNOWN BUG WORKAROUND (2026-01-22):
    ComfyUI's V3 API lazy evaluation does not work correctly for MASK type inputs.
    When a MASK input is marked with lazy=True in V3 define_schema(), ComfyUI fails
    to call check_lazy_status() and instead raises "Required input is missing: mask"
    during prompt validation - even when the input is connected but upstream is muted.

    IMAGE type inputs with lazy=True work correctly in V3 API.

    WORKAROUND: BD_CacheMask uses V1 API style (INPUT_TYPES dict) with the mask input
    placed in the "optional" dict. This allows lazy evaluation to work properly.

    If updating BD_CacheMask in an existing workflow, users must delete and re-add
    the node to refresh the schema.
"""

from comfy_api.latest import io

from .base_v3 import CacheNodeMixin
from ...utils.shared import (
    ImageSerializer,
    MaskSerializer,
    LatentSerializer,
    AudioSerializer,
    StringSerializer,
    PickleSerializer,
)


class BD_CacheImage(CacheNodeMixin, io.ComfyNode):
    """
    Cache IMAGE tensors to skip expensive image generation.

    First run: generates image, saves to cache as PNG
    Subsequent runs: loads from cache, SKIPS upstream generation!
    """

    input_name = "image"
    serializer = ImageSerializer
    node_label = "BD Cache Image"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CacheImage",
            display_name="BD Cache Image",
            category="🧠BrainDead/Cache",
            description="Cache images with lazy evaluation - skips upstream generation when cache exists.",
            inputs=[
                io.Image.Input("image", lazy=True),
                io.String.Input("cache_name", default="cached_image"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Boolean.Input("force_refresh", default=False, tooltip="Force regeneration even if cache exists"),
                io.String.Input("name_prefix", default="", optional=True, tooltip="Prefix for cache filename"),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, image, cache_name, seed, force_refresh, name_prefix="") -> io.NodeOutput:
        result, status = cls._cache_data(image, cache_name, seed, force_refresh, name_prefix)
        return io.NodeOutput(result, status)


class BD_CacheMask(CacheNodeMixin):
    """
    Cache MASK tensors to skip expensive mask generation.

    Uses V1 API style as a WORKAROUND for ComfyUI bug where V3 lazy evaluation
    does not work correctly for MASK type inputs. See module docstring for details.

    First run: generates mask, saves to cache as grayscale PNG
    Subsequent runs: loads from cache, SKIPS upstream generation!

    NOTE: If updating from a previous version, delete and re-add this node
    in your workflow to refresh the schema.
    """

    input_name = "mask"
    serializer = MaskSerializer
    node_label = "BD Cache Mask"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_name": ("STRING", {"default": "cached_mask"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_refresh": ("BOOLEAN", {"default": False, "tooltip": "Force regeneration even if cache exists"}),
            },
            "optional": {
                # WORKAROUND: mask in optional dict for V1 API lazy evaluation
                # V3 API lazy=True on MASK type does not work correctly
                "mask": ("MASK", {"lazy": True, "tooltip": "Input mask - optional when cache exists"}),
                "name_prefix": ("STRING", {"default": "", "tooltip": "Prefix for cache filename"}),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "status")
    FUNCTION = "execute"
    CATEGORY = "🧠BrainDead/Cache"
    DESCRIPTION = "Cache masks with lazy evaluation - skips upstream generation when cache exists."

    @classmethod
    def check_lazy_status(cls, cache_name, seed, force_refresh, mask=None, name_prefix=""):
        """
        V1-style check_lazy_status with explicit parameters.

        Returns [] to skip upstream evaluation (cache hit), or ["mask"] to request input.
        """
        if force_refresh:
            return ["mask"]

        cache_path = cls._get_cache_path(cache_name, seed, name_prefix)

        from ...utils.shared import check_cache_exists
        if check_cache_exists(cache_path, min_size=cls.min_cache_size):
            print(f"[{cls.node_label}] Cache HIT - SKIPPING upstream")
            return []

        return ["mask"]

    def execute(self, cache_name, seed, force_refresh, mask=None, name_prefix=""):
        result, status = self._cache_data(mask, cache_name, seed, force_refresh, name_prefix)
        return (result, status)


class BD_CacheLatent(CacheNodeMixin, io.ComfyNode):
    """Cache LATENT dicts to skip expensive latent generation."""

    input_name = "latent"
    serializer = LatentSerializer
    node_label = "BD Cache Latent"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CacheLatent",
            display_name="BD Cache Latent",
            category="🧠BrainDead/Cache",
            description="Cache latents with lazy evaluation - skips upstream generation when cache exists.",
            inputs=[
                io.Latent.Input("latent", lazy=True),
                io.String.Input("cache_name", default="cached_latent"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Boolean.Input("force_refresh", default=False),
                io.String.Input("name_prefix", default="", optional=True),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, latent, cache_name, seed, force_refresh, name_prefix="") -> io.NodeOutput:
        result, status = cls._cache_data(latent, cache_name, seed, force_refresh, name_prefix)
        return io.NodeOutput(result, status)


class BD_CacheAudio(CacheNodeMixin, io.ComfyNode):
    """Cache AUDIO dicts to skip expensive audio generation."""

    input_name = "audio"
    serializer = AudioSerializer
    node_label = "BD Cache Audio"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CacheAudio",
            display_name="BD Cache Audio",
            category="🧠BrainDead/Cache",
            description="Cache audio with lazy evaluation - skips upstream generation when cache exists.",
            inputs=[
                io.Audio.Input("audio", lazy=True),
                io.String.Input("cache_name", default="cached_audio"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Boolean.Input("force_refresh", default=False),
                io.String.Input("name_prefix", default="", optional=True),
            ],
            outputs=[
                io.Audio.Output(display_name="audio"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, audio, cache_name, seed, force_refresh, name_prefix="") -> io.NodeOutput:
        result, status = cls._cache_data(audio, cache_name, seed, force_refresh, name_prefix)
        return io.NodeOutput(result, status)


class BD_CacheString(CacheNodeMixin, io.ComfyNode):
    """Cache STRING values."""

    input_name = "text"
    serializer = StringSerializer
    node_label = "BD Cache String"
    min_cache_size = 0  # Empty strings are valid

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CacheString",
            display_name="BD Cache String",
            category="🧠BrainDead/Cache",
            description="Cache strings with lazy evaluation.",
            inputs=[
                io.String.Input("text", lazy=True, multiline=True),
                io.String.Input("cache_name", default="cached_string"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Boolean.Input("force_refresh", default=False),
                io.String.Input("name_prefix", default="", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="text"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, text, cache_name, seed, force_refresh, name_prefix="") -> io.NodeOutput:
        result, status = cls._cache_data(text, cache_name, seed, force_refresh, name_prefix)
        return io.NodeOutput(result, status)


class BD_CacheAny(CacheNodeMixin, io.ComfyNode):
    """
    Cache ANY data type using pickle serialization.

    Use this for custom types that don't have a specialized cache node.
    Falls back to pickle for maximum compatibility.
    """

    input_name = "data"
    serializer = PickleSerializer
    node_label = "BD Cache Any"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CacheAny",
            display_name="BD Cache Any",
            category="🧠BrainDead/Cache",
            description="Cache any data type using pickle serialization.",
            inputs=[
                io.AnyType.Input("data", lazy=True),
                io.String.Input("cache_name", default="cached_data"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Boolean.Input("force_refresh", default=False),
                io.String.Input("name_prefix", default="", optional=True),
            ],
            outputs=[
                io.AnyType.Output(display_name="data"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, data, cache_name, seed, force_refresh, name_prefix="") -> io.NodeOutput:
        result, status = cls._cache_data(data, cache_name, seed, force_refresh, name_prefix)
        return io.NodeOutput(result, status)


# V3 node list for extension
# Note: BD_CacheMask uses V1 API - ComfyUI V3 lazy evaluation broken for MASK type
CACHE_CORE_V3_NODES = [
    BD_CacheImage,
    # BD_CacheMask - V1 API (lazy eval workaround)
    BD_CacheLatent,
    BD_CacheAudio,
    BD_CacheString,
    BD_CacheAny,
]

# V1 compatibility - NODE_CLASS_MAPPINGS dict
CACHE_CORE_NODES = {
    "BD_CacheImage": BD_CacheImage,
    "BD_CacheMask": BD_CacheMask,
    "BD_CacheLatent": BD_CacheLatent,
    "BD_CacheAudio": BD_CacheAudio,
    "BD_CacheString": BD_CacheString,
    "BD_CacheAny": BD_CacheAny,
}

CACHE_CORE_DISPLAY_NAMES = {
    "BD_CacheImage": "BD Cache Image",
    "BD_CacheMask": "BD Cache Mask",
    "BD_CacheLatent": "BD Cache Latent",
    "BD_CacheAudio": "BD Cache Audio",
    "BD_CacheString": "BD Cache String",
    "BD_CacheAny": "BD Cache Any",
}
