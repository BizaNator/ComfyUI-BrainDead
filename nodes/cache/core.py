"""
V3 API cache nodes for common ComfyUI data types.

Each node caches a specific data type with lazy evaluation
to skip expensive upstream generation when cache is valid.
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


class BD_CacheMask(CacheNodeMixin, io.ComfyNode):
    """Cache MASK tensors to skip expensive mask generation."""

    input_name = "mask"
    serializer = MaskSerializer
    node_label = "BD Cache Mask"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CacheMask",
            display_name="BD Cache Mask",
            category="🧠BrainDead/Cache",
            description="Cache masks with lazy evaluation - skips upstream generation when cache exists.",
            inputs=[
                io.Mask.Input("mask", lazy=True),
                io.String.Input("cache_name", default="cached_mask"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Boolean.Input("force_refresh", default=False),
                io.String.Input("name_prefix", default="", optional=True),
            ],
            outputs=[
                io.Mask.Output(display_name="mask"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, mask, cache_name, seed, force_refresh, name_prefix="") -> io.NodeOutput:
        result, status = cls._cache_data(mask, cache_name, seed, force_refresh, name_prefix)
        return io.NodeOutput(result, status)


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
CACHE_CORE_V3_NODES = [
    BD_CacheImage,
    BD_CacheMask,
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
