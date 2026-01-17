"""
Core cache nodes for common ComfyUI data types.

Each node caches a specific data type with lazy evaluation
to skip expensive upstream generation when cache is valid.
"""

from .base import BaseCacheNode
from ...utils.shared import (
    ImageSerializer,
    MaskSerializer,
    LatentSerializer,
    AudioSerializer,
    StringSerializer,
    PickleSerializer,
)


class BD_CacheImage(BaseCacheNode):
    """
    Cache IMAGE tensors to skip expensive image generation.

    First run: generates image, saves to cache as PNG
    Subsequent runs: loads from cache, SKIPS upstream generation!
    """

    input_name = "image"
    input_type = "IMAGE"
    serializer = ImageSerializer
    default_cache_name = "cached_image"
    node_label = "BD Cache Image"

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    FUNCTION = "cache_image"

    def cache_image(self, image, cache_name, seed, force_refresh, name_prefix=""):
        return self._cache_data(image, cache_name, seed, force_refresh, name_prefix)


class BD_CacheMask(BaseCacheNode):
    """Cache MASK tensors to skip expensive mask generation."""

    input_name = "mask"
    input_type = "MASK"
    serializer = MaskSerializer
    default_cache_name = "cached_mask"
    node_label = "BD Cache Mask"

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "status")
    FUNCTION = "cache_mask"

    def cache_mask(self, mask, cache_name, seed, force_refresh, name_prefix=""):
        return self._cache_data(mask, cache_name, seed, force_refresh, name_prefix)


class BD_CacheLatent(BaseCacheNode):
    """Cache LATENT dicts to skip expensive latent generation."""

    input_name = "latent"
    input_type = "LATENT"
    serializer = LatentSerializer
    default_cache_name = "cached_latent"
    node_label = "BD Cache Latent"

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "status")
    FUNCTION = "cache_latent"

    def cache_latent(self, latent, cache_name, seed, force_refresh, name_prefix=""):
        return self._cache_data(latent, cache_name, seed, force_refresh, name_prefix)


class BD_CacheAudio(BaseCacheNode):
    """Cache AUDIO dicts to skip expensive audio generation."""

    input_name = "audio"
    input_type = "AUDIO"
    serializer = AudioSerializer
    default_cache_name = "cached_audio"
    node_label = "BD Cache Audio"

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "status")
    FUNCTION = "cache_audio"

    def cache_audio(self, audio, cache_name, seed, force_refresh, name_prefix=""):
        return self._cache_data(audio, cache_name, seed, force_refresh, name_prefix)


class BD_CacheString(BaseCacheNode):
    """Cache STRING values."""

    input_name = "text"
    input_type = "STRING"
    serializer = StringSerializer
    default_cache_name = "cached_string"
    node_label = "BD Cache String"
    min_cache_size = 0  # Empty strings are valid

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "status")
    FUNCTION = "cache_string"

    def cache_string(self, text, cache_name, seed, force_refresh, name_prefix=""):
        return self._cache_data(text, cache_name, seed, force_refresh, name_prefix)


class BD_CacheAny(BaseCacheNode):
    """
    Cache ANY data type using pickle serialization.

    Use this for custom types that don't have a specialized cache node.
    Falls back to pickle for maximum compatibility.
    """

    input_name = "data"
    input_type = "*"  # Accepts any type
    serializer = PickleSerializer
    default_cache_name = "cached_data"
    node_label = "BD Cache Any"

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("data", "status")
    FUNCTION = "cache_any"

    @classmethod
    def INPUT_TYPES(cls):
        from ...utils.shared import LAZY_OPTIONS
        return {
            "required": {
                "data": ("*", LAZY_OPTIONS),
                "cache_name": ("STRING", {"default": cls.default_cache_name}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_refresh": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "name_prefix": ("STRING", {"default": ""}),
            }
        }

    def cache_any(self, data, cache_name, seed, force_refresh, name_prefix=""):
        return self._cache_data(data, cache_name, seed, force_refresh, name_prefix)


# Node exports
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
