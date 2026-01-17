"""
Base classes for BrainDead cache nodes.

Provides common functionality for all cache node types,
eliminating code duplication across image/mask/latent/audio/etc caches.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Type

from ...utils.shared import (
    LAZY_OPTIONS,
    get_cache_path,
    hash_from_seed,
    check_cache_exists,
    save_to_cache,
)


class BaseCacheNode(ABC):
    """
    Abstract base class for all cache nodes.

    Subclasses must define:
    - input_name: str - The name of the data input ("image", "mask", etc.)
    - input_type: str - The ComfyUI type ("IMAGE", "MASK", etc.)
    - serializer: class - The serializer class to use
    - default_cache_name: str - Default name for cache files
    - node_label: str - Label for log messages

    The base class handles:
    - check_lazy_status logic (skip upstream if cache exists)
    - IS_CHANGED logic (force refresh handling)
    - Common cache load/save workflow
    """

    CATEGORY = "BrainDead/Cache"

    # Subclasses must override these
    input_name: str = "data"
    input_type: str = "ANY"
    serializer: Type = None
    default_cache_name: str = "cached_data"
    node_label: str = "BD Cache"
    min_cache_size: int = 100  # Minimum file size to consider cache valid

    @classmethod
    def INPUT_TYPES(cls):
        """Generate INPUT_TYPES based on class attributes."""
        return {
            "required": {
                cls.input_name: (cls.input_type, LAZY_OPTIONS),
                "cache_name": ("STRING", {"default": cls.default_cache_name}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_refresh": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "name_prefix": ("STRING", {"default": ""}),
            }
        }

    @classmethod
    def get_return_types(cls) -> Tuple[str, ...]:
        """Return the RETURN_TYPES tuple."""
        return (cls.input_type, "STRING")

    @classmethod
    def get_return_names(cls) -> Tuple[str, ...]:
        """Return the RETURN_NAMES tuple."""
        return (cls.input_name, "status")

    def _get_cache_path(self, cache_name: str, seed: int, name_prefix: str = "") -> str:
        """Build the cache file path."""
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        return get_cache_path(full_name, cache_hash, self.serializer.extension)

    def check_lazy_status(self, cache_name, seed, force_refresh, name_prefix="", **kwargs):
        """
        Return [] to skip upstream, [input_name] to evaluate upstream.

        This method is called by ComfyUI's lazy evaluation system.
        """
        if force_refresh:
            print(f"[{self.node_label}] Force refresh - will run upstream")
            return [self.input_name]

        cache_path = self._get_cache_path(cache_name, seed, name_prefix)

        if check_cache_exists(cache_path, min_size=self.min_cache_size):
            print(f"[{self.node_label}] ✓ Cache exists: {os.path.basename(cache_path)} - SKIPPING upstream")
            return []  # Empty list = don't need input, skip upstream

        print(f"[{self.node_label}] No cache found - will run upstream")
        return [self.input_name]

    @classmethod
    def IS_CHANGED(cls, cache_name, seed, force_refresh, name_prefix="", **kwargs):
        """
        Return a unique value when cache should be invalidated.
        """
        if force_refresh:
            return f"force_{time.time()}"
        return f"{name_prefix}_{cache_name}_{seed}"

    def _cache_data(self, data: Any, cache_name: str, seed: int,
                    force_refresh: bool, name_prefix: str = "") -> Tuple[Any, str]:
        """
        Common cache handling logic.

        Returns: (data, status_message)
        """
        cache_path = self._get_cache_path(cache_name, seed, name_prefix)

        # Try to load from cache if not forcing refresh
        if check_cache_exists(cache_path, min_size=self.min_cache_size) and not force_refresh:
            try:
                cached_data = self.serializer.load(cache_path)
                if cached_data is not None:
                    return (cached_data, f"Cache HIT: {os.path.basename(cache_path)}")
            except Exception as e:
                print(f"[{self.node_label}] Cache load failed: {e}")

        # Check if input data is available
        if data is None:
            return (data, "Input is None - cannot cache")

        # Save to cache
        if save_to_cache(cache_path, data, self.serializer):
            status = f"SAVED: {os.path.basename(cache_path)}"
        else:
            status = "Save failed"

        return (data, status)
