"""
V3 API base classes for BrainDead cache nodes.

Provides common functionality for all cache node types using
the ComfyUI V3 API patterns.
"""

import os
import time
from typing import Any, Type

from comfy_api.latest import io

from ...utils.shared import (
    get_cache_path,
    hash_from_seed,
    check_cache_exists,
    save_to_cache,
)


class CacheNodeMixin:
    """
    Mixin providing common cache functionality for V3 nodes.

    Subclasses must define:
    - input_name: str - The name of the data input ("image", "mask", etc.)
    - serializer: class - The serializer class to use
    - node_label: str - Label for log messages
    - min_cache_size: int - Minimum file size to consider cache valid
    """

    input_name: str = "data"
    serializer: Type = None
    node_label: str = "BD Cache"
    min_cache_size: int = 100

    @classmethod
    def _get_cache_path(cls, cache_name: str, seed: int, name_prefix: str = "") -> str:
        """Build the cache file path."""
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        return get_cache_path(full_name, cache_hash, cls.serializer.extension)

    @classmethod
    def check_lazy_status(cls, **kwargs) -> list[str]:
        """
        Return [] to skip upstream, [input_name] to evaluate upstream.

        This method is called by ComfyUI's lazy evaluation system.
        Args come in schema order as kwargs.
        """
        # Extract args - data input comes first but we don't need its value
        cache_name = kwargs.get('cache_name', '')
        seed = kwargs.get('seed', 0)
        force_refresh = kwargs.get('force_refresh', False)
        name_prefix = kwargs.get('name_prefix', '')

        if force_refresh:
            print(f"[{cls.node_label}] Force refresh - will run upstream")
            return [cls.input_name]

        cache_path = cls._get_cache_path(cache_name, seed, name_prefix)

        if check_cache_exists(cache_path, min_size=cls.min_cache_size):
            print(f"[{cls.node_label}] Cache HIT - SKIPPING upstream")
            return []  # Empty list = don't need input, skip upstream

        print(f"[{cls.node_label}] No cache - will run upstream")
        return [cls.input_name]

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> str:
        """
        Return a unique value when cache should be invalidated.
        Replaces IS_CHANGED in V3 API.
        """
        cache_name = kwargs.get('cache_name', '')
        seed = kwargs.get('seed', 0)
        force_refresh = kwargs.get('force_refresh', False)
        name_prefix = kwargs.get('name_prefix', '')

        if force_refresh:
            return f"force_{time.time()}"
        return f"{name_prefix}_{cache_name}_{seed}"

    @classmethod
    def _cache_data(cls, data: Any, cache_name: str, seed: int,
                    force_refresh: bool, name_prefix: str = "") -> tuple[Any, str]:
        """
        Common cache handling logic.

        Returns: (data, status_message)
        """
        cache_path = cls._get_cache_path(cache_name, seed, name_prefix)

        # Try to load from cache if not forcing refresh
        if check_cache_exists(cache_path, min_size=cls.min_cache_size) and not force_refresh:
            try:
                cached_data = cls.serializer.load(cache_path)
                if cached_data is not None:
                    return (cached_data, f"Cache HIT: {os.path.basename(cache_path)}")
            except Exception as e:
                print(f"[{cls.node_label}] Cache load failed: {e}")

        # Check if input data is available
        if data is None:
            return (data, "Input is None - cannot cache")

        # Save to cache
        if save_to_cache(cache_path, data, cls.serializer):
            status = f"SAVED: {os.path.basename(cache_path)}"
        else:
            status = "Save failed"

        return (data, status)
