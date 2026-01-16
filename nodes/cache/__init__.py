"""
BrainDead Cache Nodes for ComfyUI
Created by BizaNator for BrainDeadGuild.com
A Biloxi Studios Inc. Production

Universal checkpoint/cache nodes that work with ANY data type.
Generate once, reuse forever - skip expensive regeneration on subsequent runs.

Features:
- Lazy evaluation: upstream nodes are SKIPPED when cache is valid
- Seed-based cache keys: change seed to force regeneration
- Type-specific serialization: PNG for images, WAV for audio, PLY for meshes
"""

import os
import pickle

from ...utils.shared import (
    LAZY_OPTIONS,
    CACHE_DIR,
    OUTPUT_DIR,
    get_cache_path,
    hash_from_seed,
    hash_from_params,
    check_cache_exists,
    save_to_cache,
    ImageSerializer,
    MaskSerializer,
    LatentSerializer,
    AudioSerializer,
    StringSerializer,
    PickleSerializer,
    # Workflow version utilities
    WORKFLOW_VERSIONS_DIR,
    hash_workflow_structure,
    hash_workflow_full,
    auto_workflow_id,
    list_workflow_versions,
    save_workflow_version,
    load_workflow_version,
    compare_workflow_versions,
    clear_workflow_versions,
)


# =============================================================================
# BD_CacheImage
# =============================================================================

class BD_CacheImage:
    """
    Cache IMAGE tensors to skip expensive image generation on subsequent runs.

    USAGE:
    1. Connect expensive image generation output to 'image' input
    2. Set a unique cache_name (e.g., "step1_generated_face")
    3. Use 'seed' to control cache invalidation
    4. Output connects to downstream nodes

    First run: generates image, saves to cache as PNG
    Subsequent runs: loads from cache, SKIPS upstream generation!
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", LAZY_OPTIONS),
                "cache_name": ("STRING", {"default": "cached_image"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_refresh": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "name_prefix": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    FUNCTION = "cache_image"
    CATEGORY = "BrainDead/Cache"

    def check_lazy_status(self, cache_name, seed, force_refresh, image=None, name_prefix=""):
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, ImageSerializer.extension)

        if check_cache_exists(cache_path, min_size=100) and not force_refresh:
            try:
                cached_image = ImageSerializer.load(cache_path)
                if cached_image is not None:
                    status = f"Cache HIT: {os.path.basename(cache_path)}"
                    print(f"[BD Cache Image] {status}")
                    return (cached_image, status)
            except Exception as e:
                print(f"[BD Cache Image] Error: {e}")
                return ["image"]
        return ["image"]

    @classmethod
    def IS_CHANGED(cls, cache_name, seed, force_refresh, image=None, name_prefix=""):
        # When force_refresh is True, always return a unique value to force re-execution
        if force_refresh:
            import time
            return f"force_{time.time()}"
        # Otherwise return a stable hash based on cache parameters
        return f"{name_prefix}_{cache_name}_{seed}"

    def cache_image(self, image, cache_name, seed, force_refresh, name_prefix=""):
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, ImageSerializer.extension)

        if check_cache_exists(cache_path, min_size=100) and not force_refresh:
            try:
                cached_image = ImageSerializer.load(cache_path)
                if cached_image is not None:
                    return (cached_image, f"Cache HIT (main): {os.path.basename(cache_path)}")
            except:
                pass

        if image is None:
            return (image, "Input is None - cannot cache")

        if save_to_cache(cache_path, image, ImageSerializer):
            status = f"SAVED: {os.path.basename(cache_path)}"
        else:
            status = "Save failed"
        return (image, status)


# =============================================================================
# BD_CacheMask
# =============================================================================

class BD_CacheMask:
    """Cache MASK tensors to skip expensive mask generation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", LAZY_OPTIONS),
                "cache_name": ("STRING", {"default": "cached_mask"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_refresh": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "name_prefix": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "status")
    FUNCTION = "cache_mask"
    CATEGORY = "BrainDead/Cache"

    def check_lazy_status(self, cache_name, seed, force_refresh, mask=None, name_prefix=""):
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, MaskSerializer.extension)

        if check_cache_exists(cache_path, min_size=100) and not force_refresh:
            try:
                cached_mask = MaskSerializer.load(cache_path)
                if cached_mask is not None:
                    return (cached_mask, f"Cache HIT: {os.path.basename(cache_path)}")
            except:
                return ["mask"]
        return ["mask"]

    @classmethod
    def IS_CHANGED(cls, cache_name, seed, force_refresh, mask=None, name_prefix=""):
        if force_refresh:
            import time
            return f"force_{time.time()}"
        return f"{name_prefix}_{cache_name}_{seed}"

    def cache_mask(self, mask, cache_name, seed, force_refresh, name_prefix=""):
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, MaskSerializer.extension)

        if check_cache_exists(cache_path, min_size=100) and not force_refresh:
            try:
                cached_mask = MaskSerializer.load(cache_path)
                if cached_mask is not None:
                    return (cached_mask, f"Cache HIT (main): {os.path.basename(cache_path)}")
            except:
                pass

        if mask is None:
            return (mask, "Input is None - cannot cache")

        if save_to_cache(cache_path, mask, MaskSerializer):
            status = f"SAVED: {os.path.basename(cache_path)}"
        else:
            status = "Save failed"
        return (mask, status)


# =============================================================================
# BD_CacheLatent
# =============================================================================

class BD_CacheLatent:
    """Cache LATENT tensors to skip expensive VAE encoding or sampling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", LAZY_OPTIONS),
                "cache_name": ("STRING", {"default": "cached_latent"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_refresh": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "name_prefix": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "status")
    FUNCTION = "cache_latent"
    CATEGORY = "BrainDead/Cache"

    def check_lazy_status(self, cache_name, seed, force_refresh, latent=None, name_prefix=""):
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, LatentSerializer.extension)

        if check_cache_exists(cache_path, min_size=100) and not force_refresh:
            try:
                cached_latent = LatentSerializer.load(cache_path)
                if cached_latent is not None:
                    return (cached_latent, f"Cache HIT: {os.path.basename(cache_path)}")
            except:
                return ["latent"]
        return ["latent"]

    @classmethod
    def IS_CHANGED(cls, cache_name, seed, force_refresh, latent=None, name_prefix=""):
        if force_refresh:
            import time
            return f"force_{time.time()}"
        return f"{name_prefix}_{cache_name}_{seed}"

    def cache_latent(self, latent, cache_name, seed, force_refresh, name_prefix=""):
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, LatentSerializer.extension)

        if check_cache_exists(cache_path, min_size=100) and not force_refresh:
            try:
                cached_latent = LatentSerializer.load(cache_path)
                if cached_latent is not None:
                    return (cached_latent, f"Cache HIT (main): {os.path.basename(cache_path)}")
            except:
                pass

        if latent is None:
            return (latent, "Input is None - cannot cache")

        if save_to_cache(cache_path, latent, LatentSerializer):
            status = f"SAVED: {os.path.basename(cache_path)}"
        else:
            status = "Save failed"
        return (latent, status)


# =============================================================================
# BD_CacheAudio
# =============================================================================

class BD_CacheAudio:
    """Cache AUDIO data to skip expensive audio generation (TTS, voice cloning)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", LAZY_OPTIONS),
                "cache_name": ("STRING", {"default": "cached_audio"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_refresh": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "name_prefix": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "status")
    FUNCTION = "cache_audio"
    CATEGORY = "BrainDead/Cache"

    def check_lazy_status(self, cache_name, seed, force_refresh, audio=None, name_prefix=""):
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, AudioSerializer.extension)

        if check_cache_exists(cache_path, min_size=100) and not force_refresh:
            try:
                cached_audio = AudioSerializer.load(cache_path)
                if cached_audio is not None:
                    return (cached_audio, f"Cache HIT: {os.path.basename(cache_path)}")
            except:
                return ["audio"]
        return ["audio"]

    @classmethod
    def IS_CHANGED(cls, cache_name, seed, force_refresh, audio=None, name_prefix=""):
        if force_refresh:
            import time
            return f"force_{time.time()}"
        return f"{name_prefix}_{cache_name}_{seed}"

    def cache_audio(self, audio, cache_name, seed, force_refresh, name_prefix=""):
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, AudioSerializer.extension)

        if check_cache_exists(cache_path, min_size=100) and not force_refresh:
            try:
                cached_audio = AudioSerializer.load(cache_path)
                if cached_audio is not None:
                    return (cached_audio, f"Cache HIT (main): {os.path.basename(cache_path)}")
            except:
                pass

        if audio is None:
            return (audio, "Input is None - cannot cache")

        if save_to_cache(cache_path, audio, AudioSerializer):
            status = f"SAVED: {os.path.basename(cache_path)}"
        else:
            status = "Save failed"
        return (audio, status)


# =============================================================================
# BD_CacheString
# =============================================================================

class BD_CacheString:
    """Cache STRING data (prompts, generated text, etc.)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {**LAZY_OPTIONS, "multiline": True, "forceInput": True}),
                "cache_name": ("STRING", {"default": "cached_text"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_refresh": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "name_prefix": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "status")
    FUNCTION = "cache_string"
    CATEGORY = "BrainDead/Cache"

    def check_lazy_status(self, cache_name, seed, force_refresh, text=None, name_prefix=""):
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, StringSerializer.extension)

        if check_cache_exists(cache_path, min_size=1) and not force_refresh:
            try:
                cached_text = StringSerializer.load(cache_path)
                if cached_text is not None:
                    return (cached_text, f"Cache HIT: {os.path.basename(cache_path)}")
            except:
                return ["text"]
        return ["text"]

    @classmethod
    def IS_CHANGED(cls, cache_name, seed, force_refresh, text=None, name_prefix=""):
        if force_refresh:
            import time
            return f"force_{time.time()}"
        return f"{name_prefix}_{cache_name}_{seed}"

    def cache_string(self, text, cache_name, seed, force_refresh, name_prefix=""):
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, StringSerializer.extension)

        if check_cache_exists(cache_path, min_size=1) and not force_refresh:
            try:
                cached_text = StringSerializer.load(cache_path)
                if cached_text is not None:
                    return (cached_text, f"Cache HIT (main): {os.path.basename(cache_path)}")
            except:
                pass

        if text is None:
            return (text, "Input is None - cannot cache")

        if save_to_cache(cache_path, text, StringSerializer):
            status = f"SAVED: {os.path.basename(cache_path)}"
        else:
            status = "Save failed"
        return (text, status)


# =============================================================================
# BD_CacheAny
# =============================================================================

class BD_CacheAny:
    """
    Cache ANY data type using pickle serialization.
    Works with: conditioning, models, embeddings, custom types, etc.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("*", LAZY_OPTIONS),
                "cache_name": ("STRING", {"default": "cached_data"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_refresh": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "name_prefix": ("STRING", {"default": ""}),
                "extension": ("STRING", {"default": ".pkl"}),
            }
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("data", "status")
    FUNCTION = "cache_any"
    CATEGORY = "BrainDead/Cache"

    def _get_extension(self, extension):
        ext = extension.strip()
        if not ext:
            return ".pkl"
        if not ext.startswith('.'):
            ext = '.' + ext
        return ext

    def check_lazy_status(self, cache_name, seed, force_refresh, data=None, name_prefix="", extension=".pkl"):
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        ext = self._get_extension(extension)
        cache_path = get_cache_path(full_name, cache_hash, ext)

        if check_cache_exists(cache_path, min_size=10) and not force_refresh:
            try:
                cached_data = PickleSerializer.load(cache_path)
                if cached_data is not None:
                    return (cached_data, f"Cache HIT: {os.path.basename(cache_path)}")
            except:
                return ["data"]
        return ["data"]

    @classmethod
    def IS_CHANGED(cls, cache_name, seed, force_refresh, data=None, name_prefix="", extension=".pkl"):
        if force_refresh:
            import time
            return f"force_{time.time()}"
        return f"{name_prefix}_{cache_name}_{seed}"

    def cache_any(self, data, cache_name, seed, force_refresh, name_prefix="", extension=".pkl"):
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        ext = self._get_extension(extension)
        cache_path = get_cache_path(full_name, cache_hash, ext)

        if check_cache_exists(cache_path, min_size=10) and not force_refresh:
            try:
                cached_data = PickleSerializer.load(cache_path)
                if cached_data is not None:
                    return (cached_data, f"Cache HIT (main): {os.path.basename(cache_path)}")
            except:
                pass

        if data is None:
            return (data, "Input is None - cannot cache")

        if save_to_cache(cache_path, data, PickleSerializer):
            status = f"SAVED: {os.path.basename(cache_path)}"
        else:
            status = "Save failed"
        return (data, status)


# =============================================================================
# BD_CacheMesh (3D)
# =============================================================================

# Try to import trimesh
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


class BD_CacheMesh:
    """
    Cache TRIMESH objects to skip expensive mesh generation.
    Saves as PLY format for human-readable/editable files.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH", LAZY_OPTIONS),
                "cache_name": ("STRING", {"default": "cached_mesh"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_refresh": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "name_prefix": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "status")
    FUNCTION = "cache_mesh"
    CATEGORY = "BrainDead/Cache"

    def check_lazy_status(self, cache_name, seed, force_refresh, mesh=None, name_prefix=""):
        if not HAS_TRIMESH:
            return ["mesh"]

        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, ".ply")

        if check_cache_exists(cache_path, min_size=100) and not force_refresh:
            try:
                cached_mesh = trimesh.load(cache_path)
                if cached_mesh is not None and hasattr(cached_mesh, 'vertices'):
                    return (cached_mesh, f"Cache HIT: {os.path.basename(cache_path)}")
            except:
                return ["mesh"]
        return ["mesh"]

    @classmethod
    def IS_CHANGED(cls, cache_name, seed, force_refresh, mesh=None, name_prefix=""):
        if force_refresh:
            import time
            return f"force_{time.time()}"
        return f"{name_prefix}_{cache_name}_{seed}"

    def cache_mesh(self, mesh, cache_name, seed, force_refresh, name_prefix=""):
        if not HAS_TRIMESH:
            return (mesh, "ERROR: trimesh not installed")

        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, ".ply")

        if check_cache_exists(cache_path, min_size=100) and not force_refresh:
            try:
                cached_mesh = trimesh.load(cache_path)
                if cached_mesh is not None and hasattr(cached_mesh, 'vertices'):
                    return (cached_mesh, f"Cache HIT (main): {os.path.basename(cache_path)}")
            except:
                pass

        if mesh is None:
            return (mesh, "Input is None - cannot cache")

        try:
            trimesh.exchange.export.export_mesh(mesh, cache_path, file_type='ply')
            status = f"SAVED: {os.path.basename(cache_path)}"
        except Exception as e:
            status = f"Save failed: {e}"
        return (mesh, status)


# =============================================================================
# BD_ClearCache
# =============================================================================

class BD_ClearCache:
    """
    Clear cached files from BrainDead_Cache/ folder by name pattern.

    ONLY affects: BrainDead_Cache/ folder (cache nodes)
    DOES NOT affect: output/ folder (BD Save File)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pattern": ("STRING", {"default": "*"}),
                "confirm_clear": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "clear_cache"
    CATEGORY = "BrainDead/Cache"
    OUTPUT_NODE = True

    def clear_cache(self, pattern, confirm_clear):
        if not confirm_clear:
            return (f"Set confirm_clear=True to delete: {pattern}",)

        if not os.path.exists(CACHE_DIR):
            return ("Cache directory empty",)

        from glob import glob

        search_pattern = os.path.join(CACHE_DIR, f"{pattern}*")
        matching_files = glob(search_pattern)

        if not matching_files:
            return (f"No files matching: {pattern}",)

        deleted_count = 0
        deleted_size = 0
        for filepath in matching_files:
            try:
                file_size = os.path.getsize(filepath)
                os.remove(filepath)
                deleted_count += 1
                deleted_size += file_size
            except Exception as e:
                print(f"[BD Cache] Error deleting {filepath}: {e}")

        size_mb = deleted_size / (1024 * 1024)
        return (f"Deleted {deleted_count} files ({size_mb:.1f} MB)",)


# =============================================================================
# BD_SaveFile
# =============================================================================

class BD_SaveFile:
    """
    Save ANY data type to file in native format, output the file path.

    SAVES TO: ComfyUI output/ folder (NOT BrainDead_Cache)
    Files saved here are NOT affected by BD Clear Cache.

    Supported types: IMAGE (PNG), MASK (PNG), AUDIO (WAV), LATENT, STRING (TXT), TRIMESH (PLY/OBJ/GLB)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("*",),
                "filename": ("STRING", {"default": "saved_file"}),
                "skip_if_exists": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "name_prefix": ("STRING", {"default": ""}),
                "extension": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_path", "status")
    FUNCTION = "save_file"
    CATEGORY = "BrainDead/Cache"
    OUTPUT_NODE = True

    def _detect_type_and_save(self, data, filepath):
        import torch
        import numpy as np

        # IMAGE tensor
        if isinstance(data, torch.Tensor):
            shape = data.shape
            if len(shape) == 4 and shape[-1] in [3, 4]:
                if not filepath.endswith('.png'):
                    filepath = filepath.rsplit('.', 1)[0] + '.png' if '.' in filepath else filepath + '.png'
                ImageSerializer.save(filepath, data)
                return filepath, "IMAGE"
            elif len(shape) in [2, 3] and shape[-1] not in [3, 4]:
                if not filepath.endswith('.png'):
                    filepath = filepath.rsplit('.', 1)[0] + '.png' if '.' in filepath else filepath + '.png'
                MaskSerializer.save(filepath, data)
                return filepath, "MASK"

        # LATENT dict
        if isinstance(data, dict) and 'samples' in data:
            if not filepath.endswith('.latent'):
                filepath = filepath.rsplit('.', 1)[0] + '.latent' if '.' in filepath else filepath + '.latent'
            LatentSerializer.save(filepath, data)
            return filepath, "LATENT"

        # AUDIO dict
        if isinstance(data, dict) and 'waveform' in data and 'sample_rate' in data:
            if not filepath.endswith('.wav'):
                filepath = filepath.rsplit('.', 1)[0] + '.wav' if '.' in filepath else filepath + '.wav'
            AudioSerializer.save(filepath, data)
            return filepath, "AUDIO"

        # STRING
        if isinstance(data, str):
            if not filepath.endswith('.txt'):
                filepath = filepath.rsplit('.', 1)[0] + '.txt' if '.' in filepath else filepath + '.txt'
            StringSerializer.save(filepath, data)
            return filepath, "STRING"

        # TRIMESH
        if hasattr(data, 'vertices') and hasattr(data, 'faces'):
            if HAS_TRIMESH:
                if not any(filepath.endswith(ext) for ext in ['.ply', '.obj', '.glb', '.gltf', '.stl']):
                    filepath = filepath.rsplit('.', 1)[0] + '.ply' if '.' in filepath else filepath + '.ply'
                ext = filepath.rsplit('.', 1)[-1].lower()
                trimesh.exchange.export.export_mesh(data, filepath, file_type=ext)
                return filepath, "TRIMESH"

        # Fallback: pickle
        if not filepath.endswith('.pkl'):
            filepath = filepath.rsplit('.', 1)[0] + '.pkl' if '.' in filepath else filepath + '.pkl'
        PickleSerializer.save(filepath, data)
        return filepath, "GENERIC"

    def save_file(self, data, filename, skip_if_exists=True, name_prefix="", extension=""):
        if name_prefix:
            full_name = f"{name_prefix}_{filename}"
        else:
            full_name = filename

        if extension:
            ext = extension.strip()
            if not ext.startswith('.'):
                ext = '.' + ext
            full_name = full_name + ext

        filepath = os.path.join(OUTPUT_DIR, full_name)

        # Handle subdirectories
        if '/' in full_name or '\\' in full_name:
            subdir = os.path.dirname(filepath)
            os.makedirs(subdir, exist_ok=True)

        try:
            final_path, data_type = self._detect_type_and_save(data, filepath)

            if skip_if_exists and os.path.exists(final_path):
                return (final_path, f"EXISTS: {os.path.basename(final_path)}")

            status = f"Saved {data_type}: {os.path.basename(final_path)}"
            return (final_path, status)
        except Exception as e:
            return ("", f"Save failed: {e}")


# =============================================================================
# BD_LoadImage
# =============================================================================

class BD_LoadImage:
    """Load an image from a file path (STRING input)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "status")
    FUNCTION = "load_image"
    CATEGORY = "BrainDead/Cache"

    def load_image(self, file_path):
        from PIL import Image
        import torch
        import numpy as np

        if not file_path or not os.path.exists(file_path):
            empty_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_img, empty_mask, f"File not found: {file_path}")

        try:
            pil_img = Image.open(file_path)

            if pil_img.mode == 'RGBA':
                r, g, b, a = pil_img.split()
                pil_img = Image.merge('RGB', (r, g, b))
                mask_np = np.array(a).astype(np.float32) / 255.0
                mask = torch.from_numpy(mask_np).unsqueeze(0)
            elif pil_img.mode == 'L':
                mask_np = np.array(pil_img).astype(np.float32) / 255.0
                mask = torch.from_numpy(mask_np).unsqueeze(0)
                pil_img = pil_img.convert('RGB')
            else:
                pil_img = pil_img.convert('RGB')
                mask = torch.zeros((1, pil_img.height, pil_img.width), dtype=torch.float32)

            img_np = np.array(pil_img).astype(np.float32) / 255.0
            image = torch.from_numpy(img_np).unsqueeze(0)

            return (image, mask, f"Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            empty_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_img, empty_mask, f"Load failed: {e}")


# =============================================================================
# BD_LoadMesh
# =============================================================================

class BD_LoadMesh:
    """Load a 3D mesh from a file path (STRING input)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "status")
    FUNCTION = "load_mesh"
    CATEGORY = "BrainDead/Cache"

    def load_mesh(self, file_path):
        if not HAS_TRIMESH:
            return (None, "ERROR: trimesh not installed")

        if not file_path or not os.path.exists(file_path):
            return (None, f"File not found: {file_path}")

        try:
            mesh = trimesh.load(file_path)

            if isinstance(mesh, trimesh.Scene):
                meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if meshes:
                    mesh = trimesh.util.concatenate(meshes)
                else:
                    return (None, "No meshes found in scene")

            return (mesh, f"Loaded: {os.path.basename(file_path)} ({len(mesh.vertices)} verts)")
        except Exception as e:
            return (None, f"Load failed: {e}")


# =============================================================================
# BD_LoadAudio
# =============================================================================

class BD_LoadAudio:
    """Load audio from a file path (STRING input)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "status")
    FUNCTION = "load_audio"
    CATEGORY = "BrainDead/Cache"

    def load_audio(self, file_path):
        import torch

        if not file_path or not os.path.exists(file_path):
            return (None, f"File not found: {file_path}")

        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(file_path)
            audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
            duration = waveform.shape[-1] / sample_rate
            return (audio, f"Loaded: {os.path.basename(file_path)} ({duration:.1f}s)")
        except ImportError:
            return (None, "ERROR: torchaudio not installed")
        except Exception as e:
            return (None, f"Load failed: {e}")


# =============================================================================
# BD_LoadText
# =============================================================================

class BD_LoadText:
    """Load text from a file path (STRING input)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "status")
    FUNCTION = "load_text"
    CATEGORY = "BrainDead/Cache"

    def load_text(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return ("", f"File not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return (text, f"Loaded: {os.path.basename(file_path)} ({len(text)} chars)")
        except Exception as e:
            return ("", f"Load failed: {e}")


# =============================================================================
# BD_WorkflowVersionCache
# =============================================================================

# Track last known workflow hash per workflow_id to detect changes
_WORKFLOW_HASH_CACHE = {}


class BD_WorkflowVersionCache:
    """
    Automatically save workflow versions when changes are detected.

    USAGE:
    1. Add this node anywhere in your workflow
    2. Set a workflow_id or leave empty for auto-generated ID based on structure
    3. Configure max_versions to control how many versions are kept
    4. Workflow is auto-saved on first run and whenever changes are detected

    Auto-ID: When workflow_id is empty, generates ID from workflow structure hash.
    This means structurally identical workflows share version history.

    Storage: Workflow files are saved as clean JSON (drag-and-drop compatible
    with any ComfyUI instance). Metadata stored separately in .meta.json files.

    Ideal for:
    - Crash recovery
    - Version history / rollback
    - Tracking workflow evolution
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_id": ("STRING", {"default": "", "placeholder": "Leave empty for auto-detect"}),
                "max_versions": ("INT", {"default": 50, "min": 0, "max": 999}),
                "save_on_any_change": ("BOOLEAN", {"default": True}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "trigger": ("*",),  # Optional connection to force execution order
                "description": ("STRING", {"default": "", "multiline": False}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT",
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("status", "workflow_id", "version_count")
    FUNCTION = "cache_workflow_version"
    CATEGORY = "BrainDead/Cache"
    OUTPUT_NODE = True
    DESCRIPTION = """
Automatically save workflow versions for backup and recovery.

RECOMMENDED: Set a manual workflow_id (e.g., "my_project") to track
your workflow's evolution over time. Auto-generated IDs change when
you add/remove nodes, starting fresh at v1.

Inputs:
- workflow_id: Name for this workflow (leave empty for auto-detect)
- max_versions: How many versions to keep (0 = unlimited)
- save_on_any_change: Currently uses structure detection
- enabled: Toggle versioning on/off
- trigger: Optional input to control execution order
- description: Note to attach to saved version

Outputs:
- status: "Saved vN" or "No changes (vN current)"
- workflow_id: The effective ID being used
- version_count: Total saved versions

Storage: output/BrainDead_Cache/workflow_versions/
Files are clean ComfyUI JSON - drag & drop to restore.
"""

    @classmethod
    def IS_CHANGED(cls, workflow_id, max_versions, save_on_any_change, enabled,
                   trigger=None, description="", extra_pnginfo=None, prompt=None):
        # Always execute this node on every run so it can check for workflow changes
        # The main function handles deduplication via hash comparison
        import time
        return time.time()

    def cache_workflow_version(self, workflow_id, max_versions, save_on_any_change, enabled,
                                trigger=None, description="", extra_pnginfo=None, prompt=None):
        # Get workflow data from ComfyUI's extra_pnginfo
        workflow_data = None
        if extra_pnginfo and isinstance(extra_pnginfo, dict):
            workflow_data = extra_pnginfo.get('workflow')

        if not workflow_data:
            return ("No workflow data available", workflow_id or "unknown", 0)

        # Auto-detect workflow_id if not provided
        effective_id = workflow_id.strip() if workflow_id else auto_workflow_id(workflow_data)

        if not enabled:
            versions = list_workflow_versions(effective_id)
            return ("Disabled", effective_id, len(versions))

        # Calculate current hash
        if save_on_any_change:
            current_hash = hash_workflow_full(workflow_data)
        else:
            current_hash = hash_workflow_structure(workflow_data)

        cache_key = f"{effective_id}_{'full' if save_on_any_change else 'struct'}"
        last_hash = _WORKFLOW_HASH_CACHE.get(cache_key)

        # Check if we need to save a new version
        existing_versions = list_workflow_versions(effective_id)

        if last_hash == current_hash and existing_versions:
            # No change detected, skip saving
            status = f"No changes (v{existing_versions[0]['version']} current)"
            return (status, effective_id, len(existing_versions))

        # Check if the latest version has the same structure (avoid duplicate saves)
        # Use structure hash for comparison since full hash includes volatile execution data
        if existing_versions:
            latest_struct_hash = existing_versions[0].get('structure_hash', '')  # 8 chars (truncated)
            current_struct_hash = hash_workflow_structure(workflow_data)
            # Compare structure hashes - this ignores widget values that change during execution
            if latest_struct_hash == current_struct_hash[:8]:
                _WORKFLOW_HASH_CACHE[cache_key] = current_hash
                status = f"No changes (v{existing_versions[0]['version']} current)"
                return (status, effective_id, len(existing_versions))

        # Save new version
        desc = description if description else "auto-saved"
        success, message, version_num = save_workflow_version(
            effective_id, workflow_data, desc, max_versions
        )

        if success:
            _WORKFLOW_HASH_CACHE[cache_key] = current_hash
            status = f"Saved v{version_num}"
            print(f"[BD Workflow Version] {status} for '{effective_id}'")
        else:
            status = f"Error: {message}"
            print(f"[BD Workflow Version] {status}")

        # Get updated count
        updated_versions = list_workflow_versions(effective_id)
        return (status, effective_id, len(updated_versions))


# =============================================================================
# BD_WorkflowVersionList
# =============================================================================

class BD_WorkflowVersionList:
    """
    List all saved versions for a workflow with metadata.

    Outputs a formatted list showing version number, timestamp, node count,
    and hash for each saved version. Useful for reviewing history without
    filesystem access.

    Tip: Connect the workflow_id output from BD Workflow Version Cache
    to this node's workflow_id input for seamless integration.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_id": ("STRING", {"default": ""}),
                "show_hashes": ("BOOLEAN", {"default": True}),
                "max_display": ("INT", {"default": 20, "min": 1, "max": 100}),
            },
            "optional": {
                "compare_version_a": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "compare_version_b": ("INT", {"default": 0, "min": 0, "max": 9999}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("version_list", "diff_result", "total_versions")
    FUNCTION = "list_versions"
    CATEGORY = "BrainDead/Cache"
    DESCRIPTION = """
List all saved versions for a workflow.

Inputs:
- workflow_id: Connect from BD Workflow Version Cache output
- show_hashes: Include hash in output table
- max_display: Limit number of versions shown
- compare_version_a/b: Compare two versions (shows diff)

Outputs:
- version_list: Formatted table of versions
- diff_result: Node differences between compared versions
- total_versions: Count of saved versions

Tip: Connect workflow_id output from BD Workflow Version Cache
to automatically use the same ID.
"""

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always refresh the list
        import time
        return time.time()

    def list_versions(self, workflow_id, show_hashes=True, max_display=20,
                      compare_version_a=0, compare_version_b=0):
        if not workflow_id or not workflow_id.strip():
            return ("Error: workflow_id is required. Connect from BD Workflow Version Cache or enter manually.", "", 0)

        workflow_id = workflow_id.strip()
        versions = list_workflow_versions(workflow_id)

        if not versions:
            return (f"No versions found for '{workflow_id}'", "", 0)

        # Build formatted list
        lines = [f"Workflow: {workflow_id}", f"Total versions: {len(versions)}", ""]
        lines.append("Version | Timestamp           | Nodes | Hash")
        lines.append("-" * 50)

        for v in versions[:max_display]:
            timestamp = v['timestamp'][:19] if v['timestamp'] else 'Unknown'
            hash_str = f" | {v['workflow_hash']}" if show_hashes else ""
            desc_str = f" ({v['description']})" if v['description'] and v['description'] != 'auto-saved' else ""
            lines.append(f"v{v['version']:4d}   | {timestamp} | {v['node_count']:5d}{hash_str}{desc_str}")

        if len(versions) > max_display:
            lines.append(f"... and {len(versions) - max_display} more versions")

        version_list = "\n".join(lines)

        # Handle comparison if requested
        diff_result = ""
        if compare_version_a > 0 and compare_version_b > 0:
            diff = compare_workflow_versions(workflow_id, compare_version_a, compare_version_b)
            if 'error' in diff:
                diff_result = diff['error']
            else:
                diff_result = f"Comparing v{compare_version_a} -> v{compare_version_b}:\n{diff['summary']}"

        return (version_list, diff_result, len(versions))


# =============================================================================
# BD_WorkflowVersionRestore
# =============================================================================

class BD_WorkflowVersionRestore:
    """
    Restore a specific workflow version and output as JSON.

    Outputs the workflow JSON string for manual import or the file path
    for download. Use version_number to select which version to restore.
    Set version_number to 0 to restore the latest version.

    Output files are clean ComfyUI workflow JSON - drag and drop into
    any ComfyUI instance (no special nodes required to load).

    Tip: Connect the workflow_id output from BD Workflow Version Cache
    to this node's workflow_id input for seamless integration.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_id": ("STRING", {"default": ""}),
                "version_number": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "save_to_file": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "output_filename": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("workflow_json", "file_path", "status")
    FUNCTION = "restore_version"
    CATEGORY = "BrainDead/Cache"
    OUTPUT_NODE = True
    DESCRIPTION = """
Restore a saved workflow version.

Inputs:
- workflow_id: Connect from BD Workflow Version Cache output
- version_number: Which version to restore (0 = latest)
- save_to_file: Export to output/ folder for download
- output_filename: Custom filename (optional)

Outputs:
- workflow_json: Full workflow as JSON string
- file_path: Path to exported file (if save_to_file=True)
- status: Result message

To restore: Drag the exported JSON file into ComfyUI,
or copy workflow_json into a .json file manually.
"""

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Check each time in case versions changed
        import time
        return time.time()

    def restore_version(self, workflow_id, version_number, save_to_file, output_filename=""):
        import json

        if not workflow_id or not workflow_id.strip():
            return ("", "", "Error: workflow_id is required. Connect from BD Workflow Version Cache or enter manually.")

        workflow_id = workflow_id.strip()
        versions = list_workflow_versions(workflow_id)

        if not versions:
            return ("", "", f"No versions found for '{workflow_id}'")

        # Version 0 means latest
        if version_number == 0:
            target_version = versions[0]['version']
        else:
            target_version = version_number

        # Load the version
        workflow_data, metadata = load_workflow_version(workflow_id, target_version)

        if workflow_data is None:
            return ("", "", f"Error: {metadata}")

        # Convert to JSON string
        workflow_json = json.dumps(workflow_data, indent=2)

        # Optionally save to file for download
        file_path = ""
        if save_to_file:
            if output_filename:
                filename = output_filename
                if not filename.endswith('.json'):
                    filename += '.json'
            else:
                filename = f"{workflow_id}_v{target_version}.json"

            file_path = os.path.join(OUTPUT_DIR, filename)

            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(workflow_json)
            except Exception as e:
                return (workflow_json, "", f"Restored v{target_version} but failed to save file: {e}")

        timestamp = metadata.get('timestamp', 'Unknown')[:19]
        status = f"Restored v{target_version} ({timestamp}, {metadata.get('node_count', 0)} nodes)"

        if file_path:
            status += f" - Saved to {os.path.basename(file_path)}"

        return (workflow_json, file_path, status)


# =============================================================================
# BD_WorkflowVersionClear
# =============================================================================

class BD_WorkflowVersionClear:
    """
    Clear saved workflow versions.

    Can delete all versions or keep the N most recent ones.
    Requires confirm_clear=True to actually delete files.

    Tip: Connect the workflow_id output from BD Workflow Version Cache
    to this node's workflow_id input for seamless integration.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_id": ("STRING", {"default": ""}),
                "keep_latest": ("INT", {"default": 0, "min": 0, "max": 999}),
                "confirm_clear": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "clear_versions"
    CATEGORY = "BrainDead/Cache"
    OUTPUT_NODE = True
    DESCRIPTION = """
Delete saved workflow versions to free disk space.

Inputs:
- workflow_id: Connect from BD Workflow Version Cache output
- keep_latest: Keep N most recent versions (0 = delete all)
- confirm_clear: Must be True to actually delete

Safety: First shows what would be deleted. Set confirm_clear=True
only when ready to delete.

Example: keep_latest=5 deletes all but the 5 newest versions.
"""

    def clear_versions(self, workflow_id, keep_latest, confirm_clear):
        if not workflow_id or not workflow_id.strip():
            return ("Error: workflow_id is required. Connect from BD Workflow Version Cache or enter manually.",)

        workflow_id = workflow_id.strip()
        versions = list_workflow_versions(workflow_id)

        if not versions:
            return (f"No versions found for '{workflow_id}'",)

        if not confirm_clear:
            to_delete = len(versions) - keep_latest if keep_latest > 0 else len(versions)
            to_delete = max(0, to_delete)
            return (f"Would delete {to_delete} of {len(versions)} versions. Set confirm_clear=True to proceed.",)

        deleted, message = clear_workflow_versions(workflow_id, keep_latest)

        # Clear the hash cache for this workflow
        for key in list(_WORKFLOW_HASH_CACHE.keys()):
            if key.startswith(workflow_id):
                del _WORKFLOW_HASH_CACHE[key]

        return (message,)


# =============================================================================
# BD_CacheTrellis2Conditioning
# =============================================================================

class BD_CacheTrellis2Conditioning:
    """Cache Trellis2 conditioning output to skip image preprocessing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("TRELLIS2_CONDITIONING", LAZY_OPTIONS),
                "cache_name": ("STRING", {"default": "trellis2_cond"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_refresh": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "name_prefix": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("TRELLIS2_CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "status")
    FUNCTION = "cache_conditioning"
    CATEGORY = "BrainDead/Cache/Trellis2"
    DESCRIPTION = """
Cache Trellis2 conditioning to skip image preprocessing.

Place AFTER Trellis2GetConditioning node.
"""

    @classmethod
    def IS_CHANGED(cls, cache_name, seed, force_refresh, conditioning=None, name_prefix=""):
        if force_refresh:
            import time
            return f"force_{time.time()}"
        return f"{name_prefix}_{cache_name}_{seed}"

    def check_lazy_status(self, cache_name, seed, force_refresh, conditioning=None, name_prefix=""):
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, ".pkl")

        if check_cache_exists(cache_path, min_size=100) and not force_refresh:
            try:
                print(f"[BD Trellis2 Conditioning] Loading from cache: {cache_path}")
                cached_data = PickleSerializer.load(cache_path)
                if cached_data is not None:
                    print(f"[BD Trellis2 Conditioning] ✓ Using cached conditioning (skipping upstream)")
                    return (cached_data, f"Cache HIT: {os.path.basename(cache_path)}")
            except Exception as e:
                print(f"[BD Trellis2 Conditioning] Cache load failed: {e}")
                return ["conditioning"]
        if force_refresh:
            print(f"[BD Trellis2 Conditioning] Force refresh - regenerating conditioning")
        else:
            print(f"[BD Trellis2 Conditioning] No cache found - generating new conditioning")
        return ["conditioning"]

    def cache_conditioning(self, conditioning, cache_name, seed, force_refresh, name_prefix=""):
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, ".pkl")

        if check_cache_exists(cache_path, min_size=100) and not force_refresh:
            try:
                cached_data = PickleSerializer.load(cache_path)
                if cached_data is not None:
                    return (cached_data, f"Cache HIT (main): {os.path.basename(cache_path)}")
            except:
                pass

        if conditioning is None:
            return (conditioning, "Input is None - cannot cache")

        print(f"[BD Trellis2 Conditioning] Saving new cache: {cache_path}")
        if save_to_cache(cache_path, conditioning, PickleSerializer):
            status = f"SAVED: {os.path.basename(cache_path)}"
            print(f"[BD Trellis2 Conditioning] ✓ Cache saved successfully")
        else:
            status = "Save failed"
            print(f"[BD Trellis2 Conditioning] ✗ Cache save failed")
        return (conditioning, status)


# =============================================================================
# BD_CacheTrellis2Shape
# =============================================================================

class BD_CacheTrellis2Shape:
    """
    Cache Trellis2 shape result AND mesh together.
    This is the KEY node - caches expensive shape generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shape_result": ("TRELLIS2_SHAPE_RESULT", LAZY_OPTIONS),
                "mesh": ("TRIMESH", LAZY_OPTIONS),
                "cache_name": ("STRING", {"default": "trellis2_shape"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_refresh": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "name_prefix": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("TRELLIS2_SHAPE_RESULT", "TRIMESH", "STRING")
    RETURN_NAMES = ("shape_result", "mesh", "status")
    FUNCTION = "cache_shape"
    CATEGORY = "BrainDead/Cache/Trellis2"
    DESCRIPTION = """
Cache Trellis2 shape result + mesh to skip expensive generation.

Place AFTER Trellis2ImageToShape node.
This is THE most important cache - saves ~30-60s per run!

Caches both shape_result (PKL) and mesh (PLY) together.
"""

    @classmethod
    def IS_CHANGED(cls, cache_name, seed, force_refresh, shape_result=None, mesh=None, name_prefix=""):
        if force_refresh:
            import time
            return f"force_{time.time()}"
        return f"{name_prefix}_{cache_name}_{seed}"

    def check_lazy_status(self, cache_name, seed, force_refresh, shape_result=None, mesh=None, name_prefix=""):
        if not HAS_TRIMESH:
            return ["shape_result", "mesh"]

        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path_pkl = get_cache_path(full_name, cache_hash, ".pkl")
        cache_path_ply = get_cache_path(full_name, cache_hash, "_mesh.ply")

        if (check_cache_exists(cache_path_pkl, min_size=100) and
            check_cache_exists(cache_path_ply, min_size=100) and not force_refresh):
            try:
                print(f"[BD Trellis2 Shape] Loading from cache: {cache_path_pkl}")
                shape_data = PickleSerializer.load(cache_path_pkl)
                mesh_data = trimesh.load(cache_path_ply)
                if shape_data is not None and mesh_data is not None:
                    vert_count = len(mesh_data.vertices) if hasattr(mesh_data, 'vertices') else 'unknown'
                    print(f"[BD Trellis2 Shape] ✓ Using cached shape + mesh ({vert_count} verts) - skipping upstream generation")
                    return (shape_data, mesh_data, f"Cache HIT: shape + mesh")
            except Exception as e:
                print(f"[BD Trellis2 Shape] Cache load failed: {e}")
                return ["shape_result", "mesh"]
        if force_refresh:
            print(f"[BD Trellis2 Shape] Force refresh - regenerating shape + mesh")
        else:
            print(f"[BD Trellis2 Shape] No cache found - generating new shape + mesh")
        return ["shape_result", "mesh"]

    def cache_shape(self, shape_result, mesh, cache_name, seed, force_refresh, name_prefix=""):
        if not HAS_TRIMESH:
            return (shape_result, mesh, "ERROR: trimesh not installed")

        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path_pkl = get_cache_path(full_name, cache_hash, ".pkl")
        cache_path_ply = get_cache_path(full_name, cache_hash, "_mesh.ply")

        if (check_cache_exists(cache_path_pkl, min_size=100) and
            check_cache_exists(cache_path_ply, min_size=100) and not force_refresh):
            try:
                shape_data = PickleSerializer.load(cache_path_pkl)
                mesh_data = trimesh.load(cache_path_ply)
                if shape_data is not None and mesh_data is not None:
                    return (shape_data, mesh_data, f"Cache HIT (main): shape + mesh")
            except:
                pass

        if shape_result is None or mesh is None:
            return (shape_result, mesh, "Input is None - cannot cache")

        try:
            print(f"[BD Trellis2 Shape] Saving new cache...")
            # Save shape_result as pickle
            PickleSerializer.save(cache_path_pkl, shape_result)
            # Save mesh as PLY
            trimesh.exchange.export.export_mesh(mesh, cache_path_ply, file_type='ply')
            vert_count = len(mesh.vertices) if hasattr(mesh, 'vertices') else 'unknown'
            status = f"SAVED: shape + mesh ({vert_count} verts)"
            print(f"[BD Trellis2 Shape] ✓ Cache saved: {cache_path_pkl}")
        except Exception as e:
            status = f"Save failed: {e}"
            print(f"[BD Trellis2 Shape] ✗ Cache save failed: {e}")

        return (shape_result, mesh, status)


# =============================================================================
# BD_CacheTrellis2Texture
# =============================================================================

class BD_CacheTrellis2Texture:
    """
    Cache Trellis2 textured mesh output (trimesh + voxelgrid + pointcloud).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh_out": ("TRIMESH", LAZY_OPTIONS),
                "voxelgrid": ("TRELLIS2_VOXELGRID", LAZY_OPTIONS),
                "pbr_pointcloud": ("TRIMESH", LAZY_OPTIONS),
                "cache_name": ("STRING", {"default": "trellis2_texture"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_refresh": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "name_prefix": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "TRELLIS2_VOXELGRID", "TRIMESH", "STRING")
    RETURN_NAMES = ("trimesh", "voxelgrid", "pbr_pointcloud", "status")
    FUNCTION = "cache_texture"
    CATEGORY = "BrainDead/Cache/Trellis2"
    DESCRIPTION = """
Cache Trellis2 textured mesh outputs together.

Place AFTER Trellis2ShapeToTexturedMesh node.
Caches trimesh, voxelgrid, and pbr_pointcloud as single PKL.

Note: voxelgrid contains GPU tensors - may require GPU
to be available when loading from cache.
"""

    @classmethod
    def IS_CHANGED(cls, cache_name, seed, force_refresh, trimesh_out=None, voxelgrid=None, pbr_pointcloud=None, name_prefix=""):
        if force_refresh:
            import time
            return f"force_{time.time()}"
        return f"{name_prefix}_{cache_name}_{seed}"

    def check_lazy_status(self, cache_name, seed, force_refresh, trimesh_out=None, voxelgrid=None, pbr_pointcloud=None, name_prefix=""):
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, "_texture.pkl")

        if check_cache_exists(cache_path, min_size=100) and not force_refresh:
            try:
                print(f"[BD Trellis2 Texture] Loading from cache: {cache_path}")
                cached_data = PickleSerializer.load(cache_path)
                if cached_data and 'trimesh' in cached_data and 'voxelgrid' in cached_data:
                    print(f"[BD Trellis2 Texture] ✓ Using cached texture data (trimesh + voxelgrid + pointcloud) - skipping upstream")
                    return (cached_data['trimesh'], cached_data['voxelgrid'],
                           cached_data['pointcloud'], f"Cache HIT: texture data")
            except Exception as e:
                print(f"[BD Trellis2 Texture] Cache load failed: {e}")
                return ["trimesh_out", "voxelgrid", "pbr_pointcloud"]
        if force_refresh:
            print(f"[BD Trellis2 Texture] Force refresh - regenerating texture data")
        else:
            print(f"[BD Trellis2 Texture] No cache found - generating new texture data")
        return ["trimesh_out", "voxelgrid", "pbr_pointcloud"]

    def cache_texture(self, trimesh_out, voxelgrid, pbr_pointcloud, cache_name, seed, force_refresh, name_prefix=""):
        full_name = f"{name_prefix}_{cache_name}" if name_prefix else cache_name
        cache_hash = hash_from_seed(seed)
        cache_path = get_cache_path(full_name, cache_hash, "_texture.pkl")

        if check_cache_exists(cache_path, min_size=100) and not force_refresh:
            try:
                cached_data = PickleSerializer.load(cache_path)
                if cached_data and 'trimesh' in cached_data and 'voxelgrid' in cached_data:
                    return (cached_data['trimesh'], cached_data['voxelgrid'],
                           cached_data['pointcloud'], f"Cache HIT (main): texture data")
            except:
                pass

        if trimesh_out is None or voxelgrid is None:
            return (trimesh_out, voxelgrid, pbr_pointcloud, "Input is None - cannot cache")

        try:
            print(f"[BD Trellis2 Texture] Saving new cache...")
            cache_data = {
                'trimesh': trimesh_out,
                'voxelgrid': voxelgrid,
                'pointcloud': pbr_pointcloud
            }
            PickleSerializer.save(cache_path, cache_data)
            # Get voxelgrid info if available
            voxel_info = ""
            if isinstance(voxelgrid, dict):
                if 'coords' in voxelgrid:
                    voxel_info = f" ({len(voxelgrid['coords'])} voxels)"
            status = f"SAVED: texture data{voxel_info}"
            print(f"[BD Trellis2 Texture] ✓ Cache saved: {cache_path}")
        except Exception as e:
            status = f"Save failed: {e}"
            print(f"[BD Trellis2 Texture] ✗ Cache save failed: {e}")

        return (trimesh_out, voxelgrid, pbr_pointcloud, status)


# =============================================================================
# BD_TransferPointcloudColors
# =============================================================================

class BD_TransferPointcloudColors:
    """
    Transfer colors from a pointcloud to mesh vertices using nearest-neighbor lookup.

    This bypasses the TRELLIS2 UV/texture pipeline entirely by directly transferring
    the PBR colors from the pointcloud to vertex colors on the mesh.

    Perfect for: Preparing meshes for vertex-color workflows and decimation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "pointcloud": ("TRIMESH",),  # pbr_pointcloud from TRELLIS2
            },
            "optional": {
                "coordinate_fix": (["auto", "none", "mesh_to_zup", "pointcloud_to_yup"], {
                    "default": "auto",
                    "tooltip": "Fix coordinate mismatch between mesh and pointcloud"}),
                "max_distance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001,
                                           "tooltip": "Max distance for color transfer (0 = unlimited)"}),
                "default_color": ("STRING", {"default": "0.5,0.5,0.5,1.0",
                                             "tooltip": "RGBA color for vertices with no nearby points"}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "status")
    FUNCTION = "transfer_colors"
    CATEGORY = "BrainDead/Mesh"
    DESCRIPTION = """
Transfer colors from pointcloud to mesh vertices.

Use this to bypass TRELLIS2's UV/texture pipeline:
1. Connect 'trimesh' output to mesh input
2. Connect 'pbr_pointcloud' output to pointcloud input
3. Output mesh has vertex colors ready for export/decimation

The node uses KD-tree nearest-neighbor search for fast color transfer,
even on 14M+ polygon meshes (typically <1 minute).

Inputs:
- mesh: Geometry mesh from TRELLIS2 (or any TRIMESH)
- pointcloud: pbr_pointcloud from TRELLIS2 (has colors)
- coordinate_fix: Handle TRELLIS2 coordinate mismatch
  - auto: Auto-detect and fix (recommended)
  - none: No conversion
  - mesh_to_zup: Convert mesh Y-up to Z-up
  - pointcloud_to_yup: Convert pointcloud Z-up back to Y-up
- max_distance: Skip vertices farther than this from any point (0=unlimited)
- default_color: RGBA for vertices with no nearby points

Note: TRELLIS2 outputs mesh in Y-up but pointcloud in Z-up coordinates.
The 'auto' setting detects and fixes this automatically.

Output mesh can be:
- Exported directly with vertex colors (GLB/PLY)
- Passed to Blender decimation with color preservation
- Cached with BD_CacheMesh
"""

    def transfer_colors(self, mesh, pointcloud, coordinate_fix="auto", max_distance=0.0, default_color="0.5,0.5,0.5,1.0"):
        import numpy as np
        import time

        if not HAS_TRIMESH:
            return (mesh, "ERROR: trimesh not installed")

        if mesh is None:
            return (None, "ERROR: mesh is None")

        if pointcloud is None:
            return (mesh, "ERROR: pointcloud is None - no colors to transfer")

        # Parse default color
        try:
            default_rgba = [float(x.strip()) for x in default_color.split(",")]
            if len(default_rgba) == 3:
                default_rgba.append(1.0)
            default_rgba = np.array(default_rgba[:4], dtype=np.float64)
        except:
            default_rgba = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float64)

        start_time = time.time()

        # Get mesh vertices
        mesh_verts = np.array(mesh.vertices, dtype=np.float64)
        num_verts = len(mesh_verts)
        print(f"[BD Transfer Colors] Mesh: {num_verts} vertices")

        # Get pointcloud data
        if hasattr(pointcloud, 'vertices'):
            pc_verts = np.array(pointcloud.vertices, dtype=np.float64)
        else:
            return (mesh, "ERROR: pointcloud has no vertices")

        # Debug: Show bounding boxes BEFORE any conversion
        mesh_min = mesh_verts.min(axis=0)
        mesh_max = mesh_verts.max(axis=0)
        pc_min = pc_verts.min(axis=0)
        pc_max = pc_verts.max(axis=0)
        print(f"[BD Transfer Colors] Mesh bounds: min={mesh_min}, max={mesh_max}")
        print(f"[BD Transfer Colors] Pointcloud bounds: min={pc_min}, max={pc_max}")

        # Check for coordinate mismatch and apply fix
        # TRELLIS2 applies Y-up to Z-up conversion to pointcloud but NOT to mesh
        # Pointcloud: Y,Z swapped and Z negated
        # To reverse: swap Y,Z back and negate Z

        def apply_yup_to_zup(verts):
            """Convert from Y-up to Z-up (swap Y,Z and negate new Z)"""
            result = verts.copy()
            result[:, 1], result[:, 2] = verts[:, 2].copy(), -verts[:, 1].copy()
            return result

        def apply_zup_to_yup(verts):
            """Convert from Z-up back to Y-up (reverse of yup_to_zup)"""
            result = verts.copy()
            result[:, 1], result[:, 2] = -verts[:, 2].copy(), verts[:, 1].copy()
            return result

        if coordinate_fix == "auto":
            # Auto-detect: check if bounds overlap
            # If mesh Y range matches pointcloud Z range (roughly), they're misaligned
            mesh_y_range = (mesh_min[1], mesh_max[1])
            mesh_z_range = (mesh_min[2], mesh_max[2])
            pc_y_range = (pc_min[1], pc_max[1])
            pc_z_range = (pc_min[2], pc_max[2])

            # Check overlap in current space
            def ranges_overlap(r1, r2, tolerance=0.5):
                return not (r1[1] < r2[0] - tolerance or r2[1] < r1[0] - tolerance)

            current_overlap = (
                ranges_overlap((mesh_min[0], mesh_max[0]), (pc_min[0], pc_max[0])) and
                ranges_overlap((mesh_min[1], mesh_max[1]), (pc_min[1], pc_max[1])) and
                ranges_overlap((mesh_min[2], mesh_max[2]), (pc_min[2], pc_max[2]))
            )

            if current_overlap:
                print(f"[BD Transfer Colors] Auto-detect: Bounds overlap, no conversion needed")
                coordinate_fix = "none"
            else:
                # Try converting pointcloud back to Y-up
                pc_verts_yup = apply_zup_to_yup(pc_verts)
                pc_yup_min = pc_verts_yup.min(axis=0)
                pc_yup_max = pc_verts_yup.max(axis=0)

                converted_overlap = (
                    ranges_overlap((mesh_min[0], mesh_max[0]), (pc_yup_min[0], pc_yup_max[0])) and
                    ranges_overlap((mesh_min[1], mesh_max[1]), (pc_yup_min[1], pc_yup_max[1])) and
                    ranges_overlap((mesh_min[2], mesh_max[2]), (pc_yup_min[2], pc_yup_max[2]))
                )

                if converted_overlap:
                    print(f"[BD Transfer Colors] Auto-detect: Converting pointcloud to Y-up (TRELLIS2 fix)")
                    coordinate_fix = "pointcloud_to_yup"
                else:
                    # Try the other way - convert mesh to Z-up
                    mesh_verts_zup = apply_yup_to_zup(mesh_verts)
                    mesh_zup_min = mesh_verts_zup.min(axis=0)
                    mesh_zup_max = mesh_verts_zup.max(axis=0)

                    mesh_converted_overlap = (
                        ranges_overlap((mesh_zup_min[0], mesh_zup_max[0]), (pc_min[0], pc_max[0])) and
                        ranges_overlap((mesh_zup_min[1], mesh_zup_max[1]), (pc_min[1], pc_max[1])) and
                        ranges_overlap((mesh_zup_min[2], mesh_zup_max[2]), (pc_min[2], pc_max[2]))
                    )

                    if mesh_converted_overlap:
                        print(f"[BD Transfer Colors] Auto-detect: Converting mesh to Z-up")
                        coordinate_fix = "mesh_to_zup"
                    else:
                        print(f"[BD Transfer Colors] Auto-detect: Could not find matching coordinate space, trying pointcloud_to_yup")
                        coordinate_fix = "pointcloud_to_yup"

        # Apply the chosen coordinate fix
        if coordinate_fix == "pointcloud_to_yup":
            pc_verts = apply_zup_to_yup(pc_verts)
            pc_min_new = pc_verts.min(axis=0)
            pc_max_new = pc_verts.max(axis=0)
            print(f"[BD Transfer Colors] Converted pointcloud to Y-up: min={pc_min_new}, max={pc_max_new}")
        elif coordinate_fix == "mesh_to_zup":
            mesh_verts = apply_yup_to_zup(mesh_verts)
            mesh_min_new = mesh_verts.min(axis=0)
            mesh_max_new = mesh_verts.max(axis=0)
            print(f"[BD Transfer Colors] Converted mesh to Z-up: min={mesh_min_new}, max={mesh_max_new}")

        # Get pointcloud colors
        pc_colors = None
        if hasattr(pointcloud, 'colors') and pointcloud.colors is not None:
            pc_colors = np.array(pointcloud.colors)
            # Normalize to 0-1 if uint8
            if pc_colors.dtype == np.uint8:
                pc_colors = pc_colors.astype(np.float64) / 255.0

        if pc_colors is None or len(pc_colors) == 0:
            return (mesh, "ERROR: pointcloud has no colors")

        print(f"[BD Transfer Colors] Pointcloud: {len(pc_verts)} points with colors")

        # Build KD-tree for fast nearest-neighbor lookup
        print(f"[BD Transfer Colors] Building KD-tree...")
        tree_start = time.time()

        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pc_verts)
            print(f"[BD Transfer Colors] KD-tree built in {time.time() - tree_start:.2f}s (scipy)")
        except ImportError:
            # Fallback to numpy-based approach (slower but works)
            print(f"[BD Transfer Colors] scipy not available, using numpy fallback (slower)")
            tree = None

        # Query nearest neighbors for all mesh vertices
        print(f"[BD Transfer Colors] Finding nearest neighbors for {num_verts} vertices...")
        query_start = time.time()

        if tree is not None:
            # Fast scipy query
            if max_distance > 0:
                distances, indices = tree.query(mesh_verts, k=1, distance_upper_bound=max_distance)
            else:
                distances, indices = tree.query(mesh_verts, k=1)
        else:
            # Numpy fallback - process in batches to avoid memory issues
            batch_size = 10000
            indices = np.zeros(num_verts, dtype=np.int64)
            distances = np.zeros(num_verts, dtype=np.float64)

            for i in range(0, num_verts, batch_size):
                end_idx = min(i + batch_size, num_verts)
                batch_verts = mesh_verts[i:end_idx]

                # Compute distances to all pointcloud vertices
                dists = np.linalg.norm(pc_verts[np.newaxis, :, :] - batch_verts[:, np.newaxis, :], axis=2)
                batch_indices = np.argmin(dists, axis=1)
                batch_distances = dists[np.arange(len(batch_indices)), batch_indices]

                indices[i:end_idx] = batch_indices
                distances[i:end_idx] = batch_distances

                if (i // batch_size) % 10 == 0:
                    print(f"[BD Transfer Colors] Progress: {end_idx}/{num_verts}")

        print(f"[BD Transfer Colors] Nearest neighbor query: {time.time() - query_start:.2f}s")

        # Transfer colors
        print(f"[BD Transfer Colors] Transferring colors...")

        # Initialize vertex colors array
        vertex_colors = np.zeros((num_verts, 4), dtype=np.float64)

        # Handle valid indices (scipy returns len(pc_verts) for out-of-range)
        valid_mask = indices < len(pc_verts)
        if max_distance > 0:
            valid_mask &= distances <= max_distance

        # Copy colors for valid vertices
        valid_indices = indices[valid_mask]
        vertex_colors[valid_mask] = pc_colors[valid_indices]

        # Set default color for invalid vertices
        invalid_count = np.sum(~valid_mask)
        if invalid_count > 0:
            vertex_colors[~valid_mask] = default_rgba
            print(f"[BD Transfer Colors] {invalid_count} vertices used default color (no nearby points)")

        # Ensure RGBA (add alpha if needed)
        if vertex_colors.shape[1] == 3:
            alpha = np.ones((num_verts, 1), dtype=np.float64)
            vertex_colors = np.concatenate([vertex_colors, alpha], axis=1)

        # Convert to uint8 for trimesh (0-255)
        vertex_colors_uint8 = (vertex_colors * 255).clip(0, 255).astype(np.uint8)

        # Create new mesh with vertex colors
        # We need to copy the mesh and add vertex colors
        try:
            new_mesh = trimesh.Trimesh(
                vertices=mesh.vertices.copy(),
                faces=mesh.faces.copy() if hasattr(mesh, 'faces') and mesh.faces is not None else None,
                vertex_colors=vertex_colors_uint8,
                process=False  # Don't modify the mesh
            )

            # Copy other attributes if present
            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                new_mesh.vertex_normals = mesh.vertex_normals.copy()

            total_time = time.time() - start_time

            # Calculate color statistics
            unique_colors = len(np.unique(vertex_colors_uint8.view(np.uint32)))
            avg_dist = np.mean(distances[valid_mask]) if np.any(valid_mask) else 0

            status = f"Transferred colors to {num_verts} vertices ({unique_colors} unique colors, avg_dist={avg_dist:.4f}) in {total_time:.1f}s"
            print(f"[BD Transfer Colors] ✓ {status}")

            return (new_mesh, status)

        except Exception as e:
            return (mesh, f"ERROR creating colored mesh: {e}")


# =============================================================================
# BD_ExportMeshWithColors
# =============================================================================

class BD_ExportMeshWithColors:
    """
    Export a mesh with vertex colors to file (GLB, PLY, OBJ).

    Designed to work with BD_TransferPointcloudColors output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "filename": ("STRING", {"default": "mesh_colored"}),
                "format": (["glb", "ply", "obj"], {"default": "glb"}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": "", "tooltip": "Leave empty for default output folder"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_path", "status")
    FUNCTION = "export_mesh"
    CATEGORY = "BrainDead/Mesh"
    OUTPUT_NODE = True
    DESCRIPTION = """
Export mesh with vertex colors to file.

Formats:
- GLB: Best for game engines, preserves vertex colors
- PLY: Good for Blender import, preserves vertex colors
- OBJ: Basic format, vertex colors may not be preserved

Use after BD_TransferPointcloudColors to export colored mesh
for decimation in Blender.
"""

    def export_mesh(self, mesh, filename, format="glb", output_dir=""):
        import os

        if not HAS_TRIMESH:
            return ("", "ERROR: trimesh not installed")

        if mesh is None:
            return ("", "ERROR: mesh is None")

        # Determine output directory
        if not output_dir:
            import folder_paths
            output_dir = folder_paths.get_output_directory()

        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Build filename
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"

        file_path = os.path.join(output_dir, filename)

        try:
            # Check if mesh has vertex colors
            has_colors = hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors')
            color_info = ""
            if has_colors and mesh.visual.vertex_colors is not None:
                color_info = f" with {len(mesh.visual.vertex_colors)} vertex colors"
            elif hasattr(mesh, 'vertex_colors') and mesh.vertex_colors is not None:
                color_info = f" with {len(mesh.vertex_colors)} vertex colors"

            print(f"[BD Export Mesh] Exporting to {file_path}{color_info}...")

            # Export based on format
            if format == "glb":
                # GLB export - best for vertex colors
                mesh.export(file_path, file_type='glb')
            elif format == "ply":
                # PLY export - preserves vertex colors well
                mesh.export(file_path, file_type='ply')
            elif format == "obj":
                # OBJ export - basic
                mesh.export(file_path, file_type='obj')

            # Get file size
            file_size = os.path.getsize(file_path)
            size_str = f"{file_size / 1024 / 1024:.1f}MB" if file_size > 1024*1024 else f"{file_size / 1024:.1f}KB"

            vert_count = len(mesh.vertices) if hasattr(mesh, 'vertices') else 0
            face_count = len(mesh.faces) if hasattr(mesh, 'faces') and mesh.faces is not None else 0

            status = f"Exported {format.upper()}: {vert_count} verts, {face_count} faces ({size_str})"
            print(f"[BD Export Mesh] ✓ {status}")

            return (file_path, status)

        except Exception as e:
            return ("", f"ERROR: {e}")


# =============================================================================
# Node Mappings
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "BD_CacheImage": BD_CacheImage,
    "BD_CacheMask": BD_CacheMask,
    "BD_CacheLatent": BD_CacheLatent,
    "BD_CacheAudio": BD_CacheAudio,
    "BD_CacheString": BD_CacheString,
    "BD_CacheAny": BD_CacheAny,
    "BD_CacheMesh": BD_CacheMesh,
    "BD_ClearCache": BD_ClearCache,
    "BD_SaveFile": BD_SaveFile,
    "BD_LoadImage": BD_LoadImage,
    "BD_LoadMesh": BD_LoadMesh,
    "BD_LoadAudio": BD_LoadAudio,
    "BD_LoadText": BD_LoadText,
    # Trellis2 Cache Nodes
    "BD_CacheTrellis2Conditioning": BD_CacheTrellis2Conditioning,
    "BD_CacheTrellis2Shape": BD_CacheTrellis2Shape,
    "BD_CacheTrellis2Texture": BD_CacheTrellis2Texture,
    # Mesh Processing Nodes
    "BD_TransferPointcloudColors": BD_TransferPointcloudColors,
    "BD_ExportMeshWithColors": BD_ExportMeshWithColors,
    # Workflow Version Nodes
    "BD_WorkflowVersionCache": BD_WorkflowVersionCache,
    "BD_WorkflowVersionList": BD_WorkflowVersionList,
    "BD_WorkflowVersionRestore": BD_WorkflowVersionRestore,
    "BD_WorkflowVersionClear": BD_WorkflowVersionClear,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BD_CacheImage": "BD Cache Image",
    "BD_CacheMask": "BD Cache Mask",
    "BD_CacheLatent": "BD Cache Latent",
    "BD_CacheAudio": "BD Cache Audio",
    "BD_CacheString": "BD Cache String",
    "BD_CacheAny": "BD Cache Any",
    "BD_CacheMesh": "BD Cache Mesh",
    "BD_ClearCache": "BD Clear Cache",
    "BD_SaveFile": "BD Save File",
    "BD_LoadImage": "BD Load Image",
    "BD_LoadMesh": "BD Load Mesh",
    "BD_LoadAudio": "BD Load Audio",
    "BD_LoadText": "BD Load Text",
    # Trellis2 Cache Nodes
    "BD_CacheTrellis2Conditioning": "BD Cache Trellis2 Conditioning",
    "BD_CacheTrellis2Shape": "BD Cache Trellis2 Shape",
    "BD_CacheTrellis2Texture": "BD Cache Trellis2 Texture",
    # Mesh Processing Nodes
    "BD_TransferPointcloudColors": "BD Transfer Pointcloud Colors",
    "BD_ExportMeshWithColors": "BD Export Mesh With Colors",
    # Workflow Version Nodes
    "BD_WorkflowVersionCache": "BD Workflow Version Cache",
    "BD_WorkflowVersionList": "BD Workflow Version List",
    "BD_WorkflowVersionRestore": "BD Workflow Version Restore",
    "BD_WorkflowVersionClear": "BD Workflow Version Clear",
}
