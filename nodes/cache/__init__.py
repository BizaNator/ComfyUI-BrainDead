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
}
