"""
BrainDead Shared Utilities
Created by BizaNator for BrainDeadGuild.com
A Biloxi Studios Inc. Production

Core utilities providing:
- Belt-and-suspenders caching pattern
- Type-specific serializers (PNG, WAV, PKL, etc.)
- Input hashing for automatic cache invalidation
- Lazy evaluation helpers
"""

import os
import hashlib
import pickle
import json
import folder_paths
from pathlib import Path
import numpy as np

# =============================================================================
# Directory Configuration
# =============================================================================

# Cache directory (for cache nodes - can be cleared with BD Clear Cache)
CACHE_DIR = os.path.join(folder_paths.output_directory, "BrainDead_Cache")

# Output directory (for BD Save File - permanent saves, not affected by Clear Cache)
OUTPUT_DIR = folder_paths.output_directory


def ensure_cache_dir():
    """Ensure cache directory exists."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR


def get_cache_path(cache_name, data_hash, extension):
    """Generate cache file path from name and hash.

    Supports subdirectories in cache_name (e.g., "project/step1/image").
    Creates subdirectories automatically if they don't exist.
    """
    ensure_cache_dir()

    # Handle subdirectories in cache_name
    if '/' in cache_name or '\\' in cache_name:
        # Normalize path separators
        cache_name = cache_name.replace('\\', '/')
        # Split into directory and filename parts
        parts = cache_name.rsplit('/', 1)
        if len(parts) == 2:
            subdir, name = parts
            full_dir = os.path.join(CACHE_DIR, subdir)
            os.makedirs(full_dir, exist_ok=True)
            filename = f"{name}_{data_hash}{extension}"
            return os.path.join(full_dir, filename)

    # No subdirectory, use base cache dir
    filename = f"{cache_name}_{data_hash}{extension}"
    return os.path.join(CACHE_DIR, filename)


# =============================================================================
# Hashing Functions
# =============================================================================

def hash_from_seed(seed):
    """Generate deterministic hash from seed value."""
    return hashlib.md5(str(seed).encode()).hexdigest()


def hash_from_params(**kwargs):
    """Generate hash from arbitrary parameters."""
    params_dict = {k: v for k, v in sorted(kwargs.items()) if v is not None}
    params_str = json.dumps(params_dict, sort_keys=True, default=str)
    return hashlib.md5(params_str.encode()).hexdigest()


def hash_tensor(tensor):
    """Hash a PyTorch tensor or numpy array."""
    if tensor is None:
        return "none"
    try:
        if hasattr(tensor, 'cpu'):
            # PyTorch tensor
            data = tensor.cpu().numpy().tobytes()
        elif hasattr(tensor, 'tobytes'):
            # Numpy array
            data = tensor.tobytes()
        else:
            # Fallback
            data = str(tensor).encode()
        return hashlib.md5(data).hexdigest()[:16]  # Truncate for speed
    except Exception as e:
        print(f"[BrainDead] Warning: Could not hash tensor: {e}")
        return "unhashable"


def hash_image(image_tensor):
    """Hash an IMAGE tensor (B, H, W, C)."""
    return hash_tensor(image_tensor)


def hash_latent(latent_dict):
    """Hash a LATENT dict containing 'samples' tensor."""
    if latent_dict is None:
        return "none"
    if isinstance(latent_dict, dict) and 'samples' in latent_dict:
        return hash_tensor(latent_dict['samples'])
    return hash_tensor(latent_dict)


def hash_mask(mask_tensor):
    """Hash a MASK tensor."""
    return hash_tensor(mask_tensor)


def hash_string(text):
    """Hash a string."""
    if text is None:
        return "none"
    return hashlib.md5(str(text).encode()).hexdigest()[:16]


def hash_audio(audio_dict):
    """Hash an AUDIO dict containing 'waveform' and 'sample_rate'."""
    if audio_dict is None:
        return "none"
    if isinstance(audio_dict, dict):
        parts = []
        if 'waveform' in audio_dict:
            parts.append(hash_tensor(audio_dict['waveform']))
        if 'sample_rate' in audio_dict:
            parts.append(str(audio_dict['sample_rate']))
        return hashlib.md5("_".join(parts).encode()).hexdigest()[:16]
    return hash_tensor(audio_dict)


# =============================================================================
# Lazy Evaluation Helpers
# =============================================================================

def compare_revision(num):
    """Check if ComfyUI revision supports lazy evaluation."""
    try:
        import git
        repo = git.Repo(os.path.dirname(folder_paths.__file__))
        comfy_ui_revision = len(list(repo.iter_commits('HEAD')))
        return int(comfy_ui_revision) >= num
    except:
        return True  # Assume modern version


# Lazy options for INPUT_TYPES
LAZY_OPTIONS = {"lazy": True} if compare_revision(2543) else {}


# =============================================================================
# Serializers - Type-specific save/load functions
# =============================================================================

class ImageSerializer:
    """Save/load IMAGE tensors as PNG files."""
    extension = ".png"

    @staticmethod
    def save(filepath, image_tensor):
        """Save IMAGE tensor (B, H, W, C) as PNG."""
        from PIL import Image
        import torch

        # Take first image if batched
        if len(image_tensor.shape) == 4:
            img = image_tensor[0]
        else:
            img = image_tensor

        # Convert to numpy and scale to 0-255
        if hasattr(img, 'cpu'):
            img_np = img.cpu().numpy()
        else:
            img_np = img

        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

        # Save as PNG
        pil_img = Image.fromarray(img_np)
        pil_img.save(filepath, 'PNG')
        return True

    @staticmethod
    def load(filepath):
        """Load PNG as IMAGE tensor."""
        from PIL import Image
        import torch

        pil_img = Image.open(filepath).convert('RGB')
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        # Return as (1, H, W, C) tensor
        return torch.from_numpy(img_np).unsqueeze(0)


class MaskSerializer:
    """Save/load MASK tensors as PNG files."""
    extension = ".png"

    @staticmethod
    def save(filepath, mask_tensor):
        """Save MASK tensor as grayscale PNG."""
        from PIL import Image
        import torch

        # Handle batched masks
        if len(mask_tensor.shape) == 3:
            mask = mask_tensor[0]
        else:
            mask = mask_tensor

        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask

        mask_np = (mask_np * 255).clip(0, 255).astype(np.uint8)

        pil_img = Image.fromarray(mask_np, mode='L')
        pil_img.save(filepath, 'PNG')
        return True

    @staticmethod
    def load(filepath):
        """Load PNG as MASK tensor."""
        from PIL import Image
        import torch

        pil_img = Image.open(filepath).convert('L')
        mask_np = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(mask_np).unsqueeze(0)


class LatentSerializer:
    """Save/load LATENT dicts as safetensors or pickle."""
    extension = ".latent"

    @staticmethod
    def save(filepath, latent_dict):
        """Save LATENT dict."""
        import torch
        try:
            from safetensors.torch import save_file
            # Safetensors only supports tensors, extract samples
            tensors = {"samples": latent_dict["samples"]}
            save_file(tensors, filepath)
        except ImportError:
            # Fallback to pickle
            with open(filepath, 'wb') as f:
                pickle.dump(latent_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        return True

    @staticmethod
    def load(filepath):
        """Load LATENT dict."""
        import torch
        try:
            from safetensors.torch import load_file
            tensors = load_file(filepath)
            return {"samples": tensors["samples"]}
        except:
            # Fallback to pickle
            with open(filepath, 'rb') as f:
                return pickle.load(f)


class AudioSerializer:
    """Save/load AUDIO dicts as WAV files."""
    extension = ".wav"

    @staticmethod
    def save(filepath, audio_dict):
        """Save AUDIO dict as WAV."""
        import torch
        try:
            import torchaudio
            waveform = audio_dict['waveform']
            sample_rate = audio_dict['sample_rate']

            # Ensure proper shape (channels, samples)
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)

            torchaudio.save(filepath, waveform.cpu(), sample_rate)
            return True
        except ImportError:
            # Fallback to pickle
            pkl_path = filepath.replace('.wav', '.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(audio_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True

    @staticmethod
    def load(filepath):
        """Load WAV as AUDIO dict."""
        import torch
        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(filepath)
            return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        except ImportError:
            # Fallback to pickle
            pkl_path = filepath.replace('.wav', '.pkl')
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    return pickle.load(f)
            raise


class StringSerializer:
    """Save/load STRING as text files."""
    extension = ".txt"

    @staticmethod
    def save(filepath, text):
        """Save string as text file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(str(text))
        return True

    @staticmethod
    def load(filepath):
        """Load text file as string."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()


class PickleSerializer:
    """Generic pickle serializer for any Python object."""
    extension = ".pkl"

    @staticmethod
    def save(filepath, data):
        """Save any object as pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return True

    @staticmethod
    def load(filepath):
        """Load pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# Serializer registry
SERIALIZERS = {
    "IMAGE": ImageSerializer,
    "MASK": MaskSerializer,
    "LATENT": LatentSerializer,
    "AUDIO": AudioSerializer,
    "STRING": StringSerializer,
    "GENERIC": PickleSerializer,
}


# =============================================================================
# Belt-and-Suspenders Cache Check
# =============================================================================

def check_cache_exists(cache_path, min_size=10):
    """Check if cache file exists and has valid size."""
    if not os.path.exists(cache_path):
        return False
    try:
        return os.path.getsize(cache_path) >= min_size
    except:
        return False


def load_cached_data(cache_path, serializer_class):
    """Load data from cache file using appropriate serializer."""
    try:
        return serializer_class.load(cache_path)
    except Exception as e:
        print(f"[BrainDead] Error loading cache: {e}")
        return None


def save_to_cache(cache_path, data, serializer_class):
    """Save data to cache file using appropriate serializer."""
    try:
        serializer_class.save(cache_path, data)
        return True
    except Exception as e:
        print(f"[BrainDead] Error saving cache: {e}")
        return False
