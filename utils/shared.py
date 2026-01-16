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


# =============================================================================
# Workflow Version Cache Utilities
# =============================================================================

WORKFLOW_VERSIONS_DIR = os.path.join(CACHE_DIR, "workflow_versions")


def ensure_workflow_versions_dir():
    """Ensure workflow versions directory exists."""
    os.makedirs(WORKFLOW_VERSIONS_DIR, exist_ok=True)
    return WORKFLOW_VERSIONS_DIR


def hash_workflow_structure(workflow_data):
    """
    Generate a hash of the workflow structure.

    Focuses on node types and connections, excludes volatile data like seeds.
    This allows detecting structural changes while ignoring execution-specific values.
    """
    if not workflow_data:
        return "empty"

    try:
        # Extract structural elements only
        structure = {}

        if isinstance(workflow_data, dict):
            nodes = workflow_data.get('nodes', [])
            links = workflow_data.get('links', [])

            # Hash node types and their basic configuration
            node_info = []
            for node in nodes:
                if isinstance(node, dict):
                    node_info.append({
                        'type': node.get('type', ''),
                        'id': node.get('id', ''),
                        # Include widget names but not values (to detect structural changes)
                        'widgets': [w.get('name', '') if isinstance(w, dict) else str(i)
                                   for i, w in enumerate(node.get('widgets_values', []))] if 'widgets_values' in node else []
                    })

            structure['nodes'] = sorted(node_info, key=lambda x: str(x.get('id', '')))
            structure['links'] = sorted([str(l) for l in links]) if links else []
            structure['node_count'] = len(nodes)

        structure_str = json.dumps(structure, sort_keys=True, default=str)
        return hashlib.md5(structure_str.encode()).hexdigest()
    except Exception as e:
        print(f"[BrainDead] Warning: Could not hash workflow structure: {e}")
        return "unhashable"


def hash_workflow_full(workflow_data):
    """
    Generate a hash of the complete workflow including all values.

    Used for detecting any change whatsoever in the workflow.
    """
    if not workflow_data:
        return "empty"

    try:
        workflow_str = json.dumps(workflow_data, sort_keys=True, default=str)
        return hashlib.md5(workflow_str.encode()).hexdigest()
    except Exception as e:
        print(f"[BrainDead] Warning: Could not hash workflow: {e}")
        return "unhashable"


def get_workflow_version_path(workflow_id, version_num):
    """Get the path for a specific workflow version file."""
    ensure_workflow_versions_dir()
    # Sanitize workflow_id for filesystem
    safe_id = "".join(c if c.isalnum() or c in '-_' else '_' for c in str(workflow_id))
    return os.path.join(WORKFLOW_VERSIONS_DIR, f"{safe_id}_v{version_num:04d}.json")


def list_workflow_versions(workflow_id):
    """
    List all saved versions for a workflow.

    Returns list of dicts with version metadata, sorted by version number descending.
    """
    ensure_workflow_versions_dir()
    safe_id = "".join(c if c.isalnum() or c in '-_' else '_' for c in str(workflow_id))

    versions = []
    pattern = f"{safe_id}_v"

    try:
        for filename in os.listdir(WORKFLOW_VERSIONS_DIR):
            if filename.startswith(pattern) and filename.endswith('.json'):
                filepath = os.path.join(WORKFLOW_VERSIONS_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    versions.append({
                        'version': data.get('version', 0),
                        'timestamp': data.get('timestamp', ''),
                        'workflow_hash': data.get('workflow_hash', '')[:8],
                        'structure_hash': data.get('structure_hash', '')[:8],
                        'node_count': data.get('node_count', 0),
                        'description': data.get('description', ''),
                        'filepath': filepath,
                        'filename': filename,
                    })
                except Exception as e:
                    print(f"[BrainDead] Warning: Could not read version file {filename}: {e}")
    except Exception as e:
        print(f"[BrainDead] Error listing workflow versions: {e}")

    # Sort by version number descending (newest first)
    versions.sort(key=lambda x: x['version'], reverse=True)
    return versions


def save_workflow_version(workflow_id, workflow_data, description="auto-saved", max_versions=50):
    """
    Save a new workflow version.

    Returns tuple of (success, message, version_number).
    Automatically prunes old versions if max_versions is exceeded.
    """
    ensure_workflow_versions_dir()

    if not workflow_data:
        return False, "No workflow data to save", 0

    # Get existing versions to determine next version number
    existing = list_workflow_versions(workflow_id)
    next_version = (existing[0]['version'] + 1) if existing else 1

    # Build version metadata
    from datetime import datetime
    version_data = {
        'version': next_version,
        'timestamp': datetime.now().isoformat(),
        'workflow_hash': hash_workflow_full(workflow_data),
        'structure_hash': hash_workflow_structure(workflow_data),
        'node_count': len(workflow_data.get('nodes', [])) if isinstance(workflow_data, dict) else 0,
        'description': description,
        'workflow': workflow_data,
    }

    # Save the version
    version_path = get_workflow_version_path(workflow_id, next_version)
    try:
        with open(version_path, 'w', encoding='utf-8') as f:
            json.dump(version_data, f, indent=2, default=str)
    except Exception as e:
        return False, f"Failed to save version: {e}", 0

    # Prune old versions if needed
    if max_versions > 0 and len(existing) >= max_versions:
        versions_to_delete = existing[max_versions - 1:]  # Keep max_versions - 1 (new one makes max_versions)
        for v in versions_to_delete:
            try:
                os.remove(v['filepath'])
            except Exception as e:
                print(f"[BrainDead] Warning: Could not delete old version {v['filename']}: {e}")

    return True, f"Saved version {next_version}", next_version


def load_workflow_version(workflow_id, version_num):
    """
    Load a specific workflow version.

    Returns tuple of (workflow_data, metadata) or (None, error_message).
    """
    version_path = get_workflow_version_path(workflow_id, version_num)

    if not os.path.exists(version_path):
        return None, f"Version {version_num} not found"

    try:
        with open(version_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        workflow = data.get('workflow')
        metadata = {k: v for k, v in data.items() if k != 'workflow'}
        return workflow, metadata
    except Exception as e:
        return None, f"Failed to load version: {e}"


def compare_workflow_versions(workflow_id, version_a, version_b):
    """
    Compare two workflow versions and return a simple diff.

    Returns a dict with:
    - added_nodes: list of node types added in version_b
    - removed_nodes: list of node types removed in version_b
    - changed_nodes: list of node types that changed
    - summary: human-readable summary string
    """
    workflow_a, meta_a = load_workflow_version(workflow_id, version_a)
    workflow_b, meta_b = load_workflow_version(workflow_id, version_b)

    if workflow_a is None:
        return {'error': f"Could not load version {version_a}: {meta_a}"}
    if workflow_b is None:
        return {'error': f"Could not load version {version_b}: {meta_b}"}

    # Extract nodes by ID
    nodes_a = {}
    nodes_b = {}

    if isinstance(workflow_a, dict):
        for node in workflow_a.get('nodes', []):
            if isinstance(node, dict):
                nodes_a[node.get('id')] = node

    if isinstance(workflow_b, dict):
        for node in workflow_b.get('nodes', []):
            if isinstance(node, dict):
                nodes_b[node.get('id')] = node

    ids_a = set(nodes_a.keys())
    ids_b = set(nodes_b.keys())

    added_ids = ids_b - ids_a
    removed_ids = ids_a - ids_b
    common_ids = ids_a & ids_b

    # Check for changed nodes (same ID but different type)
    changed_ids = set()
    for node_id in common_ids:
        if nodes_a[node_id].get('type') != nodes_b[node_id].get('type'):
            changed_ids.add(node_id)

    added_nodes = [nodes_b[nid].get('type', 'Unknown') for nid in added_ids]
    removed_nodes = [nodes_a[nid].get('type', 'Unknown') for nid in removed_ids]
    changed_nodes = [(nodes_a[nid].get('type', 'Unknown'), nodes_b[nid].get('type', 'Unknown'))
                     for nid in changed_ids]

    # Build summary
    summary_parts = []
    if added_nodes:
        summary_parts.append(f"Added {len(added_nodes)} node(s): {', '.join(added_nodes[:5])}" +
                           ("..." if len(added_nodes) > 5 else ""))
    if removed_nodes:
        summary_parts.append(f"Removed {len(removed_nodes)} node(s): {', '.join(removed_nodes[:5])}" +
                           ("..." if len(removed_nodes) > 5 else ""))
    if changed_nodes:
        changes = [f"{old}->{new}" for old, new in changed_nodes[:3]]
        summary_parts.append(f"Changed {len(changed_nodes)} node(s): {', '.join(changes)}" +
                           ("..." if len(changed_nodes) > 3 else ""))

    if not summary_parts:
        summary_parts.append("No structural changes detected")

    return {
        'added_nodes': added_nodes,
        'removed_nodes': removed_nodes,
        'changed_nodes': changed_nodes,
        'summary': "\n".join(summary_parts),
        'version_a': version_a,
        'version_b': version_b,
    }


def clear_workflow_versions(workflow_id, keep_latest=0):
    """
    Clear workflow versions, optionally keeping the N most recent.

    Returns tuple of (deleted_count, message).
    """
    versions = list_workflow_versions(workflow_id)

    if not versions:
        return 0, "No versions found"

    to_delete = versions[keep_latest:] if keep_latest > 0 else versions
    deleted = 0

    for v in to_delete:
        try:
            os.remove(v['filepath'])
            deleted += 1
        except Exception as e:
            print(f"[BrainDead] Warning: Could not delete {v['filename']}: {e}")

    return deleted, f"Deleted {deleted} version(s)"
