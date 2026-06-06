"""
CubePart model manager - lazy-loaded PartShapeDenoiserPipeline singleton.

Vendors Roblox's ``cube_part`` library (nodes/cubepart/vendor/cube_part) so the
node is self-contained and does not depend on a separately pip-installed package.
Only third-party *libraries* (warp-lang, etc.) are expected in the venv.

Model weights (downloaded ahead of time per the BrainDead pre-download rule):
  - Roblox/cubepart            -> /srv/AI_Stuff/models/cubepart
        multi_part_dit.safetensors (~8.6 GB), vae.safetensors (~1.3 GB)
  - Qwen/Qwen3-VL-4B-Instruct  -> /srv/AI_Stuff/models/LLM/Qwen3-VL-4B-Instruct
        text encoder for the open-vocabulary part prompts (loaded local-only)
  - Qwen/Qwen-Image            -> HF cache (only transformer/config.json is read;
        the DiT architecture is built via from_config and filled by the checkpoint)
"""
import os
import sys

import numpy as np
import torch

# --- vendor cube_part onto sys.path before importing it -----------------------
_VENDOR = os.path.join(os.path.dirname(__file__), "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

HAS_CUBEPART = False
IMPORT_ERROR = None
try:
    from cube_part.pipelines import PartShapeDenoiserPipeline, ShapeInput  # noqa: F401
    from cube_part.utils.mesh import load_mesh, sample_surface, rescale
    from cube_part.utils.config import load_config
    from cube_part.systems.shape_denoiser import ShapeDenoiserSystem
    HAS_CUBEPART = True
except Exception as e:  # pragma: no cover - import-time guard
    IMPORT_ERROR = e


# HF repos for auto-download on first run.
CUBEPART_REPO = "Roblox/cubepart"
TEXT_ENCODER_REPO = "Qwen/Qwen3-VL-4B-Instruct"
TEXT_ENCODER_SUBDIR = "Qwen3-VL-4B-Instruct"  # leaf dir name under the LLM models folder

# Fallback absolute paths, used only when folder_paths / extra_model_paths is unavailable.
DEFAULT_MODEL_DIR = "/srv/AI_Stuff/models/cubepart"
DEFAULT_TEXT_ENCODER = "/srv/AI_Stuff/models/LLM/Qwen3-VL-4B-Instruct"
CONFIG_PATH = os.path.join(_VENDOR, "configs", "shape_denoiser_multimesh.yaml")
# Pipeline hard limit (num_parts in PartShapeDenoiserPipeline.input_to_part_shape)
MAX_PARTS = 8
DEFAULT_MESH_SCALE = 0.96

_pipeline = None
_pipeline_key = None


def _folder_paths():
    """ComfyUI's folder_paths module if importable (None outside ComfyUI)."""
    try:
        import folder_paths
        return folder_paths
    except Exception:
        return None


# Register a default "cubepart" model folder so it shows up in folder_paths even
# without an extra_model_paths.yaml entry. An entry in extra_model_paths.yaml
# (e.g. cubepart: /srv/AI_Stuff/models/cubepart/) is merged in front of this and
# wins. Idempotent + best-effort.
_fp_boot = _folder_paths()
if _fp_boot is not None:
    try:
        names = getattr(_fp_boot, "folder_names_and_paths", {})
        if "cubepart" not in names:
            _fp_boot.add_model_folder_path(
                "cubepart", os.path.join(_fp_boot.models_dir, "cubepart")
            )
    except Exception:
        pass


def resolve_model_dir(override: str = "") -> str:
    """Resolve the cubepart weights dir. Explicit override wins; otherwise use
    folder_paths (which respects extra_model_paths.yaml), preferring a folder that
    already holds the weights. Falls back to models_dir/cubepart, then DEFAULT."""
    ov = (override or "").strip()
    if ov and ov.lower() != "auto":
        return ov
    fp = _folder_paths()
    if fp is None:
        return DEFAULT_MODEL_DIR
    candidates = []
    try:
        candidates = list(fp.get_folder_paths("cubepart"))
    except Exception:
        candidates = []
    if not candidates:
        candidates = [os.path.join(fp.models_dir, "cubepart")]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "multi_part_dit.safetensors")):
            return c
    return candidates[0]


def resolve_text_encoder(override: str = "") -> str:
    """Resolve the Qwen3-VL text-encoder dir. Explicit override wins; otherwise
    place it under the registered LLM models folder (extra_model_paths), preferring
    a copy that already exists. Falls back to models_dir/LLM, then DEFAULT."""
    ov = (override or "").strip()
    if ov and ov.lower() != "auto":
        return ov
    fp = _folder_paths()
    if fp is None:
        return DEFAULT_TEXT_ENCODER
    roots = []
    try:
        roots = list(fp.get_folder_paths("LLM"))
    except Exception:
        roots = []
    if not roots:
        roots = [os.path.join(fp.models_dir, "LLM")]
    candidates = [os.path.join(r, TEXT_ENCODER_SUBDIR) for r in roots]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "config.json")):
            return c
    return candidates[0]


def ensure_weights(model_dir: str, text_encoder_dir: str, auto_download: bool = True):
    """Make sure both model sets are on disk, downloading from HF on first run when
    auto_download is True. Honors the user's pre-download preference: present = skip."""
    dit = os.path.join(model_dir, "multi_part_dit.safetensors")
    vae = os.path.join(model_dir, "vae.safetensors")
    if not (os.path.isfile(dit) and os.path.isfile(vae)):
        if not auto_download:
            raise FileNotFoundError(
                f"CubePart weights missing in {model_dir} and auto_download is off.\n"
                f"  huggingface-cli download {CUBEPART_REPO} --local-dir {model_dir}"
            )
        from huggingface_hub import snapshot_download
        print(f"[BD CubePart] Weights not found in {model_dir} — downloading "
              f"{CUBEPART_REPO} (~10GB, first run only)...")
        snapshot_download(repo_id=CUBEPART_REPO, local_dir=model_dir,
                          allow_patterns=["*.safetensors", "*.json"])

    if not os.path.isfile(os.path.join(text_encoder_dir, "config.json")):
        if not auto_download:
            raise FileNotFoundError(
                f"Text encoder missing in {text_encoder_dir} and auto_download is off.\n"
                f"  huggingface-cli download {TEXT_ENCODER_REPO} --local-dir {text_encoder_dir}"
            )
        from huggingface_hub import snapshot_download
        print(f"[BD CubePart] Text encoder not found in {text_encoder_dir} — downloading "
              f"{TEXT_ENCODER_REPO} (~8GB, first run only)...")
        snapshot_download(repo_id=TEXT_ENCODER_REPO, local_dir=text_encoder_dir)


def _build_pipeline(model_dir: str, text_encoder_path: str, device: str = "cuda"):
    """Construct a PartShapeDenoiserPipeline pointed at our local weights.

    We bypass the stock ``__init__`` (which re-loads the YAML and has no hook for
    a local text-encoder path) and assemble the pipeline attributes directly, so
    we can inject ``base_model_path`` for fully-offline Qwen3-VL loading.
    """
    config = load_config(CONFIG_PATH)
    config.system.pretrained_model_path = os.path.join(model_dir, "multi_part_dit.safetensors")
    config.system.shape_model.pretrained_model_path = os.path.join(model_dir, "vae.safetensors")
    if text_encoder_path and os.path.isdir(text_encoder_path):
        config.system.base_model_path = text_encoder_path
    else:
        print(
            f"[BD CubePart] WARNING: local text encoder not found at "
            f"'{text_encoder_path}'; falling back to HF id "
            f"'{config.system.base_model_type}' (may require network)."
        )
    # inference-only overrides (match the stock constructor)
    config.system.attn_implementation = "sdpa"
    config.system.gradient_checkpointing = False

    system = ShapeDenoiserSystem(config.system).eval().to(device)
    torch.set_grad_enabled(False)

    pipe = PartShapeDenoiserPipeline.__new__(PartShapeDenoiserPipeline)
    pipe.system = system
    pipe.device = torch.device(device)
    pipe.cfg = config
    pipe.extract_geometry_fn_name = "extract_geometry_coarse_to_fine"
    return pipe


def get_pipeline(model_dir: str = "", text_encoder_path: str = "",
                 auto_download: bool = True):
    """Return a cached pipeline, rebuilding only if the resolved paths change.

    `model_dir` / `text_encoder_path` may be empty/"auto" to resolve via
    folder_paths (respecting extra_model_paths.yaml), or an explicit path override.
    Missing weights are downloaded from HF on first run when auto_download is True.
    """
    global _pipeline, _pipeline_key

    model_dir = resolve_model_dir(model_dir)
    text_encoder_path = resolve_text_encoder(text_encoder_path)

    key = (model_dir, text_encoder_path)
    if _pipeline is not None and _pipeline_key == key:
        return _pipeline

    ensure_weights(model_dir, text_encoder_path, auto_download=auto_download)

    print(f"[BD CubePart] Loading pipeline from {model_dir} "
          f"(text encoder: {text_encoder_path})...")
    torch.cuda.reset_peak_memory_stats()
    _pipeline = _build_pipeline(model_dir, text_encoder_path)
    _pipeline_key = key
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    print(f"[BD CubePart] Pipeline ready (load peak VRAM: {peak:.0f} MB).")
    return _pipeline


def prepare_surface(tri_mesh=None, mesh_path: str = "", num_samples: int = 128_000,
                    mesh_scale: float = DEFAULT_MESH_SCALE, device: str = "cuda"):
    """Turn an in-memory TRIMESH (or a .glb on disk) into the (1, N, 6) surface
    tensor the shape VAE expects. Mirrors examples/run_inference.py but accepts a
    live trimesh.Trimesh so upstream BD mesh nodes can feed it directly."""
    if tri_mesh is not None:
        mesh = tri_mesh.copy()
        verts, _, _ = rescale(np.asarray(mesh.vertices, dtype=np.float64), mesh_scale=mesh_scale)
        mesh.vertices = verts
    else:
        if not mesh_path:
            raise ValueError("CubePart needs either a `mesh` (TRIMESH) or a `mesh_path`.")
        mesh, _, _ = load_mesh(mesh_path, mesh_scale=mesh_scale)

    surface = sample_surface(mesh, num_samples=num_samples)
    surface = torch.from_numpy(surface).to(device).unsqueeze(0).float()
    return surface


def parse_parts(parts_text: str):
    """Parse comma- or newline-separated part names, drop blanks, cap to MAX_PARTS.

    Returns ``(parts, dropped)`` where ``dropped`` is the count truncated past the
    8-part model limit (logged by the caller).
    """
    raw = parts_text.replace("\n", ",").split(",")
    parts = [p.strip() for p in raw if p.strip()]
    dropped = 0
    if len(parts) > MAX_PARTS:
        dropped = len(parts) - MAX_PARTS
        parts = parts[:MAX_PARTS]
    return parts, dropped
