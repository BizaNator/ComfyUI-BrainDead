"""
Make-It-Animatable inference wrapper for ComfyUI-BrainDead.

Provides functions to load MIA models and run inference for humanoid
rigging. Uses vendored MIA code from lib/autorig/mia/ for model loading
(no bpy dependency at inference time).

Vendored from ComfyUI-UniRig (PozzettiAndrea) — MIT licensed. The upstream
Make-It-Animatable project: https://github.com/jasongzy/Make-It-Animatable

Model weights download from HuggingFace (jasongzy/Make-It-Animatable) on
first use into ComfyUI's models/autorig/mia/ directory.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import MagicMock

# Mock bpy before importing external MIA modules for inference
# The inference functions don't use bpy, but transitively import it
if 'bpy' not in sys.modules:
    sys.modules['bpy'] = MagicMock()

import numpy as np
import torch
import trimesh

# ── Path layout ────────────────────────────────────────────────────────
# Source code lives at: <pack>/lib/autorig/mia/  (vendored)
# Model weights live at: <ComfyUI>/models/autorig/mia/output/best/new/
LIB_DIR = Path(__file__).parent.absolute()  # <pack>/lib/autorig/

# Resolve models dir via ComfyUI's folder_paths if available, else fallback
try:
    import folder_paths
    _MODELS_ROOT = Path(folder_paths.models_dir) / "autorig" / "mia"
except ImportError:
    # Headless / no ComfyUI context — fall back to lib-local
    _MODELS_ROOT = LIB_DIR.parent.parent / "_local_models" / "mia"

# MIA models live under output/best/new/ on HF, mirror that structure
if os.environ.get('MIA_MODELS_PATH'):
    MIA_MODELS_DIR = Path(os.environ['MIA_MODELS_PATH'])
else:
    MIA_MODELS_DIR = _MODELS_ROOT / "output" / "best" / "new"

# HuggingFace local_dir target — relative paths inside become full paths
MIA_PATH = _MODELS_ROOT

# Required model files
MIA_MODEL_FILES = [
    "bw.pth",
    "bw_normal.pth",
    "joints.pth",
    "joints_coarse.pth",
    "pose.pth",
]

# Global cache for loaded models
_MIA_MODEL_CACHE: Dict[str, Any] = {}


def ensure_mia_models() -> bool:
    """
    Ensure MIA model files are downloaded.
    Downloads from HuggingFace if not present.

    Returns:
        True if all models are available, False otherwise.
    """
    missing = [m for m in MIA_MODEL_FILES if not (MIA_MODELS_DIR / m).exists()]

    if not missing:
        return True

    print(f"[MIA] Downloading missing models: {missing}")

    try:
        from huggingface_hub import hf_hub_download

        MIA_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        for model_file in missing:
            print(f"[MIA] Downloading {model_file}...")
            hf_hub_download(
                repo_id="jasongzy/Make-It-Animatable",
                filename=f"output/best/new/{model_file}",
                local_dir=str(MIA_PATH),
                local_dir_use_symlinks=False,
            )

        print(f"[MIA] All models downloaded successfully")
        return True

    except Exception as e:
        print(f"[MIA] Error downloading models: {e}")
        return False


def load_mia_models(cache_to_gpu: bool = True) -> Dict[str, Any]:
    """
    Load all MIA models into memory.

    Args:
        cache_to_gpu: If True, keep models on GPU for faster inference.

    Returns:
        Dictionary containing loaded models and metadata.
    """
    global _MIA_MODEL_CACHE

    cache_key = f"mia_models_gpu={cache_to_gpu}"

    if cache_key in _MIA_MODEL_CACHE:
        print(f"[MIA] Using cached models")
        return _MIA_MODEL_CACHE[cache_key]

    # Ensure models are downloaded
    if not ensure_mia_models():
        raise RuntimeError("Failed to download MIA models")

    device = torch.device("cuda" if torch.cuda.is_available() and cache_to_gpu else "cpu")
    print(f"[MIA] Loading models to {device}...")

    # Import vendored MIA modules (no bpy dependency)
    from .mia import PCAE, JOINTS_NUM, KINEMATIC_TREE

    N = 32768  # Number of points to sample
    hands_resample_ratio = 0.5
    geo_resample_ratio = 0.0
    hierarchical_ratio = hands_resample_ratio + geo_resample_ratio

    # Load coarse joints model (for preprocessing)
    print(f"[MIA] Loading joints_coarse model...")
    model_coarse = PCAE(
        N=N,
        input_normal=False,
        deterministic=True,
        output_dim=JOINTS_NUM,
        predict_bw=False,
        predict_joints=True,
        predict_joints_tail=True,
    )
    model_coarse.load(str(MIA_MODELS_DIR / "joints_coarse.pth")).to(device).eval()

    # Load blend weights model
    print(f"[MIA] Loading bw model...")
    model_bw = PCAE(
        N=N,
        input_normal=False,
        input_attention=False,
        deterministic=True,
        hierarchical_ratio=hierarchical_ratio,
    )
    model_bw.load(str(MIA_MODELS_DIR / "bw.pth")).to(device).eval()

    # Load blend weights with normals model
    print(f"[MIA] Loading bw_normal model...")
    model_bw_normal = PCAE(
        N=N,
        input_normal=True,
        input_attention=True,  # Checkpoint trained with attention
        deterministic=True,
        hierarchical_ratio=hierarchical_ratio,
    )
    model_bw_normal.load(str(MIA_MODELS_DIR / "bw_normal.pth")).to(device).eval()

    # Load joints model
    print(f"[MIA] Loading joints model...")
    model_joints = PCAE(
        N=N,
        input_normal=False,
        deterministic=True,
        hierarchical_ratio=hierarchical_ratio,
        output_dim=JOINTS_NUM,
        kinematic_tree=KINEMATIC_TREE,
        predict_bw=False,
        predict_joints=True,
        predict_joints_tail=True,
        joints_attn_causal=True,  # Match original MIA config
    )
    model_joints.load(str(MIA_MODELS_DIR / "joints.pth")).to(device).eval()

    # Load pose model
    print(f"[MIA] Loading pose model...")
    model_pose = PCAE(
        N=N,
        input_normal=False,
        deterministic=True,
        hierarchical_ratio=hierarchical_ratio,
        output_dim=JOINTS_NUM,
        kinematic_tree=KINEMATIC_TREE,
        predict_bw=False,
        predict_pose_trans=True,
        pose_mode="ortho6d",  # Match original MIA config
        pose_input_joints=True,
        pose_attn_causal=True,  # Match original MIA config
    )
    model_pose.load(str(MIA_MODELS_DIR / "pose.pth")).to(device).eval()

    models = {
        "backend": "make_it_animatable",
        "model_coarse": model_coarse,
        "model_bw": model_bw,
        "model_bw_normal": model_bw_normal,
        "model_joints": model_joints,
        "model_pose": model_pose,
        "device": device,
        "cache_to_gpu": cache_to_gpu,
        "N": N,
        "hands_resample_ratio": hands_resample_ratio,
        "geo_resample_ratio": geo_resample_ratio,
    }

    _MIA_MODEL_CACHE[cache_key] = models
    print(f"[MIA] All models loaded successfully")

    return models


def run_mia_inference(
    mesh: trimesh.Trimesh,
    models: Dict[str, Any],
    output_path: str,
    no_fingers: bool = True,
    use_normal: bool = False,
    reset_to_rest: bool = True,
) -> str:
    """
    Run Make-It-Animatable inference on a mesh.

    Args:
        mesh: Input trimesh object.
        models: Loaded MIA models from load_mia_models().
        output_path: Path for output FBX file.
        no_fingers: If True, merge finger weights to hand (for models without separate fingers).
        use_normal: If True, use normals for better weights when limbs are close.
        reset_to_rest: If True, transform output to T-pose rest position.

    Returns:
        Path to output FBX file.
    """
    # Use vendored pipeline (no external dependencies)
    from .mia.pipeline import prepare_input, preprocess, infer, bw_post_process
    from .mia import BONES_IDX_DICT

    device = models["device"]
    N = models["N"]

    print(f"[MIA] Starting inference...")
    print(f"[MIA] Options: no_fingers={no_fingers}, use_normal={use_normal}, reset_to_rest={reset_to_rest}")

    # Prepare input
    print(f"[MIA] Preparing input...")
    data = prepare_input(
        mesh,
        N=N,
        hands_resample_ratio=models["hands_resample_ratio"],
        geo_resample_ratio=models["geo_resample_ratio"],
        get_normals=use_normal,
    )

    # Preprocess (normalize, coarse joint localization)
    print(f"[MIA] Preprocessing...")
    data = preprocess(
        data,
        model_coarse=models["model_coarse"],
        device=device,
        hands_resample_ratio=models["hands_resample_ratio"],
        geo_resample_ratio=models["geo_resample_ratio"],
        N=N,
    )

    # Run main inference
    print(f"[MIA] Running model inference...")
    data = infer(
        data,
        model_bw=models["model_bw"],
        model_bw_normal=models["model_bw_normal"],
        model_joints=models["model_joints"],
        model_pose=models["model_pose"],
        device=device,
        use_normal=use_normal,
    )

    # Post-process blend weights
    print(f"[MIA] Post-processing...")
    # Get head position for above-head mask
    joints = data.joints
    head_idx = BONES_IDX_DICT[f"mixamorig:Head"]
    head_y = joints[..., head_idx, 4]  # tail y position (index 3:6 is tail)
    above_head_mask = data.verts[..., 1] >= head_y

    bw = bw_post_process(
        data.bw,
        bones_idx_dict=BONES_IDX_DICT,
        above_head_mask=above_head_mask,
        no_fingers=no_fingers,
    )

    # Prepare output data for Blender export
    joints_np = data.joints.squeeze(0).numpy()
    output_data = {
        "mesh": data.mesh,
        "gs": None,
        "joints": joints_np[..., :3],
        "joints_tail": joints_np[..., 3:] if joints_np.shape[-1] > 3 else None,
        "bw": bw.squeeze(0).numpy(),
        "pose": data.pose.squeeze(0).numpy() if reset_to_rest and data.pose is not None else None,
        "bones_idx_dict": BONES_IDX_DICT,
        "pose_ignore_list": [],
    }

    # Export to FBX using MIA's Blender integration
    print(f"[MIA] Exporting to FBX...")
    _export_mia_fbx(output_data, output_path, no_fingers, reset_to_rest)

    print(f"[MIA] Inference complete: {output_path}")
    return output_path


def _export_mia_fbx(
    data: Dict[str, Any],
    output_path: str,
    remove_fingers: bool,
    reset_to_rest: bool,
) -> None:
    """
    Export MIA results to FBX using Blender.

    Uses the bundled Blender from UniRig for consistency.
    """
    import subprocess
    import tempfile
    import json

    # Use the pack's standard Blender discovery (nodes/blender/base.py).
    # Falls back to common system paths.
    BLENDER_EXE = None
    try:
        # Walk up to <pack>/nodes/blender/base.py
        _here = Path(__file__).resolve()
        _pack_root = _here.parents[2]  # <pack>/
        # Add pack to sys.path so the nodes namespace resolves
        sys.path.insert(0, str(_pack_root.parent))
        from nodes.blender.base import find_blender  # type: ignore
        BLENDER_EXE = find_blender()
    except Exception:
        pass

    if BLENDER_EXE is None:
        import shutil
        for cand in ("blender", "/usr/bin/blender", "/usr/local/bin/blender"):
            p = shutil.which(cand) if "/" not in cand else (cand if Path(cand).exists() else None)
            if p:
                BLENDER_EXE = p
                break

    if BLENDER_EXE is None:
        raise RuntimeError(
            "Blender not found for MIA FBX export. Install Blender 4.2+ "
            "system-wide, or set BD_AUTORIG_BLENDER_PATH environment var."
        )

    # Save mesh to temp file (can't pickle trimesh directly)
    temp_mesh_path = tempfile.NamedTemporaryFile(suffix=".glb", delete=False).name
    mesh = data["mesh"]
    if hasattr(mesh, 'export'):
        mesh.export(temp_mesh_path)
    else:
        # If it's already a path or vertices/faces
        import trimesh
        trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces).export(temp_mesh_path)

    # Save data using JSON + raw binary (avoids numpy pickle version issues)
    import json
    temp_dir = tempfile.mkdtemp()
    temp_json = os.path.join(temp_dir, "data.json")
    temp_bw = os.path.join(temp_dir, "bw.bin")
    temp_joints = os.path.join(temp_dir, "joints.bin")
    temp_joints_tail = os.path.join(temp_dir, "joints_tail.bin")

    # Save arrays as raw binary
    data["bw"].astype(np.float32).tofile(temp_bw)
    data["joints"].astype(np.float32).tofile(temp_joints)
    if data.get("joints_tail") is not None:
        data["joints_tail"].astype(np.float32).tofile(temp_joints_tail)

    # Save metadata as JSON
    bones_idx_dict = dict(data["bones_idx_dict"])
    json_data = {
        "mesh_path": temp_mesh_path,
        "bw_path": temp_bw,
        "bw_shape": list(data["bw"].shape),
        "joints_path": temp_joints,
        "joints_shape": list(data["joints"].shape),
        "bones_idx_dict": bones_idx_dict,
    }
    if data.get("joints_tail") is not None:
        json_data["joints_tail_path"] = temp_joints_tail
        json_data["joints_tail_shape"] = list(data["joints_tail"].shape)

    with open(temp_json, 'w') as f:
        json.dump(json_data, f)

    try:
        # Get template path - use UniRig's bundled Mixamo template
        ASSETS_DIR = LIB_DIR.parent / "assets"
        template_path = ASSETS_DIR / "animation_characters" / "mixamo.fbx"
        if not template_path.exists():
            # Fallback to MIA template if available
            mia_template = MIA_PATH / "data/Mixamo/character/Ch14_nonPBR.fbx"
            if mia_template.exists():
                template_path = mia_template
            else:
                raise FileNotFoundError(f"No Mixamo template found. Expected at: {template_path}")

        # Build Blender command - use our own script (no torch dependency)
        blender_script = LIB_DIR / "blender" / "mia_export.py"

        cmd = [
            str(BLENDER_EXE),
            "--background",
            "--python", str(blender_script),
            "--",
            "--input_path", temp_json,
            "--output_path", output_path,
            "--template_path", str(template_path),
        ]

        if remove_fingers:
            cmd.append("--remove_fingers")
        if reset_to_rest:
            cmd.append("--reset_to_rest")

        # Run Blender
        print(f"[MIA] Running Blender: {' '.join(cmd[:4])}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Always print output for debugging
        if result.stdout:
            print(f"[MIA] Blender stdout: {result.stdout[-2000:]}")  # Last 2000 chars
        if result.stderr:
            print(f"[MIA] Blender stderr: {result.stderr[-2000:]}")

        if result.returncode != 0:
            raise RuntimeError(f"Blender export failed with code {result.returncode}")

        # Verify output was created
        if not os.path.exists(output_path):
            raise RuntimeError(f"Blender completed but output file not created: {output_path}")

    finally:
        # Cleanup temp files
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if os.path.exists(temp_mesh_path):
            os.remove(temp_mesh_path)


def clear_mia_cache():
    """Clear the MIA model cache."""
    global _MIA_MODEL_CACHE
    _MIA_MODEL_CACHE.clear()
    print("[MIA] Model cache cleared")
