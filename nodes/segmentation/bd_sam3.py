"""
bd_sam3.py — BrainDead's in-house SAM3 loader + text/geometry segmentation.

Standalone (CLAUDE.md "tools stand alone" = no dependency on another node pack):
this loads SAM3 ourselves and does NOT touch comfyui-rmbg or any bundled `sam3/`
package. It uses ONLY comfy-core:

  • Model + text encoder come from the official ComfyUI-repackaged SAM 3.1 checkpoint
    (Comfy-Org/sam3.1 → checkpoints/sam3.1_multiplex_fp16.safetensors), AUTO-DOWNLOADED
    via huggingface_hub if absent. The "multiplex" file is a single checkpoint that
    comfy.sd.load_checkpoint_guess_config splits into the SAM3 detector/tracker MODEL
    AND the sam3_clip text encoder (CLIP) in one call (comfy registers a SAM3 clip_target).
  • Text segmentation mirrors comfy_extras/nodes_sam3.SAM3_Detect's text path exactly,
    reusing core's _extract_text_prompts / _refine_mask (core comfy, not a node pack).

Models are downloaded/loaded by us — that is the rule. We never require a sibling pack.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

import comfy.sd
import comfy.utils
import comfy.model_management

try:
    import folder_paths as _folder_paths
except ImportError:
    _folder_paths = None

_SAM3_REPO = "Comfy-Org/sam3.1"
_SAM3_FILE = "sam3.1_multiplex_fp16.safetensors"
_SAM3_HF_PATH = "checkpoints/" + _SAM3_FILE   # path within the HF repo
_SAM3_SIZE = 1008                              # SAM3 works in 1008×1008 space

_CACHE: dict = {}   # {"model":..., "clip":...}


# Reuse comfy-core's own text helpers (core, not a node pack). Fallback-replicate
# _extract_text_prompts if the private name ever moves.
try:
    from comfy_extras.nodes_sam3 import _extract_text_prompts, _refine_mask  # type: ignore
    _HAS_CORE_HELPERS = True
except Exception:
    _HAS_CORE_HELPERS = False

    def _extract_text_prompts(conditioning, device, dtype):
        cond_meta = conditioning[0][1]
        multi = cond_meta.get("sam3_multi_cond")
        prompts = []
        if multi is not None:
            for entry in multi:
                emb = entry["cond"].to(device=device, dtype=dtype)
                mask = entry["attention_mask"].to(device) if entry["attention_mask"] is not None else None
                if mask is None:
                    mask = torch.ones(emb.shape[0], emb.shape[1], dtype=torch.int64, device=device)
                prompts.append((emb, mask, entry.get("max_detections", 1)))
        else:
            emb = conditioning[0][0].to(device=device, dtype=dtype)
            mask = cond_meta.get("attention_mask")
            mask = mask.to(device) if mask is not None else torch.ones(
                emb.shape[0], emb.shape[1], dtype=torch.int64, device=device)
            prompts.append((emb, mask, 1))
        return prompts


def find_sam3_checkpoint(need_clip: bool = True) -> str | None:
    """An existing SAM3 checkpoint under any registered model dir, else None.

    When need_clip is True only the multiplex/full checkpoints qualify (they carry the
    sam3_clip text encoder); a detector-only `sam3.pt` is accepted only for geometry."""
    if _folder_paths is None:
        return None
    cands = [("checkpoints", _SAM3_FILE), ("checkpoints", "sam3.safetensors"),
             ("diffusion_models", _SAM3_FILE)]
    if not need_clip:
        cands += [("diffusion_models", "sam3.pt"), ("checkpoints", "sam3.pt")]
    for folder, name in cands:
        try:
            p = _folder_paths.get_full_path(folder, name)
        except Exception:
            p = None
        if p and os.path.exists(p):
            return p
    return None


def ensure_sam3_checkpoint(need_clip: bool = True) -> str:
    """Path to a SAM3 checkpoint — AUTO-DOWNLOADS the official Comfy-Org repackage if
    none is present. Returns the local path (downloaded into the HF cache)."""
    p = find_sam3_checkpoint(need_clip)
    if p:
        return p
    from huggingface_hub import hf_hub_download
    # Land it in the registered checkpoints dir (e.g. /srv/.../models/checkpoints) so it's
    # a proper local model — reusable, visible to CheckpointLoader, per the pre-download rule.
    try:
        ckpt_dir = _folder_paths.get_folder_paths("checkpoints")[0]
    except Exception:
        ckpt_dir = os.path.join(getattr(_folder_paths, "models_dir", "models"), "checkpoints")
    target = os.path.join(ckpt_dir, _SAM3_FILE)
    if os.path.exists(target):
        return target
    parent = os.path.dirname(ckpt_dir)   # repo path is "checkpoints/<file>" → lands in ckpt_dir
    print(f"[BD SAM3] {_SAM3_FILE} not found — downloading official {_SAM3_REPO} "
          f"(~1.75 GB, first use only) → {target}", flush=True)
    try:
        hf_hub_download(repo_id=_SAM3_REPO, filename=_SAM3_HF_PATH,
                        local_dir=parent, local_dir_use_symlinks=False)
    except TypeError:   # newer huggingface_hub dropped local_dir_use_symlinks
        hf_hub_download(repo_id=_SAM3_REPO, filename=_SAM3_HF_PATH, local_dir=parent)
    print(f"[BD SAM3] downloaded → {target}", flush=True)
    return target


def load_sam3(need_clip: bool = True):
    """Load (and cache) the SAM3 MODEL and its text-encoder CLIP from one checkpoint.

    need_clip=True requires the multiplex checkpoint (text prompts). need_clip=False
    accepts a detector-only checkpoint (geometry prompts only)."""
    if "model" in _CACHE and (_CACHE.get("clip") is not None or not need_clip):
        return _CACHE["model"], _CACHE.get("clip")
    path = ensure_sam3_checkpoint(need_clip)
    emb = None
    try:
        emb = _folder_paths.get_folder_paths("embeddings")
    except Exception:
        emb = None
    out = comfy.sd.load_checkpoint_guess_config(
        path, output_vae=False, output_clip=need_clip, embedding_directory=emb)
    model, clip = out[0], (out[1] if len(out) > 1 else None)
    if need_clip and clip is None:
        raise RuntimeError(
            f"SAM3 checkpoint at {path} has no text encoder — text prompts need the "
            f"multiplex checkpoint ({_SAM3_FILE} from {_SAM3_REPO}).")
    _CACHE["model"], _CACHE["clip"] = model, clip
    return model, clip


def _sam3_runtime(model):
    comfy.model_management.load_model_gpu(model)
    return (model.model.diffusion_model,
            comfy.model_management.get_torch_device(),
            model.model.get_dtype())


def segment_text(model, clip, image, prompt: str, threshold: float = 0.5,
                 refine_iterations: int = 2, max_det: int = 0) -> torch.Tensor:
    """Text-grounded SAM3 for ONE prompt → a single [1, H, W] mask (union of kept
    detections). Mirrors SAM3_Detect's text path. Returns a CPU float mask in [0,1]."""
    sam3, device, dtype = _sam3_runtime(model)
    img = image if image.ndim == 4 else image.unsqueeze(0)
    B, H, W, C = img.shape
    frame = comfy.utils.common_upscale(
        img[0:1, ..., :3].movedim(-1, 1), _SAM3_SIZE, _SAM3_SIZE, "bilinear", crop="disabled"
    ).to(device=device, dtype=dtype)
    orig_hwc = img[0].to(device=device, dtype=dtype)

    cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    acc = torch.zeros((H, W), device=device, dtype=torch.float32)
    for emb, tmask, default_max in _extract_text_prompts(cond, device, dtype):
        results = sam3(frame, text_embeddings=emb, text_mask=tmask,
                       boxes=None, threshold=threshold, orig_size=(H, W))
        scores = results["scores"][0]
        boxes = results["boxes"][0]
        masks = results["masks"][0]
        probs = scores.sigmoid()
        keep = probs > threshold
        kept_masks = masks[keep]
        kept_boxes = boxes[keep]
        kept_probs = probs[keep]
        lim = max_det if max_det and max_det > 0 else default_max if default_max else kept_masks.shape[0]
        order = kept_probs.argsort(descending=True)[:max(1, lim)] if kept_masks.shape[0] else []
        for idx in order:
            m = _refine_mask(sam3, orig_hwc, kept_masks[idx], kept_boxes[idx],
                             H, W, device, dtype, refine_iterations)[0]
            acc = torch.maximum(acc, m.float())
    return acc.unsqueeze(0).cpu()
