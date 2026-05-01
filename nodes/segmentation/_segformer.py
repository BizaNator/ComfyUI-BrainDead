"""
Shared SegFormer load + inference for human-parsing backends.

Both fashn-ai/fashn-human-parser and mattmdjaga/segformer_b2_clothes are
SegFormer semantic segmentation checkpoints with the same call shape, so
the loader and runner are factored here.

Default cache_dir is /srv/AI_Stuff/models/huggingface (shared model storage
on this host) when present; otherwise HF's own resolution is used.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F


_DEFAULT_CACHE = "/srv/AI_Stuff/models/huggingface"
_MODEL_CACHE: dict[str, tuple] = {}


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dtype(dtype: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]


def resolve_cache_dir(cache_dir: str) -> str | None:
    if cache_dir:
        return cache_dir
    if os.path.isdir(_DEFAULT_CACHE):
        return _DEFAULT_CACHE
    return None


def load_segformer(model_id: str, device: str, dtype: torch.dtype, cache_dir: str | None):
    key = f"{model_id}|{device}|{dtype}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir, use_fast=False)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_id, cache_dir=cache_dir, dtype=dtype
    ).to(device).eval()

    _MODEL_CACHE[key] = (processor, model)
    return processor, model


def _tensor_batch_to_pil_list(image: torch.Tensor, max_dim: int = 0):
    from PIL import Image
    arr = (image.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    out = []
    for i in range(arr.shape[0]):
        pil = Image.fromarray(arr[i])
        if max_dim > 0 and max(pil.size) > max_dim:
            scale = max_dim / max(pil.size)
            new_size = (int(round(pil.width * scale)), int(round(pil.height * scale)))
            pil = pil.resize(new_size, Image.LANCZOS)
        out.append(pil)
    return out


@torch.no_grad()
def run_segformer(
    processor,
    model,
    image: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    inference_size: int = 0,
    confidence_threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    image: ComfyUI tensor (B, H, W, 3) in [0, 1].
    inference_size: max image dim before processor (0 = let processor decide).
                    Smaller → sharper class boundaries (less bilinear bleed),
                    coarser detail.
    confidence_threshold: pixels with max-softmax below this go to background (class 0).

    Returns (class_map LongTensor (B, H, W), confidence FloatTensor (B, H, W)) on CPU.
    """
    b, h, w, _ = image.shape
    pil_list = _tensor_batch_to_pil_list(image, max_dim=inference_size)

    inputs = processor(images=pil_list, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)

    logits = model(pixel_values=pixel_values).logits
    upsampled = F.interpolate(logits.float(), size=(h, w), mode="bilinear", align_corners=False)

    probs = F.softmax(upsampled, dim=1)
    confidence, class_map = probs.max(dim=1)

    if confidence_threshold > 0.0:
        class_map = torch.where(
            confidence >= confidence_threshold,
            class_map,
            torch.zeros_like(class_map),
        )

    return class_map.cpu().long(), confidence.cpu().float()
