"""
bd_face_parse.py — in-house SegFormer face parsing (1038lab/segformer_face).

Standalone (CLAUDE.md "tools stand alone" = no other-node-pack dependency): we load
the model ourselves via transformers (auto-downloads on first use) — NOT through
comfyui-rmbg. 19 semantic classes including a real **Neck (17)**, **Clothing (18)** and
**Necklace (16)**, which is what makes reliable neck removal possible (SAM3 text "neck"
and MediaPipe jaw landmarks can't do it on stylized renders).
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

try:
    import folder_paths as _folder_paths
    _CACHE_DIR = os.environ.get("HF_HOME", os.path.join(_folder_paths.models_dir, "huggingface"))
except Exception:
    _CACHE_DIR = None

_REPO = "1038lab/segformer_face"

# CelebAMask-HQ 19-class face-parsing label map (same as comfyui-rmbg's Face Segment).
CLASS_MAP = {
    "Background": 0, "Skin": 1, "Nose": 2, "Eyeglasses": 3,
    "Left-eye": 4, "Right-eye": 5, "Left-eyebrow": 6, "Right-eyebrow": 7,
    "Left-ear": 8, "Right-ear": 9, "Mouth": 10, "Upper-lip": 11,
    "Lower-lip": 12, "Hair": 13, "Hat": 14, "Earring": 15,
    "Necklace": 16, "Neck": 17, "Clothing": 18,
}

_M: dict = {}


def _load():
    if "model" in _M:
        return _M["model"], _M["device"]
    from transformers import AutoModelForSemanticSegmentation
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BD FaceParse] loading {_REPO} (auto-download to {_CACHE_DIR})…", flush=True)
    model = AutoModelForSemanticSegmentation.from_pretrained(_REPO, cache_dir=_CACHE_DIR).to(dev).eval()
    _M["model"], _M["device"] = model, dev
    return model, dev


def parse(image, process_res: int = 512) -> np.ndarray:
    """image: ComfyUI IMAGE tensor [B,H,W,C] or [H,W,C]. Returns an int label map (H,W)
    at the image's native resolution. Preprocessing matches the segformer_face training
    (resize → ImageNet normalize), upsampled logits → argmax."""
    import torchvision.transforms as T
    from PIL import Image as _PIL
    model, dev = _load()
    img = image if image.ndim == 4 else image.unsqueeze(0)
    arr = (img[0, ..., :3].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    H, W = arr.shape[:2]
    x = T.Compose([
        T.Resize((process_res, process_res)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(_PIL.fromarray(arr)).unsqueeze(0).to(dev)
    with torch.no_grad():
        logits = model(x).logits
        up = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        pred = up.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
    return pred


def class_mask(pred: np.ndarray, names) -> np.ndarray:
    """uint8 (H,W) union of the given class names from a parse() label map."""
    m = np.zeros(pred.shape, np.uint8)
    for n in names:
        if n in CLASS_MAP:
            m[pred == CLASS_MAP[n]] = 255
    return m
