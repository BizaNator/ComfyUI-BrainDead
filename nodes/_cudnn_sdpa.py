"""Take cuDNN out of ComfyUI's SDPA backend priority on the cuDNN builds where its masked attention fails.

On BRAINZ (torch 2.9.1+cu130, cuDNN 9.13) masked scaled-dot-product attention through cuDNN fails intermittently
with "cuDNN Frontend error: No valid execution plans built" - seen in Qwen Image 2.1's prefix attention and in
core SAM3's detector (2026-09-23), in whichever model runs first after others have filled VRAM. The next job in
that process can then abort ComfyUI. comfy.ops keeps a module-level SDPA_BACKEND_PRIORITY list and reads it on
every call; this removes CUDNN_ATTENTION from it once, at load, for cuDNN 9.10-9.15 (the range comfy.ops already
treats as buggy on torch 2.9-2.10). Flash and memory-efficient attention take the same calls.

BD_KEEP_CUDNN_ATTENTION=1 in the environment leaves the list alone.
"""
import logging
import os


def apply():
    if os.environ.get("BD_KEEP_CUDNN_ATTENTION") == "1":
        return False
    try:
        import torch
        import comfy.ops as ops
        from torch.nn.attention import SDPBackend
        prio = getattr(ops, "SDPA_BACKEND_PRIORITY", None)
        ver = torch.backends.cudnn.version() if torch.cuda.is_available() else None
    except Exception:
        return False
    if prio is None or ver is None or not (91000 <= ver < 91500) or SDPBackend.CUDNN_ATTENTION not in prio:
        return False
    prio[:] = [b for b in prio if b != SDPBackend.CUDNN_ATTENTION]
    logging.warning("[BrainDead] cuDNN %s: cuDNN attention removed from the SDPA backend priority "
                    "(masked attention fails with 'No valid execution plans built'). "
                    "BD_KEEP_CUDNN_ATTENTION=1 keeps it.", ver)
    return True
