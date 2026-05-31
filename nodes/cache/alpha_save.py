"""
Shared alpha-channel helpers for BD save nodes.

Three save nodes (BD_SaveFile, BD_BulkSave, BD_SaveBatch) all support the same
alpha options:
  save_alpha_separately  — also write alpha as a standalone greyscale PNG
  alpha_mask             — bake a B&W mask as the saved file's transparency
  invert_alpha           — flip the mask polarity before baking

Import the helpers + the three standard io.Input definitions from this module
so all save nodes stay in sync without duplicating logic.
"""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

# ─── shared io.Input objects ─────────────────────────────────────────────────
# Import into a node's define_schema inputs list with:
#   inputs=[..., *ALPHA_SAVE_INPUTS]

ALPHA_SAVE_INPUTS: list = [
    io.Boolean.Input(
        "save_alpha_separately", default=False, optional=True,
        tooltip=(
            "Also write the alpha channel as a standalone greyscale PNG alongside the main "
            "file, named with the same suffix + '_alpha'.\n\n"
            "Source of the alpha pixels:\n"
            "  • alpha_mask wired → uses the mask (after invert_alpha)\n"
            "  • no mask wired, image has 4 channels (RGBA) → extracts the embedded A channel\n"
            "  • no mask wired, image is RGB (3ch) → no alpha to extract; this output is skipped\n\n"
            "NOTE: the _alpha.png is a RAW greyscale representation — white pixel = opaque area, "
            "black pixel = transparent area. It is NOT a composited preview; open the main RGBA PNG "
            "in a viewer that supports transparency to see the actual cut-out."
        ),
    ),
    io.Mask.Input(
        "alpha_mask", optional=True,
        tooltip=(
            "Bake this mask into the saved file's alpha channel before writing.\n\n"
            "Convention: WHITE (1.0) = OPAQUE, BLACK (0.0) = TRANSPARENT.\n"
            "  Face mask (white face, black background) → face is opaque, background is cut out.\n"
            "  Background mask (white background, black face) → use invert_alpha=True.\n\n"
            "If you already have an RGBA image (4-channel), you do NOT need to wire alpha_mask — "
            "the embedded alpha is preserved automatically when saving.\n\n"
            "The upstream image tensor is NOT modified — transparency is baked only in the saved file. "
            "Accepts batched masks (B, H, W); single (H,W) or (1,H,W) applies to all frames."
        ),
    ),
    io.Boolean.Input(
        "invert_alpha", default=False, optional=True,
        tooltip=(
            "Flip the mask polarity before baking as alpha.\n"
            "Use when your mask has white=background / black=subject and you want the subject opaque."
        ),
    ),
]


# ─── helpers ─────────────────────────────────────────────────────────────────

def get_frame_mask(
    alpha_mask: torch.Tensor | None,
    frame_idx: int,
    H: int,
    W: int,
) -> torch.Tensor | None:
    """Return a (H, W) float32 mask tensor for the requested frame, or None."""
    if alpha_mask is None:
        return None
    m = alpha_mask.float()
    # Normalise to (B, H, W)
    if m.ndim == 2:
        frame_m = m
    elif m.ndim == 3:
        frame_m = m[frame_idx] if frame_idx < m.shape[0] else m[0]
    elif m.ndim == 4:
        m = m.squeeze(-1) if m.shape[-1] == 1 else m[..., 0]
        frame_m = m[frame_idx] if frame_idx < m.shape[0] else m[0]
    else:
        return None
    # Resize if needed
    if frame_m.shape[0] != H or frame_m.shape[1] != W:
        frame_m = torch.nn.functional.interpolate(
            frame_m.unsqueeze(0).unsqueeze(0).float(),
            size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze(0).squeeze(0)
    return frame_m.clamp(0.0, 1.0)


def apply_alpha_to_frame(
    single: torch.Tensor,
    frame_mask: torch.Tensor | None,
    invert_alpha: bool,
) -> torch.Tensor:
    """Bake frame_mask into the alpha channel of a (1, H, W, C) tensor.

    Returns (1, H, W, 4) regardless of whether the input was RGB or RGBA.
    If frame_mask is None the input is returned unchanged.
    """
    if frame_mask is None:
        return single
    if invert_alpha:
        frame_mask = 1.0 - frame_mask
    alpha_ch = frame_mask.unsqueeze(-1)          # (H, W, 1)
    if single.shape[-1] == 4:
        result = single.clone()
        result[0, ..., 3:4] = alpha_ch
        return result
    # RGB → RGBA
    return torch.cat([single, alpha_ch.unsqueeze(0)], dim=-1)  # (1, H, W, 4)


def save_alpha_file(alpha_tensor: torch.Tensor, filepath: str) -> str:
    """Save a (H, W) float32 alpha tensor as a transparent-background RGBA PNG.

    Output format: RGB=white, A=alpha value.
    White areas are fully opaque; black areas are fully transparent.
    Suitable for direct engine import as a cutout/sprite (no solid background).
    """
    from PIL import Image
    alpha_np = (alpha_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    H, W = alpha_np.shape
    rgba = np.stack([
        np.full((H, W), 255, dtype=np.uint8),  # R white
        np.full((H, W), 255, dtype=np.uint8),  # G white
        np.full((H, W), 255, dtype=np.uint8),  # B white
        alpha_np,                               # A = mask
    ], axis=-1)
    Image.fromarray(rgba, mode="RGBA").save(filepath, "PNG")
    return filepath


def alpha_to_rgba_tensor(alpha_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a (H, W) float32 alpha tensor to a (1, H, W, 4) white+transparent IMAGE tensor.

    Matches the format written by save_alpha_file: white where opaque,
    transparent where black. Wire to PreviewImage or BD_SaveBatch preview_alpha.
    """
    H, W = alpha_tensor.shape[-2], alpha_tensor.shape[-1]
    a = alpha_tensor.cpu().float().reshape(H, W)
    white = torch.ones((H, W, 3), dtype=torch.float32)
    return torch.cat([white, a.unsqueeze(-1)], dim=-1).unsqueeze(0)  # (1, H, W, 4)


def alpha_file_path(main_filepath: str) -> str:
    """Derive the _alpha.png path from a main file path.

    Examples:
      /out/char_sr_light.png  →  /out/char_sr_light_alpha.png
      /out/char_sr_light      →  /out/char_sr_light_alpha.png
    """
    if "." in main_filepath:
        base, _ext = main_filepath.rsplit(".", 1)
    else:
        base = main_filepath
    return base + "_alpha.png"


def save_alpha_alongside(
    single_to_save: torch.Tensor,
    frame_mask: torch.Tensor | None,
    invert_alpha: bool,
    main_filepath: str,
    context_id: str,
    suffix: str,
    custom_vars: str,
) -> tuple[str | None, str]:
    """Save the alpha channel as a greyscale PNG next to the main file.

    Tries to resolve via context (suffix + '_alpha') first; falls back to
    deriving the path from main_filepath.

    Returns (alpha_filepath_or_None, status_note).
    """
    # Determine the alpha pixel data
    if frame_mask is not None:
        alpha_src = frame_mask if not invert_alpha else (1.0 - frame_mask)
    elif single_to_save.shape[-1] == 4:
        alpha_src = single_to_save[0, ..., 3]   # (H, W)
    else:
        return None, " [save_alpha_separately: no alpha channel]"

    # Try context-aware path first
    alpha_filepath = None
    try:
        from .save_context import resolve_context_path, get_context
        if context_id and get_context(context_id) is not None:
            alpha_filepath, _ = resolve_context_path(
                context_id, suffix + "_alpha", "png",
                node_custom_vars=custom_vars,
            )
    except Exception:
        pass

    if not alpha_filepath:
        alpha_filepath = alpha_file_path(main_filepath)

    try:
        save_alpha_file(alpha_src, alpha_filepath)
        return alpha_filepath, f" + alpha→{alpha_filepath}"
    except Exception as ae:
        return None, f" [alpha save FAILED: {ae}]"
