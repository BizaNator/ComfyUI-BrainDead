"""
Shared luminance-weighting standards for BD nodes.

Use `LUMA_STANDARDS` for the Combo dropdown options and `get_luma_weights(name)`
to fetch the actual weights tensor. Keeping this centralized so all BD nodes
expose the same selector and stay in sync.

References
----------
- BT.709 (Rec.709 / sRGB): modern HD video + sRGB display standard.
  Coefficients reflect the spectral sensitivity of human vision under modern
  display whites. THIS IS THE DEFAULT for games, AI pipelines, Unity/Unreal.

- BT.601 (Rec.601 / NTSC): legacy SD video standard. Photoshop "Desaturate"
  and many older tools still use these weights. Provided for compatibility
  when matching outputs from legacy pipelines.

- average: simple (R+G+B)/3. Not perceptually accurate but useful for
  matching naive averaging tools.
"""

import torch


LUMA_STANDARDS = ["bt709", "bt601", "average"]

_WEIGHTS = {
    "bt709":   torch.tensor([0.2126, 0.7152, 0.0722]),
    "bt601":   torch.tensor([0.2990, 0.5870, 0.1140]),
    "average": torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
}

LUMA_TOOLTIP = (
    "RGB → luminance weighting:\n"
    "bt709 (default): 0.2126 R + 0.7152 G + 0.0722 B — modern sRGB/Rec.709 "
    "standard. Use for Unity/Unreal, AI pipelines, modern display work.\n"
    "bt601: 0.299 R + 0.587 G + 0.114 B — legacy NTSC weights, matches "
    "Photoshop 'Desaturate' and older tools.\n"
    "average: (R+G+B)/3 — simple, not perceptual."
)


def get_luma_weights(name: str = "bt709") -> torch.Tensor:
    """Return the luminance weight tensor for the named standard."""
    return _WEIGHTS.get(name, _WEIGHTS["bt709"]).clone()


def luma_from_rgb(rgb: torch.Tensor, standard: str = "bt709") -> torch.Tensor:
    """Compute luminance from an RGB tensor. Last dim must be 3 (or more — extras ignored)."""
    w = get_luma_weights(standard).to(rgb.device).to(rgb.dtype)
    return (rgb[..., :3] * w).sum(dim=-1)
