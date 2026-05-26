"""
BD_ImageToGreyscale — convert an RGB/RGBA image to greyscale.

Modes
-----
luminance        : 0.2126·R + 0.7152·G + 0.0722·B  (BT.709 — modern default for games + AI)
luminance_bt601  : 0.299·R + 0.587·G + 0.114·B    (legacy NTSC weights, for matching old tools)
average          : (R + G + B) / 3
max_channel      : max(R, G, B)
red / green / blue : single channel pass-through

Output modes
------------
image  : 3-channel IMAGE with R=G=B=grey  (for SaveImage, GLSL input, etc.)
mask   : single-channel MASK tensor  (for downstream mask nodes)

Optional mask input + mask_mode:
- apply_within (default): only convert pixels INSIDE the mask to greyscale,
  pixels outside pass through UNCHANGED in original color
- cutout: multiply greyscale × mask, pixels outside are zeroed (black)
"""

import torch
from comfy_api.latest import io


_MODES = ["luminance", "luminance_bt601", "average", "max_channel", "red", "green", "blue"]
_WEIGHTS_BT709 = torch.tensor([0.2126, 0.7152, 0.0722])  # modern default — sRGB / Rec.709
_WEIGHTS_BT601 = torch.tensor([0.299, 0.587, 0.114])     # legacy NTSC, for backwards compat

_MASK_MODES = ["apply_within", "cutout"]


class BD_ImageToGreyscale(io.ComfyNode):
    """Convert an IMAGE to greyscale using selectable luma mode."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_ImageToGreyscale",
            display_name="BD Image To Greyscale",
            category="🧠BrainDead/Segmentation",
            description=(
                "Convert an RGB/RGBA image to greyscale.\n\n"
                "luminance (default): BT.709 / sRGB — 0.2126·R + 0.7152·G + 0.0722·B. "
                "Modern standard, matches what Unity/Unreal use, perceptually accurate for "
                "modern displays and AI workflows.\n"
                "luminance_bt601: legacy NTSC — 0.299·R + 0.587·G + 0.114·B. "
                "Matches older Photoshop 'Desaturate' and legacy print tools.\n"
                "average: (R+G+B)/3.\n"
                "max_channel: max(R,G,B).\n"
                "red/green/blue: single channel.\n\n"
                "rgb output: 3-channel IMAGE (R=G=B=luma, alpha=1) — "
                "plug into SaveImage, GLSL uniforms, etc.\n"
                "mask output: single-channel MASK tensor."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("mode", options=_MODES, default="luminance"),
                io.Combo.Input("mask_mode", options=_MASK_MODES, default="apply_within",
                               tooltip="How the optional mask input affects the output (only matters when mask is wired):\n"
                                       "apply_within (default): Only convert pixels inside the mask to greyscale. "
                                       "Pixels outside the mask pass through UNCHANGED from the source (keeps original color). "
                                       "Use when you want to greyscale only a region (e.g. only the skin).\n"
                                       "cutout: Multiply greyscale by mask. Pixels inside mask show grey, pixels outside "
                                       "are zeroed out (black). Use for cutout-style processing where you want to "
                                       "isolate a region and discard the rest."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Optional mask. Behavior controlled by mask_mode above. "
                                      "Soft edges in the mask give a natural feather between modified and untouched regions."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, mode: str = "luminance",
                mask_mode: str = "apply_within",
                mask: torch.Tensor | None = None) -> io.NodeOutput:
        # image: (B, H, W, C) — C is 3 or 4
        img = image[..., :3].float()  # drop alpha if present, work in float

        if mode == "luminance":
            w = _WEIGHTS_BT709.to(img.device)
            grey = (img * w).sum(dim=-1, keepdim=True)   # (B, H, W, 1)
        elif mode == "luminance_bt601":
            w = _WEIGHTS_BT601.to(img.device)
            grey = (img * w).sum(dim=-1, keepdim=True)
        elif mode == "average":
            grey = img.mean(dim=-1, keepdim=True)
        elif mode == "max_channel":
            grey = img.max(dim=-1, keepdim=True).values
        elif mode == "red":
            grey = img[..., 0:1]
        elif mode == "green":
            grey = img[..., 1:2]
        elif mode == "blue":
            grey = img[..., 2:3]
        else:
            grey = (img * _WEIGHTS_BT709.to(img.device)).sum(dim=-1, keepdim=True)

        grey = grey.clamp(0.0, 1.0)

        # Expand grey to 3-channel for IMAGE output (R=G=B=luma)
        rgb_out = grey.expand(-1, -1, -1, 3)           # (B, H, W, 3)

        # Apply optional mask
        if mask is not None:
            m = mask.float().to(img.device)
            if m.dim() == 2:
                m = m.unsqueeze(0)           # (1, H, W)
            m_3 = m.unsqueeze(-1)            # (B, H, W, 1)

            if mask_mode == "apply_within":
                # Greyscale only inside mask, original color outside.
                # Soft mask values give natural feather between converted and original.
                rgb_out = rgb_out * m_3 + img * (1.0 - m_3)
                # The mask output is the greyscale value × mask (1 inside mask, faded outside)
                # so downstream nodes see "where the conversion was applied" weighted by intensity.
                grey = grey * m_3
            else:  # "cutout" — multiply (legacy behavior)
                rgb_out = rgb_out * m_3
                grey = grey * m_3

        # mask output: squeeze to (B, H, W)
        mask_out = grey.squeeze(-1)                    # (B, H, W)
        if mask_out.shape[0] == 1:
            mask_out = mask_out.squeeze(0)             # (H, W) for single image

        return io.NodeOutput(rgb_out, mask_out)


IMAGE_TO_GREYSCALE_V3_NODES = [BD_ImageToGreyscale]

IMAGE_TO_GREYSCALE_NODES = {"BD_ImageToGreyscale": BD_ImageToGreyscale}
IMAGE_TO_GREYSCALE_DISPLAY_NAMES = {"BD_ImageToGreyscale": "BD Image To Greyscale"}
