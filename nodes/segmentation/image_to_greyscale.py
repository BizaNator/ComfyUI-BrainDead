"""
BD_ImageToGreyscale — convert an RGB/RGBA image to greyscale.

Modes
-----
luminance   : 0.299·R + 0.587·G + 0.114·B  (perceptual, matches GLSL dot)
average     : (R + G + B) / 3
max_channel : max(R, G, B)
red / green / blue : single channel pass-through

Output modes
------------
rgb  : 3-channel IMAGE with R=G=B=grey, alpha=1  (for SaveImage, GLSL input, etc.)
mask : single-channel MASK tensor  (for downstream mask nodes)

Optional alpha_mask input: when wired, multiplies the greyscale by the mask
before output (zeros out background). Does not affect output IMAGE alpha
(always 1.0 for the rgb output).
"""

import torch
from comfy_api.latest import io


_MODES = ["luminance", "average", "max_channel", "red", "green", "blue"]
_WEIGHTS = torch.tensor([0.299, 0.587, 0.114])  # BT.601 luma


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
                "luminance (default): 0.299·R + 0.587·G + 0.114·B — "
                "perceptual, matches GLSL dot(rgb, vec3(0.299,0.587,0.114)).\n"
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
                io.Mask.Input("alpha_mask", optional=True,
                              tooltip="Optional mask to multiply onto greyscale output "
                                      "(zeros out background). Does not affect rgb output alpha."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, mode: str = "luminance",
                alpha_mask: torch.Tensor | None = None) -> io.NodeOutput:
        # image: (B, H, W, C) — C is 3 or 4
        img = image[..., :3].float()  # drop alpha if present, work in float

        if mode == "luminance":
            w = _WEIGHTS.to(img.device)
            grey = (img * w).sum(dim=-1, keepdim=True)   # (B, H, W, 1)
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
            grey = (img * _WEIGHTS.to(img.device)).sum(dim=-1, keepdim=True)

        grey = grey.clamp(0.0, 1.0)

        # Apply optional alpha mask (B, H, W) or (H, W)
        if alpha_mask is not None:
            m = alpha_mask.float().to(img.device)
            if m.dim() == 2:
                m = m.unsqueeze(0)           # (1, H, W)
            m = m.unsqueeze(-1)              # (B, H, W, 1)
            grey = grey * m

        # rgb output: expand single channel to 3 (R=G=B=luma), alpha=1
        rgb_out = grey.expand(-1, -1, -1, 3)           # (B, H, W, 3)

        # mask output: squeeze to (B, H, W)
        mask_out = grey.squeeze(-1)                    # (B, H, W)
        if mask_out.shape[0] == 1:
            mask_out = mask_out.squeeze(0)             # (H, W) for single image

        return io.NodeOutput(rgb_out, mask_out)


IMAGE_TO_GREYSCALE_V3_NODES = [BD_ImageToGreyscale]

IMAGE_TO_GREYSCALE_NODES: dict = {}
IMAGE_TO_GREYSCALE_DISPLAY_NAMES: dict = {}
