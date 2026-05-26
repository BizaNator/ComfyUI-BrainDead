"""
BD_UnpackChannels — split an RGBA image into its 4 channels.

Inverse of BD_PackChannels. Takes one packed image and outputs:
- R, G, B, A as separate greyscale IMAGE outputs (replicated across RGB so they
  feed cleanly into any downstream IMAGE input)
- R, G, B, A as separate MASK outputs (single-channel, for nodes expecting MASK)
- debug_preview: 2x2 grid showing each channel tinted in its own color
  (R as red, G as green, B as blue, A as white)

Common use: pull apart the u_image3 RGBA pack used by the skin shader to verify
which channel has which data, or to feed individual channels into other nodes.
"""

import torch
from comfy_api.latest import io


def _build_unpack_debug_preview(r: torch.Tensor, g: torch.Tensor,
                                b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """2x2 grid: R as red, G as green, B as blue, A as white (matches BD_PackChannels)."""
    zeros = torch.zeros_like(r)
    rg = torch.stack([r, zeros, zeros], dim=-1)
    gg = torch.stack([zeros, g, zeros], dim=-1)
    bg = torch.stack([zeros, zeros, b], dim=-1)
    ag = torch.stack([a, a, a], dim=-1)
    top = torch.cat([rg, gg], dim=2)
    bot = torch.cat([bg, ag], dim=2)
    return torch.cat([top, bot], dim=1).clamp(0.0, 1.0)


class BD_UnpackChannels(io.ComfyNode):
    """Split RGBA image into 4 individual channels (IMAGE + MASK outputs + debug preview)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_UnpackChannels",
            display_name="BD Unpack Channels",
            category="🧠BrainDead/Segmentation",
            description=(
                "Inverse of BD_PackChannels. Splits one RGBA image into 4 individual "
                "channels. Each channel is output as both IMAGE (greyscale, 3-channel) "
                "and MASK (single-channel). Also produces a debug_preview showing all "
                "4 channels in a 2x2 grid, each tinted in its own color (R=red, G=green, "
                "B=blue, A=white) — matches BD_PackChannels debug_preview layout."
            ),
            inputs=[
                io.Image.Input("image",
                               tooltip="Source image — RGB or RGBA. If RGB, alpha output is "
                                       "the value from alpha_default."),
                io.Float.Input("alpha_default", default=1.0, min=0.0, max=1.0, step=0.05, optional=True,
                               tooltip="Value used for the alpha output when the input image is "
                                       "RGB (only 3 channels). Default 1.0 = fully opaque."),
            ],
            outputs=[
                io.Image.Output(display_name="red_image",
                                tooltip="R channel as greyscale 3-channel image (R replicated to RGB)."),
                io.Image.Output(display_name="green_image",
                                tooltip="G channel as greyscale 3-channel image."),
                io.Image.Output(display_name="blue_image",
                                tooltip="B channel as greyscale 3-channel image."),
                io.Image.Output(display_name="alpha_image",
                                tooltip="A channel as greyscale 3-channel image."),
                io.Mask.Output(display_name="red_mask",
                               tooltip="R channel as single-channel MASK."),
                io.Mask.Output(display_name="green_mask",
                               tooltip="G channel as single-channel MASK."),
                io.Mask.Output(display_name="blue_mask",
                               tooltip="B channel as single-channel MASK."),
                io.Mask.Output(display_name="alpha_mask",
                               tooltip="A channel as single-channel MASK."),
                io.Image.Output(display_name="debug_preview",
                                tooltip="2x2 grid: R top-left tinted red, G top-right green, "
                                        "B bottom-left blue, A bottom-right white."),
            ],
        )

    @classmethod
    def execute(cls, image, alpha_default=1.0) -> io.NodeOutput:
        img = image if image.ndim == 4 else image.unsqueeze(0)
        img = img.float()

        b, h, w, nchan = img.shape

        r = img[..., 0]
        g = img[..., 1] if nchan >= 2 else img[..., 0]
        bl = img[..., 2] if nchan >= 3 else img[..., 0]
        if nchan >= 4:
            a = img[..., 3]
        else:
            a = torch.full((b, h, w), float(alpha_default), dtype=img.dtype, device=img.device)

        # IMAGE outputs: replicate the single channel across RGB
        def grey3(ch: torch.Tensor) -> torch.Tensor:
            return ch.unsqueeze(-1).expand(-1, -1, -1, 3).clamp(0.0, 1.0)

        r_img = grey3(r)
        g_img = grey3(g)
        b_img = grey3(bl)
        a_img = grey3(a)

        # Debug preview: same layout/colors as BD_PackChannels
        debug_preview = _build_unpack_debug_preview(r, g, bl, a)

        # Diagnostic — match BD_PackChannels output format
        print(f"[BD_UnpackChannels] mean values: "
              f"R={float(r.mean()):.3f}, G={float(g.mean()):.3f}, "
              f"B={float(bl.mean()):.3f}, A={float(a.mean()):.3f}, "
              f"input_channels={nchan}")

        return io.NodeOutput(r_img, g_img, b_img, a_img, r, g, bl, a, debug_preview)


UNPACK_CHANNELS_V3_NODES = [BD_UnpackChannels]
UNPACK_CHANNELS_NODES = {"BD_UnpackChannels": BD_UnpackChannels}
UNPACK_CHANNELS_DISPLAY_NAMES = {"BD_UnpackChannels": "BD Unpack Channels"}
