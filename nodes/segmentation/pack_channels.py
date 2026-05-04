"""
BD_PackChannels — combine up to 4 source images/masks into the RGBA channels of one output.

Inverse of channel splitting. Common use cases:
- Game engine packed textures: roughness in R, metalness in G, AO in B, height in A
- Normal map repackaging: separate XYZ channels into RGB
- Combining grayscale outputs from segmentation/depth/normal nodes

Each channel slot accepts EITHER an IMAGE or a MASK input (or both — MASK wins
since it's already single-channel and unambiguous). When IMAGE is provided,
its luminance is computed (0.299 R + 0.587 G + 0.114 B). When neither is
provided, the channel is filled with default_value.

Output is RGB by default; toggle output_alpha to get RGBA (4-channel image).
"""

import torch

from comfy_api.latest import io


_LUMA_WEIGHTS = torch.tensor([0.299, 0.587, 0.114])


def _resize_to(t: torch.Tensor, h: int, w: int, mode: str = "bilinear") -> torch.Tensor:
    if t.ndim == 4 and t.shape[1] == h and t.shape[2] == w:
        return t
    if t.ndim == 3 and t.shape[1] == h and t.shape[2] == w:
        return t
    if t.ndim == 4:
        nchw = t.permute(0, 3, 1, 2)
        out = torch.nn.functional.interpolate(
            nchw, size=(h, w), mode=mode,
            align_corners=False if mode == "bilinear" else None,
        )
        return out.permute(0, 2, 3, 1)
    if t.ndim == 3:
        nchw = t.unsqueeze(1)
        out = torch.nn.functional.interpolate(
            nchw, size=(h, w), mode=mode,
            align_corners=False if mode == "bilinear" else None,
        )
        return out.squeeze(1)
    return t


def _resolve_channel(image: torch.Tensor | None, mask: torch.Tensor | None,
                     h: int, w: int, b: int,
                     default_value: float, invert: bool,
                     dtype: torch.dtype, device: torch.device,
                     image_source: str = "luminance") -> torch.Tensor:
    """Resolve one channel from IMAGE/MASK/default. Returns (B, H, W) float.

    image_source controls how a multi-channel IMAGE input is reduced:
      luminance — 0.299R + 0.587G + 0.114B (default for R/G/B output slots)
      red, green, blue — pick that channel directly
      alpha — use the alpha channel if image is RGBA, else fall back to luminance
              (default for ALPHA output slot)
    """
    if mask is not None:
        ch = mask
        if ch.ndim == 4:
            ch = ch.squeeze(0) if ch.shape[0] == 1 else ch[..., 0]
        if ch.ndim == 2:
            ch = ch.unsqueeze(0)
        ch = _resize_to(ch.float(), h, w, mode="bilinear")
    elif image is not None:
        img = image if image.ndim == 4 else image.unsqueeze(0)
        img = _resize_to(img.float(), h, w, mode="bilinear")
        nchan = img.shape[-1]
        if image_source == "red":
            ch = img[..., 0]
        elif image_source == "green":
            ch = img[..., 1] if nchan >= 2 else img[..., 0]
        elif image_source == "blue":
            ch = img[..., 2] if nchan >= 3 else img[..., 0]
        elif image_source == "alpha":
            if nchan >= 4:
                ch = img[..., 3]
            else:
                weights = _LUMA_WEIGHTS.to(img.device).to(img.dtype)
                ch = (img[..., :3] * weights).sum(dim=-1)
        else:
            weights = _LUMA_WEIGHTS.to(img.device).to(img.dtype)
            ch = (img[..., :3] * weights).sum(dim=-1)
    else:
        ch = torch.full((b, h, w), float(default_value), dtype=dtype, device=device)

    if ch.shape[0] != b:
        ch = ch.expand(b, -1, -1) if ch.shape[0] == 1 else ch[:b]
    if invert:
        ch = 1.0 - ch
    return ch.clamp(0.0, 1.0).to(dtype).to(device)


def _infer_size(sources: list, override_w: int, override_h: int) -> tuple[int, int, int]:
    """Pick (B, H, W) from the first non-None source, with optional overrides."""
    for s in sources:
        if s is None:
            continue
        if s.ndim == 4:
            b, h, w = s.shape[0], s.shape[1], s.shape[2]
        elif s.ndim == 3:
            b, h, w = s.shape[0], s.shape[1], s.shape[2]
        elif s.ndim == 2:
            b, h, w = 1, s.shape[0], s.shape[1]
        else:
            continue
        if override_w > 0:
            w = override_w
        if override_h > 0:
            h = override_h
        return b, h, w
    return 1, max(override_h, 64), max(override_w, 64)


class BD_PackChannels(io.ComfyNode):
    """Pack up to 4 IMAGE/MASK sources into the RGBA channels of one output image."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_PackChannels",
            display_name="BD Pack Channels",
            category="🧠BrainDead/Segmentation",
            description=(
                "Combine up to 4 source images/masks into the RGBA channels of one output. "
                "Each channel slot accepts an IMAGE (auto-converted to luminance grayscale) "
                "or a MASK (used directly). When both wired for a slot, MASK wins. When "
                "neither is wired, the channel is filled with the channel's default_value. "
                "Toggle output_alpha to get a 4-channel RGBA image; default is RGB only."
            ),
            inputs=[
                io.Image.Input("red_image", optional=True,
                               tooltip="IMAGE source for the RED channel — its grayscale luminance becomes R."),
                io.Mask.Input("red_mask", optional=True,
                              tooltip="MASK source for the RED channel (wins over red_image if both provided)."),
                io.Float.Input("red_default", default=0.0, min=0.0, max=1.0, step=0.05, optional=True,
                               tooltip="Value to fill RED with when neither red_image nor red_mask is wired."),
                io.Boolean.Input("red_invert", default=False, optional=True,
                                 tooltip="Invert the resolved RED channel before packing."),

                io.Image.Input("green_image", optional=True,
                               tooltip="IMAGE source for the GREEN channel — grayscale becomes G."),
                io.Mask.Input("green_mask", optional=True),
                io.Float.Input("green_default", default=0.0, min=0.0, max=1.0, step=0.05, optional=True),
                io.Boolean.Input("green_invert", default=False, optional=True),

                io.Image.Input("blue_image", optional=True,
                               tooltip="IMAGE source for the BLUE channel — grayscale becomes B."),
                io.Mask.Input("blue_mask", optional=True),
                io.Float.Input("blue_default", default=0.0, min=0.0, max=1.0, step=0.05, optional=True),
                io.Boolean.Input("blue_invert", default=False, optional=True),

                io.Boolean.Input("output_alpha", default=False, optional=True,
                                 tooltip="If True, output is 4-channel RGBA instead of RGB. Use the alpha_image/"
                                         "alpha_mask/alpha_default inputs to set the A channel."),
                io.Image.Input("alpha_image", optional=True,
                               tooltip="IMAGE source for the ALPHA channel. If RGBA, the alpha channel is used "
                                       "directly. If RGB, luminance is computed. Wiring this auto-enables output_alpha."),
                io.Mask.Input("alpha_mask", optional=True,
                              tooltip="MASK source for the ALPHA channel (wins over alpha_image). "
                                      "Wiring this auto-enables output_alpha."),
                io.Float.Input("alpha_default", default=1.0, min=0.0, max=1.0, step=0.05, optional=True,
                               tooltip="Value to fill ALPHA with when no alpha source is wired (default 1.0 = opaque)."),
                io.Boolean.Input("alpha_invert", default=False, optional=True),

                io.Int.Input("width_override", default=0, min=0, max=8192, step=64, optional=True,
                             tooltip="Override output width. 0 = use first non-None source's width."),
                io.Int.Input("height_override", default=0, min=0, max=8192, step=64, optional=True,
                             tooltip="Override output height. 0 = use first non-None source's height. "
                                     "All channels are resized to this dimension."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="alpha"),
            ],
        )

    @classmethod
    def execute(cls, red_image=None, red_mask=None, red_default=0.0, red_invert=False,
                green_image=None, green_mask=None, green_default=0.0, green_invert=False,
                blue_image=None, blue_mask=None, blue_default=0.0, blue_invert=False,
                output_alpha=False,
                alpha_image=None, alpha_mask=None, alpha_default=1.0, alpha_invert=False,
                width_override=0, height_override=0) -> io.NodeOutput:

        sources = [red_image, red_mask, green_image, green_mask, blue_image, blue_mask,
                   alpha_image, alpha_mask]
        b, h, w = _infer_size(sources, width_override, height_override)

        device = torch.device("cpu")
        dtype = torch.float32
        for s in sources:
            if s is not None:
                device = s.device
                dtype = s.dtype if s.dtype.is_floating_point else torch.float32
                break

        r = _resolve_channel(red_image, red_mask, h, w, b, red_default, red_invert,
                             dtype, device, image_source="luminance")
        g = _resolve_channel(green_image, green_mask, h, w, b, green_default, green_invert,
                             dtype, device, image_source="luminance")
        bl = _resolve_channel(blue_image, blue_mask, h, w, b, blue_default, blue_invert,
                              dtype, device, image_source="luminance")
        a = _resolve_channel(alpha_image, alpha_mask, h, w, b, alpha_default, alpha_invert,
                             dtype, device, image_source="alpha")

        alpha_was_wired = (alpha_image is not None) or (alpha_mask is not None)
        effective_output_alpha = bool(output_alpha) or alpha_was_wired

        if effective_output_alpha:
            stacked = torch.stack([r, g, bl, a], dim=-1)
        else:
            stacked = torch.stack([r, g, bl], dim=-1)

        return io.NodeOutput(stacked, a)


PACK_CHANNELS_V3_NODES = [BD_PackChannels]
PACK_CHANNELS_NODES = {"BD_PackChannels": BD_PackChannels}
PACK_CHANNELS_DISPLAY_NAMES = {"BD_PackChannels": "BD Pack Channels"}
