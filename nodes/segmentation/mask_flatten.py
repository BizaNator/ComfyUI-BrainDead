"""
BD_MaskFlatten — flatten an image (with optional alpha/mask) onto a plain background.

Output is always RGB (no alpha) so downstream nodes don't have to handle
transparency. Supports solid color backgrounds (white/black/RGB chroma keys),
arbitrary hex, optional background_image for composite-on-scene workflows,
and channel-routing modes that pack the image into a single RGB channel
(useful for game-engine packed-channel workflows or multi-pass debugging).

Standard alpha composite math: result = image * alpha + bg * (1 - alpha).
Semi-transparent pixels become bg-blended, so removing the bg later (chroma
key) recovers percentage-based opacity.
"""

import re

import torch

from comfy_api.latest import io


_NAMED_COLORS = {
    "white":  (1.0, 1.0, 1.0),
    "black":  (0.0, 0.0, 0.0),
    "red":    (1.0, 0.0, 0.0),
    "green":  (0.0, 1.0, 0.0),
    "blue":   (0.0, 0.0, 1.0),
    "magenta":(1.0, 0.0, 1.0),
    "cyan":   (0.0, 1.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
}

_LUMA_WEIGHTS = torch.tensor([0.299, 0.587, 0.114])


def _parse_hex_color(hex_str: str) -> tuple[float, float, float]:
    """Parse '#RRGGBB' or 'RRGGBB' into (r, g, b) floats in [0, 1]."""
    s = hex_str.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", s):
        return (0.5, 0.5, 0.5)
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return (r, g, b)


def _resolve_bg_color(name: str, hex_str: str) -> tuple[float, float, float]:
    if name == "custom":
        return _parse_hex_color(hex_str)
    return _NAMED_COLORS.get(name, (1.0, 1.0, 1.0))


def _build_checker(b: int, h: int, w: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    check = 32
    yy = torch.arange(h, device=device).view(1, h, 1, 1)
    xx = torch.arange(w, device=device).view(1, 1, w, 1)
    pattern = (((yy // check) + (xx // check)) % 2).to(dtype)
    bg = (pattern * 0.4 + 0.5).expand(b, h, w, 3).contiguous()
    return bg


def edge_pad_rgb(rgb: torch.Tensor, foreground_binary: torch.Tensor,
                 max_pixels: int = 0) -> torch.Tensor:
    """Voronoi edge padding — extend foreground RGB into transparent background.

    For each background pixel, find nearest foreground pixel and copy its RGB.
    Standard game-engine "alpha bleed" / "color spread" technique to prevent
    UV-edge halo artifacts when textures are sampled at silhouette boundaries.

    rgb: (B, H, W, 3) float [0,1]
    foreground_binary: (B, H, W) bool — True = foreground
    max_pixels: 0 = unlimited (pad everywhere); >0 = limit pad distance
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    out = rgb.clone()
    rgb_np = rgb.detach().cpu().numpy()
    fg_np = foreground_binary.detach().cpu().numpy()
    for bi in range(rgb.shape[0]):
        fg = fg_np[bi]
        if not fg.any():
            continue
        distances, indices = distance_transform_edt(~fg, return_indices=True)
        padded = rgb_np[bi][indices[0], indices[1]]
        if max_pixels > 0:
            within = distances <= max_pixels
            padded = np.where(within[..., None], padded, rgb_np[bi])
        out[bi] = torch.from_numpy(padded).to(out.device).to(out.dtype)
    return out


def _resize_to(t: torch.Tensor, h: int, w: int, mode: str = "bilinear") -> torch.Tensor:
    """Resize tensor to (h, w). Handles (B,H,W,C) IMAGE or (B,H,W) MASK."""
    if t.ndim == 4 and t.shape[1] == h and t.shape[2] == w:
        return t
    if t.ndim == 3 and t.shape[1] == h and t.shape[2] == w:
        return t
    if t.ndim == 4:
        nchw = t.permute(0, 3, 1, 2)
        resized = torch.nn.functional.interpolate(nchw, size=(h, w), mode=mode,
                                                  align_corners=False if mode == "bilinear" else None)
        return resized.permute(0, 2, 3, 1)
    if t.ndim == 3:
        nchw = t.unsqueeze(1)
        resized = torch.nn.functional.interpolate(nchw, size=(h, w), mode=mode,
                                                  align_corners=False if mode == "bilinear" else None)
        return resized.squeeze(1)
    return t


class BD_MaskFlatten(io.ComfyNode):
    """Flatten image + optional mask onto a plain background; outputs RGB always."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_MaskFlatten",
            display_name="BD Mask Flatten",
            category="🧠BrainDead/Segmentation",
            description=(
                "Flatten an image (RGB or RGBA) plus optional alpha mask onto a plain "
                "background or another image. Output is always RGB (no alpha) so downstream "
                "nodes don't need to handle transparency. Standard alpha composite math: "
                "image × alpha + bg × (1 - alpha) — semi-transparent pixels become "
                "bg-blended, so removing the bg later via chroma key gives percentage-based "
                "recovery. Channel-routing modes pack the image's grayscale into a single "
                "RGB channel (useful for game-engine packed maps)."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input(
                    "mask", optional=True,
                    tooltip="Optional alpha mask. OVERRIDES image's alpha channel if both present. "
                            "If not provided AND image is RGBA, uses image's alpha. "
                            "If not provided AND image is RGB, treats as fully opaque (alpha=1)."
                ),
                io.Image.Input(
                    "background_image", optional=True,
                    tooltip="Optional image to composite ONTO instead of a solid color. When provided, "
                            "background_color and background_color_hex are ignored. Resized to match the "
                            "input image's dimensions."
                ),
                io.Combo.Input(
                    "background_color",
                    options=["white", "black", "red", "green", "blue", "magenta", "cyan", "yellow", "checker", "custom"],
                    default="white", optional=True,
                    tooltip="Solid color background. red/green/blue/magenta = chroma keys. checker = "
                            "gray checkerboard for debug visualization. custom = use background_color_hex."
                ),
                io.String.Input(
                    "background_color_hex", default="#808080", optional=True,
                    tooltip="Hex color when background_color='custom'. Format: '#RRGGBB' or '#RGB'. "
                            "Falls back to gray if invalid."
                ),
                io.Combo.Input(
                    "flatten_mode",
                    options=["alpha_composite", "grayscale", "image_to_red", "image_to_green", "image_to_blue"],
                    default="alpha_composite", optional=True,
                    tooltip=(
                        "How to flatten:\n"
                        "  alpha_composite — standard alpha blend image RGB onto background.\n"
                        "  grayscale — convert image to luminance, alpha-composite onto background.\n"
                        "  image_to_red — put image's grayscale into ONLY the red channel "
                        "(green/blue stay as bg). Useful for packing multiple grayscale maps "
                        "into one RGB image for game engines.\n"
                        "  image_to_green / image_to_blue — same but for those channels."
                    ),
                ),
                io.Float.Input(
                    "mask_strength", default=1.0, min=0.0, max=2.0, step=0.05, optional=True,
                    tooltip="Multiply alpha values before composite. >1 makes alpha stronger (less bg "
                            "shows through), <1 weaker (more bg shows through). Clamped to [0, 1] after."
                ),
                io.Boolean.Input(
                    "invert_mask", default=False, optional=True,
                    tooltip="Invert the alpha before applying — flattens the OUTSIDE of the mask onto "
                            "background instead of the inside."
                ),
                io.Float.Input(
                    "mask_threshold", default=0.0, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="If > 0, binarize the alpha at this threshold before composite. 0 = soft "
                            "(continuous alpha), >0 = hard binary cutoff. Useful when you want sharp "
                            "edges instead of partial transparency on the flatten."
                ),
                io.Int.Input(
                    "edge_pad_pixels", default=0, min=0, max=512, step=4, optional=True,
                    tooltip="Voronoi edge padding — extend foreground RGB outward into transparent "
                            "areas by N pixels using nearest-foreground color. Standard game-engine "
                            "'alpha bleed' to prevent UV-edge halos when textures are sampled at "
                            "silhouette boundaries. 0 = off (no padding). 16-32 = typical for game "
                            "textures. Applied BEFORE flattening, so the flatten background only "
                            "shows beyond the padded zone."
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="alpha_used"),
            ],
        )

    @classmethod
    def execute(cls, image, mask=None, background_image=None,
                background_color="white", background_color_hex="#808080",
                flatten_mode="alpha_composite", mask_strength=1.0,
                invert_mask=False, mask_threshold=0.0,
                edge_pad_pixels=0) -> io.NodeOutput:

        img = image if image.ndim == 4 else image.unsqueeze(0)
        b, h, w, c = img.shape

        if mask is not None:
            alpha = mask
            if alpha.ndim == 4:
                alpha = alpha.squeeze(0) if alpha.shape[0] == 1 else alpha[..., 0]
            if alpha.ndim == 2:
                alpha = alpha.unsqueeze(0)
            alpha = _resize_to(alpha, h, w, mode="bilinear")
        elif c == 4:
            alpha = img[..., 3].clone()
        else:
            alpha = torch.ones((b, h, w), dtype=img.dtype, device=img.device)

        if invert_mask:
            alpha = 1.0 - alpha
        if mask_threshold > 0.0:
            alpha = (alpha >= mask_threshold).to(img.dtype)
        alpha = (alpha * float(mask_strength)).clamp(0.0, 1.0)

        img_rgb = img[..., :3].contiguous()

        if edge_pad_pixels > 0:
            fg_binary = (alpha > 0.5)
            img_rgb = edge_pad_rgb(img_rgb, fg_binary, max_pixels=int(edge_pad_pixels))

        if background_image is not None:
            bg = background_image if background_image.ndim == 4 else background_image.unsqueeze(0)
            bg = _resize_to(bg[..., :3], h, w, mode="bilinear")
            if bg.shape[0] != b:
                bg = bg.expand(b, -1, -1, -1) if bg.shape[0] == 1 else bg[:b]
            bg = bg.to(img.device).to(img.dtype)
        elif background_color == "checker":
            bg = _build_checker(b, h, w, img.dtype, img.device)
        else:
            r, g, bl = _resolve_bg_color(background_color, background_color_hex)
            bg = torch.zeros((b, h, w, 3), dtype=img.dtype, device=img.device)
            bg[..., 0] = r
            bg[..., 1] = g
            bg[..., 2] = bl

        alpha_b = alpha.unsqueeze(-1)

        if flatten_mode == "alpha_composite":
            result = img_rgb * alpha_b + bg * (1.0 - alpha_b)
        elif flatten_mode == "grayscale":
            weights = _LUMA_WEIGHTS.to(img.device).to(img.dtype)
            gray = (img_rgb * weights).sum(dim=-1, keepdim=True).expand(-1, -1, -1, 3)
            result = gray * alpha_b + bg * (1.0 - alpha_b)
        elif flatten_mode in ("image_to_red", "image_to_green", "image_to_blue"):
            weights = _LUMA_WEIGHTS.to(img.device).to(img.dtype)
            gray = (img_rgb * weights).sum(dim=-1)
            result = bg.clone()
            ch = {"image_to_red": 0, "image_to_green": 1, "image_to_blue": 2}[flatten_mode]
            result[..., ch] = gray * alpha + bg[..., ch] * (1.0 - alpha)
        else:
            result = img_rgb * alpha_b + bg * (1.0 - alpha_b)

        result = result.clamp(0.0, 1.0)
        return io.NodeOutput(result, alpha)


MASK_FLATTEN_V3_NODES = [BD_MaskFlatten]
MASK_FLATTEN_NODES = {"BD_MaskFlatten": BD_MaskFlatten}
MASK_FLATTEN_DISPLAY_NAMES = {"BD_MaskFlatten": "BD Mask Flatten"}
