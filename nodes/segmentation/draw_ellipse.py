"""
BD_DrawEllipse — draw a filled ellipse/circle with optional rounded feather.

Two output modes (can be used together):
  1. Color composite — fill_r/g/b blended over the input image (same as BD_DrawRect).
  2. Channel inject  — write the feathered ellipse mask into a specific R/G/B/A channel
     of an existing packed image (e.g. u_image3 RGBA shadow/highlight/dark/line pack).

Feather convention (same as BD_DrawRect / BD_MaskColorFill):
  positive = ramp expands OUTWARD from the ellipse edge (interior stays 1.0)
  negative = ramp bleeds INWARD from the edge
  0 = hard binary edge

Normalized coordinates: x_center=0.5, y_center=0.5 = image center.
radius_x is normalized to image WIDTH, radius_y to image HEIGHT.
For a visually circular result on a square image, set radius_x == radius_y.
"""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

_CHANNELS = ["none", "R", "G", "B", "A"]
_CHAN_IDX  = {"R": 0, "G": 1, "B": 2, "A": 3}


def _ellipse_u8(H: int, W: int,
                cx: int, cy: int,
                rx: int, ry: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    if rx > 0 and ry > 0:
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    return mask


def _feather(mask_u8: np.ndarray, feather: int) -> np.ndarray:
    if feather == 0 or not HAS_CV2:
        return mask_u8.astype(np.float32) / 255.0
    abs_f = abs(feather)
    ksize = abs_f * 2 + 1
    m_f   = mask_u8.astype(np.float32)
    blurred = cv2.GaussianBlur(m_f, (ksize, ksize), abs_f * 0.5)
    result  = np.maximum(m_f, blurred) if feather > 0 else np.minimum(m_f, blurred)
    return (result / 255.0).clip(0.0, 1.0)


class BD_DrawEllipse(io.ComfyNode):
    """Draw a filled ellipse/circle — color composite and/or packed-channel inject."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_DrawEllipse",
            display_name="BD Draw Ellipse",
            category="🧠BrainDead/Segmentation",
            description=(
                "Draw a filled ellipse (or circle) with optional edge feather.\n\n"
                "Two modes — use one or both:\n"
                "  Color: blend fill_r/g/b over the input image (or a blank canvas).\n"
                "  Channel inject: write the feathered mask into a specific R/G/B/A "
                "channel of a packed image (e.g. u_image3 RGBA shadow/highlight/dark/line "
                "pack for skin_shader.glsl).\n\n"
                "radius_x is normalized to image WIDTH, radius_y to HEIGHT. "
                "Set radius_x == radius_y for a visually circular result on square images.\n\n"
                "Feather: positive = expands outward (interior stays 1.0), "
                "negative = bleeds inward, 0 = hard edge."
            ),
            inputs=[
                # ── Base image (for color compositing) ───────────────────────
                io.Image.Input("image", optional=True,
                               tooltip="Base image for color-composite output. If not connected, "
                                       "a blank canvas (canvas_width × canvas_height) is used."),

                # ── Packed image (for channel injection) ──────────────────────
                io.Image.Input("packed_image", optional=True,
                               tooltip="Existing packed RGBA image. When pack_channel is not 'none', "
                                       "the ellipse mask is written into the selected channel and "
                                       "returned as packed_out. Other channels are preserved.\n\n"
                                       "Typical use: u_image3 RGBA pack "
                                       "(R=shadow, G=highlight, B=dark, A=line) for skin_shader.glsl."),
                io.Combo.Input("pack_channel", options=_CHANNELS, default="none",
                               tooltip="Which channel of packed_image to overwrite with the ellipse mask.\n"
                                       "none  = pass packed_image through unchanged.\n"
                                       "R/G/B = write mask into that colour channel.\n"
                                       "A     = write mask into the alpha channel.\n"
                                       "If packed_image has no alpha and A is selected, "
                                       "a new alpha channel is appended."),

                # ── Position / size (normalized 0–1) ─────────────────────────
                io.Float.Input("x_center", default=0.5, min=0.0, max=1.0, step=0.005,
                               tooltip="Horizontal centre (0=left, 1=right)."),
                io.Float.Input("y_center", default=0.5, min=0.0, max=1.0, step=0.005,
                               tooltip="Vertical centre (0=top, 1=bottom)."),
                io.Float.Input("radius_x", default=0.1, min=0.0, max=2.0, step=0.005,
                               tooltip="Horizontal radius as fraction of image width. "
                                       "Set equal to radius_y for a circle (on a square image)."),
                io.Float.Input("radius_y", default=0.1, min=0.0, max=2.0, step=0.005,
                               tooltip="Vertical radius as fraction of image height. "
                                       "Set equal to radius_x for a circle (on a square image)."),

                # ── Edge ─────────────────────────────────────────────────────
                io.Int.Input("feather", default=0, min=-200, max=200, step=1,
                             tooltip="Edge feather in pixels.\n"
                                     "Positive: ramp expands OUTWARD — interior stays fully opaque.\n"
                                     "Negative: ramp bleeds INWARD — fill fades toward its edge.\n"
                                     "0 = hard binary edge."),

                # ── Color composite ───────────────────────────────────────────
                io.Int.Input("fill_r", default=128, min=0, max=255, step=1,
                             tooltip="Fill red. Default 128 = mid-grey."),
                io.Int.Input("fill_g", default=128, min=0, max=255, step=1,
                             tooltip="Fill green."),
                io.Int.Input("fill_b", default=128, min=0, max=255, step=1,
                             tooltip="Fill blue."),
                io.Float.Input("opacity", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Blend strength over the base image. 1.0 = fully opaque fill."),

                # ── Canvas fallback ───────────────────────────────────────────
                io.Int.Input("canvas_width",  default=512, min=1, max=8192, step=1, optional=True,
                             tooltip="Canvas width when no image is connected."),
                io.Int.Input("canvas_height", default=512, min=1, max=8192, step=1, optional=True,
                             tooltip="Canvas height when no image is connected."),
            ],
            outputs=[
                io.Image.Output("image",
                                tooltip="Colour-composite: ellipse blended over image (or blank canvas)."),
                io.Mask.Output("mask",
                               tooltip="Feathered ellipse mask (1.0 inside, 0.0 outside)."),
                io.Image.Output("packed_out",
                                tooltip="packed_image with the ellipse mask written into pack_channel. "
                                        "Pass-through (or blank) when pack_channel='none'."),
            ],
        )

    @classmethod
    def execute(
        cls,
        x_center: float = 0.5,
        y_center: float = 0.5,
        radius_x: float = 0.1,
        radius_y: float = 0.1,
        feather: int = 0,
        fill_r: int = 128,
        fill_g: int = 128,
        fill_b: int = 128,
        opacity: float = 1.0,
        pack_channel: str = "none",
        image: torch.Tensor | None = None,
        packed_image: torch.Tensor | None = None,
        canvas_width: int = 512,
        canvas_height: int = 512,
    ) -> io.NodeOutput:

        if not HAS_CV2:
            raise RuntimeError("BD_DrawEllipse requires opencv-python.")

        # ── Resolve canvas dimensions from the first available source ─────────
        if image is not None:
            img = image if image.ndim == 4 else image.unsqueeze(0)
            img = img.float()
            B, H, W, C = img.shape
        elif packed_image is not None:
            pi = packed_image if packed_image.ndim == 4 else packed_image.unsqueeze(0)
            B, H, W = pi.shape[0], pi.shape[1], pi.shape[2]
            C = 3
            img = torch.zeros((B, H, W, 3), dtype=torch.float32)
        else:
            H, W, B, C = canvas_height, canvas_width, 1, 3
            img = torch.zeros((1, H, W, 3), dtype=torch.float32)

        # ── Geometry (pixel space) ─────────────────────────────────────────────
        cx = int(round(x_center * W))
        cy = int(round(y_center * H))
        rx = max(0, int(round(radius_x * W)))
        ry = max(0, int(round(radius_y * H)))

        ellipse_u8  = _ellipse_u8(H, W, cx, cy, rx, ry)
        ellipse_f32 = _feather(ellipse_u8, feather)   # (H, W) float32

        fill_color = np.array([fill_r / 255.0, fill_g / 255.0, fill_b / 255.0],
                               dtype=np.float32)

        # ── Normalise packed_image ─────────────────────────────────────────────
        pi_t: torch.Tensor | None = None
        if packed_image is not None:
            pi_t = packed_image if packed_image.ndim == 4 else packed_image.unsqueeze(0)
            pi_t = pi_t.float()
            # Resize to match canvas if needed
            if pi_t.shape[1] != H or pi_t.shape[2] != W:
                pi_t = torch.nn.functional.interpolate(
                    pi_t.permute(0, 3, 1, 2),
                    size=(H, W), mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)

        out_imgs:   list[torch.Tensor] = []
        out_masks:  list[torch.Tensor] = []
        out_packed: list[torch.Tensor] = []

        chan_idx = _CHAN_IDX.get(pack_channel, -1)

        for b in range(B):
            # ── Colour composite ───────────────────────────────────────────────
            frame    = img[b].cpu().numpy()
            base_rgb = frame[..., :3].astype(np.float32)
            w        = ellipse_f32[:, :, np.newaxis] * opacity
            composited = (base_rgb * (1.0 - w) + fill_color * w).clip(0.0, 1.0)
            if C == 4:
                composited = np.concatenate([composited, frame[..., 3:4]], axis=-1)
            out_imgs.append(torch.from_numpy(composited))
            out_masks.append(torch.from_numpy(ellipse_f32))

            # ── Channel inject ─────────────────────────────────────────────────
            if pi_t is not None and chan_idx >= 0:
                pi_b = pi_t[min(b, pi_t.shape[0] - 1)].cpu().numpy()   # (H, W, C_pi)
                n_ch = pi_b.shape[-1]

                if chan_idx == 3 and n_ch < 4:
                    # Append alpha channel — expand to 4-channel
                    pi_b = np.concatenate(
                        [pi_b, np.zeros((H, W, 1), dtype=np.float32)], axis=-1
                    )
                    n_ch = 4

                if chan_idx < n_ch:
                    pi_b = pi_b.copy()
                    pi_b[..., chan_idx] = ellipse_f32
                out_packed.append(torch.from_numpy(pi_b))
            elif pi_t is not None:
                # pack_channel == "none" → pass through
                out_packed.append(pi_t[min(b, pi_t.shape[0] - 1)])
            else:
                # No packed_image → blank 1×1 placeholder
                out_packed.append(torch.zeros((1, 1, 3), dtype=torch.float32))

        out_image  = torch.stack(out_imgs,   dim=0)
        out_mask   = torch.stack(out_masks,  dim=0)
        out_packed_t = torch.stack(out_packed, dim=0)

        print(f"[BD_DrawEllipse] centre=({cx},{cy}) radius=({rx},{ry}px) "
              f"feather={feather:+d} color=({fill_r},{fill_g},{fill_b}) "
              f"opacity={opacity:.2f} pack_channel={pack_channel}")

        return io.NodeOutput(out_image, out_mask, out_packed_t)


DRAW_ELLIPSE_V3_NODES      = [BD_DrawEllipse]
DRAW_ELLIPSE_NODES         = {"BD_DrawEllipse": BD_DrawEllipse}
DRAW_ELLIPSE_DISPLAY_NAMES = {"BD_DrawEllipse": "BD Draw Ellipse"}
