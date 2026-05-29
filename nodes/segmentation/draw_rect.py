"""
BD_DrawRect — draw a flat-color rectangle (rounded corners, edge feather) over an image,
or use an existing mask / packed-channel as the shape.

Shape source (priority order):
  1. mask input       — any ComfyUI mask (skips geometry)
  2. mask_from_packed — extract one R/G/B/A channel from packed_image as the shape
  3. geometry         — fallback: compute rect from x_center/y_center/rect_width/rect_height

Feather convention (same as BD_MaskColorFill / BD_DrawEllipse):
  positive = ramp expands OUTWARD from the edge (interior stays fully opaque)
  negative = ramp bleeds INWARD from the edge (exterior stays 0)
  0 = hard binary edge

Also supports channel injection: write the final mask into a specific R/G/B/A channel
of packed_image (e.g. the u_image3 RGBA pack for skin_shader.glsl).
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


def _rounded_rect_u8(H: int, W: int,
                     x1: int, y1: int, x2: int, y2: int,
                     r: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    if r <= 0:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    else:
        cv2.rectangle(mask, (x1 + r, y1), (x2 - r, y2), 255, -1)
        cv2.rectangle(mask, (x1, y1 + r), (x2, y2 - r), 255, -1)
        cv2.circle(mask, (x1 + r, y1 + r), r, 255, -1)
        cv2.circle(mask, (x2 - r, y1 + r), r, 255, -1)
        cv2.circle(mask, (x1 + r, y2 - r), r, 255, -1)
        cv2.circle(mask, (x2 - r, y2 - r), r, 255, -1)
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


def _mask_to_bhw(m: torch.Tensor, B: int, H: int, W: int) -> torch.Tensor:
    if m.ndim == 2:
        m = m.unsqueeze(0)
    elif m.ndim == 4:
        m = m[..., 0]
    m = m.float()
    if m.shape[0] != B:
        m = m.expand(B, -1, -1) if m.shape[0] == 1 else m[:B]
    if m.shape[-2:] != (H, W):
        m = torch.nn.functional.interpolate(
            m.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1)
    return m


class BD_DrawRect(io.ComfyNode):
    """Draw a flat-color rectangle (or use any mask) with optional rounded corners, feather, and channel inject."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_DrawRect",
            display_name="BD Draw Rect",
            category="🧠BrainDead/Segmentation",
            description=(
                "Draw a rectangle shape — or use any existing mask — for color compositing "
                "and/or packed-channel injection.\n\n"
                "Shape priority:\n"
                "  1. mask input — any ComfyUI mask (skips geometry)\n"
                "  2. mask_from_packed — extract one R/G/B/A channel from packed_image\n"
                "  3. geometry — x_center / y_center / rect_width / rect_height / corner_radius\n\n"
                "Feather: positive = outward from edge (interior stays 1.0), "
                "negative = inward, 0 = hard edge.\n\n"
                "pack_channel writes the final mask into a specific channel of packed_image."
            ),
            inputs=[
                # ── Shape source ─────────────────────────────────────────────
                io.Mask.Input("mask", optional=True,
                              tooltip="Shape mask. When connected, skips rect geometry and uses "
                                      "this mask directly (feather still applies).\n"
                                      "Priority: mask > mask_from_packed > geometry."),
                io.Combo.Input("mask_from_packed", options=_CHANNELS, default="none",
                               tooltip="Extract a channel from packed_image and use it as the shape mask.\n"
                                       "Ignored when mask is connected. none = fall through to geometry."),

                # ── Base image ────────────────────────────────────────────────
                io.Image.Input("image", optional=True,
                               tooltip="Base image to composite onto. If not connected, a blank canvas "
                                       "(canvas_width × canvas_height) is used."),

                # ── Packed image ──────────────────────────────────────────────
                io.Image.Input("packed_image", optional=True,
                               tooltip="Existing packed RGBA image.\n"
                                       "  pack_channel: write the final mask into this channel.\n"
                                       "  mask_from_packed: extract a channel as the shape source.\n\n"
                                       "Typical use: u_image3 RGBA pack for skin_shader.glsl."),
                io.Combo.Input("pack_channel", options=_CHANNELS, default="none",
                               tooltip="Write the final feathered mask into this channel of packed_image.\n"
                                       "none = pass packed_image through unchanged.\n"
                                       "R/G/B = overwrite that colour channel.\n"
                                       "A = overwrite alpha (appended if image has no alpha)."),

                # ── Geometry (fallback) ───────────────────────────────────────
                io.Float.Input("x_center", default=0.5, min=0.0, max=1.0, step=0.005,
                               tooltip="Horizontal center (0=left, 1=right). Ignored when mask is connected."),
                io.Float.Input("y_center", default=0.5, min=0.0, max=1.0, step=0.005,
                               tooltip="Vertical center (0=top, 1=bottom)."),
                io.Float.Input("rect_width", default=0.2, min=0.0, max=2.0, step=0.005,
                               tooltip="Width as fraction of image width. >1.0 extends beyond the edge."),
                io.Float.Input("rect_height", default=0.1, min=0.0, max=2.0, step=0.005,
                               tooltip="Height as fraction of image height."),
                io.Int.Input("corner_radius", default=0, min=0, max=400, step=1,
                             tooltip="Rounded corner radius in pixels. 0 = sharp corners. "
                                     "Clamped to half the shortest side."),

                # ── Edge ─────────────────────────────────────────────────────
                io.Int.Input("feather", default=0, min=-200, max=200, step=1,
                             tooltip="Edge feather in pixels (applied to any shape source).\n"
                                     "Positive: expands OUTWARD — interior stays fully opaque.\n"
                                     "Negative: bleeds INWARD — fill fades toward its edge.\n"
                                     "0 = hard binary edge."),

                # ── Fill color ────────────────────────────────────────────────
                io.Int.Input("fill_r", default=128, min=0, max=255, step=1,
                             tooltip="Fill red. Default 128 = mid-grey."),
                io.Int.Input("fill_g", default=128, min=0, max=255, step=1,
                             tooltip="Fill green."),
                io.Int.Input("fill_b", default=128, min=0, max=255, step=1,
                             tooltip="Fill blue."),
                io.Float.Input("opacity", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Blend strength over the base image. 1.0 = fully opaque."),

                # ── Canvas fallback ───────────────────────────────────────────
                io.Int.Input("canvas_width",  default=512, min=1, max=8192, step=1, optional=True,
                             tooltip="Canvas width when no image is connected."),
                io.Int.Input("canvas_height", default=512, min=1, max=8192, step=1, optional=True,
                             tooltip="Canvas height when no image is connected."),
            ],
            outputs=[
                io.Image.Output("image",
                                tooltip="Shape composited over image (or blank canvas)."),
                io.Mask.Output("mask",
                               tooltip="Final feathered shape mask (1.0 inside, 0.0 outside)."),
                io.Image.Output("packed_out",
                                tooltip="packed_image with mask written into pack_channel. "
                                        "Pass-through (or blank) when pack_channel='none'."),
            ],
        )

    @classmethod
    def execute(
        cls,
        x_center: float = 0.5,
        y_center: float = 0.5,
        rect_width: float = 0.2,
        rect_height: float = 0.1,
        corner_radius: int = 0,
        feather: int = 0,
        fill_r: int = 128,
        fill_g: int = 128,
        fill_b: int = 128,
        opacity: float = 1.0,
        pack_channel: str = "none",
        mask_from_packed: str = "none",
        mask: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
        packed_image: torch.Tensor | None = None,
        canvas_width: int = 512,
        canvas_height: int = 512,
    ) -> io.NodeOutput:

        if not HAS_CV2:
            raise RuntimeError("BD_DrawRect requires opencv-python.")

        # ── Resolve canvas dimensions ─────────────────────────────────────────
        if image is not None:
            img = image if image.ndim == 4 else image.unsqueeze(0)
            img = img.float()
            B, H, W, C = img.shape
        elif packed_image is not None:
            pi = packed_image if packed_image.ndim == 4 else packed_image.unsqueeze(0)
            B, H, W, C = pi.shape[0], pi.shape[1], pi.shape[2], 3
            img = torch.zeros((B, H, W, 3), dtype=torch.float32)
        elif mask is not None:
            m = mask if mask.ndim != 2 else mask.unsqueeze(0)
            B = m.shape[0] if m.ndim == 3 else 1
            H, W, C = m.shape[-2], m.shape[-1], 3
            img = torch.zeros((B, H, W, 3), dtype=torch.float32)
        else:
            H, W, B, C = canvas_height, canvas_width, 1, 3
            img = torch.zeros((1, H, W, 3), dtype=torch.float32)

        # ── Normalise packed_image ─────────────────────────────────────────────
        pi_t: torch.Tensor | None = None
        if packed_image is not None:
            pi_t = packed_image if packed_image.ndim == 4 else packed_image.unsqueeze(0)
            pi_t = pi_t.float()
            if pi_t.shape[1] != H or pi_t.shape[2] != W:
                pi_t = torch.nn.functional.interpolate(
                    pi_t.permute(0, 3, 1, 2),
                    size=(H, W), mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)

        # ── Shape source (priority: mask > mask_from_packed > geometry) ────────
        shape_source  = "geometry"
        shape_masks: torch.Tensor | None = None

        if mask is not None:
            shape_masks  = _mask_to_bhw(mask, B, H, W)
            shape_source = "mask_input"

        elif mask_from_packed != "none" and pi_t is not None:
            ch = _CHAN_IDX.get(mask_from_packed, -1)
            if ch >= 0 and ch < pi_t.shape[-1]:
                shape_masks  = pi_t[..., ch]
                shape_source = f"packed_channel_{mask_from_packed}"

        if shape_masks is None:
            cx = x_center * W;  cy = y_center * H
            hw = rect_width * W / 2.0;  hh = rect_height * H / 2.0
            x1 = max(0, int(round(cx - hw)));  y1 = max(0, int(round(cy - hh)))
            x2 = min(W, int(round(cx + hw)));  y2 = min(H, int(round(cy + hh)))
            rect_u8    = _rounded_rect_u8(H, W, x1, y1, x2, y2, corner_radius)
            rect_f32   = _feather(rect_u8, feather)
            shape_masks = torch.from_numpy(rect_f32).unsqueeze(0).expand(B, -1, -1)
            shape_source = "geometry"

        fill_color = np.array([fill_r / 255.0, fill_g / 255.0, fill_b / 255.0],
                               dtype=np.float32)
        chan_idx   = _CHAN_IDX.get(pack_channel, -1)

        out_imgs:   list[torch.Tensor] = []
        out_masks:  list[torch.Tensor] = []
        out_packed: list[torch.Tensor] = []

        for b in range(B):
            sm_f32 = shape_masks[b].cpu().numpy().astype(np.float32)

            if shape_source != "geometry" and feather != 0:
                sm_u8  = (sm_f32 * 255.0).clip(0, 255).astype(np.uint8)
                sm_f32 = _feather(sm_u8, feather)

            frame    = img[b].cpu().numpy()
            base_rgb = frame[..., :3].astype(np.float32)
            w        = sm_f32[:, :, np.newaxis] * opacity
            composited = (base_rgb * (1.0 - w) + fill_color * w).clip(0.0, 1.0)
            if C == 4:
                composited = np.concatenate([composited, frame[..., 3:4]], axis=-1)
            out_imgs.append(torch.from_numpy(composited))
            out_masks.append(torch.from_numpy(sm_f32))

            if pi_t is not None and chan_idx >= 0:
                pi_b = pi_t[min(b, pi_t.shape[0] - 1)].cpu().numpy()
                n_ch = pi_b.shape[-1]
                if chan_idx == 3 and n_ch < 4:
                    pi_b = np.concatenate(
                        [pi_b, np.zeros((H, W, 1), dtype=np.float32)], axis=-1
                    )
                if chan_idx < pi_b.shape[-1]:
                    pi_b = pi_b.copy()
                    pi_b[..., chan_idx] = sm_f32
                out_packed.append(torch.from_numpy(pi_b))
            elif pi_t is not None:
                out_packed.append(pi_t[min(b, pi_t.shape[0] - 1)])
            else:
                out_packed.append(torch.zeros((1, 1, 3), dtype=torch.float32))

        out_image    = torch.stack(out_imgs,   dim=0)
        out_mask     = torch.stack(out_masks,  dim=0)
        out_packed_t = torch.stack(out_packed, dim=0)

        print(f"[BD_DrawRect] source={shape_source} feather={feather:+d} "
              f"color=({fill_r},{fill_g},{fill_b}) opacity={opacity:.2f} "
              f"pack_channel={pack_channel}")

        return io.NodeOutput(out_image, out_mask, out_packed_t)


DRAW_RECT_V3_NODES      = [BD_DrawRect]
DRAW_RECT_NODES         = {"BD_DrawRect": BD_DrawRect}
DRAW_RECT_DISPLAY_NAMES = {"BD_DrawRect": "BD Draw Rect"}
