"""
BD_DrawEllipse — draw a filled ellipse/circle, or use an existing mask as the shape.

Shape source (pick one):
  mask input       — use any ComfyUI mask (lips from BD_FaceSocketInfill, SAM3 output, etc.)
  mask_from_packed — extract one R/G/B/A channel from packed_image as the shape mask
  geometry         — fallback: compute an ellipse from x_center/y_center/radius_x/radius_y

Feather is applied to whichever shape source is active.

Two output modes (can be used together):
  Color composite  — fill_r/g/b blended over image (or blank canvas).
  Channel inject   — write the final mask into a specific R/G/B/A channel of packed_image.

Feather convention (same as BD_DrawRect / BD_MaskColorFill):
  positive = ramp expands OUTWARD from the edge (interior stays 1.0)
  negative = ramp bleeds INWARD
  0 = hard binary edge
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

_CHANNELS     = ["none", "R", "G", "B", "A"]
_CHAN_IDX     = {"R": 0, "G": 1, "B": 2, "A": 3}


def _ellipse_u8(H: int, W: int, cx: int, cy: int, rx: int, ry: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    if rx > 0 and ry > 0:
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    return mask


def _feather(mask_u8: np.ndarray, feather: int) -> np.ndarray:
    """Edge-start feather: positive=outward, negative=inward, 0=hard."""
    if feather == 0 or not HAS_CV2:
        return mask_u8.astype(np.float32) / 255.0
    abs_f = abs(feather)
    ksize = abs_f * 2 + 1
    m_f   = mask_u8.astype(np.float32)
    blurred = cv2.GaussianBlur(m_f, (ksize, ksize), abs_f * 0.5)
    result  = np.maximum(m_f, blurred) if feather > 0 else np.minimum(m_f, blurred)
    return (result / 255.0).clip(0.0, 1.0)


def _mask_to_bhw(m: torch.Tensor, B: int, H: int, W: int) -> torch.Tensor:
    """Normalise any mask tensor to (B, H, W) float, resized if needed."""
    if m.ndim == 2:
        m = m.unsqueeze(0)          # (1, H, W)
    elif m.ndim == 4:
        m = m[..., 0]               # (B, H, W, 1) → (B, H, W)
    m = m.float()
    if m.shape[0] != B:
        m = m.expand(B, -1, -1) if m.shape[0] == 1 else m[:B]
    if m.shape[-2:] != (H, W):
        m = torch.nn.functional.interpolate(
            m.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1)
    return m


class BD_DrawEllipse(io.ComfyNode):
    """Draw a filled ellipse/circle — or use any mask — for color composite and channel inject."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_DrawEllipse",
            display_name="BD Draw Ellipse",
            category="🧠BrainDead/Segmentation",
            description=(
                "Draw with an ellipse shape OR use any existing mask as the shape.\n\n"
                "Shape priority:\n"
                "  1. mask input — any ComfyUI mask (lips, eyes, SAM3 output, etc.)\n"
                "  2. mask_from_packed — extract one R/G/B/A channel from packed_image as the mask\n"
                "  3. geometry — compute ellipse from x_center / y_center / radius_x / radius_y\n\n"
                "Feather is applied to whichever source is active.\n\n"
                "Two outputs (use one or both):\n"
                "  image — fill_r/g/b blended over image (or blank canvas)\n"
                "  packed_out — packed_image with mask written into pack_channel\n\n"
                "Feather: positive = outward from edge (interior stays 1.0), "
                "negative = inward, 0 = hard edge."
            ),
            inputs=[
                # ── Shape source ─────────────────────────────────────────────
                io.Mask.Input("mask", optional=True,
                              tooltip="Shape mask. When connected, skips ellipse geometry and uses "
                                      "this mask directly (after feather). Accepts any ComfyUI mask "
                                      "— lip mask from BD_FaceSocketInfill, SAM3 output, etc.\n\n"
                                      "Priority: mask > mask_from_packed > geometry."),
                io.Combo.Input("mask_from_packed", options=_CHANNELS, default="none",
                               tooltip="Extract a channel from packed_image and use it as the shape mask.\n"
                                       "Useful when a packed RGBA already contains a useful mask in one "
                                       "channel (e.g. extract the line alpha to use as a draw region).\n"
                                       "Ignored when mask is connected.\n"
                                       "none = fall through to geometry."),

                # ── Base image (color compositing) ────────────────────────────
                io.Image.Input("image", optional=True,
                               tooltip="Base image for color-composite output. If not connected, "
                                       "a blank canvas (canvas_width × canvas_height) is used."),

                # ── Packed image (channel injection + optional mask source) ───
                io.Image.Input("packed_image", optional=True,
                               tooltip="Existing packed RGBA image.\n"
                                       "  pack_channel: write the final mask into this channel.\n"
                                       "  mask_from_packed: extract a channel as the shape source.\n\n"
                                       "Typical use: u_image3 RGBA pack "
                                       "(R=shadow, G=highlight, B=dark, A=line) for skin_shader.glsl."),
                io.Combo.Input("pack_channel", options=_CHANNELS, default="none",
                               tooltip="Write the final (feathered) mask into this channel of packed_image.\n"
                                       "none = pass packed_image through unchanged.\n"
                                       "R/G/B = overwrite that colour channel.\n"
                                       "A = overwrite alpha (channel appended if image has no alpha)."),

                # ── Ellipse geometry (fallback when no mask connected) ────────
                io.Float.Input("x_center", default=0.5, min=0.0, max=1.0, step=0.005,
                               tooltip="Ellipse horizontal centre (0=left, 1=right). "
                                       "Ignored when mask or mask_from_packed is active."),
                io.Float.Input("y_center", default=0.5, min=0.0, max=1.0, step=0.005,
                               tooltip="Ellipse vertical centre (0=top, 1=bottom)."),
                io.Float.Input("radius_x", default=0.1, min=0.0, max=2.0, step=0.005,
                               tooltip="Horizontal radius as fraction of image width. "
                                       "Set == radius_y for a circle on square images."),
                io.Float.Input("radius_y", default=0.1, min=0.0, max=2.0, step=0.005,
                               tooltip="Vertical radius as fraction of image height."),

                # ── Edge ─────────────────────────────────────────────────────
                io.Int.Input("feather", default=0, min=-200, max=200, step=1,
                             tooltip="Edge feather in pixels (applied to any shape source).\n"
                                     "Positive: expands outward — interior stays fully opaque.\n"
                                     "Negative: bleeds inward — fill fades toward its own edge.\n"
                                     "0 = hard binary edge."),

                # ── Color composite ───────────────────────────────────────────
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
                                tooltip="Colour composite: shape blended over image (or blank canvas)."),
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
        radius_x: float = 0.1,
        radius_y: float = 0.1,
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
            raise RuntimeError("BD_DrawEllipse requires opencv-python.")

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

        # ── Resolve shape mask (priority: mask > mask_from_packed > geometry) ──
        shape_source = "geometry"
        shape_masks: torch.Tensor | None = None   # (B, H, W) float [0,1]

        if mask is not None:
            shape_masks  = _mask_to_bhw(mask, B, H, W)
            shape_source = "mask_input"

        elif mask_from_packed != "none" and pi_t is not None:
            ch = _CHAN_IDX.get(mask_from_packed, -1)
            if ch >= 0 and ch < pi_t.shape[-1]:
                shape_masks  = pi_t[..., ch]     # (B, H, W)
                shape_source = f"packed_channel_{mask_from_packed}"

        # If geometry fallback, build the ellipse once (same for all frames)
        if shape_masks is None:
            cx = int(round(x_center * W))
            cy = int(round(y_center * H))
            rx = max(0, int(round(radius_x * W)))
            ry = max(0, int(round(radius_y * H)))
            ell_u8     = _ellipse_u8(H, W, cx, cy, rx, ry)
            ell_f32    = _feather(ell_u8, feather)
            shape_masks = torch.from_numpy(ell_f32).unsqueeze(0).expand(B, -1, -1)
            shape_source = "geometry"

        fill_color = np.array([fill_r / 255.0, fill_g / 255.0, fill_b / 255.0],
                               dtype=np.float32)
        chan_idx   = _CHAN_IDX.get(pack_channel, -1)

        out_imgs:   list[torch.Tensor] = []
        out_masks:  list[torch.Tensor] = []
        out_packed: list[torch.Tensor] = []

        for b in range(B):
            # Per-frame shape mask
            sm_f32 = shape_masks[b].cpu().numpy().astype(np.float32)   # (H, W)

            # Apply feather to mask/channel sources (geometry already feathered above)
            if shape_source != "geometry" and feather != 0:
                sm_u8  = (sm_f32 * 255.0).clip(0, 255).astype(np.uint8)
                sm_f32 = _feather(sm_u8, feather)

            # ── Colour composite ───────────────────────────────────────────────
            frame    = img[b].cpu().numpy()
            base_rgb = frame[..., :3].astype(np.float32)
            w        = sm_f32[:, :, np.newaxis] * opacity
            composited = (base_rgb * (1.0 - w) + fill_color * w).clip(0.0, 1.0)
            if C == 4:
                composited = np.concatenate([composited, frame[..., 3:4]], axis=-1)
            out_imgs.append(torch.from_numpy(composited))
            out_masks.append(torch.from_numpy(sm_f32))

            # ── Channel inject ─────────────────────────────────────────────────
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

        print(f"[BD_DrawEllipse] source={shape_source} feather={feather:+d} "
              f"color=({fill_r},{fill_g},{fill_b}) opacity={opacity:.2f} "
              f"pack_channel={pack_channel}")

        return io.NodeOutput(out_image, out_mask, out_packed_t)


DRAW_ELLIPSE_V3_NODES      = [BD_DrawEllipse]
DRAW_ELLIPSE_NODES         = {"BD_DrawEllipse": BD_DrawEllipse}
DRAW_ELLIPSE_DISPLAY_NAMES = {"BD_DrawEllipse": "BD Draw Ellipse"}
