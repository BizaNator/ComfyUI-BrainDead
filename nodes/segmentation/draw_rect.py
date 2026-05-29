"""
BD_DrawRect — draw a flat-color rectangle (rounded corners, edge feather) over an image.

Coordinates are normalized 0–1 so the rect scales with any resolution.
Feather uses the same edge-start convention as BD_MaskColorFill and BD_FaceSocketInfill:
  positive = ramp expands OUTWARD from the edge (interior stays fully opaque)
  negative = ramp bleeds INWARD from the edge (exterior stays 0)

Typical use: grey-out the mouth area before feeding into a shader that needs a
neutral mid-grey target, or mask off a region for compositing.
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


def _rounded_rect_u8(H: int, W: int,
                     x1: int, y1: int, x2: int, y2: int,
                     r: int) -> np.ndarray:
    """Draw a filled rounded rectangle into a uint8 (H, W) mask."""
    mask = np.zeros((H, W), dtype=np.uint8)
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    if r <= 0:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    else:
        # Two strips to fill the body
        cv2.rectangle(mask, (x1 + r, y1), (x2 - r, y2), 255, -1)
        cv2.rectangle(mask, (x1, y1 + r), (x2, y2 - r), 255, -1)
        # Four quarter-circle corners
        cv2.circle(mask, (x1 + r, y1 + r), r, 255, -1)   # top-left
        cv2.circle(mask, (x2 - r, y1 + r), r, 255, -1)   # top-right
        cv2.circle(mask, (x1 + r, y2 - r), r, 255, -1)   # bottom-left
        cv2.circle(mask, (x2 - r, y2 - r), r, 255, -1)   # bottom-right
    return mask


def _feather(mask_u8: np.ndarray, feather: int) -> np.ndarray:
    """Apply edge feather. Positive=outward, negative=inward. Same convention as BD_MaskColorFill."""
    if feather == 0 or not HAS_CV2:
        return mask_u8.astype(np.float32) / 255.0
    abs_f = abs(feather)
    ksize = abs_f * 2 + 1
    m_f = mask_u8.astype(np.float32)
    blurred = cv2.GaussianBlur(m_f, (ksize, ksize), abs_f * 0.5)
    result = np.maximum(m_f, blurred) if feather > 0 else np.minimum(m_f, blurred)
    return (result / 255.0).clip(0.0, 1.0)


class BD_DrawRect(io.ComfyNode):
    """Draw a flat-color rectangle with optional rounded corners and edge feather."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_DrawRect",
            display_name="BD Draw Rect",
            category="🧠BrainDead/Segmentation",
            description=(
                "Draw a flat-color filled rectangle (with optional rounded corners and "
                "edge feather) over an image or onto a blank canvas.\n\n"
                "Coordinates are normalized 0–1 (x_center=0.5, y_center=0.5 = image center).\n\n"
                "Feather starts AT the edge and expands outward (positive) or inward (negative) "
                "— interior stays fully opaque regardless of feather amount.\n\n"
                "Typical use: lay a neutral mid-grey (128, 128, 128) block over the mouth "
                "before feeding into a skin shader that expects a neutral target there."
            ),
            inputs=[
                io.Image.Input("image", optional=True,
                               tooltip="Base image to draw onto. If not connected, a blank canvas "
                                       "is created using canvas_width × canvas_height (default 512×512)."),

                # ── Position / size (normalized 0–1) ──────────────────────────
                io.Float.Input("x_center", default=0.5, min=0.0, max=1.0, step=0.005,
                               tooltip="Horizontal center of the rectangle (0=left edge, 1=right edge)."),
                io.Float.Input("y_center", default=0.5, min=0.0, max=1.0, step=0.005,
                               tooltip="Vertical center of the rectangle (0=top edge, 1=bottom edge)."),
                io.Float.Input("rect_width", default=0.2, min=0.0, max=2.0, step=0.005,
                               tooltip="Width as a fraction of the image width. "
                                       ">1.0 allowed to extend beyond the image edge."),
                io.Float.Input("rect_height", default=0.1, min=0.0, max=2.0, step=0.005,
                               tooltip="Height as a fraction of the image height. "
                                       ">1.0 allowed to extend beyond the image edge."),

                # ── Shape ─────────────────────────────────────────────────────
                io.Int.Input("corner_radius", default=0, min=0, max=400, step=1,
                             tooltip="Rounded corner radius in pixels (at native resolution). "
                                     "0 = sharp square corners. Clamped to half the shortest side."),
                io.Int.Input("feather", default=0, min=-200, max=200, step=1,
                             tooltip="Edge feather radius in pixels.\n"
                                     "Positive: ramp expands OUTWARD from the edge — "
                                     "interior stays fully opaque (1.0).\n"
                                     "Negative: ramp bleeds INWARD — fill fades toward its edge.\n"
                                     "0 = hard binary edge."),

                # ── Fill color ────────────────────────────────────────────────
                io.Int.Input("fill_r", default=128, min=0, max=255, step=1,
                             tooltip="Fill red channel. Default 128 = mid-grey."),
                io.Int.Input("fill_g", default=128, min=0, max=255, step=1,
                             tooltip="Fill green channel."),
                io.Int.Input("fill_b", default=128, min=0, max=255, step=1,
                             tooltip="Fill blue channel."),
                io.Float.Input("opacity", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Blend strength of the rectangle over the base image. "
                                       "1.0 = fully opaque fill. 0.5 = 50% mix."),

                # ── Canvas fallback (no image input) ──────────────────────────
                io.Int.Input("canvas_width", default=512, min=1, max=8192, step=1, optional=True,
                             tooltip="Canvas width when no image is connected."),
                io.Int.Input("canvas_height", default=512, min=1, max=8192, step=1, optional=True,
                             tooltip="Canvas height when no image is connected."),
            ],
            outputs=[
                io.Image.Output("image",
                                tooltip="Input image (or blank canvas) with the rectangle composited over it."),
                io.Mask.Output("mask",
                               tooltip="Rectangle mask (feathered). 1.0 inside the rect, "
                                       "0.0 outside. Use as a blend mask for other nodes."),
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
        image: torch.Tensor | None = None,
        canvas_width: int = 512,
        canvas_height: int = 512,
    ) -> io.NodeOutput:

        if not HAS_CV2:
            raise RuntimeError("BD_DrawRect requires opencv-python.")

        # Resolve canvas
        if image is not None:
            img = image if image.ndim == 4 else image.unsqueeze(0)
            img = img.float()
            B, H, W, C = img.shape
        else:
            H, W = canvas_height, canvas_width
            B, C = 1, 3
            img = torch.zeros((1, H, W, 3), dtype=torch.float32)

        # Pixel-space rect bounds (clamped to image)
        cx = x_center * W
        cy = y_center * H
        hw = rect_width  * W / 2.0
        hh = rect_height * H / 2.0
        x1 = max(0, int(round(cx - hw)))
        y1 = max(0, int(round(cy - hh)))
        x2 = min(W, int(round(cx + hw)))
        y2 = min(H, int(round(cy + hh)))

        fill_color = np.array([fill_r / 255.0, fill_g / 255.0, fill_b / 255.0],
                               dtype=np.float32)

        rect_u8  = _rounded_rect_u8(H, W, x1, y1, x2, y2, corner_radius)
        rect_f32 = _feather(rect_u8, feather)   # (H, W) float32 [0, 1]

        out_imgs: list[torch.Tensor] = []
        out_masks: list[torch.Tensor] = []

        for b in range(B):
            frame = img[b].cpu().numpy()  # (H, W, C)
            base_rgb = frame[..., :3].astype(np.float32)

            w = rect_f32[:, :, np.newaxis] * opacity   # (H, W, 1) blend weight
            composited = (base_rgb * (1.0 - w) + fill_color * w).clip(0.0, 1.0)

            if C == 4:
                composited = np.concatenate([composited, frame[..., 3:4]], axis=-1)

            out_imgs.append(torch.from_numpy(composited))
            out_masks.append(torch.from_numpy(rect_f32))

        out_image = torch.stack(out_imgs,  dim=0)
        out_mask  = torch.stack(out_masks, dim=0)

        print(f"[BD_DrawRect] rect px=({x1},{y1})→({x2},{y2}) "
              f"color=({fill_r},{fill_g},{fill_b}) opacity={opacity:.2f} "
              f"corner_r={corner_radius} feather={feather:+d}")

        return io.NodeOutput(out_image, out_mask)


DRAW_RECT_V3_NODES      = [BD_DrawRect]
DRAW_RECT_NODES         = {"BD_DrawRect": BD_DrawRect}
DRAW_RECT_DISPLAY_NAMES = {"BD_DrawRect": "BD Draw Rect"}
