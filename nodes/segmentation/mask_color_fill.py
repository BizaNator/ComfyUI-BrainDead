"""
BD_MaskColorFill — fill up to 4 mask regions with flat colors.

Each slot gets an independent R, G, B color. Masks are layered bottom-to-top
(slot 1 on bottom, slot 4 on top). Overlapping regions are painted by the
higher-numbered slot.

RGBA output uses the union of all active masks as the alpha channel so the
result composites cleanly over other layers (background is transparent by
default).

Typical use: plug BD_FaceSocketInfill mask outputs into slots to build a
flat-color UV texture map for 2D flipbook animation.
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


def _to_hw(mask_t: torch.Tensor, b: int, H: int, W: int) -> np.ndarray:
    """Extract frame b from a mask tensor and return float32 (H, W)."""
    m = mask_t
    if m.ndim == 2:
        m_np = m.cpu().numpy().astype(np.float32)
    else:
        idx = min(b, m.shape[0] - 1)
        m_np = m[idx].cpu().numpy().astype(np.float32)
    if m_np.shape != (H, W) and HAS_CV2:
        m_np = cv2.resize(m_np, (W, H), interpolation=cv2.INTER_LINEAR)
    return m_np.clip(0.0, 1.0)


def _process_mask(m_np: np.ndarray, expand: int, feather: int) -> np.ndarray:
    """Dilate then edge-feather a float32 (H,W) mask.

    Positive feather: blurs outward from the edge — interior stays fully
    opaque (1.0), soft ramp extends BEYOND the fill boundary.
    np.maximum(binary, blurred) preserves the interior while letting the
    Gaussian escape outward, so the fill region shows no opacity loss.

    Negative feather: blurs inward from the edge — exterior stays 0,
    interior fades toward the boundary. np.minimum(binary, blurred).

    Zero feather: binary pass-through.
    """
    if not HAS_CV2:
        return m_np
    u8 = (m_np * 255.0).clip(0, 255).astype(np.uint8)
    if expand > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * expand + 1, 2 * expand + 1))
        u8 = cv2.dilate(u8, k)
    if feather > 0:
        inner = u8.copy()
        ksize = feather * 2 + 1
        blurred = cv2.GaussianBlur(u8, (ksize, ksize), feather * 0.5)
        u8 = np.maximum(inner, blurred)   # interior stays 255, edge escapes outward
    elif feather < 0:
        abs_fth = abs(feather)
        ksize = abs_fth * 2 + 1
        blurred = cv2.GaussianBlur(u8, (ksize, ksize), abs_fth * 0.5)
        u8 = np.minimum(u8, blurred)      # exterior stays 0, edge blends inward
    return (u8.astype(np.float32) / 255.0).clip(0.0, 1.0)


class BD_MaskColorFill(io.ComfyNode):
    """
    Fill up to 4 mask regions with flat colors and composite into one image.

    Slots are layered 1 (bottom) → 4 (top).  The RGBA output uses the union
    of all active mask slots as its alpha channel so the node output composites
    cleanly over other layers.  Set bg_alpha=1 if you want a fully opaque
    background in the RGBA output instead.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        def _slot(n: int, r: int, g: int, b: int) -> list:
            return [
                io.Mask.Input(f"mask_{n}", optional=True,
                              tooltip=f"Mask slot {n}. Leave unconnected to skip."),
                io.Int.Input(f"r_{n}", default=r, min=0, max=255, step=1, optional=True,
                             tooltip=f"Red channel for slot {n}."),
                io.Int.Input(f"g_{n}", default=g, min=0, max=255, step=1, optional=True,
                             tooltip=f"Green channel for slot {n}."),
                io.Int.Input(f"b_{n}", default=b, min=0, max=255, step=1, optional=True,
                             tooltip=f"Blue channel for slot {n}."),
                io.Int.Input(f"expand_{n}", default=0, min=0, max=60, step=1, optional=True,
                             tooltip=f"Dilate mask slot {n} by this many pixels before coloring. "
                                     f"Grows the painted region outward."),
                io.Int.Input(f"feather_{n}", default=0, min=-30, max=30, step=1, optional=True,
                             tooltip=f"Edge feather for mask slot {n}. Applied after expand.\n"
                                     f"Positive: soft ramp extends OUTWARD from the fill edge — "
                                     f"interior stays fully opaque, only the boundary bleeds out.\n"
                                     f"Negative: soft ramp goes INWARD — fill fades toward its own edge.\n"
                                     f"0 = hard binary edge."),
            ]

        return io.Schema(
            node_id="BD_MaskColorFill",
            display_name="BD Mask Color Fill",
            category="🧠BrainDead/Segmentation",
            description=(
                "Fill up to 4 mask regions with flat colors and composite them "
                "over an optional background.\n\n"
                "Slots are layered 1 (bottom) → 4 (top). Overlapping regions "
                "are painted by the higher-numbered slot.\n\n"
                "Each slot has independent expand (dilate) and feather (blur) so "
                "you can grow and soften each zone separately.\n\n"
                "bg_alpha_from_image: when ON and a background image is connected, "
                "the background image's luminance drives the RGBA alpha channel — "
                "white areas of the background become opaque, black areas transparent. "
                "Use this when your background is a head/body shape cutout.\n\n"
                "Plug BD_FaceSocketInfill mask outputs here to build flat-color "
                "UV textures for 2D flipbook animation."
            ),
            inputs=[
                io.Image.Input("background", optional=True,
                               tooltip="Base image. If not connected, bg_r/g/b solid color "
                                       "is used as background."),
                io.Int.Input("bg_r", default=0, min=0, max=255, step=1, optional=True,
                             tooltip="Background red (used when no background image)."),
                io.Int.Input("bg_g", default=0, min=0, max=255, step=1, optional=True,
                             tooltip="Background green."),
                io.Int.Input("bg_b", default=0, min=0, max=255, step=1, optional=True,
                             tooltip="Background blue."),
                io.Float.Input("bg_alpha", default=0.0, min=0.0, max=1.0, step=0.01,
                               optional=True,
                               tooltip="Alpha of the background in the RGBA output. "
                                       "0 = transparent background (default). "
                                       "1 = fully opaque. "
                                       "Ignored when bg_alpha_from_image is ON."),
                io.Boolean.Input("bg_alpha_from_image", default=False, optional=True,
                                 tooltip="When ON and a background image is connected, the background "
                                         "image's luminance is used as a CLIP MASK applied AFTER all "
                                         "slot painting. Pixels outside the background shape (luma≈0) "
                                         "become fully transparent regardless of slot masks.\n\n"
                                         "Use bg_alpha to control the non-slot areas inside the shape:\n"
                                         "  bg_alpha=0 → only slot-colored zones are visible (e.g. just "
                                         "eyes/mouth floating on a transparent head silhouette)\n"
                                         "  bg_alpha=1 → whole head shape is opaque, slot zones painted "
                                         "on top showing the background image everywhere else\n\n"
                                         "Typical use: background = head silhouette (white head, black "
                                         "outside), bg_alpha_from_image=ON, bg_alpha=0 → eye/mouth zones "
                                         "visible on transparent background, clipped to head boundary."),
                *_slot(1, 255,   0,   0),
                *_slot(2,   0, 255,   0),
                *_slot(3,   0,   0, 255),
                *_slot(4, 255, 255,   0),
            ],
            outputs=[
                io.Image.Output("image", tooltip="RGB composite."),
                io.Image.Output("rgba",
                                tooltip="RGBA — filled regions opaque, background transparent "
                                        "(unless bg_alpha > 0)."),
                io.Mask.Output("mask",  tooltip="Union of all active mask slots."),
                io.String.Output("status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        background: torch.Tensor | None = None,
        bg_r: int = 0, bg_g: int = 0, bg_b: int = 0,
        bg_alpha: float = 0.0,
        bg_alpha_from_image: bool = False,
        mask_1: torch.Tensor | None = None,
        r_1: int = 255, g_1: int = 0,   b_1: int = 0,
        expand_1: int = 0, feather_1: int = 0,
        mask_2: torch.Tensor | None = None,
        r_2: int = 0,   g_2: int = 255, b_2: int = 0,
        expand_2: int = 0, feather_2: int = 0,
        mask_3: torch.Tensor | None = None,
        r_3: int = 0,   g_3: int = 0,   b_3: int = 255,
        expand_3: int = 0, feather_3: int = 0,
        mask_4: torch.Tensor | None = None,
        r_4: int = 255, g_4: int = 255, b_4: int = 0,
        expand_4: int = 0, feather_4: int = 0,
    ) -> io.NodeOutput:

        slots = [
            (mask_1, r_1, g_1, b_1, expand_1, feather_1),
            (mask_2, r_2, g_2, b_2, expand_2, feather_2),
            (mask_3, r_3, g_3, b_3, expand_3, feather_3),
            (mask_4, r_4, g_4, b_4, expand_4, feather_4),
        ]
        active = [(m, r, g, b, ex, fth) for m, r, g, b, ex, fth in slots if m is not None]

        # Resolve canvas dimensions
        if background is not None:
            if background.ndim == 3:
                background = background.unsqueeze(0)
            B, H, W = background.shape[0], background.shape[1], background.shape[2]
        elif active:
            m0 = active[0][0]
            if m0.ndim == 2:
                H, W = m0.shape; B = 1
            else:
                B, H, W = m0.shape[0], m0.shape[1], m0.shape[2]
        else:
            blank_img  = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            blank_rgba = torch.zeros((1, 1, 1, 4), dtype=torch.float32)
            blank_msk  = torch.zeros((1, 1, 1),    dtype=torch.float32)
            return io.NodeOutput(blank_img, blank_rgba, blank_msk,
                                 "BD_MaskColorFill: no masks connected")

        bg_color = np.array([bg_r / 255.0, bg_g / 255.0, bg_b / 255.0], dtype=np.float32)

        images_out: list[torch.Tensor] = []
        rgbas_out:  list[torch.Tensor] = []
        masks_out:  list[torch.Tensor] = []

        for b in range(B):
            if background is not None:
                canvas = background[b].cpu().numpy().astype(np.float32)[..., :3].copy()
            else:
                canvas = np.full((H, W, 3), bg_color, dtype=np.float32)

            # Pre-compute bg luma for clip BEFORE painting changes the canvas
            if bg_alpha_from_image and background is not None:
                bg_luma = (0.2126 * canvas[..., 0]
                           + 0.7152 * canvas[..., 1]
                           + 0.0722 * canvas[..., 2]).clip(0.0, 1.0)
            else:
                bg_luma = None

            combined_alpha = np.full((H, W), bg_alpha, dtype=np.float32)

            for mask_t, r, g, bv, ex, fth in active:
                m_np  = _to_hw(mask_t, b, H, W)
                m_np  = _process_mask(m_np, ex, fth)
                color = np.array([r / 255.0, g / 255.0, bv / 255.0], dtype=np.float32)
                a3    = m_np[:, :, np.newaxis]
                canvas = canvas * (1.0 - a3) + color * a3
                np.maximum(combined_alpha, m_np, out=combined_alpha)

            # bg_alpha_from_image: clip combined_alpha by background luma AFTER
            # all slots are painted. Pixels outside the background shape (luma≈0)
            # become transparent even if a mask bled there. Inside the shape the
            # slot masks determine opacity — use bg_alpha=0 to see only slot zones,
            # bg_alpha=1 to keep the whole head opaque showing background pixels.
            if bg_luma is not None:
                combined_alpha *= bg_luma

            canvas         = canvas.clip(0.0, 1.0)
            combined_alpha = combined_alpha.clip(0.0, 1.0)
            rgba           = np.concatenate(
                [canvas, combined_alpha[:, :, np.newaxis]], axis=-1
            ).astype(np.float32)

            images_out.append(torch.from_numpy(canvas))
            rgbas_out.append(torch.from_numpy(rgba))
            masks_out.append(torch.from_numpy(combined_alpha))

        n_active  = len(active)
        bg_alpha_desc = "luma" if (bg_alpha_from_image and background is not None) else f"{bg_alpha:.2f}"
        status    = (f"BD_MaskColorFill: {B} frame(s), {n_active}/{len(slots)} slots active, "
                     f"bg_alpha={bg_alpha_desc}")
        print(f"[BD_MaskColorFill] {status}", flush=True)

        return io.NodeOutput(
            torch.stack(images_out),
            torch.stack(rgbas_out),
            torch.stack(masks_out),
            status,
        )


MASK_COLOR_FILL_V3_NODES      = [BD_MaskColorFill]
MASK_COLOR_FILL_NODES         = {"BD_MaskColorFill": BD_MaskColorFill}
MASK_COLOR_FILL_DISPLAY_NAMES = {"BD_MaskColorFill": "BD Mask Color Fill"}
