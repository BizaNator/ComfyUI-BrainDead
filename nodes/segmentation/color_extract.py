"""
BD_ColorExtract — extract a color (with threshold) from an image as a mask.

Common use: painted shadow extraction. You paint shadows in blue (or any color),
this node pulls all blue-tinted pixels out as a mask that can feed into BD_PackChannels
or any downstream node that wants a single-channel "where is this color" map.

Two match modes:
- hue (default): matches by color identity, ignoring brightness. "Any blue"
  whether dark or bright, saturated or muted, contributes to the mask.
  Best for painted-color extraction (blue shadows on a white base, etc.).
- rgb_distance: matches by RGB Euclidean distance. Strict, exact-color match.
  Good when you want only pixels close to a specific RGB triplet.

Output is a continuous greyscale mask in [0, 1]. Optional soft_edge for falloff,
gradient_mode for a luma-aware ramp like BD_LuminanceMask, and an optional input
mask to restrict the calculation to a region (e.g. skin only).
"""

import math
import numpy as np
import torch

from comfy_api.latest import io


_MATCH_MODES = ["hue", "rgb_distance"]


def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor (..., 3) in [0, 1] to HSV (..., 3). H is in [0, 1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc, _ = torch.max(rgb, dim=-1)
    minc, _ = torch.min(rgb, dim=-1)
    v = maxc
    delta = maxc - minc
    # Saturation
    s = torch.where(maxc > 0, delta / maxc.clamp(min=1e-6), torch.zeros_like(maxc))
    # Hue
    safe_delta = delta.clamp(min=1e-6)
    rc = (maxc - r) / safe_delta
    gc = (maxc - g) / safe_delta
    bc = (maxc - b) / safe_delta
    h = torch.zeros_like(maxc)
    h = torch.where(r == maxc, bc - gc, h)
    h = torch.where(g == maxc, 2.0 + rc - bc, h)
    h = torch.where(b == maxc, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h = torch.where(delta == 0, torch.zeros_like(h), h)
    return torch.stack([h, s, v], dim=-1)


def _mask_to_bhw(m):
    if m is None:
        return None
    if m.ndim == 4:
        return m.squeeze(0) if m.shape[0] == 1 else m[..., 0]
    if m.ndim == 2:
        return m.unsqueeze(0)
    return m


def _smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    t = ((x - edge0) / max(edge1 - edge0, 1e-6)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class BD_ColorExtract(io.ComfyNode):
    """Extract a color (with threshold) from an image as a greyscale mask."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_ColorExtract",
            display_name="BD Color Extract",
            category="🧠BrainDead/Segmentation",
            description=(
                "Extract pixels matching a target color from an image as a greyscale mask. "
                "Pulls out all pixels close to the target color (within tolerance) and outputs "
                "a mask where matching pixels are white. Use hue mode for color-identity matching "
                "(any shade of the color regardless of brightness — perfect for painted shadow "
                "extraction). Use rgb_distance for strict RGB triplet matching."
            ),
            inputs=[
                io.Image.Input("image",
                               tooltip="Source image to extract color from."),
                io.Float.Input("target_r", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Target color RED component (0-1). Default 0.0 (for blue)."),
                io.Float.Input("target_g", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Target color GREEN component. Default 0.0 (for blue)."),
                io.Float.Input("target_b", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Target color BLUE component. Default 1.0 (for blue)."),
                io.Combo.Input("match_mode", options=_MATCH_MODES, default="hue",
                               tooltip="hue (default): matches by color identity in HSV. Brightness-independent — "
                                       "'any blue' regardless of how dark/light. Best for painted color extraction. "
                                       "rgb_distance: strict RGB Euclidean distance match. Use when you want only "
                                       "pixels very close to a specific RGB triplet."),
                io.Float.Input("tolerance", default=0.15, min=0.0, max=1.0, step=0.01,
                               tooltip="How loose the match is. In hue mode: tolerance in hue units "
                                       "(0.15 ≈ ±54° of hue). In rgb_distance mode: max distance in RGB space "
                                       "(0.15 ≈ pixels within 0.15 unit sphere of target). Lower = stricter."),
                io.Float.Input("min_saturation", default=0.10, min=0.0, max=1.0, step=0.01,
                               tooltip="HUE MODE ONLY — pixels with saturation below this are ignored (treated as "
                                       "non-matching). Prevents greys/near-whites from accidentally matching the "
                                       "target hue when their hue calc is unreliable. Set to 0 to match desaturated "
                                       "pixels too. Ignored in rgb_distance mode."),
                io.Float.Input("soft_edge", default=0.05, min=0.0, max=0.5, step=0.005,
                               tooltip="Smoothstep falloff around the tolerance boundary. 0 = hard binary cutoff. "
                                       "Larger = softer mask edge."),
                io.Boolean.Input("gradient_mode", default=False,
                                 tooltip="When ON: output is a smooth gradient based on HOW CLOSE each pixel is "
                                         "to the target color. Pixel at target = 1.0, far pixel = 0.0, smooth fade. "
                                         "Ignores soft_edge. Use for 'how much blue tint does this pixel have' style "
                                         "extraction. OFF: thresholded binary-ish mask with soft_edge."),
                io.Boolean.Input("invert", default=False,
                                 tooltip="Invert the output mask (1 - mask). Use to extract everything EXCEPT "
                                         "the target color."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Optional region mask. When provided, output is zero outside this region "
                                      "(restricts extraction to e.g. skin areas)."),
            ],
            outputs=[
                io.Image.Output(display_name="color_mask",
                                tooltip="Greyscale mask (3-channel image, R=G=B=mask_value). White where pixels "
                                        "match the target color, black elsewhere."),
                io.Mask.Output(display_name="mask",
                               tooltip="Single-channel MASK version for nodes expecting MASK type."),
            ],
        )

    @classmethod
    def execute(cls, image, target_r=0.0, target_g=0.0, target_b=1.0,
                match_mode="hue", tolerance=0.15, min_saturation=0.10,
                soft_edge=0.05, gradient_mode=False, invert=False,
                mask=None) -> io.NodeOutput:

        img = image if image.ndim == 4 else image.unsqueeze(0)
        img = img[..., :3].float()  # drop alpha if present
        b, h, w, _ = img.shape

        target_rgb = torch.tensor([target_r, target_g, target_b],
                                  dtype=img.dtype, device=img.device)

        if match_mode == "hue":
            # Convert source and target to HSV
            hsv = _rgb_to_hsv(img)
            target_hsv = _rgb_to_hsv(target_rgb.view(1, 1, 1, 3))

            # Hue distance (circular)
            hue_diff = (hsv[..., 0] - target_hsv[..., 0, 0, 0, 0]).abs()
            hue_diff = torch.min(hue_diff, 1.0 - hue_diff)  # wrap around at 0/1

            sat = hsv[..., 1]

            if gradient_mode:
                # Smooth gradient: pixel at target hue with full saturation = 1, fades with distance
                hue_match = (1.0 - (hue_diff / max(tolerance, 1e-6)).clamp(0.0, 1.0))
                # Modulate by saturation (grey pixels can't strongly match)
                color_mask = hue_match * sat
                # Apply saturation floor
                color_mask = torch.where(sat < min_saturation,
                                         torch.zeros_like(color_mask), color_mask)
            else:
                # Binary-ish: pixel within tolerance AND above min_saturation
                if soft_edge > 0.0:
                    hue_passes = 1.0 - _smoothstep(tolerance - soft_edge, tolerance + soft_edge, hue_diff)
                    sat_passes = _smoothstep(min_saturation - soft_edge, min_saturation + soft_edge, sat)
                else:
                    hue_passes = (hue_diff <= tolerance).float()
                    sat_passes = (sat >= min_saturation).float()
                color_mask = hue_passes * sat_passes

        else:  # rgb_distance
            # Euclidean distance in RGB space
            diff = img - target_rgb
            dist = torch.sqrt((diff * diff).sum(dim=-1))
            # Max distance possible is sqrt(3) ≈ 1.732, but we use [0,1] tolerance
            # so normalize tolerance to RGB space (1.0 tolerance = max distance)
            tol_rgb = tolerance * math.sqrt(3.0)

            if gradient_mode:
                color_mask = (1.0 - (dist / max(tol_rgb, 1e-6)).clamp(0.0, 1.0))
            else:
                if soft_edge > 0.0:
                    soft_rgb = soft_edge * math.sqrt(3.0)
                    color_mask = 1.0 - _smoothstep(tol_rgb - soft_rgb, tol_rgb + soft_rgb, dist)
                else:
                    color_mask = (dist <= tol_rgb).float()

        # Apply optional region mask
        roi = _mask_to_bhw(mask)
        if roi is not None:
            roi = roi.to(color_mask.device).float()
            if roi.shape[0] != b:
                roi = roi.expand(b, -1, -1) if roi.shape[0] == 1 else roi[:b]
            if roi.shape[-2:] != color_mask.shape[-2:]:
                roi = torch.nn.functional.interpolate(
                    roi.unsqueeze(1), size=color_mask.shape[-2:],
                    mode="bilinear", align_corners=False
                ).squeeze(1)
            color_mask = color_mask * roi

        # Invert
        if invert:
            color_mask = 1.0 - color_mask
            if roi is not None:
                color_mask = color_mask * roi  # keep zero outside ROI even when inverted

        color_mask = color_mask.clamp(0.0, 1.0)

        # IMAGE output (3-channel greyscale)
        image_out = color_mask.unsqueeze(-1).expand(-1, -1, -1, 3).clone()

        # MASK output (single-channel)
        mask_out = color_mask.squeeze(0) if color_mask.shape[0] == 1 else color_mask

        # Diagnostic
        match_pct = float((color_mask > 0.5).float().mean() * 100)
        print(f"[BD_ColorExtract] target_rgb=({target_r:.2f},{target_g:.2f},{target_b:.2f}), "
              f"mode={match_mode}, tolerance={tolerance:.3f}, min_sat={min_saturation:.2f}, "
              f"gradient={'on' if gradient_mode else 'off'}, invert={'yes' if invert else 'no'}, "
              f"mask_active={'yes' if mask is not None else 'no'}, "
              f"matched pixels (>0.5): {match_pct:.1f}%")

        return io.NodeOutput(image_out, mask_out)


COLOR_EXTRACT_V3_NODES = [BD_ColorExtract]
COLOR_EXTRACT_NODES = {"BD_ColorExtract": BD_ColorExtract}
COLOR_EXTRACT_DISPLAY_NAMES = {"BD_ColorExtract": "BD Color Extract"}
