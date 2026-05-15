"""
BD_UVConfidenceBlend — composite N partial UV textures into one.

Each view contributes pixels weighted by its confidence (the view-cosine
mask emitted by BD_FaceTextureBake). Optional seam dilation prevents
black-bleed at UV-island boundaries when the result is sampled by a 3D
renderer with bilinear filtering.

This is the input for the Qwen-inpaint finalize step. Wire BD_FaceTextureBake's
`uv_layout_mask` into `target_mask` here and feed `inpaint_mask` to Qwen as
the paint region — it explicitly covers gaps no view managed to fill, so
Qwen knows to redraw missing sections (not just blend covered ones).
"""

import numpy as np
import torch

from comfy_api.latest import io


# 8-neighbour offsets, ordered so axis-aligned neighbours take priority over
# diagonals (axis-aligned copies look slightly cleaner along straight seams).
_EDGE_EXTEND_OFFSETS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),       # N, S, W, E
    (-1, -1), (-1, 1), (1, -1), (1, 1),     # NW, NE, SW, SE
]


def _edge_extend_texture(image: torch.Tensor, mask: torch.Tensor,
                         radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Grow the filled region by `radius` pixels via non-blurring edge extend.

    Unlike an averaging dilation, each newly-filled pixel COPIES the value
    of one filled neighbour rather than averaging all of them — so the
    boundary is extended without softening color gradients. This matters
    because the dilated ring is sampled by downstream 3D renderers; an
    averaged ring reads as a blurry halo at UV seams.

    image: (H, W, 3) float
    mask:  (H, W) bool — True where image has valid pixels
    Returns (extended_image, extended_mask).
    """
    if radius <= 0:
        return image, mask

    img = image.clone()
    m = mask.clone()
    h, w = m.shape

    for _ in range(radius):
        if m.all():
            break
        empty = ~m
        if not empty.any():
            break
        # `filled_this_iter` guards each empty pixel so it's filled by exactly
        # one neighbour (the first direction in priority order that has data).
        newly_filled = torch.zeros_like(m)
        for dy, dx in _EDGE_EXTEND_OFFSETS:
            # Shift mask + image so that [y, x] sees neighbour [y+dy, x+dx]
            shifted_m = torch.roll(m, shifts=(dy, dx), dims=(0, 1))
            shifted_img = torch.roll(img, shifts=(dy, dx), dims=(0, 1))
            # Kill wrap-around contributions at the rolled edges
            if dy > 0:
                shifted_m[:dy, :] = False
            elif dy < 0:
                shifted_m[dy:, :] = False
            if dx > 0:
                shifted_m[:, :dx] = False
            elif dx < 0:
                shifted_m[:, dx:] = False

            can_fill = empty & shifted_m & ~newly_filled
            if can_fill.any():
                img = torch.where(can_fill.unsqueeze(-1), shifted_img, img)
                newly_filled = newly_filled | can_fill

        m = m | newly_filled

    return img, m


class BD_UVConfidenceBlend(io.ComfyNode):
    """Confidence-weighted blend of N partial UV textures."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_UVConfidenceBlend",
            display_name="BD UV Confidence Blend",
            category="🧠BrainDead/FaceWrap",
            description=(
                "Composite multiple BD_FaceTextureBake outputs into one UV\n"
                "texture, weighted per-pixel by their confidence masks.\n\n"
                "Outputs:\n"
                "- uv_texture: the composite\n"
                "- confidence: combined per-pixel confidence (max across views)\n"
                "- filled_mask: binary — where the composite has data\n"
                "- inpaint_mask: binary — where Qwen SHOULD paint. If\n"
                "  target_mask is wired, this is target & ~filled (gaps inside\n"
                "  the UV layout). Otherwise it falls back to ~filled (every\n"
                "  unfilled pixel including background)."
            ),
            inputs=[
                io.Image.Input(
                    "uv_textures",
                    tooltip="Stack of BD_FaceTextureBake uv_texture outputs (N,H,W,3).",
                ),
                io.Mask.Input(
                    "confidences",
                    tooltip="Stack of BD_FaceTextureBake confidence outputs (N,H,W).",
                ),
                io.Int.Input(
                    "seam_dilate",
                    default=4,
                    min=0,
                    max=32,
                    step=1,
                    tooltip="Grow the filled region by this many pixels via "
                            "non-blurring edge extend (each new pixel copies a "
                            "filled neighbour, no averaging) to prevent "
                            "black-bleed at UV seams under bilinear sampling. "
                            "0 disables.",
                ),
                io.Float.Input(
                    "fill_threshold",
                    default=0.05,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip="Pixels with summed confidence below this are treated "
                            "as gaps in filled_mask.",
                ),
                io.Mask.Input(
                    "target_mask",
                    optional=True,
                    tooltip="Full UV-layout silhouette — where the final texture "
                            "SHOULD have data (typically BD_FaceTextureBake's "
                            "uv_layout_mask). Used to compute inpaint_mask = "
                            "target & ~filled, so Qwen explicitly paints gaps "
                            "no view covered. If omitted, inpaint_mask is just "
                            "~filled (background included).",
                ),
                io.Float.Input(
                    "confidence_gamma",
                    default=3.0,
                    min=0.1,
                    max=8.0,
                    step=0.1,
                    optional=True,
                    tooltip="Raise confidences to this power before blending. "
                            ">1 makes the highest-confidence view dominate each "
                            "texel more sharply, which cuts ghosting/blur from "
                            "averaging slightly-misaligned views. 1 = flat "
                            "linear blend (more prone to blur); 3 = default; "
                            "higher = closer to hard winner-take-all.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="uv_texture"),
                io.Mask.Output(display_name="confidence"),
                io.Mask.Output(display_name="filled_mask"),
                io.Mask.Output(display_name="inpaint_mask"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        uv_textures: torch.Tensor,
        confidences: torch.Tensor,
        seam_dilate: int = 4,
        target_mask: torch.Tensor = None,
        fill_threshold: float = 0.05,
        confidence_gamma: float = 3.0,
    ) -> io.NodeOutput:
        z = torch.zeros(1, 1, 1)
        if uv_textures is None or uv_textures.ndim != 4:
            return io.NodeOutput(
                torch.zeros(1, 1, 1, 3), z, z, z,
                "ERROR: uv_textures must be (N,H,W,3)",
            )
        if confidences is None or confidences.ndim != 3:
            return io.NodeOutput(
                torch.zeros(1, 1, 1, 3), z, z, z,
                "ERROR: confidences must be (N,H,W)",
            )
        if uv_textures.shape[0] != confidences.shape[0]:
            return io.NodeOutput(
                torch.zeros(1, 1, 1, 3), z, z, z,
                f"ERROR: batch mismatch: {uv_textures.shape[0]} textures vs "
                f"{confidences.shape[0]} confidence maps",
            )
        if uv_textures.shape[1:3] != confidences.shape[1:3]:
            return io.NodeOutput(
                torch.zeros(1, 1, 1, 3), z, z, z,
                f"ERROR: spatial size mismatch: textures {tuple(uv_textures.shape[1:3])} "
                f"vs confidences {tuple(confidences.shape[1:3])}",
            )

        device = uv_textures.device
        N, H, W, _ = uv_textures.shape

        textures = uv_textures.float()                 # (N, H, W, 3)
        conf_raw = confidences.float().clamp(min=0)     # (N, H, W) — raw view-cosine

        # gamma sharpens only the BLEND WEIGHTS — where one view is more
        # confident than another, high gamma makes it dominate that texel
        # rather than averaging in the weaker (and likely misaligned) view.
        # The raw confidence is kept separately for the fill decision so
        # gamma can't push otherwise-covered texels below fill_threshold.
        conf_weight = conf_raw.pow(confidence_gamma) if confidence_gamma != 1.0 else conf_raw

        # Weighted sum (gamma'd weights)
        weighted_sum = (textures * conf_weight.unsqueeze(-1)).sum(dim=0)  # (H, W, 3)
        conf_sum = conf_weight.sum(dim=0)                                 # (H, W)
        eps = 1e-8
        composite = weighted_sum / (conf_sum.unsqueeze(-1) + eps)         # (H, W, 3)

        # Fill decision + output confidence use the RAW (pre-gamma) cosine,
        # so "did any view see this texel" is independent of the gamma knob.
        conf_max = conf_raw.max(dim=0).values                            # (H, W)
        filled_mask = conf_max > fill_threshold

        # Zero out unfilled regions in the composite
        composite = composite * filled_mask.float().unsqueeze(-1)

        # Optional seam dilation — non-blurring edge extend into empty neighbours
        if seam_dilate > 0:
            composite, dilated_mask = _edge_extend_texture(composite, filled_mask, seam_dilate)
        else:
            dilated_mask = filled_mask

        n_filled_orig = int(filled_mask.sum().item())
        n_filled_after = int(dilated_mask.sum().item())
        total = H * W

        # inpaint_mask: where Qwen SHOULD paint.
        # With target_mask: target & ~filled (gaps inside the UV layout).
        # Without: ~filled (every unfilled pixel, includes UV background).
        if target_mask is not None and target_mask.numel() > 1:
            tm = target_mask
            if tm.ndim == 3:
                tm = tm[0]
            if tm.shape != (H, W):
                return io.NodeOutput(
                    torch.zeros(1, 1, 1, 3), z, z, z, z,
                    f"ERROR: target_mask shape {tuple(tm.shape)} != texture shape {(H, W)}",
                )
            target_bool = (tm.to(device) > 0.5)
            inpaint_bool = target_bool & ~dilated_mask
            target_coverage = f", target={100.0*int(target_bool.sum().item())/total:.1f}%"
        else:
            inpaint_bool = ~dilated_mask
            target_coverage = " (no target_mask: inpaint=~filled)"
        n_inpaint = int(inpaint_bool.sum().item())

        out_image = composite.unsqueeze(0).cpu()                       # (1, H, W, 3)
        out_confidence = conf_max.unsqueeze(0).cpu()                   # (1, H, W)
        out_filled = dilated_mask.float().unsqueeze(0).cpu()           # (1, H, W)
        out_inpaint = inpaint_bool.float().unsqueeze(0).cpu()          # (1, H, W)

        status = (
            f"blended {N} views | "
            f"texture {H}x{W} | "
            f"filled {100.0*n_filled_orig/total:.1f}% pre-dilate, "
            f"{100.0*n_filled_after/total:.1f}% post-dilate "
            f"(seam_dilate={seam_dilate}) | "
            f"inpaint={100.0*n_inpaint/total:.1f}%{target_coverage}"
        )
        return io.NodeOutput(out_image, out_confidence, out_filled, out_inpaint, status)


FACEWRAP_BLEND_V3_NODES = [BD_UVConfidenceBlend]

FACEWRAP_BLEND_NODES = {
    "BD_UVConfidenceBlend": BD_UVConfidenceBlend,
}

FACEWRAP_BLEND_DISPLAY_NAMES = {
    "BD_UVConfidenceBlend": "BD UV Confidence Blend",
}
