"""
BD_UVConfidenceBlend — composite N partial UV textures into one.

Each view contributes pixels weighted by its confidence (the view-cosine
mask emitted by BD_FaceTextureBake). Optional seam dilation prevents
black-bleed at UV-island boundaries when the result is sampled by a 3D
renderer with bilinear filtering.

This is the input for the Qwen-inpaint finalize step — the `filled_mask`
output tells Qwen what NOT to overwrite.
"""

import numpy as np
import torch
import torch.nn.functional as F

from comfy_api.latest import io


def _dilate_texture(image: torch.Tensor, mask: torch.Tensor,
                    radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Iteratively grow an image's filled region by `radius` pixels.

    Each iteration: for each unfilled pixel, take the average of its
    filled neighbors (3x3 kernel). Stops early if no more pixels can be
    filled.

    image: (H, W, 3) float
    mask:  (H, W) bool — True where image has valid pixels
    Returns (dilated_image, dilated_mask).
    """
    if radius <= 0:
        return image, mask

    img = image.permute(2, 0, 1).unsqueeze(0).float()       # (1, 3, H, W)
    m = mask.unsqueeze(0).unsqueeze(0).float()              # (1, 1, H, W)

    # Mask the image so unfilled pixels are 0 (so the kernel sees them as 0)
    img_masked = img * m

    kernel = torch.ones((1, 1, 3, 3), device=img.device, dtype=img.dtype)
    for _ in range(radius):
        if m.sum() == m.numel():
            break  # already full
        # Sum of neighbor values for each channel separately
        neighbor_sum_c = F.conv2d(img_masked, kernel.expand(3, 1, 3, 3),
                                  padding=1, groups=3)
        neighbor_count = F.conv2d(m, kernel, padding=1)
        # Fill only currently-empty pixels that have ≥1 filled neighbor
        new_fill = (m == 0) & (neighbor_count > 0)
        avg = neighbor_sum_c / neighbor_count.clamp(min=1.0)
        # Update img and mask
        img_masked = torch.where(new_fill.expand_as(img_masked), avg, img_masked)
        m = m + new_fill.float()
        m = m.clamp(max=1.0)

    out_img = img_masked.squeeze(0).permute(1, 2, 0)
    out_mask = m.squeeze(0).squeeze(0).bool()
    return out_img, out_mask


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
                "- filled_mask: binary — where the composite has data. Pass\n"
                "  this (inverted) to Qwen Image Edit as the inpaint region."
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
                    tooltip="Iteratively grow the filled region by this many pixels "
                            "to prevent black-bleed at UV seams under bilinear "
                            "sampling. 0 disables.",
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
                io.Float.Input(
                    "confidence_gamma",
                    default=1.0,
                    min=0.1,
                    max=4.0,
                    step=0.1,
                    optional=True,
                    tooltip="Raise confidences to this power before blending. "
                            ">1 makes the highest-confidence view dominate more "
                            "sharply (sharper seam); 1 = standard linear blend.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="uv_texture"),
                io.Mask.Output(display_name="confidence"),
                io.Mask.Output(display_name="filled_mask"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        uv_textures: torch.Tensor,
        confidences: torch.Tensor,
        seam_dilate: int = 4,
        fill_threshold: float = 0.05,
        confidence_gamma: float = 1.0,
    ) -> io.NodeOutput:
        if uv_textures is None or uv_textures.ndim != 4:
            return io.NodeOutput(
                torch.zeros(1, 1, 1, 3), torch.zeros(1, 1, 1), torch.zeros(1, 1, 1),
                "ERROR: uv_textures must be (N,H,W,3)",
            )
        if confidences is None or confidences.ndim != 3:
            return io.NodeOutput(
                torch.zeros(1, 1, 1, 3), torch.zeros(1, 1, 1), torch.zeros(1, 1, 1),
                "ERROR: confidences must be (N,H,W)",
            )
        if uv_textures.shape[0] != confidences.shape[0]:
            return io.NodeOutput(
                torch.zeros(1, 1, 1, 3), torch.zeros(1, 1, 1), torch.zeros(1, 1, 1),
                f"ERROR: batch mismatch: {uv_textures.shape[0]} textures vs "
                f"{confidences.shape[0]} confidence maps",
            )
        if uv_textures.shape[1:3] != confidences.shape[1:3]:
            return io.NodeOutput(
                torch.zeros(1, 1, 1, 3), torch.zeros(1, 1, 1), torch.zeros(1, 1, 1),
                f"ERROR: spatial size mismatch: textures {tuple(uv_textures.shape[1:3])} "
                f"vs confidences {tuple(confidences.shape[1:3])}",
            )

        device = uv_textures.device
        N, H, W, _ = uv_textures.shape

        textures = uv_textures.float()           # (N, H, W, 3)
        conf = confidences.float().clamp(min=0)  # (N, H, W)
        if confidence_gamma != 1.0:
            conf = conf.pow(confidence_gamma)

        # Weighted sum
        weighted_sum = (textures * conf.unsqueeze(-1)).sum(dim=0)  # (H, W, 3)
        conf_sum = conf.sum(dim=0)                                 # (H, W)
        eps = 1e-8
        composite = weighted_sum / (conf_sum.unsqueeze(-1) + eps)  # (H, W, 3)

        # Use the max across views as the "trust" signal for the output mask
        conf_max = conf.max(dim=0).values                          # (H, W)

        # filled_mask = where any view contributed meaningfully
        filled_mask = conf_max > fill_threshold

        # Zero out unfilled regions in the composite
        composite = composite * filled_mask.float().unsqueeze(-1)

        # Optional seam dilation — fill outward by averaging into empty neighbors
        if seam_dilate > 0:
            composite, dilated_mask = _dilate_texture(composite, filled_mask, seam_dilate)
        else:
            dilated_mask = filled_mask

        n_filled_orig = int(filled_mask.sum().item())
        n_filled_after = int(dilated_mask.sum().item())
        total = H * W

        out_image = composite.unsqueeze(0).cpu()                       # (1, H, W, 3)
        out_confidence = conf_max.unsqueeze(0).cpu()                   # (1, H, W)
        out_filled = dilated_mask.float().unsqueeze(0).cpu()           # (1, H, W)

        status = (
            f"blended {N} views | "
            f"texture {H}x{W} | "
            f"filled {100.0*n_filled_orig/total:.1f}% pre-dilate, "
            f"{100.0*n_filled_after/total:.1f}% post-dilate "
            f"(seam_dilate={seam_dilate})"
        )
        return io.NodeOutput(out_image, out_confidence, out_filled, status)


FACEWRAP_BLEND_V3_NODES = [BD_UVConfidenceBlend]

FACEWRAP_BLEND_NODES = {
    "BD_UVConfidenceBlend": BD_UVConfidenceBlend,
}

FACEWRAP_BLEND_DISPLAY_NAMES = {
    "BD_UVConfidenceBlend": "BD UV Confidence Blend",
}
