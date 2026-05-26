"""
BD_FaceWrapComposite — re-composite a Qwen-filled UV texture over the
original baked texture so the baked face pixels survive byte-identical.

The Qwen-fill step (Qwen Image Edit inpaint) should only invent the gap
regions; the face that BD_FaceTextureBake/BD_UVConfidenceBlend produced
is real photo data and must not drift. Even with a hard latent noise
mask, VAE round-trips and mask bleed can shift the "preserved" region a
little. This node makes the guarantee explicit:

    out = filled_mask * original_texture + (1 - filled_mask) * qwen_output

Wire it as the LAST step of the Qwen-fill subgraph (see docs/face-wrap.md).
An optional feather softens the seam between preserved and filled regions
without letting Qwen bleed deep into the baked face.
"""

import torch
import torch.nn.functional as F

from comfy_api.latest import io


def _feather_mask(mask: torch.Tensor, feather: int) -> torch.Tensor:
    """Soften a binary mask edge by `feather` px via box blur.

    mask: (H, W) float in [0,1]. Returns (H, W) float in [0,1].
    """
    if feather <= 0:
        return mask
    k = 2 * feather + 1
    m = mask.unsqueeze(0).unsqueeze(0)
    m = F.avg_pool2d(m, kernel_size=k, stride=1, padding=feather)
    return m.squeeze(0).squeeze(0).clamp(0.0, 1.0)


class BD_FaceWrapComposite(io.ComfyNode):
    """Re-composite a Qwen-filled texture over the original — form guarantee."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_FaceWrapComposite",
            display_name="BD FaceWrap Composite",
            category="🧠BrainDead/FaceWrap",
            description=(
                "Re-composite a Qwen-filled UV texture over the original\n"
                "baked texture: out = filled*original + (1-filled)*qwen.\n\n"
                "Guarantees the baked/blended face pixels are preserved\n"
                "exactly regardless of what Qwen did in the masked gap\n"
                "region. Wire as the LAST step of the Qwen-fill subgraph.\n\n"
                "Inputs:\n"
                "- original_texture: BD_UVConfidenceBlend's uv_texture\n"
                "- filled_mask:      BD_UVConfidenceBlend's filled_mask\n"
                "- qwen_output:      the Qwen Image Edit inpaint result\n\n"
                "feather softens the preserved/filled seam without letting\n"
                "Qwen bleed deep into the baked face."
            ),
            inputs=[
                io.Image.Input(
                    "original_texture",
                    tooltip="The pre-Qwen baked/blended UV texture "
                            "(BD_UVConfidenceBlend uv_texture).",
                ),
                io.Mask.Input(
                    "filled_mask",
                    tooltip="BD_UVConfidenceBlend filled_mask — 1 where real "
                            "baked data is, 0 in the gaps Qwen filled.",
                ),
                io.Image.Input(
                    "qwen_output",
                    tooltip="The Qwen Image Edit inpaint result.",
                ),
                io.Int.Input(
                    "feather",
                    default=2,
                    min=0,
                    max=64,
                    step=1,
                    tooltip="Feather the preserved/filled seam by this many "
                            "pixels (box blur of the mask). 0 = hard edge.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="uv_texture"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        original_texture: torch.Tensor,
        filled_mask: torch.Tensor,
        qwen_output: torch.Tensor,
        feather: int = 2,
    ) -> io.NodeOutput:
        if original_texture is None or original_texture.ndim != 4:
            return io.NodeOutput(original_texture, "ERROR: original_texture must be (B,H,W,3)")
        if qwen_output is None or qwen_output.ndim != 4:
            return io.NodeOutput(original_texture, "ERROR: qwen_output must be (B,H,W,3)")
        if filled_mask is None or filled_mask.ndim != 3:
            return io.NodeOutput(original_texture, "ERROR: filled_mask must be (B,H,W)")

        orig = original_texture[0].float()       # (H, W, 3)
        qwen = qwen_output[0].float()            # (H', W', 3)
        mask = filled_mask[0].float().clamp(0, 1)  # (H, W)

        H, W = orig.shape[:2]

        # Resize Qwen output / mask to the original's resolution if they drifted
        if qwen.shape[:2] != (H, W):
            qwen = F.interpolate(
                qwen.permute(2, 0, 1).unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
        if mask.shape != (H, W):
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(0).squeeze(0).clamp(0, 1)

        feathered = _feather_mask(mask, feather)
        alpha = feathered.unsqueeze(-1)  # (H, W, 1)

        composite = alpha * orig + (1.0 - alpha) * qwen

        n_preserved = int((feathered > 0.99).sum().item())
        n_filled = int((feathered < 0.01).sum().item())
        n_blend = H * W - n_preserved - n_filled
        status = (
            f"composited {H}x{W} | preserved {100.0*n_preserved/(H*W):.1f}% "
            f"| qwen-filled {100.0*n_filled/(H*W):.1f}% "
            f"| feathered seam {n_blend:,}px (feather={feather})"
        )
        return io.NodeOutput(composite.unsqueeze(0).cpu(), status)


FACEWRAP_QWEN_COMPOSITE_V3_NODES = [BD_FaceWrapComposite]

FACEWRAP_QWEN_COMPOSITE_NODES = {
    "BD_FaceWrapComposite": BD_FaceWrapComposite,
}

FACEWRAP_QWEN_COMPOSITE_DISPLAY_NAMES = {
    "BD_FaceWrapComposite": "BD FaceWrap Composite",
}
