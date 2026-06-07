"""
BD_MaskBatchIndex — pick one mask by index from a batch tensor (N, H, W).

Use case: SAM3 per_prompt_masks is a (N,H,W) batch, one mask per prompt.
This node extracts mask[i] so you can wire individual part masks into
BD_PackChannels, BD_RemoveBackground, or any single-mask input.
"""

import torch
from comfy_api.latest import io


class BD_MaskBatchIndex(io.ComfyNode):
    """Pick one mask by 0-based index from a batch (N,H,W). Use to extract
    individual parts from SAM3 per_prompt_masks before feeding BD_PackChannels."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_MaskBatchIndex",
            display_name="BD Mask Batch Index",
            category="🧠BrainDead/Segmentation",
            description=(
                "Extract one mask by 0-based index from a mask batch (N,H,W). "
                "Bridges SAM3 per_prompt_masks → single-mask inputs on BD_PackChannels, "
                "BD_RemoveBackground, etc. Negative indices count from the end."
            ),
            inputs=[
                io.Mask.Input("masks",
                              tooltip="Mask batch (N,H,W) — e.g. per_prompt_masks from BD_SAM3MultiPrompt."),
                io.Int.Input("index", default=0, min=-64, max=64,
                             tooltip="0-based index of the mask to extract. Negative counts from end. "
                                     "Clamped to [0, N-1] if out of range."),
                io.Boolean.Input("invert", default=False, optional=True,
                                 tooltip="Invert the selected mask before output."),
            ],
            outputs=[
                io.Mask.Output(display_name="mask",
                               tooltip="Single (1,H,W) mask at the selected index."),
                io.String.Output(display_name="info",
                                 tooltip="'index/total' — e.g. '2/5' confirms which mask was selected."),
            ],
        )

    @classmethod
    def execute(cls, masks, index=0, invert=False) -> io.NodeOutput:
        m = masks.detach().cpu().float()
        if m.ndim == 2:
            m = m.unsqueeze(0)  # (H,W) → (1,H,W)

        N = m.shape[0]
        # Resolve negative / clamp
        idx = int(index)
        if idx < 0:
            idx = N + idx
        idx = max(0, min(idx, N - 1))

        selected = m[idx].unsqueeze(0)  # (1,H,W)
        if invert:
            selected = 1.0 - selected

        info = f"{idx}/{N}"
        print(f"[BD MaskBatchIndex] Selected mask {idx} of {N}", flush=True)
        return io.NodeOutput(selected, info)


MASK_BATCH_INDEX_V3_NODES = [BD_MaskBatchIndex]
MASK_BATCH_INDEX_NODES = {"BD_MaskBatchIndex": BD_MaskBatchIndex}
MASK_BATCH_INDEX_DISPLAY_NAMES = {"BD_MaskBatchIndex": "BD Mask Batch Index"}
