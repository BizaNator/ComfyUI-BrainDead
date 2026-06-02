"""
BD_FillMaskHoles — fill interior holes in a binary mask.

Uses scipy.ndimage.binary_fill_holes to flood-fill any enclosed region
inside the mask boundary. Optionally applies morphological closing first
to connect near-touching perimeter edges before filling.

Use case: SAM3 correctly detects the lip/teeth/tongue perimeter but leaves
interior gaps. This node fills those enclosed holes so the mask is solid.

Pipeline position: after BD_SAM3MultiPrompt, before BD_MaskResolver.
"""

import numpy as np
import torch
from comfy_api.latest import io


def _to_hw(mask: torch.Tensor) -> np.ndarray:
    m = mask.detach().cpu().float()
    if m.ndim == 3:
        m = m[0]
    return m.numpy().astype(np.float32)


def _from_hw(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)


def _fill(arr: np.ndarray, threshold: float, closing_radius: int,
          smooth_edges: int) -> np.ndarray:
    binary = (arr >= threshold)

    if closing_radius > 0:
        import cv2
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * closing_radius + 1, 2 * closing_radius + 1),
        )
        u8 = binary.astype(np.uint8) * 255
        closed = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, k)
        binary = closed > 127

    from scipy.ndimage import binary_fill_holes
    filled = binary_fill_holes(binary).astype(np.float32)

    if smooth_edges > 0:
        import cv2
        ksize = 2 * smooth_edges + 1
        filled = cv2.GaussianBlur(filled, (ksize, ksize), smooth_edges * 0.5)

    return filled


class BD_FillMaskHoles(io.ComfyNode):
    """Fill interior holes in a mask using binary_fill_holes. Insert after SAM3 when the perimeter is correct but interior pixels are missing."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_FillMaskHoles",
            display_name="BD Fill Mask Holes",
            category="🧠BrainDead/Segmentation",
            description=(
                "Fill interior holes in a binary mask. Uses scipy binary_fill_holes to "
                "flood-fill any enclosed region inside the mask boundary. Optional morphological "
                "closing first connects near-touching edges so enclosed regions are fully closed "
                "before filling. Use after SAM3 when the perimeter is detected but interior "
                "pixels are missing."
            ),
            inputs=[
                io.Mask.Input("mask"),
                io.Float.Input(
                    "threshold", default=0.5, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="Binarize the input mask at this value before filling. "
                            "0.5 = standard binary threshold.",
                ),
                io.Int.Input(
                    "closing_radius", default=4, min=0, max=64, step=1, optional=True,
                    tooltip="Morphological closing radius (pixels) applied BEFORE fill_holes. "
                            "Closing = dilate then erode — it bridges small gaps in the mask "
                            "perimeter so that enclosed regions are properly sealed before filling. "
                            "0 = skip closing (only fill_holes). 4–8 = good for SAM3 lip/teeth gaps.",
                ),
                io.Int.Input(
                    "smooth_edges", default=0, min=0, max=16, step=1, optional=True,
                    tooltip="Gaussian blur radius applied to the filled mask for soft edges. "
                            "0 = hard binary output. 2–4 = slight feathering at boundaries.",
                ),
            ],
            outputs=[
                io.Mask.Output(display_name="filled_mask"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, mask, threshold=0.5, closing_radius=4,
                smooth_edges=0) -> io.NodeOutput:
        arr = _to_hw(mask)
        original_coverage = float((arr >= threshold).mean()) * 100.0

        filled = _fill(arr, float(threshold), int(closing_radius), int(smooth_edges))

        filled_coverage = float((filled > 0.5).mean()) * 100.0
        gained = filled_coverage - original_coverage
        status = (
            f"fill_holes: {original_coverage:.2f}% → {filled_coverage:.2f}% "
            f"(+{gained:.2f}%) closing={closing_radius}px smooth={smooth_edges}"
        )
        print(f"[BD FillMaskHoles] {status}", flush=True)

        return io.NodeOutput(_from_hw(filled), status)


FILL_MASK_HOLES_V3_NODES = [BD_FillMaskHoles]
FILL_MASK_HOLES_NODES = {"BD_FillMaskHoles": BD_FillMaskHoles}
FILL_MASK_HOLES_DISPLAY_NAMES = {"BD_FillMaskHoles": "BD Fill Mask Holes"}
