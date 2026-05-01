"""
Post-processing for HUMAN_PARSE_MAP: morphology + connected-component filtering.

Per-class binary cleanup: any pixel removed from its class becomes background (0).
Pure CPU (scipy.ndimage) — fast enough for the small mask sizes involved.
"""

import numpy as np
import torch
import scipy.ndimage as ndi

from comfy_api.latest import io

from .types import HumanParseMapInput, HumanParseMapOutput


def _clean_class_mask(mask: np.ndarray, morph_op: str, kernel_size: int, min_area: int) -> np.ndarray:
    if morph_op != "none" and kernel_size > 0:
        kernel = np.ones((kernel_size, kernel_size), dtype=bool)
        if morph_op == "open":
            mask = ndi.binary_opening(mask, kernel)
        elif morph_op == "close":
            mask = ndi.binary_closing(mask, kernel)
        elif morph_op == "open_then_close":
            mask = ndi.binary_opening(mask, kernel)
            mask = ndi.binary_closing(mask, kernel)

    if min_area > 0:
        labeled, num = ndi.label(mask)
        if num > 0:
            sizes = ndi.sum(mask, labeled, range(1, num + 1))
            keep_label = np.zeros(num + 1, dtype=bool)
            keep_label[1:] = sizes >= min_area
            mask = keep_label[labeled]

    return mask


class BD_HumanParserMaskClean(io.ComfyNode):
    """Morphological cleanup + small-component removal for a HUMAN_PARSE_MAP."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_HumanParserMaskClean",
            display_name="BD Human Parser Mask Clean",
            category="🧠BrainDead/Segmentation",
            description=(
                "Per-class morphological cleanup + min-component-area filter. "
                "Removes speckle, fills small holes, drops tiny disconnected blobs. "
                "Pixels stripped from their class fall back to background."
            ),
            inputs=[
                HumanParseMapInput("parse_map"),
                io.Combo.Input(
                    "morph_op",
                    options=["none", "open", "close", "open_then_close"],
                    default="open", optional=True,
                    tooltip="open=remove specks, close=fill small holes, open_then_close=both.",
                ),
                io.Int.Input(
                    "kernel_size", default=3, min=1, max=15, step=2, optional=True,
                    tooltip="Morphology kernel side (odd numbers work best).",
                ),
                io.Int.Input(
                    "min_area", default=64, min=0, max=100000, step=8, optional=True,
                    tooltip="Drop connected components smaller than N pixels per class. 0 = off.",
                ),
            ],
            outputs=[
                HumanParseMapOutput(display_name="parse_map"),
            ],
        )

    @classmethod
    def execute(cls, parse_map, morph_op="open", kernel_size=3, min_area=64) -> io.NodeOutput:
        cm = parse_map["class_map"].clone()
        labels = parse_map["labels"]

        cm_np = cm.numpy()
        for b in range(cm_np.shape[0]):
            frame = cm_np[b]
            for cls_idx in range(1, len(labels)):
                original = frame == cls_idx
                if not original.any():
                    continue
                cleaned = _clean_class_mask(original.copy(), morph_op, kernel_size, min_area)
                removed = original & ~cleaned
                if removed.any():
                    frame[removed] = 0

        out = {
            "class_map": torch.from_numpy(cm_np).long(),
            "labels": labels,
            "backend": parse_map["backend"],
        }
        if "confidence" in parse_map:
            out["confidence"] = parse_map["confidence"]
        return io.NodeOutput(out)


CLEANUP_V3_NODES = [BD_HumanParserMaskClean]
CLEANUP_NODES = {"BD_HumanParserMaskClean": BD_HumanParserMaskClean}
CLEANUP_DISPLAY_NAMES = {"BD_HumanParserMaskClean": "BD Human Parser Mask Clean"}
