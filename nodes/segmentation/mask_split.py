"""
Split a HUMAN_PARSE_MAP into per-region MASK outputs and a colorized preview.

Pure tensor logic — no model load, no stubs. Works with either FASHN or ATR
backend output by reading the labels list off the HUMAN_PARSE_MAP dict.
"""

import torch
from comfy_api.latest import io

from .labels import CLOTHING_GROUPS, PALETTE, label_index
from .types import HumanParseMapInput


def _group_mask(class_map: torch.Tensor, labels: list[str], group: str) -> torch.Tensor:
    members = CLOTHING_GROUPS.get(group, [])
    indices = [i for i in (label_index(labels, m) for m in members) if i is not None]
    if not indices:
        return torch.zeros_like(class_map, dtype=torch.float32)
    mask = torch.zeros_like(class_map, dtype=torch.bool)
    for i in indices:
        mask |= class_map == i
    return mask.float()


def _named_mask(class_map: torch.Tensor, labels: list[str], name: str) -> torch.Tensor:
    idx = label_index(labels, name)
    if idx is None:
        return torch.zeros_like(class_map, dtype=torch.float32)
    return (class_map == idx).float()


class BD_HumanParserMaskSplit(io.ComfyNode):
    """Split HUMAN_PARSE_MAP into the most-used clothing group masks."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_HumanParserMaskSplit",
            display_name="BD Human Parser Mask Split",
            category="🧠BrainDead/Segmentation",
            description=(
                "Split a HUMAN_PARSE_MAP into per-region masks. Group masks (all_clothing, "
                "upper_body, lower_body, skin, accessories, footwear) work across FASHN and ATR "
                "backends; missing labels are silently zero."
            ),
            inputs=[
                HumanParseMapInput("parse_map"),
            ],
            outputs=[
                io.Mask.Output(display_name="all_clothing"),
                io.Mask.Output(display_name="upper_body"),
                io.Mask.Output(display_name="lower_body"),
                io.Mask.Output(display_name="accessories"),
                io.Mask.Output(display_name="skin"),
                io.Mask.Output(display_name="footwear"),
                io.Mask.Output(display_name="hair"),
                io.Mask.Output(display_name="face"),
                io.Mask.Output(display_name="background"),
            ],
        )

    @classmethod
    def execute(cls, parse_map) -> io.NodeOutput:
        cm = parse_map["class_map"]
        labels = parse_map["labels"]
        return io.NodeOutput(
            _group_mask(cm, labels, "all_clothing"),
            _group_mask(cm, labels, "upper_body"),
            _group_mask(cm, labels, "lower_body"),
            _group_mask(cm, labels, "accessories"),
            _group_mask(cm, labels, "skin"),
            _group_mask(cm, labels, "footwear"),
            _named_mask(cm, labels, "hair"),
            _named_mask(cm, labels, "face"),
            _named_mask(cm, labels, "background"),
        )


class BD_HumanParserNamedMask(io.ComfyNode):
    """Extract a single named region by label."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        all_names = sorted({n for g in CLOTHING_GROUPS.values() for n in g} | {"hair", "face", "background"})
        return io.Schema(
            node_id="BD_HumanParserNamedMask",
            display_name="BD Human Parser Named Mask",
            category="🧠BrainDead/Segmentation",
            description="Extract a single label as a MASK. Returns zeros if the label is not in the active backend's scheme.",
            inputs=[
                HumanParseMapInput("parse_map"),
                io.Combo.Input("label", options=all_names, default="top"),
            ],
            outputs=[
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    def execute(cls, parse_map, label) -> io.NodeOutput:
        cm = parse_map["class_map"]
        labels = parse_map["labels"]
        return io.NodeOutput(_named_mask(cm, labels, label))


class BD_HumanParserPreview(io.ComfyNode):
    """Colorize a HUMAN_PARSE_MAP into an RGB IMAGE for visual debugging."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_HumanParserPreview",
            display_name="BD Human Parser Preview",
            category="🧠BrainDead/Segmentation",
            description="Colorize a HUMAN_PARSE_MAP using a fixed 18-class palette.",
            inputs=[
                HumanParseMapInput("parse_map"),
            ],
            outputs=[
                io.Image.Output(display_name="preview"),
            ],
        )

    @classmethod
    def execute(cls, parse_map) -> io.NodeOutput:
        cm = parse_map["class_map"]
        palette = torch.tensor(PALETTE, dtype=torch.float32) / 255.0
        if cm.max().item() >= palette.shape[0]:
            extra = cm.max().item() - palette.shape[0] + 1
            palette = torch.cat([palette, torch.zeros(extra, 3)], dim=0)
        rgb = palette[cm.long()]
        return io.NodeOutput(rgb)


MASK_SPLIT_V3_NODES = [BD_HumanParserMaskSplit, BD_HumanParserNamedMask, BD_HumanParserPreview]
MASK_SPLIT_NODES = {
    "BD_HumanParserMaskSplit": BD_HumanParserMaskSplit,
    "BD_HumanParserNamedMask": BD_HumanParserNamedMask,
    "BD_HumanParserPreview": BD_HumanParserPreview,
}
MASK_SPLIT_DISPLAY_NAMES = {
    "BD_HumanParserMaskSplit": "BD Human Parser Mask Split",
    "BD_HumanParserNamedMask": "BD Human Parser Named Mask",
    "BD_HumanParserPreview": "BD Human Parser Preview",
}
