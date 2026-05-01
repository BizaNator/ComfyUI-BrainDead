"""
Custom IO types for parsing nodes.

HUMAN_PARSE_MAP is a dict carrying the per-pixel class index tensor plus
the label scheme it was produced with, so downstream nodes can look up
class names without guessing the backend.

Shape contract:
    {
        "class_map": LongTensor (B, H, W) with values in [0, len(labels)),
        "labels": list[str],          # FASHN_LABELS or ATR_LABELS
        "backend": "fashn" | "atr",
    }
"""

from comfy_api.latest import io


def HumanParseMapInput(name: str, optional: bool = False, tooltip: str | None = None):
    return io.Custom("HUMAN_PARSE_MAP").Input(name, optional=optional, tooltip=tooltip)


def HumanParseMapOutput(display_name: str = "parse_map"):
    return io.Custom("HUMAN_PARSE_MAP").Output(display_name=display_name)
