"""
BrainDead Segmentation nodes — human/clothing parsing.

CATEGORY: 🧠BrainDead/Segmentation

Backends:
- BD_FashnHumanParser: fashn-ai/fashn-human-parser (SegFormer-B4, NVIDIA license)
- BD_ATRHumanParser:   mattmdjaga/segformer_b2_clothes (ATR scheme, MIT)

Both produce a HUMAN_PARSE_MAP consumed by:
- BD_HumanParserMaskSplit:  6 group masks + hair/face/background
- BD_HumanParserNamedMask:  pick one label by name
- BD_HumanParserPreview:    colorized RGB preview
"""

from .fashn_parser import (
    FASHN_PARSER_NODES,
    FASHN_PARSER_DISPLAY_NAMES,
    FASHN_PARSER_V3_NODES,
)
from .atr_parser import (
    ATR_PARSER_NODES,
    ATR_PARSER_DISPLAY_NAMES,
    ATR_PARSER_V3_NODES,
)
from .mask_split import (
    MASK_SPLIT_NODES,
    MASK_SPLIT_DISPLAY_NAMES,
    MASK_SPLIT_V3_NODES,
)

SEGMENTATION_V3_NODES = [
    *FASHN_PARSER_V3_NODES,
    *ATR_PARSER_V3_NODES,
    *MASK_SPLIT_V3_NODES,
]

NODE_CLASS_MAPPINGS = {
    **FASHN_PARSER_NODES,
    **ATR_PARSER_NODES,
    **MASK_SPLIT_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **FASHN_PARSER_DISPLAY_NAMES,
    **ATR_PARSER_DISPLAY_NAMES,
    **MASK_SPLIT_DISPLAY_NAMES,
}
