"""
BrainDead Segmentation nodes — human/clothing parsing.

CATEGORY: 🧠BrainDead/Segmentation

Backends:
- BD_FashnHumanParser: fashn-ai/fashn-human-parser (SegFormer-B4, NVIDIA license)
- BD_ATRHumanParser:   mattmdjaga/segformer_b2_clothes (ATR scheme, MIT)

Both produce a HUMAN_PARSE_MAP consumed by:
- BD_HumanParserMaskClean: morphology + min-area cleanup (optional, before split)
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
from .cleanup import (
    CLEANUP_NODES,
    CLEANUP_DISPLAY_NAMES,
    CLEANUP_V3_NODES,
)
from .mask_split import (
    MASK_SPLIT_NODES,
    MASK_SPLIT_DISPLAY_NAMES,
    MASK_SPLIT_V3_NODES,
)
from .seethrough_export import (
    SEETHROUGH_EXPORT_NODES,
    SEETHROUGH_EXPORT_DISPLAY_NAMES,
    SEETHROUGH_EXPORT_V3_NODES,
)
from .seethrough_pipeline import (
    SEETHROUGH_PIPELINE_NODES,
    SEETHROUGH_PIPELINE_DISPLAY_NAMES,
    SEETHROUGH_PIPELINE_V3_NODES,
)
from .sam3_multiprompt import (
    SAM3_MULTIPROMPT_NODES,
    SAM3_MULTIPROMPT_DISPLAY_NAMES,
    SAM3_MULTIPROMPT_V3_NODES,
)
from .parts_remap import (
    PARTS_REMAP_NODES,
    PARTS_REMAP_DISPLAY_NAMES,
    PARTS_REMAP_V3_NODES,
)
from .mask_resolver import (
    MASK_RESOLVER_NODES,
    MASK_RESOLVER_DISPLAY_NAMES,
    MASK_RESOLVER_V3_NODES,
)

SEGMENTATION_V3_NODES = [
    *FASHN_PARSER_V3_NODES,
    *ATR_PARSER_V3_NODES,
    *CLEANUP_V3_NODES,
    *MASK_SPLIT_V3_NODES,
    *SEETHROUGH_EXPORT_V3_NODES,
    *SEETHROUGH_PIPELINE_V3_NODES,
    *SAM3_MULTIPROMPT_V3_NODES,
    *PARTS_REMAP_V3_NODES,
    *MASK_RESOLVER_V3_NODES,
]

NODE_CLASS_MAPPINGS = {
    **FASHN_PARSER_NODES,
    **ATR_PARSER_NODES,
    **CLEANUP_NODES,
    **MASK_SPLIT_NODES,
    **SEETHROUGH_EXPORT_NODES,
    **SEETHROUGH_PIPELINE_NODES,
    **SAM3_MULTIPROMPT_NODES,
    **PARTS_REMAP_NODES,
    **MASK_RESOLVER_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **FASHN_PARSER_DISPLAY_NAMES,
    **ATR_PARSER_DISPLAY_NAMES,
    **CLEANUP_DISPLAY_NAMES,
    **MASK_SPLIT_DISPLAY_NAMES,
    **SEETHROUGH_EXPORT_DISPLAY_NAMES,
    **SEETHROUGH_PIPELINE_DISPLAY_NAMES,
    **SAM3_MULTIPROMPT_DISPLAY_NAMES,
    **PARTS_REMAP_DISPLAY_NAMES,
    **MASK_RESOLVER_DISPLAY_NAMES,
}
