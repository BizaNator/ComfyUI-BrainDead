"""
BrainDead Segmentation nodes — human/clothing parsing + the BD_Parts* pipeline.

CATEGORY: 🧠BrainDead/Segmentation

Primary pipeline (SAM3 + Lotus-2 + QwenVL → rebuild loop):
- BD_SAM3MultiPrompt   → multi-prompt grounded segmentation
- BD_PartsRefine       → IoU dedup of overlapping prompts
- BD_PartsBuilder      → build PARTS_BUNDLE from masks + image + depth
- BD_PartsCompose      → flatten bundle to single image
- BD_PartsExport       → per-tag PNG + composite + layered PSD

Auxiliary (legacy / different segmentation paths):
- BD_FashnHumanParser, BD_ATRHumanParser → HUMAN_PARSE_MAP
- BD_HumanParserMaskClean / Split / NamedMask / Preview
- BD_MaskResolver — categorical skin/clothes/accessories resolution
- BD_MaskFlatten, BD_PackChannels, BD_DerivePBR — asset prep
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
from .sam3_multiprompt import (
    SAM3_MULTIPROMPT_NODES,
    SAM3_MULTIPROMPT_DISPLAY_NAMES,
    SAM3_MULTIPROMPT_V3_NODES,
)
from .mask_resolver import (
    MASK_RESOLVER_NODES,
    MASK_RESOLVER_DISPLAY_NAMES,
    MASK_RESOLVER_V3_NODES,
)
from .mask_flatten import (
    MASK_FLATTEN_NODES,
    MASK_FLATTEN_DISPLAY_NAMES,
    MASK_FLATTEN_V3_NODES,
)
from .pack_channels import (
    PACK_CHANNELS_NODES,
    PACK_CHANNELS_DISPLAY_NAMES,
    PACK_CHANNELS_V3_NODES,
)
from .derive_pbr import (
    DERIVE_PBR_NODES,
    DERIVE_PBR_DISPLAY_NAMES,
    DERIVE_PBR_V3_NODES,
)
from .parts_builder import (
    PARTS_BUILDER_NODES,
    PARTS_BUILDER_DISPLAY_NAMES,
    PARTS_BUILDER_V3_NODES,
)
from .parts_refine import (
    PARTS_REFINE_NODES,
    PARTS_REFINE_DISPLAY_NAMES,
    PARTS_REFINE_V3_NODES,
)
from .parts_compose import (
    PARTS_COMPOSE_NODES,
    PARTS_COMPOSE_DISPLAY_NAMES,
    PARTS_COMPOSE_V3_NODES,
)
from .parts_export import (
    PARTS_EXPORT_NODES,
    PARTS_EXPORT_DISPLAY_NAMES,
    PARTS_EXPORT_V3_NODES,
)
from .parts_batch_edit import (
    PARTS_BATCH_EDIT_NODES,
    PARTS_BATCH_EDIT_DISPLAY_NAMES,
    PARTS_BATCH_EDIT_V3_NODES,
)

SEGMENTATION_V3_NODES = [
    *FASHN_PARSER_V3_NODES,
    *ATR_PARSER_V3_NODES,
    *CLEANUP_V3_NODES,
    *MASK_SPLIT_V3_NODES,
    *SAM3_MULTIPROMPT_V3_NODES,
    *MASK_RESOLVER_V3_NODES,
    *MASK_FLATTEN_V3_NODES,
    *PACK_CHANNELS_V3_NODES,
    *DERIVE_PBR_V3_NODES,
    *PARTS_BUILDER_V3_NODES,
    *PARTS_REFINE_V3_NODES,
    *PARTS_COMPOSE_V3_NODES,
    *PARTS_EXPORT_V3_NODES,
    *PARTS_BATCH_EDIT_V3_NODES,
]

NODE_CLASS_MAPPINGS = {
    **FASHN_PARSER_NODES,
    **ATR_PARSER_NODES,
    **CLEANUP_NODES,
    **MASK_SPLIT_NODES,
    **SAM3_MULTIPROMPT_NODES,
    **MASK_RESOLVER_NODES,
    **MASK_FLATTEN_NODES,
    **PACK_CHANNELS_NODES,
    **DERIVE_PBR_NODES,
    **PARTS_BUILDER_NODES,
    **PARTS_REFINE_NODES,
    **PARTS_COMPOSE_NODES,
    **PARTS_EXPORT_NODES,
    **PARTS_BATCH_EDIT_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **FASHN_PARSER_DISPLAY_NAMES,
    **ATR_PARSER_DISPLAY_NAMES,
    **CLEANUP_DISPLAY_NAMES,
    **MASK_SPLIT_DISPLAY_NAMES,
    **SAM3_MULTIPROMPT_DISPLAY_NAMES,
    **MASK_RESOLVER_DISPLAY_NAMES,
    **MASK_FLATTEN_DISPLAY_NAMES,
    **PACK_CHANNELS_DISPLAY_NAMES,
    **DERIVE_PBR_DISPLAY_NAMES,
    **PARTS_BUILDER_DISPLAY_NAMES,
    **PARTS_REFINE_DISPLAY_NAMES,
    **PARTS_COMPOSE_DISPLAY_NAMES,
    **PARTS_EXPORT_DISPLAY_NAMES,
    **PARTS_BATCH_EDIT_DISPLAY_NAMES,
}
