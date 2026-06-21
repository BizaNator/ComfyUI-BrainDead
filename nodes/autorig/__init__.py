"""
🧠BrainDead/AutoRig — auto-rigging nodes for ComfyUI.

Wraps Make-It-Animatable and UniRig (both delivered by the sibling
ComfyUI-UniRig pack) with V3 nodes that fit the BrainDead pipeline pattern,
and adds a Mixamo → UEFN_Mannequin bone-name remap pass so the rigged FBX
ends up with the canonical Fortnite/UEFN skeleton naming.

Nodes:
  • BD_AutoRigMIA      — fast humanoid rigging via Make-It-Animatable (<1s)
  • BD_AutoRigUniRig   — general autoregressive-transformer rigging
  • BD_MixamoToUEFN    — bone-name remap only, FBX in → FBX out
  • BD_AutoRigUEFN     — full UEFN skeleton conversion (step 2 of pipeline)

Pipeline:
    Mesh → BD_AutoRigMIA  →  BD_AutoRigUEFN  →  UEFN FBX (ready for import)

    BD_AutoRigMIA produces a Mixamo-rigged FBX.
    BD_AutoRigUEFN transfers weights from the bundled SKM_UEFN_Mannequin
    reference via Blender's Data Transfer modifier, producing a character
    bound to the genuine UEFN/Fortnite armature.

    BD_MixamoToUEFN is a lightweight alternative that only renames bones
    (no weight transfer) — use when you have an existing Mixamo FBX and
    want the name convention only.
"""

from .bone_remap import (
    BD_MixamoToUEFN,
    BONE_REMAP_V3_NODES,
    BONE_REMAP_NODES,
    BONE_REMAP_DISPLAY_NAMES,
)
from .mia_autorig import (
    BD_AutoRigMIA,
    MIA_V3_NODES,
    MIA_NODES,
    MIA_DISPLAY_NAMES,
)
from .unirig_autorig import (
    BD_AutoRigUniRig,
    UNIRIG_V3_NODES,
    UNIRIG_NODES,
    UNIRIG_DISPLAY_NAMES,
)
from .uefn_skeleton import (
    BD_AutoRigUEFN,
    UEFN_SKEL_V3_NODES,
    UEFN_SKEL_NODES,
    UEFN_SKEL_DISPLAY_NAMES,
)


AUTORIG_V3_NODES = [
    *BONE_REMAP_V3_NODES,
    *MIA_V3_NODES,
    *UNIRIG_V3_NODES,
    *UEFN_SKEL_V3_NODES,
]

AUTORIG_NODES = {
    **BONE_REMAP_NODES,
    **MIA_NODES,
    **UNIRIG_NODES,
    **UEFN_SKEL_NODES,
}

AUTORIG_DISPLAY_NAMES = {
    **BONE_REMAP_DISPLAY_NAMES,
    **MIA_DISPLAY_NAMES,
    **UNIRIG_DISPLAY_NAMES,
    **UEFN_SKEL_DISPLAY_NAMES,
}

__all__ = [
    "BD_MixamoToUEFN",
    "BD_AutoRigMIA",
    "BD_AutoRigUniRig",
    "BD_AutoRigUEFN",
    "AUTORIG_V3_NODES",
    "AUTORIG_NODES",
    "AUTORIG_DISPLAY_NAMES",
]
