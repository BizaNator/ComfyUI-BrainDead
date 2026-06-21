"""
🧠BrainDead/AutoRig — auto-rigging nodes for ComfyUI.

Wraps Make-It-Animatable and UniRig (both delivered by the sibling
ComfyUI-UniRig pack) with V3 nodes that fit the BrainDead pipeline pattern,
and adds a Mixamo → UEFN_Mannequin bone-name remap pass so the rigged FBX
ends up with the canonical Fortnite/UEFN skeleton naming.

Nodes:
  • BD_AutoRigMIA      — fast humanoid rigging via Make-It-Animatable (<1s)
  • BD_AutoRigUniRig   — general autoregressive-transformer rigging
  • BD_MixamoToUEFN    — bone-name remap, FBX in → FBX out

Workflow:
    Mesh → BD_AutoRigMIA  (remap_to_uefn=True)  →  UEFN-ready FBX
    Mesh → BD_AutoRigUniRig (skeleton_template=mixamo, remap_to_uefn=True)
                                                  →  UEFN-ready FBX

After this node pack's FBX comes back to Blender, run PoseFixer_v1 from the
BrainDeadBlender uefn_pipeline to retarget the rest pose from T-pose to the
canonical UEFN A-pose.
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


AUTORIG_V3_NODES = [
    *BONE_REMAP_V3_NODES,
    *MIA_V3_NODES,
    *UNIRIG_V3_NODES,
]

AUTORIG_NODES = {
    **BONE_REMAP_NODES,
    **MIA_NODES,
    **UNIRIG_NODES,
}

AUTORIG_DISPLAY_NAMES = {
    **BONE_REMAP_DISPLAY_NAMES,
    **MIA_DISPLAY_NAMES,
    **UNIRIG_DISPLAY_NAMES,
}

__all__ = [
    "BD_MixamoToUEFN",
    "BD_AutoRigMIA",
    "BD_AutoRigUniRig",
    "AUTORIG_V3_NODES",
    "AUTORIG_NODES",
    "AUTORIG_DISPLAY_NAMES",
]
