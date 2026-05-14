"""
BrainDead FaceWrap — multi-view UV texture pipeline.

CATEGORY: 🧠BrainDead/FaceWrap

Pipeline (see docs/face-wrap.md):
- BD_FaceLandmarks       → MediaPipe FaceMesh on a 4-view batch
- BD_FlameFit            → landmark-only FLAME fit (future)
- BD_FlameTextureBake    → per-view UV bake via nvdiffrast (future)
- BD_UVConfidenceBlend   → cosine-weighted composite + gap mask (future)
- BD_UVTransfer          → cross-mesh UV→UV warp for CC5/Metahuman (future)
"""

from .landmarks import (
    FACEWRAP_LANDMARKS_NODES,
    FACEWRAP_LANDMARKS_DISPLAY_NAMES,
    FACEWRAP_LANDMARKS_V3_NODES,
)

FACEWRAP_V3_NODES = [
    *FACEWRAP_LANDMARKS_V3_NODES,
]

NODE_CLASS_MAPPINGS = {
    **FACEWRAP_LANDMARKS_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **FACEWRAP_LANDMARKS_DISPLAY_NAMES,
}
