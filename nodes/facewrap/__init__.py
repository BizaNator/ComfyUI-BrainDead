"""
BrainDead FaceWrap — multi-view UV texture pipeline.

CATEGORY: 🧠BrainDead/FaceWrap

Pipeline (see docs/face-wrap.md):
- BD_FaceLandmarks       → MediaPipe FaceLandmarker on a 4-view batch
- BD_FaceFit             → assemble landmarks + canonical face mesh
- BD_FaceTextureBake     → per-view UV bake via nvdiffrast (future)
- BD_UVConfidenceBlend   → cosine-weighted composite + gap mask (future)
- BD_UVTransfer          → cross-mesh UV→UV warp for CC5/Metahuman (future)
"""

from .landmarks import (
    FACEWRAP_LANDMARKS_NODES,
    FACEWRAP_LANDMARKS_DISPLAY_NAMES,
    FACEWRAP_LANDMARKS_V3_NODES,
)
from .face_fit import (
    FACEWRAP_FACE_FIT_NODES,
    FACEWRAP_FACE_FIT_DISPLAY_NAMES,
    FACEWRAP_FACE_FIT_V3_NODES,
)

FACEWRAP_V3_NODES = [
    *FACEWRAP_LANDMARKS_V3_NODES,
    *FACEWRAP_FACE_FIT_V3_NODES,
]

NODE_CLASS_MAPPINGS = {
    **FACEWRAP_LANDMARKS_NODES,
    **FACEWRAP_FACE_FIT_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **FACEWRAP_LANDMARKS_DISPLAY_NAMES,
    **FACEWRAP_FACE_FIT_DISPLAY_NAMES,
}
