"""
Make-It-Animatable vendored code.

This is a subset of the MIA codebase needed for inference, with bpy dependencies removed.
Original: https://github.com/jasongzy/Make-It-Animatable
"""

from .model import PCAE
from .dataset_mixamo import (
    MIXAMO_JOINTS,
    JOINTS_NUM,
    KINEMATIC_TREE,
    BONES_IDX_DICT,
    Joint,
)
