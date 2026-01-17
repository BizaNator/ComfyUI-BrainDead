"""
Custom types for BrainDead mesh nodes.

Provides TRIMESH type wrapper to match TRELLIS2 output.
"""

from comfy_api.latest import io


# TRIMESH type wrapper - use io.Custom("TRIMESH") for input/output
# This matches TRELLIS2's "TRIMESH" type
def TrimeshInput(name: str, **kwargs):
    """Create a TRIMESH input (matches TRELLIS2 output type)."""
    return io.Custom("TRIMESH").Input(name, **kwargs)


def TrimeshOutput(display_name: str = "mesh"):
    """Create a TRIMESH output (matches TRELLIS2 input type)."""
    return io.Custom("TRIMESH").Output(display_name=display_name)
