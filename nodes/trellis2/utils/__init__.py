"""
TRELLIS2 utilities for BrainDead nodes.

This module handles:
- Path setup to import trellis2 from ComfyUI-TRELLIS2
- Model manager for lazy loading
- Helper functions
"""

import sys
from pathlib import Path

# Add ComfyUI-TRELLIS2 to path so we can import trellis2
TRELLIS2_NODE_PATH = Path(__file__).parent.parent.parent.parent.parent / "ComfyUI-TRELLIS2"
if TRELLIS2_NODE_PATH.exists():
    trellis2_str = str(TRELLIS2_NODE_PATH)
    if trellis2_str not in sys.path:
        sys.path.insert(0, trellis2_str)
        print(f"[BD TRELLIS2] Added to path: {trellis2_str}")

# Check if trellis2 is available
try:
    import trellis2
    HAS_TRELLIS2 = True
    print(f"[BD TRELLIS2] trellis2 package available")
except ImportError as e:
    HAS_TRELLIS2 = False
    print(f"[BD TRELLIS2] WARNING: trellis2 not available: {e}")

__all__ = ["HAS_TRELLIS2", "TRELLIS2_NODE_PATH"]
