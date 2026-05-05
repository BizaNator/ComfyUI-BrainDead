"""
PARTS_BUNDLE — the canonical wrapper type passed between BD_Parts* nodes.

Shape:
  {
    "tag2pinfo": {
        tag: {
            "img":          numpy uint8 (H, W, 4) RGBA  — cropped to bbox
            "xyxy":         [x1, y1, x2, y2]            — position in source coords
            "tag":          str                         — the part's label
            "depth_median": float                       — for back-to-front sort
            "depth":        numpy uint8 (h, w) optional — per-part depth crop
        },
        ...
    },
    "frame_size": (H, W)  — source image dimensions
  }

Producers: BD_PartsBuilder
Consumers: BD_PartsRefine (passthrough), BD_PartsCompose, BD_PartsExport,
           BD_PartsBatchEdit (mutates in place)
"""

PARTS_BUNDLE = "PARTS_BUNDLE"


def ensure_bundle(bundle, *, source: str = "input"):
    """Validate that `bundle` matches the PARTS_BUNDLE shape. Returns the dict."""
    if not isinstance(bundle, dict) or "tag2pinfo" not in bundle:
        raise ValueError(
            f"{source}: expected PARTS_BUNDLE wrapper dict with 'tag2pinfo' key, "
            f"got {type(bundle).__name__}"
        )
    return bundle


def frame_size(bundle) -> tuple[int, int]:
    """Return (H, W) — falls back to (0, 0) if missing."""
    fs = bundle.get("frame_size", (0, 0))
    if not fs or fs == (0, 0):
        return 0, 0
    return int(fs[0]), int(fs[1])


def empty_bundle(H: int = 0, W: int = 0) -> dict:
    """Construct an empty PARTS_BUNDLE — used by error paths."""
    return {"tag2pinfo": {}, "frame_size": (H, W)}
