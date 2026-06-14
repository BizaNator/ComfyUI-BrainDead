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

import os as _os

PARTS_BUNDLE = "PARTS_BUNDLE"

# Path to the bundled default character category table shipped with the node.
CATEGORY_TABLE_DEFAULT_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
    "config", "parts_categories_character.txt"
)


def _load_file_safe(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def parse_category_table(text: str) -> dict[str, tuple[str, str]]:
    """Parse 'tag = slug:region' lines.  Returns {tag: (slug, region)}.

    # comment lines and blank lines are ignored.
    'tag = slug' (no colon) sets slug only; region stays ''.
    Empty / all-comment text → {} (passthrough: tag used as slug, region='').
    """
    result: dict[str, tuple[str, str]] = {}
    for line in (text or "").splitlines():
        line = line.split("#")[0].strip()
        if not line or "=" not in line:
            continue
        tag, rest = line.split("=", 1)
        tag = tag.strip()
        rest = rest.strip()
        if not tag:
            continue
        if ":" in rest:
            slug, region = rest.split(":", 1)
            result[tag] = (slug.strip(), region.strip())
        elif rest:
            result[tag] = (rest, "")
    return result


# Pre-loaded default so the widget has it as default value at schema build time.
DEFAULT_CATEGORY_TABLE = _load_file_safe(CATEGORY_TABLE_DEFAULT_PATH)


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
