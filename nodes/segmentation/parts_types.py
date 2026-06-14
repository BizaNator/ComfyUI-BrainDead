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


def _fuzzy_match(pattern: str, tag_lower: str) -> bool:
    """True if all words in pattern appear (as substrings) in tag_lower.

    Single-word patterns: simple substring check.
    Multi-word patterns: every word must be present — allows "left sneaker"
    to match "left blue sneaker" even though the words aren't consecutive.
    """
    if " " not in pattern:
        return pattern in tag_lower
    return all(w in tag_lower for w in pattern.split())


class CategoryTable:
    """Result of parse_category_table().

    Lookup order per tag:
      1. Exact match (case-sensitive, O(1)).
      2. Substring match via ~ lines (case-insensitive).
         Among all matching ~patterns, the longest wins (most-specific).
      3. Default supplied by caller (passthrough: tag as slug, region='').
    """

    __slots__ = ("_exact", "_fuzzy")

    def __init__(self, exact: dict, fuzzy: list):
        self._exact = exact
        # Sort descending by pattern length so longest wins when multiple match.
        self._fuzzy: list[tuple[str, tuple[str, str]]] = sorted(
            fuzzy, key=lambda kv: len(kv[0]), reverse=True
        )

    def get(self, tag: str, default: tuple = ("", "")) -> tuple[str, str]:
        if tag in self._exact:
            return self._exact[tag]
        tag_lower = tag.lower()
        for pattern, value in self._fuzzy:
            if _fuzzy_match(pattern, tag_lower):
                return value
        return default

    def __bool__(self) -> bool:
        return bool(self._exact or self._fuzzy)

    def __len__(self) -> int:
        return len(self._exact) + len(self._fuzzy)


def parse_category_table(text: str) -> CategoryTable:
    """Parse a category table into a CategoryTable.

    Line formats:
      tag = slug:region       exact match (case-sensitive)
      ~pattern = slug:region  substring match (case-insensitive); longest wins
      tag = slug              slug only, region=''
      # comment               ignored
      blank lines             ignored

    Empty / all-comment text → empty CategoryTable (passthrough mode).
    """
    exact: dict[str, tuple[str, str]] = {}
    fuzzy: list[tuple[str, tuple[str, str]]] = []
    for line in (text or "").splitlines():
        line = line.split("#")[0].strip()
        if not line or "=" not in line:
            continue
        is_fuzzy = line.startswith("~")
        if is_fuzzy:
            line = line[1:].strip()
        tag, rest = line.split("=", 1)
        tag = tag.strip()
        rest = rest.strip()
        if not tag:
            continue
        if ":" in rest:
            slug, region = rest.split(":", 1)
            value: tuple[str, str] = (slug.strip(), region.strip())
        elif rest:
            value = (rest, "")
        else:
            continue
        if is_fuzzy:
            fuzzy.append((tag.lower(), value))
        else:
            exact[tag] = value
    return CategoryTable(exact, fuzzy)


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
