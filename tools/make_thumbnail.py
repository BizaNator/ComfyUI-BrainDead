#!/usr/bin/env python3
"""
Canonical BrainDead workflow-template thumbnail generator.

Produces the consistent 1180x680 "card" thumbnail used for every
ComfyUI-BrainDead example_workflows template, so every agent makes identical
thumbnails that ACTUALLY SHOW in the ComfyUI template browser (jpg only,
sanitized text, correct name).

It combines two looks:
  * the clean BrainDead card (accent bar, wordmark, title, subtitle, bullets,
    chips, footer) — always on top, always readable; and
  * a faded BACKGROUND that "shows the actual nodes":
      - auto-drawn stylized node graph from the workflow JSON (default: the
        sibling `<name>.json` next to the output `<name>.jpg`), OR
      - a provided screenshot / result image (`"background": "path.jpg"`).

THE RENDERER LIVES IN utils/thumbnail.py. This file is only the CLI. The node
BD_SaveWorkflowImage renders from that same module, so the card cannot drift
between the two. Do not reimplement drawing here.

Usage:
    python3 tools/make_thumbnail.py <out.jpg> '<json-config>'
    python3 tools/make_thumbnail.py <out.jpg> --file config.json

Config (JSON):
    {
      "title":    "Background Removal",                 # required
      "subtitle": "SAM3 + pymatting alpha matting",
      "bullets":  ["Load Image -> BD Remove Background", ...],   # 3-7 short lines
      "chips":    ["RGBA","white BG","black BG"],        # optional tag chips
      "footnote": "Models: SAM3 (auto-download)",        # optional grey line
      "workflow": "example_workflows/background_removal.json",  # optional override
      "background": "/path/to/result.jpg",   # optional image bg (wins over graph)
      "no_graph": false                      # set true for a pure clean card
    }

Also honoured (shared with the node, all optional):
      width, height, quality, footer, accent, wordmark_color, bg,
      logo_path, logo_height, logo_margin, logo_corner, no_logo

Rules baked in (do NOT reimplement per-thumbnail — call this script):
  - 1180x680, JPEG q88, `.jpg` only, name == sibling `<name>.json`.
  - Text is sanitized (emoji/CJK dropped — DejaVu renders them as tofu boxes).
"""
import importlib.util
import json
import os
import sys

# Load utils/thumbnail.py by path: importing the `utils` PACKAGE would execute its
# __init__, which pulls in ComfyUI's folder_paths and fails outside ComfyUI. The
# CLI has to keep working from a plain shell.
_MOD = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "utils", "thumbnail.py"))
_spec = importlib.util.spec_from_file_location("bd_thumbnail", _MOD)
thumbnail = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(thumbnail)

make = thumbnail.make


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    out = sys.argv[1]
    cfg = json.load(open(sys.argv[3])) if sys.argv[2] == "--file" else json.loads(sys.argv[2])
    print("wrote", make(out, cfg))


if __name__ == "__main__":
    main()
