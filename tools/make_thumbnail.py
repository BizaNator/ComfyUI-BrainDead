#!/usr/bin/env python3
"""
Canonical BrainDead workflow-template thumbnail generator.

Produces the consistent 1180x680 "card" style used for every ComfyUI-BrainDead
example_workflows thumbnail, so any agent makes identical-looking thumbnails.

Usage:
    python3 tools/make_thumbnail.py <out.jpg> '<json-config>'
    python3 tools/make_thumbnail.py <out.jpg> --file config.json

Config (JSON):
    {
      "title":    "Background Removal",          # big heading (required)
      "subtitle": "SAM3 + pymatting alpha matting",   # one line under title
      "bullets":  ["Load Image -> BD Remove Background",
                   "SAM3 segments the subject", ...],  # 3-7 short lines
      "chips":    ["albedo","normal","roughness","metallic"],  # optional tag chips
      "footnote": "Models: SAM3 (auto-download)"     # optional small grey line above footer
    }

Rules baked in (do NOT reimplement per-thumbnail — call this script):
  - 1180x680, dark bg, purple BrainDead wordmark (drawn dot, NO emoji glyphs —
    emoji render as tofu boxes in DejaVu, so all text is sanitized to latin-1).
  - JPEG quality 88. Footer always shows the BrainDead links line.
"""
import json
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1180, 680
BG = (24, 24, 30)
ACCENT = (170, 120, 255)
TITLE = (240, 240, 245)
SUB = (200, 200, 210)
BODY = (186, 186, 196)
GREY = (150, 150, 160)
FOOTER = "BrainDeadGuild.com  ·  BrainDead.TV  ·  github.com/BizaNator/ComfyUI-BrainDead"
_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf"


def _font(size, bold=False):
    try:
        return ImageFont.truetype(_FONT % ("-Bold" if bold else ""), size)
    except Exception:
        return ImageFont.load_default()


def _clean(s):
    """Drop glyphs DejaVu can't render (emoji -> tofu). Keep a few nice symbols."""
    keep = {"→": "->", "·": "·", "—": "-", "–": "-"}
    out = []
    for ch in str(s):
        if ch in keep:
            out.append(keep[ch])
        elif ord(ch) < 0x250:  # latin + latin-1 supplement + latin extended-A
            out.append(ch)
        # else: drop (emoji, CJK, etc.)
    return "".join(out)


# A small palette so chips look distinct without per-call colors.
_CHIP_COLORS = [(210, 150, 110), (120, 120, 230), (150, 150, 155), (90, 90, 100),
                (110, 190, 140), (200, 140, 200), (200, 190, 110), (120, 200, 210)]


def make(out_path, cfg):
    title = _clean(cfg.get("title", "BrainDead"))
    subtitle = _clean(cfg.get("subtitle", ""))
    bullets = [_clean(b) for b in cfg.get("bullets", [])][:8]
    chips = [_clean(c) for c in cfg.get("chips", [])][:6]
    footnote = _clean(cfg.get("footnote", ""))

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    x = 64

    # top accent bar + wordmark
    d.rectangle([0, 0, W, 6], fill=ACCENT)
    d.ellipse([x, 58, x + 24, 82], fill=ACCENT)
    d.text((x + 36, 52), "BrainDead", font=_font(34, True), fill=ACCENT)

    # title + subtitle
    d.text((x, 104), title, font=_font(56, True), fill=TITLE)
    if subtitle:
        d.text((x, 176), subtitle, font=_font(30, True), fill=SUB)

    # bullets
    y = 244
    fb = _font(23)
    for b in bullets:
        d.ellipse([x, y + 9, x + 9, y + 18], fill=ACCENT)
        d.text((x + 22, y), b, font=fb, fill=BODY)
        y += 40

    # optional footnote line
    if footnote:
        d.text((x, y + 6), footnote, font=_font(20), fill=GREY)

    # optional chips row (just above footer)
    if chips:
        cx = x
        for i, c in enumerate(chips):
            w = 22 + len(c) * 11
            col = _CHIP_COLORS[i % len(_CHIP_COLORS)]
            d.rounded_rectangle([cx, H - 116, cx + w, H - 74], 8, fill=col)
            d.text((cx + 14, H - 107), c, font=_font(20, True), fill=(20, 20, 25))
            cx += w + 18

    # footer
    d.rectangle([0, H - 56, W, H], fill=(16, 16, 20))
    d.text((x, H - 42), FOOTER, font=_font(20), fill=GREY)

    img.save(out_path, "JPEG", quality=88)
    return out_path


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    out = sys.argv[1]
    if sys.argv[2] == "--file":
        cfg = json.load(open(sys.argv[3]))
    else:
        cfg = json.loads(sys.argv[2])
    print("wrote", make(out, cfg))


if __name__ == "__main__":
    main()
