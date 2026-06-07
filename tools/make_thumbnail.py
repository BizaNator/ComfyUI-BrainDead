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

Rules baked in (do NOT reimplement per-thumbnail — call this script):
  - 1180x680, JPEG q88, `.jpg` only, name == sibling `<name>.json`.
  - Text is sanitized (emoji/CJK dropped — DejaVu renders them as tofu boxes).
"""
import json
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1180, 680
BG = (24, 24, 30)
ACCENT = (170, 120, 255)          # purple — dot, top bar, bullets, BD node title bars
WORDMARK = (22, 163, 74)          # braindead.tv green (#16a34a) — "BrainDead" text only
TITLE = (240, 240, 245)
SUB = (200, 200, 210)
BODY = (188, 188, 198)
GREY = (150, 150, 160)
FOOTER = "BrainDeadGuild.com  ·  BrainDead.TV  ·  github.com/BizaNator/ComfyUI-BrainDead"
_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf"
# Brand watermark — bottom-right corner of every thumbnail. Lives in the repo
# (tools/assets/) so it's always available at build time.
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "bizanator_logo.png")
LOGO_H = 60          # rendered height (px); aspect preserved
LOGO_MARGIN = 18     # px from the right + bottom edges
_CHIP_COLORS = [(210, 150, 110), (120, 120, 230), (150, 150, 155), (90, 90, 100),
                (110, 190, 140), (200, 140, 200), (200, 190, 110), (120, 200, 210)]


def _font(size, bold=False):
    try:
        return ImageFont.truetype(_FONT % ("-Bold" if bold else ""), size)
    except Exception:
        return ImageFont.load_default()


def _clean(s):
    keep = {"→": "->", "·": "·", "—": "-", "–": "-"}
    out = []
    for ch in str(s):
        if ch in keep:
            out.append(keep[ch])
        elif ord(ch) < 0x250:
            out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------- node graph --
def _bezier(d, p0, p3, color, width=2):
    x1, y1 = p0
    x2, y2 = p3
    dx = max(40, abs(x2 - x1) * 0.4)
    c1 = (x1 + dx, y1)
    c2 = (x2 - dx, y2)
    pts = []
    for i in range(21):
        t = i / 20.0
        mt = 1 - t
        x = mt**3 * x1 + 3 * mt**2 * t * c1[0] + 3 * mt * t**2 * c2[0] + t**3 * x2
        y = mt**3 * y1 + 3 * mt**2 * t * c1[1] + 3 * mt * t**2 * c2[1] + t**3 * y2
        pts.append((x, y))
    d.line(pts, fill=color, width=width, joint="curve")


def _draw_graph(base, wf_path):
    """Draw a stylized LiteGraph-style node graph from a workflow JSON, faded
    behind the card. Returns True if drawn."""
    try:
        wf = json.load(open(wf_path))
        nodes = wf["nodes"]
    except Exception:
        return False
    nodes = [n for n in nodes if n.get("type") != "MarkdownNote"]
    if not nodes:
        return False

    def nsize(n):
        s = n.get("size", [200, 100])
        return (s.get("0", s[0]) if isinstance(s, dict) else s[0],
                s.get("1", s[1]) if isinstance(s, dict) else s[1])

    xs, ys, xe, ye = [], [], [], []
    for n in nodes:
        x, y = n["pos"][0], n["pos"][1]
        w, h = nsize(n)
        xs.append(x); ys.append(y); xe.append(x + w); ye.append(y + h)
    minx, miny, maxx, maxy = min(xs), min(ys), max(xe), max(ye)
    gw, gh = maxx - minx, maxy - miny

    # fit graph into the canvas with margins; bias slightly right so the left
    # text column stays clean.
    pad = 40
    avail_w, avail_h = W - 2 * pad, H - 2 * pad
    scale = min(avail_w / gw, avail_h / gh) if gw and gh else 1.0
    scale = min(scale, 0.9)
    off_x = pad + (avail_w - gw * scale) / 2 + 80
    off_y = pad + (avail_h - gh * scale) / 2

    def tx(x): return off_x + (x - minx) * scale
    def ty(y): return off_y + (y - miny) * scale

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    TITLE_H = 22 * scale + 6
    SLOT_H = 20 * scale + 4

    def port_pos(node, slot, is_input):
        x, y = node["pos"][0], node["pos"][1]
        w, _ = nsize(node)
        px = tx(x) if is_input else tx(x + w)
        py = ty(y) + TITLE_H + slot * SLOT_H + SLOT_H / 2
        return (px, py)

    byid = {n["id"]: n for n in nodes}
    # wires first (behind boxes)
    for L in wf.get("links", []):
        try:
            _, sn, ss, dn, ds, _t = L[:6]
            if sn not in byid or dn not in byid:
                continue
            p0 = port_pos(byid[sn], ss, False)
            p3 = port_pos(byid[dn], ds, True)
            _bezier(d, p0, p3, (90, 200, 200, 130), width=max(1, int(2 * scale)))
        except Exception:
            continue

    fnt = _font(max(9, int(15 * scale)), True)
    for n in nodes:
        x, y = tx(n["pos"][0]), ty(n["pos"][1])
        w, h = nsize(n)
        x2, y2 = x + w * scale, y + h * scale
        is_bd = str(n.get("type", "")).startswith("BD_")
        title_col = (130, 80, 175, 235) if is_bd else (70, 86, 120, 235)
        body_col = (40, 40, 52, 200)
        d.rounded_rectangle([x, y, x2, y2], radius=6, fill=body_col,
                            outline=(80, 80, 100, 200), width=1)
        d.rounded_rectangle([x, y, x2, y + TITLE_H], radius=6, fill=title_col)
        d.rectangle([x, y + TITLE_H - 6, x2, y + TITLE_H], fill=title_col)
        d.ellipse([x + 6, y + TITLE_H/2 - 4, x + 14, y + TITLE_H/2 + 4],
                  fill=(120, 220, 150, 255))
        title = _clean(n.get("title") or n.get("type", ""))
        maxc = max(4, int((w * scale - 28) / (9 * scale)))
        d.text((x + 20, y + 3), title[:maxc], font=fnt, fill=(235, 235, 240, 255))
        # ports
        for s, _ in enumerate(n.get("inputs", [])):
            p = port_pos(n, s, True)
            d.ellipse([p[0]-3, p[1]-3, p[0]+3, p[1]+3], fill=(150, 200, 210, 230))
        for s, _ in enumerate(n.get("outputs", [])):
            p = port_pos(n, s, False)
            d.ellipse([p[0]-3, p[1]-3, p[0]+3, p[1]+3], fill=(150, 200, 210, 230))

    # fade the graph so it reads as a background
    alpha = layer.split()[3].point(lambda a: int(a * 0.60))
    layer.putalpha(alpha)
    base.alpha_composite(layer)
    return True


def _draw_image_bg(base, img_path):
    try:
        bg = Image.open(img_path).convert("RGB")
    except Exception:
        return False
    # cover-fit
    r = max(W / bg.width, H / bg.height)
    bg = bg.resize((int(bg.width * r), int(bg.height * r)), Image.LANCZOS)
    bg = bg.crop((0, 0, W, H))
    dark = Image.new("RGBA", (W, H), (24, 24, 30, 150))
    base.alpha_composite(bg.convert("RGBA"))
    base.alpha_composite(dark)
    return True


def _left_scrim(base):
    """Left-to-right dark gradient so the text column stays readable over any bg."""
    scrim = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    px = scrim.load()
    for x in range(W):
        a = int(235 * max(0.0, 1 - (x / (W * 0.62))))  # opaque left -> clear ~62%
        for y in range(H):
            px[x, y] = (BG[0], BG[1], BG[2], a)
    base.alpha_composite(scrim)


# ---------------------------------------------------------------------- card --
def make(out_path, cfg):
    title = _clean(cfg.get("title", "BrainDead"))
    subtitle = _clean(cfg.get("subtitle", ""))
    bullets = [_clean(b) for b in cfg.get("bullets", [])][:8]
    chips = [_clean(c) for c in cfg.get("chips", [])][:6]
    footnote = _clean(cfg.get("footnote", ""))

    base = Image.new("RGBA", (W, H), BG + (255,))

    # background layer: image > graph > none
    drew_bg = False
    if cfg.get("background"):
        drew_bg = _draw_image_bg(base, cfg["background"])
    if not drew_bg and not cfg.get("no_graph"):
        wf = cfg.get("workflow")
        if not wf:
            cand = os.path.splitext(out_path)[0] + ".json"
            wf = cand if os.path.exists(cand) else None
        if wf:
            drew_bg = _draw_graph(base, wf)
    if drew_bg:
        _left_scrim(base)

    d = ImageDraw.Draw(base)
    x = 64
    d.rectangle([0, 0, W, 6], fill=ACCENT + (255,))
    d.ellipse([x, 58, x + 24, 82], fill=ACCENT + (255,))
    d.text((x + 36, 52), "BrainDead", font=_font(34, True), fill=WORDMARK + (255,))
    d.text((x, 104), title, font=_font(56, True), fill=TITLE + (255,))
    if subtitle:
        d.text((x, 176), subtitle, font=_font(30, True), fill=SUB + (255,))

    y = 244
    fb = _font(23)
    for b in bullets:
        d.ellipse([x, y + 9, x + 9, y + 18], fill=ACCENT + (255,))
        d.text((x + 22, y), b, font=fb, fill=BODY + (255,))
        y += 40
    if footnote:
        d.text((x, y + 6), footnote, font=_font(20), fill=GREY + (255,))

    if chips:
        cx = x
        for i, c in enumerate(chips):
            w = 22 + len(c) * 11
            col = _CHIP_COLORS[i % len(_CHIP_COLORS)]
            d.rounded_rectangle([cx, H - 116, cx + w, H - 74], 8, fill=col + (255,))
            d.text((cx + 14, H - 107), c, font=_font(20, True), fill=(20, 20, 25, 255))
            cx += w + 18

    d.rectangle([0, H - 56, W, H], fill=(16, 16, 20, 255))
    d.text((x, H - 42), FOOTER, font=_font(20), fill=GREY + (255,))

    _watermark(base)
    base.convert("RGB").save(out_path, "JPEG", quality=88)
    return out_path


def _watermark(base):
    """Composite the BizaNator brand logo in the bottom-right corner."""
    try:
        logo = Image.open(LOGO_PATH).convert("RGBA")
    except Exception:
        return
    w = max(1, round(logo.width * LOGO_H / logo.height))
    logo = logo.resize((w, LOGO_H), Image.LANCZOS)
    base.alpha_composite(logo, dest=(W - w - LOGO_MARGIN, H - LOGO_H - LOGO_MARGIN))


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    out = sys.argv[1]
    cfg = json.load(open(sys.argv[3])) if sys.argv[2] == "--file" else json.loads(sys.argv[2])
    print("wrote", make(out, cfg))


if __name__ == "__main__":
    main()
