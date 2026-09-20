"""Shared BrainDead thumbnail/card renderer.

Lifted verbatim out of tools/make_thumbnail.py so the CLI and the
BD_SaveWorkflowImage node render from ONE implementation. They had to be one
thing: a node that reimplemented the card would drift from the template
thumbnails the moment either side was touched, which is exactly how the UI and
API workflow exports ended up disagreeing.

Behaviour for the existing CLI is unchanged -- every default here is the value it
used. What is new is that the sizes, colours, footer, watermark placement and the
graph source are parameters rather than module constants, because the node exposes
them as inputs.

Two accepted background sources, in priority order, same as before:
  * an explicit image (a screenshot, a render, a photo of a tree -- anything)
  * an auto-drawn stylized LiteGraph-ish node graph from a workflow

`draw_graph` takes a path OR an already-loaded dict, since a node receives the
graph from ComfyUI's extra_pnginfo rather than off disk.
"""
import json
import os

from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------------ defaults --
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

# Brand watermark. Lives in the repo so it is always available at build time.
_HERE = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.normpath(os.path.join(_HERE, "..", "tools", "assets", "bizanator_logo.png"))
LOGO_H = 60          # rendered height (px); aspect preserved
LOGO_MARGIN = 18     # px from the edges
LOGO_CORNER = "bottom-right"

_CHIP_COLORS = [(210, 150, 110), (120, 120, 230), (150, 150, 155), (90, 90, 100),
                (110, 190, 140), (200, 140, 200), (200, 190, 110), (120, 200, 210)]


def font(size, bold=False):
    try:
        return ImageFont.truetype(_FONT % ("-Bold" if bold else ""), size)
    except Exception:
        return ImageFont.load_default()


def clean(s):
    """Drop glyphs DejaVu renders as tofu boxes (emoji, CJK), map a few to ASCII."""
    keep = {"→": "->", "·": "·", "—": "-", "–": "-"}
    out = []
    for ch in str(s):
        if ch in keep:
            out.append(keep[ch])
        elif ord(ch) < 0x250:
            out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------- node graph --
def bezier(d, p0, p3, color, width=2):
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


def _load_workflow(src):
    """Accept a path, a JSON string, or an already-parsed dict."""
    if isinstance(src, dict):
        return src
    if not src:
        return None
    try:
        if isinstance(src, str) and src.lstrip().startswith("{"):
            return json.loads(src)
        return json.load(open(src))
    except Exception:
        return None


def draw_graph(base, workflow, size=None, accent_bd=(130, 80, 175, 235),
               fade=0.60, bias_x=80):
    """Stylized LiteGraph-style graph behind the card. True if drawn.

    `workflow` may be a path, a JSON string, or a dict — a node gets the graph
    from extra_pnginfo, never off disk.
    """
    cw, ch = size or (W, H)
    wf = _load_workflow(workflow)
    try:
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

    # fit with margins; bias right so the left text column stays clean
    pad = 40
    avail_w, avail_h = cw - 2 * pad, ch - 2 * pad
    scale = min(avail_w / gw, avail_h / gh) if gw and gh else 1.0
    scale = min(scale, 0.9)
    off_x = pad + (avail_w - gw * scale) / 2 + bias_x
    off_y = pad + (avail_h - gh * scale) / 2

    def tx(x): return off_x + (x - minx) * scale
    def ty(y): return off_y + (y - miny) * scale

    layer = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
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
    for L in wf.get("links", []):
        try:
            _, sn, ss, dn, ds, _t = L[:6]
            if sn not in byid or dn not in byid:
                continue
            p0 = port_pos(byid[sn], ss, False)
            p3 = port_pos(byid[dn], ds, True)
            bezier(d, p0, p3, (90, 200, 200, 130), width=max(1, int(2 * scale)))
        except Exception:
            continue

    fnt = font(max(9, int(15 * scale)), True)
    for n in nodes:
        x, y = tx(n["pos"][0]), ty(n["pos"][1])
        w, h = nsize(n)
        x2, y2 = x + w * scale, y + h * scale
        is_bd = str(n.get("type", "")).startswith("BD_")
        title_col = accent_bd if is_bd else (70, 86, 120, 235)
        body_col = (40, 40, 52, 200)
        d.rounded_rectangle([x, y, x2, y2], radius=6, fill=body_col,
                            outline=(80, 80, 100, 200), width=1)
        d.rounded_rectangle([x, y, x2, y + TITLE_H], radius=6, fill=title_col)
        d.rectangle([x, y + TITLE_H - 6, x2, y + TITLE_H], fill=title_col)
        d.ellipse([x + 6, y + TITLE_H / 2 - 4, x + 14, y + TITLE_H / 2 + 4],
                  fill=(120, 220, 150, 255))
        title = clean(n.get("title") or n.get("type", ""))
        maxc = max(4, int((w * scale - 28) / (9 * scale)))
        d.text((x + 20, y + 3), title[:maxc], font=fnt, fill=(235, 235, 240, 255))
        for s, _ in enumerate(n.get("inputs", [])):
            p = port_pos(n, s, True)
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=(150, 200, 210, 230))
        for s, _ in enumerate(n.get("outputs", [])):
            p = port_pos(n, s, False)
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=(150, 200, 210, 230))

    a = layer.split()[3].point(lambda v: int(v * fade))
    layer.putalpha(a)
    base.alpha_composite(layer)
    return True


def draw_image_bg(base, img, size=None, dim=150):
    """Cover-fit an image behind the card. `img` may be a path or a PIL Image."""
    cw, ch = size or (W, H)
    try:
        bg = img if isinstance(img, Image.Image) else Image.open(img)
        bg = bg.convert("RGB")
    except Exception:
        return False
    r = max(cw / bg.width, ch / bg.height)
    bg = bg.resize((int(bg.width * r), int(bg.height * r)), Image.LANCZOS)
    bg = bg.crop((0, 0, cw, ch))
    base.alpha_composite(bg.convert("RGBA"))
    base.alpha_composite(Image.new("RGBA", (cw, ch), BG + (dim,)))
    return True


def left_scrim(base, size=None, bg=BG):
    """Left-to-right dark gradient so the text column stays readable over any bg."""
    cw, ch = size or (W, H)
    scrim = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    px = scrim.load()
    for x in range(cw):
        a = int(235 * max(0.0, 1 - (x / (cw * 0.62))))
        for y in range(ch):
            px[x, y] = (bg[0], bg[1], bg[2], a)
    base.alpha_composite(scrim)


def watermark(base, size=None, logo_path=None, logo_h=LOGO_H,
              margin=LOGO_MARGIN, corner=LOGO_CORNER):
    """Composite the brand logo into one corner. Silently skips if unreadable."""
    cw, ch = size or (W, H)
    src = logo_path or LOGO_PATH
    if not src:
        return False
    try:
        # a PIL Image (piped in as tensors by a node) or a path on disk
        logo = src.convert("RGBA") if isinstance(src, Image.Image) else Image.open(src).convert("RGBA")
    except Exception:
        return False
    w = max(1, round(logo.width * logo_h / logo.height))
    logo = logo.resize((w, logo_h), Image.LANCZOS)
    x = margin if "left" in corner else cw - w - margin
    y = margin if "top" in corner else ch - logo_h - margin
    base.alpha_composite(logo, dest=(int(x), int(y)))
    return True



def _fit(d, text, fnt, max_w):
    """Truncate with an ellipsis so text cannot run off the card.

    Neither the original CLI nor the first cut of the node clipped anything, so a
    long subtitle -- which is exactly what stamp_version produces when it appends a
    workflow name -- simply ran past the right edge and off the image.
    """
    if not text:
        return text
    try:
        if d.textlength(text, font=fnt) <= max_w:
            return text
    except Exception:
        return text
    ell = "..."
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if d.textlength(text[:mid] + ell, font=fnt) <= max_w:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + ell


# ---------------------------------------------------------------------- card --
def render(cfg, out_path=None):
    """Render the card. Returns a PIL RGB Image; also writes it if out_path given.

    cfg keys (all optional except title):
      title, subtitle, bullets[], chips[], footnote, footer
      wordmark      top line, default "BrainDead"; empty hides it and the dot
      wordmark_size point size for it, default 34
      workflow      path | JSON string | dict   -> auto-drawn graph background
      background    path | PIL.Image            -> composited OVER the graph
      image_blend   0..1  1.0 = image only (default), 0.0 = graph only, 0.5 = both
      no_graph      bool
      width, height, quality
      accent, wordmark_color, bg               (r,g,b) tuples
      logo_path, logo_height, logo_margin, logo_corner, no_logo
    """
    cw = int(cfg.get("width") or W)
    chh = int(cfg.get("height") or H)
    size = (cw, chh)
    bg = tuple(cfg.get("bg") or BG)
    accent = tuple(cfg.get("accent") or ACCENT)
    wordmark = tuple(cfg.get("wordmark_color") or WORDMARK)
    footer = cfg.get("footer", FOOTER)

    title = clean(cfg.get("title", "BrainDead"))
    subtitle = clean(cfg.get("subtitle", ""))
    bullets = [clean(b) for b in (cfg.get("bullets") or [])][:8]
    chips = [clean(c) for c in (cfg.get("chips") or [])][:6]
    footnote = clean(cfg.get("footnote", ""))

    base = Image.new("RGBA", size, bg + (255,))

    # Background. The graph is NOT an either/or fallback: when a workflow is
    # available it always draws, and a supplied image composites over it at
    # image_blend. Connecting an image then enriches the card instead of silently
    # discarding the graph -- which is exactly what happened when a LoadImage was
    # wired into the example template and the graph vanished from its own thumbnail.
    #   image_blend 1.0 = image only, 0.0 = graph only, 0.5 = both.
    drew = False
    wf = cfg.get("workflow")
    if not wf and out_path:
        cand = os.path.splitext(out_path)[0] + ".json"
        wf = cand if os.path.exists(cand) else None
    if wf and not cfg.get("no_graph"):
        drew = draw_graph(base, wf, size)

    img = cfg.get("background")
    if img is not None:
        blend = cfg.get("image_blend")
        blend = 1.0 if blend is None else max(0.0, min(1.0, float(blend)))
        if blend > 0.0:
            if drew and blend < 1.0:
                over = Image.new("RGBA", size, (0, 0, 0, 0))
                if draw_image_bg(over, img, size):
                    a = over.split()[3].point(lambda v: int(v * blend))
                    over.putalpha(a)
                    base.alpha_composite(over)
            else:
                drew = draw_image_bg(base, img, size) or drew
    if drew:
        left_scrim(base, size, bg)

    d = ImageDraw.Draw(base)
    x = 64
    d.rectangle([0, 0, cw, 6], fill=accent + (255,))
    mark = cfg.get("wordmark")
    mark = "BrainDead" if mark is None else str(mark)
    if mark:
        ms = int(cfg.get("wordmark_size") or 34)
        d.ellipse([x, 58, x + 24, 82], fill=accent + (255,))
        d.text((x + 36, 52), clean(mark), font=font(ms, True), fill=wordmark + (255,))
    avail = cw - x - 40
    d.text((x, 104), _fit(d, title, font(56, True), avail), font=font(56, True), fill=TITLE + (255,))
    if subtitle:
        d.text((x, 176), _fit(d, subtitle, font(30, True), avail), font=font(30, True), fill=SUB + (255,))

    y = 244
    fb = font(23)
    for b in bullets:
        d.ellipse([x, y + 9, x + 9, y + 18], fill=accent + (255,))
        d.text((x + 22, y), b, font=fb, fill=BODY + (255,))
        y += 40
    if footnote:
        d.text((x, y + 6), footnote, font=font(20), fill=GREY + (255,))

    if chips:
        cx = x
        for i, c in enumerate(chips):
            cf = font(20, True)
            try:
                w = int(d.textlength(c, font=cf)) + 28
            except Exception:
                w = 22 + len(c) * 11
            col = _CHIP_COLORS[i % len(_CHIP_COLORS)]
            d.rounded_rectangle([cx, chh - 116, cx + w, chh - 74], 8, fill=col + (255,))
            d.text((cx + 14, chh - 107), c, font=cf, fill=(20, 20, 25, 255))
            cx += w + 18

    d.rectangle([0, chh - 56, cw, chh], fill=(16, 16, 20, 255))
    if footer:
        d.text((x, chh - 42), clean(footer), font=font(20), fill=GREY + (255,))

    if not cfg.get("no_logo"):
        watermark(base, size,
                  logo_path=cfg.get("logo_path"),
                  logo_h=int(cfg.get("logo_height") or LOGO_H),
                  margin=int(cfg.get("logo_margin") or LOGO_MARGIN),
                  corner=cfg.get("logo_corner") or LOGO_CORNER)

    img = base.convert("RGB")
    if out_path:
        img.save(out_path, "JPEG", quality=int(cfg.get("quality") or 88))
    return img


def make(out_path, cfg):
    """Back-compat entry point for tools/make_thumbnail.py. Returns the path."""
    render(cfg, out_path)
    return out_path
