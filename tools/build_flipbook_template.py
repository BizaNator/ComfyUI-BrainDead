#!/usr/bin/env python3
"""
Build the BD-atlas_flipbook template — the *proper* home for BD Atlas Pack:
tile a sequence of frames (or a bunch of packed textures) into a sprite sheet.

  4x Load Image -> ImageBatch chain -> BD Atlas Pack (grid)  -> sheet + UV layout
                                    -> BD Atlas Pack (strip)  -> 1-row sprite strip

Pulls live object_info so widgets_values stay aligned. Honors the BrainDead template
styling convention (circled-number titles + 'ℹ️ About — 🧠 BrainDead …').
"""
import os, json, urllib.request

SRV = "http://127.0.0.1:8188"
EW = os.path.join(os.path.dirname(__file__), "..", "example_workflows")
CONN = {"IMAGE", "MASK", "LATENT", "LOTUS2_MODEL", "CLIP", "VAE", "MODEL", "*"}

_cache = {}
def oi(t):
    if t not in _cache:
        _cache[t] = json.loads(urllib.request.urlopen(f"{SRV}/object_info/{t}", timeout=20).read())[t]
    return _cache[t]

def widget_defaults(t):
    d = oi(t); order = d['input_order'].get('required', []) + d['input_order'].get('optional', [])
    alld = {**d['input'].get('required', {}), **d['input'].get('optional', {})}
    out = []
    for k in order:
        spec = alld[k]; typ = spec[0]
        if isinstance(typ, str) and typ in CONN:
            continue
        if isinstance(typ, list):
            default = (spec[1].get('default') if len(spec) > 1 and isinstance(spec[1], dict) else None)
            if default is None: default = typ[0] if typ else None
        else:
            default = spec[1].get('default') if len(spec) > 1 and isinstance(spec[1], dict) else None
        out.append((k, default))
    return out

def out_names(t): return oi(t).get('output_name', [])
def in_conn_names(t):
    d = oi(t); order = d['input_order'].get('required', []) + d['input_order'].get('optional', [])
    alld = {**d['input'].get('required', {}), **d['input'].get('optional', {})}
    return [k for k in order if isinstance(alld[k][0], str) and alld[k][0] in CONN]

nid = 0; lid = 0; nodes = []; links = []
def add(type_, pos, size, overrides=None, title=None):
    global nid; nid += 1
    if type_ == "MarkdownNote":
        n = {"id": nid, "type": type_, "pos": list(pos), "size": list(size), "flags": {},
             "order": nid - 1, "mode": 0, "inputs": [], "outputs": [],
             "properties": {}, "widgets_values": [overrides["__md__"]]}
        if title: n["title"] = title
        nodes.append(n); return n
    widgets = [overrides.get(k, d) if overrides else d for k, d in widget_defaults(type_)]
    conns = in_conn_names(type_)
    ispec = {**oi(type_)['input'].get('required', {}), **oi(type_)['input'].get('optional', {})}
    n = {"id": nid, "type": type_, "pos": list(pos), "size": list(size), "flags": {},
         "order": nid - 1, "mode": 0,
         "inputs": [{"name": c, "type": ispec[c][0], "link": None} for c in conns],
         "outputs": [{"name": o, "type": oi(type_)['output'][i], "links": None, "slot_index": i}
                     for i, o in enumerate(out_names(type_))],
         "properties": {"Node name for S&R": type_}, "widgets_values": widgets}
    if title: n["title"] = title
    nodes.append(n); return n

def link(src, oname, dst, iname):
    global lid; lid += 1
    so = [o['name'] for o in src['outputs']].index(oname)
    di = [i['name'] for i in dst['inputs']].index(iname)
    links.append([lid, src['id'], so, dst['id'], di, src['outputs'][so]['type']])
    o = src['outputs'][so]; o['links'] = (o['links'] or []) + [lid]
    dst['inputs'][di]['link'] = lid

C = lambda i: chr(0x2460 + i - 1)

# ── frames ──
frames = [add("LoadImage", (40, 40 + i * 230), (300, 200), {"image": "example.png", "__img2__": "image"},
              title=f"{C(i + 1)} Frame {i + 1}") for i in range(4)]
# LoadImage carries a trailing upload-widget entry; ensure 2 values
for f in frames:
    f["widgets_values"] = ["example.png", "image"]

# ── batch the frames (ImageBatch only takes 2 → chain) ──
b1 = add("ImageBatch", (380, 120), (220, 60), title="Batch 1+2")
b2 = add("ImageBatch", (380, 360), (220, 60), title="Batch +3")
b3 = add("ImageBatch", (380, 600), (220, 60), title="Batch +4")
link(frames[0], "IMAGE", b1, "image1"); link(frames[1], "IMAGE", b1, "image2")
link(b1, "IMAGE", b2, "image1");        link(frames[2], "IMAGE", b2, "image2")
link(b2, "IMAGE", b3, "image1");        link(frames[3], "IMAGE", b3, "image2")

# ── two atlas configs off the same frame batch ──
grid = add("BD_AtlasPack", (660, 60), (300, 320),
           {"columns": 0, "padding": 4, "fit_mode": "contain", "background_hex": "#000000"},
           title="⑤ BD Atlas Pack — grid (auto 2×2)")
strip = add("BD_AtlasPack", (660, 460), (300, 320),
            {"columns": 4, "rows": 1, "padding": 0, "fit_mode": "contain", "background_hex": "#000000"},
            title="⑥ BD Atlas Pack — sprite strip (1 row)")
link(b3, "IMAGE", grid, "images")
link(b3, "IMAGE", strip, "images")

save = add("SaveImage", (1010, 60), (300, 270), {"filename_prefix": "atlas/flipbook_sheet"},
           title="⑦ Save Atlas Sheet")
pv_grid = add("PreviewImage", (1010, 360), (300, 240), title="⑧ Grid Sheet")
pv_strip = add("PreviewImage", (1010, 620), (300, 200), title="⑨ Sprite Strip")
pv_layout = add("PreviewAny", (1010, 840), (300, 200), title="⑩ Layout JSON (per-cell UV rects)")
link(grid, "atlas", save, "images")
link(grid, "atlas", pv_grid, "images")
link(strip, "atlas", pv_strip, "images")
link(grid, "layout", pv_layout, "source")

md = ("## Atlas / Flipbook Packing\n\n"
      "The proper home for **BD Atlas Pack** — tile a sequence of frames (or a bunch of "
      "**packed** textures) into one sprite sheet.\n\n"
      "- 4× **Load Image** → **Image Batch** chain → one IMAGE batch of frames\n"
      "- **BD Atlas Pack (grid)** — auto cols×rows (4 → 2×2), padding\n"
      "- **BD Atlas Pack (strip)** — `rows=1` → a horizontal sprite strip for UV-animated "
      "shaders / flipbook materials\n"
      "- **layout** output = JSON with per-cell pixel + normalised-**UV** rects for the engine\n\n"
      "Swap Load Image for any frame source (or wire packed-texture outputs into the "
      "`image_1..image_8` slots instead of the batch).")
add("MarkdownNote", (40, 980), (520, 240), {"__md__": md},
    title="ℹ️ About — 🧠 BrainDead Atlas / Flipbook")

wf = {"id": "bd-atlas-flipbook", "revision": 0, "last_node_id": nid, "last_link_id": lid,
      "nodes": nodes, "links": links, "groups": [], "config": {}, "extra": {}, "version": 0.4}
out_path = os.path.join(EW, "BD-atlas_flipbook.json")
json.dump(wf, open(out_path, "w"), indent=2)
print(f"wrote {out_path}: {nid} nodes, {lid} links")
