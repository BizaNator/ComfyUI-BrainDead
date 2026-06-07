#!/usr/bin/env python3
"""
Build the BD-game_engine_packing template:
  Load -> Remove BG -> SAM3 isolate 3 parts -> per part: greyscale -> normalize luma
  -> center median luma  =>  BD Pack Channels (R/G/B) AND BD Atlas Pack (grid).

Pulls live input_order + defaults from object_info so widgets_values always align with
the current nodes. Re-run any time a node's inputs change. Honors the BrainDead template
styling convention (circled-number titles + 'ℹ️ About — 🧠 BrainDead …').
"""
import json, math, os, sys, urllib.request

SRV = "http://127.0.0.1:8188"
EW = os.path.join(os.path.dirname(__file__), "..", "example_workflows")
CONN = {"IMAGE", "MASK", "LATENT", "LOTUS2_MODEL", "CLIP", "VAE", "MODEL"}

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
    if type_ == "MarkdownNote":   # frontend-only node, not in object_info
        n = {"id": nid, "type": type_, "pos": list(pos), "size": list(size), "flags": {},
             "order": nid - 1, "mode": 0, "inputs": [], "outputs": [],
             "properties": {}, "widgets_values": [overrides["__md__"]]}
        if title: n["title"] = title
        nodes.append(n); return n
    widgets = [overrides.get(k, d) if overrides else d for k, d in widget_defaults(type_)]
    conns = in_conn_names(type_)
    n = {"id": nid, "type": type_, "pos": list(pos), "size": list(size), "flags": {},
         "order": nid - 1, "mode": 0,
         "inputs": [{"name": c, "type": oi(type_)['input'].get('required', {}).get(c, oi(type_)['input'].get('optional', {}).get(c))[0], "link": None} for c in conns],
         "outputs": [{"name": o, "type": oi(type_)['output'][i], "links": None, "slot_index": i}
                     for i, o in enumerate(out_names(type_))],
         "properties": {"Node name for S&R": type_}}
    if title: n["title"] = title
    if type_ != "MarkdownNote":
        n["widgets_values"] = widgets
    else:
        n["widgets_values"] = [overrides["__md__"]]
    nodes.append(n); return n

def link(src, oname, dst, iname):
    global lid; lid += 1
    so = [o['name'] for o in src['outputs']].index(oname)
    di = [i['name'] for i in dst['inputs']].index(iname)
    typ = src['outputs'][so]['type']
    links.append([lid, src['id'], so, dst['id'], di, typ])
    o = src['outputs'][so]
    o['links'] = (o['links'] or []) + [lid]
    dst['inputs'][di]['link'] = lid
    return lid

C = lambda i: chr(0x2460 + i - 1)  # ① ...

# ── column layout ──
load = add("LoadImage", (40, 80), (300, 314), {"image": "example.png"}, title="① Load Image")
rbg  = add("BD_RemoveBackground", (380, 80), (320, 200),
           {"prompts": "subject\nperson\nforeground", "crop_to_content": False,
            "edge_refine": "guided"}, title="② BD Remove Background")
sam  = add("BD_SAM3MultiPrompt", (740, 80), (320, 200),
           {"prompts": "lips\nteeth\ntongue"}, title="③ SAM3 Isolate Parts (lips/teeth/tongue)")

# 3 parallel object chains R(lips) G(teeth) B(tongue)
chan = ["R · lips", "G · teeth", "B · tongue"]
ys = [40, 300, 560]
idx_nodes, grey_nodes, norm_nodes, cen_nodes = [], [], [], []
for k, (label, y) in enumerate(zip(chan, ys)):
    mi = add("BD_MaskBatchIndex", (1100, y), (250, 110), {"index": k},
             title=f"{C(4 + k)} Pick mask [{k}] ({label})")
    gs = add("BD_ImageToGreyscale", (1380, y), (250, 130),
             {"mode": "luminance", "mask_mode": "apply_within"},
             title=f"Greyscale ({label})")
    nm = add("BD_NormalizeLuma", (1660, y), (250, 180),
             {"apply_to_mask_only": True}, title=f"Normalize Luma ({label})")
    cm = add("BD_CenterMedianLuma", (1940, y), (250, 180),
             {"apply_to_mask_only": True, "calc_from_mask": True}, title=f"Center Median ({label})")
    idx_nodes.append(mi); grey_nodes.append(gs); norm_nodes.append(nm); cen_nodes.append(cm)

pack = add("BD_PackChannels", (2220, 80), (300, 260), {}, title="⑦ BD Pack Channels (R/G/B)")
atlas = add("BD_AtlasPack", (2220, 420), (300, 320),
            {"columns": 3, "fit_mode": "contain", "padding": 8, "background_hex": "#000000"},
            title="⑧ BD Atlas Pack (grid)")
save = add("SaveImage", (2560, 80), (300, 270), {"filename_prefix": "packing/rgb_pack"},
           title="⑨ Save RGB Pack")
prev_atlas = add("PreviewImage", (2560, 420), (300, 300), {}, title="⑩ Atlas Preview")
prev_dbg = add("PreviewImage", (2560, 740), (300, 240), {}, title="Pack Debug (R/G/B/A)")

md = ("## Game-Engine Packing\n\n"
      "Remove BG → **SAM3** isolates parts → per part: **greyscale → normalize luma → "
      "center median** (each object's luma balanced *within its own mask*) → packed two ways:\n\n"
      "- **BD Pack Channels** — R=lips, G=teeth, B=tongue in one RGB texture\n"
      "- **BD Atlas Pack** — the same balanced parts tiled into a grid atlas (cols×rows, "
      "padding, `layout` JSON has per-cell UV rects)\n\n"
      "Swap the SAM3 prompts + add channels/cells for your own parts (e.g. the LipViseme atlas).")
about = add("MarkdownNote", (40, 440), (340, 300), {"__md__": md},
            title="ℹ️ About — 🧠 BrainDead Game-Engine Packing")

# ── wiring ──
link(load, "IMAGE", rbg, "image")
link(load, "IMAGE", sam, "image")
link(rbg, "mask", sam, "silhouette_mask")
for k in range(3):
    link(sam, "per_prompt_masks", idx_nodes[k], "masks")
    link(rbg, "rgb_white_bg", grey_nodes[k], "image")
    link(idx_nodes[k], "mask", grey_nodes[k], "mask")
    link(grey_nodes[k], "image", norm_nodes[k], "image")
    link(idx_nodes[k], "mask", norm_nodes[k], "mask")
    link(norm_nodes[k], "image", cen_nodes[k], "image")
    link(idx_nodes[k], "mask", cen_nodes[k], "mask")
# pack R/G/B
link(cen_nodes[0], "image", pack, "red_image")
link(cen_nodes[1], "image", pack, "green_image")
link(cen_nodes[2], "image", pack, "blue_image")
# atlas image_1/2/3
link(cen_nodes[0], "image", atlas, "image_1")
link(cen_nodes[1], "image", atlas, "image_2")
link(cen_nodes[2], "image", atlas, "image_3")
# outputs
link(pack, "image", save, "images")
link(atlas, "atlas", prev_atlas, "images")
link(pack, "debug_preview", prev_dbg, "images")

wf = {"id": "bd-game-engine-packing", "revision": 0, "last_node_id": nid, "last_link_id": lid,
      "nodes": nodes, "links": links, "groups": [], "config": {}, "extra": {}, "version": 0.4}
out_path = os.path.join(EW, "BD-game_engine_packing.json")
json.dump(wf, open(out_path, "w"), indent=2)
print(f"wrote {out_path}: {nid} nodes, {lid} links")
