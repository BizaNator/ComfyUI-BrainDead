#!/usr/bin/env python3
"""
Build BD-isolate_part: Load → BD Remove Background (SAM3 prompt = what to KEEP) → save the
clean white-bg crop. A reusable "isolate a named subject/part → clean PNG" workflow, and stage 1
of the SAM3→Trellis chain (run_part_to_3d.py). Override the prompt at run time:
    run_bd.py --workflow isolate_part --image x.png --set "BD_RemoveBackground.prompts=tank top"
"""
import json, os, urllib.request

SRV = "http://127.0.0.1:8188"
EW = os.path.join(os.path.dirname(__file__), "..", "example_workflows")
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
_cache = {}

def oi(t):
    if t not in _cache:
        _cache[t] = json.loads(urllib.request.urlopen(f"{SRV}/object_info/{t}", timeout=20).read())[t]
    return _cache[t]

def _order_specs(t):
    d = oi(t)
    return (d['input_order'].get('required', []) + d['input_order'].get('optional', []),
            {**d['input'].get('required', {}), **d['input'].get('optional', {})})

def _is_widget(spec):
    return isinstance(spec[0], list) or (isinstance(spec[0], str) and spec[0] in WIDGET_TYPES)

def widget_defaults(t):
    order, alld = _order_specs(t); out = []
    for k in order:
        spec = alld[k]
        if not _is_widget(spec):
            continue
        meta = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        if isinstance(spec[0], list):
            df = meta.get('default', spec[0][0] if spec[0] else None)
        elif spec[0] == "COMBO":
            opts = meta.get('options', []); df = meta.get('default', opts[0] if opts else None)
        else:
            df = meta.get('default')
        out.append((k, df))
        if k in ("seed", "noise_seed") or meta.get("control_after_generate"):
            out.append(("__control_after_generate__", "fixed"))
    return out

def conns(t):
    order, alld = _order_specs(t)
    return [(k, alld[k][0]) for k in order if not _is_widget(alld[k])]

def out_pairs(t):
    d = oi(t); return list(zip(d.get('output_name', []), d.get('output', [])))

nid = 0; lid = 0; nodes = []; links = []
def add(type_, pos, size, overrides=None, title=None):
    global nid; nid += 1
    if type_ == "MarkdownNote":
        n = {"id": nid, "type": type_, "pos": list(pos), "size": list(size), "flags": {}, "order": nid - 1,
             "mode": 0, "inputs": [], "outputs": [], "properties": {}, "widgets_values": [overrides["__md__"]]}
        if title: n["title"] = title
        nodes.append(n); return n
    widgets = [overrides.get(k, d) if overrides else d for k, d in widget_defaults(type_)]
    n = {"id": nid, "type": type_, "pos": list(pos), "size": list(size), "flags": {}, "order": nid - 1, "mode": 0,
         "inputs": [{"name": c, "type": ty, "link": None} for c, ty in conns(type_)],
         "outputs": [{"name": o, "type": ty, "links": None, "slot_index": i} for i, (o, ty) in enumerate(out_pairs(type_))],
         "properties": {"Node name for S&R": type_}, "widgets_values": widgets}
    if title: n["title"] = title
    nodes.append(n); return n

def link(src, oname, dst, iname):
    global lid; lid += 1
    so = [o['name'] for o in src['outputs']].index(oname); di = [i['name'] for i in dst['inputs']].index(iname)
    links.append([lid, src['id'], so, dst['id'], di, src['outputs'][so]['type']])
    o = src['outputs'][so]; o['links'] = (o['links'] or []) + [lid]; dst['inputs'][di]['link'] = lid

load = add("LoadImage", (40, 80), (300, 314), {"image": "example.png"}, title="① Load Character")
rmbg = add("BD_RemoveBackground", (380, 80), (320, 520),
           {"prompts": "subject\nperson", "crop_to_content": True, "output_size_mode": "pad_square", "output_size": 1024},
           title="② Isolate (SAM3 prompt = what to keep)")
sv = add("SaveImage", (740, 80), (300, 300), {"filename_prefix": "isolate/part"}, title="Save Isolated (white BG)")
pv = add("PreviewImage", (740, 400), (300, 280), {}, title="Preview")
md = ("## Isolate Part / Subject\n\n"
      "`BD Remove Background` (SAM3 + matting). The **prompts** are *what to keep* — set to a clothing "
      "item or part (`tank top`, `jacket`) to isolate just that, or `person` for the whole subject.\n\n"
      "Saves the clean **white-bg** crop. Stage 1 of the SAM3→Trellis chain "
      "(`run_part_to_3d.py`): isolate → feed the PNG into the Trellis Unreal FBX dispatcher.\n\n"
      "Run: `run_bd.py --workflow isolate_part --image x.png --set \"BD_RemoveBackground.prompts=tank top\"`\n\n"
      "---\n**BrainDeadGuild** — created by **BizaNator**")
add("MarkdownNote", (40, 440), (320, 260), {"__md__": md}, title="ℹ️ About — 🧠 BrainDead Isolate Part")

link(load, "IMAGE", rmbg, "image")
link(rmbg, "rgb_white_bg", sv, "images")
link(rmbg, "rgb_white_bg", pv, "images")

wf = {"id": "bd-isolate-part", "revision": 0, "last_node_id": nid, "last_link_id": lid,
      "nodes": nodes, "links": links, "groups": [], "config": {}, "extra": {}, "version": 0.4}
out = os.path.join(EW, "BD-isolate_part.json")
json.dump(wf, open(out, "w"), indent=2)
print(f"wrote {out}: {nid} nodes, {lid} links")
