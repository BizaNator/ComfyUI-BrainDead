#!/usr/bin/env python3
"""
Build BD-pixal3d_image_to_3d (textured): Load -> Pixal3D Preprocess -> Pixal3D Image->3D
(mesh + voxelgrid) -> CuMesh Simplify -> OVoxel Texture Bake (PBR) -> Preview3D + Export +
Save diffuse/normal. Pulls live object_info so widgets align. Styled per the BrainDead
convention (circled-number titles + 'ℹ️ About — 🧠 BrainDead …').
"""
import json, os, urllib.request

SRV = "http://127.0.0.1:8188"
EW = os.path.join(os.path.dirname(__file__), "..", "example_workflows")
# Widget inputs: scalar types + combos (combos come as a list of options OR the string "COMBO"
# with options in the 2nd element). Everything else (IMAGE/TRIMESH/VOXELGRID/...) is a socket.
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}

_cache = {}
def oi(t):
    if t not in _cache:
        _cache[t] = json.loads(urllib.request.urlopen(f"{SRV}/object_info/{t}", timeout=20).read())[t]
    return _cache[t]

def _order_specs(t):
    d = oi(t)
    order = d['input_order'].get('required', []) + d['input_order'].get('optional', [])
    alld = {**d['input'].get('required', {}), **d['input'].get('optional', {})}
    return order, alld

def _is_widget(spec):
    typ = spec[0]
    return isinstance(typ, list) or (isinstance(typ, str) and typ in WIDGET_TYPES)

def widget_defaults(t):
    order, alld = _order_specs(t)
    out = []
    for k in order:
        spec = alld[k]
        if not _is_widget(spec):
            continue
        meta = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        if isinstance(spec[0], list):                 # list-style combo
            df = meta.get('default', spec[0][0] if spec[0] else None)
        elif spec[0] == "COMBO":                       # string-style combo (options in meta)
            opts = meta.get('options', [])
            df = meta.get('default', opts[0] if opts else None)
        else:
            df = meta.get('default')
        out.append((k, df))
        # ComfyUI inserts a hidden 'control_after_generate' widget right after a seed/noise_seed
        # (or any widget flagged control_after_generate). Emit it or every later widget shifts.
        if k in ("seed", "noise_seed") or meta.get("control_after_generate"):
            out.append(("__control_after_generate__", "fixed"))
    return out

def conns(t):
    order, alld = _order_specs(t)
    return [(k, alld[k][0]) for k in order if not _is_widget(alld[k])]

def out_pairs(t):
    d = oi(t)
    return list(zip(d.get('output_name', []), d.get('output', [])))

nid = 0; lid = 0; nodes = []; links = []
def add(type_, pos, size, overrides=None, title=None):
    global nid; nid += 1
    if type_ == "MarkdownNote":
        n = {"id": nid, "type": type_, "pos": list(pos), "size": list(size), "flags": {},
             "order": nid - 1, "mode": 0, "inputs": [], "outputs": [], "properties": {},
             "widgets_values": [overrides["__md__"]]}
        if title: n["title"] = title
        nodes.append(n); return n
    widgets = [overrides.get(k, d) if overrides else d for k, d in widget_defaults(type_)]
    n = {"id": nid, "type": type_, "pos": list(pos), "size": list(size), "flags": {},
         "order": nid - 1, "mode": 0,
         "inputs": [{"name": c, "type": ty, "link": None} for c, ty in conns(type_)],
         "outputs": [{"name": o, "type": ty, "links": None, "slot_index": i}
                     for i, (o, ty) in enumerate(out_pairs(type_))],
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

load = add("LoadImage", (40, 80), (300, 314), {"image": "example.png"}, title="① Load Image")
pre  = add("BD_Pixal3DPreprocess", (380, 80), (300, 240), {}, title="② Pixal3D Preprocess (FOV + bg)")
gen  = add("BD_Pixal3DImageTo3D", (720, 80), (320, 460), {}, title="③ Pixal3D Image→3D (mesh + voxelgrid)")
# All-in-one bake: decimate + UV unwrap + PBR bake from the voxelgrid (the manual
# Simplify→UVUnwrap→TextureBake chain produced a scrambled atlas — this one is clean).
bake = add("BD_OVoxelBake", (1100, 80), (320, 240),
           {"decimation_target": 50000, "texture_size": 2048},
           title="④ OVoxel Bake (decimate + UV + PBR, all-in-one)")
# Pixal3D's mesh comes out lying down / facing up — rotate_x=180 stands it upright facing
# forward (matches Trellis2's convention). UV/material/COLOR_0 ride along.
orient = add("BD_OrientMesh", (1460, 80), (300, 180),
             {"rotate_x": 180.0}, title="⑤ Orient Mesh (Pixal3D → upright/forward)")
prev3d = add("BD_Preview3D", (1800, 80), (300, 120), {}, title="⑥ Preview 3D (textured mesh)")
export = add("BD_ExportMeshWithColors", (1800, 240), (300, 260),
             {"filename": "pixal3d_textured", "format": "glb"}, title="⑦ Export Mesh (.glb)")
sv_d = add("SaveImage", (1800, 540), (300, 270), {"filename_prefix": "pixal3d/diffuse"}, title="⑧ Save Diffuse")
sv_n = add("SaveImage", (1800, 840), (300, 270), {"filename_prefix": "pixal3d/normal"}, title="⑨ Save Normal")
pv_pre = add("PreviewImage", (380, 360), (300, 280), {}, title="Preprocessed Input")

md = ("## Pixal3D Image → Textured 3D\n\n"
      "TencentARC **Pixal3D** single-image → textured mesh.\n\n"
      "- **Preprocess** — FOV estimate + background removal → pixal3d_input\n"
      "- **Image→3D** — `pipeline_type=1024_cascade` generates the **mesh + coloured "
      "voxelgrid** (shape + texture passes)\n"
      "- **OVoxel Bake** (all-in-one) — decimate + UV unwrap + bake the voxelgrid → "
      "**diffuse / normal / metallic / roughness** PBR maps + a clean UV'd mesh\n"
      "- **Orient Mesh** — Pixal3D comes out lying down / facing up; `rotate_x=180` stands it "
      "upright facing forward (UV/material/vertex colors preserved)\n"
      "- **Preview 3D** (textured) + **Export .glb** + save diffuse/normal\n\n"
      "Needs the Pixal3D model at `TencentARC/Pixal3D` (auto-downloads on first run). "
      "GPU + a few minutes per run.")
add("MarkdownNote", (40, 440), (320, 300), {"__md__": md},
    title="ℹ️ About — 🧠 BrainDead Pixal3D Textured")

# wiring
link(load, "IMAGE", pre, "image")
link(pre, "pixal3d_input", gen, "pixal3d_input")
link(pre, "preprocessed_image", pv_pre, "images")
link(gen, "voxelgrid", bake, "voxelgrid")   # all-in-one bake makes its own decimated+UV'd mesh
link(bake, "mesh", orient, "mesh")          # fix Pixal3D orientation on the final textured mesh
link(orient, "mesh", prev3d, "mesh")
link(orient, "mesh", export, "mesh")
link(bake, "diffuse", export, "diffuse")   # embed baked atlas into the glb (textured export, not white)
link(bake, "normal", export, "normal")
link(bake, "diffuse", sv_d, "images")
link(bake, "normal", sv_n, "images")

wf = {"id": "bd-pixal3d-textured", "revision": 0, "last_node_id": nid, "last_link_id": lid,
      "nodes": nodes, "links": links, "groups": [], "config": {}, "extra": {}, "version": 0.4}
out = os.path.join(EW, "BD-pixal3d_image_to_3d.json")
json.dump(wf, open(out, "w"), indent=2)
print(f"wrote {out}: {nid} nodes, {lid} links")
