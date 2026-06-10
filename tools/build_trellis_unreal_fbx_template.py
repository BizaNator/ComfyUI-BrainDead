#!/usr/bin/env python3
"""
Build BD-trellis2_unreal_fbx: the Unreal/Blender character FBX pipeline.

Load → Remove BG (RMBG) → Get Conditioning → Image→Shape → Shape→Textured Mesh →
OVoxel Bake (decimate + UV + PBR) → Sample Vertex Colors (→ color_field) →
Pack Bundle (mesh + color_field + PBR maps) → **BD Blender Export Mesh (format=FBX,
flat_shading=True)** → single game-ready FBX with embedded textures AND vertex colors.

Requires Blender (bundled at lib/blender) — FBX is written by Blender's exporter, which is
the only path that puts BOTH a texture and vertex colors in one file. flat_shading=True gives
the clean low-poly no-smoothing look. Pulls live object_info so widgets align (incl. the
hidden control_after_generate companion after seed widgets).
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
        if isinstance(spec[0], list):
            df = meta.get('default', spec[0][0] if spec[0] else None)
        elif spec[0] == "COMBO":
            opts = meta.get('options', [])
            df = meta.get('default', opts[0] if opts else None)
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

def add_rmbg(pos, title):
    """RMBG (comfyui-rmbg) — hand-built so the template doesn't hard-require its object_info."""
    global nid; nid += 1
    n = {"id": nid, "type": "RMBG", "pos": list(pos), "size": [300, 250], "flags": {},
         "order": nid - 1, "mode": 0,
         "inputs": [{"name": "image", "type": "IMAGE", "link": None}],
         "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": None, "slot_index": 0},
                     {"name": "MASK", "type": "MASK", "links": None, "slot_index": 1},
                     {"name": "MASK_IMAGE", "type": "IMAGE", "links": None, "slot_index": 2}],
         "properties": {"Node name for S&R": "RMBG"},
         "widgets_values": ["RMBG-2.0", 1.0, 1024, 0, 0, False, False, "Alpha", "#222222"],
         "title": title}
    nodes.append(n); return n

def link(src, oname, dst, iname):
    global lid; lid += 1
    so = [o['name'] for o in src['outputs']].index(oname)
    di = [i['name'] for i in dst['inputs']].index(iname)
    links.append([lid, src['id'], so, dst['id'], di, src['outputs'][so]['type']])
    o = src['outputs'][so]; o['links'] = (o['links'] or []) + [lid]
    dst['inputs'][di]['link'] = lid

C = lambda i: chr(0x2460 + i - 1)

load = add("LoadImage", (40, 80), (300, 314), {"image": "example.png"}, title="① Load Character (head or body)")
rmbg = add_rmbg((380, 80), "② Remove Background (RMBG-2.0)")
cond = add("BD_Trellis2GetConditioning", (720, 80), (300, 260), {}, title="③ Get Conditioning")
shape = add("BD_Trellis2ImageToShape", (1060, 80), (300, 300), {}, title="④ Image → Shape")
tex = add("BD_Trellis2ShapeToTexturedMesh", (1400, 80), (300, 300), {}, title="⑤ Shape → Textured Mesh (+ voxelgrid)")
bake = add("BD_OVoxelBake", (1740, 80), (300, 240),
           {"decimation_target": 3000, "texture_size": 2048},   # low-poly target ~3k tris (bump to 5000 if chunky)
           title="⑥ OVoxel Bake (decimate + UV + PBR • ~3k tris low-poly)")
samp = add("BD_SampleVoxelgridColors", (2080, 80), (300, 200), {"sampling_mode": "smooth"},
           title="⑦ Sample Vertex Colors (→ color_field)")
pack = add("BD_PackBundle", (2420, 80), (300, 240), {"name": "character"},
           title="⑧ Pack Bundle (mesh + color_field + PBR)")
fbx = add("BD_BlenderExportMesh", (2760, 80), (320, 320),
          {"output_dir": "unreal_fbx", "format": "FBX", "flat_shading": True,
           "solidify_mode": "NONE"},
          title="⑨ Blender Export FBX (flat • embedded textures + vertex colors)")

prev3d = add("BD_Preview3D", (2080, 320), (300, 260), {}, title="Preview 3D (low-poly textured)")
sv_d = add("SaveImage", (1740, 360), (300, 270), {"filename_prefix": "unreal_fbx/diffuse"}, title="Save Diffuse")
sv_n = add("SaveImage", (1740, 660), (300, 270), {"filename_prefix": "unreal_fbx/normal"}, title="Save Normal")
pv_pre = add("PreviewImage", (380, 360), (300, 280), {}, title="Preprocessed Input")

md = ("## TRELLIS2 → Unreal FBX (Blender)\n\n"
      "Single-image character (head or body) → **game-ready FBX** for the Blender→Unreal pipeline.\n\n"
      "**Flow:** Load → **RMBG** → Conditioning → Image→Shape → Shape→Textured Mesh → "
      "**OVoxel Bake** (decimate + UV + PBR) → **Sample Vertex Colors** (→ color_field) → "
      "**Pack Bundle** → **BD Blender Export Mesh (FBX)**.\n\n"
      "- **Requires Blender** (bundled). FBX is the only single-file format that carries **both** "
      "embedded textures **and** vertex colors — exactly what Blender needs to continue the "
      "character pipeline into Unreal.\n"
      "- **Low-poly:** OVoxel Bake `decimation_target=3000` tris (heads/bodies); bump to ~5000 "
      "if too chunky. **`flat_shading=True`** → clean **no-smoothing** faceted look.\n"
      "- The FBX embeds baseColor/normal (+ metallic/roughness) and the source-accurate "
      "**COLOR_0** vertex colors (from the voxelgrid color_field).\n\n"
      "**Sharper low-poly:** for explicit sharp-edge marking / planar flat-plane grouping, insert "
      "**BD Planar Grouping**, **BD Blender Edge Marking** or **BD Blender Merge Planes** between "
      "OVoxel Bake and Sample Vertex Colors — or do it in Blender with the BrainDead toolkit after "
      "import.\n\n"
      "Models (TRELLIS.2-4B) auto-download on first run. ~12–16GB VRAM.\n\n"
      "---\n**BrainDeadGuild** — created by **BizaNator**")
add("MarkdownNote", (40, 440), (320, 360), {"__md__": md},
    title="ℹ️ About — 🧠 BrainDead TRELLIS2 Unreal FBX")

# wiring
link(load, "IMAGE", rmbg, "image")
link(load, "IMAGE", cond, "image")            # conditioning does its own alpha/crop from image+mask
link(rmbg, "MASK", cond, "mask")
link(load, "IMAGE", pv_pre, "images")
link(cond, "conditioning", shape, "conditioning")
link(cond, "conditioning", tex, "conditioning")   # tex needs BOTH conditioning + shape_result
link(shape, "shape_result", tex, "shape_result")
link(tex, "voxelgrid", bake, "voxelgrid")
link(bake, "mesh", samp, "mesh")
link(tex, "voxelgrid", samp, "voxelgrid")     # sampler reads the same voxelgrid (mesh is in its space)
link(samp, "mesh", pack, "mesh")
link(samp, "color_field", pack, "color_field")
link(bake, "diffuse", pack, "diffuse")
link(bake, "normal", pack, "normal")
link(bake, "metallic", pack, "metallic")
link(bake, "roughness", pack, "roughness")
link(bake, "alpha", pack, "alpha")
link(pack, "bundle", fbx, "bundle")
link(bake, "mesh", prev3d, "mesh")
link(bake, "diffuse", sv_d, "images")
link(bake, "normal", sv_n, "images")

wf = {"id": "bd-trellis2-unreal-fbx", "revision": 0, "last_node_id": nid, "last_link_id": lid,
      "nodes": nodes, "links": links, "groups": [], "config": {}, "extra": {}, "version": 0.4}
out = os.path.join(EW, "BD-trellis2_unreal_fbx.json")
json.dump(wf, open(out, "w"), indent=2)
print(f"wrote {out}: {nid} nodes, {lid} links")
