#!/usr/bin/env python3
"""
Build BD-parts_builder_2: Qwen Image 2.1 layer decomposition of a character head (UEFN-430).

  ① Load headshot (RGBA: its alpha is the head mask)
  ② Presence vote - BD Parts Vocabulary -> AILab_QwenVL (Qwen3-VL-8B) true/false over a CLOSED list
  ③ Depth - Lotus2 raw_linear (brighter = nearer) - only breaks paint-order ties
  ④ BD Parts Builder 2 - per voted item a VISIBLE + COMPLETE 2.1 edit -> parts / complete_parts / base_image
  ⑤ BD Parts Builder 2 Plates - bald -> mouthless + eyeless, frame-proven, plate frame
  ⑥ BD Parts Builder 2 Assemble - the engine stack (back parts -> eyeless plate -> front parts)
  ⑦ Export visible layers + head without items (PSD rebuilds the source)
  ⑧ Export the engine stack (PSD)

New nodes exist only where this branch is loaded, so read /object_info from that server:
    python3 tools/build_parts_builder_2.py [--server http://127.0.0.1:8188]
"""
import argparse
import json
import os
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--server", default="http://127.0.0.1:8188")
args = ap.parse_args()
SRV = args.server.rstrip("/")
EW = os.path.join(os.path.dirname(__file__), "..", "example_workflows")
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
_cache = {}

HF = "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/"
MODELS = {
    "UNETLoader": [{"name": "qwen_image_2.1_bf16.safetensors", "url": HF + "diffusion_models/qwen_image_2.1_bf16.safetensors",
                    "directory": "diffusion_models"}],
    "CLIPLoader": [{"name": "qwen3vl_8b_bf16.safetensors", "url": HF + "text_encoders/qwen3vl_8b_bf16.safetensors",
                    "directory": "text_encoders"}],
    "VAELoader": [{"name": "qwen_image_2.1_vae_bf16.safetensors", "url": HF + "vae/qwen_image_2.1_vae_bf16.safetensors",
                   "directory": "vae"}],
}


def oi(t):
    if t not in _cache:
        _cache[t] = json.loads(urllib.request.urlopen(f"{SRV}/object_info/{t}", timeout=30).read())[t]
    return _cache[t]


def _order_specs(t):
    d = oi(t)
    return (d["input_order"].get("required", []) + d["input_order"].get("optional", []),
            {**d["input"].get("required", {}), **d["input"].get("optional", {})})


def _is_widget(spec):
    meta = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if meta.get("forceInput"):
        return False                                   # a socket-only input (e.g. Assemble plate_frame)
    return isinstance(spec[0], list) or (isinstance(spec[0], str) and spec[0] in WIDGET_TYPES)


def widget_defaults(t):
    order, alld = _order_specs(t)
    out = []
    for k in order:
        spec = alld[k]
        if not _is_widget(spec):
            continue
        meta = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        if isinstance(spec[0], list):
            df = meta.get("default", spec[0][0] if spec[0] else None)
        elif spec[0] == "COMBO":
            opts = meta.get("options", [])
            df = meta.get("default", opts[0] if opts else None)
        else:
            df = meta.get("default")
        out.append((k, df))
        if k in ("seed", "noise_seed") or meta.get("control_after_generate"):
            out.append(("__control_after_generate__", "fixed"))
    return out


def widget_type(t, name):
    _, alld = _order_specs(t)
    s = alld[name][0]
    return "COMBO" if isinstance(s, list) else s


def conns(t):
    order, alld = _order_specs(t)
    return [(k, alld[k][0]) for k in order if not _is_widget(alld[k])]


def out_pairs(t):
    d = oi(t)
    return list(zip(d.get("output_name", []), d.get("output", [])))


nid = 0
lid = 0
nodes, links, groups = [], [], []


def add(type_, pos, size, overrides=None, title=None):
    global nid
    nid += 1
    if type_ == "MarkdownNote":
        n = {"id": nid, "type": type_, "pos": list(pos), "size": list(size), "flags": {}, "order": nid - 1,
             "mode": 0, "inputs": [], "outputs": [], "properties": {}, "widgets_values": [overrides["__md__"]]}
        if title:
            n["title"] = title
        nodes.append(n)
        return n
    wd = widget_defaults(type_)
    unknown = set(overrides or {}) - {k for k, _ in wd}
    assert not unknown, (type_, unknown)
    widgets = [overrides.get(k, d) if overrides else d for k, d in wd]
    n = {"id": nid, "type": type_, "pos": list(pos), "size": list(size), "flags": {}, "order": nid - 1, "mode": 0,
         "inputs": [{"name": c, "type": ty, "link": None} for c, ty in conns(type_)],
         "outputs": [{"name": o, "type": ty, "links": None, "slot_index": i} for i, (o, ty) in enumerate(out_pairs(type_))],
         "properties": {"Node name for S&R": type_}, "widgets_values": widgets}
    if type_ in MODELS:
        n["properties"]["models"] = MODELS[type_]
    if title:
        n["title"] = title
    nodes.append(n)
    return n


def link(src, oname, dst, iname):
    """Link an output to a socket input, or to a WIDGET input (the input entry is added with its widget ref)."""
    global lid
    lid += 1
    so = [o["name"] for o in src["outputs"]].index(oname)
    names = [i["name"] for i in dst["inputs"]]
    if iname not in names:
        dst["inputs"].append({"name": iname, "type": widget_type(dst["type"], iname), "widget": {"name": iname}, "link": None})
        names.append(iname)
    di = names.index(iname)
    links.append([lid, src["id"], so, dst["id"], di, src["outputs"][so]["type"]])
    o = src["outputs"][so]
    o["links"] = (o["links"] or []) + [lid]
    dst["inputs"][di]["link"] = lid


def group(title, x, y, w, h, color="#3f789e"):
    groups.append({"id": len(groups) + 1, "title": title, "bounding": [x, y, w, h], "color": color,
                   "font_size": 24, "flags": {}})


SEED = 20260931
# ── ① input + models ─────────────────────────────────────────────────────────
group("① Input - headshot with alpha", 0, 0, 760, 520)
load = add("LoadImage", (20, 50), (340, 440), {"image": "example.png"}, title="① Load Headshot (RGBA)")
alpha = add("InvertMask", (390, 50), (230, 60), {}, title="head mask (alpha)")
prev_in = add("PreviewImage", (390, 150), (340, 340), {}, title="Headshot")

group("Models - Qwen Image 2.1 (bf16, qwen3vl_8b, RGBA VAE)", 0, 560, 760, 480, "#335")
unet = add("UNETLoader", (20, 610), (360, 90), {"unet_name": "qwen_image_2.1_bf16.safetensors"}, title="Qwen Image 2.1")
cache = add("QwenImage21Cache", (400, 610), (320, 90), {}, title="Qwen Image 2.1 Cache")
clip = add("CLIPLoader", (20, 740), (360, 110), {"clip_name": "qwen3vl_8b_bf16.safetensors", "type": "qwen_image"},
           title="qwen3vl_8b (qwen_image)")
vae = add("VAELoader", (400, 740), (320, 70), {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}, title="Qwen 2.1 VAE (RGBA)")

# ── ② vote, ③ depth ──────────────────────────────────────────────────────────
group("② Presence vote - closed list (2.1 draws anything it is asked for)", 800, 0, 1080, 520)
voc = add("BD_PartsVocabulary", (820, 50), (400, 440), {}, title="② Parts Vocabulary")
vl = add("AILab_QwenVL", (1240, 50), (380, 440), {"model_name": "Qwen3-VL-8B-Instruct", "quantization": "None (FP16)", "preset_prompt": "\U0001f5bc️ Tags",
                                                  "max_tokens": 400, "keep_model_loaded": False, "seed": 1},
         title="Qwen3-VL vote")
vote = add("PreviewAny", (1640, 50), (220, 440), {}, title="vote")

group("③ Depth - Lotus2, brighter = nearer (paint-order tie-break only)", 800, 560, 1080, 480, "#335")
lm = add("BD_Lotus2ModelLoader", (820, 610), (360, 200), {"task": "depth", "cpu_offload": False}, title="Lotus2 depth")
lp = add("BD_Lotus2Predict", (1200, 610), (320, 200), {}, title="③ Depth")
dprev = add("PreviewImage", (1540, 610), (320, 400), {}, title="depth")

# ── ④ Parts Builder 2 ────────────────────────────────────────────────────────
group("④ Parts Builder 2 - visible + complete layers, paint order, head without items", 1920, 0, 1500, 1040)
pb = add("BD_PartsBuilder2", (1940, 50), (440, 960), {"seed": SEED}, title="④ Parts Builder 2 (Qwen 2.1)")
lprev = add("PreviewImage", (2400, 50), (500, 460), {}, title="layers (visible, then complete)")
bprev = add("PreviewImage", (2400, 540), (500, 470), {}, title="head without items")
rep1 = add("PreviewAny", (2920, 50), (480, 960), {}, title="Parts Builder 2 report")

# ── ⑤ plates, ⑥ assemble ─────────────────────────────────────────────────────
group("⑤ Plates - bald -> mouthless + eyeless (cut onto white, frame-proven)", 0, 1080, 1880, 900)
pl = add("BD_PartsBuilder2Plates", (20, 1130), (440, 820), {"seed": SEED + 2}, title="⑤ Parts Builder 2 Plates (Qwen 2.1)")
pv_b = add("PreviewImage", (480, 1130), (340, 400), {}, title="bald")
pv_m = add("PreviewImage", (840, 1130), (340, 400), {}, title="mouthless / browless")
pv_e = add("PreviewImage", (1200, 1130), (340, 400), {}, title="eyeless (engine base)")
rep2 = add("PreviewAny", (1560, 1130), (300, 820), {}, title="Plates report")
sv_b = add("SaveImage", (480, 1560), (340, 390), {"filename_prefix": "parts_builder_2/plate_bald"}, title="Save bald")
sv_m = add("SaveImage", (840, 1560), (340, 390), {"filename_prefix": "parts_builder_2/plate_mouthless"}, title="Save mouthless")
sv_e = add("SaveImage", (1200, 1560), (340, 390), {"filename_prefix": "parts_builder_2/plate_eyeless"}, title="Save eyeless")

group("⑥ Assemble - the engine stack vs the source", 1920, 1080, 1500, 900)
asm = add("BD_PartsBuilder2Assemble", (1940, 1130), (360, 300), {}, title="⑥ Parts Builder 2 Assemble")
pv_r = add("PreviewImage", (2320, 1130), (520, 520), {}, title="reassembled")
pv_d = add("PreviewImage", (2860, 1130), (520, 520), {}, title="difference (red = off)")
rep3 = add("PreviewAny", (1940, 1460), (360, 490), {}, title="Assemble report")

# ── ⑦ ⑧ exports ──────────────────────────────────────────────────────────────
group("⑦ ⑧ Export - PSD + PNGs + manifest (BD Parts Export)", 0, 2020, 1880, 700, "#353")
ex1 = add("BD_PartsExport", (20, 2070), (440, 620), {"filename": "visible_layers", "name_prefix": "parts_builder_2"},
          title="⑦ Export visible layers + head without items")
ex2 = add("BD_PartsExport", (480, 2070), (440, 620), {"filename": "engine_stack", "name_prefix": "parts_builder_2"},
          title="⑧ Export engine stack")

md = ("## BD Parts Builder 2\n\n"
      "Qwen Image 2.1 layer decomposition of a character head - prompts only, no SAM, every layer in the "
      "source's own pixels.\n\n"
      "**Flow**\n"
      "1. **Load** a headshot with alpha (e.g. HeadMasters MASTER 1).\n"
      "2. **Vote** - Qwen3-VL answers true/false for a CLOSED list (BD Parts Vocabulary). 2.1 draws any item it "
      "is asked for, present or not, so only voted items are requested.\n"
      "3. **Parts Builder 2** - per item a *visible* edit (where it is; pixels come from the source, so the "
      "visible layers rebuild it exactly) and a *complete* edit (the whole object, hidden parts included, split "
      "into front and _back). Paint order is read off the source. base_image = head without items.\n"
      "4. **Plates** - bald (its alpha is the matte), then mouthless and eyeless, each ONE edit of the bald, cut "
      "onto white and frame-proven. Small heads (a hat set the framing) are re-framed; work_size 2048 for more "
      "pixels.\n"
      "5. **Assemble** - the engine stack: back parts -> eyeless plate -> front parts, compared with the source.\n"
      "6. **Export** - visible layers + head without items (the PSD rebuilds the source) and the engine stack.\n\n"
      "**Models** - `qwen_image_2.1_bf16`, `qwen3vl_8b_bf16` (CLIPLoader type qwen_image), "
      "`qwen_image_2.1_vae_bf16` (Comfy-Org/Qwen-Image-2.1, auto-download links on the loaders); "
      "`Qwen3-VL-8B-Instruct` (AILab QwenVL); Lotus2 depth. An edit takes ~15 s at 1024 (40 steps) on the "
      "studio card; a head with 6 items is ~16 edits plus the plates.\n\n"
      "**Rules learned the hard way** - never feed a stage 2.1's raw output (noise amplifies); at most 2 "
      "generations from the source; never ask for an item the vote did not find.\n\n"
      "---\n**BrainDeadGuild** - created by **BizaNator**\n"
      "[BrainDeadGuild.com](https://BrainDeadGuild.com) · [BrainDead.TV](https://BrainDead.TV) · "
      "[GitHub](https://github.com/BizaNator/ComfyUI-BrainDead) · [Discord](https://braindeadguild.com/discord)\n")
add("MarkdownNote", (960, 2070), (900, 620), {"__md__": md}, title="ℹ️ About — 🧠 BrainDead Parts Builder 2")

# ── links ────────────────────────────────────────────────────────────────────
link(load, "IMAGE", prev_in, "images")
link(load, "MASK", alpha, "mask")
link(unet, "MODEL", cache, "model")
link(voc, "vlm_prompt", vl, "custom_prompt")
link(load, "IMAGE", vl, "image")
link(vl, "RESPONSE", vote, "source")
link(lm, "model", lp, "model")
link(load, "IMAGE", lp, "image")
link(lp, "map", dprev, "images")
for dst in (pb, pl):
    link(load, "IMAGE", dst, "image")
    link(cache, "MODEL", dst, "model")
    link(clip, "CLIP", dst, "clip")
    link(vae, "VAE", dst, "vae")
    link(alpha, "MASK", dst, "head_mask")
link(vl, "RESPONSE", pb, "presence")
link(voc, "vocabulary", pb, "vocabulary")
link(lp, "raw_linear", pb, "depth_image")
link(pb, "layers_preview", lprev, "images")
link(pb, "base_image", bprev, "images")
link(pb, "report", rep1, "source")
link(pb, "parts", pl, "parts")
link(pb, "complete_parts", pl, "complete_parts")
for o, p, s in (("bald", pv_b, sv_b), ("mouthless", pv_m, sv_m), ("eyeless", pv_e, sv_e)):
    link(pl, o, p, "images")
    link(pl, o, s, "images")
link(pl, "report", rep2, "source")
link(pb, "complete_parts", asm, "complete_parts")
link(pl, "eyeless", asm, "plate")
link(pl, "matte", asm, "matte")
link(pl, "plate_frame", asm, "plate_frame")
link(pl, "plate_source", asm, "plate_source")
link(pl, "plate_head", asm, "plate_head")
link(asm, "reassembled", pv_r, "images")
link(asm, "difference", pv_d, "images")
link(asm, "report", rep3, "source")
link(pb, "parts", ex1, "parts")
link(pb, "base_image", ex1, "base_image")
link(asm, "engine_parts", ex2, "parts")

wf = {"id": "bd-parts-builder-2", "revision": 0, "last_node_id": nid, "last_link_id": lid,
      "nodes": nodes, "links": links, "groups": groups, "config": {}, "extra": {"ds": {"scale": 0.35, "offset": [100, 100]}},
      "version": 0.4}
out = os.path.join(EW, "BD-parts_builder_2.json")
json.dump(wf, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"wrote {out}: {nid} nodes, {lid} links, {len(groups)} groups")
