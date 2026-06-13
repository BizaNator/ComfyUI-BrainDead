#!/usr/bin/env python3
"""
flatten_subgraph.py — expand a ComfyUI subgraph-wrapped workflow into a FLAT UI-graph so it can
be run via the API (run_bd.py / run_workflow.py). ComfyUI's frontend expands subgraphs before
submitting to /prompt; this does the same headless for single-level subgraphs.

It inlines the subgraph definition's internal nodes and rewires the subgraph's boundary I/O:
  - each subgraph INPUT is bound to a source you pass (a LoadImage, or a Primitive carrying a
    literal value),
  - the subgraph OUTPUT is wired to a SaveImage.

Usage (as a library): flatten(subgraph_workflow_dict, bindings, save_prefix) -> flat UI-graph dict
  bindings: {input_name: ("image", <path-or-None>) | ("int"|"bool"|"string"|"float", value)}

CLI: python3 flatten_subgraph.py "<subgraph.json>" --out flat.json \
        --image-input image --save-prefix mannequin --bind seed=int:0 --bind switch=bool:false ...
"""
import argparse, json, os

# Pure-UI / non-functional nodes to drop when flattening.
SKIP = {"Note", "MarkdownNote", "PreviewImage", "PreviewAny", "Image Comparer (rgthree)",
        "MaskPreview", "Reroute"}
_PRIM = {"int": "PrimitiveInt", "float": "PrimitiveFloat", "bool": "PrimitiveBoolean",
         "string": "PrimitiveStringMultiline"}


def flatten(wf, bindings, save_prefix="flatten_out"):
    sg = wf["definitions"]["subgraphs"][0]
    internal = {n["id"]: n for n in sg["nodes"]}
    keep = {nid: n for nid, n in internal.items() if n["type"] not in SKIP}

    # Resolve Reroutes: map (node,slot)->(node,slot) skipping reroute chains.
    feed = {}                     # (target_id, target_slot) -> (origin_id, origin_slot)
    for l in sg["links"]:
        feed[(l["target_id"], l["target_slot"])] = (l["origin_id"], l["origin_slot"])

    def trace(origin):
        oid, oslot = origin
        seen = 0
        while oid in internal and internal[oid]["type"] == "Reroute" and seen < 20:
            src = feed.get((oid, 0))
            if not src:
                break
            oid, oslot = src; seen += 1
        return oid, oslot

    # ComfyUI subgraph boundary convention: links from origin_id=-10 come from subgraph INPUT
    # [origin_slot]; links to target_id=-20 go to subgraph OUTPUT [target_slot].
    IN_ID, OUT_ID = -10, -20

    nid = max(internal) + 1000           # fresh ids for the nodes we add
    lid = (max((l["id"] for l in sg["links"]), default=0)) + 1000
    extra_nodes, links = [], []

    def add_node(t, widgets, outs, pos=(0, 0)):
        nonlocal nid
        n = {"id": nid, "type": t, "pos": list(pos), "size": [210, 80], "flags": {}, "order": 0,
             "mode": 0, "inputs": [], "properties": {"Node name for S&R": t},
             "outputs": [{"name": o, "type": ty, "links": []} for o, ty in outs],
             "widgets_values": widgets}
        nid += 1; extra_nodes.append(n); return n

    def link(src, sslot, dst, dslot, ty):
        nonlocal lid
        links.append([lid, src, sslot, dst, dslot, ty]); lid += 1

    # ---- bind each subgraph INPUT (by index) to a concrete source node -----------------------
    src_for_input = {}            # input index -> (node_id, slot)
    load_img = None
    for k, inp in enumerate(sg["inputs"]):
        kind, val = bindings.get(inp["name"], ("skip", None))
        if kind == "image":
            if load_img is None:
                ln = add_node("LoadImage", [os.path.basename(val) if val else "example.png", "image"],
                              [("IMAGE", "IMAGE"), ("MASK", "MASK")])
                load_img = ln["id"]
            src_for_input[k] = (load_img, 0)
        elif kind in _PRIM:
            p = add_node(_PRIM[kind], [val], [("VALUE", inp["type"])])
            src_for_input[k] = (p["id"], 0)
        # 'skip' → leave unbound (consumer keeps its own widget/default)

    def resolve(oid, oslot):
        """Resolve a link origin through Reroutes and the input boundary → a real (node, slot)."""
        oid, oslot = trace((oid, oslot))
        if oid == IN_ID:
            return src_for_input.get(oslot)
        return (oid, oslot) if oid in keep else None

    # ---- internal links (rewire boundary origins) -------------------------------------------
    for l in sg["links"]:
        oid, oslot, tid, tslot, ty = l["origin_id"], l["origin_slot"], l["target_id"], l["target_slot"], l["type"]
        if tid == OUT_ID or tid not in keep:
            continue
        src = resolve(oid, oslot)
        if src:
            link(src[0], src[1], tid, tslot, ty)

    # ---- output → SaveImage ------------------------------------------------------------------
    save = add_node("SaveImage", [save_prefix], [])
    save["outputs"] = []
    save["inputs"] = [{"name": "images", "type": "IMAGE", "link": None}]
    for l in sg["links"]:
        if l["target_id"] == OUT_ID and l["target_slot"] == 0:
            src = resolve(l["origin_id"], l["origin_slot"])
            if src:
                link(src[0], src[1], save["id"], 0, "IMAGE"); break

    # Rebuild every node's inputs[].link / outputs[].links from the final links array — workflow_to_api
    # reads those fields (not just the links list), and the kept internal nodes still carry stale
    # boundary link ids that we've rewired.
    allnodes = list(keep.values()) + extra_nodes
    byid = {n["id"]: n for n in allnodes}
    for n in allnodes:
        for inp in n.get("inputs", []):
            inp["link"] = None
        for o in n.get("outputs", []):
            o["links"] = []
    for lk in links:
        lkid, oid, oslot, tid, tslot, ty = lk
        tn, snode = byid.get(tid), byid.get(oid)
        if tn and tslot < len(tn.get("inputs", [])):
            tn["inputs"][tslot]["link"] = lkid
        if snode and oslot < len(snode.get("outputs", [])):
            snode["outputs"][oslot].setdefault("links", []).append(lkid)

    flat = {"last_node_id": nid, "last_link_id": lid, "nodes": allnodes,
            "links": links, "groups": [], "config": {}, "extra": {}, "version": 0.4}
    return flat


def _parse_bind(s):
    name, _, rest = s.partition("=")
    kind, _, val = rest.partition(":")
    if kind == "int": val = int(val)
    elif kind == "float": val = float(val)
    elif kind == "bool": val = val.lower() == "true"
    return name, (kind, val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subgraph")
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-input", default="image", help="subgraph input name to bind to a LoadImage")
    ap.add_argument("--save-prefix", default="flatten_out")
    ap.add_argument("--bind", action="append", default=[], help="name=kind:value (kind=int/float/bool/string/image)")
    args = ap.parse_args()
    wf = json.load(open(args.subgraph))
    bindings = {args.image_input: ("image", None)}
    for b in args.bind:
        name, kv = _parse_bind(b)
        bindings[name] = ("image", kv[1]) if kv[0] == "image" else kv
    flat = flatten(wf, bindings, args.save_prefix)
    json.dump(flat, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}: {len(flat['nodes'])} nodes, {len(flat['links'])} links")


if __name__ == "__main__":
    main()
