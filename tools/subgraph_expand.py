#!/usr/bin/env python3
"""subgraph_expand.py — recursive UI-graph subgraph inliner (ComfyUI-BrainDead).

Expands subgraph INSTANCE nodes (UUID-typed) in a UI-format workflow graph by
inlining their definition's nodes and remapping boundary links, recursively to a
bounded depth. ComfyUI subgraph defs (definitions.subgraphs[]) declare ordered
inputs[]/outputs[] whose linkIds wire the boundary (-10 inputNode / -20 outputNode)
into the interior; a top-level instance node's inputs[]/outputs[] carry the
external links in the same slot order. Boundary ORDER is preserved exactly —
RinChar's silent-failure case (GLSLShader with 36 ordered boundary inputs) is the
canonical test.

API:
    expanded = expand_graph(workflow_dict, max_depth=4)
returns a NEW workflow dict {nodes, links, ...} with no UUID-typed nodes left.
"""
import copy

BOUND_IN, BOUND_OUT = -10, -20


def _defs_by_uuid(workflow):
    defs = workflow.get("definitions", {}).get("subgraphs", [])
    return {d.get("id"): d for d in defs}


def _link_map(links):
    return {l.get("id"): l for l in links}


def _to_dict(l):
    """ComfyUI top-level links are arrays [id, origin_id, origin_slot, target_id, target_slot, type];
    subgraph-internal links are already dicts."""
    if isinstance(l, dict):
        return l
    return {"id": l[0], "origin_id": l[1], "origin_slot": l[2],
            "target_id": l[3], "target_slot": l[4], "type": l[5] if len(l) > 5 else "*"}


def _to_array(l):
    return [l["id"], l["origin_id"], l.get("origin_slot", 0),
            l["target_id"], l.get("target_slot", 0), l.get("type", "*")]


def expand_graph(workflow, max_depth=4):
    g = copy.deepcopy(workflow)
    defs = _defs_by_uuid(g)
    if not defs:
        return g
    nodes = list(g.get("nodes", []))
    links = [_to_dict(l) for l in g.get("links", [])]
    for depth in range(max_depth):
        uuid_types = {n.get("type") for n in nodes if n.get("type") in defs}
        if not uuid_types:
            break
        nodes, links = _expand_once(nodes, links, defs)
        # rewrite slot_def refs NOW (not at the end) so the NEXT nesting level's
        # boundary lookups resolve against the fresh ids (BUG 1c fix)
        nodes = _rewrite_slot_defs(nodes, links)
    nodes, links = _trace_reroutes(nodes, links)
    nodes, links = _inline_primitives(nodes, links)
    return {**g, "nodes": nodes, "links": [_to_array(l) for l in links]}


def _rewrite_slot_defs(nodes, links):
    in_fix = {}
    out_fix = {}
    for l in links:
        in_fix[(l["target_id"], l.get("target_slot", 0))] = l["id"]
        out_fix.setdefault((l["origin_id"], l.get("origin_slot", 0)), []).append(l["id"])
    for n in nodes:
        for idx, slot in enumerate(n.get("inputs", []) or []):
            new_lid = in_fix.get((n.get("id"), slot.get("slot_index", idx))) or in_fix.get((n.get("id"), idx))
            if new_lid is not None:
                slot["link"] = new_lid
        for idx, slot in enumerate(n.get("outputs", []) or []):
            new_lids = out_fix.get((n.get("id"), slot.get("slot_index", idx))) or out_fix.get((n.get("id"), idx))
            if new_lids is not None:
                slot["links"] = new_lids
    return nodes


def _trace_reroutes(nodes, links):
    """Reroute nodes are dropped by the API converter — rewire links THROUGH them
    to the true upstream producer (BUG 1a fix)."""
    rr_ids = {n.get("id") for n in nodes if n.get("type") == "Reroute"}
    if not rr_ids:
        return nodes, links
    by_target = {}
    for l in links:
        by_target.setdefault(l["target_id"], []).append(l)

    def upstream(nid, guard=0):
        if guard > 64:
            return nid, 0
        ups = [l for l in by_target.get(nid, [])]
        if not ups:
            return nid, 0
        u = ups[0]
        if u["origin_id"] in rr_ids:
            return upstream(u["origin_id"], guard + 1)
        return u["origin_id"], u.get("origin_slot", 0)

    new_nodes = [n for n in nodes if n.get("type") != "Reroute"]
    new_links = []
    for l in links:
        if l["origin_id"] in rr_ids:
            o, os_ = upstream(l["origin_id"])
            if o in rr_ids:
                continue
            l = {**l, "origin_id": o, "origin_slot": os_}
        if l["target_id"] in rr_ids:
            continue  # inbound edge to a deleted reroute is absorbed by the rewiring above
        new_links.append(l)
    return new_nodes, new_links


def _inline_primitives(nodes, links):
    """PrimitiveNode literal values must be inlined into consumer widget slots
    (frontend-only nodes; BUG 1b fix)."""
    prims = {n.get("id"): n for n in nodes if n.get("type") == "PrimitiveNode"}
    if not prims:
        return nodes, links
    val_of = {pid: (n.get("widgets_values") or [None])[0] for pid, n in prims.items()}
    consumer_links = [l for l in links if l["origin_id"] in prims]
    by_id = {n.get("id"): n for n in nodes}
    for l in consumer_links:
        tgt = by_id.get(l["target_id"])
        if tgt is None:
            continue
        # find the linked input slot; primitives feed widget-converted inputs whose
        # slot_def carries the link id — write the literal into the matching
        # widgets_values position (index among that node's widget slots)
        tslot = l.get("target_slot", 0)
        wv = tgt.get("widgets_values")
        if not isinstance(wv, list) or not wv:
            continue
        # map slot to widget position: widget-converted inputs appear as input slots
        # with a link; their widget index follows the node's widget layout — the
        # linked input slot index among inputs that are widget-like
        inputs = tgt.get("inputs", []) or []
        widget_like = [i for i, s in enumerate(inputs) if s.get("widget") is not None or s.get("link") is not None]
        if tslot < len(inputs):
            try:
                wpos = widget_like.index(tslot)
            except ValueError:
                wpos = tslot
        else:
            wpos = tslot
        if wpos < len(wv):
            wv[wpos] = val_of[l["origin_id"]]
        else:
            wv.append(val_of[l["origin_id"]])
    keep_links = [l for l in links if l["origin_id"] not in prims]
    keep_nodes = [n for n in nodes if n.get("type") != "PrimitiveNode"]
    return keep_nodes, keep_links


def _expand_once(nodes, links, defs):
    new_nodes = [n for n in nodes if n.get("type") not in defs]
    new_links = []
    in_fix = {}    # (target_id, target_slot) -> new link id  (for inputs[].link rewrite)
    out_fix = {}   # (origin_id, origin_slot) -> [new link ids] (for outputs[].links rewrite)

    def _tgt(l):
        in_fix[(l["target_id"], l.get("target_slot", 0))] = l["id"]
    def _src(l):
        out_fix.setdefault((l["origin_id"], l.get("origin_slot", 0)), []).append(l["id"])

    for l in links:
        # keep links that don't touch instances yet; boundary links are rebuilt per instance
        if l.get("target_id") in {n.get("id") for n in nodes if n.get("type") in defs} or \
           l.get("origin_id") in {n.get("id") for n in nodes if n.get("type") in defs}:
            continue
        new_links.append(l)
        _tgt(l); _src(l)

    for inst in [n for n in nodes if n.get("type") in defs]:
        sg = defs[inst["type"]]
        iid = inst["id"]
        fresh = lambda x: f"{iid}__{x}"
        # incremental lookup of links emitted so far (sibling-instance interiors
        # emitted earlier in THIS pass must resolve as boundary sources)
        ext_link_map = {l["id"]: l for l in new_links}

        # 1. inline interior nodes (boundary marker nodes are not real nodes)
        for sn in sg.get("nodes", []):
            nn = copy.deepcopy(sn)
            nn["id"] = fresh(sn["id"])
            if "properties" in nn and isinstance(nn["properties"], dict):
                nn["properties"] = {**nn["properties"], "subgraph_instance": iid}
            new_nodes.append(nn)

        # 2. inline interior links (boundary links handled below)
        for sl in sg.get("links", []):
            if sl.get("origin_id") in (BOUND_IN, BOUND_OUT) or sl.get("target_id") in (BOUND_IN, BOUND_OUT):
                continue
            nl = dict(sl)
            nl["id"] = fresh(sl["id"])
            nl["origin_id"] = fresh(sl["origin_id"])
            nl["target_id"] = fresh(sl["target_id"])
            new_links.append(nl)
            _tgt(nl); _src(nl)

        # 3. boundary INPUTS: external link into instance slot i fans out to the
        #    def's interior targets via inputs[i].linkIds (ORDER PRESERVED)
        inst_inputs = inst.get("inputs", [])
        for slot_idx, sg_in in enumerate(sg.get("inputs", [])):
            ext = None
            if slot_idx < len(inst_inputs):
                lid = inst_inputs[slot_idx].get("link")
                if lid is not None:
                    ext = ext_link_map.get(lid) or _link_map(links).get(lid)
            if ext is None:
                # also accept links already targeting this instance node directly
                # (e.g. from a sibling instance's inlined interior, BUG-1c class)
                cands = [l for l in new_links if l["target_id"] == iid and l.get("target_slot", 0) == slot_idx]
                if cands:
                    ext = cands[0]
                    new_links.remove(ext)
            if ext is None:
                continue
            for ilid in sg_in.get("linkIds", []):
                il = _link_map(sg.get("links", [])).get(ilid)
                if il is None:
                    continue
                nl = {
                    "id": fresh(f"bin_{slot_idx}_{ilid}"),
                    "origin_id": ext["origin_id"],
                    "origin_slot": ext.get("origin_slot", 0),
                    "target_id": fresh(il["target_id"]),
                    "target_slot": il.get("target_slot", 0),
                    "type": il.get("type", ext.get("type", "*")),
                }
                new_links.append(nl)
                _tgt(nl); _src(nl)

        # 4. boundary OUTPUTS: interior sources via outputs[j].linkIds fan out to
        #    the instance's external consumers (ORDER PRESERVED)
        inst_outputs = inst.get("outputs", [])
        for slot_idx, sg_out in enumerate(sg.get("outputs", [])):
            ext_links = []
            if slot_idx < len(inst_outputs):
                for lid in inst_outputs[slot_idx].get("links", []) or []:
                    e = _link_map(links).get(lid)
                    if e is not None:
                        ext_links.append(e)
            if not ext_links:
                continue
            for ilid in sg_out.get("linkIds", []):
                il = _link_map(sg.get("links", [])).get(ilid)
                if il is None:
                    continue
                for ext in ext_links:
                    nl = {
                        "id": fresh(f"bout_{slot_idx}_{ilid}_{ext['id']}"),
                        "origin_id": fresh(il["origin_id"]),
                        "origin_slot": il.get("origin_slot", 0),
                        "target_id": ext["target_id"],
                        "target_slot": ext.get("target_slot", 0),
                        "type": il.get("type", ext.get("type", "*")),
                    }
                    new_links.append(nl)
                    _tgt(nl); _src(nl)

    # post-pass sweep: any link still targeting a RAW instance id (emitted by a
    # sibling instance AFTER that instance was processed) gets rewired into the
    # target instance's boundary interior (order-independent, BUG-1c class)
    raw_ids = {n.get("id") for n in nodes if n.get("type") in defs}
    leftover = [l for l in new_links if l["target_id"] in raw_ids]
    new_links = [l for l in new_links if l["target_id"] not in raw_ids]
    for l in leftover:
        inst = next(n for n in nodes if n.get("id") == l["target_id"])
        sg = defs[inst["type"]]
        iid = inst["id"]
        fresh = lambda x: f"{iid}__{x}"
        slot_idx = l.get("target_slot", 0)
        if slot_idx >= len(sg.get("inputs", [])):
            continue
        sg_in = sg["inputs"][slot_idx]
        for ilid in sg_in.get("linkIds", []):
            il = _link_map(sg.get("links", [])).get(ilid)
            if il is None:
                continue
            nl = {
                "id": fresh(f"bin_{slot_idx}_{ilid}_x{l['id']}"),
                "origin_id": l["origin_id"],
                "origin_slot": l.get("origin_slot", 0),
                "target_id": fresh(il["target_id"]),
                "target_slot": il.get("target_slot", 0),
                "type": il.get("type", l.get("type", "*")),
            }
            new_links.append(nl)
    return new_nodes, new_links
