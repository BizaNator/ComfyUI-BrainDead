#!/usr/bin/env python3
"""
ComfyUI Subgraph-Aware Workflow Runner
======================================
This is the canonical copy. `/opt/comfyui/run_workflow_sg.py` on BRAINZ is a
shim that executes this file -- edit it here, not there, or the two drift and
only one of them has the slot fix.

Converts a saved workflow (UI *graph* format) to API format, submits it,
monitors progress, and returns output file paths -- just like run_workflow.py,
but it ALSO understands modern ComfyUI **subgraphs**.

How it differs from run_workflow.py
-----------------------------------
run_workflow.py assumes every node in `workflow["nodes"]` is a real ComfyUI
node whose `type` exists in `/object_info`. Modern workflows instead carry a
top-level `definitions.subgraphs` array -- reusable sub-graphs, each with its
own nodes/links and an input/output interface -- and the main graph contains
"instance" nodes whose `type` is a **UUID** referencing one of those
definitions. run_workflow.py can't expand those: it logs
`[WARN] Node <id> (<uuid>): not found in object_info -- skipping` and silently
DROPS them (including active ones), producing a broken, incomplete API prompt.

This runner first **flattens** the graph -- recursively inlining every subgraph
instance's internal nodes (subgraphs can nest), namespacing node IDs as
`instance:node` to stay unique, and rewiring the subgraph interface (input/output
slots) so links crossing the boundary connect to the parent graph correctly --
then converts the flat graph to an API prompt the same way the ComfyUI front-end
"Queue Prompt" does:

  * mode=4 (bypass) nodes are passed-through (output slot -> matching input by
    type/index; dead if no match), not dropped while active.
  * mode=2 (mute) and mode=4 nodes are excluded from the final prompt.
  * Reroute / PrimitiveNode / Note / MarkdownNote are pure-frontend and are
    resolved/skipped (Reroute passes its input through to consumers).
  * subgraph instances inherit each internal node's own widgets_values, with
    instance-level proxyWidget overrides applied when present.

Same CLI surface as run_workflow.py:
    python3 tools/workflow_to_api.py <workflow.json> [--server http://localhost:8188]
    python3 tools/workflow_to_api.py <workflow.json> --output-dir /tmp/out
    python3 tools/workflow_to_api.py <workflow.json> --dry-run        # print API prompt
    python3 tools/workflow_to_api.py <workflow.json> --oracle <api.json>  # compare-only

The optional --oracle flag diffs the flattened API prompt (by structure /
class_type / edges, not literal IDs) against a UI-produced API export to validate
correctness without a GPU render.
"""

import argparse
import json
import time
import uuid
import sys
import re
import urllib.request
import urllib.parse
import urllib.error
from collections import Counter
from pathlib import Path


DEFAULT_SERVER = "http://10.15.0.20:8188"
CONNECTION_TYPES = {
    "IMAGE", "MASK", "MODEL", "LATENT", "CONDITIONING", "VAE", "CLIP",
    "CONTROL_NET", "UPSCALE_MODEL", "AUDIO", "VIDEO", "MESH", "TRIMESH",
    "HUMAN_PARSE_MAP", "PARTS_BUNDLE", "LOTUS2_MODEL", "TRELLIS2_SHAPE",
    "TRELLIS2_VOXELGRID", "TRELLIS2_TEXTURE",
}

# Pure-frontend node types that never appear in an API prompt.
FRONTEND_TYPES = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode"}

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# Subgraph virtual boundary node IDs (ComfyUI convention).
SG_INPUT_NODE = -10   # interface inputs originate here
SG_OUTPUT_NODE = -20  # interface outputs terminate here


# --------------------------------------------------------------------------- #
# HTTP plumbing (identical behaviour to run_workflow.py)
# --------------------------------------------------------------------------- #
def api_get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def api_post(url, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        try:
            err_json = json.loads(err_body)
            node_errors = err_json.get("node_errors", {})
            if node_errors:
                print(f"\nValidation errors in {len(node_errors)} node(s):", file=sys.stderr)
                for nid, nerr in list(node_errors.items())[:5]:
                    ct = nerr.get("class_type", "?")
                    for e2 in nerr.get("errors", []):
                        print(f"  Node {nid} ({ct}): {e2['details']}", file=sys.stderr)
        except Exception:
            print(f"  Response: {err_body[:500]}", file=sys.stderr)
        raise


WIDGET_PRIMITIVE_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def is_connection_type(inp_schema):
    """Return True if this input schema represents a node-link slot (not a widget)."""
    if not isinstance(inp_schema, (list, tuple)) or not inp_schema:
        return False
    first = inp_schema[0]
    if isinstance(first, list):
        return False  # V1-style COMBO dropdown [["opt1","opt2"], {...}]
    if isinstance(first, str):
        if first == "*":
            return True   # wildcard socket (BD_SaveFile.data) -- never a widget
        if first in WIDGET_PRIMITIVE_TYPES:
            return False
        if first in CONNECTION_TYPES:
            return True
        if first.isupper() and first.replace("_", "").isalnum():
            return True
    return False


# --------------------------------------------------------------------------- #
# Subgraph flattening
# --------------------------------------------------------------------------- #
class FlatGraph:
    """A flattened node/link space built by recursively inlining subgraphs.

    After flattening, every entry in `nodes` is a *real* ComfyUI node (its type
    exists in object_info) keyed by a unique string id (possibly namespaced like
    "38:369" or "7127:c3d:50" for nested instances). `links` is a normalised
    list of edges: (link_id, src_node, src_slot, dst_node, dst_slot, type).
    Node link references inside `inputs[].link` / `outputs[].links` use the same
    normalised link ids.
    """

    def __init__(self):
        self.nodes = {}            # node_id(str) -> node dict (graph-format)
        self.links = {}            # link_id(int) -> [id, src, src_slot, dst, dst_slot, type]
        self._next_link = 1

    def add_link(self, src, src_slot, dst, dst_slot, ltype):
        lid = self._next_link
        self._next_link += 1
        self.links[lid] = [lid, src, src_slot, dst, dst_slot, ltype]
        return lid


def _normalise_subgraph_link(lnk):
    """Subgraph links are dict-form; return (id, origin_id, origin_slot, target_id, target_slot, type)."""
    if isinstance(lnk, dict):
        return (lnk["id"], lnk["origin_id"], lnk["origin_slot"],
                lnk["target_id"], lnk["target_slot"], lnk.get("type"))
    # list-form fallback
    return (lnk[0], lnk[1], lnk[2], lnk[3], lnk[4], lnk[5] if len(lnk) > 5 else None)


def flatten(graph, subgraphs, flat, prefix="", boundary_in=None, boundary_out=None):
    """Recursively inline `graph` (a graph or a subgraph definition) into `flat`.

    prefix       : namespace prefix for node ids ("" at top level, "38:" inside
                   instance 38, "7127:c3d:" two levels deep, ...).
    boundary_in  : dict slot_index -> (parent_node_id, parent_slot, type) describing
                   what feeds each interface INPUT of this subgraph from the parent.
    boundary_out : dict slot_index -> list of (parent_node_id, parent_slot) consumers
                   that should be wired to this subgraph's interface OUTPUTs.
                   (Output wiring is resolved lazily via `out_sources`, see below.)

    Returns: a dict slot_index -> (flat_node_id, flat_slot, type) giving the real
    producer behind each interface OUTPUT slot, so the *caller* can connect parent
    consumers to it.
    """
    boundary_in = boundary_in or {}

    is_subgraph = "inputNode" in graph or "inputs" in graph and "nodes" in graph and prefix
    nodes = graph["nodes"]
    raw_links = graph["links"]

    # Build a per-scope link table (normalised tuples) keyed by link id.
    local_links = {}
    for lnk in raw_links:
        lid, oid, oslot, tid, tslot, ltype = _normalise_subgraph_link(lnk)
        local_links[lid] = (oid, oslot, tid, tslot, ltype)

    # A link records the target slot index it had AT SAVE TIME. That is NOT the
    # current inputs-array index whenever a node hides, collapses or reorders its
    # optional inputs -- BD_PartsBuilder saves mask_labels at slot 11 while its
    # array holds four entries, so a positional write lands on depth_image (or is
    # dropped) and the node fails type validation at queue time while the prompt
    # still reports success. Each node's own inputs[].link IS authoritative, so
    # derive link_id -> index from it and prefer that over the recorded slot.
    slot_of = {}   # local_node_id -> {link_id: inputs-array index}
    for _n in nodes:
        _m = {}
        for _i, _inp in enumerate(_n.get("inputs", []) or []):
            _lk = _inp.get("link")
            if _lk is not None:
                _m[_lk] = _i
        if _m:
            slot_of[_n["id"]] = _m

    def real_tslot(lid, tid, recorded):
        return slot_of.get(tid, {}).get(lid, recorded)

    sub_by_id = {s["id"]: s for s in subgraphs}

    def nid(local_id):
        return f"{prefix}{local_id}"

    # First pass: place every concrete (non-subgraph-instance) node into flat,
    # remembering, for each local node, the mapping local_id -> flat_id.
    # Subgraph-instance nodes are recursed into; we record their interface
    # output sources so links crossing OUT of them can be rewired.
    instance_out_sources = {}   # local_instance_id -> {out_slot: (flat_src, flat_slot, type)}

    # We need to know, for a subgraph instance, how its interface inputs are fed.
    # That depends on parent links targeting the instance node -- which live in
    # THIS scope's local_links. Precompute per local node the inbound links.
    inbound = {}   # local_node_id -> {target_slot: (origin_id, origin_slot, type)}
    for lid, (oid, oslot, tid, tslot, ltype) in local_links.items():
        inbound.setdefault(tid, {})[real_tslot(lid, tid, tslot)] = (oid, oslot, ltype)

    # Resolve, within this scope, the flat producer behind (local_node, out_slot).
    # For a normal node this is just (nid(node), out_slot). For a nested subgraph
    # instance it's whatever its interface output resolved to.
    def resolve_producer(local_node_id, out_slot, ltype):
        if local_node_id in instance_out_sources:
            src = instance_out_sources[local_node_id].get(out_slot)
            return src  # may be None if interface output is unconnected internally
        if local_node_id in (SG_INPUT_NODE,):
            # producer is a boundary input of THIS subgraph -> defer to parent
            return ("__BOUNDARY_IN__", out_slot, ltype)
        return (nid(local_node_id), out_slot, ltype)

    # We must process subgraph instances first (so their out-sources exist before
    # we wire their consumers). Order: recurse all instances, then add links.
    for node in nodes:
        ntype = node["type"]
        if ntype in sub_by_id:
            sub = sub_by_id[ntype]
            inst_prefix = f"{prefix}{node['id']}:"

            # Compute boundary_in for the child: for each interface input slot,
            # what (in THIS scope) feeds the instance node's matching input slot?
            child_boundary_in = {}
            iface_inputs = sub.get("inputs", [])
            iface_outputs = sub.get("outputs", [])
            for slot_idx, iface in enumerate(iface_inputs):
                inst_slot = slot_idx   # interface/instance input slots align by index
                fed = inbound.get(node["id"], {}).get(inst_slot)
                if fed is not None:
                    oid, oslot, ltype = fed
                    prod = resolve_producer(oid, oslot, ltype or iface.get("type"))
                    if prod is not None:
                        child_boundary_in[slot_idx] = prod

            inst_mode = node.get("mode", 0)
            if inst_mode == 2:
                # Muted instance: contributes nothing; all interface outputs are dead.
                instance_out_sources[node["id"]] = {i: None for i in range(len(iface_outputs))}
                continue
            if inst_mode == 4:
                # Bypassed instance: behaves like a bypassed node -- each interface
                # OUTPUT passes through to the instance INPUT with a matching type
                # (else same index). Internal nodes are NOT inlined.
                out_src = {}
                for o_idx, oiface in enumerate(iface_outputs):
                    otype = oiface.get("type")
                    match = None
                    if otype and otype != "*":
                        for i_idx, iiface in enumerate(iface_inputs):
                            if iiface.get("type") == otype:
                                match = i_idx
                                break
                    if match is None and o_idx < len(iface_inputs):
                        match = o_idx
                    out_src[o_idx] = child_boundary_in.get(match) if match is not None else None
                instance_out_sources[node["id"]] = out_src
                continue

            out_src = flatten(sub, subgraphs, flat,
                              prefix=inst_prefix,
                              boundary_in=child_boundary_in)
            # Apply proxyWidget overrides from the instance, if any widgets_values present.
            _apply_proxy_widgets(node, inst_prefix, flat)
            instance_out_sources[node["id"]] = out_src

    # Second pass: place concrete nodes.
    for node in nodes:
        ntype = node["type"]
        if ntype in sub_by_id:
            continue  # already recursed
        new = json.loads(json.dumps(node))   # deep copy
        new["id"] = nid(node["id"])
        # Reset link references; we rewrite them from local_links below.
        for inp in new.get("inputs", []) or []:
            inp["link"] = None
        for out in new.get("outputs", []) or []:
            out["links"] = []
        flat.nodes[new["id"]] = new

    # Third pass: translate every link in this scope into flat.links, rewiring
    # boundary crossings.
    pending_out = {}   # interface out_slot -> (flat_src, flat_slot, type)

    for lid, (oid, oslot, tid, tslot, ltype) in local_links.items():
        # Resolve source side to a flat producer (handles nested instances + boundary-in).
        src = resolve_producer(oid, oslot, ltype)

        # If source is a boundary input of THIS subgraph, pull from parent.
        if src and src[0] == "__BOUNDARY_IN__":
            parent = boundary_in.get(src[1])
            if parent is None:
                continue   # unconnected interface input -> dead link
            src = parent

        if src is None:
            continue   # unconnected nested interface output -> dead link

        # Resolve target side.
        if tid == SG_OUTPUT_NODE:
            # link feeds an interface OUTPUT slot -> record producer for caller.
            pending_out[tslot] = src
            continue
        if tid in sub_by_id_ids(nodes, sub_by_id):
            # link targets a subgraph instance input -> already handled as boundary_in
            continue

        dst_id = nid(tid)
        if dst_id not in flat.nodes:
            continue   # target was a subgraph instance / dropped

        # Register the flat link and patch node slot references.
        dst_slot = real_tslot(lid, tid, tslot)
        flat_lid = flat.add_link(src[0], src[1], dst_id, dst_slot, ltype or src[2])
        _attach_link(flat, flat_lid, src[0], src[1], dst_id, dst_slot)

    # Build the interface output map for the caller.
    out_map = {}
    iface_outputs = graph.get("outputs", [])
    for slot_idx in range(len(iface_outputs)):
        out_map[slot_idx] = pending_out.get(slot_idx)
    return out_map


def sub_by_id_ids(nodes, sub_by_id):
    return {n["id"] for n in nodes if n["type"] in sub_by_id}


def _attach_link(flat, lid, src_id, src_slot, dst_id, dst_slot):
    """Patch the graph-format slot references on the flat nodes for a new link."""
    src = flat.nodes.get(src_id)
    if src is not None:
        outs = src.setdefault("outputs", [])
        while len(outs) <= src_slot:
            outs.append({"name": "", "type": "*", "links": []})
        outs[src_slot].setdefault("links", [])
        outs[src_slot]["links"].append(lid)
    dst = flat.nodes.get(dst_id)
    if dst is not None:
        ins = dst.setdefault("inputs", [])
        # dst_slot has already been resolved to a real inputs-array index by the
        # caller (see real_tslot in flatten); do not re-derive it here.
        if dst_slot < len(ins):
            ins[dst_slot]["link"] = lid


def _apply_proxy_widgets(inst_node, inst_prefix, flat):
    """Push instance-level promoted widget overrides into the inlined internal nodes.

    proxyWidgets = [[internal_node_id, widget_name], ...] maps each promoted widget
    (in instance order) to the internal node + widget it controls. If the instance
    carries widgets_values, those override the internal node's own values; if it is
    empty (the common case), the internal node keeps its own widgets_values.
    """
    proxy = (inst_node.get("properties") or {}).get("proxyWidgets") or []
    values = inst_node.get("widgets_values") or []
    if not proxy or not values:
        return
    for (target_id, _wname), val in zip(proxy, values):
        if str(_wname).startswith("$$"):
            continue   # canvas-only proxy widget (e.g. $$canvas-image-preview)
        fid = f"{inst_prefix}{target_id}"
        node = flat.nodes.get(fid)
        if node is None:
            continue
        # We do not know the internal widget index reliably; record an override
        # keyed by widget name for the converter to apply.
        node.setdefault("_widget_overrides", {})[_wname] = val


# --------------------------------------------------------------------------- #
# Link resolution through pass-through nodes (Reroute / bypass / mute)
# --------------------------------------------------------------------------- #
def build_resolver(flat, subgraph_ids):
    """Return resolve(node_id, out_slot) -> (real_node_id, real_slot) or None.

    Traces transitively through:
      * Reroute / PrimitiveNode  (pass input straight to output)
      * mode=4 bypassed nodes    (output slot -> matching input slot by type, else index)
    Skips mode=2 muted nodes (they produce nothing).
    """
    nodes = flat.nodes
    links = flat.links

    def input_link_source(node_id, in_slot):
        node = nodes.get(node_id)
        if not node:
            return None
        ins = node.get("inputs") or []
        if in_slot >= len(ins):
            return None
        lid = ins[in_slot].get("link")
        if lid is None or lid not in links:
            return None
        _, src, sslot, _, _, _ = links[lid]
        return (src, sslot)

    def bypass_input_for_output(node, out_slot):
        """For a bypassed node, find the input slot feeding output `out_slot`.

        ComfyUI matches by type: the first input whose type equals the output's
        type; falls back to same index when types are wildcards/unknown.
        """
        ins = node.get("inputs") or []
        outs = node.get("outputs") or []
        otype = outs[out_slot].get("type") if out_slot < len(outs) else None
        if otype and otype != "*":
            for idx, inp in enumerate(ins):
                if inp.get("type") == otype:
                    return idx
        # fallback: same index if it exists and is linkable
        if out_slot < len(ins):
            return out_slot
        return None

    cache = {}

    def resolve(node_id, out_slot, _seen=None):
        key = (node_id, out_slot)
        if key in cache:
            return cache[key]
        _seen = _seen or set()
        if key in _seen:
            return None   # cycle guard
        _seen = _seen | {key}

        node = nodes.get(node_id)
        if node is None:
            return None
        ntype = node.get("type")
        mode = node.get("mode", 0)

        # Muted node -> nothing.
        if mode == 2:
            cache[key] = None
            return None

        is_passthrough = ntype in ("Reroute", "PrimitiveNode")
        is_bypass = mode == 4

        if is_passthrough:
            src = input_link_source(node_id, 0)
            res = resolve(src[0], src[1], _seen) if src else None
            cache[key] = res
            return res

        if is_bypass:
            in_slot = bypass_input_for_output(node, out_slot)
            if in_slot is None:
                cache[key] = None
                return None
            src = input_link_source(node_id, in_slot)
            res = resolve(src[0], src[1], _seen) if src else None
            cache[key] = res
            return res

        # Real, active node -> it IS the producer.
        cache[key] = (node_id, out_slot)
        return (node_id, out_slot)

    return resolve


# --------------------------------------------------------------------------- #
# Flat graph -> API prompt
# --------------------------------------------------------------------------- #
def widget_values_by_name(node, all_inputs_schema):
    """Map widget name -> saved value by walking the schema in declaration order.

    `widgets_values` is positional, written in the order the frontend lays the
    widgets out, which is the schema's declaration order (required then optional).
    A widget that has been converted to a link KEEPS its slot, so link state
    cannot be used to walk the list -- the value is read and discarded.

    Widget-ness comes from the schema, with the graph allowed to override it: a
    dynamic combo (COMFY_DYNAMICCOMBO_V3) reads as a socket type but is a widget,
    and the graph marks that slot accordingly. Its expansion `parent.sub` slots
    are not declared in the schema at all and are spliced in after the parent.

    The node's `inputs` array is NOT used for ordering. It is not a complete
    widget list -- BD-parts_pointclick saves BD_PartsExport with four entries
    against sixteen widget values.

    A seed INT is followed in `widgets_values` by a UI-only control_after_generate
    value that has no input entry. It is dropped when the list is longer than the
    number of widgets still to fill.
    """
    vals = node.get("widgets_values")
    if isinstance(vals, dict):
        return dict(vals)          # newer saves may store it keyed already
    q = list(vals or [])
    if not q:
        return {}
    entries = node.get("inputs") or []
    marked_widget = {e["name"] for e in entries if "widget" in e}
    names = []
    for name, sch in all_inputs_schema.items():
        if is_connection_type(sch) and name not in marked_widget:
            continue
        names.append(name)
        # A dynamic combo expands into `parent.sub` slots the schema never
        # declares; the widget ones sit directly after the parent.
        prefix = name + "."
        names.extend(e["name"] for e in entries
                     if e["name"].startswith(prefix) and "widget" in e)
    out = {}
    for idx, name in enumerate(names):
        if not q:
            break
        out[name] = q.pop(0)
        sch = all_inputs_schema.get(name)
        remaining = len(names) - idx - 1
        if (sch and isinstance(sch, (list, tuple)) and sch
                and isinstance(sch[0], str) and sch[0] == "INT"
                and "seed" in name.lower()
                and len(q) > remaining):
            q.pop(0)               # control_after_generate
    return out


def flat_to_api(flat, object_info):
    """Convert the flattened graph to an API prompt, resolving pass-through nodes."""
    nodes = flat.nodes
    links = flat.links
    resolve = build_resolver(flat, set())

    api = {}
    errors = []

    for node_id, node in nodes.items():
        ntype = node["type"]
        mode = node.get("mode", 0)

        if ntype in FRONTEND_TYPES:
            continue
        if mode in (2, 4):   # muted / bypassed -> excluded from prompt
            continue

        schema = object_info.get(ntype)
        if schema is None:
            errors.append(f"Node {node_id} ({ntype}): not found in object_info -- skipping")
            continue

        # input_name -> resolved [src_node, src_slot] for linked inputs.
        link_map = {}
        for slot_def in node.get("inputs", []) or []:
            lk = slot_def.get("link")
            if lk is None or lk not in links:
                continue
            _, src, sslot, _, _, _ = links[lk]
            real = resolve(src, sslot)
            if real is not None:
                link_map[slot_def["name"]] = [str(real[0]), real[1]]

        overrides = node.get("_widget_overrides", {})

        all_inputs_schema = {
            **schema.get("input", {}).get("required", {}),
            **schema.get("input", {}).get("optional", {}),
        }
        widget_vals = widget_values_by_name(node, all_inputs_schema)

        api_inputs = {}
        for inp_name, inp_schema in all_inputs_schema.items():
            if inp_name in link_map:
                api_inputs[inp_name] = link_map[inp_name]
            elif inp_name in overrides:
                api_inputs[inp_name] = overrides[inp_name]
            elif inp_name in widget_vals:
                api_inputs[inp_name] = widget_vals[inp_name]
            # else: unlinked socket, or a widget the save predates -> omit and
            # let the server apply its own default.

        # Dynamic (graph-only) inputs not present in the schema. A dynamic combo
        # expands into sub-inputs the schema never declares -- ResizeImageMaskNode
        # declares `resize_type` (COMFY_DYNAMICCOMBO_V3) and the graph carries
        # `resize_type.match` (a socket) and `resize_type.crop` (a widget). Both
        # forms have to be forwarded or the server falls back to a default the
        # user never chose.
        schema_names = set(all_inputs_schema.keys())
        for slot_def in node.get("inputs", []) or []:
            name = slot_def["name"]
            if name in schema_names:
                continue
            if name in link_map:
                api_inputs[name] = link_map[name]
            elif (name in widget_vals and "." in name
                  and name.split(".", 1)[0] in schema_names):
                # `parent.sub` is a dynamic combo's sub-widget and belongs in the
                # prompt. A bare name that the schema does not declare is a
                # frontend-only widget -- LoadImage's `upload` button -- and the
                # UI's own export leaves it out.
                api_inputs[name] = widget_vals[name]

        api[str(node_id)] = {
            "class_type": ntype,
            "inputs": api_inputs,
            "_meta": {"title": node.get("title", ntype)},
        }

    if errors:
        for e in errors:
            print(f"  [WARN] {e}", file=sys.stderr)

    return api


def workflow_to_api(workflow, object_info):
    """Top-level: flatten subgraphs (if any) then convert to API prompt."""
    subgraphs = (workflow.get("definitions") or {}).get("subgraphs", [])
    flat = FlatGraph()
    flatten(workflow, subgraphs, flat, prefix="")
    return flat_to_api(flat, object_info)


# --------------------------------------------------------------------------- #
# Oracle comparison (structural, ID-agnostic)
# --------------------------------------------------------------------------- #
def _edge_signature(api):
    """Return (class_type_counter, edge_class_set) describing an API prompt's
    structure independent of literal node IDs.

    edge = (src_class_type, src_slot, dst_class_type, dst_input_name)
    """
    ct = Counter(v["class_type"] for v in api.values())
    cls = {nid: v["class_type"] for nid, v in api.items()}
    edges = Counter()
    for nid, v in api.items():
        dst_ct = v["class_type"]
        for inp_name, val in v["inputs"].items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                src_id, src_slot = val[0], val[1]
                src_ct = cls.get(src_id, "<EXTERNAL>")
                edges[(src_ct, src_slot, dst_ct, inp_name)] += 1
    return ct, edges


def compare_to_oracle(mine, oracle):
    mc, me = _edge_signature(mine)
    oc, oe = _edge_signature(oracle)

    print("\n=== ORACLE COMPARISON (structural) ===")
    print(f"  nodes: mine={len(mine)}  oracle={len(oracle)}")

    only_mine_ct = mc - oc
    only_oracle_ct = oc - mc
    print(f"  class_type multiset match: {only_mine_ct == Counter() and only_oracle_ct == Counter()}")
    if only_mine_ct:
        print("    extra class_types in mine:", dict(only_mine_ct))
    if only_oracle_ct:
        print("    missing class_types vs oracle:", dict(only_oracle_ct))

    only_mine_e = me - oe
    only_oracle_e = oe - me
    print(f"  edges: mine={sum(me.values())}  oracle={sum(oe.values())}")
    print(f"  edge multiset match: {only_mine_e == Counter() and only_oracle_e == Counter()}")
    matched = sum((me & oe).values())
    total = sum((me | oe).values())
    print(f"  edge overlap: {matched}/{total} ({100*matched/total:.1f}%)")
    if only_oracle_e:
        print(f"  edges in oracle but NOT mine ({sum(only_oracle_e.values())}):")
        for e, c in list(only_oracle_e.most_common(15)):
            print(f"    -{c}x  {e}")
    if only_mine_e:
        print(f"  edges in mine but NOT oracle ({sum(only_mine_e.values())}):")
        for e, c in list(only_mine_e.most_common(15)):
            print(f"    +{c}x  {e}")

    ok = (only_mine_ct == Counter() and only_oracle_ct == Counter()
          and only_mine_e == Counter() and only_oracle_e == Counter())
    print(f"\n  RESULT: {'MATCH' if ok else 'MISMATCH'}")
    return ok


# --------------------------------------------------------------------------- #
# Submit / poll / collect (identical behaviour to run_workflow.py)
# --------------------------------------------------------------------------- #
def queue_prompt(prompt_api, server, client_id, ui_graph=None):
    """Submit, and hand ComfyUI the UI graph for provenance.

    BD save nodes write a workflow sidecar beside each output, but only from what
    the CLIENT posts -- the web front-end always sends
    extra_data.extra_pnginfo.workflow; a headless caller has to do it explicitly.
    Omit it and every sidecar records "workflow": null, which looks fine in a
    directory listing and is worthless when someone needs to reproduce the run.
    """
    payload = {"client_id": client_id, "prompt": prompt_api}
    if ui_graph is not None:
        payload["extra_data"] = {"extra_pnginfo": {"workflow": ui_graph}}
    result = api_post(f"{server}/prompt", payload)
    return result.get("prompt_id")


def wait_for_completion(prompt_id, server, poll_interval=3.0, timeout=1800,
                        max_conn_retries=20, conn_retry_delay=5.0):
    deadline = time.time() + timeout
    last_status = ""
    conn_failures = 0
    while time.time() < deadline:
        try:
            queue = api_get(f"{server}/queue")
            conn_failures = 0
        except Exception:
            conn_failures += 1
            if conn_failures >= max_conn_retries:
                print(f"\n  Server unreachable after {conn_failures} attempts -- giving up", file=sys.stderr)
                return None
            status = f"server down, retry {conn_failures}/{max_conn_retries}..."
            if status != last_status:
                print(f"\r  Status: {status}                    ", end="", flush=True)
                last_status = status
            time.sleep(conn_retry_delay)
            continue

        running = [p[1] for p in queue.get("queue_running", [])]
        pending = [p[1] for p in queue.get("queue_pending", [])]

        if prompt_id in running:
            status = f"running  (pending={len(pending)})"
        elif prompt_id in pending:
            status = f"queued   (position {pending.index(prompt_id)+1}/{len(pending)})"
        else:
            try:
                hist = api_get(f"{server}/history/{prompt_id}")
            except Exception:
                hist = {}
            if prompt_id in hist:
                print(f"\r  Status: done                              ")
                return hist[prompt_id]
            status = "not in queue/history (server restarted?)"

        if status != last_status:
            print(f"\r  Status: {status}                    ", end="", flush=True)
            last_status = status

        time.sleep(poll_interval)

    return None


def collect_outputs(history_entry, server, output_dir=None):
    outputs = history_entry.get("outputs", {})
    results = []
    for node_id, node_output in outputs.items():
        for media_type in ("images", "gifs", "audio"):
            for item in node_output.get(media_type, []):
                fname = item.get("filename", "")
                subfolder = item.get("subfolder", "")
                ftype = item.get("type", "output")
                params = urllib.parse.urlencode({"filename": fname, "subfolder": subfolder, "type": ftype})
                url = f"{server}/view?{params}"
                results.append({"node_id": node_id, "filename": fname, "subfolder": subfolder, "url": url})
                if output_dir:
                    dest_dir = Path(output_dir) / subfolder if subfolder else Path(output_dir)
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = dest_dir / fname
                    if not dest.exists():
                        print(f"  Downloading {fname}...")
                        urllib.request.urlretrieve(url, dest)
                        print(f"  Saved -> {dest}")
                    else:
                        print(f"  Already exists: {dest}")
    return results


def main():
    ap = argparse.ArgumentParser(description="Run a subgraph-aware ComfyUI workflow and collect outputs.")
    ap.add_argument("workflow", help="Path to workflow JSON (UI graph format, may contain subgraphs)")
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--output-dir", default=None, help="Download outputs to this directory")
    ap.add_argument("--timeout", type=int, default=1800, help="Max seconds to wait (default 1800)")
    ap.add_argument("--dry-run", action="store_true", help="Convert to API format and print, don't submit")
    ap.add_argument("--oracle", default=None, help="Compare flattened API prompt against this UI API export and exit")
    args = ap.parse_args()

    print(f"Server  : {args.server}")
    print(f"Workflow: {args.workflow}")

    with open(args.workflow) as f:
        workflow = json.load(f)

    n_sub = len((workflow.get("definitions") or {}).get("subgraphs", []))
    n_inst = sum(1 for n in workflow.get("nodes", [])
                 if isinstance(n.get("type"), str) and UUID_RE.match(n["type"]))
    print(f"Subgraph defs: {n_sub}   instances in main graph: {n_inst}")

    print("Fetching node schemas...", end=" ", flush=True)
    object_info = api_get(f"{args.server}/object_info")
    print(f"OK ({len(object_info)} node types)")

    print("Flattening + converting workflow...", end=" ", flush=True)
    api_prompt = workflow_to_api(workflow, object_info)
    print(f"OK ({len(api_prompt)} nodes)")

    if args.oracle:
        with open(args.oracle) as f:
            oracle = json.load(f)
        ok = compare_to_oracle(api_prompt, oracle)
        sys.exit(0 if ok else 2)

    if args.dry_run:
        print(json.dumps(api_prompt, indent=2)[:3000], "...")
        return

    client_id = str(uuid.uuid4())
    print(f"Queuing prompt (client={client_id[:8]}...)...", end=" ", flush=True)
    # hand ComfyUI the UI graph so BD save nodes can write a real sidecar
    prompt_id = queue_prompt(api_prompt, args.server, client_id, ui_graph=workflow)
    if not prompt_id:
        print("FAILED -- server returned no prompt_id", file=sys.stderr)
        sys.exit(1)
    print(f"OK -> prompt_id={prompt_id}")

    print(f"Waiting for completion (timeout={args.timeout}s)...")
    history = wait_for_completion(prompt_id, args.server, timeout=args.timeout)
    if history is None:
        print("TIMEOUT -- workflow did not complete in time", file=sys.stderr)
        sys.exit(1)

    status = history.get("status", {})
    if status.get("status_str") == "error":
        print("\nWORKFLOW ERROR:")
        for msg in status.get("messages", []):
            print(f"  {msg}")
        sys.exit(1)

    print("Collecting outputs...")
    results = collect_outputs(history, args.server, args.output_dir)
    if not results:
        print("  No output files found.")
    else:
        print(f"\n=== {len(results)} output file(s) ===")
        for r in results:
            print(f"  [{r['node_id']}] {r['subfolder']}/{r['filename']}" if r['subfolder'] else f"  [{r['node_id']}] {r['filename']}")

    print("\n--- JSON RESULT ---")
    print(json.dumps({
        "prompt_id": prompt_id,
        "status": status.get("status_str", "unknown"),
        "outputs": results,
    }))


if __name__ == "__main__":
    main()
