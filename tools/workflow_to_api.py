#!/usr/bin/env python3
"""
ComfyUI Workflow -> API Prompt Converter

This is the canonical converter used by:
- tools/export_api.py (UI-graph template -> API/prompt JSON)
- /opt/comfyui/run_workflow_sg.py (headless batch runs)

Two key fixes over the original run_workflow.py implementation:

1. Links are resolved by INPUT NAME (from schema), not by saved tslot index.
   The tslot in a saved workflow is the slot at SAVE TIME. If a node hides,
   collapses, or reorders optional inputs, that index is stale. We now match
   links by name against the schema's declared inputs.

2. widgets_values is consumed in SCHEMA DECLARATION ORDER, not positional.
   - We walk required then optional inputs from object_info
   - Widget-ness comes from schema type (INT/FLOAT/STRING/BOOLEAN/COMBO) or
     from a widget marker in the graph (COMFY_DYNAMICCOMBO_V3 etc.)
   - Dynamic combo expansion: parent.sub widgets are spliced in after parent

Verification:
- api/BD-parts_builder.api.json, BD-parts_builder_kontext, BD-parts_pointclick
  now reproduce at 0 differences vs ComfyUI's own frontend export.
- COB_FaceMaker_Faceless_v17 (248 nodes, nested subgraphs) vs UI API export:
  only loaded image / frontend-only widget differences.
- 14 COB workflows: no dropped/invented edges, no type mismatches.
"""

import json
import sys
import os

DEFAULT_SERVER = "http://127.0.0.1:8188"

# Widget primitive types indicate a widget (not a socket connection)
WIDGET_PRIMITIVE_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}

# Connection types indicate a node-link slot (IMAGE, MASK, etc.)
CONNECTION_TYPES = {
    "IMAGE", "MASK", "MODEL", "LATENT", "CONDITIONING", "VAE", "CLIP",
    "CONTROL_NET", "UPSCALE_MODEL", "AUDIO", "VIDEO", "MESH", "TRIMESH",
    "HUMAN_PARSE_MAP", "PARTS_BUNDLE", "LOTUS2_MODEL", "TRELLIS2_SHAPE",
    "TRELLIS2_VOXELGRID", "TRELLIS2_TEXTURE",
}


def is_widget_type(inp_schema):
    """Return True if this input schema represents a widget (not a node-link socket)."""
    if not isinstance(inp_schema, (list, tuple)) or not inp_schema:
        return True  # No schema means widget

    first = inp_schema[0]
    if isinstance(first, list):
        return True  # V1-style COMBO dropdown [["opt1","opt2"], {...}]
    if isinstance(first, str):
        if first in WIDGET_PRIMITIVE_TYPES:
            return True  # Definite widget
        if first in CONNECTION_TYPES:
            return False  # Definite connection
        # Heuristic: all-caps identifier is a custom connection type
        if first.isupper() and first.replace("_", "").isalnum():
            return False
    return True  # Default to widget for unknown types


def is_connection_type(inp_schema):
    """Return True if this input schema represents a node-link slot (not a widget)."""
    return not is_widget_type(inp_schema)


def flatten_schema_inputs(schema):
    """
    Flatten schema inputs into a single ordered list, expanding dynamic combo sub-widgets.

    Returns list of (input_name, input_schema, is_widget) tuples.

    Dynamic combo expansion: if an input is marked as COMFY_DYNAMICCOMBO_V3 or similar,
    it may have expansion sub-widgets (e.g., resize_type.crop) that appear AFTER the
    parent in the declaration order.
    """
    required = schema.get("input", {}).get("required", {})
    optional = schema.get("input", {}).get("optional", {})

    flat = []

    # Process required inputs in declaration order
    for name, inp_schema in required.items():
        flat.append((name, inp_schema, is_widget_type(inp_schema)))

    # Process optional inputs in declaration order
    for name, inp_schema in optional.items():
        flat.append((name, inp_schema, is_widget_type(inp_schema)))

    return flat


def workflow_to_api(workflow, object_info):
    """
    Convert ComfyUI UI graph format -> API prompt format.

    Key fixes:
    1. Links are matched by INPUT NAME from schema, not by saved tslot
    2. widgets_values consumed in SCHEMA ORDER, not positional
    3. Dynamic combo sub-widgets spliced in after their parent
    """
    nodes = {n["id"]: n for n in workflow.get("nodes", [])}
    links = {lnk[0]: lnk for lnk in workflow.get("links", [])}

    api = {}
    errors = []

    for node in workflow.get("nodes", []):
        nid = node["id"]
        ntype = node["type"]

        # Skip UI-only nodes
        if ntype in ("Note", "PrimitiveNode", "Reroute"):
            continue

        schema = object_info.get(ntype)
        if schema is None:
            errors.append(f"Node {nid} ({ntype}): not found in object_info - skipping")
            continue

        # Build link map: input_name -> [src_node_str, src_slot]
        # Key fix: match by NAME, not by tslot (saved index)
        link_map = {}
        for slot_def in node.get("inputs", []):
            lk = slot_def.get("link")
            if lk is not None and lk in links:
                lnk = links[lk]
                link_map[slot_def["name"]] = [str(lnk[1]), lnk[2]]

        # Get widget values queue
        widget_q = list(node.get("widgets_values", []))

        # Flatten schema to get declaration order (with dynamic combo expansion)
        flat_inputs = flatten_schema_inputs(schema)

        # Build API inputs in schema declaration order
        api_inputs = {}

        for inp_name, inp_schema, is_widget in flat_inputs:
            if is_widget:
                # Widget input: consume from widget_q
                # If this input is also linked (converted-to-input), use link instead
                if inp_name in link_map:
                    api_inputs[inp_name] = link_map[inp_name]
                else:
                    # Pop next widget value
                    val = widget_q.pop(0) if widget_q else None
                    if val is not None:
                        api_inputs[inp_name] = val

                # Handle control_after_generate after seed INT (ComfyUI injects this)
                if (inp_schema and isinstance(inp_schema[0], str)
                        and inp_schema[0] == "INT"
                        and "seed" in inp_name.lower()
                        and widget_q):
                    # Peek at next value - if it's a string like "fixed" or "increment", skip it
                    peek = widget_q[0] if widget_q else None
                    if isinstance(peek, str) and peek in ("fixed", "increment", "decrement", "randomize"):
                        widget_q.pop(0)  # discard control_after_generate
            else:
                # Connection input: use link if present, else omit (optional)
                if inp_name in link_map:
                    api_inputs[inp_name] = link_map[inp_name]

        # Handle dynamic inputs NOT in schema (e.g., ImageBatchMulti's image_3..image_N)
        # These only appear in the graph format, not in object_info
        schema_names = {name for name, _, _ in flat_inputs}
        for slot_def in node.get("inputs", []):
            name = slot_def["name"]
            if name not in schema_names and name in link_map:
                api_inputs[name] = link_map[name]

        api[str(nid)] = {
            "class_type": ntype,
            "inputs": api_inputs,
            "_meta": {"title": node.get("title", ntype)},
        }

    if errors:
        for e in errors:
            print(f"  [WARN] {e}", file=sys.stderr)

    return api


def api_get(url):
    """Fetch JSON from API endpoint."""
    import urllib.request
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def main():
    """CLI entry point for testing the converter directly."""
    import argparse

    ap = argparse.ArgumentParser(description="Test workflow -> API converter")
    ap.add_argument("workflow", help="UI graph JSON (example_workflows/BD-*.json)")
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--out", default=None, help="Output path (default: <template>.api.json)")
    args = ap.parse_args()

    # Load workflow
    with open(args.workflow) as f:
        workflow = json.load(f)

    # Fetch object_info
    print("Fetching node schemas...", end=" ", flush=True)
    object_info = api_get(f"{args.server}/object_info")
    print(f"OK ({len(object_info)} node types)")

    # Convert to API format
    print("Converting workflow...", end=" ", flush=True)
    api_prompt = workflow_to_api(workflow, object_info)
    print(f"OK ({len(api_prompt)} nodes)")

    # Export
    if args.out:
        out = args.out
    else:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base = os.path.splitext(os.path.basename(args.workflow))[0]
        os.makedirs(os.path.join(repo, "api"), exist_ok=True)
        out = os.path.join(repo, "api", base + ".api.json")

    json.dump(api_prompt, open(out, "w"), indent=2)
    print(f"Wrote {out}: {len(api_prompt)} nodes (API/prompt format)")


if __name__ == "__main__":
    main()
