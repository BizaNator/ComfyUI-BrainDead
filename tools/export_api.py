#!/usr/bin/env python3
"""
Export a UI-graph template (example_workflows/BD-*.json) to API/prompt format
(<name>.api.json) — the stable, self-contained contract handed to downstream agents.

Reuses workflow_to_api.py from the same node pack (queries the live server's object_info),
so the export always matches the currently-loaded node schemas. Regenerate this whenever
the UI template changes — the UI .json stays the single source of truth; the .api.json
is derived.

Usage:
    python3 tools/export_api.py example_workflows/BD-trellis2_unreal_fbx.json
    python3 tools/export_api.py <template.json> --server http://127.0.0.1:8188 --out <path>
"""
import argparse
import json
import os
import sys

# Import the canonical converter from this node pack
tools_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, tools_dir)
from workflow_to_api import workflow_to_api, api_get


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("template", help="UI-graph template JSON (example_workflows/BD-*.json)")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--out", default=None, help="Output path (default: <template>.api.json)")
    ap.add_argument("--expand-subgraphs", action="store_true",
                    help="Recursively inline subgraph instances (UUID-typed nodes) before conversion. "
                         "Without this flag, instances are skipped as 'not found in object_info'.")
    ap.add_argument("--max-depth", type=int, default=4, help="Max subgraph nesting depth to expand")
    args = ap.parse_args()

    workflow = json.load(open(args.template))
    if args.expand_subgraphs:
        from subgraph_expand import expand_graph
        before = len(workflow.get("nodes", []))
        workflow = expand_graph(workflow, max_depth=args.max_depth)
        print(f"subgraph expansion: {before} -> {len(workflow['nodes'])} nodes")
    object_info = api_get(f"{args.server}/object_info")
    api = workflow_to_api(workflow, object_info)

    # export-time asserts (the server ACCEPTS broken graphs silently — this is the
    # only place to catch them):
    #   A) every link ref resolves to an emitted node
    #   B) every emitted node is reachable from a declared input node (LoadImage etc.)
    emitted = set(api.keys())
    dangling = []
    for nid, node in api.items():
        for inp, val in node.get("inputs", {}).items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                if val[0] not in emitted:
                    dangling.append((nid, inp, val[0]))
    if dangling:
        print(f"ASSERT FAIL: {len(dangling)} dangling link refs:")
        for d in dangling[:20]:
            print("   ", d)
        sys.exit(2)

    def _walk(roots):
        # FORWARD walk from input roots: link origin -> its targets
        fwd = {}
        for nid, node in api.items():
            for inp, val in node.get("inputs", {}).items():
                if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str) and val[0] in emitted:
                    fwd.setdefault(val[0], []).append(nid)
        seen, stack = set(), list(roots)
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(fwd.get(cur, []))
        return seen
    roots = [nid for nid, node in api.items()
             if not any(isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)
                        for v in node.get("inputs", {}).values())]
    if roots:
        reachable = _walk(roots)
        stranded = emitted - reachable
        if stranded:
            print(f"ASSERT WARN: {len(stranded)} orphaned nodes (inbound-linked but unreachable from any source root):")
            for s in sorted(stranded, key=str)[:15]:
                print("   ", s, api[s].get("class_type"))
    else:
        print("ASSERT WARN: no source-root nodes in export")

    # API exports live in api/ (NOT example_workflows/ — ComfyUI scans that for UI templates and
    # would try to load the API-format json as a graph -> empty-canvas error in the browser).
    if args.out:
        out = args.out
    else:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base = os.path.splitext(os.path.basename(args.template))[0]
        os.makedirs(os.path.join(repo, "api"), exist_ok=True)
        out = os.path.join(repo, "api", base + ".api.json")
    json.dump(api, open(out, "w"), indent=2)
    print(f"wrote {out}: {len(api)} nodes (API/prompt format)")


if __name__ == "__main__":
    main()
