#!/usr/bin/env python3
"""
Export a UI-graph template (example_workflows/BD-*.json) to API/prompt format
(<name>.api.json) — the stable, self-contained contract handed to downstream agents.

Reuses run_workflow.py's graph→API converter (queries the live server's object_info), so the
export always matches the currently-loaded node schemas. Regenerate this whenever the UI
template changes — the UI .json stays the single source of truth; the .api.json is derived.

Usage:
    python3 tools/export_api.py example_workflows/BD-trellis2_unreal_fbx.json
    python3 tools/export_api.py <template.json> --server http://127.0.0.1:8188 --out <path>
"""
import argparse, importlib.util, json, os, sys

RUN_WORKFLOW = "/opt/comfyui/run_workflow.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_workflow", RUN_WORKFLOW)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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

    rw = load_runner()
    workflow = json.load(open(args.template))
    if args.expand_subgraphs:
        tools_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, tools_dir)
        from subgraph_expand import expand_graph
        before = len(workflow.get("nodes", []))
        workflow = expand_graph(workflow, max_depth=args.max_depth)
        print(f"subgraph expansion: {before} -> {len(workflow['nodes'])} nodes")
    object_info = rw.api_get(f"{args.server}/object_info")
    api = rw.workflow_to_api(workflow, object_info)

    # API exports live in api/ (NOT example_workflows/ — ComfyUI scans that for UI templates and
    # would try to load the API-format json as a graph → empty-canvas error in the browser).
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
