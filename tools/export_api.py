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
    args = ap.parse_args()

    rw = load_runner()
    workflow = json.load(open(args.template))
    object_info = rw.api_get(f"{args.server}/object_info")
    api = rw.workflow_to_api(workflow, object_info)

    out = args.out or os.path.splitext(args.template)[0] + ".api.json"
    json.dump(api, open(out, "w"), indent=2)
    print(f"wrote {out}: {len(api)} nodes (API/prompt format)")


if __name__ == "__main__":
    main()
