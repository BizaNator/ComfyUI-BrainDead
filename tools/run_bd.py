#!/usr/bin/env python3
"""
run_bd.py — generic launcher for ANY BrainDead workflow.

One dispatcher instead of a bespoke run_<x>.py per workflow. It:
  - resolves --workflow (a path, or a BD-<name> template basename, or a COB API json),
  - uploads --image and points every LoadImage at it,
  - applies arbitrary --set "Selector.input=value" overrides (Selector = node class_type OR node id),
  - submits, waits, and collects the saved image outputs (from /history),
  - optionally copies them into the studio character folder (Characters/<name>/models/<part>/…),
  - prints a JSON contract.

Examples:
    # isolate a clothing part with SAM3-based background removal
    run_bd.py --workflow background_removal --image jojo.png \
        --set "BD_RemoveBackground.prompts=tank top" --output-dir /tmp/iso

    # run any template, override a node input by id
    run_bd.py --workflow BD-sam3_parts_segmentation --image x.png --set "13.prompts=top, jacket"

For the FBX/3D stage use tools/run_unreal_fbx.py (specialized: embeds maps, copies fbx).
See tools/run_part_to_3d.py for the SAM3→Trellis chain.
"""
import argparse, importlib.util, json, os, shutil, sys, time, uuid
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EW = os.path.join(REPO, "example_workflows")
API_DIR = os.path.join(REPO, "api")
STUDIO_WF = "/mnt/tank/Studio/Brains/Workflows"
CHAR_BASE = "/mnt/tank/Studio/Brains/Characters"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_workflow", "/opt/comfyui/run_workflow.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def resolve_workflow(ref):
    """Path | BD-<name>(.json) template | <name>.api.json | COB name → a loaded workflow + is_api."""
    cands = [ref,
             os.path.join(EW, ref), os.path.join(EW, ref + ".json"),
             os.path.join(EW, f"BD-{ref}.json"),
             os.path.join(API_DIR, ref + ".api.json"), os.path.join(API_DIR, f"BD-{ref}.api.json"),
             os.path.join(STUDIO_WF, ref), os.path.join(STUDIO_WF, ref + ".json")]
    for p in cands:
        if os.path.isfile(p):
            wf = json.load(open(p))
            is_api = not (isinstance(wf, dict) and isinstance(wf.get("nodes"), list))
            return wf, is_api, p
    raise SystemExit(f"workflow not found: {ref}\n  tried: " + "\n         ".join(cands))


def _post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def upload_image(path, server):
    name = os.path.basename(path)
    b = "----bd" + uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n").encode() + open(path, "rb").read() + \
           (f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{b}--\r\n").encode()
    req = urllib.request.Request(f"{server}/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    res = json.loads(urllib.request.urlopen(req, timeout=120).read())
    sub = res.get("subfolder", "")
    return f"{sub}/{res['name']}" if sub else res["name"]


def _parse_val(s):
    low = s.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    if s[:1] in "[{":
        try:
            return json.loads(s)
        except Exception:
            pass
    return s


def apply_set(api, selector_value):
    """selector = '<class_type|node_id>.<input>'; sets matching nodes' input to the parsed value."""
    sel, _, val = selector_value.partition("=")
    ref, _, inp = sel.rpartition(".")
    if not ref or not inp:
        raise SystemExit(f"--set must be 'NodeType.input=value' or 'id.input=value' (got {selector_value!r})")
    pv = _parse_val(val)
    hits = 0
    for nid, node in api.items():
        if nid == ref or node.get("class_type") == ref:
            node.setdefault("inputs", {})[inp] = pv
            hits += 1
    if not hits:
        print(f"  [warn] --set matched no node for {ref!r}", file=sys.stderr)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", required=True, help="path | BD-<name> template | <name>.api.json | COB")
    ap.add_argument("--image", default=None, help="upload + point every LoadImage at it")
    ap.add_argument("--set", action="append", default=[], dest="sets",
                    help="override 'NodeType.input=value' or 'id.input=value' (repeatable)")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--output-dir", default=None, help="copy saved image outputs here")
    ap.add_argument("--name", default=None, help="character name (for the character-folder copy)")
    ap.add_argument("--part", default="body")
    ap.add_argument("--char-base", default=CHAR_BASE, help="empty to skip the character-folder copy")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    rw = _load_runner()
    wf, is_api, path = resolve_workflow(args.workflow)
    api = wf if is_api else rw.workflow_to_api(wf, rw.api_get(f"{args.server}/object_info"))

    if args.image:
        up = upload_image(args.image, args.server)
        for n in api.values():
            if n.get("class_type") == "LoadImage":
                n.setdefault("inputs", {})["image"] = up
    for s in args.sets:
        apply_set(api, s)

    pid = _post(f"{args.server}/prompt", {"client_id": uuid.uuid4().hex, "prompt": api}).get("prompt_id")
    if not pid:
        print(json.dumps({"status": "error", "error": "submit rejected (check --set / inputs)"})); sys.exit(1)
    deadline = time.time() + args.timeout
    hist = None
    while time.time() < deadline:
        h = _get(f"{args.server}/history/{pid}")
        if pid in h:
            hist = h[pid]; break
        time.sleep(3.0)
    if hist is None:
        print(json.dumps({"status": "error", "error": "timeout", "prompt_id": pid})); sys.exit(1)

    # collect saved image outputs from /history (filename/subfolder on disk under the output dir)
    out_base = "/srv/AI_Stuff/outputs"
    saved = []
    for nid, out in (hist.get("outputs") or {}).items():
        for img in out.get("images", []):
            if img.get("type") == "temp":
                continue
            p = os.path.join(out_base, img.get("subfolder", ""), img["filename"])
            if os.path.exists(p):
                saved.append({"node": nid, "path": p})

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for s in saved:
            d = os.path.join(args.output_dir, os.path.basename(s["path"]))
            shutil.copy2(s["path"], d); s["copied"] = d

    char_outputs = []
    if args.char_base and args.name:
        dest = os.path.join(args.char_base, args.name, "models", args.part, "images")
        os.makedirs(dest, exist_ok=True)
        for s in saved:
            d = os.path.join(dest, os.path.basename(s["path"]))
            shutil.copy2(s["path"], d); char_outputs.append(d)

    print(json.dumps({
        "status": "success" if saved else "no_outputs",
        "prompt_id": pid, "workflow": path,
        "outputs": [s["path"] for s in saved],
        "character_outputs": char_outputs,
    }, indent=2))
    sys.exit(0 if saved else 2)


if __name__ == "__main__":
    main()
