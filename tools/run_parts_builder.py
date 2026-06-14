#!/usr/bin/env python3
"""
run_parts_builder.py — studio-pipeline handoff wrapper for the BD-parts_builder
workflow (Qwen-VL + SAM3 + BD_PartsBuilder + BD_PartsExport).

Self-contained: submits the frozen API contract (BD-parts_builder.api.json) to a
ComfyUI server, then drops the per-part PNGs into the canonical character
layout:

    Characters/<name>/images/parts/<slug>/<n>_<descriptor>.png

Slug routing is driven by BD_PartsExport's category_table (see
ComfyUI-BrainDead/config/parts_categories_character.txt). Tags that don't match
any entry fall through to the table's default region/slug — they get saved but
in the catch-all bucket so they're not lost.

Usage:
    run_parts_builder.py --image /path/<char>_composite.png --name <char>
    run_parts_builder.py --image x.png --name Bob --server http://10.15.0.20:8188

The wrapper does NOT trellis the parts — that's the next pipeline stage. After
this completes, the runner can iterate parts/<slug>/ dirs and fire
run_unreal_fbx.py per part.
"""
import argparse, json, os, shutil, sys, time, uuid
import urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
STUDIO_API = "/mnt/tank/Studio/Brains/Workflows/COB_PartsBuilder_API.json"
REPO_API   = os.path.join(HERE, "..", "api", "BD-parts_builder.api.json")
DEFAULT_API = STUDIO_API if os.path.exists(STUDIO_API) else REPO_API
CHAR_BASE  = "/mnt/tank/Studio/Brains/Characters"


def _post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def upload_image(path, server):
    name = os.path.basename(path)
    boundary = "----bd" + uuid.uuid4().hex
    with open(path, "rb") as f:
        filedata = f.read()
    body = b""
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'.encode()
    body += b"Content-Type: application/octet-stream\r\n\r\n" + filedata + b"\r\n"
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{server}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        res = json.loads(r.read())
    sub = res.get("subfolder", "")
    return f"{sub}/{res['name']}" if sub else res["name"]


def find_nodes(api, class_type):
    return [nid for nid, n in api.items() if n.get("class_type") == class_type]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Character composite image")
    ap.add_argument("--name", required=True, help="Character slug")
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--category-table-path", default="",
                    help="Override BD_PartsExport.category_table_path. Default = bundled config.")
    ap.add_argument("--filename-template",
                    default="%character%/images/parts/%region%/%slug%/%suffix%",
                    help="BD_PartsExport.filename template. %%region%%/%%slug%% gives the canonical layout.")
    ap.add_argument("--output-base", default="/srv/AI_Stuff/outputs",
                    help="ComfyUI output dir on disk (parts land under here first).")
    ap.add_argument("--char-base", default=CHAR_BASE,
                    help="Studio Characters root. Empty disables the character-folder copy.")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--no-region-tier", action="store_true",
                    help="Drop the %%region%% tier from the path template, "
                         "yielding parts/<slug>/<n>.png (skips intermediate region group).")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        sys.exit(f"image not found: {args.image}")
    if not os.path.exists(args.api):
        sys.exit(f"workflow api not found: {args.api}")
    api = json.load(open(args.api))
    if "prompt" in api and isinstance(api["prompt"], dict):
        api = api["prompt"]

    # ── 1. upload + point LoadImage at it ─────────────────────────────────
    uploaded = upload_image(args.image, args.server)
    for nid in find_nodes(api, "LoadImage"):
        api[nid]["inputs"]["image"] = uploaded

    # ── 2. wire BD_PartsExport for the canonical save layout ──────────────
    template = args.filename_template
    if args.no_region_tier:
        template = template.replace("/%region%/", "/")
    for nid in find_nodes(api, "BD_PartsExport"):
        api[nid]["inputs"]["filename"] = template
        # BD_PartsExport reads BD_SaveContext for %character% etc., so set it
        # via name_prefix only if context isn't wired. Most workflows DO wire
        # one — leaving these defaults alone keeps that working.
        if args.category_table_path:
            api[nid]["inputs"]["category_table_path"] = args.category_table_path

    # ── 3. set BD_SaveContext (if present) so %character% resolves ────────
    for nid in find_nodes(api, "BD_SaveContext"):
        api[nid]["inputs"].update({
            "character": args.name,
            "name": "parts",
            "project": "PartsBuilder",
            "base_dir": "",
            "subfolder": "",
        })

    # ── 4. submit + wait ──────────────────────────────────────────────────
    submit = _post(f"{args.server}/prompt", {"prompt": api})
    if "node_errors" in submit and submit["node_errors"]:
        print("Node errors:", json.dumps(submit["node_errors"], indent=2), file=sys.stderr)
    pid = submit.get("prompt_id")
    if not pid:
        sys.exit(f"submit rejected: {submit}")
    print(f"submitted pid={pid}")

    t0 = time.time()
    while time.time() - t0 < args.timeout:
        time.sleep(8)
        try:
            hist = _get(f"{args.server}/history/{pid}")
            if pid in hist:
                status = hist[pid].get("status", {})
                if status.get("status_str") == "success":
                    break
                if status.get("status_str") == "error":
                    sys.exit(f"workflow errored: {status}")
        except (urllib.error.URLError, json.JSONDecodeError):
            pass
    else:
        sys.exit(f"timeout after {args.timeout}s")

    # ── 5. report which slugs landed on disk ──────────────────────────────
    char_dir = os.path.join(args.char_base, args.name)
    parts_dir = os.path.join(char_dir, "images", "parts")
    found = {}
    if os.path.isdir(parts_dir):
        for region in sorted(os.listdir(parts_dir)):
            region_dir = os.path.join(parts_dir, region)
            if not os.path.isdir(region_dir): continue
            # Two valid shapes: parts/<region>/<slug>/*.png  OR  parts/<slug>/*.png
            sub = [d for d in os.listdir(region_dir)
                   if os.path.isdir(os.path.join(region_dir, d))]
            if sub:
                for slug in sub:
                    pngs = [f for f in os.listdir(os.path.join(region_dir, slug))
                            if f.lower().endswith(".png")]
                    if pngs:
                        found[f"{region}/{slug}"] = len(pngs)
            else:
                pngs = [f for f in os.listdir(region_dir) if f.lower().endswith(".png")]
                if pngs:
                    found[region] = len(pngs)

    print(json.dumps({
        "status": "success",
        "prompt_id": pid,
        "name": args.name,
        "parts_dir": parts_dir,
        "found": found,
        "total_slugs": len(found),
        "total_pngs": sum(found.values()),
    }, indent=2))


if __name__ == "__main__":
    main()
