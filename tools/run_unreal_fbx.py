#!/usr/bin/env python3
"""
run_unreal_fbx.py — studio-pipeline handoff wrapper for the TRELLIS2 → Unreal FBX workflow.

Self-contained: submits the frozen API contract (BD-trellis2_unreal_fbx.api.json) to a ComfyUI
server and returns the path to the game-ready FBX (embedded PBR textures + vertex colors) plus
the PBR map PNGs. The studio-pipeline agent only needs this script + the .api.json + a reachable
ComfyUI — no run_workflow.py / object_info dependency.

Contract:
    in : a single character image (head or body), a character name
    out: <output-base>/<output-dir>/<name>.fbx  (+ <name>_diffuse/normal/metallic/roughness/alpha.png)

Usage:
    python3 run_unreal_fbx.py --image /path/Letti.png --name Letti
    python3 run_unreal_fbx.py --image x.png --name Bob --decimation 5000 --server http://10.15.0.20:8188
"""
import argparse, json, os, shutil, sys, time, uuid
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
# Registered studio executable (COB convention); falls back to the BrainDead repo source.
STUDIO_API = "/mnt/tank/Studio/Brains/Workflows/COB_3d_TrellisUnrealFBX_v01_API.json"
REPO_API = os.path.join(HERE, "..", "api", "BD-trellis2_unreal_fbx.api.json")
DEFAULT_API = STUDIO_API if os.path.exists(STUDIO_API) else REPO_API
CHAR_BASE = "/mnt/tank/Studio/Brains/Characters"


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def _post(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def upload_image(path, server):
    """POST /upload/image (multipart) → returns the server-side filename to feed LoadImage."""
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
    req = urllib.request.Request(f"{server}/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        res = json.loads(r.read())
    sub = res.get("subfolder", "")
    return f"{sub}/{res['name']}" if sub else res["name"]


def find_nodes(api, class_type):
    return [nid for nid, n in api.items() if n.get("class_type") == class_type]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Character image (head or body)")
    ap.add_argument("--name", required=True, help="Character name → <name>.fbx")
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--decimation", type=int, default=None,
                    help="Override target tri count (template default 3000; bump to ~5000 if chunky)")
    ap.add_argument("--detail-strength", type=float, default=None,
                    help="Override albedo detail-normal strength (template default 1.0; 0=off, ~1.5-2 for more)")
    ap.add_argument("--output-dir", default="unreal_fbx", help="Subfolder under ComfyUI output/")
    ap.add_argument("--output-base", default="/srv/AI_Stuff/outputs",
                    help="ComfyUI output directory on disk (to resolve the FBX path)")
    ap.add_argument("--part", default="body",
                    help="Body part → Characters/<name>/models/<part>/unreal/ (e.g. body, head, arm_left)")
    ap.add_argument("--char-base", default=CHAR_BASE,
                    help="Studio Characters root. Set empty to skip the character-folder copy.")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    api = json.load(open(args.api))

    # 1. upload the image, point LoadImage at it
    uploaded = upload_image(args.image, args.server)
    for nid in find_nodes(api, "LoadImage"):
        api[nid]["inputs"]["image"] = uploaded

    # 2. naming + output dir — every asset is named "<name>_<part>" (mesh, maps, thumbnail)
    base = f"{args.name}_{args.part}"
    for nid in find_nodes(api, "BD_PackBundle"):
        api[nid]["inputs"]["name"] = base
    for nid in find_nodes(api, "BD_BlenderExportMesh"):
        api[nid]["inputs"]["output_dir"] = args.output_dir
    for nid in find_nodes(api, "SaveImage"):
        pre = api[nid]["inputs"].get("filename_prefix", "")
        tex = pre.split("/")[-1] if "/" in pre else pre  # diffuse / normal / preview
        api[nid]["inputs"]["filename_prefix"] = f"{args.output_dir}/{base}_{tex}"

    # 3. optional poly-count / detail-normal overrides
    if args.decimation is not None:
        for nid in find_nodes(api, "BD_OVoxelBake"):
            api[nid]["inputs"]["decimation_target"] = args.decimation
    if args.detail_strength is not None:
        for nid in find_nodes(api, "BD_DetailNormalFromAlbedo"):
            api[nid]["inputs"]["detail_strength"] = args.detail_strength

    # 4. submit + poll
    client_id = uuid.uuid4().hex
    pid = _post(f"{args.server}/prompt", {"client_id": client_id, "prompt": api}).get("prompt_id")
    if not pid:
        print(json.dumps({"status": "error", "error": "no prompt_id (submit rejected)"})); sys.exit(1)
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        hist = _get(f"{args.server}/history/{pid}")
        if pid in hist:
            break
        time.sleep(3.0)
    else:
        print(json.dumps({"status": "error", "error": "timeout", "prompt_id": pid})); sys.exit(1)

    # 5. resolve outputs on disk (ComfyUI output dir) — all named "<name>_<part>"
    out_dir = os.path.join(args.output_base, args.output_dir)
    fbx = os.path.join(out_dir, f"{base}.fbx")
    maps = {t: os.path.join(out_dir, f"{base}_{t}.png")
            for t in ("diffuse", "normal", "metallic", "roughness", "alpha")}
    maps = {k: v for k, v in maps.items() if os.path.exists(v)}
    # DAM preview thumbnail (BD_MeshPreview grid → SaveImage, auto-numbered)
    import glob as _glob
    _prev = sorted(_glob.glob(os.path.join(out_dir, f"{base}_preview*.png")), key=os.path.getmtime)
    preview = _prev[-1] if _prev else None

    # 6. copy into the studio character folder (Seam 1 bridge) — canonical layout for Blender→Unreal
    #    Characters/<name>/models/<part>/<name>_<part>.png   ← SOURCE image used to generate the mesh
    #    Characters/<name>/models/<part>/unreal/<name>_<part>.{fbx, _diffuse.png, …, _thumbnail.png}
    char_fbx, char_maps, char_preview, char_source = None, {}, None, None
    if args.char_base and os.path.exists(fbx):
        part_dir = os.path.join(args.char_base, args.name, "models", args.part)
        dest = os.path.join(part_dir, "unreal")
        os.makedirs(dest, exist_ok=True)
        # save the SOURCE image that generated this mesh, alongside the part
        if args.image and os.path.exists(args.image):
            char_source = os.path.join(part_dir, f"{base}{os.path.splitext(args.image)[1] or '.png'}")
            shutil.copy2(args.image, char_source)
        char_fbx = os.path.join(dest, f"{base}.fbx")
        shutil.copy2(fbx, char_fbx)
        for t, p in maps.items():
            d = os.path.join(dest, f"{base}_{t}.png")
            shutil.copy2(p, d); char_maps[t] = d
        if preview and os.path.exists(preview):
            char_preview = os.path.join(dest, f"{base}_thumbnail.png")
            shutil.copy2(preview, char_preview)   # DAM renders this; no Blender needed

    result = {
        "status": "success" if os.path.exists(fbx) else "incomplete",
        "prompt_id": pid,
        "name": args.name,
        "part": args.part,
        "asset": base,
        "fbx": fbx if os.path.exists(fbx) else None,
        "maps": maps,
        "preview": preview,
        "character_source": char_source,      # <name>_<part>.png used to generate the mesh
        "character_fbx": char_fbx,            # canonical studio location (Blender→Unreal picks this up)
        "character_maps": char_maps,
        "character_preview": char_preview,    # DAM thumbnail (textured-look mesh render)
        "decimation_target": args.decimation or "template default (3000)",
    }
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "success" else 2)


if __name__ == "__main__":
    main()
