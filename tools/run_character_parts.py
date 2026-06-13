#!/usr/bin/env python3
"""
run_character_parts.py — multi-part character → 3D orchestrator.

Every named part gets a 3D version, named <name>_<part>, in the studio character folder.

Two paths (use either or both):
  • 2D→Trellis (per item): for each --part-prompt, resolve a source image — a PartBuilder/SkinMaker/
    FaceMaker export if one exists (--part-images dir, matched by tag), else SAM3-isolate from
    --image — then run_unreal_fbx → Characters/<name>/models/<part>/unreal/<name>_<part>.fbx.
  • CubePart (geometry split): one --cubepart-mesh + --cube-parts → BD_CubePartSegment splits the
    mesh into canonically-aligned sub-meshes; ONE graph segments once and exports every part.

Examples:
  run_character_parts.py --name jojo_rhoads --image jojo_body.png \
      --part-prompts "tank top, jacket, shorts" --part-images Characters/jojo_rhoads/images/parts
  run_character_parts.py --name jojo_rhoads --cubepart-mesh body.glb \
      --cube-parts "head, torso, left arm, right arm, left leg, right leg"
"""
import argparse, glob, json, os, re, subprocess, sys, tempfile, time, uuid
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_BD = os.path.join(HERE, "run_bd.py")
RUN_FBX = os.path.join(HERE, "run_unreal_fbx.py")
CHAR_BASE = "/mnt/tank/Studio/Brains/Characters"


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def _run_json(cmd):
    p = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    if p.stderr.strip():
        print(p.stderr, file=sys.stderr)
    out = p.stdout.strip()
    i = out.find("{")
    if i < 0:
        return {"status": "error", "raw": out}, p.returncode
    obj, _ = json.JSONDecoder().raw_decode(out[i:])
    return obj, p.returncode


def find_partbuilder_image(part_images_dir, part):
    """Look for an existing PartBuilder/SkinMaker/FaceMaker export for this part (matched by tag)."""
    if not part_images_dir or not os.path.isdir(part_images_dir):
        return None
    slug = _slug(part)
    tokens = [t for t in slug.split("_") if len(t) > 2]
    cands = glob.glob(os.path.join(part_images_dir, "*.png")) + glob.glob(os.path.join(part_images_dir, "*.psd"))
    # exact slug first, else any file whose name contains all/most tokens
    exact = [c for c in cands if _slug(os.path.splitext(os.path.basename(c))[0]) == slug]
    if exact:
        return exact[0]
    scored = sorted(((sum(t in os.path.basename(c).lower() for t in tokens), c) for c in cands), reverse=True)
    return scored[0][1] if scored and scored[0][0] >= max(1, len(tokens) - 1) else None


# ---- CubePart: one graph, segment once, export every part ----------------------------------------
_OI = {}
def _node_defaults(server, ct):
    """Widget-input defaults for a node (API /prompt requires ALL required inputs present)."""
    if ct not in _OI:
        d = json.loads(urllib.request.urlopen(f"{server}/object_info/{ct}", timeout=20).read())[ct]
        a = {**d['input'].get('required', {}), **d['input'].get('optional', {})}
        out = {}
        for k, v in a.items():
            t = v[0]; meta = v[1] if len(v) > 1 and isinstance(v[1], dict) else {}
            if isinstance(t, list):
                out[k] = meta.get('default', t[0] if t else None)
            elif t == "COMBO":
                opts = meta.get('options', []); out[k] = meta.get('default', opts[0] if opts else None)
            elif t in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                out[k] = meta.get('default')
            # connection types (TRIMESH/VOXELGRID/IMAGE/…) are wired explicitly, not defaulted
        _OI[ct] = out
    return dict(_OI[ct])


def _node(server, ct, **inputs):
    return {"class_type": ct, "inputs": {**_node_defaults(server, ct), **inputs}}


def cubepart_all(mesh_path, cube_parts, name, args):
    """Build an inline API graph that segments the mesh once and exports each part. Returns result."""
    parts = [p.strip() for p in cube_parts.split(",") if p.strip()][:8]  # CubePart caps at 8
    sv = args.server
    # CubePart emits clean, canonically-aligned but UNCOLOURED geometry in its own normalized
    # space — it does NOT carry the source texture, and its space doesn't align with the source
    # mesh, so colour must be projected on in Blender (per the studio projection-bake step).
    # Here we deliver: CuMesh-decimated sub-mesh + a 4-angle geometry thumbnail per part.
    api = {"1": _node(sv, "BD_CubePartSegment", mesh_path=mesh_path, parts=", ".join(parts), seed=0)}
    dec = args.decimation or 5000
    nid = 2
    for i, p in enumerate(parts):
        gp, cm, ori, ex, pv, si = (str(nid + k) for k in range(6)); nid += 6
        slug = _slug(p)
        api[gp] = _node(sv, "BD_CubePartGetPart", parts=["1", 0], index=i)
        api[cm] = _node(sv, "BD_CuMeshSimplify", mesh=[gp, 0], target_faces=dec)
        # CubePart output is Z-up; glTF/Blender + the thumbnail camera are Y-up → rotate_x=-90 so the
        # exported glb imports upright AND the thumbnail renders upright.
        api[ori] = _node(sv, "BD_OrientMesh", mesh=[cm, 0], rotate_x=-90.0)
        api[ex] = _node(sv, "BD_ExportMeshWithColors", mesh=[ori, 0], format="glb",
                        auto_increment=False, filename=f"{name}_{slug}_seg", name_prefix="")
        api[pv] = _node(sv, "BD_MeshPreview", mesh=[ori, 0], shading="solid", views=4,
                        tile_size=512, background="dark")
        api[si] = _node(sv, "SaveImage", images=[pv, 0], filename_prefix=f"{name}_{slug}_segthumb")
    # submit + poll
    pid = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"{args.server}/prompt", data=json.dumps({"client_id": uuid.uuid4().hex, "prompt": api}).encode(),
        headers={"Content-Type": "application/json"}), timeout=60).read()).get("prompt_id")
    if not pid:
        return {"status": "error", "stage": "cubepart-submit"}
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        h = json.loads(urllib.request.urlopen(f"{args.server}/history/{pid}", timeout=30).read())
        if pid in h:
            break
        time.sleep(3.0)
    else:
        return {"status": "error", "stage": "cubepart-timeout", "prompt_id": pid}
    # collect textured glbs + thumbnails → char folder
    import shutil
    res = []
    for p in parts:
        slug = _slug(p)
        def _newest(pat):
            g = sorted(glob.glob(os.path.join(args.output_base, pat)), key=os.path.getmtime)
            return g[-1] if g else None
        gpath = _newest(f"{name}_{slug}_seg*.glb")
        tpath = _newest(f"{name}_{slug}_segthumb*.png")
        cglb, cthumb = None, None
        if args.char_base and gpath:
            dest = os.path.join(args.char_base, name, "models", slug, "unreal")
            os.makedirs(dest, exist_ok=True)
            cglb = os.path.join(dest, f"{name}_{slug}_seg.glb"); shutil.copy2(gpath, cglb)
            if tpath:
                cthumb = os.path.join(dest, f"{name}_{slug}_seg_thumbnail.png"); shutil.copy2(tpath, cthumb)
        res.append({"part": slug, "glb": gpath, "character_glb": cglb, "character_thumbnail": cthumb})
    return {"status": "success", "prompt_id": pid, "parts": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--image", default=None, help="full character image (SAM3-isolate fallback source)")
    ap.add_argument("--part-prompts", default="", help="comma-separated parts to 3D (2D→Trellis path)")
    ap.add_argument("--part-images", default=None, help="dir of PartBuilder/SkinMaker exports (<tag>.png)")
    ap.add_argument("--cubepart-mesh", default=None, help="base mesh to split with CubePart")
    ap.add_argument("--cube-parts",
                    default="head, torso, left arm, right arm, left leg, right leg, left hand, right hand",
                    help="comma-separated names for CubePart split (max 8; default = the 8 standard body parts)")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--output-dir", default="char_parts")
    ap.add_argument("--output-base", default="/srv/AI_Stuff/outputs")
    ap.add_argument("--char-base", default=CHAR_BASE)
    ap.add_argument("--decimation", type=int, default=None)
    ap.add_argument("--detail-strength", type=float, default=None)
    ap.add_argument("--timeout", type=int, default=2400)
    args = ap.parse_args()

    results = {"name": args.name, "parts": [], "cubepart": None}

    # ---- 2D→Trellis per part ----
    prompts = [p.strip() for p in args.part_prompts.split(",") if p.strip()]
    stage_dir = tempfile.mkdtemp(prefix="charparts_")
    for part in prompts:
        slug = _slug(part)
        src = find_partbuilder_image(args.part_images, part)
        source_type = "partbuilder" if src else "isolate"
        if not src:
            if not args.image:
                results["parts"].append({"part": slug, "status": "skipped", "reason": "no part image / no --image"})
                continue
            iso, _ = _run_json([RUN_BD, "--workflow", "isolate_part", "--image", args.image,
                                "--set", f"BD_RemoveBackground.prompts={part}",
                                "--output-dir", stage_dir, "--char-base", "",
                                "--server", args.server, "--timeout", str(args.timeout)])
            if iso.get("status") != "success" or not iso.get("outputs"):
                results["parts"].append({"part": slug, "status": "isolate_failed", "detail": iso}); continue
            src = iso["outputs"][0]
        fcmd = [RUN_FBX, "--image", src, "--name", args.name, "--part", slug,
                "--char-base", args.char_base, "--server", args.server, "--timeout", str(args.timeout)]
        if args.decimation is not None:
            fcmd += ["--decimation", str(args.decimation)]
        if args.detail_strength is not None:
            fcmd += ["--detail-strength", str(args.detail_strength)]
        fbx, _ = _run_json(fcmd)
        results["parts"].append({"part": slug, "source": source_type, "status": fbx.get("status"),
                                 "character_source": fbx.get("character_source"),
                                 "character_fbx": fbx.get("character_fbx"),
                                 "character_thumbnail": fbx.get("character_preview")})

    # ---- CubePart geometry split ----
    if args.cubepart_mesh and args.cube_parts:
        results["cubepart"] = cubepart_all(args.cubepart_mesh, args.cube_parts, args.name, args)

    ok = all(p.get("status") in ("success", None) for p in results["parts"])
    print(json.dumps(results, indent=2))
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
