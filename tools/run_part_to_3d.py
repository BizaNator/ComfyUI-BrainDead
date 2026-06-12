#!/usr/bin/env python3
"""
run_part_to_3d.py — SAM3 → Trellis chain: isolate a named part from a character image, then
generate its game-ready 3D FBX into the character folder.

Stage 1 (run_bd.py + BD-isolate_part): SAM3 keeps only `--prompts` (e.g. "tank top") → clean
            white-bg PNG.
Stage 2 (run_unreal_fbx.py): Trellis2 → textured low-poly FBX (+ maps + DAM thumbnail) copied to
            Characters/<name>/models/<part>/unreal/.

Example:
    run_part_to_3d.py --image jojo_body.png --prompts "tank top" --name jojo_rhoads --part top
"""
import argparse, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_BD = os.path.join(HERE, "run_bd.py")
RUN_FBX = os.path.join(HERE, "run_unreal_fbx.py")


def _run_json(cmd):
    """Run a tool, echo its stderr, return the parsed JSON it printed on stdout (last block)."""
    p = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    if p.stderr.strip():
        print(p.stderr, file=sys.stderr)
    out = p.stdout.strip()
    start = out.find("{")
    if start < 0:
        raise SystemExit(f"no JSON from {cmd[0]}:\n{out}")
    # raw_decode parses the first complete JSON object from `start` (handles nested objects);
    # rfind would wrongly grab an inner object like character_maps.
    obj, _ = json.JSONDecoder().raw_decode(out[start:])
    return obj, p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="character image to isolate the part from")
    ap.add_argument("--prompts", required=True, help="SAM3 prompt(s) for the part to keep (e.g. 'tank top')")
    ap.add_argument("--name", required=True, help="character name")
    ap.add_argument("--part", required=True, help="part slug → Characters/<name>/models/<part>/…")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--decimation", type=int, default=None)
    ap.add_argument("--detail-strength", type=float, default=None)
    ap.add_argument("--char-base", default="/mnt/tank/Studio/Brains/Characters")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    # ---- Stage 1: isolate the part → clean white-bg PNG ----------------------
    stage_dir = tempfile.mkdtemp(prefix="part2x3d_")
    iso, rc = _run_json([RUN_BD, "--workflow", "isolate_part", "--image", args.image,
                         "--set", f"BD_RemoveBackground.prompts={args.prompts}",
                         "--output-dir", stage_dir, "--char-base", "",
                         "--server", args.server, "--timeout", str(args.timeout)])
    if iso.get("status") != "success" or not iso.get("outputs"):
        print(json.dumps({"status": "error", "stage": "isolate", "detail": iso})); sys.exit(1)
    isolated = iso["outputs"][0]
    print(f"[chain] isolated '{args.prompts}' -> {isolated}", file=sys.stderr)

    # ---- Stage 2: Trellis → FBX into the character folder --------------------
    fbx_cmd = [RUN_FBX, "--image", isolated, "--name", args.name, "--part", args.part,
               "--char-base", args.char_base, "--server", args.server, "--timeout", str(args.timeout)]
    if args.decimation is not None:
        fbx_cmd += ["--decimation", str(args.decimation)]
    if args.detail_strength is not None:
        fbx_cmd += ["--detail-strength", str(args.detail_strength)]
    fbx, rc2 = _run_json(fbx_cmd)

    print(json.dumps({
        "status": fbx.get("status"),
        "part_prompt": args.prompts,
        "isolated_image": isolated,
        "fbx": fbx.get("fbx"),
        "character_fbx": fbx.get("character_fbx"),
        "character_thumbnail": fbx.get("character_preview"),
        "character_maps": fbx.get("character_maps"),
    }, indent=2))
    sys.exit(0 if fbx.get("status") == "success" else 2)


if __name__ == "__main__":
    main()
