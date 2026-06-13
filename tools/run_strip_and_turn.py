#!/usr/bin/env python3
"""
run_strip_and_turn.py — strip a character to a nude mannequin, then turn it for 3D-reference views.

Stage 1 (BD-clothing_remover via run_bd): Qwen clothing remover → clean nude mannequin, saved as
        Characters/<name>/models/body/<name>_mannequin.png.
Stage 2 (COB_CharacterTurner_api via run_bd): feed the mannequin to the turner's BODY input
        (node 11; LoadFace is left as-is) → all turnaround views collected into
        Characters/<name>/models/body/images/.

Example:
    run_strip_and_turn.py --image jojo_body.png --name jojo_rhoads
"""
import argparse, json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_BD = os.path.join(HERE, "run_bd.py")
TURNER = "COB_CharacterTurner_api"
TURNER_BODY_NODE = "11"   # "Load Flux Upscale Body T Pose" (LoadFace=43 left untouched)


def _run_json(cmd):
    p = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    if p.stderr.strip():
        print(p.stderr, file=sys.stderr)
    out = p.stdout.strip(); i = out.find("{")
    if i < 0:
        return {"status": "error", "raw": out}, p.returncode
    obj, _ = json.JSONDecoder().raw_decode(out[i:])
    return obj, p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="clothed character image")
    ap.add_argument("--name", required=True, help="character name")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--char-base", default="/mnt/tank/Studio/Brains/Characters")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--no-turn", action="store_true", help="strip only, skip the turner")
    args = ap.parse_args()

    stage = tempfile.mkdtemp(prefix="strip_")
    body_dir = os.path.join(args.char_base, args.name, "models", "body")

    # ---- Stage 1: strip → nude mannequin -----------------------------------------------------
    strip, _ = _run_json([RUN_BD, "--workflow", "clothing_remover", "--image", args.image,
                          "--output-dir", stage, "--char-base", "",
                          "--server", args.server, "--timeout", str(args.timeout)])
    if strip.get("status") != "success" or not strip.get("outputs"):
        print(json.dumps({"status": "error", "stage": "strip", "detail": strip})); sys.exit(1)
    mannequin = strip["outputs"][0]
    os.makedirs(body_dir, exist_ok=True)
    char_mannequin = os.path.join(body_dir, f"{args.name}_mannequin.png")
    shutil.copy2(mannequin, char_mannequin)
    print(f"[chain] mannequin -> {char_mannequin}", file=sys.stderr)

    result = {"status": "success", "name": args.name,
              "mannequin": mannequin, "character_mannequin": char_mannequin, "turn_views": []}

    # ---- Stage 2: turnaround views from the mannequin ----------------------------------------
    if not args.no_turn:
        turn, _ = _run_json([RUN_BD, "--workflow", TURNER, "--image", mannequin,
                             "--image-node", TURNER_BODY_NODE, "--name", args.name, "--part", "body",
                             "--char-base", args.char_base, "--output-dir", stage,
                             "--server", args.server, "--timeout", str(args.timeout)])
        if turn.get("status") == "success":
            result["turn_views"] = turn.get("character_outputs", [])
        else:
            result["turn_status"] = turn.get("status"); result["turn_detail"] = turn

    print(json.dumps(result, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
