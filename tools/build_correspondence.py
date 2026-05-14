#!/usr/bin/env python3
"""Build a vertex correspondence file for BD_UVTransfer.

Given:
  - source mesh (e.g. lib/facewrap/canonical_face_model.obj — the MediaPipe
    canonical face mesh, or eventually a FLAME .obj)
  - target mesh (CC5 head, Metahuman head, custom rig, etc.)

Produces a `.npz` mapping each target vertex to a UV coordinate in the
source mesh's UV layout, plus the target mesh's own UV info (so the
warp node doesn't need to load the target .obj again at runtime).

Algorithm:
  1. Build a BVH over the source mesh.
  2. For each target vertex, query the closest point on the source surface
     and get back (source_face_id, source_barycentric).
  3. Interpolate the source mesh's UV at that point.
  4. Mark target vertices farther than `--max-distance` as invalid (e.g.
     scalp/back of head when source is a face-only mesh).

Both meshes must be roughly aligned in 3D space and in the same scale.
For the canonical face mesh (≈ -8..+8 unit cube, Y-up), align the target
mesh to that range before running this. The output mask flags unaligned
target verts so the warp node can avoid sampling them.

Usage:
    python tools/build_correspondence.py \\
        --source lib/facewrap/canonical_face_model.obj \\
        --target /path/to/cc5_head.obj \\
        --output /srv/AI_Stuff/models/facewrap/correspondences/cc5.npz \\
        [--max-distance 0.5]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def parse_obj(path: str) -> dict:
    """Parse .obj into verts/uvs/faces/face_uvs/uv_to_vert.

    Same layout as nodes/facewrap/face_fit.py:_parse_canonical_obj —
    duplicated here so the tool is standalone.
    """
    verts, uvs, faces_v, faces_vt = [], [], [], []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]
            if tag == "v":
                verts.append([float(x) for x in parts[1:4]])
            elif tag == "vt":
                uvs.append([float(x) for x in parts[1:3]])
            elif tag == "f":
                vi, vti = [], []
                for tok in parts[1:4]:
                    sub = tok.split("/")
                    vi.append(int(sub[0]) - 1)
                    vti.append(int(sub[1]) - 1 if len(sub) > 1 and sub[1] else int(sub[0]) - 1)
                faces_v.append(vi)
                faces_vt.append(vti)

    verts = np.asarray(verts, dtype=np.float32)
    uvs = np.asarray(uvs, dtype=np.float32) if uvs else np.zeros((0, 2), dtype=np.float32)
    faces = np.asarray(faces_v, dtype=np.int32)
    face_uvs = np.asarray(faces_vt, dtype=np.int32) if faces_vt else faces.copy()

    # Build uv→vert
    n_uv = uvs.shape[0]
    uv_to_vert = np.full(max(n_uv, 1), -1, dtype=np.int32)
    if n_uv:
        for vi3, vi_uv in zip(faces.reshape(-1), face_uvs.reshape(-1)):
            if uv_to_vert[vi_uv] == -1:
                uv_to_vert[vi_uv] = vi3
            elif uv_to_vert[vi_uv] != vi3:
                raise ValueError(
                    f"UV index {vi_uv} maps to multiple verts in {path}: "
                    f"{uv_to_vert[vi_uv]} and {vi3}"
                )
        n_orphan = int((uv_to_vert == -1).sum())
        if n_orphan:
            print(f"WARNING: {n_orphan} unreferenced UVs in {path} → set to vert 0")
            uv_to_vert[uv_to_vert == -1] = 0

    return {
        "verts": verts,
        "uvs": uvs,
        "faces": faces,
        "face_uvs": face_uvs,
        "uv_to_vert": uv_to_vert,
    }


def build(args):
    src = parse_obj(args.source)
    tgt = parse_obj(args.target)

    print(f"source: {len(src['verts'])} verts, {len(src['faces'])} faces, "
          f"{len(src['uvs'])} uvs")
    print(f"target: {len(tgt['verts'])} verts, {len(tgt['faces'])} faces, "
          f"{len(tgt['uvs'])} uvs")

    if src["uvs"].size == 0:
        print(f"ERROR: source mesh has no UVs (need 'vt' lines in {args.source})")
        sys.exit(2)
    if tgt["uvs"].size == 0:
        print(f"ERROR: target mesh has no UVs (need 'vt' lines in {args.target})")
        sys.exit(2)

    # --- Build BVH over the SOURCE mesh and query closest point per target vert ---
    try:
        import torch
        import cumesh
    except ImportError as e:
        print(f"ERROR: needs torch + cumesh ({e})")
        sys.exit(3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("ERROR: CUDA required for cumesh BVH")
        sys.exit(3)

    src_verts = torch.from_numpy(src["verts"]).to(device).float()
    src_faces = torch.from_numpy(src["faces"]).to(device).int()
    tgt_verts = torch.from_numpy(tgt["verts"]).to(device).float()

    print("Building BVH over source mesh...")
    bvh = cumesh.cuBVH(src_verts, src_faces)
    print("Querying closest source-surface point per target vertex...")
    dist, src_face_id, src_bary = bvh.unsigned_distance(tgt_verts, return_uvw=True)

    dist_np = dist.cpu().numpy()
    src_face_id_np = src_face_id.cpu().numpy().astype(np.int32)
    src_bary_np = src_bary.cpu().numpy()  # (V_target, 3)

    # Interpolate source UV at each closest-point match
    src_face_uv_idx = src["face_uvs"][src_face_id_np]   # (V_target, 3) source-UV indices
    src_uv_per_face = src["uvs"][src_face_uv_idx]       # (V_target, 3, 2)
    target_source_uv = (src_uv_per_face * src_bary_np[:, :, None]).sum(axis=1)  # (V_target, 2)

    valid_mask = dist_np < args.max_distance
    n_valid = int(valid_mask.sum())
    n_total = len(tgt["verts"])
    print(f"closest-point matches: {n_valid}/{n_total} verts within "
          f"{args.max_distance} units ({100*n_valid/n_total:.1f}%)")
    print(f"distance percentiles: 25%={np.percentile(dist_np,25):.3f}, "
          f"50%={np.percentile(dist_np,50):.3f}, "
          f"95%={np.percentile(dist_np,95):.3f}, "
          f"max={dist_np.max():.3f}")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        args.output,
        # Per-target-vertex source-UV lookup
        target_source_uv=target_source_uv.astype(np.float32),
        valid_mask=valid_mask,
        distances=dist_np.astype(np.float32),
        # Target mesh data (so the runtime node doesn't need the .obj)
        target_verts=tgt["verts"],
        target_uvs=tgt["uvs"],
        target_faces=tgt["faces"],
        target_face_uvs=tgt["face_uvs"],
        uv_to_vert=tgt["uv_to_vert"],
        # Provenance for debugging
        source_path=str(Path(args.source).resolve()),
        target_path=str(Path(args.target).resolve()),
        max_distance=args.max_distance,
    )
    size_kb = os.path.getsize(args.output) / 1024
    print(f"wrote {args.output} ({size_kb:.0f} KB)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Source mesh .obj (with UVs)")
    p.add_argument("--target", required=True, help="Target mesh .obj (with UVs)")
    p.add_argument("--output", required=True, help="Output .npz path")
    p.add_argument("--max-distance", type=float, default=0.5,
                   help="Target verts farther than this from any source-mesh "
                        "surface point are flagged invalid. Units match the "
                        "input meshes — tune so face area is mostly valid and "
                        "non-face area (scalp/back of head) is mostly invalid "
                        "when the source is face-only.")
    args = p.parse_args()
    build(args)


if __name__ == "__main__":
    main()
