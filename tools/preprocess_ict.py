#!/usr/bin/env python3
"""Preprocess ICT-FaceKit's generic_neutral_mesh.obj into a clean skin mesh
for the face-wrap pipeline.

ICT-FaceKit's neutral mesh is a full head, but:
  - it's mostly QUADS (the pipeline wants triangles)
  - it uses UDIM tiles — each material on its own U-tile
  - only M_Face (tile u=0) + M_BackHead (tile u=1) are actual head skin;
    the rest is teeth / gums / eyeballs / lashes (interior parts we don't
    bake photo texture onto)

This tool:
  1. Parses the .obj, triangulating quads (a b c d -> a b c + a c d)
  2. Keeps only the M_Face + M_BackHead faces
  3. Packs the two UDIM tiles into one [0,1] atlas: U is scaled x0.5 so
     M_Face lands in u[0,0.5] and M_BackHead in u[0.5,1] — no overlap,
     single square texture, no runtime UDIM handling
  4. Remaps vertex / UV indices to the filtered subset
  5. Remaps ICT's 68 iBUG landmark vertex indices to the subset
  6. Writes a clean all-triangle ict_head_skin.obj + ict_landmarks_68.json

Run once; the outputs get bundled in lib/facewrap/ict/.

Usage:
    python tools/preprocess_ict.py \\
        --ict-dir /tmp/ICT-FaceKit/FaceXModel \\
        --out-dir lib/facewrap/ict
"""

import argparse
import json
import os
import sys

import numpy as np


SKIN_MATERIALS = ("M_Face", "M_BackHead")
# U-scale so the two UDIM tiles (u in [0,1] and [1,2]) fit in one [0,1] atlas.
U_SCALE = 0.5


def parse_ict_obj(path: str):
    """Parse ICT .obj with quad triangulation + per-face material tracking.

    Returns (verts, uvs, tris_v, tris_vt, tri_materials) — tris are all
    triangles (quads split), still indexing into the FULL vert/uv arrays.
    """
    verts, uvs = [], []
    tris_v, tris_vt, tri_mtl = [], [], []
    cur_mtl = None

    for line in open(path):
        p = line.split()
        if not p:
            continue
        tag = p[0]
        if tag == "v":
            verts.append([float(x) for x in p[1:4]])
        elif tag == "vt":
            uvs.append([float(x) for x in p[1:3]])
        elif tag == "usemtl":
            cur_mtl = p[1]
        elif tag == "f":
            vi, vti = [], []
            for tok in p[1:]:
                sub = tok.split("/")
                vi.append(int(sub[0]) - 1)
                vti.append(int(sub[1]) - 1 if len(sub) > 1 and sub[1] else int(sub[0]) - 1)
            # Fan-triangulate: a b c d ... -> (a,b,c), (a,c,d), ...
            for k in range(1, len(vi) - 1):
                tris_v.append([vi[0], vi[k], vi[k + 1]])
                tris_vt.append([vti[0], vti[k], vti[k + 1]])
                tri_mtl.append(cur_mtl)

    return (
        np.asarray(verts, dtype=np.float32),
        np.asarray(uvs, dtype=np.float32),
        np.asarray(tris_v, dtype=np.int64),
        np.asarray(tris_vt, dtype=np.int64),
        tri_mtl,
    )


def build(args):
    obj_path = os.path.join(args.ict_dir, "generic_neutral_mesh.obj")
    idx_path = os.path.join(args.ict_dir, "vertex_indices.json")
    if not os.path.exists(obj_path):
        print(f"ERROR: {obj_path} not found")
        sys.exit(2)
    if not os.path.exists(idx_path):
        print(f"ERROR: {idx_path} not found")
        sys.exit(2)

    verts, uvs, tris_v, tris_vt, tri_mtl = parse_ict_obj(obj_path)
    print(f"parsed: {len(verts)} verts, {len(uvs)} uvs, {len(tris_v)} tris "
          f"(after quad triangulation)")

    # --- Filter to skin materials ---
    keep = np.array([m in SKIN_MATERIALS for m in tri_mtl], dtype=bool)
    skin_tris_v = tris_v[keep]
    skin_tris_vt = tris_vt[keep]
    print(f"skin tris ({'+'.join(SKIN_MATERIALS)}): {keep.sum()} of {len(tris_v)}")

    # --- Remap vertices to the used subset ---
    used_v = np.unique(skin_tris_v.reshape(-1))
    used_vt = np.unique(skin_tris_vt.reshape(-1))
    v_remap = np.full(len(verts), -1, dtype=np.int64)
    v_remap[used_v] = np.arange(len(used_v))
    vt_remap = np.full(len(uvs), -1, dtype=np.int64)
    vt_remap[used_vt] = np.arange(len(used_vt))

    new_verts = verts[used_v]
    new_uvs = uvs[used_vt].copy()
    new_faces = v_remap[skin_tris_v]
    new_face_uvs = vt_remap[skin_tris_vt]

    # --- Pack the two UDIM tiles into one [0,1] atlas (U x0.5) ---
    # M_Face is u[0,1] -> u[0,0.5]; M_BackHead is u[1,2] -> u[0.5,1].
    new_uvs[:, 0] = new_uvs[:, 0] * U_SCALE
    print(f"UV after pack: u[{new_uvs[:,0].min():.3f},{new_uvs[:,0].max():.3f}] "
          f"v[{new_uvs[:,1].min():.3f},{new_uvs[:,1].max():.3f}]")

    # --- Remap the 68 iBUG landmark vertex indices ---
    idx_json = json.load(open(idx_path))
    lm68_full = np.asarray(idx_json["idx_to_landmark_verts"], dtype=np.int64)
    lm68_new = v_remap[lm68_full]
    n_lost = int((lm68_new == -1).sum())
    if n_lost > 0:
        lost = [i for i, x in enumerate(lm68_new) if x == -1]
        print(f"ERROR: {n_lost} of 68 landmark verts are not in the skin "
              f"materials (iBUG indices {lost}). Cannot build a usable fit.")
        sys.exit(3)
    print(f"all 68 iBUG landmark verts remapped into the skin subset")

    # --- Write outputs ---
    os.makedirs(args.out_dir, exist_ok=True)
    out_obj = os.path.join(args.out_dir, "ict_head_skin.obj")
    with open(out_obj, "w") as f:
        f.write("# ICT-FaceKit head skin (M_Face + M_BackHead), triangulated,\n")
        f.write("# UDIM tiles packed into a single [0,1] atlas (U x0.5).\n")
        f.write("# Generated by tools/preprocess_ict.py — see lib/facewrap/NOTICE.md\n")
        for vx in new_verts:
            f.write(f"v {vx[0]:.6f} {vx[1]:.6f} {vx[2]:.6f}\n")
        for vt in new_uvs:
            f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
        for fv, fvt in zip(new_faces, new_face_uvs):
            f.write(f"f {fv[0]+1}/{fvt[0]+1} {fv[1]+1}/{fvt[1]+1} {fv[2]+1}/{fvt[2]+1}\n")
    print(f"wrote {out_obj}: {len(new_verts)} verts, {len(new_uvs)} uvs, "
          f"{len(new_faces)} tris ({os.path.getsize(out_obj)/1024:.0f} KB)")

    out_lm = os.path.join(args.out_dir, "ict_landmarks_68.json")
    with open(out_lm, "w") as f:
        json.dump({
            "idx_to_landmark_verts": lm68_new.tolist(),
            "note": "68 iBUG landmark vertex indices into ict_head_skin.obj",
            "source": "ICT-FaceKit vertex_indices.json, remapped to the skin subset",
        }, f, indent=2)
    print(f"wrote {out_lm}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ict-dir", required=True,
                   help="ICT-FaceKit FaceXModel directory (has generic_neutral_mesh.obj)")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for ict_head_skin.obj + ict_landmarks_68.json")
    args = p.parse_args()
    build(args)


if __name__ == "__main__":
    main()
