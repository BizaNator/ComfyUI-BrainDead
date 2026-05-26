#!/usr/bin/env python3
"""Convert a FLAME .pkl (+ texture-space UV) into a clean .npz for BD_FlameFit.

The FLAME .pkl is serialized with chumpy, which doesn't run on modern
Python/numpy (uses removed `inspect.getargspec` and `numpy.int`). Rather
than fight that at runtime, this tool unpickles it ONCE with a tiny stub
that pulls the numpy arrays straight out of the chumpy wrapper, and saves
a plain .npz the runtime node can `np.load` with zero chumpy dependency.

It also merges in the FLAME UV layout from FLAME_texture.npz (the .pkl
has faces but no texture coords).

Output .npz keys:
    v_template   (V, 3)        neutral mesh template
    shapedirs    (V, 3, n_sh)  shape + expression basis (float32)
    posedirs     (V, 3, 36)    pose-corrective blendshapes
    weights      (V, J)        LBS skinning weights
    J_regressor  (J, V)        joint regressor (densified from sparse)
    kintree      (2, J)        kinematic tree
    faces        (F, 3)        triangle vertex indices
    uvs          (V_uv, 2)     texture coords
    face_uvs     (F, 3)        per-face UV indices

Usage:
    python tools/convert_flame.py \\
        --pkl     /srv/AI_Stuff/models/flame/_extracted/flame2023/FLAME2023/flame2023.pkl \\
        --texture /srv/AI_Stuff/models/flame/_extracted/texspace/FLAME_texture.npz \\
        --output  /srv/AI_Stuff/models/flame/flame2023_facewrap.npz
"""

import argparse
import os
import pickle
import sys

import numpy as np


class _ChStub:
    """Stand-in for chumpy.ch.Ch — captures the pickled state, exposes the array.

    A chumpy Ch pickles a state dict whose 'x' key is the underlying ndarray.
    """
    def __setstate__(self, state):
        if isinstance(state, dict) and "x" in state:
            object.__setattr__(self, "_arr", np.asarray(state["x"]))
        else:
            object.__setattr__(self, "_arr", np.asarray(state))

    @property
    def arr(self):
        return object.__getattribute__(self, "_arr")


class _FlameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "chumpy" or module.startswith("chumpy."):
            return _ChStub
        return super().find_class(module, name)


def _arr(v):
    """Coerce a pkl value (ndarray, _ChStub, or scipy sparse) to a dense ndarray."""
    if isinstance(v, _ChStub):
        return v.arr
    if hasattr(v, "toarray"):  # scipy sparse
        return v.toarray()
    return np.asarray(v)


def convert(args):
    if not os.path.exists(args.pkl):
        print(f"ERROR: FLAME pkl not found: {args.pkl}")
        sys.exit(2)
    if not os.path.exists(args.texture):
        print(f"ERROR: FLAME_texture.npz not found: {args.texture}")
        sys.exit(2)

    with open(args.pkl, "rb") as f:
        flame = _FlameUnpickler(f, encoding="latin1").load()

    keys = ("v_template", "shapedirs", "posedirs", "weights",
            "J_regressor", "kintree_table", "f")
    missing = [k for k in keys if k not in flame]
    if missing:
        print(f"ERROR: FLAME pkl missing keys: {missing}")
        sys.exit(3)

    v_template = _arr(flame["v_template"]).astype(np.float32)
    shapedirs = _arr(flame["shapedirs"]).astype(np.float32)
    posedirs = _arr(flame["posedirs"]).astype(np.float32)
    weights = _arr(flame["weights"]).astype(np.float32)
    j_regressor = _arr(flame["J_regressor"]).astype(np.float32)
    kintree = _arr(flame["kintree_table"]).astype(np.int64)
    faces = _arr(flame["f"]).astype(np.int64)

    # posedirs sometimes ships as (V*3, n) — normalise to (V, 3, n)
    if posedirs.ndim == 2:
        posedirs = posedirs.reshape(v_template.shape[0], 3, -1)

    print(f"FLAME model:")
    print(f"  v_template  {v_template.shape}")
    print(f"  shapedirs   {shapedirs.shape}  (shape + expression basis)")
    print(f"  posedirs    {posedirs.shape}")
    print(f"  weights     {weights.shape}  ({weights.shape[1]} joints)")
    print(f"  J_regressor {j_regressor.shape}")
    print(f"  kintree     {kintree.shape}")
    print(f"  faces       {faces.shape}")

    # UV layout from FLAME_texture.npz
    tex = np.load(args.texture)
    if "vt" not in tex or "ft" not in tex:
        print(f"ERROR: {args.texture} missing 'vt' / 'ft'")
        sys.exit(3)
    uvs = tex["vt"].astype(np.float32)
    face_uvs = tex["ft"].astype(np.int64)
    print(f"  uvs         {uvs.shape}  (from FLAME_texture.npz)")
    print(f"  face_uvs    {face_uvs.shape}")

    if face_uvs.shape[0] != faces.shape[0]:
        print(f"WARNING: face_uvs count ({face_uvs.shape[0]}) != faces "
              f"({faces.shape[0]}) — UV/topology mismatch")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(
        args.output,
        v_template=v_template,
        shapedirs=shapedirs,
        posedirs=posedirs,
        weights=weights,
        J_regressor=j_regressor,
        kintree=kintree,
        faces=faces,
        uvs=uvs,
        face_uvs=face_uvs,
    )
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nwrote {args.output} ({size_mb:.1f} MB)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pkl", required=True, help="FLAME .pkl (e.g. flame2023.pkl)")
    p.add_argument("--texture", required=True,
                   help="FLAME_texture.npz (provides the UV layout: vt, ft)")
    p.add_argument("--output", required=True, help="Output .npz path")
    args = p.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
