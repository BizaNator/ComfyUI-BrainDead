"""
BD_FaceFit — assemble a LANDMARKS_BATCH into a FACE_FIT.

Loads the MediaPipe canonical face mesh (468 verts + UVs + 898 faces,
Apache-2.0, bundled in lib/facewrap/) and attaches the per-view 2D + 3D
landmark positions from BD_FaceLandmarks. The output FACE_FIT is what
the bake node consumes.

No optimization in v1: MediaPipe's landmarks already encode subject-
specific shape, and the 4x4 transform gives per-view pose. The bake step
only needs each vertex's 2D projection in the source photo, which is
exactly what LANDMARKS_BATCH["views"][i]["landmarks_2d"][:468] gives us.

Future swap-in: BD_FlameFit (FLAME 2023) or BD_ICTFit (ICT-FaceKit) can
emit the same FACE_FIT type using a different mesh + topology.
"""

import os
from pathlib import Path

import numpy as np

from comfy_api.latest import io

from .types import (
    LandmarksBatchInput,
    FaceFitOutput,
)


# Default location of the bundled canonical face mesh.
_PACK_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CANONICAL_OBJ = str(_PACK_ROOT / "lib" / "facewrap" / "canonical_face_model.obj")


def _parse_canonical_obj(path: str) -> dict:
    """Parse a MediaPipe-style canonical face .obj.

    Returns dict with:
        verts:     (V, 3) float32 — vertex positions
        uvs:       (V, 2) float32 — per-vertex UV (raw .obj convention, V-up)
        faces:     (F, 3) int32   — vertex indices per triangle
        face_uvs:  (F, 3) int32   — UV indices per triangle (== faces when
                                     vertex/UV are 1:1)
    """
    verts = []
    uvs = []
    faces_v = []
    faces_vt = []

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
                # Each token is "v", "v/vt", "v/vt/vn"; .obj is 1-indexed.
                vi, vti = [], []
                for tok in parts[1:4]:
                    sub = tok.split("/")
                    vi.append(int(sub[0]) - 1)
                    vti.append(int(sub[1]) - 1 if len(sub) > 1 and sub[1] else int(sub[0]) - 1)
                faces_v.append(vi)
                faces_vt.append(vti)

    verts_arr = np.asarray(verts, dtype=np.float32)
    uvs_arr = np.asarray(uvs, dtype=np.float32)
    faces_arr = np.asarray(faces_v, dtype=np.int32)
    face_uvs_arr = np.asarray(faces_vt, dtype=np.int32)

    # Build uv→vert lookup so the bake node can rasterize in UV-index space
    # and look up the corresponding 3D-vertex attribute per UV index.
    # Each UV index belongs to exactly one 3D vertex (UV islands duplicate
    # the vertex in UV space; the reverse mapping is many-to-one).
    n_uv = uvs_arr.shape[0]
    uv_to_vert = np.full(n_uv, -1, dtype=np.int32)
    for vi3, vi_uv in zip(faces_arr.reshape(-1), face_uvs_arr.reshape(-1)):
        if uv_to_vert[vi_uv] == -1:
            uv_to_vert[vi_uv] = vi3
        elif uv_to_vert[vi_uv] != vi3:
            # Inconsistent — bail loud rather than silently produce wrong textures
            raise ValueError(
                f"Inconsistent UV→vert mapping: UV index {vi_uv} maps to both "
                f"vertex {uv_to_vert[vi_uv]} and {vi3}"
            )

    # Any UV indices not referenced by any face stay at -1; warn but don't fail.
    n_orphan = int((uv_to_vert == -1).sum())
    if n_orphan > 0:
        print(f"[BD FaceFit] WARNING: {n_orphan}/{n_uv} UVs are unreferenced; "
              f"setting them to vertex 0.")
        uv_to_vert[uv_to_vert == -1] = 0

    return {
        "verts": verts_arr,
        "uvs": uvs_arr,
        "faces": faces_arr,
        "face_uvs": face_uvs_arr,
        "uv_to_vert": uv_to_vert,
    }


# Module-level cache: the canonical mesh is small (~45 KB) and never changes.
_CANONICAL_CACHE: dict | None = None


def _load_canonical(path: str) -> dict:
    global _CANONICAL_CACHE
    if _CANONICAL_CACHE is not None and _CANONICAL_CACHE.get("path") == path:
        return _CANONICAL_CACHE
    data = _parse_canonical_obj(path)
    data["path"] = path
    _CANONICAL_CACHE = data
    return data


class BD_FaceFit(io.ComfyNode):
    """Assemble LANDMARKS_BATCH + canonical face mesh into a FACE_FIT."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_FaceFit",
            display_name="BD Face Fit",
            category="🧠BrainDead/FaceWrap",
            description=(
                "Combine LANDMARKS_BATCH with the canonical face mesh into a\n"
                "FACE_FIT, the input for BD_FaceTextureBake.\n\n"
                "v1 uses MediaPipe's 468-vertex canonical mesh (Apache-2.0,\n"
                "bundled in lib/facewrap/). No optimization — MediaPipe's\n"
                "landmarks already encode subject-specific 2D + 3D positions.\n\n"
                "Future BD_FlameFit / BD_ICTFit nodes can emit the same\n"
                "FACE_FIT type using a different mesh."
            ),
            inputs=[
                LandmarksBatchInput(
                    "landmarks_batch",
                    tooltip="Output of BD_FaceLandmarks (per-view 478 landmarks).",
                ),
                io.String.Input(
                    "canonical_obj_path",
                    default="",
                    optional=True,
                    tooltip=f"Override path to canonical_face_model.obj. Empty = "
                            f"use bundled lib/facewrap/canonical_face_model.obj.",
                ),
            ],
            outputs=[
                FaceFitOutput(display_name="face_fit"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        landmarks_batch,
        canonical_obj_path: str = "",
    ) -> io.NodeOutput:
        if not isinstance(landmarks_batch, dict) or "views" not in landmarks_batch:
            return io.NodeOutput(None, "ERROR: invalid LANDMARKS_BATCH input")

        # Resolve canonical mesh path
        path = canonical_obj_path.strip() if canonical_obj_path else ""
        if not path:
            path = DEFAULT_CANONICAL_OBJ
        path = os.path.expanduser(os.path.expandvars(path))
        if not os.path.exists(path):
            return io.NodeOutput(None, f"ERROR: canonical mesh not found: {path}")

        try:
            canonical = _load_canonical(path)
        except Exception as e:
            return io.NodeOutput(None, f"ERROR: failed to parse {path}: {e}")

        n_verts = canonical["verts"].shape[0]
        n_faces = canonical["faces"].shape[0]

        # Build per-view dicts
        in_views = landmarks_batch["views"]
        n_lm = landmarks_batch.get("n_landmarks", 478)
        out_views = []
        for v in in_views:
            lm2d_full = v["landmarks_2d"]
            lm3d_full = v["landmarks_3d"]
            # Truncate to canonical mesh vertex count (drop iris/pupil landmarks 468..)
            if lm2d_full.shape[0] < n_verts:
                # If somehow fewer landmarks than vertices, pad with zeros
                pad2d = np.zeros((n_verts, 2), dtype=np.float32)
                pad2d[: lm2d_full.shape[0]] = lm2d_full
                pad3d = np.zeros((n_verts, 3), dtype=np.float32)
                pad3d[: lm3d_full.shape[0]] = lm3d_full
                verts_2d = pad2d
                verts_3d = pad3d
            else:
                verts_2d = lm2d_full[:n_verts].astype(np.float32)
                verts_3d = lm3d_full[:n_verts].astype(np.float32)

            out_views.append({
                "verts_2d": verts_2d,
                "verts_3d": verts_3d,
                "transform_4x4": v["transform_4x4"].astype(np.float32),
                "detected": bool(v["detected"]),
                "view_hint": v["view_hint"],
                "image_size": tuple(v["image_size"]),
                "yaw_estimate": float(v["yaw_estimate"]),
            })

        face_fit = {
            "canonical_verts": canonical["verts"],
            "canonical_uvs": canonical["uvs"],
            "faces": canonical["faces"],
            "face_uvs": canonical["face_uvs"],
            "uv_to_vert": canonical["uv_to_vert"],
            "mesh_source": "mediapipe_canonical",
            "views": out_views,
        }

        n_detected = sum(1 for v in out_views if v["detected"])
        hints = ", ".join(v["view_hint"] for v in out_views)
        status = (
            f"canonical: {n_verts} verts, {n_faces} faces | "
            f"views: {n_detected}/{len(out_views)} detected | "
            f"hints: [{hints}]"
        )
        return io.NodeOutput(face_fit, status)


FACEWRAP_FACE_FIT_V3_NODES = [BD_FaceFit]

FACEWRAP_FACE_FIT_NODES = {
    "BD_FaceFit": BD_FaceFit,
}

FACEWRAP_FACE_FIT_DISPLAY_NAMES = {
    "BD_FaceFit": "BD Face Fit",
}
