"""
BD_FaceFit — assemble a LANDMARKS_BATCH into a FACE_FIT.

Two mesh sources, both emitting the same FACE_FIT contract:

- mediapipe_canonical (default): MediaPipe's 468-vertex canonical face mesh
  (Apache-2.0, bundled). The 468 vertices ARE FaceLandmarker landmarks
  0..467, so the fit is pure assembly — no optimization. Face-only: no
  ears, scalp, or back of head.

- ict_facekit: ICT-FaceKit's head-skin mesh (MIT, bundled — preprocessed
  by tools/preprocess_ict.py to a 14k-vert triangle mesh with a single
  [0,1] UV atlas). Full head incl. ears + scalp + back. Per view, the
  ICT head is Procrustes-fitted (similarity transform: scale + rotation
  + translation) to MediaPipe's 68 iBUG landmarks. The fit poses ICT's
  *neutral* shape — it's not per-subject shape-accurate, but it gives the
  bake somewhere to project ear / scalp / rear pixels onto.

Future swap-in: BD_FlameFit (FLAME 2023) can emit the same FACE_FIT type.
"""

import json
import os
from pathlib import Path

import numpy as np

from comfy_api.latest import io

from .types import (
    LandmarksBatchInput,
    FaceFitOutput,
)


# Bundled mesh asset locations
_PACK_ROOT = Path(__file__).resolve().parent.parent.parent
_LIB = _PACK_ROOT / "lib" / "facewrap"
DEFAULT_CANONICAL_OBJ = str(_LIB / "canonical_face_model.obj")
DEFAULT_ICT_OBJ = str(_LIB / "ict" / "ict_head_skin.obj")
DEFAULT_ICT_LANDMARKS = str(_LIB / "ict" / "ict_landmarks_68.json")

MESH_SOURCES = ["mediapipe_canonical", "ict_facekit"]

# MediaPipe FaceMesh (468) index for each of the 68 iBUG landmarks, in
# standard iBUG-68 order (jaw, brows, nose, eyes, outer lip, inner lip).
# Used to Procrustes-fit the ICT head, whose 68 landmark verts are also
# stored in iBUG-68 order.
_MP_TO_IBUG68 = [
    # jaw line 0-16
    127, 234, 93, 132, 58, 172, 136, 150, 152, 379, 365, 397, 288, 361, 323, 454, 356,
    # right eyebrow 17-21
    70, 63, 105, 66, 107,
    # left eyebrow 22-26
    336, 296, 334, 293, 300,
    # nose bridge 27-30
    168, 197, 5, 4,
    # lower nose 31-35
    75, 97, 2, 326, 305,
    # right eye 36-41
    33, 160, 158, 133, 153, 144,
    # left eye 42-47
    362, 385, 387, 263, 373, 380,
    # outer lip 48-59
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    # inner lip 60-67
    78, 81, 13, 311, 308, 402, 14, 178,
]


def _parse_triangle_obj(path: str) -> dict:
    """Parse an all-triangle .obj with per-corner UVs.

    Works for both the MediaPipe canonical mesh and the preprocessed ICT
    head skin (tools/preprocess_ict.py emits pure triangles).

    Returns dict with: verts (V,3), uvs (V_uv,2), faces (F,3),
    face_uvs (F,3), uv_to_vert (V_uv,).
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
                # .obj is 1-indexed; tokens are "v", "v/vt", or "v/vt/vn".
                # We only read the first 3 corners — input must be triangulated.
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

    # uv→vert lookup: rasterization runs in UV-index space, but vertex
    # attributes (the per-view 2D/3D positions) are indexed by vertex.
    # Each UV index belongs to exactly one vertex (UV islands duplicate a
    # vertex in UV space; the reverse mapping is many-to-one).
    n_uv = uvs_arr.shape[0]
    uv_to_vert = np.full(n_uv, -1, dtype=np.int32)
    for vi3, vi_uv in zip(faces_arr.reshape(-1), face_uvs_arr.reshape(-1)):
        if uv_to_vert[vi_uv] == -1:
            uv_to_vert[vi_uv] = vi3
        elif uv_to_vert[vi_uv] != vi3:
            raise ValueError(
                f"Inconsistent UV→vert mapping: UV index {vi_uv} maps to both "
                f"vertex {uv_to_vert[vi_uv]} and {vi3}"
            )

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


# Path-keyed mesh cache — both bundled meshes are small and never change.
_MESH_CACHE: dict[str, dict] = {}


def _load_mesh(path: str) -> dict:
    if path in _MESH_CACHE:
        return _MESH_CACHE[path]
    data = _parse_triangle_obj(path)
    _MESH_CACHE[path] = data
    return data


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity transform mapping src → dst (Umeyama 1991).

    src, dst: (N, 3) corresponding point sets.
    Returns (s, R, t) such that  s * (R @ src.T).T + t  ≈  dst.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    var_src = (src_c ** 2).sum() / n
    s = float(np.trace(np.diag(D) @ S) / var_src) if var_src > 1e-12 else 1.0
    t = mu_dst - s * (R @ mu_src)
    return s, R, t


def _tps_refine(
    procrustes_verts: np.ndarray,
    landmark_idx: np.ndarray,
    mp_targets: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Landmark-exact thin-plate-spline refinement of a Procrustes fit.

    The rigid Umeyama fit gets the gross pose/scale right but poses ICT's
    *neutral* shape — individual landmarks land several px off the subject,
    smearing the baked face. This warps the Procrustes-aligned verts so the
    68 landmark verts hit their MediaPipe targets exactly.

    Done on the RESIDUAL (target − procrustes_landmark), so the correction
    is small and TPS extrapolation stays mild. A Gaussian distance falloff
    damps the correction for verts far from any landmark (ears / scalp /
    back of head have no landmark constraints) — those stay close to the
    pure-Procrustes pose.

    procrustes_verts: (V, 3) — rigid-fit verts
    landmark_idx:     (68,)  — indices of the landmark verts within V
    mp_targets:       (68, 3) — MediaPipe landmark targets (same metric space)
    strength:         0..1 blend of the correction (1 = full landmark-exact)
    Returns (V, 3) refined verts.
    """
    if strength <= 0.0:
        return procrustes_verts

    try:
        from scipy.interpolate import RBFInterpolator
        from scipy.spatial import cKDTree
    except ImportError:
        print("[BD FaceFit] scipy unavailable — skipping TPS refine, using rigid fit")
        return procrustes_verts

    ctrl = procrustes_verts[landmark_idx].astype(np.float64)   # (68, 3)
    residual = mp_targets.astype(np.float64) - ctrl            # (68, 3)

    # TPS maps 3D position → 3D residual; smoothing=0 → exact hit at controls.
    rbf = RBFInterpolator(ctrl, residual, kernel="thin_plate_spline", smoothing=0.0)
    correction = rbf(procrustes_verts.astype(np.float64))      # (V, 3)

    # Gaussian falloff by distance to the nearest landmark control point.
    # falloff_dist ≈ the control-point spread, so face-region verts get the
    # full correction and far verts (ears/scalp/back) fade toward rigid pose.
    tree = cKDTree(ctrl)
    dist, _ = tree.query(procrustes_verts, k=1)                # (V,)
    ctrl_centroid = ctrl.mean(axis=0)
    falloff_dist = float(np.linalg.norm(ctrl - ctrl_centroid, axis=1).mean()) + 1e-6
    falloff = np.exp(-((dist / falloff_dist) ** 2))            # (V,) 1 near → 0 far

    refined = procrustes_verts + (strength * falloff[:, None] * correction)
    return refined.astype(np.float32)


def _assemble_canonical_views(mesh: dict, in_views: list) -> list:
    """mediapipe_canonical: the 468 mesh verts ARE landmarks 0..467."""
    n_verts = mesh["verts"].shape[0]
    out = []
    for v in in_views:
        lm2d = v["landmarks_2d"]
        lm3d = v["landmarks_3d"]
        if lm2d.shape[0] >= n_verts:
            verts_2d = lm2d[:n_verts].astype(np.float32)
            verts_3d = lm3d[:n_verts].astype(np.float32)
        else:
            # Fewer landmarks than mesh verts — pad (shouldn't happen for 478)
            verts_2d = np.zeros((n_verts, 2), dtype=np.float32)
            verts_2d[: lm2d.shape[0]] = lm2d
            verts_3d = np.zeros((n_verts, 3), dtype=np.float32)
            verts_3d[: lm3d.shape[0]] = lm3d
        out.append({
            "verts_2d": verts_2d,
            "verts_3d": verts_3d,
            "transform_4x4": v["transform_4x4"].astype(np.float32),
            "detected": bool(v["detected"]),
            "view_hint": v["view_hint"],
            "image_size": tuple(v["image_size"]),
            "yaw_estimate": float(v["yaw_estimate"]),
        })
    return out


def _fit_ict_views(mesh: dict, ict_lm68_idx: np.ndarray, in_views: list,
                   landmark_warp: float = 1.0) -> list:
    """ict_facekit: fit the ICT head to MediaPipe's landmarks per view.

    Two stages:
      1. Rigid Umeyama similarity transform (scale + rotation + translation)
         — gets the gross pose/scale right, including the ear/scalp/back
         verts that have no landmark constraints.
      2. Landmark-exact TPS refinement (_tps_refine, strength=landmark_warp)
         — warps the result so the 68 landmark verts hit MediaPipe's targets
         exactly, fixing the face distortion that the neutral-shape rigid
         fit leaves behind. landmark_warp=0 = pure rigid (old behaviour).

    MediaPipe's landmarks_3d are normalized (x by width, y by height, z by
    width). We lift them to a common pixel-scale metric space [x*W, y*H, z*W]
    so a single-scale similarity transform fits all three axes. The fitted
    verts_3d are in pixel space; verts_2d is just its first two columns.
    """
    ict_verts = mesh["verts"].astype(np.float64)        # (V, 3)
    ict_lm = ict_verts[ict_lm68_idx]                     # (68, 3)
    mp_idx = np.asarray(_MP_TO_IBUG68, dtype=np.int64)
    n_verts = ict_verts.shape[0]

    out = []
    for v in in_views:
        h, w = v["image_size"]
        if not v["detected"]:
            out.append({
                "verts_2d": np.zeros((n_verts, 2), dtype=np.float32),
                "verts_3d": np.zeros((n_verts, 3), dtype=np.float32),
                "transform_4x4": v["transform_4x4"].astype(np.float32),
                "detected": False,
                "view_hint": v["view_hint"],
                "image_size": (h, w),
                "yaw_estimate": float(v["yaw_estimate"]),
            })
            continue

        # MediaPipe iBUG-68 landmarks → common pixel-scale metric space
        mp_lm = v["landmarks_3d"][mp_idx].astype(np.float64)   # (68, 3)
        mp_lm_metric = mp_lm * np.array([w, h, w], dtype=np.float64)

        # Stage 1: rigid similarity fit
        s, R, t = _umeyama(ict_lm, mp_lm_metric)
        verts_3d = s * (ict_verts @ R.T) + t                   # (V, 3) pixel space

        # Stage 2: landmark-exact TPS refinement of the residual
        verts_3d = _tps_refine(
            verts_3d.astype(np.float32), ict_lm68_idx,
            mp_lm_metric.astype(np.float32), strength=landmark_warp,
        )

        verts_3d = verts_3d.astype(np.float32)
        verts_2d = verts_3d[:, :2].copy()

        out.append({
            "verts_2d": verts_2d,
            "verts_3d": verts_3d,
            "transform_4x4": v["transform_4x4"].astype(np.float32),
            "detected": True,
            "view_hint": v["view_hint"],
            "image_size": (h, w),
            "yaw_estimate": float(v["yaw_estimate"]),
        })
    return out


class BD_FaceFit(io.ComfyNode):
    """Assemble LANDMARKS_BATCH + a head mesh into a FACE_FIT."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_FaceFit",
            display_name="BD Face Fit",
            category="🧠BrainDead/FaceWrap",
            description=(
                "Combine a LANDMARKS_BATCH with a head mesh into a FACE_FIT,\n"
                "the input for BD_FaceTextureBake.\n\n"
                "mesh_source:\n"
                "- mediapipe_canonical: 468-vert face shell (Apache-2.0). The\n"
                "  verts ARE the landmarks — pure assembly, no optimization.\n"
                "  Face only: no ears / scalp / back of head.\n"
                "- ict_facekit: ICT-FaceKit head skin (MIT) — full head incl.\n"
                "  ears + scalp + back. Procrustes-fits the ICT neutral head\n"
                "  to MediaPipe's 68 iBUG landmarks per view (poses the\n"
                "  neutral shape; not per-subject shape-accurate, but gives\n"
                "  the bake somewhere to land ear/scalp/rear pixels).\n\n"
                "Both emit the same FACE_FIT contract — the bake / blend /\n"
                "transfer nodes don't care which mesh produced it."
            ),
            inputs=[
                LandmarksBatchInput(
                    "landmarks_batch",
                    tooltip="Output of BD_FaceLandmarks (per-view 478 landmarks).",
                ),
                io.Combo.Input(
                    "mesh_source",
                    options=MESH_SOURCES,
                    default="mediapipe_canonical",
                    tooltip="Which head mesh to fit. ict_facekit adds ear/"
                            "scalp/back coverage the canonical face shell lacks.",
                ),
                io.String.Input(
                    "mesh_obj_path",
                    default="",
                    optional=True,
                    tooltip="Override path to the mesh .obj. Empty = use the "
                            "bundled asset for the selected mesh_source.",
                ),
                io.Float.Input(
                    "ict_landmark_warp",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    optional=True,
                    tooltip="ict_facekit only: strength of the landmark-exact "
                            "TPS warp applied after the rigid fit. 1.0 = the "
                            "68 landmarks hit MediaPipe's targets exactly "
                            "(fixes neutral-shape face distortion); 0.0 = pure "
                            "rigid Procrustes (the older, more distorted fit). "
                            "Ignored for mediapipe_canonical.",
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
        mesh_source: str = "mediapipe_canonical",
        mesh_obj_path: str = "",
        ict_landmark_warp: float = 1.0,
    ) -> io.NodeOutput:
        if not isinstance(landmarks_batch, dict) or "views" not in landmarks_batch:
            return io.NodeOutput(None, "ERROR: invalid LANDMARKS_BATCH input")

        if mesh_source not in MESH_SOURCES:
            return io.NodeOutput(None, f"ERROR: unknown mesh_source '{mesh_source}'")

        # Resolve the mesh .obj path
        path = mesh_obj_path.strip() if mesh_obj_path else ""
        if not path:
            path = DEFAULT_CANONICAL_OBJ if mesh_source == "mediapipe_canonical" else DEFAULT_ICT_OBJ
        path = os.path.expanduser(os.path.expandvars(path))
        if not os.path.exists(path):
            return io.NodeOutput(None, f"ERROR: mesh not found: {path}")

        try:
            mesh = _load_mesh(path)
        except Exception as e:
            return io.NodeOutput(None, f"ERROR: failed to parse {path}: {e}")

        in_views = landmarks_batch["views"]

        # Build per-view data according to the mesh source
        if mesh_source == "mediapipe_canonical":
            out_views = _assemble_canonical_views(mesh, in_views)
        else:  # ict_facekit
            if not os.path.exists(DEFAULT_ICT_LANDMARKS):
                return io.NodeOutput(None, f"ERROR: ICT landmark file not found: "
                                           f"{DEFAULT_ICT_LANDMARKS}")
            try:
                ict_lm68_idx = np.asarray(
                    json.load(open(DEFAULT_ICT_LANDMARKS))["idx_to_landmark_verts"],
                    dtype=np.int64,
                )
            except Exception as e:
                return io.NodeOutput(None, f"ERROR: bad ICT landmark file: {e}")
            if ict_lm68_idx.shape[0] != 68:
                return io.NodeOutput(None, f"ERROR: expected 68 ICT landmark verts, "
                                           f"got {ict_lm68_idx.shape[0]}")
            out_views = _fit_ict_views(mesh, ict_lm68_idx, in_views,
                                       landmark_warp=ict_landmark_warp)

        n_verts = mesh["verts"].shape[0]
        n_faces = mesh["faces"].shape[0]

        face_fit = {
            "canonical_verts": mesh["verts"],
            "canonical_uvs": mesh["uvs"],
            "faces": mesh["faces"],
            "face_uvs": mesh["face_uvs"],
            "uv_to_vert": mesh["uv_to_vert"],
            "mesh_source": mesh_source,
            "views": out_views,
        }

        n_detected = sum(1 for v in out_views if v["detected"])
        hints = ", ".join(v["view_hint"] for v in out_views)
        status = (
            f"{mesh_source}: {n_verts} verts, {n_faces} faces | "
            f"views: {n_detected}/{len(out_views)} detected | "
            f"hints: [{hints}]"
        )
        if mesh_source == "ict_facekit":
            status += f" | landmark_warp={ict_landmark_warp:.2f}"
        return io.NodeOutput(face_fit, status)


FACEWRAP_FACE_FIT_V3_NODES = [BD_FaceFit]

FACEWRAP_FACE_FIT_NODES = {
    "BD_FaceFit": BD_FaceFit,
}

FACEWRAP_FACE_FIT_DISPLAY_NAMES = {
    "BD_FaceFit": "BD Face Fit",
}
