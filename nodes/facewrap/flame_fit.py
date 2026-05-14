"""
BD_FlameFit — fit the FLAME head model to MediaPipe landmarks per view.

FLAME is a proper morphable model: a neutral template plus a learned
shape + expression basis plus pose blendshapes plus linear-blend
skinning. Unlike the ICT path (which warps a *neutral* mesh and so
leaves the cheeks/forehead distorted), fitting FLAME's shape basis
produces a *subject-accurate* face — the basis only spans real face
shapes.

Per view:
  1. FLAME forward — given shape β, expression ψ, pose θ → posed mesh.
  2. The official mediapipe_landmark_embedding gives, for 105 MediaPipe
     landmarks, the FLAME (triangle, barycentric) they sit on. Those
     105 FLAME-surface points are the fit targets.
  3. Adam optimises β, ψ, θ + a weak-perspective camera (scale, 2D
     translation) to minimise the 2D landmark reprojection error, with
     light L2 regularisation on β / ψ.

Emits the same FACE_FIT contract as BD_FaceFit — bake / blend / transfer
are unchanged. FLAME is a full head (face + scalp + neck), so it covers
far more than the canonical face shell.

Requires the converted model: tools/convert_flame.py → flame2023_facewrap.npz
(default /srv/AI_Stuff/models/flame/) plus the official
mediapipe_landmark_embedding.npz alongside it.
"""

import os

import numpy as np

from comfy_api.latest import io

from .types import LandmarksBatchOutput  # noqa: F401  (kept for parity)
from .types import FaceFitOutput, LandmarksBatchInput


DEFAULT_FLAME_NPZ = "/srv/AI_Stuff/models/flame/flame2023_facewrap.npz"
DEFAULT_EMBEDDING_NPZ = "/srv/AI_Stuff/models/flame/mediapipe_landmark_embedding.npz"

# FLAME 2023 shapedirs is 400 components: 300 shape + 100 expression.
_N_SHAPE_TOTAL = 300
_N_EXPR_TOTAL = 100


# ----------------------------------------------------------------------------
# Model loading (cached — numpy only, no chumpy)
# ----------------------------------------------------------------------------
_FLAME_CACHE: dict[str, dict] = {}
_EMBED_CACHE: dict[str, dict] = {}


def _load_flame(path: str) -> dict:
    if path in _FLAME_CACHE:
        return _FLAME_CACHE[path]
    d = np.load(path)
    flame = {k: d[k] for k in d.files}
    _FLAME_CACHE[path] = flame
    return flame


def _load_embedding(path: str) -> dict:
    if path in _EMBED_CACHE:
        return _EMBED_CACHE[path]
    d = np.load(path)
    emb = {
        "lmk_face_idx": d["lmk_face_idx"].astype(np.int64),
        "lmk_b_coords": d["lmk_b_coords"].astype(np.float32),
        "landmark_indices": d["landmark_indices"].astype(np.int64),
    }
    _EMBED_CACHE[path] = emb
    return emb


# ----------------------------------------------------------------------------
# FLAME forward (torch)
# ----------------------------------------------------------------------------
def _batch_rodrigues(axis_angle):
    """axis_angle: (J, 3) → rotation matrices (J, 3, 3)."""
    import torch
    angle = torch.norm(axis_angle + 1e-8, dim=-1, keepdim=True)   # (J, 1)
    axis = axis_angle / angle
    cos = torch.cos(angle).unsqueeze(-1)                           # (J, 1, 1)
    sin = torch.sin(angle).unsqueeze(-1)
    J = axis_angle.shape[0]
    zero = torch.zeros(J, device=axis_angle.device, dtype=axis_angle.dtype)
    ax, ay, az = axis[:, 0], axis[:, 1], axis[:, 2]
    K = torch.stack([
        torch.stack([zero, -az, ay], dim=-1),
        torch.stack([az, zero, -ax], dim=-1),
        torch.stack([-ay, ax, zero], dim=-1),
    ], dim=-2)                                                     # (J, 3, 3)
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).expand(J, 3, 3)
    return eye + sin * K + (1.0 - cos) * (K @ K)


def _flame_forward(flame_t, shape, expr, pose):
    """FLAME forward pass.

    flame_t: dict of torch tensors — v_template (V,3), shapedirs (V,3,400),
             posedirs (V,3,36), weights (V,J), J_regressor (J,V), kintree (2,J)
    shape:   (n_shape,) torch — shape coefficients
    expr:    (n_expr,)  torch — expression coefficients
    pose:    (J, 3)     torch — per-joint axis-angle [global, neck, jaw, leye, reye]
    Returns v_final (V, 3) torch.
    """
    import torch
    device = flame_t["v_template"].device
    V = flame_t["v_template"].shape[0]
    n_joints = flame_t["weights"].shape[1]

    # 1. Shape + expression — pack into the 400-wide coefficient vector
    n_basis = flame_t["shapedirs"].shape[2]
    coeffs = torch.zeros(n_basis, device=device, dtype=shape.dtype)
    coeffs[: shape.shape[0]] = shape
    coeffs[_N_SHAPE_TOTAL: _N_SHAPE_TOTAL + expr.shape[0]] = expr
    v_shaped = flame_t["v_template"] + torch.einsum(
        "vnc,c->vn", flame_t["shapedirs"], coeffs)

    # 2. Joints from the shaped mesh
    J = flame_t["J_regressor"] @ v_shaped                          # (n_joints, 3)

    # 3. Pose blendshapes — posedirs driven by (R - I) of the non-root joints
    R = _batch_rodrigues(pose)                                      # (n_joints, 3, 3)
    eye = torch.eye(3, device=device, dtype=R.dtype)
    pose_feature = (R[1:] - eye).reshape(-1)                        # ((n_joints-1)*9,)
    v_posed = v_shaped + torch.einsum(
        "vnc,c->vn", flame_t["posedirs"], pose_feature)

    # 4. Linear blend skinning
    parents = flame_t["kintree"][0].long()                          # (n_joints,)
    rel_J = J.clone()
    rel_J[1:] = J[1:] - J[parents[1:]]

    def _to4x4(R3, t3):
        M = torch.zeros(R3.shape[0], 4, 4, device=device, dtype=R3.dtype)
        M[:, :3, :3] = R3
        M[:, :3, 3] = t3
        M[:, 3, 3] = 1.0
        return M

    local = _to4x4(R, rel_J)                                        # (n_joints, 4, 4)
    world = [local[0]]
    for i in range(1, n_joints):
        world.append(world[parents[i]] @ local[i])
    world = torch.stack(world)                                      # (n_joints, 4, 4)

    # Strip the rest-pose offset so the neutral pose skins to identity
    rest = _to4x4(eye.expand(n_joints, 3, 3), J)
    T = world @ torch.inverse(rest)                                 # (n_joints, 4, 4)

    # Blend per vertex
    Tv = torch.einsum("vj,jxy->vxy", flame_t["weights"], T)         # (V, 4, 4)
    v_homo = torch.cat([v_posed, torch.ones(V, 1, device=device, dtype=v_posed.dtype)],
                       dim=-1)                                       # (V, 4)
    v_final = torch.einsum("vxy,vy->vx", Tv, v_homo)[:, :3]
    return v_final


def _landmark_positions(v_final, faces_t, lmk_face_idx_t, lmk_b_coords_t):
    """Barycentric-interpolate the 105 embedded landmark 3D positions."""
    import torch  # noqa: F401
    tri = v_final[faces_t[lmk_face_idx_t]]                          # (105, 3, 3)
    return (tri * lmk_b_coords_t.unsqueeze(-1)).sum(dim=1)          # (105, 3)


# ----------------------------------------------------------------------------
# Per-view fit
# ----------------------------------------------------------------------------
def _fit_flame_view(flame_t, faces_t, emb_t, mp_lm_2d, image_size,
                    n_shape, n_expr, iters, device):
    """Staged Adam fit of FLAME + a weak-perspective camera to a view.

    Joint optimisation of shape+expr+pose+camera from zero is badly
    under-constrained (168 params, 105 landmarks) and lands in bad minima.
    Instead, fit coarse-to-fine:
      Stage A — camera + GLOBAL pose only (shape/expr frozen at neutral).
                Orients + scales the neutral head to the landmarks.
      Stage B — + shape. Identity refines onto an already-aligned head.
      Stage C — + expression + jaw/neck pose. Fine detail.
    Plus a good closed-form init for scale/translation so Stage A starts close.

    mp_lm_2d: (105, 2) MediaPipe landmark pixel coords for this view.
    Returns (v_final_np, verts_2d_np, verts_3d_np, rmse_px).
    """
    import torch

    h, w = image_size
    target = torch.from_numpy(mp_lm_2d.astype(np.float32)).to(device)   # (105, 2)
    flip = torch.tensor([1.0, -1.0], device=device)

    shape = torch.zeros(n_shape, device=device, requires_grad=True)
    expr = torch.zeros(n_expr, device=device, requires_grad=True)
    pose = torch.zeros(5, 3, device=device, requires_grad=True)

    # --- Closed-form init for the weak-perspective camera ---
    # Project the NEUTRAL FLAME landmarks (Y-flipped) and match their 2D
    # spread + centroid to the target's, so Stage A starts well-aligned.
    with torch.no_grad():
        v0 = _flame_forward(flame_t, shape, expr, pose)
        lmk0 = _landmark_positions(v0, faces_t,
                                   emb_t["lmk_face_idx"], emb_t["lmk_b_coords"])
        lmk0_xy = lmk0[:, :2] * flip
        src_c = lmk0_xy.mean(dim=0)
        dst_c = target.mean(dim=0)
        src_spread = (lmk0_xy - src_c).abs().mean()
        dst_spread = (target - dst_c).abs().mean()
        scale0 = float(dst_spread / (src_spread + 1e-8))
        trans0 = (dst_c - scale0 * src_c)
    scale = torch.tensor(scale0, device=device, requires_grad=True)
    trans = trans0.clone().detach().requires_grad_(True)

    def project(v3d):
        return scale * (v3d[:, :2] * flip) + trans

    def run_stage(active_params, lr_groups, n_iters, reg_shape, reg_expr, reg_pose):
        opt = torch.optim.Adam(lr_groups)
        for _ in range(n_iters):
            opt.zero_grad()
            v_final = _flame_forward(flame_t, shape, expr, pose)
            lmk3d = _landmark_positions(v_final, faces_t,
                                        emb_t["lmk_face_idx"], emb_t["lmk_b_coords"])
            proj = project(lmk3d)
            loss = ((proj - target) ** 2).sum(dim=-1).mean()
            loss = loss + reg_shape * (shape ** 2).mean() \
                        + reg_expr * (expr ** 2).mean() \
                        + reg_pose * (pose[1:] ** 2).mean()
            loss.backward()
            # Freeze inactive params by zeroing their grads
            for p in (shape, expr, pose, scale, trans):
                if p not in active_params and p.grad is not None:
                    p.grad.zero_()
            opt.step()

    # Split the iteration budget across the three stages
    n_a = max(40, iters // 4)
    n_b = max(60, iters // 2)
    n_c = iters - n_a - n_b

    # Stage A — camera + global pose. shape/expr stay 0.
    run_stage(
        active_params={pose, scale, trans},
        lr_groups=[{"params": [pose], "lr": 0.05},
                   {"params": [scale], "lr": scale0 * 0.05 + 1.0},
                   {"params": [trans], "lr": 3.0}],
        n_iters=n_a, reg_shape=0.0, reg_expr=0.0, reg_pose=5e-2,
    )
    # Stage B — + shape.
    run_stage(
        active_params={shape, pose, scale, trans},
        lr_groups=[{"params": [shape], "lr": 0.03},
                   {"params": [pose], "lr": 0.02},
                   {"params": [scale], "lr": scale0 * 0.02 + 0.5},
                   {"params": [trans], "lr": 1.5}],
        n_iters=n_b, reg_shape=2e-3, reg_expr=0.0, reg_pose=5e-2,
    )
    # Stage C — + expression + jaw/neck pose.
    run_stage(
        active_params={shape, expr, pose, scale, trans},
        lr_groups=[{"params": [shape], "lr": 0.01},
                   {"params": [expr], "lr": 0.02},
                   {"params": [pose], "lr": 0.01},
                   {"params": [scale], "lr": scale0 * 0.01 + 0.25},
                   {"params": [trans], "lr": 0.75}],
        n_iters=n_c, reg_shape=2e-3, reg_expr=3e-3, reg_pose=2e-2,
    )

    with torch.no_grad():
        v_final = _flame_forward(flame_t, shape, expr, pose)
        lmk3d = _landmark_positions(v_final, faces_t,
                                    emb_t["lmk_face_idx"], emb_t["lmk_b_coords"])
        rmse = float(((project(lmk3d) - target) ** 2).sum(dim=-1).mean().item()) ** 0.5
        verts_2d = project(v_final)                                 # (V, 2) pixels
        # verts_3d: x,y in pixel space (matches verts_2d), z = scaled depth.
        # FLAME faces +Z; camera looks toward -Z. The bake's view-cosine wants
        # camera-facing normals to have negative z → negate.
        z = -scale * v_final[:, 2]
        verts_3d = torch.stack([verts_2d[:, 0], verts_2d[:, 1], z], dim=-1)

    return (v_final.cpu().numpy(),
            verts_2d.cpu().numpy().astype(np.float32),
            verts_3d.cpu().numpy().astype(np.float32),
            rmse)


class BD_FlameFit(io.ComfyNode):
    """Fit the FLAME head model to MediaPipe landmarks → FACE_FIT."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_FlameFit",
            display_name="BD Flame Fit",
            category="🧠BrainDead/FaceWrap",
            description=(
                "Fit the FLAME head model to a LANDMARKS_BATCH and emit a\n"
                "FACE_FIT — drop-in alternative to BD_FaceFit.\n\n"
                "FLAME is a proper morphable model: fitting its shape +\n"
                "expression basis produces a SUBJECT-ACCURATE face (the\n"
                "basis only spans real face shapes), unlike warping a\n"
                "neutral mesh. Full head — face + scalp + neck.\n\n"
                "Per view: Adam fits shape/expression/pose + a weak-\n"
                "perspective camera to the 105 MediaPipe landmarks via the\n"
                "official mediapipe_landmark_embedding.\n\n"
                "Requires the converted model (tools/convert_flame.py →\n"
                "flame2023_facewrap.npz) + mediapipe_landmark_embedding.npz."
            ),
            inputs=[
                LandmarksBatchInput(
                    "landmarks_batch",
                    tooltip="Output of BD_FaceLandmarks.",
                ),
                io.String.Input(
                    "flame_npz_path",
                    default="",
                    optional=True,
                    tooltip=f"Path to flame2023_facewrap.npz. Empty = "
                            f"{DEFAULT_FLAME_NPZ}",
                ),
                io.String.Input(
                    "embedding_npz_path",
                    default="",
                    optional=True,
                    tooltip=f"Path to mediapipe_landmark_embedding.npz. Empty = "
                            f"{DEFAULT_EMBEDDING_NPZ}",
                ),
                io.Int.Input(
                    "shape_coeffs",
                    default=100,
                    min=10,
                    max=300,
                    step=10,
                    optional=True,
                    tooltip="Number of FLAME shape basis components to fit. "
                            "More = finer identity, slower, more overfit risk.",
                ),
                io.Int.Input(
                    "expr_coeffs",
                    default=50,
                    min=0,
                    max=100,
                    step=10,
                    optional=True,
                    tooltip="Number of FLAME expression components to fit.",
                ),
                io.Int.Input(
                    "iterations",
                    default=300,
                    min=50,
                    max=2000,
                    step=50,
                    optional=True,
                    tooltip="Adam iterations per view.",
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
        flame_npz_path: str = "",
        embedding_npz_path: str = "",
        shape_coeffs: int = 100,
        expr_coeffs: int = 50,
        iterations: int = 300,
    ) -> io.NodeOutput:
        if not isinstance(landmarks_batch, dict) or "views" not in landmarks_batch:
            return io.NodeOutput(None, "ERROR: invalid LANDMARKS_BATCH input")

        try:
            import torch
        except ImportError as e:
            return io.NodeOutput(None, f"ERROR: torch unavailable ({e})")

        flame_path = (flame_npz_path.strip() or DEFAULT_FLAME_NPZ)
        emb_path = (embedding_npz_path.strip() or DEFAULT_EMBEDDING_NPZ)
        flame_path = os.path.expanduser(os.path.expandvars(flame_path))
        emb_path = os.path.expanduser(os.path.expandvars(emb_path))
        if not os.path.exists(flame_path):
            return io.NodeOutput(None, f"ERROR: FLAME model not found: {flame_path} "
                                       f"(run tools/convert_flame.py)")
        if not os.path.exists(emb_path):
            return io.NodeOutput(None, f"ERROR: MP embedding not found: {emb_path}")

        try:
            flame = _load_flame(flame_path)
            emb = _load_embedding(emb_path)
        except Exception as e:
            return io.NodeOutput(None, f"ERROR: failed to load FLAME assets: {e}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pack the model into torch tensors once
        flame_t = {
            "v_template": torch.from_numpy(flame["v_template"]).float().to(device),
            "shapedirs": torch.from_numpy(flame["shapedirs"]).float().to(device),
            "posedirs": torch.from_numpy(flame["posedirs"]).float().to(device),
            "weights": torch.from_numpy(flame["weights"]).float().to(device),
            "J_regressor": torch.from_numpy(flame["J_regressor"]).float().to(device),
            "kintree": torch.from_numpy(flame["kintree"]).long().to(device),
        }
        faces_t = torch.from_numpy(flame["faces"]).long().to(device)
        emb_t = {
            "lmk_face_idx": torch.from_numpy(emb["lmk_face_idx"]).long().to(device),
            "lmk_b_coords": torch.from_numpy(emb["lmk_b_coords"]).float().to(device),
        }
        mp_lmk_idx = emb["landmark_indices"]   # (105,) which MP landmarks

        n_shape = min(shape_coeffs, _N_SHAPE_TOTAL)
        n_expr = min(expr_coeffs, _N_EXPR_TOTAL)
        n_verts = flame["v_template"].shape[0]

        in_views = landmarks_batch["views"]
        out_views = []
        reports = []
        for i, v in enumerate(in_views):
            h, w = v["image_size"]
            if not v["detected"]:
                out_views.append({
                    "verts_2d": np.zeros((n_verts, 2), dtype=np.float32),
                    "verts_3d": np.zeros((n_verts, 3), dtype=np.float32),
                    "transform_4x4": v["transform_4x4"].astype(np.float32),
                    "detected": False,
                    "view_hint": v["view_hint"],
                    "image_size": (h, w),
                    "yaw_estimate": float(v["yaw_estimate"]),
                })
                continue

            # MediaPipe's 105 embedded landmarks, in pixel coords
            mp_lm_2d = v["landmarks_2d"][mp_lmk_idx]                 # (105, 2)
            _v_final, verts_2d, verts_3d, rmse = _fit_flame_view(
                flame_t, faces_t, emb_t, mp_lm_2d, (h, w),
                n_shape, n_expr, iterations, device,
            )
            out_views.append({
                "verts_2d": verts_2d,
                "verts_3d": verts_3d,
                "transform_4x4": v["transform_4x4"].astype(np.float32),
                "detected": True,
                "view_hint": v["view_hint"],
                "image_size": (h, w),
                "yaw_estimate": float(v["yaw_estimate"]),
            })
            reports.append(f"{i}:{v['view_hint']} rmse={rmse:.1f}px")

        face_fit = {
            "canonical_verts": flame["v_template"].astype(np.float32),
            "canonical_uvs": flame["uvs"].astype(np.float32),
            "faces": flame["faces"].astype(np.int32),
            "face_uvs": flame["face_uvs"].astype(np.int32),
            "uv_to_vert": _build_uv_to_vert(flame["faces"], flame["face_uvs"]),
            "mesh_source": "flame",
            "views": out_views,
        }

        n_detected = sum(1 for v in out_views if v["detected"])
        status = (
            f"flame: {n_verts} verts, {flame['faces'].shape[0]} faces | "
            f"views: {n_detected}/{len(out_views)} fitted | "
            f"shape={n_shape} expr={n_expr} iters={iterations} | "
            f"{', '.join(reports)}"
        )
        return io.NodeOutput(face_fit, status)


def _build_uv_to_vert(faces: np.ndarray, face_uvs: np.ndarray) -> np.ndarray:
    """uv→vert lookup (same contract as the .obj parser in face_fit.py)."""
    n_uv = int(face_uvs.max()) + 1
    uv_to_vert = np.full(n_uv, -1, dtype=np.int32)
    for vi3, vi_uv in zip(faces.reshape(-1), face_uvs.reshape(-1)):
        if uv_to_vert[vi_uv] == -1:
            uv_to_vert[vi_uv] = vi3
        elif uv_to_vert[vi_uv] != vi3:
            raise ValueError(
                f"Inconsistent UV→vert mapping: UV index {vi_uv} maps to both "
                f"vertex {uv_to_vert[vi_uv]} and {vi3}")
    n_orphan = int((uv_to_vert == -1).sum())
    if n_orphan > 0:
        print(f"[BD FlameFit] WARNING: {n_orphan}/{n_uv} UVs unreferenced; "
              f"setting to vertex 0.")
        uv_to_vert[uv_to_vert == -1] = 0
    return uv_to_vert


FACEWRAP_FLAME_FIT_V3_NODES = [BD_FlameFit]

FACEWRAP_FLAME_FIT_NODES = {
    "BD_FlameFit": BD_FlameFit,
}

FACEWRAP_FLAME_FIT_DISPLAY_NAMES = {
    "BD_FlameFit": "BD Flame Fit",
}
