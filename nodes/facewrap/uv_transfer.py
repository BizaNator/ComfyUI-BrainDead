"""
BD_UVTransfer — cross-mesh UV→UV texture warp.

Given a source texture in some source UV layout (e.g. the canonical face
UV from BD_UVConfidenceBlend) and a precomputed correspondence between
the source and target meshes, warp the source texture into the target's
UV layout (CC5, Metahuman, etc.).

The correspondence is a `.npz` file with at minimum:
    target_source_uv : (V_target, 2) float — for each target vertex, the
                         UV coordinate in the source texture that this
                         vertex corresponds to.
    target_faces     : (F_target, 3)   int — target mesh face indices
                         (vertex indices)
    target_uvs       : (V_target, 2) float — target mesh UV coords
                         (per-vertex, V-up .obj convention)
    target_face_uvs  : (F_target, 3)   int — target mesh per-face UV indices
                         (== target_faces if 1:1 vertex/UV)
    uv_to_vert       : (V_target_uv,)  int — UV→vertex lookup (target side)
    valid_mask       : (V_target,)     bool — False where the target vertex
                         had no good source-mesh correspondence
                         (e.g. scalp/rear of head when source is face-only)

Build a correspondence with `tools/build_correspondence.py` (one-time
per target rig).
"""

import os

import numpy as np
import torch

from comfy_api.latest import io


def _bilinear_sample_uv(texture: torch.Tensor, uv: torch.Tensor,
                        valid: torch.Tensor) -> torch.Tensor:
    """Bilinear-sample texture (H, W, 3) at UV coords in [0,1].

    uv: (M, 2) — uv[:,0] = u, uv[:,1] = v (V-up convention; we flip internally)
    valid: (M,) bool
    """
    h, w, c = texture.shape
    out = torch.zeros((uv.shape[0], c), dtype=texture.dtype, device=texture.device)
    if not valid.any():
        return out

    # Texture is stored Y-down (image array convention), UV is V-up: flip V
    u = uv[valid, 0].clamp(0.0, 1.0)
    v = (1.0 - uv[valid, 1]).clamp(0.0, 1.0)
    x = u * (w - 1)
    y = v * (h - 1)

    x0 = torch.floor(x).long().clamp(0, w - 1)
    x1 = (x0 + 1).clamp(0, w - 1)
    y0 = torch.floor(y).long().clamp(0, h - 1)
    y1 = (y0 + 1).clamp(0, h - 1)

    wx1 = (x - x0.float()).unsqueeze(-1)
    wx0 = 1.0 - wx1
    wy1 = (y - y0.float()).unsqueeze(-1)
    wy0 = 1.0 - wy1

    c00 = texture[y0, x0]
    c01 = texture[y0, x1]
    c10 = texture[y1, x0]
    c11 = texture[y1, x1]

    sampled = c00 * wx0 * wy0 + c01 * wx1 * wy0 + c10 * wx0 * wy1 + c11 * wx1 * wy1
    out[valid] = sampled
    return out


class BD_UVTransfer(io.ComfyNode):
    """Warp a texture from one UV layout to another via vertex correspondence."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_UVTransfer",
            display_name="BD UV Transfer",
            category="🧠BrainDead/FaceWrap",
            description=(
                "Retarget a UV texture from one mesh's UV layout to another's\n"
                "using a precomputed vertex-correspondence file (.npz).\n\n"
                "Build the correspondence ONCE per target rig with\n"
                "`tools/build_correspondence.py`; reuse it for every head.\n\n"
                "Typical use: BD_UVConfidenceBlend's canonical-UV output →\n"
                "this node → CC5 or Metahuman UV layout → Qwen Image Edit\n"
                "finalize, all consistent with the chosen target rig."
            ),
            inputs=[
                io.Image.Input(
                    "source_texture",
                    tooltip="Source UV texture (e.g. from BD_UVConfidenceBlend).",
                ),
                io.String.Input(
                    "correspondence_path",
                    default="",
                    tooltip="Path to .npz built by tools/build_correspondence.py.",
                ),
                io.Int.Input(
                    "output_size",
                    default=2048,
                    min=256,
                    max=8192,
                    step=128,
                    tooltip="Target-UV texture resolution.",
                ),
                io.Mask.Input(
                    "source_mask",
                    optional=True,
                    tooltip="Optional: source filled_mask. If provided, target\n"
                            "texels mapping to unfilled source regions become 0\n"
                            "in the output mask.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="uv_texture"),
                io.Mask.Output(display_name="filled_mask"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        source_texture: torch.Tensor,
        correspondence_path: str,
        output_size: int = 2048,
        source_mask=None,
    ) -> io.NodeOutput:
        try:
            import nvdiffrast.torch as dr
        except ImportError as e:
            empty_img = torch.zeros((1, output_size, output_size, 3))
            empty_mask = torch.zeros((1, output_size, output_size))
            return io.NodeOutput(empty_img, empty_mask, f"ERROR: nvdiffrast not installed ({e})")

        if not correspondence_path or not os.path.exists(correspondence_path):
            empty_img = torch.zeros((1, output_size, output_size, 3))
            empty_mask = torch.zeros((1, output_size, output_size))
            return io.NodeOutput(empty_img, empty_mask,
                                 f"ERROR: correspondence file not found: {correspondence_path}")

        try:
            corr = np.load(correspondence_path)
        except Exception as e:
            empty_img = torch.zeros((1, output_size, output_size, 3))
            empty_mask = torch.zeros((1, output_size, output_size))
            return io.NodeOutput(empty_img, empty_mask, f"ERROR: bad correspondence: {e}")

        required = ("target_source_uv", "target_uvs", "target_face_uvs", "uv_to_vert")
        for k in required:
            if k not in corr:
                empty_img = torch.zeros((1, output_size, output_size, 3))
                empty_mask = torch.zeros((1, output_size, output_size))
                return io.NodeOutput(empty_img, empty_mask,
                                     f"ERROR: correspondence missing '{k}'")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            empty_img = torch.zeros((1, output_size, output_size, 3))
            empty_mask = torch.zeros((1, output_size, output_size))
            return io.NodeOutput(empty_img, empty_mask, "ERROR: CUDA required")

        # Source texture: take first frame from (B,H,W,3)
        if source_texture is None or source_texture.ndim != 4 or source_texture.shape[0] == 0:
            empty_img = torch.zeros((1, output_size, output_size, 3))
            empty_mask = torch.zeros((1, output_size, output_size))
            return io.NodeOutput(empty_img, empty_mask, "ERROR: invalid source_texture")
        src_tex = source_texture[0].to(device, dtype=torch.float32)

        # ---- Build per-UV-vertex source-UV-lookup attribute ----
        target_source_uv = torch.from_numpy(np.asarray(corr["target_source_uv"], dtype=np.float32)).to(device)
        target_uvs = torch.from_numpy(np.asarray(corr["target_uvs"], dtype=np.float32)).to(device)
        target_face_uvs = torch.from_numpy(np.asarray(corr["target_face_uvs"], dtype=np.int32)).to(device)
        uv_to_vert = torch.from_numpy(np.asarray(corr["uv_to_vert"], dtype=np.int64)).to(device)
        valid_mask_v = (
            torch.from_numpy(np.asarray(corr["valid_mask"], dtype=np.bool_)).to(device)
            if "valid_mask" in corr.files
            else torch.ones(target_source_uv.shape[0], dtype=torch.bool, device=device)
        )

        # Per-UV-index attributes (target-side rasterization runs in UV index space)
        src_uv_per_uv = target_source_uv[uv_to_vert]   # (V_target_uv, 2)
        valid_per_uv = valid_mask_v[uv_to_vert].float().unsqueeze(-1)  # (V_target_uv, 1)
        attr = torch.cat([src_uv_per_uv, valid_per_uv], dim=-1)        # (V_target_uv, 3)

        # ---- Rasterize target mesh in target-UV space ----
        uvs_ndc = target_uvs.clone()
        uvs_ndc[:, 1] = 1.0 - uvs_ndc[:, 1]
        rast_pos = torch.cat([
            uvs_ndc * 2.0 - 1.0,
            torch.zeros_like(uvs_ndc[:, :1]),
            torch.ones_like(uvs_ndc[:, :1]),
        ], dim=-1).unsqueeze(0)

        ctx = dr.RasterizeCudaContext(device=device)
        rast, _ = dr.rasterize(
            ctx, rast_pos, target_face_uvs,
            resolution=[output_size, output_size],
        )
        tri_mask = rast[0, ..., 3] > 0

        # Interpolate the (source_u, source_v, valid) attribute
        interp, _ = dr.interpolate(attr.unsqueeze(0), rast, target_face_uvs)
        # interp: (1, H, W, 3) — channels = (source_u, source_v, target_valid)
        sample_uv = interp[0, ..., :2]      # (H, W, 2)
        target_valid_f = interp[0, ..., 2]  # (H, W) — interpolated valid_mask

        # Pixels where ALL 3 verts of the triangle are valid: valid_f == 1.0
        # Partial validity (e.g. 0.67) means triangle straddles the source-mesh
        # boundary; drop them to avoid garbage at the boundary.
        target_in_valid = target_valid_f > 0.99
        valid = tri_mask & target_in_valid

        # ---- Sample the source texture at the per-texel source UV ----
        sample_uv_flat = sample_uv.reshape(-1, 2)
        valid_flat = valid.reshape(-1)
        sampled = _bilinear_sample_uv(src_tex, sample_uv_flat, valid_flat)
        out_tex = sampled.reshape(output_size, output_size, 3)

        # Optionally mask further by source_mask (only count target texels whose
        # source UV actually has data)
        out_mask = valid.float()
        if source_mask is not None and source_mask.ndim == 3 and source_mask.shape[0] > 0:
            src_mask = source_mask[0].to(device, dtype=torch.float32)
            h_src, w_src = src_mask.shape
            # Sample source_mask at sample_uv using nearest-neighbor for binary mask
            uv_clamped = sample_uv.clamp(0.0, 1.0)
            sx = (uv_clamped[..., 0] * (w_src - 1)).long().clamp(0, w_src - 1)
            sy = ((1.0 - uv_clamped[..., 1]) * (h_src - 1)).long().clamp(0, h_src - 1)
            src_mask_sampled = src_mask[sy, sx]
            out_mask = out_mask * (src_mask_sampled > 0.05).float()
            out_tex = out_tex * out_mask.unsqueeze(-1)

        n_filled = int(out_mask.bool().sum().item())
        total = output_size * output_size
        coverage = 100.0 * n_filled / total
        status = (
            f"transferred to target UV ({output_size}x{output_size}) | "
            f"target valid coverage {coverage:.1f}% | "
            f"V_target_uv={target_uvs.shape[0]}, F_target={target_face_uvs.shape[0]}"
        )

        return io.NodeOutput(
            out_tex.unsqueeze(0).cpu(),
            out_mask.unsqueeze(0).cpu(),
            status,
        )


FACEWRAP_TRANSFER_V3_NODES = [BD_UVTransfer]

FACEWRAP_TRANSFER_NODES = {
    "BD_UVTransfer": BD_UVTransfer,
}

FACEWRAP_TRANSFER_DISPLAY_NAMES = {
    "BD_UVTransfer": "BD UV Transfer",
}
