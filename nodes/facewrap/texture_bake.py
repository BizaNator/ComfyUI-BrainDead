"""
BD_FaceTextureBake — per-view bake of a source photo into the canonical
UV layout using nvdiffrast.

For each output UV texel:
  1. Rasterize the canonical UV layout in NDC → per-texel (face_id, barycentric)
  2. Interpolate the 3 per-corner verts_2d positions → source-image coord
  3. Bilinear-sample the source image at that coord
  4. Compute per-texel confidence = front-facingness of the underlying
     triangle, clamped to [0, 1]

Reuses the UV-rasterization recipe from `nodes/mesh/ovoxel_texture_bake.py`.
"""

import numpy as np
import torch

from comfy_api.latest import io

from .types import FaceFitInput


def _bilinear_sample(image: torch.Tensor, xy: torch.Tensor,
                     valid: torch.Tensor) -> torch.Tensor:
    """Bilinear-sample image at fractional pixel coords.

    image: (H, W, 3) float
    xy:    (M, 2) float, columns = (x, y) in image-pixel space
    valid: (M,) bool, only valid coords are sampled (others get 0)
    Returns (M, 3) float.
    """
    h, w, c = image.shape
    out = torch.zeros((xy.shape[0], c), dtype=image.dtype, device=image.device)
    if not valid.any():
        return out

    x = xy[valid, 0]
    y = xy[valid, 1]

    x0 = torch.floor(x).long().clamp(0, w - 1)
    x1 = (x0 + 1).clamp(0, w - 1)
    y0 = torch.floor(y).long().clamp(0, h - 1)
    y1 = (y0 + 1).clamp(0, h - 1)

    wx1 = (x - x0.float()).unsqueeze(-1)
    wx0 = 1.0 - wx1
    wy1 = (y - y0.float()).unsqueeze(-1)
    wy0 = 1.0 - wy1

    c00 = image[y0, x0]
    c01 = image[y0, x1]
    c10 = image[y1, x0]
    c11 = image[y1, x1]

    sampled = (
        c00 * wx0 * wy0 +
        c01 * wx1 * wy0 +
        c10 * wx0 * wy1 +
        c11 * wx1 * wy1
    )
    out[valid] = sampled
    return out


def _per_face_view_cosine(verts_3d: torch.Tensor,
                          faces: torch.Tensor) -> torch.Tensor:
    """For each triangle, compute cos(angle between face normal and camera dir).

    MediaPipe 3D landmark convention: x right, y down, z depth-relative
    (smaller z = closer to camera). Outward face normals on a front-facing
    head point in -z direction, so the cosine = max(0, -normal_z).

    verts_3d: (V, 3) float
    faces:    (F, 3) int  — vertex indices (NOT UV indices)
    Returns:  (F,) float in [0, 1]
    """
    v0 = verts_3d[faces[:, 0]]
    v1 = verts_3d[faces[:, 1]]
    v2 = verts_3d[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    n = torch.cross(e1, e2, dim=-1)
    n = torch.nn.functional.normalize(n, dim=-1)
    # Camera dir = -z in MediaPipe convention; cos(normal, -z) = -n_z
    cos = (-n[:, 2]).clamp(min=0.0, max=1.0)
    return cos


class BD_FaceTextureBake(io.ComfyNode):
    """Bake a source photo into the canonical UV layout for one view."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_FaceTextureBake",
            display_name="BD Face Texture Bake",
            category="🧠BrainDead/FaceWrap",
            description=(
                "Bake one source photo into the canonical UV layout using\n"
                "nvdiffrast. Outputs a partial UV texture + a confidence mask\n"
                "(per-texel front-facingness, clamped to [0,1]).\n\n"
                "Pair with BD_UVConfidenceBlend to composite multiple views."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Source photo batch — bake samples from images[view_index].",
                ),
                FaceFitInput(
                    "face_fit",
                    tooltip="FACE_FIT from BD_FaceFit. Provides canonical mesh "
                            "+ per-view 2D vertex positions.",
                ),
                io.Int.Input(
                    "view_index",
                    default=0,
                    min=0,
                    max=63,
                    step=1,
                    tooltip="Which view of the FACE_FIT to bake.",
                ),
                io.Int.Input(
                    "texture_size",
                    default=2048,
                    min=256,
                    max=8192,
                    step=128,
                    tooltip="Output UV texture resolution.",
                ),
                io.Float.Input(
                    "min_confidence",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    optional=True,
                    tooltip="Texels with face-cosine below this are zeroed in the mask.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="uv_texture"),
                io.Mask.Output(display_name="confidence"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        face_fit,
        view_index: int = 0,
        texture_size: int = 2048,
        min_confidence: float = 0.0,
    ) -> io.NodeOutput:
        try:
            import nvdiffrast.torch as dr
        except ImportError as e:
            empty_img = torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, texture_size, texture_size), dtype=torch.float32)
            return io.NodeOutput(empty_img, empty_mask, f"ERROR: nvdiffrast not installed ({e})")

        # Validate FACE_FIT
        if not isinstance(face_fit, dict) or "views" not in face_fit:
            empty_img = torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, texture_size, texture_size), dtype=torch.float32)
            return io.NodeOutput(empty_img, empty_mask, "ERROR: invalid FACE_FIT")

        if view_index < 0 or view_index >= len(face_fit["views"]):
            empty_img = torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, texture_size, texture_size), dtype=torch.float32)
            return io.NodeOutput(empty_img, empty_mask,
                                 f"ERROR: view_index {view_index} out of range "
                                 f"(have {len(face_fit['views'])} views)")

        view = face_fit["views"][view_index]
        if not view["detected"]:
            empty_img = torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, texture_size, texture_size), dtype=torch.float32)
            return io.NodeOutput(empty_img, empty_mask,
                                 f"View {view_index} ({view['view_hint']}) not detected — "
                                 f"returning empty texture.")

        # Pick the source image — use view_index if available, else fall back to 0
        if images is None or images.ndim != 4 or images.shape[0] == 0:
            empty_img = torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, texture_size, texture_size), dtype=torch.float32)
            return io.NodeOutput(empty_img, empty_mask, "ERROR: invalid IMAGE input")
        img_idx = min(view_index, images.shape[0] - 1)
        src_image = images[img_idx]  # (H, W, 3)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            empty_img = torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, texture_size, texture_size), dtype=torch.float32)
            return io.NodeOutput(empty_img, empty_mask, "ERROR: CUDA not available")

        # ---- Pack mesh + view data into CUDA tensors ----
        uvs = torch.from_numpy(face_fit["canonical_uvs"]).to(device, dtype=torch.float32)
        face_uvs = torch.from_numpy(face_fit["face_uvs"]).to(device, dtype=torch.int32)
        faces_v = torch.from_numpy(face_fit["faces"]).to(device, dtype=torch.int64)
        uv_to_vert = torch.from_numpy(face_fit["uv_to_vert"]).to(device, dtype=torch.int64)

        verts_2d_per_3d = torch.from_numpy(view["verts_2d"]).to(device, dtype=torch.float32)
        verts_3d = torch.from_numpy(view["verts_3d"]).to(device, dtype=torch.float32)
        src_image_gpu = src_image.to(device, dtype=torch.float32)

        # Per-UV-index 2D positions in source-image pixel space
        verts_2d_per_uv = verts_2d_per_3d[uv_to_vert]  # (V_uv, 2)

        # ---- Set up nvdiffrast rasterization in UV space ----
        # UV → NDC: flip V (OpenGL convention) then map [0,1] → [-1,1]
        uvs_ndc = uvs.clone()
        uvs_ndc[:, 1] = 1.0 - uvs_ndc[:, 1]
        # (V_uv, 4) homogeneous clip-space positions: (x, y, z=0, w=1)
        rast_pos = torch.cat([
            uvs_ndc * 2.0 - 1.0,
            torch.zeros_like(uvs_ndc[:, :1]),
            torch.ones_like(uvs_ndc[:, :1]),
        ], dim=-1).unsqueeze(0)  # (1, V_uv, 4)

        try:
            ctx = dr.RasterizeCudaContext(device=device)
            rast, _ = dr.rasterize(
                ctx, rast_pos, face_uvs,
                resolution=[texture_size, texture_size],
            )
            # rast: (1, H, W, 4) — last channel = triangle_id+1 (0 = empty)
            tri_mask = rast[0, ..., 3] > 0

            # Interpolate verts_2d_per_uv across the UV-rasterized mesh
            attrs_2d = verts_2d_per_uv.unsqueeze(0)  # (1, V_uv, 2)
            interp_2d, _ = dr.interpolate(attrs_2d, rast, face_uvs)
            sample_xy = interp_2d[0]  # (H, W, 2) in source-image pixel space

            # ---- Bilinear-sample the source image ----
            h_src, w_src, _ = src_image_gpu.shape
            in_bounds = (
                (sample_xy[..., 0] >= 0) & (sample_xy[..., 0] < w_src - 1) &
                (sample_xy[..., 1] >= 0) & (sample_xy[..., 1] < h_src - 1)
            )
            valid = tri_mask & in_bounds
            sample_xy_flat = sample_xy.reshape(-1, 2)
            valid_flat = valid.reshape(-1)
            sampled = _bilinear_sample(src_image_gpu, sample_xy_flat, valid_flat)
            texture = sampled.reshape(texture_size, texture_size, 3)

            # ---- Per-face confidence from view-cosine ----
            face_cos = _per_face_view_cosine(verts_3d, faces_v)  # (F,)
            # rast's triangle_id is +1; subtract 1 to get face index (-1 = miss)
            face_idx = (rast[0, ..., 3] - 1).long()
            face_idx_clamped = face_idx.clamp(min=0)
            tex_cos = face_cos[face_idx_clamped]  # (H, W)
            tex_cos = tex_cos * tri_mask.float() * in_bounds.float()

            if min_confidence > 0:
                tex_cos = torch.where(
                    tex_cos < min_confidence,
                    torch.zeros_like(tex_cos),
                    tex_cos,
                )

            # Black out texels below the confidence floor
            texture = texture * (tex_cos > 0).float().unsqueeze(-1)

        finally:
            # Free intermediate tensors
            pass

        # Output shapes: ComfyUI IMAGE = (B,H,W,3), MASK = (B,H,W) on CPU
        out_image = texture.unsqueeze(0).cpu()
        out_mask = tex_cos.unsqueeze(0).cpu()

        n_filled = int((tex_cos > 0).sum().item())
        n_total = texture_size * texture_size
        coverage = 100.0 * n_filled / n_total
        status = (
            f"baked view {view_index} ({view['view_hint']}) | "
            f"texture {texture_size}x{texture_size} | "
            f"coverage {coverage:.1f}% ({n_filled:,}/{n_total:,})"
        )
        return io.NodeOutput(out_image, out_mask, status)


FACEWRAP_TEXTURE_BAKE_V3_NODES = [BD_FaceTextureBake]

FACEWRAP_TEXTURE_BAKE_NODES = {
    "BD_FaceTextureBake": BD_FaceTextureBake,
}

FACEWRAP_TEXTURE_BAKE_DISPLAY_NAMES = {
    "BD_FaceTextureBake": "BD Face Texture Bake",
}
