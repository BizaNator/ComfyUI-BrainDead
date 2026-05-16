"""
BD_FaceTextureBake — bake source photos into the canonical UV layout
using nvdiffrast.

By default bakes EVERY detected view in the FACE_FIT and outputs batches
(uv_textures, confidences) that plug straight into BD_UVConfidenceBlend.
Set view_index >= 0 to bake just one view (for debugging / inspection).

For each output UV texel of a given view:
  1. Rasterize the canonical UV layout in NDC → per-texel (face_id, barycentric)
  2. Interpolate the 3 per-corner verts_2d positions → source-image coord
  3. Bilinear-sample the source image at that coord
  4. Confidence = front-facingness of the underlying triangle, clamped [0,1]

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


def _bake_one_view(
    ctx,
    dr,
    src_image_gpu: torch.Tensor,
    view: dict,
    uvs_ndc: torch.Tensor,
    face_uvs: torch.Tensor,
    faces_v: torch.Tensor,
    uv_to_vert: torch.Tensor,
    texture_size: int,
    min_confidence: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Bake a single view into the canonical UV layout.

    Returns (texture (H,W,3), confidence (H,W), uv_layout (H,W bool), n_filled_texels).
    uv_layout is view-independent (rasterized from UV coords).
    """
    verts_2d_per_3d = torch.from_numpy(view["verts_2d"]).to(device, dtype=torch.float32)
    verts_3d = torch.from_numpy(view["verts_3d"]).to(device, dtype=torch.float32)

    # Per-UV-index 2D positions in source-image pixel space
    verts_2d_per_uv = verts_2d_per_3d[uv_to_vert]  # (V_uv, 2)

    # (V_uv, 4) homogeneous clip-space positions: (x, y, z=0, w=1)
    rast_pos = torch.cat([
        uvs_ndc * 2.0 - 1.0,
        torch.zeros_like(uvs_ndc[:, :1]),
        torch.ones_like(uvs_ndc[:, :1]),
    ], dim=-1).unsqueeze(0)  # (1, V_uv, 4)

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

    # Bilinear-sample the source image
    h_src, w_src, _ = src_image_gpu.shape
    in_bounds = (
        (sample_xy[..., 0] >= 0) & (sample_xy[..., 0] < w_src - 1) &
        (sample_xy[..., 1] >= 0) & (sample_xy[..., 1] < h_src - 1)
    )
    valid = tri_mask & in_bounds
    sampled = _bilinear_sample(
        src_image_gpu, sample_xy.reshape(-1, 2), valid.reshape(-1)
    )
    texture = sampled.reshape(texture_size, texture_size, 3)

    # Per-face confidence from view-cosine
    face_cos = _per_face_view_cosine(verts_3d, faces_v)  # (F,)
    face_idx = (rast[0, ..., 3] - 1).long()
    tex_cos = face_cos[face_idx.clamp(min=0)]  # (H, W)
    tex_cos = tex_cos * tri_mask.float() * in_bounds.float()

    if min_confidence > 0:
        tex_cos = torch.where(tex_cos < min_confidence,
                              torch.zeros_like(tex_cos), tex_cos)

    texture = texture * (tex_cos > 0).float().unsqueeze(-1)
    n_filled = int((tex_cos > 0).sum().item())
    return texture, tex_cos, tri_mask, n_filled


class BD_FaceTextureBake(io.ComfyNode):
    """Bake source photos into the canonical UV layout — all views by default."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_FaceTextureBake",
            display_name="BD Face Texture Bake",
            category="🧠BrainDead/FaceWrap",
            description=(
                "Bake source photos into the canonical UV layout using\n"
                "nvdiffrast. By DEFAULT bakes every detected view and outputs\n"
                "(uv_textures, confidences) batches that plug straight into\n"
                "BD_UVConfidenceBlend.\n\n"
                "uv_layout_mask is view-independent: it marks every UV-atlas\n"
                "texel that maps to a mesh triangle (the full target region).\n"
                "Wire it into BD_UVConfidenceBlend.target_mask so the\n"
                "inpaint_mask covers gaps no view managed to fill.\n\n"
                "Set view_index >= 0 to bake just one view (debug/inspection);\n"
                "the output is still a 1-length batch so wiring is unchanged.\n\n"
                "Undetected views (rear / failed detection) are skipped — they\n"
                "carry no usable landmarks, so they contribute nothing to a\n"
                "face-UV bake."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Source photo batch — same order/count as the "
                            "LANDMARKS_BATCH the FACE_FIT was built from.",
                ),
                FaceFitInput(
                    "face_fit",
                    tooltip="FACE_FIT from BD_FaceFit. Provides canonical mesh "
                            "+ per-view 2D vertex positions.",
                ),
                io.Int.Input(
                    "view_index",
                    default=-1,
                    min=-1,
                    max=63,
                    step=1,
                    tooltip="-1 = bake ALL detected views (default). "
                            ">= 0 = bake only that view index.",
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
                io.Image.Output(display_name="uv_textures"),
                io.Mask.Output(display_name="confidences"),
                io.Mask.Output(display_name="uv_layout_mask"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        face_fit,
        view_index: int = -1,
        texture_size: int = 2048,
        min_confidence: float = 0.0,
    ) -> io.NodeOutput:
        empty_img = torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32)
        empty_mask = torch.zeros((1, texture_size, texture_size), dtype=torch.float32)

        try:
            import nvdiffrast.torch as dr
        except ImportError as e:
            return io.NodeOutput(empty_img, empty_mask, empty_mask, f"ERROR: nvdiffrast not installed ({e})")

        if not isinstance(face_fit, dict) or "views" not in face_fit:
            return io.NodeOutput(empty_img, empty_mask, empty_mask, "ERROR: invalid FACE_FIT")

        if images is None or images.ndim != 4 or images.shape[0] == 0:
            return io.NodeOutput(empty_img, empty_mask, empty_mask, "ERROR: invalid IMAGE input")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            return io.NodeOutput(empty_img, empty_mask, empty_mask, "ERROR: CUDA not available")

        all_views = face_fit["views"]

        # Decide which views to bake
        if view_index >= 0:
            if view_index >= len(all_views):
                return io.NodeOutput(empty_img, empty_mask, empty_mask,
                                     f"ERROR: view_index {view_index} out of range "
                                     f"(have {len(all_views)} views)")
            candidate_indices = [view_index]
        else:
            candidate_indices = list(range(len(all_views)))

        # Filter to detected views only
        bake_indices = [i for i in candidate_indices if all_views[i]["detected"]]
        skipped = [i for i in candidate_indices if not all_views[i]["detected"]]

        if not bake_indices:
            hint = (f"view {view_index}" if view_index >= 0
                    else "any view")
            return io.NodeOutput(empty_img, empty_mask, empty_mask,
                                 f"No detected views to bake ({hint} not detected).")

        # ---- Pack the shared canonical mesh into CUDA tensors once ----
        uvs = torch.from_numpy(face_fit["canonical_uvs"]).to(device, dtype=torch.float32)
        face_uvs = torch.from_numpy(face_fit["face_uvs"]).to(device, dtype=torch.int32)
        faces_v = torch.from_numpy(face_fit["faces"]).to(device, dtype=torch.int64)
        uv_to_vert = torch.from_numpy(face_fit["uv_to_vert"]).to(device, dtype=torch.int64)

        # UV → NDC: flip V (OpenGL convention) then map [0,1] → [-1,1]
        uvs_ndc = uvs.clone()
        uvs_ndc[:, 1] = 1.0 - uvs_ndc[:, 1]

        ctx = dr.RasterizeCudaContext(device=device)

        textures = []
        confidences = []
        per_view_report = []
        uv_layout = None  # view-independent; captured from first bake
        total_texels = texture_size * texture_size

        for i in bake_indices:
            view = all_views[i]
            img_idx = min(i, images.shape[0] - 1)
            src_image_gpu = images[img_idx].to(device, dtype=torch.float32)

            texture, tex_cos, tri_mask, n_filled = _bake_one_view(
                ctx, dr, src_image_gpu, view,
                uvs_ndc, face_uvs, faces_v, uv_to_vert,
                texture_size, min_confidence, device,
            )
            textures.append(texture.cpu())
            confidences.append(tex_cos.cpu())
            if uv_layout is None:
                uv_layout = tri_mask.float().cpu()
            cov = 100.0 * n_filled / total_texels
            per_view_report.append(f"{i}:{view['view_hint']}={cov:.0f}%")

        out_textures = torch.stack(textures, dim=0)      # (N, H, W, 3)
        out_confidences = torch.stack(confidences, dim=0)  # (N, H, W)
        out_uv_layout = uv_layout.unsqueeze(0)           # (1, H, W)

        mode = f"view {view_index}" if view_index >= 0 else "all views"
        status = (
            f"baked {len(bake_indices)} view(s) [{mode}] @ "
            f"{texture_size}x{texture_size} | coverage {', '.join(per_view_report)}"
        )
        if skipped:
            skipped_hints = ", ".join(
                f"{i}:{all_views[i]['view_hint']}" for i in skipped
            )
            status += f" | skipped undetected: [{skipped_hints}]"

        return io.NodeOutput(out_textures, out_confidences, out_uv_layout, status)


FACEWRAP_TEXTURE_BAKE_V3_NODES = [BD_FaceTextureBake]

FACEWRAP_TEXTURE_BAKE_NODES = {
    "BD_FaceTextureBake": BD_FaceTextureBake,
}

FACEWRAP_TEXTURE_BAKE_DISPLAY_NAMES = {
    "BD_FaceTextureBake": "BD Face Texture Bake",
}
