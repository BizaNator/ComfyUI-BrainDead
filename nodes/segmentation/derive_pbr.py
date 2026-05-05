"""
BD_DerivePBR — heuristic PBR map derivation from image + depth (+ optional normal).

Computes roughness, metallic, AO, and normal maps from a 2D image plus a depth
map (from DepthAnythingV2, Lotus, MiDaS, or whatever upstream depth estimator).
Heuristics are explicitly marked — these are GUESSES from a 2D image, not
measurements. For hero assets, hand-tune downstream; for background characters,
the heuristics are typically "good enough" once tuned.

Outputs each map separately PLUS pre-packed Unreal/Unity-style ORM/ARM maps
ready to drop into a game engine.

Heuristics used:
- Roughness: high-frequency detail (Laplacian magnitude) — busy regions are
  rougher; smooth painted areas are smoother. Optional luminance modulation.
- Metallic: low saturation + high luminance threshold (defaults to OFF since
  this is the least reliable heuristic). Most character workflows hand-paint
  metallic regions instead.
- AO: depth gradient magnitude → cavities (steep depth changes) get darker.
  Optional dilation for a softer ambient occlusion look.
- Normal: cross-product of depth's partial derivatives (Sobel) when explicit
  normal_map isn't provided. Encoded RGB tangent-space.
"""

import numpy as np
import torch

from comfy_api.latest import io


_LUMA_WEIGHTS = torch.tensor([0.299, 0.587, 0.114])


def _to_2d_float(t: torch.Tensor) -> np.ndarray:
    """Single-channel float numpy (H, W) from MASK or single-channel IMAGE."""
    arr = t.detach().cpu().float().numpy()
    if arr.ndim == 4:
        arr = arr[0]
        if arr.shape[-1] in (3, 4):
            arr = arr[..., 0]
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = arr[..., 0]
        elif arr.shape[0] == 1:
            arr = arr[0]
    return arr.astype(np.float32)


def _to_image_tensor(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return torch.from_numpy(arr.astype(np.float32)).clamp(0.0, 1.0).unsqueeze(0)


def _to_mask_tensor(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] in (3, 4) else arr[0]
    return torch.from_numpy(arr.astype(np.float32)).clamp(0.0, 1.0).unsqueeze(0)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (rgb * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(axis=-1)


def _saturation(rgb: np.ndarray) -> np.ndarray:
    cmax = rgb.max(axis=-1)
    cmin = rgb.min(axis=-1)
    return np.where(cmax > 1e-3, (cmax - cmin) / np.maximum(cmax, 1e-3), 0.0)


def _laplacian_magnitude(gray: np.ndarray, kernel: int = 3) -> np.ndarray:
    """High-frequency detail proxy. Output normalized to [0, 1]."""
    from scipy.ndimage import laplace, gaussian_filter
    smooth = gaussian_filter(gray, sigma=1.0)
    lap = np.abs(laplace(smooth))
    p99 = np.percentile(lap, 99) if lap.max() > 0 else 1.0
    return np.clip(lap / max(p99, 1e-6), 0.0, 1.0)


def _depth_gradient_magnitude(depth: np.ndarray, blur_sigma: float = 1.0) -> np.ndarray:
    """For AO: bigger depth changes = creases = stronger occlusion."""
    from scipy.ndimage import gaussian_filter, sobel
    smoothed = gaussian_filter(depth, sigma=blur_sigma)
    dx = sobel(smoothed, axis=1)
    dy = sobel(smoothed, axis=0)
    mag = np.sqrt(dx * dx + dy * dy)
    p99 = np.percentile(mag, 99) if mag.max() > 0 else 1.0
    return np.clip(mag / max(p99, 1e-6), 0.0, 1.0)


def _sharpen(arr: np.ndarray, amount: float, radius: float = 2.0) -> np.ndarray:
    """Unsharp mask. amount=0 → identity, amount=1 → strong sharpen."""
    if amount <= 0:
        return arr
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(arr, sigma=radius)
    return np.clip(arr + amount * (arr - blurred), 0.0, 1.0)


def _laplacian_edges(rgb_or_gray: np.ndarray) -> np.ndarray:
    """Edge magnitude from RGB or grayscale, normalized [0, 1]."""
    from scipy.ndimage import laplace, gaussian_filter
    if rgb_or_gray.ndim == 3:
        gray = _luminance(rgb_or_gray)
    else:
        gray = rgb_or_gray
    smooth = gaussian_filter(gray, sigma=1.0)
    lap = np.abs(laplace(smooth))
    p99 = np.percentile(lap, 99) if lap.max() > 0 else 1.0
    return np.clip(lap / max(p99, 1e-6), 0.0, 1.0)


def _normal_from_depth(depth: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Cross-product of depth partial derivatives → tangent-space normal RGB.

    Output shape (H, W, 3), values in [0, 1] (encoded normal: (n+1)/2).
    """
    from scipy.ndimage import sobel
    dz_dx = sobel(depth, axis=1) * strength
    dz_dy = sobel(depth, axis=0) * strength
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(depth) * 0.5
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    norm = np.maximum(norm, 1e-6)
    nx /= norm
    ny /= norm
    nz /= norm
    encoded = np.stack([nx, ny, nz], axis=-1) * 0.5 + 0.5
    return np.clip(encoded, 0.0, 1.0)


def _metallic_heuristic(rgb: np.ndarray, lum_threshold: float, sat_threshold: float,
                       softness: float) -> np.ndarray:
    """Low-saturation + high-luminance pixels are *candidate* metal. Heuristic only."""
    lum = _luminance(rgb)
    sat = _saturation(rgb)
    lum_score = np.clip((lum - lum_threshold) / max(softness, 0.01), 0.0, 1.0)
    sat_score = np.clip((sat_threshold - sat) / max(softness, 0.01), 0.0, 1.0)
    return lum_score * sat_score


class BD_DerivePBR(io.ComfyNode):
    """Derive PBR maps (roughness, metallic, AO, normal) from image + depth via heuristics."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_DerivePBR",
            display_name="BD Derive PBR Maps",
            category="🧠BrainDead/Segmentation",
            description=(
                "Heuristic PBR map derivation from a 2D image + depth map. Outputs roughness, "
                "metallic, AO, normal as separate maps + game-engine-ready packed ORM/ARM maps. "
                "These are HEURISTICS not measurements — for hero assets hand-tune downstream; "
                "for background characters, tuning the parameters is usually 'good enough'. Pair "
                "with an existing depth estimator (DepthAnythingV2, Lotus, MiDaS) upstream."
            ),
            inputs=[
                io.Image.Input("image",
                               tooltip="Source character/object image. Used for roughness (high-freq detail) "
                                       "and metallic (low-sat / high-lum heuristic) and as albedo passthrough."),
                io.Image.Input("depth",
                               tooltip="Depth map IMAGE from upstream estimator (DepthAnythingV2 / Lotus / MiDaS / "
                                       "Metric3D — all output IMAGE type). Closer surfaces darker by convention. "
                                       "Used for AO (cavity detection) and normal (when no explicit normal_map provided)."),
                io.Image.Input("normal_map", optional=True,
                               tooltip="Optional explicit normal map IMAGE (e.g. from Metric3D-NormalMapPreprocessor or "
                                       "BAE-NormalMapPreprocessor). When provided, used directly. When NOT provided, "
                                       "normals are derived from the depth map's gradients."),
                io.Mask.Input("silhouette_mask", optional=True,
                              tooltip="Optional character/object silhouette MASK (white = subject, black = background). "
                                      "Multiplies all output maps so background pixels are zeroed in roughness/metallic, "
                                      "set to 1.0 in AO (no occlusion outside subject), and set to neutral (0.5, 0.5, 1.0) "
                                      "in normal map. Source: BiRefNetRMBG, BodySegment, or your existing skin/body chain."),
                io.Mask.Input("aux_shading_alpha", optional=True,
                              tooltip="Optional shading alpha map — e.g. the alpha output of your skin_details "
                                      "GLSL shader, where low alpha = highlights, high alpha = shaded zones. "
                                      "When provided: shaded areas get +roughness +AO -metallic; highlights get the "
                                      "opposite. This is FAR more accurate than the image-only heuristic for stylized art."),
                io.Image.Input("aux_detail_texture", optional=True,
                              tooltip="Optional surface detail texture (lineart, manga lines, surface bumps). When "
                                      "provided, edge-detected lines boost roughness — useful for stylized character "
                                      "art where line detail represents real surface variation."),
                io.Boolean.Input("invert_shading_alpha", default=False, optional=True,
                                 tooltip="Set True if your shading alpha uses the opposite convention "
                                         "(low alpha = shaded, high alpha = highlight)."),
                io.Float.Input("shading_to_roughness", default=0.4, min=0.0, max=2.0, step=0.05, optional=True,
                               tooltip="How much shaded areas (high alpha) add to roughness. 0 = ignore shading for "
                                       "roughness; 0.4 = moderate boost; 1.0+ = strong (shaded → fully rough)."),
                io.Float.Input("shading_to_ao", default=0.5, min=0.0, max=2.0, step=0.05, optional=True,
                               tooltip="How much shading darkens AO (multiplicatively combined with depth-based AO). "
                                       "0 = ignore; 0.5 = moderate; 1.0 = full (deeply shaded → near-black AO)."),
                io.Float.Input("detail_to_roughness", default=0.5, min=0.0, max=2.0, step=0.05, optional=True,
                               tooltip="How much detail texture lines boost roughness. Lines = surface detail = "
                                       "rougher. 0 = ignore; 0.5 = moderate; >1 = strong line emphasis."),
                io.Float.Input("depth_sharpen", default=0.0, min=0.0, max=3.0, step=0.1, optional=True,
                               tooltip="Unsharp mask the depth map before AO/normal derivation. Useful when your "
                                       "depth estimator (Lotus, Metric3D) gives smooth/chunky depth. 0 = no sharpen, "
                                       "0.5-1.0 = moderate edge enhancement, 2-3 = aggressive."),

                io.Combo.Input("roughness_mode",
                               options=["high_freq_detail", "luminance_inverted", "combined", "off"],
                               default="combined", optional=True,
                               tooltip="high_freq_detail: Laplacian magnitude → busy areas rougher (good for "
                                       "stylized art).\nluminance_inverted: bright = smooth, dark = rough (works "
                                       "for photo-lit subjects).\ncombined: average of both.\noff: skip (output "
                                       "uses roughness_default)."),
                io.Float.Input("roughness_default", default=0.5, min=0.0, max=1.0, step=0.05, optional=True,
                               tooltip="Used when roughness_mode=off, or as a baseline blended with the heuristic."),
                io.Float.Input("roughness_strength", default=1.0, min=0.0, max=2.0, step=0.05, optional=True,
                               tooltip="Multiplier on the computed roughness map. <1 makes everything smoother, "
                                       ">1 boosts roughness contrast."),

                io.Combo.Input("metallic_mode",
                               options=["off", "low_sat_high_lum"],
                               default="off", optional=True,
                               tooltip="off: outputs metallic_default everywhere (typically 0). "
                                       "low_sat_high_lum: detect metallic candidates by bright + desaturated pixels. "
                                       "STRONGLY recommend wiring metallic_zone_mask too — heuristic alone gets false "
                                       "positives on bright skin/white cloth. With zone_mask=clothing+accessories, "
                                       "the heuristic becomes useful (bright desaturated pixels INSIDE clothing/accessories "
                                       "= metal candidates)."),
                io.Mask.Input("metallic_zone_mask", optional=True,
                              tooltip="Optional MASK restricting where metallic CAN occur. Multiplies the metallic "
                                      "output, so only pixels inside the zone can be metal. Wire your clothing or "
                                      "accessories mask here (from a SAM3 'accessories,jewelry,metal' prompt, "
                                      "ClothesSegment, MaskResolver, etc.) "
                                      "— most game-character metal lives on jewelry/buckles/armor/details which align "
                                      "with the accessories category. Without this, metallic detection fires everywhere "
                                      "and produces false positives."),
                io.Float.Input("metallic_default", default=0.0, min=0.0, max=1.0, step=0.05, optional=True,
                               tooltip="Constant metallic value used everywhere when metallic_mode=off."),
                io.Float.Input("metallic_lum_threshold", default=0.6, min=0.0, max=1.0, step=0.05, optional=True,
                               tooltip="Luminance above this counts toward metallic (when metallic_mode=low_sat_high_lum)."),
                io.Float.Input("metallic_sat_threshold", default=0.2, min=0.0, max=1.0, step=0.05, optional=True,
                               tooltip="Saturation below this counts toward metallic (when metallic_mode=low_sat_high_lum)."),

                io.Float.Input("ao_strength", default=0.7, min=0.0, max=2.0, step=0.05, optional=True,
                               tooltip="How much the depth gradients affect AO. 0 = pure white AO (no occlusion), "
                                       "1 = full strength, >1 = exaggerated. AO is OUTPUT WHITE = no occlusion, "
                                       "DARK = occluded (Unreal/Unity convention)."),
                io.Float.Input("ao_blur", default=2.0, min=0.0, max=20.0, step=0.5, optional=True,
                               tooltip="Gaussian blur radius for depth before computing gradients. Higher = softer AO."),

                io.Float.Input("normal_strength", default=2.0, min=0.1, max=10.0, step=0.1, optional=True,
                               tooltip="Strength multiplier when deriving normals from depth. Higher = more "
                                       "pronounced normals. Only used when normal_map is NOT provided."),
                io.Combo.Input("albedo_treatment",
                               options=["edge_pad", "silhouette_clip", "passthrough"],
                               default="edge_pad", optional=True,
                               tooltip="How to clean up the albedo output (was passing input through raw, which "
                                       "leaked transparent-area garbage colors as halo).\n"
                                       "  edge_pad — extend foreground colors outward into background using "
                                       "Voronoi nearest-neighbor (game-engine standard 'alpha bleed'). Prevents UV "
                                       "edge halos. Requires silhouette_mask.\n"
                                       "  silhouette_clip — multiply by silhouette_mask (background goes to black). "
                                       "Quick but causes black halos at UV edges in-engine.\n"
                                       "  passthrough — original behavior (use input RGB as-is)."),
                io.Int.Input("albedo_edge_pad_pixels", default=32, min=0, max=512, step=4, optional=True,
                             tooltip="When albedo_treatment=edge_pad: how far to extend foreground colors into "
                                     "transparent area. 0 = unlimited (pad everywhere — slow on huge images), "
                                     "16-32 = typical for game textures, 64+ = aggressive."),
            ],
            outputs=[
                io.Image.Output(display_name="albedo"),
                io.Image.Output(display_name="normal"),
                io.Mask.Output(display_name="roughness"),
                io.Mask.Output(display_name="metallic"),
                io.Mask.Output(display_name="ao"),
                io.Image.Output(display_name="packed_orm"),
                io.Image.Output(display_name="packed_arm"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, image, depth, normal_map=None, silhouette_mask=None,
                aux_shading_alpha=None, aux_detail_texture=None,
                invert_shading_alpha=False,
                shading_to_roughness=0.4, shading_to_ao=0.5, detail_to_roughness=0.5,
                depth_sharpen=0.0,
                roughness_mode="combined", roughness_default=0.5, roughness_strength=1.0,
                metallic_mode="off", metallic_zone_mask=None, metallic_default=0.0,
                metallic_lum_threshold=0.6, metallic_sat_threshold=0.2,
                ao_strength=0.7, ao_blur=2.0,
                normal_strength=2.0,
                albedo_treatment="edge_pad", albedo_edge_pad_pixels=32) -> io.NodeOutput:
        img = image if image.ndim == 4 else image.unsqueeze(0)
        b, h, w, _ = img.shape
        rgb = img[0, ..., :3].detach().cpu().float().numpy()

        depth_arr = _to_2d_float(depth)
        if depth_arr.shape != (h, w):
            from PIL import Image as PILImage
            depth_arr = np.asarray(
                PILImage.fromarray((depth_arr * 255).clip(0, 255).astype(np.uint8), mode="L")
                .resize((w, h), PILImage.BILINEAR)
            ).astype(np.float32) / 255.0

        if depth_sharpen > 0:
            depth_arr = _sharpen(depth_arr, amount=float(depth_sharpen), radius=2.0)

        shading_arr = None
        if aux_shading_alpha is not None:
            shading_arr = _to_2d_float(aux_shading_alpha)
            if shading_arr.shape != (h, w):
                from PIL import Image as PILImage
                shading_arr = np.asarray(
                    PILImage.fromarray((shading_arr * 255).clip(0, 255).astype(np.uint8), mode="L")
                    .resize((w, h), PILImage.BILINEAR)
                ).astype(np.float32) / 255.0
            if invert_shading_alpha:
                shading_arr = 1.0 - shading_arr

        detail_edges = None
        if aux_detail_texture is not None:
            detail_t = aux_detail_texture if aux_detail_texture.ndim == 4 else aux_detail_texture.unsqueeze(0)
            detail_arr = detail_t[0, ..., :3].detach().cpu().float().numpy()
            if detail_arr.shape[:2] != (h, w):
                from PIL import Image as PILImage
                detail_arr = np.asarray(
                    PILImage.fromarray((detail_arr * 255).clip(0, 255).astype(np.uint8))
                    .resize((w, h), PILImage.BILINEAR)
                ).astype(np.float32) / 255.0
            detail_edges = _laplacian_edges(detail_arr)

        if normal_map is not None:
            normal_arr = normal_map if normal_map.ndim == 4 else normal_map.unsqueeze(0)
            normal_arr = normal_arr[0, ..., :3].detach().cpu().float().numpy()
            if normal_arr.shape[:2] != (h, w):
                from PIL import Image as PILImage
                normal_arr = np.asarray(
                    PILImage.fromarray((normal_arr * 255).clip(0, 255).astype(np.uint8))
                    .resize((w, h), PILImage.BILINEAR)
                ).astype(np.float32) / 255.0
        else:
            normal_arr = _normal_from_depth(depth_arr, strength=float(normal_strength))

        if roughness_mode == "off":
            roughness = np.full((h, w), float(roughness_default), dtype=np.float32)
        else:
            gray = _luminance(rgb)
            r_high = _laplacian_magnitude(gray)
            r_lum = 1.0 - gray
            if roughness_mode == "high_freq_detail":
                rough_raw = r_high
            elif roughness_mode == "luminance_inverted":
                rough_raw = r_lum
            else:
                rough_raw = (r_high + r_lum) * 0.5
            roughness = float(roughness_default) + rough_raw * float(roughness_strength)

        if shading_arr is not None and shading_to_roughness > 0:
            roughness = roughness + shading_arr * float(shading_to_roughness)
        if detail_edges is not None and detail_to_roughness > 0:
            roughness = roughness + detail_edges * float(detail_to_roughness)
        roughness = np.clip(roughness, 0.0, 1.0)

        if metallic_mode == "off":
            metallic = np.full((h, w), float(metallic_default), dtype=np.float32)
        else:
            metallic = _metallic_heuristic(
                rgb, float(metallic_lum_threshold), float(metallic_sat_threshold), softness=0.1
            )
            if shading_arr is not None:
                metallic = metallic * (1.0 - shading_arr)
        if metallic_zone_mask is not None:
            mz_arr = _to_2d_float(metallic_zone_mask)
            if mz_arr.shape != (h, w):
                from PIL import Image as PILImage
                mz_arr = np.asarray(
                    PILImage.fromarray((mz_arr * 255).clip(0, 255).astype(np.uint8), mode="L")
                    .resize((w, h), PILImage.BILINEAR)
                ).astype(np.float32) / 255.0
            metallic = metallic * mz_arr

        ao_depth_raw = _depth_gradient_magnitude(depth_arr, blur_sigma=float(ao_blur))
        ao = np.clip(1.0 - ao_depth_raw * float(ao_strength), 0.0, 1.0)
        if shading_arr is not None and shading_to_ao > 0:
            ao_shading = np.clip(1.0 - shading_arr * float(shading_to_ao), 0.0, 1.0)
            ao = ao * ao_shading

        if silhouette_mask is not None:
            sil_arr = _to_2d_float(silhouette_mask)
            if sil_arr.shape != (h, w):
                from PIL import Image as PILImage
                sil_arr = np.asarray(
                    PILImage.fromarray((sil_arr * 255).clip(0, 255).astype(np.uint8), mode="L")
                    .resize((w, h), PILImage.NEAREST)
                ).astype(np.float32) / 255.0
            roughness = roughness * sil_arr
            metallic = metallic * sil_arr
            ao = np.where(sil_arr > 0.01, ao, 1.0)
            mask3 = np.stack([sil_arr] * 3, axis=-1)
            normal_arr = normal_arr * mask3 + np.array([0.5, 0.5, 1.0]) * (1.0 - mask3)

        packed_orm = np.stack([ao, roughness, metallic], axis=-1)
        packed_arm = np.stack([ao, roughness, metallic], axis=-1)

        albedo_arr = rgb.copy()
        albedo_treatment_str = albedo_treatment
        if silhouette_mask is not None:
            sil_for_albedo = _to_2d_float(silhouette_mask)
            if sil_for_albedo.shape != (h, w):
                from PIL import Image as PILImage
                sil_for_albedo = np.asarray(
                    PILImage.fromarray((sil_for_albedo * 255).clip(0, 255).astype(np.uint8), mode="L")
                    .resize((w, h), PILImage.NEAREST)
                ).astype(np.float32) / 255.0
            if albedo_treatment == "edge_pad" and sil_for_albedo.max() > 0.1:
                from scipy.ndimage import distance_transform_edt
                fg = sil_for_albedo > 0.5
                if fg.any():
                    distances, indices = distance_transform_edt(~fg, return_indices=True)
                    padded = albedo_arr[indices[0], indices[1]]
                    if albedo_edge_pad_pixels > 0:
                        within = distances <= albedo_edge_pad_pixels
                        albedo_arr = np.where(within[..., None], padded, albedo_arr)
                    else:
                        albedo_arr = padded
                albedo_treatment_str = f"edge_pad ({albedo_edge_pad_pixels}px)"
            elif albedo_treatment == "silhouette_clip":
                albedo_arr = albedo_arr * sil_for_albedo[..., None]
                albedo_treatment_str = "silhouette_clip"
        else:
            albedo_treatment_str = "passthrough (no silhouette_mask)"

        aux_str = []
        if shading_arr is not None:
            aux_str.append(f"shading_alpha (mean={float(shading_arr.mean()):.3f})")
        if detail_edges is not None:
            aux_str.append(f"detail_texture (mean={float(detail_edges.mean()):.3f})")
        if depth_sharpen > 0:
            aux_str.append(f"depth_sharpen={depth_sharpen:.1f}")
        aux_summary = (" | aux: " + ", ".join(aux_str)) if aux_str else ""

        zone_str = " (zone-restricted)" if metallic_zone_mask is not None else ""
        status = (
            f"roughness mode={roughness_mode} default={roughness_default:.2f} strength={roughness_strength:.2f} "
            f"mean={float(roughness.mean()):.3f}\n"
            f"metallic mode={metallic_mode}{zone_str} mean={float(metallic.mean()):.3f}\n"
            f"ao depth_strength={ao_strength:.2f} blur={ao_blur:.1f} mean={float(ao.mean()):.3f}\n"
            f"normal source={'explicit' if normal_map is not None else f'depth (strength={normal_strength:.2f})'}\n"
            f"albedo treatment={albedo_treatment_str}"
            f"{aux_summary}"
        )
        print(f"[BD DerivePBR] {status}", flush=True)

        return io.NodeOutput(
            _to_image_tensor(albedo_arr),
            _to_image_tensor(normal_arr),
            _to_mask_tensor(roughness),
            _to_mask_tensor(metallic),
            _to_mask_tensor(ao),
            _to_image_tensor(packed_orm),
            _to_image_tensor(packed_arm),
            status,
        )


DERIVE_PBR_V3_NODES = [BD_DerivePBR]
DERIVE_PBR_NODES = {"BD_DerivePBR": BD_DerivePBR}
DERIVE_PBR_DISPLAY_NAMES = {"BD_DerivePBR": "BD Derive PBR Maps"}
