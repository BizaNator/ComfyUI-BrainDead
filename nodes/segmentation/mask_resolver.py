"""
BD_MaskResolver — priority-based mask separation.

Python port of the user's GLSL "Mask Resolver" shader. Takes overlapping
category masks (skin / clothes / accessories) plus a silhouette and the
original image, and outputs clean mutually-exclusive masks that sum to
(roughly) the silhouette.

Algorithm (per pixel inside silhouette):
  1. Threshold each input mask to clean up gray noise.
  2. Clamp each category to the silhouette.
  3. Score each category = mask × priority × color_modulation.
       Color modulation: skin-toned pixels boost skin score / penalize clothes;
       non-skin colors boost clothes/accessories / penalize skin.
  4. Winner-takes-all (or soft-blend) at each pixel.
  5. Pixels in silhouette that no category claimed → "residual / gap":
       fill from nearest neighbors (and color hints) so the silhouette is fully covered.

The output masks are guaranteed non-overlapping. The shader's debug overlay
output is preserved as `debug_overlay` IMAGE.
"""

import numpy as np
import torch

from comfy_api.latest import io


def _to_2d_uint8(mask: torch.Tensor) -> np.ndarray:
    arr = mask.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB → CIE LAB (D65). Input in [0,1], returns L in [0,100], a/b in ~[-128,127]."""
    rgb_lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = rgb_lin.reshape(-1, 3) @ M.T
    xyz_n = xyz / np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    f = np.where(xyz_n > 0.008856, np.cbrt(np.clip(xyz_n, 0, None)), 7.787 * xyz_n + 16.0 / 116.0)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=-1).reshape(rgb.shape)


def _adaptive_skin_likelihood(rgb: np.ndarray, skin_mask_binary: np.ndarray,
                              tolerance: float = 25.0, min_samples: int = 50) -> tuple[np.ndarray, dict]:
    """Sample reference skin from skin_mask, return per-pixel LAB-distance likelihood.

    Returns (likelihood (H,W) in [0,1], debug_info dict with sampled ref + count).
    Likelihood = 1 / (1 + (deltaE / tolerance)^2) — Lorentzian falloff so a pixel
    AT tolerance distance scores 0.5, double tolerance scores 0.2, etc.
    """
    debug = {"ref_lab": None, "ref_rgb": None, "sample_count": 0, "skipped": True}
    skin_pixels = rgb[skin_mask_binary]
    if len(skin_pixels) < min_samples:
        return np.zeros(rgb.shape[:2], dtype=np.float32), debug

    skin_lab = _rgb_to_lab(skin_pixels)
    img_lab = _rgb_to_lab(rgb)

    ref_lab = np.median(skin_lab, axis=0)
    delta = img_lab - ref_lab[None, None, :]
    distance = np.sqrt(np.sum(delta * delta, axis=-1))
    likelihood = 1.0 / (1.0 + (distance / max(tolerance, 1e-3)) ** 2)

    debug["ref_lab"] = ref_lab.tolist()
    debug["ref_rgb"] = (np.median(skin_pixels, axis=0) * 255).astype(int).tolist()
    debug["sample_count"] = int(len(skin_pixels))
    debug["skipped"] = False
    return likelihood.astype(np.float32), debug


def _skin_tone_likelihood(rgb: np.ndarray) -> np.ndarray:
    """Verbatim port of the GLSL `skinToneLikelihood` function. rgb in [0,1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    val = cmax
    sat = np.where(cmax > 1e-3, delta / np.maximum(cmax, 1e-3), 0.0)

    hue = np.zeros_like(r)
    safe_delta = np.maximum(delta, 1e-6)
    mask_r = (cmax == r) & (delta > 1e-3)
    mask_g = (cmax == g) & (delta > 1e-3) & (~mask_r)
    mask_b = (cmax == b) & (delta > 1e-3) & (~mask_r) & (~mask_g)
    hue[mask_r] = (g[mask_r] - b[mask_r]) / safe_delta[mask_r]
    hue[mask_g] = 2.0 + (b[mask_g] - r[mask_g]) / safe_delta[mask_g]
    hue[mask_b] = 4.0 + (r[mask_b] - g[mask_b]) / safe_delta[mask_b]
    hue = np.mod(hue / 6.0, 1.0)

    hue_score = _smoothstep(0.14, 0.10, hue) + _smoothstep(0.92, 0.96, hue)
    hue_score = np.clip(hue_score, 0.0, 1.0)
    sat_score = _smoothstep(0.05, 0.15, sat) * _smoothstep(0.85, 0.65, sat)
    val_score = _smoothstep(0.12, 0.25, val) * _smoothstep(0.98, 0.88, val)
    rgb_order = ((b <= g) & (g <= r)).astype(np.float32) * 0.3
    return np.clip(hue_score * sat_score * val_score + rgb_order, 0.0, 1.0)


def _clothing_likelihood(rgb: np.ndarray, skin: np.ndarray) -> np.ndarray:
    cmax = np.maximum(np.maximum(rgb[..., 0], rgb[..., 1]), rgb[..., 2])
    cmin = np.minimum(np.minimum(rgb[..., 0], rgb[..., 1]), rgb[..., 2])
    sat = np.where(cmax > 1e-3, (cmax - cmin) / np.maximum(cmax, 1e-3), 0.0)
    darkness = _smoothstep(0.2, 0.05, cmax)
    saturated_non_skin = sat * (1.0 - skin)
    return np.clip((1.0 - skin) * 0.5 + darkness * 0.3 + saturated_non_skin * 0.4, 0.0, 1.0)


def _to_mask_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)


def _to_image_tensor(arr_uint8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr_uint8.astype(np.float32) / 255.0).unsqueeze(0)


class BD_MaskResolver(io.ComfyNode):
    """Priority-based mask separation. Takes overlapping skin/clothes/accessories masks → clean mutually-exclusive outputs."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_MaskResolver",
            display_name="BD Mask Resolver",
            category="🧠BrainDead/Segmentation",
            description=(
                "Priority-based mask separation. Given a silhouette plus overlapping category masks "
                "(skin, clothes, accessories) and the original image, produces clean mutually-exclusive "
                "masks via priority + HSV color voting + neighbor-aware gap-fill. Python port of the "
                "Mask Resolver GLSL shader."
            ),
            inputs=[
                io.Mask.Input("silhouette",
                              tooltip="Full character outline mask (white = character body, black = background). Defines the universe of pixels to assign to a category. Anything outside silhouette becomes background in all outputs."),
                io.Mask.Input("skin_mask",
                              tooltip="Mask of pixels you believe are skin (e.g. from BD_SAM3MultiPrompt with positive skin prompts). Will be clamped to silhouette."),
                io.Mask.Input("clothes_mask",
                              tooltip="Mask of pixels you believe are clothing. Will be clamped to silhouette."),
                io.Image.Input("original_image",
                               tooltip="Original character image. Sampled per-pixel for HSV-based skin-tone detection (see color_weight). Resized to silhouette resolution if mismatched."),
                io.Mask.Input("accessories_mask", optional=True,
                              tooltip="Optional mask for hard accessories (gloves, boots, belts, weapons). Defaults to all-zero if not provided. Highest default priority — wins overlaps with skin/clothes."),

                io.Float.Input("silhouette_threshold", default=0.4, min=0.0, max=1.0, step=0.05, optional=True,
                               tooltip="Noise floor on the silhouette mask. Pixels in silhouette ABOVE this value count as character body, below count as background. Higher = stricter silhouette interpretation. NOT include/exclude — it's a noise filter."),
                io.Float.Input("skin_threshold", default=0.5, min=0.0, max=1.0, step=0.05, optional=True,
                               tooltip="Noise floor on the skin_mask input. Pixels above count as 'skin candidate', below are zeroed. Higher = only confidently-detected skin counts. Lower = pick up faint partial detections (and noise)."),
                io.Float.Input("clothes_threshold", default=0.5, min=0.0, max=1.0, step=0.05, optional=True,
                               tooltip="Noise floor on the clothes_mask input. Same semantics as skin_threshold."),
                io.Float.Input("accessories_threshold", default=0.5, min=0.0, max=1.0, step=0.05, optional=True,
                               tooltip="Noise floor on the accessories_mask input. Same semantics as skin_threshold."),

                io.Float.Input("skin_priority", default=1.0, min=0.1, max=10.0, step=0.1, optional=True,
                               tooltip="Priority weight for skin when multiple masks claim a pixel. Default ranking: accessories(3) > clothes(2) > skin(1) — meaning a pixel claimed by both clothes and skin goes to clothes. Raise skin_priority above 2.0 to flip it."),
                io.Float.Input("clothes_priority", default=2.0, min=0.1, max=10.0, step=0.1, optional=True,
                               tooltip="Priority weight for clothes. See skin_priority for the ranking convention."),
                io.Float.Input("accessories_priority", default=3.0, min=0.1, max=10.0, step=0.1, optional=True,
                               tooltip="Priority weight for accessories. See skin_priority for the ranking convention."),

                io.Float.Input("edge_softness", default=0.3, min=0.0, max=1.0, step=0.05, optional=True,
                               tooltip="Smoothstep width around each input threshold. 0 = hard binary thresholds (sharp edges), higher = anti-aliased gradient at mask boundaries."),
                io.Float.Input("claim_threshold", default=0.15, min=0.0, max=2.0, step=0.05, optional=True,
                               tooltip="A pixel is 'claimed' by the winning category only if the winning score exceeds this. Pixels with weak/uncertain scores below claim_threshold become residual and are eligible for gap_fill. Raise this (e.g. 0.5+) to make residual generous (more gaps to fill); lower it to make claim aggressive (almost no residual). Default 0.15 = mild claim."),
                io.Float.Input("gap_fill_strength", default=0.8, min=0.0, max=1.0, step=0.05, optional=True,
                               tooltip="How aggressively to assign residual pixels via neighbor voting. 0 = leave residual untouched, 1 = fully assign residual to the strongest neighbor category. Only has an effect when residual is non-empty (controlled by claim_threshold)."),
                io.Int.Input("gap_neighbor_radius", default=8, min=1, max=64, step=1, optional=True,
                               tooltip="Radius (in pixels) of the neighborhood used to vote on residual pixels. Higher = looks further to find dominant category. Default 8 covers small gaps; raise to 16-32 for character art at high res."),
                io.Float.Input("color_weight", default=0.6, min=0.0, max=1.0, step=0.05, optional=True,
                               tooltip="How much skin-tone detection influences scoring. 0 = ignore image color, only use input masks + priorities. 1 = strong color influence (skin pixels boosted toward skin, non-skin pixels boosted toward clothes)."),
                io.Combo.Input("skin_color_mode", options=["adaptive_lab", "fixed_hsv", "both"],
                               default="adaptive_lab", optional=True,
                               tooltip="How to detect skin-toned pixels in the original image:\n"
                                       "  adaptive_lab — sample reference skin color from confident skin_mask pixels, "
                                       "compute LAB ΔE distance per pixel. Self-tunes per character (works on pale "
                                       "white through dark African through reddish Navajo skin).\n"
                                       "  fixed_hsv — hard-coded HSV ranges from the original GLSL shader. Fails on "
                                       "very pale/dark skin. Fallback for when adaptive can't sample enough pixels.\n"
                                       "  both — multiply the two likelihoods. Strictest mode."),
                io.Float.Input("adaptive_tolerance", default=25.0, min=2.0, max=100.0, step=1.0, optional=True,
                               tooltip="LAB ΔE distance at which adaptive likelihood = 0.5. Smaller = stricter "
                                       "(only pixels very close to sampled skin tone count). Larger = more permissive. "
                                       "ΔE 2.3 = just-noticeable difference, 25 = clearly different but related "
                                       "(typical skin variation), 50+ = very loose."),
                io.Float.Input("adaptive_sample_threshold", default=0.7, min=0.0, max=1.0, step=0.05, optional=True,
                               tooltip="Only skin_mask pixels above this value are used as reference samples for adaptive mode. "
                                       "Higher = use only the most confident skin pixels (better reference, fewer samples). "
                                       "Lower = use more pixels including soft edges (more samples, may include non-skin)."),
                io.Int.Input("adaptive_min_samples", default=50, min=10, max=10000, step=10, optional=True,
                               tooltip="Minimum number of confident skin pixels needed for adaptive mode to activate. "
                                       "If fewer, adaptive returns zeros and falls back to fixed_hsv (in 'both' mode) or off."),

                io.Combo.Input("overlap_mode", options=["priority", "soft_blend"], default="priority", optional=True,
                               tooltip="priority = winner takes all (each pixel goes to exactly ONE category). soft_blend = distribute proportionally by score (each pixel split fractionally; outputs may sum to >100% in overlap zones)."),
                io.Boolean.Input("output_debug_viz", default=False, optional=True,
                                 tooltip="If True, debug_overlay shows color-coded overlap map: green=skin, blue=clothes, red=access, yellow=skin+clothes, magenta=skin+access, cyan=clothes+access, white=triple, gray=residual gap. Tinted by skin-tone confidence. If False, debug_overlay just shows the residual mask as RGB grayscale."),
            ],
            outputs=[
                io.Mask.Output(display_name="clean_skin"),
                io.Mask.Output(display_name="clean_clothes"),
                io.Mask.Output(display_name="clean_accessories"),
                io.Mask.Output(display_name="residual"),
                io.Image.Output(display_name="debug_overlay"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, silhouette, skin_mask, clothes_mask, original_image,
                accessories_mask=None,
                silhouette_threshold=0.4, skin_threshold=0.5,
                clothes_threshold=0.5, accessories_threshold=0.5,
                skin_priority=1.0, clothes_priority=2.0, accessories_priority=3.0,
                edge_softness=0.3, claim_threshold=0.15,
                gap_fill_strength=0.8, gap_neighbor_radius=8, color_weight=0.6,
                skin_color_mode="adaptive_lab", adaptive_tolerance=25.0,
                adaptive_sample_threshold=0.7, adaptive_min_samples=50,
                overlap_mode="priority", output_debug_viz=False) -> io.NodeOutput:

        sil_u8 = _to_2d_uint8(silhouette)
        skin_u8 = _to_2d_uint8(skin_mask)
        cloth_u8 = _to_2d_uint8(clothes_mask)
        h, w = sil_u8.shape

        sil_mean_in = sil_u8.mean() / 255.0
        sil_max_in = sil_u8.max() / 255.0
        silhouette_blank = sil_mean_in < 0.005 or sil_max_in < 0.1
        if silhouette_blank:
            print(
                f"[BD MaskResolver] WARNING: silhouette appears blank (mean={sil_mean_in:.4f}, "
                f"max={sil_max_in:.3f}). Treating entire image as silhouette to avoid empty output.",
                flush=True,
            )
            sil_u8 = np.full_like(sil_u8, 255)

        # Resize all masks to silhouette resolution
        from PIL import Image as PILImage
        def _match(arr):
            if arr.shape != (h, w):
                return np.asarray(PILImage.fromarray(arr, mode="L").resize((w, h), PILImage.NEAREST))
            return arr
        skin_u8 = _match(skin_u8)
        cloth_u8 = _match(cloth_u8)
        if accessories_mask is not None:
            access_u8 = _match(_to_2d_uint8(accessories_mask))
        else:
            access_u8 = np.zeros_like(sil_u8)

        # Original image to (H, W, 3) numpy in [0, 1]
        img_t = original_image
        if img_t.ndim == 4:
            img_t = img_t[0]
        img_np = img_t.detach().cpu().numpy()
        if img_np.shape[:2] != (h, w):
            img_np = np.asarray(
                PILImage.fromarray((img_np[..., :3] * 255).clip(0, 255).astype(np.uint8))
                .resize((w, h), PILImage.LANCZOS)
            ).astype(np.float32) / 255.0

        # Threshold inputs (with soft edges via smoothstep)
        edge_soft = max(edge_softness, 1e-3)
        sil = _smoothstep(silhouette_threshold - edge_soft, silhouette_threshold + edge_soft, sil_u8 / 255.0)
        skin = _smoothstep(skin_threshold - edge_soft, skin_threshold + edge_soft, skin_u8 / 255.0)
        cloth = _smoothstep(clothes_threshold - edge_soft, clothes_threshold + edge_soft, cloth_u8 / 255.0)
        access = _smoothstep(accessories_threshold - edge_soft, accessories_threshold + edge_soft, access_u8 / 255.0)

        # Clamp to silhouette
        in_sil = sil >= 0.01
        skin = np.minimum(skin, sil)
        cloth = np.minimum(cloth, sil)
        access = np.minimum(access, sil)

        # Color confidence — adaptive (sampled from skin_mask) vs fixed HSV ranges.
        adaptive_debug = {}
        if skin_color_mode in ("adaptive_lab", "both"):
            skin_ref_binary = (skin_u8 / 255.0) >= adaptive_sample_threshold
            skin_color_conf_adaptive, adaptive_debug = _adaptive_skin_likelihood(
                img_np[..., :3], skin_ref_binary,
                tolerance=adaptive_tolerance, min_samples=adaptive_min_samples,
            )
        else:
            skin_color_conf_adaptive = None

        if skin_color_mode == "fixed_hsv" or (skin_color_mode == "adaptive_lab" and adaptive_debug.get("skipped")):
            skin_color_conf = _skin_tone_likelihood(img_np[..., :3])
        elif skin_color_mode == "adaptive_lab":
            skin_color_conf = skin_color_conf_adaptive
        elif skin_color_mode == "both":
            fixed = _skin_tone_likelihood(img_np[..., :3])
            if adaptive_debug.get("skipped"):
                skin_color_conf = fixed
            else:
                skin_color_conf = fixed * skin_color_conf_adaptive
        else:
            skin_color_conf = _skin_tone_likelihood(img_np[..., :3])

        cloth_color_conf = _clothing_likelihood(img_np[..., :3], skin_color_conf)

        # Priority + color modulation
        cw = float(np.clip(color_weight, 0.0, 1.0))
        skin_boost = 1.0 + cw * skin_color_conf
        cloth_boost = 1.0 + cw * cloth_color_conf
        skin_penalty = 1.0 - cw * cloth_color_conf * 0.5
        cloth_penalty = 1.0 - cw * skin_color_conf * 0.5

        skin_score = skin * skin_priority * skin_boost * skin_penalty
        cloth_score = cloth * clothes_priority * cloth_boost * cloth_penalty
        access_score = access * accessories_priority * cloth_boost

        max_score = np.maximum(np.maximum(skin_score, cloth_score), access_score)

        # Per-pixel classification — claim_threshold controls when a pixel is
        # confidently in a category vs. eligible for gap fill as residual.
        ct = float(claim_threshold)
        if overlap_mode == "soft_blend":
            total = np.maximum(skin_score + cloth_score + access_score, 1e-3)
            out_skin = (skin_score / total) * sil
            out_cloth = (cloth_score / total) * sil
            out_access = (access_score / total) * sil
            claimed = (max_score >= ct) & in_sil
            out_resid = sil * (in_sil & ~claimed).astype(np.float32)
        else:
            out_skin = np.zeros_like(sil)
            out_cloth = np.zeros_like(sil)
            out_access = np.zeros_like(sil)
            claimed = (max_score >= ct) & in_sil
            wins_skin = claimed & (skin_score >= max_score - 1e-3) & (skin_score > ct)
            wins_cloth = claimed & ~wins_skin & (cloth_score >= max_score - 1e-3) & (cloth_score > ct)
            wins_access = claimed & ~wins_skin & ~wins_cloth & (access_score >= max_score - 1e-3) & (access_score > ct)
            out_skin[wins_skin] = sil[wins_skin]
            out_cloth[wins_cloth] = sil[wins_cloth]
            out_access[wins_access] = sil[wins_access]
            out_resid = sil * (in_sil & ~claimed).astype(np.float32)

        # Gap fill: assign residual pixels from neighbors + color
        if gap_fill_strength > 0.01 and out_resid.any():
            from scipy.ndimage import uniform_filter
            kernel = max(3, 2 * int(gap_neighbor_radius) + 1)
            # Local average of each category mask gives "neighbor voting"
            n_skin = uniform_filter(skin, size=kernel) * skin_priority * skin_boost * skin_penalty
            n_cloth = uniform_filter(cloth, size=kernel) * clothes_priority * cloth_boost * cloth_penalty
            n_access = uniform_filter(access, size=kernel) * accessories_priority * cloth_boost
            n_max = np.maximum(np.maximum(n_skin, n_cloth), n_access)

            gap = (in_sil & (out_resid > 0.01))
            strong_neighbor = (n_max > 0.05) & gap
            color_decided = ~strong_neighbor & gap & (cw > 0.1)

            n_wins_skin = strong_neighbor & (n_skin >= n_max - 1e-3)
            n_wins_cloth = strong_neighbor & ~n_wins_skin & (n_cloth >= n_max - 1e-3)
            n_wins_access = strong_neighbor & ~n_wins_skin & ~n_wins_cloth & (n_access >= n_max - 1e-3)

            gf = float(gap_fill_strength)
            out_skin[n_wins_skin] = sil[n_wins_skin] * gf
            out_cloth[n_wins_cloth] = sil[n_wins_cloth] * gf
            out_access[n_wins_access] = sil[n_wins_access] * gf
            out_resid[strong_neighbor] *= (1.0 - gf)

            # Color-only fallback
            color_skin = color_decided & (skin_color_conf > 0.6)
            color_cloth = color_decided & ~color_skin & (cloth_color_conf > 0.6)
            out_skin[color_skin] = sil[color_skin] * gf * skin_color_conf[color_skin]
            out_cloth[color_cloth] = sil[color_cloth] * gf * cloth_color_conf[color_cloth]
            out_resid[color_skin] *= (1.0 - gf * skin_color_conf[color_skin])
            out_resid[color_cloth] *= (1.0 - gf * cloth_color_conf[color_cloth])

        # Debug viz
        if output_debug_viz:
            debug = np.zeros((h, w, 3), dtype=np.float32)
            has_skin = out_skin > 0.3
            has_cloth = out_cloth > 0.3
            has_access = out_access > 0.3
            count = has_skin.astype(int) + has_cloth.astype(int) + has_access.astype(int)
            debug[count >= 2] = [1, 1, 0]    # default 2-overlap = yellow (will refine)
            debug[has_skin & has_cloth & ~has_access] = [1, 1, 0]
            debug[has_skin & has_access & ~has_cloth] = [1, 0, 1]
            debug[has_cloth & has_access & ~has_skin] = [0, 1, 1]
            debug[count == 3] = [1, 1, 1]
            debug[has_skin & (count == 1)] = [0, 1, 0]
            debug[has_cloth & (count == 1)] = [0, 0, 1]
            debug[has_access & (count == 1)] = [1, 0, 0]
            uncategorized = (out_resid > 0.1) & in_sil & (count == 0)
            debug[uncategorized] = [0.3, 0.3, 0.3]
            tint = np.stack([skin_color_conf, skin_color_conf * 0.6, np.zeros_like(skin_color_conf)], axis=-1)
            debug = debug * (1.0 - cw * 0.3) + tint * (cw * 0.3)
            debug = debug * sil[..., None]
            debug_t = _to_image_tensor((debug * 255).clip(0, 255).astype(np.uint8))
        else:
            r_resid = (out_resid * 255).clip(0, 255).astype(np.uint8)
            debug_t = _to_image_tensor(np.stack([r_resid, r_resid, r_resid], axis=-1))

        cov_skin = 100.0 * (out_skin > 0.1).mean()
        cov_cloth = 100.0 * (out_cloth > 0.1).mean()
        cov_access = 100.0 * (out_access > 0.1).mean()
        cov_resid = 100.0 * (out_resid > 0.1).mean()
        if skin_color_mode in ("adaptive_lab", "both") and not adaptive_debug.get("skipped", True):
            ref_rgb = adaptive_debug.get("ref_rgb", [0, 0, 0])
            ref_lab = adaptive_debug.get("ref_lab", [0, 0, 0])
            n = adaptive_debug.get("sample_count", 0)
            color_status = (
                f"\n  adaptive ref: RGB={tuple(ref_rgb)} "
                f"LAB=(L{ref_lab[0]:.1f}, a{ref_lab[1]:+.1f}, b{ref_lab[2]:+.1f}) "
                f"from {n} samples (tol={adaptive_tolerance:.1f} ΔE)"
            )
        elif skin_color_mode in ("adaptive_lab", "both"):
            color_status = (
                f"\n  adaptive: SKIPPED (only {len(skin_u8[skin_u8/255 >= adaptive_sample_threshold])} "
                f"confident pixels, need {adaptive_min_samples}) → fell back to fixed_hsv"
            )
        else:
            color_status = "\n  using fixed_hsv ranges"

        status = (
            f"resolver mode={overlap_mode} color_weight={cw:.2f} skin_color={skin_color_mode} "
            f"claim_th={ct:.2f} gap_fill={gap_fill_strength:.2f} (radius={gap_neighbor_radius}px)\n"
            f"  skin={cov_skin:.2f}%  clothes={cov_cloth:.2f}%  access={cov_access:.2f}%  residual={cov_resid:.2f}%"
            + color_status
        )
        print(f"[BD MaskResolver] {status}", flush=True)

        return io.NodeOutput(
            _to_mask_tensor((out_skin * 255).clip(0, 255).astype(np.uint8)),
            _to_mask_tensor((out_cloth * 255).clip(0, 255).astype(np.uint8)),
            _to_mask_tensor((out_access * 255).clip(0, 255).astype(np.uint8)),
            _to_mask_tensor((out_resid * 255).clip(0, 255).astype(np.uint8)),
            debug_t,
            status,
        )


MASK_RESOLVER_V3_NODES = [BD_MaskResolver]
MASK_RESOLVER_NODES = {"BD_MaskResolver": BD_MaskResolver}
MASK_RESOLVER_DISPLAY_NAMES = {"BD_MaskResolver": "BD Mask Resolver"}
