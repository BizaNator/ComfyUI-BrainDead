"""
Shared matting / edge-refinement / alpha-cleanup library for BrainDead segmentation
nodes. Composable parts so BD_RemoveBackground, BD MP SAM3 Face Segment,
BD SAM3 Multi-Prompt, etc. don't each re-implement matting logic.

Conventions: alpha is a torch tensor (1,H,W) float [0,1]; image is (1,H,W,3)
float [0,1]. Functions are pure + best-effort (degrade gracefully if an optional
dep is missing). All edge-refiners operate on the boundary ROI and LOCK the
confident interior/exterior so they refine the edge, not the whole image.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


# ── VitMatte (lazy, auto-downloads from HF cache; shared across nodes) ─────────
_VITMATTE_REPOS = {"small": "hustvl/vitmatte-small-composition-1k",
                   "base": "hustvl/vitmatte-base-composition-1k"}
_VITMATTE: dict = {}


def load_vitmatte(variant: str = "small"):
    """Lazy-load + cache VitMatte (transformers). Returns (model, proc, device) or
    (None, None, None) on failure. Auto-downloads to the HF cache on first use."""
    if variant in _VITMATTE:
        return _VITMATTE[variant]
    try:
        import torch as _t
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor
        repo = _VITMATTE_REPOS.get(variant, _VITMATTE_REPOS["small"])
        dev = "cuda" if _t.cuda.is_available() else "cpu"
        print(f"[BD matting] loading VitMatte '{variant}' ({repo}) — HF cache if missing", flush=True)
        proc = VitMatteImageProcessor.from_pretrained(repo)
        model = VitMatteForImageMatting.from_pretrained(repo).to(dev).eval()
        _VITMATTE[variant] = (model, proc, dev)
    except Exception as e:
        print(f"[BD matting] VitMatte load failed ({e})", flush=True)
        _VITMATTE[variant] = (None, None, None)
    return _VITMATTE[variant]


def _roi(mask_bool: np.ndarray, pad: int, shape):
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return None
    H, W = shape
    return (max(0, int(ys.min()) - pad), min(H, int(ys.max()) + pad + 1),
            max(0, int(xs.min()) - pad), min(W, int(xs.max()) + pad + 1))


# ── alpha estimation / refinement ─────────────────────────────────────────────
def closed_form_alpha(img_np: np.ndarray, mask_np: np.ndarray,
                      erode: int = 8, dilate: int = 8) -> np.ndarray:
    """pymatting closed-form alpha from an auto-trimap. img_np HWC[0,1], mask_np HW."""
    try:
        import pymatting
    except Exception as e:
        print(f"[BD matting] closed_form needs pymatting ({e}) — passthrough", flush=True)
        return mask_np
    if not HAS_CV2:
        return mask_np
    binary = (mask_np * 255).astype(np.uint8)
    k_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
    k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
    fg = cv2.erode(binary, k_e)
    bg = cv2.dilate(binary, k_d)
    trimap = np.full(binary.shape, 0.5, np.float32)
    trimap[fg > 127] = 1.0
    trimap[bg <= 127] = 0.0
    try:
        return pymatting.estimate_alpha_cf(img_np.astype(np.float64), trimap).astype(np.float32)
    except Exception as e:
        print(f"[BD matting] closed_form failed ({e}) — passthrough", flush=True)
        return mask_np


def guided_refine(alpha: torch.Tensor, image: torch.Tensor,
                  radius: int = 6, eps: float = 1e-4) -> torch.Tensor:
    """Edge-aware SOFT refine on the boundary ROI only; confident fg/bg are LOCKED so
    it snaps the edge without fading the whole image. (1,H,W)."""
    if not (HAS_CV2 and hasattr(cv2, "ximgproc")):
        print("[BD matting] guided needs cv2.ximgproc — skipping", flush=True)
        return alpha
    a = alpha[0].cpu().numpy().astype(np.float32)
    band = (a > 0.02) & (a < 0.98)
    roi = _roi(band, max(radius * 3, 16), a.shape)
    if roi is None:
        return alpha
    y0, y1, x0, x1 = roi
    guide = np.ascontiguousarray(image[0, y0:y1, x0:x1, :3].cpu().numpy().astype(np.float32))
    filt = cv2.ximgproc.guidedFilter(guide, a[y0:y1, x0:x1], int(radius), float(eps))
    out = a.copy()
    sub = out[y0:y1, x0:x1]
    b = (sub > 0.02) & (sub < 0.98)          # only the uncertain band moves
    sub[b] = np.clip(filt[b], 0.0, 1.0)
    out[y0:y1, x0:x1] = sub
    return torch.from_numpy(out).unsqueeze(0)


def vitmatte_refine(alpha: torch.Tensor, image: torch.Tensor,
                    variant: str = "small", band: int = 10) -> torch.Tensor:
    """VitMatte deep matting on the boundary ROI → SOFT alpha. Cleanest on hair/soft
    edges. Falls back to the input alpha if VitMatte/deps are unavailable. (1,H,W)."""
    if not HAS_CV2:
        return alpha
    model, proc, dev = load_vitmatte(variant)
    if model is None:
        return alpha
    a = alpha[0].cpu().numpy().astype(np.float32)
    base = (a > 0.5).astype(np.uint8) * 255
    roi = _roi(base > 0, max(band * 3, 32), a.shape)
    if roi is None:
        return alpha
    y0, y1, x0, x1 = roi
    m = base[y0:y1, x0:x1]
    g = (image[0, y0:y1, x0:x1, :3].cpu().numpy() * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    fg = cv2.erode(m, k); bg = cv2.dilate(m, k)
    tri = np.full(m.shape, 128, np.uint8); tri[fg > 0] = 255; tri[bg == 0] = 0
    try:
        import torch as _t
        from PIL import Image as _PIL
        inp = proc(images=_PIL.fromarray(g), trimaps=_PIL.fromarray(tri), return_tensors="pt")
        inp = {kk: vv.to(dev) for kk, vv in inp.items()}
        with _t.no_grad():
            al = model(**inp).alphas[0, 0].float().cpu().numpy()
        al = al[:m.shape[0], :m.shape[1]]
    except Exception as e:
        print(f"[BD matting] vitmatte failed ({e}) — skipping", flush=True)
        return alpha
    out = a.copy(); out[y0:y1, x0:x1] = np.clip(al, 0.0, 1.0)
    return torch.from_numpy(out.astype(np.float32)).unsqueeze(0)


# ── background colour key (punch out bg-coloured gaps between legs/fingers) ────
def key_background(alpha: torch.Tensor, image: torch.Tensor,
                   tolerance: float = 0.10, max_area_frac: float = 0.04,
                   include_enclosed: bool = False, bg_color=None) -> torch.Tensor:
    """Remove background showing through gaps in the mask.

    By default removes ONLY bg-coloured regions reachable from the image border —
    i.e. OPEN gaps (between legs, around an arm) connected to the real background.
    Enclosed bg-coloured islands (between fingers in a fist) are removed only when
    `include_enclosed=True`, and even then only those smaller than `max_area_frac`.
    This protects legitimate interior details (a white nametag, buttons) that are
    enclosed by the subject and NOT connected to the background.

    Samples the bg colour from the removed region (alpha<0.1) unless `bg_color` is
    given; matches in LAB within `tolerance`. (1,H,W)."""
    if not HAS_CV2:
        return alpha
    a = alpha[0].cpu().numpy().astype(np.float32)
    inside = a > 0.5
    if not inside.any():
        return alpha
    img = image[0, :, :, :3].cpu().numpy().astype(np.float32)
    if bg_color is not None:
        bg = np.asarray(bg_color, np.float32)
    else:
        out = a < 0.1
        bg = (np.median(img[out].reshape(-1, 3), axis=0) if out.any()
              else np.array([1.0, 1.0, 1.0], np.float32))
    lab = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor((np.clip(bg, 0, 1)[None, None] * 255).astype(np.uint8),
                          cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    dist = np.linalg.norm(lab - bg_lab[None, None, :], axis=-1) / 255.0
    bg_like = (dist < float(tolerance))            # all bg-coloured pixels (whole frame)
    if not (bg_like & inside).any():
        return alpha
    H, W = a.shape
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(bg_like.astype(np.uint8), 8)

    # (1) bg-colour reachable from the image border = outer background + OPEN gaps
    #     (e.g. between the legs, open to the bottom) — remove its in-mask part, any size.
    border = np.concatenate([lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]])
    border_labels = set(int(v) for v in np.unique(border) if v != 0)
    flood = np.isin(lbl, list(border_labels)) if border_labels else np.zeros_like(bg_like)
    remove = flood & inside

    # (2) ENCLOSED bg-colour islands inside the mask (e.g. between fingers) — opt-in,
    #     and only those smaller than max_area_frac. OFF by default so interior
    #     details (white nametag, buttons) that are NOT background stay intact.
    if include_enclosed:
        enclosed = (bg_like & inside & ~flood).astype(np.uint8)
        if enclosed.any():
            n2, lbl2, stats2, _ = cv2.connectedComponentsWithStats(enclosed, 8)
            max_area = float(max_area_frac) * H * W
            for i in range(1, n2):
                if stats2[i, cv2.CC_STAT_AREA] <= max_area:
                    remove |= (lbl2 == i)

    out_a = a.copy()
    out_a[remove] = 0.0
    return torch.from_numpy(out_a).unsqueeze(0)


# ── alpha cleanup ──────────────────────────────────────────────────────────────
def clean_alpha(alpha: torch.Tensor, sharpen: float = 0.0, floor: float = 0.0) -> torch.Tensor:
    """Crisp/denoise the matte. `floor` zeros faint background alpha (kills ghosting);
    `sharpen` (0..1) tightens the edge ramp via smoothstep (higher = crisper)."""
    a = alpha.clamp(0.0, 1.0)
    if floor > 0:
        a = torch.where(a < floor, torch.zeros_like(a), a)
    if sharpen > 0:
        s = min(0.49, float(sharpen) * 0.45)
        lo, hi = s, 1.0 - s
        a = ((a - lo) / max(1e-6, (hi - lo))).clamp(0.0, 1.0)
        a = a * a * (3.0 - 2.0 * a)
    return a


def erode_alpha(alpha: torch.Tensor, px: int) -> torch.Tensor:
    """Shrink the alpha edge inward by `px` (cuts a thin fringe halo)."""
    if px <= 0 or not HAS_CV2:
        return alpha
    a = (alpha[0].cpu().numpy() * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    a = cv2.erode(a, k)
    return torch.from_numpy(a.astype(np.float32) / 255.0).unsqueeze(0)


def decontaminate(image: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Estimate true foreground colour (remove bg colour spill at soft edges) so a
    white/coloured halo doesn't show when composited. Only changes RGB. (1,H,W,3)."""
    try:
        from pymatting import estimate_foreground_ml
    except Exception as e:
        print(f"[BD matting] decontaminate needs pymatting ({e}) — skipping", flush=True)
        return image
    img_np = image[0, :, :, :3].detach().cpu().float().numpy()
    a_np = alpha[0].detach().cpu().float().numpy()
    try:
        fg = estimate_foreground_ml(img_np, a_np)
    except Exception as e:
        print(f"[BD matting] decontaminate failed ({e}) — skipping", flush=True)
        return image
    return torch.from_numpy(np.clip(fg, 0.0, 1.0).astype(np.float32)).unsqueeze(0)


# ── sticker / compositing helpers ──────────────────────────────────────────────
def hex_rgb(s: str, default=(1.0, 1.0, 1.0)):
    try:
        s = s.strip().lstrip("#")
        return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)
    except Exception:
        return default


def make_sticker(image: torch.Tensor, alpha: torch.Tensor,
                 outline_px: int, color_hex: str) -> torch.Tensor:
    """Die-cut sticker: subject + a coloured outline band, transparent outside → RGBA."""
    a = alpha[0].cpu().numpy().astype(np.float32)
    base = (a > 0.5).astype(np.uint8)
    if outline_px > 0 and HAS_CV2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outline_px + 1, 2 * outline_px + 1))
        dil = cv2.dilate(base, k).astype(np.float32)
    else:
        dil = base.astype(np.float32)
    band = np.clip(dil - a, 0.0, 1.0)[..., None]
    rgb = image[0, :, :, :3].cpu().numpy().astype(np.float32)
    col = np.array(hex_rgb(color_hex), np.float32)[None, None, :]
    out_rgb = np.clip(rgb * a[..., None] + col * band, 0.0, 1.0)
    rgba = np.concatenate([out_rgb, dil[..., None]], axis=-1)
    return torch.from_numpy(rgba).unsqueeze(0)
