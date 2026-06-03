"""
face_mp_shared.py — shared mask generation for all BD MP Face nodes.

Extracted from face_mask_mediapipe.py so BD MP Face Mask, BD MP Face Export,
and BD MP Face Infill all produce identical masks from the same landmark data.

Public API
----------
_masks_from_landmarks(lm, H, W, **expand_kwargs) → dict[str, np.ndarray]
    Generate all 18 region masks from a pre-detected MediaPipe landmark list.
    Takes the landmarks directly — MediaPipe detection is the caller's responsibility.

detect_landmarks_robust(np_img, model_path, ...) → (landmarks, meta)
    Run MediaPipe FaceLandmarker with a sanity guard against degenerate tiny
    detections (the whole 478-point mesh collapsing into a small central region —
    a transient MediaPipe failure seen inside long-lived processes). Retries with a
    fresh landmarker, then on padded copies, remapping coords back to the ORIGINAL
    image space. Use this instead of a bare landmarker.detect() call.

_init_mp_idx() → None
    Lazy-initialise the landmark index dicts (call after confirming mediapipe is loaded).

_KEYS, _BBOX_FEATURES, _OVAL_PATH, _NOSE_INDICES, _MP_IDX — landmark constants.
Helper drawing functions: _pts, _blank, _fill_convex, _fill_poly_ordered, _union,
    _subtract, _ellipse_k, _hair_mask, _forehead_mask, _ear_mask.
"""

from __future__ import annotations

import os

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import folder_paths as _folder_paths
except ImportError:
    _folder_paths = None


def _model_roots() -> list[str]:
    """Every configured ComfyUI model root, generically — `models_dir` plus the parent
    of each registered folder path (so extra_model_paths.yaml roots are picked up too).
    No hardcoded machine paths: works wherever the user pointed ComfyUI's models dir."""
    roots: list[str] = []
    if _folder_paths is None:
        return roots
    md = getattr(_folder_paths, "models_dir", None)
    if md:
        roots.append(md)
    try:
        for _name, _entry in getattr(_folder_paths, "folder_names_and_paths", {}).items():
            for _p in _entry[0]:
                parent = os.path.dirname(os.path.normpath(_p))   # .../models/checkpoints → .../models
                if parent and parent not in roots:
                    roots.append(parent)
    except Exception:
        pass
    return roots


def find_mediapipe_model(filename: str = "face_landmarker.task") -> str:
    """Locate a MediaPipe model file under any configured ComfyUI model root, in a
    `mediapipe/` subfolder. Generic + machine-agnostic (uses folder_paths, never a
    hardcoded path). Falls back to `<models_dir>/mediapipe/<filename>` if not found."""
    for base_dir in _model_roots():
        candidate = os.path.join(base_dir, "mediapipe", filename)
        if os.path.exists(candidate):
            return candidate
    md = getattr(_folder_paths, "models_dir", "models") if _folder_paths else "models"
    return os.path.join(md, "mediapipe", filename)

try:
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as _FLC
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    _FLC = None


# ── Landmark index sets ────────────────────────────────────────────────────────

def _conn_verts(connections) -> list[int]:
    s = set()
    for c in connections:
        s.add(c.start)
        s.add(c.end)
    return sorted(s)


_OVAL_PATH = [
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
    148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
    127, 162, 21, 54, 103, 67, 109,
]

_LEFT_EAR_OVAL  = [93, 234, 127, 162, 58, 132]
_RIGHT_EAR_OVAL = [251, 389, 356, 454, 323, 361]

_NOSE_INDICES = sorted({
    168, 6, 197, 195, 5, 4, 1, 2,
    19, 94, 141, 370,
    51, 45, 131, 134, 102, 48, 115, 49,
    281, 275, 360, 363, 331, 278, 344, 279,
    98, 327,
})

# Outer lip perimeter in winding order: left corner → upper lip → right corner
# → lower lip → back. Excludes inner-contour indices that cause a double
# cupid's-bow when all lip landmarks are hulled together. (Ported from
# face_socket_infill so all BD MP nodes draw the same organic lip shape.)
_OUTER_LIP_IDX = [
     61, 185,  40,  39,  37,   0, 267, 269, 270, 409, 291,   # upper (L→R)
    375, 321, 405, 314,  17,  84, 181,  91, 146,             # lower (R→L)
]

# Band/erosion defaults are expressed in 1536px-native pixels and scaled by
# max(H,W)/1536 inside _masks_from_landmarks (same convention as BD MP FaceInfill).
_REF_DIM = 1536.0

_MP_IDX: dict[str, list[int]] = {}


# ── Robust detection (tiny-detection guard + retry) ─────────────────────────────

def _lm_span(lm) -> float:
    """Larger of the x/y normalized spans of a landmark list.

    A healthy face fills a large fraction of the frame in at least one axis; a
    degenerate detection collapses small in BOTH axes, so max(x_span, y_span) is
    a robust "is this detection plausible" signal that won't reject narrow side
    profiles.
    """
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def _lm_remap(lm, W: int, H: int, px: int, py: int) -> list:
    """Remap landmarks detected on a padded image back to the ORIGINAL image's
    normalized [0,1] space. Critical: downstream UV calibration assumes coords
    are normalized to the unpadded source image.
    """
    from types import SimpleNamespace
    Wp, Hp = W + 2 * px, H + 2 * py
    return [
        SimpleNamespace(x=(p.x * Wp - px) / W,
                        y=(p.y * Hp - py) / H,
                        z=getattr(p, "z", 0.0))
        for p in lm
    ]


def detect_landmarks_robust(
    np_img: np.ndarray,
    model_path: str,
    min_conf: float = 0.3,
    min_span: float = 0.35,
    direct_retries: int = 3,
    pad_fracs: tuple = (0.35, 0.6),
):
    """Detect MediaPipe face landmarks, guarding against degenerate tiny detections.

    Background: inside ComfyUI's long-lived process, MediaPipe FaceLandmarker
    occasionally returns a degenerate result where the entire 478-point mesh
    collapses into a small central region (~0.18 span) of a face that actually
    fills the frame. The identical image re-detects correctly in a fresh process,
    so the cause is transient internal state, aggravated by frame-filling heads
    whose silhouette touches the image edges.

    Strategy (cheapest first, stops at the first plausible result):
      1. Up to `direct_retries` direct detections, each with a freshly created
         landmarker — clears transient state.
      2. If still tiny, detect on padded copies (margin so the face no longer
         touches the edges), remapping landmarks back to original space.
      3. Return the largest-span result seen.

    Returns (landmarks, meta):
      landmarks : list of objects with .x/.y/.z normalized to the ORIGINAL image,
                  or None if no face was detected at all.
      meta      : {"quality": "ok"|"degraded"|"none", "span": float,
                   "method": str, "attempts": int}
    """
    import mediapipe as mp
    from mediapipe.tasks import python as _mpt
    from mediapipe.tasks.python import vision as _mpv

    H, W = np_img.shape[:2]

    def _run(arr):
        opts = _mpv.FaceLandmarkerOptions(
            base_options=_mpt.BaseOptions(model_asset_path=model_path),
            running_mode=_mpv.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_conf,
            min_face_presence_confidence=min_conf,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        with _mpv.FaceLandmarker.create_from_options(opts) as lmk:
            res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                      data=np.ascontiguousarray(arr)))
        return res.face_landmarks[0] if res.face_landmarks else None

    best, best_span, best_method, attempts = None, -1.0, "none", 0

    # Phase 1 — direct detection, fresh landmarker each attempt
    for _ in range(max(1, direct_retries)):
        attempts += 1
        lm = _run(np_img)
        if lm is not None:
            s = _lm_span(lm)
            if s > best_span:
                best, best_span, best_method = lm, s, "direct"
            if s >= min_span:
                return best, {"quality": "ok", "span": round(s, 4),
                              "method": "direct", "attempts": attempts}

    # Phase 2 — padded retries (margin around a frame-filling face)
    for frac in pad_fracs:
        px, py = int(W * frac), int(H * frac)
        if px <= 0 or py <= 0:
            continue
        padded = np.pad(np_img, ((py, py), (px, px), (0, 0)),
                        mode="constant", constant_values=0)
        attempts += 1
        lm = _run(padded)
        if lm is None:
            continue
        lm = _lm_remap(lm, W, H, px, py)
        s = _lm_span(lm)
        if s > best_span:
            best, best_span, best_method = lm, s, f"pad:{frac}"
        if s >= min_span:
            return best, {"quality": "ok", "span": round(s, 4),
                          "method": f"pad:{frac}", "attempts": attempts}

    if best is None:
        return None, {"quality": "none", "span": 0.0,
                      "method": "none", "attempts": attempts}
    return best, {"quality": "degraded", "span": round(best_span, 4),
                  "method": best_method, "attempts": attempts}


def _init_mp_idx() -> None:
    # Mutate the existing dict IN PLACE (do not rebind) so modules that did
    # `from .face_mp_shared import _MP_IDX` keep a valid reference after init.
    if _MP_IDX or not HAS_MEDIAPIPE:
        return
    _MP_IDX.update({
        'left_eye':   _conn_verts(_FLC.FACE_LANDMARKS_LEFT_EYE),
        'right_eye':  _conn_verts(_FLC.FACE_LANDMARKS_RIGHT_EYE),
        'left_brow':  _conn_verts(_FLC.FACE_LANDMARKS_LEFT_EYEBROW),
        'right_brow': _conn_verts(_FLC.FACE_LANDMARKS_RIGHT_EYEBROW),
        'lips':       _conn_verts(_FLC.FACE_LANDMARKS_LIPS),
        'left_iris':  _conn_verts(_FLC.FACE_LANDMARKS_LEFT_IRIS),
        'right_iris': _conn_verts(_FLC.FACE_LANDMARKS_RIGHT_IRIS),
    })


_KEYS = [
    'face_oval', 'skin',
    'left_eye', 'right_eye', 'eyes',
    'left_brow', 'right_brow', 'brows',
    'left_iris', 'right_iris', 'irises',
    'lips', 'nose',
    'left_ear', 'right_ear', 'ears',
    'forehead', 'hair',
]

_BBOX_FEATURES = ['none'] + _KEYS


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _pts(indices: list[int], lm, H: int, W: int) -> np.ndarray:
    return np.array([[int(lm[i].x * W), int(lm[i].y * H)] for i in indices], dtype=np.int32)


def _ellipse_k(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _blank(H: int, W: int) -> np.ndarray:
    return np.zeros((H, W), dtype=np.uint8)


def _fill_convex(pts: np.ndarray, H: int, W: int, expand: int = 0) -> np.ndarray:
    out = _blank(H, W)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(out, hull, 255)
        if expand > 0:
            out = cv2.dilate(out, _ellipse_k(expand))
        elif expand < 0:
            out = cv2.erode(out, _ellipse_k(-expand))
    return out


def _fill_poly_ordered(ordered: list[int], lm, H: int, W: int, expand: int = 0) -> np.ndarray:
    p = _pts(ordered, lm, H, W)
    out = _blank(H, W)
    cv2.fillPoly(out, [p], 255)
    if expand > 0:
        out = cv2.dilate(out, _ellipse_k(expand))
    elif expand < 0:
        out = cv2.erode(out, _ellipse_k(-expand))
    return out


def _ellipse_k_xy(rx: int, ry: int) -> np.ndarray:
    """Elliptical structuring element with independent x/y radii."""
    rx = max(1, rx)
    ry = max(1, ry)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rx + 1, 2 * ry + 1))


def _smooth_1d(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving-average smooth of a 1-D float array. Pads with edge values."""
    if len(arr) < window:
        return arr.copy()
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(arr.astype(np.float32), window // 2, mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]


def _fill_arch_band(pts: np.ndarray, H: int, W: int,
                    expand_x: int = 0, expand_y: int = 0,
                    band_half: int = 12) -> np.ndarray:
    """Fill a smooth arch band through sorted brow landmarks.

    Two fixes vs. a convex hull (which bulges UP over the arch and rides high
    above the actual eyebrow):
      1. Arch shape — upper/lower edges follow the landmark curve instead of
         bridging across the arch with a straight baseline.
      2. Smooth Y — MediaPipe brow points jitter; a moving average removes it.

    band_half — base half-height of the band in native px; expand_y adds on top.
    Ported from face_socket_infill._fill_arch_band so every BD MP node matches.
    """
    out = _blank(H, W)
    if len(pts) < 2:
        return out
    pts_sorted = pts[np.argsort(pts[:, 0])].astype(np.float32)
    window = min(len(pts_sorted), 5)
    pts_sorted[:, 1] = _smooth_1d(pts_sorted[:, 1], window)
    if expand_x > 0:
        left  = pts_sorted[0].copy();  left[0]  = max(0.0, left[0] - expand_x)
        right = pts_sorted[-1].copy(); right[0] = min(float(W - 1), right[0] + expand_x)
        pts_sorted = np.vstack([left, pts_sorted, right])
    half_h = max(band_half, expand_y)
    upper  = pts_sorted.copy(); upper[:, 1] = np.clip(upper[:, 1] - half_h, 0, H - 1)
    lower  = pts_sorted.copy(); lower[:, 1] = np.clip(lower[:, 1] + half_h, 0, H - 1)
    band_poly = np.vstack([upper, lower[::-1]]).astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(out, [band_poly], 255)
    return out


def _fill_brow_envelope(pts: np.ndarray, H: int, W: int,
                        expand_x: int = 0, expand_y: int = 0,
                        band_floor: int = 10) -> np.ndarray:
    """Fill an eyebrow band that follows the actual upper/lower landmark edges.

    Unlike a fixed-height arch band (which collapses the brow to a thin centerline
    strip) or a convex hull (which bulges up over the arch), this interpolates the
    UPPER landmark envelope and the LOWER landmark envelope across the brow width
    and fills between them — so the band auto-adapts to the brow's real thickness
    and taper. band_floor guarantees a minimum half-height where the landmarks are
    sparse/flat; expand_x/expand_y pad the band outward.

    Note: matches the MASK to the landmark cloud. It does NOT correct for art whose
    painted brow is offset from MediaPipe's (human-trained) landmarks — for that,
    refine with SAM3 (BD MP Face Refine).
    """
    out = _blank(H, W)
    if len(pts) < 3:
        return _fill_arch_band(pts, H, W, expand_x, expand_y, max(band_floor, 1))
    ps = pts[np.argsort(pts[:, 0])].astype(np.float32)
    cl = _smooth_1d(ps[:, 1], min(len(ps), 5))
    up_pts = ps[ps[:, 1] <= cl]
    lo_pts = ps[ps[:, 1] >  cl]
    if len(up_pts) < 2:
        up_pts = ps
    if len(lo_pts) < 2:
        lo_pts = ps
    xs = np.linspace(ps[:, 0].min() - expand_x, ps[:, 0].max() + expand_x, 64)
    up_y = np.interp(xs, up_pts[:, 0], up_pts[:, 1]) - expand_y
    lo_y = np.interp(xs, lo_pts[:, 0], lo_pts[:, 1]) + expand_y
    mid  = (up_y + lo_y) / 2.0
    thin = (lo_y - up_y) < 2 * band_floor
    up_y[thin] = mid[thin] - band_floor
    lo_y[thin] = mid[thin] + band_floor
    upper = np.stack([xs, np.clip(up_y, 0, H - 1)], axis=1)
    lower = np.stack([xs, np.clip(lo_y, 0, H - 1)], axis=1)
    poly = np.vstack([upper, lower[::-1]]).astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(out, [poly], 255)
    return out


def _fill_lip_shape(pts: np.ndarray, H: int, W: int,
                    expand_x: int = 0, expand_y: int = 0,
                    lip_band: int = 6) -> np.ndarray:
    """Fill lip polygon from ordered outer-lip contour landmarks (_OUTER_LIP_IDX).

    pts must arrive in perimeter winding order — the outer contour naturally
    carries the cupid's bow; inner-contour indices are excluded. Expansion pushes
    each point outward from the centroid via sign(delta); lip_band enforces a
    minimum half-height so a closed mouth still yields a usable band.
    Ported from face_socket_infill._fill_lip_shape.
    """
    out = _blank(H, W)
    if len(pts) < 6:
        return out
    pts_f = pts.astype(np.float32).copy()
    cx = float(pts_f[:, 0].mean())
    cy = float(pts_f[:, 1].mean())
    if expand_x > 0 or expand_y > 0:
        dx = pts_f[:, 0] - cx
        dy = pts_f[:, 1] - cy
        pts_f[:, 0] = np.clip(pts_f[:, 0] + np.sign(dx) * expand_x, 0, W - 1)
        pts_f[:, 1] = np.clip(pts_f[:, 1] + np.sign(dy) * expand_y, 0, H - 1)
    if lip_band > 0:
        above = pts_f[:, 1] < cy
        below = pts_f[:, 1] > cy
        pts_f[above, 1] = np.minimum(pts_f[above, 1], cy - lip_band)
        pts_f[below, 1] = np.maximum(pts_f[below, 1], cy + lip_band)
        np.clip(pts_f[:, 1], 0, H - 1, out=pts_f[:, 1])
    cv2.fillPoly(out, [pts_f.astype(np.int32).reshape(-1, 1, 2)], 255)
    return out


def _union(*masks: np.ndarray) -> np.ndarray:
    out = masks[0].copy()
    for m in masks[1:]:
        np.maximum(out, m, out=out)
    return out


def _subtract(base: np.ndarray, *features: np.ndarray) -> np.ndarray:
    r = base.astype(np.int16)
    for f in features:
        r -= f.astype(np.int16)
    return np.clip(r, 0, 255).astype(np.uint8)


def _hair_mask(oval_pts: np.ndarray, H: int, W: int, expand: int = 20) -> np.ndarray:
    if len(oval_pts) == 0:
        return _blank(H, W)
    min_y = int(oval_pts[:, 1].min())
    left_x = int(oval_pts[:, 0].min())
    right_x = int(oval_pts[:, 0].max())
    top_thresh = min_y + (H // 2 - min_y) * 0.30
    top_pts = oval_pts[oval_pts[:, 1] <= top_thresh]
    if len(top_pts) == 0:
        top_pts = oval_pts
    poly = np.vstack([
        np.array([[left_x, 0], [right_x, 0]], dtype=np.int32),
        top_pts.astype(np.int32),
    ])
    out = _blank(H, W)
    hull = cv2.convexHull(poly)
    cv2.fillConvexPoly(out, hull, 255)
    if expand > 0:
        out = cv2.dilate(out, _ellipse_k(expand))
    return out


def _forehead_mask(oval_mask: np.ndarray, brows_mask: np.ndarray, H: int) -> np.ndarray:
    upper = oval_mask.copy()
    upper[int(H * 0.40):] = 0
    return _subtract(upper, brows_mask)


def _ear_mask(ear_oval_pts: np.ndarray, face_cx: int, H: int, W: int,
              side: str, expand: int) -> np.ndarray:
    m = _fill_convex(ear_oval_pts, H, W, expand=expand)
    clip = _blank(H, W)
    if side == 'left':
        clip[:, :face_cx] = 255
    else:
        clip[:, face_cx:] = 255
    return np.minimum(m, clip)


def _bbox_from_mask(mask: np.ndarray) -> dict | None:
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any():
        return None
    y1, y2 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    x1, x2 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return {"x": float(x1), "y": float(y1),
            "width": float(x2 - x1 + 1), "height": float(y2 - y1 + 1)}


# ── Core mask generation ───────────────────────────────────────────────────────

def _masks_from_landmarks(
    lm,
    H: int,
    W: int,
    face_expand: int = 0,
    feature_expand: int = 0,
    iris_expand: int = 4,
    subtract_nose: bool = False,
    ear_expand: int = 25,
    hair_expand: int = 20,
    head_mask_np: np.ndarray | None = None,
    tight_features: bool = True,
    brow_band: int = 12,
    eye_inset: int = 2,
    lip_band: int = 6,
) -> dict[str, np.ndarray]:
    """
    Generate all 18 region masks from a pre-detected MediaPipe landmark list.

    Parameters mirror BD MP Face Mask inputs. Call _init_mp_idx() before first use.

    tight_features (default True) selects the feature-drawing method shared with
    BD MP FaceInfill, so every BD MP node produces identical zones:
      - brows: ENVELOPE band fitted to the actual upper/lower brow landmark edges
        (auto-thickness, floored at brow_band) — not a thin centerline strip nor a
        convex hull that rides high above the eyebrow
      - eyes:  eyelid convex hull eroded inward by eye_inset (socket sits inside
        the lid edge)
      - lips:  organic outer-lip contour (cupid's bow preserved)
    brow_band (the minimum band half-height floor) / eye_inset / lip_band are in
    1536px-native pixels and scaled by max(H,W)/1536. Set tight_features=False for
    the legacy convex-hull behaviour.

    NOTE: brows track the landmark cloud. For stylized art whose painted eyebrow is
    offset from MediaPipe's human-trained landmarks (and the offset varies by pose),
    no landmark-only band will match precisely — refine with SAM3 (BD MP Face Refine)
    for pixel-accurate brows; MediaPipe still provides the Blender JSON.
    """
    scale = max(H, W) / _REF_DIM
    brow_band_px = max(1, round(brow_band * scale))
    eye_inset_px = max(0, round(eye_inset * scale))
    lip_band_px  = max(0, round(lip_band * scale))

    face_oval = _fill_poly_ordered(_OVAL_PATH, lm, H, W, expand=face_expand)
    oval_pts  = _pts(_OVAL_PATH, lm, H, W)
    face_cx   = int(oval_pts[:, 0].mean())

    if tight_features:
        # Eyes — eyelid hull eroded inward so the socket sits just inside the lid.
        def _eye(key):
            hull = _fill_convex(_pts(_MP_IDX[key], lm, H, W), H, W, expand=feature_expand)
            if eye_inset_px > 0:
                eroded = cv2.erode(hull, _ellipse_k_xy(eye_inset_px, eye_inset_px))
                return eroded if eroded.any() else hull
            return hull
        left_eye, right_eye = _eye('left_eye'), _eye('right_eye')

        # Brows — envelope band fitted to the actual upper/lower landmark edges
        # (auto-thickness), floored at brow_band so it never collapses to a strip.
        left_brow  = _fill_brow_envelope(_pts(_MP_IDX['left_brow'],  lm, H, W), H, W,
                                         expand_x=feature_expand, expand_y=feature_expand,
                                         band_floor=brow_band_px)
        right_brow = _fill_brow_envelope(_pts(_MP_IDX['right_brow'], lm, H, W), H, W,
                                         expand_x=feature_expand, expand_y=feature_expand,
                                         band_floor=brow_band_px)

        # Lips — organic outer-lip contour (winding order, cupid's bow kept).
        lips = _fill_lip_shape(_pts(_OUTER_LIP_IDX, lm, H, W), H, W,
                               expand_x=feature_expand, expand_y=feature_expand,
                               lip_band=lip_band_px)
    else:
        left_eye  = _fill_convex(_pts(_MP_IDX['left_eye'],  lm, H, W), H, W, expand=feature_expand)
        right_eye = _fill_convex(_pts(_MP_IDX['right_eye'], lm, H, W), H, W, expand=feature_expand)
        left_brow  = _fill_convex(_pts(_MP_IDX['left_brow'],  lm, H, W), H, W, expand=feature_expand)
        right_brow = _fill_convex(_pts(_MP_IDX['right_brow'], lm, H, W), H, W, expand=feature_expand)
        lips = _fill_convex(_pts(_MP_IDX['lips'], lm, H, W), H, W, expand=feature_expand)

    eyes  = _union(left_eye, right_eye)
    brows = _union(left_brow, right_brow)
    nose  = _fill_convex(_pts(_NOSE_INDICES, lm, H, W), H, W)

    if len(lm) > 477 and 'left_iris' in _MP_IDX:
        left_iris  = _fill_convex(_pts(_MP_IDX['left_iris'],  lm, H, W), H, W, expand=iris_expand)
        right_iris = _fill_convex(_pts(_MP_IDX['right_iris'], lm, H, W), H, W, expand=iris_expand)
    else:
        left_iris  = _blank(H, W)
        right_iris = _blank(H, W)
    irises = _union(left_iris, right_iris)

    left_ear  = _ear_mask(_pts(_LEFT_EAR_OVAL,  lm, H, W), face_cx, H, W, 'left',  ear_expand)
    right_ear = _ear_mask(_pts(_RIGHT_EAR_OVAL, lm, H, W), face_cx, H, W, 'right', ear_expand)
    ears = _union(left_ear, right_ear)

    skin_base      = head_mask_np if head_mask_np is not None else face_oval
    skin_subtracts = [eyes, brows, lips]
    if subtract_nose:
        skin_subtracts.append(nose)
    skin     = _subtract(skin_base, *skin_subtracts)
    hair     = _hair_mask(oval_pts, H, W, expand=hair_expand)
    forehead = _forehead_mask(face_oval, brows, H)

    return {
        'face_oval': face_oval, 'skin': skin,
        'left_eye': left_eye,   'right_eye': right_eye,   'eyes': eyes,
        'left_brow': left_brow, 'right_brow': right_brow, 'brows': brows,
        'left_iris': left_iris, 'right_iris': right_iris, 'irises': irises,
        'lips': lips,           'nose': nose,
        'left_ear': left_ear,   'right_ear': right_ear,   'ears': ears,
        'forehead': forehead,   'hair': hair,
    }
