"""
BD_FaceSocketInfill — one-shot face socket creator for 2D animation flipbook textures.

All expand values are 1536px-normalised: a value of 6 means 6 native pixels at
1536px, 3 pixels at 768px, 12 pixels at 3072px.  Set values as if working at
1536px — they scale automatically with image resolution.

expand_x / expand_y control horizontal and vertical dilation independently via
an elliptical structuring element — letting lips expand wide (laugh lines) while
brows stay tight vertically so they don't bleed into the eye sockets below.

Per-zone overrides (eyes_expand_x, brows_expand_y, …) take priority over the
master expand_x / expand_y when set ≥ 0.  −1 means "use master value."

Two-image mode:
  image  — detection guide. MediaPipe runs on this. Use your cleanest, most
           representative frame (e.g. neutral lighting, no motion blur).
  image1 — fill target (optional). The actual image that gets infilled. When
           not connected, falls back to image (single-image behaviour).
  Typical use: image = line-art / ILM guide, image1 = albedo / skin layer.

fill_mode:
  flat     — fill_r / fill_g / fill_b (default white, good for UV textures)
  surround — Gaussian-blurred surrounding skin
  inpaint  — OpenCV Telea inpaint per zone (cleanest). Each zone is processed
             independently so brows can't contaminate the eye socket.
"""

from __future__ import annotations

import os
import numpy as np
import torch
from comfy_api.latest import io

_MODEL_PATH = "/srv/AI_Stuff/models/mediapipe/face_landmarker.task"

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mpt
    from mediapipe.tasks.python import vision as _mpv
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as _FLC
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    _FLC = None


# ── Landmark index sets ───────────────────────────────────────────────────────

# Outer lip perimeter in winding order: left corner → upper lip → right corner
# → lower lip → back.  Excludes inner contour indices that cause the
# double-cupid's-bow when all lip landmarks are used together.
_OUTER_LIP_IDX = [
     61, 185,  40,  39,  37,   0, 267, 269, 270, 409, 291,   # upper (L→R)
    375, 321, 405, 314,  17,  84, 181,  91, 146,             # lower (R→L)
]

_NOSE_INDICES = sorted({
    168, 6, 197, 195, 5, 4, 1, 2,
    19, 94, 141, 370,
    51, 45, 131, 134, 102, 48, 115, 49,
    281, 275, 360, 363, 331, 278, 344, 279,
    98, 327,
})

_MP_IDX: dict[str, list[int]] = {}


def _init_mp_idx() -> None:
    global _MP_IDX
    if _MP_IDX or not HAS_MEDIAPIPE:
        return

    def _verts(connections) -> list[int]:
        s = set()
        for c in connections:
            s.add(c.start)
            s.add(c.end)
        return sorted(s)

    _MP_IDX = {
        'left_eye':   _verts(_FLC.FACE_LANDMARKS_LEFT_EYE),
        'right_eye':  _verts(_FLC.FACE_LANDMARKS_RIGHT_EYE),
        'left_brow':  _verts(_FLC.FACE_LANDMARKS_LEFT_EYEBROW),
        'right_brow': _verts(_FLC.FACE_LANDMARKS_RIGHT_EYEBROW),
        'lips':       _verts(_FLC.FACE_LANDMARKS_LIPS),
        'face_oval':  _verts(_FLC.FACE_LANDMARKS_FACE_OVAL),
    }
    # Iris ring landmarks (indices 468-477, only present when FaceLandmarker
    # returns 478-point mesh).  Hardcoded — FLC has no FACE_LANDMARKS_*_IRIS.
    # 468=right center, 469-472=right ring, 473=left center, 474-477=left ring.
    _MP_IDX['left_iris']  = [474, 475, 476, 477]   # ring only — centre at 473
    _MP_IDX['right_iris'] = [469, 470, 471, 472]   # ring only — centre at 468


# ── Mask helpers ──────────────────────────────────────────────────────────────

def _pts(indices: list[int], lm, H: int, W: int) -> np.ndarray:
    return np.array([[int(lm[i].x * W), int(lm[i].y * H)] for i in indices], dtype=np.int32)


def _ellipse_k_xy(rx: int, ry: int) -> np.ndarray:
    """Elliptical structuring element with independent x/y radii."""
    rx = max(1, rx)
    ry = max(1, ry)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rx + 1, 2 * ry + 1))


def _fill_convex(pts: np.ndarray, H: int, W: int,
                 expand_x: int = 0, expand_y: int = 0) -> np.ndarray:
    out = np.zeros((H, W), dtype=np.uint8)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(out, hull, 255)
        if expand_x > 0 or expand_y > 0:
            out = cv2.dilate(out, _ellipse_k_xy(max(1, expand_x), max(1, expand_y)))
    return out


def _smooth_1d(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving-average smooth of a 1-D float array.  Pads with edge values."""
    if len(arr) < window:
        return arr.copy()
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(arr.astype(np.float32), window // 2, mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]


def _fill_arch_band(pts: np.ndarray, H: int, W: int,
                    expand_x: int = 0, expand_y: int = 0,
                    band_half: int = 12) -> np.ndarray:
    """
    Fill a smooth arch band through sorted brow landmarks.

    Two fixes vs. convex hull:
    1. Arch shape — upper/lower edges follow the landmark curve instead of
       bridging across the arch with a straight baseline.
    2. Smooth Y — MediaPipe brow points have per-point Y jitter; a moving
       average over the sorted sequence removes the squiggle.

    band_half  — base half-height of the band in pixels (already scaled to
                 native resolution by the caller).  expand_y adds on top.
    """
    out = np.zeros((H, W), dtype=np.uint8)
    if len(pts) < 2:
        return out

    pts_sorted = pts[np.argsort(pts[:, 0])].astype(np.float32)

    # Smooth Y to remove point-to-point jitter along the arch
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


def _fill_iris_ellipse(pts: np.ndarray, H: int, W: int,
                       expand_x: int = 0, expand_y: int = 0) -> np.ndarray:
    """
    Fit an ellipse to iris ring landmarks and draw it.

    The 4-point iris ring gives a stable, blink-independent centre and radius.
    expand_x / expand_y grow the fitted radius independently so the socket can
    be made wider than tall (or vice versa) to match the character style.
    """
    out = np.zeros((H, W), dtype=np.uint8)
    if len(pts) < 2:
        return out
    cx = int(round(float(pts[:, 0].mean())))
    cy = int(round(float(pts[:, 1].mean())))
    dists = np.sqrt(((pts.astype(np.float64) - [cx, cy]) ** 2).sum(axis=1))
    r = int(round(float(dists.mean())))
    rx = max(2, r + expand_x)
    ry = max(2, r + expand_y)
    cv2.ellipse(out, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    return out


def _fill_lip_plane(lm, H: int, W: int,
                    lip_band_px: int = 12,
                    expand_x_px: int = 0,
                    expand_y_px: int = 0) -> np.ndarray:
    """
    Rotated rectangle plane along the mouth axis.

    Uses landmark 61 (left corner) and 291 (right corner) to compute the
    mouth axis angle, then draws a clean 4-point rectangle centred on the
    lip region.  Unlike _fill_lip_shape, this produces a flat geometric
    polygon with no cupid's bow — suitable for pre-drawing a stylised
    lip plane before Qwen edit runs.
    """
    out = np.zeros((H, W), dtype=np.uint8)
    lc = np.array([lm[61].x * W,  lm[61].y * H], dtype=np.float32)   # left corner
    rc = np.array([lm[291].x * W, lm[291].y * H], dtype=np.float32)   # right corner
    tc = np.array([lm[0].x * W,   lm[0].y * H],  dtype=np.float32)    # philtrum dip
    bc = np.array([lm[17].x * W,  lm[17].y * H], dtype=np.float32)    # lower lip centre

    dx, dy = rc - lc
    angle_deg = float(np.degrees(np.arctan2(dy, dx)))
    cx = float((lc[0] + rc[0]) / 2.0)
    cy = float((tc[1] + bc[1]) / 2.0)

    width  = float(np.sqrt(dx ** 2 + dy ** 2)) + expand_x_px * 2.0
    height = float(max(lip_band_px * 2, abs(float(tc[1]) - float(bc[1])))) + expand_y_px * 2.0

    box = cv2.boxPoints(((cx, cy), (width, height), angle_deg)).astype(np.int32)
    cv2.fillConvexPoly(out, box, 255)
    return out


def _fill_lip_shape(pts: np.ndarray, H: int, W: int,
                    expand_x: int = 0, expand_y: int = 0,
                    lip_band: int = 6) -> np.ndarray:
    """
    Fill lip polygon from ordered outer lip contour landmarks.

    pts must arrive in perimeter winding order via _OUTER_LIP_IDX — NOT
    sorted by angle or index.  The outer contour naturally carries the
    cupid's bow; inner contour indices are excluded, fixing the previous
    double-cupid and box issues.

    Expansion pushes each point outward from the centroid using sign(delta):
      corners      → expand_x sideways (tapers naturally along corner edge)
      top-centre   → expand_y upward
      bottom-centre → expand_y downward

    lip_band — minimum half-height from centroid (native px).
    """
    out = np.zeros((H, W), dtype=np.uint8)
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


def _blank(H: int, W: int) -> np.ndarray:
    return np.zeros((H, W), dtype=np.uint8)


def _feather_mask(mask_u8: np.ndarray, feather: int) -> np.ndarray:
    """Binary uint8 mask → float32 [0,1].

    Positive feather: ramp starts AT the edge and expands OUTWARD.
    Interior stays fully opaque (255); only the boundary bleeds out.
    Uses np.maximum(inner, blurred) so the interior is never dimmed.

    Negative feather: ramp goes INWARD from the edge.
    Exterior stays 0; fill fades toward its own boundary.
    Uses np.minimum(mask, blurred).

    Zero: hard binary edge pass-through.
    """
    if feather == 0:
        return mask_u8.astype(np.float32) / 255.0
    abs_f = abs(feather)
    ksize = abs_f * 2 + 1
    m_f = mask_u8.astype(np.float32)
    blurred = cv2.GaussianBlur(m_f, (ksize, ksize), abs_f * 0.5)
    if feather > 0:
        result = np.maximum(m_f, blurred)   # interior stays 255, edge escapes outward
    else:
        result = np.minimum(m_f, blurred)   # exterior stays 0, edge blends inward
    return (result / 255.0).clip(0.0, 1.0)


# ── Per-frame mask extraction ─────────────────────────────────────────────────

_MASK_KEYS = [
    'left_eye', 'right_eye', 'eyes',
    'left_brow', 'right_brow', 'brows',
    'lips', 'nose', 'face_oval',
    'lip_plane',  # rotated rectangle along 61→291 mouth axis
]
_ZONES = ('eyes', 'brows', 'lips', 'nose')


def _process_frame(
    frame_rgb: np.ndarray,
    landmarker,
    zone_expand: dict[str, tuple[int, int]],   # zone → (expand_x, expand_y)
    eye_mode: str = "iris",
    brow_band: int = 12,                        # native-px half-height of arch band
    eye_inset: int = 3,                         # native-px erosion inside eyelid (iris mode)
    lip_band: int = 6,                          # native-px minimum half-height of lip band
    lip_mode: str = "organic",                  # "organic" | "plane"
) -> tuple[dict[str, np.ndarray], str]:
    H, W = frame_rgb.shape[:2]
    blank = _blank(H, W)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result   = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return {k: blank for k in _MASK_KEYS}, "no face detected"

    lm = result.face_landmarks[0]
    has_iris = len(lm) > 477  # 478-point mesh includes iris ring landmarks

    def _zone(key: str, feat_key: str) -> np.ndarray:
        ex, ey = zone_expand[key]
        return _fill_convex(_pts(_MP_IDX[feat_key], lm, H, W), H, W,
                            expand_x=ex, expand_y=ey)

    def _brow_zone(feat_key: str) -> np.ndarray:
        ex, ey = zone_expand['brows']
        return _fill_arch_band(_pts(_MP_IDX[feat_key], lm, H, W), H, W,
                               expand_x=ex, expand_y=ey, band_half=brow_band)

    def _eye_zone(iris_key: str, eyelid_key: str) -> np.ndarray:
        ex, ey = zone_expand['eyes']
        eyelid_mask = _fill_convex(_pts(_MP_IDX[eyelid_key], lm, H, W), H, W,
                                   expand_x=ex, expand_y=ey)
        if eye_mode == "iris" and eye_inset > 0:
            # Eyelid hull is the outer shape reference; erode it inward so
            # the socket sits just inside the lid edge.  Feathering then
            # provides the outside sample zone beyond the lid.
            k = _ellipse_k_xy(eye_inset, eye_inset)
            eroded = cv2.erode(eyelid_mask, k)
            return eroded if eroded.any() else eyelid_mask
        return eyelid_mask

    left_eye  = _eye_zone('left_iris',  'left_eye')
    right_eye = _eye_zone('right_iris', 'right_eye')
    eyes      = _union(left_eye, right_eye)

    left_brow  = _brow_zone('left_brow')
    right_brow = _brow_zone('right_brow')
    brows      = _union(left_brow, right_brow)

    lip_plane = _fill_lip_plane(lm, H, W,
                                lip_band_px=lip_band,
                                expand_x_px=zone_expand['lips'][0],
                                expand_y_px=zone_expand['lips'][1])
    if lip_mode == "plane":
        lips = lip_plane
    elif lip_mode == "contour":
        # exact outer lip polygon — no forced band, no expansion
        lips = _fill_lip_shape(_pts(_OUTER_LIP_IDX, lm, H, W), H, W,
                               expand_x=0, expand_y=0, lip_band=0)
    else:  # organic
        lips = _fill_lip_shape(_pts(_OUTER_LIP_IDX, lm, H, W), H, W,
                               expand_x=zone_expand['lips'][0], expand_y=zone_expand['lips'][1],
                               lip_band=lip_band)
    nose = _fill_convex(_pts(_NOSE_INDICES,   lm, H, W), H, W,
                        expand_x=zone_expand['nose'][0], expand_y=zone_expand['nose'][1])

    # Face oval — filled convex hull of the face perimeter landmarks.
    # No expand applied (oval is a large-area shape; the caller handles subtraction).
    face_oval = _fill_convex(_pts(_MP_IDX['face_oval'], lm, H, W), H, W,
                             expand_x=0, expand_y=0)

    return {
        'left_eye': left_eye, 'right_eye': right_eye, 'eyes': eyes,
        'left_brow': left_brow, 'right_brow': right_brow, 'brows': brows,
        'lips': lips, 'nose': nose, 'face_oval': face_oval,
        'lip_plane': lip_plane,
    }, "ok"


# ── Fill helpers ──────────────────────────────────────────────────────────────

def _surround_fill(frame_u8: np.ndarray, radius: int) -> np.ndarray:
    ksize = max(3, radius * 2 + 1) | 1   # ensure odd
    return cv2.GaussianBlur(frame_u8.astype(np.float32), (ksize, ksize), radius * 0.5) / 255.0


def _inpaint_zone(
    frame_u8: np.ndarray,
    zone_mask_u8: np.ndarray,
    expand_x: int,
    expand_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inpaint one zone and return (inpainted_u8, expanded_mask_u8).

    Fringe margin uses the same X/Y anisotropy as the expansion so the
    boundary reliably sits on clean skin regardless of direction.
    """
    fringe_x = max(3, expand_x // 2)
    fringe_y = max(3, expand_y // 2)
    inpaint_mask = cv2.dilate(zone_mask_u8, _ellipse_k_xy(fringe_x, fringe_y))
    radius       = max(12, max(expand_x, expand_y))
    inpainted    = cv2.inpaint(frame_u8, inpaint_mask, radius, cv2.INPAINT_TELEA)
    return inpainted, inpaint_mask


# ── Face data NPZ → mask dict ─────────────────────────────────────────────────

# Maps _MASK_KEYS to the zone they belong to (for expand lookup)
_KEY_TO_ZONE = {
    'left_eye': 'eyes', 'right_eye': 'eyes', 'eyes': 'eyes',
    'left_brow': 'brows', 'right_brow': 'brows', 'brows': 'brows',
    'lips': 'lips', 'nose': 'nose',
    'face_oval': None, 'lip_plane': None,
}


def _masks_from_npz(
    npz_data,
    H: int,
    W: int,
    zone_expand: dict[str, tuple[int, int]],
) -> dict[str, np.ndarray]:
    """Load NPZ face data masks and apply zone_expand dilation.

    Returned dict has same keys as _process_frame so downstream code
    is identical in both MediaPipe and face-data modes.
    lip_plane is blank (requires raw landmarks to compute).
    """
    blank = np.zeros((H, W), dtype=np.uint8)

    def _load(key: str) -> np.ndarray:
        if key not in npz_data.files:
            return blank.copy()
        arr = npz_data[key]
        if arr.ndim != 2:
            return blank.copy()
        if arr.shape[:2] != (H, W):
            arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_NEAREST)
        zone = _KEY_TO_ZONE.get(key)
        if zone and zone in zone_expand:
            ex, ey = zone_expand[zone]
            if ex > 0 or ey > 0:
                arr = cv2.dilate(arr, _ellipse_k_xy(max(1, ex), max(1, ey)))
        return arr

    masks: dict[str, np.ndarray] = {k: _load(k) for k in _MASK_KEYS if k != 'lip_plane'}

    # Recompute unions from individual masks (consistent with _process_frame)
    masks['eyes']  = np.maximum(masks['left_eye'],  masks['right_eye'])
    masks['brows'] = np.maximum(masks['left_brow'], masks['right_brow'])
    masks['lip_plane'] = blank.copy()  # requires landmarks — not available from NPZ

    return masks


# ── Node ─────────────────────────────────────────────────────────────────────

_FILL_MODES = ["flat", "surround", "inpaint"]


class BD_FaceSocketInfill(io.ComfyNode):
    """
    One-shot face socket creator for 2D animation flipbook textures.

    expand_x / expand_y give independent horizontal / vertical dilation via
    an elliptical kernel so each zone can be tuned separately:
      lips_expand_x=12, lips_expand_y=4  →  catches laugh lines, not chin
      brows_expand_x=8, brows_expand_y=2 →  catches side hairs, not eye socket

    Per-zone overrides take priority over master when set ≥ 0; −1 = use master.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_FaceSocketInfill",
            display_name="BD MP Face Infill",
            category="🧠BrainDead/Segmentation",
            description=(
                "Fills face feature sockets for 2D animation flipbook textures.\n\n"
                "expand_x / expand_y use an elliptical dilation kernel so you can\n"
                "expand lips wide (laugh lines) without growing brows into eye sockets.\n\n"
                "Per-zone overrides (eyes_expand_x, brows_expand_y, …) take priority\n"
                "when set ≥ 0; −1 falls back to the master expand_x / expand_y.\n\n"
                "inpaint processes each zone independently (brows before eyes) so\n"
                "they can't contaminate each other's fill."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Detection guide — MediaPipe runs on this to locate feature zones. "
                            "Use your cleanest, most representative image (e.g. neutral pose, "
                            "good lighting). Does NOT need to be the image you want to fill.\n\n"
                            "If face_data_path is wired, MediaPipe is skipped entirely and this "
                            "image is used ONLY as the fill target (same as image1 behaviour)."),
                io.Image.Input(
                    "image1", optional=True,
                    tooltip="Fill target (optional). The image that actually gets infilled. "
                            "Masks are computed from 'image' but applied to 'image1'. "
                            "Leave unconnected to fill 'image' itself (original behaviour).\n\n"
                            "Typical use: image = ILM / line-art guide, image1 = albedo / skin layer.\n"
                            "Resolution should match 'image'; if different it is resized to match."),
                io.String.Input(
                    "face_data_path", default="", optional=True,
                    tooltip="Path to a .mpface.npz file saved by BD MP Save Face Data. "
                            "When wired (connect npz_path output), MediaPipe is bypassed — "
                            "the saved eye/brow/lip/nose/oval masks are used directly. "
                            "Enables infilling on modified images where MediaPipe would fail "
                            "(closed mouth, removed eyes, delighted, albedo-prepped, etc.).\n\n"
                            "NOTE: masks in the NPZ already include the expand from BD MP Face Mask. "
                            "Set expand_x=0, expand_y=0 here to avoid double-expansion, "
                            "or leave as-is to add extra dilation on top."),
                io.Float.Input("detection_confidence", default=0.5, min=0.1, max=1.0,
                               step=0.05, optional=True,
                               tooltip="Face detection confidence (ignored when face_data_path is set)."),

                # ── Zone toggles ─────────────────────────────────────────────
                io.Boolean.Input("eyes",  default=True,  optional=True),
                io.Boolean.Input("brows", default=True,  optional=True),
                io.Boolean.Input("lips",  default=True,  optional=True),
                io.Boolean.Input("nose",  default=False, optional=True),
                io.Boolean.Input("oval_subtract_sockets", default=True, optional=True,
                                 tooltip="face_oval output only. When ON (default): socket zones "
                                         "(eyes, brows, lips, nose — whichever are active) are "
                                         "subtracted from the face_oval mask so the oval doesn't "
                                         "overlap the socket areas. Use for blending layers where "
                                         "the oval should be the skin/non-socket region.\n"
                                         "OFF: raw face oval with no subtraction."),

                # ── Eye fill strategy ─────────────────────────────────────────
                io.Combo.Input("eye_mode", options=["iris", "eyelid"], default="iris",
                               optional=True,
                               tooltip="iris: eyelid hull eroded inward by eye_inset — socket "
                                       "sits just inside the lid edge, feather provides outside "
                                       "sample zone. eyelid: raw eyelid hull (may clip lashes)."),
                io.Int.Input("eye_inset", default=2, min=0, max=15, step=1, optional=True,
                             tooltip="Pixels to erode the eyelid hull inward (at 1536px native). "
                                     "2 = socket sits 2px inside the lid edge at 1536px. "
                                     "0 = exact eyelid boundary."),

                # ── Master expand (1536px-normalised) ──────────────────────────
                io.Int.Input("expand_x", default=6, min=0, max=60, step=1, optional=True,
                             tooltip="Master horizontal expansion in pixels at 1536px. "
                                     "Scales proportionally at other resolutions."),
                io.Int.Input("expand_y", default=6, min=0, max=60, step=1, optional=True,
                             tooltip="Master vertical expansion. Reduce to stop zones "
                                     "bleeding into adjacent areas (e.g. brows → eyes)."),

                # ── Per-zone expand overrides (−1 = use master) ───────────────
                io.Int.Input("eyes_expand_x",  default=-1, min=-1, max=60, step=1, optional=True,
                             tooltip="Eye horizontal override (−1 = master expand_x)."),
                io.Int.Input("eyes_expand_y",  default=-1, min=-1, max=60, step=1, optional=True,
                             tooltip="Eye vertical override."),
                io.Int.Input("brows_expand_x", default=-1, min=-1, max=60, step=1, optional=True,
                             tooltip="Brow horizontal override. Raise to catch side hairs."),
                io.Int.Input("brows_expand_y", default=-1, min=-1, max=60, step=1, optional=True,
                             tooltip="Brow vertical override. Keep low to avoid eye socket bleed."),
                io.Int.Input("brow_band", default=12, min=2, max=60, step=1, optional=True,
                             tooltip="Base half-height of the brow arch band in 1536px-normalised "
                                     "pixels. Sets minimum band thickness independently of "
                                     "expand_y. Increase if brows look too thin."),
                io.Int.Input("lip_band", default=6, min=0, max=40, step=1, optional=True,
                             tooltip="Minimum half-height of the lip band from centroid "
                                     "(1536px-normalised). Ensures full coverage even when the "
                                     "mouth is nearly closed. expand_y adds on top of this."),
                io.Int.Input("lips_expand_x",  default=-1, min=-1, max=60, step=1, optional=True,
                             tooltip="Lip horizontal override. Raise to cover laugh lines."),
                io.Int.Input("lips_expand_y",  default=-1, min=-1, max=60, step=1, optional=True,
                             tooltip="Lip vertical override."),
                io.Int.Input("nose_expand_x",  default=-1, min=-1, max=60, step=1, optional=True),
                io.Int.Input("nose_expand_y",  default=-1, min=-1, max=60, step=1, optional=True),

                # ── Feather ───────────────────────────────────────────────────
                io.Int.Input("feather", default=3, min=-30, max=30, step=1, optional=True,
                             tooltip="Master feather radius for socket edges.\n"
                                     "Positive: ramp starts AT the edge and expands OUTWARD — "
                                     "interior stays fully opaque (1.0), only the boundary bleeds out.\n"
                                     "Negative: ramp goes INWARD from the edge — fill fades toward "
                                     "its own boundary, exterior stays 0.\n"
                                     "0 = hard binary edge."),
                io.Int.Input("eyes_feather",  default=-1, min=-1, max=30, step=1, optional=True,
                             tooltip="Eye feather override. −1 = use master feather "
                                     "(inherits positive/negative direction). 0+ = outward override."),
                io.Int.Input("brows_feather", default=-1, min=-1, max=30, step=1, optional=True,
                             tooltip="Brow feather override. −1 = use master feather."),
                io.Int.Input("lips_feather",  default=-1, min=-1, max=30, step=1, optional=True,
                             tooltip="Lip feather override. −1 = use master feather."),
                io.Int.Input("nose_feather",  default=-1, min=-1, max=30, step=1, optional=True,
                             tooltip="Nose feather override. −1 = use master feather."),

                # ── Fill ─────────────────────────────────────────────────────
                io.Combo.Input("fill_mode", options=_FILL_MODES, default="flat", optional=True,
                               tooltip="flat: fill_r/g/b. surround: skin-tone blur. "
                                       "inpaint: per-zone Telea."),
                io.Boolean.Input("fill_from_guide", default=False, optional=True,
                                 tooltip="surround / inpaint only. When ON and image1 is connected: "
                                         "sample fill COLORS from the detection guide (image) rather "
                                         "than the fill target (image1). The result is still applied "
                                         "to image1.\n\n"
                                         "Use when image1 already has pre-filled socket areas — "
                                         "those pre-fills would otherwise bias the surrounding-color "
                                         "sample and bleed into the fill. Sampling from the clean "
                                         "guide avoids that contamination.\n\n"
                                         "Has no effect when image1 is not connected, or with "
                                         "fill_mode=flat."),
                io.Int.Input("fill_r", default=255, min=0, max=255, step=1, optional=True),
                io.Int.Input("fill_g", default=255, min=0, max=255, step=1, optional=True),
                io.Int.Input("fill_b", default=255, min=0, max=255, step=1, optional=True),

                # ── Lip plane mode ────────────────────────────────────────────
                io.Combo.Input(
                    "lip_mode", options=["organic", "contour", "plane"], default="organic",
                    optional=True,
                    tooltip="Controls the shape used for lip socket fill:\n"
                            "organic (default): MediaPipe outer lip contour with lip_band "
                            "minimum height and expand applied — can appear wider/taller "
                            "than the actual lip pixels.\n"
                            "contour: exact outer lip polygon with zero expansion and no "
                            "forced minimum height. Tight crop that follows the MediaPipe "
                            "landmark boundary precisely.\n"
                            "plane: Rotated rectangle aligned to the 61→291 mouth axis — "
                            "a flat geometric polygon for stylised pre-draw before Qwen.\n\n"
                            "The lip_plane output always emits the rotated rectangle "
                            "regardless of this setting."),
            ],
            outputs=[
                io.Image.Output("socket_image",
                                tooltip="RGB — socket regions replaced by fill."),
                io.Image.Output("alpha_image",
                                tooltip="RGBA — socket regions fully transparent."),
                io.Mask.Output("socket_mask",
                               tooltip="Feathered union of all active zone masks."),
                io.Mask.Output("left_eye"),
                io.Mask.Output("right_eye"),
                io.Mask.Output("eyes"),
                io.Mask.Output("left_brow"),
                io.Mask.Output("right_brow"),
                io.Mask.Output("brows"),
                io.Mask.Output("lips"),
                io.Mask.Output("nose"),
                io.Mask.Output("face_oval",
                               tooltip="MediaPipe face oval mask. By default, active socket zones "
                                       "are subtracted (oval_subtract_sockets=ON) so this gives "
                                       "the skin/non-socket face region for blending."),
                io.Mask.Output("lip_plane",
                               tooltip="Rotated rectangle aligned to the 61→291 mouth axis. "
                                       "Always a flat geometric polygon regardless of lip_mode. "
                                       "Use to pre-draw a stylised midgrey plane before Qwen."),
                io.String.Output("status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        image1: torch.Tensor | None = None,
        face_data_path: str = "",
        detection_confidence: float = 0.5,
        eyes: bool = True,
        brows: bool = True,
        lips: bool = True,
        nose: bool = False,
        oval_subtract_sockets: bool = True,
        expand_x: int = 6,
        expand_y: int = 6,
        eyes_expand_x: int = -1,
        eyes_expand_y: int = -1,
        brows_expand_x: int = -1,
        brows_expand_y: int = -1,
        brow_band: int = 12,
        lip_band: int = 6,
        lips_expand_x: int = -1,
        lips_expand_y: int = -1,
        nose_expand_x: int = -1,
        nose_expand_y: int = -1,
        feather: int = 3,
        eyes_feather: int = -1,
        brows_feather: int = -1,
        lips_feather: int = -1,
        nose_feather: int = -1,
        eye_mode: str = "iris",
        eye_inset: int = 2,
        fill_mode: str = "flat",
        fill_from_guide: bool = False,
        fill_r: int = 255,
        fill_g: int = 255,
        fill_b: int = 255,
        lip_mode: str = "organic",
    ) -> io.NodeOutput:

        _blank1     = torch.zeros((1, 1, 1),    dtype=torch.float32)
        _blank_img  = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        _blank_rgba = torch.zeros((1, 1, 1, 4), dtype=torch.float32)

        def _fail(msg: str) -> io.NodeOutput:
            return io.NodeOutput(_blank_img, _blank_rgba, _blank1,
                                 *([_blank1] * len(_MASK_KEYS)), msg)

        # ── Face data bypass — skip MediaPipe when NPZ is provided ───────────
        _face_data_npz = None
        _npz_path = (face_data_path or "").strip()
        if _npz_path:
            # Accept .json companion path too
            if _npz_path.endswith(".mpface.json"):
                _npz_path = _npz_path.replace(".mpface.json", ".mpface.npz")
            if os.path.exists(_npz_path):
                try:
                    _face_data_npz = np.load(_npz_path, allow_pickle=False)
                    print(f"[BD MP FaceInfill] Using face data: {os.path.basename(_npz_path)}")
                except Exception as e:
                    print(f"[BD MP FaceInfill] face_data_path load failed ({e}), falling back to MediaPipe")
            else:
                print(f"[BD MP FaceInfill] face_data_path not found: {_npz_path}, falling back to MediaPipe")

        _use_face_data = _face_data_npz is not None

        if not _use_face_data:
            if not HAS_MEDIAPIPE or not HAS_CV2:
                missing = [p for p, h in [("mediapipe", HAS_MEDIAPIPE), ("opencv-python", HAS_CV2)] if not h]
                return _fail(f"BD_FaceSocketInfill: missing — {', '.join(missing)}")
            if not os.path.exists(_MODEL_PATH):
                return _fail(f"BD_FaceSocketInfill: model not found at {_MODEL_PATH}")
        elif not HAS_CV2:
            return _fail("BD_FaceSocketInfill: opencv-python missing (needed even in face-data mode)")

        if not _use_face_data:
            _init_mp_idx()

        if image.ndim == 3:
            image = image.unsqueeze(0)
        B, H, W, C = image.shape
        scale = max(H, W) / 1536.0

        # Normalise image1 if provided
        if image1 is not None:
            if image1.ndim == 3:
                image1 = image1.unsqueeze(0)

        # ── Resolve per-zone (expand_x, expand_y) at native resolution ────────
        def _eff(override: int, master: int) -> int:
            return max(0, round((override if override >= 0 else master) * scale))

        zone_expand: dict[str, tuple[int, int]] = {
            'eyes':  (_eff(eyes_expand_x,  expand_x), _eff(eyes_expand_y,  expand_y)),
            'brows': (_eff(brows_expand_x, expand_x), _eff(brows_expand_y, expand_y)),
            'lips':  (_eff(lips_expand_x,  expand_x), _eff(lips_expand_y,  expand_y)),
            'nose':  (_eff(nose_expand_x,  expand_x), _eff(nose_expand_y,  expand_y)),
        }
        zone_feather: dict[str, int] = {
            'eyes':  eyes_feather  if eyes_feather  >= 0 else feather,
            'brows': brows_feather if brows_feather >= 0 else feather,
            'lips':  lips_feather  if lips_feather  >= 0 else feather,
            'nose':  nose_feather  if nose_feather  >= 0 else feather,
        }

        flat_rgb = np.array([fill_r / 255.0, fill_g / 255.0, fill_b / 255.0], dtype=np.float32)

        batches: dict[str, list[torch.Tensor]] = {k: [] for k in _MASK_KEYS + ['socket_soft']}
        socket_images: list[torch.Tensor] = []
        alpha_images:  list[torch.Tensor] = []
        statuses:      list[str] = []

        # ── Build MediaPipe landmarker (only when not using face data) ────────
        from contextlib import nullcontext
        if not _use_face_data:
            base_opts = _mpt.BaseOptions(model_asset_path=_MODEL_PATH)
            _mp_opts = _mpv.FaceLandmarkerOptions(
                base_options=base_opts,
                running_mode=_mpv.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=detection_confidence,
                min_face_presence_confidence=detection_confidence,
                min_tracking_confidence=detection_confidence,
            )
            _landmarker_ctx = _mpv.FaceLandmarker.create_from_options(_mp_opts)
        else:
            _landmarker_ctx = nullcontext(None)

        with _landmarker_ctx as landmarker:
            for b in range(B):
                # ── Detection frame ──────────────────────────────────────────
                detect_f  = image[b].cpu().numpy()
                detect_u8 = (detect_f * 255.0).clip(0, 255).astype(np.uint8)
                if C == 4:
                    detect_u8 = detect_u8[..., :3]
                detect_rgb_u8 = detect_u8.copy()

                # ── Fill frame (image1 if provided, else image0) ─────────────
                if image1 is not None:
                    b1 = min(b, image1.shape[0] - 1)
                    fill_f  = image1[b1].cpu().numpy()
                    fill_u8 = (fill_f * 255.0).clip(0, 255).astype(np.uint8)
                    if fill_f.shape[-1] == 4:
                        fill_u8 = fill_u8[..., :3]
                    if fill_u8.shape[:2] != (H, W):
                        fill_u8 = cv2.resize(fill_u8, (W, H), interpolation=cv2.INTER_LINEAR)
                    fill_rgb_u8  = fill_u8.copy()
                    fill_frame_f = fill_f[..., :3].astype(np.float32)
                    if fill_frame_f.shape[:2] != (H, W):
                        fill_frame_f = cv2.resize(fill_frame_f, (W, H), interpolation=cv2.INTER_LINEAR)
                else:
                    fill_rgb_u8  = detect_rgb_u8
                    fill_frame_f = detect_f[..., :3].astype(np.float32)

                brow_band_px = max(2, round(brow_band * scale))
                lip_band_px  = max(0, round(lip_band  * scale))
                eye_inset_px = max(0, round(eye_inset  * scale))

                # ── Masks: from face data NPZ or MediaPipe ───────────────────
                if _use_face_data:
                    masks = _masks_from_npz(_face_data_npz, H, W, zone_expand)
                    status = f"ok (face data: {os.path.basename(_npz_path)})"
                else:
                    masks, status = _process_frame(
                        detect_rgb_u8, landmarker, zone_expand,
                        eye_mode, brow_band_px, eye_inset_px, lip_band_px, lip_mode,
                    )
                statuses.append(status)

                # Per-zone feather applied to individual mask outputs so they
                # match the soft edges used in the fill blend.
                _zone_for = {
                    'left_eye': 'eyes',  'right_eye': 'eyes',  'eyes': 'eyes',
                    'left_brow': 'brows', 'right_brow': 'brows', 'brows': 'brows',
                    'lips': 'lips', 'nose': 'nose',
                    'lip_plane': 'lips',
                }
                for k in _MASK_KEYS:
                    if k == 'face_oval':
                        # Build oval mask with optional socket subtraction.
                        oval_u8 = masks['face_oval'].copy()
                        if oval_subtract_sockets:
                            sockets_u8 = np.zeros_like(oval_u8)
                            if eyes:  np.maximum(sockets_u8, masks['eyes'],  out=sockets_u8)
                            if brows: np.maximum(sockets_u8, masks['brows'], out=sockets_u8)
                            if lips:  np.maximum(sockets_u8, masks['lips'],  out=sockets_u8)
                            if nose:  np.maximum(sockets_u8, masks['nose'],  out=sockets_u8)
                            oval_u8 = np.where(sockets_u8 > 127, 0, oval_u8).astype(np.uint8)
                        batches['face_oval'].append(
                            torch.from_numpy(_feather_mask(oval_u8, feather))
                        )
                    else:
                        fth = zone_feather[_zone_for[k]]
                        batches[k].append(
                            torch.from_numpy(_feather_mask(masks[k], fth))
                        )

                # ── Active zones, ordered for sequential inpaint ───────────────
                zone_info = [
                    (brows, 'brows', masks['brows']),   # brows first — above eyes
                    (eyes,  'eyes',  masks['eyes']),
                    (lips,  'lips',  masks['lips']),
                    (nose,  'nose',  masks['nose']),
                ]

                combined_soft = np.zeros((H, W), dtype=np.float32)

                # Source for surround / inpaint color sampling.
                # fill_from_guide=True samples from image0 (clean detect guide)
                # so pre-filled socket areas in image1 don't contaminate the sample.
                fill_src_u8 = detect_rgb_u8 if (fill_from_guide and image1 is not None) else fill_rgb_u8

                if fill_mode == "inpaint":
                    result_u8 = fill_src_u8.copy()
                    for enabled, zone, zone_mask in zone_info:
                        if not enabled or not zone_mask.any():
                            continue
                        ex, ey = zone_expand[zone]
                        result_u8, exp_mask = _inpaint_zone(result_u8, zone_mask, ex, ey)
                        zone_soft = _feather_mask(exp_mask, zone_feather[zone])
                        np.maximum(combined_soft, zone_soft, out=combined_soft)
                    fill_np = result_u8.astype(np.float32) / 255.0

                elif fill_mode == "surround":
                    max_ex = max(ex for ex, ey in zone_expand.values())
                    fill_np = _surround_fill(fill_src_u8, max(15, max_ex * 4))
                    for enabled, zone, zone_mask in zone_info:
                        if not enabled or not zone_mask.any():
                            continue
                        zone_soft = _feather_mask(zone_mask, zone_feather[zone])
                        np.maximum(combined_soft, zone_soft, out=combined_soft)

                else:  # flat
                    fill_np = np.full((H, W, 3), flat_rgb, dtype=np.float32)
                    for enabled, zone, zone_mask in zone_info:
                        if not enabled or not zone_mask.any():
                            continue
                        zone_soft = _feather_mask(zone_mask, zone_feather[zone])
                        np.maximum(combined_soft, zone_soft, out=combined_soft)

                batches['socket_soft'].append(torch.from_numpy(combined_soft))

                alpha_2d = combined_soft[:, :, np.newaxis]
                blended  = (fill_frame_f * (1.0 - alpha_2d) + fill_np * alpha_2d).clip(0.0, 1.0)
                socket_images.append(torch.from_numpy(blended))

                alpha_ch = (1.0 - combined_soft)[:, :, np.newaxis]
                alpha_images.append(torch.from_numpy(
                    np.concatenate([blended, alpha_ch], axis=-1).astype(np.float32)
                ))

        socket_image = torch.stack(socket_images, dim=0)
        alpha_image  = torch.stack(alpha_images,  dim=0)
        out = {k: torch.stack(batches[k], dim=0) for k in _MASK_KEYS + ['socket_soft']}

        detected = sum(1 for s in statuses if s == "ok")
        ex_str = " ".join(f"{z}={zone_expand[z]}" for z in _ZONES)
        ft_str = " ".join(f"{z}={zone_feather[z]}" for z in _ZONES)
        status_str = (
            f"BD_FaceSocketInfill: {detected}/{B} | {fill_mode} | "
            f"expand({ex_str}) feather({ft_str})"
        )
        if detected < B:
            status_str += f" | missed: {[i for i,s in enumerate(statuses) if s != 'ok']}"
        print(f"[BD_FaceSocketInfill] {status_str}", flush=True)

        return io.NodeOutput(
            socket_image, alpha_image, out['socket_soft'],
            out['left_eye'], out['right_eye'], out['eyes'],
            out['left_brow'], out['right_brow'], out['brows'],
            out['lips'], out['nose'], out['face_oval'],
            out['lip_plane'],
            status_str,
        )


FACE_SOCKET_INFILL_V3_NODES      = [BD_FaceSocketInfill]
FACE_SOCKET_INFILL_NODES         = {"BD_FaceSocketInfill": BD_FaceSocketInfill}
FACE_SOCKET_INFILL_DISPLAY_NAMES = {"BD_FaceSocketInfill": "BD MP Face Infill"}
