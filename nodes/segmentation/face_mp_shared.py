"""
face_mp_shared.py — shared mask generation for all BD MP Face nodes.

Extracted from face_mask_mediapipe.py so BD MP Face Mask, BD MP Face Export,
and BD MP Face Infill all produce identical masks from the same landmark data.

Public API
----------
_masks_from_landmarks(lm, H, W, **expand_kwargs) → dict[str, np.ndarray]
    Generate all 18 region masks from a pre-detected MediaPipe landmark list.
    Takes the landmarks directly — MediaPipe detection is the caller's responsibility.

_init_mp_idx() → None
    Lazy-initialise the landmark index dicts (call after confirming mediapipe is loaded).

_KEYS, _BBOX_FEATURES, _OVAL_PATH, _NOSE_INDICES, _MP_IDX — landmark constants.
Helper drawing functions: _pts, _blank, _fill_convex, _fill_poly_ordered, _union,
    _subtract, _ellipse_k, _hair_mask, _forehead_mask, _ear_mask.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

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

_MP_IDX: dict[str, list[int]] = {}


def _init_mp_idx() -> None:
    global _MP_IDX
    if _MP_IDX or not HAS_MEDIAPIPE:
        return
    _MP_IDX = {
        'left_eye':   _conn_verts(_FLC.FACE_LANDMARKS_LEFT_EYE),
        'right_eye':  _conn_verts(_FLC.FACE_LANDMARKS_RIGHT_EYE),
        'left_brow':  _conn_verts(_FLC.FACE_LANDMARKS_LEFT_EYEBROW),
        'right_brow': _conn_verts(_FLC.FACE_LANDMARKS_RIGHT_EYEBROW),
        'lips':       _conn_verts(_FLC.FACE_LANDMARKS_LIPS),
        'left_iris':  _conn_verts(_FLC.FACE_LANDMARKS_LEFT_IRIS),
        'right_iris': _conn_verts(_FLC.FACE_LANDMARKS_RIGHT_IRIS),
    }


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
) -> dict[str, np.ndarray]:
    """
    Generate all 18 region masks from a pre-detected MediaPipe landmark list.

    Parameters mirror BD MP Face Mask inputs. Defaults (0 expand) produce
    tight reference masks suitable for UV mapping and Blender import.
    Call _init_mp_idx() before the first use.
    """
    face_oval = _fill_poly_ordered(_OVAL_PATH, lm, H, W, expand=face_expand)
    oval_pts  = _pts(_OVAL_PATH, lm, H, W)
    face_cx   = int(oval_pts[:, 0].mean())

    left_eye  = _fill_convex(_pts(_MP_IDX['left_eye'],  lm, H, W), H, W, expand=feature_expand)
    right_eye = _fill_convex(_pts(_MP_IDX['right_eye'], lm, H, W), H, W, expand=feature_expand)
    eyes      = _union(left_eye, right_eye)

    left_brow  = _fill_convex(_pts(_MP_IDX['left_brow'],  lm, H, W), H, W, expand=feature_expand)
    right_brow = _fill_convex(_pts(_MP_IDX['right_brow'], lm, H, W), H, W, expand=feature_expand)
    brows      = _union(left_brow, right_brow)

    lips = _fill_convex(_pts(_MP_IDX['lips'], lm, H, W), H, W, expand=feature_expand)
    nose = _fill_convex(_pts(_NOSE_INDICES,   lm, H, W), H, W)

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
