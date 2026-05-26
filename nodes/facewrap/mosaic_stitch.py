"""
BD_FaceMosaicCompose — Photoshop-style horizontal mosaic stitch of 4 head views.

Unlike BD_FaceStripCompose which TPS-warps every view through a cylindrical
projection (heavy deformation; each face becomes a cylindrical unwrap), this
node does the literal "cut and paste in Photoshop" thing:

  - For each face view, compute a 2D similarity transform (rotate + scale +
    translate) that makes the eyes horizontal, normalizes face height, and
    lands the eye-center at a configurable target X column in the strip.
  - Build a feathered face-hull alpha mask per view (filled face-oval
    landmark polygon, Gaussian-blurred edge).
  - Composite all views by alpha-weighted average — overlaps blend smoothly
    via the feathering instead of a global warp.

The result keeps each face recognizable as itself. Rear photos are slotted
at the strip edges using the same silhouette-driven perspective warp as
BD_FaceStripCompose (with optional rear_mask).
"""

import numpy as np
import torch
import cv2

from comfy_api.latest import io

from .types import LandmarksBatchInput
from .strip_stitch import _slot_rear, _rear_mask_soft, _rear_gate_factor


# --- MediaPipe FaceMesh landmark indices ---------------------------------------

# Eye landmark indices (both eyes combined) — used for the eye-center anchor.
_EYE_LM_ALL = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,           # subject's right eye (image-left)
    362, 382, 381, 380, 374, 373, 390, 249, 263,        # subject's left eye  (image-right)
]
_RIGHT_EYE_OUTER = 33     # subject's right eye outer corner (low x in canonical, image-left)
_LEFT_EYE_OUTER = 263     # subject's left eye outer corner

_CHIN_LM = 152
_FOREHEAD_LM = 10

# Anchor landmarks for shared-feature alignment between front and side views.
# These canonical indices fall on the temple/tragion area that's visible in
# BOTH the front view and the respective side view — exactly where a
# Photoshop stitch would line up its two layers.
_LEFT_ANCHOR_LM = 454     # subject's left tragion (canonical +x, strip-right)
_RIGHT_ANCHOR_LM = 234    # subject's right tragion (canonical −x, strip-left)

# Face-oval boundary (used to build a face-shape alpha mask). 36 indices that
# trace the perimeter of the face mesh — covers forehead, sides, and chin.
_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]


# --- Per-view transform + alpha builders ---------------------------------------

def _per_view_similarity(
    landmarks_2d: np.ndarray,    # (N, 2) MediaPipe pixel coords in source image
    target_eye_xy: tuple[float, float],
    target_face_height_px: float,
) -> np.ndarray:
    """Build a 2x3 cv2 affine: source image → strip canvas.

    The transform rotates the source image so the eye-line is horizontal,
    scales so chin↔forehead distance = target_face_height_px, and translates
    so the eye-center lands at target_eye_xy.
    """
    eye_pts = landmarks_2d[_EYE_LM_ALL]
    eye_center = eye_pts.mean(axis=0)               # (2,) image px

    re_corner = landmarks_2d[_RIGHT_EYE_OUTER]
    le_corner = landmarks_2d[_LEFT_EYE_OUTER]
    dx = float(le_corner[0] - re_corner[0])
    dy = float(le_corner[1] - re_corner[1])
    eye_line_angle_deg = np.degrees(np.arctan2(dy, dx))   # 0 = horizontal

    chin = landmarks_2d[_CHIN_LM]
    forehead = landmarks_2d[_FOREHEAD_LM]
    face_height_src = float(np.linalg.norm(chin - forehead))
    face_height_src = max(face_height_src, 1.0)

    scale = float(target_face_height_px) / face_height_src

    # cv2.getRotationMatrix2D rotates around 'center' by 'angle' degrees (CCW)
    # and scales by 'scale'. We negate the eye-line angle so the rotation
    # straightens the eyes (makes them horizontal).
    M = cv2.getRotationMatrix2D(
        center=(float(eye_center[0]), float(eye_center[1])),
        angle=-eye_line_angle_deg,
        scale=scale,
    )
    # After rotation+scale, eye_center is unchanged in image space. Add a
    # translation so it lands at target_eye_xy in strip space.
    M[0, 2] += float(target_eye_xy[0]) - float(eye_center[0])
    M[1, 2] += float(target_eye_xy[1]) - float(eye_center[1])
    return M


def _side_view_anchored_similarity(
    landmarks_2d: np.ndarray,            # side view's landmarks_2d
    anchor_lm_idx: int,                  # shared landmark index (e.g. 454 or 234)
    anchor_target_xy: tuple[float, float], # where it should land in strip space
    target_face_height_px: float,
) -> np.ndarray:
    """Like _per_view_similarity but uses an EYE-aligned rotation + scale,
    then translates so a SHARED anchor landmark (not the eye-center) lands
    at anchor_target_xy. This is the 'align via shared landmark' mode.
    """
    eye_pts = landmarks_2d[_EYE_LM_ALL]
    eye_center = eye_pts.mean(axis=0)
    re_corner = landmarks_2d[_RIGHT_EYE_OUTER]
    le_corner = landmarks_2d[_LEFT_EYE_OUTER]
    dx = float(le_corner[0] - re_corner[0])
    dy = float(le_corner[1] - re_corner[1])
    eye_line_angle_deg = np.degrees(np.arctan2(dy, dx))

    chin = landmarks_2d[_CHIN_LM]
    forehead = landmarks_2d[_FOREHEAD_LM]
    face_height_src = max(float(np.linalg.norm(chin - forehead)), 1.0)
    scale = float(target_face_height_px) / face_height_src

    M = cv2.getRotationMatrix2D(
        center=(float(eye_center[0]), float(eye_center[1])),
        angle=-eye_line_angle_deg,
        scale=scale,
    )
    # Apply M (without final translate) to the anchor landmark to see where
    # it would land. Then translate so it lands at anchor_target_xy instead.
    anchor_src = landmarks_2d[anchor_lm_idx].astype(np.float64)
    anchor_after = M @ np.array([anchor_src[0], anchor_src[1], 1.0])
    M[0, 2] += float(anchor_target_xy[0]) - float(anchor_after[0])
    M[1, 2] += float(anchor_target_xy[1]) - float(anchor_after[1])
    return M


def _apply_affine_to_point(M: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to a 2D point. Returns (2,) float."""
    return (M[:, :2] @ xy + M[:, 2]).astype(np.float64)


def _face_oval_mask(landmarks_2d: np.ndarray, h: int, w: int) -> np.ndarray:
    """Filled face-oval polygon mask in source image space. Returns uint8."""
    pts = landmarks_2d[_FACE_OVAL].astype(np.int32)
    pts = pts.reshape(-1, 1, 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _feather_mask_float(mask_u8: np.ndarray, radius: int) -> np.ndarray:
    """Gaussian-blur a binary mask, return (h,w) float in [0,1]."""
    if radius <= 0:
        return (mask_u8.astype(np.float32) / 255.0)
    k = 2 * radius + 1
    blurred = cv2.GaussianBlur(mask_u8, (k, k), radius / 2.0)
    return blurred.astype(np.float32) / 255.0


# --- Node ----------------------------------------------------------------------


class BD_FaceMosaicCompose(io.ComfyNode):
    """Photoshop-style horizontal mosaic stitch of 4 head views."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_FaceMosaicCompose",
            display_name="BD Face Mosaic Compose",
            category="🧠BrainDead/FaceWrap",
            description=(
                "Photoshop-style mosaic stitch of 4 head-view photos: cut, "
                "align, paste side-by-side with feathered overlap.\n\n"
                "Per detected view (front/left/right): a similarity transform "
                "(rotate + scale + translate) is applied so eyes are "
                "horizontal, face_height matches face_height_ratio * "
                "strip_height, and the eye-center lands at the configured "
                "target X column in the strip. Each face stays geometrically "
                "intact — no cylindrical unwrap, no TPS deformation.\n\n"
                "A feathered face-oval alpha is built per view; views are "
                "composited by alpha-weighted average, so overlaps blend "
                "smoothly across the feather band.\n\n"
                "Rear photo: split at rear_split_x, slotted at the strip "
                "edges. If rear_mask (head silhouette from BD_SAM3MultiPrompt "
                "or similar) is provided, each rear half is perspective-"
                "warped so its silhouette fills the slot (same as "
                "BD_FaceStripCompose's rear path). Without it: cv2.resize.\n\n"
                "When the result is good enough that Qwen finalize is only "
                "doing cleanup, this is the node you want. When you need a "
                "geometrically correct cylindrical unwrap of the whole head, "
                "use BD_FaceStripCompose instead."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Source photo batch — typically 4 images: front, "
                            "left, right, rear.",
                ),
                LandmarksBatchInput(
                    "landmarks_batch",
                    tooltip="From BD_FaceLandmarks. Provides per-view "
                            "landmark positions for the similarity transform.",
                ),
                io.Int.Input(
                    "front_index", default=-1, min=-1, max=63, step=1,
                    tooltip="Image index for the FRONT slot. -1 = auto.",
                ),
                io.Int.Input(
                    "left_index", default=-1, min=-1, max=63, step=1,
                    tooltip="Image index for the LEFT slot (subject's left, "
                            "lands at strip-right by convention). -1 = auto.",
                ),
                io.Int.Input(
                    "right_index", default=-1, min=-1, max=63, step=1,
                    tooltip="Image index for the RIGHT slot (subject's right, "
                            "lands at strip-left). -1 = auto.",
                ),
                io.Int.Input(
                    "rear_index", default=-1, min=-1, max=63, step=1,
                    tooltip="Image index for the REAR slot. -1 = auto.",
                ),
                io.Int.Input(
                    "strip_width", default=4096, min=512, max=16384, step=128,
                ),
                io.Int.Input(
                    "strip_height", default=1024, min=256, max=4096, step=64,
                ),
                io.Float.Input(
                    "front_center_x", default=0.50, min=0.0, max=1.0, step=0.01,
                    tooltip="Strip-X column for the FRONT view's eye-center "
                            "(0.5 = strip center).",
                ),
                io.Float.Input(
                    "left_center_x", default=0.68, min=0.0, max=1.0, step=0.01,
                    tooltip="Strip-X column for the LEFT view (subject's left "
                            "face). 0.68 lets the left view's hull overlap "
                            "the front by ~⅓ — that's the overlap region "
                            "Qwen smooths and where shared landmarks meet.",
                ),
                io.Float.Input(
                    "right_center_x", default=0.32, min=0.0, max=1.0, step=0.01,
                    tooltip="Strip-X column for the RIGHT view. 0.32 mirrors "
                            "left_center_x — same overlap with front.",
                ),
                io.Float.Input(
                    "eye_target_y", default=0.40, min=0.0, max=1.0, step=0.01,
                    tooltip="Strip-Y row where every view's eye-center lands. "
                            "0.4 = 40% from top. Lower → more headroom above; "
                            "higher → more neck/chin below.",
                ),
                io.Float.Input(
                    "face_height_ratio", default=0.60, min=0.10, max=1.0, step=0.05,
                    tooltip="Target face height (chin↔forehead distance) as a "
                            "fraction of strip height. 0.60 makes the face "
                            "large enough that adjacent views overlap "
                            "meaningfully at their face hulls (which is where "
                            "shared landmark alignment happens).",
                ),
                io.Boolean.Input(
                    "align_to_front",
                    default=True,
                    optional=True,
                    tooltip="When ON: each side view's transform is anchored "
                            "to a shared landmark (subject's-left tragion for "
                            "the LEFT view, subject's-right tragion for the "
                            "RIGHT view) so it lands at the SAME strip-space "
                            "position as the front view placed it. This is "
                            "literal 'align by face points' stitching — the "
                            "left side of the front photo and the inner edge "
                            "of the left-profile photo meet at the same "
                            "subject feature. When OFF: each view independently "
                            "targets its configured center_x (looser fit but "
                            "easier to position manually).",
                ),
                io.Int.Input(
                    "feather_radius", default=30, min=0, max=200, step=2,
                    optional=True,
                    tooltip="Gaussian-blur radius for the per-view face-hull "
                            "alpha. 30 = moderate feathered selection — soft "
                            "enough that adjacent views crossfade smoothly "
                            "in the overlap region, but each view's face is "
                            "still visible as itself. Increase for more "
                            "blending; decrease (e.g. 10) for visible seams.",
                ),
                io.Float.Input(
                    "rear_extent", default=0.15, min=0.0, max=0.30, step=0.01,
                    optional=True,
                    tooltip="Fraction of strip width per rear half (each side).",
                ),
                io.Float.Input(
                    "rear_split_x", default=0.5, min=0.0, max=1.0, step=0.01,
                    optional=True,
                    tooltip="Column in the rear photo to split for the two halves.",
                ),
                io.Boolean.Input(
                    "rear_flip_lr", default=False, optional=True,
                    tooltip="Swap which rear half goes to which strip edge.",
                ),
                io.Mask.Input(
                    "rear_mask", optional=True,
                    tooltip="Head silhouette for rear photo (e.g. from "
                            "BD_SAM3MultiPrompt('head')). When provided, each "
                            "rear half is perspective-warped so its silhouette "
                            "fills the slot from outer edge to back-midline. "
                            "Without it, rear is plain cv2.resize-stretched.",
                ),
                io.Float.Input(
                    "rear_confidence", default=0.5, min=0.0, max=1.0, step=0.05,
                    optional=True,
                    tooltip="Weight applied to the rear contribution in the "
                            "alpha-weighted composite.",
                ),
                io.Float.Input(
                    "rear_blend_band", default=0.06, min=0.0, max=0.20, step=0.01,
                    optional=True,
                    tooltip="Smooth-blend band width at the rear-slot boundary.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="strip_image"),
                io.Mask.Output(display_name="coverage_mask"),
                io.Mask.Output(display_name="inpaint_mask"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        landmarks_batch,
        front_index: int = -1,
        left_index: int = -1,
        right_index: int = -1,
        rear_index: int = -1,
        strip_width: int = 4096,
        strip_height: int = 1024,
        front_center_x: float = 0.50,
        left_center_x: float = 0.68,
        right_center_x: float = 0.32,
        eye_target_y: float = 0.40,
        face_height_ratio: float = 0.60,
        align_to_front: bool = True,
        feather_radius: int = 30,
        rear_extent: float = 0.15,
        rear_split_x: float = 0.5,
        rear_flip_lr: bool = False,
        rear_mask: torch.Tensor = None,
        rear_confidence: float = 0.5,
        rear_blend_band: float = 0.06,
    ) -> io.NodeOutput:
        zero1 = torch.zeros(1, strip_height, strip_width)
        empty_img = torch.zeros(1, strip_height, strip_width, 3)

        if not isinstance(landmarks_batch, dict) or "views" not in landmarks_batch:
            return io.NodeOutput(empty_img, zero1, zero1, "ERROR: invalid LANDMARKS_BATCH")
        if images is None or images.ndim != 4:
            return io.NodeOutput(empty_img, zero1, zero1, "ERROR: images must be (B,H,W,3)")

        views = landmarks_batch["views"]
        if len(views) == 0:
            return io.NodeOutput(empty_img, zero1, zero1, "ERROR: empty views list")

        images_np = images.detach().cpu().numpy().astype(np.float32)
        n_imgs = images_np.shape[0]
        n_views = len(views)

        # --- Resolve slot → image index (same logic as BD_FaceStripCompose) ---
        def auto_pick(target_hint: str) -> int:
            for i, v in enumerate(views):
                if target_hint == "rear":
                    if not v["detected"] or v["view_hint"] == "rear":
                        return i
                else:
                    if v["detected"] and v["view_hint"] == target_hint:
                        return i
            return -1

        slot_idx = {
            "front": front_index if front_index >= 0 else auto_pick("front"),
            "left":  left_index  if left_index  >= 0 else auto_pick("left"),
            "right": right_index if right_index >= 0 else auto_pick("right"),
            "rear":  rear_index  if rear_index  >= 0 else auto_pick("rear"),
        }
        for k, v in list(slot_idx.items()):
            if v >= 0 and (v >= n_views or v >= n_imgs):
                slot_idx[k] = -1

        target_face_height_px = float(face_height_ratio) * float(strip_height)
        eye_y_px = float(eye_target_y) * float(strip_height)

        # Per-view target eye-X in strip pixels
        view_target_x = {
            "front": float(front_center_x) * strip_width,
            "left":  float(left_center_x)  * strip_width,
            "right": float(right_center_x) * strip_width,
        }

        # Accumulators for alpha-weighted compositing
        accum_color = np.zeros((strip_height, strip_width, 3), dtype=np.float32)
        accum_alpha = np.zeros((strip_height, strip_width), dtype=np.float32)
        per_slot_report = []

        # First pass: process FRONT view and stash its transform so side views
        # can anchor to shared landmarks when align_to_front is enabled.
        front_M = None
        front_idx = slot_idx["front"]
        anchor_lm_for_slot = {
            "left":  _LEFT_ANCHOR_LM,   # subject's-left tragion
            "right": _RIGHT_ANCHOR_LM,  # subject's-right tragion
        }

        def _process_slot(slot_hint: str) -> None:
            nonlocal front_M, accum_color, accum_alpha
            idx = slot_idx[slot_hint]
            if idx < 0:
                per_slot_report.append(f"{slot_hint}=skip(no source)")
                return
            view = views[idx]
            if not view["detected"]:
                per_slot_report.append(f"{slot_hint}=skip(img {idx}: face not detected)")
                return

            src_img = images_np[idx]
            h_src, w_src = src_img.shape[:2]
            lm = view["landmarks_2d"]
            if lm.shape[0] < max(_FACE_OVAL) + 1:
                per_slot_report.append(f"{slot_hint}=skip(landmarks too few)")
                return

            anchored = False
            try:
                if slot_hint == "front":
                    M = _per_view_similarity(
                        lm,
                        target_eye_xy=(view_target_x[slot_hint], eye_y_px),
                        target_face_height_px=target_face_height_px,
                    )
                    front_M = M
                elif (align_to_front and front_M is not None
                      and slot_hint in anchor_lm_for_slot):
                    # Compute where the front view placed this shared anchor,
                    # then build a side-view transform that lands the side's
                    # version of that same canonical landmark at the same xy.
                    anchor_idx = anchor_lm_for_slot[slot_hint]
                    front_view = views[front_idx]
                    front_anchor_src_xy = front_view["landmarks_2d"][anchor_idx].astype(np.float64)
                    front_anchor_strip_xy = _apply_affine_to_point(front_M, front_anchor_src_xy)
                    M = _side_view_anchored_similarity(
                        lm,
                        anchor_lm_idx=anchor_idx,
                        anchor_target_xy=(float(front_anchor_strip_xy[0]),
                                          float(front_anchor_strip_xy[1])),
                        target_face_height_px=target_face_height_px,
                    )
                    anchored = True
                else:
                    M = _per_view_similarity(
                        lm,
                        target_eye_xy=(view_target_x[slot_hint], eye_y_px),
                        target_face_height_px=target_face_height_px,
                    )
                warped = cv2.warpAffine(
                    src_img, M, (strip_width, strip_height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
                hull_src = _face_oval_mask(lm, h_src, w_src)
                hull_dst = cv2.warpAffine(
                    hull_src, M, (strip_width, strip_height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
                alpha = _feather_mask_float(hull_dst, feather_radius)
            except Exception as e:
                per_slot_report.append(f"{slot_hint}=ERR({e.__class__.__name__})")
                return

            accum_color += warped * alpha[..., None]
            accum_alpha += alpha
            tag = ",anchored" if anchored else ""
            per_slot_report.append(
                f"{slot_hint}=img{idx}(was '{view['view_hint']}'{tag})"
            )

        # Process front first, then the side views (which depend on front_M)
        _process_slot("front")
        _process_slot("left")
        _process_slot("right")

        # --- Rear slot (reuse strip_stitch's _slot_rear) ---
        rear_img = None
        rear_idx_used = slot_idx["rear"]
        if rear_idx_used >= 0 and rear_idx_used < n_imgs:
            rear_img = images_np[rear_idx_used]

        if rear_img is not None and rear_extent > 0:
            rear_silhouette_np = None
            if rear_mask is not None:
                if hasattr(rear_mask, "detach"):
                    m_np = rear_mask.detach().cpu().numpy()
                else:
                    m_np = np.asarray(rear_mask)
                if m_np.size > 1:
                    m_np = m_np.astype(np.float32)
                    if m_np.ndim == 3:
                        m_np = m_np[0]
                    if m_np.shape != rear_img.shape[:2]:
                        m_np = cv2.resize(
                            m_np, (rear_img.shape[1], rear_img.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    rear_silhouette_np = (m_np > 0.5)

            rear_strip, rear_strip_mask = _slot_rear(
                rear_img, rear_split_x, strip_width, strip_height, rear_extent,
                rear_blend_band=rear_blend_band, rear_flip_lr=rear_flip_lr,
                rear_mask=rear_silhouette_np,
            )
            rear_w = rear_strip_mask * rear_confidence
            accum_color += rear_strip * rear_w[..., None]
            accum_alpha += rear_w
            tag = ",mask" if rear_silhouette_np is not None else ",no-mask"
            per_slot_report.append(
                f"rear=img{rear_idx_used}(slot={rear_extent:.2f}"
                f"{',flip' if rear_flip_lr else ''}{tag})"
            )
        else:
            per_slot_report.append("rear=skip")

        # --- Final composite ---
        eps = 1e-6
        composite = accum_color / np.maximum(accum_alpha, eps)[..., None]
        composite = np.clip(composite, 0.0, 1.0)

        coverage = (accum_alpha > 0.02).astype(np.float32)
        # Inpaint = anywhere coverage is weak — that's where Qwen needs to draw
        inpaint = (accum_alpha < 0.1).astype(np.float32)

        out_image = torch.from_numpy(composite).unsqueeze(0).float()
        out_coverage = torch.from_numpy(coverage).unsqueeze(0).float()
        out_inpaint = torch.from_numpy(inpaint).unsqueeze(0).float()

        cov_pct = 100.0 * float(coverage.mean())
        inp_pct = 100.0 * float(inpaint.mean())
        auto_hints = ",".join(v["view_hint"] for v in views)
        status = (
            f"mosaic {strip_height}x{strip_width} | "
            f"slots: {' | '.join(per_slot_report)} | "
            f"auto-classified: [{auto_hints}] | "
            f"coverage={cov_pct:.1f}% | inpaint={inp_pct:.1f}% | "
            f"face_height={face_height_ratio:.2f}, feather={feather_radius}"
        )
        return io.NodeOutput(out_image, out_coverage, out_inpaint, status)


FACEWRAP_MOSAIC_V3_NODES = [BD_FaceMosaicCompose]

FACEWRAP_MOSAIC_NODES = {
    "BD_FaceMosaicCompose": BD_FaceMosaicCompose,
}

FACEWRAP_MOSAIC_DISPLAY_NAMES = {
    "BD_FaceMosaicCompose": "BD Face Mosaic Compose",
}
