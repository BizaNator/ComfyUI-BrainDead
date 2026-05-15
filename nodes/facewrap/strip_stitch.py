"""
BD_FaceStripCompose — landmark-anchored cylindrical unwrap of the head
into a horizontal strip, from front/left/right/rear views.

For each detected view: TPS-warp source image → strip-space using the 478
MediaPipe FaceMesh landmark correspondences. The canonical strip-space
target positions are computed once from canonical_face_model.obj by
projecting (x, y, z) → (longitude, latitude). Per-view confidences are
cosine-bell weighted around the view's strip-center, so each side view
dominates only its own slice of the strip while front fills the middle.

Rear views (undetected by MediaPipe — no usable face landmarks) are
split at rear_split_x and slotted onto the strip's left and right edges
with a fixed low confidence. The seam pixels between rear-slot and the
adjacent warped side view are emitted into inpaint_mask so the downstream
Qwen finalize can explicitly redraw those transitions.

This bypasses 3D mesh fitting entirely — the failure mode that ICT/FLAME
were hitting on stylized characters.
"""

import numpy as np
import torch
from pathlib import Path

from comfy_api.latest import io

from .types import LandmarksBatchInput


# ----------------------------------------------------------------------------
# Canonical landmark layout (cached at module level)
# ----------------------------------------------------------------------------

_CANONICAL_OBJ = Path(__file__).resolve().parent.parent.parent / "lib" / "facewrap" / "canonical_face_model.obj"
_CANONICAL_VERTS_CACHE: np.ndarray | None = None
_STRIP_COORDS_CACHE: dict[str, np.ndarray] = {}


def _load_canonical_verts() -> np.ndarray:
    """Load (478, 3) vertex positions from canonical_face_model.obj."""
    global _CANONICAL_VERTS_CACHE
    if _CANONICAL_VERTS_CACHE is not None:
        return _CANONICAL_VERTS_CACHE
    if not _CANONICAL_OBJ.exists():
        raise FileNotFoundError(f"Missing canonical mesh: {_CANONICAL_OBJ}")
    verts = []
    with open(_CANONICAL_OBJ, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    _CANONICAL_VERTS_CACHE = np.asarray(verts, dtype=np.float32)
    return _CANONICAL_VERTS_CACHE


def _canonical_strip_coords(mode: str = "cylindrical") -> np.ndarray:
    """Project canonical landmarks to strip [0,1] coords. Returns (478, 2)."""
    if mode in _STRIP_COORDS_CACHE:
        return _STRIP_COORDS_CACHE[mode]
    verts = _load_canonical_verts()                  # (478, 3)
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

    if mode == "cylindrical":
        # Canonical orientation: +x subject-left, +z toward viewer.
        # Use atan2(x, z) — subject's LEFT (canonical +x) lands at strip_x≈0.75
        # (right of strip), subject's RIGHT (canonical −x) lands at strip_x≈0.25
        # (left of strip). This matches natural mirror-view reading: when you
        # look at a front photo, subject's left is on your right. So the front
        # photo drops into the strip in the same orientation it has in the
        # photo, no flip.
        lon = np.arctan2(x, z)                       # [-π, π]
        strip_x = (lon + np.pi) / (2.0 * np.pi)      # [0, 1], front≈0.5
    else:
        raise ValueError(f"unknown canonical_mode: {mode}")

    # latitude: linear in y, INVERTED so the canonical mesh's anatomical
    # Y-up convention (chin at low y, forehead at high y) maps to image
    # Y-down (forehead at strip top, chin at strip bottom). Without this
    # flip the warped face renders upside down in the strip.
    y_min, y_max = y.min(), y.max()
    y_span = y_max - y_min
    margin = 0.05 * y_span
    strip_y = ((y_max + margin) - y) / (y_span + 2 * margin)
    strip_y = np.clip(strip_y, 0.0, 1.0)

    out = np.stack([strip_x, strip_y], axis=-1).astype(np.float32)
    _STRIP_COORDS_CACHE[mode] = out
    return out


# ----------------------------------------------------------------------------
# Per-view warp + confidence
# ----------------------------------------------------------------------------

# Canonical strip-x centers per view hint, matching the cylindrical projection:
# atan2(x, z) → subject's left (canonical +x) lands at strip_x ≈ 0.75 (strip-right),
# subject's right (canonical −x) lands at strip_x ≈ 0.25 (strip-left). So the
# LEFT-VIEW photo (which shows the subject's left side) should peak at 0.75,
# the RIGHT-VIEW photo at 0.25. This is the natural mirror-view reading.
_VIEW_CENTERS = {
    "front": 0.50,
    "left":  0.75,
    "right": 0.25,
}


def _warp_view_to_strip(
    src_image: np.ndarray,                # (H_src, W_src, 3) uint8 or float
    src_landmarks_px: np.ndarray,         # (N_src, 2) pixel coords in src; N_src may be 478
    canonical_strip: np.ndarray,          # (N_canon, 2) in [0,1]; typically 468
    strip_w: int,
    strip_h: int,
) -> np.ndarray:
    """TPS-warp the source image into strip-space.

    Fits a TPS in the direction (strip_pixel → src_pixel) so we can evaluate
    at every strip pixel and use cv2.remap to bilinearly sample the source.
    Returns (strip_h, strip_w, 3) float32 in [0, 1].

    Source landmarks are clipped to the canonical mesh's vertex count —
    MediaPipe FaceLandmarker emits 478 points (incl. iris refinement) but
    canonical_face_model.obj is the legacy 468-vertex layout.
    """
    import cv2
    from scipy.interpolate import RBFInterpolator

    n = min(canonical_strip.shape[0], src_landmarks_px.shape[0])
    src_used = src_landmarks_px[:n]
    canon_used = canonical_strip[:n]

    target_strip_px = canon_used * np.array([strip_w, strip_h], dtype=np.float32)

    # We want a function strip_xy → src_xy. Fit on strip-space controls.
    tps = RBFInterpolator(
        target_strip_px,
        src_used,
        kernel="thin_plate_spline",
        smoothing=0.0,
    )

    # Evaluate at every strip pixel
    yy, xx = np.meshgrid(np.arange(strip_h), np.arange(strip_w), indexing="ij")
    strip_grid = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
    src_lookup = tps(strip_grid).reshape(strip_h, strip_w, 2).astype(np.float32)

    src_for_cv = src_image
    if src_for_cv.dtype != np.float32:
        src_for_cv = src_for_cv.astype(np.float32)
    if src_for_cv.max() > 1.5:
        src_for_cv = src_for_cv / 255.0

    warped = cv2.remap(
        src_for_cv,
        src_lookup[..., 0],
        src_lookup[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0),
    )
    return warped


def _view_confidence_bell(
    view_hint: str,
    strip_w: int,
    strip_h: int,
    bell_sigma: float,
    rear_extent: float = 0.0,
) -> np.ndarray:
    """Cosine-bell confidence centered on the view's canonical strip-x.

    If rear_extent > 0, the bell is gated to zero within the rear-slot
    regions [0, rear_extent] and [1 - rear_extent, 1] so the rear slot has
    clean ownership of those columns instead of bleeding side-view content.
    Returns (strip_h, strip_w) float32.
    """
    center = _VIEW_CENTERS.get(view_hint, 0.5)
    x_norm = np.linspace(0.0, 1.0, strip_w, dtype=np.float32)

    # wrapped distance on the circle [0,1]
    d = np.abs(x_norm - center)
    d = np.minimum(d, 1.0 - d)               # wrap-aware: max distance = 0.5

    # cosine bell: peak=1 at d=0, falls off, zero past d=bell_sigma
    half = max(bell_sigma, 1e-3)
    norm = np.clip(d / half, 0.0, 1.0)
    bell_1d = 0.5 * (1.0 + np.cos(np.pi * norm))   # cos bell, 1→0 over [0, half]
    bell_1d = np.where(d > half, 0.0, bell_1d)

    if rear_extent > 0.0:
        in_rear = (x_norm < rear_extent) | (x_norm > 1.0 - rear_extent)
        bell_1d = np.where(in_rear, 0.0, bell_1d)

    return np.broadcast_to(bell_1d[None, :], (strip_h, strip_w)).copy()


def _slot_rear(
    rear_image: np.ndarray,                   # (H_src, W_src, 3) float in [0,1]
    rear_split_x: float,
    strip_w: int,
    strip_h: int,
    rear_extent: float = 0.125,                # fraction of strip width per edge
) -> tuple[np.ndarray, np.ndarray]:
    """Split rear image at rear_split_x and slot the halves onto strip edges.

    Returns (rear_strip_image, rear_strip_mask).
    rear_strip_image: (strip_h, strip_w, 3) with content only in the edge slots.
    rear_strip_mask:  (strip_h, strip_w) — 1.0 where rear was placed.
    """
    import cv2
    h_src, w_src = rear_image.shape[:2]
    split_col = int(np.clip(round(rear_split_x * w_src), 1, w_src - 1))

    # In a rear photo, photographer is behind the subject facing the same
    # direction → image-LEFT shows subject's-LEFT back-of-head. With the new
    # projection, subject's left maps to strip-RIGHT (and back of head wraps
    # around past the right edge to wrap-around point at strip_x=1). So:
    # - left half of rear image  → rightmost strip slot
    # - right half of rear image → leftmost strip slot
    left_half = rear_image[:, :split_col, :]
    right_half = rear_image[:, split_col:, :]

    slot_w = max(1, int(round(rear_extent * strip_w)))
    left_resized = cv2.resize(left_half, (slot_w, strip_h), interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(right_half, (slot_w, strip_h), interpolation=cv2.INTER_AREA)

    out_img = np.zeros((strip_h, strip_w, 3), dtype=np.float32)
    out_mask = np.zeros((strip_h, strip_w), dtype=np.float32)

    out_img[:, :slot_w, :] = right_resized
    out_img[:, strip_w - slot_w:, :] = left_resized
    out_mask[:, :slot_w] = 1.0
    out_mask[:, strip_w - slot_w:] = 1.0
    return out_img, out_mask


# ----------------------------------------------------------------------------
# Node
# ----------------------------------------------------------------------------


class BD_FaceStripCompose(io.ComfyNode):
    """Cylindrical-unwrap stitch of front/left/right/rear photos into one strip."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_FaceStripCompose",
            display_name="BD Face Strip Compose",
            category="🧠BrainDead/FaceWrap",
            description=(
                "Stitch 4 head-view photos into a single horizontal head-wrap\n"
                "strip via landmark-anchored cylindrical unwrap.\n\n"
                "Per detected view (front/left/right): TPS warp on all 478\n"
                "MediaPipe landmarks → strip space. Cosine-bell confidence\n"
                "around the view's strip-center keeps each side dominant in\n"
                "its own slice. Rear views (undetected) are split at\n"
                "rear_split_x and slotted onto the strip edges with low\n"
                "confidence; the seam between rear-slot and the adjacent\n"
                "side view is added to inpaint_mask so Qwen redraws it.\n\n"
                "Outputs:\n"
                "- strip_image: stitched composite\n"
                "- coverage_mask: where the strip has any data\n"
                "- inpaint_mask: where Qwen SHOULD paint (low-coverage areas\n"
                "  AND the rear-seam bands)\n"
                "- status: per-view diagnostic"
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Source photo batch — same order/count as the "
                            "LANDMARKS_BATCH (typically 4 images: "
                            "front, left, right, rear).",
                ),
                LandmarksBatchInput(
                    "landmarks_batch",
                    tooltip="From BD_FaceLandmarks. By default, view_hint "
                            "per view picks which strip slot each image goes "
                            "to. Override with the per-slot index inputs below "
                            "when MediaPipe's yaw classifier misfires.",
                ),
                io.Int.Input(
                    "front_index",
                    default=-1,
                    min=-1,
                    max=63,
                    step=1,
                    tooltip="Image index for the FRONT slot. -1 = auto "
                            "(use the first view classified as 'front'). "
                            "Set to e.g. 0 to force image 0 into the front slot "
                            "even if MediaPipe classified it differently.",
                ),
                io.Int.Input(
                    "left_index",
                    default=-1,
                    min=-1,
                    max=63,
                    step=1,
                    tooltip="Image index for the LEFT slot (subject's left "
                            "side, lands at strip-right). -1 = auto.",
                ),
                io.Int.Input(
                    "right_index",
                    default=-1,
                    min=-1,
                    max=63,
                    step=1,
                    tooltip="Image index for the RIGHT slot (subject's right "
                            "side, lands at strip-left). -1 = auto.",
                ),
                io.Int.Input(
                    "rear_index",
                    default=-1,
                    min=-1,
                    max=63,
                    step=1,
                    tooltip="Image index for the REAR slot. -1 = auto "
                            "(first undetected view, or first view classified "
                            "as 'rear').",
                ),
                io.Int.Input(
                    "strip_width",
                    default=4096,
                    min=512,
                    max=16384,
                    step=128,
                    tooltip="Output strip width (longitude direction).",
                ),
                io.Int.Input(
                    "strip_height",
                    default=1024,
                    min=256,
                    max=4096,
                    step=64,
                    tooltip="Output strip height (latitude direction).",
                ),
                io.Float.Input(
                    "rear_split_x",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Column in the rear image where to split (0=left "
                            "edge, 1=right edge). Right of the split → leftmost "
                            "strip slot; left of the split → rightmost strip slot.",
                ),
                io.Float.Input(
                    "rear_extent",
                    default=0.125,
                    min=0.0,
                    max=0.25,
                    step=0.01,
                    optional=True,
                    tooltip="Fraction of strip width occupied by each rear half. "
                            "0.125 → each rear half takes 1/8 of the strip "
                            "(combined rear = 1/4).",
                ),
                io.Float.Input(
                    "rear_confidence",
                    default=0.1,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    optional=True,
                    tooltip="Fixed confidence value applied to the rear slots. "
                            "Low → side views dominate the seam; 0 → rear "
                            "is purely a fill that Qwen will paint over.",
                ),
                io.Float.Input(
                    "rear_seam_width",
                    default=0.04,
                    min=0.0,
                    max=0.2,
                    step=0.01,
                    optional=True,
                    tooltip="Width of the inpaint-flagged seam band on each "
                            "side of the rear slots, as a fraction of strip "
                            "width. 0 disables.",
                ),
                io.Float.Input(
                    "bell_sigma",
                    default=0.30,
                    min=0.05,
                    max=0.50,
                    step=0.01,
                    optional=True,
                    tooltip="Half-width of the per-view cosine confidence bell "
                            "(fraction of strip width). 0.30 = each view "
                            "contributes within ±30% of strip width around its "
                            "center. Wider = smoother blend; narrower = "
                            "sharper handoff between views.",
                ),
                io.Combo.Input(
                    "canonical_mode",
                    options=["cylindrical"],
                    default="cylindrical",
                    optional=True,
                    tooltip="How to project the canonical face landmarks into "
                            "strip space. cylindrical = atan2(x,z) longitude, "
                            "y latitude. (equirectangular / JSON layout: future.)",
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
        rear_split_x: float = 0.5,
        rear_extent: float = 0.125,
        rear_confidence: float = 0.1,
        rear_seam_width: float = 0.04,
        bell_sigma: float = 0.30,
        canonical_mode: str = "cylindrical",
    ) -> io.NodeOutput:
        z = torch.zeros(1, strip_height, strip_width)
        empty_img = torch.zeros(1, strip_height, strip_width, 3)

        if not isinstance(landmarks_batch, dict) or "views" not in landmarks_batch:
            return io.NodeOutput(empty_img, z, z, "ERROR: invalid LANDMARKS_BATCH")
        if images is None or images.ndim != 4:
            return io.NodeOutput(empty_img, z, z, "ERROR: images must be (B,H,W,3)")

        views = landmarks_batch["views"]
        if len(views) == 0:
            return io.NodeOutput(empty_img, z, z, "ERROR: empty views list")

        try:
            canonical_strip = _canonical_strip_coords(canonical_mode)
        except Exception as e:
            return io.NodeOutput(empty_img, z, z, f"ERROR: canonical projection failed: {e}")

        images_np = images.detach().cpu().numpy()                # (B, H, W, 3) float [0,1]
        n_imgs = images_np.shape[0]
        n_views = len(views)

        # --- Resolve slot → image index ---
        # Explicit (front/left/right/rear)_index overrides auto-classification.
        # auto picks the first view classified as that hint (rear: first
        # undetected or hint=='rear').
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
        # Clamp to valid range
        for k, v in list(slot_idx.items()):
            if v >= 0 and (v >= n_views or v >= n_imgs):
                slot_idx[k] = -1

        # Per-view accumulators
        accum_color = np.zeros((strip_height, strip_width, 3), dtype=np.float32)
        accum_weight = np.zeros((strip_height, strip_width), dtype=np.float32)
        per_slot_report = []

        # --- Warp the three landmark-driven slots (front, left, right) ---
        for slot_hint in ("front", "left", "right"):
            idx = slot_idx[slot_hint]
            if idx < 0:
                per_slot_report.append(f"{slot_hint}=skip(no source)")
                continue

            view = views[idx]
            if not view["detected"]:
                per_slot_report.append(f"{slot_hint}=skip(img {idx}: face not detected)")
                continue

            src_img = images_np[idx]
            try:
                warped = _warp_view_to_strip(
                    src_img,
                    view["landmarks_2d"],
                    canonical_strip,
                    strip_width,
                    strip_height,
                )
            except Exception as e:
                per_slot_report.append(f"{slot_hint}=ERR(img {idx}: {e.__class__.__name__})")
                continue

            conf = _view_confidence_bell(
                slot_hint, strip_width, strip_height, bell_sigma,
                rear_extent=rear_extent,
            )
            accum_color += warped * conf[..., None]
            accum_weight += conf
            per_slot_report.append(f"{slot_hint}=img{idx}(was '{view['view_hint']}')")

        # Rear slot
        rear_img = None
        rear_idx_used = slot_idx["rear"]
        if rear_idx_used >= 0 and rear_idx_used < n_imgs:
            rear_img = images_np[rear_idx_used]

        rear_mask = np.zeros((strip_height, strip_width), dtype=np.float32)
        if rear_img is not None and rear_extent > 0:
            rear_strip, rear_mask = _slot_rear(
                rear_img, rear_split_x, strip_width, strip_height, rear_extent,
            )
            rear_w = rear_mask * rear_confidence
            accum_color += rear_strip * rear_w[..., None]
            accum_weight += rear_w
            per_slot_report.append(f"rear=img{rear_idx_used}(slot={rear_extent:.2f})")
        else:
            per_slot_report.append("rear=skip")

        # Final composite — weighted average
        eps = 1e-6
        composite = accum_color / np.maximum(accum_weight, eps)[..., None]
        composite = np.clip(composite, 0.0, 1.0)

        coverage = (accum_weight > eps).astype(np.float32)

        # Inpaint mask = where coverage is weak, OR the rear-seam band
        inpaint = (coverage < 0.05).astype(np.float32)

        slot_w = max(1, int(round(rear_extent * strip_width)))
        seam_w = max(0, int(round(rear_seam_width * strip_width)))
        if seam_w > 0 and rear_extent > 0:
            # left rear slot ends at column [slot_w], right rear slot starts at strip_w - slot_w
            l0 = max(0, slot_w - seam_w // 2)
            l1 = min(strip_width, slot_w + seam_w // 2 + seam_w % 2)
            r0 = max(0, strip_width - slot_w - seam_w // 2)
            r1 = min(strip_width, strip_width - slot_w + seam_w // 2 + seam_w % 2)
            inpaint[:, l0:l1] = 1.0
            inpaint[:, r0:r1] = 1.0

        out_image = torch.from_numpy(composite).unsqueeze(0).float()        # (1, H, W, 3)
        out_coverage = torch.from_numpy(coverage).unsqueeze(0).float()      # (1, H, W)
        out_inpaint = torch.from_numpy(inpaint).unsqueeze(0).float()        # (1, H, W)

        cov_pct = 100.0 * float(coverage.mean())
        inp_pct = 100.0 * float(inpaint.mean())
        auto_hints = ",".join(v["view_hint"] for v in views)
        status = (
            f"strip {strip_height}x{strip_width} | "
            f"slots: {' | '.join(per_slot_report)} | "
            f"auto-classified: [{auto_hints}] | "
            f"coverage={cov_pct:.1f}% | inpaint={inp_pct:.1f}% | "
            f"canonical={canonical_mode}"
        )
        return io.NodeOutput(out_image, out_coverage, out_inpaint, status)


FACEWRAP_STRIP_V3_NODES = [BD_FaceStripCompose]

FACEWRAP_STRIP_NODES = {
    "BD_FaceStripCompose": BD_FaceStripCompose,
}

FACEWRAP_STRIP_DISPLAY_NAMES = {
    "BD_FaceStripCompose": "BD Face Strip Compose",
}
