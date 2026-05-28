"""
BD_FaceSkinRefine — dedicated face skin refinement pipeline.

Algorithm:
  1. Each wired feature (eyes, brows, lips, nose) is matched against the SAM3
     candidate batch by IoU. With exclusive=True each candidate is claimed once
     (lips first → eyes → brows → nose — most-distinct-first ordering).
  2. The matched SAM3 segment is intersected with the MediaPipe feature hull →
     SAM3 gives pixel accuracy, MediaPipe prevents bleed outside the region.
  3. Refined skin = face_oval − all refined feature masks.
     face_oval is the inner face plate boundary (no hair) — no SAM3 matching needed for it.
     feature_expand dilates each refined feature before subtracting (edge buffer).
  4. If silhouette_mask is wired, all output masks are clipped to it (hard head boundary).
  5. If image is wired, masked_image (skin only) and masked_face (full face plate) are
     composited and emitted — same pattern as BD_SAM3MultiPrompt masked_image output.

Why a dedicated node instead of BD_MaskCorrelate:
  BD_MaskCorrelate is generic — getting the skin case right required chaining
  slot_modes + subtract_slots + correct target wiring. This node bakes in the
  correct strategy (face_oval outer boundary, intersect for features, fixed
  priority order) so you just wire MediaPipe outputs and SAM3 batch.
"""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_hw(mask: torch.Tensor) -> np.ndarray:
    m = mask.detach().cpu().float()
    if m.ndim == 3:
        m = m[0]
    return m.numpy().astype(np.float32)


def _to_mask_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ab = (a > 0.5).astype(np.float32)
    bb = (b > 0.5).astype(np.float32)
    inter = (ab * bb).sum()
    union = np.maximum(ab + bb, 1e-6).clip(0, 1).sum()
    return float(inter / union) if union > 0 else 0.0


def _dilate(arr: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return arr
    try:
        import cv2
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
        u8 = (arr * 255).clip(0, 255).astype(np.uint8)
        return cv2.dilate(u8, k).astype(np.float32) / 255.0
    except ImportError:
        return arr


def _resize_to(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    if arr.shape == (H, W):
        return arr
    try:
        import cv2
        u8 = (arr * 255).clip(0, 255).astype(np.uint8)
        return cv2.resize(u8, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    except ImportError:
        from PIL import Image
        pil = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8), mode="L")
        return np.asarray(pil.resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0


def _make_bg(bg_mode: str, H: int, W: int) -> np.ndarray:
    if bg_mode == "white":
        return np.ones((H, W, 3), dtype=np.float32)
    if bg_mode == "checker":
        sz = 16
        rows = (np.arange(H) // sz)[:, None]
        cols = (np.arange(W) // sz)[None, :]
        light = np.where((rows + cols) % 2 == 0, 0.7, 0.4).astype(np.float32)
        return np.stack([light, light, light], axis=-1)
    return np.zeros((H, W, 3), dtype=np.float32)


def _composite(img_np: np.ndarray, alpha: np.ndarray, bg_mode: str) -> torch.Tensor:
    """Alpha-composite img_np with mask. Returns (1, H, W, 3 or 4) float32 tensor."""
    H, W = alpha.shape
    a = alpha[..., None]
    if bg_mode == "transparent":
        rgba = np.concatenate([img_np[..., :3] * a, a], axis=-1)
        return torch.from_numpy(rgba.clip(0, 1).astype(np.float32)).unsqueeze(0)
    bg = _make_bg(bg_mode, H, W)
    composited = img_np[..., :3] * a + bg * (1.0 - a)
    return torch.from_numpy(composited.clip(0, 1).astype(np.float32)).unsqueeze(0)


# Feature registry: (slot_name, debug_color_rgb, exclusive_priority)
# Higher priority → first pick in exclusive mode
_FEATURES = [
    ("lips",       (0.95, 0.30, 0.35), 5),
    ("left_eye",   (0.30, 0.60, 1.00), 4),
    ("right_eye",  (0.20, 0.50, 0.90), 4),
    ("left_brow",  (0.80, 0.58, 0.18), 3),
    ("right_brow", (0.68, 0.46, 0.14), 3),
    ("nose",       (0.30, 0.82, 0.42), 2),
]

_SKIN_COLOR = (0.95, 0.78, 0.60)


def _match_best(
    t_arr: np.ndarray,
    cands_hw: list[np.ndarray],
    used_cands: set[int],
    min_iou: float,
    target_expand: int,
    clip_expand: int,
    max_target_fill: float,
    exclusive: bool,
) -> tuple[np.ndarray | None, int, float, str]:
    """Find and intersect the best SAM3 candidate for a feature target.

    Returns (refined_or_None, best_j, best_iou, note).
    refined is None on skip (full-white) or no match.

    Strategy: MediaPipe identifies the zone (IoU matching via expanded hull).
    The SAM3 candidate provides the pixel-accurate shape — it is clipped by
    the clip-expanded hull (not the raw hull) so candidate edge detail outside
    the tight MediaPipe boundary is preserved while wild bleed-out is still blocked.
    """
    fill = float((t_arr > 0.5).mean())
    if max_target_fill < 1.0 and fill > max_target_fill:
        return None, -1, fill, f"SKIPPED fill={fill:.1%}"

    # Expand for IoU matching — wider than clip so we find the right candidate
    # even when MediaPipe is tight and only partially overlaps the SAM3 segment.
    t_query = _dilate(t_arr, target_expand)

    best_iou, best_j = -1.0, -1
    for j, c in enumerate(cands_hw):
        if exclusive and j in used_cands:
            continue
        iou = _iou(t_query, c)
        if iou > best_iou:
            best_iou, best_j = iou, j

    if best_j >= 0 and best_iou >= min_iou:
        # Clip guard: candidate is clipped to the clip-expanded hull, not the
        # raw hull. This lets SAM3's pixel-accurate edges show through at the
        # boundary while still blocking bleed beyond clip_expand pixels.
        t_clip = _dilate(t_arr, clip_expand) if clip_expand != target_expand else t_query
        refined = np.minimum(t_clip, cands_hw[best_j]).clip(0, 1)
        return refined, best_j, best_iou, f"cand={best_j} IoU={best_iou:.3f}"

    return None, -1, best_iou, f"no match (best IoU={best_iou:.3f})"


# ── Node ─────────────────────────────────────────────────────────────────────

class BD_FaceSkinRefine(io.ComfyNode):
    """
    Refine face feature masks with SAM3, then compute pixel-accurate skin.

    Wire MediaPipe feature masks and a SAM3 segment batch.
    Each feature is matched to the closest SAM3 segment (IoU, intersect mode).
    Skin = face_oval − refined_eyes − refined_brows − refined_lips − refined_nose.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_FaceSkinRefine",
            display_name="BD Face Skin Refine",
            category="🧠BrainDead/Segmentation",
            description=(
                "Refine MediaPipe face feature masks using SAM3 segments, then compute "
                "skin = face_oval − refined features.\n\n"
                "Each feature (eye, brow, lips, nose) is matched to the best-overlapping SAM3 "
                "segment by IoU. The matched segment is intersected with the MediaPipe hull — "
                "SAM3 adds pixel-accuracy, MediaPipe prevents bleed. "
                "face_oval is the inner face-plate boundary (no hair); no SAM3 matching is attempted for it.\n\n"
                "Wire silhouette_mask (head outline) to clip all outputs to the head boundary. "
                "Wire image to get masked_image (skin only) and masked_face (full face plate) composited outputs."
            ),
            inputs=[
                io.Mask.Input(
                    "candidates",
                    tooltip="SAM3 segment batch (B, H, W). Each frame is one candidate segment.",
                ),
                io.Mask.Input(
                    "face_oval",
                    tooltip="MediaPipe face oval — the INNER face plate (covers the face surface, "
                            "not hair or the full head silhouette). Outer boundary for skin: "
                            "refined skin = face_oval minus all refined feature masks.",
                ),
                io.Image.Input(
                    "image", optional=True,
                    tooltip="Original character image. When wired:\n"
                            "  • Used as the debug overlay base (shows mask colours over the render)\n"
                            "  • Produces masked_image (skin region) and masked_face (full face plate)\n"
                            "Leave unwired to composite the debug overlay on black.",
                ),
                io.Mask.Input(
                    "silhouette_mask", optional=True,
                    tooltip="Head silhouette (white=head, black=background). When wired, all output "
                            "masks are clipped to this boundary after refinement.\n\n"
                            "Note: face_oval is the INNER face plate (no hair); silhouette_mask is the "
                            "OUTER head boundary (full head shape including hair). They are different. "
                            "Use head mask from BD_SAM3MultiPrompt here.",
                ),
                io.Combo.Input(
                    "masked_image_bg",
                    options=["transparent", "white", "black", "checker"],
                    default="transparent", optional=True,
                    tooltip="Background for masked_image and masked_face outputs when image is wired.\n"
                            "  transparent → RGBA PNG, background is fully transparent (alpha=0)\n"
                            "  white / black → RGB composite over solid colour\n"
                            "  checker → grey checkerboard (visually indicates transparency)"
                ),
                io.Mask.Input("left_eye",   optional=True,
                              tooltip="Left eye mask from BD_MediaPipeFaceMask (subject's left = image right)."),
                io.Mask.Input("right_eye",  optional=True,
                              tooltip="Right eye mask from BD_MediaPipeFaceMask."),
                io.Mask.Input("left_brow",  optional=True,
                              tooltip="Left eyebrow mask from BD_MediaPipeFaceMask."),
                io.Mask.Input("right_brow", optional=True,
                              tooltip="Right eyebrow mask from BD_MediaPipeFaceMask."),
                io.Mask.Input("lips",       optional=True,
                              tooltip="Lips mask from BD_MediaPipeFaceMask."),
                io.Mask.Input("nose",       optional=True,
                              tooltip="Nose mask from BD_MediaPipeFaceMask."),
                io.Float.Input(
                    "min_iou", default=0.05, min=0.0, max=1.0, step=0.01, optional=True,
                    tooltip="Minimum IoU for a SAM3 candidate to be accepted as a feature match. "
                            "0.05 is permissive. Raise to 0.15+ if wrong segments are being matched.",
                ),
                io.Int.Input(
                    "target_expand", default=4, min=0, max=60, step=1, optional=True,
                    tooltip="Pixels to dilate each feature mask BEFORE computing IoU. "
                            "Used only for finding the right SAM3 candidate — does not affect the "
                            "final clip boundary. 4–8 px is usually enough.",
                ),
                io.Int.Input(
                    "clip_expand", default=4, min=0, max=40, step=1, optional=True,
                    tooltip="How far outside the raw MediaPipe hull the SAM3 candidate is allowed to extend. "
                            "SAM3 provides the pixel-accurate shape; this guards against wild bleed. "
                            "Matches target_expand by default. "
                            "Raise to let SAM3's edges show through more freely. "
                            "Lower to stay closer to the MediaPipe shape.",
                ),
                io.Int.Input(
                    "feature_expand", default=0, min=0, max=20, step=1, optional=True,
                    tooltip="Pixels to dilate each REFINED feature mask BEFORE subtracting from skin. "
                            "Creates an edge buffer — useful so the skin shader has a small gap between "
                            "skin paint and the eye/lip regions. 0–4 px typical.",
                ),
                io.Float.Input(
                    "max_target_fill", default=0.95, min=0.0, max=1.0, step=0.01, optional=True,
                    tooltip="Reject feature targets covering more than this fraction of the image. "
                            "Catches SAM3 full-white 'not found' results from invert_negative. "
                            "Set to 1.0 to disable.",
                ),
                io.Boolean.Input(
                    "exclusive", default=True, optional=True,
                    tooltip="Each SAM3 candidate can only be matched to ONE feature (highest-IoU wins). "
                            "Matching order: lips → eyes → brows → nose. "
                            "Recommended ON — face features don't overlap in SAM3's output.",
                ),
                io.Float.Input(
                    "overlay_alpha", default=0.55, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="Colour overlay opacity on the debug image.",
                ),
            ],
            outputs=[
                io.Mask.Output(display_name="skin",
                               tooltip="Refined skin: face_oval minus all refined feature masks "
                                       "(with feature_expand buffer if set). Clipped to silhouette_mask if wired."),
                io.Mask.Output(display_name="left_eye",
                               tooltip="SAM3-refined left eye, clipped to MediaPipe hull. "
                                       "Falls back to original MediaPipe mask if no match."),
                io.Mask.Output(display_name="right_eye",   tooltip="SAM3-refined right eye."),
                io.Mask.Output(display_name="left_brow",   tooltip="SAM3-refined left brow."),
                io.Mask.Output(display_name="right_brow",  tooltip="SAM3-refined right brow."),
                io.Mask.Output(display_name="lips",        tooltip="SAM3-refined lips."),
                io.Mask.Output(display_name="nose",        tooltip="SAM3-refined nose."),
                io.Image.Output(display_name="debug_overlay",
                                tooltip="Colour-coded overlay: skin in warm tone, each feature in its "
                                        "own colour. Wire to PreviewImage to inspect the segmentation."),
                io.String.Output(display_name="match_info",
                                 tooltip="Per-feature match summary: candidate index, IoU, fallback notes."),
                io.Image.Output(
                    display_name="masked_image",
                    tooltip="Original image with the head/body silhouette applied AND feature holes cut out "
                            "(eyes, brows, lips, nose removed). Uses silhouette_mask as the outer boundary "
                            "if wired, falls back to face_oval if not. "
                            "Requires image to be wired. Background controlled by masked_image_bg.",
                ),
                io.Image.Output(
                    display_name="masked_skin",
                    tooltip="Original image composited with the skin mask as alpha — shows only the skin "
                            "region (face_oval minus all refined feature holes). Tighter cut than masked_image "
                            "since it uses face_oval rather than the full silhouette. "
                            "Requires image to be wired. Background controlled by masked_image_bg.",
                ),
                io.Mask.Output(
                    display_name="head_mask",
                    tooltip="Head/face boundary with feature holes cut out — same alpha as masked_image. "
                            "Outer boundary is silhouette_mask if wired, else face_oval. "
                            "Eye/brow/lip/nose regions are subtracted so this is a paintable 'skin shape' mask. "
                            "Wire to SaveImage or BD_PackChannels to use as a standalone cutout.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        candidates: torch.Tensor,
        face_oval: torch.Tensor,
        image: torch.Tensor | None = None,
        silhouette_mask: torch.Tensor | None = None,
        masked_image_bg: str = "transparent",
        left_eye:   torch.Tensor | None = None,
        right_eye:  torch.Tensor | None = None,
        left_brow:  torch.Tensor | None = None,
        right_brow: torch.Tensor | None = None,
        lips:       torch.Tensor | None = None,
        nose:       torch.Tensor | None = None,
        min_iou: float = 0.05,
        target_expand: int = 4,
        clip_expand: int = 4,
        feature_expand: int = 0,
        max_target_fill: float = 0.95,
        exclusive: bool = True,
        overlay_alpha: float = 0.55,
    ) -> io.NodeOutput:

        # Normalise face_oval → (H, W)
        face_oval_arr = _to_hw(face_oval)
        H, W = face_oval_arr.shape

        # Normalise candidates → list of (H, W) arrays
        cands_t = candidates
        if cands_t.ndim == 2:
            cands_t = cands_t.unsqueeze(0)
        cands_hw: list[np.ndarray] = (
            [_resize_to(_to_hw(cands_t[b:b+1]), H, W) for b in range(cands_t.shape[0])]
            if cands_t.ndim == 3 else []
        )

        # Normalise silhouette_mask → (H, W) or None
        sil_arr: np.ndarray | None = None
        if silhouette_mask is not None:
            sil_arr = _resize_to(_to_hw(silhouette_mask), H, W)

        # Prepare image for overlay and compositing — (H, W, 3) float32
        img_np: np.ndarray | None = None
        if image is not None:
            img_t = image.detach().cpu().float()
            if img_t.ndim == 4:
                img_t = img_t[0]
            if img_t.shape[-1] == 4:
                img_t = img_t[..., :3]
            img_np_raw = img_t.numpy().astype(np.float32)
            if img_np_raw.shape[:2] != (H, W):
                luma = _resize_to(img_np_raw.mean(axis=-1), H, W)
                img_np = np.stack([luma] * 3, axis=-1)
            else:
                img_np = img_np_raw

        feature_inputs: dict[str, torch.Tensor | None] = {
            "left_eye":   left_eye,
            "right_eye":  right_eye,
            "left_brow":  left_brow,
            "right_brow": right_brow,
            "lips":       lips,
            "nose":       nose,
        }

        blank = np.zeros((H, W), dtype=np.float32)
        refined: dict[str, np.ndarray] = {}
        used_cands: set[int] = set()
        info_lines: list[str] = []

        # Match features in priority order (highest priority → first exclusive pick)
        for feat_name, _color, _priority in sorted(_FEATURES, key=lambda x: x[2], reverse=True):
            t_tensor = feature_inputs.get(feat_name)
            if t_tensor is None:
                refined[feat_name] = blank.copy()
                continue

            t_arr = _resize_to(_to_hw(t_tensor), H, W)

            if not cands_hw:
                # No candidates — pass MediaPipe mask through unchanged
                refined[feat_name] = t_arr
                info_lines.append(f"  {feat_name}: no candidates → original")
                continue

            result, best_j, best_iou, note = _match_best(
                t_arr, cands_hw, used_cands, min_iou, target_expand,
                clip_expand, max_target_fill, exclusive,
            )

            if result is not None:
                if exclusive:
                    used_cands.add(best_j)
                refined[feat_name] = result
                info_lines.append(f"  {feat_name}: matched {note}")
            else:
                # Fallback: keep the original MediaPipe mask
                refined[feat_name] = t_arr
                info_lines.append(f"  {feat_name}: fallback → original ({note})")

        # Compute skin = face_oval − refined features
        skin_arr = face_oval_arr.copy()
        subtracted_names = []
        for feat_name, _color, _priority in _FEATURES:
            feat_mask = refined[feat_name]
            if feat_mask.max() < 0.01:
                continue
            sub_mask = _dilate(feat_mask, feature_expand) if feature_expand > 0 else feat_mask
            skin_arr = np.maximum(0.0, skin_arr - sub_mask)
            subtracted_names.append(feat_name)

        if subtracted_names:
            info_lines.append(f"  skin: face_oval − ({', '.join(subtracted_names)})")

        # Apply silhouette clip to all masks
        if sil_arr is not None:
            skin_arr = np.minimum(skin_arr, sil_arr)
            face_oval_arr = np.minimum(face_oval_arr, sil_arr)
            for feat_name in refined:
                refined[feat_name] = np.minimum(refined[feat_name], sil_arr)
            info_lines.append("  silhouette_mask: applied to all outputs")

        # Build debug overlay
        base_rgb = img_np if img_np is not None else None
        canvas = base_rgb.copy() if base_rgb is not None else np.zeros((H, W, 3), dtype=np.float32)

        # Skin layer (behind features)
        skin_binary = (skin_arr > 0.5).astype(np.float32)
        a = overlay_alpha * 0.65   # slightly dimmer for skin so features stand out
        for c, cv in enumerate(_SKIN_COLOR):
            canvas[..., c] = canvas[..., c] * (1 - skin_binary * a) + cv * skin_binary * a

        # Feature layers (on top)
        for feat_name, color, _priority in _FEATURES:
            feat_mask = refined[feat_name]
            binary = (feat_mask > 0.5).astype(np.float32)
            if binary.max() < 0.01:
                continue
            for c, cv in enumerate(color):
                canvas[..., c] = canvas[..., c] * (1 - binary * overlay_alpha) + cv * binary * overlay_alpha

        debug_overlay = torch.from_numpy(canvas.clip(0, 1)).unsqueeze(0)

        wired_count = sum(1 for v in feature_inputs.values() if v is not None)
        n_matched = sum(1 for l in info_lines if "matched" in l)
        header = (
            f"BD_FaceSkinRefine: {n_matched}/{wired_count} features matched "
            f"from {len(cands_hw)} candidates "
            f"(min_iou={min_iou}, target_expand={target_expand}, clip_expand={clip_expand})"
        )
        match_info = header + "\n" + "\n".join(info_lines)
        print(f"[BD_FaceSkinRefine] {match_info}", flush=True)

        # Union of all refined features (eyes + brows + lips + nose)
        union_features = np.zeros((H, W), dtype=np.float32)
        for feat_name in ["left_eye", "right_eye", "left_brow", "right_brow", "lips", "nose"]:
            union_features = np.maximum(union_features, refined[feat_name])

        # head_mask: outer boundary (silhouette or face_oval) WITH feature holes cut out.
        # This is the useful downstream mask — not the raw silhouette (which the user already has).
        outer_boundary = sil_arr if sil_arr is not None else face_oval_arr
        head_mask_arr = np.maximum(0.0, outer_boundary - union_features)

        # Masked image outputs — require image to be wired
        if img_np is not None:
            # masked_image: same alpha as head_mask (head with feature holes)
            masked_image = _composite(img_np, head_mask_arr.clip(0, 1), masked_image_bg)
            # masked_skin: face_oval minus features (the refined skin plate only, tighter boundary)
            masked_skin = _composite(img_np, skin_arr.clip(0, 1), masked_image_bg)
        else:
            channels = 4 if masked_image_bg == "transparent" else 3
            placeholder = torch.zeros((1, H, W, channels), dtype=torch.float32)
            masked_image = placeholder
            masked_skin  = placeholder

        return io.NodeOutput(
            _to_mask_tensor(skin_arr),
            _to_mask_tensor(refined["left_eye"]),
            _to_mask_tensor(refined["right_eye"]),
            _to_mask_tensor(refined["left_brow"]),
            _to_mask_tensor(refined["right_brow"]),
            _to_mask_tensor(refined["lips"]),
            _to_mask_tensor(refined["nose"]),
            debug_overlay,
            match_info,
            masked_image,
            masked_skin,
            _to_mask_tensor(head_mask_arr),
        )


FACE_SKIN_REFINE_V3_NODES = [BD_FaceSkinRefine]
FACE_SKIN_REFINE_NODES = {"BD_FaceSkinRefine": BD_FaceSkinRefine}
FACE_SKIN_REFINE_DISPLAY_NAMES = {"BD_FaceSkinRefine": "BD Face Skin Refine"}
