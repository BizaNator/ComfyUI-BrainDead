"""
BD_MaskCorrelate — match coarse guide masks to a batch of precise candidate masks by IoU.

Primary use case: MediaPipe gives you WHERE a face feature is (approximate convex-hull
polygon), SAM3 gives you WHAT'S in that region (pixel-accurate segment). This node
pairs each guide with the best-overlapping SAM3 segment, then combines them with the
chosen mode.

Algorithm (per target slot):
  1. Compute IoU between the target and every candidate in the batch.
  2. Accept the highest-IoU candidate if it meets min_iou threshold.
  3. Combine: replace | intersect | union | weighted_blend.
  4. If no candidate meets threshold → fall back (original target or blank).

Compared to BD_MaskResolver: no skin-tone heuristics, no priority weights, no fixed
category names. Pure geometric matching — works with any mask pair (face features,
body parts, object segments, etc.).
"""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_hw(mask: torch.Tensor) -> np.ndarray:
    """Normalise any mask tensor → (H, W) float32 in [0, 1]."""
    m = mask.detach().cpu().float()
    if m.ndim == 3:
        m = m[0]
    return m.numpy().astype(np.float32)


def _to_mask_tensor(arr: np.ndarray) -> torch.Tensor:
    """(H, W) float32 → (1, H, W) ComfyUI MASK tensor."""
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two (H, W) float masks, binarised at 0.5."""
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


def _combine(target: np.ndarray, candidate: np.ndarray, mode: str) -> np.ndarray:
    if mode == "replace":
        return candidate
    elif mode == "intersect":
        return np.minimum(target, candidate)
    elif mode == "union":
        return np.maximum(target, candidate)
    elif mode == "weighted_blend":
        # Average, biased toward candidate where candidate is confident
        conf = candidate.clip(0, 1)
        return target * (1.0 - conf * 0.5) + candidate * conf
    return candidate


# ── Node ─────────────────────────────────────────────────────────────────────

_N = 8   # max target slots


class BD_MaskCorrelate(io.ComfyNode):
    """
    Match coarse guide masks to precise candidate masks by IoU overlap.
    Wire MediaPipe feature masks as targets, SAM3 segment batch as candidates.
    For each target the best-overlapping candidate is found and combined with it.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        target_inputs = []
        for i in range(1, _N + 1):
            target_inputs.append(
                io.Mask.Input(
                    f"target_{i}", optional=True,
                    tooltip=f"Target slot {i} — coarse guide mask (e.g. from BD_MediaPipeFaceMask). "
                            f"Leave unwired to skip this slot.",
                )
            )

        return io.Schema(
            node_id="BD_MaskCorrelate",
            display_name="BD Mask Correlate",
            category="🧠BrainDead/Segmentation",
            description=(
                "Match coarse guide masks (targets) to a batch of precise candidate masks by IoU. "
                "For each wired target the candidate with highest overlap is selected and combined "
                "with the target using the chosen mode. Typical use: MediaPipe brow/eye/lip masks "
                "as targets, SAM3 segment batch as candidates → pixel-accurate feature masks."
            ),
            inputs=[
                io.Mask.Input(
                    "candidates",
                    tooltip="Batch of precise candidate masks (B, H, W) — e.g. SAM3 segment output. "
                            "Each frame in the batch is a separate candidate segment.",
                ),
                io.String.Input(
                    "labels", multiline=True,
                    default="left_brow\nright_brow\nleft_eye\nright_eye\nlips",
                    optional=True,
                    tooltip="One label per line, aligned with target_1..target_N slots. "
                            "Used only for the match_info status string.",
                ),
                io.Float.Input(
                    "min_iou", default=0.05, min=0.0, max=1.0, step=0.01,
                    optional=True,
                    tooltip="Minimum IoU for a candidate to be accepted as a match. "
                            "0.05 is permissive (any reasonable overlap counts). Raise to 0.2+ "
                            "if candidates are bleeding into the wrong target regions.",
                ),
                io.Combo.Input(
                    "mode",
                    options=["intersect", "replace", "union", "weighted_blend"],
                    default="intersect",
                    optional=True,
                    tooltip=(
                        "How to combine the accepted candidate with the original target:\n"
                        "  intersect — candidate clipped to target region (safest; prevents "
                        "candidate bleeding outside the coarse guide area).\n"
                        "  replace   — use the raw candidate mask (full SAM3 segment, ignores "
                        "target shape).\n"
                        "  union     — expand target by candidate shape (most generous).\n"
                        "  weighted_blend — smooth blend biased toward confident candidate areas."
                    ),
                ),
                io.Combo.Input(
                    "fallback",
                    options=["original", "blank"],
                    default="original",
                    optional=True,
                    tooltip=(
                        "What to output for a target slot when no candidate meets min_iou:\n"
                        "  original — return the unmodified target mask.\n"
                        "  blank    — return an empty mask (signals 'no confident match')."
                    ),
                ),
                io.Int.Input(
                    "target_expand", default=0, min=0, max=60, step=1,
                    optional=True,
                    tooltip="Pixels to dilate each target mask BEFORE computing IoU. "
                            "Useful when MediaPipe landmarks produce a tight hull that doesn't "
                            "fully overlap the actual SAM3 segment. 4–10 px is usually enough.",
                ),
                io.Boolean.Input(
                    "exclusive", default=False,
                    optional=True,
                    tooltip="When True each candidate can only be matched to ONE target "
                            "(assigned in slot order 1→N, highest IoU wins). "
                            "When False the same SAM3 segment can match multiple targets "
                            "(safe for non-overlapping features like left/right brow).",
                ),
                *target_inputs,
            ],
            outputs=[
                *[io.Mask.Output(display_name=f"refined_{i}",
                                 tooltip=f"Refined mask for target slot {i}. "
                                         f"Blank (zeros) if slot was not wired.")
                  for i in range(1, _N + 1)],
                io.String.Output(display_name="match_info",
                                 tooltip="Per-slot match summary: label, best candidate index, IoU, mode used."),
            ],
        )

    @classmethod
    def execute(
        cls,
        candidates: torch.Tensor,
        labels: str = "",
        min_iou: float = 0.05,
        mode: str = "intersect",
        fallback: str = "original",
        target_expand: int = 0,
        exclusive: bool = False,
        **kwargs,
    ) -> io.NodeOutput:

        # Parse targets from kwargs
        targets: list[torch.Tensor | None] = []
        for i in range(1, _N + 1):
            targets.append(kwargs.get(f"target_{i}"))

        wired = [(i, t) for i, t in enumerate(targets) if t is not None]

        # Parse labels
        label_list = [l.strip() for l in (labels or "").strip().split("\n") if l.strip()]

        # Normalise candidates to list of (H, W) float32 arrays
        cands_t = candidates
        if cands_t.ndim == 2:
            cands_t = cands_t.unsqueeze(0)
        if cands_t.ndim == 3:
            # (B, H, W) — each frame is one candidate
            cands_hw = [_to_hw(cands_t[b:b+1]) for b in range(cands_t.shape[0])]
        else:
            cands_hw = []

        blank_shape = None
        if wired:
            first = wired[0][1]
            m = first
            if m.ndim == 3:
                m = m[0]
            blank_shape = (int(m.shape[-2]), int(m.shape[-1]))
        elif cands_hw:
            blank_shape = cands_hw[0].shape
        else:
            blank_shape = (64, 64)

        H, W = blank_shape

        # Resize candidates to target resolution
        cands_hw = [_resize_to(c, H, W) for c in cands_hw]

        blank = np.zeros((H, W), dtype=np.float32)

        # Output slot arrays — initialised to blank
        out_masks: list[np.ndarray] = [blank.copy() for _ in range(_N)]
        info_lines: list[str] = []
        used_cands: set[int] = set()

        if not cands_hw:
            # No candidates — pass targets through
            for slot_i, t in wired:
                out_masks[slot_i] = _to_hw(t)
            match_info = "BD_MaskCorrelate: no candidates provided — targets passed through unchanged."
        else:
            for slot_i, t_tensor in wired:
                label = label_list[slot_i] if slot_i < len(label_list) else f"target_{slot_i+1}"
                t_arr = _resize_to(_to_hw(t_tensor), H, W)
                t_query = _dilate(t_arr, target_expand)

                # Find best candidate
                best_iou = -1.0
                best_j = -1
                for j, c in enumerate(cands_hw):
                    if exclusive and j in used_cands:
                        continue
                    iou = _iou(t_query, c)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j

                if best_j >= 0 and best_iou >= min_iou:
                    if exclusive:
                        used_cands.add(best_j)
                    refined = _combine(t_arr, cands_hw[best_j], mode)
                    out_masks[slot_i] = refined.clip(0, 1)
                    info_lines.append(
                        f"  {label}: matched candidate {best_j} (IoU={best_iou:.3f}, mode={mode})"
                    )
                else:
                    # No match
                    if fallback == "blank":
                        out_masks[slot_i] = blank.copy()
                        info_lines.append(
                            f"  {label}: no match (best IoU={best_iou:.3f} < {min_iou}) → blank"
                        )
                    else:
                        out_masks[slot_i] = t_arr
                        info_lines.append(
                            f"  {label}: no match (best IoU={best_iou:.3f} < {min_iou}) → original"
                        )

            n_matched = sum(1 for l in info_lines if "matched" in l)
            header = (
                f"BD_MaskCorrelate: {n_matched}/{len(wired)} slots matched "
                f"from {len(cands_hw)} candidates (min_iou={min_iou}, mode={mode})"
            )
            match_info = header + "\n" + "\n".join(info_lines)

        print(f"[BD_MaskCorrelate] {match_info}", flush=True)

        outputs = [_to_mask_tensor(out_masks[i]) for i in range(_N)]
        return io.NodeOutput(*outputs, match_info)


MASK_CORRELATE_V3_NODES = [BD_MaskCorrelate]
MASK_CORRELATE_NODES = {"BD_MaskCorrelate": BD_MaskCorrelate}
MASK_CORRELATE_DISPLAY_NAMES = {"BD_MaskCorrelate": "BD Mask Correlate"}
