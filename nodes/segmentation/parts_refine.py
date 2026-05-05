"""
BD_PartsRefine — IoU-based deduplication of tagged masks.

When a VLM (QwenVL etc.) emits redundant/synonymous descriptors and they're
fed to SAM3 as separate prompts, you get overlapping masks pointing at the
same physical region (e.g. "shoe", "sneaker", "athletic shoe", "left shoe"
might all mask roughly the same pixels).

This node:
  1. Computes pairwise IoU across all input masks.
  2. Groups masks whose IoU exceeds `iou_threshold` into clusters.
  3. Per cluster, picks a canonical tag (shortest by default — heuristic for
     "more general"; ties broken by first-encountered) and merges geometry
     per `merge_strategy`.
  4. Optionally caps total kept parts at `max_parts` (largest area first).
  5. Emits a color-coded debug overlay so you can sanity-check the refinement.
"""

import re

import numpy as np
import torch

from comfy_api.latest import io


def _iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> float:
    a_b = (a > 0.5).float()
    b_b = (b > 0.5).float()
    inter = (a_b * b_b).sum().item()
    union = (a_b + b_b).clamp(0, 1).sum().item()
    return inter / max(union, eps)


def _normalize_masks(masks: torch.Tensor) -> torch.Tensor:
    m = masks
    if m.dim() == 4 and m.shape[-1] == 1:
        m = m[..., 0]
    if m.dim() == 4:
        m = m.reshape(-1, m.shape[-2], m.shape[-1])
    if m.dim() == 2:
        m = m.unsqueeze(0)
    return m.float()


def _cluster_by_iou(masks: torch.Tensor, threshold: float) -> list[list[int]]:
    """Union-find clustering: index i and j join if IoU(masks[i], masks[j]) >= threshold."""
    n = masks.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if _iou(masks[i], masks[j]) >= threshold:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def _safe_label(s: str) -> str:
    return re.sub(r"\s+", "_", s.strip()) or "part"


def _pick_canonical(indices: list[int], labels: list[str], areas: list[int]) -> int:
    """Pick the index whose label is most general (shortest, alphabetically first on tie)."""
    best = indices[0]
    for i in indices[1:]:
        if (len(labels[i]), labels[i]) < (len(labels[best]), labels[best]):
            best = i
    return best


def _merge_masks(masks: torch.Tensor, indices: list[int], strategy: str) -> torch.Tensor:
    if strategy == "keep_largest":
        areas = [(masks[i] > 0.5).sum().item() for i in indices]
        return masks[indices[int(np.argmax(areas))]]
    # union: pixel-wise max
    stack = torch.stack([masks[i] for i in indices], dim=0)
    return stack.max(dim=0).values


def _color_for(idx: int) -> np.ndarray:
    """Deterministic color from index — hash-based HSV → RGB."""
    h = (idx * 0.6180339887) % 1.0  # golden ratio for good spread
    s, v = 0.85, 0.95
    h6 = h * 6.0
    c = v * s
    x = c * (1 - abs((h6 % 2) - 1))
    m = v - c
    if h6 < 1:    r, g, b = c, x, 0
    elif h6 < 2:  r, g, b = x, c, 0
    elif h6 < 3:  r, g, b = 0, c, x
    elif h6 < 4:  r, g, b = 0, x, c
    elif h6 < 5:  r, g, b = x, 0, c
    else:         r, g, b = c, 0, x
    return np.array([(r + m), (g + m), (b + m)], dtype=np.float32)


def _build_debug_overlay(image: torch.Tensor, refined_masks: torch.Tensor,
                         labels: list[str], alpha: float = 0.55) -> torch.Tensor:
    """Paint each refined mask with a unique color overlaid on the source image."""
    img = image
    if img.dim() == 4:
        img = img[0]
    img_np = img.detach().cpu().numpy().astype(np.float32)
    if img_np.shape[-1] == 4:
        img_np = img_np[..., :3]
    H, W = img_np.shape[:2]

    out = img_np.copy()
    if refined_masks.shape[-2:] != (H, W):
        refined_masks = torch.nn.functional.interpolate(
            refined_masks.unsqueeze(0), size=(H, W), mode="nearest"
        )[0]

    for i, m in enumerate(refined_masks):
        color = _color_for(i)
        m_np = (m.detach().cpu().numpy() > 0.5).astype(np.float32)
        if m_np.sum() == 0:
            continue
        out = np.where(
            m_np[..., None] > 0,
            out * (1 - alpha) + color[None, None, :] * alpha,
            out,
        )
    return torch.from_numpy(out.clip(0, 1).astype(np.float32)).unsqueeze(0)


class BD_PartsRefine(io.ComfyNode):
    """Dedupe overlapping tagged masks via IoU clustering."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_PartsRefine",
            display_name="BD Parts Refine",
            category="🧠BrainDead/Segmentation",
            description=(
                "Cluster tagged masks by pairwise IoU. Per cluster, pick a canonical tag "
                "(shortest label = most general) and merge geometry (union or keep_largest). "
                "Use to dedupe SAM3 outputs from synonymous VLM descriptors "
                "(e.g. 'shoe' + 'sneaker' + 'left shoe' → one entry).\n\n"
                "Optional max_parts caps the kept count by area (largest first). "
                "Optional debug overlay paints each refined mask with a unique color."
            ),
            inputs=[
                io.Mask.Input("masks", tooltip="MASK batch (B, H, W) — one per input prompt."),
                io.String.Input(
                    "labels", multiline=True, default="",
                    tooltip="Parallel labels, one per line. Auto-named part_NN if shorter than batch.",
                ),
                io.Float.Input("iou_threshold", default=0.7, min=0.0, max=1.0, step=0.05,
                               tooltip="Masks with IoU above this merge into one cluster. "
                                       "Lower = more aggressive dedup. 0.7 is a good starting point."),
                io.Combo.Input(
                    "merge_strategy", options=["union", "keep_largest"], default="union",
                    tooltip="union: pixel-wise max across cluster (more inclusive). "
                            "keep_largest: pick the cluster member with the most pixels (cleaner edges)."
                ),
                io.Int.Input("max_parts", default=0, min=0, max=64,
                             tooltip="Cap total refined parts (largest area kept). 0 = no cap."),
                io.Int.Input("min_pixels", default=64, min=1, max=10_000_000,
                             tooltip="Drop masks smaller than this before clustering. "
                                     "Filters noise/false-positives from low-confidence prompts."),
                io.Image.Input("image", optional=True,
                               tooltip="Optional source IMAGE — required if output_debug_viz=True."),
                io.Boolean.Input("output_debug_viz", default=False, optional=True,
                                 tooltip="Render a color-coded overlay of the refined masks "
                                         "on the source image. Requires `image` wired."),
            ],
            outputs=[
                io.Mask.Output(display_name="refined_masks"),
                io.String.Output(display_name="refined_labels"),
                io.Image.Output(display_name="debug_overlay"),
                io.String.Output(display_name="summary"),
            ],
        )

    @classmethod
    def execute(cls, masks, labels="", iou_threshold=0.7, merge_strategy="union",
                max_parts=0, min_pixels=64, image=None,
                output_debug_viz=False) -> io.NodeOutput:
        m = _normalize_masks(masks)
        n = m.shape[0]

        raw_labels = [l.strip() for l in (labels or "").strip().split("\n") if l.strip()]
        names: list[str] = []
        for i in range(n):
            names.append(raw_labels[i] if i < len(raw_labels) else f"part_{i:02d}")

        if len(raw_labels) != n:
            print(
                f"[BD PartsRefine] WARNING: got {n} masks but {len(raw_labels)} labels — "
                f"label↔mask alignment is broken. Likely causes:\n"
                f"  1. SAM3 returned multi-instance for one prompt (now patched in BD_SAM3MultiPrompt — restart dev).\n"
                f"  2. Wrong STRING wired to `labels` (must be the SAME prompts list given to SAM3, "
                f"     OR the refined_labels output from a previous PartsRefine).\n"
                f"  3. CSV used in labels — must be NEWLINE-separated.\n"
                f"  Auto-named extras: {[f'part_{i:02d}' for i in range(len(raw_labels), n)]}",
                flush=True,
            )

        # Drop below min_pixels first
        keep_idx = []
        dropped = []
        for i in range(n):
            area = int((m[i] > 0.5).sum().item())
            if area >= min_pixels:
                keep_idx.append(i)
            else:
                dropped.append((names[i], area))

        if not keep_idx:
            empty = torch.zeros((0, m.shape[1], m.shape[2]), dtype=torch.float32)
            blank = torch.zeros((1, m.shape[1], m.shape[2], 3), dtype=torch.float32)
            return io.NodeOutput(empty, "", blank,
                                 f"BD PartsRefine: no masks survived min_pixels={min_pixels}. "
                                 f"Dropped: {[d[0] for d in dropped]}")

        m_kept = m[keep_idx]
        names_kept = [names[i] for i in keep_idx]
        areas_kept = [int((m_kept[i] > 0.5).sum().item()) for i in range(len(keep_idx))]

        clusters = _cluster_by_iou(m_kept, iou_threshold)

        refined_masks = []
        refined_labels = []
        cluster_info = []
        for cluster in clusters:
            canonical = _pick_canonical(cluster, names_kept, areas_kept)
            merged = _merge_masks(m_kept, cluster, merge_strategy)
            refined_masks.append(merged)
            refined_labels.append(_safe_label(names_kept[canonical]))
            cluster_info.append((canonical, cluster, int((merged > 0.5).sum().item())))

        # Sort by area desc, apply max_parts cap
        order = sorted(range(len(refined_masks)),
                       key=lambda i: -cluster_info[i][2])
        if max_parts > 0:
            order = order[:max_parts]

        final_masks = torch.stack([refined_masks[i] for i in order], dim=0)
        final_labels = [refined_labels[i] for i in order]
        final_info = [cluster_info[i] for i in order]

        debug = torch.zeros((1, m.shape[1], m.shape[2], 3), dtype=torch.float32)
        if output_debug_viz and image is not None:
            debug = _build_debug_overlay(image, final_masks, final_labels)

        summary_lines = [
            f"BD PartsRefine: {n} → {len(final_masks)} parts "
            f"(iou>={iou_threshold}, strategy={merge_strategy}, max={max_parts or '∞'})",
        ]
        for label, (canon_idx, cluster_idx, area) in zip(final_labels, final_info):
            members = [names_kept[i] for i in cluster_idx]
            summary_lines.append(
                f"  {label:24s} ← {len(cluster_idx)} mask(s) {members}  area={area}"
            )
        if dropped:
            summary_lines.append(
                f"  Dropped {len(dropped)} below min_pixels={min_pixels}: "
                f"{[d[0] for d in dropped]}"
            )
        summary = "\n".join(summary_lines)
        labels_str = "\n".join(final_labels)

        print(f"[BD PartsRefine] {summary}", flush=True)
        return io.NodeOutput(final_masks, labels_str, debug, summary)


PARTS_REFINE_V3_NODES = [BD_PartsRefine]
PARTS_REFINE_NODES = {"BD_PartsRefine": BD_PartsRefine}
PARTS_REFINE_DISPLAY_NAMES = {"BD_PartsRefine": "BD Parts Refine"}
