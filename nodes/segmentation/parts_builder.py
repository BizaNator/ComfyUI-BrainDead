"""
BD_PartsBuilder — crop, label, resize, and build a PARTS_BUNDLE from a
segmentation source (SAM3 multi-prompt is primary; Sapiens2 supported via
the optional sapiens2_labels input).

PARTS_BUNDLE feeds the rest of the BD_Parts* pipeline:
  PartsBuilder → PartsRefine → PartsCompose / PartsExport / PartsBatchEdit

Auxiliary outputs:
  image_batch + label_list — parallel per-part IMAGE batch (padded uniform
  shape) and newline-separated labels. Useful for batch upscale + BD_BulkSave
  workflows that don't need the structured bundle.

Selection presets (Sapiens2 taxonomy, used when sapiens2_labels is wired):
  - all       : every class except Background
  - clothing  : Apparel, Eyeglass, Lower_Clothing, Upper_Clothing,
                Left_Shoe, Right_Shoe, Left_Sock, Right_Sock
  - bodyparts : Face_Neck, Hair, Torso, all L/R limbs, lips/teeth/tongue
  - specific  : whitelist from selection_list (csv or newline)
For SAM3 source: use 'all' (or 'specific' with a whitelist). Presets only
match Sapiens2's 29-class names.

Crop: bbox + N px padding. Resize: native or longest-side N (preserves
aspect). Square-pad option centers each crop in a square canvas — required
to batch parts of different aspect ratios into a single tensor.
"""

import json
import re

import numpy as np
import torch
from PIL import Image

from comfy_api.latest import io

from .parts_types import PARTS_BUNDLE


SAPIENS2_LABELS = "SAPIENS2_LABELS"

# Mirror of Sapiens2 SEG_PARTS taxonomy. If Sapiens2 ever extends this list,
# update both entries. Index = class id used by Sapiens2 dense head.
SAPIENS2_SEG_PARTS = (
    "Background", "Apparel", "Eyeglass", "Face_Neck", "Hair",
    "Left_Foot", "Left_Hand", "Left_Lower_Arm", "Left_Lower_Leg",
    "Left_Shoe", "Left_Sock", "Left_Upper_Arm", "Left_Upper_Leg",
    "Lower_Clothing", "Right_Foot", "Right_Hand", "Right_Lower_Arm",
    "Right_Lower_Leg", "Right_Shoe", "Right_Sock", "Right_Upper_Arm",
    "Right_Upper_Leg", "Torso", "Upper_Clothing",
    "Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue",
)

CLOTHING_LABELS = (
    "Apparel", "Eyeglass", "Lower_Clothing", "Upper_Clothing",
    "Left_Shoe", "Right_Shoe", "Left_Sock", "Right_Sock",
)

BODYPARTS_LABELS = (
    "Face_Neck", "Hair", "Torso",
    "Left_Upper_Arm", "Left_Lower_Arm", "Left_Hand",
    "Right_Upper_Arm", "Right_Lower_Arm", "Right_Hand",
    "Left_Upper_Leg", "Left_Lower_Leg", "Left_Foot",
    "Right_Upper_Leg", "Right_Lower_Leg", "Right_Foot",
    "Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue",
)

OUTPUT_SIZE_PRESETS = {"native": 0, "512": 512, "1024": 1024, "2048": 2048}


def _parse_hex(s: str) -> tuple[float, float, float]:
    s = (s or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6 or not all(c in "0123456789abcdefABCDEF" for c in s):
        return (0.5, 0.5, 0.5)
    return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)


def _bg_color(background: str, hex_str: str) -> tuple[float, float, float]:
    if background == "white":
        return (1.0, 1.0, 1.0)
    if background == "black":
        return (0.0, 0.0, 0.0)
    if background == "hex":
        return _parse_hex(hex_str)
    return (0.0, 0.0, 0.0)


def _resolve_selection(selection_mode: str, selection_list: str,
                       available_labels: list[str]) -> list[str]:
    if selection_mode == "clothing":
        wanted = list(CLOTHING_LABELS)
    elif selection_mode == "bodyparts":
        wanted = list(BODYPARTS_LABELS)
    elif selection_mode == "specific":
        raw = re.split(r"[,\n]", selection_list or "")
        wanted = [w.strip() for w in raw if w.strip()]
    else:
        wanted = [l for l in available_labels if l != "Background"]
    avail_set = set(available_labels)
    return [w for w in wanted if w in avail_set]


def _per_class_masks_from_labels(labels) -> tuple[torch.Tensor, list[str]]:
    """From SAPIENS2_LABELS dict, return (masks (C,H,W) float, label_names list).

    Honors `selected_part_ids` if present (advanced node). Always drops Background.
    """
    if not isinstance(labels, dict) or "class_ids" not in labels:
        raise ValueError("sapiens2_labels: expected SAPIENS2_LABELS dict with 'class_ids'.")
    class_ids = labels["class_ids"]
    if not isinstance(class_ids, torch.Tensor):
        class_ids = torch.as_tensor(class_ids)
    if class_ids.dim() == 4:
        class_ids = class_ids[0, 0]
    elif class_ids.dim() == 3:
        class_ids = class_ids[0]
    parts = labels.get("parts", SAPIENS2_SEG_PARTS)
    selected_ids = labels.get("selected_part_ids")
    if selected_ids:
        ids_to_emit = [int(i) for i in selected_ids]
    else:
        ids_to_emit = list(range(1, len(parts)))
    masks = []
    names = []
    cid = class_ids.long()
    for pid in ids_to_emit:
        if pid < 0 or pid >= len(parts):
            continue
        m = (cid == pid).float()
        masks.append(m)
        names.append(parts[pid])
    if not masks:
        return torch.zeros((0, cid.shape[0], cid.shape[1])), []
    return torch.stack(masks, dim=0), names


def _per_class_masks_from_batch(masks: torch.Tensor, labels_str: str) -> tuple[torch.Tensor, list[str]]:
    if masks is None:
        raise ValueError("masks input is required when sapiens2_labels is not wired.")
    m = masks
    if m.dim() == 4 and m.shape[-1] == 1:
        m = m[..., 0]
    if m.dim() == 4:
        m = m.reshape(-1, m.shape[-2], m.shape[-1])
    if m.dim() == 2:
        m = m.unsqueeze(0)
    raw = re.split(r"[,\n]", labels_str or "")
    names = [n.strip() for n in raw if n.strip()]
    if len(names) < m.shape[0]:
        names += [f"part_{i:02d}" for i in range(len(names), m.shape[0])]
    elif len(names) > m.shape[0]:
        names = names[: m.shape[0]]
    return m.float(), names


def _bbox_of(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    nz = (mask > 0.5).nonzero()
    if nz.numel() == 0:
        return None
    y_min = int(nz[:, 0].min())
    y_max = int(nz[:, 0].max())
    x_min = int(nz[:, 1].min())
    x_max = int(nz[:, 1].max())
    return (x_min, y_min, x_max + 1, y_max + 1)


def _resize_keep_aspect(rgba: np.ndarray, target_long: int) -> np.ndarray:
    h, w = rgba.shape[:2]
    if max(h, w) == target_long:
        return rgba
    scale = target_long / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    if rgba.shape[2] == 4:
        pil = Image.fromarray(rgba.astype(np.uint8), mode="RGBA")
    else:
        pil = Image.fromarray(rgba.astype(np.uint8), mode="RGB")
    pil = pil.resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(pil)


def _pad_canvas(rgba: np.ndarray, out_h: int, out_w: int,
                bg_rgb: tuple[float, float, float], alpha_mode: bool) -> np.ndarray:
    h, w = rgba.shape[:2]
    if h == out_h and w == out_w:
        return rgba
    c = rgba.shape[2]
    if alpha_mode:
        canvas = np.zeros((out_h, out_w, c), dtype=np.uint8)
    else:
        bg = np.array([int(round(v * 255)) for v in bg_rgb], dtype=np.uint8)
        canvas = np.tile(bg.reshape(1, 1, 3), (out_h, out_w, 1))
        if c == 4:
            canvas = np.concatenate([canvas, np.full((out_h, out_w, 1), 255, dtype=np.uint8)], axis=-1)
    top = max(0, (out_h - h) // 2)
    left = max(0, (out_w - w) // 2)
    canvas[top:top + h, left:left + w] = rgba
    return canvas


def _composite(image_rgb: np.ndarray, mask_2d: np.ndarray,
               background: str, bg_rgb: tuple[float, float, float]) -> np.ndarray:
    """image_rgb: HxWx3 float [0,1]. mask_2d: HxW float [0,1].
    Returns HxWx{3 or 4} uint8.
    """
    if background == "alpha":
        rgba = np.concatenate([image_rgb, mask_2d[..., None]], axis=-1)
        return (rgba * 255.0).clip(0, 255).astype(np.uint8)
    bg = np.array(bg_rgb, dtype=np.float32).reshape(1, 1, 3)
    a = mask_2d[..., None]
    comp = image_rgb * a + bg * (1.0 - a)
    return (comp * 255.0).clip(0, 255).astype(np.uint8)


class BD_PartsBuilder(io.ComfyNode):
    """Build a PARTS_BUNDLE from a segmentation source (SAM3 batch or Sapiens2)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_PartsBuilder",
            display_name="BD Parts Builder",
            category="🧠BrainDead/Segmentation",
            description=(
                "Build a PARTS_BUNDLE from per-prompt SAM3 masks (preferred) or Sapiens2 labels. "
                "Each part gets its own RGBA crop + bbox + tag + per-part depth (when depth_image "
                "wired). Primary output is `parts` for the BD_Parts* pipeline. Auxiliary outputs "
                "(image_batch, label_list, bbox_list) feed BD_BulkSave / ImageUpscaleWithModel "
                "for batch-only workflows.\n\n"
                "Selection presets only match Sapiens2's 29-class taxonomy. For SAM3 source "
                "use 'all' or 'specific'."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Custom(SAPIENS2_LABELS).Input(
                    "sapiens2_labels", optional=True,
                    tooltip="Preferred input from Sapiens2Segmentation. Carries class_ids tensor "
                            "+ part name taxonomy. When wired, `masks` and `mask_labels` are ignored.",
                ),
                io.Mask.Input(
                    "masks", optional=True,
                    tooltip="Generic MASK batch (B,H,W) — one mask per class. Used when "
                            "sapiens2_labels is not wired (e.g. SAM3 multi-prompt output).",
                ),
                io.String.Input(
                    "mask_labels", multiline=True, default="", optional=True,
                    tooltip="Parallel labels for `masks` (newline or comma separated). "
                            "Auto-named part_00, part_01, … if shorter than the batch.",
                ),
                io.Image.Input(
                    "depth_image", optional=True,
                    tooltip="Optional depth IMAGE (e.g. from BD_Lotus2Predict.raw_linear). When wired, "
                            "each part's depth_median is computed from the median depth value within its "
                            "mask region (used by Composite for back-to-front sort), AND the cropped depth "
                            "map for that part is stored in the parts dict so Export can save it. Use the "
                            "raw_linear output (0=near, 1=far) for proper sorting.",
                ),
                io.Mask.Input(
                    "combined_mask", optional=True,
                    tooltip="Optional silhouette mask. When wired, every per-class/per-prompt mask is "
                            "intersected with this before bbox + crop. Sources:\n"
                            "  • BD_SAM3MultiPrompt.combined_mask — drops pixels removed by negatives / "
                            "vote / color filter. The authoritative final layer.\n"
                            "  • Sapiens2SegmentationAdvanced.foreground_mask — confines per-class to body.\n\n"
                            "WARNING: do NOT wire an inverted mask here. If SAM3's invert_combined=True "
                            "is set upstream, you'll get the BACKGROUND silhouette — every real part will "
                            "be zeroed and you'll see only a single full-image leftover.",
                ),
                io.Combo.Input(
                    "selection_mode",
                    options=["all", "clothing", "bodyparts", "specific"],
                    default="all",
                    tooltip="all = every class except Background. clothing/bodyparts = curated "
                            "subsets (Sapiens2 taxonomy). specific = whitelist from selection_list.",
                ),
                io.String.Input(
                    "selection_list", multiline=True, default="", optional=True,
                    tooltip="When selection_mode=specific: comma or newline separated label names "
                            "to include (e.g. 'Left_Shoe,Right_Shoe,Upper_Clothing').",
                ),
                io.Int.Input("padding_pixels", default=5, min=0, max=512,
                             tooltip="Pixels of padding around each part's bbox (in source resolution)."),
                io.Combo.Input(
                    "output_size", options=list(OUTPUT_SIZE_PRESETS.keys()) + ["custom"],
                    default="native",
                    tooltip="native = keep crop's natural size (per-part may differ — batched by "
                            "padding to common max). 512/1024/2048 = scale longest side, preserve aspect.",
                ),
                io.Int.Input("output_size_custom", default=1024, min=64, max=8192, optional=True,
                             tooltip="Used when output_size=custom."),
                io.Combo.Input(
                    "square_pad", options=["pad_to_square", "none"],
                    default="pad_to_square",
                    tooltip="pad_to_square: center each crop in NxN canvas (uniform aspect, "
                            "ideal for upscaler batching and T3D). none: keep aspect, pad batch to "
                            "the rectangular common-bbox of all crops.",
                ),
                io.Combo.Input(
                    "background", options=["alpha", "white", "black", "hex"],
                    default="alpha",
                    tooltip="alpha = RGBA, mask baked to alpha channel (best for T3D). "
                            "white/black/hex = composite onto solid background (best for 2D pipeline).",
                ),
                io.String.Input("background_hex", default="#888888", optional=True,
                                tooltip="Hex color when background=hex. Accepts #RGB or #RRGGBB."),
                io.Int.Input("min_pixels", default=64, min=1, max=10_000_000,
                             tooltip="Drop parts whose mask area is fewer than this many pixels."),
            ],
            outputs=[
                io.Image.Output(display_name="image_batch"),
                io.String.Output(display_name="label_list"),
                io.String.Output(display_name="bbox_list"),
                io.Custom(PARTS_BUNDLE).Output(display_name="parts"),
            ],
        )

    @classmethod
    def execute(cls, image, sapiens2_labels=None, masks=None, mask_labels="",
                depth_image=None, combined_mask=None,
                selection_mode="all", selection_list="",
                padding_pixels=5, output_size="native", output_size_custom=1024,
                square_pad="pad_to_square", background="alpha", background_hex="#888888",
                min_pixels=64) -> io.NodeOutput:

        if image.dim() == 4:
            img = image[0]
        else:
            img = image
        img_np = img.detach().cpu().numpy().astype(np.float32)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        if img_np.shape[-1] == 4:
            img_np = img_np[..., :3]
        H, W = img_np.shape[:2]

        if sapiens2_labels is not None:
            class_masks, names = _per_class_masks_from_labels(sapiens2_labels)
            available = list(sapiens2_labels.get("parts", SAPIENS2_SEG_PARTS))
        else:
            class_masks, names = _per_class_masks_from_batch(masks, mask_labels)
            available = list(names)

        wanted = _resolve_selection(selection_mode, selection_list, available)
        if not wanted:
            print(f"[BD BulkMaskExtract] No labels matched selection_mode='{selection_mode}'. "
                  f"Available: {available}", flush=True)
            empty = torch.zeros((1, 1, 1, 4 if background == "alpha" else 3), dtype=torch.float32)
            return io.NodeOutput(empty, "", "[]")

        if output_size == "custom":
            target_long = int(output_size_custom)
        else:
            target_long = OUTPUT_SIZE_PRESETS[output_size]

        bg_rgb = _bg_color(background, background_hex)
        alpha_mode = (background == "alpha")

        if class_masks.shape[-2:] != (H, W):
            mh, mw = class_masks.shape[-2:]
            class_masks = torch.nn.functional.interpolate(
                class_masks.unsqueeze(0), size=(H, W), mode="nearest"
            )[0]

        if combined_mask is not None:
            clip = combined_mask
            if clip.dim() == 4 and clip.shape[-1] == 1:
                clip = clip[..., 0]
            if clip.dim() == 4:
                clip = clip[0, 0]
            elif clip.dim() == 3:
                clip = clip[0]
            if clip.shape[-2:] != (H, W):
                clip = torch.nn.functional.interpolate(
                    clip.unsqueeze(0).unsqueeze(0), size=(H, W), mode="nearest"
                )[0, 0]
            class_masks = class_masks * clip.unsqueeze(0).to(class_masks.dtype)

        depth_np = None
        if depth_image is not None:
            d = depth_image
            if d.dim() == 4:
                d = d[0]
            d_np = d.detach().cpu().numpy().astype(np.float32)
            if d_np.ndim == 3 and d_np.shape[-1] >= 3:
                d_np = d_np[..., :3].mean(axis=-1)
            elif d_np.ndim == 3:
                d_np = d_np[..., 0]
            if d_np.shape[:2] != (H, W):
                pil = Image.fromarray((d_np * 255.0).clip(0, 255).astype(np.uint8), mode="L")
                pil = pil.resize((W, H), Image.BILINEAR)
                d_np = np.asarray(pil).astype(np.float32) / 255.0
            depth_np = d_np

        crops = []
        out_labels = []
        out_bboxes = []
        skipped = []
        tag2pinfo: dict[str, dict] = {}

        name_to_idx = {n: i for i, n in enumerate(names)}
        for label in wanted:
            idx = name_to_idx.get(label)
            if idx is None:
                continue
            mask = class_masks[idx]
            area = int((mask > 0.5).sum().item())
            if area < min_pixels:
                skipped.append((label, area))
                continue
            bb = _bbox_of(mask)
            if bb is None:
                continue
            x1, y1, x2, y2 = bb
            x1 = max(0, x1 - padding_pixels)
            y1 = max(0, y1 - padding_pixels)
            x2 = min(W, x2 + padding_pixels)
            y2 = min(H, y2 + padding_pixels)
            if x2 <= x1 or y2 <= y1:
                continue
            sub_img = img_np[y1:y2, x1:x2, :]
            sub_mask = mask[y1:y2, x1:x2].detach().cpu().numpy().astype(np.float32)

            # Native-size RGBA crop for the parts wrapper (Iterator/GetPart/SetPart consume this).
            # Always RGBA with mask baked to alpha — matches SeeThrough's per-part shape.
            native_rgba = _composite(sub_img, sub_mask, "alpha", (0.0, 0.0, 0.0))
            part_info: dict = {
                "img": native_rgba,
                "xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "tag": label,
                "depth_median": 0.5,
            }

            if depth_np is not None:
                sub_depth = depth_np[y1:y2, x1:x2]
                masked_vals = sub_depth[sub_mask > 0.5]
                if masked_vals.size:
                    part_info["depth_median"] = float(np.median(masked_vals))
                # Store grayscale depth crop (uint8) for Export.save_depth.
                part_info["depth"] = (sub_depth * 255.0).clip(0, 255).astype(np.uint8)

            tag2pinfo[label] = part_info

            comp = _composite(sub_img, sub_mask, background, bg_rgb)
            if target_long > 0:
                comp = _resize_keep_aspect(comp, target_long)

            crops.append(comp)
            out_labels.append(label)
            out_bboxes.append([int(x1), int(y1), int(x2), int(y2)])

        if not crops:
            print(f"[BD BulkMaskExtract] No usable crops after filtering. "
                  f"Skipped (below min_pixels): {skipped}", flush=True)
            empty_c = 4 if alpha_mode else 3
            empty = torch.zeros((1, 1, 1, empty_c), dtype=torch.float32)
            empty_parts = {"tag2pinfo": {}, "frame_size": (H, W)}
            return io.NodeOutput(empty, "", "[]", empty_parts)

        if square_pad == "pad_to_square":
            side = max(max(c.shape[0], c.shape[1]) for c in crops)
            out_h = out_w = side
        else:
            out_h = max(c.shape[0] for c in crops)
            out_w = max(c.shape[1] for c in crops)

        padded = [_pad_canvas(c, out_h, out_w, bg_rgb, alpha_mode) for c in crops]
        batch = np.stack(padded, axis=0).astype(np.float32) / 255.0
        batch_t = torch.from_numpy(batch)

        labels_str = "\n".join(out_labels)
        bbox_str = json.dumps(out_bboxes)

        skipped_str = (f" | skipped {len(skipped)} below min_pixels: "
                       f"{[s[0] for s in skipped]}") if skipped else ""
        clip_str = " | combined_mask=on" if combined_mask is not None else ""
        print(
            f"[BD BulkMaskExtract] {len(crops)} parts @ {out_h}x{out_w} "
            f"({'RGBA' if alpha_mode else 'RGB'}, {background}, "
            f"target_long={target_long or 'native'}, square={square_pad})"
            f"{clip_str}{skipped_str}",
            flush=True,
        )
        parts_bundle = {"tag2pinfo": tag2pinfo, "frame_size": (H, W)}
        return io.NodeOutput(batch_t, labels_str, bbox_str, parts_bundle)


PARTS_BUILDER_V3_NODES = [BD_PartsBuilder]
PARTS_BUILDER_NODES = {"BD_PartsBuilder": BD_PartsBuilder}
PARTS_BUILDER_DISPLAY_NAMES = {"BD_PartsBuilder": "BD Parts Builder"}
