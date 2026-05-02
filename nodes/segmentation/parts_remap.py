"""
BD_PartsRemap26to11 — translate a 26-tag SeeThrough parts dict into BrainPed's
11-part rig vocabulary (Head, Spine, Hips, RightArm, LeftArm, RightHand,
LeftHand, RightLeg, LeftLeg, RightFoot, LeftFoot).

Mirrors BrainPed's LAYER_TO_PART_MAP from rig_processor/layerdiff_mapping.py.

For L/R-ambiguous parts (topwear/bottomwear/legwear/footwear in SeeThrough),
the merged composite is masked to the BrainPed segment's own crop_bounds so
each side gets only its own pixels.

Output is the same SEETHROUGH_PARTS dict shape — just with BrainPed-style tag
names — so all downstream BD nodes (Iterator, Composite, ExtractSkinMask, etc.)
work unchanged.
"""

import copy
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from comfy_api.latest import io


SEETHROUGH_PARTS = "SEETHROUGH_PARTS"

# Mirror of BrainPed's rig_processor/layerdiff_mapping.py::LAYER_TO_PART_MAP.
# Keep in sync if BrainPed updates its taxonomy.
LAYER_TO_PART_MAP: dict[str, list[str]] = {
    "Head": [
        "back hair", "front hair", "hairb", "hairf",
        "face", "ears", "ears-l", "ears-r",
        "eyebrow-l", "eyebrow-r",
        "eyelash-l", "eyelash-r",
        "eyewhite-l", "eyewhite-r",
        "irides-l", "irides-r",
        "mouth", "nose", "neck",
    ],
    "Spine":     ["topwear", "neckwear"],
    "Hips":      ["bottomwear"],
    "RightArm":  ["topwear"],
    "LeftArm":   ["topwear"],
    "RightHand": ["handwear-r"],
    "LeftHand":  ["handwear-l"],
    "RightLeg":  ["legwear", "bottomwear"],
    "LeftLeg":   ["legwear", "bottomwear"],
    "RightFoot": ["footwear"],
    "LeftFoot":  ["footwear"],
}

DEFAULT_IGNORED_LAYERS = {"tail", "wings", "objects", "earwear"}

LR_AMBIGUOUS_PARTS = {
    "Spine", "RightArm", "LeftArm",
    "RightLeg", "LeftLeg", "RightFoot", "LeftFoot",
}

DEFAULT_SEGMENTS_DIR = "/opt/brainped-work/BrainPed_Rig_Processing/config/segments"
DEFAULT_CANVAS_SIZE = 4096


def _ensure_parts(parts):
    if not isinstance(parts, dict) or "tag2pinfo" not in parts:
        raise ValueError(
            f"Expected SEETHROUGH_PARTS wrapper, got {type(parts).__name__}"
        )
    return parts["tag2pinfo"], tuple(parts.get("frame_size", (0, 0)))


def _load_brainped_bounds(segments_dir: str) -> dict[str, tuple[int, int, int, int]]:
    """Read crop_bounds for each BrainPed part from config/segments/{Part}.json."""
    bounds: dict[str, tuple[int, int, int, int]] = {}
    p = Path(segments_dir)
    if not p.is_dir():
        return bounds
    for json_path in p.glob("*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            cb = data.get("crop_bounds")
            if cb and all(k in cb for k in ("left", "top", "right", "bottom")):
                bounds[json_path.stem] = (
                    int(cb["left"]), int(cb["top"]),
                    int(cb["right"]), int(cb["bottom"]),
                )
        except (json.JSONDecodeError, OSError):
            continue
    return bounds


def _composite_part(source_infos: list[dict], canvas_h: int, canvas_w: int,
                    src_frame_h: int, src_frame_w: int) -> tuple[np.ndarray, np.ndarray]:
    """Composite source SeeThrough parts onto a canvas at canvas_size, depth-sorted back-to-front.

    Scales each source part's xyxy from src_frame coords to canvas_size coords.
    Returns (rgba_uint8 (H, W, 4), skin_mask_uint8 (H, W)) — the skin mask is the union
    of any per-part skin_masks present in source_infos.
    """
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)
    skin_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    has_any_skin = False
    if not source_infos:
        return canvas.astype(np.uint8), skin_canvas.astype(np.uint8)

    sx = canvas_w / max(src_frame_w, 1)
    sy = canvas_h / max(src_frame_h, 1)

    ordered = sorted(source_infos, key=lambda d: -float(d.get("depth_median", 1.0)))
    for info in ordered:
        img = info.get("img")
        xyxy = info.get("xyxy")
        if img is None or xyxy is None:
            continue
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            continue
        x1, y1, x2, y2 = xyxy
        cx1 = max(0, int(round(x1 * sx)))
        cy1 = max(0, int(round(y1 * sy)))
        cx2 = min(canvas_w, int(round(x2 * sx)))
        cy2 = min(canvas_h, int(round(y2 * sy)))
        tw, th = cx2 - cx1, cy2 - cy1
        if tw <= 0 or th <= 0:
            continue
        if arr.shape[:2] != (th, tw):
            pil = Image.fromarray(arr.astype(np.uint8),
                                  mode="RGBA" if arr.shape[2] == 4 else "RGB")
            pil = pil.resize((tw, th), Image.LANCZOS)
            arr = np.asarray(pil)
        arr_f = arr.astype(np.float32) / 255.0
        if arr_f.shape[2] == 3:
            arr_f = np.concatenate([arr_f, np.ones((th, tw, 1), dtype=np.float32)], axis=-1)

        src_rgb = arr_f[..., :3]
        src_a = arr_f[..., 3:4]
        dst = canvas[cy1:cy2, cx1:cx2]
        dst_rgb = dst[..., :3]
        dst_a = dst[..., 3:4]
        out_a = src_a + dst_a * (1.0 - src_a)
        safe_a = np.maximum(out_a, 1e-6)
        canvas[cy1:cy2, cx1:cx2, :3] = (src_rgb * src_a + dst_rgb * dst_a * (1.0 - src_a)) / safe_a
        canvas[cy1:cy2, cx1:cx2, 3:4] = out_a

        sk = info.get("skin_mask")
        if sk is not None:
            sk_arr = np.asarray(sk).astype(np.float32) / 255.0
            if sk_arr.shape != (th, tw):
                sk_pil = Image.fromarray((sk_arr * 255).clip(0, 255).astype(np.uint8), mode="L").resize(
                    (tw, th), Image.NEAREST)
                sk_arr = np.asarray(sk_pil).astype(np.float32) / 255.0
            skin_canvas[cy1:cy2, cx1:cx2] = np.maximum(skin_canvas[cy1:cy2, cx1:cx2], sk_arr)
            has_any_skin = True

    rgba = (canvas * 255.0).clip(0, 255).astype(np.uint8)
    skin = (skin_canvas * 255.0).clip(0, 255).astype(np.uint8) if has_any_skin else np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    return rgba, skin


def _apply_bbox_mask(canvas_rgba: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Zero alpha outside bbox. canvas is HxWx4 uint8."""
    h, w = canvas_rgba.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    mask[y1:y2, x1:x2] = True
    out = canvas_rgba.copy()
    out[..., 3] = np.where(mask, out[..., 3], 0)
    return out


def _crop_to_bbox(canvas_rgba: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop canvas to bbox region, return cropped RGBA + the actual bbox used."""
    h, w = canvas_rgba.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 4), dtype=np.uint8), (0, 0, 1, 1)
    return canvas_rgba[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


class BD_PartsRemap26to11(io.ComfyNode):
    """Translate a SeeThrough 26-tag parts dict into BrainPed's 11-part rig vocabulary."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_PartsRemap26to11",
            display_name="BD Parts Remap 26 → 11 (BrainPed)",
            category="🧠BrainDead/Segmentation",
            description=(
                "Convert a SeeThrough-shaped SEETHROUGH_PARTS dict (26 fine-grained tags) into "
                "BrainPed's 11-part rig vocabulary (Head, Spine, Hips, R/L Arm/Hand/Leg/Foot). "
                "L/R-ambiguous parts (Spine, arms, legs, feet) get cropped to BrainPed's "
                "per-segment crop_bounds so each side gets only its own pixels."
            ),
            inputs=[
                io.Custom(SEETHROUGH_PARTS).Input("parts"),
                io.String.Input(
                    "segments_config_dir", default=DEFAULT_SEGMENTS_DIR, optional=True,
                    tooltip="Path to BrainPed's config/segments/ dir. Reads {Part}.json files for crop_bounds. "
                            "If not found, L/R-ambiguous parts will use a simple half-and-half heuristic.",
                ),
                io.Int.Input(
                    "canvas_size", default=DEFAULT_CANVAS_SIZE, min=512, max=8192, step=128, optional=True,
                    tooltip="Output canvas resolution (square). BrainPed expects 4096; SeeThrough natively emits 1024.",
                ),
                io.Boolean.Input(
                    "crop_to_part_bbox", default=True, optional=True,
                    tooltip="If True, each output part is cropped tight to its bbox (smaller img per part). "
                            "If False, each part is full-canvas with alpha defining the part region.",
                ),
                io.String.Input(
                    "ignored_layers", default=",".join(sorted(DEFAULT_IGNORED_LAYERS)),
                    optional=True, multiline=True,
                    tooltip="Comma-separated SeeThrough tags to drop entirely (not part of biped rig).",
                ),
            ],
            outputs=[
                io.Custom(SEETHROUGH_PARTS).Output(display_name="parts"),
                io.String.Output(display_name="summary"),
            ],
        )

    @classmethod
    def execute(cls, parts, segments_config_dir=DEFAULT_SEGMENTS_DIR,
                canvas_size=DEFAULT_CANVAS_SIZE, crop_to_part_bbox=True,
                ignored_layers=",".join(sorted(DEFAULT_IGNORED_LAYERS))) -> io.NodeOutput:
        tag2pinfo, frame_size = _ensure_parts(parts)
        if frame_size == (0, 0):
            raise ValueError("parts['frame_size'] missing — needed to scale into BrainPed canvas.")
        src_h, src_w = frame_size

        ignored = {t.strip() for t in (ignored_layers or "").split(",") if t.strip()}
        bounds = _load_brainped_bounds(segments_config_dir)
        bounds_loaded = len(bounds) > 0

        out_tag2pinfo: dict[str, dict] = {}
        summary_lines = [
            f"Source: {len(tag2pinfo)} tags @ {src_w}x{src_h} | "
            f"Target canvas: {canvas_size}x{canvas_size} | "
            f"Bounds loaded: {bounds_loaded} from {segments_config_dir if bounds_loaded else '(none)'}"
        ]

        for part_name, source_layer_names in LAYER_TO_PART_MAP.items():
            collected: list[dict] = []
            for layer_name in source_layer_names:
                if layer_name in ignored:
                    continue
                if layer_name in tag2pinfo:
                    collected.append(tag2pinfo[layer_name])

            if not collected:
                summary_lines.append(f"  {part_name:12s} ← (no source layers found)")
                continue

            canvas, skin_canvas = _composite_part(collected, canvas_size, canvas_size, src_h, src_w)

            if part_name in LR_AMBIGUOUS_PARTS and part_name in bounds:
                canvas = _apply_bbox_mask(canvas, bounds[part_name])
                bx1, by1, bx2, by2 = bounds[part_name]
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(canvas_size, bx2), min(canvas_size, by2)
                trim_mask = np.zeros_like(skin_canvas, dtype=bool)
                trim_mask[by1:by2, bx1:bx2] = True
                skin_canvas = np.where(trim_mask, skin_canvas, 0)

            alpha = canvas[..., 3]
            if not alpha.any():
                summary_lines.append(f"  {part_name:12s} ← empty after merge/crop")
                continue

            if crop_to_part_bbox:
                ys, xs = np.where(alpha > 0)
                if len(xs) == 0:
                    continue
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
                cropped = canvas[y1:y2, x1:x2]
                cropped_skin = skin_canvas[y1:y2, x1:x2]
                xyxy = (x1, y1, x2, y2)
            elif part_name in bounds:
                cropped, xyxy = _crop_to_bbox(canvas, bounds[part_name])
                cropped_skin = skin_canvas[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            else:
                cropped = canvas
                cropped_skin = skin_canvas
                xyxy = (0, 0, canvas_size, canvas_size)

            depth_medians = [float(s.get("depth_median", 0.5)) for s in collected
                             if s.get("depth_median") is not None]
            dm = float(np.mean(depth_medians)) if depth_medians else 0.5

            new_info = {
                "img": cropped,
                "xyxy": list(xyxy),
                "depth_median": dm,
                "tag": part_name,
                "_source_layers": [n for n in source_layer_names if n in tag2pinfo and n not in ignored],
            }
            if cropped_skin.any():
                new_info["skin_mask"] = cropped_skin

            out_tag2pinfo[part_name] = new_info
            skin_str = f" skin={100.0 * (cropped_skin > 0).mean():.1f}%" if cropped_skin.any() else ""
            summary_lines.append(
                f"  {part_name:12s} ← {new_info['_source_layers']} "
                f"→ {cropped.shape[1]}x{cropped.shape[0]} @ {xyxy}{skin_str}"
            )

        out = copy.copy(parts)
        out["tag2pinfo"] = out_tag2pinfo
        out["frame_size"] = (canvas_size, canvas_size)
        summary = "\n".join(summary_lines)
        print(f"[BD PartsRemap 26→11] {summary}", flush=True)
        return io.NodeOutput(out, summary)


PARTS_REMAP_V3_NODES = [BD_PartsRemap26to11]
PARTS_REMAP_NODES = {"BD_PartsRemap26to11": BD_PartsRemap26to11}
PARTS_REMAP_DISPLAY_NAMES = {"BD_PartsRemap26to11": "BD Parts Remap 26 → 11 (BrainPed)"}
