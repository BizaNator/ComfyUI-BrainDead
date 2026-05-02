"""
SeeThrough parts pipeline: extract per-part skin masks, get/set individual parts,
and composite parts back into a flat image.

All nodes operate on SeeThrough's SEETHROUGH_PARTS wrapper:
    {"tag2pinfo": {tag: part_info, ...}, "frame_size": (H, W)}

Each part_info contains:
    img:           HxWx4 uint8 RGBA (cropped to part's bbox)
    xyxy:          [x1, y1, x2, y2] position in frame_size coordinate space
    depth:         HxW uint8 depth map
    depth_median:  float z-order (back-to-front sort key)
    tag:           string label
    skin_mask:     HxW uint8 (added by BD_SeeThroughExtractSkinMask)
"""

import copy

import numpy as np
import torch
from PIL import Image

from comfy_api.latest import io

from .types import HumanParseMapInput  # unused but keeps import structure consistent


SEETHROUGH_PARTS = "SEETHROUGH_PARTS"

DEFAULT_NON_SKIN_TAGS = (
    "hairf,hairb,front hair,back hair,"
    "eyewhite-r,eyewhite-l,irides-r,irides-l,"
    "eyelash-r,eyelash-l,eyebrow-r,eyebrow-l,"
    "eyer,eyel,browl,browr,"
    "headwear,eyewear,earwear,"
    "tail,wings,objects"
)

DEFAULT_CLOTHING_TAGS = (
    "neckwear,topwear,bottomwear,legwear,footwear,"
    "handwear-r,handwear-l,handwear,"
    "headwear,eyewear,earwear"
)


def _parse_tag_csv(csv: str) -> set[str]:
    return {t.strip() for t in (csv or "").split(",") if t.strip()}


# ---------- helpers ----------


def _ensure_parts(parts) -> tuple[dict, tuple[int, int]]:
    if not isinstance(parts, dict) or "tag2pinfo" not in parts:
        raise ValueError(
            f"Expected SEETHROUGH_PARTS wrapper {{'tag2pinfo': ..., 'frame_size': ...}}, "
            f"got {type(parts).__name__}"
        )
    tag2pinfo = parts["tag2pinfo"]
    frame_size = tuple(parts.get("frame_size", (0, 0)))
    return tag2pinfo, frame_size


def _mask_to_2d_uint8(mask) -> np.ndarray:
    """Accepts torch MASK (B,H,W) float[0,1] or numpy. Returns (H,W) uint8 0..255."""
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr


def _np_to_image_tensor(arr: np.ndarray) -> torch.Tensor:
    """RGB uint8 (H,W,3) → ComfyUI IMAGE (1,H,W,3) float32 in [0,1]."""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)


def _np_to_mask_tensor(arr: np.ndarray) -> torch.Tensor:
    """Single-channel uint8 (H,W) → ComfyUI MASK (1,H,W) float32 in [0,1]."""
    if arr.ndim == 3:
        arr = arr[..., -1] if arr.shape[2] == 4 else arr[..., 0]
    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)


def _image_tensor_to_np(image: torch.Tensor) -> np.ndarray:
    """ComfyUI IMAGE (B,H,W,C) → uint8 numpy (H,W,C). Takes batch[0]."""
    arr = image.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = (arr.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return arr


def _resize_global_to_frame(global_mask: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
    if global_mask.shape == (frame_h, frame_w):
        return global_mask
    pil = Image.fromarray(global_mask, mode="L").resize((frame_w, frame_h), Image.BILINEAR)
    return np.asarray(pil)


# ---------- 1. ExtractSkinMask ----------


class BD_SeeThroughExtractSkinMask(io.ComfyNode):
    """Slice a global skin mask into per-part skin masks, intersected with each part's alpha."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_SeeThroughExtractSkinMask",
            display_name="BD SeeThrough Extract Skin Mask",
            category="🧠BrainDead/Segmentation",
            description=(
                "For each SeeThrough part, crop the global skin mask to the part's bbox and "
                "intersect with the part's alpha. Result stored as part_info['skin_mask']. "
                "Use a global skin mask from your existing SAM3 chain (or any MASK source)."
            ),
            inputs=[
                io.Custom(SEETHROUGH_PARTS).Input("parts"),
                io.Mask.Input("global_skin_mask",
                              tooltip="Global skin mask at frame_size resolution. Resized if mismatched."),
                io.Float.Input("alpha_threshold", default=0.04, min=0.0, max=1.0, step=0.01, optional=True,
                               tooltip="Part alpha values above this count as part-pixels (matches SeeThrough's >10/255 default)."),
                io.Float.Input("skin_threshold", default=0.5, min=0.0, max=1.0, step=0.05, optional=True,
                               tooltip="Global mask values above this count as skin."),
                io.String.Input("excluded_tags", default=DEFAULT_NON_SKIN_TAGS, optional=True, multiline=True,
                                tooltip="Comma-separated tag names that always get empty skin_mask (eye details, hair, accessories — where SAM3's broad mask would false-positive)."),
                io.Boolean.Input("subtract_skin_from_clothing_alpha", default=False, optional=True,
                                 tooltip="For clothing tags, subtract global skin pixels from the part's own alpha. Fixes 'topwear includes belly' bleed at the source. Modifies part.img alpha in-place."),
                io.String.Input("clothing_tags", default=DEFAULT_CLOTHING_TAGS, optional=True, multiline=True,
                                tooltip="Tags treated as clothing for the alpha-subtract option above."),
            ],
            outputs=[
                io.Custom(SEETHROUGH_PARTS).Output(display_name="parts"),
                io.String.Output(display_name="summary"),
            ],
        )

    @classmethod
    def execute(cls, parts, global_skin_mask, alpha_threshold=0.04, skin_threshold=0.5,
                excluded_tags=DEFAULT_NON_SKIN_TAGS,
                subtract_skin_from_clothing_alpha=False,
                clothing_tags=DEFAULT_CLOTHING_TAGS) -> io.NodeOutput:
        tag2pinfo, frame_size = _ensure_parts(parts)
        if frame_size == (0, 0):
            raise ValueError("parts['frame_size'] missing or zero — cannot map global mask.")
        frame_h, frame_w = frame_size

        global_u8 = _mask_to_2d_uint8(global_skin_mask)
        global_u8 = _resize_global_to_frame(global_u8, frame_h, frame_w)
        skin_thresh_u8 = int(skin_threshold * 255)
        alpha_thresh_u8 = int(alpha_threshold * 255)

        excluded = _parse_tag_csv(excluded_tags)
        clothing = _parse_tag_csv(clothing_tags)

        out = copy.copy(parts)
        out_tag2pinfo = {}
        summary_lines = []
        for tag, info in tag2pinfo.items():
            new_info = dict(info)
            img = info.get("img")
            xyxy = info.get("xyxy")
            if img is None or xyxy is None:
                out_tag2pinfo[tag] = new_info
                continue
            arr = np.asarray(img).copy()
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            x1 = max(0, min(x1, frame_w))
            x2 = max(0, min(x2, frame_w))
            y1 = max(0, min(y1, frame_h))
            y2 = max(0, min(y2, frame_h))
            if x2 <= x1 or y2 <= y1:
                new_info["skin_mask"] = np.zeros(arr.shape[:2], dtype=np.uint8)
                out_tag2pinfo[tag] = new_info
                continue

            if tag in excluded:
                new_info["skin_mask"] = np.zeros(arr.shape[:2], dtype=np.uint8)
                out_tag2pinfo[tag] = new_info
                summary_lines.append(f"  {tag:14s} skin= 0.00% (excluded)")
                continue

            global_crop = global_u8[y1:y2, x1:x2]
            if global_crop.shape != arr.shape[:2]:
                pil = Image.fromarray(global_crop, mode="L").resize(
                    (arr.shape[1], arr.shape[0]), Image.BILINEAR)
                global_crop = np.asarray(pil)

            if arr.shape[2] == 4:
                alpha = arr[..., -1]
            else:
                alpha = np.full(arr.shape[:2], 255, dtype=np.uint8)
            skin_bool = (global_crop >= skin_thresh_u8) & (alpha >= alpha_thresh_u8)
            skin_mask = skin_bool.astype(np.uint8) * 255
            new_info["skin_mask"] = skin_mask

            if subtract_skin_from_clothing_alpha and tag in clothing and arr.shape[2] == 4:
                arr[..., -1] = np.where(skin_bool, 0, arr[..., -1]).astype(np.uint8)
                new_info["img"] = arr

            cov = 100.0 * (skin_mask > 0).mean()
            extra = " (alpha trimmed)" if (subtract_skin_from_clothing_alpha and tag in clothing) else ""
            summary_lines.append(f"  {tag:14s} skin={cov:5.2f}%{extra}")
            out_tag2pinfo[tag] = new_info

        out["tag2pinfo"] = out_tag2pinfo
        summary = f"Extracted skin masks for {len(out_tag2pinfo)} parts:\n" + "\n".join(summary_lines)
        print(f"[BD SeeThroughExtractSkinMask] {summary}", flush=True)
        return io.NodeOutput(out, summary)


# ---------- 2. GetPart ----------


class BD_SeeThroughGetPart(io.ComfyNode):
    """Extract one tag from SEETHROUGH_PARTS as IMAGE + body/skin masks + bbox."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_SeeThroughGetPart",
            display_name="BD SeeThrough Get Part",
            category="🧠BrainDead/Segmentation",
            description=(
                "Pull a single tag's image and masks out of SEETHROUGH_PARTS. "
                "Skin mask requires running BD_SeeThroughExtractSkinMask first (else returns zeros)."
            ),
            inputs=[
                io.Custom(SEETHROUGH_PARTS).Input("parts"),
                io.String.Input("tag", default="face",
                                tooltip="Exact tag name from SeeThrough (e.g. 'face', 'hairf', 'top', 'arm-l')."),
                io.Combo.Input("background",
                               options=["white", "black", "transparent", "checker"],
                               default="white", optional=True,
                               tooltip="Color shown behind transparent pixels in the IMAGE output. SeeThrough's RGB outside the part is meaningless (alpha is the truth) — pick a color so downstream nodes see something sensible. Use 'transparent' to keep raw RGB."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="body_mask"),
                io.Mask.Output(display_name="skin_mask"),
                io.Int.Output(display_name="x1"),
                io.Int.Output(display_name="y1"),
                io.Int.Output(display_name="x2"),
                io.Int.Output(display_name="y2"),
                io.Float.Output(display_name="depth_median"),
            ],
        )

    @classmethod
    def execute(cls, parts, tag, background="white") -> io.NodeOutput:
        tag2pinfo, _ = _ensure_parts(parts)
        if tag not in tag2pinfo:
            available = ", ".join(sorted(tag2pinfo.keys()))
            raise ValueError(f"Tag '{tag}' not in parts. Available: {available}")
        info = tag2pinfo[tag]
        img = np.asarray(info["img"])
        xyxy = info.get("xyxy") or [0, 0, img.shape[1], img.shape[0]]
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        dm = float(info.get("depth_median", 0.5))

        h, w = img.shape[:2]
        if img.shape[2] == 4:
            alpha = img[..., 3:4].astype(np.float32) / 255.0
            rgb = img[..., :3].astype(np.float32)
        else:
            alpha = np.ones((h, w, 1), dtype=np.float32)
            rgb = img.astype(np.float32)

        if background == "transparent":
            composited = rgb
        else:
            if background == "white":
                bg = np.full((h, w, 3), 255.0, dtype=np.float32)
            elif background == "black":
                bg = np.zeros((h, w, 3), dtype=np.float32)
            elif background == "checker":
                check = 64
                yy, xx = np.indices((h, w))
                pattern = (((yy // check) + (xx // check)) % 2).astype(np.float32)
                bg = (pattern * 96 + 160)[..., None].repeat(3, axis=-1)
            else:
                bg = np.full((h, w, 3), 255.0, dtype=np.float32)
            composited = rgb * alpha + bg * (1.0 - alpha)
        composited = composited.clip(0, 255).astype(np.uint8)

        image_t = _np_to_image_tensor(composited)
        body_mask_t = _np_to_mask_tensor(img[..., -1] if img.shape[2] == 4
                                         else np.full((h, w), 255, dtype=np.uint8))
        skin = info.get("skin_mask")
        if skin is None:
            skin = np.zeros((h, w), dtype=np.uint8)
        skin_mask_t = _np_to_mask_tensor(np.asarray(skin))

        return io.NodeOutput(image_t, body_mask_t, skin_mask_t, x1, y1, x2, y2, dm)


# ---------- 3. SetPart ----------


class BD_SeeThroughSetPart(io.ComfyNode):
    """Replace one tag's image (and optionally skin_mask) inside SEETHROUGH_PARTS.

    MUTATES the parts dict IN PLACE. This is intentional and required for the
    iterator pattern: each Run mutates one tag, the cached parts dict accumulates
    all updates across runs, and Composite reads the fully-accumulated state.

    Position is preserved as-is. If the new image has different dimensions than the
    original crop, that's treated as an upscale — the part is simply stored at the
    new resolution; BD_SeeThroughComposite handles the rescaling at composite time.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_SeeThroughSetPart",
            display_name="BD SeeThrough Set Part",
            category="🧠BrainDead/Segmentation",
            description=(
                "Replace a tag's image in-place inside SEETHROUGH_PARTS. "
                "MUTATES the parts dict so iteration accumulates across Runs. "
                "xyxy stays fixed; resolution change is allowed (Composite scales at output time)."
            ),
            inputs=[
                io.Custom(SEETHROUGH_PARTS).Input("parts"),
                io.String.Input("tag", default="face"),
                io.Image.Input("image"),
                io.Mask.Input("skin_mask", optional=True,
                              tooltip="Optional replacement skin mask. Must match incoming image size."),
                io.Boolean.Input("derive_alpha_from_image", default=True, optional=True,
                                 tooltip="If incoming IMAGE is 3-channel, keep the original alpha (rescaled to match). If 4-channel, use its alpha."),
                io.Boolean.Input("skip_if_missing", default=True, optional=True,
                                 tooltip="If True, silently skip when the tag isn't in parts (logs to console). If False, raise — useful only for strict workflows where the tag must exist."),
            ],
            outputs=[
                io.Custom(SEETHROUGH_PARTS).Output(display_name="parts"),
            ],
        )

    @classmethod
    def execute(cls, parts, tag, image, skin_mask=None, derive_alpha_from_image=True,
                skip_if_missing=True) -> io.NodeOutput:
        tag2pinfo, _ = _ensure_parts(parts)
        if tag not in tag2pinfo:
            if skip_if_missing:
                print(f"[BD SeeThrough Set Part] tag='{tag}' not in parts (have {len(tag2pinfo)}: "
                      f"{sorted(tag2pinfo.keys())}). Skipping.", flush=True)
                return io.NodeOutput(parts)
            available = ", ".join(sorted(tag2pinfo.keys()))
            raise ValueError(f"Tag '{tag}' not in parts. Available: {available}")

        new_rgb = _image_tensor_to_np(image)
        new_h, new_w = new_rgb.shape[:2]

        old_info = tag2pinfo[tag]
        old_img = np.asarray(old_info["img"])
        old_h, old_w = old_img.shape[:2]

        if old_img.shape[2] == 4:
            old_alpha = old_img[..., -1]
            if (old_h, old_w) != (new_h, new_w):
                old_alpha = np.asarray(
                    Image.fromarray(old_alpha, mode="L").resize((new_w, new_h), Image.BILINEAR)
                )
        else:
            old_alpha = np.full((new_h, new_w), 255, dtype=np.uint8)

        if derive_alpha_from_image and new_rgb.shape[2] == 4:
            alpha = new_rgb[..., -1]
            new_rgb = new_rgb[..., :3]
        else:
            alpha = old_alpha
            if new_rgb.shape[2] == 4:
                new_rgb = new_rgb[..., :3]

        new_img_rgba = np.concatenate([new_rgb, alpha[..., None]], axis=-1).astype(np.uint8)
        old_info["img"] = new_img_rgba

        if skin_mask is not None:
            sk = _mask_to_2d_uint8(skin_mask)
            if sk.shape != (new_h, new_w):
                sk = np.asarray(Image.fromarray(sk, mode="L").resize((new_w, new_h), Image.BILINEAR))
            old_info["skin_mask"] = sk
        elif "skin_mask" in old_info:
            existing_sk = np.asarray(old_info["skin_mask"])
            if existing_sk.shape != (new_h, new_w):
                old_info["skin_mask"] = np.asarray(
                    Image.fromarray(existing_sk, mode="L").resize((new_w, new_h), Image.NEAREST)
                )

        if "depth" in old_info and old_info["depth"] is not None:
            existing_depth = np.asarray(old_info["depth"])
            if existing_depth.shape != (new_h, new_w):
                old_info["depth"] = np.asarray(
                    Image.fromarray(existing_depth, mode="L").resize((new_w, new_h), Image.BILINEAR)
                )

        return io.NodeOutput(parts)


# ---------- 4. Composite ----------


class BD_SeeThroughComposite(io.ComfyNode):
    """Composite all parts back to a flat RGBA image, sorted back-to-front by depth_median."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_SeeThroughComposite",
            display_name="BD SeeThrough Composite",
            category="🧠BrainDead/Segmentation",
            description=(
                "Assemble parts back into a flat image at output_size. Parts are placed at "
                "scaled xyxy positions and resized to fit. Use this for final output or to "
                "validate that upscaled parts re-assemble correctly."
            ),
            inputs=[
                io.Custom(SEETHROUGH_PARTS).Input("parts"),
                io.Int.Input("output_size", default=0, min=0, max=8192, step=64, optional=True,
                             tooltip="Max side length of output canvas. 0 = use frame_size as-is."),
                io.Boolean.Input("emit_skin_mask", default=False, optional=True,
                                 tooltip="If true, also emit a composited global skin mask MASK from per-part skin masks."),
                io.Boolean.Input("trigger", default=True, optional=True,
                                 tooltip="If false, returns blank canvas without compositing. Wire is_last from Part Iterator here to only compose on the final iteration."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="alpha"),
                io.Mask.Output(display_name="skin_mask"),
            ],
        )

    @classmethod
    def execute(cls, parts, output_size=0, emit_skin_mask=False, trigger=True) -> io.NodeOutput:
        if not trigger:
            tag2pinfo, frame_size = _ensure_parts(parts)
            h, w = frame_size if frame_size != (0, 0) else (64, 64)
            blank_img = torch.zeros((1, h, w, 3))
            blank_mask = torch.zeros((1, h, w))
            return io.NodeOutput(blank_img, blank_mask, blank_mask)
        tag2pinfo, frame_size = _ensure_parts(parts)
        if frame_size == (0, 0):
            raise ValueError("parts['frame_size'] missing — Composite needs canvas dims.")
        frame_h, frame_w = frame_size

        if output_size <= 0:
            out_h, out_w = frame_h, frame_w
            scale = 1.0
        else:
            scale = output_size / max(frame_h, frame_w)
            out_h = int(round(frame_h * scale))
            out_w = int(round(frame_w * scale))

        canvas = np.zeros((out_h, out_w, 4), dtype=np.float32)
        skin_canvas = np.zeros((out_h, out_w), dtype=np.float32) if emit_skin_mask else None

        ordered = sorted(
            tag2pinfo.items(),
            key=lambda kv: float(kv[1].get("depth_median", 1.0)),
            reverse=True,
        )

        for tag, info in ordered:
            img = info.get("img")
            xyxy = info.get("xyxy")
            if img is None or xyxy is None:
                continue
            arr = np.asarray(img)
            if arr.ndim != 3 or arr.shape[2] not in (3, 4):
                continue
            x1, y1, x2, y2 = (int(round(v * scale)) for v in xyxy)
            x1 = max(0, min(x1, out_w))
            x2 = max(0, min(x2, out_w))
            y1 = max(0, min(y1, out_h))
            y2 = max(0, min(y2, out_h))
            tw, th = x2 - x1, y2 - y1
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
            dst = canvas[y1:y2, x1:x2]
            dst_rgb = dst[..., :3]
            dst_a = dst[..., 3:4]
            out_a = src_a + dst_a * (1.0 - src_a)
            safe_a = np.maximum(out_a, 1e-6)
            out_rgb = (src_rgb * src_a + dst_rgb * dst_a * (1.0 - src_a)) / safe_a
            canvas[y1:y2, x1:x2, :3] = out_rgb
            canvas[y1:y2, x1:x2, 3:4] = out_a

            if skin_canvas is not None:
                sk = info.get("skin_mask")
                if sk is not None:
                    sk_arr = np.asarray(sk).astype(np.float32) / 255.0
                    if sk_arr.shape != (th, tw):
                        sk_pil = Image.fromarray((sk_arr * 255).astype(np.uint8), mode="L").resize(
                            (tw, th), Image.NEAREST)
                        sk_arr = np.asarray(sk_pil).astype(np.float32) / 255.0
                    skin_canvas[y1:y2, x1:x2] = np.maximum(skin_canvas[y1:y2, x1:x2], sk_arr)

        image_t = torch.from_numpy(canvas[..., :3]).unsqueeze(0)
        alpha_t = torch.from_numpy(canvas[..., 3]).unsqueeze(0)
        skin_t = (torch.from_numpy(skin_canvas) if skin_canvas is not None
                  else torch.zeros_like(alpha_t.squeeze(0))).unsqueeze(0)
        return io.NodeOutput(image_t, alpha_t, skin_t)


# ---------- 5. PartIterator ----------


import time

_PART_ITERATOR_STATE: dict[str, dict] = {}


def _filter_tag_list(all_tags, filter_mode: str, custom_csv: str) -> list[str]:
    excl = _parse_tag_csv(DEFAULT_NON_SKIN_TAGS)
    cloth = _parse_tag_csv(DEFAULT_CLOTHING_TAGS)
    ordered = sorted(all_tags)
    if filter_mode == "all":
        return ordered
    if filter_mode == "skin_likely":
        return [t for t in ordered if t not in excl]
    if filter_mode == "clothing_only":
        return [t for t in ordered if t in cloth]
    if filter_mode == "non_skin_only":
        return [t for t in ordered if t in excl]
    if filter_mode == "custom":
        custom = _parse_tag_csv(custom_csv)
        return [t for t in ordered if t in custom]
    return ordered


class BD_SeeThroughPartIterator(io.ComfyNode):
    """Cycle through SEETHROUGH_PARTS one tag per execution.

    Drop-in replacement for `BD SeeThrough Get Part` when you want to process
    every (filtered) tag with the same downstream chain — Qwen upscale, GLSL
    skin tinting, etc. Each Run advances to the next tag. Wire `is_last` to
    stop your queue loop or trigger the final composite.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_SeeThroughPartIterator",
            display_name="BD SeeThrough Part Iterator",
            category="🧠BrainDead/Segmentation",
            description=(
                "Iterate one tag from SEETHROUGH_PARTS per execution. Same outputs as Get Part "
                "plus iteration metadata. Filter to skin-likely / clothing-only / etc. "
                "Pattern matches BD Prompt Iterator: each queued Run cycles forward, set reset=True to start over."
            ),
            inputs=[
                io.Custom(SEETHROUGH_PARTS).Input("parts"),
                io.Combo.Input("mode", options=["sequential", "manual", "single"], default="sequential", optional=True),
                io.Combo.Input("filter",
                               options=["all", "skin_likely", "clothing_only", "non_skin_only", "custom"],
                               default="all", optional=True,
                               tooltip="skin_likely = everything not in excluded list (face, neck, arms…). clothing_only = neckwear/topwear/etc."),
                io.String.Input("custom_tags", default="", multiline=True, optional=True,
                                tooltip="Used when filter=custom. Comma-separated tag names."),
                io.Int.Input("manual_index", default=0, min=0, max=999, step=1, optional=True),
                io.Boolean.Input("reset", default=False, optional=True,
                                 tooltip="Edge-triggered: when this transitions False→True, the next Run starts at index 0 then advances normally. Leave True across runs and only the first one resets; toggle to False then back to True to reset again."),
                io.String.Input("workflow_id", default="default", optional=True,
                                tooltip="Use distinct ids if you have multiple iterators running."),
                io.Combo.Input("background", options=["white", "black", "transparent", "checker"],
                               default="white", optional=True,
                               tooltip="Background under transparent pixels in the IMAGE output."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="body_mask"),
                io.Mask.Output(display_name="skin_mask"),
                io.String.Output(display_name="tag"),
                io.Int.Output(display_name="current_index"),
                io.Int.Output(display_name="total_count"),
                io.Boolean.Output(display_name="is_last"),
                io.Int.Output(display_name="x1"),
                io.Int.Output(display_name="y1"),
                io.Int.Output(display_name="x2"),
                io.Int.Output(display_name="y2"),
                io.Float.Output(display_name="depth_median"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, parts, mode="sequential", filter="all", custom_tags="",
                           manual_index=0, reset=False, workflow_id="default",
                           background="white") -> str:
        if mode == "sequential":
            return f"seq_{time.time()}"
        return f"{mode}_{manual_index}_{workflow_id}_{filter}_{background}"

    @classmethod
    def execute(cls, parts, mode="sequential", filter="all", custom_tags="",
                manual_index=0, reset=False, workflow_id="default",
                background="white") -> io.NodeOutput:
        global _PART_ITERATOR_STATE

        tag2pinfo, _ = _ensure_parts(parts)
        candidates = _filter_tag_list(list(tag2pinfo.keys()), filter, custom_tags)
        total = len(candidates)
        if total == 0:
            raise ValueError(f"No tags match filter='{filter}'. Available: {sorted(tag2pinfo.keys())}")

        state = _PART_ITERATOR_STATE.setdefault(
            workflow_id, {"index": 0, "iteration": 0, "last_reset_value": False}
        )
        prev_reset = state.get("last_reset_value", False)
        if reset and not prev_reset:
            state["index"] = 0
            state["iteration"] = 0
        state["last_reset_value"] = reset

        if mode == "manual":
            current_index = max(0, min(manual_index, total - 1))
        elif mode == "single":
            current_index = 0
        else:
            current_index = state["index"] % total
            state["index"] = (current_index + 1) % total
            if state["index"] == 0:
                state["iteration"] += 1

        tag = candidates[current_index]
        is_last = (current_index == total - 1)

        getpart_out = BD_SeeThroughGetPart.execute(parts, tag, background=background)
        image_t, body_mask_t, skin_mask_t, x1, y1, x2, y2, dm = getpart_out.args

        status_lines = [
            f"[{current_index + 1}/{total}] tag='{tag}' filter={filter}"
            + (f" iter={state['iteration'] + 1}" if mode == "sequential" else "")
            + (" (LAST)" if is_last else ""),
            f"  filtered ({total}): {', '.join(candidates)}",
            f"  all in parts ({len(tag2pinfo)}): {', '.join(sorted(tag2pinfo.keys()))}",
        ]
        status = "\n".join(status_lines)

        print(f"[BD SeeThroughPartIterator] {status}", flush=True)
        return io.NodeOutput(image_t, body_mask_t, skin_mask_t, tag,
                             current_index, total, is_last,
                             x1, y1, x2, y2, dm, status)


class BD_SeeThroughListTags(io.ComfyNode):
    """Inspect SEETHROUGH_PARTS: list every detected tag, its bbox, depth_median, and skin coverage if present."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_SeeThroughListTags",
            display_name="BD SeeThrough List Tags",
            category="🧠BrainDead/Segmentation",
            description=(
                "Diagnostic: report all detected tags + per-tag stats. SeeThrough has a fixed "
                "19-tag vocabulary (hair, headwear, face, eyes, eyewear, ears, earwear, nose, "
                "mouth, neck, neckwear, topwear, handwear, bottomwear, legwear, footwear, tail, "
                "wings, objects). There is no separate 'arm' or 'leg' tag — bare skin lives "
                "inside topwear/handwear/legwear/etc. silhouettes."
            ),
            inputs=[
                io.Custom(SEETHROUGH_PARTS).Input("parts"),
            ],
            outputs=[
                io.String.Output(display_name="all_tags_csv"),
                io.String.Output(display_name="skin_likely_csv"),
                io.String.Output(display_name="clothing_csv"),
                io.String.Output(display_name="report"),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    def execute(cls, parts) -> io.NodeOutput:
        tag2pinfo, frame_size = _ensure_parts(parts)
        excl = _parse_tag_csv(DEFAULT_NON_SKIN_TAGS)
        cloth = _parse_tag_csv(DEFAULT_CLOTHING_TAGS)

        all_tags = sorted(tag2pinfo.keys())
        skin_likely = [t for t in all_tags if t not in excl]
        clothing = [t for t in all_tags if t in cloth]

        lines = [
            f"frame_size: {frame_size}",
            f"total tags: {len(all_tags)}",
            "",
            f"{'tag':<16}{'size':<14}{'xyxy':<28}{'depth':<8}{'skin%'}",
            "─" * 76,
        ]
        for tag in all_tags:
            info = tag2pinfo[tag]
            img = np.asarray(info.get("img"))
            sz = f"{img.shape[1]}x{img.shape[0]}" if img.ndim == 3 else "?"
            xyxy = info.get("xyxy", "?")
            xy_str = str(xyxy)
            dm = info.get("depth_median")
            dm_str = f"{dm:.3f}" if isinstance(dm, (int, float)) else "?"
            sk = info.get("skin_mask")
            if sk is not None:
                sk_arr = np.asarray(sk)
                sk_pct = f"{100.0 * (sk_arr > 0).mean():.1f}%"
            else:
                sk_pct = "—"
            lines.append(f"{tag:<16}{sz:<14}{xy_str:<28}{dm_str:<8}{sk_pct}")

        lines.append("")
        lines.append(f"skin_likely ({len(skin_likely)}): {', '.join(skin_likely)}")
        lines.append(f"clothing    ({len(clothing)}):    {', '.join(clothing)}")
        report = "\n".join(lines)
        print(f"[BD SeeThrough List Tags]\n{report}", flush=True)

        return io.NodeOutput(
            ",".join(all_tags),
            ",".join(skin_likely),
            ",".join(clothing),
            report,
            len(all_tags),
        )


SEETHROUGH_PIPELINE_V3_NODES = [
    BD_SeeThroughExtractSkinMask,
    BD_SeeThroughGetPart,
    BD_SeeThroughSetPart,
    BD_SeeThroughComposite,
    BD_SeeThroughPartIterator,
    BD_SeeThroughListTags,
]

SEETHROUGH_PIPELINE_NODES = {n.__name__: n for n in SEETHROUGH_PIPELINE_V3_NODES}
SEETHROUGH_PIPELINE_DISPLAY_NAMES = {
    "BD_SeeThroughExtractSkinMask": "BD SeeThrough Extract Skin Mask",
    "BD_SeeThroughGetPart":         "BD SeeThrough Get Part",
    "BD_SeeThroughSetPart":         "BD SeeThrough Set Part",
    "BD_SeeThroughComposite":       "BD SeeThrough Composite",
    "BD_SeeThroughPartIterator":    "BD SeeThrough Part Iterator",
    "BD_SeeThroughListTags":        "BD SeeThrough List Tags",
}
