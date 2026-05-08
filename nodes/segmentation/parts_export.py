"""
BD_PartsExport — write a PARTS_BUNDLE to disk: per-part PNGs, optional
per-part depth PNGs, optional flattened composite PNG, optional layered PSD.

Path resolution:
  - If `context_id` is wired (or auto-picked from a single registered
    BD_SaveContext), each per-tag file uses the context's template with
    `%suffix%` filled from the tag name. Composite/PSD use the same
    template with `%suffix%` = `_composite` / empty.
  - Without context: legacy folder layout — `{name_prefix}_{filename}_NNN/`
    with `{tag}.png` inside it.

PSD: raw compression (Photoshop chokes on pytoshop's zip variant).
"""

import os
import re
from glob import glob

import numpy as np
import torch
from PIL import Image

import folder_paths
import comfy.utils
from comfy_api.latest import io

from .parts_types import PARTS_BUNDLE, ensure_bundle, frame_size as _frame_size


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "tag"


def _resolve_legacy_folder(filename: str, name_prefix: str, auto_increment: bool) -> tuple[str, str]:
    output_base = folder_paths.get_output_directory()
    full_name = f"{name_prefix}_{filename}" if name_prefix else filename
    full_name = full_name.replace("\\", "/")
    if "/" in full_name:
        subdir, base = full_name.rsplit("/", 1)
        parent = os.path.join(output_base, subdir)
    else:
        parent = output_base
        base = full_name
    os.makedirs(parent, exist_ok=True)

    if auto_increment:
        existing = glob(os.path.join(parent, f"{base}_*"))
        nums = []
        for p in existing:
            m = re.search(rf"{re.escape(base)}_(\d+)$", os.path.basename(p))
            if m:
                nums.append(int(m.group(1)))
        next_num = (max(nums) + 1) if nums else 1
        folder = os.path.join(parent, f"{base}_{next_num:03d}")
    else:
        folder = os.path.join(parent, base)
    os.makedirs(folder, exist_ok=True)
    return folder, base


def _save_layered_psd(parts: dict, out_path: str, output_size: int = 0,
                      include_masks: bool = False,
                      base_image: "torch.Tensor | None" = None) -> int:
    """Write all parts as a single layered PSD. Returns layer count.
    Uses RAW compression — Photoshop fails to open pytoshop's zip variant.

    output_size: if > 0, the canvas (and per-layer xyxy/dims) is scaled so the
    longest frame_size edge equals output_size. Layer pixel dims are scaled
    to match the new bbox dims at that canvas resolution. 0 = use frame_size.

    include_masks: if True, ALSO add a `{tag}_mask` layer per part containing
    the original SAM3 visibility mask (white = was visible). Mask layers are
    placed at the same scaled xyxy as their part, sized identically, and start
    with visibility OFF — drag them in PS to use as layer masks if needed.

    base_image: optional (1, H, W, 3) IMAGE that gets added as the BOTTOM
    layer of the PSD covering the full canvas. Resized to canvas dim if needed.
    """
    from pytoshop import enums
    from pytoshop.user.nested_layers import Image as PsdImage, nested_layers_to_psd

    tag2pinfo = parts.get("tag2pinfo", {})
    frame_h, frame_w = _frame_size(parts)

    # Compute scale factor from frame_size to canvas
    if output_size > 0 and frame_h > 0 and frame_w > 0:
        scale = output_size / max(frame_h, frame_w)
    else:
        scale = 1.0
    canvas_w = int(round(frame_w * scale)) if frame_w > 0 else 0
    canvas_h = int(round(frame_h * scale)) if frame_h > 0 else 0

    sorted_items = sorted(
        tag2pinfo.items(),
        key=lambda kv: -float(kv[1].get("depth_median", 0.5)),
    )

    layers = []

    # Bottom layer: base_image (e.g. nude mannequin) covering full canvas.
    # Added FIRST so it sits at the bottom of the layer stack in PS.
    if base_image is not None:
        bi = base_image[0] if base_image.dim() == 4 else base_image
        bi_np = bi.detach().cpu().numpy()
        bi_uint = (bi_np * 255.0).clip(0, 255).astype(np.uint8)
        if bi_uint.shape[-1] == 3:
            alpha_full = np.full(bi_uint.shape[:2], 255, dtype=np.uint8)
            bi_uint = np.concatenate([bi_uint, alpha_full[..., None]], axis=-1)
        # Resize to canvas
        if bi_uint.shape[:2] != (canvas_h, canvas_w):
            pil = Image.fromarray(bi_uint, mode="RGBA")
            pil = pil.resize((canvas_w, canvas_h), Image.LANCZOS)
            bi_uint = np.asarray(pil)
        layers.append(PsdImage(
            name="base",
            top=0, left=0,
            bottom=canvas_h, right=canvas_w,
            channels={
                -1: bi_uint[..., 3],
                0:  bi_uint[..., 0],
                1:  bi_uint[..., 1],
                2:  bi_uint[..., 2],
            },
        ))

    for tag, info in sorted_items:
        img = info.get("img")
        xyxy = info.get("xyxy")
        if img is None or xyxy is None:
            continue
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            continue
        if arr.shape[2] == 3:
            alpha = np.full(arr.shape[:2], 255, dtype=np.uint8)
            arr = np.concatenate([arr, alpha[..., None]], axis=-1)
        # Scale xyxy from frame_size space to canvas space
        x1 = int(round(int(xyxy[0]) * scale))
        y1 = int(round(int(xyxy[1]) * scale))
        x2 = int(round(int(xyxy[2]) * scale))
        y2 = int(round(int(xyxy[3]) * scale))
        bbox_w, bbox_h = max(1, x2 - x1), max(1, y2 - y1)
        tag_safe = _safe_filename(tag)
        # Resize layer pixels to match bbox dims at the chosen canvas resolution.
        if arr.shape[:2] != (bbox_h, bbox_w):
            pil = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
            pil = pil.resize((bbox_w, bbox_h), Image.LANCZOS)
            arr = np.asarray(pil)
        layers.append(PsdImage(
            name=tag_safe,
            top=y1, left=x1,
            bottom=y1 + bbox_h, right=x1 + bbox_w,
            channels={
                -1: arr[..., 3],
                0:  arr[..., 0],
                1:  arr[..., 1],
                2:  arr[..., 2],
            },
        ))

        # Optional: add a `{tag}_mask` layer with the ORIGINAL SAM3 visibility
        # mask, sized identically to the part, visibility OFF.
        if include_masks:
            orig = info.get("original_alpha")
            if orig is not None:
                mask_arr = np.asarray(orig).astype(np.uint8)
                # Resize to match the part's bbox dims at canvas resolution
                if mask_arr.shape != (bbox_h, bbox_w):
                    mask_arr = np.asarray(
                        Image.fromarray(mask_arr, mode="L").resize(
                            (bbox_w, bbox_h), Image.BILINEAR,
                        )
                    )
                # Render as a grayscale-looking layer (RGB = mask value, alpha=255).
                # Visibility off so user can toggle on when they want to use it.
                gray_rgb = np.stack([mask_arr] * 3, axis=-1)
                full_alpha = np.full(mask_arr.shape, 255, dtype=np.uint8)
                layers.append(PsdImage(
                    name=f"{tag_safe}_mask",
                    visible=False,
                    top=y1, left=x1,
                    bottom=y1 + bbox_h, right=x1 + bbox_w,
                    channels={
                        -1: full_alpha,
                        0:  gray_rgb[..., 0],
                        1:  gray_rgb[..., 1],
                        2:  gray_rgb[..., 2],
                    },
                ))

    # Fallback canvas if frame_size missing
    if canvas_w <= 0 or canvas_h <= 0:
        canvas_w = max((int(info.get("xyxy", [0, 0, 0, 0])[2])
                        for info in tag2pinfo.values()), default=64)
        canvas_h = max((int(info.get("xyxy", [0, 0, 0, 0])[3])
                        for info in tag2pinfo.values()), default=64)

    psd = nested_layers_to_psd(layers, color_mode=3, size=(canvas_w, canvas_h),
                               compression=enums.Compression.raw)
    with open(out_path, "wb") as f:
        psd.write(f)
    return len(layers)


class BD_PartsExport(io.ComfyNode):
    """Export a PARTS_BUNDLE to disk: per-tag PNGs + optional composite + layered PSD."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_PartsExport",
            display_name="BD Parts Export",
            category="🧠BrainDead/Segmentation",
            is_output_node=True,
            description=(
                "Write a PARTS_BUNDLE to disk. Per-tag RGBA PNGs, optional per-tag depth PNGs "
                "(when bundle has 'depth' from BD_PartsBuilder + depth_image), optional flat "
                "composite PNG, and optional layered PSD (raw-compressed for Photoshop "
                "compatibility).\n\n"
                "Wire `context_id` from BD_SaveContext for template-based naming "
                "(suffix = tag for per-part, '_composite' for composite, '' for psd). "
                "Without context, uses legacy folder layout: {name_prefix}_{filename}_NNN/{tag}.png"
            ),
            inputs=[
                io.Custom(PARTS_BUNDLE).Input("parts"),
                io.String.Input(
                    "filename", default="parts",
                    tooltip="Legacy mode: base name for the output folder. "
                            "Context mode: feeds %name% / %filename% in the template.",
                ),
                io.String.Input(
                    "name_prefix", default="", optional=True,
                    tooltip="Legacy mode: prepended to filename, supports subdirs (e.g. 'Characters/Letti'). "
                            "Context mode: feeds %name_prefix% in the template.",
                ),
                io.Boolean.Input("auto_increment", default=True, optional=True,
                                 tooltip="Legacy mode only: append _001/_002/... to folder name."),
                io.String.Input("context_id", default="", optional=True,
                                tooltip="Match a BD_SaveContext id. Empty + exactly one registered = auto-pick. "
                                        "Per-tag files use suffix=tag; composite uses suffix='_composite'."),
                io.Boolean.Input("save_pngs", default=True, optional=True,
                                 tooltip="Write one RGBA PNG per tag."),
                io.Boolean.Input("save_depth", default=False, optional=True,
                                 tooltip="Also write {tag}_depth.png grayscale per part. "
                                         "Requires bundle to carry per-part depth (wire depth_image into BD_PartsBuilder)."),
                io.Boolean.Input("save_masks", default=False, optional=True,
                                 tooltip="Also write {tag}_mask.png — single-channel grayscale of "
                                         "the ORIGINAL SAM3 mask (the visible region in the source, "
                                         "before Qwen redraw), resized to match the rebuilt part "
                                         "dims. Drop into Photoshop to re-apply the original "
                                         "visibility cut on a redrawn layer. White = was visible, "
                                         "Black = was occluded.\n\n"
                                         "If the bundle was never edited (no PartsBatchEdit step), "
                                         "falls back to the part's current alpha channel."),
                io.Boolean.Input("save_composite", default=True, optional=True,
                                 tooltip="Also write {filename}_composite.png (RGBA) and "
                                         "{filename}_composite_alpha.png (mask)."),
                io.Int.Input("composite_size", default=0, min=0, max=8192, step=64, optional=True,
                             tooltip="Max side length of composite PNG AND PSD canvas. Scales xyxy "
                                     "and layer dims to match. 0 = use frame_size as-is. Set to e.g. "
                                     "4096 for 4K production output."),
                io.Boolean.Input("save_psd", default=False, optional=True,
                                 tooltip="Also write {filename}.psd — single layered file with one "
                                         "layer per tag at its xyxy position. Layer order: back-to-front "
                                         "by depth_median. RAW compression (Photoshop-compatible)."),
                io.Image.Input(
                    "base_image", optional=True,
                    tooltip="Optional base IMAGE (e.g. nude mannequin) painted UNDER all parts. "
                            "Becomes the bottom layer of the PSD (full canvas, visible) and the "
                            "base of the composite PNG. Resized to canvas dim if needed.",
                ),
            ],
            outputs=[
                io.Custom(PARTS_BUNDLE).Output(display_name="parts"),
                io.String.Output(display_name="output_dir"),
                io.String.Output(display_name="summary"),
                io.Image.Output(display_name="composite_image"),
                io.Image.Output(display_name="parts_image_batch"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **_) -> str:
        import time
        return f"parts_export_{time.time()}"

    @classmethod
    def execute(cls, parts, filename="parts", name_prefix="",
                auto_increment=True, context_id="",
                save_pngs=True, save_depth=False, save_masks=False,
                save_composite=True, composite_size=0,
                save_psd=False, base_image=None) -> io.NodeOutput:
        ensure_bundle(parts, source="BD_PartsExport.parts")

        from ..cache.save_context import resolve_context_path, get_context, auto_pick_context

        effective_ctx_id = context_id
        if not effective_ctx_id:
            picked = auto_pick_context()
            if picked is not None:
                effective_ctx_id = picked

        ctx = get_context(effective_ctx_id) if effective_ctx_id else None
        use_context = ctx is not None

        if not use_context:
            folder, base = _resolve_legacy_folder(filename, name_prefix, auto_increment)

        tag2pinfo = parts["tag2pinfo"]
        summary_lines: list[str] = []
        written = 0

        # Per-tag PNGs (and optional depth)
        if save_pngs:
            for tag, info in tag2pinfo.items():
                if not isinstance(info, dict):
                    continue
                img = info.get("img")
                if img is None:
                    continue
                arr = np.asarray(img)
                if arr.ndim != 3 or arr.shape[2] not in (3, 4):
                    continue
                mode = "RGBA" if arr.shape[2] == 4 else "RGB"
                tag_safe = _safe_filename(tag)
                if use_context:
                    png_path, _ = resolve_context_path(
                        effective_ctx_id, f"_{tag_safe}", "png",
                        node_filename=filename, node_name_prefix=name_prefix,
                    )
                    os.makedirs(os.path.dirname(png_path), exist_ok=True)
                else:
                    png_path = os.path.join(folder, f"{tag_safe}.png")
                Image.fromarray(arr.astype(np.uint8), mode=mode).save(png_path, optimize=True)
                written += 1

                if save_depth and info.get("depth") is not None:
                    depth_arr = np.asarray(info["depth"])
                    if depth_arr.dtype != np.uint8:
                        if depth_arr.max() <= 1.0:
                            depth_arr = (depth_arr * 255.0).clip(0, 255).astype(np.uint8)
                        else:
                            depth_arr = depth_arr.clip(0, 255).astype(np.uint8)
                    if use_context:
                        depth_path, _ = resolve_context_path(
                            effective_ctx_id, f"_{tag_safe}_depth", "png",
                            node_filename=filename, node_name_prefix=name_prefix,
                        )
                        os.makedirs(os.path.dirname(depth_path), exist_ok=True)
                    else:
                        depth_path = os.path.join(folder, f"{tag_safe}_depth.png")
                    Image.fromarray(depth_arr, mode="L").save(depth_path, optimize=True)

                if save_masks:
                    # Prefer the ORIGINAL SAM3 mask stashed by PartsBatchEdit
                    # (the pre-edit visibility shape resized to match rebuilt dims).
                    # Fall back to the current alpha if the part was never edited.
                    orig = info.get("original_alpha")
                    if orig is not None:
                        mask_arr = np.asarray(orig).astype(np.uint8)
                    elif arr.shape[-1] == 4:
                        mask_arr = arr[..., 3].astype(np.uint8)
                    else:
                        mask_arr = None
                    if mask_arr is not None:
                        if use_context:
                            mask_path, _ = resolve_context_path(
                                effective_ctx_id, f"_{tag_safe}_mask", "png",
                                node_filename=filename, node_name_prefix=name_prefix,
                            )
                            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                        else:
                            mask_path = os.path.join(folder, f"{tag_safe}_mask.png")
                        Image.fromarray(mask_arr, mode="L").save(mask_path, optimize=True)

        # Composite + alpha
        composite_tensor = None
        if save_composite or save_psd:
            from .parts_compose import BD_PartsCompose
            comp_out = BD_PartsCompose.execute(parts, output_size=composite_size, trigger=True)
            composite_tensor, alpha_tensor = comp_out.args

            # If base_image is wired, blend the parts composite OVER the base.
            # Resize base to canvas dim if needed.
            if base_image is not None:
                base_t = base_image if base_image.dim() == 4 else base_image.unsqueeze(0)
                ch, cw = composite_tensor.shape[1], composite_tensor.shape[2]
                if base_t.shape[1] != ch or base_t.shape[2] != cw:
                    base_chw = base_t.movedim(-1, 1)
                    base_chw = comfy.utils.common_upscale(base_chw, cw, ch, "lanczos", "disabled")
                    base_t = base_chw.movedim(1, -1)
                base_rgb = base_t[..., :3]
                a3 = alpha_tensor[..., None] if alpha_tensor.dim() == 3 else alpha_tensor.unsqueeze(-1)
                composite_tensor = composite_tensor * a3 + base_rgb * (1.0 - a3)
                # Alpha after compositing onto base = fully opaque (we have a base now)
                alpha_tensor = torch.ones_like(alpha_tensor)

            if save_composite:
                rgb = (composite_tensor[0].cpu().numpy() * 255).astype(np.uint8)
                a = (alpha_tensor[0].cpu().numpy() * 255).astype(np.uint8)
                rgba = np.concatenate([rgb, a[..., None]], axis=-1)

                if use_context:
                    comp_path, _ = resolve_context_path(
                        effective_ctx_id, "_composite", "png",
                        node_filename=filename, node_name_prefix=name_prefix,
                    )
                    alpha_path, _ = resolve_context_path(
                        effective_ctx_id, "_composite_alpha", "png",
                        node_filename=filename, node_name_prefix=name_prefix,
                    )
                    os.makedirs(os.path.dirname(comp_path), exist_ok=True)
                else:
                    comp_path = os.path.join(folder, f"{base}_composite.png")
                    alpha_path = os.path.join(folder, f"{base}_composite_alpha.png")

                Image.fromarray(rgba, mode="RGBA").save(comp_path, optimize=True)
                Image.fromarray(a, mode="L").save(alpha_path, optimize=True)
                summary_lines.append(f"  composite: {os.path.basename(comp_path)}  ({rgba.shape})")

        # Layered PSD
        if save_psd:
            if use_context:
                psd_path, _ = resolve_context_path(
                    effective_ctx_id, "", "psd",
                    node_filename=filename, node_name_prefix=name_prefix,
                )
                os.makedirs(os.path.dirname(psd_path), exist_ok=True)
            else:
                psd_path = os.path.join(folder, f"{base}.psd")
            try:
                n_layers = _save_layered_psd(
                    parts, psd_path, output_size=int(composite_size),
                    include_masks=bool(save_masks),
                    base_image=base_image,
                )
                summary_lines.append(f"  layered PSD: {os.path.basename(psd_path)}  ({n_layers} layers, raw)")
            except Exception as e:
                summary_lines.append(f"  PSD save FAILED: {e}")

        out_dir = (os.path.dirname(psd_path) if save_psd
                   else (folder if not use_context else os.path.dirname(comp_path) if save_composite
                         else os.path.dirname(png_path) if save_pngs else "."))

        if composite_tensor is None:
            h, w = _frame_size(parts) or (64, 64)
            h, w = (h or 64), (w or 64)
            composite_tensor = torch.zeros((1, h, w, 3))

        # Build parts_image_batch — N parts as RGBA batch where:
        #   RGB = full Qwen rebuild (info["img"] RGB, NOT cut to mask)
        #   alpha = original SAM3 visibility mask (info["original_alpha"] if set,
        #           else info["img"]'s alpha as fallback for un-edited bundles)
        # Padded to common max-dim shape for downstream batched processing.
        tag2pinfo = parts["tag2pinfo"]
        parts_for_batch = []
        for tag, info in tag2pinfo.items():
            if not isinstance(info, dict):
                continue
            img = info.get("img")
            if img is None:
                continue
            arr = np.asarray(img)
            if arr.ndim != 3 or arr.shape[2] not in (3, 4):
                continue
            # RGB always from info["img"]
            if arr.shape[2] == 4:
                rgb = arr[..., :3]
            else:
                rgb = arr
            # Alpha preference: original SAM3 mask (resized to match) if available
            orig = info.get("original_alpha")
            if orig is not None:
                a = np.asarray(orig).astype(np.uint8)
                if a.shape != rgb.shape[:2]:
                    a = np.asarray(
                        Image.fromarray(a, mode="L").resize(
                            (rgb.shape[1], rgb.shape[0]), Image.BILINEAR,
                        )
                    )
            elif arr.shape[2] == 4:
                a = arr[..., 3]
            else:
                a = np.full(rgb.shape[:2], 255, dtype=np.uint8)
            rgba = np.concatenate([rgb, a[..., None]], axis=-1).astype(np.uint8)
            parts_for_batch.append(rgba)

        if parts_for_batch:
            max_h = max(r.shape[0] for r in parts_for_batch)
            max_w = max(r.shape[1] for r in parts_for_batch)
            padded = []
            for rgba in parts_for_batch:
                h_, w_ = rgba.shape[:2]
                pad_t = (max_h - h_) // 2
                pad_l = (max_w - w_) // 2
                canvas = np.zeros((max_h, max_w, 4), dtype=np.uint8)
                canvas[pad_t:pad_t + h_, pad_l:pad_l + w_] = rgba
                padded.append(canvas)
            parts_image_batch = torch.from_numpy(
                np.stack(padded, axis=0).astype(np.float32) / 255.0
            )
        else:
            parts_image_batch = torch.zeros((1, 1, 1, 4))

        header = (f"BD PartsExport: {written} per-tag PNGs to {out_dir}"
                  + (f" via context='{effective_ctx_id}'" if use_context else ""))
        summary = header + ("\n" + "\n".join(summary_lines) if summary_lines else "")
        print(f"[BD PartsExport] {summary}", flush=True)
        return io.NodeOutput(parts, out_dir, summary, composite_tensor, parts_image_batch)


PARTS_EXPORT_V3_NODES = [BD_PartsExport]
PARTS_EXPORT_NODES = {"BD_PartsExport": BD_PartsExport}
PARTS_EXPORT_DISPLAY_NAMES = {"BD_PartsExport": "BD Parts Export"}
