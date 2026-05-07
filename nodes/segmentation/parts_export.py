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


def _save_layered_psd(parts: dict, out_path: str, output_size: int = 0) -> int:
    """Write all parts as a single layered PSD. Returns layer count.
    Uses RAW compression — Photoshop fails to open pytoshop's zip variant.

    output_size: if > 0, the canvas (and per-layer xyxy/dims) is scaled so the
    longest frame_size edge equals output_size. Layer pixel dims are scaled
    to match the new bbox dims at that canvas resolution. 0 = use frame_size.
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
        # Resize layer pixels to match bbox dims at the chosen canvas resolution.
        if arr.shape[:2] != (bbox_h, bbox_w):
            pil = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
            pil = pil.resize((bbox_w, bbox_h), Image.LANCZOS)
            arr = np.asarray(pil)
        layers.append(PsdImage(
            name=_safe_filename(tag),
            top=y1, left=x1,
            bottom=y1 + bbox_h, right=x1 + bbox_w,
            channels={
                -1: arr[..., 3],
                0:  arr[..., 0],
                1:  arr[..., 1],
                2:  arr[..., 2],
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
            ],
            outputs=[
                io.Custom(PARTS_BUNDLE).Output(display_name="parts"),
                io.String.Output(display_name="output_dir"),
                io.String.Output(display_name="summary"),
                io.Image.Output(display_name="composite_image"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **_) -> str:
        import time
        return f"parts_export_{time.time()}"

    @classmethod
    def execute(cls, parts, filename="parts", name_prefix="",
                auto_increment=True, context_id="",
                save_pngs=True, save_depth=False,
                save_composite=True, composite_size=0,
                save_psd=False) -> io.NodeOutput:
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

        # Composite + alpha
        composite_tensor = None
        if save_composite or save_psd:
            from .parts_compose import BD_PartsCompose
            comp_out = BD_PartsCompose.execute(parts, output_size=composite_size, trigger=True)
            composite_tensor, alpha_tensor = comp_out.args

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
                n_layers = _save_layered_psd(parts, psd_path, output_size=int(composite_size))
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

        header = (f"BD PartsExport: {written} per-tag PNGs to {out_dir}"
                  + (f" via context='{effective_ctx_id}'" if use_context else ""))
        summary = header + ("\n" + "\n".join(summary_lines) if summary_lines else "")
        print(f"[BD PartsExport] {summary}", flush=True)
        return io.NodeOutput(parts, out_dir, summary, composite_tensor)


PARTS_EXPORT_V3_NODES = [BD_PartsExport]
PARTS_EXPORT_NODES = {"BD_PartsExport": BD_PartsExport}
PARTS_EXPORT_DISPLAY_NAMES = {"BD_PartsExport": "BD Parts Export"}
