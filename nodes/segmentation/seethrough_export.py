"""
Export SeeThrough's per-tag layers as individual RGBA PNG files.

SeeThrough's SEETHROUGH_PARTS is a wrapper dict from SeeThrough_PostProcess:
  {"tag2pinfo": {tag: part_info, ...}, "frame_size": (H, W)}

Each part_info has:
  - img:           HxWx4 uint8 RGBA (cropped to part's bbox after PostProcess)
  - xyxy:          [x1, y1, x2, y2] position in the original padded canvas
  - depth:         HxW uint8 depth map (matches img crop)
  - depth_median:  float z-order
  - tag:           string label (may contain "-l" / "-r" suffixes)

Each invocation creates a folder named like the standard BD output pattern,
and writes one PNG per tag inside it. Optionally also writes the depth maps.
"""

import os
import re
from glob import glob

import numpy as np
from PIL import Image

import folder_paths
from comfy_api.latest import io


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "tag"


def _resolve_output_folder(filename: str, name_prefix: str, auto_increment: bool) -> tuple[str, str]:
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


class BD_SeeThroughExportPNGs(io.ComfyNode):
    """Export each SeeThrough tag as a separate RGBA PNG file."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_SeeThroughExportPNGs",
            display_name="BD SeeThrough Export PNGs",
            category="🧠BrainDead/Segmentation",
            description=(
                "Take a SEETHROUGH_PARTS dict (from SeeThrough_PostProcess) and write each tag "
                "as its own RGBA PNG. Creates an output folder per run; one file per tag "
                "(plus optional depth grayscale)."
            ),
            inputs=[
                io.Custom("SEETHROUGH_PARTS").Input("parts"),
                io.String.Input(
                    "filename", default="seethrough",
                    tooltip="Base name for the output folder; tags become {tag}.png inside it.",
                ),
                io.String.Input(
                    "name_prefix", default="", optional=True,
                    tooltip="Prepended to filename. Supports subdirs (e.g. 'Characters/Letti').",
                ),
                io.Boolean.Input(
                    "auto_increment", default=True, optional=True,
                    tooltip="Append _001/_002/... so re-runs don't overwrite.",
                ),
                io.Boolean.Input(
                    "save_depth", default=False, optional=True,
                    tooltip="Also save per-tag depth maps as {tag}_depth.png (grayscale).",
                ),
                io.Boolean.Input(
                    "save_composite", default=True, optional=True,
                    tooltip="Also save the assembled character as {filename}_composite.png "
                            "(plus _composite_alpha.png for the assembled mask).",
                ),
                io.Int.Input(
                    "composite_size", default=0, min=0, max=8192, step=64, optional=True,
                    tooltip="Max side length of the composite. 0 = use frame_size as-is. "
                            "Set to e.g. 3072 if you upscaled parts 3x and want a matching output.",
                ),
                io.Boolean.Input(
                    "save_skin_composite", default=False, optional=True,
                    tooltip="Also save the assembled skin mask (union of per-part skin_masks) as "
                            "{filename}_composite_skin.png. Requires ExtractSkinMask upstream.",
                ),
            ],
            outputs=[
                io.Custom("SEETHROUGH_PARTS").Output(display_name="parts"),
                io.String.Output(display_name="output_dir"),
                io.String.Output(display_name="tag_summary"),
                io.Image.Output(display_name="composite_image"),
            ],
        )

    @classmethod
    def execute(cls, parts, filename="seethrough", name_prefix="",
                auto_increment=True, save_depth=False,
                save_composite=True, composite_size=0,
                save_skin_composite=False) -> io.NodeOutput:
        if not isinstance(parts, dict) or "tag2pinfo" not in parts:
            raise ValueError(
                f"BD_SeeThroughExportPNGs expects a SEETHROUGH_PARTS wrapper dict "
                f'with key "tag2pinfo", got {type(parts).__name__}'
            )
        tag2pinfo = parts["tag2pinfo"]

        folder, base = _resolve_output_folder(filename, name_prefix, auto_increment)

        written = []
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
            png_path = os.path.join(folder, f"{tag_safe}.png")
            Image.fromarray(arr.astype(np.uint8), mode=mode).save(png_path, optimize=True)
            written.append((tag, tag_safe, arr.shape, info.get("xyxy"), info.get("depth_median")))

            if save_depth and info.get("depth") is not None:
                depth_arr = np.asarray(info["depth"])
                if depth_arr.dtype != np.uint8:
                    if depth_arr.max() <= 1.0:
                        depth_arr = (depth_arr * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        depth_arr = depth_arr.clip(0, 255).astype(np.uint8)
                Image.fromarray(depth_arr, mode="L").save(
                    os.path.join(folder, f"{tag_safe}_depth.png"), optimize=True
                )

        summary_lines = [f"Exported {len(written)} tags to: {folder}"]
        for tag, safe, shape, xyxy, dm in written:
            dm_str = f"z={dm:.3f}" if isinstance(dm, (int, float)) else "z=?"
            xyxy_str = f"xyxy={xyxy}" if xyxy is not None else ""
            summary_lines.append(f"  {tag:20s} → {safe}.png  {shape}  {xyxy_str}  {dm_str}")

        composite_tensor = None
        if save_composite or save_skin_composite:
            from .seethrough_pipeline import BD_SeeThroughComposite
            comp_out = BD_SeeThroughComposite.execute(
                parts, output_size=composite_size,
                emit_skin_mask=save_skin_composite, trigger=True,
            )
            composite_tensor, alpha_tensor, skin_tensor = comp_out.args

            if save_composite:
                rgb = (composite_tensor[0].cpu().numpy() * 255).astype(np.uint8)
                a = (alpha_tensor[0].cpu().numpy() * 255).astype(np.uint8)
                rgba = np.concatenate([rgb, a[..., None]], axis=-1)
                comp_path = os.path.join(folder, f"{base}_composite.png")
                Image.fromarray(rgba, mode="RGBA").save(comp_path, optimize=True)
                Image.fromarray(a, mode="L").save(
                    os.path.join(folder, f"{base}_composite_alpha.png"), optimize=True
                )
                summary_lines.append(f"  → composite: {base}_composite.png  ({rgba.shape})")

            if save_skin_composite:
                sk = (skin_tensor[0].cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(sk, mode="L").save(
                    os.path.join(folder, f"{base}_composite_skin.png"), optimize=True
                )
                summary_lines.append(f"  → skin composite: {base}_composite_skin.png")

        if composite_tensor is None:
            tag2pinfo, frame_size = parts["tag2pinfo"], parts.get("frame_size", (64, 64))
            h, w = frame_size if frame_size != (0, 0) else (64, 64)
            composite_tensor = __import__("torch").zeros((1, h, w, 3))

        summary = "\n".join(summary_lines)
        print(f"[BD SeeThrough Export] {summary}", flush=True)

        return io.NodeOutput(parts, folder, summary, composite_tensor)


SEETHROUGH_EXPORT_V3_NODES = [BD_SeeThroughExportPNGs]
SEETHROUGH_EXPORT_NODES = {"BD_SeeThroughExportPNGs": BD_SeeThroughExportPNGs}
SEETHROUGH_EXPORT_DISPLAY_NAMES = {"BD_SeeThroughExportPNGs": "BD SeeThrough Export PNGs"}
