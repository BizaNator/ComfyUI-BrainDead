"""
BD_AtlasPack — tile multiple images into a single grid atlas (rows x columns, padding).

Inverse-of-a-spritesheet: takes N images and lays them out in a uniform cols x rows
grid with configurable per-cell size, fit mode, padding, and background colour.

Common use cases:
- Viseme / sprite atlases (e.g. 7 lip visemes -> one 2048x1024 sheet)
- Game-engine texture sheets where each cell is a separate baked map
- Contact sheets / preview grids for batched outputs

Images are collected from THREE sources and concatenated IN THIS ORDER:
  1. images          — a batched IMAGE (B, H, W, C)
  2. image_1..image_8 — individual IMAGE slots (each one cell)
  3. masks           — a batched MASK; mask[i] becomes the alpha for cell i
                       (falls back to an RGBA image's own alpha, else fully opaque)

Outputs the packed atlas IMAGE (RGB, or RGBA when output_alpha is on), the matching
single-channel mask atlas, and a JSON `layout` string with per-cell pixel rects +
normalised UV rects — handy for feeding a game engine / shader the cell coordinates.
"""

import json
import math

import torch
import torch.nn.functional as F

from comfy_api.latest import io


def _hex_to_rgb(s: str) -> tuple[float, float, float]:
    s = (s or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return (0.0, 0.0, 0.0)
    try:
        return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore
    except ValueError:
        return (0.0, 0.0, 0.0)


def _img_frames(t: torch.Tensor | None) -> list[torch.Tensor]:
    """Normalise an IMAGE input to a list of (H, W, C) float CPU tensors (one per frame)."""
    if t is None:
        return []
    x = t.detach().to("cpu").float()
    if x.ndim == 3:           # (H, W, C)
        x = x.unsqueeze(0)
    if x.ndim != 4:
        return []
    return [x[i] for i in range(x.shape[0])]


def _mask_frames(m: torch.Tensor | None) -> list[torch.Tensor]:
    """Normalise a MASK input to a list of (H, W) float CPU tensors."""
    if m is None:
        return []
    x = m.detach().to("cpu").float()
    if x.ndim == 2:           # (H, W)
        x = x.unsqueeze(0)
    if x.ndim == 4:           # (B, H, W, 1) or (B, 1, H, W)
        x = x.squeeze(-1) if x.shape[-1] == 1 else x[:, 0]
    if x.ndim != 3:
        return []
    return [x[i] for i in range(x.shape[0])]


def _resize_hw(img: torch.Tensor, th: int, tw: int, mode: str = "bilinear") -> torch.Tensor:
    """Resize an (H, W, C) tensor to (th, tw, C)."""
    nchw = img.permute(2, 0, 1).unsqueeze(0)
    out = F.interpolate(nchw, size=(th, tw), mode=mode,
                        align_corners=False if mode == "bilinear" else None)
    return out.squeeze(0).permute(1, 2, 0)


def _fit_into_cell(img: torch.Tensor, mask: torch.Tensor,
                   ch: int, cw: int, fit: str,
                   bg: tuple[float, float, float]) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit one (H,W,3) image + (H,W) mask into a (ch, cw) cell. Returns (cell_rgb, cell_mask)."""
    H, W = img.shape[0], img.shape[1]
    cell = torch.empty((ch, cw, 3), dtype=torch.float32)
    cell[..., 0], cell[..., 1], cell[..., 2] = bg[0], bg[1], bg[2]
    cmask = torch.zeros((ch, cw), dtype=torch.float32)
    if H == 0 or W == 0:
        return cell, cmask

    if fit == "stretch":
        cell = _resize_hw(img, ch, cw).clamp(0, 1)
        cmask = _resize_hw(mask.unsqueeze(-1), ch, cw)[..., 0].clamp(0, 1)
        return cell, cmask

    scale = (max(cw / W, ch / H) if fit == "cover" else min(cw / W, ch / H))
    nw, nh = max(1, round(W * scale)), max(1, round(H * scale))
    r_img = _resize_hw(img, nh, nw).clamp(0, 1)
    r_msk = _resize_hw(mask.unsqueeze(-1), nh, nw)[..., 0].clamp(0, 1)

    if fit == "cover":   # center-crop the oversized image down to the cell
        y0, x0 = max(0, (nh - ch) // 2), max(0, (nw - cw) // 2)
        cell = r_img[y0:y0 + ch, x0:x0 + cw, :]
        cmask = r_msk[y0:y0 + ch, x0:x0 + cw]
    else:                # contain — center-paste onto the bg-filled cell
        y0, x0 = (ch - nh) // 2, (cw - nw) // 2
        cell[y0:y0 + nh, x0:x0 + nw, :] = r_img
        cmask[y0:y0 + nh, x0:x0 + nw] = r_msk
    return cell, cmask


class BD_AtlasPack(io.ComfyNode):
    """Tile a batch + individual images into a single cols x rows grid atlas."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_AtlasPack",
            display_name="BD Atlas Pack",
            category="🧠BrainDead/Segmentation",
            description=(
                "Pack multiple images into one grid atlas (rows x columns) with padding. "
                "Sources are concatenated in order: the `images` batch, then image_1..image_8, "
                "then `masks` (batch) supplies each cell's alpha (falls back to an RGBA image's "
                "own alpha, else opaque). Each image is fit into a uniform cell via contain / "
                "cover / stretch. Outputs the atlas image, a single-channel mask atlas, and a "
                "JSON layout string with per-cell pixel + normalised-UV rects for game engines."
            ),
            inputs=[
                io.Image.Input("images", optional=True,
                               tooltip="Batched IMAGE (B,H,W,C) — each frame becomes one cell, in batch order."),
                io.Image.Input("image_1", optional=True, tooltip="Individual image for a cell (appended after the batch)."),
                io.Image.Input("image_2", optional=True),
                io.Image.Input("image_3", optional=True),
                io.Image.Input("image_4", optional=True),
                io.Image.Input("image_5", optional=True),
                io.Image.Input("image_6", optional=True),
                io.Image.Input("image_7", optional=True),
                io.Image.Input("image_8", optional=True),
                io.Mask.Input("masks", optional=True,
                              tooltip="Batched MASK — mask[i] is the alpha for cell i (aligned to the concatenated "
                                      "image order). If absent, an RGBA image's own alpha is used, else opaque."),
                io.Int.Input("columns", default=0, min=0, max=64, step=1, optional=True,
                             tooltip="Number of columns. 0 = auto (≈ square: ceil(sqrt(count)))."),
                io.Int.Input("rows", default=0, min=0, max=64, step=1, optional=True,
                             tooltip="Number of rows. 0 = auto (ceil(count / columns)). Expanded if too small to fit all cells."),
                io.Int.Input("cell_width", default=0, min=0, max=8192, step=8, optional=True,
                             tooltip="Pixel width of each cell. 0 = use the widest input image."),
                io.Int.Input("cell_height", default=0, min=0, max=8192, step=8, optional=True,
                             tooltip="Pixel height of each cell. 0 = use the tallest input image."),
                io.Combo.Input("fit_mode", options=["contain", "cover", "stretch"], default="contain", optional=True,
                               tooltip="contain — fit whole image in cell, pad with background (no crop). "
                                       "cover — fill cell, center-crop overflow. stretch — resize to cell, ignore aspect."),
                io.Int.Input("padding", default=0, min=0, max=512, step=1, optional=True,
                             tooltip="Pixels of background between cells AND around the outer border."),
                io.String.Input("background_hex", default="#000000", optional=True,
                                tooltip="Background / padding fill colour (hex). Mask atlas pad regions are always 0."),
                io.Combo.Input("order", options=["row_major", "column_major"], default="row_major", optional=True,
                               tooltip="row_major — fill left-to-right then down. column_major — fill top-to-bottom then right."),
                io.Boolean.Input("output_alpha", default=False, optional=True,
                                 tooltip="If True, the atlas IMAGE is RGBA (alpha = the mask atlas). Default is RGB."),
            ],
            outputs=[
                io.Image.Output(display_name="atlas"),
                io.Mask.Output(display_name="mask"),
                io.String.Output(display_name="layout"),
            ],
        )

    @classmethod
    def execute(cls, images=None, image_1=None, image_2=None, image_3=None, image_4=None,
                image_5=None, image_6=None, image_7=None, image_8=None, masks=None,
                columns=0, rows=0, cell_width=0, cell_height=0, fit_mode="contain",
                padding=0, background_hex="#000000", order="row_major",
                output_alpha=False) -> io.NodeOutput:

        # ── 1. collect images: batch first, then individual slots ──
        frames: list[torch.Tensor] = list(_img_frames(images))
        for slot in (image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8):
            frames.extend(_img_frames(slot))

        bg = _hex_to_rgb(background_hex)

        if not frames:
            cw = max(cell_width, 64)
            ch = max(cell_height, 64)
            blank = torch.empty((1, ch, cw, 3), dtype=torch.float32)
            blank[..., 0], blank[..., 1], blank[..., 2] = bg
            print("[BD_AtlasPack] no images wired — emitting blank tile")
            return io.NodeOutput(blank, torch.zeros((1, ch, cw), dtype=torch.float32),
                                 json.dumps({"count": 0, "columns": 0, "rows": 0}))

        # per-cell mask + rgb sources
        masklist = _mask_frames(masks)
        rgb_frames, mask_for_frame = [], []
        for i, f in enumerate(frames):
            nchan = f.shape[-1]
            rgb_frames.append(f[..., :3] if nchan >= 3 else f[..., :1].expand(-1, -1, 3))
            if i < len(masklist):
                mask_for_frame.append(masklist[i])
            elif nchan >= 4:
                mask_for_frame.append(f[..., 3])
            else:
                mask_for_frame.append(torch.ones(f.shape[0], f.shape[1], dtype=torch.float32))

        n = len(rgb_frames)

        # ── 2. grid geometry ──
        cols = int(columns) if columns > 0 else max(1, math.ceil(math.sqrt(n)))
        rws = int(rows) if rows > 0 else math.ceil(n / cols)
        while cols * rws < n:        # never silently drop cells
            rws += 1
        cw = int(cell_width) if cell_width > 0 else max(f.shape[1] for f in rgb_frames)
        ch = int(cell_height) if cell_height > 0 else max(f.shape[0] for f in rgb_frames)
        pad = int(padding)

        atlas_w = cols * cw + (cols + 1) * pad
        atlas_h = rws * ch + (rws + 1) * pad
        atlas = torch.empty((atlas_h, atlas_w, 3), dtype=torch.float32)
        atlas[..., 0], atlas[..., 1], atlas[..., 2] = bg
        mask_atlas = torch.zeros((atlas_h, atlas_w), dtype=torch.float32)

        # ── 3. place each cell ──
        cells_meta = []
        for idx in range(n):
            if order == "column_major":
                c, r = idx // rws, idx % rws
            else:
                r, c = idx // cols, idx % cols
            x = pad + c * (cw + pad)
            y = pad + r * (ch + pad)
            cell_rgb, cell_msk = _fit_into_cell(rgb_frames[idx], mask_for_frame[idx], ch, cw, fit_mode, bg)
            atlas[y:y + ch, x:x + cw, :] = cell_rgb
            mask_atlas[y:y + ch, x:x + cw] = cell_msk
            cells_meta.append({
                "index": idx, "row": r, "col": c,
                "x": x, "y": y, "w": cw, "h": ch,
                "uv": [round(x / atlas_w, 6), round(y / atlas_h, 6),
                       round((x + cw) / atlas_w, 6), round((y + ch) / atlas_h, 6)],
            })

        atlas = atlas.clamp(0, 1).unsqueeze(0)
        mask_atlas = mask_atlas.clamp(0, 1).unsqueeze(0)
        if output_alpha:
            atlas = torch.cat([atlas, mask_atlas.unsqueeze(-1)], dim=-1)

        layout = {
            "count": n, "columns": cols, "rows": rws,
            "cell_width": cw, "cell_height": ch, "padding": pad,
            "atlas_width": atlas_w, "atlas_height": atlas_h,
            "fit_mode": fit_mode, "order": order,
            "cells": cells_meta,
        }
        print(f"[BD_AtlasPack] {n} images -> {cols}x{rws} grid, cell {cw}x{ch}, "
              f"atlas {atlas_w}x{atlas_h}, pad {pad}, rgba={'yes' if output_alpha else 'no'}")
        return io.NodeOutput(atlas, mask_atlas, json.dumps(layout))


ATLAS_PACK_V3_NODES = [BD_AtlasPack]
ATLAS_PACK_NODES = {"BD_AtlasPack": BD_AtlasPack}
ATLAS_PACK_DISPLAY_NAMES = {"BD_AtlasPack": "BD Atlas Pack"}
