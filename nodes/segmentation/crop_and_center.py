"""
BD_CropAndCenter — crop a part to its content bbox, then re-place it (at original
scale) onto a fixed-size canvas at a chosen anchor.

Why: game engines extract assets from a **common pivot**. A part masked out of a
1024×1024 image might only occupy a 200×768 region sitting wherever it was in the
source. To pack several parts into the same channel-pack / atlas and have them
*overlap on a shared pivot*, each part must be cropped to its own content and then
re-centered (or edge-anchored) on a canvas of the same size (normally the original).

  pants @ (somewhere) in 1024×1024  ->  crop to 200×768  ->  paste centered on 1024×1024

Detection: if a `mask` is wired, its bbox defines the content; otherwise the bbox is
taken from the image's alpha (if RGBA) or luminance (cutout greyscale). No scaling is
applied unless `scale_to_fit` is on. Output keeps the input's channel count.
"""

import json

import numpy as np
import torch

from comfy_api.latest import io


_ANCHORS = ["center", "top", "bottom", "left", "right",
            "top_left", "top_right", "bottom_left", "bottom_right"]


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


def _bbox(active: np.ndarray):
    rows = np.any(active, axis=1)
    cols = np.any(active, axis=0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return int(x1), int(y1), int(x2) + 1, int(y2) + 1


def _anchor_offset(anchor: str, cw: int, ch: int, bw: int, bh: int) -> tuple[int, int]:
    h = "left" if "left" in anchor else ("right" if "right" in anchor else "center")
    v = "top" if "top" in anchor else ("bottom" if "bottom" in anchor else "center")
    x = 0 if h == "left" else (cw - bw if h == "right" else (cw - bw) // 2)
    y = 0 if v == "top" else (ch - bh if v == "bottom" else (ch - bh) // 2)
    return x, y


def _resize_hw(t: torch.Tensor, th: int, tw: int) -> torch.Tensor:
    nchw = t.permute(2, 0, 1).unsqueeze(0)
    out = torch.nn.functional.interpolate(nchw, size=(th, tw), mode="bilinear", align_corners=False)
    return out.squeeze(0).permute(1, 2, 0)


class BD_CropAndCenter(io.ComfyNode):
    """Crop a part to its content bbox and re-place it (no rescale) on a fixed canvas at an anchor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CropAndCenter",
            display_name="BD Crop and Center",
            category="🧠BrainDead/Segmentation",
            description=(
                "Crop an image to the bounding box of its mask (like AILab Crop-to-Object), with "
                "optional centering. Bbox comes from the wired mask, else the image's alpha (RGBA) "
                "or its difference from the border background colour.\n"
                "  canvas_width = canvas_height = 0  → output the TIGHT CROP (just the bbox region).\n"
                "  canvas_width / canvas_height > 0  → place the crop on that canvas at the anchor "
                "(center / edges / corners), with optional scale_to_fit — this is the centering option.\n"
                "flatten_to_mask makes content show only inside the mask (background fills the rest of "
                "the crop). Outputs the cropped image, its mask, and a transform JSON."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask", optional=True,
                              tooltip="Mask defining the crop region (its bounding box). If absent, the bbox is "
                                      "taken from the image's alpha (RGBA) or its difference from the border bg."),
                io.Combo.Input("anchor", options=_ANCHORS, default="center", optional=True,
                               tooltip="Placement on the canvas (only used when canvas_width/height > 0). "
                                       "center = shared pivot; top/bottom/left/right + corners anchor by that edge."),
                io.Int.Input("canvas_width", default=0, min=0, max=8192, step=8, optional=True,
                             tooltip="0 = output the TIGHT CROP (no canvas). >0 = place the crop on a canvas this "
                                     "wide at the anchor. (If only one of width/height is >0, the other uses the input size.)"),
                io.Int.Input("canvas_height", default=0, min=0, max=8192, step=8, optional=True,
                             tooltip="0 = tight crop (no canvas). >0 = canvas height for centering."),
                io.Int.Input("pad", default=0, min=0, max=1024, step=1, optional=True,
                             tooltip="Pixels of margin added around the mask bbox before cropping (AILab-style padding)."),
                io.Float.Input("threshold", default=0.02, min=0.0, max=1.0, step=0.01, optional=True,
                               tooltip="Mask/alpha value above which a pixel counts as content (for the bbox)."),
                io.Boolean.Input("scale_to_fit", default=False, optional=True,
                                 tooltip="Canvas mode only. On = scale the crop to FIT the canvas (longest side → "
                                         "canvas), preserving aspect (up or down), then anchor. Off = keep original "
                                         "pixel scale (re-pivot only)."),
                io.String.Input("background_hex", default="#000000", optional=True,
                                tooltip="Background fill for masked-out / padded regions."),
                io.Boolean.Input("flatten_to_mask", default=True, optional=True,
                                 tooltip="On = content shows only inside the mask; the background colour fills the "
                                         "rest of the crop (and alpha = mask for RGBA). Off = output the raw "
                                         "rectangular crop (exactly like AILab Crop-to-Object)."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.String.Output(display_name="transform"),
            ],
        )

    @classmethod
    def execute(cls, image, mask=None, anchor="center", canvas_width=0, canvas_height=0,
                pad=0, threshold=0.02, scale_to_fit=False, background_hex="#000000",
                flatten_to_mask=True) -> io.NodeOutput:
        img = (image if image.ndim == 4 else image.unsqueeze(0)).detach().to("cpu").float()
        B, H, W, C = img.shape
        bg = _hex_to_rgb(background_hex)
        bg_t = torch.tensor(bg, dtype=torch.float32)
        thr = float(threshold)
        use_canvas = (int(canvas_width) > 0 or int(canvas_height) > 0)

        # mask frames aligned to the batch (optional)
        mframes = None
        if mask is not None:
            m = mask.detach().to("cpu").float()
            if m.ndim == 2:
                m = m.unsqueeze(0)
            if m.ndim == 4:
                m = m.squeeze(-1) if m.shape[-1] == 1 else m[:, 0]
            mframes = m

        # ── 1. per-frame: detect content → crop tight to the mask bbox → optional flatten ──
        crops, cmasks, transforms = [], [], []
        for i in range(B):
            frame = img[i]                       # (H, W, C)
            if mframes is not None:
                mm = mframes[min(i, mframes.shape[0] - 1)]
                if mm.shape != (H, W):
                    mm = _resize_hw(mm.unsqueeze(-1), H, W)[..., 0]
                content_mask = mm.clamp(0, 1)
            elif C >= 4:
                content_mask = frame[..., 3].clamp(0, 1)
            else:
                # No mask/alpha: content = pixels DIFFERING from the border background colour
                # (raw luminance reads a white/light bg as content).
                rgb = frame[..., :3]
                border = torch.cat([rgb[0, :], rgb[-1, :], rgb[:, 0], rgb[:, -1]], dim=0).reshape(-1, 3)
                diff = (rgb - border.median(dim=0).values).abs().amax(dim=-1).clamp(0, 1)
                content_mask = (diff >= thr).float()
            active = (content_mask >= thr).numpy()

            bb = _bbox(active)
            if bb is None:                       # nothing detected → keep the whole frame
                x1, y1, x2, y2 = 0, 0, W, H
            else:
                x1, y1, x2, y2 = bb
                x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
                x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)

            crop = frame[y1:y2, x1:x2, :].clone()             # the TIGHT crop (AILab core)
            cmask = content_mask[y1:y2, x1:x2].clamp(0, 1)
            if flatten_to_mask:                               # content only inside the mask
                cm3 = cmask.unsqueeze(-1)
                crop[..., :3] = crop[..., :3] * cm3 + bg_t * (1.0 - cm3)
                if C >= 4:
                    crop[..., 3] = cmask
            crops.append(crop); cmasks.append(cmask)
            transforms.append({"frame": i, "src_bbox": [x1, y1, x2, y2],
                               "crop_size": [x2 - x1, y2 - y1]})

        # ── 2a. TIGHT-CROP output (canvas off) — pad batch to a common max size so frames stack ──
        if not use_canvas:
            mh = max(c.shape[0] for c in crops); mw = max(c.shape[1] for c in crops)
            out_imgs, out_masks = [], []
            for crop, cmask, t in zip(crops, cmasks, transforms):
                bh, bw = crop.shape[0], crop.shape[1]
                ox, oy = _anchor_offset(anchor, mw, mh, bw, bh)
                canvas = torch.empty((mh, mw, C), dtype=torch.float32)
                canvas[..., 0], canvas[..., 1], canvas[..., 2] = bg
                if C >= 4:
                    canvas[..., 3] = 0.0
                mcanvas = torch.zeros((mh, mw), dtype=torch.float32)
                canvas[oy:oy + bh, ox:ox + bw, :] = crop
                mcanvas[oy:oy + bh, ox:ox + bw] = cmask
                out_imgs.append(canvas.clamp(0, 1)); out_masks.append(mcanvas)
                t.update({"output": [mw, mh], "placed_at": [ox, oy], "mode": "crop"})
            print(f"[BD_CropAndCenter] {B} frame(s) → TIGHT CROP {mw}x{mh} "
                  f"(flatten={'yes' if flatten_to_mask else 'no'})")
            return io.NodeOutput(torch.stack(out_imgs, 0), torch.stack(out_masks, 0),
                                 json.dumps(transforms if B > 1 else transforms[0]))

        # ── 2b. CANVAS output (centering) — place each crop on the canvas at the anchor ──
        cw = int(canvas_width) if canvas_width > 0 else W
        ch = int(canvas_height) if canvas_height > 0 else H
        out_imgs, out_masks = [], []
        for crop, cmask, t in zip(crops, cmasks, transforms):
            bh, bw = crop.shape[0], crop.shape[1]
            scale = 1.0
            if scale_to_fit:
                scale = min(cw / bw, ch / bh)         # fit longest side (up or down), keep aspect
                nh, nw = max(1, round(bh * scale)), max(1, round(bw * scale))
                crop = _resize_hw(crop, nh, nw)
                cmask = _resize_hw(cmask.unsqueeze(-1), nh, nw)[..., 0]
                bh, bw = nh, nw
            canvas = torch.empty((ch, cw, C), dtype=torch.float32)
            canvas[..., 0], canvas[..., 1], canvas[..., 2] = bg
            if C >= 4:
                canvas[..., 3] = 0.0
            mcanvas = torch.zeros((ch, cw), dtype=torch.float32)
            ox, oy = _anchor_offset(anchor, cw, ch, bw, bh)
            sx, sy = max(0, -ox), max(0, -oy)
            dx, dy = max(0, ox), max(0, oy)
            pw = min(bw - sx, cw - dx); ph = min(bh - sy, ch - dy)
            if pw > 0 and ph > 0:
                canvas[dy:dy + ph, dx:dx + pw, :] = crop[sy:sy + ph, sx:sx + pw, :]
                mcanvas[dy:dy + ph, dx:dx + pw] = cmask[sy:sy + ph, sx:sx + pw]
            out_imgs.append(canvas.clamp(0, 1)); out_masks.append(mcanvas)
            t.update({"output": [cw, ch], "placed_at": [int(dx), int(dy)],
                      "scale": round(float(scale), 4), "anchor": anchor, "mode": "canvas"})
        print(f"[BD_CropAndCenter] {B} frame(s) → CANVAS {cw}x{ch} anchor={anchor} "
              f"scale_to_fit={'yes' if scale_to_fit else 'no'}")
        return io.NodeOutput(torch.stack(out_imgs, 0), torch.stack(out_masks, 0),
                             json.dumps(transforms if B > 1 else transforms[0]))


CROP_AND_CENTER_V3_NODES = [BD_CropAndCenter]
CROP_AND_CENTER_NODES = {"BD_CropAndCenter": BD_CropAndCenter}
CROP_AND_CENTER_DISPLAY_NAMES = {"BD_CropAndCenter": "BD Crop and Center"}
