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
                "Crop a part to its content bounding box, then paste it — at original scale — "
                "onto a fixed-size canvas at a chosen anchor (center / edges / corners). Gives "
                "every part a common pivot so they overlap correctly when channel-packed or "
                "atlased. Content bbox comes from the wired mask, else the image's alpha (RGBA) "
                "or luminance (cutout). Canvas defaults to the input size; no rescale unless "
                "scale_to_fit is on. Outputs the recentered image, its mask, and a transform JSON."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask", optional=True,
                              tooltip="Optional mask defining the content bbox. If absent, bbox is taken "
                                      "from the image's alpha (RGBA) or luminance."),
                io.Combo.Input("anchor", options=_ANCHORS, default="center", optional=True,
                               tooltip="Where to place the cropped content on the canvas. center = shared pivot "
                                       "(parts overlap). top/bottom/left/right + corners anchor by that edge."),
                io.Int.Input("canvas_width", default=0, min=0, max=8192, step=8, optional=True,
                             tooltip="Target canvas width. 0 = use the input image width (the original size)."),
                io.Int.Input("canvas_height", default=0, min=0, max=8192, step=8, optional=True,
                             tooltip="Target canvas height. 0 = use the input image height."),
                io.Int.Input("pad", default=0, min=0, max=1024, step=1, optional=True,
                             tooltip="Pixels of margin added around the detected content before placing."),
                io.Float.Input("threshold", default=0.02, min=0.0, max=1.0, step=0.01, optional=True,
                               tooltip="Pixel value above which a pixel counts as content (for bbox detection)."),
                io.Boolean.Input("scale_to_fit", default=False, optional=True,
                                 tooltip="If the cropped content is larger than the canvas, scale it down to fit "
                                         "(preserves aspect). Off = keep original scale and clip any overflow."),
                io.String.Input("background_hex", default="#000000", optional=True,
                                tooltip="Canvas fill colour. Mask + (for RGBA) alpha pad regions are always 0."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.String.Output(display_name="transform"),
            ],
        )

    @classmethod
    def execute(cls, image, mask=None, anchor="center", canvas_width=0, canvas_height=0,
                pad=0, threshold=0.02, scale_to_fit=False, background_hex="#000000") -> io.NodeOutput:
        img = (image if image.ndim == 4 else image.unsqueeze(0)).detach().to("cpu").float()
        B, H, W, C = img.shape
        bg = _hex_to_rgb(background_hex)
        cw = int(canvas_width) if canvas_width > 0 else W
        ch = int(canvas_height) if canvas_height > 0 else H
        thr = float(threshold)

        # mask frames aligned to the batch (optional)
        mframes = None
        if mask is not None:
            m = mask.detach().to("cpu").float()
            if m.ndim == 2:
                m = m.unsqueeze(0)
            if m.ndim == 4:
                m = m.squeeze(-1) if m.shape[-1] == 1 else m[:, 0]
            mframes = m

        out_imgs, out_masks, transforms = [], [], []
        for i in range(B):
            frame = img[i]                       # (H, W, C)
            # ── detect content ──
            if mframes is not None:
                mm = mframes[min(i, mframes.shape[0] - 1)]
                if mm.shape != (H, W):
                    mm = _resize_hw(mm.unsqueeze(-1), H, W)[..., 0]
                active = (mm >= thr).numpy()
                content_mask = mm.clamp(0, 1)
            elif C >= 4:
                content_mask = frame[..., 3].clamp(0, 1)
                active = (content_mask >= thr).numpy()
            else:
                content_mask = (frame[..., :3] * torch.tensor([0.2126, 0.7152, 0.0722])).sum(-1).clamp(0, 1)
                active = (content_mask >= thr).numpy()

            bb = _bbox(active)
            # ── build canvas ──
            canvas = torch.empty((ch, cw, C), dtype=torch.float32)
            canvas[..., 0], canvas[..., 1], canvas[..., 2] = bg
            if C >= 4:
                canvas[..., 3] = 0.0
            mcanvas = torch.zeros((ch, cw), dtype=torch.float32)

            if bb is None:
                out_imgs.append(canvas); out_masks.append(mcanvas)
                transforms.append({"frame": i, "bbox": None, "note": "no content"})
                continue

            x1, y1, x2, y2 = bb
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
            crop = frame[y1:y2, x1:x2, :]
            cmask = content_mask[y1:y2, x1:x2]
            bh, bw = crop.shape[0], crop.shape[1]

            scale = 1.0
            if scale_to_fit and (bw > cw or bh > ch):
                scale = min(cw / bw, ch / bh)
                nh, nw = max(1, round(bh * scale)), max(1, round(bw * scale))
                crop = _resize_hw(crop, nh, nw)
                cmask = _resize_hw(cmask.unsqueeze(-1), nh, nw)[..., 0]
                bh, bw = nh, nw

            ox, oy = _anchor_offset(anchor, cw, ch, bw, bh)
            # clip to canvas (overflow when not scaling)
            sx, sy = max(0, -ox), max(0, -oy)
            dx, dy = max(0, ox), max(0, oy)
            pw = min(bw - sx, cw - dx)
            ph = min(bh - sy, ch - dy)
            if pw > 0 and ph > 0:
                canvas[dy:dy + ph, dx:dx + pw, :] = crop[sy:sy + ph, sx:sx + pw, :]
                mcanvas[dy:dy + ph, dx:dx + pw] = cmask[sy:sy + ph, sx:sx + pw]

            out_imgs.append(canvas.clamp(0, 1))
            out_masks.append(mcanvas.clamp(0, 1))
            transforms.append({
                "frame": i, "src_bbox": [x1, y1, x2, y2], "content_size": [bw, bh],
                "canvas": [cw, ch], "anchor": anchor, "placed_at": [int(dx), int(dy)],
                "scale": round(float(scale), 4),
            })

        out_img = torch.stack(out_imgs, dim=0)
        out_mask = torch.stack(out_masks, dim=0)
        print(f"[BD_CropAndCenter] {B} frame(s) -> {cw}x{ch} canvas, anchor={anchor}, "
              f"scale_to_fit={'yes' if scale_to_fit else 'no'}")
        return io.NodeOutput(out_img, out_mask, json.dumps(transforms if B > 1 else transforms[0]))


CROP_AND_CENTER_V3_NODES = [BD_CropAndCenter]
CROP_AND_CENTER_NODES = {"BD_CropAndCenter": BD_CropAndCenter}
CROP_AND_CENTER_DISPLAY_NAMES = {"BD_CropAndCenter": "BD Crop and Center"}
