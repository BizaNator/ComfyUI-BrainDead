"""
BD_CropToMask — crop an image to the bounding box of its active mask region.

Finds the bounding box of the active region (all channels or a specific channel),
adds proportional padding, optionally enforces a square crop, then resizes to
the target resolution.

For lip atlas normalization, set channel="R" (lips only) and use a consistent
fixed_size_px across all 7 visemes so every cell shows the lips at the same
scale and center UV position, regardless of how much teeth/tongue is visible.
"""

import numpy as np
import torch
from comfy_api.latest import io


def _to_hw(mask: torch.Tensor) -> np.ndarray:
    m = mask.detach().cpu().float()
    if m.ndim == 3:
        m = m[0]
    return m.numpy().astype(np.float32)


def _bbox_of_mask(binary: np.ndarray):
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return int(x1), int(y1), int(x2) + 1, int(y2) + 1


def _add_padding(x1, y1, x2, y2, pad_px: int, H: int, W: int):
    return (max(0, x1 - pad_px), max(0, y1 - pad_px),
            min(W, x2 + pad_px), min(H, y2 + pad_px))


def _make_square(x1, y1, x2, y2, H: int, W: int):
    bw, bh = x2 - x1, y2 - y1
    size = max(bw, bh)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(W, x1 + size)
    y2 = min(H, y1 + size)
    if x2 - x1 < size:
        x1 = max(0, x2 - size)
    if y2 - y1 < size:
        y1 = max(0, y2 - size)
    return x1, y1, x2, y2


def _center_fixed(cx: int, cy: int, size: int, H: int, W: int):
    """Return unclamped square crop coords centered on (cx,cy). May exceed image bounds — caller pads."""
    half = size // 2
    return cx - half, cy - half, cx - half + size, cy - half + size


def _pad_and_crop(img: torch.Tensor, x1: int, y1: int, x2: int, y2: int) -> torch.Tensor:
    """Extract crop [y1:y2, x1:x2] from img, zero-padding any out-of-bounds regions."""
    B, H, W, C = img.shape
    size = x2 - x1  # assumed square

    # Intersect with image bounds
    ix1, iy1 = max(0, x1), max(0, y1)
    ix2, iy2 = min(W, x2), min(H, y2)

    # Destination paste coords
    dx1 = ix1 - x1
    dy1 = iy1 - y1

    out = torch.zeros(B, size, size, C, dtype=img.dtype, device=img.device)
    if ix2 > ix1 and iy2 > iy1:
        out[:, dy1:dy1+(iy2-iy1), dx1:dx1+(ix2-ix1), :] = img[:, iy1:iy2, ix1:ix2, :]
    return out


class BD_CropToMask(io.ComfyNode):
    """Crop image to bounding box of active region, with optional channel selection and fixed-size mode for consistent UV atlas normalization."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CropToMask",
            display_name="BD Crop To Mask",
            category="🧠BrainDead/Segmentation",
            description=(
                "Crop an image to the bounding box of its active region, add proportional "
                "padding, optionally enforce a square crop, then resize to a target resolution. "
                "Use channel='R' to center/size the crop on lips only (not teeth/tongue). "
                "Use fixed_size_px to force the same crop size across all visemes for consistent "
                "UV atlas framing — lips always appear at the same scale and center position."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask", optional=True),
                io.Combo.Input(
                    "channel",
                    options=["all", "R", "G", "B", "A"],
                    default="R", optional=True,
                    tooltip="Which image channel drives the bounding box detection.\n"
                            "  all — union of R/G/B (lips+teeth+tongue)\n"
                            "  R   — lips only (recommended for consistent atlas framing)\n"
                            "  G   — teeth only\n"
                            "  B   — tongue only\n"
                            "  A   — alpha/depth channel",
                ),
                io.Float.Input(
                    "threshold", default=0.05, min=0.0, max=1.0, step=0.01, optional=True,
                    tooltip="Pixel value above which a pixel counts as active.",
                ),
                io.Float.Input(
                    "padding_pct", default=0.15, min=0.0, max=0.5, step=0.01, optional=True,
                    tooltip="Padding on each side as a fraction of the larger bbox dimension. "
                            "0.15 = 15%% per side. Used when fixed_size_px=0.",
                ),
                io.Int.Input(
                    "fixed_size_px", default=0, min=0, max=4096, step=8, optional=True,
                    tooltip="When >0, force the crop to exactly this square size (in source pixels) "
                            "centered on the detected channel centroid. Use the SAME value across "
                            "all 7 visemes for a consistent atlas — every cell covers the same "
                            "mouth region at the same scale.\n\n"
                            "Tip: run once with fixed_size_px=0 to see crop sizes in the status "
                            "output, then set fixed_size_px to the largest bbox+padding size.",
                ),
                io.Boolean.Input(
                    "square", default=True, optional=True,
                    tooltip="Enforce a square crop. Ignored when fixed_size_px>0 (already square).",
                ),
                io.Int.Input(
                    "output_size", default=512, min=64, max=4096, step=64, optional=True,
                    tooltip="Output resolution (square).",
                ),
                io.Combo.Input(
                    "resize_mode",
                    options=["fit", "stretch"],
                    default="fit", optional=True,
                    tooltip="fit — scale to fill output_size preserving aspect ratio, then pad edges with black. "
                            "Lips keep their natural shape; horizontal visemes get top/bottom bars, "
                            "vertical visemes get left/right bars.\n"
                            "stretch — squash to exact output_size (may distort lip shape).",
                ),
                io.Combo.Input(
                    "fallback",
                    options=["full_image", "center_crop"],
                    default="full_image", optional=True,
                ),
            ],
            outputs=[
                io.Image.Output(display_name="cropped"),
                io.String.Output(display_name="crop_box"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, image, mask=None, channel="R", threshold=0.05,
                padding_pct=0.15, fixed_size_px=0, square=True,
                output_size=512, resize_mode="fit", fallback="full_image") -> io.NodeOutput:
        import torch.nn.functional as F

        img = image if image.ndim == 4 else image.unsqueeze(0)
        B, H, W, C = img.shape
        thr = float(threshold)

        # Select detection channel
        if mask is not None:
            act = _to_hw(mask) >= thr
        else:
            rgb = img[0, ..., :3].detach().cpu().float().numpy()
            ch = channel.upper()
            if ch == "R" and C >= 1:
                act = rgb[..., 0] >= thr
            elif ch == "G" and C >= 2:
                act = rgb[..., 1] >= thr
            elif ch == "B" and C >= 3:
                act = rgb[..., 2] >= thr
            elif ch == "A" and C >= 4:
                act = img[0, ..., 3].detach().cpu().float().numpy() >= thr
            else:
                act = rgb.max(axis=-1) >= thr

        bbox = _bbox_of_mask(act)
        status_note = "bbox found"

        if bbox is None:
            if fallback == "center_crop":
                size = min(H, W)
                x1 = (W - size) // 2; y1 = (H - size) // 2
                bbox = (x1, y1, x1 + size, y1 + size)
            else:
                bbox = (0, 0, W, H)
            status_note = f"no active pixels → {fallback}"

        x1, y1, x2, y2 = bbox

        if int(fixed_size_px) > 0:
            # Fixed-size mode: center on bbox centroid, zero-pad if needed
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            size = int(fixed_size_px)
            x1, y1, x2, y2 = _center_fixed(cx, cy, size, H, W)
            status_note += f" fixed={size}px"
        else:
            bw, bh = x2 - x1, y2 - y1
            pad_px = int(max(bw, bh) * float(padding_pct))
            x1, y1, x2, y2 = _add_padding(x1, y1, x2, y2, pad_px, H, W)
            if square:
                x1, y1, x2, y2 = _make_square(x1, y1, x2, y2, H, W)

        crop_box = f"{x1},{y1},{x2},{y2}"
        cw, ch_val = x2 - x1, y2 - y1

        # Use pad-and-crop when fixed_size_px may exceed image bounds
        if int(fixed_size_px) > 0:
            cropped = _pad_and_crop(img, x1, y1, x2, y2)
        else:
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(W, x2), min(H, y2)
            cropped = img[:, y1c:y2c, x1c:x2c, :]

        sz = int(output_size)
        nchw = cropped.permute(0, 3, 1, 2).float()

        if resize_mode == "fit":
            # Scale to fit within sz×sz preserving aspect ratio, then center-pad
            _, _, ch_h, ch_w = nchw.shape
            scale = min(sz / ch_w, sz / ch_h)
            new_w = max(1, int(ch_w * scale))
            new_h = max(1, int(ch_h * scale))
            scaled = F.interpolate(nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
            pad_top = (sz - new_h) // 2
            pad_left = (sz - new_w) // 2
            canvas = torch.zeros(nchw.shape[0], nchw.shape[1], sz, sz,
                                 dtype=nchw.dtype, device=nchw.device)
            canvas[:, :, pad_top:pad_top+new_h, pad_left:pad_left+new_w] = scaled
            result = canvas.permute(0, 2, 3, 1).clamp(0, 1)
        else:
            resized = F.interpolate(nchw, size=(sz, sz), mode="bilinear", align_corners=False)
            result = resized.permute(0, 2, 3, 1).clamp(0, 1)

        status = (
            f"ch={channel} crop {crop_box} ({cw}×{ch_val}) → {sz}×{sz} "
            f"pad={float(padding_pct)*100:.0f}%% {status_note}"
        )
        print(f"[BD CropToMask] {status}", flush=True)

        return io.NodeOutput(result, crop_box, status)


CROP_TO_MASK_V3_NODES = [BD_CropToMask]
CROP_TO_MASK_NODES = {"BD_CropToMask": BD_CropToMask}
CROP_TO_MASK_DISPLAY_NAMES = {"BD_CropToMask": "BD Crop To Mask"}
