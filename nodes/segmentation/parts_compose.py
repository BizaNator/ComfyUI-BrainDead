"""
BD_PartsCompose — flatten a PARTS_BUNDLE back into a single RGBA image.

Parts are placed at their xyxy positions on a canvas sized from frame_size
(or scaled to output_size if specified). Painting order is back-to-front by
depth_median (larger = farther). Standard alpha compositing.
"""

import numpy as np
import torch
from PIL import Image

from comfy_api.latest import io

from .parts_types import PARTS_BUNDLE, ensure_bundle, frame_size as _frame_size


class BD_PartsCompose(io.ComfyNode):
    """Composite all parts in a PARTS_BUNDLE into one flat RGBA image."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_PartsCompose",
            display_name="BD Parts Compose",
            category="🧠BrainDead/Segmentation",
            description=(
                "Assemble all parts in a PARTS_BUNDLE back into a flat image at output_size. "
                "Each part is placed at its scaled xyxy position. Back-to-front order by "
                "depth_median (larger = farther; closer parts paint on top). Standard alpha "
                "compositing — every part's RGBA mask drives the blend."
            ),
            inputs=[
                io.Custom(PARTS_BUNDLE).Input("parts"),
                io.Int.Input("output_size", default=0, min=0, max=8192, step=64, optional=True,
                             tooltip="Max side length of output canvas. 0 = use frame_size as-is. "
                                     "Set to e.g. 3072 if you upscaled per-part 3x."),
                io.Boolean.Input("trigger", default=True, optional=True,
                                 tooltip="If false, returns blank canvas — useful when piping is_last "
                                         "from an iterator to delay composite until the final pass."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="alpha"),
            ],
        )

    @classmethod
    def execute(cls, parts, output_size=0, trigger=True) -> io.NodeOutput:
        ensure_bundle(parts, source="BD_PartsCompose.parts")
        frame_h, frame_w = _frame_size(parts)

        if not trigger:
            h, w = (frame_h or 64), (frame_w or 64)
            return io.NodeOutput(torch.zeros((1, h, w, 3)), torch.zeros((1, h, w)))

        if frame_h == 0 or frame_w == 0:
            raise ValueError("BD_PartsCompose: parts['frame_size'] missing — needed for canvas dims.")

        if output_size <= 0:
            out_h, out_w, scale = frame_h, frame_w, 1.0
        else:
            scale = output_size / max(frame_h, frame_w)
            out_h = int(round(frame_h * scale))
            out_w = int(round(frame_w * scale))

        canvas = np.zeros((out_h, out_w, 4), dtype=np.float32)

        # Back-to-front: larger depth_median first (farther → painted under closer parts).
        ordered = sorted(
            parts["tag2pinfo"].items(),
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
            canvas[y1:y2, x1:x2, :3] = (src_rgb * src_a + dst_rgb * dst_a * (1.0 - src_a)) / safe_a
            canvas[y1:y2, x1:x2, 3:4] = out_a

        image_t = torch.from_numpy(canvas[..., :3]).unsqueeze(0)
        alpha_t = torch.from_numpy(canvas[..., 3]).unsqueeze(0)
        return io.NodeOutput(image_t, alpha_t)


PARTS_COMPOSE_V3_NODES = [BD_PartsCompose]
PARTS_COMPOSE_NODES = {"BD_PartsCompose": BD_PartsCompose}
PARTS_COMPOSE_DISPLAY_NAMES = {"BD_PartsCompose": "BD Parts Compose"}
