"""
BD_Pixal3DPreprocess - Image preprocessing and camera estimation for Pixal3D.
"""
import gc
import math

import numpy as np
import torch
from PIL import Image

from comfy_api.latest import io

from .utils import HAS_MOGE, compute_camera_distance


class BD_Pixal3DPreprocess(io.ComfyNode):
    """
    Preprocess image and estimate camera parameters for Pixal3D image-to-3D.

    Handles mask compositing, square cropping, and camera FOV estimation.
    Connect the output to BD Pixal3D Image to 3D.

    FOV Modes:
    - auto_moge: Use MoGe-2 monocular depth to estimate camera intrinsics (recommended)
    - manual: Specify camera_angle_x in radians (try 0.2 if MoGe gives distorted results)

    The preprocessed image is composed on a black background (Pixal3D default).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_Pixal3DPreprocess",
            display_name="BD Pixal3D Preprocess",
            category="🧠BrainDead/Pixal3D",
            description="""Preprocess image and estimate camera for Pixal3D image-to-3D.

Connect the output PIXAL3D_INPUT to BD Pixal3D Image to 3D.

FOV estimation:
- auto_moge: MoGe-2 estimates camera from depth cues (recommended)
- manual: Set camera_angle_x directly (try 0.2 rad if MoGe distorts)

Provides a preprocessed_image preview showing the 512x512 input Pixal3D will see.""",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip="Object mask. If omitted, uses full image rectangle.",
                ),
                io.Combo.Input(
                    "background",
                    options=["black", "gray"],
                    default="black",
                    tooltip="Compositing background color. Pixal3D was trained with black.",
                ),
                io.Combo.Input(
                    "fov_mode",
                    options=["auto_moge", "manual"],
                    default="auto_moge",
                    tooltip="Camera FOV estimation method",
                ),
                io.Float.Input(
                    "manual_fov",
                    default=0.2,
                    min=0.05,
                    max=2.0,
                    step=0.01,
                    tooltip="Camera FOV in radians. Only used when fov_mode=manual.",
                ),
                io.Float.Input(
                    "mesh_scale",
                    default=1.0,
                    min=0.1,
                    max=5.0,
                    step=0.1,
                    tooltip="Mesh scale factor for camera distance calculation.",
                ),
                io.Int.Input(
                    "extend_pixel",
                    default=0,
                    min=-64,
                    max=64,
                    tooltip="Expand/shrink the FOV pixel range. 0 is standard.",
                ),
            ],
            outputs=[
                io.Custom("PIXAL3D_INPUT").Output(display_name="pixal3d_input"),
                io.Image.Output(display_name="preprocessed_image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor | None = None,
        background: str = "black",
        fov_mode: str = "auto_moge",
        manual_fov: float = 0.2,
        mesh_scale: float = 1.0,
        extend_pixel: int = 0,
    ) -> io.NodeOutput:

        IMAGE_RESOLUTION = 512

        # --- Tensor → PIL ---
        if image.dim() == 4:
            img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            img_np = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np, "RGB")

        # --- Apply mask to produce RGBA ---
        if mask is not None:
            if mask.dim() == 3:
                mask_np = mask[0].cpu().numpy()
            elif mask.dim() == 2:
                mask_np = mask.cpu().numpy()
            else:
                mask_np = mask[0].cpu().numpy()

            if mask_np.shape[:2] != (pil_image.height, pil_image.width):
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
                mask_pil = mask_pil.resize(
                    (pil_image.width, pil_image.height), Image.LANCZOS
                )
                mask_np = np.array(mask_pil) / 255.0

            alpha_np = (mask_np * 255).astype(np.uint8)
            rgba = np.dstack([img_np, alpha_np])
            pil_image = Image.fromarray(rgba, "RGBA")

        # --- Smart crop: tight bbox around non-transparent pixels + 5% margin ---
        pil_image = _crop_to_subject(pil_image, margin_ratio=0.05)

        # --- Composite onto background ---
        bg_color = (0, 0, 0) if background == "black" else (128, 128, 128)
        if pil_image.mode == "RGBA":
            bg = Image.new("RGB", pil_image.size, bg_color)
            bg.paste(pil_image, mask=pil_image.split()[3])
            pil_composited = bg
        else:
            pil_composited = pil_image.convert("RGB")

        # Resize to 512x512
        pil_composited = pil_composited.resize(
            (IMAGE_RESOLUTION, IMAGE_RESOLUTION), Image.LANCZOS
        )

        # --- Camera estimation ---
        if fov_mode == "manual":
            camera_angle_x = float(manual_fov)
            distance = compute_camera_distance(
                camera_angle_x, mesh_scale, extend_pixel, IMAGE_RESOLUTION
            )
            print(
                f"[BD Pixal3D] Manual FOV: {math.degrees(camera_angle_x):.2f}°, "
                f"distance={distance:.4f}"
            )
        else:
            if not HAS_MOGE:
                raise ImportError(
                    "moge not available. "
                    "Install: pip install git+https://github.com/microsoft/MoGe.git"
                )
            from .utils import get_moge_model

            moge_model = get_moge_model()

            img_f32 = np.array(pil_composited).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_f32).permute(2, 0, 1).cuda()

            with torch.no_grad():
                output = moge_model.infer(img_tensor)

            intrinsics = output["intrinsics"].squeeze().cpu().numpy()
            fx_normalized = intrinsics[0, 0]
            fx = fx_normalized * IMAGE_RESOLUTION
            camera_angle_x = 2.0 * math.atan(IMAGE_RESOLUTION / (2.0 * fx))
            distance = compute_camera_distance(
                camera_angle_x, mesh_scale, extend_pixel, IMAGE_RESOLUTION
            )
            print(
                f"[BD Pixal3D] MoGe FOV: {math.degrees(camera_angle_x):.2f}°, "
                f"distance={distance:.4f}"
            )

        camera_params = {
            "camera_angle_x": camera_angle_x,
            "distance": distance,
            "mesh_scale": mesh_scale,
        }

        pixal3d_input = {
            "image": pil_composited,
            "camera_params": camera_params,
        }

        # Preview tensor
        preview_np = np.array(pil_composited).astype(np.float32) / 255.0
        preview_tensor = torch.from_numpy(preview_np).unsqueeze(0)

        gc.collect()
        torch.cuda.empty_cache()

        return io.NodeOutput(pixal3d_input, preview_tensor)


def _crop_to_subject(image: Image.Image, margin_ratio: float = 0.05) -> Image.Image:
    """Crop to non-transparent bounding box with margin, keep square aspect."""
    if image.mode != "RGBA":
        return image

    alpha = np.array(image)[:, :, 3]
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)

    if not (rows.any() and cols.any()):
        return image

    rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

    h_margin = int((rmax - rmin) * margin_ratio)
    w_margin = int((cmax - cmin) * margin_ratio)
    rmin = max(0, rmin - h_margin)
    rmax = min(image.height, rmax + h_margin + 1)
    cmin = max(0, cmin - w_margin)
    cmax = min(image.width, cmax + w_margin + 1)

    # Expand to square from center
    side = max(rmax - rmin, cmax - cmin)
    r_center = (rmin + rmax) // 2
    c_center = (cmin + cmax) // 2
    rmin = max(0, r_center - side // 2)
    rmax = min(image.height, rmin + side)
    cmin = max(0, c_center - side // 2)
    cmax = min(image.width, cmin + side)

    return image.crop((cmin, rmin, cmax, rmax))


# V3 node list
PIXAL3D_PREPROCESS_V3_NODES = [BD_Pixal3DPreprocess]

PIXAL3D_PREPROCESS_NODES = {
    "BD_Pixal3DPreprocess": BD_Pixal3DPreprocess,
}

PIXAL3D_PREPROCESS_DISPLAY_NAMES = {
    "BD_Pixal3DPreprocess": "BD Pixal3D Preprocess",
}
