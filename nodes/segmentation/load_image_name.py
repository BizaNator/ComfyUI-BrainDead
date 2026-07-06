"""
BD_LoadImageWithName — LoadImage that also outputs the scene name.

Standard LoadImage only returns IMAGE + MASK; the selected filename is a widget
value that never reaches downstream nodes. This wrapper additionally outputs the
file's basename (no extension) as a STRING so it can be wired into an export
node's `filename` — e.g. so each scene's parts export lands in a folder named
after the source image with zero manual typing.
"""

import os
import hashlib

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths
import node_helpers
import comfy.model_management

from comfy_api.latest import io


def _resolve_path(image: str) -> str:
    """Accept either an input-dir image name OR an absolute filesystem path.

    The Combo widget lists input-dir files, but API callers (run_workflow.py)
    often want to point at a scene PNG anywhere on disk without pre-uploading.
    """
    s = str(image).split(" [", 1)[0]
    if os.path.isabs(s) and os.path.isfile(s):
        return s
    return folder_paths.get_annotated_filepath(image)


def _load(image: str):
    image_path = _resolve_path(image)
    img = node_helpers.pillow(Image.open, image_path)

    output_images, output_masks = [], []
    w = h = None
    dtype = comfy.model_management.intermediate_dtype()

    for i in ImageSequence.Iterator(img):
        i = node_helpers.pillow(ImageOps.exif_transpose, i)
        if i.mode == "I":
            i = i.point(lambda v: v * (1 / 255))
        rgb = i.convert("RGB")
        if len(output_images) == 0:
            w, h = rgb.size
        if rgb.size[0] != w or rgb.size[1] != h:
            continue
        arr = np.array(rgb).astype(np.float32) / 255.0
        t = torch.from_numpy(arr)[None,]
        if "A" in i.getbands():
            m = np.array(i.getchannel("A")).astype(np.float32) / 255.0
            m = 1.0 - torch.from_numpy(m)
        elif i.mode == "P" and "transparency" in i.info:
            m = np.array(i.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0
            m = 1.0 - torch.from_numpy(m)
        else:
            m = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        output_images.append(t.to(dtype=dtype))
        output_masks.append(m.unsqueeze(0).to(dtype=dtype))
        if img.format == "MPO":
            break

    if len(output_images) > 1:
        out_img = torch.cat(output_images, dim=0)
        out_mask = torch.cat(output_masks, dim=0)
    else:
        out_img = output_images[0]
        out_mask = output_masks[0]
    return out_img, out_mask


def _scene_name(image: str) -> str:
    """basename without extension, filename-safe, of the selected image."""
    base = os.path.basename(str(image))
    # annotated paths may look like 'subdir/name.png [input]' — strip the tag
    base = base.split(" [", 1)[0]
    stem = os.path.splitext(base)[0].strip()
    return stem or "scene"


class BD_LoadImageWithName(io.ComfyNode):
    """Load an image and also output its filename (no extension) as a STRING."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir)
                 if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return io.Schema(
            node_id="BD_LoadImageWithName",
            display_name="BD Load Image + Name",
            category="🧠BrainDead/Segmentation",
            description=(
                "Same as Load Image, plus a STRING output with the file's name "
                "(no extension). Wire scene_name into an export node's filename so "
                "each scene's outputs land in a folder named after the source image."
            ),
            inputs=[
                io.Combo.Input("image", options=sorted(files), upload=io.UploadType.image),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.String.Output(display_name="scene_name"),
            ],
        )

    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        out_img, out_mask = _load(image)
        return io.NodeOutput(out_img, out_mask, _scene_name(image))

    @classmethod
    def fingerprint_inputs(cls, image) -> str:
        image_path = _resolve_path(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.hexdigest()

    @classmethod
    def validate_inputs(cls, image) -> bool | str:
        s = str(image).split(" [", 1)[0]
        if os.path.isabs(s):
            return True if os.path.isfile(s) else f"Invalid image file: {image}"
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


LOAD_IMAGE_NAME_V3_NODES = [BD_LoadImageWithName]
LOAD_IMAGE_NAME_NODES = {"BD_LoadImageWithName": BD_LoadImageWithName}
LOAD_IMAGE_NAME_DISPLAY_NAMES = {"BD_LoadImageWithName": "BD Load Image + Name"}
