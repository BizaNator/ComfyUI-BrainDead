"""
BD_RigPreview — render a rigged FBX with visible bone skeleton as a ComfyUI IMAGE.

Uses headless Blender (EEVEE) to create:
  - Orange spheres at every joint
  - Blue-white cylinders along every bone
  - Semi-transparent blue character mesh (configurable opacity)

Output modes:
  all_views   — 2×2 grid: front / side / back / perspective (default)
  front       — orthographic front only
  side        — orthographic right-side only
  back        — orthographic back only
  perspective — 3/4 perspective view only
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from comfy_api.latest import io

from ..blender.base import BlenderNodeMixin

_PACK_ROOT      = Path(__file__).resolve().parent.parent.parent
_BLENDER_SCRIPT = _PACK_ROOT / "lib" / "blender" / "rig_preview.py"

_ALL_VIEWS = ["front", "side", "back", "perspective"]


class BD_RigPreview(io.ComfyNode, BlenderNodeMixin):
    """Visualize a rigged FBX skeleton as a ComfyUI image.

    Renders the bone structure (joints + bones as geometry) with an optional
    semi-transparent character mesh overlay using headless Blender EEVEE.
    Accepts output from BD_AutoRigMIA, BD_MixamoToUEFN, or BD_AutoRigUEFN.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_RigPreview",
            display_name="BD Rig Preview",
            category="🧠BrainDead/AutoRig",
            description=(
                "Render a rigged FBX as a bone-visualization image. "
                "Joints appear as orange spheres, bones as cylinders, "
                "and the character mesh as a semi-transparent overlay. "
                "Use after BD_AutoRigMIA or BD_AutoRigUEFN to inspect the skeleton."
            ),
            inputs=[
                io.String.Input(
                    "fbx_path",
                    tooltip="Rigged FBX — output of BD_AutoRigMIA, BD_MixamoToUEFN, or BD_AutoRigUEFN.",
                ),
                io.Combo.Input(
                    "view_mode",
                    options=["all_views", "perspective", "front", "side", "back"],
                    default="all_views",
                    tooltip=(
                        "all_views: 2×2 grid with front/side/back/perspective. "
                        "Single-view options render one camera at full resolution."
                    ),
                ),
                io.Float.Input(
                    "mesh_opacity",
                    default=0.25,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="0 = bones only; 1 = fully opaque mesh. 0.2–0.35 recommended.",
                ),
                io.Int.Input(
                    "resolution",
                    default=768,
                    min=256,
                    max=2048,
                    step=64,
                    tooltip=(
                        "Per-view resolution. all_views produces a 2× grid "
                        "(e.g. 768 → 1536×1536 output)."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(display_name="preview"),
            ],
        )

    @classmethod
    def execute(
        cls,
        fbx_path: str,
        view_mode: str = "all_views",
        mesh_opacity: float = 0.25,
        resolution: int = 768,
    ) -> io.NodeOutput:
        fbx_path = str(Path(fbx_path).resolve())
        if not os.path.exists(fbx_path):
            raise FileNotFoundError(f"BD_RigPreview: fbx_path not found: {fbx_path}")

        if not _BLENDER_SCRIPT.exists():
            raise FileNotFoundError(f"BD_RigPreview: script missing: {_BLENDER_SCRIPT}")

        ok, err = cls._check_blender()
        if not ok:
            raise RuntimeError(f"BD_RigPreview: Blender unavailable — {err}")

        script = _BLENDER_SCRIPT.read_text(encoding="utf-8")
        tmp = tempfile.mkdtemp(prefix="bd_rigpreview_")
        try:
            output_base = os.path.join(tmp, "rig.png")

            ok, msg, lines = cls._run_blender_script(
                script=script,
                input_path=fbx_path,
                output_path=output_base,
                extra_args={
                    "VIEW_MODE":    view_mode,
                    "MESH_OPACITY": str(mesh_opacity),
                    "RESOLUTION":   str(resolution),
                },
                timeout=300,
            )
            if not ok:
                tail = "\n".join(lines[-25:]) if lines else msg
                raise RuntimeError(f"BD_RigPreview Blender failed:\n{tail}")

            pil_img = _load_result(tmp, output_base, view_mode)
            arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0)   # [1, H, W, 3]
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

        return io.NodeOutput(tensor)


def _load_result(tmp_dir: str, output_base: str, view_mode: str) -> Image.Image:
    """Load rendered images and composite into final PIL image."""
    stem = output_base[:-4]  # strip .png

    if view_mode == "all_views":
        imgs = []
        for vname in _ALL_VIEWS:
            p = f"{stem}_{vname}.png"
            if os.path.exists(p):
                imgs.append(Image.open(p).convert("RGBA"))

        if not imgs:
            raise RuntimeError("BD_RigPreview: no rendered view images found in temp dir")

        W, H   = imgs[0].size
        ncols  = 2
        nrows  = (len(imgs) + 1) // 2
        bg     = (15, 15, 20, 255)
        grid   = Image.new("RGBA", (W * ncols, H * nrows), bg)
        for i, img in enumerate(imgs):
            grid.paste(img, ((i % ncols) * W, (i // ncols) * H), img)
        return grid.convert("RGB")

    # Single view
    if not os.path.exists(output_base):
        raise RuntimeError(f"BD_RigPreview: rendered image not found: {output_base}")
    return Image.open(output_base).convert("RGB")


RIG_PREVIEW_V3_NODES      = [BD_RigPreview]
RIG_PREVIEW_NODES         = {"BD_RigPreview": BD_RigPreview}
RIG_PREVIEW_DISPLAY_NAMES = {"BD_RigPreview": "BD Rig Preview"}
