"""
Pixal3D model manager - lazy-loaded pipeline and MoGe singletons.
"""
import math

import torch

HAS_PIXAL3D = False
HAS_MOGE = False

try:
    from pixal3d.pipelines import Pixal3DImageTo3DPipeline
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    HAS_PIXAL3D = True
except ImportError:
    pass

try:
    from moge.model.v2 import MoGeModel
    HAS_MOGE = True
except ImportError:
    pass


DEFAULT_MODEL_PATH = "TencentARC/Pixal3D"
MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"

# DinoV3 image conditioning configs for each pipeline stage
_IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}

_pipeline: "Pixal3DImageTo3DPipeline | None" = None
_pipeline_model_path: str | None = None
_moge_model = None


def get_pipeline(model_path: str = DEFAULT_MODEL_PATH) -> "Pixal3DImageTo3DPipeline":
    global _pipeline, _pipeline_model_path

    if _pipeline is not None and _pipeline_model_path == model_path:
        return _pipeline

    print(f"[BD Pixal3D] Loading pipeline from {model_path}...")

    # Skip the auto-download of briaai/RMBG-2.0 during from_pretrained — it's a gated
    # repo and HF_TOKEN isn't in the service env. We load it from the local copy instead.
    import pixal3d.pipelines.rembg as _p3d_rembg

    class _NullRembg:
        def __init__(self, **kwargs): pass
        def __call__(self, image): return image

    _orig_birefnet = getattr(_p3d_rembg, "BiRefNet", None)
    _p3d_rembg.BiRefNet = _NullRembg
    try:
        pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)
    finally:
        if _orig_birefnet is not None:
            _p3d_rembg.BiRefNet = _orig_birefnet

    # Load rembg from a local copy under ComfyUI's configured models dir — avoids the
    # gated HF download. Generic (folder_paths), not a hardcoded machine path.
    import os as _os
    try:
        import folder_paths as _fp
        _models_dir = _fp.models_dir
    except Exception:
        _models_dir = "models"
    _RMBG_LOCAL = _os.path.join(_models_dir, "RMBG", "RMBG-2.0")
    print(f"[BD Pixal3D] Loading BiRefNet rembg from {_RMBG_LOCAL}...")
    pipeline.rembg_model = _p3d_rembg.BiRefNet(model_name=_RMBG_LOCAL)

    print("[BD Pixal3D] Building DinoV3 projection models...")
    pipeline.image_cond_model_ss = _build_dino_model(_IMAGE_COND_CONFIGS["ss"])
    pipeline.image_cond_model_shape_512 = _build_dino_model(_IMAGE_COND_CONFIGS["shape_512"])
    pipeline.image_cond_model_shape_1024 = _build_dino_model(_IMAGE_COND_CONFIGS["shape_1024"])
    pipeline.image_cond_model_tex_1024 = _build_dino_model(_IMAGE_COND_CONFIGS["tex_1024"])

    pipeline.low_vram = False
    pipeline.cuda()
    pipeline.image_cond_model_ss.cuda()
    pipeline.image_cond_model_shape_512.cuda()
    pipeline.image_cond_model_shape_1024.cuda()
    pipeline.image_cond_model_tex_1024.cuda()

    print("[BD Pixal3D] Pre-loading NAF upsampler models...")
    for attr in ['image_cond_model_shape_512', 'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
        model = getattr(pipeline, attr, None)
        if model is not None and getattr(model, 'use_naf_upsample', False):
            model._load_naf()

    _pipeline = pipeline
    _pipeline_model_path = model_path
    print(f"[BD Pixal3D] Pipeline ready.")
    return _pipeline


def get_moge_model():
    global _moge_model
    if _moge_model is not None:
        return _moge_model
    print(f"[BD Pixal3D] Loading MoGe-2 for camera estimation ({MOGE_MODEL_NAME})...")
    _moge_model = MoGeModel.from_pretrained(MOGE_MODEL_NAME).cuda()
    _moge_model.eval()
    return _moge_model


def _build_dino_model(config: dict):
    model = DinoV3ProjFeatureExtractor(**config)
    model.eval()
    return model


def compute_camera_distance(
    camera_angle_x: float,
    mesh_scale: float,
    extend_pixel: int,
    image_resolution: int,
) -> float:
    """
    Compute camera distance from FOV angle.
    Matches the distance_from_fov logic in Pixal3D inference.py.
    """
    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    rotation_matrix = torch.tensor([
        [1.0,  0.0,  0.0],
        [0.0,  0.0, -1.0],
        [0.0,  1.0,  0.0],
    ])
    gp = (grid_point.float() @ rotation_matrix.T) / mesh_scale / 2.0
    xw = float(gp[0].item())
    yw = float(gp[1].item())

    target_x = float(0 - extend_pixel)
    f_pixels = (16.0 / math.tan(camera_angle_x / 2.0)) * image_resolution / 32.0
    x_ndc = target_x - image_resolution / 2.0
    distance = f_pixels * xw / x_ndc - yw
    return float(distance)
