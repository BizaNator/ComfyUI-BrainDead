"""
BD_Lotus2ModelLoader + BD_Lotus2Predict — depth/normal estimation via Lotus-2.

Lotus-2 is FLUX.1-dev + two LoRAs (core_predictor + detail_sharpener) + a tiny
2-conv residual module (LCM = Local Continuity Module). The pipeline runs:
  VAE encode → empty prompt encode → 1-step transformer w/ core LoRA →
  LCM module → N-step denoise w/ sharpener LoRA → VAE decode.

Upstream: https://github.com/EnVision-Research/Lotus-2 (Apache-2.0)
Pipeline + LCM module + LoRA loader code in this file is adapted from upstream.

Weights layout (default — auto-downloaded if missing):
  /srv/AI_Stuff/models/lotus2/{depth,normal}/
    lotus-2_core_predictor_{task}.safetensors
    lotus-2_detail_sharpener_{task}.safetensors
    lotus-2_lcm_{task}.safetensors

FLUX.1-dev base loaded from HF cache (HF_HOME=/srv/AI_Stuff/models/huggingface).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn

from comfy_api.latest import io


LOTUS2_MODEL = "LOTUS2_MODEL"

DEFAULT_LOTUS2_REPO = "jingheya/Lotus-2"
DEFAULT_FLUX_REPO = "black-forest-labs/FLUX.1-dev"
DEFAULT_LOTUS2_ROOT = "/srv/AI_Stuff/models/lotus2"

CORE_PREDICTOR_FILENAME = {
    "depth":  "lotus-2_core_predictor_depth.safetensors",
    "normal": "lotus-2_core_predictor_normal.safetensors",
}
LCM_FILENAME = {
    "depth":  "lotus-2_lcm_depth.safetensors",
    "normal": "lotus-2_lcm_normal.safetensors",
}
DETAIL_SHARPENER_FILENAME = {
    "depth":  "lotus-2_detail_sharpener_depth.safetensors",
    "normal": "lotus-2_detail_sharpener_normal.safetensors",
}

# LoRA target modules — matches upstream infer.py::load_lora_and_lcm_weights
_LORA_TARGET_MODULES = [
    "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "ff.net.0.proj", "ff.net.2",
    "ff_context.net.0.proj", "ff_context.net.2",
]


# ============================================================================
# Local Continuity Module (LCM) — bridges core_predictor and detail_sharpener
# Adapted from upstream infer.py:118-130
# ============================================================================

class Local_Continuity_Module(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.lcm = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(num_channels * 2, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lcm_dtype = next(self.lcm.parameters()).dtype
        if x.dtype != lcm_dtype:
            x = x.to(dtype=lcm_dtype)
        return x + self.lcm(x)


# ============================================================================
# Resize helpers — adapted from upstream utils/image_utils.py
# ============================================================================

def _resize_to_multiple_of_16(image_tensor: torch.Tensor) -> torch.Tensor:
    h, w = image_tensor.shape[2], image_tensor.shape[3]
    min_side = min(h, w)
    scale = (min_side // 16) * 16 / min_side
    new_h = (int(h * scale) // 16) * 16
    new_w = (int(w * scale) // 16) * 16
    return torch.nn.functional.interpolate(
        image_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False,
    )


def _resize_first(image_tensor: torch.Tensor, process_res: Optional[int]) -> torch.Tensor:
    if process_res:
        max_edge = max(image_tensor.shape[2], image_tensor.shape[3])
        if max_edge > process_res:
            scale = process_res / max_edge
            new_h = int(image_tensor.shape[2] * scale)
            new_w = int(image_tensor.shape[3] * scale)
            image_tensor = torch.nn.functional.interpolate(
                image_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False,
            )
    return _resize_to_multiple_of_16(image_tensor)


def _resize_to(image_tensor: torch.Tensor, target_hw: tuple) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        image_tensor, size=target_hw, mode="bilinear", align_corners=False,
    )


# ============================================================================
# Lotus-2 pipeline — adapted from upstream pipeline.py
# ============================================================================

def _build_lotus2_pipeline_class():
    """Lazy import diffusers + define Lotus2Pipeline subclass.
    Wrapped in a function so import errors only surface when the loader runs.
    """
    from diffusers import FluxPipeline
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

    class Lotus2Pipeline(FluxPipeline):
        @torch.no_grad()
        def __call__(
            self,
            rgb_in: torch.FloatTensor,
            prompt: Union[str, List[str]] = "",
            num_inference_steps: int = 10,
            output_type: str = "pt",
            process_res: Optional[int] = None,
            timestep_core_predictor: int = 1,
            guidance_scale: float = 3.5,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        ):
            batch_size = rgb_in.shape[0]
            input_size = rgb_in.shape[2:]
            rgb_in = _resize_first(rgb_in, process_res)
            height, width = rgb_in.shape[2:]

            self._guidance_scale = guidance_scale
            self._joint_attention_kwargs = joint_attention_kwargs
            self._interrupt = False
            device = self._execution_device

            prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
                prompt=prompt, prompt_2=None, device=device,
            )

            rgb_in = rgb_in.to(device=device, dtype=self.dtype)
            rgb_latents = self.vae.encode(rgb_in).latent_dist.sample()
            rgb_latents = (rgb_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

            packed_rgb_latents = self._pack_latents(
                rgb_latents,
                batch_size=rgb_latents.shape[0],
                num_channels_latents=rgb_latents.shape[1],
                height=rgb_latents.shape[2],
                width=rgb_latents.shape[3],
            )

            latent_image_ids_core = self._prepare_latent_image_ids(
                batch_size, rgb_latents.shape[2] // 2, rgb_latents.shape[3] // 2,
                device, rgb_latents.dtype,
            )
            latent_image_ids = self._prepare_latent_image_ids(
                batch_size, rgb_latents.shape[2] // 2, rgb_latents.shape[3] // 2,
                device, rgb_latents.dtype,
            )

            timestep_core = torch.tensor(timestep_core_predictor).expand(batch_size).to(
                device=rgb_in.device, dtype=rgb_in.dtype,
            )

            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            image_seq_len = packed_rgb_latents.shape[1]
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.base_image_seq_len,
                self.scheduler.config.max_image_seq_len,
                self.scheduler.config.base_shift,
                self.scheduler.config.max_shift,
            )
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
            )
            self._num_timesteps = len(timesteps)

            if self.transformer.config.guidance_embeds:
                guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                guidance = guidance.expand(packed_rgb_latents.shape[0])
            else:
                guidance = None

            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {}

            # Stage 1: core predictor (1 step)
            self.transformer.set_adapter("core_predictor")
            latents = self.transformer(
                hidden_states=packed_rgb_latents,
                timestep=timestep_core / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids_core,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = self.local_continuity_module(latents)

            # Stage 2: detail sharpener (N-step denoise)
            self.transformer.set_adapter("detail_sharpener")
            latents = self._pack_latents(
                latents,
                batch_size=latents.shape[0],
                num_channels_latents=latents.shape[1],
                height=latents.shape[2],
                width=latents.shape[3],
            )

            for i, t in enumerate(timesteps):
                if getattr(self, "_interrupt", False):
                    continue
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            latents = latents.to(dtype=self.dtype)
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            # image is (B, 3, H, W) in [-1, 1] roughly

            # Resize back to input size
            image = _resize_to(image, input_size)
            return image

    return Lotus2Pipeline


# ============================================================================
# Weight resolution + downloading
# ============================================================================

def _resolve_lotus2_weight(task: str, kind: str, override_path: str) -> str:
    """Returns local path to the requested Lotus-2 weight. Auto-downloads if missing.
    kind ∈ {core, lcm, sharpener}.
    """
    fname = {
        "core":      CORE_PREDICTOR_FILENAME[task],
        "lcm":       LCM_FILENAME[task],
        "sharpener": DETAIL_SHARPENER_FILENAME[task],
    }[kind]

    if override_path and override_path.strip():
        p = Path(os.path.expanduser(os.path.expandvars(override_path.strip())))
        if p.is_dir():
            p = p / fname
        if not p.exists():
            raise FileNotFoundError(f"Lotus-2 {kind} weight override not found: {p}")
        return str(p)

    target_dir = Path(DEFAULT_LOTUS2_ROOT) / task
    target = target_dir / fname
    if target.exists():
        return str(target)

    # Auto-download
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[BD Lotus2] Downloading {fname} to {target_dir}", flush=True)
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=DEFAULT_LOTUS2_REPO,
        filename=fname,
        local_dir=str(target_dir),
    )
    return path


def _load_lora_and_lcm(transformer, task: str, core_path: str, lcm_path: str,
                       sharpener_path: str):
    """Load core_predictor + detail_sharpener LoRAs as named adapters,
    plus the LCM bridge module. Adapted from upstream infer.py."""
    from peft import LoraConfig, set_peft_model_state_dict
    from diffusers.utils import convert_unet_state_dict_to_peft

    Lotus2Pipeline = _build_lotus2_pipeline_class()
    lora_rank = 128 if task == "depth" else 256
    device = transformer.device
    weight_dtype = transformer.dtype

    # Stage 1: core predictor LoRA
    core_cfg = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=_LORA_TARGET_MODULES,
    )
    transformer.add_adapter(core_cfg, adapter_name="core_predictor")
    core_sd = Lotus2Pipeline.lora_state_dict(core_path)
    core_xfm_sd = {
        k.replace("transformer.", ""): v
        for k, v in core_sd.items() if k.startswith("transformer.")
    }
    core_xfm_sd = convert_unet_state_dict_to_peft(core_xfm_sd)
    set_peft_model_state_dict(transformer, core_xfm_sd, adapter_name="core_predictor")
    for name, param in transformer.named_parameters():
        if "core_predictor" in name:
            param.requires_grad = False

    # LCM bridge
    lcm = Local_Continuity_Module(transformer.config.in_channels // 4)
    lcm_sd = torch.load(lcm_path, map_location="cpu", weights_only=True)
    lcm.load_state_dict(lcm_sd)
    lcm.requires_grad_(False)
    lcm.to(device=device, dtype=weight_dtype)

    # Stage 2: detail sharpener LoRA
    sharp_cfg = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=_LORA_TARGET_MODULES,
    )
    transformer.add_adapter(sharp_cfg, adapter_name="detail_sharpener")
    sharp_sd = Lotus2Pipeline.lora_state_dict(sharpener_path)
    sharp_xfm_sd = {
        k.replace("transformer.", ""): v
        for k, v in sharp_sd.items() if k.startswith("transformer.")
    }
    sharp_xfm_sd = convert_unet_state_dict_to_peft(sharp_xfm_sd)
    set_peft_model_state_dict(transformer, sharp_xfm_sd, adapter_name="detail_sharpener")
    for name, param in transformer.named_parameters():
        if "detail_sharpener" in name:
            param.requires_grad = False

    return transformer, lcm


# ============================================================================
# Module-level cache so reloading the same model doesn't re-init FLUX
# ============================================================================
_PIPELINE_CACHE: Dict[str, Any] = {}


def _cache_key(task, flux_repo, dtype, device, cpu_offload):
    return f"{task}|{flux_repo}|{dtype}|{device}|offload={cpu_offload}"


def _resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(name)


# ============================================================================
# Nodes
# ============================================================================

class BD_Lotus2ModelLoader(io.ComfyNode):
    """Load FLUX.1-dev base + Lotus-2 task-specific LoRAs + LCM bridge."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_Lotus2ModelLoader",
            display_name="BD Lotus-2 Model Loader",
            category="🧠BrainDead/Depth",
            description=(
                "Load Lotus-2 (FLUX.1-dev + LoRAs + LCM bridge) for depth or normal "
                "estimation. Diffusion-based, SOTA-quality monocular geometry prediction. "
                "FLUX.1-dev is loaded from HF cache (~24 GB). Lotus-2 weights "
                "(~1.5 GB depth or ~2.9 GB normal core+sharpener) auto-download to "
                "/srv/AI_Stuff/models/lotus2/{task}/ if missing.\n\n"
                "Reuse the loaded model across multiple Predict calls — first load is slow, "
                "subsequent same-task calls hit the in-memory cache."
            ),
            inputs=[
                io.Combo.Input("task", options=["depth", "normal"], default="depth",
                               tooltip="Which Lotus-2 LoRA pair to load. depth = ~1.5 GB, "
                                       "normal = ~2.9 GB (different LoRA rank)."),
                io.Combo.Input("dtype", options=["bf16", "fp16", "fp32"], default="bf16",
                               tooltip="Inference precision. bf16 recommended on Blackwell/H100."),
                io.Combo.Input("device", options=["auto", "cuda", "cpu"], default="auto"),
                io.String.Input(
                    "flux_repo_id", default=DEFAULT_FLUX_REPO, optional=True,
                    tooltip="HF repo id for the FLUX.1-dev base. Loaded via HF cache "
                            "(HF_HOME env). Override only if you have a fork.",
                ),
                io.String.Input(
                    "lotus2_weights_dir", default="", optional=True,
                    tooltip="Override directory containing the three Lotus-2 weight files. "
                            f"Default: {DEFAULT_LOTUS2_ROOT}/{{task}}/",
                ),
                io.Boolean.Input(
                    "cpu_offload", default=False, optional=True,
                    tooltip="Enable diffusers model CPU offload. Each pipeline module "
                            "(transformer / VAE / text encoders) lives on CPU and moves to "
                            "GPU only when needed. Peak VRAM drops from ~35 GB to ~27 GB "
                            "(largest single module + activations) at the cost of ~1.5x "
                            "slower inference per call. Turn on when running Lotus-2 "
                            "alongside other large models (Qwen-Image-Edit, FLUX gen, etc).",
                ),
            ],
            outputs=[
                io.Custom(LOTUS2_MODEL).Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, task="depth", dtype="bf16", device="auto",
                flux_repo_id=DEFAULT_FLUX_REPO, lotus2_weights_dir="",
                cpu_offload=False) -> io.NodeOutput:
        weight_dtype = _resolve_dtype(dtype)
        torch_device = _resolve_device(device)

        key = _cache_key(task, flux_repo_id, dtype, str(torch_device), cpu_offload)
        if key in _PIPELINE_CACHE:
            print(f"[BD Lotus2] Cache hit: {key}", flush=True)
            return io.NodeOutput(_PIPELINE_CACHE[key])

        from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
        Lotus2Pipeline = _build_lotus2_pipeline_class()

        print(f"[BD Lotus2] Loading FLUX.1-dev base from {flux_repo_id} (dtype={dtype})", flush=True)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            flux_repo_id, subfolder="scheduler", num_train_timesteps=10,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            flux_repo_id, subfolder="transformer", torch_dtype=weight_dtype,
        )
        transformer.requires_grad_(False)
        transformer.to(device=torch_device, dtype=weight_dtype)

        core_p = _resolve_lotus2_weight(task, "core", lotus2_weights_dir)
        lcm_p = _resolve_lotus2_weight(task, "lcm", lotus2_weights_dir)
        sharp_p = _resolve_lotus2_weight(task, "sharpener", lotus2_weights_dir)
        print(f"[BD Lotus2] Loading {task} LoRAs + LCM:\n"
              f"  core:      {core_p}\n"
              f"  lcm:       {lcm_p}\n"
              f"  sharpener: {sharp_p}", flush=True)
        transformer, lcm = _load_lora_and_lcm(transformer, task, core_p, lcm_p, sharp_p)

        pipeline = Lotus2Pipeline.from_pretrained(
            flux_repo_id,
            scheduler=scheduler,
            transformer=transformer,
            torch_dtype=weight_dtype,
        )
        pipeline.local_continuity_module = lcm
        if cpu_offload:
            print(f"[BD Lotus2] Enabling model CPU offload (target device: {torch_device})", flush=True)
            pipeline.enable_model_cpu_offload(device=torch_device)
        else:
            pipeline = pipeline.to(torch_device)
        pipeline.set_progress_bar_config(disable=True)

        model = {
            "pipeline": pipeline,
            "lcm": lcm,
            "task": task,
            "dtype": weight_dtype,
            "device": torch_device,
            "cache_key": key,
        }
        _PIPELINE_CACHE[key] = model
        print(f"[BD Lotus2] Loaded. Cache key: {key}", flush=True)
        return io.NodeOutput(model)


def _to_pipeline_input(image: torch.Tensor) -> torch.Tensor:
    """ComfyUI IMAGE (B,H,W,C in [0,1]) → pipeline input (B,3,H,W in [-1,1])."""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    img = image[..., :3].permute(0, 3, 1, 2).contiguous().float()
    return img * 2.0 - 1.0


def _from_pipeline_output(out: torch.Tensor) -> torch.Tensor:
    """Pipeline output (B,3,H,W in ~[-1,1]) → ComfyUI IMAGE (B,H,W,C in [0,1])."""
    out = (out / 2.0 + 0.5).clamp(0, 1)
    return out.permute(0, 2, 3, 1).contiguous().float().cpu()


def _colorize_depth(depth_01: torch.Tensor) -> torch.Tensor:
    """depth_01: (B,H,W,3) — already 3-channel grayscale-ish from VAE decode.
    Apply matplotlib Spectral colormap (reversed: near=red, far=blue)."""
    try:
        import matplotlib
        cm = matplotlib.colormaps["Spectral"]
        # average channels to scalar depth, normalize per-image
        mono = depth_01.mean(dim=-1)  # (B, H, W)
        out = []
        for b in range(mono.shape[0]):
            d = mono[b].numpy()
            d = (d - d.min()) / max(d.max() - d.min(), 1e-6)
            colored = cm(1.0 - d, bytes=False)[..., :3]  # reverse for near=warm
            out.append(torch.from_numpy(colored).float())
        return torch.stack(out, dim=0)
    except Exception:
        return depth_01


class BD_Lotus2Predict(io.ComfyNode):
    """Run Lotus-2 inference on an image. Outputs map (RGB), raw [0,1], and colorized preview."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_Lotus2Predict",
            display_name="BD Lotus-2 Predict",
            category="🧠BrainDead/Depth",
            description=(
                "Run a loaded Lotus-2 model on an image. For depth: outputs a 3-channel "
                "depth map (the model produces RGB-encoded depth — channels are nearly "
                "identical), a [0,1] normalized linear depth, and a colorized preview. "
                "For normal: outputs the normal map directly (R=X, G=Y, B=Z, mapped to [0,1]).\n\n"
                "process_res caps the longest input edge before inference (1024 is the "
                "model's training resolution — going higher costs VRAM with marginal quality)."
            ),
            inputs=[
                io.Custom(LOTUS2_MODEL).Input("model"),
                io.Image.Input("image"),
                io.Int.Input("num_inference_steps", default=10, min=1, max=50,
                             tooltip="Detail sharpener denoising steps. Default 10 from upstream. "
                                     "Quality plateaus around 10-15 for most images."),
                io.Int.Input("process_res", default=1024, min=512, max=2048, step=64,
                             tooltip="Cap longest input edge at this resolution. Output is "
                                     "resized back to input size after inference."),
                io.Float.Input("guidance_scale", default=3.5, min=0.0, max=10.0, step=0.1),
                io.Int.Input("timestep_core_predictor", default=1, min=0, max=1000,
                             tooltip="Core predictor stage's effective timestep (default 1)."),
            ],
            outputs=[
                io.Image.Output(display_name="map"),
                io.Image.Output(display_name="raw_linear"),
                io.Image.Output(display_name="colorized_preview"),
            ],
        )

    @classmethod
    def execute(cls, model, image, num_inference_steps=10, process_res=1024,
                guidance_scale=3.5, timestep_core_predictor=1) -> io.NodeOutput:
        if not isinstance(model, dict) or "pipeline" not in model:
            raise ValueError("BD_Lotus2Predict: 'model' must come from BD_Lotus2ModelLoader.")
        pipeline = model["pipeline"]
        task = model["task"]

        rgb_in = _to_pipeline_input(image).to(model["device"])
        with torch.inference_mode():
            out = pipeline(
                rgb_in=rgb_in,
                prompt="",
                num_inference_steps=int(num_inference_steps),
                output_type="pt",
                process_res=int(process_res),
                timestep_core_predictor=int(timestep_core_predictor),
                guidance_scale=float(guidance_scale),
            )
        map_01 = _from_pipeline_output(out)  # (B, H, W, 3)

        if task == "depth":
            mono = map_01.mean(dim=-1, keepdim=True)
            mn = mono.amin(dim=(1, 2, 3), keepdim=True)
            mx = mono.amax(dim=(1, 2, 3), keepdim=True)
            raw = (mono - mn) / (mx - mn).clamp(min=1e-6)
            raw_rgb = raw.repeat(1, 1, 1, 3)
            preview = _colorize_depth(map_01)
        else:
            raw_rgb = map_01
            preview = map_01

        print(f"[BD Lotus2 Predict] task={task} steps={num_inference_steps} "
              f"process_res={process_res} → out {tuple(map_01.shape)}", flush=True)
        return io.NodeOutput(map_01, raw_rgb, preview)


LOTUS2_V3_NODES = [BD_Lotus2ModelLoader, BD_Lotus2Predict]
LOTUS2_NODES = {
    "BD_Lotus2ModelLoader": BD_Lotus2ModelLoader,
    "BD_Lotus2Predict": BD_Lotus2Predict,
}
LOTUS2_DISPLAY_NAMES = {
    "BD_Lotus2ModelLoader": "BD Lotus-2 Model Loader",
    "BD_Lotus2Predict": "BD Lotus-2 Predict",
}
