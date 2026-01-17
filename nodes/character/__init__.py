"""
V3 API BrainDead Character Nodes for ComfyUI
Created by BizaNator for BrainDeadGuild.com
A Biloxi Studios Inc. Production

Advanced character consistency tools for Qwen-Image models.
Customizable system prompts for maintaining character identity across image edits.
"""

import math
import torch

import node_helpers
import comfy.utils
from comfy_api.latest import io

from .qwen_tokenizer import QwenImageTokenizer


# Default templates as module constants
DEFAULT_CHARACTER_TEMPLATE = """<|im_start|>system
You are a Prompt optimizer specialized in character consistency for image editing. Your primary goal is to preserve character identity while implementing requested changes.

Character Consistency Priority (CRITICAL):
1. FACIAL FEATURES (HIGHEST PRIORITY): Preserve exact facial structure, face shape, jawline, cheekbones, nose shape, lip shape, eye shape and spacing
2. HAIR: Maintain hair texture, hairstyle, hair color, hair length, and any hair accessories or decorations
3. EYES: Keep exact eye color, eye shape, eyebrow shape and color, eyelash style
4. SKIN: Preserve skin tone, skin texture, any facial markings, freckles, moles, or scars
5. DISTINCTIVE FEATURES: Maintain tattoos, piercings, birthmarks, facial hair style, or unique characteristics
6. CLOTHING/STYLE: Adapt clothing and accessories as requested while keeping character recognizable

Task Requirements:
1. When modifying the image, ALWAYS explicitly describe which facial and character features must remain unchanged
2. For brief inputs, add details that enhance the scene while strictly preserving all character-identifying features
3. If text rendering is required, enclose in quotes with position specification
4. Prioritize character recognition over scene/background changes
5. Limit response to 200 words, focusing on character preservation

Process: First identify all distinctive character features from the input image, then explain how the requested changes will be applied while maintaining these exact features. Add "Ultra HD, 4K, cinematic composition" for quality enhancement.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>
<|im_start|>assistant
"""

DEFAULT_T2I_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)

DEFAULT_MULTI_TEMPLATE = """<|im_start|>system
You are analyzing multiple reference images for character-consistent image editing.

MULTI-IMAGE ANALYSIS PROTOCOL:
Picture 1: <|vision_start|><|image_pad|><|vision_end|> - Primary reference for character identity
Picture 2: <|vision_start|><|image_pad|><|vision_end|> - Secondary reference for style/pose
Picture 3: <|vision_start|><|image_pad|><|vision_end|> - Additional context reference

CHARACTER CONSISTENCY (FROM PICTURE 1):
- Face structure, facial features, eye color/shape
- Hair style, color, texture
- Skin tone and distinctive marks
- Body proportions and build

STYLE TRANSFER (FROM PICTURE 2/3):
- Pose and composition
- Clothing style (if requested)
- Environment and lighting
- Art style and rendering

TASK: Combine elements while maintaining character identity from Picture 1.
User request: {}

Generate image maintaining character consistency, Ultra HD, 4K quality.<|im_end|>
<|im_start|>assistant
"""

DEFAULT_IDENTITY_TEMPLATE = """<|im_start|>system
CRITICAL IDENTITY PRESERVATION PROTOCOL:

IMAGE 1 - PRIMARY IDENTITY SOURCE (MUST PRESERVE):
Picture 1: <|vision_start|><|image_pad|><|vision_end|>
This contains the CHARACTER IDENTITY that must be EXACTLY preserved:
- PRECISE facial structure, bone structure, face shape
- EXACT eye color, eye shape, eye spacing, eyebrow shape
- IDENTICAL nose shape, nostril shape, nose bridge
- EXACT mouth shape, lip fullness, lip color
- PRECISE jawline, chin shape, cheekbone structure
- IDENTICAL skin tone, skin texture, any marks/moles/freckles
- EXACT hair color, hair texture, hairline, hair style

IMAGE 2 - SECONDARY REFERENCE (APPLY SELECTIVELY):
Picture 2: <|vision_start|><|image_pad|><|vision_end|>
Apply ONLY non-identity aspects: pose, clothing, style, environment

IMAGE 3 - ADDITIONAL CONTEXT (MINIMAL INFLUENCE):
Picture 3: <|vision_start|><|image_pad|><|vision_end|>
Use for composition/background only, NO identity influence

GENERATION RULES:
1. Picture 1's face MUST appear UNCHANGED in output
2. Preserve Picture 1's exact identity regardless of other changes
3. Background should be clean, artifact-free, no noise
4. Maintain clear subject-background separation

User request: {}

Generate maintaining EXACT identity from Picture 1.<|im_end|>
<|im_start|>assistant
"""


class BD_QwenCharacterEdit(io.ComfyNode):
    """
    Enhanced Qwen-Image encoder with customizable system prompts for character consistency.
    Supports up to 3 reference images for multi-view character preservation.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_QwenCharacterEdit",
            display_name="BD Qwen Character Edit",
            category="🧠BrainDead/Character",
            description="Enhanced Qwen-Image encoder with customizable system prompts for character consistency. Supports up to 3 reference images.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_template", multiline=True, default=DEFAULT_CHARACTER_TEMPLATE),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
                io.Boolean.Input("enable_resize", default=True, optional=True),
                io.Boolean.Input("enable_vl_resize", default=True, optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.Image.Output(display_name="image1"),
                io.Image.Output(display_name="image2"),
                io.Image.Output(display_name="image3"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_template, vae=None, image=None, image2=None, image3=None,
                enable_resize=True, enable_vl_resize=True) -> io.NodeOutput:
        tokenizer = QwenImageTokenizer()

        input_images = [image, image2, image3]
        ref_latents = []
        images_vl = []
        processed_images = []
        image_prompt = ""

        image_count = sum(1 for img in input_images if img is not None)

        active_image_indices = []
        for i, img in enumerate(input_images):
            if img is not None:
                active_image_indices.append(i)
                image_prompt += f"Picture {i+1}: <|vision_start|><|image_pad|><|vision_end|>"
                if len(active_image_indices) < image_count:
                    image_prompt += "\n"

        template = system_template
        if image_count > 1 and "Picture" not in template:
            template_parts = template.split("<|vision_start|><|image_pad|><|vision_end|>")
            if len(template_parts) > 1:
                template = template_parts[0] + image_prompt + template_parts[1]
            else:
                template = template.replace(
                    "<|im_start|>user\n",
                    f"<|im_start|>user\n{image_prompt}\n"
                )

        tokenizer.llama_template_images = template

        for i, img in enumerate(input_images):
            if img is not None:
                samples = img.movedim(-1, 1)

                if enable_resize and vae is not None:
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8
                    s = comfy.utils.common_upscale(samples, width, height, "bicubic", "center")
                    vae_image = s.movedim(1, -1)
                    ref_latents.append(vae.encode(vae_image[:, :, :, :3]))
                    processed_images.append(vae_image)
                else:
                    processed_images.append(img)

                if enable_vl_resize:
                    total = int(384 * 384)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by)
                    height = round(samples.shape[2] * scale_by)
                    s = comfy.utils.common_upscale(samples, width, height, "bicubic", "center")
                    vl_image = s.movedim(1, -1)
                    images_vl.append(vl_image[:, :, :, :3])
                else:
                    images_vl.append(img[:, :, :, :3])

        original_tokenizer = clip.tokenizer
        clip.tokenizer = tokenizer

        full_prompt = image_prompt + prompt if image_count > 1 else prompt
        tokens = clip.tokenize(full_prompt, images=images_vl)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"reference_latents": ref_latents}, append=True
            )

        clip.tokenizer = original_tokenizer

        while len(processed_images) < 3:
            empty_image = torch.zeros((1, 64, 64, 3))
            processed_images.append(empty_image)

        if len(ref_latents) > 0:
            latent_out = {"samples": ref_latents[0]}
        else:
            latent_out = {"samples": torch.zeros(1, 4, 128, 128)}

        return io.NodeOutput(conditioning, processed_images[0], processed_images[1], processed_images[2], latent_out)


class BD_QwenT2ICustom(io.ComfyNode):
    """Qwen-Image text-to-image encoder with custom system prompt."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_QwenT2ICustom",
            display_name="BD Qwen T2I Custom",
            category="🧠BrainDead/Character",
            description="Qwen-Image text-to-image encoder with custom system prompt.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_template", multiline=True, default=DEFAULT_T2I_TEMPLATE),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_template) -> io.NodeOutput:
        tokenizer = QwenImageTokenizer()
        tokenizer.llama_template = system_template

        original_tokenizer = clip.tokenizer
        clip.tokenizer = tokenizer

        try:
            tokens = clip.tokenize(prompt, images=[])
            conditioning = clip.encode_from_tokens_scheduled(tokens)
        finally:
            clip.tokenizer = original_tokenizer

        return io.NodeOutput(conditioning)


class BD_QwenMultiImage(io.ComfyNode):
    """Multi-image Qwen encoder with weight mode control."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_QwenMultiImage",
            display_name="BD Qwen Multi-Image",
            category="🧠BrainDead/Character",
            description="Multi-image Qwen encoder with weight mode control for character-consistent image editing.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_template", multiline=True, default=DEFAULT_MULTI_TEMPLATE),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
                io.Boolean.Input("enable_resize", default=True, optional=True),
                io.Boolean.Input("enable_vl_resize", default=True, optional=True),
                io.Combo.Input("weight_mode", options=["comfy", "diffusers", "raw"], default="diffusers", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.Image.Output(display_name="image1"),
                io.Image.Output(display_name="image2"),
                io.Image.Output(display_name="image3"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_template, vae=None,
                image1=None, image2=None, image3=None,
                enable_resize=True, enable_vl_resize=True,
                weight_mode="diffusers") -> io.NodeOutput:

        tokenizer = QwenImageTokenizer()

        if weight_mode in ["diffusers", "raw"]:
            tokenizer.disable_weights = True

        input_images = [image1, image2, image3]
        ref_latents = []
        images_vl = []
        processed_images = []

        for img in input_images:
            if img is not None:
                samples = img.movedim(-1, 1)

                if enable_resize and vae is not None:
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8
                    s = comfy.utils.common_upscale(samples, width, height, "bicubic", "center")
                    vae_image = s.movedim(1, -1)
                    ref_latents.append(vae.encode(vae_image[:, :, :, :3]))
                    processed_images.append(vae_image)
                else:
                    processed_images.append(img)

                if enable_vl_resize:
                    total = int(384 * 384)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by)
                    height = round(samples.shape[2] * scale_by)
                    s = comfy.utils.common_upscale(samples, width, height, "bicubic", "center")
                    vl_image = s.movedim(1, -1)
                    images_vl.append(vl_image[:, :, :, :3])
                else:
                    images_vl.append(img[:, :, :, :3])

        tokenizer.llama_template_images = system_template

        original_tokenizer = clip.tokenizer
        clip.tokenizer = tokenizer

        tokens = clip.tokenize(prompt, images=images_vl)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"reference_latents": ref_latents}, append=True
            )

        clip.tokenizer = original_tokenizer

        while len(processed_images) < 3:
            empty_image = torch.zeros((1, 64, 64, 3))
            processed_images.append(empty_image)

        latent_out = {"samples": ref_latents[0] if len(ref_latents) > 0 else torch.zeros(1, 4, 128, 128)}

        return io.NodeOutput(conditioning, processed_images[0], processed_images[1], processed_images[2], latent_out)


class BD_QwenIdentityLock(io.ComfyNode):
    """Identity-focused Qwen encoder with strong character preservation."""

    STRENGTH_VALUES = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.5", "1.8", "2.0"]
    ROLES = ["character", "style", "pose", "environment", "reference", "control"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_QwenIdentityLock",
            display_name="BD Qwen Identity Lock",
            category="🧠BrainDead/Character",
            description="Identity-focused Qwen encoder with strong character preservation using weighted multi-image input.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_template", multiline=True, default=DEFAULT_IDENTITY_TEMPLATE),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
                io.Combo.Input("image1_strength", options=cls.STRENGTH_VALUES, default="1.5", optional=True),
                io.Combo.Input("image2_strength", options=cls.STRENGTH_VALUES, default="0.6", optional=True),
                io.Combo.Input("image3_strength", options=cls.STRENGTH_VALUES, default="0.4", optional=True),
                io.Combo.Input("image1_role", options=cls.ROLES, default="character", optional=True),
                io.Combo.Input("image2_role", options=cls.ROLES, default="style", optional=True),
                io.Combo.Input("image3_role", options=cls.ROLES, default="reference", optional=True),
                io.Boolean.Input("enable_resize", default=True, optional=True),
                io.Boolean.Input("enable_vl_resize", default=True, optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.Image.Output(display_name="image1"),
                io.Image.Output(display_name="image2"),
                io.Image.Output(display_name="image3"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def _apply_weight_to_latent(cls, latent, strength, role):
        """Apply weight multiplier to latent based on strength and role."""
        strength_value = float(strength)
        role_multipliers = {
            "character": 1.2,
            "style": 0.8,
            "pose": 0.75,
            "environment": 0.5,
            "reference": 0.7,
            "control": 1.1
        }
        final_strength = strength_value * role_multipliers.get(role, 1.0)
        return latent * final_strength

    @classmethod
    def execute(cls, clip, prompt, system_template, vae=None,
                image1=None, image2=None, image3=None,
                image1_strength="1.5", image2_strength="0.6", image3_strength="0.4",
                image1_role="character", image2_role="style", image3_role="reference",
                enable_resize=True, enable_vl_resize=True) -> io.NodeOutput:

        tokenizer = QwenImageTokenizer()

        input_images = [image1, image2, image3]
        strengths = [image1_strength, image2_strength, image3_strength]
        roles = [image1_role, image2_role, image3_role]

        ref_latents = []
        weighted_latents = []
        images_vl = []
        processed_images = []

        for i, (img, strength, role) in enumerate(zip(input_images, strengths, roles)):
            if img is not None:
                samples = img.movedim(-1, 1)

                if vae is not None and enable_resize:
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8
                    s = comfy.utils.common_upscale(samples, width, height, "bicubic", "center")
                    vae_image = s.movedim(1, -1)

                    latent = vae.encode(vae_image[:, :, :, :3])
                    weighted_latent = cls._apply_weight_to_latent(latent, strength, role)
                    weighted_latents.append(weighted_latent)
                    ref_latents.append(latent)
                    processed_images.append(vae_image)
                else:
                    processed_images.append(img)

                if enable_vl_resize:
                    total = int(384 * 384)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by)
                    height = round(samples.shape[2] * scale_by)
                    s = comfy.utils.common_upscale(samples, width, height, "bicubic", "center")
                    vl_image = s.movedim(1, -1)
                    images_vl.append(vl_image[:, :, :, :3])
                else:
                    images_vl.append(img[:, :, :, :3])

        tokenizer.llama_template_images = system_template
        tokenizer.image_weights = [float(s) for s in strengths]
        tokenizer.image_roles = roles

        original_tokenizer = clip.tokenizer
        clip.tokenizer = tokenizer

        tokens = clip.tokenize(prompt, images=images_vl)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        if len(weighted_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"reference_latents": weighted_latents}, append=True
            )

        weight_metadata = {
            "image_strengths": strengths,
            "image_roles": roles,
        }
        conditioning = node_helpers.conditioning_set_values(
            conditioning, {"weight_metadata": weight_metadata}, append=True
        )

        clip.tokenizer = original_tokenizer

        while len(processed_images) < 3:
            empty_image = torch.zeros((1, 64, 64, 3))
            processed_images.append(empty_image)

        latent_out = {"samples": weighted_latents[0] if len(weighted_latents) > 0 else torch.zeros(1, 4, 128, 128)}

        return io.NodeOutput(conditioning, processed_images[0], processed_images[1], processed_images[2], latent_out)


# =============================================================================
# V3 Node List for Extension
# =============================================================================

CHARACTER_V3_NODES = [
    BD_QwenCharacterEdit,
    BD_QwenT2ICustom,
    BD_QwenMultiImage,
    BD_QwenIdentityLock,
]

# =============================================================================
# V1 Compatibility - Node Mappings
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "BD_QwenCharacterEdit": BD_QwenCharacterEdit,
    "BD_QwenT2ICustom": BD_QwenT2ICustom,
    "BD_QwenMultiImage": BD_QwenMultiImage,
    "BD_QwenIdentityLock": BD_QwenIdentityLock,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BD_QwenCharacterEdit": "BD Qwen Character Edit",
    "BD_QwenT2ICustom": "BD Qwen T2I Custom",
    "BD_QwenMultiImage": "BD Qwen Multi-Image",
    "BD_QwenIdentityLock": "BD Qwen Identity Lock",
}
