"""
BD Fashn Human Parser — SegFormer-B4 from fashn-ai/fashn-human-parser.

Weights ship under NVIDIA Source Code License (research/non-commercial
leaning). For commercial pipelines prefer BD_ATRHumanParser (MIT weights).
"""

from comfy_api.latest import io

from .labels import FASHN_LABELS
from .types import HumanParseMapOutput
from ._segformer import load_segformer, resolve_cache_dir, resolve_device, resolve_dtype, run_segformer


class BD_FashnHumanParser(io.ComfyNode):
    """SegFormer-B4 human parser (FASHN, 18 classes)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_FashnHumanParser",
            display_name="BD Fashn Human Parser",
            category="🧠BrainDead/Segmentation",
            description=(
                "Parse a human image into 18 semantic regions using fashn-ai/fashn-human-parser "
                "(SegFormer-B4). Outputs a HUMAN_PARSE_MAP for the mask-split node. "
                "Weights are NVIDIA SegFormer license — prefer ATR backend for commercial work."
            ),
            inputs=[
                io.Image.Input("image"),
                io.String.Input(
                    "model_id", default="fashn-ai/fashn-human-parser",
                    tooltip="HuggingFace repo id"
                ),
                io.Combo.Input("device", options=["auto", "cuda", "cpu"], default="auto", optional=True),
                io.Combo.Input("dtype", options=["fp16", "bf16", "fp32"], default="fp16", optional=True),
                io.String.Input(
                    "cache_dir", default="", optional=True,
                    tooltip="Override HF cache dir. Blank → /srv/AI_Stuff/models/huggingface (if present), else HF default."
                ),
            ],
            outputs=[
                HumanParseMapOutput(display_name="parse_map"),
            ],
        )

    @classmethod
    def execute(cls, image, model_id, device="auto", dtype="fp16", cache_dir="") -> io.NodeOutput:
        device = resolve_device(device)
        torch_dtype = resolve_dtype(dtype)
        processor, model = load_segformer(model_id, device, torch_dtype, resolve_cache_dir(cache_dir))

        class_map = run_segformer(processor, model, image, device, torch_dtype)

        return io.NodeOutput({
            "class_map": class_map,
            "labels": FASHN_LABELS,
            "backend": "fashn",
        })


FASHN_PARSER_V3_NODES = [BD_FashnHumanParser]
FASHN_PARSER_NODES = {"BD_FashnHumanParser": BD_FashnHumanParser}
FASHN_PARSER_DISPLAY_NAMES = {"BD_FashnHumanParser": "BD Fashn Human Parser"}
