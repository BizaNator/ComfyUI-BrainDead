"""
BD ATR Human Parser — ATR/LIP-scheme SegFormer (default mattmdjaga/segformer_b2_clothes).

MIT-licensed weights, friendlier for commercial pipelines than FASHN.
Same HUMAN_PARSE_MAP output shape so downstream nodes are interchangeable.
"""

from comfy_api.latest import io

from .labels import ATR_LABELS
from .types import HumanParseMapOutput
from ._segformer import load_segformer, resolve_cache_dir, resolve_device, resolve_dtype, run_segformer


class BD_ATRHumanParser(io.ComfyNode):
    """SegFormer human parser using ATR-scheme labels (default: segformer_b2_clothes, MIT)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_ATRHumanParser",
            display_name="BD ATR Human Parser",
            category="🧠BrainDead/Segmentation",
            description=(
                "Parse a human image into 18 semantic regions using an ATR-scheme SegFormer. "
                "Default weights (mattmdjaga/segformer_b2_clothes) are MIT-licensed. "
                "Outputs a HUMAN_PARSE_MAP for the mask-split node."
            ),
            inputs=[
                io.Image.Input("image"),
                io.String.Input(
                    "model_id", default="mattmdjaga/segformer_b2_clothes",
                    tooltip="HuggingFace repo id (must use ATR 18-class scheme)"
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
            "labels": ATR_LABELS,
            "backend": "atr",
        })


ATR_PARSER_V3_NODES = [BD_ATRHumanParser]
ATR_PARSER_NODES = {"BD_ATRHumanParser": BD_ATRHumanParser}
ATR_PARSER_DISPLAY_NAMES = {"BD_ATRHumanParser": "BD ATR Human Parser"}
