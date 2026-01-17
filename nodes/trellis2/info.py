"""
V3 API TRELLIS2 information/helper nodes.

BD_Trellis2DualConditioning - Info node explaining dual conditioning workflow
"""

from comfy_api.latest import io


class BD_Trellis2DualConditioning(io.ComfyNode):
    """
    INFO NODE: Explains dual conditioning workflow for TRELLIS2.

    This is a documentation/helper node. For actual dual conditioning,
    use two separate TRELLIS.2 Get Conditioning nodes.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_Trellis2DualConditioning",
            display_name="BD TRELLIS2 Dual Conditioning (Info)",
            category="🧠BrainDead/TRELLIS2",
            description="INFO: How to use dual conditioning for TRELLIS2. Explains workflows for using different images for shape vs texture.",
            is_output_node=True,
            inputs=[
                io.String.Input("info_trigger", default="Connect to see workflow info", multiline=True, optional=True),
            ],
            outputs=[
                io.String.Output(display_name="workflow_info"),
            ],
        )

    @classmethod
    def execute(cls, info_trigger: str = "") -> io.NodeOutput:
        info = """
ADVANCED TRELLIS2 WORKFLOWS
===========================

1. DUAL CONDITIONING (different images for shape vs texture)
------------------------------------------------------------
Use case: Traced outline for shape, clean image for texture.

Setup:
- Add TWO "TRELLIS.2 Get Conditioning" nodes
- Shape image (outline) -> first conditioning
- Texture image (clean) -> second conditioning

Connections:
- First conditioning -> "Image to Shape" (conditioning)
- First conditioning -> "Shape to Textured Mesh" (conditioning)
- Second conditioning -> "Shape to Textured Mesh" (texture_conditioning)


2. FAST SHAPE + DETAILED TEXTURE (different resolutions)
--------------------------------------------------------
Use case: Fast shape at 512, detailed voxelgrid at 1024.

Setup:
- Add TWO "Load TRELLIS.2 Models" nodes
- First: resolution=512 (fast shape generation)
- Second: resolution=1024_cascade (detailed texture voxelgrid)

Connections:
- First model config -> "Image to Shape" (model_config)
- First model config -> "Shape to Textured Mesh" (model_config)
- Second model config -> "Shape to Textured Mesh" (texture_model_config)

This gives ~2x faster shape generation while maintaining
high-resolution color data for vertex color sampling.


3. COMBINED WORKFLOW (both features)
------------------------------------
- 512 model -> shape generation (fast)
- 1024_cascade model -> texture generation (detailed)
- Outline image -> shape conditioning
- Clean image -> texture conditioning

All four optional inputs can be used together!
"""
        print(f"[BD Dual Conditioning Info] Workflow information displayed")
        return io.NodeOutput(info)


# V3 node list for extension
TRELLIS2_INFO_V3_NODES = [
    BD_Trellis2DualConditioning,
]

# V1 compatibility - NODE_CLASS_MAPPINGS dict
TRELLIS2_INFO_NODES = {
    "BD_Trellis2DualConditioning": BD_Trellis2DualConditioning,
}

TRELLIS2_INFO_DISPLAY_NAMES = {
    "BD_Trellis2DualConditioning": "BD TRELLIS2 Dual Conditioning (Info)",
}
