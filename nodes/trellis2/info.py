"""
TRELLIS2 information/helper nodes.

BD_Trellis2DualConditioning - Info node explaining dual conditioning workflow
"""


class BD_Trellis2DualConditioning:
    """
    INFO NODE: Explains dual conditioning workflow for TRELLIS2.

    This is a documentation/helper node. For actual dual conditioning,
    use two separate TRELLIS.2 Get Conditioning nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "info_trigger": ("STRING", {"default": "Connect to see workflow info", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("workflow_info",)
    FUNCTION = "show_info"
    CATEGORY = "BrainDead/TRELLIS2"
    OUTPUT_NODE = True
    DESCRIPTION = """
INFO: How to use dual conditioning for TRELLIS2

This node explains the dual-conditioning workflow where you use:
- One image for SHAPE (e.g., traced outline, edge detection)
- Another image for TEXTURE (e.g., clean reference render)

WORKFLOW:
=========
1. Create TWO "TRELLIS.2 Get Conditioning" nodes:
   - First one: Connect your SHAPE image (traced outline)
   - Second one: Connect your TEXTURE image (clean reference)

2. "TRELLIS.2 Image to Shape" node:
   - Connect the SHAPE conditioning
   - This determines the 3D geometry

3. "TRELLIS.2 Shape to Textured Mesh" node:
   - Connect SHAPE conditioning to 'conditioning' (required)
   - Connect TEXTURE conditioning to 'texture_conditioning' (optional)
   - This applies texture from the clean image to the generated shape

This workflow is now supported natively in the TRELLIS2 nodes!
"""

    def show_info(self, info_trigger=""):
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
        return (info,)


# Node exports
TRELLIS2_INFO_NODES = {
    "BD_Trellis2DualConditioning": BD_Trellis2DualConditioning,
}

TRELLIS2_INFO_DISPLAY_NAMES = {
    "BD_Trellis2DualConditioning": "BD TRELLIS2 Dual Conditioning (Info)",
}
