#!/usr/bin/env python3
"""
Regenerate ALL ComfyUI-BrainDead workflow-template thumbnails in one command.

Each thumbnail is the combined card (make_thumbnail.py) over the auto-drawn node
graph from its sibling example_workflows/<name>.json. Add a CONFIGS entry when
you add a template (key = the .json basename, no extension).

    python3 tools/regen_all_thumbnails.py            # regenerate into example_workflows/
    python3 tools/regen_all_thumbnails.py --deploy    # also copy .jpg into the stable install

Reminder: thumbnails are browser-cached (Cache-Control max-age=86400). After
regenerating, hard-refresh (Ctrl+Shift+R) the Browse Templates panel to see them.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_thumbnail import make

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EW = os.path.join(REPO, "example_workflows")
DOCS_IMG = os.path.join(REPO, "docs", "images")
STABLE_EW = "/opt/comfyui/stable/custom_nodes/ComfyUI-BrainDead/example_workflows"

# template name -> short slug used by the GitHub README gallery (docs/images/workflow_<slug>.jpg).
GALLERY_SLUG = {
    "BD-CubePart_Part_Decomposition": "cubepart", "BD-trellis2_shape_to_texture": "trellis2",
    "BD-ovoxel_pbr_bake": "ovoxel", "BD-pixal3d_image_to_3d": "pixal3d",
    "BD-parts_builder": "partsbuilder", "BD-lotus2_depth_normal": "lotus2",
    "BD-facewrap_pipeline": "facewrap", "BD-glsl_skin_tinting": "glsl",
    "BD-character_consistency": "character", "BD-background_removal": "bgremoval",
    "BD-face_segmentation": "faceseg", "BD-channel_operations": "channels", "BD-mask_tools": "masks", "BD-pbr_from_image": "pbr",
    "BD-game_engine_packing": "packing", "BD-atlas_flipbook": "flipbook",
    "BD-trellis2_unreal_fbx": "unrealfbx", "BD-isolate_part": "isolatepart",
    "BD-clothing_remover": "clothingremover",
    "BD-autorig_mia_to_uefn": "autorignuefn",
    "BD-autorig_mia_uefn_full": "autoriguefnfull",
    "BD-autorig_rig_preview": "autorigpreview",
    "BD-autorig_hymotion": "autorighymotion",
    "BD-autorig_anim_retarget": "autoriganim",
}

# name (== <name>.json) -> card config. Background auto-draws from the sibling json.
CONFIGS = {
    "BD-CubePart_Part_Decomposition": {
        "title": "CubePart", "subtitle": "Part-Controllable 3D Decomposition",
        "bullets": ["Load Mesh -> CubePart Segment (up to 8 parts)",
                    "One clean mesh per named part",
                    "Mesh Preview grid + interactive Preview 3D",
                    "Get Part -> Export (Save Context)"]},
    "BD-trellis2_shape_to_texture": {
        "title": "TRELLIS2", "subtitle": "Shape -> Textured Game-Ready Mesh",
        "bullets": ["Image -> Remove BG -> Conditioning",
                    "Image-to-Shape -> Shape-to-Textured-Mesh",
                    "CuMesh Simplify (sharp-edge GPU decimate)",
                    "OVoxel Texture Bake -> PBR maps", "Export GLB + Preview 3D"],
        "chips": ["albedo", "normal", "rough", "metal"]},
    "BD-ovoxel_pbr_bake": {
        "title": "OVoxel PBR Bake", "subtitle": "Bake PBR maps from any mesh",
        "bullets": ["Load Mesh -> Mesh to OVoxel", "Voxelize + extract PBR attributes",
                    "OVoxel Bake: simplify, UV-unwrap, bake", "Save maps + Export GLB"],
        "chips": ["albedo", "normal", "rough", "metal"]},
    "BD-pixal3d_image_to_3d": {
        "title": "Pixal3D", "subtitle": "Image -> Textured 3D",
        "bullets": ["Load Image -> Preprocess (FOV + bg)", "Image-to-3D -> mesh + voxelgrid",
                    "OVoxel Bake (decimate + UV + PBR, all-in-one)",
                    "Orient Mesh (upright / forward)", "Export GLB + Preview 3D"],
        "chips": ["albedo", "normal", "rough", "metal"],
        "background": "screenshots/pixal3d_bg.png"},
    "BD-isolate_part": {
        "title": "Isolate Part / Subject", "subtitle": "SAM3 + matting -> clean white-bg crop",
        "bullets": ["Load -> BD Remove Background (SAM3 prompt = what to keep)",
                    "Set prompts to a part (tank top / jacket) or 'person'",
                    "pymatting refines alpha at fine edges",
                    "Saves the clean white-bg crop -- stage 1 of SAM3->Trellis"],
        "chips": ["SAM3", "matting", "white BG"]},
    "BD-trellis2_unreal_fbx": {
        "title": "TRELLIS2 -> Unreal FBX", "subtitle": "Low-poly game-ready FBX (Blender)",
        "bullets": ["Image -> RMBG -> Trellis2 -> OVoxel Bake",
                    "Detail Normal (albedo micro-detail)",
                    "Sample vertex colors -> Pack Bundle",
                    "Blender FBX: textures + vertex colors in ONE file",
                    "~3k tris, flat-shaded -- studio Seam-4 dispatcher"],
        "chips": ["FBX", "low-poly", "vcol", "Unreal"],
        "background": "screenshots/trellis_unreal_fbx_bg.png"},
    "BD-parts_builder": {
        "title": "BD Parts Builder", "subtitle": "Prompt-driven part segmentation + edit",
        "bullets": ["Image -> Lotus2 depth -> QwenVL tags", "SAM3 Multi-Prompt -> Parts Refine",
                    "Fill Holes -> Parts Builder", "PartsBatchEdit (Qwen Inpaint)",
                    "Parts Export (per-part RGBA + PSD + category table)"]},
    "BD-lotus2_depth_normal": {
        "title": "Lotus-2", "subtitle": "FLUX diffusion depth + normal",
        "bullets": ["Load Image -> Lotus-2 Loader", "Predict depth or normal",
                    "Far higher quality than feedforward estimators",
                    "Map + raw + colorized previews"],
        "chips": ["depth", "normal"]},
    "BD-facewrap_pipeline": {
        "title": "FaceWrap", "subtitle": "Face landmark + socket pipeline",
        "bullets": ["MediaPipe face landmarks", "SAM3-guided face region masks",
                    "Socket infill (UV-ready fill)", "UV export"]},
    "BD-glsl_skin_tinting": {
        "title": "GLSL Skin Tinting", "subtitle": "GPU 4-output skin-tone shader",
        "bullets": ["Load Image -> BD GLSL Batch", "ILM / SR+Parts / Unity / Unreal outputs",
                    "Canonical skin tones", "Save Batch"],
        "chips": ["ILM", "Unity", "Unreal"]},
    "BD-character_consistency": {
        "title": "Character Consistency", "subtitle": "Qwen-Image identity-locked editing",
        "bullets": ["Multi-view edit with identity lock", "Prompt iteration across views",
                    "Save context naming"]},
    "BD-background_removal": {
        "title": "Background Removal", "subtitle": "SAM3 + pymatting alpha matting",
        "bullets": ["Load Image -> BD Remove Background", "SAM3 segments the subject from a text prompt",
                    "pymatting refines alpha at hair / fine edges",
                    "Outputs: RGBA + white & black composites",
                    "BD Mask Batch Index -> extract any channel"],
        "chips": ["RGBA", "white BG", "black BG"]},
    "BD-face_segmentation": {
        "title": "Face Segmentation", "subtitle": "MediaPipe + SAM3 anatomy masks",
        "bullets": ["Load Image -> BD MP SAM3 Face Segment", "25+ anatomy masks (eyes, lips, brows, skin...)",
                    "BD MP Face Infill -> UV-ready socket fill", "Per-part mask previews"],
        "chips": ["eyes", "lips", "brows", "skin", "teeth"],
        # real workflow screenshot used as the background instead of the auto node-graph
        "background": "screenshots/face_segmentation_bg.png"},
    "BD-channel_operations": {
        "title": "Channel Operations", "subtitle": "Pack / unpack / merge image channels",
        "bullets": ["Load Image -> BD Unpack Channels (R/G/B/A)", "BD Pack Channels -> round-trip recombine",
                    "BD Channel Merge -> inject into one channel", "Per-channel previews"],
        "chips": ["R", "G", "B", "A"]},
    "BD-mask_tools": {
        "title": "Mask Tools", "subtitle": "Luminance, flatten, crop, fill-holes",
        "bullets": ["BD Luminance Mask -> mask from brightness", "BD Mask Flatten -> merge a mask stack",
                    "BD Crop to Mask -> tight bounding-box crop", "BD Fill Mask Holes -> close interior gaps",
                    "Each node is a standalone section"]},
    "BD-pbr_from_image": {
        "title": "Image -> Full PBR", "subtitle": "albedo / normal / roughness / metallic / AO",
        "bullets": ["Remove BG -> silhouette mask",
                    "Lotus-2 Depth + Normal (masked, neutral bg)",
                    "SAM3 finds metal zones -> metallic",
                    "Derive PBR -> albedo/normal/rough/metal/AO",
                    "Packed ORM + ARM for game engines"],
        "chips": ["albedo", "normal", "rough", "metal", "AO"],
        # real workflow screenshot used as the background instead of the auto node-graph
        "background": "screenshots/pbr_from_image_bg.png"},
    "BD-game_engine_packing": {
        "title": "Game-Engine Packing", "subtitle": "isolate -> balance luma -> channel + atlas pack",
        "bullets": ["Remove BG -> SAM3 isolates parts (shirt/pants/jacket)",
                    "Per part: greyscale cutout -> normalize -> center median",
                    "BD Pack Channels -> R/G/B in one texture (no bleed)",
                    "Atlas more typically used on packed images / flipbooks",
                    "Each channel holds only its part"],
        "chips": ["R", "G", "B", "pack"],
        "background": "screenshots/game_engine_packing_bg.png"},
    "BD-atlas_flipbook": {
        "title": "Atlas / Flipbook", "subtitle": "tile frames / packed textures into a sprite sheet",
        "bullets": ["Load frames -> Image Batch -> BD Atlas Pack",
                    "Grid sheet (auto cols x rows, padding)",
                    "Sprite strip (rows=1) for UV-animated shaders",
                    "layout JSON = per-cell UV rects for the engine",
                    "Or wire packed textures into image_1..image_8"],
        "chips": ["grid", "strip", "UV", "flipbook"]},
    "BD-clothing_remover": {
        "title": "Clothing Remover", "subtitle": "Qwen Image Edit + ControlNet pose guidance",
        "bullets": ["Load Image -> Sapiens2 Pose (optional) -> ControlNet",
                    "Qwen Image Edit Plus — prompt-driven clothing swap / removal",
                    "Switch: use pose control or run unguided",
                    "LoRA stack (detail + style) + CFGNorm quality pass",
                    "KSampler -> VAEDecode -> Save"],
        "chips": ["Qwen", "ControlNet", "LoRA", "pose"]},
    "BD-autorig_mia_to_uefn": {
        "title": "AutoRig: MIA -> UEFN", "subtitle": "ML humanoid rigging in <1s",
        "bullets": ["Load mesh (GLB/OBJ) -> Make-It-Animatable (GPU)",
                    "Predicts Mixamo 52-bone skeleton + blend weights",
                    "Bone remap: Mixamo -> UEFN_Mannequin naming",
                    "Output FBX ready for weight-transfer or UEFN import"],
        "chips": ["MIA", "Mixamo", "UEFN", "FBX"]},
    "BD-autorig_mia_uefn_full": {
        "title": "AutoRig: Full UEFN Pipeline", "subtitle": "Mesh -> genuine UEFN skeleton (2 steps)",
        "bullets": ["Step 1: BD AutoRig MIA — ML Mixamo rigging (<1s GPU)",
                    "Step 2: BD AutoRig UEFN — Blender Data Transfer",
                    "Bundled SKM_UEFN_Mannequin as weight donor",
                    "Scale match + align + bake + transfer + bind",
                    "Output FBX importable directly into UEFN / Fortnite"],
        "chips": ["MIA", "UEFN", "Blender", "FBX"]},
    "BD-autorig_rig_preview": {
        "title": "AutoRig Rig Preview", "subtitle": "Full pipeline + bone visualization image",
        "bullets": ["Load Mesh -> BD AutoRig MIA (Mixamo rig)",
                    "BD AutoRig UEFN (weight transfer -> UEFN skeleton)",
                    "BD Rig Preview: headless Blender EEVEE render",
                    "2x2 grid: front / side / back / perspective views",
                    "Orange = joints, blue-white = bones, transparent = mesh"],
        "chips": ["EEVEE", "4-view", "bones", "joints"]},
    "BD-autorig_hymotion": {
        "title": "AutoRig + HunyuanMotion", "subtitle": "Text-to-motion generation + FBX export",
        "bullets": ["Load Qwen3-8B GGUF + HY-Motion-1.0-Lite network",
                    "Encode motion prompt -> Generate motion data",
                    "HYMotionPreview: skeleton animation strip (IMAGE)",
                    "HYMotionExportFBX: animated SMPL-H FBX output",
                    "Pair with BD AutoRig for full character pipeline"],
        "chips": ["HunyuanMotion", "Qwen3", "motion", "FBX"]},
    "BD-autorig_anim_retarget": {
        "title": "AutoRig + Anim Retarget", "subtitle": "Mesh → rig → motion → animated FBX",
        "bullets": ["BD AutoRig MIA (ML Mixamo rig) → BD AutoRig UEFN (skeleton)",
                    "HunyuanMotion text-to-motion (Qwen3-8B + HY-Motion-1.0-Lite)",
                    "HYMotionExportFBX: SMPL-H animated FBX",
                    "BD Anim Retarget: SMPL-H → UEFN bone mapping + Blender bake",
                    "Output: animated character FBX for UEFN / Fortnite"],
        "chips": ["MIA", "UEFN", "HunyuanMotion", "retarget"]},
}


def main():
    deploy = "--deploy" in sys.argv
    made, skipped = 0, []
    for name, cfg in CONFIGS.items():
        if not os.path.exists(os.path.join(EW, name + ".json")):
            skipped.append(name); continue
        out = os.path.join(EW, name + ".jpg")
        cfg = dict(cfg)
        if cfg.get("background") and not os.path.isabs(cfg["background"]):
            cfg["background"] = os.path.join(EW, cfg["background"])  # resolve vs example_workflows/
        make(out, cfg)
        if deploy and os.path.isdir(STABLE_EW):
            shutil.copy2(out, os.path.join(STABLE_EW, name + ".jpg"))
        # keep the GitHub README gallery image in sync
        slug = GALLERY_SLUG.get(name)
        if slug:
            os.makedirs(DOCS_IMG, exist_ok=True)
            shutil.copy2(out, os.path.join(DOCS_IMG, f"workflow_{slug}.jpg"))
        made += 1
        print(f"  ok {name}")
    # warn about any template json without a config entry
    have = set(CONFIGS)
    for f in os.listdir(EW):
        if f.endswith(".json") and not f.endswith(".api.json") and f[:-5] not in have:
            print(f"  [WARN] {f} has no CONFIGS entry — no thumbnail generated")
    print(f"\nregenerated {made} thumbnail(s){' + deployed to stable' if deploy else ''}"
          f"{'; skipped (no json): ' + ', '.join(skipped) if skipped else ''}")
    print("Hard-refresh (Ctrl+Shift+R) Browse Templates — thumbnails are browser-cached 24h.")


if __name__ == "__main__":
    main()
