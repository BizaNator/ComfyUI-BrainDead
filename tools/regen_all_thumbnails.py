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
    "BD-sam3_parts_segmentation": "sam3", "BD-lotus2_depth_normal": "lotus2",
    "BD-facewrap_pipeline": "facewrap", "BD-glsl_skin_tinting": "glsl",
    "BD-character_consistency": "character", "BD-background_removal": "bgremoval",
    "BD-face_segmentation": "faceseg", "BD-channel_operations": "channels", "BD-mask_tools": "masks",
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
        "bullets": ["Load Image -> Preprocess (MoGe FOV)", "Image-to-3D -> mesh + voxelgrid",
                    "CuMesh Simplify (sharp-edge GPU decimate)",
                    "OVoxel Texture Bake -> PBR maps", "Export GLB + Preview 3D"],
        "chips": ["albedo", "normal", "rough", "metal"]},
    "BD-sam3_parts_segmentation": {
        "title": "SAM3 Parts", "subtitle": "Prompt-driven part segmentation + edit",
        "bullets": ["Image -> Lotus2 depth -> QwenVL tags", "SAM3 Multi-Prompt -> Parts Refine",
                    "Fill Holes -> Parts Builder", "PartsBatchEdit (Qwen Inpaint)",
                    "Parts Export (per-part + PSD)"]},
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
        if f.endswith(".json") and f[:-5] not in have:
            print(f"  [WARN] {f} has no CONFIGS entry — no thumbnail generated")
    print(f"\nregenerated {made} thumbnail(s){' + deployed to stable' if deploy else ''}"
          f"{'; skipped (no json): ' + ', '.join(skipped) if skipped else ''}")
    print("Hard-refresh (Ctrl+Shift+R) Browse Templates — thumbnails are browser-cached 24h.")


if __name__ == "__main__":
    main()
