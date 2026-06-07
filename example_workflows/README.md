# BrainDead Workflow Templates

Workflows here ship with the node pack and appear in ComfyUI's
**Workflow → Browse Templates** browser under "ComfyUI-BrainDead"
(ComfyUI scans `*/example_workflows/*.json` and serves them as templates).

**Full authoring guide:** [`.claude/skills/braindead-templates/SKILL.md`](../.claude/skills/braindead-templates/SKILL.md)
(assignable to a subagent to bulk-create templates, one per function set).

## The two rules that bite

1. **snake_case file names** — no spaces, dots, or special characters
   (`cubepart_part_decomposition.json`). Spaces break the thumbnail URL → no thumbnail shows.
2. **Thumbnail must be `.jpg`**, same basename as the `.json`, **no `-1` suffix**. The frontend
   hardcodes `mediaSubtype:'jpg'` for custom-node templates and requests
   `/api/workflow_templates/<pack>/<name>.jpg` — a `.png` (or the official repo's `-1.webp`) is
   never loaded. QwenVL/QwenTTS confirm: `<name>.json` + `<name>.jpg`.

## Each template should have

- The *main* nodes of one function set, wired end to end, with **portable inputs**
  (BD Load Mesh picker / LoadImage — no machine-specific absolute paths).
- A **`MarkdownNote`** node ("About — …") with a description, the step-by-step flow, model/VRAM
  notes, and the BrainDead links block (below).
- A landscape `.jpg` thumbnail (~1180×680): a rendered result or a node-graph screenshot.

## BrainDead links block (paste into every MarkdownNote)

```
---
**BrainDeadGuild** — created by **BizaNator**
[BrainDeadGuild.com](https://BrainDeadGuild.com) · [BrainDead.TV](https://BrainDead.TV) · [GitHub](https://github.com/BizaNator/ComfyUI-BrainDead) · [Discord](https://braindeadguild.com/discord)
```

## Verify (after adding a file, restart `comfyui-stable` first)

- `GET /workflow_templates` → name listed under `ComfyUI-BrainDead`.
- `GET /api/workflow_templates/ComfyUI-BrainDead/<name>.json` and `.jpg` → 200.
- Hard-refresh the browser (Ctrl+Shift+R) — the template browser caches thumbnails.

## Templates

| Template (file) | What it shows |
|-----------------|---------------|
| `BD-CubePart_Part_Decomposition` | Load Mesh → CubePart Segment → Mesh Preview (grid) + Preview 3D + Get Part → Export (Save Context). |
| `BD-trellis2_shape_to_texture` | Load Image → Remove BG → Trellis2 Conditioning → Image-to-Shape → Shape-to-Textured-Mesh → Preview 3D. |
| `BD-ovoxel_pbr_bake` | Load Mesh → Mesh-to-OVoxel → OVoxel Bake → albedo/normal/roughness/metallic + Export GLB. |
| `BD-pixal3d_image_to_3d` | Load Image → Pixal3D Preprocess (MoGe FOV) → Image-to-3D → CuMesh Simplify + OVoxel Bake. |
| `BD-sam3_parts_segmentation` | Load Image → scale → Lotus2 depth → QwenVL tags → SAM3 Multi-Prompt → Parts Refine → Fill Holes → Parts Builder → PartsBatchEdit (Qwen Inpaint) → Parts Export. Mirrors the full COB_PartBuilder pipeline. |
| `BD-lotus2_depth_normal` | Load Image → Lotus-2 Loader → Predict (depth/normal) → previews. |
| `BD-facewrap_pipeline` | FaceWrap nodes end to end. |
| `BD-glsl_skin_tinting` | Load Image → GLSL Batch (skin tint) → Save Batch. |
| `BD-character_consistency` | Qwen-Image character consistency pipeline. |
| `BD-background_removal` | Load Image → BD Remove Background (SAM3 + pymatting) → RGBA + white/black composites + BD Mask Batch Index (channel-extraction demo). |
| `BD-face_segmentation` | Load Image → BD MP SAM3 Face Segment (25+ anatomy masks) → BD MP Face Infill (UV-ready socket fill) → previews. |
| `BD-channel_operations` | Load Image → BD Unpack Channels → BD Pack Channels (round-trip R/G/B) → BD Channel Merge (inject into single channel) → previews. |
| `BD-mask_tools` | Load Image → BD Luminance Mask + BD Mask Flatten + BD Crop to Mask + BD Fill Mask Holes → previews. Each node is a standalone section. |
