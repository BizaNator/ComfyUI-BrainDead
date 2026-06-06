# BrainDead Workflow Templates

Workflows in this folder ship with the node pack and appear in ComfyUI's
**Workflow → Browse Templates** browser under "ComfyUI-BrainDead"
(ComfyUI scans `*/example_workflows/*.json` and serves them as templates).

## Authoring a template — one per major function set

For each major set of nodes (CubePart, Segmentation, OVoxel/PBR bake, FaceWrap,
GLSL, …) add a `<Title>.json` + matching `<Title>.png` here.

1. **`<Title>.json`** — the workflow in ComfyUI UI graph format. The file name
   (minus `.json`) is the template's display name, so name it readably
   (e.g. `CubePart Part Decomposition.json`).
   - Showcase the *main* nodes of that function set, wired end to end.
   - Include a **`MarkdownNote`** node titled "About — …" with: one-line
     description, the step-by-step flow, model/VRAM notes, and the BrainDead
     links block (below).
   - Prefer portable inputs (e.g. **BD Load Mesh** with a picker, or LoadImage)
     over machine-specific absolute paths so the template runs on any install.
2. **`<Title>.jpg`** — the thumbnail. **Must be `.jpg`** with the same basename
   as the `.json`: the ComfyUI frontend hardcodes `mediaSubtype: 'jpg'` for
   custom-node templates and requests `/api/workflow_templates/<pack>/<Title>.jpg`
   (a `.png` is never loaded). ~1180×680 landscape reads well (QwenVL/QwenTTS use
   ~2:1 node-graph screenshots). The CubePart card is composited from its BD Mesh
   Preview grid + title + links.

## BrainDead links block (paste into every MarkdownNote)

```
---
**BrainDeadGuild** — created by **BizaNator**
[BrainDeadGuild.com](https://BrainDeadGuild.com) · [BrainDead.TV](https://BrainDead.TV) · [GitHub](https://github.com/BizaNator/ComfyUI-BrainDead) · [Discord](https://braindeadguild.com/discord)
```

## Verify

- `GET /workflow_templates` → should list the file under `ComfyUI-BrainDead`.
- `GET /api/workflow_templates/ComfyUI-BrainDead/<Title>.json` (and `.png`) → 200.
- A **restart** is needed after adding a new `example_workflows/` file (the
  static serving route is registered at startup).

## Templates

| Template | What it shows |
|----------|---------------|
| **CubePart Part Decomposition** | Load Mesh → CubePart Segment → Mesh Preview (thumbnail grid) + Preview 3D + Get Part → Export (Save Context). |
