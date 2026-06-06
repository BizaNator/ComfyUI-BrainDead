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
| `cubepart_part_decomposition` | Load Mesh → CubePart Segment → Mesh Preview (thumbnail grid) + Preview 3D + Get Part → Export (Save Context). |
