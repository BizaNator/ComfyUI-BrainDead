---
name: braindead-templates
description: "Create ComfyUI workflow templates for ComfyUI-BrainDead that show in Browse Templates (with thumbnails), one per major node function-set. Use when asked to: add a BrainDead template, make a workflow template, create example_workflows, ship a sample workflow, bulk-create templates, add a thumbnail to a template, or onboard a workflow into the pack. Covers the custom-node example_workflows mechanism (simpler than the official Comfy-Org repo: no index.json / bundles / i18n)."
---

# Authoring BrainDead Workflow Templates

Templates ship in **`example_workflows/`** at the repo root and appear in ComfyUI under
**Workflow → Browse Templates → ComfyUI-BrainDead**. ComfyUI auto-discovers
`*/example_workflows/*.json` and serves them — there is **no** `index.json`, `bundles.json`,
version bump, or i18n step (those are only for the official Comfy-Org/workflow_templates repo).
Goal: one polished template per major function set so users learn each pipeline.

## What custom-node templates support vs the official repo

The frontend builds every custom-node template entry as a **hardcoded literal** — it never reads an
index.json for our pack:
`{ name, title: name, description: name, mediaType: 'image', mediaSubtype: 'jpg', sourceModule }`.

| Official feature | Custom-node `example_workflows`? | How we get it |
|------------------|----------------------------------|---------------|
| Static thumbnail | ✅ | `<name>.jpg` (jpg only, no `-1`) |
| Description in browser | ❌ (browser shows the filename) | put it in a **MarkdownNote** node *inside* the workflow |
| Title/tags in browser | ❌ (title = filename) | choose a clear snake_case filename |
| Animated/video thumbnail | ❌ (`mediaType` hardcoded `image`) | — (official repo only) |
| Hover **effects** (`thumbnailVariant`: zoomHover / compareSlider / hoverDissolve) | ❌ (not set) | — (official repo only) |
| **Model metadata** (auto-download) | ✅ — lives in the workflow JSON, not index.json | `models` array in each loader node's `properties` |
| **Node versions** (`cnr_id`/`ver`) | ✅ — in the workflow JSON | node `properties` (only if the pack/node is in the ComfyUI registry) |
| **Blueprint subgraphs** | ✅ — in the workflow JSON | `definitions.subgraphs` (package a sub-pipeline as one node) |
| index.json metadata, bundles, i18n | ❌ | — (official repo only) |

So: **everything that lives in the workflow JSON we can and should do** (description-via-note, model
metadata, node versions, subgraphs). The browser-chrome extras (animation, hover effects, rich
title/description/tags) require getting the template into the **official Comfy-Org/workflow_templates
repo** (submit with `requiresCustomNodes: ["comfyui-braindead"]`) — pursue that for flagship
templates; otherwise accept a static jpg + in-canvas note. Treat the official `adding-templates` skill
as the spec for that path.

## Hard rules (these are the bugs people hit)

1. **snake_case file names** — no spaces, dots (other than `.json`/`.jpg`), or special chars.
   `cubepart_part_decomposition.json`, NOT `CubePart Part Decomposition.json`. Spaces break the
   thumbnail URL the frontend builds → no thumbnail shows.
2. **Thumbnail MUST be `.jpg`**, same basename as the `.json`, **no `-1` suffix**.
   The frontend hardcodes `mediaSubtype:'jpg'` for custom-node templates and requests
   `/api/workflow_templates/ComfyUI-BrainDead/<name>.jpg`. A `.png` (or `-1.webp`, which is the
   *official-repo* convention) is never loaded for custom nodes. (Confirmed against QwenVL/QwenTTS,
   which ship `<name>.json` + `<name>.jpg`.)
3. **Restart `comfyui-stable` after adding a new file** — the static serving route for
   `example_workflows/` is registered at startup. (`/workflow_templates` *listing* updates live, but
   serving the file needs the restart.)
4. **Double-quotes only** in JSON. Build JSON with a script (`json.dump`), never hand-type braces.
5. **Widget order = node input DEFINITION order** in the UI. BD nodes are authored required-first,
   so a node's `widgets_values` array must be in **required-then-optional** order, skipping
   connection-type inputs and any input that is wired (a linked widget is removed from the array).
   Get the order from `/object_info/<NodeId>` (`required` keys then `optional` keys). Mis-ordering
   shifts every widget (e.g. guidance_scale shows the steps value). If a node interleaves optional
   widgets before required ones, fix the node schema (required-first) rather than the template.

## Steps

### 1. Pick the function set + nodes
One template per pipeline (see the table at the end). Wire the *main* nodes of that set end to end
with **portable inputs**: BD Load Mesh (picker) / LoadImage / BD-internal generators — never a
machine-specific absolute path (the template must run on any install).

### 2. Build the workflow JSON (UI graph format)
Either **export from ComfyUI** (Save → Export; ideally launched with `--disable-all-custom-nodes`
off so BD nodes load) OR hand-author with a Python builder. UI graph format:
`{id, revision, last_node_id, last_link_id, nodes:[...], links:[...], groups, config, extra, version}`.
- Each node: `{id, type, pos, size, flags, order, mode, inputs:[{name,type,link}],
  outputs:[{name,type,links,slot_index}], properties:{"Node name for S&R":type}, widgets_values:[...]}`.
- `links`: `[link_id, src_node, src_slot, dst_node, dst_slot, type]`.
- Always include a **`MarkdownNote`** node (frontend-only; the runner skips it) titled "About — …"
  with the description, step-by-step flow, model/VRAM notes, and the BrainDead links block (below).

### 3. Validate the wiring
`python3 /opt/comfyui/run_workflow.py <wf>.json --server http://127.0.0.1:8188 --dry-run`
confirms every node converts and widgets map to the right inputs (it uses required-then-optional,
which matches the UI once nodes are required-first). For runnable smoke tests, set a real input and
drop `--dry-run`.

### 4. Make the thumbnail (`.jpg`)
A landscape card (~1180×680) reads well. Two good sources:
- A rendered **result** (e.g. composite the BD Mesh Preview grid + a title + the links footer).
- A **node-graph screenshot** (QwenVL/QwenTTS style) — capture the canvas, or draw a stylized
  node diagram. Save as JPEG quality ~90, keep under a few hundred KB.
Name it `<same_snake_case_basename>.jpg` next to the `.json`.

### 5. Embed model auto-download metadata (required when the workflow has loader nodes)
For **every** standard loader node (`UNETLoader`, `VAELoader`, `CLIPLoader`, checkpoint loaders, etc.),
add a `"models"` array to the node's `"properties"` so ComfyUI offers to download missing models on
load — this lives in the workflow JSON, so it works for custom-node templates:
```json
"properties": {
  "Node name for S&R": "VAELoader",
  "models": [{ "name": "wan_2.1_vae.safetensors",
               "url": "https://huggingface.co/.../wan_2.1_vae.safetensors?download=true",
               "hash": "<sha256>", "hash_type": "SHA256", "directory": "vae" }]
}
```
`name` must match `widgets_values` **exactly**. BD nodes that auto-download internally (CubePart,
Pixal3D, Lotus-2…) don't need this — note that in the MarkdownNote instead.

### 5b. Embed node versions (when the pack is in the ComfyUI registry)
Pin nodes that need a specific version with `cnr_id` + `ver` in `properties`
(`{"cnr_id":"comfy-core","ver":"0.3.26"}`). Only add a BD `cnr_id` once the pack is published to the
registry — an unresolvable `cnr_id` triggers "missing node" warnings, so omit it if unsure.

### 5c. Blueprint subgraphs (optional, for complex pipelines)
Package a reusable sub-pipeline as a single subgraph node via the workflow's `definitions.subgraphs`
(see the official `importing-subgraphs` skill). Useful when a function set has a repeated multi-node
block (e.g. an OVoxel bake chain) — keeps the template readable. Subgraphs live in the JSON, so they
ship fine for custom-node templates.

### 6. Deploy + restart + verify
- Copy the `.json` + `.jpg` into **both** dev and `/opt/comfyui/stable/.../example_workflows/`.
- `sudo systemctl restart comfyui-stable`.
- Verify:
  - `curl -s :8188/workflow_templates` → name listed under `ComfyUI-BrainDead`.
  - `curl -o /dev/null -w "%{http_code}" ":8188/api/workflow_templates/ComfyUI-BrainDead/<name>.json"` → 200.
  - same for `<name>.jpg` → 200.
- In the browser, hard-refresh (Ctrl+Shift+R) — the template browser caches thumbnails.

## BrainDead links block (paste into every MarkdownNote)

```
---
**BrainDeadGuild** — created by **BizaNator**
[BrainDeadGuild.com](https://BrainDeadGuild.com) · [BrainDead.TV](https://BrainDead.TV) · [GitHub](https://github.com/BizaNator/ComfyUI-BrainDead) · [Discord](https://braindeadguild.com/discord)
```

## Function sets to template (assign one per subagent)

| Template | Main nodes |
|----------|-----------|
| `cubepart_part_decomposition` ✅ | Load Mesh → CubePart Segment → Mesh Preview + Preview 3D + Get Part → Export |
| `ovoxel_pbr_bake` | Trellis2 Texture → OVoxel Bake → albedo/normal/rough/metallic → Export |
| `sam3_parts_pipeline` | SAM3 Multi-Prompt → Parts pipeline → PartsCompose / PartsExport |
| `trellis2_shape_to_texture` | Trellis2 Generate Shape → Shape-to-Textured-Mesh |
| `pixal3d_image_to_3d` | Pixal3D Preprocess → Image to 3D → CuMesh/OVoxel bake |
| `facewrap_pipeline` | FaceWrap nodes end to end |
| `glsl_skin_tinting` | GLSL Batch skin-tint → Save Batch |
| `lotus2_depth_normal` | Lotus-2 Loader → Predict (depth/normal) |

## Quick reference

| File / Dir | Purpose |
|------------|---------|
| `example_workflows/<name>.json` | Workflow (UI graph format), name = display title |
| `example_workflows/<name>.jpg` | Thumbnail (jpg, same basename, no `-1`) |
| `example_workflows/README.md` | Index + this convention |
| `/opt/comfyui/run_workflow.py` | Graph→API convert + run/dry-run for validation |
