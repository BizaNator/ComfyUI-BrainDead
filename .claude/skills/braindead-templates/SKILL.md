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

### 4. Make the thumbnail (`.jpg`) — ALWAYS use `tools/make_thumbnail.py`
Do NOT hand-roll thumbnail code — every thumbnail must use the canonical generator so the whole set
stays visually identical (1180×680 dark card: accent bar, purple "BrainDead" wordmark, big title,
subtitle, bullet flow, optional tag chips, BrainDead footer). It also **sanitizes text** (emoji/CJK
are dropped — DejaVu renders them as tofu boxes, a bug we hit repeatedly).

```bash
python3 tools/make_thumbnail.py example_workflows/<name>.jpg '{
  "title": "Background Removal",
  "subtitle": "SAM3 + pymatting alpha matting",
  "bullets": ["Load Image -> BD Remove Background",
              "SAM3 segments the subject from a text prompt",
              "pymatting refines alpha at hair / fine edges",
              "Outputs: RGBA + white & black composites"],
  "chips": ["RGBA","white BG","black BG"]
}'
```
Fields: `title` (required), `subtitle`, `bullets` (3–7 short `A -> B` lines), `chips` (optional tags),
`footnote` (optional grey line). Output is JPEG q88 — name it `<same_snake_case_basename>.jpg`.

**Background (combines the card with "the actual nodes"):**
- By default it **auto-draws the node graph** from the sibling `<name>.json` (faded LiteGraph-style
  boxes + wires behind the card) — so write the `.json` first, then run the generator pointed at
  `<name>.jpg` and it just works. Override with `"workflow": "path.json"`.
- Provide `"background": "path.jpg"` to use a **screenshot or result image** instead (cover-fit +
  darkened). This wins over the node graph — use it when you have a nice output/result render.
- Set `"no_graph": true` for a pure clean card (no background).

Rules baked in (don't re-derive): **no emoji glyphs** (sanitized), `.jpg` only, 1180×680, JPEG q88,
left scrim keeps text readable over any background. Always `<same_snake_case_basename>.jpg`.

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

## Testing protocol — do this before shipping any template

**Never ask the user to test a template. Run it yourself end-to-end. Every error below has already been hit; know them before you start.**

### Step 1 — Audit widget values against live schemas

```bash
python3 tools/audit_workflows.py --server http://127.0.0.1:8188
```

If issues are reported, run:
```bash
python3 tools/audit_workflows.py --rebuild --server http://127.0.0.1:8188
```

`--rebuild` resets ALL widget values to schema defaults. Always safe for templates.
`--fix` only patches out-of-range values; use it only when you need to preserve specific values.

**After any rebuild, audit again (`--rebuild` is not idempotent on the first run if schemas change):**
```bash
python3 tools/audit_workflows.py --server http://127.0.0.1:8188
# Must report: Total issues: 0
```

### Step 2 — Upload a test input and run end-to-end (NOT dry-run)

Dry-run only verifies node existence and widget count. It does NOT catch:
- Missing Python packages
- Wrong model file names
- Incompatible library versions
- `is_output_node` missing on terminal nodes
- Runtime errors of any kind

**Always run with a real input and wait for `status: success`.**

```bash
# 1. Upload test image (required — API validates LoadImage against known files)
python3 -c "
import urllib.request, json
with open('/opt/comfyui/stable/input/Agent_A_Horse_front_00001.png', 'rb') as f:
    img_data = f.read()
boundary = 'b123'
body = (f'--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"test.png\"\r\nContent-Type: image/png\r\n\r\n').encode() + img_data + f'\r\n--{boundary}--\r\n'.encode()
req = urllib.request.Request('http://127.0.0.1:8188/upload/image', data=body, headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
print(json.load(urllib.request.urlopen(req)))
"

# 2. Patch the workflow to use the uploaded image, run it
python3 -c "
import json, subprocess
wf = json.load(open('example_workflows/<name>.json'))
for node in wf['nodes']:
    if node['type'] == 'LoadImage':
        node['widgets_values'][0] = 'test.png'
    if node['type'] == 'BD_LoadMesh':
        # Point file_path at a real mesh; set model_file to '(none — use file_path)'
        node['widgets_values'][0] = '(none — use file_path)'
        node['widgets_values'][2] = '/opt/comfyui/stable/input/3d/Stanford_Bunny.stl'
json.dump(wf, open('/tmp/test_wf.json', 'w'), indent=2)
"
/opt/comfyui/stable/venv/bin/python /opt/comfyui/run_workflow.py \
    /tmp/test_wf.json --server http://127.0.0.1:8188 --timeout 300
# Must end with: status: success
```

### Step 3 — Verify output was actually written

```bash
ls -lt /srv/AI_Stuff/outputs/ | head -5
# Should show a fresh file timestamped within the last few minutes
```

A `status: success` with no output file means the workflow ran but the terminal node
isn't marked as an output node. Fix: add `is_output_node=True` to the node's schema.

---

## Known pitfalls and their fixes

### Widget value corruption — V3 COMBO type

**Symptom:** ComfyUI reports `pipeline_type: '49152' not in ['1024_cascade', '1536_cascade']` or similar — COMBO options getting integer/float values, and subsequent inputs getting COMBO strings.

**Root cause:** The ComfyUI V3 API represents COMBO inputs as `['COMBO', {options:[...]}]` (string key), not `[['opt1','opt2'], {}]` (list key, V1 format). Old audit tools and converters only detected V1 COMBOs, silently skipping V3 ones and shifting every subsequent widget value by +1 per skipped COMBO.

**Fix:** Run `--rebuild` which uses the corrected audit tool. The tool normalizes both formats via `_normalize_spec()`.

**Rule:** When building widget_values manually, iterate through `schema['input']['required']` then `schema['input']['optional']`, and for EACH input: if `spec[0]` is a list → V1 COMBO widget; if `spec[0] == 'COMBO'` → V3 COMBO widget (get options from `spec[1]['options']`); if `spec[0]` in `{STRING, INT, FLOAT, BOOLEAN, IMAGEUPLOAD}` → primitive widget; otherwise → skip (socket-only connection type). Then after any INT input whose name contains `"seed"`, insert one extra slot for `control_after_generate`.

### Widget value corruption — control_after_generate

**Symptom:** All widget values after a seed input are shifted by one (e.g. `max_tokens` receives `'1024_cascade'`, `ss_guidance_strength` receives `49152`).

**Root cause:** ComfyUI's frontend injects a virtual `control_after_generate` widget (options: `fixed`, `increment`, `decrement`, `randomize`) into the UI immediately after every INT input whose name contains `seed`. This virtual slot IS stored in `widgets_values` but is NOT present in the Python node schema (`/object_info`). Any tool that reads `widgets_values` and maps them to schema inputs must explicitly skip this slot after consuming each seed INT.

**Fix:** The audit tool (`tools/audit_workflows.py`) and `run_workflow.py` both implement this skip. When rebuilding manually, always insert `"fixed"` (or another valid option) at position `seed_index + 1` in `widgets_values`.

**Rule:** For every INT input with `"seed"` in its name → consume value, then consume one more value (control_after_generate) and discard/map it separately.

### LoadImage rejects filenames not in its option list

**Symptom:** API returns 400 with `image - Invalid image file: my_file.png` even though the file exists.

**Root cause:** ComfyUI validates LoadImage's `image` widget against a live scan of the input directory. If the file was copied there directly (not uploaded through the API), it may not appear in the scan, or the file permissions differ.

**Fix:** Always upload via `POST /upload/image` before running the workflow. This both copies the file and registers it with ComfyUI.

### Terminal node rejects workflow — prompt_no_outputs

**Symptom:** API returns 400 or workflow validation fails with "prompt has no outputs" when BD_SaveBatch (or another custom save node) is the only terminal node.

**Root cause:** ComfyUI requires at least one node with `is_output_node=True` (or a node that has `OUTPUT_NODE = True` in V1). ComfyUI's built-in SaveImage has this; custom save nodes must declare it explicitly.

**Fix:** Add `is_output_node=True` to `io.Schema(...)` in the node's `define_schema()`. For V1 API: `OUTPUT_NODE = True` as a class attribute.

**Rule:** Every node that saves files to disk and appears as a terminal node in templates MUST have `is_output_node=True`.

### Missing packages in stable venv

**Symptom:** `ImportError: X not available` or `ModuleNotFoundError: No module named 'X'` during execution.

**Root cause:** The stable venv and dev venv are separate. Packages installed in dev are not automatically available in stable.

**Fix procedure:**
1. Check if it's in dev: `find /opt/comfyui/dev/venv/lib -name "X" -type d`
2. If found and it's a pure-Python package: `cp -r /opt/comfyui/dev/venv/lib/python3.12/site-packages/X /opt/comfyui/stable/venv/lib/python3.12/site-packages/`
3. If it's a compiled extension (`.so` files): install via pip instead — don't copy, the `.so` may be ABI-specific: `/opt/comfyui/stable/venv/bin/pip install X`
4. After install: verify with `/opt/comfyui/stable/venv/bin/python -c "import X; print('OK')"`
5. Verify PyTorch version unchanged: `/opt/comfyui/stable/venv/bin/python -c "import torch; print(torch.__version__)"` → must be `2.9.1+cu130`

**Packages required by BrainDead that are NOT in a stock ComfyUI install:**
- `moge` — install via: `pip install git+https://github.com/microsoft/MoGe.git --no-deps`
- `utils3d` — must install the git version, NOT PyPI (PyPI lacks `.pt` submodule): `pip install "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183" --no-deps`
- `pipeline` — git version: `pip install "pipeline @ git+https://github.com/EasternJournalist/pipeline.git@866f059d2a05cde05e4a52211ec5051fd5f276d6" --no-deps`
- `natten` — pip installable: `pip install natten`
- `pixal3d` — copy from dev venv (private/vendored package)

These are all listed in `requirements.txt`. ComfyUI Manager installs them automatically on node pack install.

### BiRefNet Config incompatibility with transformers >= 4.40

**Symptom:** `AttributeError: 'Config' object has no attribute 'get_text_config'` when loading RMBG-2.0.

**Root cause:** transformers >= 4.40 calls `get_text_config()` during `tie_weights()` on all `PreTrainedModel` submodules. BiRefNet's internal `Config` class (a plain Python dataclass used for training hyperparameters, not a transformers config) gets assigned as `self.config` inside backbone submodules, so transformers tries to call `get_text_config()` on it and fails.

**Fix:** The `get_pipeline()` function in `nodes/pixal3d/utils.py` automatically patches `birefnet.py` on disk on first use — adds a `get_text_config(self, decoder=False): return self` stub to the `Config` class. This is a one-time patch that persists.

If encountering this with a different model: add `get_text_config(self, decoder=False): return self` to the model's internal Config class.

### MoGe/model downloading fails — server has no HuggingFace access

**Symptom:** `Failed to resolve 'huggingface.co'` DNS error when loading MoGe model, even though the model is in the HF cache.

**Root cause:** `from_pretrained()` attempts a network HEAD request to check for model updates even when the model is cached locally.

**Fix:** `get_moge_model()` in `nodes/pixal3d/utils.py` tries `local_files_only=True` first, only falling back to network if the model is not cached. The model is pre-downloaded to `/srv/AI_Stuff/models/huggingface/hub/models--Ruicheng--moge-2-vitl/` (1.3GB).

---

## Checklist — before shipping a template

```
[ ] tools/audit_workflows.py reports 0 issues
[ ] Tested end-to-end (not --dry-run) with status: success
[ ] Output file confirmed in /srv/AI_Stuff/outputs/ or ComfyUI output dir
[ ] All required packages installed in stable venv (checked with import test)
[ ] Terminal node has is_output_node=True (or is a built-in output node)
[ ] LoadImage uses a filename that exists in /object_info (uploaded via API)
[ ] Model files in character_consistency or other model-dependent templates
    point at actually installed models (check /srv/AI_Stuff/models/)
[ ] Thumbnail is <name>.jpg (same basename as .json, no -1 suffix, snake_case)
[ ] Synced to stable: rsync example_workflows/*.json to stable
[ ] ComfyUI stable restarted after sync
```

## Quick reference

| File / Dir | Purpose |
|------------|---------|
| `example_workflows/<name>.json` | Workflow (UI graph format), name = display title |
| `example_workflows/<name>.jpg` | Thumbnail (jpg, same basename, no `-1`) |
| `example_workflows/README.md` | Index + this convention |
| `tools/audit_workflows.py` | Widget value auditor + `--rebuild` / `--fix` mode |
| `/opt/comfyui/run_workflow.py` | Graph→API convert + run (no --dry-run for real testing) |
