# ComfyUI-BrainDead Roadmap

Tracked work — features queued, design notes, and decisions captured outside of
session memory. When picking up a future task, start here, then verify against
current code with `git log`, `git status`, and the relevant module.

Last updated: 2026-05-07

---

## Parts pipeline — Get/Set node split

**Status:** queued. User asked for these explicitly so workflow logic can move
out of `BD_PartsBatchEdit` and into visible workflow wiring.

**Why:** `BD_PartsBatchEdit` does too much in one node — Qwen Edit Plus encoder,
ReferenceLatent append, ModelSamplingAuraFlow patch, CFGNorm patch, KSampler,
tonemap, alpha mode resolution, save metadata. Hard to swap models, debug per
part, or run alternative editing chains. Splitting it gives users full control
over the per-part edit while keeping single-execution semantics.

**Design — 5 new nodes:**

| Node | Purpose |
|------|---------|
| `BD_PartsGetBatch` | parts_bundle → image_batch (uniform-padded), tags STRING (newline), bbox_list, depth_batch (when present). One execution gives you all parts as parallel arrays for batched downstream processing. |
| `BD_PartsSetBatch` | parts_bundle + edited image_batch + tags STRING → mutated parts_bundle. Pairs with GetBatch — user wires their editing subgraph between the two. |
| `BD_PartsGetPart` | parts_bundle + tag → image, body_mask, x1/y1/x2/y2, depth_median. For multi-Run iterator chains where the user wants tight per-part wiring. |
| `BD_PartsSetPart` | parts_bundle + tag + image → mutated parts_bundle. Mutates in place; pairs with GetPart and an external loop driver. |
| `BD_PartsIterator` | parts_bundle → emits one (tag, image, body_mask, depth_median, current_index, total_count, is_last) per Run. Same multi-Run pattern as the deleted SeeThrough Iterator. |

**Workflow patterns:**

```
BATCH PATTERN (single execution, batched Qwen call):
  parts_bundle ──→ BD_PartsGetBatch
                     ├─ image_batch
                     ├─ tags (parallel, for label-into-prompt template)
                     └─ bboxes
                          ↓
                  [user's Qwen Image Edit batched subgraph]
                          ↓
                       edited_batch
                          ↓
  parts_bundle ──→ BD_PartsSetBatch
                     image_batch=edited_batch
                     ↓
                  rebuilt parts_bundle ──→ Compose / Export

ITERATOR PATTERN (multi-Run, per-part fine control):
  BD_PartsIterator ──→ tag, image, ... (one per queue Run)
                          ↓
                  [user's per-part edit subgraph]
                          ↓
                       new_image
                          ↓
  BD_PartsSetPart (parts, tag, new_image) → mutates bundle
                          ↓
  BD_PartsCompose (trigger=is_last) → final composite
```

**Open design questions:**

1. **Per-item conditioning in batch mode**: Qwen Image Edit Plus takes one prompt
   per batch call. To get `"rebuild this {tag} in same style"` per part, the
   conditioning needs item-aligned prompts. ComfyUI supports this via per-item
   conditioning batches but it's clunky to wire. Options:
   - Single template prompt applied to all (lose per-tag specificity)
   - Multiple TextEncodeQwenImageEditPlus nodes batched (workflow gets noisy)
   - Build a `BD_PartsPromptTemplate` helper that takes a template + tag list
     and emits a per-item conditioning batch
   - User decides per-workflow

2. **Padding shape for batch mode**: `image_batch` requires uniform shape.
   PartsBuilder already handles this via `square_pad=pad_to_square`. We can
   reuse that, or expose dimension control on `BD_PartsGetBatch` directly so
   user picks how to pad.

3. **Image_batch → tags alignment**: tags STRING parallel to image_batch order.
   When image_batch dims vary or are reordered downstream, alignment breaks.
   Document that user shouldn't reorder. Or emit a labels TENSOR / dict that
   binds tag to batch index more strictly.

**Implementation notes:**

- Reuse `PARTS_BUNDLE` type and `parts_types.py` helpers (`ensure_bundle`,
  `frame_size`, `empty_bundle`).
- Node files: `parts_get_batch.py`, `parts_set_batch.py`, `parts_get_part.py`,
  `parts_set_part.py`, `parts_iterator.py` in `nodes/segmentation/`.
- Iterator state can live in a module-level dict keyed by `workflow_id` (same
  pattern `BD_ForEachRun` uses in `nodes/prompt/__init__.py`).
- All Get/Set nodes should support time-based `fingerprint_inputs` for
  reliable per-Run mutation visibility (same pattern as `BD_SaveFile`).

**Migration:**

- `BD_PartsBatchEdit` stays — useful as a turnkey "all-in-one" for users who
  don't want to wire the Qwen subgraph themselves. Becomes the reference
  implementation of Get/SetBatch + the Qwen recipe. Document it as such.

---

## BD_CallSubgraphRepeat

**Status:** future research task. Lower priority than Get/Set split — the split
already gives 90% of the benefit (user-owned editing chain), this is the cherry
on top.

**Concept:** invoke a saved ComfyUI subgraph N times within one execution,
threading per-iteration inputs through it and collecting outputs as a batch.
Lets users define edit logic as a subgraph and `BD_CallSubgraphRepeat` invokes
it generically — no model-specific wrapper code needed.

**Use case:** chain `BD_PartsGetBatch` → `BD_CallSubgraphRepeat(subgraph=user_edit, iterate_over=image_batch + tags)` → `BD_PartsSetBatch` to run any per-part edit logic without writing a node.

**Research needed:**

- ComfyUI v3 subgraphs: do they expose a programmatic invocation API? Last
  checked, they're a UI-level grouping primitive, not callable from inside
  another node's `execute()`.
- Alternative: rgthree has Loop nodes (ForLoopOpen / ForLoopClose) that
  iterate an existing subgraph — could we reuse their pattern?
- Tradeoff: a generic subgraph caller is invasive (touches the executor); a
  specific batch driver (per-model) is simpler but less reusable.

**If feasible**, this likely supersedes any model-specific wrappers we'd build
later (Flux Kontext edit, etc.).

---

## Mannequin pass

**Status:** mentioned but never built. Originally planned for the
GLSL skin tinting pipeline.

**Concept:** for character workflows where in-game clothing-swap needs the
underlying skin to show through:

1. Pass 1: clothed character → BD_SAM3MultiPrompt → PARTS_BUNDLE → clothed_parts
2. Run Qwen Edit "remove all clothing" on the source → mannequin character
3. Pass 2: mannequin character → BD_SAM3MultiPrompt with `"skin"` prompt → mannequin_parts
4. Merge: mannequin's bare-skin parts get `depth_median += 0.5` (push back) and
   composited UNDER the clothed parts via BD_PartsCompose

**Why:** users tinting skin tones at runtime via GLSL shader need the bare body
visible behind clothing. Without this pass, clothed regions of the character
have no underlying skin pixels to tint.

**Likely shape:** `BD_PartsMergeMannequin(clothed_bundle, mannequin_bundle)` →
combined bundle with mannequin layered under. Could be a small node (~50 lines)
once Get/Set split is done — uses existing PARTS_BUNDLE primitives.

---

## BrainPed reintegration (deferred)

**Status:** stage 2.5 of BrainPed currently broken — it called the old
SeeThrough/PartsRemap26to11 path which is deleted. Left intentionally because
the user pivoted away from SeeThrough for stylized character processing.

**When picking up:** see `/home/home/.claude-code-shared/projects/-opt-comfyui/memory/project_brainped_integration.md` for the historical contract. The new shape should be:

1. BrainPed sends per-segment SAM3 prompts (from `config/segments/{Part}.json`)
2. ComfyUI runs the SAM3 + BD_Parts* pipeline
3. New `BD_PartsRemap*` adapter maps free-form tags into BrainPed's 11-part
   schema using crop_bounds for L/R disambiguation
4. Per-part PNG layers returned in BrainPed's expected format

The new adapter is NOT `BD_PartsRemap26to11` (deleted — that mapped SeeThrough's
specific 26-tag taxonomy). It's a fresh node that takes free-form tags + the
per-segment crop_bounds dict.

---

## Smaller queued items

- **Reinstate `auto_from_white_bg + true_inpaint` warning**: the modes are
  incompatible (true_inpaint output is full source crop, white-detection
  inverts the alpha). Add a tooltip warning + log line, do NOT auto-redirect
  (user explicitly forbade hidden auto-fixes).
- **PSD scale uniformity audit**: at high `composite_size` (e.g. 6144),
  user observed parts have non-uniform relative scale. Likely interaction
  between per-part Qwen output dims, `flatten_pad_factor` margin, and the
  uniform scale-to-bbox math. Fix by standardizing per-part canvas dims OR
  by enabling `auto_crop_to_content=True` paired with `flatten_pad_factor`.
- **PSD layer mask channels**: currently mask info goes in as separate hidden
  layers. Pytoshop supports `user_layer_mask` (`channel_id=-2`) for actual
  per-layer mask channels. Stretch goal — would let users right-click in PS
  to "use as layer mask" with one click.
- **Sapiens2 5b weights**: 83 GB, on-demand download via HF. User has 0.4b +
  1b for normal+seg+pose+pointmap downloaded. 5b is queued if quality demands.
- **Lotus-2 normal pipeline**: weights downloaded, node loader supports
  `task=normal`, but no PBR-derivation workflow built yet that uses
  `BD_Lotus2Predict (normal)` → `BD_DerivePBR.normal_map`. Worth wiring up
  as a workflow example in README.

---

## Reference

- Pipeline architecture details: `README.md` (Segmentation Nodes section)
- Node design rules: `CLAUDE.md` (V3 API, category emoji, type rename
  notes, output node patterns)
- Auto-memory: `/home/home/.claude-code-shared/projects/-opt-comfyui/memory/`
  - `project_braindead_segmentation_state.md` — current pipeline + quirks
  - `feedback_node_design_principles.md` — user's design preferences
  - `reference_paths_and_endpoints.md` — services, models, sudo perms
- Snapshot rule (production): per `/opt/comfyui/CLAUDE.md`, snapshot stable
  before pull/restart.
