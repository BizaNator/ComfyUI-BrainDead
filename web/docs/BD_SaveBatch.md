# BD Save Batch

Save each frame of an IMAGE batch as a separate file using a BD_SaveContext, with independent save/pass/preview outputs and per-frame label filtering.

## Overview

Companion to `BD_GLSLBatch` (and any other node producing image batches). Given a batch of N images and a list of N labels, writes N files using the context's filename template with each label as the suffix. Has separate filter fields for what to **save to disk** vs what to **pass downstream** vs what to **preview** — three independent flows out of one node.

Compared to `BD_BulkSave` (one labels-per-slot, N slots wired): `BD_SaveBatch` is one slot with one batch tensor — much cleaner when you already have a batch in hand.

## Inputs

| Input | Type | Description |
|-------|------|-------------|
| `images` | IMAGE (batch) | Batch tensor `(B, H, W, C)`. Each frame becomes a separate file. |
| `labels` | STRING (multiline) | One label per line, aligned with batch frames. Typically wired from `BD_GLSLBatch.iteration_names`. |
| `save_only` | STRING | Comma-separated filter of which frames to **save to disk**. Indices (`0,3`), labels (`light,zombie`), or mixed (`0,zombie`). Empty = save all. |
| `pass_only` | STRING | Comma-separated filter of which frames to **pass to `passed_images` output**. Same syntax as `save_only`. Empty = pass all. Independent from save_only. |
| `label_prefix` | STRING | Prepended to each label before becoming the suffix. Default `_` → label `light` becomes suffix `_light`. Use to add channel name too: `_head_sr` → `_head_sr_light`. |
| `context_id` | STRING | Match a BD_SaveContext id. Empty + exactly one context registered = auto-pick. |
| `format` | COMBO | `png` / `jpg` / `webp`. |
| `skip_if_exists` | BOOL | If True, don't overwrite existing files. |
| `custom_vars` | STRING (multiline) | Extra context variables layered over the saved batch. |
| `save_alpha_separately` | BOOL | When ON and the saved image has an alpha channel, also writes the alpha as a standalone greyscale PNG alongside each main file (same suffix + `_alpha`). No effect on images without alpha. |
| `alpha_mask` | MASK (optional) | If wired, bakes this mask into the saved file's alpha channel (white=opaque, black=transparent) before writing. **Does not modify the upstream tensor** — the alpha is applied only in the file written to disk. Accepts batched masks — each frame gets its own slice. |
| `invert_alpha` | BOOL | Invert the alpha_mask before baking: transparent areas become opaque and vice versa. No effect when alpha_mask is not wired. |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `saved_count` | INT | Number of files actually written. |
| `saved_paths` | STRING | Newline-joined absolute paths of saved files. |
| `passed_images` | IMAGE (batch) | Frames matching `pass_only` filter. Wire downstream for post-processing of selected frames. |
| `passed_labels` | STRING | Newline-joined labels of the frames in `passed_images`, in order. |
| `preview_images` | IMAGE (batch) | **Unfiltered passthrough of the full input batch.** Wire to a PreviewImage node during testing to visually verify before saving. Disconnect in production. |
| `status` | STRING | Human-readable summary of what was saved/skipped/errored/passed. |

## Three independent flows

```
images IN ─┬─→ save_only filter ─→ files on disk (saved_count, saved_paths)
           ├─→ pass_only filter ─→ passed_images (for downstream post-processing)
           └─→ unfiltered ──────→ preview_images (for testing only)
```

## Filter Syntax

Comma-separated. Each token is parsed as an integer index first; if not numeric, matched against the labels list:

| Filter text | Means |
|-------------|-------|
| (empty) | all frames |
| `0` | frame at index 0 |
| `light` | frame whose label is `light` |
| `0,3` | indices 0 and 3 |
| `light,zombie` | by label |
| `0,zombie` | mixed — both refer to same frame here, deduplicated |
| `99,light` | `99` out of range silently dropped, `light` kept |

Invalid tokens are silently dropped — check the console log: `(save_only='light,xombi' → 1/4)` will tell you if a typo killed your filter.

## Common Wiring for Multi-Tone Skin Pipeline

Given `BD_GLSLBatch` outputs `fc0_batch`, `fc1_batch`, `fc2_batch`, `fc3_batch`, `iteration_names`:

```
BD_GLSLBatch.fc1_batch        → BD_SaveBatch #1.images         (SR — all 4 tones)
BD_GLSLBatch.iteration_names  → BD_SaveBatch #1.labels
                                 save_only=""    (empty, all)
                                 label_prefix="_head_sr"

BD_GLSLBatch.fc0_batch        → BD_SaveBatch #2.images         (ILM — light+zombie)
BD_GLSLBatch.iteration_names  → BD_SaveBatch #2.labels
                                 save_only="light,zombie"
                                 label_prefix="_head_ilm"

BD_GLSLBatch.fc2_batch        → BD_SaveBatch #3.images         (Unity layer — light+zombie)
BD_GLSLBatch.iteration_names  → BD_SaveBatch #3.labels
                                 save_only="light,zombie"
                                 label_prefix="_head_unityLayer"

BD_GLSLBatch.fc3_batch        → BD_SaveBatch #4.images         (Unreal albedo — light only)
BD_GLSLBatch.iteration_names  → BD_SaveBatch #4.labels
                                 save_only="light"
                                 label_prefix="_head_unrealAlbedo"
```

Total: 4 (fc1) + 2 (fc0) + 2 (fc2) + 1 (fc3) = **9 files** per workflow run.

## Pass-and-Save Pattern

```
save_only="light,zombie"     # save 2 files (light + zombie tone variants)
pass_only="zombie"           # pass ONLY the zombie tone downstream for post-processing
```

This lets you, e.g., save 2 ILM variants to disk AND send just the zombie ILM into a `BD_PartsCompose` node for further work — independent filters, one node call.

## Alpha Features

### Save alpha separately (`save_alpha_separately`)

When the image has 4 channels (RGBA), each main save also writes a greyscale `_alpha.png`:

```
{char}_head_v1_sr_light_profile_1.png        ← RGBA main file
{char}_head_v1_sr_light_profile_alpha_1.png  ← greyscale alpha
```

The alpha source is:
- the `alpha_mask` input (after `invert_alpha`) if wired
- the image's own A channel if no mask is wired

### Save with transparency (`alpha_mask`)

Wire any B&W mask to `alpha_mask` to bake it as the transparency layer in the saved PNG. The upstream `images` tensor is **not modified** — only the bytes written to disk carry the alpha.

| Pixel value | Meaning in saved PNG |
|---|---|
| White (1.0) | Opaque |
| Black (0.0) | Transparent |
| `invert_alpha=True` | Swap above |

Common use: saving a character's skin render with its segmentation mask as real PNG transparency, so it imports clean into Unreal/Unity without needing a separate import step.

```
GLSL output (RGB)  ──────────────────────→ BD_SaveBatch.images
skin segmentation mask  ─────────────────→ BD_SaveBatch.alpha_mask
                         save_alpha_separately=True
                         invert_alpha=False
```

Produces per-tone:
- `{char}_head_v1_sr_light.png` — RGBA PNG, mask as alpha (transparent BG)
- `{char}_head_v1_sr_light_alpha.png` — greyscale copy of just the alpha

## Notes

- `images` is plural to signal batch convention (ComfyUI IMAGE type can be single or batched; the plural name tells consumers to expect ≥1 frames).
- `preview_images` is always the full unfiltered batch — perfect for wiring to `PreviewImage` during testing, then disconnecting or bypassing the downstream preview node in production.
- If `save_only` is empty but `pass_only` is set, you save all files but only pass some downstream. If both are set, they're independent — overlap is allowed and harmless.
- Order of saved/passed frames is always ascending by batch index, regardless of the order tokens appear in the filter string.
- When `alpha_mask` is a single-frame mask `(1, H, W)`, it applies to all frames. A batched mask `(B, H, W)` applies per-frame.
