# BD GLSL Batch (iterate uniforms)

Run a GLSL shader N times with different uniform values in a single workflow execution, batching the outputs.

## Overview

ComfyUI's standard GLSL shader node runs once per workflow execution. To produce N variants (e.g., 4 skin tones), you'd normally need N separate workflow runs — burning workflow setup overhead + cache invalidation each time.

`BD_GLSLBatch` wraps ComfyUI's internal `_render_shader_batch()` so you can specify per-iteration overrides for any float or int uniform, run the shader N times, and get back batched fc0/fc1/fc2/fc3 tensors — all in one execution.

Iteration syntax follows the BrainDead convention (multiline string, one item per newline) — matches `BD_SAM3MultiPrompt`, `BD_PromptIterator`, `BD_PartsExport`.

## Inputs

| Input | Type | Description |
|-------|------|-------------|
| `fragment_shader` | STRING (multiline) | The fragment shader source code (GLSL ES 3.0). |
| `u_image0` ... `u_image4` | IMAGE (optional) | Bound to `sampler2D u_image0` etc. If the input is a batch, frame `i` is used for iteration `i`. Single images broadcast across all iterations. |
| `floats` | STRING (multiline) | Base float uniform values, one per line: `u_float0=0.48`. Indexes 0..19. Missing slots default to 0.0. |
| `ints` | STRING (multiline) | Base int uniform values, one per line: `u_int0=255`. Indexes 0..12. Missing slots default to 0. |
| `vary_ints` | STRING (multiline) | Per-iteration int overrides. Format: `u_intN=v1,v2,v3,...` (one uniform per line). Single-value lines are constants. |
| `vary_floats` | STRING (multiline) | Per-iteration float overrides. Same format. |
| `iteration_names` | STRING (multiline) | One name per iteration, for downstream filename use. Defaults to `1`, `2`, ... if empty. |
| `iterations_override` | INT (optional) | If > 0, force this iteration count. 0 = auto-detect from the longest vary list. |
| `width` / `height` | INT (optional) | Output dimensions. 0 = use `u_image0`'s dimensions. |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `fc0_batch` | IMAGE batch (B=N, H, W, C) | All N iterations' fc0 outputs stacked along batch dim |
| `fc1_batch` | IMAGE batch | fc1 outputs (typically the SR Preview / tinted variant) |
| `fc2_batch` | IMAGE batch | fc2 outputs |
| `fc3_batch` | IMAGE batch | fc3 outputs |
| `iteration_names` | STRING | Newline-joined names, in order (route to BD_BulkSave for per-iteration filenames) |
| `iteration_count` | INT | N, useful for downstream node sizing |

## Vary Syntax

```
u_int3=255,240,160,140        ← iteration 1=255, 2=240, 3=160, 4=140
u_int4=255,190,110,190
u_float0=0.48                  ← single value = constant across iterations
```

**Iteration count rules:**
- Auto-detected as the **longest** multi-value list across all vary fields
- Single-value lines are broadcast (used as a constant for every iteration)
- If `iterations_override > 0`, that value wins
- Lines with fewer values than the iteration count pad with the last value

## Common Use: 4 Skin Tone Variants

For the BrainDead skin shader pipeline, produce all 4 skin-tone variants of `fc1` (SR Preview) in one shader call:

```
fragment_shader  = <paste the skin shader v2.20+>
u_image0         = character RGB + body mask
u_image1         = skin mask
u_image3         = u_image3 RGBA pack (shadow + overlays + lines)
u_image4         = mannequin

floats =
    u_float0=0.48
    u_float1=0.5
    u_float2=0.40
    u_float4=0.5
    u_float6=1.0
    u_float8=1.0
    u_float15=0.70
    u_float16=0.40
    u_float17=0.80

ints =
    u_int2=1
    u_int7=1

vary_ints =
    u_int3=255,240,160,140
    u_int4=255,190,110,190
    u_int5=255,150,80,130

iteration_names =
    light
    medium
    dark
    zombie
```

Output: 4-image batch in each of fc0/fc1/fc2/fc3, plus `iteration_names = "light\nmedium\ndark\nzombie"`.

Pipe `fc1_batch` into `BD_BulkSave` (or split with `ImageBatchSplit` → BD_SaveContext) using `iteration_names` to write `_light`, `_medium`, `_dark`, `_zombie` suffixes per file.

## Performance

- **Shader compilation:** happens once per `BD_GLSLBatch` call (not per iteration)
- **Workflow submission overhead:** eliminated (no re-submit per iteration)
- **Per-iteration cost:** roughly equivalent to running the standard GLSL node once
- **Typical speedup vs. 4× workflow re-runs:** 5-15× faster for tone-variant cases, depending on how much upstream work the workflow does

## Common Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| Output is single image, not batch | All vary lines have only 1 value | Use multiple comma-separated values, or set `iterations_override` |
| `fc1_batch` shape mismatch in downstream | Image inputs have different sizes | Set `width`/`height` explicitly, or pre-resize images |
| Wrong values per iteration | Vary lists have unequal lengths | All multi-value vary lines should have the same count |
| Node fails to load on ComfyUI restart | This won't happen — imports are deferred to execute() time | n/a |

## Notes

- This node wraps ComfyUI's internal `_render_shader_batch()` directly. The GL context is created and torn down per call (one call per iteration in this implementation) — a future optimization could batch all iterations into a single GL session for further speedup.
- Same `_render_shader_batch` is what the standard `GLSLShader` node uses — output should be **bit-identical** to running that node N times with the same uniforms.
- Boolean uniforms and curve LUTs are not exposed by this node (yet) — only image, int, and float varying. Add if needed.

## Relationship to BD Cache nodes

BD_GLSLBatch makes the upstream BD_CacheImage / BD_CacheMask nodes **largely redundant for the multi-tone use case**, because all 4 iterations now run inside ONE workflow execution — the expensive upstream work (SAM3, Lotus2, Qwen, Wan2) only runs once per character regardless of iteration count.

| Scenario | Are BD Cache nodes needed? |
|----------|----------------------------|
| Process all 4 tones in one workflow run (BD_GLSLBatch primary use case) | No — upstream runs once anyway |
| Re-submit workflow for same character with same seed (parameter tweak) | Yes — cache skips the expensive upstream |
| Iterate on the GLSL shader code, re-run repeatedly | Yes — same reason |
| Different view of same character (front vs profile) | Cache misses anyway (different inputs) |

Recommended: **keep cache nodes** for safety. They're effectively zero-cost when not hitting (just passthrough) and protect against accidental re-work when you re-submit during iteration/debugging.

## Pairs With

- **BD_SaveBatch** — wire `fc[0-3]_batch` → `images` and `iteration_names` → `labels` for per-tone file saves with `save_only` filtering.
