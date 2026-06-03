# BD Parts Batch Edit (Qwen)

Iterate every part in a PARTS_BUNDLE through Qwen Image Edit in a single workflow execution, with internal AuraFlow sampling patches and two inpaint strategies.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `parts` | PARTS_BUNDLE | Bundle from `BD_PartsBuilder` or `BD_PartsRefine`. Mutated in-place and passed through. |
| `model` | MODEL | Qwen Image Edit 2509 (or compatible) model. |
| `clip` | CLIP | CLIP text encoder for the model. |
| `vae` | VAE | VAE for encoding/decoding. |
| `positive_prompt` | STRING (multiline) | Edit instruction applied to every part. |
| `negative_prompt` | STRING (multiline) | Negative conditioning. |
| `seed` | INT | Base seed. Incremented per part for variety. |
| `inpaint_mode` | COMBO | `flatten_redraw` (composite on white, encode with Qwen Edit, latent upscale, Reinhard tonemap) or `true_inpaint` (source crop + enclosed hole detection, prefill, noise_mask). |
| `context_extend_factor` | FLOAT | CropAndStitch-style bbox expansion for surrounding context in `true_inpaint` mode. |
| `flatten_pad_factor` | FLOAT | White padding around part image before encoding. Default 1.25 (25% breathing room). |
| `alpha_after_edit` | COMBO | How to compute post-edit alpha: `auto_from_white_bg` (detect white pixels, default), `fill_holes`, `original_part`, `original_dilated`, `bbox_full`. |
| `tonemap_reinhard_multiplier` | FLOAT | Reinhard tonemap strength applied after generation. Default 2.0 (matches manual recipe). |
| `model_sampling_shift` | FLOAT | ModelSamplingAuraFlow shift. Default 3.0. |
| `cfg_norm_strength` | FLOAT | CFGNorm strength. Default 0.85. |
| `steps` | INT | Sampling steps. |
| `cfg` | FLOAT | CFG scale. Qwen Lightning recipe: 1.0. |
| `sampler_name` | COMBO | Sampler. Qwen Lightning recipe: `euler`. |
| `scheduler` | COMBO | Scheduler. Qwen Lightning recipe: `simple`. |
| `latent_upscale_factor` | FLOAT | Latent upscale factor applied before decode. Default 1.5. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `parts` | PARTS_BUNDLE | Mutated bundle with edited part images and stashed original alphas. |
| `summary` | STRING | Per-part edit results, timing, and any errors. |
| `image_batch` | IMAGE | All edited part images stacked as a batch for preview. |

## Inpaint modes

**`flatten_redraw`** — Clean full-part redraw:
1. Composite part on white background (+ `flatten_pad_factor` margin)
2. Encode with Qwen Image Edit
3. Latent upscale → decode
4. Reinhard tonemap
5. Detect alpha from white bg pixels

**`true_inpaint`** — Fill enclosed holes only:
1. Expand source crop by `context_extend_factor`
2. Detect enclosed holes (pixels inside the part bbox but outside the mask)
3. Prefill holes with nearby color
4. Generate with noise_mask limited to hole region
5. Keep all visible source pixels unchanged

## Usage

- Default settings match the manual Qwen Lightning recipe: cfg=1.0, euler sampler, simple scheduler, 4 steps, shift=3.0, cfg_norm=0.85, tonemap=2.0.
- `alpha_after_edit=auto_from_white_bg` works when `background=white` was set in `BD_PartsBuilder`. The node detects near-white pixels and removes them as background.
- Original alpha is stashed in the bundle before editing — `BD_PartsExport.save_masks` and `save_masked_pngs` use the pre-edit alpha automatically.
- All parts run inside ONE execute() call. No workflow re-submission overhead — typically 4–8x faster than running Qwen Edit per part in separate workflow runs.
