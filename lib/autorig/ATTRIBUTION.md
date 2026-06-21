# lib/autorig — Vendored Make-It-Animatable Inference

This directory vendors the Make-It-Animatable inference code so
`ComfyUI-BrainDead` has no runtime dependency on `ComfyUI-UniRig`.

## Sources

- **`mia/`** (model architecture + pipeline) — from
  [jasongzy/Make-It-Animatable](https://github.com/jasongzy/Make-It-Animatable),
  MIT licensed. The subset shipped here is what's needed for inference;
  training code is omitted.

- **`mia_inference.py`** — adapted from
  [PozzettiAndrea/ComfyUI-UniRig/lib/mia_inference.py](https://github.com/PozzettiAndrea/ComfyUI-UniRig),
  MIT licensed. Changes:
    - `MIA_MODELS_DIR` repointed to `<ComfyUI>/models/autorig/mia/`
      following the BrainDead model-folder convention
    - `_export_mia_fbx` Blender discovery uses our pack's
      `nodes/blender/base.py::find_blender()` instead of the UniRig
      bundled Blender
    - Headless / non-ComfyUI fallback for model dir

- **`blender_export_fbx.py`** — vendored verbatim from PozzettiAndrea's
  pack, runs inside Blender to convert MIA inference output into an FBX
  with a posed armature.

- **`model_cache.py`** — vendored verbatim. Caches loaded MIA models per
  process so repeated inference calls don't reload from disk.

## License

All vendored code is MIT licensed. See the upstream repositories above
for the original LICENSE files.

## How to update

When upstream MIA gets fixes that we care about:

```bash
# from ComfyUI-BrainDead/
rm -rf lib/autorig/mia lib/autorig/mia_inference.py lib/autorig/blender_export_fbx.py lib/autorig/model_cache.py
scp -r home@brainz:/opt/comfyui/dev/custom_nodes/ComfyUI-UniRig/lib/mia lib/autorig/mia
scp home@brainz:/opt/comfyui/dev/custom_nodes/ComfyUI-UniRig/lib/{mia_inference.py,blender_export_fbx.py,model_cache.py} lib/autorig/
# then re-apply the path adjustments in mia_inference.py (see top of file)
```

Then `git diff` to confirm only the intended changes were re-applied.
