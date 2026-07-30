# DEPLOY — opt-in MIA predicted-skeleton export (deploy-pending, owner-gated)

Commits: `c93a324` (initial opt-in feature) + follow-up frame-fit fix
(`fix(autorig): frame-fit apply_predicted joints into template world`).
The initial feature is deployed on BRAINZ; the frame-fit fix fixes smoke-2
(predicted joints landed ~h/2 low in MIA's normalized mesh-centered frame).
**DO NOT deploy anything further until the owner greenlights.**

## What changed (3 files)

- `lib/autorig/mia_inference.py`
  - `run_mia_inference(..., apply_predicted=False)` — new kwarg.
  - `_export_mia_fbx`: serializes MIA's predicted rest→current pose to
    `pose.bin` + `pose_path/pose_shape` in the export JSON (same raw-float32
    format `braindead_blender/autorig_runner.py` already uses). Pose is
    serialized when `reset_to_rest` OR `apply_predicted` is on.
  - Appends `--apply_predicted` to the Blender CLI when set.
- `lib/autorig/blender/mia_export.py`
  - Parses `--apply_predicted`; loads `pose.bin` when present.
  - NEW (opt-in): `--apply_predicted` moves template bones to MIA's predicted
    joint positions; `--apply_predicted --reset_to_rest` additionally FK-poses
    the skeleton by the predicted ortho6d per-bone rotations (numpy mirror of
    `mia/utils.py::pose_local_to_global`) so the exported skeleton lands in
    the INPUT MESH'S OWN POSE (A-pose route — no downstream T→A march).
  - FRAME FIX (follow-up): predicted joints arrive in MIA's NORMALIZED,
    MESH-CENTERED frame (recentered, ~2-unit scale, Y-up). `_frame_fit`
    (Kabsch rotation + height-span-anchored scale + least-squares
    translation) maps them into the template armature's world frame before
    application — Kabsch rotation absorbs the axis swap and MIA's per-mesh
    hips-alignment rotation; height-anchored scale preserves predicted
    proportions (a plain argmin Kabsch scale drags limbs toward template
    proportions); pose rotations are conjugated by the fit rotation. Fit
    diagnostics (scale/translation/max residual) are printed per run.
  - UNCHANGED default: no flag → fixed T-pose template armature verbatim,
    `--reset_to_rest` alone stays the historical no-op (verified: pre/post
    probes of unflagged export are identical to 0.0).
- `nodes/autorig/mia_autorig.py`
  - `BD_AutoRigMIA` gains `apply_predicted` (BOOLEAN, default **off**),
    plumbed node → `_run_mia_inference` → `run_mia_inference` → CLI flag.
  - `reset_to_rest` tooltip updated to document the legacy no-op / interplay.

## Why

Forensic (run054): the workflow always returned the fixed T-pose template
skeleton (`lib/assets/animation_characters/mixamo.fbx` imported verbatim;
predicted joints loaded but never applied; `--reset_to_rest` parsed but never
used). That silently breaks weight transfer for non-T-posed inputs (run053:
whole hand collapsed onto `thumb_03`).

## Test evidence (harness: `B:\Brains\Characters\_autorig_runs\_mia_export_harness\`)

Synthetic `data.json`/`bw.bin`/`joints.bin`/`joints_tail.bin`/`pose.bin` per
the runner format, run through Blender 4.3 with the legacy and new scripts.
joints are fabricated in a TRUE MIA-normalized frame (recentered, 1.099x
normalized scale, Y-up) with 1.25x-lengthened arms — the exact frame that
smoked out the offset bug:
- legacy file vs new file, no flags: bone heads identical (max diff 0.0).
- `--apply_predicted`: frame fit recovered scale 0.9098 (true inverse of the
  1.099 normalization) and the ~h/2 re-grounding translation; legs land
  1.6 mm from template positions, skeleton height exact (ratio 1.0009),
  LeftArm head z 1.442 vs 1.441 template, and the predicted 1.25x arm
  proportions survive EXACTLY (all four arm segments ratio 1.250, leg
  segment 1.000).
- `--apply_predicted --reset_to_rest` (45deg about MIA-Y on
  `mixamorig:LeftArm`): rotation conjugates into world Z (axis Z component
  1.000), descendants rigid to 0.00000, pivot unmoved, non-descendants
  byte-identical to the unposed run.
- `python -m py_compile` clean on all touched files.

## Deploy steps (BRAINZ, when approved)

1. Pull this commit on BRAINZ into the pack install used by ComfyUI
   (the copy under `/opt/comfyui` — typically
   `/opt/comfyui/custom_nodes/ComfyUI-BrainDead`; confirm with
   `find /opt/comfyui -name mia_autorig.py`).
2. Restart the ComfyUI service on BRAINZ (node schemas + lib code are loaded
   at process start; a browser refresh is NOT enough).
3. Smoke test WITHOUT touching the driver's default: queue the example
   workflow `example_workflows/BD-autorig_mia_to_uefn.json` unchanged →
   output must match current behavior (fixed T-pose template skeleton).
4. A-pose route test: same workflow with `apply_predicted=true` +
   `reset_to_rest=true` on an A-posed GLB → exported skeleton should land in
   the mesh's own pose AND in the template-world frame — check the
   `frame fit` log line (scale should be near the character's inverse
   normalization, translation should re-ground by ~half the skeleton
   height) and that hand joints sit inside the hand geometry at the right
   height (arm joints ~shoulder height in world units, NOT ~0.5).

## Risks / notes

- The Blender addon side (`B:\Brains\Tools\BrainDeadBlender`,
  `_build_workflow`) does NOT pass `apply_predicted` yet — ComfyUI treats
  missing optional inputs as default (off), so existing driver runs are
  unaffected. Wiring the addon setting through is a separate, deliberate step.
- `pose` is rotation-only (ortho6d, no translation — matches the shipped MIA
  config `pose_mode="ortho6d"`); global translation is not modeled.
- FBX files embed creation timestamps, so "byte-identical" was verified
  semantically (bone head/tail arrays), not by raw bytes.
- The vendored copy in BrainDeadBlender (`braindead_blender/autorig_vendor/
  mia_export.py`) is an OLDER file that doesn't even parse these flags; it
  was not touched. If the local-venv path should get the same capability,
  port separately.
