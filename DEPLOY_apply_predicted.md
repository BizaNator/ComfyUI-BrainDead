# DEPLOY — opt-in MIA predicted-skeleton export (deploy-pending, owner-gated)

Commit: see `git log -1` (message: `feat(autorig): opt-in apply_predicted …`).
**DO NOT deploy until the owner greenlights.** The live pack runs on BRAINZ at
`/opt/comfyui` — nothing here has been pushed there.

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
the runner format, run through Blender 4.3 with the legacy and new scripts:
- legacy file vs new file, no flags: bone heads identical (max diff 0.0).
- `--apply_predicted` (+0.1 X offset joints): all 65 heads moved exactly
  (max err 1e-6).
- `--apply_predicted --reset_to_rest` (45° about-Y on `mixamorig:LeftArm`):
  true subtree matches the analytic 45°-about-pivot prediction (max err 1e-6);
  all other bones unmoved (max err 2e-6).
- `python -m py_compile` clean on all three touched files.

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
   the mesh's own pose (check hand joints sit inside the hand geometry).

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
