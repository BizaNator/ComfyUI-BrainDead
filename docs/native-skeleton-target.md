# Native Fortnite Skeleton

`BD_TargetFortniteSkeleton` / **BD Native Fortnite Skeleton** adds the measured Character Device completion stage to the existing autorig flow. The target defaults to `NATIVE_DEVICE`; `FAB_UEFN` is an explicit legacy option. It runs Blender 5.1.2 on CPU and allocates no model or GPU.

Connect a rigged `.fbx` or `.blend` path with canonical bone names to `source_asset`. Each queue run re-reads the files, so modifying an asset at the same path cannot reuse an earlier cached result. The node creates a unique output folder, reposes Basis and every morph with existing weights and full reference matrices, exports the complete rig, and reimports its FBXs for checks. It returns full FBX path, editable blend path and conversion-report path.

For full characters, leave **require_fingers** on: all thirty phalanges must already carry weights. Missing weights stop the job before export. Disable this only for standalone parts such as a head, torso or feet. The node does not generate absent weights, repair UVs, or retarget animation clips.

The studio defaults resolve `B:\Brains\Tools\BrainDeadBlender\scripts\utils\convert_skeleton_target.py` and the measured reference under `B:\Brains\Characters\_uefn_reference\native_contracts\codex_device_v01\` (NAS mount equivalents on BRAINZ). Inputs allow overriding the script, FBX/provenance pair and Blender executable when deployed elsewhere. Native provenance must match the target asset and FBX hash. Jobs never overwrite an earlier bundle.

Existing `BD_AutoRigUEFN` keeps its node ID and behavior, and is labeled **Fab UEFN Skeleton** to identify its actual reference. Add the native node after that stage. It can fail on an older rig's missing finger weights; that failure is the intended signal to repair or replace its hands before native delivery.

Offline validation executed the real node class through the installed ComfyUI V3 API and real CPU Blender, producing a named native FBX with eight morphs and all thirty finger weights. A deliberately removed finger-weight group fails without exporting. This does not establish live workflow loading or Device/NPC playback. The running ComfyUI service was not restarted or reloaded; comfy-lead owns safe activation.

Run `python3 -m unittest discover -s tests -v` for validation. The two Blender integration tests run when the studio reference bundle and reviewed v02 are available; otherwise they skip explicitly. `BDB_NATIVE_TEST_FBX`, `BDB_NATIVE_TEST_BLEND` and `BDB_NATIVE_TEST_BLENDER` can select equivalent test inputs. The package does not bundle the separate BrainDead Blender converter or proprietary reference assets; configure those paths before using the node outside the studio deployment.
