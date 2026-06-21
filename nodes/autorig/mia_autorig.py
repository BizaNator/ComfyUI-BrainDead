"""
BD_AutoRigMIA — One-shot auto-rigger using Make-It-Animatable.

Wraps the sibling pack `ComfyUI-UniRig`'s `mia_inference.run_mia_inference()`
with a single-node convenience interface: takes a trimesh, returns a path to a
rigged FBX (Mixamo skeleton). Optionally chains BD_MixamoToUEFN immediately so
the output is already UEFN-named.

Use this when you have a single mesh and want a UEFN-ready rig in one step
with no separate model-loading wire-up.

For workflow flexibility (re-use a loaded model across many meshes) use
the upstream MIALoadModel + MIAAutoRig nodes from the UniRig pack and feed
the FBX into BD_MixamoToUEFN.

Upstream:
  Make-It-Animatable — https://github.com/jasongzy/Make-It-Animatable (MIT)
  ComfyUI-UniRig    — https://github.com/PozzettiAndrea/ComfyUI-UniRig
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional

import folder_paths
from comfy_api.latest import io

from ..mesh.types import TrimeshInput
from ..blender.base import BlenderNodeMixin


# ── Sibling pack discovery (ComfyUI-UniRig provides MIA inference) ──────────
_THIS = Path(__file__).resolve()
_PACK_ROOT = _THIS.parent.parent.parent           # ComfyUI-BrainDead/
_CUSTOM_NODES_DIR = _PACK_ROOT.parent             # ComfyUI/custom_nodes/
_UNIRIG_PACK = _CUSTOM_NODES_DIR / "ComfyUI-UniRig"
_UNIRIG_LIB = _UNIRIG_PACK / "lib"


def _ensure_unirig_lib_on_path() -> bool:
    """Add the sibling UniRig lib dir to sys.path if present. Returns True on
    success, False if the pack isn't installed."""
    if not _UNIRIG_LIB.is_dir():
        return False
    if str(_UNIRIG_LIB) not in sys.path:
        sys.path.insert(0, str(_UNIRIG_LIB))
    if str(_UNIRIG_PACK) not in sys.path:
        sys.path.insert(0, str(_UNIRIG_PACK))
    return True


# Cache the loaded MIA model across invocations so we don't pay HF-load cost
# every call.
_MIA_MODEL_CACHE: dict = {}
_MIA_MODEL_LOCK = threading.Lock()


def _load_mia_model(device: str = "auto"):
    """Load the Make-It-Animatable model and memoize it process-wide."""
    if not _ensure_unirig_lib_on_path():
        raise RuntimeError(
            "ComfyUI-UniRig pack is not installed in custom_nodes/. "
            "Install from https://github.com/PozzettiAndrea/ComfyUI-UniRig "
            "or use the upstream MIA nodes directly."
        )

    key = device
    with _MIA_MODEL_LOCK:
        if key in _MIA_MODEL_CACHE:
            return _MIA_MODEL_CACHE[key]

        # Import lazily so a fresh ComfyUI install doesn't crash at module
        # load time when the sibling pack hasn't been added yet.
        from mia_inference import load_mia_models  # type: ignore

        print(f"[BD_AutoRigMIA] Loading MIA model on device={device}...")
        t0 = time.time()
        # PozzettiAndrea's load_mia_models() takes a device string.
        model = load_mia_models(device=device)
        print(f"[BD_AutoRigMIA] Model loaded in {time.time()-t0:.1f}s.")
        _MIA_MODEL_CACHE[key] = model
        return model


def _run_mia_inference(mesh, model, output_path: str, *,
                        no_fingers: bool, use_normal: bool,
                        reset_to_rest: bool) -> str:
    """Thin wrapper around the sibling pack's run_mia_inference()."""
    if not _ensure_unirig_lib_on_path():
        raise RuntimeError("ComfyUI-UniRig pack not on path.")
    from mia_inference import run_mia_inference  # type: ignore
    return run_mia_inference(
        mesh=mesh,
        models=model,
        output_path=output_path,
        no_fingers=no_fingers,
        use_normal=use_normal,
        reset_to_rest=reset_to_rest,
    )


class BD_AutoRigMIA(io.ComfyNode):
    """One-shot Make-It-Animatable auto-rigger. Trimesh in → rigged FBX path
    out. Auto-loads + memoizes the MIA model. Optionally appends a UEFN
    bone-name remap so the output is UEFN-ready.

    Per upstream: <1s on humanoid characters with a warm model. First call
    pays HF download + model load (~10-20s)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_AutoRigMIA",
            display_name="BD AutoRig (Make-It-Animatable)",
            category="🧠BrainDead/AutoRig",
            description=(
                "Convenience wrapper: takes a trimesh, calls Make-It-Animatable "
                "inference (loaded from sibling ComfyUI-UniRig pack), and "
                "returns a path to a rigged FBX. Optionally also runs the "
                "Mixamo → UEFN bone-name remap so downstream UE imports see "
                "the canonical skeleton convention.\n\n"
                "MIA is humanoid-only and outputs a Mixamo skeleton "
                "(52 bones with fingers — or 22 bones if no_fingers=True). "
                "For non-humanoid meshes use BD AutoRig (UniRig)."
            ),
            inputs=[
                TrimeshInput("mesh"),
                io.String.Input(
                    "fbx_name",
                    default="",
                    tooltip="Output filename (no extension). Empty = timestamped.",
                    optional=True,
                ),
                io.Combo.Input(
                    "device",
                    options=["auto", "cuda", "cpu"],
                    default="auto",
                    optional=True,
                ),
                io.Boolean.Input(
                    "no_fingers",
                    default=True,
                    tooltip="Merge finger weights into the hand bone. Recommended "
                            "for our 25-head pipeline — the studio characters "
                            "don't have animated fingers.",
                    optional=True,
                ),
                io.Boolean.Input(
                    "use_normal",
                    default=False,
                    tooltip="Use surface normals for tighter skinning weights. "
                            "Helps when limbs touch (arms-to-torso pose).",
                    optional=True,
                ),
                io.Boolean.Input(
                    "reset_to_rest",
                    default=True,
                    tooltip="Transform output mesh into T-pose rest position. "
                            "Required for downstream PoseFixer to retarget cleanly.",
                    optional=True,
                ),
                io.Boolean.Input(
                    "remap_to_uefn",
                    default=True,
                    tooltip="Append the Mixamo → UEFN bone-name remap step so "
                            "the output FBX uses UEFN_Mannequin bone names "
                            "(pelvis, spine_01, upperarm_l, hand_l, ...).",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="fbx_path"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        fbx_name: str = "",
        device: str = "auto",
        no_fingers: bool = True,
        use_normal: bool = False,
        reset_to_rest: bool = True,
        remap_to_uefn: bool = True,
    ) -> io.NodeOutput:
        out_dir = Path(folder_paths.get_output_directory())
        out_dir.mkdir(parents=True, exist_ok=True)

        if fbx_name:
            stem = fbx_name
        else:
            stem = f"rigged_mia_{time.strftime('%Y%m%d_%H%M%S')}"
        mia_fbx = out_dir / f"{stem}_mixamo.fbx"

        model = _load_mia_model(device=device)
        t0 = time.time()
        result = _run_mia_inference(
            mesh=mesh,
            model=model,
            output_path=str(mia_fbx),
            no_fingers=no_fingers,
            use_normal=use_normal,
            reset_to_rest=reset_to_rest,
        )
        print(f"[BD_AutoRigMIA] Inference complete in {time.time()-t0:.2f}s → "
               f"{result}")

        if not os.path.exists(result):
            raise RuntimeError(f"MIA inference returned {result} but file missing")

        if not remap_to_uefn:
            return io.NodeOutput(str(result))

        # Chain the remap node inline
        from .bone_remap import BD_MixamoToUEFN
        remap_out = BD_MixamoToUEFN.execute(
            input_fbx=str(result),
            output_name=f"{stem}_uefn",
        )
        # NodeOutput is a tuple-ish wrapper; .result[0] holds the string
        uefn_fbx = remap_out.result[0] if hasattr(remap_out, "result") else str(remap_out)
        return io.NodeOutput(str(uefn_fbx))


MIA_V3_NODES = [BD_AutoRigMIA]
MIA_NODES = {"BD_AutoRigMIA": BD_AutoRigMIA}
MIA_DISPLAY_NAMES = {"BD_AutoRigMIA": "BD AutoRig (Make-It-Animatable)"}
