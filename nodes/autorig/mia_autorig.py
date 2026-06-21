"""
BD_AutoRigMIA — fast humanoid auto-rigging using vendored Make-It-Animatable.

NO dependency on ComfyUI-UniRig pack. The MIA inference code lives in
<pack>/lib/autorig/, vendored from PozzettiAndrea's ComfyUI-UniRig
(MIT licensed). Model weights auto-download from HuggingFace
(jasongzy/Make-It-Animatable) on first use into
<ComfyUI>/models/autorig/mia/.

Workflow:
    Mesh (TRIMESH) → BD_AutoRigMIA → FBX path (rigged, Mixamo skeleton)
    Then chain through BD_MixamoToUEFN to rename to UEFN_Mannequin.

Upstream:
    Make-It-Animatable — https://github.com/jasongzy/Make-It-Animatable (MIT)
    MIA ComfyUI wrapper code vendored from
    https://github.com/PozzettiAndrea/ComfyUI-UniRig (MIT)
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


# ── Vendored MIA discovery ──────────────────────────────────────────────────
_THIS = Path(__file__).resolve()
_PACK_ROOT = _THIS.parent.parent.parent           # ComfyUI-BrainDead/
_VENDORED_LIB = _PACK_ROOT / "lib" / "autorig"    # vendored MIA inference


def _ensure_vendored_lib_on_path() -> None:
    """Add the pack root to sys.path so `from lib.autorig.mia_inference`
    resolves. Also make sure the autorig lib dir itself is reachable so
    its relative imports work."""
    lib_root = str(_VENDORED_LIB.parent)  # <pack>/lib
    if lib_root not in sys.path:
        sys.path.insert(0, lib_root)
    pack_str = str(_PACK_ROOT)
    if pack_str not in sys.path:
        sys.path.insert(0, pack_str)


# Per-process model cache so successive calls reuse loaded weights.
_MIA_MODEL_CACHE: dict = {}
_MIA_MODEL_LOCK = threading.Lock()


def _load_mia_model(device: str = "auto"):
    """Auto-download weights + load model. Memoized per device."""
    _ensure_vendored_lib_on_path()
    key = device
    with _MIA_MODEL_LOCK:
        if key in _MIA_MODEL_CACHE:
            return _MIA_MODEL_CACHE[key]
        # Lazy import — the vendored module pulls in PyTorch + MIA arch
        from autorig.mia_inference import (  # type: ignore
            ensure_mia_models, load_mia_models,
        )
        if not ensure_mia_models():
            raise RuntimeError(
                "Failed to download Make-It-Animatable model weights "
                "from HuggingFace. Check internet connectivity + write "
                "permissions on <ComfyUI>/models/autorig/."
            )
        print(f"[BD_AutoRigMIA] Loading MIA model (device={device}) ...")
        t0 = time.time()
        cache_to_gpu = device in ("auto", "cuda")
        model = load_mia_models(cache_to_gpu=cache_to_gpu)
        print(f"[BD_AutoRigMIA] Loaded in {time.time() - t0:.1f}s.")
        _MIA_MODEL_CACHE[key] = model
        return model


def _run_mia_inference(mesh, model, output_path: str, *,
                        no_fingers: bool, use_normal: bool,
                        reset_to_rest: bool) -> str:
    """Wrap the vendored run_mia_inference call."""
    _ensure_vendored_lib_on_path()
    from autorig.mia_inference import run_mia_inference  # type: ignore
    return run_mia_inference(
        mesh=mesh,
        models=model,
        output_path=output_path,
        no_fingers=no_fingers,
        use_normal=use_normal,
        reset_to_rest=reset_to_rest,
    )


class BD_AutoRigMIA(io.ComfyNode):
    """Fast humanoid auto-rigging via Make-It-Animatable (vendored).

    Trimesh in → rigged FBX path out, with optional inline Mixamo → UEFN
    bone-name remap. Per upstream: <1s/char on humanoid with warm model
    (first call pays HF download + model load, ~10–20s).

    Model weights auto-download from HuggingFace on first use.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_AutoRigMIA",
            display_name="BD AutoRig (Make-It-Animatable)",
            category="🧠BrainDead/AutoRig",
            description=(
                "Make-It-Animatable auto-rigger for humanoid meshes.\n\n"
                "Predicts skeleton joint positions and blend skinning "
                "weights from a Trimesh in one pass. Outputs a Mixamo-"
                "compatible rigged FBX (52 bones, or 22 if no_fingers=True).\n\n"
                "Model weights auto-download from HuggingFace on first "
                "use to <ComfyUI>/models/autorig/mia/. Subsequent calls "
                "reuse the cached weights. No dependency on ComfyUI-"
                "UniRig — inference code is vendored.\n\n"
                "Chain output through BD Mixamo → UEFN Rename for the "
                "canonical UEFN_Mannequin skeleton naming, or set "
                "remap_to_uefn=True (default) to do it inline."
            ),
            inputs=[
                TrimeshInput("mesh"),
                io.String.Input(
                    "fbx_name",
                    default="",
                    tooltip="Output filename stem (no extension). Empty = "
                            "timestamped.",
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
                    tooltip="Merge finger weights into the hand bone. "
                            "Recommended for the studio 27-char batch — "
                            "characters don't have animated fingers.",
                    optional=True,
                ),
                io.Boolean.Input(
                    "use_normal",
                    default=False,
                    tooltip="Use surface normals for tighter skinning. "
                            "Helps when limbs touch (arms-to-torso poses).",
                    optional=True,
                ),
                io.Boolean.Input(
                    "reset_to_rest",
                    default=True,
                    tooltip="Transform output mesh into T-pose rest "
                            "position. Required for downstream PoseFixer "
                            "to retarget cleanly.",
                    optional=True,
                ),
                io.Boolean.Input(
                    "remap_to_uefn",
                    default=True,
                    tooltip="Append the Mixamo → UEFN bone-name remap "
                            "step inline. Output FBX uses pelvis, "
                            "spine_01..05, upperarm_l, etc.",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="fbx_path"),
            ],
            is_output_node=True,
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

        stem = fbx_name or f"rigged_mia_{time.strftime('%Y%m%d_%H%M%S')}"
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
        print(f"[BD_AutoRigMIA] Inference complete in "
               f"{time.time() - t0:.2f}s → {result}")

        if not os.path.exists(result):
            raise RuntimeError(
                f"MIA inference returned {result} but file missing")

        if not remap_to_uefn:
            return io.NodeOutput(str(result))

        # Inline chain the bone-name remap
        from .bone_remap import BD_MixamoToUEFN
        remap_out = BD_MixamoToUEFN.execute(
            input_fbx=str(result),
            output_name=f"{stem}_uefn",
        )
        uefn_fbx = (remap_out.result[0]
                     if hasattr(remap_out, "result")
                     else str(remap_out))
        return io.NodeOutput(str(uefn_fbx))


MIA_V3_NODES = [BD_AutoRigMIA]
MIA_NODES = {"BD_AutoRigMIA": BD_AutoRigMIA}
MIA_DISPLAY_NAMES = {"BD_AutoRigMIA": "BD AutoRig (Make-It-Animatable)"}
