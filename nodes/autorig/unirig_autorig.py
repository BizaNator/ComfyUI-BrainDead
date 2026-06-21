"""
BD_AutoRigUniRig — auto-rig using UniRig (SIGGRAPH 2025), wrapping the
sibling ComfyUI-UniRig pack.

UniRig is more general than Make-It-Animatable — handles humans, animals,
and arbitrary articulated objects. For our humanoid 27-char batch MIA is
preferred (faster + better humanoid quality), but UniRig is the fallback
for any char where MIA gives a bad skeleton.

Upstream: https://github.com/VAST-AI-Research/UniRig (MIT)
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


# Same sibling-pack discovery as the MIA wrapper.
_THIS = Path(__file__).resolve()
_PACK_ROOT = _THIS.parent.parent.parent
_CUSTOM_NODES_DIR = _PACK_ROOT.parent
_UNIRIG_PACK = _CUSTOM_NODES_DIR / "ComfyUI-UniRig"
_UNIRIG_LIB = _UNIRIG_PACK / "lib"


def _ensure_unirig_lib_on_path() -> bool:
    if not _UNIRIG_LIB.is_dir():
        return False
    if str(_UNIRIG_LIB) not in sys.path:
        sys.path.insert(0, str(_UNIRIG_LIB))
    if str(_UNIRIG_PACK) not in sys.path:
        sys.path.insert(0, str(_UNIRIG_PACK))
    return True


_UNIRIG_MODEL_CACHE: dict = {}
_UNIRIG_MODEL_LOCK = threading.Lock()


def _load_unirig_model(skeleton_template: str = "mixamo",
                        device: str = "auto"):
    """Load + memoize the UniRig skeleton + skinning models for the given
    template ('mixamo' for humanoids, 'articulationxl' for general)."""
    if not _ensure_unirig_lib_on_path():
        raise RuntimeError(
            "ComfyUI-UniRig pack is not installed in custom_nodes/. "
            "Install from https://github.com/PozzettiAndrea/ComfyUI-UniRig."
        )
    key = (skeleton_template, device)
    with _UNIRIG_MODEL_LOCK:
        if key in _UNIRIG_MODEL_CACHE:
            return _UNIRIG_MODEL_CACHE[key]

        # PozzettiAndrea's model_loaders.py provides load_unirig_models()
        from model_loaders import load_unirig_models  # type: ignore

        print(f"[BD_AutoRigUniRig] Loading UniRig model "
               f"(template={skeleton_template}, device={device}) ...")
        t0 = time.time()
        model = load_unirig_models(
            skeleton_template=skeleton_template,
            device=device,
        )
        print(f"[BD_AutoRigUniRig] Loaded in {time.time()-t0:.1f}s.")
        _UNIRIG_MODEL_CACHE[key] = model
        return model


def _run_unirig_inference(mesh, models, output_path: str,
                            skeleton_template: str) -> str:
    if not _ensure_unirig_lib_on_path():
        raise RuntimeError("ComfyUI-UniRig pack not on path.")
    # The PozzettiAndrea pack exposes the inference call inside auto_rig.py
    # of its nodes pkg. Use a thin shim.
    nodes_pkg = _UNIRIG_PACK / "nodes"
    if str(nodes_pkg) not in sys.path:
        sys.path.insert(0, str(nodes_pkg))
    from auto_rig import UniRigAutoRig  # type: ignore

    # Build the node instance and call its auto_rig() method. UniRig's
    # auto_rig() returns a tuple — first element is the output FBX path.
    inst = UniRigAutoRig()
    result = inst.auto_rig(
        trimesh=mesh,
        model=models,
        skeleton_template=skeleton_template,
        fbx_name=Path(output_path).stem,
    )
    if isinstance(result, tuple) and result:
        return result[0]
    return str(result)


class BD_AutoRigUniRig(io.ComfyNode):
    """UniRig auto-rigger (more general than MIA; fallback for non-humanoid
    chars). Output is FBX with the chosen skeleton template's bone names."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_AutoRigUniRig",
            display_name="BD AutoRig (UniRig)",
            category="🧠BrainDead/AutoRig",
            description=(
                "UniRig (SIGGRAPH 2025) auto-rigger via the sibling "
                "ComfyUI-UniRig pack. More general than Make-It-Animatable — "
                "handles humans, animals, props.\n\n"
                "skeleton_template:\n"
                "  • mixamo — 52-bone humanoid (default for our pipeline; "
                "    chain with BD Mixamo → UEFN Rename for UEFN output)\n"
                "  • articulationxl — generic articulated objects (no remap "
                "    target; for non-humanoid props)\n"
            ),
            inputs=[
                TrimeshInput("mesh"),
                io.Combo.Input(
                    "skeleton_template",
                    options=["mixamo", "articulationxl"],
                    default="mixamo",
                ),
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
                    "remap_to_uefn",
                    default=True,
                    tooltip="Run the Mixamo → UEFN bone-name remap step after "
                            "rigging. Only meaningful when skeleton_template = "
                            "'mixamo'. Disabled automatically for "
                            "articulationxl.",
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
        skeleton_template: str = "mixamo",
        fbx_name: str = "",
        device: str = "auto",
        remap_to_uefn: bool = True,
    ) -> io.NodeOutput:
        out_dir = Path(folder_paths.get_output_directory())
        out_dir.mkdir(parents=True, exist_ok=True)

        if fbx_name:
            stem = fbx_name
        else:
            stem = f"rigged_unirig_{time.strftime('%Y%m%d_%H%M%S')}"
        out_fbx = out_dir / f"{stem}_{skeleton_template}.fbx"

        model = _load_unirig_model(skeleton_template=skeleton_template,
                                     device=device)
        t0 = time.time()
        result = _run_unirig_inference(
            mesh=mesh,
            models=model,
            output_path=str(out_fbx),
            skeleton_template=skeleton_template,
        )
        print(f"[BD_AutoRigUniRig] Inference complete in "
               f"{time.time()-t0:.2f}s → {result}")

        if not os.path.exists(result):
            raise RuntimeError(
                f"UniRig inference returned {result} but file missing")

        if not (remap_to_uefn and skeleton_template == "mixamo"):
            return io.NodeOutput(str(result))

        from .bone_remap import BD_MixamoToUEFN
        remap_out = BD_MixamoToUEFN.execute(
            input_fbx=str(result),
            output_name=f"{stem}_uefn",
        )
        uefn_fbx = remap_out.result[0] if hasattr(remap_out, "result") else str(remap_out)
        return io.NodeOutput(str(uefn_fbx))


UNIRIG_V3_NODES = [BD_AutoRigUniRig]
UNIRIG_NODES = {"BD_AutoRigUniRig": BD_AutoRigUniRig}
UNIRIG_DISPLAY_NAMES = {"BD_AutoRigUniRig": "BD AutoRig (UniRig)"}
