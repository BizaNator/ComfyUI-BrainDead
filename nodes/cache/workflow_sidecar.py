"""
Shared workflow-sidecar helper for BD save nodes.

Three save nodes (BD_SaveFile, BD_BulkSave, BD_SaveBatch) write plain PNGs with
NO embedded metadata (unlike ComfyUI's stock SaveImage/PreviewImage, which embed
the full workflow+prompt graph into every PNG's text chunks by default). That
keeps delivered images clean, but leaves no trace of how they were generated.

WORKFLOW_SIDECAR_INPUTS adds one opt-out toggle so each save node can instead
write a `<image>.json` file next to the image containing the workflow + prompt
graph — full traceability without touching the pixel file at all.

Import into a node's define_schema with:
    inputs=[..., *WORKFLOW_SIDECAR_INPUTS]
    hidden=[io.Hidden.extra_pnginfo, io.Hidden.prompt]
"""

from __future__ import annotations

import json
import os

from comfy_api.latest import io

EMBED_WORKFLOW_INPUT = io.Boolean.Input(
    "embed_workflow", default=False, optional=True,
    tooltip=(
        "Also write the workflow + prompt into the PNG's own text chunks, exactly as "
        "ComfyUI's stock SaveImage does, so the image opens by drag-and-drop. OFF by "
        "default: embedding a large graph into every delivered product is waste, and the "
        "sidecar already carries the same data. Turn it on for the one image per run you "
        "want to reopen. PNG only -- JPEG cannot carry text chunks."
    ),
)

WORKFLOW_SIDECAR_INPUTS: list = [
    io.Boolean.Input(
        "save_workflow_sidecar", default=True, optional=True,
        tooltip=(
            "Write a <filename>.json sidecar next to the saved image containing the "
            "workflow + prompt graph (same data ComfyUI's stock SaveImage embeds into "
            "the PNG itself). The image file is never touched — this only adds a "
            "companion .json. Set False to save with no metadata at all."
        ),
    ),
]


def write_workflow_sidecar(filepath: str, extra_pnginfo, prompt) -> str:
    """Write a `<filepath minus ext>.json` sidecar with the workflow + prompt graph.

    If that name would be `filepath` itself -- the caller passed a .json, as
    BD_PartsExport does with its manifest -- `_workflow` is appended instead, so
    the sidecar can never overwrite the file it belongs to.

    Returns the sidecar path on success, or "" if there was nothing to write
    (no extra_pnginfo/prompt available, e.g. running outside a normal queued
    execution) or the write failed.
    """
    if not filepath:
        return ""
    workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
    if not workflow and not prompt:
        return ""
    sidecar_path = filepath.rsplit(".", 1)[0] + ".json" if "." in filepath else filepath + ".json"
    if os.path.abspath(sidecar_path) == os.path.abspath(filepath):
        # The caller's own file is already a .json -- BD_PartsExport sits the
        # sidecar beside its manifest. Writing there would clobber it, or be
        # clobbered by it, depending on which write lands last.
        sidecar_path = filepath.rsplit(".", 1)[0] + "_workflow.json"
    try:
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump({"workflow": workflow, "prompt": prompt}, f)
        return sidecar_path
    except Exception as e:
        print(f"[BD Save] workflow sidecar failed for {filepath}: {e}", flush=True)
        return ""


def embed_workflow_chunks(extra_pnginfo, prompt):
    """A PngInfo carrying workflow+prompt, or None when there is nothing to embed.

    Pass as `pnginfo=` to PIL's PNG save. Returns None rather than an empty PngInfo
    so callers can tell "nothing to embed" from "embedded".
    """
    try:
        from PIL.PngImagePlugin import PngInfo
    except Exception:
        return None
    wf = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
    if wf is None and prompt is None:
        return None
    meta = PngInfo()
    if wf is not None:
        meta.add_text("workflow", json.dumps(wf))
    if prompt is not None:
        meta.add_text("prompt", json.dumps(prompt))
    return meta
