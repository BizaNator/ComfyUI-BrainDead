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
    try:
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump({"workflow": workflow, "prompt": prompt}, f)
        return sidecar_path
    except Exception as e:
        print(f"[BD Save] workflow sidecar failed for {filepath}: {e}", flush=True)
        return ""
