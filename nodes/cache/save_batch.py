"""
BD_SaveBatch — save each frame of an IMAGE batch as a separate file using a SaveContext.

Companion to BD_GLSLBatch (which produces image-batch outputs, one frame per iteration).
Given a batch of N images and a list of N labels, writes N files using the context's
filename template with each label as the suffix.

Compared to BD_BulkSave (one labels-per-slot, N slots wired): BD_SaveBatch is one slot
with one batch tensor, much cleaner when you already have a batch in hand.

Alpha features
--------------
save_alpha_separately:
    When the image has an alpha channel (C==4), also saves the alpha as a standalone
    greyscale PNG alongside every main file (suffix + "_alpha"). Useful for importing
    the alpha mask separately in DCC tools or for manual inspection.

alpha_mask (MASK, optional) + invert_alpha:
    If wired, replaces (or adds) the alpha channel of the saved image with this mask
    before writing. Black=transparent, white=opaque. Use invert_alpha to flip.
    Combined with save_alpha_separately you get both:
      • an RGBA PNG with the mask baked in as transparency
      • a separate greyscale _alpha.png of just that mask
    This is the "save with transparency" path — the source image stays as-is
    upstream; only the saved file has the transparency applied.
"""

import os
import torch

from comfy_api.latest import io
from .save_context import resolve_context_path, get_context, auto_pick_context
from .file_ops import BD_SaveFile  # for _detect_type_and_save
from .alpha_save import (
    ALPHA_SAVE_INPUTS,
    get_frame_mask,
    apply_alpha_to_frame,
    save_alpha_alongside,
)



class BD_SaveBatch(io.ComfyNode):
    """Save each frame of an IMAGE batch as a separate file using BD_SaveContext."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_SaveBatch",
            display_name="BD Save Batch",
            category="🧠BrainDead/Cache",
            description=(
                "Save each frame of an IMAGE batch as a separate file. "
                "Wire a batch (B>=1) and a labels list — one file per frame, "
                "with the corresponding label as the suffix.\n\n"
                "Pairs well with BD_GLSLBatch: feed fc1_batch + iteration_names "
                "to produce 4 distinct files (e.g. _light, _medium, _dark, _zombie) "
                "in one node, one wire."
            ),
            inputs=[
                io.Image.Input("images",
                               tooltip="IMAGE batch tensor (B, H, W, C). Each frame is saved as a separate file."),
                io.String.Input("labels", multiline=True,
                                default="light\nmedium\ndark\nzombie",
                                tooltip="One label per line, aligned with batch frames. "
                                        "If the labels list has FEWER entries than the batch size, "
                                        "remaining frames save with empty suffix (or numeric fallback). "
                                        "If MORE labels than frames, extras are ignored."),
                io.String.Input("save_only", default="", optional=True,
                                tooltip="Comma-separated FILTER of which frames to SAVE TO DISK. Two formats accepted:\n"
                                        "  Indices: '0,3' = save only frames 0 and 3 (skip 1, 2)\n"
                                        "  Labels:  'light,zombie' = match labels list, save those\n"
                                        "Mix is fine: '0,zombie'. Empty (default) = save ALL frames.\n"
                                        "Use to avoid duplicate copies of tone-independent outputs "
                                        "(e.g. fc3 only needs index 0; fc0/fc2 only need indices 0 and 3)."),
                io.String.Input("pass_only", default="", optional=True,
                                tooltip="Comma-separated FILTER of which frames to PASS DOWNSTREAM via the "
                                        "passed_images output. Same syntax as save_only — indices, label names, "
                                        "or mixed.\n"
                                        "Empty (default) = pass ALL frames downstream.\n"
                                        "Independent from save_only — you can SAVE different frames than you PASS.\n"
                                        "Example: save_only='light,zombie' + pass_only='zombie' = saves 2 to disk, "
                                        "but only the zombie frame flows downstream for post-processing."),
                io.String.Input("label_prefix", default="_", optional=True,
                                tooltip="Prepended to each label before becoming the suffix. "
                                        "Default '_' so label='light' → suffix='_light'."),
                io.String.Input("context_id", default="", optional=True,
                                tooltip="Match a BD_SaveContext id. Empty + exactly one context registered = auto-pick."),
                io.Combo.Input("format", options=["png", "jpg", "webp"], default="png", optional=True,
                               tooltip="Output file format."),
                io.Boolean.Input("skip_if_exists", default=False, optional=True,
                                 tooltip="If True, don't overwrite existing files (reports their path)."),
                io.String.Input("custom_vars", multiline=True, default="", optional=True,
                                tooltip="Extra context variables layered over the saved batch. "
                                        "One per line as key=value. Example:\n  subfolder=SR\n  pass=4k"),
                *ALPHA_SAVE_INPUTS,
            ],
            outputs=[
                io.Int.Output(display_name="saved_count",
                              tooltip="Number of files successfully written."),
                io.String.Output(display_name="saved_paths",
                                 tooltip="Newline-joined list of absolute paths of saved files, in batch order."),
                io.Image.Output(display_name="passed_images",
                                tooltip="IMAGE batch containing only the frames matching pass_only filter "
                                        "(or the full input batch if pass_only is empty). Wire downstream for "
                                        "post-processing of selected frames (e.g. one tone to parts compose)."),
                io.String.Output(display_name="passed_labels",
                                 tooltip="Newline-joined labels of the frames passed downstream, in order."),
                io.Image.Output(display_name="preview_images",
                                tooltip="UNFILTERED passthrough of the full input batch (all N frames). "
                                        "Useful for wiring to a PreviewImage node during testing to visually "
                                        "verify the GLSL output before saving. Disconnect or bypass the "
                                        "downstream preview node in production. Independent from save_only/pass_only."),
                io.String.Output(display_name="status",
                                 tooltip="Human-readable summary of what was saved/skipped/errored/passed."),
            ],
        )

    @staticmethod
    def _resolve_indices(filter_str: str, label_list: list[str], n_frames: int) -> list[int]:
        """Parse a comma-separated filter into a sorted list of valid frame indices.

        Tokens are either integers (0-based index) or label names from label_list.
        Empty/None filter returns ALL indices (0..n_frames-1).
        Invalid tokens are silently dropped.
        """
        if not filter_str or not filter_str.strip():
            return list(range(n_frames))
        keep: set[int] = set()
        for tok_raw in filter_str.split(","):
            tok = tok_raw.strip()
            if not tok:
                continue
            try:
                idx = int(tok)
                if 0 <= idx < n_frames:
                    keep.add(idx)
                continue
            except ValueError:
                pass
            if tok in label_list:
                keep.add(label_list.index(tok))
        return sorted(keep)

    @classmethod
    def execute(cls, images: torch.Tensor, labels: str,
                save_only: str = "", pass_only: str = "",
                label_prefix: str = "_", context_id: str = "",
                format: str = "png", skip_if_exists: bool = False,
                custom_vars: str = "",
                save_alpha_separately: bool = False,
                alpha_mask: torch.Tensor | None = None,
                invert_alpha: bool = False) -> io.NodeOutput:

        # Coerce input to (B, H, W, C)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            empty_batch = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            return io.NodeOutput(0, "", empty_batch, "",
                                 f"BD_SaveBatch: expected (B,H,W,C), got shape {tuple(images.shape)}")

        n_frames = images.shape[0]
        label_list = [l.strip() for l in (labels or "").strip().split("\n")]
        # Trim to batch size — extras ignored
        label_list = label_list[:n_frames]
        # Pad missing
        while len(label_list) < n_frames:
            label_list.append(f"frame{len(label_list):02d}")

        # Resolve both filters independently
        indices_to_save = cls._resolve_indices(save_only, label_list, n_frames)
        indices_to_pass = cls._resolve_indices(pass_only, label_list, n_frames)

        # Resolve context
        effective_ctx_id = context_id
        if not effective_ctx_id:
            picked = auto_pick_context()
            if picked is not None:
                effective_ctx_id = picked

        ctx_resolved = bool(effective_ctx_id and get_context(effective_ctx_id) is not None)
        if not ctx_resolved:
            return io.NodeOutput(
                0, "",
                f"BD_SaveBatch: no usable BD_SaveContext (context_id='{context_id}', auto-pick=None). "
                f"Add a BD_SaveContext upstream first."
            )

        ext = format if format != "jpg" else "jpg"

        saved_paths = []
        status_lines = []
        skipped = 0
        errors = 0

        H, W = images.shape[1], images.shape[2]

        for i in indices_to_save:
            label = label_list[i]
            suffix = (label_prefix or "") + label if label else ""
            try:
                filepath, rel_path = resolve_context_path(
                    effective_ctx_id, suffix, ext,
                    node_custom_vars=custom_vars,
                )
                if filepath:
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                if skip_if_exists and os.path.exists(filepath):
                    saved_paths.append(filepath)
                    status_lines.append(f"  frame={i} suffix='{suffix}' EXISTS → {rel_path}")
                    skipped += 1
                    continue

                # Slice this frame as its own (1, H, W, C) tensor
                single = images[i:i+1]

                frame_mask = get_frame_mask(alpha_mask, i, H, W)
                single_to_save = apply_alpha_to_frame(single, frame_mask, invert_alpha)

                final_path, data_type = BD_SaveFile._detect_type_and_save(single_to_save, filepath)
                saved_paths.append(final_path)
                rel_final = os.path.relpath(final_path).replace("\\", "/")
                alpha_note = ""

                if save_alpha_separately:
                    alpha_path, alpha_note = save_alpha_alongside(
                        single_to_save, frame_mask, invert_alpha,
                        final_path, effective_ctx_id, suffix, custom_vars,
                    )
                    if alpha_path:
                        saved_paths.append(alpha_path)

                status_lines.append(
                    f"  frame={i} suffix='{suffix}' {data_type} → {rel_final}{alpha_note}"
                )
            except Exception as e:
                errors += 1
                status_lines.append(f"  frame={i} suffix='{suffix}' ERROR: {e}")

        # Build the downstream pass batch
        if indices_to_pass:
            passed_images = images[indices_to_pass]  # shape: (len(pass), H, W, C)
            passed_labels = "\n".join(label_list[i] for i in indices_to_pass)
        else:
            # Edge case: pass_only had only invalid tokens. Pass empty placeholder.
            passed_images = torch.zeros((1, images.shape[1], images.shape[2], images.shape[3]),
                                       dtype=images.dtype, device=images.device)
            passed_labels = ""

        # Build a filter-status summary
        filter_notes = []
        if save_only and save_only.strip():
            filter_notes.append(f"save_only='{save_only}' → {len(indices_to_save)}/{n_frames}")
        if pass_only and pass_only.strip():
            filter_notes.append(f"pass_only='{pass_only}' → {len(indices_to_pass)}/{n_frames}")
        else:
            filter_notes.append(f"pass_only=all → {len(indices_to_pass)}/{n_frames}")
        filter_note = " (" + "; ".join(filter_notes) + ")" if filter_notes else ""

        header = (
            f"BD_SaveBatch: saved={len(saved_paths) - skipped} skipped={skipped} "
            f"errors={errors} passed={len(indices_to_pass)} "
            f"context='{effective_ctx_id}'{filter_note}"
        )
        status = header + "\n" + "\n".join(status_lines)
        print(f"[BD_SaveBatch] {header}", flush=True)

        # preview_images = unfiltered passthrough of the input batch
        preview_images = images

        return io.NodeOutput(
            len(saved_paths) - skipped,
            "\n".join(saved_paths),
            passed_images,
            passed_labels,
            preview_images,
            status,
        )


SAVE_BATCH_V3_NODES = [BD_SaveBatch]
SAVE_BATCH_NODES = {"BD_SaveBatch": BD_SaveBatch}
SAVE_BATCH_DISPLAY_NAMES = {"BD_SaveBatch": "BD Save Batch"}
