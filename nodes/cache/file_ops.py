"""
V3 API file operation nodes for saving and loading various data types.

BD_ClearCache - Clear cache files by pattern
BD_SaveFile - Save data to output folder
BD_LoadImage - Load image from file path
BD_LoadMesh - Load 3D mesh from file path
BD_LoadAudio - Load audio from file path
BD_LoadText - Load text from file path
"""

import os
from glob import glob

from comfy_api.latest import io

from ...utils.shared import (
    CACHE_DIR,
    OUTPUT_DIR,
    ImageSerializer,
    MaskSerializer,
    LatentSerializer,
    AudioSerializer,
    StringSerializer,
    PickleSerializer,
)

# Check for optional trimesh support
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


class BD_ClearCache(io.ComfyNode):
    """
    Clear cached files from BrainDead_Cache/ folder by name pattern.

    ONLY affects: BrainDead_Cache/ folder (cache nodes)
    DOES NOT affect: output/ folder (BD Save File)
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_ClearCache",
            display_name="BD Clear Cache",
            category="🧠BrainDead/Cache",
            description="Clear cached files from BrainDead_Cache folder by pattern.",
            is_output_node=True,
            inputs=[
                io.String.Input("pattern", default="*", tooltip="File pattern to match (e.g., 'image_*' or '*')"),
                io.Boolean.Input("confirm_clear", default=False, tooltip="Must be True to actually delete files"),
            ],
            outputs=[
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, pattern: str, confirm_clear: bool) -> io.NodeOutput:
        if not confirm_clear:
            return io.NodeOutput(f"Set confirm_clear=True to delete: {pattern}")

        if not os.path.exists(CACHE_DIR):
            return io.NodeOutput("Cache directory empty")

        search_pattern = os.path.join(CACHE_DIR, f"{pattern}*")
        matching_files = glob(search_pattern)

        if not matching_files:
            return io.NodeOutput(f"No files matching: {pattern}")

        deleted_count = 0
        deleted_size = 0
        for filepath in matching_files:
            try:
                file_size = os.path.getsize(filepath)
                os.remove(filepath)
                deleted_count += 1
                deleted_size += file_size
            except Exception as e:
                print(f"[BD Cache] Error deleting {filepath}: {e}")

        size_mb = deleted_size / (1024 * 1024)
        return io.NodeOutput(f"Deleted {deleted_count} files ({size_mb:.1f} MB)")


class BD_SaveFile(io.ComfyNode):
    """
    Save ANY data type to file in native format, output the file path.

    SAVES TO: ComfyUI output/ folder (NOT BrainDead_Cache)
    Files saved here are NOT affected by BD Clear Cache.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_SaveFile",
            display_name="BD Save File",
            category="🧠BrainDead/Cache",
            description="Save any data type to output folder. Supports IMAGE, MASK, AUDIO, LATENT, STRING, TRIMESH. "
                        "Optionally use a save context (from BD_SaveContext) for template-based paths.",
            is_output_node=True,
            inputs=[
                io.AnyType.Input("data"),
                io.String.Input("filename", default="saved_file",
                                tooltip="With context: overrides %name% / %filename% in template if set "
                                        "(non-'saved_file'). Without context: legacy filename."),
                io.Boolean.Input("skip_if_exists", default=True),
                io.String.Input("name_prefix", default="", optional=True,
                                tooltip="With context: exposed as %name_prefix% in template. Without context: "
                                        "legacy prefix prepended to filename."),
                io.String.Input("extension", default="", optional=True,
                                tooltip="Override auto-detected extension."),
                io.String.Input("context_id", default="", optional=True,
                                tooltip="If set AND a BD_SaveContext with this id is registered: path is resolved "
                                        "from the context's template + this node's suffix/filename/name_prefix. "
                                        "If empty AND exactly ONE context is registered, that one is auto-used. "
                                        "If empty AND zero or multiple contexts: legacy filename-based behavior."),
                io.String.Input("suffix", default="", optional=True,
                                tooltip="Per-save suffix appended after filename (e.g. '_albedo', '_shoes', '_head'). "
                                        "Wire from BD_ForEachRun.label or Iterator.tag for per-iteration filenames. "
                                        "With context: also exposed as %suffix% in the template. "
                                        "Without context: appended literally to filename — caller controls separator "
                                        "(convention: leading underscore)."),
                io.String.Input("custom_vars", multiline=True, default="", optional=True,
                                tooltip="Per-save extra variables, one per line as key=value. Layered ON TOP of "
                                        "the context's custom_vars (this node's keys override context for matching "
                                        "names). Examples:\n  subfolder=textures\n  materials=metal\n  pass=normal\n"
                                        "These become %subfolder%, %materials%, %pass% in the template. Empty values "
                                        "resolve cleanly (// → /). Undefined vars stay as %var% literals so you spot typos."),
            ],
            outputs=[
                io.AnyType.Output(display_name="data"),
                io.String.Output(display_name="file_path"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def _detect_type_and_save(cls, data, filepath: str) -> tuple[str, str]:
        import torch
        import numpy as np

        # IMAGE tensor
        if isinstance(data, torch.Tensor):
            shape = data.shape
            if len(shape) == 4 and shape[-1] in [3, 4]:
                if not filepath.endswith('.png'):
                    filepath = filepath.rsplit('.', 1)[0] + '.png' if '.' in filepath else filepath + '.png'
                ImageSerializer.save(filepath, data)
                return filepath, "IMAGE"
            elif len(shape) in [2, 3] and shape[-1] not in [3, 4]:
                if not filepath.endswith('.png'):
                    filepath = filepath.rsplit('.', 1)[0] + '.png' if '.' in filepath else filepath + '.png'
                MaskSerializer.save(filepath, data)
                return filepath, "MASK"

        # LATENT dict
        if isinstance(data, dict) and 'samples' in data:
            if not filepath.endswith('.latent'):
                filepath = filepath.rsplit('.', 1)[0] + '.latent' if '.' in filepath else filepath + '.latent'
            LatentSerializer.save(filepath, data)
            return filepath, "LATENT"

        # AUDIO dict
        if isinstance(data, dict) and 'waveform' in data and 'sample_rate' in data:
            if not filepath.endswith('.wav'):
                filepath = filepath.rsplit('.', 1)[0] + '.wav' if '.' in filepath else filepath + '.wav'
            AudioSerializer.save(filepath, data)
            return filepath, "AUDIO"

        # STRING
        if isinstance(data, str):
            if not filepath.endswith('.txt'):
                filepath = filepath.rsplit('.', 1)[0] + '.txt' if '.' in filepath else filepath + '.txt'
            StringSerializer.save(filepath, data)
            return filepath, "STRING"

        # TRIMESH
        if hasattr(data, 'vertices') and hasattr(data, 'faces'):
            if HAS_TRIMESH:
                if not any(filepath.endswith(ext) for ext in ['.ply', '.obj', '.glb', '.gltf', '.stl']):
                    filepath = filepath.rsplit('.', 1)[0] + '.ply' if '.' in filepath else filepath + '.ply'
                ext = filepath.rsplit('.', 1)[-1].lower()
                trimesh.exchange.export.export_mesh(data, filepath, file_type=ext)
                return filepath, "TRIMESH"

        # Fallback: pickle
        if not filepath.endswith('.pkl'):
            filepath = filepath.rsplit('.', 1)[0] + '.pkl' if '.' in filepath else filepath + '.pkl'
        PickleSerializer.save(filepath, data)
        return filepath, "GENERIC"

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> str:
        # Save nodes are side-effect operations — force re-execution each Run so
        # downstream context changes (e.g. BD_SaveContext updates) are picked up
        # instead of returning a cached file_path from the previous Run.
        import time
        return f"savefile_{time.time()}"

    @classmethod
    def execute(cls, data, filename: str, skip_if_exists: bool = True,
                name_prefix: str = "", extension: str = "",
                context_id: str = "", suffix: str = "", custom_vars: str = "") -> io.NodeOutput:
        from .save_context import resolve_context_path, get_context, auto_pick_context

        effective_ctx_id = context_id
        if not effective_ctx_id:
            picked = auto_pick_context()
            if picked is not None:
                effective_ctx_id = picked

        if effective_ctx_id and get_context(effective_ctx_id) is not None:
            try:
                ext = (extension.strip().lstrip('.') if extension else "png") or "png"
                filepath, _ = resolve_context_path(
                    effective_ctx_id, suffix, ext,
                    node_filename=filename, node_name_prefix=name_prefix,
                    node_custom_vars=custom_vars,
                )
                if '/' in filepath or '\\' in filepath:
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                try:
                    final_path, data_type = cls._detect_type_and_save(data, filepath)
                    auto_str = " (auto-picked)" if not context_id else ""
                    if skip_if_exists and os.path.exists(final_path):
                        return io.NodeOutput(data, final_path, f"EXISTS via context='{effective_ctx_id}'{auto_str}: {os.path.basename(final_path)}")
                    return io.NodeOutput(data, final_path, f"Saved {data_type} via context='{effective_ctx_id}'{auto_str} suffix='{suffix}': {os.path.basename(final_path)}")
                except Exception as e:
                    return io.NodeOutput(data, "", f"Save (context) failed: {e}")
            except ValueError as ve:
                return io.NodeOutput(data, "", f"Save context error: {ve}")

        if name_prefix:
            full_name = f"{name_prefix}_{filename}"
        else:
            full_name = filename

        # Apply suffix in legacy mode too (e.g. ForEachRun's `_shoes` `_shirt` labels).
        # Suffix is appended literally — caller controls separator (typical convention: leading underscore).
        if suffix:
            full_name = f"{full_name}{suffix}"

        if extension:
            ext = extension.strip()
            if not ext.startswith('.'):
                ext = '.' + ext
            full_name = full_name + ext

        filepath = os.path.join(OUTPUT_DIR, full_name)

        if '/' in full_name or '\\' in full_name:
            subdir = os.path.dirname(filepath)
            os.makedirs(subdir, exist_ok=True)

        try:
            final_path, data_type = cls._detect_type_and_save(data, filepath)

            if skip_if_exists and os.path.exists(final_path):
                return io.NodeOutput(data, final_path, f"EXISTS: {os.path.basename(final_path)}")

            status = f"Saved {data_type}: {os.path.basename(final_path)}"
            return io.NodeOutput(data, final_path, status)
        except Exception as e:
            return io.NodeOutput(data, "", f"Save failed: {e}")


class BD_BulkSave(io.ComfyNode):
    """Bulk-save N inputs in ONE Run using a save context.

    Wire up to 16 typed inputs (any data type) and a parallel labels list — each
    wired input is saved with the corresponding label as suffix, all in a single
    execute() call. No queueing. The node IS the loop.

    Use for: "save 7 PBR maps with 7 different suffixes in one Run."
    Sibling node BD_ForEachRun emits one (data, label) per Run instead — use that
    when downstream needs more than save (e.g., per-iteration upscale + process).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        inputs = [
            io.String.Input("labels", multiline=True, default="albedo\nnormal\nroughness\nmetallic\nao",
                            tooltip="One label per line, aligned with WIRED inputs (after skipping empty slots). "
                                    "If labels list is shorter than wired inputs, missing labels save with empty suffix."),
            io.String.Input("label_prefix", default="_", optional=True,
                            tooltip="Prepended to each label before becoming the save suffix. Default '_' so "
                                    "label='albedo' → suffix='_albedo' → file ends in '_albedo'."),
            io.String.Input("context_id", default="", optional=True,
                            tooltip="Match a BD_SaveContext id. Empty + exactly one context registered = auto-pick."),
            io.Combo.Input("format", options=["png", "jpg", "webp"], default="png", optional=True),
            io.Int.Input("jpg_quality", default=95, min=1, max=100, step=1, optional=True),
            io.Boolean.Input("skip_if_exists", default=False, optional=True,
                             tooltip="If True, don't overwrite existing files (still reports their path)."),
            io.String.Input("custom_vars", multiline=True, default="", optional=True,
                            tooltip="Extra variables applied to ALL bulk-saved files in this batch. "
                                    "One per line as key=value. Layered ON TOP of the context's custom_vars. "
                                    "Examples:\n  subfolder=PBR\n  materials=metal\n  pass=4k\n"
                                    "Become %subfolder%, %materials%, %pass% in the template. Empty values OK."),
        ]
        for i in range(1, 17):
            inputs.append(io.AnyType.Input(f"input_{i}", optional=True,
                                           tooltip=f"Input slot #{i}. Wire any data type. Empty slots are skipped."))
        return io.Schema(
            node_id="BD_BulkSave",
            display_name="BD Bulk Save",
            category="🧠BrainDead/Cache",
            is_output_node=True,
            description=(
                "Save N inputs in ONE Run using a save context. Wire BD_DerivePBR's 7 outputs to "
                "input_1..input_7, set labels accordingly, and all 7 files save in a single Run. "
                "No queueing. Replaces N parallel BD_SaveFile nodes for batch save scenarios."
            ),
            inputs=inputs,
            outputs=[
                io.Int.Output(display_name="saved_count"),
                io.String.Output(display_name="filepaths"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> str:
        # Same reasoning as BD_SaveFile — force re-execution every Run so context
        # updates upstream are reflected. Without this, wiring an output (e.g. status
        # to PreviewText) was the only way to get the node to run.
        import time
        return f"bulksave_{time.time()}"

    @classmethod
    def execute(cls, labels="", label_prefix="_", context_id="",
                format="png", jpg_quality=95, skip_if_exists=False,
                custom_vars="", **inputs) -> io.NodeOutput:
        from .save_context import resolve_context_path, get_context, auto_pick_context

        wired = []
        for i in range(1, 17):
            v = inputs.get(f"input_{i}")
            if v is not None:
                wired.append((i, v))
        if not wired:
            return io.NodeOutput(0, "", "BD_BatchSave: no inputs wired")

        effective_ctx_id = context_id
        if not effective_ctx_id:
            picked = auto_pick_context()
            if picked is not None:
                effective_ctx_id = picked
        # Resolve context, but DON'T bail if it's missing. Fall back to a flat
        # save in OUTPUT_DIR so data is never silently lost.
        ctx_resolved = bool(effective_ctx_id and get_context(effective_ctx_id) is not None)
        fallback_warning = None
        if not ctx_resolved:
            fallback_warning = (
                f"WARNING: no usable BD_SaveContext (context_id='{context_id}', "
                f"auto-pick returned None). Falling back to flat save in "
                f"{OUTPUT_DIR}/bd_bulksave_fallback/<label>_<idx>.<ext>. "
                f"Add a BD_SaveContext upstream named 'default' for proper paths."
            )
            print(f"[BD BulkSave] {fallback_warning}", flush=True)

        label_list = [l.strip() for l in (labels or "").strip().split("\n")]
        ext = format if format != "jpg" else "jpg"

        saved_paths = []
        status_lines = []
        skipped = 0
        errors = 0
        fallback_dir = os.path.join(OUTPUT_DIR, "bd_bulksave_fallback")

        for i, (slot, data) in enumerate(wired):
            label_raw = label_list[i] if i < len(label_list) else ""
            suffix = (label_prefix or "") + label_raw if label_raw else ""
            try:
                if ctx_resolved:
                    filepath, rel_path = resolve_context_path(
                        effective_ctx_id, suffix, ext,
                        node_custom_vars=custom_vars,
                    )
                else:
                    # Fallback: flat save with auto-numbered filename
                    os.makedirs(fallback_dir, exist_ok=True)
                    base_name = (label_raw or f"slot{slot:02d}").strip("_")
                    filepath = os.path.join(
                        fallback_dir,
                        f"{base_name}_{i:03d}.{ext}",
                    )
                    rel_path = os.path.relpath(filepath, OUTPUT_DIR).replace("\\", "/")

                if '/' in filepath or '\\' in filepath:
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)

                if skip_if_exists and os.path.exists(filepath):
                    saved_paths.append(filepath)
                    status_lines.append(f"  [{i + 1}/{len(wired)}] slot=input_{slot} suffix='{suffix}' EXISTS → {rel_path}")
                    skipped += 1
                    continue

                final_path, data_type = BD_SaveFile._detect_type_and_save(data, filepath)
                saved_paths.append(final_path)
                rel_final = os.path.relpath(final_path, OUTPUT_DIR).replace("\\", "/")
                status_lines.append(f"  [{i + 1}/{len(wired)}] slot=input_{slot} suffix='{suffix}' {data_type} → {rel_final}")
            except Exception as e:
                errors += 1
                status_lines.append(f"  [{i + 1}/{len(wired)}] slot=input_{slot} ERROR: {e}")

        if ctx_resolved:
            auto_str = " (auto-picked)" if not context_id else ""
            header = (
                f"saved={len(saved_paths) - skipped} skipped={skipped} errors={errors} "
                f"context='{effective_ctx_id}'{auto_str}"
            )
        else:
            header = (
                f"saved={len(saved_paths) - skipped} skipped={skipped} errors={errors} "
                f"context=FALLBACK (data preserved in {fallback_dir})"
            )
        summary = header + "\n" + "\n".join(status_lines)
        if fallback_warning:
            summary = fallback_warning + "\n" + summary
        print(f"[BD BulkSave] {summary}", flush=True)
        return io.NodeOutput(len(saved_paths), "\n".join(saved_paths), summary)


class BD_LoadImage(io.ComfyNode):
    """Load an image from a file path (STRING input)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_LoadImage",
            display_name="BD Load Image",
            category="🧠BrainDead/Cache",
            description="Load an image from a file path string.",
            inputs=[
                io.String.Input("file_path", default="", force_input=True),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, file_path: str) -> io.NodeOutput:
        from PIL import Image
        import torch
        import numpy as np

        if not file_path or not os.path.exists(file_path):
            empty_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return io.NodeOutput(empty_img, empty_mask, f"File not found: {file_path}")

        try:
            pil_img = Image.open(file_path)

            if pil_img.mode == 'RGBA':
                r, g, b, a = pil_img.split()
                pil_img = Image.merge('RGB', (r, g, b))
                mask_np = np.array(a).astype(np.float32) / 255.0
                mask = torch.from_numpy(mask_np).unsqueeze(0)
            elif pil_img.mode == 'L':
                mask_np = np.array(pil_img).astype(np.float32) / 255.0
                mask = torch.from_numpy(mask_np).unsqueeze(0)
                pil_img = pil_img.convert('RGB')
            else:
                pil_img = pil_img.convert('RGB')
                mask = torch.zeros((1, pil_img.height, pil_img.width), dtype=torch.float32)

            img_np = np.array(pil_img).astype(np.float32) / 255.0
            image = torch.from_numpy(img_np).unsqueeze(0)

            return io.NodeOutput(image, mask, f"Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            empty_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return io.NodeOutput(empty_img, empty_mask, f"Load failed: {e}")


class BD_LoadMesh(io.ComfyNode):
    """Load a 3D mesh from a file path (STRING input)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_LoadMesh",
            display_name="BD Load Mesh",
            category="🧠BrainDead/Cache",
            description="Load a 3D mesh from a file path. Supports PLY, OBJ, GLB, STL.",
            inputs=[
                io.String.Input("file_path", default="", force_input=True),
            ],
            outputs=[
                io.Mesh.Output(display_name="mesh"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, file_path: str) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput(None, "ERROR: trimesh not installed")

        if not file_path or not os.path.exists(file_path):
            return io.NodeOutput(None, f"File not found: {file_path}")

        try:
            mesh = trimesh.load(file_path)

            if isinstance(mesh, trimesh.Scene):
                meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if meshes:
                    mesh = trimesh.util.concatenate(meshes)
                else:
                    return io.NodeOutput(None, "No meshes found in scene")

            return io.NodeOutput(mesh, f"Loaded: {os.path.basename(file_path)} ({len(mesh.vertices)} verts)")
        except Exception as e:
            return io.NodeOutput(None, f"Load failed: {e}")


class BD_LoadAudio(io.ComfyNode):
    """Load audio from a file path (STRING input)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_LoadAudio",
            display_name="BD Load Audio",
            category="🧠BrainDead/Cache",
            description="Load audio from a file path. Supports WAV, MP3, FLAC.",
            inputs=[
                io.String.Input("file_path", default="", force_input=True),
            ],
            outputs=[
                io.Audio.Output(display_name="audio"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, file_path: str) -> io.NodeOutput:
        if not file_path or not os.path.exists(file_path):
            return io.NodeOutput(None, f"File not found: {file_path}")

        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(file_path)
            audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
            duration = waveform.shape[-1] / sample_rate
            return io.NodeOutput(audio, f"Loaded: {os.path.basename(file_path)} ({duration:.1f}s)")
        except ImportError:
            return io.NodeOutput(None, "ERROR: torchaudio not installed")
        except Exception as e:
            return io.NodeOutput(None, f"Load failed: {e}")


class BD_LoadText(io.ComfyNode):
    """Load text from a file path (STRING input)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_LoadText",
            display_name="BD Load Text",
            category="🧠BrainDead/Cache",
            description="Load text content from a file path.",
            inputs=[
                io.String.Input("file_path", default="", force_input=True),
            ],
            outputs=[
                io.String.Output(display_name="text"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, file_path: str) -> io.NodeOutput:
        if not file_path or not os.path.exists(file_path):
            return io.NodeOutput("", f"File not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return io.NodeOutput(text, f"Loaded: {os.path.basename(file_path)} ({len(text)} chars)")
        except Exception as e:
            return io.NodeOutput("", f"Load failed: {e}")


# V3 node list for extension
FILE_OPS_V3_NODES = [
    BD_ClearCache,
    BD_SaveFile,
    BD_BulkSave,
    BD_LoadImage,
    BD_LoadMesh,
    BD_LoadAudio,
    BD_LoadText,
]

# V1 compatibility - NODE_CLASS_MAPPINGS dict
FILE_OPS_NODES = {
    "BD_ClearCache": BD_ClearCache,
    "BD_SaveFile": BD_SaveFile,
    "BD_BulkSave": BD_BulkSave,
    "BD_LoadImage": BD_LoadImage,
    "BD_LoadMesh": BD_LoadMesh,
    "BD_LoadAudio": BD_LoadAudio,
    "BD_LoadText": BD_LoadText,
}

FILE_OPS_DISPLAY_NAMES = {
    "BD_ClearCache": "BD Clear Cache",
    "BD_SaveFile": "BD Save File",
    "BD_BulkSave": "BD Bulk Save",
    "BD_LoadImage": "BD Load Image",
    "BD_LoadMesh": "BD Load Mesh",
    "BD_LoadAudio": "BD Load Audio",
    "BD_LoadText": "BD Load Text",
}
