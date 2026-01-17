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
            description="Save any data type to output folder. Supports IMAGE, MASK, AUDIO, LATENT, STRING, TRIMESH.",
            is_output_node=True,
            inputs=[
                io.AnyType.Input("data"),
                io.String.Input("filename", default="saved_file"),
                io.Boolean.Input("skip_if_exists", default=True),
                io.String.Input("name_prefix", default="", optional=True),
                io.String.Input("extension", default="", optional=True, tooltip="Override auto-detected extension"),
            ],
            outputs=[
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
    def execute(cls, data, filename: str, skip_if_exists: bool = True,
                name_prefix: str = "", extension: str = "") -> io.NodeOutput:
        if name_prefix:
            full_name = f"{name_prefix}_{filename}"
        else:
            full_name = filename

        if extension:
            ext = extension.strip()
            if not ext.startswith('.'):
                ext = '.' + ext
            full_name = full_name + ext

        filepath = os.path.join(OUTPUT_DIR, full_name)

        # Handle subdirectories
        if '/' in full_name or '\\' in full_name:
            subdir = os.path.dirname(filepath)
            os.makedirs(subdir, exist_ok=True)

        try:
            final_path, data_type = cls._detect_type_and_save(data, filepath)

            if skip_if_exists and os.path.exists(final_path):
                return io.NodeOutput(final_path, f"EXISTS: {os.path.basename(final_path)}")

            status = f"Saved {data_type}: {os.path.basename(final_path)}"
            return io.NodeOutput(final_path, status)
        except Exception as e:
            return io.NodeOutput("", f"Save failed: {e}")


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
    BD_LoadImage,
    BD_LoadMesh,
    BD_LoadAudio,
    BD_LoadText,
]

# V1 compatibility - NODE_CLASS_MAPPINGS dict
FILE_OPS_NODES = {
    "BD_ClearCache": BD_ClearCache,
    "BD_SaveFile": BD_SaveFile,
    "BD_LoadImage": BD_LoadImage,
    "BD_LoadMesh": BD_LoadMesh,
    "BD_LoadAudio": BD_LoadAudio,
    "BD_LoadText": BD_LoadText,
}

FILE_OPS_DISPLAY_NAMES = {
    "BD_ClearCache": "BD Clear Cache",
    "BD_SaveFile": "BD Save File",
    "BD_LoadImage": "BD Load Image",
    "BD_LoadMesh": "BD Load Mesh",
    "BD_LoadAudio": "BD Load Audio",
    "BD_LoadText": "BD Load Text",
}
