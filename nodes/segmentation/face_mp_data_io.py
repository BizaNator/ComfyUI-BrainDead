"""
BD MP Save Face Data / BD MP Load Face Data — persist MediaPipe face mask data.

Saves all face region masks (BD MP Face Mask outputs, BD MP Face Refine outputs,
head_mask, and pre-resolution image) to NPZ + JSON so the data survives image
processing steps (eye removal, mouth close, delighting, albedo prep) that break
subsequent MediaPipe detection.

context_id integration:
  Wire a BD_SaveContext context_id to auto-resolve the save path.
  When context_id is set, output_dir/name are ignored — the context template
  determines the full path (same pattern as BD_SaveFile/BD_BulkSave).
  BD MP Load Face Data can use the same context_id to auto-find saved files.

NPZ format: {region_name: uint8 (H, W)} for each present mask.
JSON sidecar: image dimensions, saved regions, per-region bboxes — readable
              from Blender Python (import numpy as np, json).

Wiring BD MP Face Refine → BD MP Save Face Data:
  BD MP Face Refine's mask outputs (skin, left_eye, right_eye, left_brow,
  right_brow, lips, nose, head_mask) wire directly to the matching inputs on
  BD MP Save Face Data — refined SAM3-accurate masks replace the raw MP masks.
"""

from __future__ import annotations

import os
import json
import numpy as np
import torch
from glob import glob

from comfy_api.latest import io


# ── Mask region names (matches BD MP Face Mask output order) ──────────────────
_MP_KEYS = [
    'face_oval', 'skin',
    'left_eye', 'right_eye', 'eyes',
    'left_brow', 'right_brow', 'brows',
    'left_iris', 'right_iris', 'irises',
    'lips', 'nose',
    'left_ear', 'right_ear', 'ears',
    'forehead', 'hair',
]
_EXTRA_KEYS = ['head_mask', 'masked_skin']
_ALL_MASK_KEYS = _MP_KEYS + _EXTRA_KEYS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mask_to_u8(t: torch.Tensor | None) -> np.ndarray | None:
    if t is None:
        return None
    m = t.detach().cpu().float()
    if m.ndim == 3:
        m = m[0]
    return (m.numpy() * 255.0).clip(0, 255).astype(np.uint8)


def _u8_to_mask(arr: np.ndarray, B: int = 1) -> torch.Tensor:
    t = torch.from_numpy(arr.astype(np.float32) / 255.0)
    return t.unsqueeze(0).expand(B, -1, -1).contiguous()


def _blank_mask(H: int, W: int, B: int = 1) -> torch.Tensor:
    return torch.zeros((B, H, W), dtype=torch.float32)


def _blank_image(H: int, W: int, B: int = 1) -> torch.Tensor:
    return torch.zeros((B, H, W, 3), dtype=torch.float32)


def _image_to_u8(t: torch.Tensor | None) -> np.ndarray | None:
    if t is None:
        return None
    img = t.detach().cpu().float()
    if img.ndim == 4:
        img = img[0]
    return (img[..., :3].numpy() * 255.0).clip(0, 255).astype(np.uint8)


def _u8_to_image(arr: np.ndarray, B: int = 1) -> torch.Tensor:
    t = torch.from_numpy(arr.astype(np.float32) / 255.0)
    return t.unsqueeze(0).expand(B, -1, -1, -1).contiguous()


def _bbox(mask_u8: np.ndarray) -> dict | None:
    rows = np.any(mask_u8 > 0, axis=1)
    cols = np.any(mask_u8 > 0, axis=0)
    if not rows.any():
        return None
    y1, y2 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    x1, x2 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return {"x": x1, "y": y1, "width": x2 - x1 + 1, "height": y2 - y1 + 1,
            "cx": round((x1 + x2) / 2.0, 1), "cy": round((y1 + y2) / 2.0, 1)}


def _try_context_path(context_id: str, suffix: str) -> tuple[str, str] | None:
    """Try to resolve a save path from context_id. Returns (npz_path, json_path) or None."""
    try:
        from ..cache.save_context import resolve_context_path
        full_path, _ = resolve_context_path(context_id, suffix, "npz")
        npz_path = full_path
        json_path = full_path.replace(".npz", ".json").replace(".mpface.npz", ".mpface.json")
        # Ensure .mpface suffix in stem
        if not npz_path.endswith(".mpface.npz"):
            base = npz_path[:-4]  # strip .npz
            npz_path = base + ".mpface.npz"
            json_path = base + ".mpface.json"
        return npz_path, json_path
    except Exception as e:
        print(f"[BD MP SaveFaceData] context_id '{context_id}' lookup failed: {e}")
        return None


# ── Save node ─────────────────────────────────────────────────────────────────

class BD_SaveMPFaceData(io.ComfyNode):
    """
    Save all MediaPipe face mask outputs + extras to disk as NPZ + JSON.

    Wire BD MP Face Mask outputs to the matching inputs. Also accepts refined
    masks from BD MP Face Refine (skin, left_eye, right_eye, left_brow,
    right_brow, lips, nose, head_mask) — wiring these overwrites the raw MP
    masks with the SAM3-accurate refined versions.

    context_id: wire a BD_SaveContext context_id to auto-resolve the save path.
    When set, output_dir and name are ignored.

    JSON sidecar is Blender-readable — bboxes, region centers, image dims.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_SaveMPFaceData",
            display_name="BD MP Save Face Data",
            category="🧠BrainDead/Segmentation",
            description=(
                "Save MediaPipe face mask data to disk (NPZ + JSON). "
                "Wire BD MP Face Mask outputs directly to matching inputs. "
                "Wire BD MP Face Refine masks to overwrite with SAM3-refined versions. "
                "Use context_id (from BD_SaveContext) for auto-path resolution. "
                "JSON sidecar has per-region bboxes readable from Blender Python."
            ),
            is_output_node=True,
            inputs=[
                # context_id — optional, replaces output_dir + name when set
                io.String.Input("context_id", default="",
                                optional=True,
                                tooltip="BD_SaveContext context_id. When wired/set, output_dir and name "
                                        "are ignored — the context template determines the full path."),
                # MP mask outputs — all optional so any subset can be wired
                io.Mask.Input("face_oval",   optional=True),
                io.Mask.Input("skin",        optional=True,
                              tooltip="From BD MP Face Mask or BD MP Face Refine (SAM3-refined)."),
                io.Mask.Input("left_eye",    optional=True),
                io.Mask.Input("right_eye",   optional=True),
                io.Mask.Input("eyes",        optional=True),
                io.Mask.Input("left_brow",   optional=True),
                io.Mask.Input("right_brow",  optional=True),
                io.Mask.Input("brows",       optional=True),
                io.Mask.Input("left_iris",   optional=True),
                io.Mask.Input("right_iris",  optional=True),
                io.Mask.Input("irises",      optional=True),
                io.Mask.Input("lips",        optional=True),
                io.Mask.Input("nose",        optional=True),
                io.Mask.Input("left_ear",    optional=True),
                io.Mask.Input("right_ear",   optional=True),
                io.Mask.Input("ears",        optional=True),
                io.Mask.Input("forehead",    optional=True),
                io.Mask.Input("hair",        optional=True),
                # Extras — also outputs of BD MP Face Refine
                io.Mask.Input("head_mask",   optional=True,
                              tooltip="External head silhouette (SAM3/BD MP Face Refine)."),
                io.Mask.Input("masked_skin", optional=True,
                              tooltip="Refined skin mask (BD MP Face Refine 'skin' output or custom)."),
                # Pre-resolution image for Blender UV projection
                io.Image.Input("image", optional=True,
                               tooltip="Pre-resolution image for texture generation / Blender UV projection."),
                # Fallback save location (used only when context_id is empty)
                io.String.Input("output_dir", default="mediapipe_data",
                                optional=True,
                                tooltip="Subdirectory under ComfyUI output/ (ignored when context_id is set)."),
                io.String.Input("name", default="face",
                                optional=True,
                                tooltip="Base filename (ignored when context_id is set)."),
                io.Boolean.Input("auto_increment", default=True, optional=True,
                                 tooltip="Append _001, _002… to avoid overwriting."),
            ],
            outputs=[
                io.String.Output(display_name="npz_path"),
                io.String.Output(display_name="json_path"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        context_id: str = "",
        face_oval=None, skin=None,
        left_eye=None, right_eye=None, eyes=None,
        left_brow=None, right_brow=None, brows=None,
        left_iris=None, right_iris=None, irises=None,
        lips=None, nose=None,
        left_ear=None, right_ear=None, ears=None,
        forehead=None, hair=None,
        head_mask=None, masked_skin=None,
        image=None,
        output_dir: str = "mediapipe_data",
        name: str = "face",
        auto_increment: bool = True,
    ) -> io.NodeOutput:

        mask_inputs = {
            'face_oval': face_oval, 'skin': skin,
            'left_eye': left_eye, 'right_eye': right_eye, 'eyes': eyes,
            'left_brow': left_brow, 'right_brow': right_brow, 'brows': brows,
            'left_iris': left_iris, 'right_iris': right_iris, 'irises': irises,
            'lips': lips, 'nose': nose,
            'left_ear': left_ear, 'right_ear': right_ear, 'ears': ears,
            'forehead': forehead, 'hair': hair,
            'head_mask': head_mask, 'masked_skin': masked_skin,
        }

        arrays: dict[str, np.ndarray] = {}
        H, W = 0, 0
        for key, tensor in mask_inputs.items():
            arr = _mask_to_u8(tensor)
            if arr is not None:
                arrays[key] = arr
                if H == 0:
                    H, W = arr.shape[:2]

        img_arr = _image_to_u8(image)
        if img_arr is not None:
            if H == 0:
                H, W = img_arr.shape[:2]
            arrays['image'] = img_arr

        if not arrays:
            return io.NodeOutput("", "", "ERROR: No masks or image wired — nothing to save")

        # Resolve save paths
        npz_path = json_path = ""
        cid = (context_id or "").strip()
        if cid:
            result = _try_context_path(cid, "_mpface")
            if result:
                npz_path, json_path = result
                os.makedirs(os.path.dirname(npz_path), exist_ok=True)

        if not npz_path:
            # Fallback: explicit output_dir + name
            import folder_paths
            base_out = folder_paths.get_output_directory()
            out_dir = os.path.join(base_out, output_dir.replace("\\", "/"))
            os.makedirs(out_dir, exist_ok=True)
            base_name = name
            if auto_increment:
                pattern = os.path.join(out_dir, f"{base_name}_*.mpface.npz")
                existing = glob(pattern)
                nums = []
                for f in existing:
                    stem = os.path.basename(f).replace(".mpface.npz", "")
                    try:
                        nums.append(int(stem.split("_")[-1]))
                    except ValueError:
                        pass
                idx = max(nums) + 1 if nums else 1
                base_name = f"{base_name}_{idx:03d}"
            npz_path  = os.path.join(out_dir, f"{base_name}.mpface.npz")
            json_path = os.path.join(out_dir, f"{base_name}.mpface.json")

        np.savez_compressed(npz_path, **arrays)

        bboxes: dict = {}
        for key, arr in arrays.items():
            if key != 'image':
                bboxes[key] = _bbox(arr)

        meta = {
            "format": "mpface_v1",
            "image_height": H,
            "image_width": W,
            "saved_regions": [k for k in arrays if k != 'image'],
            "has_image": 'image' in arrays,
            "bboxes": bboxes,
        }
        with open(json_path, 'w') as f:
            json.dump(meta, f, indent=2)

        n_masks = len(arrays) - (1 if 'image' in arrays else 0)
        status = (
            f"Saved {n_masks} masks ({H}×{W})"
            + (" + image" if 'image' in arrays else "")
            + f" → {os.path.basename(npz_path)}"
        )
        print(f"[BD MP SaveFaceData] {status}")
        return io.NodeOutput(npz_path, json_path, status)


# ── Load node ─────────────────────────────────────────────────────────────────

class BD_LoadMPFaceData(io.ComfyNode):
    """
    Load previously saved MediaPipe face mask data.

    Outputs the same 18 region masks as BD MP Face Mask, plus head_mask,
    masked_skin, and the pre-resolution image. Missing regions → blank masks.

    Supply file_path explicitly, or set context_id and leave file_path empty
    to auto-locate the most recently saved .mpface.npz for that context.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_LoadMPFaceData",
            display_name="BD MP Load Face Data",
            category="🧠BrainDead/Segmentation",
            description=(
                "Reload MediaPipe face mask data saved by BD MP Save Face Data. "
                "Outputs all 18 region masks + head_mask + masked_skin + image. "
                "Wire context_id and leave file_path empty for auto-locate, "
                "or type the .mpface.npz path directly. No MediaPipe needed."
            ),
            inputs=[
                io.String.Input("file_path", default="",
                                tooltip="Path to .mpface.npz (or .mpface.json). "
                                        "Leave empty to auto-locate via context_id."),
                io.String.Input("context_id", default="",
                                optional=True,
                                tooltip="BD_SaveContext context_id. When file_path is empty, "
                                        "resolves the path automatically from the context template. "
                                        "Also accepts auto-pick when only one context is registered."),
                io.Int.Input("frame_index", default=0, min=0, max=63, optional=True,
                             tooltip="Which saved frame to load (always 0 for single-frame saves)."),
            ],
            outputs=[
                io.Mask.Output(display_name="face_oval"),
                io.Mask.Output(display_name="skin"),
                io.Mask.Output(display_name="left_eye"),
                io.Mask.Output(display_name="right_eye"),
                io.Mask.Output(display_name="eyes"),
                io.Mask.Output(display_name="left_brow"),
                io.Mask.Output(display_name="right_brow"),
                io.Mask.Output(display_name="brows"),
                io.Mask.Output(display_name="left_iris"),
                io.Mask.Output(display_name="right_iris"),
                io.Mask.Output(display_name="irises"),
                io.Mask.Output(display_name="lips"),
                io.Mask.Output(display_name="nose"),
                io.Mask.Output(display_name="left_ear"),
                io.Mask.Output(display_name="right_ear"),
                io.Mask.Output(display_name="ears"),
                io.Mask.Output(display_name="forehead"),
                io.Mask.Output(display_name="hair"),
                io.Mask.Output(display_name="head_mask"),
                io.Mask.Output(display_name="masked_skin"),
                io.Image.Output(display_name="image"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, file_path: str = "", context_id: str = "",
                frame_index: int = 0) -> io.NodeOutput:

        npz_path = file_path.strip()

        # Auto-locate via context_id when file_path is empty
        if not npz_path:
            cid = (context_id or "").strip()
            if not cid:
                # Try auto-pick (single registered context or 'default')
                try:
                    from ..cache.save_context import auto_pick_context
                    cid = auto_pick_context() or ""
                except Exception:
                    pass

            if cid:
                result = _try_context_path(cid, "_mpface")
                if result:
                    candidate, _ = result
                    if os.path.exists(candidate):
                        npz_path = candidate
                    else:
                        # Search the directory for any .mpface.npz
                        search_dir = os.path.dirname(candidate)
                        if os.path.isdir(search_dir):
                            hits = sorted(glob(os.path.join(search_dir, "*.mpface.npz")))
                            if hits:
                                npz_path = hits[-1]  # most recent alphabetically

        # Accept .json path — find companion NPZ
        if npz_path.endswith(".mpface.json"):
            npz_path = npz_path.replace(".mpface.json", ".mpface.npz")

        if not npz_path or not os.path.exists(npz_path):
            msg = f"ERROR: file not found: {npz_path or '(no path resolved)'}"
            return cls._empty_output(msg)

        try:
            data = np.load(npz_path, allow_pickle=False)
        except Exception as e:
            return cls._empty_output(f"ERROR loading NPZ: {e}")

        H, W = 1, 1
        for key in data.files:
            arr = data[key]
            if arr.ndim >= 2:
                H, W = arr.shape[:2]
                break

        def _get(key: str) -> torch.Tensor:
            if key in data.files:
                arr = data[key]
                if arr.ndim == 2:
                    return _u8_to_mask(arr)
            return _blank_mask(H, W)

        def _get_img() -> torch.Tensor:
            if 'image' in data.files:
                arr = data['image']
                if arr.ndim == 3:
                    return _u8_to_image(arr)
            return _blank_image(H, W)

        outputs = [_get(k) for k in _MP_KEYS]
        outputs.append(_get('head_mask'))
        outputs.append(_get('masked_skin'))
        outputs.append(_get_img())

        saved = [k for k in data.files if k != 'image']
        has_img = 'image' in data.files
        status = (
            f"Loaded {len(saved)} masks ({H}×{W})"
            + (" + image" if has_img else "")
            + f" from {os.path.basename(npz_path)}"
        )
        print(f"[BD MP LoadFaceData] {status}")
        return io.NodeOutput(*outputs, status)

    @classmethod
    def _empty_output(cls, msg: str) -> io.NodeOutput:
        H, W = 8, 8
        blanks = [_blank_mask(H, W)] * (len(_MP_KEYS) + 2)
        return io.NodeOutput(*blanks, _blank_image(H, W), msg)


# ── Registration ──────────────────────────────────────────────────────────────

FACE_MP_DATA_IO_V3_NODES = [BD_SaveMPFaceData, BD_LoadMPFaceData]

FACE_MP_DATA_IO_NODES = {
    "BD_SaveMPFaceData": BD_SaveMPFaceData,
    "BD_LoadMPFaceData": BD_LoadMPFaceData,
}

FACE_MP_DATA_IO_DISPLAY_NAMES = {
    "BD_SaveMPFaceData": "BD MP Save Face Data",
    "BD_LoadMPFaceData": "BD MP Load Face Data",
}
