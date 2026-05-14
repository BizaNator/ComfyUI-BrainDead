"""
Custom types for the face-wrap pipeline.

LANDMARKS_BATCH and FLAME_FIT are opaque to ComfyUI — they're plain Python
dicts at runtime; the contract is enforced by the consuming node.

LANDMARKS_BATCH dict shape (one entry per input image):
    {
        "views": [
            {
                "landmarks_2d":  np.ndarray (N, 2) float — pixel coords
                "landmarks_3d":  np.ndarray (N, 3) float — MediaPipe normalized
                                  (x,y in [0,1] image-space, z depth-rel)
                "detected":      bool — False for rear views / failed detection
                "view_hint":     str — "front" | "left" | "right" | "rear" | "unknown"
                "image_size":    (h, w)
                "yaw_estimate":  float — degrees, signed (left>0, right<0)
            },
            ...
        ],
        "model": str — "face_mesh_full" or "face_mesh_lite"
        "n_landmarks": int — 478 (with iris refine) or 468 (without)
    }

FLAME_FIT dict shape (single, joint across all input views):
    {
        "verts":          torch.Tensor (V, 3) float — fitted FLAME mesh verts
        "faces":          torch.Tensor (F, 3) int   — FLAME topology
        "uvs":            torch.Tensor (V_uv, 2)    — FLAME canonical UV
        "face_uv_idx":    torch.Tensor (F, 3) int   — per-face UV indices
        "shape_params":   torch.Tensor (n_shape,)
        "exp_params":     torch.Tensor (n_exp,)
        "cameras":        list of {
                              "K": (3,3) intrinsics,
                              "R": (3,3) rotation,
                              "t": (3,) translation,
                              "view_hint": str,
                              "image_size": (h, w),
                          }
    }
"""

from comfy_api.latest import io


LANDMARKS_BATCH_TYPE = "BD_LANDMARKS_BATCH"
FLAME_FIT_TYPE = "BD_FLAME_FIT"


def LandmarksBatchInput(name: str, optional: bool = False, tooltip: str = ""):
    return io.Custom(LANDMARKS_BATCH_TYPE).Input(name, optional=optional, tooltip=tooltip)


def LandmarksBatchOutput(display_name: str = "landmarks"):
    return io.Custom(LANDMARKS_BATCH_TYPE).Output(display_name=display_name)


def FlameFitInput(name: str, optional: bool = False, tooltip: str = ""):
    return io.Custom(FLAME_FIT_TYPE).Input(name, optional=optional, tooltip=tooltip)


def FlameFitOutput(display_name: str = "flame_fit"):
    return io.Custom(FLAME_FIT_TYPE).Output(display_name=display_name)
