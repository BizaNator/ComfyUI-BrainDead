"""
Custom types for the face-wrap pipeline.

LANDMARKS_BATCH and FACE_FIT are opaque to ComfyUI — they're plain Python
dicts at runtime; the contract is enforced by the consuming node.

LANDMARKS_BATCH dict shape:
    {
        "views": [
            {
                "landmarks_2d":  np.ndarray (N, 2) float — pixel coords
                "landmarks_3d":  np.ndarray (N, 3) float — MediaPipe normalized
                                  (x,y in [0,1] image-space, z depth-rel)
                "transform_4x4": np.ndarray (4, 4) float — head pose in camera space
                "detected":      bool — False for rear views / failed detection
                "view_hint":     str — "front" | "left" | "right" | "rear" | "unknown"
                "image_size":    (h, w)
                "yaw_estimate":  float — degrees, signed (left>0, right<0)
            },
            ...
        ],
        "model":       str — e.g. "face_landmarker_v1"
        "n_landmarks": int — 478 (with iris refine) or 468 (without)
        "model_path":  str — resolved path to the .task model bundle
    }

FACE_FIT dict shape:
    {
        # Canonical mesh — shared across all views
        "canonical_verts":  np.ndarray (V, 3) float — neutral pose vertices
        "canonical_uvs":    np.ndarray (V, 2) float — per-vertex UV (V flipped
                                                        to OpenGL convention)
        "faces":            np.ndarray (F, 3) int   — vertex indices per triangle
        "face_uvs":         np.ndarray (F, 3) int   — UV indices per triangle
                                                        (may differ from `faces`
                                                        when the mesh has UV
                                                        islands / per-corner UVs)
        "uv_to_vert":       np.ndarray (V_uv,) int  — for each UV index, the
                                                        corresponding 3D vertex
                                                        index (many-to-one)
        "mesh_source":      str — "mediapipe_canonical" | future: "flame" | "ict_facekit"

        # Per-view data — same length and ordering as LANDMARKS_BATCH["views"]
        "views": [
            {
                "verts_2d":      np.ndarray (V, 2) float — pixel coords in source image
                "verts_3d":      np.ndarray (V, 3) float — MediaPipe 3D landmarks
                "transform_4x4": np.ndarray (4, 4) float
                "detected":      bool
                "view_hint":     str
                "image_size":    (h, w)
                "yaw_estimate":  float
            },
            ...
        ]
    }

Note: V is the canonical mesh's vertex count (468 for MediaPipe). When the
input LANDMARKS_BATCH has N=478 (refine_landmarks=True), the extra iris/pupil
landmarks at indices 468..477 are dropped — they're not mesh vertices.
"""

from comfy_api.latest import io


LANDMARKS_BATCH_TYPE = "BD_LANDMARKS_BATCH"
FACE_FIT_TYPE = "BD_FACE_FIT"


def LandmarksBatchInput(name: str, optional: bool = False, tooltip: str = ""):
    return io.Custom(LANDMARKS_BATCH_TYPE).Input(name, optional=optional, tooltip=tooltip)


def LandmarksBatchOutput(display_name: str = "landmarks"):
    return io.Custom(LANDMARKS_BATCH_TYPE).Output(display_name=display_name)


def FaceFitInput(name: str, optional: bool = False, tooltip: str = ""):
    return io.Custom(FACE_FIT_TYPE).Input(name, optional=optional, tooltip=tooltip)


def FaceFitOutput(display_name: str = "face_fit"):
    return io.Custom(FACE_FIT_TYPE).Output(display_name=display_name)
