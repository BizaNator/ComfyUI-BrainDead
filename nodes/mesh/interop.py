"""
TRIMESH <-> native MESH interop adapters.

BrainDead (and TRELLIS2 / Hunyuan3d-2-1 / GeometryPack) pass meshes as a full
`trimesh.Trimesh` object under the custom "TRIMESH" type, which carries vertex
colors, UVs, materials and processing/export. ComfyUI's *built-in* 3D nodes
(VAEDecodeHunyuan3D, SaveGLB, VoxelToMesh...) use the native `MESH` type — a thin
container of just batched vertices/faces tensors (B, N, 3).

These two nodes bridge the gap so BD meshes can feed native nodes and native
results can enter the BD pipeline. Geometry survives the round trip; colors/UVs
do NOT survive a trip through native MESH (the format cannot represent them).

- BD_TrimeshToMesh : TRIMESH -> MESH   (hand geometry to native nodes / SaveGLB)
- BD_MeshToTrimesh : MESH   -> TRIMESH (pull a native Hunyuan3D mesh into BD)
"""

import numpy as np
import torch

from comfy_api.latest import io

from .types import TrimeshInput, TrimeshOutput

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def _native_mesh(vertices: torch.Tensor, faces: torch.Tensor):
    """Construct a native comfy MESH object (batched B,N,3) from tensors."""
    from comfy_api.latest import MESH
    return MESH(vertices, faces)


class BD_TrimeshToMesh(io.ComfyNode):
    """
    Convert a BrainDead TRIMESH into ComfyUI's native MESH type.

    Use this to feed BD/TRELLIS2 meshes into ComfyUI's built-in 3D nodes
    (e.g. Save 3D Model / SaveGLB). Outputs batched vertices/faces tensors
    (shape [1, N, 3] / [1, M, 3]) — the layout native nodes expect.

    NOTE: native MESH carries geometry only. Vertex colors, UVs and materials
    are dropped (the format can't hold them) — keep using BD Export Mesh With
    Colors when you need those preserved.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_TrimeshToMesh",
            display_name="BD Trimesh → MESH",
            category="🧠BrainDead/Mesh",
            description="Convert BD/TRELLIS2 TRIMESH to native ComfyUI MESH (geometry only) "
                        "for built-in 3D nodes like Save 3D Model.",
            inputs=[
                TrimeshInput("mesh"),
            ],
            outputs=[
                io.Mesh.Output(display_name="MESH"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, mesh) -> io.NodeOutput:
        if mesh is None:
            return io.NodeOutput(None, "ERROR: mesh is None")
        try:
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            v = torch.from_numpy(verts).unsqueeze(0)          # (1, N, 3)
            f = torch.from_numpy(faces).unsqueeze(0)          # (1, M, 3)
            native = _native_mesh(v, f)
            status = f"TRIMESH → MESH: {verts.shape[0]:,} verts, {faces.shape[0]:,} faces (colors/UVs dropped)"
            print(f"[BD Trimesh→MESH] {status}", flush=True)
            return io.NodeOutput(native, status)
        except Exception as e:
            return io.NodeOutput(None, f"ERROR: {e}")


class BD_MeshToTrimesh(io.ComfyNode):
    """
    Convert ComfyUI's native MESH into a BrainDead TRIMESH.

    Use this to pull a mesh produced by built-in 3D nodes (e.g. VAE Decode
    Hunyuan3D / Voxel To Mesh) into the BD pipeline (CuMesh, Blender, UV unwrap,
    color sampling, export...). Native MESH is batched; when it holds more than
    one mesh, set `batch_index` to choose which one.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_MeshToTrimesh",
            display_name="BD MESH → Trimesh",
            category="🧠BrainDead/Mesh",
            description="Convert native ComfyUI MESH to BD/TRELLIS2 TRIMESH for the BD mesh pipeline.",
            inputs=[
                io.Mesh.Input("MESH"),
                io.Int.Input("batch_index", default=0, min=0, max=4096, optional=True,
                             tooltip="Which mesh to extract when the native MESH is batched."),
            ],
            outputs=[
                TrimeshOutput(display_name="mesh"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, MESH, batch_index: int = 0) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput(None, "ERROR: trimesh not installed")
        if MESH is None:
            return io.NodeOutput(None, "ERROR: MESH is None")
        try:
            verts_t = MESH.vertices
            faces_t = MESH.faces
            # Native MESH is batched (B, N, 3); accept un-batched (N, 3) too.
            if hasattr(verts_t, "dim") and verts_t.dim() == 3:
                b = max(0, min(batch_index, verts_t.shape[0] - 1))
                verts_t = verts_t[b]
                faces_t = faces_t[b]
            verts = np.asarray(verts_t.detach().cpu().numpy() if hasattr(verts_t, "detach") else verts_t,
                               dtype=np.float32)
            faces = np.asarray(faces_t.detach().cpu().numpy() if hasattr(faces_t, "detach") else faces_t,
                               dtype=np.int64)
            tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            status = f"MESH → TRIMESH: {len(tm.vertices):,} verts, {len(tm.faces):,} faces"
            print(f"[BD MESH→Trimesh] {status}", flush=True)
            return io.NodeOutput(tm, status)
        except Exception as e:
            return io.NodeOutput(None, f"ERROR: {e}")


MESH_INTEROP_V3_NODES = [BD_TrimeshToMesh, BD_MeshToTrimesh]

MESH_INTEROP_NODES = {
    "BD_TrimeshToMesh": BD_TrimeshToMesh,
    "BD_MeshToTrimesh": BD_MeshToTrimesh,
}

MESH_INTEROP_DISPLAY_NAMES = {
    "BD_TrimeshToMesh": "BD Trimesh → MESH",
    "BD_MeshToTrimesh": "BD MESH → Trimesh",
}
