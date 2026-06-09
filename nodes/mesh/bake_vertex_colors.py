"""
BD_BakeVertexColorsFromTexture - Sample a texture through a mesh's UVs into per-vertex
colors (COLOR_0), without touching geometry or the existing UV/material.

Single responsibility: texture + UV -> vertex colors. Works on ANY mesh that carries UVs.
The diffuse/atlas is sampled at each vertex's UV coordinate; the result is written to
`vertex_attributes['COLOR_0']` so the mesh now carries BOTH the texture AND matching
vertex colors. Non-destructive: the original TextureVisuals (UV + baked material) is kept
intact, so nothing is "forced onto" or stripped from the mesh.

Sampling from the texture (rather than re-sampling a voxelgrid) inherits the full texture
resolution at each vertex and is guaranteed consistent with the baked atlas.
"""

import numpy as np

from comfy_api.latest import io

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

from .types import TrimeshInput, TrimeshOutput


def _image_to_np(img):
    """ComfyUI IMAGE tensor [B,H,W,C] float 0-1 -> uint8 HxWx3 numpy (or None)."""
    if img is None:
        return None
    arr = img
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        return None
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return arr[..., :3]


def _texture_from_mesh(mesh):
    """Pull the baseColorTexture off a TextureVisuals mesh -> uint8 HxWx3 (or None)."""
    vis = getattr(mesh, "visual", None)
    mat = getattr(vis, "material", None) if vis is not None else None
    tex = getattr(mat, "baseColorTexture", None) if mat is not None else None
    if tex is None:
        return None
    return np.asarray(tex.convert("RGB"), dtype=np.uint8)


def _sample(tex, uv, bilinear=True):
    """Sample uint8 HxWx3 tex at UV (N,2) in [0,1], origin bottom-left -> (N,3) uint8."""
    H, W = tex.shape[:2]
    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(uv[:, 1], 0.0, 1.0)
    # glTF/UV origin is bottom-left; image rows go top-down -> flip V.
    fy = (1.0 - v) * (H - 1)
    fx = u * (W - 1)
    if not bilinear:
        return tex[np.rint(fy).astype(int), np.rint(fx).astype(int)]
    x0 = np.floor(fx).astype(int); y0 = np.floor(fy).astype(int)
    x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1)
    wx = (fx - x0)[:, None]; wy = (fy - y0)[:, None]
    t = tex.astype(np.float32)
    top = t[y0, x0] * (1 - wx) + t[y0, x1] * wx
    bot = t[y1, x0] * (1 - wx) + t[y1, x1] * wx
    return np.clip(top * (1 - wy) + bot * wy, 0, 255).astype(np.uint8)


class BD_BakeVertexColorsFromTexture(io.ComfyNode):
    """Sample a diffuse/atlas through the mesh UVs into per-vertex COLOR_0 (non-destructive)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_BakeVertexColorsFromTexture",
            display_name="BD Bake Vertex Colors From Texture",
            category="🧠BrainDead/Mesh",
            description="Sample a texture through a mesh's UVs into per-vertex colors (COLOR_0). "
                        "Runs on ANY UV'd mesh. If no texture is wired, uses the mesh's own "
                        "baseColorTexture. Keeps the UV + baked material intact (additive), so the "
                        "mesh ends up carrying both the texture and matching vertex colors. Wire the "
                        "result into BD Export Mesh to get a glb with texture + COLOR_0.",
            inputs=[
                TrimeshInput("mesh"),
                io.Image.Input("texture", optional=True,
                               tooltip="Diffuse/atlas to sample (e.g. BD_OVoxelBake 'diffuse'). "
                                       "If omitted, the mesh's own baseColorTexture is used."),
                io.Combo.Input("sampling", options=["bilinear", "nearest"], default="bilinear",
                               optional=True, tooltip="Bilinear is smoother; nearest is exact texel."),
            ],
            outputs=[
                TrimeshOutput(display_name="mesh"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, mesh, texture=None, sampling: str = "bilinear") -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "ERROR: trimesh not installed")
        if mesh is None:
            return io.NodeOutput(None, "ERROR: mesh is None")

        # Resolve UVs (required) without rasterizing the visual.
        vis = getattr(mesh, "visual", None)
        uv = getattr(vis, "uv", None) if vis is not None else None
        if uv is None or len(uv) != len(mesh.vertices):
            return io.NodeOutput(mesh, "ERROR: mesh has no per-vertex UVs (need a UV-unwrapped/"
                                       "textured mesh, e.g. from BD_OVoxelBake)")
        uv = np.asarray(uv, dtype=np.float32)

        # Resolve texture: explicit input wins, else the mesh's own baseColorTexture.
        tex = _image_to_np(texture)
        src = "wired texture"
        if tex is None:
            tex = _texture_from_mesh(mesh)
            src = "mesh baseColorTexture"
        if tex is None:
            return io.NodeOutput(mesh, "ERROR: no texture wired and mesh has no baseColorTexture")

        colors = _sample(tex, uv, bilinear=(sampling != "nearest"))  # (N,3) uint8
        rgba = np.concatenate([colors, np.full((len(colors), 1), 255, np.uint8)], axis=1)

        # Additive + non-destructive: copy geometry, keep the original visual (UV + material),
        # attach COLOR_0. We do NOT replace visual with ColorVisuals (that would drop UV/texture).
        out = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
        out.visual = mesh.visual  # preserve UV + baked material
        out.vertex_attributes["COLOR_0"] = rgba

        status = (f"Baked COLOR_0 from {src} ({tex.shape[1]}x{tex.shape[0]}, {sampling}) "
                  f"onto {len(rgba):,} verts; UV + material kept")
        print(f"[BD BakeVertexColors] {status}")
        return io.NodeOutput(out, status)


BAKE_VERTEX_COLORS_V3_NODES = [BD_BakeVertexColorsFromTexture]
BAKE_VERTEX_COLORS_NODES = {"BD_BakeVertexColorsFromTexture": BD_BakeVertexColorsFromTexture}
BAKE_VERTEX_COLORS_DISPLAY_NAMES = {
    "BD_BakeVertexColorsFromTexture": "BD Bake Vertex Colors From Texture",
}
