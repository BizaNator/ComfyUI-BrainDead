"""
Mesh preview / thumbnailing nodes.

- BD_MeshPreview : render a TRIMESH_LIST (e.g. CubePart parts) — or a single
  TRIMESH — to a labeled contact-sheet IMAGE (+ a per-part IMAGE batch) using
  pyrender's headless EGL renderer. One image, all parts at a glance.
- BD_Preview3D  : export a (color-coded) mesh to a temp GLB and show it in the
  same interactive three.js viewer BD_MeshInspector uses. Feed CubePart `parts`
  (auto per-part colors) or any TRIMESH.

EGL offscreen rendering is selected before pyrender is imported so this works on
a headless server (verified stable alongside torch CUDA).
"""

import colorsys
import hashlib
import math
import os

import numpy as np
import torch

from comfy_api.latest import io

from .types import TrimeshInput

# Force headless EGL for pyrender (must be set before pyrender/OpenGL import).
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

_HAS_PYRENDER = None  # resolved lazily on first render


def _pyrender():
    global _HAS_PYRENDER
    import pyrender  # noqa
    _HAS_PYRENDER = True
    return pyrender


def _palette_rgba(n: int):
    out = []
    for i in range(max(n, 1)):
        h = (i / max(n, 1)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.95)
        out.append([int(r * 255), int(g * 255), int(b * 255), 255])
    return out


def _gather_meshes(meshes, mesh):
    """Flatten a TRIMESH_LIST + optional single TRIMESH into a list of Trimesh."""
    out = []
    if meshes:
        for m in meshes:
            if isinstance(m, trimesh.Trimesh) and len(m.vertices):
                out.append(m)
    if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices):
        out.append(mesh)
    return out


def _look_at(eye, target, up=(0.0, 1.0, 0.0)):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    z = eye - target
    z /= (np.linalg.norm(z) + 1e-9)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0] = x
    m[:3, 1] = y
    m[:3, 2] = z
    m[:3, 3] = eye
    return m


def _shade_mesh(src: "trimesh.Trimesh", shading: str, color_rgba):
    """Return a copy normalized to a unit box with per-vertex colors applied."""
    m = src.copy()
    verts = np.asarray(m.vertices, dtype=np.float64)
    center = (verts.min(0) + verts.max(0)) * 0.5
    extent = float((verts.max(0) - verts.min(0)).max()) or 1.0
    m.vertices = (verts - center) / extent  # fit in ~[-0.5, 0.5]

    if shading == "textured":
        # Real texture first: keep the mesh's UV + baseColor material so pyrender renders the
        # actual texture (full atlas resolution). Fall back to vertex colors if there's no
        # usable texture (or only a 1x1 placeholder), then to solid.
        import trimesh.visual as _tv
        vis = getattr(src, "visual", None)
        mat = getattr(vis, "material", None)
        uv = getattr(vis, "uv", None)
        bc = getattr(mat, "baseColorTexture", None) if mat is not None else None
        if (isinstance(vis, _tv.TextureVisuals) and bc is not None and uv is not None
                and min(bc.size) > 2):
            return m  # textured — visual (UV + material) survives the vertex normalization
        shading = "vertex_colors"  # no usable texture → fall back

    if shading == "normals":
        n = np.asarray(m.vertex_normals, dtype=np.float64)
        cols = ((n * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        cols = np.concatenate([cols, np.full((len(cols), 1), 255, np.uint8)], axis=1)
        m.visual.vertex_colors = cols
    elif shading == "solid":
        m.visual.vertex_colors = np.tile([200, 200, 205, 255], (len(m.vertices), 1))
    elif shading == "vertex_colors":
        # Render the mesh's OWN colors (the textured/character look) for a DAM-ready preview.
        # Read COLOR_0 from vertex_attributes first — accessing TextureVisuals.vertex_colors
        # directly would rasterize the texture and is slow/lossy.
        import trimesh.visual as _tv
        cols = None
        try:
            if hasattr(src, "vertex_attributes") and "COLOR_0" in src.vertex_attributes:
                cols = np.asarray(src.vertex_attributes["COLOR_0"])
            elif src.visual is not None and not isinstance(src.visual, _tv.TextureVisuals):
                vc = src.visual.vertex_colors
                if vc is not None:
                    cols = np.asarray(vc)
        except Exception:
            cols = None
        if cols is not None and len(cols) == len(m.vertices):
            if cols.dtype != np.uint8:
                cols = (np.clip(cols, 0, 1) * 255).astype(np.uint8) if cols.max() <= 1.0 else cols.astype(np.uint8)
            if cols.shape[1] == 3:
                cols = np.concatenate([cols, np.full((len(cols), 1), 255, np.uint8)], axis=1)
            m.visual.vertex_colors = cols
        else:  # no vertex colors on the mesh — fall back to neutral solid
            m.visual.vertex_colors = np.tile([200, 200, 205, 255], (len(m.vertices), 1))
    else:  # part_colors
        m.visual.vertex_colors = np.tile(np.asarray(color_rgba, np.uint8), (len(m.vertices), 1))
    return m


def _bg_color(name: str):
    return {
        "dark": [0.10, 0.10, 0.12, 1.0],
        "white": [1.0, 1.0, 1.0, 1.0],
        "transparent": [0.0, 0.0, 0.0, 0.0],
    }.get(name, [0.10, 0.10, 0.12, 1.0])


def _render_one(pyrender, m, tile, shading, bg, az, el, color_rgba):
    """Render a single normalized mesh to an (tile, tile, 3) uint8 array."""
    sm = _shade_mesh(m, shading, color_rgba)
    use_vc = sm.visual.kind == "vertex" if hasattr(sm.visual, "kind") else True
    pm = pyrender.Mesh.from_trimesh(sm, smooth=(shading != "normals"))
    scene = pyrender.Scene(bg_color=_bg_color(bg), ambient_light=[0.35, 0.35, 0.35])
    scene.add(pm)

    radius = float(np.linalg.norm(np.asarray(sm.vertices), axis=1).max()) or 0.6
    yfov = math.pi / 4.0
    dist = radius / math.tan(yfov / 2.0) * 1.4
    a, e = math.radians(az), math.radians(el)
    eye = np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)]) * dist
    pose = _look_at(eye, [0, 0, 0])
    scene.add(pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0), pose=pose)
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.0), pose=pose)
    # second fill light from the opposite side
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.5),
              pose=_look_at(-eye, [0, 0, 0]))

    r = pyrender.OffscreenRenderer(tile, tile)
    try:
        flags = pyrender.RenderFlags.RGBA if bg == "transparent" else pyrender.RenderFlags.NONE
        color, _ = r.render(scene, flags=flags)
    finally:
        r.delete()
    return color[..., :3].astype(np.uint8)


def _label_tile(arr_u8, text, bg):
    if not HAS_PIL or not text:
        return arr_u8
    img = Image.fromarray(arr_u8)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    w, h = img.size
    pad = 3
    tb = d.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    d.rectangle([0, h - th - 2 * pad, min(w, tw + 2 * pad), h], fill=(0, 0, 0))
    d.text((pad, h - th - pad), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def _to_image_tensor(arr_list):
    """List of (H,W,3) uint8 → IMAGE batch tensor (N,H,W,3) float32 [0,1]."""
    stack = np.stack(arr_list, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(stack)


class BD_MeshPreview(io.ComfyNode):
    """
    Render meshes to a labeled contact-sheet thumbnail grid.

    Wire CubePart `parts` (TRIMESH_LIST) to see every segment at once, each
    colorized and labeled (wire `part_names` into `labels`). Also accepts a
    single TRIMESH. Outputs the montage `grid` (IMAGE) plus a per-mesh
    `thumbnails` IMAGE batch for downstream use. Headless GPU (EGL) render.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_MeshPreview",
            display_name="BD Mesh Preview (Thumbnails)",
            category="🧠BrainDead/Mesh",
            is_output_node=True,
            description="Render a TRIMESH_LIST (e.g. CubePart parts) or a TRIMESH to a labeled "
                        "contact-sheet IMAGE + per-mesh IMAGE batch. Shows the grid inline.",
            inputs=[
                # Required first, then optional — keeps frontend widget order
                # (definition order) aligned with required-then-optional.
                io.Int.Input("tile_size", default=256, min=64, max=1024, step=32),
                io.Int.Input("columns", default=0, min=0, max=16,
                    tooltip="Grid columns. 0 = auto (square-ish)."),
                io.Combo.Input("shading", options=["textured", "vertex_colors", "part_colors", "normals", "solid"],
                               default="textured"),
                io.Combo.Input("background", options=["dark", "white", "transparent"],
                               default="dark"),
                io.Float.Input("azimuth", default=35.0, min=-180.0, max=180.0, step=5.0),
                io.Float.Input("elevation", default=20.0, min=-89.0, max=89.0, step=5.0),
                io.Custom("TRIMESH_LIST").Input("meshes", optional=True,
                    tooltip="List of meshes (e.g. CubePart `parts`)."),
                TrimeshInput("mesh", optional=True),
                io.String.Input("labels", default="", multiline=True, optional=True,
                    tooltip="Optional per-mesh labels (newline-separated), e.g. CubePart `part_names`."),
                io.Int.Input("views", default=1, min=1, max=8, optional=True,
                    tooltip="For a SINGLE mesh: render this many angles (turnaround) into the grid — "
                            "e.g. 4 = front/right/back/left. 1 = single view. Ignored for a mesh list."),
            ],
            outputs=[
                io.Image.Output(display_name="grid"),
                io.Image.Output(display_name="thumbnails"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, meshes=None, mesh=None, labels="", tile_size=256, columns=0,
                shading="part_colors", background="dark", azimuth=35.0,
                elevation=20.0, views=1) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput(_blank(), _blank(), "ERROR: trimesh not installed")
        items = _gather_meshes(meshes, mesh)
        if not items:
            return io.NodeOutput(_blank(), _blank(), "No meshes to preview.")

        MAX = 64
        dropped = max(0, len(items) - MAX)
        items = items[:MAX]
        names = [s.strip() for s in labels.split("\n")] if labels else []
        palette = _palette_rgba(len(items))

        try:
            pr = _pyrender()
        except Exception as e:
            return io.NodeOutput(_blank(), _blank(), f"ERROR: pyrender unavailable ({e})")

        # Turnaround: for a SINGLE mesh, render `views` evenly-spaced angles into the grid
        # (e.g. 4 = front/right/back/left). For a mesh list (CubePart), 1 view per part.
        n_views = max(1, int(views)) if len(items) == 1 else 1
        tiles = []
        for i, m in enumerate(items):
            for v in range(n_views):
                az = azimuth + v * (360.0 / n_views)
                try:
                    arr = _render_one(pr, m, tile_size, shading, background,
                                      az, elevation, palette[i])
                except Exception as e:
                    arr = np.zeros((tile_size, tile_size, 3), np.uint8)
                    print(f"[BD MeshPreview] render failed (mesh {i}, view {v}): {e}", flush=True)
                # Explicit label wins; auto "part N" only for multi-mesh grids (e.g. CubePart).
                # Turnaround angles + a single unlabeled mesh stay clean (DAM/catalog thumbnail).
                if n_views == 1 and i < len(names):
                    label = names[i]
                elif n_views == 1 and len(items) > 1:
                    label = f"part {i}"
                else:
                    label = ""
                tiles.append(_label_tile(arr, label, background))

        thumbs = _to_image_tensor(tiles)

        n = len(tiles)
        cols = columns if columns > 0 else max(1, int(math.ceil(math.sqrt(n))))
        rows = int(math.ceil(n / cols))
        t = tile_size
        canvas = np.zeros((rows * t, cols * t, 3), np.uint8)
        if background == "white":
            canvas[:] = 255
        elif background == "dark":
            canvas[:] = (26, 26, 31)
        for idx, tile in enumerate(tiles):
            rr, cc = divmod(idx, cols)
            canvas[rr * t:(rr + 1) * t, cc * t:(cc + 1) * t] = tile
        grid = _to_image_tensor([canvas])

        status = f"Previewed {n} mesh(es), {cols}×{rows} grid @ {t}px, shading={shading}"
        if dropped:
            status += f" (capped, {dropped} not shown)"
        print(f"[BD MeshPreview] {status}", flush=True)

        # Save the grid to temp so it renders inline in the node (PreviewImage-style).
        ui = {}
        if HAS_PIL:
            try:
                import folder_paths
                tdir = folder_paths.get_temp_directory()
                os.makedirs(tdir, exist_ok=True)
                fid = hashlib.sha1(canvas.tobytes()).hexdigest()[:16]
                fname = f"bd_meshpreview_{fid}.png"
                Image.fromarray(canvas).save(os.path.join(tdir, fname))
                ui = {"images": [{"filename": fname, "subfolder": "", "type": "temp"}]}
            except Exception as e:
                print(f"[BD MeshPreview] inline preview save failed: {e}", flush=True)

        return io.NodeOutput(grid, thumbs, status, ui=ui)


class BD_Preview3D(io.ComfyNode):
    """
    Interactive 3D preview in the BrainDead three.js viewer.

    Exports the mesh to a temp GLB and shows it in the same in-node viewer as
    BD_MeshInspector. Wire CubePart `parts` (TRIMESH_LIST) to view all segments
    color-coded in one rotatable scene, or a single TRIMESH (e.g. `combined`).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_Preview3D",
            display_name="BD Preview 3D",
            category="🧠BrainDead/Mesh",
            is_output_node=True,
            description="Show a mesh (or CubePart parts, auto color-coded) in the in-node "
                        "three.js 3D viewer.",
            inputs=[
                io.Custom("TRIMESH_LIST").Input("meshes", optional=True,
                    tooltip="List of meshes (e.g. CubePart `parts`); colored per-part."),
                TrimeshInput("mesh", optional=True),
                io.Boolean.Input("color_parts", default=True, optional=True,
                    tooltip="Assign a distinct color per part when a list is given."),
            ],
            outputs=[
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, meshes=None, mesh=None, color_parts=True) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput("ERROR: trimesh not installed")

        items = _gather_meshes(meshes, mesh)
        if not items:
            return io.NodeOutput("No mesh to preview.")

        if len(items) == 1 and not (meshes and color_parts):
            combined = items[0].copy()
        else:
            palette = _palette_rgba(len(items))
            colored = []
            for i, m in enumerate(items):
                c = m.copy()
                if color_parts:
                    c.visual.vertex_colors = np.tile(np.asarray(palette[i], np.uint8),
                                                     (len(c.vertices), 1))
                colored.append(c)
            combined = trimesh.util.concatenate(colored)

        has_colors = bool(getattr(getattr(combined, "visual", None), "kind", "") == "vertex")

        import folder_paths
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        h = hashlib.sha1(np.asarray(combined.vertices, np.float32).tobytes()
                         + np.asarray(combined.faces).tobytes()).hexdigest()[:16]
        filename = f"bd_preview3d_{h}.glb"
        filepath = os.path.join(temp_dir, filename)
        try:
            combined.export(filepath, file_type="glb")
        except Exception as e:
            return io.NodeOutput(f"ERROR: export failed - {e}")

        status = (f"Preview: {len(items)} mesh(es), "
                  f"{len(combined.vertices):,} verts, {len(combined.faces):,} faces")
        print(f"[BD Preview3D] {status} -> {filename}", flush=True)
        return io.NodeOutput(
            status,
            ui={
                "mesh_file": [filename],
                "view_type": ["temp"],
                "subfolder": [""],
                # Flat per-part colors render reliably via vertex_colors mode
                # (MeshBasicMaterial); full_material leans on the glb PBR material.
                "initial_mode": ["vertex_colors" if has_colors else "full_material"],
                "has_colors": [has_colors],
                "has_uvs": [False],
            },
        )


def _blank():
    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


MESH_PREVIEW_V3_NODES = [BD_MeshPreview, BD_Preview3D]

MESH_PREVIEW_NODES = {
    "BD_MeshPreview": BD_MeshPreview,
    "BD_Preview3D": BD_Preview3D,
}

MESH_PREVIEW_DISPLAY_NAMES = {
    "BD_MeshPreview": "BD Mesh Preview (Thumbnails)",
    "BD_Preview3D": "BD Preview 3D",
}
