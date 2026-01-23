"""
BD_ApplyColorField - Apply COLOR_FIELD data to any mesh.

Takes a COLOR_FIELD (voxelgrid color data) and applies it to a mesh at any
pipeline stage. This allows preserving original voxelgrid colors through
topology-changing operations like decimation, remeshing, etc.

Workflow:
1. BD_SampleVoxelgridColors outputs COLOR_FIELD
2. Run mesh through decimation, remeshing, etc.
3. BD_ApplyColorField applies original colors to modified mesh
"""

import numpy as np
import time

from comfy_api.latest import io

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

from .types import TrimeshInput, TrimeshOutput, ColorFieldInput


def split_vertices_by_face_with_colors(mesh, face_colors):
    """
    Split vertices so each face has its own unique vertices.
    This prevents color bleeding at shared vertices.
    """
    faces = mesh.faces
    vertices = mesh.vertices
    normals = mesh.vertex_normals if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None else None

    num_faces = len(faces)

    # Create new vertices - 3 unique vertices per triangle face
    new_vertices = vertices[faces.flatten()].reshape(-1, 3)
    new_faces = np.arange(num_faces * 3).reshape(-1, 3)

    # Create vertex colors - same color for all 3 vertices of each face
    new_vertex_colors = np.repeat(face_colors, 3, axis=0)

    # Handle normals if present
    new_normals = None
    if normals is not None:
        new_normals = normals[faces.flatten()].reshape(-1, 3)

    # Create new mesh with split vertices
    new_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        process=False
    )

    # Set vertex colors
    vertex_colors_uint8 = (new_vertex_colors * 255).clip(0, 255).astype(np.uint8)
    new_mesh.visual = trimesh.visual.ColorVisuals(
        mesh=new_mesh,
        vertex_colors=vertex_colors_uint8
    )

    if new_normals is not None:
        new_mesh.vertex_normals = new_normals

    return new_mesh


class BD_ApplyColorField(io.ComfyNode):
    """
    Apply COLOR_FIELD data to any mesh.

    This allows reapplying original voxelgrid colors after topology-changing
    operations like decimation, remeshing, etc.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_ApplyColorField",
            display_name="BD Apply Color Field",
            category="🧠BrainDead/Mesh",
            description="""Apply COLOR_FIELD (voxelgrid colors) to any mesh.

Use this to reapply original colors after mesh processing:
1. BD_SampleVoxelgridColors outputs COLOR_FIELD
2. Process mesh (decimate, remesh, etc.)
3. BD_ApplyColorField applies colors to processed mesh

Modes:
- face: 1 solid color per face, NO bleeding (works with edge marking)
- sharp: Per-vertex nearest neighbor
- smooth: Per-vertex k=4 weighted blend""",
            inputs=[
                TrimeshInput("mesh"),
                ColorFieldInput("color_field", optional=False),
                io.Combo.Input(
                    "sampling_mode",
                    options=["face", "sharp", "smooth"],
                    default="face",
                    optional=True,
                    tooltip="face=per-face solid | sharp=nearest | smooth=blended",
                ),
                io.Boolean.Input(
                    "preserve_material",
                    default=True,
                    optional=True,
                    tooltip="Save original UVs in metadata for downstream PBR reconstruction. Colors always applied natively to mesh.visual.vertex_colors.",
                ),
                io.String.Input(
                    "default_color",
                    default="0.5,0.5,0.5,1.0",
                    optional=True,
                    tooltip="RGBA for vertices beyond distance threshold",
                ),
                io.Float.Input(
                    "distance_threshold",
                    default=3.0,
                    min=0.5,
                    max=20.0,
                    step=0.5,
                    optional=True,
                    tooltip="Max voxel-units distance before using default color",
                ),
            ],
            outputs=[
                TrimeshOutput(display_name="mesh"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        color_field,
        sampling_mode: str = "face",
        preserve_material: bool = True,
        default_color: str = "0.5,0.5,0.5,1.0",
        distance_threshold: float = 3.0,
    ) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "ERROR: trimesh not installed")

        if mesh is None:
            return io.NodeOutput(None, "ERROR: mesh is None")

        if color_field is None:
            return io.NodeOutput(mesh, "ERROR: color_field is None")

        start_time = time.time()

        # Parse default color
        try:
            default_rgba = np.array([float(x.strip()) for x in default_color.split(",")][:4], dtype=np.float32)
            if len(default_rgba) == 3:
                default_rgba = np.append(default_rgba, 1.0)
        except Exception:
            default_rgba = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)

        # Extract color field data
        positions = color_field.get('positions')
        colors = color_field.get('colors')
        voxel_size = color_field.get('voxel_size', 1.0)

        if positions is None or colors is None:
            return io.NodeOutput(mesh, "ERROR: color_field missing positions or colors")

        # Ensure numpy arrays
        if hasattr(positions, 'cpu'):
            positions = positions.cpu().numpy()
        positions = np.asarray(positions, dtype=np.float32)

        if hasattr(colors, 'cpu'):
            colors = colors.cpu().numpy()
        colors = np.asarray(colors, dtype=np.float32)

        # Add alpha if not present
        if colors.shape[1] == 3:
            alpha = np.ones((len(colors), 1), dtype=np.float32)
            colors = np.hstack([colors, alpha])

        print(f"[BD ApplyColorField] Color field: {len(positions)} voxels, voxel_size={voxel_size}")

        # Get mesh data
        mesh_verts = np.array(mesh.vertices, dtype=np.float32)
        mesh_faces = mesh.faces if hasattr(mesh, 'faces') and mesh.faces is not None else None

        print(f"[BD ApplyColorField] Mesh: {len(mesh_verts)} vertices, {len(mesh_faces) if mesh_faces is not None else 0} faces")

        # Apply Y/Z axis swap for TRELLIS coordinate system
        mesh_verts_transformed = mesh_verts.copy()
        mesh_verts_transformed[:, 1] = -mesh_verts[:, 2]
        mesh_verts_transformed[:, 2] = mesh_verts[:, 1]

        # Offset to match voxel space
        mesh_in_voxel_space = mesh_verts_transformed + 0.5

        # Build KD-tree from color field positions
        from scipy.spatial import cKDTree
        tree = cKDTree(positions)

        max_dist = voxel_size * distance_threshold

        # Check if mesh has TextureVisuals (PBR material + UVs)
        has_material = (
            hasattr(mesh, 'visual')
            and hasattr(mesh.visual, 'material')
            and mesh.visual.material is not None
        )
        if has_material and preserve_material:
            print(f"[BD ApplyColorField] Mesh has PBR material - preserving UVs in metadata, applying colors natively")

        if sampling_mode == "face" and mesh_faces is not None:
            # Face mode: sample at face centers
            face_centers = mesh_in_voxel_space[mesh_faces].mean(axis=1)
            num_faces = len(face_centers)

            print(f"[BD ApplyColorField] Face mode: sampling {num_faces} face centers...")

            distances, indices = tree.query(face_centers, k=1, workers=-1)
            face_colors = colors[indices].copy()

            # Apply distance threshold
            far_faces = distances > max_dist
            face_colors[far_faces] = default_rgba

            total_time = time.time() - start_time
            unique_colors = len(np.unique(face_colors.view(np.float32).reshape(-1, 4), axis=0))
            far_pct = 100 * far_faces.sum() / num_faces

            if has_material and preserve_material:
                # Apply colors natively via split vertices (solid per-face colors)
                # Store old UVs in metadata for potential downstream PBR reconstruction
                new_mesh = split_vertices_by_face_with_colors(mesh, face_colors)
                if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                    new_mesh.metadata['preserved_uvs'] = mesh.visual.uv.copy()
                if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                    new_mesh.metadata['had_pbr_material'] = True

                status = f"Face mode (UVs preserved in metadata): {num_faces} faces ({unique_colors} unique colors), {far_pct:.1f}% beyond threshold, {total_time:.1f}s"
            else:
                # Create face-split mesh with solid colors (replaces material)
                new_mesh = split_vertices_by_face_with_colors(mesh, face_colors)
                status = f"Face mode: {num_faces} faces ({unique_colors} unique colors), {far_pct:.1f}% beyond threshold, {total_time:.1f}s"

            print(f"[BD ApplyColorField] {status}")
            return io.NodeOutput(new_mesh, status)

        elif sampling_mode == "sharp":
            # Sharp mode: k=1 nearest neighbor per vertex
            print(f"[BD ApplyColorField] Sharp mode: k=1 nearest neighbor...")

            distances, indices = tree.query(mesh_in_voxel_space, k=1, workers=-1)
            vertex_colors = colors[indices].copy()

            far_verts = distances > max_dist
            vertex_colors[far_verts] = default_rgba
            far_pct = 100 * far_verts.sum() / len(mesh_verts)

            vertex_colors_uint8 = (vertex_colors * 255).clip(0, 255).astype(np.uint8)

            new_mesh = trimesh.Trimesh(
                vertices=mesh.vertices.copy(),
                faces=mesh_faces.copy() if mesh_faces is not None else None,
                process=False,
            )
            new_mesh.visual = trimesh.visual.ColorVisuals(
                mesh=new_mesh,
                vertex_colors=vertex_colors_uint8,
            )

            if has_material and preserve_material:
                # Store old UVs in metadata for potential downstream PBR reconstruction
                if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                    new_mesh.metadata['preserved_uvs'] = mesh.visual.uv.copy()
                if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                    new_mesh.metadata['had_pbr_material'] = True

            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                new_mesh.vertex_normals = mesh.vertex_normals.copy()

            total_time = time.time() - start_time
            unique_colors = len(np.unique(vertex_colors_uint8.view(np.uint32).reshape(-1)))

            mat_note = " (UVs preserved in metadata)" if (has_material and preserve_material) else ""
            status = f"Sharp mode{mat_note}: {len(mesh_verts)} verts ({unique_colors} unique colors), {far_pct:.1f}% beyond threshold, {total_time:.1f}s"
            print(f"[BD ApplyColorField] {status}")

            return io.NodeOutput(new_mesh, status)

        else:
            # Smooth mode: k=4 weighted average
            print(f"[BD ApplyColorField] Smooth mode: k=4 weighted...")

            distances, indices = tree.query(mesh_in_voxel_space, k=4, workers=-1)

            distances_safe = np.maximum(distances, 1e-10)
            weights = 1.0 / distances_safe
            weights = weights / weights.sum(axis=1, keepdims=True)

            vertex_colors = np.zeros((len(mesh_verts), 4), dtype=np.float32)
            for i in range(4):
                vertex_colors += colors[indices[:, i]] * weights[:, i:i+1]

            far_verts = distances[:, 0] > max_dist
            vertex_colors[far_verts] = default_rgba
            far_pct = 100 * far_verts.sum() / len(mesh_verts)

            vertex_colors_uint8 = (vertex_colors * 255).clip(0, 255).astype(np.uint8)

            new_mesh = trimesh.Trimesh(
                vertices=mesh.vertices.copy(),
                faces=mesh_faces.copy() if mesh_faces is not None else None,
                process=False,
            )
            new_mesh.visual = trimesh.visual.ColorVisuals(
                mesh=new_mesh,
                vertex_colors=vertex_colors_uint8,
            )

            if has_material and preserve_material:
                # Store old UVs in metadata for potential downstream PBR reconstruction
                if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                    new_mesh.metadata['preserved_uvs'] = mesh.visual.uv.copy()
                if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                    new_mesh.metadata['had_pbr_material'] = True

            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                new_mesh.vertex_normals = mesh.vertex_normals.copy()

            total_time = time.time() - start_time
            unique_colors = len(np.unique(vertex_colors_uint8.view(np.uint32).reshape(-1)))

            mat_note = " (UVs preserved in metadata)" if (has_material and preserve_material) else ""
            status = f"Smooth mode{mat_note}: {len(mesh_verts)} verts ({unique_colors} unique colors), {far_pct:.1f}% beyond threshold, {total_time:.1f}s"
            print(f"[BD ApplyColorField] {status}")

            return io.NodeOutput(new_mesh, status)


# V3 node list
COLOR_FIELD_V3_NODES = [BD_ApplyColorField]

# V1 compatibility
COLOR_FIELD_NODES = {
    "BD_ApplyColorField": BD_ApplyColorField,
}

COLOR_FIELD_DISPLAY_NAMES = {
    "BD_ApplyColorField": "BD Apply Color Field",
}
