"""
V3 API Voxelgrid sampling nodes for extracting colors/PBR from TRELLIS2 voxelgrids.

BD_SampleVoxelgridColors - Sample colors from voxelgrid to mesh vertices
BD_SampleVoxelgridPBR - Sample full PBR attributes (color, metallic, roughness)
"""

import numpy as np
import time

from comfy_api.latest import io

# Check for optional trimesh support
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


class BD_SampleVoxelgridColors(io.ComfyNode):
    """
    Sample colors from TRELLIS2 voxelgrid directly to mesh vertices.

    This is the CORRECT way to get colors from TRELLIS2 - uses the voxelgrid
    structure directly instead of the misaligned pointcloud.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_SampleVoxelgridColors",
            display_name="BD Sample Voxelgrid Colors",
            category="🧠BrainDead/Mesh",
            description="Sample colors from TRELLIS2 voxelgrid to mesh vertices. Modes: smooth (k=4 weighted), sharp (k=1 nearest), face (per-face).",
            inputs=[
                io.Mesh.Input("mesh"),
                io.Custom("TRELLIS2_VOXELGRID").Input("voxelgrid"),
                io.Combo.Input("sampling_mode", options=["smooth", "sharp", "face"], default="smooth", optional=True,
                              tooltip="smooth=k=4 weighted, sharp=k=1 nearest, face=per-face colors"),
                io.String.Input("default_color", default="0.5,0.5,0.5,1.0", optional=True),
                io.Float.Input("distance_threshold", default=3.0, min=1.0, max=10.0, step=0.5, optional=True,
                              tooltip="Max voxels distance before using default color"),
            ],
            outputs=[
                io.Mesh.Output(display_name="mesh"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, mesh, voxelgrid, sampling_mode: str = "smooth",
                default_color: str = "0.5,0.5,0.5,1.0",
                distance_threshold: float = 3.0) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "ERROR: trimesh not installed")

        if mesh is None:
            return io.NodeOutput(None, "ERROR: mesh is None")

        if voxelgrid is None or not isinstance(voxelgrid, dict):
            return io.NodeOutput(mesh, "ERROR: voxelgrid is None or invalid")

        start_time = time.time()

        # Parse default color
        try:
            default_rgba = np.array([float(x.strip()) for x in default_color.split(",")][:4], dtype=np.float32)
            if len(default_rgba) == 3:
                default_rgba = np.append(default_rgba, 1.0)
        except:
            default_rgba = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)

        # Extract voxelgrid data
        coords = voxelgrid.get('coords')
        attrs = voxelgrid.get('attrs')
        voxel_size = voxelgrid.get('voxel_size', 1.0)
        layout = voxelgrid.get('layout', {})

        if coords is None or attrs is None:
            return io.NodeOutput(mesh, "ERROR: voxelgrid missing coords or attrs")

        print(f"[BD Sample Voxelgrid] Voxelgrid: {len(coords)} voxels, voxel_size={voxel_size}")
        print(f"[BD Sample Voxelgrid] Attrs shape: {attrs.shape}, layout: {layout}")

        # Get mesh vertices
        mesh_verts = np.array(mesh.vertices, dtype=np.float32)
        num_verts = len(mesh_verts)
        print(f"[BD Sample Voxelgrid] Mesh: {num_verts} vertices")

        # Use KD-tree for sparse voxel sampling
        print(f"[BD Sample Voxelgrid] Using KD-tree sparse sampling, mode={sampling_mode}")
        vertex_colors = cls._sample_with_kdtree(
            mesh_verts, coords, attrs, voxel_size, layout, default_rgba,
            mesh_faces=mesh.faces if hasattr(mesh, 'faces') else None,
            sampling_mode=sampling_mode,
            distance_threshold=distance_threshold
        )

        # Create new mesh with vertex colors
        try:
            vertex_colors_uint8 = (vertex_colors * 255).clip(0, 255).astype(np.uint8)

            if vertex_colors_uint8.shape[1] == 3:
                alpha = np.full((len(vertex_colors_uint8), 1), 255, dtype=np.uint8)
                vertex_colors_uint8 = np.hstack([vertex_colors_uint8, alpha])

            new_mesh = trimesh.Trimesh(
                vertices=mesh.vertices.copy(),
                faces=mesh.faces.copy() if hasattr(mesh, 'faces') and mesh.faces is not None else None,
                process=False
            )

            new_mesh.visual = trimesh.visual.ColorVisuals(
                mesh=new_mesh,
                vertex_colors=vertex_colors_uint8
            )

            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                new_mesh.vertex_normals = mesh.vertex_normals.copy()

            total_time = time.time() - start_time
            vertex_colors_contiguous = np.ascontiguousarray(vertex_colors_uint8)
            unique_colors = len(np.unique(vertex_colors_contiguous.view(np.uint32)))

            status = f"Sampled {num_verts} vertices ({unique_colors} unique colors) in {total_time:.1f}s"
            print(f"[BD Sample Voxelgrid] {status}")

            return io.NodeOutput(new_mesh, status)

        except Exception as e:
            return io.NodeOutput(mesh, f"ERROR creating colored mesh: {e}")

    @classmethod
    def _sample_with_kdtree(cls, mesh_verts, coords, attrs, voxel_size, layout, default_rgba,
                             mesh_faces=None, sampling_mode="smooth", distance_threshold=3.0):
        """Sample colors using KD-tree nearest neighbor lookup on sparse voxels."""
        from scipy.spatial import cKDTree

        # The mesh and voxelgrid have swapped Y/Z axes with negation
        mesh_verts_transformed = mesh_verts.copy()
        mesh_verts_transformed[:, 1] = -mesh_verts[:, 2]
        mesh_verts_transformed[:, 2] = mesh_verts[:, 1]
        print(f"[BD Sample Voxelgrid] Applied Y=-Z, Z=Y transform to mesh")

        # Convert coords to world positions
        if hasattr(coords, 'cpu'):
            coords_np = coords.cpu().numpy()
        else:
            coords_np = np.array(coords)
        voxel_world_positions = coords_np.astype(np.float32) * voxel_size

        mesh_in_voxel_space = mesh_verts_transformed + 0.5

        print(f"[BD Sample Voxelgrid] Building KD-tree from {len(voxel_world_positions)} sparse voxels")

        tree = cKDTree(voxel_world_positions)

        # Get attrs and convert to colors
        if hasattr(attrs, 'cpu'):
            attrs_np = attrs.cpu().numpy()
        else:
            attrs_np = np.array(attrs)

        base_color_slice = layout.get('base_color', slice(0, 3))
        alpha_slice = layout.get('alpha', slice(5, 6))

        if isinstance(base_color_slice, slice):
            rgb_raw = attrs_np[:, base_color_slice]
        else:
            rgb_raw = attrs_np[:, :3]

        rgb = np.clip(rgb_raw, 0, 1)

        if isinstance(alpha_slice, slice):
            alpha_raw = attrs_np[:, alpha_slice].flatten()
            alpha = np.clip(alpha_raw, 0, 1)
        else:
            alpha = np.ones(len(rgb), dtype=np.float32)

        voxel_colors = np.column_stack([rgb, alpha]).astype(np.float32)

        max_dist = voxel_size * distance_threshold

        if sampling_mode == "face" and mesh_faces is not None:
            print(f"[BD Sample Voxelgrid] Face mode: computing {len(mesh_faces)} face centers...")
            face_centers = mesh_in_voxel_space[mesh_faces].mean(axis=1)

            distances, indices = tree.query(face_centers, k=1, workers=-1)
            face_colors = voxel_colors[indices]

            vertex_colors = np.full((len(mesh_in_voxel_space), 4), default_rgba, dtype=np.float32)
            for face_idx, face in enumerate(mesh_faces):
                for vert_idx in face:
                    vertex_colors[vert_idx] = face_colors[face_idx]

            far_faces = distances > max_dist
            print(f"[BD Sample Voxelgrid] Faces beyond {max_dist:.6f} threshold: {far_faces.sum()} ({100*far_faces.sum()/len(far_faces):.1f}%)")

        elif sampling_mode == "sharp":
            print(f"[BD Sample Voxelgrid] Sharp mode: k=1 nearest neighbor...")
            distances, indices = tree.query(mesh_in_voxel_space, k=1, workers=-1)

            vertex_colors = voxel_colors[indices]

            far_vertices = distances > max_dist
            vertex_colors[far_vertices] = default_rgba
            print(f"[BD Sample Voxelgrid] Vertices beyond {max_dist:.6f} threshold: {far_vertices.sum()} ({100*far_vertices.sum()/len(far_vertices):.1f}%)")

        else:  # smooth (default)
            print(f"[BD Sample Voxelgrid] Smooth mode: k=4 weighted...")
            distances, indices = tree.query(mesh_in_voxel_space, k=4, workers=-1)

            distances_safe = np.maximum(distances, 1e-10)
            weights = 1.0 / distances_safe
            weights = weights / weights.sum(axis=1, keepdims=True)

            vertex_colors = np.zeros((len(mesh_in_voxel_space), 4), dtype=np.float32)
            for i in range(4):
                vertex_colors += voxel_colors[indices[:, i]] * weights[:, i:i+1]

            far_vertices = distances[:, 0] > max_dist
            vertex_colors[far_vertices] = default_rgba
            print(f"[BD Sample Voxelgrid] Vertices beyond {max_dist:.6f} threshold: {far_vertices.sum()} ({100*far_vertices.sum()/len(far_vertices):.1f}%)")

        return vertex_colors


class BD_SampleVoxelgridPBR(io.ComfyNode):
    """
    Sample full PBR attributes from TRELLIS2 voxelgrid to mesh vertices.

    Extracts all PBR channels: base_color, metallic, roughness, alpha.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_SampleVoxelgridPBR",
            display_name="BD Sample Voxelgrid PBR",
            category="🧠BrainDead/Mesh",
            description="Sample full PBR attributes (color, metallic, roughness) from TRELLIS2 voxelgrid to mesh vertices.",
            inputs=[
                io.Mesh.Input("mesh"),
                io.Custom("TRELLIS2_VOXELGRID").Input("voxelgrid"),
                io.Combo.Input("sampling_mode", options=["smooth", "sharp", "face"], default="sharp", optional=True,
                              tooltip="smooth=k=4 weighted, sharp=k=1 nearest, face=per-face colors"),
                io.String.Input("default_color", default="0.5,0.5,0.5,1.0", optional=True),
                io.Float.Input("default_metallic", default=0.0, min=0.0, max=1.0, step=0.05, optional=True),
                io.Float.Input("default_roughness", default=0.5, min=0.0, max=1.0, step=0.05, optional=True),
                io.Float.Input("distance_threshold", default=3.0, min=1.0, max=10.0, step=0.5, optional=True,
                              tooltip="Max voxels distance before using default values"),
            ],
            outputs=[
                io.Mesh.Output(display_name="mesh"),
                io.String.Output(display_name="metallic_json"),
                io.String.Output(display_name="roughness_json"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, mesh, voxelgrid, sampling_mode: str = "sharp",
                default_color: str = "0.5,0.5,0.5,1.0", default_metallic: float = 0.0,
                default_roughness: float = 0.5, distance_threshold: float = 3.0) -> io.NodeOutput:
        import json

        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "[]", "[]", "ERROR: trimesh not installed")

        if mesh is None:
            return io.NodeOutput(None, "[]", "[]", "ERROR: mesh is None")

        if voxelgrid is None or not isinstance(voxelgrid, dict):
            return io.NodeOutput(mesh, "[]", "[]", "ERROR: voxelgrid is None or invalid")

        start_time = time.time()

        # Parse default color
        try:
            default_rgba = np.array([float(x.strip()) for x in default_color.split(",")][:4], dtype=np.float32)
            if len(default_rgba) == 3:
                default_rgba = np.append(default_rgba, 1.0)
        except:
            default_rgba = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)

        # Extract voxelgrid data
        coords = voxelgrid.get('coords')
        attrs = voxelgrid.get('attrs')
        voxel_size = voxelgrid.get('voxel_size', 1.0)
        layout = voxelgrid.get('layout', {})

        if coords is None or attrs is None:
            return io.NodeOutput(mesh, "[]", "[]", "ERROR: voxelgrid missing coords or attrs")

        print(f"[BD Sample PBR] Voxelgrid: {len(coords)} voxels, voxel_size={voxel_size}")
        print(f"[BD Sample PBR] Attrs shape: {attrs.shape}, layout: {layout}")

        mesh_verts = np.array(mesh.vertices, dtype=np.float32)
        mesh_faces = mesh.faces if hasattr(mesh, 'faces') else None
        num_verts = len(mesh_verts)
        print(f"[BD Sample PBR] Mesh: {num_verts} vertices")

        vertex_colors, metallic_arr, roughness_arr = cls._sample_pbr_kdtree(
            mesh_verts, coords, attrs, voxel_size, layout,
            default_rgba, default_metallic, default_roughness,
            mesh_faces=mesh_faces,
            sampling_mode=sampling_mode,
            distance_threshold=distance_threshold
        )

        try:
            vertex_colors_uint8 = (vertex_colors * 255).clip(0, 255).astype(np.uint8)
            if vertex_colors_uint8.shape[1] == 3:
                alpha = np.full((len(vertex_colors_uint8), 1), 255, dtype=np.uint8)
                vertex_colors_uint8 = np.hstack([vertex_colors_uint8, alpha])

            new_mesh = trimesh.Trimesh(
                vertices=mesh.vertices.copy(),
                faces=mesh.faces.copy() if mesh_faces is not None else None,
                process=False
            )

            new_mesh.visual = trimesh.visual.ColorVisuals(
                mesh=new_mesh,
                vertex_colors=vertex_colors_uint8
            )

            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                new_mesh.vertex_normals = mesh.vertex_normals.copy()

            total_time = time.time() - start_time

            metallic_json = json.dumps(metallic_arr.tolist())
            roughness_json = json.dumps(roughness_arr.tolist())

            vertex_colors_contiguous = np.ascontiguousarray(vertex_colors_uint8)
            unique_colors = len(np.unique(vertex_colors_contiguous.view(np.uint32)))

            status = (f"Sampled {num_verts} verts ({unique_colors} colors) | "
                      f"Metallic: [{metallic_arr.min():.2f}, {metallic_arr.max():.2f}] | "
                      f"Roughness: [{roughness_arr.min():.2f}, {roughness_arr.max():.2f}] | "
                      f"{total_time:.1f}s")
            print(f"[BD Sample PBR] {status}")

            return io.NodeOutput(new_mesh, metallic_json, roughness_json, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return io.NodeOutput(mesh, "[]", "[]", f"ERROR creating PBR mesh: {e}")

    @classmethod
    def _sample_pbr_kdtree(cls, mesh_verts, coords, attrs, voxel_size, layout,
                           default_rgba, default_metallic, default_roughness,
                           mesh_faces=None, sampling_mode="sharp", distance_threshold=3.0):
        """Sample all PBR channels using KD-tree nearest neighbor lookup."""
        from scipy.spatial import cKDTree

        mesh_verts_transformed = mesh_verts.copy()
        mesh_verts_transformed[:, 1] = -mesh_verts[:, 2]
        mesh_verts_transformed[:, 2] = mesh_verts[:, 1]

        if hasattr(coords, 'cpu'):
            coords_np = coords.cpu().numpy()
        else:
            coords_np = np.array(coords)
        voxel_world_positions = coords_np.astype(np.float32) * voxel_size

        mesh_in_voxel_space = mesh_verts_transformed + 0.5

        print(f"[BD Sample PBR] Building KD-tree from {len(voxel_world_positions)} sparse voxels")

        tree = cKDTree(voxel_world_positions)

        if hasattr(attrs, 'cpu'):
            attrs_np = attrs.cpu().numpy()
        else:
            attrs_np = np.array(attrs)

        base_color_slice = layout.get('base_color', slice(0, 3))
        metallic_slice = layout.get('metallic', slice(3, 4))
        roughness_slice = layout.get('roughness', slice(4, 5))
        alpha_slice = layout.get('alpha', slice(5, 6))

        rgb = np.clip(attrs_np[:, base_color_slice], 0, 1) if isinstance(base_color_slice, slice) else np.clip(attrs_np[:, :3], 0, 1)
        metallic = np.clip(attrs_np[:, metallic_slice].flatten(), 0, 1) if isinstance(metallic_slice, slice) else np.full(len(attrs_np), default_metallic)
        roughness = np.clip(attrs_np[:, roughness_slice].flatten(), 0, 1) if isinstance(roughness_slice, slice) else np.full(len(attrs_np), default_roughness)
        alpha = np.clip(attrs_np[:, alpha_slice].flatten(), 0, 1) if isinstance(alpha_slice, slice) else np.ones(len(attrs_np))

        voxel_colors = np.column_stack([rgb, alpha]).astype(np.float32)

        print(f"[BD Sample PBR] PBR ranges - Metallic: [{metallic.min():.3f}, {metallic.max():.3f}], Roughness: [{roughness.min():.3f}, {roughness.max():.3f}]")

        max_dist = voxel_size * distance_threshold
        num_verts = len(mesh_in_voxel_space)

        if sampling_mode == "face" and mesh_faces is not None:
            face_centers = mesh_in_voxel_space[mesh_faces].mean(axis=1)
            distances, indices = tree.query(face_centers, k=1, workers=-1)

            face_colors = voxel_colors[indices]
            face_metallic = metallic[indices]
            face_roughness = roughness[indices]

            vertex_colors = np.full((num_verts, 4), default_rgba, dtype=np.float32)
            vertex_metallic = np.full(num_verts, default_metallic, dtype=np.float32)
            vertex_roughness = np.full(num_verts, default_roughness, dtype=np.float32)

            for face_idx, face in enumerate(mesh_faces):
                for vert_idx in face:
                    vertex_colors[vert_idx] = face_colors[face_idx]
                    vertex_metallic[vert_idx] = face_metallic[face_idx]
                    vertex_roughness[vert_idx] = face_roughness[face_idx]

            far_faces = distances > max_dist
            print(f"[BD Sample PBR] Face mode: {far_faces.sum()} faces beyond threshold ({100*far_faces.sum()/len(far_faces):.1f}%)")

        elif sampling_mode == "sharp":
            distances, indices = tree.query(mesh_in_voxel_space, k=1, workers=-1)

            vertex_colors = voxel_colors[indices]
            vertex_metallic = metallic[indices]
            vertex_roughness = roughness[indices]

            far_vertices = distances > max_dist
            vertex_colors[far_vertices] = default_rgba
            vertex_metallic[far_vertices] = default_metallic
            vertex_roughness[far_vertices] = default_roughness

            print(f"[BD Sample PBR] Sharp mode: {far_vertices.sum()} verts beyond threshold ({100*far_vertices.sum()/num_verts:.1f}%)")

        else:  # smooth
            distances, indices = tree.query(mesh_in_voxel_space, k=4, workers=-1)

            distances_safe = np.maximum(distances, 1e-10)
            weights = 1.0 / distances_safe
            weights = weights / weights.sum(axis=1, keepdims=True)

            vertex_colors = np.zeros((num_verts, 4), dtype=np.float32)
            vertex_metallic = np.zeros(num_verts, dtype=np.float32)
            vertex_roughness = np.zeros(num_verts, dtype=np.float32)

            for i in range(4):
                vertex_colors += voxel_colors[indices[:, i]] * weights[:, i:i+1]
                vertex_metallic += metallic[indices[:, i]] * weights[:, i]
                vertex_roughness += roughness[indices[:, i]] * weights[:, i]

            far_vertices = distances[:, 0] > max_dist
            vertex_colors[far_vertices] = default_rgba
            vertex_metallic[far_vertices] = default_metallic
            vertex_roughness[far_vertices] = default_roughness

            print(f"[BD Sample PBR] Smooth mode: {far_vertices.sum()} verts beyond threshold ({100*far_vertices.sum()/num_verts:.1f}%)")

        return vertex_colors, vertex_metallic, vertex_roughness


# V3 node list for extension
MESH_SAMPLING_V3_NODES = [BD_SampleVoxelgridColors, BD_SampleVoxelgridPBR]

# V1 compatibility - NODE_CLASS_MAPPINGS dict
MESH_SAMPLING_NODES = {
    "BD_SampleVoxelgridColors": BD_SampleVoxelgridColors,
    "BD_SampleVoxelgridPBR": BD_SampleVoxelgridPBR,
}

MESH_SAMPLING_DISPLAY_NAMES = {
    "BD_SampleVoxelgridColors": "BD Sample Voxelgrid Colors",
    "BD_SampleVoxelgridPBR": "BD Sample Voxelgrid PBR",
}
