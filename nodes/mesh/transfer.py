"""
Color transfer nodes for transferring colors between meshes.

BD_TransferPointcloudColors - Transfer from pointcloud to mesh (deprecated)
BD_TransferColorsPymeshlab - Transfer using pymeshlab
BD_TransferVertexColors - BVH-based vertex color transfer
"""

import numpy as np
import time

# Check for optional trimesh support
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


class BD_TransferPointcloudColors:
    """
    Transfer colors from a pointcloud to mesh vertices using nearest-neighbor lookup.

    This bypasses the TRELLIS2 UV/texture pipeline entirely by directly transferring
    the PBR colors from the pointcloud to vertex colors on the mesh.

    Perfect for: Preparing meshes for vertex-color workflows and decimation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "pointcloud": ("TRIMESH",),  # pbr_pointcloud from TRELLIS2
            },
            "optional": {
                "coordinate_fix": (["auto", "none", "mesh_to_zup", "pointcloud_to_yup"], {
                    "default": "auto",
                    "tooltip": "Fix coordinate mismatch between mesh and pointcloud"}),
                "max_distance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001,
                                           "tooltip": "Max distance for color transfer (0 = unlimited)"}),
                "default_color": ("STRING", {"default": "0.5,0.5,0.5,1.0",
                                             "tooltip": "RGBA color for vertices with no nearby points"}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "status")
    FUNCTION = "transfer_colors"
    CATEGORY = "BrainDead/Mesh"
    DESCRIPTION = """
Transfer colors from pointcloud to mesh vertices.

Use this to bypass TRELLIS2's UV/texture pipeline:
1. Connect 'trimesh' output to mesh input
2. Connect 'pbr_pointcloud' output to pointcloud input
3. Output mesh has vertex colors ready for export/decimation

The node uses KD-tree nearest-neighbor search for fast color transfer,
even on 14M+ polygon meshes (typically <1 minute).

Inputs:
- mesh: Geometry mesh from TRELLIS2 (or any TRIMESH)
- pointcloud: pbr_pointcloud from TRELLIS2 (has colors)
- coordinate_fix: Handle TRELLIS2 coordinate mismatch
  - auto: Auto-detect and fix (recommended)
  - none: No conversion
  - mesh_to_zup: Convert mesh Y-up to Z-up
  - pointcloud_to_yup: Convert pointcloud Z-up back to Y-up
- max_distance: Skip vertices farther than this from any point (0=unlimited)
- default_color: RGBA for vertices with no nearby points

Note: TRELLIS2 outputs mesh in Y-up but pointcloud in Z-up coordinates.
The 'auto' setting detects and fixes this automatically.

Output mesh can be:
- Exported directly with vertex colors (GLB/PLY)
- Passed to Blender decimation with color preservation
- Cached with BD_CacheMesh
"""

    def transfer_colors(self, mesh, pointcloud, coordinate_fix="auto", max_distance=0.0, default_color="0.5,0.5,0.5,1.0"):
        if not HAS_TRIMESH:
            return (mesh, "ERROR: trimesh not installed")

        if mesh is None:
            return (None, "ERROR: mesh is None")

        if pointcloud is None:
            return (mesh, "ERROR: pointcloud is None - no colors to transfer")

        # Parse default color
        try:
            default_rgba = [float(x.strip()) for x in default_color.split(",")]
            if len(default_rgba) == 3:
                default_rgba.append(1.0)
            default_rgba = np.array(default_rgba[:4], dtype=np.float64)
        except:
            default_rgba = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float64)

        start_time = time.time()

        # Get mesh vertices
        mesh_verts = np.array(mesh.vertices, dtype=np.float64)
        num_verts = len(mesh_verts)
        print(f"[BD Transfer Colors] Mesh: {num_verts} vertices")

        # Get pointcloud data
        if hasattr(pointcloud, 'vertices'):
            pc_verts = np.array(pointcloud.vertices, dtype=np.float64)
        else:
            return (mesh, "ERROR: pointcloud has no vertices")

        # Debug: Show bounding boxes BEFORE any conversion
        mesh_min = mesh_verts.min(axis=0)
        mesh_max = mesh_verts.max(axis=0)
        pc_min = pc_verts.min(axis=0)
        pc_max = pc_verts.max(axis=0)
        print(f"[BD Transfer Colors] Mesh bounds: min={mesh_min}, max={mesh_max}")
        print(f"[BD Transfer Colors] Pointcloud bounds: min={pc_min}, max={pc_max}")

        # Check for coordinate mismatch and apply fix
        def apply_yup_to_zup(verts):
            """Convert from Y-up to Z-up (swap Y,Z and negate new Z)"""
            result = verts.copy()
            result[:, 1], result[:, 2] = verts[:, 2].copy(), -verts[:, 1].copy()
            return result

        def apply_zup_to_yup(verts):
            """Convert from Z-up back to Y-up (reverse of yup_to_zup)"""
            result = verts.copy()
            result[:, 1], result[:, 2] = -verts[:, 2].copy(), verts[:, 1].copy()
            return result

        if coordinate_fix == "auto":
            # Auto-detect: check if bounds overlap
            def ranges_overlap(r1, r2, tolerance=0.5):
                return not (r1[1] < r2[0] - tolerance or r2[1] < r1[0] - tolerance)

            current_overlap = (
                ranges_overlap((mesh_min[0], mesh_max[0]), (pc_min[0], pc_max[0])) and
                ranges_overlap((mesh_min[1], mesh_max[1]), (pc_min[1], pc_max[1])) and
                ranges_overlap((mesh_min[2], mesh_max[2]), (pc_min[2], pc_max[2]))
            )

            if current_overlap:
                print(f"[BD Transfer Colors] Auto-detect: Bounds overlap, no conversion needed")
                coordinate_fix = "none"
            else:
                # Try converting pointcloud back to Y-up
                pc_verts_yup = apply_zup_to_yup(pc_verts)
                pc_yup_min = pc_verts_yup.min(axis=0)
                pc_yup_max = pc_verts_yup.max(axis=0)

                converted_overlap = (
                    ranges_overlap((mesh_min[0], mesh_max[0]), (pc_yup_min[0], pc_yup_max[0])) and
                    ranges_overlap((mesh_min[1], mesh_max[1]), (pc_yup_min[1], pc_yup_max[1])) and
                    ranges_overlap((mesh_min[2], mesh_max[2]), (pc_yup_min[2], pc_yup_max[2]))
                )

                if converted_overlap:
                    print(f"[BD Transfer Colors] Auto-detect: Converting pointcloud to Y-up (TRELLIS2 fix)")
                    coordinate_fix = "pointcloud_to_yup"
                else:
                    # Try the other way - convert mesh to Z-up
                    mesh_verts_zup = apply_yup_to_zup(mesh_verts)
                    mesh_zup_min = mesh_verts_zup.min(axis=0)
                    mesh_zup_max = mesh_verts_zup.max(axis=0)

                    mesh_converted_overlap = (
                        ranges_overlap((mesh_zup_min[0], mesh_zup_max[0]), (pc_min[0], pc_max[0])) and
                        ranges_overlap((mesh_zup_min[1], mesh_zup_max[1]), (pc_min[1], pc_max[1])) and
                        ranges_overlap((mesh_zup_min[2], mesh_zup_max[2]), (pc_min[2], pc_max[2]))
                    )

                    if mesh_converted_overlap:
                        print(f"[BD Transfer Colors] Auto-detect: Converting mesh to Z-up")
                        coordinate_fix = "mesh_to_zup"
                    else:
                        print(f"[BD Transfer Colors] Auto-detect: Could not find matching coordinate space, trying pointcloud_to_yup")
                        coordinate_fix = "pointcloud_to_yup"

        # Apply the chosen coordinate fix
        if coordinate_fix == "pointcloud_to_yup":
            pc_verts = apply_zup_to_yup(pc_verts)
            print(f"[BD Transfer Colors] Converted pointcloud to Y-up")
        elif coordinate_fix == "mesh_to_zup":
            mesh_verts = apply_yup_to_zup(mesh_verts)
            print(f"[BD Transfer Colors] Converted mesh to Z-up")

        # Get pointcloud colors
        pc_colors = None
        if hasattr(pointcloud, 'colors') and pointcloud.colors is not None:
            pc_colors = np.array(pointcloud.colors)
            if pc_colors.dtype == np.uint8:
                pc_colors = pc_colors.astype(np.float64) / 255.0

        if pc_colors is None or len(pc_colors) == 0:
            return (mesh, "ERROR: pointcloud has no colors")

        print(f"[BD Transfer Colors] Pointcloud: {len(pc_verts)} points with colors")

        # Build KD-tree for fast nearest-neighbor lookup
        print(f"[BD Transfer Colors] Building KD-tree...")
        tree_start = time.time()

        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pc_verts)
            print(f"[BD Transfer Colors] KD-tree built in {time.time() - tree_start:.2f}s (scipy)")
        except ImportError:
            print(f"[BD Transfer Colors] scipy not available, using numpy fallback (slower)")
            tree = None

        # Query nearest neighbors for all mesh vertices
        print(f"[BD Transfer Colors] Finding nearest neighbors for {num_verts} vertices...")
        query_start = time.time()

        if tree is not None:
            if max_distance > 0:
                distances, indices = tree.query(mesh_verts, k=1, distance_upper_bound=max_distance)
            else:
                distances, indices = tree.query(mesh_verts, k=1)
        else:
            # Numpy fallback - process in batches
            batch_size = 10000
            indices = np.zeros(num_verts, dtype=np.int64)
            distances = np.zeros(num_verts, dtype=np.float64)

            for i in range(0, num_verts, batch_size):
                end_idx = min(i + batch_size, num_verts)
                batch_verts = mesh_verts[i:end_idx]
                dists = np.linalg.norm(pc_verts[np.newaxis, :, :] - batch_verts[:, np.newaxis, :], axis=2)
                batch_indices = np.argmin(dists, axis=1)
                batch_distances = dists[np.arange(len(batch_indices)), batch_indices]
                indices[i:end_idx] = batch_indices
                distances[i:end_idx] = batch_distances

        print(f"[BD Transfer Colors] Nearest neighbor query: {time.time() - query_start:.2f}s")

        # Transfer colors
        vertex_colors = np.zeros((num_verts, 4), dtype=np.float64)
        valid_mask = indices < len(pc_verts)
        if max_distance > 0:
            valid_mask &= distances <= max_distance

        valid_indices = indices[valid_mask]
        vertex_colors[valid_mask] = pc_colors[valid_indices]

        invalid_count = np.sum(~valid_mask)
        if invalid_count > 0:
            vertex_colors[~valid_mask] = default_rgba
            print(f"[BD Transfer Colors] {invalid_count} vertices used default color")

        # Ensure RGBA
        if vertex_colors.shape[1] == 3:
            alpha = np.ones((num_verts, 1), dtype=np.float64)
            vertex_colors = np.concatenate([vertex_colors, alpha], axis=1)

        vertex_colors_uint8 = (vertex_colors * 255).clip(0, 255).astype(np.uint8)

        # Create new mesh with vertex colors
        try:
            new_mesh = trimesh.Trimesh(
                vertices=mesh.vertices.copy(),
                faces=mesh.faces.copy() if hasattr(mesh, 'faces') and mesh.faces is not None else None,
                vertex_colors=vertex_colors_uint8,
                process=False
            )

            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                new_mesh.vertex_normals = mesh.vertex_normals.copy()

            total_time = time.time() - start_time
            unique_colors = len(np.unique(vertex_colors_uint8.view(np.uint32)))
            avg_dist = np.mean(distances[valid_mask]) if np.any(valid_mask) else 0

            status = f"Transferred colors to {num_verts} vertices ({unique_colors} unique colors, avg_dist={avg_dist:.4f}) in {total_time:.1f}s"
            print(f"[BD Transfer Colors] {status}")

            return (new_mesh, status)

        except Exception as e:
            return (mesh, f"ERROR creating colored mesh: {e}")


class BD_TransferColorsPymeshlab:
    """
    Transfer colors from TRELLIS2 pointcloud to mesh using pymeshlab.

    This uses MeshLab's proven point-to-mesh color transfer algorithm.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "pointcloud": ("TRIMESH",),  # pbr_pointcloud from TRELLIS2
            },
            "optional": {
                "max_distance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                                           "tooltip": "Max distance for color transfer. 0 = automatic"}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "status")
    FUNCTION = "transfer_colors"
    CATEGORY = "BrainDead/Mesh"
    DESCRIPTION = """
Transfer colors from TRELLIS2 pointcloud to mesh using pymeshlab.

This is the RELIABLE approach - uses MeshLab's proven algorithms
for point cloud to mesh color transfer.

Connect:
1. 'trimesh' from TRELLIS.2 Shape to Textured Mesh -> mesh
2. 'pbr_pointcloud' from TRELLIS.2 Shape to Textured Mesh -> pointcloud
"""

    def transfer_colors(self, mesh, pointcloud, max_distance=0.0):
        import tempfile
        import os

        if not HAS_TRIMESH:
            return (mesh, "ERROR: trimesh not installed")

        try:
            import pymeshlab
        except ImportError:
            return (mesh, "ERROR: pymeshlab not installed")

        if mesh is None:
            return (None, "ERROR: mesh is None")

        if pointcloud is None:
            return (mesh, "ERROR: pointcloud is None")

        start_time = time.time()

        print(f"[BD Pymeshlab Transfer] Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces) if mesh.faces is not None else 0} faces")
        print(f"[BD Pymeshlab Transfer] Pointcloud: {len(pointcloud.vertices)} points")

        # Check if pointcloud has colors
        pc_colors = None
        if hasattr(pointcloud, 'colors') and pointcloud.colors is not None:
            pc_colors = pointcloud.colors
            print(f"[BD Pymeshlab Transfer] Pointcloud has {len(pc_colors)} colors")
        elif hasattr(pointcloud, 'visual') and hasattr(pointcloud.visual, 'vertex_colors'):
            pc_colors = pointcloud.visual.vertex_colors
            print(f"[BD Pymeshlab Transfer] Pointcloud visual has {len(pc_colors)} vertex colors")
        else:
            return (mesh, "ERROR: pointcloud has no colors")

        # Save to temp files for pymeshlab
        with tempfile.TemporaryDirectory() as tmpdir:
            pc_path = os.path.join(tmpdir, "pointcloud.ply")
            mesh_path = os.path.join(tmpdir, "mesh.ply")
            output_path = os.path.join(tmpdir, "output.ply")

            # Export pointcloud with colors
            pc_export = trimesh.PointCloud(vertices=pointcloud.vertices, colors=pc_colors)
            pc_export.export(pc_path)

            # Export mesh
            mesh.export(mesh_path)

            # Use pymeshlab to transfer colors
            ms = pymeshlab.MeshSet()

            # Load pointcloud first (will be source, mesh 0)
            ms.load_new_mesh(pc_path)

            # Load target mesh (will be mesh 1, becomes current)
            ms.load_new_mesh(mesh_path)

            # Transfer vertex colors from pointcloud to mesh
            try:
                print(f"[BD Pymeshlab Transfer] Transferring colors...")

                if max_distance > 0:
                    ms.apply_filter('transfer_attributes_per_vertex',
                                    sourcemesh=0,
                                    targetmesh=1,
                                    colortransfer=True,
                                    maxdist=pymeshlab.PercentageValue(max_distance * 100))
                else:
                    ms.apply_filter('transfer_attributes_per_vertex',
                                    sourcemesh=0,
                                    targetmesh=1,
                                    colortransfer=True)

            except Exception as e:
                print(f"[BD Pymeshlab Transfer] Filter error: {e}")
                try:
                    ms.apply_filter('vertex_attribute_transfer',
                                    sourcemesh=0,
                                    targetmesh=1,
                                    colortransfer=True)
                except Exception as e2:
                    return (mesh, f"ERROR: pymeshlab transfer failed: {e2}")

            # Save result
            ms.save_current_mesh(output_path)

            # Load result back into trimesh
            result_mesh = trimesh.load(output_path, process=False)

            total_time = time.time() - start_time

            # Count unique colors
            if hasattr(result_mesh, 'visual') and hasattr(result_mesh.visual, 'vertex_colors'):
                colors = result_mesh.visual.vertex_colors
                if colors is not None:
                    colors_rgba = np.ascontiguousarray(colors[:, :4].astype(np.uint8))
                    unique_count = len(np.unique(colors_rgba.view(np.uint32)))
                    status = f"Transferred colors: {len(result_mesh.vertices)} vertices, {unique_count} unique colors in {total_time:.1f}s"
                else:
                    status = f"Transferred: {len(result_mesh.vertices)} vertices (no colors detected) in {total_time:.1f}s"
            else:
                status = f"Transferred: {len(result_mesh.vertices)} vertices in {total_time:.1f}s"

            print(f"[BD Pymeshlab Transfer] {status}")

            return (result_mesh, status)


class BD_TransferVertexColors:
    """
    Transfer vertex colors from source mesh to target mesh using BVH spatial lookup.

    For each face in target mesh, finds the nearest face on source mesh
    and copies its color directly (no interpolation) for sharp boundaries.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_mesh": ("TRIMESH", {"tooltip": "High-poly mesh with vertex colors"}),
                "target_mesh": ("TRIMESH", {"tooltip": "Decimated mesh to receive colors"}),
            },
            "optional": {
                "transfer_mode": (["face_center", "vertex_nearest", "barycentric"], {
                    "default": "face_center",
                    "tooltip": "face_center=sharp boundaries, vertex_nearest=per-vertex, barycentric=smooth interpolation"
                }),
                "default_color": ("STRING", {"default": "1.0,0.0,1.0,1.0", "tooltip": "Fallback color (magenta) for missing coverage"}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "status")
    FUNCTION = "transfer_colors"
    CATEGORY = "BrainDead/Mesh"
    DESCRIPTION = """
Transfer vertex colors from source mesh to target mesh using spatial lookup.

Transfer modes:
- face_center: For each target face, find nearest source face and copy color
               (sharp color boundaries, best for game assets)
- vertex_nearest: For each target vertex, find nearest source vertex color
                  (may cause slight bleeding at boundaries)
- barycentric: Interpolate colors using barycentric coordinates
               (smoothest but may blur sharp edges)

Typical workflow:
1. Sample colors to high-poly mesh (BD_SampleVoxelgridColors)
2. Decimate mesh (BD_SmartDecimate)
3. Transfer colors back (BD_TransferVertexColors)
"""

    def transfer_colors(self, source_mesh, target_mesh, transfer_mode="face_center",
                        default_color="1.0,0.0,1.0,1.0"):
        from scipy.spatial import cKDTree

        if not HAS_TRIMESH:
            return (target_mesh, "ERROR: trimesh not installed")

        if source_mesh is None or target_mesh is None:
            return (target_mesh, "ERROR: source or target mesh is None")

        start_time = time.time()

        # Parse default color
        try:
            default_rgba = np.array([float(x.strip()) for x in default_color.split(",")][:4], dtype=np.float32)
            if len(default_rgba) == 3:
                default_rgba = np.append(default_rgba, 1.0)
        except:
            default_rgba = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)

        # Get source vertex colors
        source_colors = None
        if hasattr(source_mesh, 'visual') and hasattr(source_mesh.visual, 'vertex_colors'):
            source_colors = np.array(source_mesh.visual.vertex_colors, dtype=np.float32)
            if source_colors.max() > 1.0:
                source_colors = source_colors / 255.0
        elif hasattr(source_mesh, 'vertex_colors') and source_mesh.vertex_colors is not None:
            source_colors = np.array(source_mesh.vertex_colors, dtype=np.float32)
            if source_colors.max() > 1.0:
                source_colors = source_colors / 255.0

        if source_colors is None:
            return (target_mesh, "ERROR: Source mesh has no vertex colors")

        # Ensure RGBA
        if source_colors.shape[1] == 3:
            source_colors = np.column_stack([source_colors, np.ones(len(source_colors))])

        source_verts = np.array(source_mesh.vertices, dtype=np.float32)
        source_faces = np.array(source_mesh.faces)
        target_verts = np.array(target_mesh.vertices, dtype=np.float32)
        target_faces = np.array(target_mesh.faces) if hasattr(target_mesh, 'faces') else None

        print(f"[BD Transfer Colors] Source: {len(source_verts)} verts, {len(source_faces)} faces")
        print(f"[BD Transfer Colors] Target: {len(target_verts)} verts, {len(target_faces) if target_faces is not None else 0} faces")
        print(f"[BD Transfer Colors] Mode: {transfer_mode}")

        if transfer_mode == "face_center" and target_faces is not None:
            target_colors = self._transfer_face_center(
                source_verts, source_faces, source_colors,
                target_verts, target_faces, default_rgba
            )
        elif transfer_mode == "barycentric" and target_faces is not None:
            target_colors = self._transfer_barycentric(
                source_mesh, source_colors,
                target_verts, default_rgba
            )
        else:
            target_colors = self._transfer_vertex_nearest(
                source_verts, source_colors, target_verts, default_rgba
            )

        # Create new mesh with transferred colors
        try:
            target_colors_uint8 = (target_colors * 255).clip(0, 255).astype(np.uint8)

            new_mesh = trimesh.Trimesh(
                vertices=target_mesh.vertices.copy(),
                faces=target_mesh.faces.copy() if target_faces is not None else None,
                process=False
            )

            new_mesh.visual = trimesh.visual.ColorVisuals(
                mesh=new_mesh,
                vertex_colors=target_colors_uint8
            )

            if hasattr(target_mesh, 'vertex_normals') and target_mesh.vertex_normals is not None:
                new_mesh.vertex_normals = target_mesh.vertex_normals.copy()

            total_time = time.time() - start_time

            # Count unique colors
            colors_contiguous = np.ascontiguousarray(target_colors_uint8)
            unique_colors = len(np.unique(colors_contiguous.view(np.uint32)))

            status = f"Transferred colors: {len(target_verts)} verts ({unique_colors} unique colors) | mode={transfer_mode} | {total_time:.1f}s"
            print(f"[BD Transfer Colors] {status}")

            return (new_mesh, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return (target_mesh, f"ERROR: {e}")

    def _transfer_face_center(self, source_verts, source_faces, source_colors,
                               target_verts, target_faces, default_rgba):
        """Transfer colors using face center lookup (sharp boundaries)."""
        from scipy.spatial import cKDTree

        # Compute source face centers and colors
        source_face_centers = source_verts[source_faces].mean(axis=1)
        source_face_colors = source_colors[source_faces].mean(axis=1)

        print(f"[BD Transfer Colors] Building KD-tree from {len(source_face_centers)} source face centers")

        tree = cKDTree(source_face_centers)

        # Compute target face centers
        target_face_centers = target_verts[target_faces].mean(axis=1)

        # Find nearest source face for each target face
        distances, indices = tree.query(target_face_centers, k=1, workers=-1)

        # Get colors for target faces
        target_face_colors = source_face_colors[indices]

        # Apply to target vertices
        target_colors = np.full((len(target_verts), 4), default_rgba, dtype=np.float32)

        for face_idx, face in enumerate(target_faces):
            for vert_idx in face:
                target_colors[vert_idx] = target_face_colors[face_idx]

        num_default = np.all(target_colors == default_rgba, axis=1).sum()
        print(f"[BD Transfer Colors] Face center: {len(target_faces)} faces mapped, {num_default} verts with default color")

        return target_colors

    def _transfer_vertex_nearest(self, source_verts, source_colors, target_verts, default_rgba):
        """Transfer colors using nearest vertex lookup."""
        from scipy.spatial import cKDTree

        print(f"[BD Transfer Colors] Building KD-tree from {len(source_verts)} source vertices")

        tree = cKDTree(source_verts)
        distances, indices = tree.query(target_verts, k=1, workers=-1)

        target_colors = source_colors[indices]

        print(f"[BD Transfer Colors] Vertex nearest: max distance = {distances.max():.6f}")

        return target_colors

    def _transfer_barycentric(self, source_mesh, source_colors, target_verts, default_rgba):
        """Transfer colors using barycentric interpolation on source mesh."""
        # Use trimesh's closest point functionality
        closest_points, distances, face_indices = source_mesh.nearest.on_surface(target_verts)

        source_faces = np.array(source_mesh.faces)
        target_colors = np.full((len(target_verts), 4), default_rgba, dtype=np.float32)

        for i, (point, face_idx) in enumerate(zip(closest_points, face_indices)):
            if face_idx is None or face_idx < 0:
                continue

            # Get triangle vertices
            tri_verts = source_mesh.vertices[source_faces[face_idx]]
            tri_colors = source_colors[source_faces[face_idx]]

            # Compute barycentric coordinates
            bary = self._barycentric_coords(point, tri_verts)

            # Interpolate color
            target_colors[i] = (bary[0] * tri_colors[0] +
                               bary[1] * tri_colors[1] +
                               bary[2] * tri_colors[2])

        print(f"[BD Transfer Colors] Barycentric: processed {len(target_verts)} vertices")

        return target_colors

    def _barycentric_coords(self, p, tri):
        """Compute barycentric coordinates for point p in triangle tri."""
        v0, v1, v2 = tri[0], tri[1], tri[2]

        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = p - v0

        d00 = np.dot(v0v1, v0v1)
        d01 = np.dot(v0v1, v0v2)
        d11 = np.dot(v0v2, v0v2)
        d20 = np.dot(v0p, v0v1)
        d21 = np.dot(v0p, v0v2)

        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-10:
            return np.array([1/3, 1/3, 1/3])

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        return np.array([u, v, w]).clip(0, 1)


# Node exports
MESH_TRANSFER_NODES = {
    "BD_TransferPointcloudColors": BD_TransferPointcloudColors,
    "BD_TransferColorsPymeshlab": BD_TransferColorsPymeshlab,
    "BD_TransferVertexColors": BD_TransferVertexColors,
}

MESH_TRANSFER_DISPLAY_NAMES = {
    "BD_TransferPointcloudColors": "BD Transfer Pointcloud Colors (deprecated)",
    "BD_TransferColorsPymeshlab": "BD Transfer Colors (Pymeshlab)",
    "BD_TransferVertexColors": "BD Transfer Vertex Colors",
}
