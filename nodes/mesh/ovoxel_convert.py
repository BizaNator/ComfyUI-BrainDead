"""
BD_MeshToOVoxel - Convert a generic textured mesh into VOXELGRID format.

Uses o_voxel.convert to voxelize geometry and extract PBR attributes,
producing a VOXELGRID that can be used with BD_OVoxelTextureBake or
BD_OVoxelBake for texture baking.
"""

import gc

from comfy_api.latest import io


class BD_MeshToOVoxel(io.ComfyNode):
    """
    Convert a textured mesh (GLB/trimesh) to VOXELGRID format.

    Voxelizes mesh geometry using flexible dual grid and extracts PBR
    attributes from the mesh material/textures into a sparse voxel tensor.

    The output VOXELGRID can be used with BD_OVoxelBake or
    BD_OVoxelTextureBake for texture baking onto a different mesh.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_MeshToOVoxel",
            display_name="BD Mesh to OVoxel",
            category="🧠BrainDead/Mesh",
            description="""Convert a textured mesh to VOXELGRID format using o_voxel.

Voxelizes mesh geometry (flexible dual grid) and extracts PBR material
attributes into a sparse voxel tensor.

Use cases:
- Retexture a mesh onto different topology
- Create voxelgrid from imported GLB/FBX for baking
- Convert between mesh and voxel representations

Outputs VOXELGRID compatible with BD_OVoxelBake and BD_OVoxelTextureBake.""",
            inputs=[
                io.Custom("TRIMESH").Input(
                    "mesh",
                    tooltip="Textured mesh to voxelize (needs PBR material/textures for attribute extraction)",
                ),
                io.Int.Input(
                    "grid_size",
                    default=512,
                    min=128,
                    max=1024,
                    step=64,
                    tooltip="Voxel grid resolution. Higher = more detail, more memory",
                ),
                io.Float.Input(
                    "face_weight",
                    default=1.0,
                    min=0.1,
                    max=5.0,
                    step=0.1,
                    tooltip="QEF face term weight for geometry voxelization",
                ),
                io.Float.Input(
                    "boundary_weight",
                    default=0.2,
                    min=0.01,
                    max=2.0,
                    step=0.01,
                    tooltip="QEF boundary term weight",
                ),
                io.Float.Input(
                    "regularization_weight",
                    default=0.01,
                    min=0.001,
                    max=0.5,
                    step=0.001,
                    tooltip="QEF regularization weight",
                ),
            ],
            outputs=[
                io.Custom("VOXELGRID").Output(display_name="voxelgrid"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        grid_size: int = 512,
        face_weight: float = 1.0,
        boundary_weight: float = 0.2,
        regularization_weight: float = 0.01,
    ) -> io.NodeOutput:
        import torch
        import numpy as np

        if mesh is None:
            return io.NodeOutput(None, "ERROR: No mesh provided")

        try:
            import trimesh
            import o_voxel.convert

            device = torch.device('cuda')

            # Normalize mesh to [-0.5, 0.5] AABB
            vertices = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.int32)

            # Center and scale to unit cube
            bbox_min = vertices.min(axis=0)
            bbox_max = vertices.max(axis=0)
            center = (bbox_min + bbox_max) / 2
            extent = (bbox_max - bbox_min).max()
            scale = 0.9 / extent if extent > 0 else 1.0  # 0.9 to leave margin

            normalized_verts = (vertices - center) * scale

            # Convert to Y-up if Z-up (standard ComfyUI mesh convention)
            # Z-up → Y-up: swap Y/Z, negate new Y
            verts_yup = normalized_verts.copy()
            verts_yup[:, 1], verts_yup[:, 2] = -normalized_verts[:, 2], normalized_verts[:, 1]

            verts_t = torch.from_numpy(verts_yup).to(device).float()
            faces_t = torch.from_numpy(faces).to(device).int()

            print(f"[BD Mesh to OVoxel] Input: {len(vertices):,} verts, {len(faces):,} faces")
            print(f"[BD Mesh to OVoxel] Grid size: {grid_size}, extent: {extent:.4f}")

            # Geometry voxelization
            print("[BD Mesh to OVoxel] Voxelizing geometry...")
            aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
            coords, dual_vertices, intersected_flag = o_voxel.convert.mesh_to_flexible_dual_grid(
                vertices=verts_t,
                faces=faces_t,
                grid_size=grid_size,
                aabb=aabb,
                face_weight=face_weight,
                boundary_weight=boundary_weight,
                regularization_weight=regularization_weight,
            )
            print(f"[BD Mesh to OVoxel] Geometry: {coords.shape[0]:,} voxels")

            # Material attribute voxelization
            print("[BD Mesh to OVoxel] Extracting PBR attributes...")

            # Build a trimesh Scene with normalized geometry for attribute extraction
            norm_mesh = trimesh.Trimesh(
                vertices=verts_yup,
                faces=faces,
                process=False,
            )
            # Transfer visual from original mesh
            if hasattr(mesh, 'visual'):
                norm_mesh.visual = mesh.visual.copy()

            # Extract volumetric attributes
            attr_coords, attr_dict = o_voxel.convert.textured_mesh_to_volumetric_attr(
                norm_mesh,
                grid_size=grid_size,
                aabb=aabb,
                verbose=True,
            )

            # Build attrs tensor from attribute dict
            # Standard PBR layout: base_color(3) + metallic(1) + roughness(1) + alpha(1) = 6 channels
            pbr_layout = {
                'base_color': slice(0, 3),
                'metallic': slice(3, 4),
                'roughness': slice(4, 5),
                'alpha': slice(5, 6),
            }

            n_voxels = attr_coords.shape[0]
            attrs = torch.zeros(n_voxels, 6, device=device, dtype=torch.float32)

            if 'base_color' in attr_dict:
                bc = attr_dict['base_color'].to(device).float()
                if bc.max() > 1.0:
                    bc = bc / 255.0
                attrs[:, 0:3] = bc
            if 'metallic' in attr_dict:
                m = attr_dict['metallic'].to(device).float()
                if m.max() > 1.0:
                    m = m / 255.0
                attrs[:, 3:4] = m
            if 'roughness' in attr_dict:
                r = attr_dict['roughness'].to(device).float()
                if r.max() > 1.0:
                    r = r / 255.0
                attrs[:, 4:5] = r
            if 'alpha' in attr_dict:
                a = attr_dict['alpha'].to(device).float()
                if a.max() > 1.0:
                    a = a / 255.0
                attrs[:, 5:6] = a
            else:
                attrs[:, 5:6] = 1.0  # Default fully opaque

            # Use attr_coords for final alignment (same as mesh2ovox reference)
            voxel_size = 1.0 / grid_size

            # Build VOXELGRID output dict
            voxelgrid = {
                'coords': attr_coords.to(device),
                'attrs': attrs,
                'original_vertices': verts_t,
                'original_faces': faces_t,
                'voxel_size': voxel_size,
                'layout': pbr_layout,
            }

            # Cleanup
            del dual_vertices, intersected_flag
            gc.collect()
            torch.cuda.empty_cache()

            status = f"Voxelized: {n_voxels:,} voxels @ {grid_size}³ | {len(vertices):,} verts, {len(faces):,} faces"
            print(f"[BD Mesh to OVoxel] {status}")

            return io.NodeOutput(voxelgrid, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            return io.NodeOutput(None, f"ERROR: {e}")


# V3 node list
OVOXEL_CONVERT_V3_NODES = [BD_MeshToOVoxel]

# V1 compatibility
OVOXEL_CONVERT_NODES = {
    "BD_MeshToOVoxel": BD_MeshToOVoxel,
}

OVOXEL_CONVERT_DISPLAY_NAMES = {
    "BD_MeshToOVoxel": "BD Mesh to OVoxel",
}
