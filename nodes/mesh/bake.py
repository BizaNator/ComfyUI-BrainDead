"""
BD_BakeTextures - Bake PBR textures from voxelgrid onto UV-mapped mesh.

Outputs individual texture maps for maximum flexibility.
"""

import os
import gc

from comfy_api.latest import io

from ..blender.base import HAS_TRIMESH

if HAS_TRIMESH:
    import trimesh
    import numpy as np

from .types import TrimeshInput, TrimeshOutput


class BD_BakeTextures(io.ComfyNode):
    """
    Bake PBR textures from TRELLIS2 voxelgrid onto UV-mapped mesh.

    Takes a mesh WITH UVs (from BD_UVUnwrap) and samples colors/PBR
    attributes from the voxelgrid, outputting individual texture maps.

    Supports two voxelgrids: clean (normal colors) and outlined (edge lines).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_BakeTextures",
            display_name="BD Bake Textures",
            category="🧠BrainDead/Mesh",
            description="""Bake PBR textures from voxelgrid onto UV-mapped mesh.

Input mesh MUST have UVs (use BD_UVUnwrap first).

Outputs individual texture maps:
- diffuse: Base color (RGB)
- diffuse_outlined: Base color with edge lines
- normal: Normal map from mesh normals
- metallic/roughness: PBR attributes
- emission: Emission/glow (if available in voxelgrid)
- ao: Ambient occlusion (placeholder white)
- alpha: Transparency mask

Tip: Use BD_SampleVoxelgridColors BEFORE decimation for color edges.""",
            inputs=[
                TrimeshInput("mesh", tooltip="UV-mapped mesh (from BD_UVUnwrap)"),
                io.Custom("TRELLIS2_VOXELGRID").Input(
                    "voxelgrid",
                    tooltip="Clean voxelgrid for PBR attributes",
                ),
                io.Custom("TRELLIS2_VOXELGRID").Input(
                    "voxelgrid_outlined",
                    optional=True,
                    tooltip="Outlined voxelgrid for edge-lined diffuse",
                ),
                io.Int.Input(
                    "texture_size",
                    default=2048,
                    min=512,
                    max=8192,
                    step=512,
                    tooltip="Output texture resolution",
                ),
                io.Boolean.Input(
                    "bake_normal",
                    default=True,
                    tooltip="Generate normal map from mesh normals",
                ),
                io.Boolean.Input(
                    "inpaint_seams",
                    default=True,
                    tooltip="Fill UV seam gaps with inpainting",
                ),
                io.Int.Input(
                    "inpaint_radius",
                    default=3,
                    min=1,
                    max=10,
                    optional=True,
                    tooltip="Inpainting radius for seam filling",
                ),
            ],
            outputs=[
                TrimeshOutput(display_name="mesh"),
                io.Image.Output(display_name="diffuse"),
                io.Image.Output(display_name="diffuse_outlined"),
                io.Image.Output(display_name="normal"),
                io.Image.Output(display_name="metallic"),
                io.Image.Output(display_name="roughness"),
                io.Image.Output(display_name="emission"),
                io.Image.Output(display_name="ao"),
                io.Image.Output(display_name="alpha"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        voxelgrid,
        voxelgrid_outlined=None,
        texture_size: int = 2048,
        bake_normal: bool = True,
        inpaint_seams: bool = True,
        inpaint_radius: int = 3,
    ) -> io.NodeOutput:
        # Error return helper (10 outputs: mesh, diffuse, diffuse_outlined, normal, metallic, roughness, emission, ao, alpha, status)
        def error_return(msg):
            return io.NodeOutput(None, None, None, None, None, None, None, None, None, msg)

        if not HAS_TRIMESH:
            return error_return("ERROR: trimesh not installed")

        if mesh is None:
            return error_return("ERROR: No input mesh")

        # Check for UVs
        if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'uv'):
            return io.NodeOutput(mesh, None, None, None, None, None, None, None, None, "ERROR: Mesh has no UVs")
        if mesh.visual.uv is None or len(mesh.visual.uv) == 0:
            return io.NodeOutput(mesh, None, None, None, None, None, None, None, None, "ERROR: Mesh has no UVs")

        # Check voxelgrid
        if voxelgrid is None or 'attrs' not in voxelgrid:
            return io.NodeOutput(mesh, None, None, None, None, None, None, None, None, "ERROR: Invalid voxelgrid")

        print(f"[BD Bake] Input: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")
        print(f"[BD Bake] UVs: {len(mesh.visual.uv)} coordinates")
        print(f"[BD Bake] Texture size: {texture_size}x{texture_size}")

        try:
            import torch
            import cv2
            import cumesh as CuMesh
            import nvdiffrast.torch as dr
            from flex_gemm.ops.grid_sample import grid_sample_3d
            from PIL import Image
        except ImportError as e:
            return io.NodeOutput(mesh, None, None, None, None, None, None, None, None, f"ERROR: Missing dependency: {e}")

        try:
            # Get mesh data
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
            faces = torch.tensor(mesh.faces, dtype=torch.int32).cuda()
            uvs = torch.tensor(mesh.visual.uv, dtype=torch.float32).cuda()

            # Convert Z-up to Y-up for voxel sampling (TRELLIS uses Y-up internally)
            vertices_yup = vertices.clone()
            vertices_yup[:, 1], vertices_yup[:, 2] = -vertices[:, 2].clone(), vertices[:, 1].clone()

            # Get voxel data
            attr_volume = voxelgrid['attrs']
            if isinstance(attr_volume, np.ndarray):
                attr_volume = torch.from_numpy(attr_volume)
            attr_volume = attr_volume.cuda()

            coords = voxelgrid['coords']
            if isinstance(coords, np.ndarray):
                coords = torch.from_numpy(coords)
            coords = coords.cuda()

            voxel_size = voxelgrid['voxel_size']
            attr_layout = voxelgrid['layout']

            print(f"[BD Bake] Voxelgrid layout: {list(attr_layout.keys())}")

            # Get original mesh for BVH (if available)
            orig_vertices = voxelgrid.get('original_vertices')
            orig_faces = voxelgrid.get('original_faces')

            if orig_vertices is not None:
                if isinstance(orig_vertices, np.ndarray):
                    orig_vertices = torch.from_numpy(orig_vertices)
                orig_vertices = orig_vertices.cuda()

                if isinstance(orig_faces, np.ndarray):
                    orig_faces = torch.from_numpy(orig_faces)
                orig_faces = orig_faces.cuda()

            # AABB
            aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32, device='cuda')

            # Grid size
            if voxel_size is not None:
                if isinstance(voxel_size, float):
                    voxel_size = torch.tensor([voxel_size] * 3, device='cuda')
                elif isinstance(voxel_size, (list, tuple, np.ndarray)):
                    voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device='cuda')
                grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
            else:
                grid_size = torch.tensor([1024, 1024, 1024], dtype=torch.int32, device='cuda')
                voxel_size = (aabb[1] - aabb[0]) / grid_size

            # Build BVH if we have original mesh
            bvh = None
            if orig_vertices is not None and orig_faces is not None:
                print("[BD Bake] Building BVH from original mesh...")
                bvh = CuMesh.cuBVH(orig_vertices, orig_faces)

            print("[BD Bake] Rasterizing in UV space...")

            # Setup nvdiffrast
            ctx = dr.RasterizeCudaContext()

            # Prepare UVs for rasterization (convert 0-1 to -1 to 1 clip space)
            uvs_rast = torch.cat([
                uvs * 2 - 1,
                torch.zeros_like(uvs[:, :1]),
                torch.ones_like(uvs[:, :1])
            ], dim=-1).unsqueeze(0)

            rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)

            # Rasterize in chunks to handle large meshes
            chunk_size = 100000
            for i in range(0, faces.shape[0], chunk_size):
                rast_chunk, _ = dr.rasterize(
                    ctx, uvs_rast, faces[i:i+chunk_size],
                    resolution=[texture_size, texture_size],
                )
                mask_chunk = rast_chunk[..., 3:4] > 0
                rast_chunk[..., 3:4] += i
                rast = torch.where(mask_chunk, rast_chunk, rast)
                del rast_chunk, mask_chunk

            del ctx, uvs_rast
            torch.cuda.empty_cache()

            mask = rast[0, ..., 3] > 0

            # Interpolate 3D positions from UV pixels
            pos = dr.interpolate(vertices_yup.unsqueeze(0), rast, faces)[0][0]
            valid_pos = pos[mask]

            # Map positions to original mesh surface if BVH available
            if bvh is not None:
                _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
                orig_tri_verts = orig_vertices[orig_faces[face_id.long()]]
                valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
                del face_id, uvw, orig_tri_verts

            # Sample voxel attributes for texture
            print("[BD Bake] Sampling PBR attributes from voxelgrid...")
            attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
            attrs[mask] = grid_sample_3d(
                attr_volume,
                torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
                shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
                grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
                mode='trilinear',
            )

            # Sample outlined voxelgrid if provided
            attrs_outlined = None
            if voxelgrid_outlined is not None and 'attrs' in voxelgrid_outlined:
                print("[BD Bake] Sampling outlined voxelgrid...")
                attr_volume_outlined = voxelgrid_outlined['attrs']
                if isinstance(attr_volume_outlined, np.ndarray):
                    attr_volume_outlined = torch.from_numpy(attr_volume_outlined)
                attr_volume_outlined = attr_volume_outlined.cuda()

                coords_outlined = voxelgrid_outlined['coords']
                if isinstance(coords_outlined, np.ndarray):
                    coords_outlined = torch.from_numpy(coords_outlined)
                coords_outlined = coords_outlined.cuda()

                attrs_outlined = torch.zeros(texture_size, texture_size, attr_volume_outlined.shape[1], device='cuda')
                attrs_outlined[mask] = grid_sample_3d(
                    attr_volume_outlined,
                    torch.cat([torch.zeros_like(coords_outlined[:, :1]), coords_outlined], dim=-1),
                    shape=torch.Size([1, attr_volume_outlined.shape[1], *grid_size.tolist()]),
                    grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
                    mode='trilinear',
                )
                del attr_volume_outlined, coords_outlined

            # Bake normal map from vertex normals
            normal_map = None
            if bake_normal:
                print("[BD Bake] Baking normal map...")
                if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                    normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32).cuda()
                else:
                    mesh_copy = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
                    normals = torch.tensor(mesh_copy.vertex_normals, dtype=torch.float32).cuda()

                # Convert normals to Y-up
                normals_yup = normals.clone()
                normals_yup[:, 1], normals_yup[:, 2] = -normals[:, 2].clone(), normals[:, 1].clone()

                # Interpolate normals in UV space
                normal_map = torch.zeros(texture_size, texture_size, 3, device='cuda')
                interp_normals = dr.interpolate(normals_yup.unsqueeze(0), rast, faces)[0][0]
                normal_map = interp_normals

                # Normalize
                normal_map = normal_map / (normal_map.norm(dim=-1, keepdim=True) + 1e-8)

                del normals, normals_yup, interp_normals

            del pos, rast, vertices_yup, valid_pos
            if bvh is not None:
                del bvh
            torch.cuda.empty_cache()

            print("[BD Bake] Building texture maps...")
            mask_np = mask.cpu().numpy()

            # Extract PBR channels from clean voxelgrid
            base_color = attrs[..., attr_layout['base_color']].cpu().numpy()
            metallic = attrs[..., attr_layout['metallic']].cpu().numpy()
            roughness = attrs[..., attr_layout['roughness']].cpu().numpy()

            # Extract alpha (transparency) - Microsoft TRELLIS exports this
            if 'alpha' in attr_layout:
                alpha = attrs[..., attr_layout['alpha']].cpu().numpy()
                print("[BD Bake] Found alpha channel in voxelgrid")
            else:
                # Default to opaque
                alpha = np.ones((texture_size, texture_size), dtype=np.float32)
                print("[BD Bake] No alpha in voxelgrid, using opaque")

            # Extract emission if available (check both 'emission' and 'emissive' keys)
            emission_key = None
            if 'emission' in attr_layout:
                emission_key = 'emission'
            elif 'emissive' in attr_layout:
                emission_key = 'emissive'

            if emission_key is not None:
                emission = attrs[..., attr_layout[emission_key]].cpu().numpy()
                print(f"[BD Bake] Found {emission_key} channel in voxelgrid")
            else:
                # Default to black (no emission) - standard TRELLIS2 pipeline uses 6 channels without emission
                emission = np.zeros((texture_size, texture_size, 3), dtype=np.float32)
                print("[BD Bake] No emission in voxelgrid (standard 6-channel layout), using black")

            # Extract outlined base color if available
            base_color_outlined = None
            if attrs_outlined is not None:
                outlined_layout = voxelgrid_outlined.get('layout', attr_layout)
                base_color_outlined = attrs_outlined[..., outlined_layout['base_color']].cpu().numpy()
                del attrs_outlined

            del attrs, mask
            gc.collect()
            torch.cuda.empty_cache()

            # Inpaint UV seams
            if inpaint_seams:
                print(f"[BD Bake] Inpainting UV seams (radius={inpaint_radius})...")
                mask_inv = (~mask_np).astype(np.uint8)

                # Helper to inpaint single or multi-channel
                def inpaint_texture(tex, mask_inv, radius):
                    if tex.ndim == 2:
                        tex_uint8 = np.clip(tex * 255, 0, 255).astype(np.uint8)
                        tex_uint8 = cv2.inpaint(tex_uint8, mask_inv, radius, cv2.INPAINT_TELEA)
                        return tex_uint8.astype(np.float32) / 255.0
                    elif tex.shape[-1] == 1:
                        tex_uint8 = np.clip(tex[..., 0] * 255, 0, 255).astype(np.uint8)
                        tex_uint8 = cv2.inpaint(tex_uint8, mask_inv, radius, cv2.INPAINT_TELEA)
                        return tex_uint8.astype(np.float32) / 255.0
                    else:
                        tex_uint8 = np.clip(tex * 255, 0, 255).astype(np.uint8)
                        tex_uint8 = cv2.inpaint(tex_uint8, mask_inv, radius, cv2.INPAINT_TELEA)
                        return tex_uint8.astype(np.float32) / 255.0

                base_color = inpaint_texture(base_color, mask_inv, inpaint_radius)
                metallic = inpaint_texture(metallic, mask_inv, inpaint_radius)
                roughness = inpaint_texture(roughness, mask_inv, inpaint_radius)
                alpha = inpaint_texture(alpha, mask_inv, inpaint_radius)
                emission = inpaint_texture(emission, mask_inv, inpaint_radius)

                if base_color_outlined is not None:
                    base_color_outlined = inpaint_texture(base_color_outlined, mask_inv, inpaint_radius)

                # Inpaint normal map
                if normal_map is not None:
                    normal_np = normal_map.cpu().numpy()
                    normal_01 = (normal_np + 1) / 2
                    normal_uint8 = np.clip(normal_01 * 255, 0, 255).astype(np.uint8)
                    normal_uint8 = cv2.inpaint(normal_uint8, mask_inv, inpaint_radius, cv2.INPAINT_TELEA)
                    normal_map = torch.from_numpy((normal_uint8.astype(np.float32) / 255.0) * 2 - 1)

            # Helper to ensure 3-channel for display
            def ensure_rgb(tex):
                if tex.ndim == 2:
                    return np.stack([tex] * 3, axis=-1)
                elif tex.shape[-1] == 1:
                    return np.concatenate([tex] * 3, axis=-1)
                return tex

            # Convert to ComfyUI IMAGE format (BHWC, float32, 0-1 range)
            diffuse_tensor = torch.from_numpy(ensure_rgb(base_color)).unsqueeze(0).float()

            if base_color_outlined is not None:
                diffuse_outlined_tensor = torch.from_numpy(ensure_rgb(base_color_outlined)).unsqueeze(0).float()
            else:
                diffuse_outlined_tensor = diffuse_tensor.clone()

            # Normal map (convert from -1,1 to 0,1 for display)
            if normal_map is not None:
                normal_np = normal_map.cpu().numpy() if isinstance(normal_map, torch.Tensor) else normal_map
                normal_01 = (normal_np + 1) / 2
                normal_tensor = torch.from_numpy(normal_01).unsqueeze(0).float()
            else:
                normal_tensor = torch.full((1, texture_size, texture_size, 3), 0.5)
                normal_tensor[..., 2] = 1.0

            metallic_tensor = torch.from_numpy(ensure_rgb(metallic)).unsqueeze(0).float()
            roughness_tensor = torch.from_numpy(ensure_rgb(roughness)).unsqueeze(0).float()
            emission_tensor = torch.from_numpy(ensure_rgb(emission)).unsqueeze(0).float()

            # AO - placeholder white (full ambient light)
            # TODO: Could compute screen-space AO or raycast AO in future
            ao_np = np.ones((texture_size, texture_size, 3), dtype=np.float32)
            ao_tensor = torch.from_numpy(ao_np).unsqueeze(0).float()

            # Alpha (grayscale -> RGB for display)
            alpha_tensor = torch.from_numpy(ensure_rgb(alpha)).unsqueeze(0).float()

            # Sample vertex colors from baked texture at UV coordinates
            # This provides colors for direct viewing and PLY export
            print("[BD Bake] Sampling vertex colors from baked texture...")
            uvs_np = np.array(mesh.visual.uv)

            # Convert UV (0-1) to texture pixel coordinates
            tex_coords_x = np.clip((uvs_np[:, 0] * (texture_size - 1)).astype(int), 0, texture_size - 1)
            tex_coords_y = np.clip(((1 - uvs_np[:, 1]) * (texture_size - 1)).astype(int), 0, texture_size - 1)  # Flip Y

            # Sample diffuse texture at vertex UVs
            base_color_rgb = ensure_rgb(base_color)
            vertex_colors_rgb = base_color_rgb[tex_coords_y, tex_coords_x]

            # Sample alpha at vertex UVs
            alpha_single = alpha if alpha.ndim == 2 else alpha[..., 0]
            vertex_alpha = alpha_single[tex_coords_y, tex_coords_x]

            # Build RGBA vertex colors (0-255 uint8)
            vertex_colors_rgba = np.concatenate([
                np.clip(vertex_colors_rgb * 255, 0, 255).astype(np.uint8),
                np.clip(vertex_alpha[..., np.newaxis] * 255, 0, 255).astype(np.uint8)
            ], axis=-1)

            # Build result mesh with vertex colors (works universally)
            result_mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                vertex_normals=mesh.vertex_normals if hasattr(mesh, 'vertex_normals') else None,
                process=False,
            )

            # Set vertex colors - this works for direct viewing and PLY export
            result_mesh.visual = trimesh.visual.ColorVisuals(
                mesh=result_mesh,
                vertex_colors=vertex_colors_rgba
            )

            # Store UVs in metadata for potential texture application later
            result_mesh.metadata['uv'] = mesh.visual.uv
            result_mesh.metadata['has_baked_textures'] = True

            baked_maps = ["diffuse", "normal", "metallic", "roughness", "alpha"]
            if voxelgrid_outlined is not None:
                baked_maps.append("diffuse_outlined")
            if emission_key is not None:
                baked_maps.append("emission")

            status = f"Baked {texture_size}x{texture_size}: {', '.join(baked_maps)}"
            print(f"[BD Bake] {status}")

            # Cleanup
            del vertices, faces, uvs, attr_volume, coords
            if orig_vertices is not None:
                del orig_vertices, orig_faces
            gc.collect()
            torch.cuda.empty_cache()

            return io.NodeOutput(
                result_mesh,
                diffuse_tensor,
                diffuse_outlined_tensor,
                normal_tensor,
                metallic_tensor,
                roughness_tensor,
                emission_tensor,
                ao_tensor,
                alpha_tensor,
                status,
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return io.NodeOutput(mesh, None, None, None, None, None, None, None, None, f"ERROR: {e}")


# V3 node list
BAKE_V3_NODES = [BD_BakeTextures]

# V1 compatibility
BAKE_NODES = {
    "BD_BakeTextures": BD_BakeTextures,
}

BAKE_DISPLAY_NAMES = {
    "BD_BakeTextures": "BD Bake Textures",
}
