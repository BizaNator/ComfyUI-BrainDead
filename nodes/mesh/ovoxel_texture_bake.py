"""
BD_OVoxelTextureBake - Bake PBR textures onto a pre-processed mesh using o_voxel.

This is the "bake-only" node. It takes a mesh that already has UVs
(from your own decimation + UV unwrap pipeline) and bakes PBR textures
using BVH projection + trilinear grid_sample_3d from the voxelgrid.

Separates mesh processing from texture baking so you can control
decimation/edge preservation independently.
"""

import gc

from comfy_api.latest import io


class BD_OVoxelTextureBake(io.ComfyNode):
    """
    Bake PBR textures onto a pre-processed mesh from voxelgrid data.

    Takes a TRIMESH with UVs (your own simplification + unwrap) and a
    VOXELGRID, then bakes textures using BVH projection + trilinear
    sampling from the sparse voxel tensor.

    This gives you full control over mesh topology while using o_voxel's
    high-quality texture baking.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_OVoxelTextureBake",
            display_name="BD OVoxel Texture Bake",
            category="🧠BrainDead/Mesh",
            description="""Bake PBR textures onto a pre-processed mesh using o_voxel.

Takes YOUR mesh (already decimated + UV unwrapped) and bakes textures from
the TRELLIS2 voxelgrid using BVH projection + trilinear interpolation.

Pipeline:
1. Build BVH from original high-res mesh (in voxelgrid)
2. Rasterize YOUR mesh in UV space (nvdiffrast)
3. Project texel positions onto original mesh surface (BVH)
4. Trilinear sample voxel attributes at projected positions
5. Inpaint UV seams
6. Output individual PBR texture maps

Use after: BD_CuMeshSimplify → BD_UVUnwrap → THIS NODE""",
            inputs=[
                io.Custom("TRIMESH").Input(
                    "mesh",
                    tooltip="Pre-processed mesh WITH UVs (from your own simplify + unwrap pipeline)",
                ),
                io.Custom("VOXELGRID").Input(
                    "voxelgrid",
                    tooltip="TRELLIS2 voxelgrid with PBR attributes and original mesh data",
                ),
                io.Int.Input(
                    "texture_size",
                    default=2048,
                    min=512,
                    max=4096,
                    step=512,
                    tooltip="Output texture resolution",
                ),
                io.Int.Input(
                    "inpaint_radius",
                    default=3,
                    min=1,
                    max=10,
                    step=1,
                    tooltip="Inpainting radius for UV seam filling",
                ),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.Image.Output(display_name="diffuse"),
                io.Image.Output(display_name="normal"),
                io.Image.Output(display_name="metallic"),
                io.Image.Output(display_name="roughness"),
                io.Image.Output(display_name="alpha"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        voxelgrid,
        texture_size: int = 2048,
        inpaint_radius: int = 3,
    ) -> io.NodeOutput:
        import torch
        import numpy as np
        from PIL import Image

        def make_placeholder(h=64, w=64, color=(0.5, 0.5, 0.5)):
            img = torch.full((1, h, w, 3), 0.0, dtype=torch.float32)
            img[..., 0] = color[0]
            img[..., 1] = color[1]
            img[..., 2] = color[2]
            return img

        def error_return(msg):
            print(f"[BD OVoxel Texture Bake] ERROR: {msg}")
            ph = make_placeholder()
            ph_n = make_placeholder(color=(0.5, 0.5, 1.0))
            ph_b = make_placeholder(color=(0.0, 0.0, 0.0))
            ph_w = make_placeholder(color=(1.0, 1.0, 1.0))
            return io.NodeOutput(mesh, ph, ph_n, ph_b, ph, ph_w, f"ERROR: {msg}")

        if mesh is None:
            return error_return("No mesh provided")
        if voxelgrid is None:
            return error_return("No voxelgrid provided")

        # Check mesh has UVs
        has_uvs = (
            hasattr(mesh, 'visual')
            and hasattr(mesh.visual, 'uv')
            and mesh.visual.uv is not None
            and len(mesh.visual.uv) > 0
        )
        if not has_uvs:
            return error_return("Mesh has no UVs - run UV unwrap first")

        # Check voxelgrid has required data
        for key in ('attrs', 'coords', 'original_vertices', 'original_faces', 'voxel_size', 'layout'):
            if key not in voxelgrid:
                return error_return(f"Voxelgrid missing '{key}'")

        try:
            import cv2
            import cumesh
            import nvdiffrast.torch as dr
            from flex_gemm.ops.grid_sample import grid_sample_3d

            device = torch.device('cuda')

            # Get mesh data
            mesh_verts = torch.from_numpy(np.array(mesh.vertices, dtype=np.float32)).to(device)
            mesh_faces = torch.from_numpy(np.array(mesh.faces, dtype=np.int32)).to(device)
            mesh_uvs = torch.from_numpy(np.array(mesh.visual.uv, dtype=np.float32)).to(device)

            # Get voxelgrid data
            orig_vertices = voxelgrid['original_vertices']
            if isinstance(orig_vertices, np.ndarray):
                orig_vertices = torch.from_numpy(orig_vertices)
            orig_vertices = orig_vertices.to(device).float()

            orig_faces = voxelgrid['original_faces']
            if isinstance(orig_faces, np.ndarray):
                orig_faces = torch.from_numpy(orig_faces)
            orig_faces = orig_faces.to(device).int()

            attr_volume = voxelgrid['attrs']
            if isinstance(attr_volume, np.ndarray):
                attr_volume = torch.from_numpy(attr_volume)
            attr_volume = attr_volume.to(device).float()

            coords = voxelgrid['coords']
            if isinstance(coords, np.ndarray):
                coords = torch.from_numpy(coords)
            coords = coords.to(device)

            voxel_size = voxelgrid['voxel_size']
            layout = voxelgrid['layout']
            aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], device=device, dtype=torch.float32)

            # Compute grid_size from voxel_size
            if isinstance(voxel_size, (int, float)):
                grid_size = torch.tensor([round(1.0 / voxel_size)] * 3, device=device)
            else:
                vs = torch.tensor(voxel_size, device=device) if not isinstance(voxel_size, torch.Tensor) else voxel_size.to(device)
                grid_size = torch.round((aabb[1] - aabb[0]) / vs).int()

            print(f"[BD OVoxel Texture Bake] Mesh: {mesh_verts.shape[0]:,} verts, {mesh_faces.shape[0]:,} faces, {mesh_uvs.shape[0]:,} UVs")
            print(f"[BD OVoxel Texture Bake] Original: {orig_vertices.shape[0]:,} verts, {orig_faces.shape[0]:,} faces")
            print(f"[BD OVoxel Texture Bake] Voxels: {coords.shape[0]:,}, texture: {texture_size}x{texture_size}")

            # Convert mesh to Y-up for BVH (o_voxel operates in Y-up)
            # ComfyUI TRIMESH is Z-up, voxelgrid original is Y-up
            mesh_verts_yup = mesh_verts.clone()
            mesh_verts_yup[:, 1], mesh_verts_yup[:, 2] = -mesh_verts[:, 2].clone(), mesh_verts[:, 1].clone()

            # Build BVH from original high-res mesh
            print("[BD OVoxel Texture Bake] Building BVH from original mesh...")
            bvh = cumesh.cuBVH(orig_vertices, orig_faces)

            # Rasterize in UV space
            print("[BD OVoxel Texture Bake] Rasterizing UV space...")
            ctx = dr.RasterizeCudaContext()

            # UVs need V-flip for rasterization (glTF convention → NDC)
            uvs_ndc = mesh_uvs.clone()
            uvs_ndc[:, 1] = 1.0 - uvs_ndc[:, 1]
            uvs_rast = torch.cat([
                uvs_ndc * 2 - 1,
                torch.zeros_like(uvs_ndc[:, :1]),
                torch.ones_like(uvs_ndc[:, :1])
            ], dim=-1).unsqueeze(0)

            # Rasterize in chunks for memory efficiency
            rast = torch.zeros((1, texture_size, texture_size, 4), device=device, dtype=torch.float32)
            chunk_size = 100000
            for i in range(0, mesh_faces.shape[0], chunk_size):
                rast_chunk, _ = dr.rasterize(
                    ctx, uvs_rast, mesh_faces[i:i+chunk_size],
                    resolution=[texture_size, texture_size],
                )
                mask_chunk = rast_chunk[..., 3:4] > 0
                rast_chunk[..., 3:4] += i
                rast = torch.where(mask_chunk, rast_chunk, rast)

            mask = rast[0, ..., 3] > 0
            print(f"[BD OVoxel Texture Bake] UV coverage: {mask.sum().item():,}/{texture_size*texture_size:,} texels ({mask.float().mean().item()*100:.1f}%)")

            # Interpolate 3D positions in UV space (Y-up coordinates)
            pos = dr.interpolate(mesh_verts_yup.unsqueeze(0), rast, mesh_faces)[0][0]
            valid_pos = pos[mask]

            # BVH project to original mesh surface for accurate sampling
            print("[BD OVoxel Texture Bake] BVH projecting to original surface...")
            _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
            orig_tri_verts = orig_vertices[orig_faces[face_id.long()]]
            valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

            # Trilinear sample from voxel attrs
            print("[BD OVoxel Texture Bake] Sampling voxel attributes...")
            attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device=device)
            attrs[mask] = grid_sample_3d(
                attr_volume,
                torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
                shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
                grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
                mode='trilinear',
            )

            # Clean up heavy GPU data
            del bvh, pos, valid_pos, face_id, uvw, orig_tri_verts
            del mesh_verts_yup, uvs_rast, uvs_ndc, rast
            torch.cuda.empty_cache()

            # Extract PBR channels
            mask_np = mask.cpu().numpy()
            mask_inv = (~mask_np).astype(np.uint8)

            base_color = np.clip(attrs[..., layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
            metallic_raw = np.clip(attrs[..., layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
            roughness_raw = np.clip(attrs[..., layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
            alpha_raw = np.clip(attrs[..., layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)

            del attrs
            torch.cuda.empty_cache()

            # Inpaint UV seams
            print("[BD OVoxel Texture Bake] Inpainting UV seams...")
            base_color = cv2.inpaint(base_color, mask_inv, inpaint_radius, cv2.INPAINT_TELEA)
            metallic_raw = cv2.inpaint(metallic_raw, mask_inv, 1, cv2.INPAINT_TELEA)
            roughness_raw = cv2.inpaint(roughness_raw, mask_inv, 1, cv2.INPAINT_TELEA)
            alpha_raw = cv2.inpaint(alpha_raw, mask_inv, 1, cv2.INPAINT_TELEA)

            # Ensure correct shapes for single-channel
            if metallic_raw.ndim == 2:
                metallic_raw = metallic_raw[..., None]
            if roughness_raw.ndim == 2:
                roughness_raw = roughness_raw[..., None]
            if alpha_raw.ndim == 2:
                alpha_raw = alpha_raw[..., None]

            # Convert to ComfyUI IMAGE format [B, H, W, C] float32
            diffuse_img = torch.from_numpy(base_color.astype(np.float32) / 255.0).unsqueeze(0)
            metallic_img = torch.from_numpy(metallic_raw.astype(np.float32).repeat(3, axis=-1) / 255.0).unsqueeze(0)
            roughness_img = torch.from_numpy(roughness_raw.astype(np.float32).repeat(3, axis=-1) / 255.0).unsqueeze(0)
            alpha_img = torch.from_numpy(alpha_raw.astype(np.float32).repeat(3, axis=-1) / 255.0).unsqueeze(0)

            # Bake normal map from mesh vertex normals
            normal_img = cls._bake_normal_map(mesh, mesh_verts, mesh_faces, mesh_uvs, texture_size)
            if normal_img is None:
                normal_img = make_placeholder(texture_size, texture_size, (0.5, 0.5, 1.0))

            # Apply PBR material to output mesh
            import trimesh as trimesh_lib
            material = trimesh_lib.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(np.concatenate([base_color, (alpha_raw * np.ones_like(alpha_raw))], axis=-1)),
                baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
                metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic_raw), roughness_raw, metallic_raw], axis=-1)),
                metallicFactor=1.0,
                roughnessFactor=1.0,
                alphaMode='OPAQUE',
                doubleSided=True,
            )

            result_mesh = trimesh_lib.Trimesh(
                vertices=mesh.vertices.copy(),
                faces=mesh.faces.copy(),
                process=False,
                visual=trimesh_lib.visual.TextureVisuals(
                    uv=mesh.visual.uv.copy(),
                    material=material,
                ),
            )

            # Cleanup
            del mesh_verts, mesh_faces, mesh_uvs, orig_vertices, orig_faces, attr_volume, coords
            gc.collect()
            torch.cuda.empty_cache()

            status = f"Baked {texture_size}x{texture_size} | UV coverage: {mask_np.sum()*100/(texture_size*texture_size):.1f}%"
            print(f"[BD OVoxel Texture Bake] {status}")

            return io.NodeOutput(result_mesh, diffuse_img, normal_img, metallic_img, roughness_img, alpha_img, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            return error_return(str(e))

    @classmethod
    def _bake_normal_map(cls, mesh, mesh_verts, mesh_faces, mesh_uvs, texture_size):
        """Bake normal map from mesh vertex normals using nvdiffrast."""
        import torch
        import numpy as np

        try:
            import nvdiffrast.torch as dr

            if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
                return None

            normals = torch.from_numpy(np.array(mesh.vertex_normals, dtype=np.float32)).cuda()

            # Rasterize in UV space
            uvs_ndc = mesh_uvs.clone()
            uvs_ndc[:, 1] = 1.0 - uvs_ndc[:, 1]
            uvs_rast = torch.cat([
                uvs_ndc * 2 - 1,
                torch.zeros_like(uvs_ndc[:, :1]),
                torch.ones_like(uvs_ndc[:, :1])
            ], dim=-1).unsqueeze(0)

            ctx = dr.RasterizeCudaContext()
            rast, _ = dr.rasterize(ctx, uvs_rast, mesh_faces, resolution=[texture_size, texture_size])
            mask = rast[0, ..., 3] > 0

            # Interpolate normals
            normal_map = dr.interpolate(normals.unsqueeze(0), rast, mesh_faces)[0][0]

            # Normalize and convert to [0, 1]
            normal_np = normal_map.cpu().numpy()
            norms = np.linalg.norm(normal_np, axis=-1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            normal_np = normal_np / norms
            normal_np = (normal_np * 0.5 + 0.5)

            mask_np = mask.cpu().numpy()
            normal_np[~mask_np] = [0.5, 0.5, 1.0]

            return torch.from_numpy(normal_np.astype(np.float32)).unsqueeze(0)

        except Exception as e:
            print(f"[BD OVoxel Texture Bake] Warning: normal map baking failed: {e}")
            return None


# V3 node list
OVOXEL_TEXTURE_BAKE_V3_NODES = [BD_OVoxelTextureBake]

# V1 compatibility
OVOXEL_TEXTURE_BAKE_NODES = {
    "BD_OVoxelTextureBake": BD_OVoxelTextureBake,
}

OVOXEL_TEXTURE_BAKE_DISPLAY_NAMES = {
    "BD_OVoxelTextureBake": "BD OVoxel Texture Bake",
}
