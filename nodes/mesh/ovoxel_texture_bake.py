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

            # Bake tangent-space normal map from high-poly → low-poly transfer
            print("[BD OVoxel Texture Bake] Baking tangent-space normal map...")
            normal_img = cls._bake_tangent_normal_map(
                mesh_verts_yup, mesh_faces, mesh_uvs,
                orig_vertices, orig_faces, face_id,
                rast, mask, texture_size,
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

            # Inpaint normal map seams
            if normal_img is not None:
                normal_np = (normal_img[0].numpy() * 255).clip(0, 255).astype(np.uint8)
                normal_np = cv2.inpaint(normal_np, mask_inv, inpaint_radius, cv2.INPAINT_TELEA)
                normal_img = torch.from_numpy(normal_np.astype(np.float32) / 255.0).unsqueeze(0)
            else:
                normal_img = make_placeholder(texture_size, texture_size, (0.5, 0.5, 1.0))

            # Apply PBR material to output mesh (including normal map)
            import trimesh as trimesh_lib
            normal_tex = None
            if normal_img is not None:
                normal_np_uint8 = (normal_img[0].numpy() * 255).clip(0, 255).astype(np.uint8)
                normal_tex = Image.fromarray(normal_np_uint8)

            material = trimesh_lib.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(np.concatenate([base_color, (alpha_raw * np.ones_like(alpha_raw))], axis=-1)),
                baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
                metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic_raw), roughness_raw, metallic_raw], axis=-1)),
                normalTexture=normal_tex,
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
    def _bake_tangent_normal_map(
        cls, mesh_verts_yup, mesh_faces, mesh_uvs,
        orig_vertices, orig_faces, face_id,
        rast, mask, texture_size,
    ):
        """
        Bake a tangent-space normal map from high-poly → low-poly transfer.

        For each UV texel:
        1. Get the high-poly face normal (from BVH face_id)
        2. Compute the low-poly tangent frame (from UV gradients)
        3. Transform high-poly normal into tangent space

        This captures geometric detail lost during decimation.
        """
        import torch
        import numpy as np

        try:
            device = mesh_verts_yup.device

            # 1. Compute ALL high-poly face normals
            v0 = orig_vertices[orig_faces[:, 0]]
            v1 = orig_vertices[orig_faces[:, 1]]
            v2 = orig_vertices[orig_faces[:, 2]]
            hp_edge1 = v1 - v0
            hp_edge2 = v2 - v0
            hp_face_normals = torch.cross(hp_edge1, hp_edge2, dim=-1)
            hp_face_normals = hp_face_normals / (torch.norm(hp_face_normals, dim=-1, keepdim=True) + 1e-8)

            # Get per-texel high-poly normal (face_id is from BVH on the valid texels)
            hp_normals_valid = hp_face_normals[face_id.long()]  # [N_valid, 3]

            # 2. Compute per-triangle tangent frames on the low-poly mesh
            # Using mesh_verts_yup (same coord space as orig_vertices)
            lv0 = mesh_verts_yup[mesh_faces[:, 0]]
            lv1 = mesh_verts_yup[mesh_faces[:, 1]]
            lv2 = mesh_verts_yup[mesh_faces[:, 2]]
            luv0 = mesh_uvs[mesh_faces[:, 0]]
            luv1 = mesh_uvs[mesh_faces[:, 1]]
            luv2 = mesh_uvs[mesh_faces[:, 2]]

            # Position deltas
            dp1 = lv1 - lv0
            dp2 = lv2 - lv0

            # UV deltas
            duv1 = luv1 - luv0
            duv2 = luv2 - luv0

            # Tangent/bitangent from UV gradients
            det = duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]
            # Avoid division by zero for degenerate UVs
            det = torch.where(det.abs() < 1e-8, torch.ones_like(det), det)
            inv_det = 1.0 / det

            tangent = inv_det.unsqueeze(-1) * (duv2[:, 1:2] * dp1 - duv1[:, 1:2] * dp2)
            bitangent = inv_det.unsqueeze(-1) * (-duv2[:, 0:1] * dp1 + duv1[:, 0:1] * dp2)

            # Face normal from cross product (geometric normal of low-poly triangle)
            lp_face_normal = torch.cross(dp1, dp2, dim=-1)
            lp_face_normal = lp_face_normal / (torch.norm(lp_face_normal, dim=-1, keepdim=True) + 1e-8)

            # Orthogonalize tangent frame (Gram-Schmidt)
            tangent = tangent - (tangent * lp_face_normal).sum(-1, keepdim=True) * lp_face_normal
            tangent = tangent / (torch.norm(tangent, dim=-1, keepdim=True) + 1e-8)
            bitangent = torch.cross(lp_face_normal, tangent, dim=-1)
            bitangent = bitangent / (torch.norm(bitangent, dim=-1, keepdim=True) + 1e-8)

            # 3. Get per-texel low-poly triangle index from rast
            # rast[..., 3] is 1-indexed face ID (with chunk offset already applied)
            tri_id_map = rast[0, ..., 3].long() - 1  # [H, W], 0-indexed, -1 for empty
            valid_tri_ids = tri_id_map[mask]  # [N_valid]

            # Clamp to valid range (safety)
            valid_tri_ids = valid_tri_ids.clamp(0, mesh_faces.shape[0] - 1)

            # Get tangent frame for each valid texel
            T = tangent[valid_tri_ids]          # [N_valid, 3]
            B = bitangent[valid_tri_ids]        # [N_valid, 3]
            N = lp_face_normal[valid_tri_ids]   # [N_valid, 3]

            # 4. Transform high-poly normals to tangent space
            # TBN^T * world_normal = tangent_space_normal
            ts_x = (hp_normals_valid * T).sum(-1)   # tangent component
            ts_y = (hp_normals_valid * B).sum(-1)   # bitangent component
            ts_z = (hp_normals_valid * N).sum(-1)   # normal component

            tangent_normals = torch.stack([ts_x, ts_y, ts_z], dim=-1)

            # Normalize (handle edge cases where projection gives zero-length)
            tn_len = torch.norm(tangent_normals, dim=-1, keepdim=True)
            tangent_normals = torch.where(
                tn_len > 1e-6,
                tangent_normals / tn_len,
                torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(tangent_normals),
            )

            # Ensure Z is positive (normal should point outward in tangent space)
            # If Z is negative, the high-poly normal faces away from the low-poly surface
            tangent_normals[:, 2] = tangent_normals[:, 2].abs()

            # 5. Encode as [0, 1] range
            normal_map = torch.full((texture_size, texture_size, 3), 0.0, device=device)
            normal_map[..., :] = torch.tensor([0.5, 0.5, 1.0], device=device)  # default flat
            normal_map[mask] = tangent_normals * 0.5 + 0.5

            n_valid = mask.sum().item()
            flat_pct = ((tangent_normals[:, 2] > 0.99).sum().item() / max(n_valid, 1)) * 100
            print(f"[BD OVoxel Texture Bake] Normal map: {n_valid:,} texels, {flat_pct:.0f}% flat")

            # Free GPU memory before returning
            del hp_face_normals, hp_normals_valid, tangent, bitangent, lp_face_normal
            del T, B, N, tangent_normals
            torch.cuda.empty_cache()

            return normal_map.cpu().unsqueeze(0).float()

        except Exception as e:
            import traceback
            print(f"[BD OVoxel Texture Bake] Warning: tangent-space normal map failed: {e}")
            traceback.print_exc()
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
