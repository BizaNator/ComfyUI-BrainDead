"""
BD_OVoxelBake - Bake PBR textures using Microsoft's o_voxel reference implementation.

This wraps o_voxel.postprocess.to_glb() which handles the complete pipeline:
- Multi-stage mesh simplification with proper topology repair
- CuMesh UV unwrapping with vertex normal computation
- BVH projection for accurate texture sampling
- Trilinear grid_sample_3d interpolation from sparse voxel tensor
- Proper coordinate system conversion for GLB

This replaces the old BD_CuMeshSimplify → BD_UVUnwrap → manual bake pipeline.
"""

import gc

from comfy_api.latest import io


class BD_OVoxelBake(io.ComfyNode):
    """
    Bake PBR textures using Microsoft's o_voxel reference implementation.

    Takes the TRELLIS2 voxelgrid and produces a textured mesh with individual
    PBR texture maps. Uses the same pipeline as Microsoft's official demo.

    Handles mesh cleaning, UV unwrapping, BVH projection, and trilinear
    texture baking in one integrated step.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_OVoxelBake",
            display_name="BD OVoxel Bake",
            category="🧠BrainDead/Mesh",
            description="""Bake PBR textures using Microsoft's o_voxel reference implementation.

Takes TRELLIS2 voxelgrid directly and produces textured mesh + individual PBR maps.
Uses the same pipeline as Microsoft's official TRELLIS.2 demo.

Pipeline (all handled internally):
1. Multi-stage CuMesh simplification with topology repair
2. CuMesh UV unwrapping with vertex normals
3. BVH projection to original mesh for accurate sampling
4. Trilinear interpolation from sparse voxel tensor
5. UV seam inpainting
6. Proper coordinate system conversion

Replaces the old BD_CuMeshSimplify → BD_UVUnwrap → manual bake pipeline.""",
            inputs=[
                io.Custom("VOXELGRID").Input(
                    "voxelgrid",
                    tooltip="TRELLIS2 voxelgrid with PBR attributes (needs original_vertices/faces and attrs tensor)",
                ),
                io.Int.Input(
                    "decimation_target",
                    default=50000,
                    min=1000,
                    max=2000000,
                    step=1000,
                    tooltip="Target face count for mesh simplification",
                ),
                io.Int.Input(
                    "texture_size",
                    default=2048,
                    min=512,
                    max=4096,
                    step=512,
                    tooltip="Output texture resolution",
                ),
                io.Boolean.Input(
                    "remesh",
                    default=False,
                    tooltip="Enable dual-contouring remesh (better topology, slower)",
                ),
                io.Float.Input(
                    "remesh_band",
                    default=1.0,
                    min=0.5,
                    max=3.0,
                    step=0.5,
                    optional=True,
                    tooltip="Remesh narrow band width (only used if remesh=True)",
                ),
                io.Float.Input(
                    "remesh_project",
                    default=0.9,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    optional=True,
                    tooltip="Project back to original surface (0=none, 1=full)",
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
        voxelgrid,
        decimation_target: int = 50000,
        texture_size: int = 2048,
        remesh: bool = False,
        remesh_band: float = 1.0,
        remesh_project: float = 0.9,
    ) -> io.NodeOutput:
        import torch
        import numpy as np
        from PIL import Image

        # Placeholder for error cases
        def make_placeholder(h=64, w=64, color=(0.5, 0.5, 0.5)):
            img = torch.full((1, h, w, 3), 0.0, dtype=torch.float32)
            img[..., 0] = color[0]
            img[..., 1] = color[1]
            img[..., 2] = color[2]
            return img

        def error_return(msg):
            print(f"[BD OVoxel Bake] ERROR: {msg}")
            ph = make_placeholder()
            ph_n = make_placeholder(color=(0.5, 0.5, 1.0))
            ph_b = make_placeholder(color=(0.0, 0.0, 0.0))
            ph_w = make_placeholder(color=(1.0, 1.0, 1.0))
            return io.NodeOutput(None, ph, ph_n, ph_b, ph, ph_w, f"ERROR: {msg}")

        if voxelgrid is None:
            return error_return("No voxelgrid provided")

        if 'attrs' not in voxelgrid or 'coords' not in voxelgrid:
            return error_return("Voxelgrid missing attrs or coords")

        if 'original_vertices' not in voxelgrid or 'original_faces' not in voxelgrid:
            return error_return("Voxelgrid missing original_vertices/faces")

        try:
            import o_voxel.postprocess

            device = torch.device('cuda')

            # Get tensors from voxelgrid dict
            vertices = voxelgrid['original_vertices']
            if isinstance(vertices, np.ndarray):
                vertices = torch.from_numpy(vertices)
            vertices = vertices.to(device).float()

            faces = voxelgrid['original_faces']
            if isinstance(faces, np.ndarray):
                faces = torch.from_numpy(faces)
            faces = faces.to(device).int()

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

            print(f"[BD OVoxel Bake] Input: {vertices.shape[0]:,} verts, {faces.shape[0]:,} faces")
            print(f"[BD OVoxel Bake] Voxels: {coords.shape[0]:,}, layout: {list(layout.keys())}")
            print(f"[BD OVoxel Bake] Target: {decimation_target:,} faces, texture: {texture_size}x{texture_size}")
            print(f"[BD OVoxel Bake] Remesh: {remesh}" + (f" (band={remesh_band}, project={remesh_project})" if remesh else ""))

            # Call o_voxel reference implementation
            textured_mesh = o_voxel.postprocess.to_glb(
                vertices=vertices,
                faces=faces,
                attr_volume=attr_volume,
                coords=coords,
                attr_layout=layout,
                voxel_size=voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=decimation_target,
                texture_size=texture_size,
                remesh=remesh,
                remesh_band=remesh_band,
                remesh_project=remesh_project,
                verbose=True,
                use_tqdm=True,
            )

            # Clean up voxel tensors (keep vertices/faces for normal baking)
            del attr_volume, coords
            gc.collect()
            torch.cuda.empty_cache()

            print(f"[BD OVoxel Bake] Result: {len(textured_mesh.vertices):,} verts, {len(textured_mesh.faces):,} faces")

            # Bake tangent-space normal map from high-poly → simplified mesh
            print("[BD OVoxel Bake] Baking tangent-space normal map...")
            normal_img = cls._bake_tangent_normal_map(textured_mesh, vertices, faces, texture_size)

            # Clean up remaining GPU tensors
            del vertices, faces
            gc.collect()
            torch.cuda.empty_cache()

            # Extract textures from the PBR material
            material = textured_mesh.visual.material
            diffuse_img = None
            metallic_img = None
            roughness_img = None
            alpha_img = None

            # Extract base color texture (RGBA)
            if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                bc_tex = material.baseColorTexture
                if isinstance(bc_tex, Image.Image):
                    bc_np = np.array(bc_tex).astype(np.float32) / 255.0
                    if bc_np.shape[-1] == 4:
                        # RGBA - split into diffuse RGB and alpha
                        diffuse_np = bc_np[..., :3]
                        alpha_np = bc_np[..., 3:4]
                        # Expand alpha to 3 channels for IMAGE output
                        alpha_img = torch.from_numpy(alpha_np.repeat(3, axis=-1)).unsqueeze(0)
                    else:
                        diffuse_np = bc_np[..., :3]
                    diffuse_img = torch.from_numpy(diffuse_np).unsqueeze(0)

            # Extract metallic-roughness texture (R=0, G=roughness, B=metallic)
            if hasattr(material, 'metallicRoughnessTexture') and material.metallicRoughnessTexture is not None:
                mr_tex = material.metallicRoughnessTexture
                if isinstance(mr_tex, Image.Image):
                    mr_np = np.array(mr_tex).astype(np.float32) / 255.0
                    # Green = roughness, Blue = metallic
                    if mr_np.ndim == 3 and mr_np.shape[-1] >= 3:
                        rough_np = mr_np[..., 1:2].repeat(3, axis=-1)
                        metal_np = mr_np[..., 2:3].repeat(3, axis=-1)
                        roughness_img = torch.from_numpy(rough_np).unsqueeze(0)
                        metallic_img = torch.from_numpy(metal_np).unsqueeze(0)

            # Fallbacks for missing textures
            if diffuse_img is None:
                diffuse_img = make_placeholder(texture_size, texture_size)
            if normal_img is None:
                normal_img = make_placeholder(texture_size, texture_size, (0.5, 0.5, 1.0))
            if metallic_img is None:
                metallic_img = make_placeholder(texture_size, texture_size, (0.0, 0.0, 0.0))
            if roughness_img is None:
                roughness_img = make_placeholder(texture_size, texture_size, (0.5, 0.5, 0.5))
            if alpha_img is None:
                alpha_img = make_placeholder(texture_size, texture_size, (1.0, 1.0, 1.0))

            status = f"Baked {texture_size}x{texture_size}: {len(textured_mesh.vertices):,} verts, {len(textured_mesh.faces):,} faces"
            if remesh:
                status += " (remeshed)"
            print(f"[BD OVoxel Bake] {status}")

            return io.NodeOutput(textured_mesh, diffuse_img, normal_img, metallic_img, roughness_img, alpha_img, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            return error_return(str(e))

    @classmethod
    def _bake_tangent_normal_map(cls, textured_mesh, orig_vertices, orig_faces, texture_size):
        """
        Bake a tangent-space normal map from high-poly → simplified mesh transfer.

        The textured_mesh (from to_glb) is in Z-up GLB coordinates.
        The orig_vertices/faces are in Y-up voxelgrid coordinates.
        We convert the simplified mesh to Y-up for BVH projection.
        """
        import torch
        import numpy as np

        try:
            import cumesh
            import nvdiffrast.torch as dr

            uvs = textured_mesh.visual.uv
            if uvs is None:
                print("[BD OVoxel Bake] No UVs on mesh, skipping normal map")
                return None

            device = orig_vertices.device

            # Get simplified mesh data
            mesh_verts = torch.from_numpy(np.array(textured_mesh.vertices, dtype=np.float32)).to(device)
            mesh_faces = torch.from_numpy(np.array(textured_mesh.faces, dtype=np.int32)).to(device)
            mesh_uvs = torch.from_numpy(np.array(uvs, dtype=np.float32)).to(device)

            # Convert simplified mesh from Z-up (GLB) to Y-up (voxelgrid space) for BVH
            # to_glb does: [x, z, -y] (Y-up → Z-up), so reverse: [x, -z, y]
            mesh_verts_yup = mesh_verts.clone()
            mesh_verts_yup[:, 1] = -mesh_verts[:, 2]
            mesh_verts_yup[:, 2] = mesh_verts[:, 1]

            # Build BVH from original high-res mesh
            bvh = cumesh.cuBVH(orig_vertices, orig_faces)

            # Rasterize simplified mesh in UV space
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

            # Interpolate 3D positions in UV space (Y-up)
            pos = dr.interpolate(mesh_verts_yup.unsqueeze(0), rast, mesh_faces)[0][0]
            valid_pos = pos[mask]

            # BVH project to original surface
            _, face_id, _ = bvh.unsigned_distance(valid_pos, return_uvw=True)

            # Compute high-poly face normals
            v0 = orig_vertices[orig_faces[:, 0]]
            v1 = orig_vertices[orig_faces[:, 1]]
            v2 = orig_vertices[orig_faces[:, 2]]
            hp_face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
            hp_face_normals = hp_face_normals / (torch.norm(hp_face_normals, dim=-1, keepdim=True) + 1e-8)

            hp_normals_valid = hp_face_normals[face_id.long()]

            # Compute low-poly tangent frames from UV gradients (in Y-up space)
            lv0 = mesh_verts_yup[mesh_faces[:, 0]]
            lv1 = mesh_verts_yup[mesh_faces[:, 1]]
            lv2 = mesh_verts_yup[mesh_faces[:, 2]]
            luv0 = mesh_uvs[mesh_faces[:, 0]]
            luv1 = mesh_uvs[mesh_faces[:, 1]]
            luv2 = mesh_uvs[mesh_faces[:, 2]]

            dp1 = lv1 - lv0
            dp2 = lv2 - lv0
            duv1 = luv1 - luv0
            duv2 = luv2 - luv0

            det = duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]
            det = torch.where(det.abs() < 1e-8, torch.ones_like(det), det)
            inv_det = 1.0 / det

            tangent = inv_det.unsqueeze(-1) * (duv2[:, 1:2] * dp1 - duv1[:, 1:2] * dp2)
            bitangent = inv_det.unsqueeze(-1) * (-duv2[:, 0:1] * dp1 + duv1[:, 0:1] * dp2)

            lp_face_normal = torch.cross(dp1, dp2, dim=-1)
            lp_face_normal = lp_face_normal / (torch.norm(lp_face_normal, dim=-1, keepdim=True) + 1e-8)

            # Gram-Schmidt orthogonalization
            tangent = tangent - (tangent * lp_face_normal).sum(-1, keepdim=True) * lp_face_normal
            tangent = tangent / (torch.norm(tangent, dim=-1, keepdim=True) + 1e-8)
            bitangent = torch.cross(lp_face_normal, tangent, dim=-1)
            bitangent = bitangent / (torch.norm(bitangent, dim=-1, keepdim=True) + 1e-8)

            # Get per-texel tangent frame
            tri_id_map = rast[0, ..., 3].long() - 1
            valid_tri_ids = tri_id_map[mask].clamp(0, mesh_faces.shape[0] - 1)

            T = tangent[valid_tri_ids]
            B = bitangent[valid_tri_ids]
            N = lp_face_normal[valid_tri_ids]

            # Transform high-poly normals to tangent space
            ts_x = (hp_normals_valid * T).sum(-1)
            ts_y = (hp_normals_valid * B).sum(-1)
            ts_z = (hp_normals_valid * N).sum(-1)

            tangent_normals = torch.stack([ts_x, ts_y, ts_z], dim=-1)
            tn_len = torch.norm(tangent_normals, dim=-1, keepdim=True)
            tangent_normals = torch.where(
                tn_len > 1e-6,
                tangent_normals / tn_len,
                torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(tangent_normals),
            )
            tangent_normals[:, 2] = tangent_normals[:, 2].abs()

            # Encode to texture
            normal_map = torch.full((texture_size, texture_size, 3), 0.0, device=device)
            normal_map[..., :] = torch.tensor([0.5, 0.5, 1.0], device=device)
            normal_map[mask] = tangent_normals * 0.5 + 0.5

            n_valid = mask.sum().item()
            flat_pct = ((tangent_normals[:, 2] > 0.99).sum().item() / max(n_valid, 1)) * 100
            print(f"[BD OVoxel Bake] Normal map: {n_valid:,} texels, {flat_pct:.0f}% flat")

            # Inpaint UV seams
            try:
                import cv2
                mask_np = mask.cpu().numpy()
                mask_inv = (~mask_np).astype(np.uint8)
                normal_np = (normal_map.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                normal_np = cv2.inpaint(normal_np, mask_inv, 3, cv2.INPAINT_TELEA)
                normal_img = torch.from_numpy(normal_np.astype(np.float32) / 255.0).unsqueeze(0)
            except ImportError:
                normal_img = normal_map.cpu().unsqueeze(0).float()

            # Cleanup
            del bvh, pos, valid_pos, face_id, hp_face_normals, hp_normals_valid
            del tangent, bitangent, lp_face_normal, T, B, N, tangent_normals
            del mesh_verts, mesh_faces, mesh_uvs, mesh_verts_yup, rast
            torch.cuda.empty_cache()

            return normal_img

        except Exception as e:
            import traceback
            print(f"[BD OVoxel Bake] Warning: tangent-space normal map failed: {e}")
            traceback.print_exc()
            return None


# V3 node list
OVOXEL_BAKE_V3_NODES = [BD_OVoxelBake]

# V1 compatibility
OVOXEL_BAKE_NODES = {
    "BD_OVoxelBake": BD_OVoxelBake,
}

OVOXEL_BAKE_DISPLAY_NAMES = {
    "BD_OVoxelBake": "BD OVoxel Bake",
}
