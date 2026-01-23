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

from .types import TrimeshInput, TrimeshOutput, ColorFieldInput


class BD_BakeTextures(io.ComfyNode):
    """
    Bake PBR textures from voxelgrid or COLOR_FIELD onto UV-mapped mesh.

    Takes a mesh WITH UVs (from BD_UVUnwrap) and samples colors/PBR
    attributes from the voxelgrid or COLOR_FIELD, outputting individual texture maps.

    Supports two color sources:
    - voxelgrid: Direct TRELLIS2 voxelgrid (dense grid sampling)
    - color_field: COLOR_FIELD from BD_SampleVoxelgridColors (sparse KD-tree sampling)

    COLOR_FIELD is useful when:
    - Mesh has been decimated/remeshed (topology changed)
    - You want to preserve original voxelgrid quality through pipeline
    - You don't want to pass full voxelgrid through workflow
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_BakeTextures",
            display_name="BD Bake Textures",
            category="🧠BrainDead/Mesh",
            description="""Bake PBR textures from voxelgrid or COLOR_FIELD onto UV-mapped mesh.

Input mesh MUST have UVs (use BD_UVUnwrap first).

Color sources (use ONE):
- voxelgrid: Direct TRELLIS2 voxelgrid (full PBR)
- color_field: From BD_SampleVoxelgridColors (preserves quality through pipeline)

Outputs individual texture maps:
- diffuse: Base color (RGB)
- diffuse_outlined: Base color with edge lines
- normal: Normal map from mesh normals
- metallic/roughness: PBR attributes
- emission: Emission/glow (if available)
- ao: Ambient occlusion (placeholder white)
- alpha: Transparency mask

Tip: Use COLOR_FIELD to bake from original 10M+ voxel data after decimation.""",
            inputs=[
                TrimeshInput("mesh", tooltip="UV-mapped mesh (from BD_UVUnwrap)"),
                io.Custom("VOXELGRID").Input(
                    "voxelgrid",
                    optional=True,
                    tooltip="Clean voxelgrid for PBR attributes (or use color_field)",
                ),
                io.Custom("VOXELGRID").Input(
                    "voxelgrid_outlined",
                    optional=True,
                    tooltip="Outlined voxelgrid for edge-lined diffuse",
                ),
                ColorFieldInput(
                    "color_field",
                    optional=True,
                    tooltip="COLOR_FIELD from BD_SampleVoxelgridColors (alternative to voxelgrid)",
                ),
                ColorFieldInput(
                    "color_field_outlined",
                    optional=True,
                    tooltip="Outlined COLOR_FIELD for edge-lined diffuse",
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
                io.Boolean.Input(
                    "fix_normals",
                    default=True,
                    optional=True,
                    tooltip="Fix inconsistent face winding (TRELLIS2 meshes often have ~50% flipped faces)",
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
        voxelgrid=None,
        voxelgrid_outlined=None,
        color_field=None,
        color_field_outlined=None,
        texture_size: int = 2048,
        bake_normal: bool = True,
        inpaint_seams: bool = True,
        inpaint_radius: int = 3,
        fix_normals: bool = True,
    ) -> io.NodeOutput:
        import torch

        # Create placeholder images for error cases (prevents downstream SaveImage crashes)
        def make_placeholder(color=(0.5, 0.5, 0.5)):
            """Create a 64x64 placeholder image tensor."""
            img = torch.full((1, 64, 64, 3), 0.0, dtype=torch.float32)
            img[..., 0] = color[0]
            img[..., 1] = color[1]
            img[..., 2] = color[2]
            return img

        # Error return helper with placeholder images
        def error_return(msg, return_mesh=None):
            print(f"[BD Bake] {msg}")
            placeholder_gray = make_placeholder((0.5, 0.5, 0.5))
            placeholder_normal = make_placeholder((0.5, 0.5, 1.0))  # Default normal (0,0,1)
            placeholder_black = make_placeholder((0.0, 0.0, 0.0))
            placeholder_white = make_placeholder((1.0, 1.0, 1.0))
            return io.NodeOutput(
                return_mesh,
                placeholder_gray,   # diffuse
                placeholder_gray,   # diffuse_outlined
                placeholder_normal, # normal
                placeholder_black,  # metallic
                placeholder_gray,   # roughness
                placeholder_black,  # emission
                placeholder_white,  # ao
                placeholder_white,  # alpha
                msg
            )

        if not HAS_TRIMESH:
            return error_return("ERROR: trimesh not installed")

        if mesh is None:
            return error_return("ERROR: No input mesh")

        # Check for UVs
        if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'uv'):
            return error_return("ERROR: Mesh has no UVs", mesh)
        if mesh.visual.uv is None or len(mesh.visual.uv) == 0:
            return error_return("ERROR: Mesh has no UVs", mesh)

        # Fix inconsistent face normals (TRELLIS2 FlexiCubes often produces ~50% flipped faces)
        # The mesh may have split vertices (from UV unwrap) preventing adjacency-based fix.
        # Solution: merge vertices temporarily, fix normals, map flips back to original.
        if fix_normals:
            try:
                n_faces = len(mesh.faces)
                # Save original face winding
                original_faces = mesh.faces.copy()

                # Create a merged copy to build proper face adjacency
                merged = mesh.copy()
                merged.merge_vertices(merge_tex=False, merge_norm=False)

                # Check if merge helped connectivity
                if len(merged.faces) == n_faces:
                    # Same face count - faces map 1:1
                    # Save pre-fix faces
                    pre_fix_faces = merged.faces.copy()
                    trimesh.repair.fix_normals(merged)
                    post_fix_faces = merged.faces

                    # Find which faces were flipped (vertex order reversed)
                    flipped_mask = np.any(pre_fix_faces != post_fix_faces, axis=1)
                    n_flipped = flipped_mask.sum()

                    if n_flipped > 0:
                        # Apply the same flips to the original mesh
                        mesh.faces[flipped_mask] = mesh.faces[flipped_mask][:, ::-1]
                        print(f"[BD Bake] Fixed face normals: flipped {n_flipped}/{n_faces} faces ({100*n_flipped/n_faces:.1f}%)")
                    else:
                        # fix_normals didn't flip anything - try component-based heuristic
                        # For each connected component, check if average normal points outward
                        from scipy.sparse.csgraph import connected_components
                        from scipy.sparse import csr_matrix

                        adj = merged.face_adjacency
                        if len(adj) > 0:
                            data = np.ones(len(adj) * 2)
                            row = np.concatenate([adj[:, 0], adj[:, 1]])
                            col = np.concatenate([adj[:, 1], adj[:, 0]])
                            graph = csr_matrix((data, (row, col)), shape=(n_faces, n_faces))
                            n_components, labels = connected_components(graph, directed=False)

                            if n_components > 1:
                                mesh_center = mesh.vertices.mean(axis=0)
                                face_normals = mesh.face_normals
                                face_centers = mesh.triangles_center
                                total_flipped = 0

                                for comp_id in range(n_components):
                                    comp_mask = labels == comp_id
                                    comp_normals = face_normals[comp_mask]
                                    comp_centers = face_centers[comp_mask]

                                    # Check if component normals point away from mesh center
                                    to_outside = comp_centers - mesh_center
                                    dots = np.sum(comp_normals * to_outside, axis=1)
                                    avg_dot = dots.mean()

                                    if avg_dot < 0:  # Normals point inward on average
                                        # Flip all faces in this component
                                        comp_indices = np.where(comp_mask)[0]
                                        mesh.faces[comp_indices] = mesh.faces[comp_indices][:, ::-1]
                                        total_flipped += len(comp_indices)

                                if total_flipped > 0:
                                    print(f"[BD Bake] Fixed face normals (component heuristic): flipped {total_flipped}/{n_faces} faces ({100*total_flipped/n_faces:.1f}%) across {n_components} components")
                                else:
                                    print(f"[BD Bake] Face normals appear consistent ({n_components} components)")
                            else:
                                print(f"[BD Bake] Face normals consistent (single component)")
                        else:
                            print(f"[BD Bake] No face adjacency available for normals fix")
                else:
                    # Face count changed after merge (degenerate faces removed)
                    # Fall back to simple fix_normals on original
                    trimesh.repair.fix_normals(mesh)
                    print(f"[BD Bake] Fixed face normals (simple mode, face count changed during merge)")

                del merged
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[BD Bake] Warning: fix_normals failed: {e}")

        # Check color source - need either voxelgrid or color_field
        use_color_field = color_field is not None and 'positions' in color_field
        use_voxelgrid = voxelgrid is not None and 'attrs' in voxelgrid

        # Debug logging for color_field inputs
        print(f"[BD Bake] color_field: {type(color_field).__name__}, keys={list(color_field.keys()) if isinstance(color_field, dict) else 'N/A'}")
        print(f"[BD Bake] color_field_outlined: {type(color_field_outlined).__name__}, keys={list(color_field_outlined.keys()) if isinstance(color_field_outlined, dict) else 'N/A'}")
        print(f"[BD Bake] use_color_field={use_color_field}, use_voxelgrid={use_voxelgrid}")

        if not use_color_field and not use_voxelgrid:
            return error_return("ERROR: Need voxelgrid or color_field", mesh)

        # Prefer color_field if both provided (it's the deferred/preserved data)
        if use_color_field:
            return cls._bake_from_color_field(
                mesh, color_field, color_field_outlined, texture_size, bake_normal, inpaint_seams, inpaint_radius
            )

        print(f"[BD Bake] Input: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")
        print(f"[BD Bake] UVs: {len(mesh.visual.uv)} coordinates")
        print(f"[BD Bake] Texture size: {texture_size}x{texture_size}")

        try:
            import torch
            import cv2
            import nvdiffrast.torch as dr
            from PIL import Image
        except ImportError as e:
            return error_return(f"ERROR: Missing dependency: {e}", mesh)

        try:
            from scipy.spatial import cKDTree

            # Get mesh data
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
            faces = torch.tensor(mesh.faces, dtype=torch.int32).cuda()
            uvs = torch.tensor(mesh.visual.uv, dtype=torch.float32).cuda()

            # DIAGNOSTIC: Check UV and vertex ranges
            uvs_np_diag = mesh.visual.uv
            verts_np_diag = mesh.vertices
            print(f"[BD Bake] === DIAGNOSTICS ===")
            print(f"[BD Bake] UV range: U=[{uvs_np_diag[:, 0].min():.4f}, {uvs_np_diag[:, 0].max():.4f}], V=[{uvs_np_diag[:, 1].min():.4f}, {uvs_np_diag[:, 1].max():.4f}]")
            print(f"[BD Bake] Vertex range (Z-up): X=[{verts_np_diag[:, 0].min():.4f}, {verts_np_diag[:, 0].max():.4f}], Y=[{verts_np_diag[:, 1].min():.4f}, {verts_np_diag[:, 1].max():.4f}], Z=[{verts_np_diag[:, 2].min():.4f}, {verts_np_diag[:, 2].max():.4f}]")

            # Check for UV anomalies
            uv_out_of_range = ((uvs_np_diag < 0) | (uvs_np_diag > 1)).sum()
            uv_nan = np.isnan(uvs_np_diag).sum()
            uv_inf = np.isinf(uvs_np_diag).sum()
            print(f"[BD Bake] UV anomalies: out_of_range={uv_out_of_range}, NaN={uv_nan}, Inf={uv_inf}")

            if uv_out_of_range > 0:
                print(f"[BD Bake] WARNING: {uv_out_of_range} UV coordinates outside [0,1] range!")

            # Convert Z-up mesh to TRELLIS Y-up space (same as sampling.py)
            # Mesh from TRELLIS is exported as Z-up, need to convert back to Y-up for voxel alignment
            vertices_yup = vertices.clone()
            vertices_yup[:, 1], vertices_yup[:, 2] = -vertices[:, 2].clone(), vertices[:, 1].clone()

            # DIAGNOSTIC: Check Y-up vertex range
            v_yup_np = vertices_yup.cpu().numpy()
            print(f"[BD Bake] Vertex range (Y-up): X=[{v_yup_np[:, 0].min():.4f}, {v_yup_np[:, 0].max():.4f}], Y=[{v_yup_np[:, 1].min():.4f}, {v_yup_np[:, 1].max():.4f}], Z=[{v_yup_np[:, 2].min():.4f}, {v_yup_np[:, 2].max():.4f}]")

            # Get voxel data
            coords = voxelgrid['coords']
            if hasattr(coords, 'cpu'):
                coords_np = coords.cpu().numpy()
            else:
                coords_np = np.array(coords)

            attrs = voxelgrid['attrs']
            if hasattr(attrs, 'cpu'):
                attrs_np = attrs.cpu().numpy()
            else:
                attrs_np = np.array(attrs)

            voxel_size = voxelgrid['voxel_size']
            if isinstance(voxel_size, (torch.Tensor,)):
                voxel_size = float(voxel_size.item() if voxel_size.numel() == 1 else voxel_size[0].item())
            voxel_size = float(voxel_size)

            attr_layout = voxelgrid['layout']
            print(f"[BD Bake] Voxelgrid layout: {list(attr_layout.keys())}")

            # Convert voxel coords to world positions (same as sampling.py)
            voxel_world_positions = coords_np.astype(np.float32) * voxel_size
            print(f"[BD Bake] Voxel positions range: {voxel_world_positions.min(axis=0)} to {voxel_world_positions.max(axis=0)}")

            # Build KD-tree for voxel sampling (same as sampling.py)
            print(f"[BD Bake] Building KD-tree from {len(voxel_world_positions)} voxels...")
            voxel_tree = cKDTree(voxel_world_positions)

            print("[BD Bake] Rasterizing in UV space...")

            # Setup nvdiffrast
            ctx = dr.RasterizeCudaContext()

            # Prepare UVs for rasterization (convert 0-1 to -1 to 1 clip space)
            # glTF convention: V=0 at TOP of texture, V=1 at BOTTOM
            # nvdiffrast: clip Y=1 → row 0 (top), clip Y=-1 → row H-1 (bottom)
            # So V=0 should map to clip Y=1: clip_y = 1 - 2*v = -(2*v - 1)
            uv_clip = uvs * 2 - 1
            uv_clip[:, 1] = -uv_clip[:, 1]  # Flip Y for glTF convention
            uvs_rast = torch.cat([
                uv_clip,
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

            # Interpolate 3D positions from UV pixels (in Y-up space)
            pos = dr.interpolate(vertices_yup.unsqueeze(0), rast, faces)[0][0]
            valid_pos = pos[mask]

            # Convert positions to voxel space (same as sampling.py)
            # Y-up mesh is in [-0.5, 0.5], shift to [0, 1] to match voxel positions
            valid_pos_voxel_space = valid_pos.cpu().numpy() + 0.5
            print(f"[BD Bake] Sampling positions range: {valid_pos_voxel_space.min(axis=0)} to {valid_pos_voxel_space.max(axis=0)}")

            # Sample voxel attributes using KD-tree (same approach as sampling.py)
            print("[BD Bake] Sampling PBR attributes from voxelgrid using KD-tree...")
            distances, indices = voxel_tree.query(valid_pos_voxel_space, k=1, workers=-1)

            # DIAGNOSTIC: Check KD-tree query results
            print(f"[BD Bake] KD-tree distances: min={distances.min():.6f}, max={distances.max():.6f}, mean={distances.mean():.6f}")
            print(f"[BD Bake] KD-tree indices: min={indices.min()}, max={indices.max()}, unique={len(np.unique(indices))}")

            # Get sampled attributes
            sampled_attrs = attrs_np[indices]

            # DIAGNOSTIC: Check sampled colors
            color_slice = attr_layout.get('base_color', slice(0, 3))
            sampled_colors = sampled_attrs[:, color_slice] if isinstance(color_slice, slice) else sampled_attrs[:, color_slice[0]:color_slice[-1]+1]
            print(f"[BD Bake] Sampled colors: min={sampled_colors.min():.4f}, max={sampled_colors.max():.4f}, mean={sampled_colors.mean():.4f}")

            # Build full texture arrays
            num_channels = attrs_np.shape[1]
            attrs_texture = np.zeros((texture_size, texture_size, num_channels), dtype=np.float32)
            mask_np_temp = mask.cpu().numpy()
            attrs_texture[mask_np_temp] = sampled_attrs

            # Convert back to tensor for consistency with rest of code
            attrs = torch.from_numpy(attrs_texture).cuda()

            # Sample outlined voxelgrid if provided
            attrs_outlined = None
            if voxelgrid_outlined is not None and 'attrs' in voxelgrid_outlined:
                print("[BD Bake] Sampling outlined voxelgrid using KD-tree...")
                coords_outlined = voxelgrid_outlined['coords']
                if hasattr(coords_outlined, 'cpu'):
                    coords_outlined_np = coords_outlined.cpu().numpy()
                else:
                    coords_outlined_np = np.array(coords_outlined)

                attrs_outlined_vg = voxelgrid_outlined['attrs']
                if hasattr(attrs_outlined_vg, 'cpu'):
                    attrs_outlined_np = attrs_outlined_vg.cpu().numpy()
                else:
                    attrs_outlined_np = np.array(attrs_outlined_vg)

                voxel_size_outlined = voxelgrid_outlined.get('voxel_size', voxel_size)
                if isinstance(voxel_size_outlined, (torch.Tensor,)):
                    voxel_size_outlined = float(voxel_size_outlined.item() if voxel_size_outlined.numel() == 1 else voxel_size_outlined[0].item())

                voxel_positions_outlined = coords_outlined_np.astype(np.float32) * float(voxel_size_outlined)
                tree_outlined = cKDTree(voxel_positions_outlined)

                distances_outlined, indices_outlined = tree_outlined.query(valid_pos_voxel_space, k=1, workers=-1)
                sampled_attrs_outlined = attrs_outlined_np[indices_outlined]

                num_channels_outlined = attrs_outlined_np.shape[1]
                attrs_outlined_texture = np.zeros((texture_size, texture_size, num_channels_outlined), dtype=np.float32)
                attrs_outlined_texture[mask_np_temp] = sampled_attrs_outlined

                attrs_outlined = torch.from_numpy(attrs_outlined_texture).cuda()
                del coords_outlined_np, attrs_outlined_np, voxel_positions_outlined, tree_outlined

            # Bake normal map from vertex normals
            normal_map = None
            if bake_normal:
                print("[BD Bake] Baking normal map...")
                if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                    normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32).cuda()
                else:
                    mesh_copy = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
                    normals = torch.tensor(mesh_copy.vertex_normals, dtype=torch.float32).cuda()

                # Convert normals to Y-up (same transform as vertices)
                normals_yup = normals.clone()
                normals_yup[:, 1], normals_yup[:, 2] = -normals[:, 2].clone(), normals[:, 1].clone()

                # Interpolate normals in UV space
                normal_map = torch.zeros(texture_size, texture_size, 3, device='cuda')
                interp_normals = dr.interpolate(normals_yup.unsqueeze(0), rast, faces)[0][0]
                normal_map = interp_normals

                # Normalize
                normal_map = normal_map / (normal_map.norm(dim=-1, keepdim=True) + 1e-8)

                del normals, normals_yup, interp_normals

            del pos, rast, vertices_yup, valid_pos, valid_pos_voxel_space
            del voxel_tree, coords_np, attrs_np, voxel_world_positions
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
            # glTF convention: V=0 at top (row 0), V=1 at bottom (row H-1)
            tex_coords_x = np.clip((uvs_np[:, 0] * (texture_size - 1)).astype(int), 0, texture_size - 1)
            tex_coords_y = np.clip((uvs_np[:, 1] * (texture_size - 1)).astype(int), 0, texture_size - 1)

            # DIAGNOSTIC: Check texture sampling coordinates
            print(f"[BD Bake] Texture coords X: min={tex_coords_x.min()}, max={tex_coords_x.max()}")
            print(f"[BD Bake] Texture coords Y: min={tex_coords_y.min()}, max={tex_coords_y.max()}")

            # Sample diffuse texture at vertex UVs
            base_color_rgb = ensure_rgb(base_color)
            vertex_colors_rgb = base_color_rgb[tex_coords_y, tex_coords_x]

            # DIAGNOSTIC: Check sampled vertex colors
            print(f"[BD Bake] Vertex colors (from texture): min={vertex_colors_rgb.min():.4f}, max={vertex_colors_rgb.max():.4f}, mean={vertex_colors_rgb.mean():.4f}")

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
            del vertices, faces, uvs
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
            return error_return(f"ERROR: {e}", mesh)

    @classmethod
    def _bake_from_color_field(
        cls,
        mesh,
        color_field,
        color_field_outlined=None,
        texture_size: int = 2048,
        bake_normal: bool = True,
        inpaint_seams: bool = True,
        inpaint_radius: int = 3,
    ) -> io.NodeOutput:
        """
        Bake textures from COLOR_FIELD using KD-tree sampling.

        This path is used when color_field is provided instead of voxelgrid.
        Uses scipy KD-tree for efficient sparse point sampling.
        """
        import torch
        import cv2
        import nvdiffrast.torch as dr
        from scipy.spatial import cKDTree

        print(f"[BD Bake] Using COLOR_FIELD path (KD-tree sampling)")
        print(f"[BD Bake] Input: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")
        print(f"[BD Bake] UVs: {len(mesh.visual.uv)} coordinates")
        print(f"[BD Bake] Texture size: {texture_size}x{texture_size}")

        try:
            # Extract color field data
            cf_positions = color_field.get('positions')
            cf_colors = color_field.get('colors')
            cf_attrs = color_field.get('attributes', {})
            cf_voxel_size = color_field.get('voxel_size', 1.0)

            if hasattr(cf_positions, 'cpu'):
                cf_positions = cf_positions.cpu().numpy()
            cf_positions = np.asarray(cf_positions, dtype=np.float32)

            if hasattr(cf_colors, 'cpu'):
                cf_colors = cf_colors.cpu().numpy()
            cf_colors = np.asarray(cf_colors, dtype=np.float32)

            print(f"[BD Bake] COLOR_FIELD: {len(cf_positions)} voxels, voxel_size={cf_voxel_size}")

            # Build KD-tree from color field positions
            print("[BD Bake] Building KD-tree from COLOR_FIELD...")
            tree = cKDTree(cf_positions)

            # Extract PBR attributes from color_field if available
            cf_metallic = cf_attrs.get('metallic')
            cf_roughness = cf_attrs.get('roughness')
            cf_alpha = cf_attrs.get('alpha')

            if cf_metallic is not None:
                if hasattr(cf_metallic, 'cpu'):
                    cf_metallic = cf_metallic.cpu().numpy()
                cf_metallic = np.asarray(cf_metallic, dtype=np.float32).flatten()

            if cf_roughness is not None:
                if hasattr(cf_roughness, 'cpu'):
                    cf_roughness = cf_roughness.cpu().numpy()
                cf_roughness = np.asarray(cf_roughness, dtype=np.float32).flatten()

            if cf_alpha is not None:
                if hasattr(cf_alpha, 'cpu'):
                    cf_alpha = cf_alpha.cpu().numpy()
                cf_alpha = np.asarray(cf_alpha, dtype=np.float32).flatten()

            # Get mesh data
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
            faces = torch.tensor(mesh.faces, dtype=torch.int32).cuda()
            uvs = torch.tensor(mesh.visual.uv, dtype=torch.float32).cuda()

            # Convert Z-up to Y-up for voxel sampling (TRELLIS uses Y-up internally)
            vertices_yup = vertices.clone()
            vertices_yup[:, 1], vertices_yup[:, 2] = -vertices[:, 2].clone(), vertices[:, 1].clone()

            print("[BD Bake] Rasterizing in UV space...")

            # Setup nvdiffrast
            ctx = dr.RasterizeCudaContext()

            # Prepare UVs for rasterization (convert 0-1 to -1 to 1 clip space)
            # glTF convention: V=0 at TOP of texture, V=1 at BOTTOM
            # nvdiffrast: clip Y=1 → row 0 (top), clip Y=-1 → row H-1 (bottom)
            # So V=0 should map to clip Y=1: clip_y = -(2*v - 1)
            uv_clip = uvs * 2 - 1
            uv_clip[:, 1] = -uv_clip[:, 1]  # Flip Y for glTF convention
            uvs_rast = torch.cat([
                uv_clip,
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
            mask_np = mask.cpu().numpy()

            # Interpolate 3D positions from UV pixels
            pos = dr.interpolate(vertices_yup.unsqueeze(0), rast, faces)[0][0]
            valid_pos = pos[mask].cpu().numpy()

            # Apply coordinate transform to match COLOR_FIELD space
            # Add 0.5 offset like in sampling.py
            valid_pos_cf = valid_pos + 0.5

            print(f"[BD Bake] Sampling {valid_pos_cf.shape[0]:,} texture pixels from COLOR_FIELD...")

            # Query KD-tree for nearest color field voxel
            distances, indices = tree.query(valid_pos_cf, k=1, workers=-1)

            # Sample colors
            sampled_colors = cf_colors[indices]

            # Build texture arrays
            base_color = np.zeros((texture_size, texture_size, 3), dtype=np.float32)
            base_color[mask_np] = sampled_colors[:, :3] if sampled_colors.shape[1] >= 3 else sampled_colors

            # Sample metallic
            if cf_metallic is not None:
                metallic = np.zeros((texture_size, texture_size), dtype=np.float32)
                metallic[mask_np] = cf_metallic[indices]
            else:
                metallic = np.zeros((texture_size, texture_size), dtype=np.float32)
                print("[BD Bake] No metallic in COLOR_FIELD, using 0.0")

            # Sample roughness
            if cf_roughness is not None:
                roughness = np.zeros((texture_size, texture_size), dtype=np.float32)
                roughness[mask_np] = cf_roughness[indices]
            else:
                roughness = np.full((texture_size, texture_size), 0.5, dtype=np.float32)
                print("[BD Bake] No roughness in COLOR_FIELD, using 0.5")

            # Sample alpha
            if cf_alpha is not None:
                alpha = np.ones((texture_size, texture_size), dtype=np.float32)
                alpha[mask_np] = cf_alpha[indices]
            else:
                alpha = np.ones((texture_size, texture_size), dtype=np.float32)
                print("[BD Bake] No alpha in COLOR_FIELD, using 1.0")

            # No emission in COLOR_FIELD (not typically stored)
            emission = np.zeros((texture_size, texture_size, 3), dtype=np.float32)

            # Sample outlined color field if provided
            base_color_outlined = None
            print(f"[BD Bake] === Checking color_field_outlined ===")
            print(f"[BD Bake] color_field_outlined type: {type(color_field_outlined)}")
            print(f"[BD Bake] color_field_outlined is None: {color_field_outlined is None}")

            if color_field_outlined is not None:
                print(f"[BD Bake] color_field_outlined is dict: {isinstance(color_field_outlined, dict)}")
                if isinstance(color_field_outlined, dict):
                    print(f"[BD Bake] color_field_outlined keys: {list(color_field_outlined.keys())}")
                    has_positions = 'positions' in color_field_outlined
                    print(f"[BD Bake] has 'positions' key: {has_positions}")
                    if has_positions:
                        print(f"[BD Bake] positions shape: {np.asarray(color_field_outlined['positions']).shape}")
                        print(f"[BD Bake] colors shape: {np.asarray(color_field_outlined['colors']).shape}")
                else:
                    print(f"[BD Bake] color_field_outlined repr: {repr(color_field_outlined)[:200]}")

            # Check condition explicitly
            condition_met = (color_field_outlined is not None and
                           isinstance(color_field_outlined, dict) and
                           'positions' in color_field_outlined)
            print(f"[BD Bake] Will sample outlined: {condition_met}")

            if condition_met:
                print("[BD Bake] Sampling outlined COLOR_FIELD...")
                cfo_positions = color_field_outlined.get('positions')
                cfo_colors = color_field_outlined.get('colors')

                if hasattr(cfo_positions, 'cpu'):
                    cfo_positions = cfo_positions.cpu().numpy()
                cfo_positions = np.asarray(cfo_positions, dtype=np.float32)

                if hasattr(cfo_colors, 'cpu'):
                    cfo_colors = cfo_colors.cpu().numpy()
                cfo_colors = np.asarray(cfo_colors, dtype=np.float32)

                print(f"[BD Bake] Outlined positions: {cfo_positions.shape}, colors: {cfo_colors.shape}")
                print(f"[BD Bake] Outlined color sample (first 3): {cfo_colors[:3]}")
                print(f"[BD Bake] Clean color sample (first 3): {cf_colors[:3]}")

                tree_outlined = cKDTree(cfo_positions)
                _, indices_outlined = tree_outlined.query(valid_pos_cf, k=1, workers=-1)
                sampled_outlined = cfo_colors[indices_outlined]

                base_color_outlined = np.zeros((texture_size, texture_size, 3), dtype=np.float32)
                base_color_outlined[mask_np] = sampled_outlined[:, :3] if sampled_outlined.shape[1] >= 3 else sampled_outlined

                # Check if outlined is different from clean
                diff = np.abs(base_color - base_color_outlined).mean()
                print(f"[BD Bake] Mean difference between clean and outlined textures: {diff:.6f}")
            else:
                print("[BD Bake] NOT sampling outlined - condition not met!")

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
                interp_normals = dr.interpolate(normals_yup.unsqueeze(0), rast, faces)[0][0]
                normal_map = interp_normals / (interp_normals.norm(dim=-1, keepdim=True) + 1e-8)

                del normals, normals_yup, interp_normals

            del pos, rast, vertices_yup, valid_pos
            torch.cuda.empty_cache()

            # Inpaint UV seams
            if inpaint_seams:
                print(f"[BD Bake] Inpainting UV seams (radius={inpaint_radius})...")
                mask_inv = (~mask_np).astype(np.uint8)

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

            # AO - placeholder white
            ao_np = np.ones((texture_size, texture_size, 3), dtype=np.float32)
            ao_tensor = torch.from_numpy(ao_np).unsqueeze(0).float()

            # Alpha
            alpha_tensor = torch.from_numpy(ensure_rgb(alpha)).unsqueeze(0).float()

            # Sample vertex colors from baked texture at UV coordinates
            print("[BD Bake] Sampling vertex colors from baked texture...")
            uvs_np = np.array(mesh.visual.uv)

            # glTF convention: V=0 at top (row 0), V=1 at bottom (row H-1)
            tex_coords_x = np.clip((uvs_np[:, 0] * (texture_size - 1)).astype(int), 0, texture_size - 1)
            tex_coords_y = np.clip((uvs_np[:, 1] * (texture_size - 1)).astype(int), 0, texture_size - 1)

            base_color_rgb = ensure_rgb(base_color)
            vertex_colors_rgb = base_color_rgb[tex_coords_y, tex_coords_x]

            alpha_single = alpha if alpha.ndim == 2 else alpha[..., 0]
            vertex_alpha = alpha_single[tex_coords_y, tex_coords_x]

            vertex_colors_rgba = np.concatenate([
                np.clip(vertex_colors_rgb * 255, 0, 255).astype(np.uint8),
                np.clip(vertex_alpha[..., np.newaxis] * 255, 0, 255).astype(np.uint8)
            ], axis=-1)

            # Build result mesh
            result_mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                vertex_normals=mesh.vertex_normals if hasattr(mesh, 'vertex_normals') else None,
                process=False,
            )

            result_mesh.visual = trimesh.visual.ColorVisuals(
                mesh=result_mesh,
                vertex_colors=vertex_colors_rgba
            )

            result_mesh.metadata['uv'] = mesh.visual.uv
            result_mesh.metadata['has_baked_textures'] = True

            baked_maps = ["diffuse", "normal", "metallic", "roughness", "alpha"]
            if color_field_outlined is not None:
                baked_maps.append("diffuse_outlined")

            status = f"Baked {texture_size}x{texture_size} from COLOR_FIELD ({len(cf_positions):,} voxels): {', '.join(baked_maps)}"
            print(f"[BD Bake] {status}")

            # Cleanup
            del vertices, faces, uvs
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
            # Create placeholder images to prevent downstream SaveImage crashes
            placeholder_gray = torch.full((1, 64, 64, 3), 0.5, dtype=torch.float32)
            placeholder_normal = torch.full((1, 64, 64, 3), 0.5, dtype=torch.float32)
            placeholder_normal[..., 2] = 1.0  # Default normal (0,0,1)
            placeholder_black = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            placeholder_white = torch.ones((1, 64, 64, 3), dtype=torch.float32)
            return io.NodeOutput(
                mesh,
                placeholder_gray,   # diffuse
                placeholder_gray,   # diffuse_outlined
                placeholder_normal, # normal
                placeholder_black,  # metallic
                placeholder_gray,   # roughness
                placeholder_black,  # emission
                placeholder_white,  # ao
                placeholder_white,  # alpha
                f"ERROR (COLOR_FIELD): {e}"
            )


# V3 node list
BAKE_V3_NODES = [BD_BakeTextures]

# V1 compatibility
BAKE_NODES = {
    "BD_BakeTextures": BD_BakeTextures,
}

BAKE_DISPLAY_NAMES = {
    "BD_BakeTextures": "BD Bake Textures",
}
