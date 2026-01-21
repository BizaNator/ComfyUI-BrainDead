"""
BD_MeshExportBundle - Export all mesh LODs and textures as organized bundle.

Uses standard output folder with filename/name_prefix pattern.
"""

import os
import json
from glob import glob
from datetime import datetime

import folder_paths
from comfy_api.latest import io

from ..blender.base import BlenderNodeMixin, HAS_TRIMESH

if HAS_TRIMESH:
    import trimesh
    import numpy as np

from .types import TrimeshInput


class BD_MeshExportBundle(BlenderNodeMixin, io.ComfyNode):
    """
    Export all mesh LODs and textures as an organized bundle.

    Creates a complete asset package with:
    - Multiple mesh LODs (original, highpoly, lowpoly)
    - All PBR textures as separate PNG files
    - Multiple export formats (GLB, FBX, PLY)
    - Manifest JSON with metadata

    Uses standard output pattern: output/{name_prefix}_{filename}/
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_MeshExportBundle",
            display_name="BD Export Bundle",
            category="🧠BrainDead/Mesh",
            description="""Export complete asset bundle with meshes and textures.

Output: output/{name_prefix}_{filename}/ (or output/{filename}/ if no prefix)

Files created:
- {filename}_lowpoly.glb/fbx/ply - Low-poly with UVs
- {filename}_highpoly.glb/fbx - High-poly with UVs (optional)
- {filename}_original.ply - Original with vertex colors (optional)
- {filename}_diffuse.png, _normal.png, _alpha.png, etc.
- {filename}_manifest.json""",
            inputs=[
                TrimeshInput("lowpoly_mesh", tooltip="Low-poly mesh with UVs and materials"),
                TrimeshInput("highpoly_mesh", optional=True, tooltip="High-poly mesh with UVs (optional)"),
                TrimeshInput("original_mesh", optional=True, tooltip="Original mesh with vertex colors (optional)"),
                io.Image.Input("diffuse", optional=True, tooltip="Diffuse/albedo texture"),
                io.Image.Input("diffuse_outlined", optional=True, tooltip="Diffuse with edge lines"),
                io.Image.Input("normal", optional=True, tooltip="Normal map"),
                io.Image.Input("metallic", optional=True, tooltip="Metallic map"),
                io.Image.Input("roughness", optional=True, tooltip="Roughness map"),
                io.Image.Input("emission", optional=True, tooltip="Emission map"),
                io.Image.Input("ao", optional=True, tooltip="Ambient occlusion map"),
                io.Image.Input("alpha", optional=True, tooltip="Alpha/transparency map"),
                io.String.Input(
                    "filename",
                    default="mesh_bundle",
                    tooltip="Base name for exported files and folder",
                ),
                io.String.Input(
                    "name_prefix",
                    default="",
                    optional=True,
                    tooltip="Prepended to filename. Supports subdirs (e.g., 'Project/Name')",
                ),
                io.Boolean.Input(
                    "auto_increment",
                    default=True,
                    optional=True,
                    tooltip="Auto-increment folder name to avoid overwriting",
                ),
                io.Combo.Input(
                    "formats",
                    options=["glb", "glb+ply", "glb+fbx", "glb+fbx+ply", "ply_only"],
                    default="glb+ply",
                    tooltip="Export formats for meshes",
                ),
                io.Boolean.Input(
                    "export_original",
                    default=False,
                    tooltip="Export original high-poly mesh (can be large)",
                ),
                io.Boolean.Input(
                    "export_highpoly",
                    default=True,
                    tooltip="Export high-poly LOD mesh",
                ),
            ],
            outputs=[
                io.String.Output(display_name="output_path"),
                io.String.Output(display_name="file_list"),
                io.String.Output(display_name="status"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        lowpoly_mesh,
        highpoly_mesh=None,
        original_mesh=None,
        diffuse=None,
        diffuse_outlined=None,
        normal=None,
        metallic=None,
        roughness=None,
        emission=None,
        ao=None,
        alpha=None,
        filename: str = "mesh_bundle",
        name_prefix: str = "",
        auto_increment: bool = True,
        formats: str = "glb+ply",
        export_original: bool = False,
        export_highpoly: bool = True,
    ) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput("", "", "ERROR: trimesh not installed")

        if lowpoly_mesh is None:
            return io.NodeOutput("", "", "ERROR: No lowpoly mesh provided")

        # Setup output directory using standard pattern
        output_base = folder_paths.get_output_directory()

        # Concatenate name_prefix + filename (same pattern as BD_ExportMeshWithColors)
        full_name = f"{name_prefix}_{filename}" if name_prefix else filename

        # Handle subdirectories if full_name contains path separators
        full_name = full_name.replace('\\', '/')
        if '/' in full_name:
            parts = full_name.rsplit('/', 1)
            subdir, base_filename = parts
            parent_dir = os.path.join(output_base, subdir)
        else:
            parent_dir = output_base
            base_filename = full_name

        os.makedirs(parent_dir, exist_ok=True)

        # Auto-increment to avoid overwriting existing bundles
        if auto_increment:
            # Find existing folders with this pattern
            pattern = os.path.join(parent_dir, f"{base_filename}_*")
            existing = [d for d in glob(pattern) if os.path.isdir(d)]

            if existing:
                # Extract numbers and find max
                numbers = []
                for d in existing:
                    try:
                        num_str = os.path.basename(d).replace(f"{base_filename}_", "")
                        numbers.append(int(num_str))
                    except:
                        pass
                next_num = max(numbers) + 1 if numbers else 1
            else:
                next_num = 1

            folder_name = f"{base_filename}_{next_num:03d}"
        else:
            folder_name = base_filename

        output_dir = os.path.join(parent_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"[BD Export Bundle] Output directory: {output_dir}")
        print(f"[BD Export Bundle] Base name: {filename}")
        print(f"[BD Export Bundle] Formats: {formats}")

        exported_files = []
        manifest = {
            "name": filename,
            "created": datetime.now().isoformat(),
            "meshes": {},
            "textures": {},
        }

        # Parse formats
        export_glb = "glb" in formats
        export_fbx = "fbx" in formats
        export_ply = "ply" in formats or formats == "ply_only"

        try:
            import torch
            from PIL import Image

            # Helper to build mesh with PBR material for GLB export
            def build_textured_mesh(mesh, diffuse_tex, normal_tex=None, metallic_tex=None,
                                   roughness_tex=None, alpha_tex=None):
                """Build a mesh with TextureVisuals for GLB export."""
                # Check if mesh has UVs (either in visual or metadata)
                uvs = None
                if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                    uvs = mesh.visual.uv
                elif hasattr(mesh, 'metadata') and 'uv' in mesh.metadata:
                    uvs = mesh.metadata['uv']

                if uvs is None or diffuse_tex is None:
                    return mesh  # Can't build textured mesh without UVs or diffuse

                # Convert diffuse tensor to PIL Image
                if isinstance(diffuse_tex, torch.Tensor):
                    diffuse_np = diffuse_tex[0].cpu().numpy()
                    diffuse_np = np.clip(diffuse_np * 255, 0, 255).astype(np.uint8)
                else:
                    diffuse_np = diffuse_tex

                # Handle alpha
                if alpha_tex is not None:
                    if isinstance(alpha_tex, torch.Tensor):
                        alpha_np = alpha_tex[0].cpu().numpy()
                        alpha_np = np.clip(alpha_np * 255, 0, 255).astype(np.uint8)
                        if alpha_np.ndim == 3:
                            alpha_np = alpha_np[..., 0]  # Take first channel
                    else:
                        alpha_np = alpha_tex
                        if alpha_np.ndim == 3:
                            alpha_np = alpha_np[..., 0]
                    # Add alpha channel to diffuse
                    if diffuse_np.shape[-1] == 3:
                        diffuse_np = np.concatenate([diffuse_np, alpha_np[..., np.newaxis]], axis=-1)

                diffuse_img = Image.fromarray(diffuse_np)

                # Build metallic-roughness texture (glTF format: R=0, G=roughness, B=metallic)
                mr_tex = None
                if metallic_tex is not None or roughness_tex is not None:
                    h, w = diffuse_np.shape[:2]

                    if metallic_tex is not None:
                        if isinstance(metallic_tex, torch.Tensor):
                            metallic_np = metallic_tex[0].cpu().numpy()
                            metallic_np = np.clip(metallic_np * 255, 0, 255).astype(np.uint8)
                        else:
                            metallic_np = metallic_tex
                        if metallic_np.ndim == 3:
                            metallic_np = metallic_np[..., 0]
                    else:
                        metallic_np = np.zeros((h, w), dtype=np.uint8)

                    if roughness_tex is not None:
                        if isinstance(roughness_tex, torch.Tensor):
                            roughness_np = roughness_tex[0].cpu().numpy()
                            roughness_np = np.clip(roughness_np * 255, 0, 255).astype(np.uint8)
                        else:
                            roughness_np = roughness_tex
                        if roughness_np.ndim == 3:
                            roughness_np = roughness_np[..., 0]
                    else:
                        roughness_np = np.full((h, w), 128, dtype=np.uint8)

                    mr_np = np.stack([
                        np.zeros_like(metallic_np),  # R = 0
                        roughness_np,                 # G = roughness
                        metallic_np                   # B = metallic
                    ], axis=-1)
                    mr_tex = Image.fromarray(mr_np)

                # Create PBR material
                has_alpha = diffuse_np.shape[-1] == 4
                material = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=diffuse_img,
                    baseColorFactor=np.array([1.0, 1.0, 1.0, 1.0]),
                    metallicRoughnessTexture=mr_tex,
                    metallicFactor=1.0,
                    roughnessFactor=1.0,
                    alphaMode='BLEND' if has_alpha else 'OPAQUE',
                    doubleSided=True,
                )

                # Build new mesh with texture visuals
                textured_mesh = trimesh.Trimesh(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    vertex_normals=mesh.vertex_normals if hasattr(mesh, 'vertex_normals') else None,
                    process=False,
                )
                textured_mesh.visual = trimesh.visual.TextureVisuals(
                    uv=uvs,
                    material=material,
                )
                return textured_mesh

            # Export lowpoly mesh
            print(f"[BD Export Bundle] Exporting lowpoly ({len(lowpoly_mesh.vertices):,} verts)...")

            if export_glb:
                path = os.path.join(output_dir, f"{filename}_lowpoly.glb")
                # Build textured mesh if we have diffuse texture
                if diffuse is not None:
                    glb_mesh = build_textured_mesh(lowpoly_mesh, diffuse, normal, metallic, roughness, alpha)
                    glb_mesh.export(path, file_type='glb')
                else:
                    lowpoly_mesh.export(path, file_type='glb')
                exported_files.append(f"{filename}_lowpoly.glb")
                manifest["meshes"]["lowpoly_glb"] = {
                    "file": f"{filename}_lowpoly.glb",
                    "vertices": len(lowpoly_mesh.vertices),
                    "faces": len(lowpoly_mesh.faces),
                    "has_uvs": True if diffuse is not None else (hasattr(lowpoly_mesh.visual, 'uv') and lowpoly_mesh.visual.uv is not None),
                    "has_material": diffuse is not None,
                }

            if export_ply:
                path = os.path.join(output_dir, f"{filename}_lowpoly.ply")
                # PLY uses vertex colors directly from mesh
                lowpoly_mesh.export(path, file_type='ply')
                exported_files.append(f"{filename}_lowpoly.ply")
                manifest["meshes"]["lowpoly_ply"] = {
                    "file": f"{filename}_lowpoly.ply",
                    "vertices": len(lowpoly_mesh.vertices),
                    "faces": len(lowpoly_mesh.faces),
                    "has_vertex_colors": hasattr(lowpoly_mesh.visual, 'vertex_colors') and lowpoly_mesh.visual.vertex_colors is not None,
                }

            if export_fbx:
                path = os.path.join(output_dir, f"{filename}_lowpoly.fbx")
                # For FBX, use textured mesh if available
                if diffuse is not None:
                    fbx_mesh = build_textured_mesh(lowpoly_mesh, diffuse, normal, metallic, roughness, alpha)
                    success = cls._export_fbx_via_blender(fbx_mesh, path)
                else:
                    success = cls._export_fbx_via_blender(lowpoly_mesh, path)
                if success:
                    exported_files.append(f"{filename}_lowpoly.fbx")
                    manifest["meshes"]["lowpoly_fbx"] = {
                        "file": f"{filename}_lowpoly.fbx",
                        "vertices": len(lowpoly_mesh.vertices),
                        "faces": len(lowpoly_mesh.faces),
                    }

            # Export highpoly mesh
            if export_highpoly and highpoly_mesh is not None:
                print(f"[BD Export Bundle] Exporting highpoly ({len(highpoly_mesh.vertices):,} verts)...")

                if export_glb:
                    path = os.path.join(output_dir, f"{filename}_highpoly.glb")
                    # Build textured mesh if we have diffuse texture
                    if diffuse is not None:
                        glb_mesh = build_textured_mesh(highpoly_mesh, diffuse, normal, metallic, roughness, alpha)
                        glb_mesh.export(path, file_type='glb')
                    else:
                        highpoly_mesh.export(path, file_type='glb')
                    exported_files.append(f"{filename}_highpoly.glb")
                    manifest["meshes"]["highpoly_glb"] = {
                        "file": f"{filename}_highpoly.glb",
                        "vertices": len(highpoly_mesh.vertices),
                        "faces": len(highpoly_mesh.faces),
                        "has_material": diffuse is not None,
                    }

                if export_fbx:
                    path = os.path.join(output_dir, f"{filename}_highpoly.fbx")
                    # For FBX, use textured mesh if available
                    if diffuse is not None:
                        fbx_mesh = build_textured_mesh(highpoly_mesh, diffuse, normal, metallic, roughness, alpha)
                        success = cls._export_fbx_via_blender(fbx_mesh, path)
                    else:
                        success = cls._export_fbx_via_blender(highpoly_mesh, path)
                    if success:
                        exported_files.append(f"{filename}_highpoly.fbx")
                        manifest["meshes"]["highpoly_fbx"] = {
                            "file": f"{filename}_highpoly.fbx",
                            "vertices": len(highpoly_mesh.vertices),
                            "faces": len(highpoly_mesh.faces),
                        }

            # Export original mesh (PLY only - vertex colors, no UV)
            if export_original and original_mesh is not None:
                print(f"[BD Export Bundle] Exporting original ({len(original_mesh.vertices):,} verts)...")
                path = os.path.join(output_dir, f"{filename}_original.ply")
                original_mesh.export(path, file_type='ply')
                exported_files.append(f"{filename}_original.ply")
                manifest["meshes"]["original_ply"] = {
                    "file": f"{filename}_original.ply",
                    "vertices": len(original_mesh.vertices),
                    "faces": len(original_mesh.faces),
                    "vertex_colors": hasattr(original_mesh.visual, 'vertex_colors') and original_mesh.visual.vertex_colors is not None,
                }

            # Export textures
            texture_map = {
                "diffuse": diffuse,
                "diffuse_outlined": diffuse_outlined,
                "normal": normal,
                "metallic": metallic,
                "roughness": roughness,
                "emission": emission,
                "ao": ao,
                "alpha": alpha,
            }

            for tex_name, tex_tensor in texture_map.items():
                if tex_tensor is not None:
                    print(f"[BD Export Bundle] Exporting {tex_name} texture...")
                    path = os.path.join(output_dir, f"{filename}_{tex_name}.png")

                    # Convert ComfyUI IMAGE tensor (BHWC) to PIL Image
                    if isinstance(tex_tensor, torch.Tensor):
                        # Take first batch, convert to numpy, scale to 0-255
                        img_np = tex_tensor[0].cpu().numpy()
                        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

                        # Handle different channel counts
                        if img_np.shape[-1] == 1:
                            # Grayscale
                            img = Image.fromarray(img_np[..., 0], mode='L')
                        elif img_np.shape[-1] == 3:
                            # RGB
                            img = Image.fromarray(img_np, mode='RGB')
                        elif img_np.shape[-1] == 4:
                            # RGBA
                            img = Image.fromarray(img_np, mode='RGBA')
                        else:
                            print(f"[BD Export Bundle] Warning: Unexpected channels for {tex_name}")
                            continue

                        img.save(path, 'PNG')
                        exported_files.append(f"{filename}_{tex_name}.png")
                        manifest["textures"][tex_name] = {
                            "file": f"{filename}_{tex_name}.png",
                            "width": img.width,
                            "height": img.height,
                        }

            # Write manifest
            manifest_path = os.path.join(output_dir, f"{filename}_manifest.json")
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            exported_files.append(f"{filename}_manifest.json")

            # Build status
            mesh_count = len([k for k in manifest["meshes"]])
            tex_count = len([k for k in manifest["textures"]])
            status = f"Exported {mesh_count} meshes, {tex_count} textures to {output_dir}"

            file_list = "\n".join(exported_files)
            print(f"[BD Export Bundle] {status}")
            print(f"[BD Export Bundle] Files:\n{file_list}")

            return io.NodeOutput(output_dir, file_list, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return io.NodeOutput(output_dir, "", f"ERROR: {e}")

    @classmethod
    def _export_fbx_via_blender(cls, mesh, output_path: str) -> bool:
        """Export mesh to FBX using Blender (better FBX support than trimesh)."""
        import tempfile

        available, msg = cls._check_blender()
        if not available:
            print(f"[BD Export Bundle] FBX export skipped: {msg}")
            return False

        # Save mesh to temp GLB, then convert via Blender
        try:
            fd, temp_glb = tempfile.mkstemp(suffix='.glb')
            os.close(fd)
            mesh.export(temp_glb, file_type='glb')

            script = f'''
import bpy
import os

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import GLB
bpy.ops.import_scene.gltf(filepath="{temp_glb}")

# Export FBX
bpy.ops.export_scene.fbx(
    filepath="{output_path}",
    use_selection=False,
    apply_scale_options='FBX_SCALE_ALL',
    axis_forward='-Z',
    axis_up='Y',
)
print("[BD Export Bundle] FBX export complete")
'''

            success, message, _ = cls._run_blender_script(
                script,
                temp_glb,  # Dummy input
                output_path,
                timeout=120,
            )

            os.remove(temp_glb)
            return success and os.path.exists(output_path)

        except Exception as e:
            print(f"[BD Export Bundle] FBX export failed: {e}")
            return False


# V3 node list
EXPORT_BUNDLE_V3_NODES = [BD_MeshExportBundle]

# V1 compatibility
EXPORT_BUNDLE_NODES = {
    "BD_MeshExportBundle": BD_MeshExportBundle,
}

EXPORT_BUNDLE_DISPLAY_NAMES = {
    "BD_MeshExportBundle": "BD Export Bundle",
}
