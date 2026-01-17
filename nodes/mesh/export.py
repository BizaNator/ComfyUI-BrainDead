"""
Mesh export node for saving meshes with vertex colors.

BD_ExportMeshWithColors - Export mesh with vertex colors to GLB/PLY/OBJ
"""

import os
import glob

# Check for optional trimesh support
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


class BD_ExportMeshWithColors:
    """
    Export a mesh with vertex colors to file (GLB, PLY, OBJ).

    Designed to work with BD_SampleVoxelgridColors output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "filename": ("STRING", {"default": "mesh_colored"}),
                "format": (["glb", "ply", "obj"], {"default": "glb"}),
            },
            "optional": {
                "name_prefix": ("STRING", {"default": "", "tooltip": "Prepended to filename: {name_prefix}_{filename}. Supports subdirs (e.g., 'Project/Name')"}),
                "auto_increment": ("BOOLEAN", {"default": True, "tooltip": "Auto-increment filename to avoid overwriting"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_path", "status")
    FUNCTION = "export_mesh"
    CATEGORY = "BrainDead/Mesh"
    OUTPUT_NODE = True
    DESCRIPTION = """
Export mesh with vertex colors to file.

Formats:
- GLB: Best for game engines, preserves vertex colors
- PLY: Good for Blender import, preserves vertex colors
- OBJ: Basic format, vertex colors may not be preserved

Use after BD_SampleVoxelgridColors to export colored mesh
for decimation in Blender.

Options:
- name_prefix: Prepended to filename ({prefix}_{filename})
  Supports subdirs: "Project/Name" + "mesh" = Project/Name_mesh_001.ext
- auto_increment: Adds _001, _002 etc. to avoid overwriting
"""

    def export_mesh(self, mesh, filename, format="glb", name_prefix="", auto_increment=True):
        if not HAS_TRIMESH:
            return ("", "ERROR: trimesh not installed")

        if mesh is None:
            return ("", "ERROR: mesh is None")

        import folder_paths
        base_output_dir = folder_paths.get_output_directory()

        # Concatenate name_prefix + filename (same pattern as cache nodes)
        full_name = f"{name_prefix}_{filename}" if name_prefix else filename

        # Handle subdirectories if full_name contains path separators
        full_name = full_name.replace('\\', '/')
        if '/' in full_name:
            parts = full_name.rsplit('/', 1)
            subdir, base_filename = parts
            output_dir = os.path.join(base_output_dir, subdir)
        else:
            output_dir = base_output_dir
            base_filename = full_name

        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Auto-increment to avoid overwriting
        if auto_increment:
            # Find existing files with this pattern
            pattern = os.path.join(output_dir, f"{base_filename}_*.{format}")
            existing = glob.glob(pattern)

            if existing:
                # Extract numbers and find max
                numbers = []
                for f in existing:
                    try:
                        # Extract _NNN before extension
                        num_str = os.path.basename(f).replace(f".{format}", "").split("_")[-1]
                        numbers.append(int(num_str))
                    except:
                        pass
                next_num = max(numbers) + 1 if numbers else 1
            else:
                next_num = 1

            final_filename = f"{base_filename}_{next_num:03d}.{format}"
        else:
            final_filename = f"{base_filename}.{format}"

        file_path = os.path.join(output_dir, final_filename)

        try:
            # Check if mesh has vertex colors
            has_colors = hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors')
            color_info = ""
            if has_colors and mesh.visual.vertex_colors is not None:
                color_info = f" with {len(mesh.visual.vertex_colors)} vertex colors"
            elif hasattr(mesh, 'vertex_colors') and mesh.vertex_colors is not None:
                color_info = f" with {len(mesh.vertex_colors)} vertex colors"

            print(f"[BD Export Mesh] Exporting to {file_path}{color_info}...")

            # Export based on format
            if format == "glb":
                mesh.export(file_path, file_type='glb')
            elif format == "ply":
                mesh.export(file_path, file_type='ply')
            elif format == "obj":
                mesh.export(file_path, file_type='obj')

            # Get file size
            file_size = os.path.getsize(file_path)
            size_str = f"{file_size / 1024 / 1024:.1f}MB" if file_size > 1024*1024 else f"{file_size / 1024:.1f}KB"

            vert_count = len(mesh.vertices) if hasattr(mesh, 'vertices') else 0
            face_count = len(mesh.faces) if hasattr(mesh, 'faces') and mesh.faces is not None else 0

            status = f"Exported {format.upper()}: {vert_count} verts, {face_count} faces ({size_str})"
            print(f"[BD Export Mesh] {status}")

            return (file_path, status)

        except Exception as e:
            return ("", f"ERROR: {e}")


# Node exports
MESH_EXPORT_NODES = {
    "BD_ExportMeshWithColors": BD_ExportMeshWithColors,
}

MESH_EXPORT_DISPLAY_NAMES = {
    "BD_ExportMeshWithColors": "BD Export Mesh With Colors",
}
