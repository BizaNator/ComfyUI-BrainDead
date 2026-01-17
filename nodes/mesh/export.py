"""
V3 API Mesh export node for saving meshes with vertex colors.

BD_ExportMeshWithColors - Export mesh with vertex colors to GLB/PLY/OBJ
"""

import os
from glob import glob

from comfy_api.latest import io

# Check for optional trimesh support
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


class BD_ExportMeshWithColors(io.ComfyNode):
    """
    Export a mesh with vertex colors to file (GLB, PLY, OBJ).

    Designed to work with BD_SampleVoxelgridColors output.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_ExportMeshWithColors",
            display_name="BD Export Mesh With Colors",
            category="🧠BrainDead/Mesh",
            description="Export mesh with vertex colors to GLB/PLY/OBJ. Use after BD_SampleVoxelgridColors.",
            is_output_node=True,
            inputs=[
                io.Mesh.Input("mesh"),
                io.String.Input("filename", default="mesh_colored"),
                io.Combo.Input("format", options=["glb", "ply", "obj"], default="glb"),
                io.String.Input("name_prefix", default="", optional=True, tooltip="Prepended to filename. Supports subdirs (e.g., 'Project/Name')"),
                io.Boolean.Input("auto_increment", default=True, optional=True, tooltip="Auto-increment filename to avoid overwriting"),
            ],
            outputs=[
                io.String.Output(display_name="file_path"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, mesh, filename: str, format: str = "glb",
                name_prefix: str = "", auto_increment: bool = True) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput("", "ERROR: trimesh not installed")

        if mesh is None:
            return io.NodeOutput("", "ERROR: mesh is None")

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
            existing = glob(pattern)

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

            return io.NodeOutput(file_path, status)

        except Exception as e:
            return io.NodeOutput("", f"ERROR: {e}")


# V3 node list for extension
MESH_EXPORT_V3_NODES = [BD_ExportMeshWithColors]

# V1 compatibility - NODE_CLASS_MAPPINGS dict
MESH_EXPORT_NODES = {
    "BD_ExportMeshWithColors": BD_ExportMeshWithColors,
}

MESH_EXPORT_DISPLAY_NAMES = {
    "BD_ExportMeshWithColors": "BD Export Mesh With Colors",
}
