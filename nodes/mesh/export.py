"""
V3 API Mesh export node for saving meshes with vertex colors.

BD_ExportMeshWithColors - Export mesh with vertex colors to GLB/PLY/OBJ
"""

import os
from glob import glob

import numpy as np
from comfy_api.latest import io

# Check for optional trimesh support
try:
    import trimesh
    import trimesh.visual
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

# Import custom TRIMESH type (matches TRELLIS2)
from .types import TrimeshInput, TrimeshOutput


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
            description="Export mesh with vertex colors to GLB/PLY/OBJ. Use after BD_SampleVoxelgridColors. "
                       "Wire context_id from BD_SaveContext for template-based naming.",
            is_output_node=True,
            inputs=[
                TrimeshInput("mesh"),
                io.String.Input("filename", default="mesh_colored"),
                io.Combo.Input("format", options=["glb", "ply", "obj"], default="glb"),
                io.String.Input("name_prefix", default="", optional=True, tooltip="Prepended to filename. Supports subdirs (e.g., 'Project/Name')"),
                io.Boolean.Input("auto_increment", default=True, optional=True, tooltip="Auto-increment filename to avoid overwriting"),
                io.String.Input("context_id", default="", optional=True,
                                tooltip="BD_SaveContext id for template-based naming + foldering. Empty + exactly "
                                        "one registered context = auto-pick. When it resolves, the template (with "
                                        "%suffix%) drives the path; filename/name_prefix pass through as "
                                        "%filename%/%name_prefix%."),
                io.String.Input("suffix", default="", optional=True,
                                tooltip="Per-save suffix → %suffix% in the template (e.g. '_body', '_combined'). "
                                        "Wirable — e.g. connect BD CubePart Get Part's `name` output. Only used when context_id resolves."),
                io.String.Input("context_custom_vars", multiline=True, default="", optional=True,
                                tooltip="Extra key=value template vars (one per line), layered on top of the context "
                                        "(e.g. subfolder=parts). Only used when context_id resolves."),
            ],
            outputs=[
                io.String.Output(display_name="file_path"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, mesh, filename: str, format: str = "glb",
                name_prefix: str = "", auto_increment: bool = True,
                context_id: str = "", suffix: str = "",
                context_custom_vars: str = "") -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput("", "ERROR: trimesh not installed")

        if mesh is None:
            return io.NodeOutput("", "ERROR: mesh is None")

        import folder_paths
        base_output_dir = folder_paths.get_output_directory()

        # --- Path resolution -------------------------------------------------
        # If a BD_SaveContext is wired (or auto-picked from a single registered
        # context), resolve the path from its template so naming/foldering follow
        # the shared BD save system. Otherwise fall back to filename/name_prefix.
        from ..cache.save_context import (
            resolve_context_path, get_context, auto_pick_context,
        )
        effective_ctx_id = context_id.strip() if context_id else ""
        if not effective_ctx_id:
            picked = auto_pick_context()
            if picked is not None:
                effective_ctx_id = picked
        use_context = bool(effective_ctx_id) and get_context(effective_ctx_id) is not None

        if use_context:
            # Context owns auto-increment (set on BD_SaveContext); ignore node toggle.
            file_path, _rel = resolve_context_path(
                effective_ctx_id, suffix, format,
                node_filename=filename, node_name_prefix=name_prefix,
                node_custom_vars=context_custom_vars,
            )
            output_dir = os.path.dirname(file_path)
            os.makedirs(output_dir, exist_ok=True)
        else:
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
            # Extract vertex colors — check vertex_attributes first (safe for TextureVisuals meshes).
            # Accessing TextureVisuals.vertex_colors triggers rasterization and can silently
            # convert the visual to ColorVisuals, destroying UV on the mesh object.
            vc = None
            color_info = ""
            if hasattr(mesh, 'vertex_attributes') and 'COLOR_0' in mesh.vertex_attributes:
                raw = np.array(mesh.vertex_attributes['COLOR_0'])
                if len(raw) == len(mesh.vertices):
                    vc = raw
                    color_info = f" with {len(vc)} vertex colors (vertex_attributes)"
            if vc is None and hasattr(mesh, 'visual') and mesh.visual is not None:
                try:
                    import trimesh.visual as _tv_ex
                    if not isinstance(mesh.visual, _tv_ex.TextureVisuals):
                        raw = mesh.visual.vertex_colors
                        if raw is not None and len(raw) == len(mesh.vertices):
                            vc = np.array(raw)
                            color_info = f" with {len(vc)} vertex colors"
                except Exception:
                    pass

            print(f"[BD Export Mesh] Exporting to {file_path}{color_info}...")

            # Export based on format
            if format == "glb":
                # Export with both TEXCOORD_0 (UV) and COLOR_0 (vertex colors).
                # DO NOT use ColorVisuals — it destroys UV. Use vertex_attributes instead.
                export_mesh = trimesh.Trimesh(
                    vertices=mesh.vertices.copy(),
                    faces=mesh.faces.copy(),
                    process=False,
                )
                import trimesh.visual as _tv_ex2
                if hasattr(mesh, "visual") and isinstance(mesh.visual, _tv_ex2.TextureVisuals):
                    try:
                        _uv = mesh.visual.uv
                        if _uv is not None and len(_uv) == len(export_mesh.vertices):
                            import trimesh.visual.material as _mat_ex
                            import io as _io_ex
                            from PIL import Image as _PILex
                            _buf = _io_ex.BytesIO()
                            _PILex.new("RGBA", (1, 1), (255, 255, 255, 255)).save(_buf, format="PNG")
                            _buf.seek(0)
                            _mat = _mat_ex.PBRMaterial(baseColorTexture=_PILex.open(_buf))
                            export_mesh.visual = _tv_ex2.TextureVisuals(uv=_uv.copy(), material=_mat)
                            print("[BD Export Mesh] TextureVisuals with 1x1 placeholder — TEXCOORD_0 will be written")
                    except Exception as _e:
                        print("[BD Export Mesh] UV copy failed: " + str(_e))
                if vc is not None:
                    export_mesh.vertex_attributes["COLOR_0"] = vc
                export_mesh.export(file_path, file_type='glb')
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
