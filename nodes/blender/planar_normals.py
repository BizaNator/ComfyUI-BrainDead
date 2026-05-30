"""
BD_BlenderPlanarNormals - Planar group normal assignment for stylized flat-panel look.

Detects connected face groups within an angle threshold, computes each group's
average flat normal, and assigns it as custom split normals to all faces in the group.
Sharp edges are marked between groups.

Result: Many quads that visually appear as flat panels even with curved underlying geometry.
Output GLB/OBJ preserves the per-loop normals for Blender/Unreal import.
"""

import os
import tempfile
import datetime

from comfy_api.latest import io
import folder_paths

from ..mesh.types import TrimeshInput, TrimeshOutput
from .base import BlenderNodeMixin, HAS_TRIMESH

if HAS_TRIMESH:
    import trimesh


PLANAR_NORMALS_SCRIPT = '''
import bpy
import bmesh
import math
import os
import sys
from collections import deque
import mathutils

def log(msg):
    print(msg)
    sys.stdout.flush()

log("[BD PlanarNormals] Starting...")

input_path = os.environ["BLENDER_INPUT_PATH"]
output_path = os.environ["BLENDER_OUTPUT_PATH"]
sidecar_path = os.environ.get("BLENDER_ARG_SIDECAR_PATH", "")
angle_threshold = float(os.environ.get("BLENDER_ARG_ANGLE_THRESHOLD", "15.0"))
fill_holes = os.environ.get("BLENDER_ARG_FILL_HOLES", "True") == "True"

angle_limit = math.radians(angle_threshold)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

ext = os.path.splitext(input_path)[1].lower()
if ext == ".ply":
    bpy.ops.wm.ply_import(filepath=input_path)
elif ext == ".obj":
    bpy.ops.wm.obj_import(filepath=input_path)
elif ext in (".glb", ".gltf"):
    bpy.ops.import_scene.gltf(filepath=input_path)
else:
    raise ValueError(f"Unsupported format: {ext}")

obj = bpy.context.active_object
if obj is None:
    obj = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

log(f"[BD PlanarNormals] Input: {len(obj.data.vertices)} verts, {len(obj.data.polygons)} faces")

if fill_holes:
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.mesh.select_non_manifold(extend=False, use_boundary=True,
                                      use_wire=False, use_multi_face=False,
                                      use_non_contiguous=False)
    try:
        bpy.ops.mesh.fill_holes(sides=100)
        log("[BD PlanarNormals] Holes filled")
    except Exception:
        pass
    bpy.ops.object.mode_set(mode="OBJECT")

bm = bmesh.new()
bm.from_mesh(obj.data)
bm.faces.ensure_lookup_table()
bm.edges.ensure_lookup_table()
bm.normal_update()

num_faces = len(bm.faces)
log(f"[BD PlanarNormals] Grouping {num_faces} faces at {angle_threshold}deg threshold...")

# BFS planar grouping — each group is a connected set of faces within angle_limit
face_to_group = [-1] * num_faces
group_normals_list = []  # avg normal per group
group_id = 0

for start_idx in range(num_faces):
    if face_to_group[start_idx] != -1:
        continue

    queue = deque([bm.faces[start_idx]])
    face_to_group[start_idx] = group_id
    ref_normal = bm.faces[start_idx].normal.copy()
    acc_normal = ref_normal.copy()
    face_count = 1

    while queue:
        face = queue.popleft()
        for edge in face.edges:
            for neighbor in edge.link_faces:
                nidx = neighbor.index
                if face_to_group[nidx] != -1:
                    continue
                if ref_normal.angle(neighbor.normal, 0.0) <= angle_limit:
                    face_to_group[nidx] = group_id
                    acc_normal += neighbor.normal
                    face_count += 1
                    queue.append(neighbor)

    avg = acc_normal / face_count
    group_normals_list.append(avg.normalized() if avg.length > 1e-6 else mathutils.Vector((0, 0, 1)))
    group_id += 1

log(f"[BD PlanarNormals] {group_id} planar groups found")

# Mark sharp edges between different groups
sharp_count = 0
for edge in bm.edges:
    if len(edge.link_faces) == 2:
        g0 = face_to_group[edge.link_faces[0].index]
        g1 = face_to_group[edge.link_faces[1].index]
        if g0 != g1:
            edge.smooth = False
            sharp_count += 1
        else:
            edge.smooth = True
    else:
        # Boundary edge — always sharp
        edge.smooth = False

log(f"[BD PlanarNormals] {sharp_count} sharp edges marked between groups")

bm.to_mesh(obj.data)
bm.free()

# Compute custom split normals — one per loop (face-corner)
# All loops of a face get the group's average flat normal
try:
    obj.data.calc_normals_split()
except Exception:
    pass

custom_normals = []
for poly in obj.data.polygons:
    gid = face_to_group[poly.index]
    n = group_normals_list[gid]
    for _ in poly.loop_indices:
        custom_normals.append((n.x, n.y, n.z))

obj.data.normals_split_custom_set(custom_normals)
log(f"[BD PlanarNormals] Applied {len(custom_normals)} custom split normals")

# Export pipeline output (PLY — triangulated, for TRIMESH)
ext_out = os.path.splitext(output_path)[1].lower()
if ext_out == ".ply":
    bpy.ops.wm.ply_export(filepath=output_path, export_colors="SRGB", ascii_format=False)
elif ext_out in (".glb", ".gltf"):
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format="GLB",
        export_all_vertex_colors=True,
        export_normals=True,
    )

# Export sidecar GLB with full custom normals for Blender/Unreal import
if sidecar_path:
    bpy.ops.export_scene.gltf(
        filepath=sidecar_path,
        export_format="GLB",
        export_all_vertex_colors=True,
        export_normals=True,
    )
    log(f"[BD PlanarNormals] Saved normals GLB: {sidecar_path}")

log(f"[BD PlanarNormals] Done — {group_id} groups, {sharp_count} sharp edges")
log(f"[BD PlanarNormals] Pipeline output: {output_path}")
'''


class BD_BlenderPlanarNormals(BlenderNodeMixin, io.ComfyNode):
    """
    Group mesh faces into planar regions and assign flat normals per group.

    Faces connected to each other within the angle threshold are treated as one
    flat panel. Each panel gets its area-weighted average normal applied uniformly
    to every face-corner (loop) within it. Edges between panels are marked sharp.

    The resulting mesh LOOKS like flat stylized panels even though it is built from
    many small quads or triangles. Custom split normals are written into the exported
    GLB/OBJ so the look survives import into Blender or Unreal.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_BlenderPlanarNormals",
            display_name="BD Blender Planar Normals",
            category="🧠BrainDead/Blender",
            description=(
                "Group faces by angle threshold and assign each group a single flat normal. "
                "Creates the stylized 'flat panel' look from high-res quad meshes. "
                "Saves a GLB with proper custom split normals for Unreal/Blender import."
            ),
            inputs=[
                TrimeshInput("mesh"),
                io.Float.Input(
                    "angle_threshold",
                    default=15.0,
                    min=0.5,
                    max=89.0,
                    step=0.5,
                    tooltip=(
                        "Faces within this angle of each other are grouped into one flat panel. "
                        "Lower = more groups / sharper facets. Higher = fewer groups / smoother look."
                    ),
                ),
                io.Boolean.Input(
                    "fill_holes",
                    default=True,
                    optional=True,
                    tooltip="Fill open boundary edges before processing",
                ),
                io.Int.Input(
                    "timeout",
                    default=300,
                    min=60,
                    max=1800,
                    optional=True,
                    tooltip="Max Blender processing time in seconds",
                ),
            ],
            outputs=[
                TrimeshOutput(display_name="mesh"),
                io.String.Output(display_name="glb_path"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh,
        angle_threshold: float,
        fill_holes: bool = True,
        timeout: int = 300,
    ) -> io.NodeOutput:
        if not HAS_TRIMESH:
            return io.NodeOutput(mesh, "", "ERROR: trimesh not installed")

        available, msg = cls._check_blender()
        if not available:
            return io.NodeOutput(mesh, "", f"ERROR: {msg}")

        if mesh is None:
            return io.NodeOutput(None, "", "ERROR: No input mesh")

        orig_faces = len(mesh.faces) if hasattr(mesh, "faces") else 0

        input_path = None
        output_path = None
        try:
            input_path = cls._mesh_to_temp_file(mesh, suffix=".ply")
            fd, output_path = tempfile.mkstemp(suffix=".ply")
            os.close(fd)

            # Sidecar GLB with proper custom normals saved to output dir
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = folder_paths.get_output_directory()
            sidecar_path = os.path.join(out_dir, f"planar_normals_{ts}.glb")

            success, message, log_lines = cls._run_blender_script(
                PLANAR_NORMALS_SCRIPT,
                input_path,
                output_path,
                extra_args={
                    "angle_threshold": angle_threshold,
                    "fill_holes": fill_holes,
                    "sidecar_path": sidecar_path,
                },
                timeout=timeout,
            )

            if not success:
                ctx = "\n".join(log_lines[-10:]) if log_lines else ""
                print(f"[BD PlanarNormals] FAILED: {message}")
                if ctx:
                    print(f"[BD PlanarNormals] Log:\n{ctx}")
                return io.NodeOutput(mesh, "", f"ERROR: {message}")

            result_mesh = cls._load_mesh_from_file(output_path)
            new_faces = len(result_mesh.faces)

            # Pull group/sharp counts from log
            groups_line = next((l for l in reversed(log_lines) if "groups found" in l), "")
            sharp_line = next((l for l in reversed(log_lines) if "sharp edges marked" in l), "")
            num_groups = groups_line.split()[2] if groups_line else "?"
            num_sharp = sharp_line.split()[1] if sharp_line else "?"

            status = (
                f"PlanarNormals: {orig_faces:,} faces | "
                f"{num_groups} groups @ {angle_threshold}° | "
                f"{num_sharp} sharp edges"
            )
            glb_out = sidecar_path if os.path.exists(sidecar_path) else ""
            return io.NodeOutput(result_mesh, glb_out, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return io.NodeOutput(mesh, "", f"ERROR: {e}")

        finally:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            if output_path and os.path.exists(output_path):
                os.remove(output_path)


# Registration
PLANAR_NORMALS_V3_NODES = [BD_BlenderPlanarNormals]

PLANAR_NORMALS_NODES = {
    "BD_BlenderPlanarNormals": BD_BlenderPlanarNormals,
}

PLANAR_NORMALS_DISPLAY_NAMES = {
    "BD_BlenderPlanarNormals": "BD Blender Planar Normals",
}
