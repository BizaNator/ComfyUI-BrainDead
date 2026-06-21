"""
Headless Blender: render a rigged FBX with visible bone skeleton.

Creates custom bone visualization geometry (orange spheres at joints, white-blue
cylinders along bones) that renders in EEVEE, overlaid on the character mesh
(optional, semi-transparent). No viewport overlays needed — all geometry.

Output modes:
  "all_views"   — 4 renders: front/side/back/perspective (orthographic + persp)
                  written to {OUTPUT_BASE}_front.png, _side.png, _back.png, _perspective.png
  "front"       — orthographic front view
  "side"        — orthographic right-side view
  "back"        — orthographic back view
  "perspective" — 3/4 perspective view

Environment variables:
  BLENDER_INPUT_PATH       — input FBX (rigged, from BD_AutoRigMIA or BD_AutoRigUEFN)
  BLENDER_OUTPUT_PATH      — output path base (used as stem for all_views, literal for single)
  BLENDER_ARG_VIEW_MODE    — one of the modes above (default: all_views)
  BLENDER_ARG_MESH_OPACITY — 0.0 (hide mesh) to 1.0 (opaque) (default: 0.25)
  BLENDER_ARG_RESOLUTION   — render resolution in pixels (default: 768)
"""

import bpy
import os
import math
from mathutils import Vector

INPUT_FBX   = os.environ["BLENDER_INPUT_PATH"]
OUTPUT_BASE = os.environ["BLENDER_OUTPUT_PATH"]
VIEW_MODE   = os.environ.get("BLENDER_ARG_VIEW_MODE", "all_views")
OPACITY     = float(os.environ.get("BLENDER_ARG_MESH_OPACITY", "0.25"))
RESOLUTION  = int(os.environ.get("BLENDER_ARG_RESOLUTION", "768"))


def log(msg):
    print(msg, flush=True)


# ── Scene setup ───────────────────────────────────────────────────────────────

bpy.ops.wm.read_factory_settings(use_empty=True)

log(f"[RigPreview] Importing: {INPUT_FBX}")
bpy.ops.import_scene.fbx(filepath=INPUT_FBX, automatic_bone_orientation=True, use_anim=False)
bpy.context.view_layer.update()

arms   = [o for o in bpy.data.objects if o.type == "ARMATURE"]
meshes = [o for o in bpy.data.objects if o.type == "MESH"]

if not arms:
    raise RuntimeError("No armature found in input FBX")

arm = arms[0]
arm.data.pose_position = "REST"
bpy.context.view_layer.update()

log(f"[RigPreview] Armature '{arm.name}': {len(arm.data.bones)} bones")
log(f"[RigPreview] Meshes: {[m.name for m in meshes]}")


# ── Materials ─────────────────────────────────────────────────────────────────

def make_emit_mat(name, color_rgb, alpha=1.0, strength=2.5):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out  = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (*color_rgb, 1.0)
    emit.inputs["Strength"].default_value = strength
    if alpha < 1.0:
        transp = nodes.new("ShaderNodeBsdfTransparent")
        mix    = nodes.new("ShaderNodeMixShader")
        mix.inputs["Fac"].default_value = alpha
        links.new(transp.outputs[0], mix.inputs[1])
        links.new(emit.outputs[0],   mix.inputs[2])
        links.new(mix.outputs[0],    out.inputs[0])
        mat.blend_method  = "BLEND"
        mat.shadow_method = "NONE"
    else:
        links.new(emit.outputs[0], out.inputs[0])
    return mat


joint_mat = make_emit_mat("BD_Joint", (1.0, 0.55, 0.1))       # orange
bone_mat  = make_emit_mat("BD_Bone",  (0.75, 0.75, 1.0))      # blue-white
mesh_mat  = make_emit_mat("BD_Mesh",  (0.2, 0.4, 0.75), alpha=max(0.02, min(1.0, OPACITY)))


# ── Character mesh: semi-transparent ─────────────────────────────────────────

for mobj in meshes:
    mobj.data.materials.clear()
    if OPACITY > 0.01:
        mobj.data.materials.append(mesh_mat)

# Hide the armature object from the final render — we draw our own bones
arm.hide_render = True


# ── Bone visualization geometry ───────────────────────────────────────────────

JOINT_R = 0.018   # sphere radius (suitable for ~1.8m tall character)
BONE_R  = JOINT_R * 0.35


def add_bone_geometry(bone):
    head_w = arm.matrix_world @ bone.head_local
    tail_w = arm.matrix_world @ bone.tail_local
    safe   = bone.name.replace(":", "_").replace("|", "_")[:40]

    # Joint sphere at head
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=JOINT_R, location=head_w, segments=10, ring_count=8
    )
    sph = bpy.context.active_object
    sph.name = f"jnt_{safe}"
    sph.data.materials.append(joint_mat)

    # Bone cylinder head→tail
    d = tail_w - head_w
    L = d.length
    if L < 0.005:
        return
    mid = (head_w + tail_w) / 2

    bpy.ops.mesh.primitive_cylinder_add(
        radius=BONE_R, depth=L, location=mid, vertices=8
    )
    cyl = bpy.context.active_object
    cyl.name = f"bne_{safe}"

    # Orient cylinder along d
    z   = Vector((0, 0, 1))
    dn  = d.normalized()
    ax  = z.cross(dn)
    if ax.length > 1e-4:
        ax.normalize()
        angle = math.acos(max(-1.0, min(1.0, z.dot(dn))))
        cyl.rotation_mode = "AXIS_ANGLE"
        cyl.rotation_axis_angle = (angle, *ax)
        bpy.context.view_layer.update()

    cyl.data.materials.append(bone_mat)


for bone in arm.data.bones:
    add_bone_geometry(bone)

bpy.context.view_layer.update()
log(f"[RigPreview] Created bone geometry for {len(arm.data.bones)} bones")


# ── Scene bounds ──────────────────────────────────────────────────────────────

def scene_bounds():
    coords = []
    for obj in bpy.data.objects:
        if obj.hide_render or obj.type not in ("MESH",):
            continue
        bpy.context.view_layer.update()
        bbox = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        coords.extend(bbox)
    if not coords:
        return Vector((0, 0, 0)), Vector((1, 1, 2))
    mn = Vector((min(c.x for c in coords), min(c.y for c in coords), min(c.z for c in coords)))
    mx = Vector((max(c.x for c in coords), max(c.y for c in coords), max(c.z for c in coords)))
    return mn, mx


mn, mx = scene_bounds()
center = (mn + mx) / 2
w = mx.x - mn.x
h = mx.z - mn.z   # character height
d = mx.y - mn.y
sz = max(w, h, d)

# Camera back-distance for a ~50mm lens framing
dist = sz * 1.5
log(f"[RigPreview] Bounds: min={mn}, max={mx}, center={center}, sz={sz:.3f}")


# ── World (dark background) ────────────────────────────────────────────────────

world = bpy.data.worlds.new("BD_World")
world.use_nodes = True
world.node_tree.nodes.clear()
bg   = world.node_tree.nodes.new("ShaderNodeBackground")
out  = world.node_tree.nodes.new("ShaderNodeOutputWorld")
bg.inputs["Color"].default_value    = (0.06, 0.06, 0.08, 1.0)
bg.inputs["Strength"].default_value = 0.3
world.node_tree.links.new(bg.outputs[0], out.inputs[0])
bpy.context.scene.world = world


# ── Render settings ───────────────────────────────────────────────────────────

scene = bpy.context.scene
scene.render.engine         = "BLENDER_EEVEE_NEXT"
scene.render.resolution_x   = RESOLUTION
scene.render.resolution_y   = RESOLUTION
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode  = "RGBA"
scene.eevee.use_shadows     = False
scene.eevee.use_bloom        = False


# ── Camera render helper ──────────────────────────────────────────────────────

cx, cy, cz = center

# Orthographic scale: tall enough to show character + a little padding
orth_scale = max(w, h) * 1.25


def render_view(cam_loc, look_at, out_path, ortho=False):
    cam_data = bpy.data.cameras.new("BD_Cam")
    cam_obj  = bpy.data.objects.new("BD_Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.location = Vector(cam_loc)

    direction = Vector(look_at) - Vector(cam_loc)
    cam_obj.rotation_mode       = "QUATERNION"
    cam_obj.rotation_quaternion = direction.to_track_quat("-Z", "Y")

    if ortho:
        cam_data.type        = "ORTHO"
        cam_data.ortho_scale = orth_scale
    else:
        cam_data.type  = "PERSP"
        cam_data.lens  = 50.0

    scene.camera = cam_obj
    bpy.context.view_layer.update()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)

    bpy.data.objects.remove(cam_obj, do_unlink=True)
    bpy.data.cameras.remove(cam_data)
    log(f"[RigPreview] → {out_path}")


# ── Render ────────────────────────────────────────────────────────────────────

stem = OUTPUT_BASE
if stem.endswith(".png"):
    stem = stem[:-4]
elif stem.endswith(".jpg"):
    stem = stem[:-4]

VIEWS = {
    "front":       (Vector((cx, cy - dist, cz)), center, True),
    "back":        (Vector((cx, cy + dist, cz)), center, True),
    "side":        (Vector((cx + dist, cy, cz)), center, True),
    "perspective": (Vector((cx + dist * 0.65, cy - dist * 0.65, cz + dist * 0.3)), center, False),
}

if VIEW_MODE == "all_views":
    for vname, (loc, tgt, ortho) in VIEWS.items():
        render_view(loc, tgt, f"{stem}_{vname}.png", ortho)
elif VIEW_MODE in VIEWS:
    loc, tgt, ortho = VIEWS[VIEW_MODE]
    render_view(loc, tgt, OUTPUT_BASE, ortho)
else:
    raise ValueError(f"Unknown VIEW_MODE: {VIEW_MODE!r}")

log("[RigPreview] DONE")
