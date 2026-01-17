"""
Base class for Blender-based ComfyUI nodes.

Provides utilities for running Blender headlessly via subprocess,
passing mesh data through temporary files.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

# Find Blender executable - check common locations
# Prefer newer Blender versions first
BLENDER_PATHS = [
    # BrainDead bundled Blender 5.0.1 (preferred)
    "/opt/comfyui/dev/custom_nodes/ComfyUI-BrainDead/lib/blender/blender-5.0.1-linux-x64/blender",
    "/opt/comfyui/stable/custom_nodes/ComfyUI-BrainDead/lib/blender/blender-5.0.1-linux-x64/blender",
    # GeometryPack bundled Blender 4.2.3
    "/opt/comfyui/dev/custom_nodes/ComfyUI-GeometryPack/_blender/blender-4.2.3-linux-x64/blender",
    "/opt/comfyui/stable/custom_nodes/ComfyUI-GeometryPack/_blender/blender-4.2.3-linux-x64/blender",
    # UniRig bundled Blender 4.2.3
    "/opt/comfyui/dev/custom_nodes/ComfyUI-UniRig/lib/blender/blender-4.2.3-linux-x64/blender",
    "/opt/comfyui/stable/custom_nodes/ComfyUI-UniRig/lib/blender/blender-4.2.3-linux-x64/blender",
    # System Blender
    "/usr/bin/blender",
    "/usr/local/bin/blender",
]

def find_blender() -> Optional[str]:
    """Find a working Blender executable."""
    for path in BLENDER_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None

BLENDER_PATH = find_blender()

# Check for trimesh
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


class BlenderNodeMixin:
    """
    Mixin class providing Blender subprocess utilities.

    Subclasses should:
    1. Set BLENDER_SCRIPT class attribute with the Python script to run
    2. Call _run_blender_script() with input/output paths
    """

    BLENDER_SCRIPT: str = ""  # Override in subclass

    @classmethod
    def _check_blender(cls) -> tuple[bool, str]:
        """Check if Blender is available."""
        if not BLENDER_PATH:
            return False, "Blender not found. Install GeometryPack or system Blender."
        return True, BLENDER_PATH

    @classmethod
    def _run_blender_script(
        cls,
        script: str,
        input_path: str,
        output_path: str,
        extra_args: Optional[dict] = None,
        timeout: int = 300,
    ) -> tuple[bool, str, list[str]]:
        """
        Run a Blender Python script headlessly with real-time output.

        Args:
            script: Python script content to execute in Blender
            input_path: Path to input mesh file
            output_path: Path to write output mesh file
            extra_args: Optional dict of extra arguments to pass via environment
            timeout: Maximum execution time in seconds

        Returns:
            Tuple of (success, message, log_lines)
        """
        import time
        import select

        available, blender_path = cls._check_blender()
        if not available:
            return False, blender_path, []

        # Create temporary script file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name

        log_lines = []
        process = None

        try:
            # Build environment with extra args
            env = os.environ.copy()
            env['BLENDER_INPUT_PATH'] = input_path
            env['BLENDER_OUTPUT_PATH'] = output_path
            if extra_args:
                for key, value in extra_args.items():
                    env[f'BLENDER_ARG_{key.upper()}'] = str(value)

            print(f"[BD Blender] Starting Blender: {blender_path}")
            print(f"[BD Blender] Input: {input_path}")
            print(f"[BD Blender] Output: {output_path}")
            print(f"[BD Blender] Timeout: {timeout}s")

            # Run Blender with Popen for real-time output
            process = subprocess.Popen(
                [
                    blender_path,
                    '--background',
                    '--python', script_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                env=env,
            )

            start_time = time.time()

            # Read output in real-time with timeout checking
            while True:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    process.kill()
                    process.wait()
                    return False, f"Blender timed out after {timeout}s", log_lines

                # Check if process finished
                retcode = process.poll()

                # Read available output (non-blocking)
                try:
                    line = process.stdout.readline()
                    if line:
                        line = line.rstrip()
                        log_lines.append(line)
                        # Print lines that look like progress/status
                        if line.startswith('[BD') or 'Error' in line or 'faces' in line.lower():
                            print(line)
                    elif retcode is not None:
                        # Process finished and no more output
                        break
                    else:
                        # No output but still running, brief sleep
                        time.sleep(0.1)
                except:
                    if retcode is not None:
                        break
                    time.sleep(0.1)

            # Get remaining output (don't re-print - already printed in real-time)
            remaining = process.stdout.read()
            if remaining:
                for line in remaining.strip().split('\n'):
                    if line:
                        log_lines.append(line)

            if retcode != 0:
                # Find error in log
                error_lines = [l for l in log_lines if 'error' in l.lower() or 'Error' in l]
                error_msg = error_lines[-1] if error_lines else f"Exit code {retcode}"
                return False, f"Blender failed: {error_msg}", log_lines

            if not os.path.exists(output_path):
                return False, "Blender did not produce output file", log_lines

            # Find final status in log
            status_lines = [l for l in log_lines if l.startswith('[BD')]
            final_status = status_lines[-1] if status_lines else "Success"

            return True, final_status, log_lines

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Blender error: {e}", log_lines
        finally:
            # Kill process if still running
            if process and process.poll() is None:
                process.kill()
                process.wait()
            # Clean up script file
            if os.path.exists(script_path):
                os.remove(script_path)

    @classmethod
    def _mesh_to_temp_file(cls, mesh, suffix: str = '.ply') -> str:
        """
        Save a trimesh mesh to a temporary file.

        Args:
            mesh: trimesh.Trimesh object
            suffix: File extension (.ply, .obj, .glb)

        Returns:
            Path to temporary file
        """
        if not HAS_TRIMESH:
            raise RuntimeError("trimesh not installed")

        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

        # Export with colors if available
        trimesh.exchange.export.export_mesh(mesh, path, file_type=suffix[1:])
        return path

    @classmethod
    def _load_mesh_from_file(cls, path: str):
        """
        Load a mesh from a file path.

        Args:
            path: Path to mesh file

        Returns:
            trimesh.Trimesh object
        """
        import os

        if not HAS_TRIMESH:
            raise RuntimeError("trimesh not installed")

        file_size = os.path.getsize(path) / (1024 * 1024)  # MB
        print(f"[BD Blender] Loading result mesh ({file_size:.1f}MB)...")

        mesh = trimesh.load(path, force='mesh')

        # Handle scene vs single mesh
        if isinstance(mesh, trimesh.Scene):
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if meshes:
                mesh = trimesh.util.concatenate(meshes)
            else:
                raise RuntimeError("No meshes found in loaded file")

        print(f"[BD Blender] Loaded: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")
        return mesh


# Blender script templates
DECIMATE_SCRIPT = '''
import bpy
import os
import sys

def log(msg):
    """Print with immediate flush for real-time output."""
    print(msg)
    sys.stdout.flush()

# Get paths from environment
input_path = os.environ['BLENDER_INPUT_PATH']
output_path = os.environ['BLENDER_OUTPUT_PATH']
ratio = float(os.environ.get('BLENDER_ARG_RATIO', '0.5'))
use_collapse = os.environ.get('BLENDER_ARG_USE_COLLAPSE', 'True') == 'True'
use_symmetry = os.environ.get('BLENDER_ARG_USE_SYMMETRY', 'False') == 'True'

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh
ext = os.path.splitext(input_path)[1].lower()
if ext == '.ply':
    bpy.ops.wm.ply_import(filepath=input_path)
elif ext == '.obj':
    bpy.ops.wm.obj_import(filepath=input_path)
elif ext in ['.glb', '.gltf']:
    bpy.ops.import_scene.gltf(filepath=input_path)
elif ext == '.stl':
    bpy.ops.wm.stl_import(filepath=input_path)
else:
    raise ValueError(f"Unsupported format: {ext}")

# Get imported object
obj = bpy.context.active_object
if obj is None:
    obj = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
bpy.context.view_layer.objects.active = obj

# Add decimate modifier
modifier = obj.modifiers.new(name='Decimate', type='DECIMATE')
if use_collapse:
    modifier.decimate_type = 'COLLAPSE'
    modifier.ratio = ratio
    modifier.use_symmetry = use_symmetry
else:
    modifier.decimate_type = 'UNSUBDIV'
    modifier.iterations = max(1, int((1 - ratio) * 10))

# Apply modifier
bpy.ops.object.modifier_apply(modifier=modifier.name)

# Export result
ext_out = os.path.splitext(output_path)[1].lower()
if ext_out == '.ply':
    bpy.ops.wm.ply_export(filepath=output_path, export_colors='SRGB')
elif ext_out == '.obj':
    bpy.ops.wm.obj_export(filepath=output_path)
elif ext_out in ['.glb', '.gltf']:
    bpy.ops.export_scene.gltf(filepath=output_path)
elif ext_out == '.stl':
    bpy.ops.wm.stl_export(filepath=output_path)

log(f"[BD Decimate] Saved to {output_path}")
'''

REMESH_SCRIPT = '''
import bpy
import os
import sys

def log(msg):
    """Print with immediate flush for real-time output."""
    print(msg)
    sys.stdout.flush()

# Get paths from environment
input_path = os.environ['BLENDER_INPUT_PATH']
output_path = os.environ['BLENDER_OUTPUT_PATH']
mode = os.environ.get('BLENDER_ARG_MODE', 'VOXEL')
voxel_size = float(os.environ.get('BLENDER_ARG_VOXEL_SIZE', '0.01'))
octree_depth = int(os.environ.get('BLENDER_ARG_OCTREE_DEPTH', '6'))
smooth_iterations = int(os.environ.get('BLENDER_ARG_SMOOTH_ITERATIONS', '0'))

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh
ext = os.path.splitext(input_path)[1].lower()
if ext == '.ply':
    bpy.ops.wm.ply_import(filepath=input_path)
elif ext == '.obj':
    bpy.ops.wm.obj_import(filepath=input_path)
elif ext in ['.glb', '.gltf']:
    bpy.ops.import_scene.gltf(filepath=input_path)
elif ext == '.stl':
    bpy.ops.wm.stl_import(filepath=input_path)
else:
    raise ValueError(f"Unsupported format: {ext}")

# Get imported object
obj = bpy.context.active_object
if obj is None:
    obj = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
bpy.context.view_layer.objects.active = obj

# Add remesh modifier
modifier = obj.modifiers.new(name='Remesh', type='REMESH')
modifier.mode = mode
if mode == 'VOXEL':
    modifier.voxel_size = voxel_size
else:
    modifier.octree_depth = octree_depth
modifier.use_smooth_shade = smooth_iterations > 0

# Apply modifier
bpy.ops.object.modifier_apply(modifier=modifier.name)

# Optional smoothing
if smooth_iterations > 0:
    smooth_mod = obj.modifiers.new(name='Smooth', type='SMOOTH')
    smooth_mod.iterations = smooth_iterations
    bpy.ops.object.modifier_apply(modifier=smooth_mod.name)

# Export result
ext_out = os.path.splitext(output_path)[1].lower()
if ext_out == '.ply':
    bpy.ops.wm.ply_export(filepath=output_path, export_colors='SRGB')
elif ext_out == '.obj':
    bpy.ops.wm.obj_export(filepath=output_path)
elif ext_out in ['.glb', '.gltf']:
    bpy.ops.export_scene.gltf(filepath=output_path)
elif ext_out == '.stl':
    bpy.ops.wm.stl_export(filepath=output_path)

log(f"[BD Remesh] Saved to {output_path}")
'''

REPAIR_SCRIPT = '''
import bpy
import bmesh
import os
import sys

def log(msg):
    """Print with immediate flush for real-time output."""
    print(msg)
    sys.stdout.flush()

# Get paths from environment
input_path = os.environ['BLENDER_INPUT_PATH']
output_path = os.environ['BLENDER_OUTPUT_PATH']
fill_holes = os.environ.get('BLENDER_ARG_FILL_HOLES', 'True') == 'True'
remove_doubles = os.environ.get('BLENDER_ARG_REMOVE_DOUBLES', 'True') == 'True'
merge_distance = float(os.environ.get('BLENDER_ARG_MERGE_DISTANCE', '0.0001'))
recalc_normals = os.environ.get('BLENDER_ARG_RECALC_NORMALS', 'True') == 'True'
make_manifold = os.environ.get('BLENDER_ARG_MAKE_MANIFOLD', 'False') == 'True'

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh
ext = os.path.splitext(input_path)[1].lower()
if ext == '.ply':
    bpy.ops.wm.ply_import(filepath=input_path)
elif ext == '.obj':
    bpy.ops.wm.obj_import(filepath=input_path)
elif ext in ['.glb', '.gltf']:
    bpy.ops.import_scene.gltf(filepath=input_path)
elif ext == '.stl':
    bpy.ops.wm.stl_import(filepath=input_path)
else:
    raise ValueError(f"Unsupported format: {ext}")

# Get imported object
obj = bpy.context.active_object
if obj is None:
    obj = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

# Enter edit mode
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')

# Remove doubles
if remove_doubles:
    bpy.ops.mesh.remove_doubles(threshold=merge_distance)

# Fill holes
if fill_holes:
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold(extend=False)
    bpy.ops.mesh.fill_holes(sides=100)

# Recalculate normals
if recalc_normals:
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)

# Make manifold (3D print toolbox approach)
if make_manifold:
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.fill_holes(sides=100)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=merge_distance)

# Return to object mode
bpy.ops.object.mode_set(mode='OBJECT')

# Export result
ext_out = os.path.splitext(output_path)[1].lower()
if ext_out == '.ply':
    bpy.ops.wm.ply_export(filepath=output_path, export_colors='SRGB')
elif ext_out == '.obj':
    bpy.ops.wm.obj_export(filepath=output_path)
elif ext_out in ['.glb', '.gltf']:
    bpy.ops.export_scene.gltf(filepath=output_path)
elif ext_out == '.stl':
    bpy.ops.wm.stl_export(filepath=output_path)

log(f"[BD Repair] Saved to {output_path}")
'''

TRANSFER_COLORS_SCRIPT = '''
import bpy
import os
import sys
import numpy as np

def log(msg):
    """Print with immediate flush for real-time output."""
    print(msg)
    sys.stdout.flush()

# Get paths from environment
source_path = os.environ['BLENDER_INPUT_PATH']  # Source mesh with colors
target_path = os.environ.get('BLENDER_ARG_TARGET_PATH', '')  # Target mesh
output_path = os.environ['BLENDER_OUTPUT_PATH']
max_distance = float(os.environ.get('BLENDER_ARG_MAX_DISTANCE', '0.1'))

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import source mesh
ext = os.path.splitext(source_path)[1].lower()
if ext == '.ply':
    bpy.ops.wm.ply_import(filepath=source_path)
elif ext == '.obj':
    bpy.ops.wm.obj_import(filepath=source_path)
elif ext in ['.glb', '.gltf']:
    bpy.ops.import_scene.gltf(filepath=source_path)
else:
    raise ValueError(f"Unsupported format: {ext}")

source_obj = bpy.context.active_object
if source_obj is None:
    source_obj = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
source_obj.name = 'Source'

# Import target mesh
ext = os.path.splitext(target_path)[1].lower()
if ext == '.ply':
    bpy.ops.wm.ply_import(filepath=target_path)
elif ext == '.obj':
    bpy.ops.wm.obj_import(filepath=target_path)
elif ext in ['.glb', '.gltf']:
    bpy.ops.import_scene.gltf(filepath=target_path)
else:
    raise ValueError(f"Unsupported format: {ext}")

target_obj = bpy.context.active_object
if target_obj is None:
    target_obj = [o for o in bpy.context.scene.objects if o.type == 'MESH' and o.name != 'Source'][0]
target_obj.name = 'Target'

# Set target as active
bpy.context.view_layer.objects.active = target_obj
target_obj.select_set(True)
source_obj.select_set(True)

# Ensure source has vertex colors
if not source_obj.data.vertex_colors:
    # Try to create from material colors or use white
    source_obj.data.vertex_colors.new(name='Col')

# Ensure target has vertex colors layer
if not target_obj.data.vertex_colors:
    target_obj.data.vertex_colors.new(name='Col')

# Use data transfer modifier
modifier = target_obj.modifiers.new(name='DataTransfer', type='DATA_TRANSFER')
modifier.object = source_obj
modifier.use_vert_data = True
modifier.data_types_verts = {'VGROUP_WEIGHTS'}  # Will also transfer colors
modifier.use_loop_data = True
modifier.data_types_loops = {'VCOL'}
modifier.loop_mapping = 'NEAREST_POLYNOR'  # Best for similar topology
modifier.max_distance = max_distance
modifier.use_max_distance = True

# Apply modifier
bpy.ops.object.modifier_apply(modifier=modifier.name)

# Make target the only selected object
source_obj.select_set(False)
bpy.context.view_layer.objects.active = target_obj

# Export result
ext_out = os.path.splitext(output_path)[1].lower()
if ext_out == '.ply':
    bpy.ops.wm.ply_export(filepath=output_path, export_colors='SRGB')
elif ext_out == '.obj':
    bpy.ops.wm.obj_export(filepath=output_path)
elif ext_out in ['.glb', '.gltf']:
    bpy.ops.export_scene.gltf(filepath=output_path)
else:
    bpy.ops.wm.ply_export(filepath=output_path, export_colors='SRGB')

log(f"[BD Transfer] Saved to {output_path}")
'''
