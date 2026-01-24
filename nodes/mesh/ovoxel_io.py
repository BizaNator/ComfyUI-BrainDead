"""
BD_ExportOVoxel - Export VOXELGRID to .vxz format with mesh sidecar.
BD_LoadOVoxel - Load VOXELGRID from .vxz + sidecar files.

Uses o_voxel's native VXZ format (SVO + compression) for the voxelgrid
attributes, with a separate .mesh.npz sidecar for the original mesh data
(which VXZ cannot store).
"""

import os

import numpy as np
import torch

from comfy_api.latest import io

try:
    from o_voxel.io.vxz import write_vxz, read_vxz
    HAS_OVOXEL_IO = True
except ImportError:
    HAS_OVOXEL_IO = False


class BD_ExportOVoxel(io.ComfyNode):
    """
    Export VOXELGRID to o_voxel's native .vxz compressed format.

    Produces two files:
    - <name>.vxz: Compressed sparse voxel octree with PBR attributes
    - <name>.mesh.npz: Original high-poly mesh data (vertices, faces, metadata)

    Both files are needed to fully reconstruct the VOXELGRID for re-baking.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_ExportOVoxel",
            display_name="BD Export OVoxel",
            category="🧠BrainDead/Mesh",
            description="""Export VOXELGRID to o_voxel's native .vxz compressed format.

Saves the voxelgrid as a compressed sparse voxel octree (.vxz) alongside
a mesh sidecar (.mesh.npz) containing the original high-poly mesh data.

Both files are required to reload the full VOXELGRID with BD_LoadOVoxel.

Compression options:
- zstd: Fast compression/decompression, good ratio (recommended)
- lzma: Best compression ratio, slower
- deflate: Standard zlib, moderate speed/ratio""",
            inputs=[
                io.Custom("VOXELGRID").Input(
                    "voxelgrid",
                    tooltip="VOXELGRID from TRELLIS2 texture generation",
                ),
                io.String.Input(
                    "output_dir",
                    default="mesh_export",
                    tooltip="Output directory name (relative to ComfyUI output/)",
                ),
                io.String.Input(
                    "filename",
                    default="voxelgrid",
                    tooltip="Base filename (without extension)",
                ),
                io.Combo.Input(
                    "compression",
                    options=["zstd", "lzma", "deflate"],
                    default="zstd",
                    tooltip="Compression algorithm (zstd=fast, lzma=smallest, deflate=standard)",
                ),
            ],
            outputs=[
                io.String.Output(display_name="vxz_path"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        voxelgrid,
        output_dir: str = "mesh_export",
        filename: str = "voxelgrid",
        compression: str = "zstd",
    ) -> io.NodeOutput:
        if not HAS_OVOXEL_IO:
            return io.NodeOutput("", "ERROR: o_voxel.io not available")

        if voxelgrid is None:
            return io.NodeOutput("", "ERROR: No voxelgrid input")

        # Validate VOXELGRID dict
        required_keys = ['coords', 'attrs', 'voxel_size', 'layout']
        for key in required_keys:
            if key not in voxelgrid:
                return io.NodeOutput("", f"ERROR: VOXELGRID missing '{key}'")

        # Set up output directory
        import folder_paths
        output_base = folder_paths.get_output_directory()
        out_dir = os.path.join(output_base, output_dir)
        os.makedirs(out_dir, exist_ok=True)

        vxz_path = os.path.join(out_dir, f"{filename}.vxz")
        mesh_path = os.path.join(out_dir, f"{filename}.mesh.npz")

        try:
            coords_np = np.asarray(voxelgrid['coords'])  # (N, 3) float32
            attrs_np = np.asarray(voxelgrid['attrs'])     # (N, 6) float in [-1, 1]
            voxel_size = float(voxelgrid['voxel_size'])
            layout = voxelgrid['layout']

            n_voxels = len(coords_np)
            print(f"[BD ExportOVoxel] Exporting {n_voxels:,} voxels, compression={compression}")

            # --- VXZ: coords + attrs ---
            # Coords must be int tensor for VXZ (they're grid indices)
            coords_int = torch.from_numpy(coords_np.astype(np.int32)).int()

            # Split attrs by layout and quantize to uint8
            # attrs are in [-1, 1] → normalize to [0, 1] → scale to [0, 255]
            attrs_normalized = (attrs_np + 1.0) * 0.5  # [0, 1]
            attrs_uint8_np = (attrs_normalized * 255.0).clip(0, 255).astype(np.uint8)

            # Build named attr dict from layout
            attr_dict = {}
            layout_order = []
            for name, slc in layout.items():
                if isinstance(slc, slice):
                    channel_data = attrs_uint8_np[:, slc]
                    attr_dict[name] = torch.from_numpy(channel_data.copy())
                    layout_order.append((name, slc.start, slc.stop))

            print(f"[BD ExportOVoxel] Attrs: {', '.join(f'{k}({v.shape[1]}ch)' for k, v in attr_dict.items())}")

            # Write VXZ
            write_vxz(
                vxz_path,
                coords_int,
                attr_dict,
                chunk_size=256,
                compression=compression,
                attr_interleave='as_is',
            )

            vxz_size = os.path.getsize(vxz_path)
            raw_size = n_voxels * (3 * 4 + attrs_np.shape[1])  # approx raw bytes
            ratio = raw_size / vxz_size if vxz_size > 0 else 0
            print(f"[BD ExportOVoxel] VXZ: {vxz_size / (1024*1024):.1f} MB (ratio: {ratio:.1f}x)")

            # --- Sidecar: mesh + metadata ---
            sidecar_data = {
                'voxel_size': np.array([voxel_size], dtype=np.float32),
            }

            # Layout as serializable arrays (slice objects can't be saved to npz)
            layout_names = []
            layout_starts = []
            layout_stops = []
            for name, start, stop in layout_order:
                layout_names.append(name)
                layout_starts.append(start)
                layout_stops.append(stop)
            sidecar_data['layout_names'] = np.array(layout_names)
            sidecar_data['layout_starts'] = np.array(layout_starts, dtype=np.int32)
            sidecar_data['layout_stops'] = np.array(layout_stops, dtype=np.int32)

            # Original mesh (if present)
            if 'original_vertices' in voxelgrid and voxelgrid['original_vertices'] is not None:
                orig_verts = voxelgrid['original_vertices']
                if isinstance(orig_verts, torch.Tensor):
                    orig_verts = orig_verts.cpu().numpy()
                sidecar_data['original_vertices'] = orig_verts.astype(np.float32)

            if 'original_faces' in voxelgrid and voxelgrid['original_faces'] is not None:
                orig_faces = voxelgrid['original_faces']
                if isinstance(orig_faces, torch.Tensor):
                    orig_faces = orig_faces.cpu().numpy()
                sidecar_data['original_faces'] = orig_faces.astype(np.int32)

            np.savez_compressed(mesh_path, **sidecar_data)
            mesh_size = os.path.getsize(mesh_path)
            print(f"[BD ExportOVoxel] Mesh sidecar: {mesh_size / (1024*1024):.1f} MB")

            # Status
            total_mb = (vxz_size + mesh_size) / (1024 * 1024)
            n_orig_verts = len(sidecar_data.get('original_vertices', []))
            status = f"{filename}.vxz: {n_voxels:,} voxels, {total_mb:.1f} MB total ({compression})"
            if n_orig_verts > 0:
                status += f" | mesh: {n_orig_verts:,} verts"

            print(f"[BD ExportOVoxel] Done: {status}")
            return io.NodeOutput(vxz_path, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return io.NodeOutput("", f"ERROR: {e}")


class BD_LoadOVoxel(io.ComfyNode):
    """
    Load a VOXELGRID from exported .vxz + .mesh.npz files.

    Reconstructs the full VOXELGRID dict from:
    - <name>.vxz: Compressed voxel attributes
    - <name>.mesh.npz: Original mesh data + metadata

    The loaded VOXELGRID can be used with BD_OVoxelBake, BD_OVoxelTextureBake,
    or BD_SampleVoxelgridColors for re-baking without regenerating.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_LoadOVoxel",
            display_name="BD Load OVoxel",
            category="🧠BrainDead/Mesh",
            description="""Load a VOXELGRID from exported .vxz + .mesh.npz files.

Reconstructs the full VOXELGRID dict for use with baking nodes.
Expects both <name>.vxz and <name>.mesh.npz in the same directory.

Use with BD_ExportOVoxel to cache expensive TRELLIS2 texture generations
and re-bake with different decimation/UV settings without regenerating.""",
            inputs=[
                io.String.Input(
                    "vxz_path",
                    default="",
                    tooltip="Full path to the .vxz file (or relative to ComfyUI output/)",
                ),
            ],
            outputs=[
                io.Custom("VOXELGRID").Output(display_name="voxelgrid"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, vxz_path: str = "") -> io.NodeOutput:
        if not HAS_OVOXEL_IO:
            return io.NodeOutput(None, "ERROR: o_voxel.io not available")

        if not vxz_path:
            return io.NodeOutput(None, "ERROR: No vxz_path provided")

        # Resolve path (support relative to output dir)
        if not os.path.isabs(vxz_path):
            import folder_paths
            output_base = folder_paths.get_output_directory()
            vxz_path = os.path.join(output_base, vxz_path)

        if not os.path.exists(vxz_path):
            return io.NodeOutput(None, f"ERROR: File not found: {vxz_path}")

        # Derive sidecar path
        base_path = vxz_path
        if base_path.endswith('.vxz'):
            base_path = base_path[:-4]
        mesh_path = base_path + '.mesh.npz'

        if not os.path.exists(mesh_path):
            return io.NodeOutput(None, f"ERROR: Mesh sidecar not found: {mesh_path}")

        try:
            print(f"[BD LoadOVoxel] Loading: {vxz_path}")

            # --- Read VXZ ---
            coords_int, attr_dict = read_vxz(vxz_path)
            # coords_int: torch.Tensor int (N, 3)
            # attr_dict: dict of torch.Tensor uint8

            n_voxels = len(coords_int)
            print(f"[BD LoadOVoxel] VXZ: {n_voxels:,} voxels, attrs: {list(attr_dict.keys())}")

            # --- Read sidecar ---
            sidecar = np.load(mesh_path, allow_pickle=False)
            voxel_size = float(sidecar['voxel_size'].flat[0])

            # Reconstruct layout from saved arrays
            layout_names = [str(n) for n in sidecar['layout_names']]
            layout_starts = sidecar['layout_starts'].astype(int)
            layout_stops = sidecar['layout_stops'].astype(int)
            layout = {}
            for name, start, stop in zip(layout_names, layout_starts, layout_stops):
                layout[name] = slice(int(start), int(stop))

            # Reconstruct attrs tensor from named dict
            # Total channels from layout
            total_channels = max(s.stop for s in layout.values())
            attrs_uint8 = torch.zeros(n_voxels, total_channels, dtype=torch.uint8)
            for name, slc in layout.items():
                if name in attr_dict:
                    attrs_uint8[:, slc] = attr_dict[name]
                else:
                    print(f"[BD LoadOVoxel] WARNING: attr '{name}' in layout but not in VXZ file")

            # Convert uint8 [0, 255] → float [-1, 1]
            attrs_float = (attrs_uint8.float() / 255.0) * 2.0 - 1.0
            attrs_np = attrs_float.numpy()

            # Coords: int tensor → float32 numpy
            coords_np = coords_int.numpy().astype(np.float32)

            # Original mesh
            original_vertices = None
            original_faces = None
            if 'original_vertices' in sidecar:
                original_vertices = torch.from_numpy(
                    sidecar['original_vertices'].astype(np.float32)
                )
            if 'original_faces' in sidecar:
                original_faces = torch.from_numpy(
                    sidecar['original_faces'].astype(np.int32)
                ).long()

            # Build VOXELGRID dict
            voxelgrid = {
                'coords': coords_np,
                'attrs': attrs_np,
                'voxel_size': voxel_size,
                'layout': layout,
                'original_vertices': original_vertices,
                'original_faces': original_faces,
            }

            # Status
            n_verts = len(original_vertices) if original_vertices is not None else 0
            status = f"Loaded {n_voxels:,} voxels, voxel_size={voxel_size:.6f}"
            if n_verts > 0:
                status += f" | mesh: {n_verts:,} verts"
            status += f" | channels: {list(layout.keys())}"

            print(f"[BD LoadOVoxel] {status}")
            return io.NodeOutput(voxelgrid, status)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return io.NodeOutput(None, f"ERROR: {e}")


# V3 node list
OVOXEL_IO_V3_NODES = [BD_ExportOVoxel, BD_LoadOVoxel]

# V1 compatibility
OVOXEL_IO_NODES = {
    "BD_ExportOVoxel": BD_ExportOVoxel,
    "BD_LoadOVoxel": BD_LoadOVoxel,
}

OVOXEL_IO_DISPLAY_NAMES = {
    "BD_ExportOVoxel": "BD Export OVoxel",
    "BD_LoadOVoxel": "BD Load OVoxel",
}
