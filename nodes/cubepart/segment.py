"""
BD_CubePartSegment - Roblox CubePart open-vocabulary part decomposition.

Takes one mesh + up to 8 free-text part names and generates one clean mesh per
part (canonically aligned), suitable for rigging / game engines.
"""
import colorsys
import gc

import numpy as np
import torch
import trimesh

from comfy_api.latest import io

from .utils import (
    DEFAULT_MODEL_DIR,
    DEFAULT_TEXT_ENCODER,
    HAS_CUBEPART,
    IMPORT_ERROR,
    MAX_PARTS,
    get_pipeline,
    parse_parts,
    prepare_surface,
)


def _palette(n: int):
    """`n` visually-distinct RGBA uint8 colors (matches run_inference.py)."""
    cols = []
    for i in range(max(n, 1)):
        h = (i / max(n, 1)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.95)
        cols.append([int(r * 255), int(g * 255), int(b * 255), 255])
    return cols


class BD_CubePartSegment(io.ComfyNode):
    """
    Decompose a mesh into semantic parts with Roblox CubePart.

    Wire a TRIMESH from any BD mesh source (Pixal3D / Trellis2 / mesh cache), or
    set `mesh_path` to load a .glb from disk. Provide up to 8 comma- or
    newline-separated part names (open vocabulary, e.g. "body, left wheel, gun").

    Outputs (all derived from the same per-part result, so they always agree):
    - parts (TRIMESH_LIST): list of per-part trimesh.Trimesh, in input-name order.
      Feed BD_CubePartGetPart to pull one part into the rest of the mesh pipeline.
    - combined (TRIMESH): all parts merged into one mesh, each face-colored by part
      for visualization. A deterministic concatenation of `parts`, nothing more.
    - part_names (STRING): newline-joined names of the parts actually produced.

    Note: parts are returned in CubePart's normalized frame (~unit cube), the same
    space the input mesh is rescaled into; orientation matches the input mesh.
    Requires Roblox/cubepart + Qwen/Qwen3-VL-4B-Instruct weights (pre-downloaded).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_CubePartSegment",
            display_name="BD CubePart Segment",
            category="🧠BrainDead/CubePart",
            description="Open-vocabulary part decomposition (Roblox CubePart): "
                        "mesh + up to 8 part names -> one mesh per part.",
            # Required inputs are listed first, then optional — so the frontend's
            # widget order (definition order) matches required-then-optional and
            # serialized widgets_values line up in both the UI and headless runs.
            inputs=[
                io.String.Input(
                    "parts", multiline=True,
                    default="body, left wheel, right wheel",
                    tooltip="Up to 8 part names, comma- or newline-separated. "
                            "Open vocabulary. Extra names past 8 are dropped (logged).",
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=2**31 - 1,
                    tooltip="Random seed for reproducibility.",
                ),
                io.Float.Input(
                    "guidance_scale", default=7.5, min=0.0, max=20.0, step=0.1,
                    tooltip="Classifier-free guidance strength.",
                ),
                io.Int.Input(
                    "num_inference_steps", default=50, min=1, max=200,
                    tooltip="Diffusion denoising steps.",
                ),
                io.Float.Input(
                    "resolution_base", default=8.5, min=6.0, max=10.0, step=0.5,
                    tooltip="Marching-cubes grid resolution base (higher = finer, slower).",
                ),
                io.Combo.Input(
                    "scheduler", options=["dpm_solver", "euler", "heun"],
                    default="dpm_solver",
                    tooltip="Sampling scheduler.",
                ),
                io.Float.Input(
                    "timeshift", default=4.0, min=1.0, max=10.0, step=0.5,
                    tooltip="Flow-matching timestep shift.",
                ),
                io.Int.Input(
                    "num_samples", default=128_000, min=16_000, max=256_000, step=8_000,
                    tooltip="Surface points sampled from the input mesh for encoding.",
                ),
                io.Custom("TRIMESH").Input(
                    "mesh", optional=True,
                    tooltip="Input mesh from a BD mesh source. If unset, mesh_path is used.",
                ),
                io.String.Input(
                    "mesh_path", default="", optional=True,
                    tooltip="Path to a .glb on disk. Used only when no `mesh` is wired.",
                ),
                io.Boolean.Input(
                    "auto_download", default=True, optional=True,
                    tooltip="Download missing weights from HF on first run (Roblox/cubepart, "
                            "Qwen/Qwen3-VL-4B-Instruct). Off = error if not pre-downloaded.",
                ),
                io.String.Input(
                    "model_dir", default="", optional=True,
                    tooltip="Override the cubepart weights dir. Empty = auto-resolve via "
                            "ComfyUI folder_paths / extra_model_paths.yaml ('cubepart' key, "
                            "else models/cubepart).",
                ),
                io.String.Input(
                    "text_encoder_path", default="", optional=True,
                    tooltip="Override the Qwen3-VL-4B-Instruct dir (loaded offline). Empty = "
                            "auto-resolve under the LLM models folder.",
                ),
            ],
            outputs=[
                io.Custom("TRIMESH_LIST").Output(display_name="parts"),
                io.Custom("TRIMESH").Output(display_name="combined"),
                io.String.Output(display_name="part_names"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mesh=None,
        parts: str = "",
        mesh_path: str = "",
        seed: int = 0,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        resolution_base: float = 8.5,
        scheduler: str = "dpm_solver",
        timeshift: float = 4.0,
        num_samples: int = 128_000,
        auto_download: bool = True,
        model_dir: str = "",
        text_encoder_path: str = "",
    ) -> io.NodeOutput:
        if not HAS_CUBEPART:
            raise ImportError(
                f"cube_part failed to import: {IMPORT_ERROR}\n"
                "Ensure warp-lang + jaxtyping are installed in this venv and the "
                "vendored source exists under nodes/cubepart/vendor/."
            )

        # Empty = auto-resolve via folder_paths (extra_model_paths.yaml); str() guards
        # against a widget int leaking into a path (ComfyUI quirk; see Pixal3D).
        model_dir = str(model_dir).strip()
        text_encoder_path = str(text_encoder_path).strip()

        part_names, dropped = parse_parts(parts)
        if not part_names:
            raise ValueError("CubePart needs at least one part name in `parts`.")
        if dropped:
            print(f"[BD CubePart] WARNING: {dropped} part name(s) past the "
                  f"{MAX_PARTS}-part limit were dropped: kept {part_names}")

        surface = prepare_surface(
            tri_mesh=mesh, mesh_path=str(mesh_path).strip(),
            num_samples=num_samples, device="cuda",
        )

        pipe = get_pipeline(model_dir, text_encoder_path, auto_download=auto_download)
        from .utils import ShapeInput  # re-exported from cube_part.pipelines

        print(f"[BD CubePart] Segmenting into {len(part_names)} parts "
              f"(steps={num_inference_steps}, guidance={guidance_scale}, seed={seed})...")
        torch.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats()

        latents, _ = pipe.encode_shape(surface)
        part_meshes = pipe.input_to_part_shape(
            ShapeInput(prompt=[part_names], latents=latents),
            guidance_scale=guidance_scale,
            resolution_base=resolution_base,
            scheduler_type=scheduler,
            timeshift=timeshift,
            num_inference_steps=num_inference_steps,
            seed=seed,
            output_mesh=True,
        )

        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"[BD CubePart] Generation peak VRAM: {peak:.0f} MB")

        # Build the canonical per-part list (aligned with part_names; drop empties).
        out_meshes = []
        out_names = []
        for i, vf in enumerate(part_meshes):
            verts, faces = vf if vf is not None else (None, None)
            name = part_names[i] if i < len(part_names) else f"part_{i:02d}"
            if verts is None or len(verts) == 0 or faces is None or len(faces) == 0:
                print(f"[BD CubePart]   part {i} '{name}': empty, skipped")
                continue
            tm = trimesh.Trimesh(
                vertices=np.asarray(verts, dtype=np.float32),
                faces=np.asarray(faces),
                process=False,
            )
            out_meshes.append(tm)
            out_names.append(name)
            print(f"[BD CubePart]   part {i} '{name}': "
                  f"{len(tm.vertices):,} verts, {len(tm.faces):,} faces")

        # combined = deterministic colored concatenation of the canonical list.
        # Use VERTEX colors (not face) so they export as glTF COLOR_0 and render in
        # the three.js viewer / any glb consumer.
        palette = _palette(len(out_meshes))
        colored = []
        for j, tm in enumerate(out_meshes):
            c = tm.copy()
            c.visual.vertex_colors = np.tile(palette[j % len(palette)], (len(c.vertices), 1))
            colored.append(c)
        if colored:
            combined = trimesh.util.concatenate(colored)
        else:
            combined = trimesh.Trimesh()  # empty fallback; never None

        gc.collect()
        torch.cuda.empty_cache()

        return io.NodeOutput(out_meshes, combined, "\n".join(out_names))


CUBEPART_SEGMENT_V3_NODES = [BD_CubePartSegment]

CUBEPART_SEGMENT_NODES = {
    "BD_CubePartSegment": BD_CubePartSegment,
}

CUBEPART_SEGMENT_DISPLAY_NAMES = {
    "BD_CubePartSegment": "BD CubePart Segment",
}
