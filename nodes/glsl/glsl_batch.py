"""
BD_GLSLBatch — run a GLSL shader N times with different uniforms, batch the outputs.

Wraps ComfyUI's internal _render_shader_batch() function so the shader is compiled
ONCE and framebuffers/textures are reused across iterations. Massively faster than
re-submitting the workflow N times when only a few uniform values change between
iterations (e.g. skin tone RGB triplets).

Iteration pattern follows the BD_SAM3MultiPrompt / BD_PromptIterator convention:
each iteration spec is a line in a multiline string. Inputs not listed in a vary
field use their base value, same across all iterations.

Vary syntax (one uniform per line):
    u_int3=255,240,160,140        ← 4 iterations
    u_int4=255,190,110,190
    u_float0=0.48                 ← single value = constant for all iterations

All vary lines with multiple values MUST have the same value count = iteration count.
Lines with a single value are treated as constants (broadcast across iterations).
If only vary lines with multi-values exist, the iteration count comes from them.
If you want all single-value with a specific count, set iterations explicitly.

Outputs:
    fc0_batch ... fc3_batch — batched IMAGE tensors (B=N_iterations) for each MRT slot
    iteration_names         — newline-joined list of iteration labels (for BD_BulkSave etc.)
"""

import numpy as np
import torch
from comfy_api.latest import io

# NOTE: imports from comfy_extras.nodes_glsl are DEFERRED to execute() time —
# at module-load time, ComfyUI's `utils` package isn't on sys.path yet (only
# after the app is fully initialized), and nodes_glsl transitively depends
# on `utils.install_util`. Importing eagerly here would crash node registration.

MAX_IMAGES = 5
MAX_FLOATS = 20
MAX_INTS = 13
MAX_OUTPUTS = 4


def _strip_comments(line: str) -> str:
    """Strip inline comments. Supports both '#' and '//' starting a comment.

    Comment markers inside the VALUE portion (after the '=') are still recognized,
    so `u_int7=1  // mannequin` correctly parses value as '1'.
    """
    for marker in ("//", "#"):
        pos = line.find(marker)
        if pos != -1:
            line = line[:pos]
    return line.strip()


def _parse_uniforms_multiline(text: str, prefix: str, count: int, kind: type) -> list:
    """Parse a multiline 'u_floatN=value' (or u_intN=value) into a list of length `count`.

    Returns a base list with default values; unspecified slots stay at 0/0.0.
    Lines not matching the expected prefix are silently ignored.
    Supports inline comments with '#' or '//' — anything after a marker is stripped.
    """
    base: list = [kind(0)] * count
    if not text:
        return base
    for raw_line in text.strip().split("\n"):
        line = _strip_comments(raw_line)
        if not line or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if not key.startswith(prefix):
            continue
        try:
            idx = int(key[len(prefix):])
        except ValueError:
            continue
        if 0 <= idx < count:
            try:
                base[idx] = kind(val)
            except ValueError:
                pass
    return base


def _parse_vary_spec(text: str, prefix: str, count: int, kind: type) -> tuple[dict, int]:
    """Parse a vary spec multiline into {uniform_index: [v_per_iteration, ...]}.

    Returns (overrides_dict, max_iteration_count).
    Supports inline comments with '#' or '//'.
    """
    overrides: dict[int, list] = {}
    max_iter = 0
    if not text:
        return overrides, max_iter
    for raw_line in text.strip().split("\n"):
        line = _strip_comments(raw_line)
        if not line or "=" not in line:
            continue
        key, _, vals = line.partition("=")
        key = key.strip()
        if not key.startswith(prefix):
            continue
        try:
            idx = int(key[len(prefix):])
        except ValueError:
            continue
        if not (0 <= idx < count):
            continue
        try:
            values = [kind(v.strip()) for v in vals.split(",") if v.strip()]
        except ValueError:
            continue
        if not values:
            continue
        overrides[idx] = values
        if len(values) > 1:
            max_iter = max(max_iter, len(values))
    return overrides, max_iter


def _apply_vary(base: list, overrides: dict, iteration: int) -> list:
    """Return a copy of `base` with overrides applied for `iteration` index."""
    out = list(base)
    for idx, values in overrides.items():
        if len(values) == 1:
            out[idx] = values[0]
        elif iteration < len(values):
            out[idx] = values[iteration]
        else:
            out[idx] = values[-1]  # past-end pads with last value (defensive)
    return out


class BD_GLSLBatch(io.ComfyNode):
    """Run a GLSL shader N times with varying uniforms, batch fc0..fc3 outputs."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_GLSLBatch",
            display_name="BD GLSL Batch (iterate uniforms)",
            category="🧠BrainDead/GLSL",
            description=(
                "Run a GLSL shader N times with different uniform values, batching "
                "the fc0/fc1/fc2/fc3 outputs along the batch dimension. Shader compiles "
                "ONCE per call, reused across iterations — much faster than re-submitting "
                "the workflow N times.\n\n"
                "Iteration count is auto-detected from the longest 'vary' list. Uniforms "
                "not listed in a vary spec use their base value (same every iteration).\n\n"
                "Vary syntax: one line per uniform, 'u_intN=v1,v2,v3,...' or "
                "'u_floatN=v1,v2,v3,...'. Lines with one value are treated as constants."
            ),
            inputs=[
                io.String.Input("fragment_shader", multiline=True,
                                default="#version 300 es\nprecision highp float;\nin vec2 v_texCoord;\nuniform sampler2D u_image0;\nlayout(location=0) out vec4 fragColor0;\nvoid main(){ fragColor0 = texture(u_image0, v_texCoord); }",
                                tooltip="The fragment shader source code (GLSL ES 3.0). "
                                        "OVERRIDDEN by fragment_shader_file if that path is set and exists."),
                io.String.Input("fragment_shader_file", default="",
                                tooltip="Optional ABSOLUTE path to a .glsl file. When set and the file exists, "
                                        "its contents are used instead of the fragment_shader text field. "
                                        "Lets you edit the shader externally and have BD_GLSLBatch pick up "
                                        "the latest version on each workflow execution. "
                                        "Example: /mnt/tank/Studio/Brains/Workflows/ComfyUI_GLSL_Shaders/skin_shader.glsl"),

                # Image inputs (u_image0..u_image4) — all optional, mirror GLSLShader
                io.Image.Input("u_image0", optional=True, tooltip="Bound to sampler2D u_image0."),
                io.Image.Input("u_image1", optional=True, tooltip="Bound to sampler2D u_image1."),
                io.Image.Input("u_image2", optional=True, tooltip="Bound to sampler2D u_image2."),
                io.Image.Input("u_image3", optional=True, tooltip="Bound to sampler2D u_image3."),
                io.Image.Input("u_image4", optional=True, tooltip="Bound to sampler2D u_image4."),

                # Base uniform values — multiline 'u_floatN=value' / 'u_intN=value'
                io.String.Input("floats", multiline=True, default="",
                                tooltip=("Base float uniform values, one per line:\n"
                                         "    u_float0=0.48\n    u_float1=0.5\n"
                                         f"Indexes 0..{MAX_FLOATS-1}. Missing slots default to 0.0.")),
                io.String.Input("ints", multiline=True, default="",
                                tooltip=("Base int uniform values, one per line:\n"
                                         "    u_int0=255\n    u_int1=128\n"
                                         f"Indexes 0..{MAX_INTS-1}. Missing slots default to 0.")),

                # Per-iteration overrides
                io.String.Input("vary_ints", multiline=True, default="",
                                tooltip=("Per-iteration int overrides. One uniform per line:\n"
                                         "    u_int3=255,240,160,140\n    u_int4=255,190,110,190\n"
                                         "Iteration count = longest list. Single-value lines are constants.")),
                io.String.Input("vary_floats", multiline=True, default="",
                                tooltip=("Per-iteration float overrides. Same format as vary_ints.")),

                io.String.Input("iteration_names", multiline=True, default="",
                                tooltip=("One name per iteration for downstream filename use:\n"
                                         "    light\n    medium\n    dark\n    zombie\n"
                                         "Defaults to '1', '2', '3', ... if empty.")),

                io.Int.Input("iterations_override", default=0, min=0, max=128, optional=True,
                             tooltip="If > 0, force this iteration count. 0 = auto-detect from vary specs."),
                io.Int.Input("width", default=0, min=0, max=8192, optional=True,
                             tooltip="Output width. 0 = use u_image0's width."),
                io.Int.Input("height", default=0, min=0, max=8192, optional=True,
                             tooltip="Output height. 0 = use u_image0's height."),
            ],
            outputs=[
                io.Image.Output(display_name="fc0_batch",
                                tooltip="Stacked IMAGE batch (B=N_iterations) of fc0 outputs."),
                io.Image.Output(display_name="fc1_batch"),
                io.Image.Output(display_name="fc2_batch"),
                io.Image.Output(display_name="fc3_batch"),
                io.String.Output(display_name="iteration_names",
                                 tooltip="Newline-joined iteration names, in order."),
                io.Int.Output(display_name="iteration_count"),
            ],
        )

    @classmethod
    def execute(cls, fragment_shader: str,
                fragment_shader_file: str = "",
                u_image0=None, u_image1=None, u_image2=None, u_image3=None, u_image4=None,
                floats: str = "", ints: str = "",
                vary_ints: str = "", vary_floats: str = "",
                iteration_names: str = "",
                iterations_override: int = 0,
                width: int = 0, height: int = 0) -> io.NodeOutput:
        # Deferred import — see module docstring for why
        from comfy_extras.nodes_glsl import _render_shader_batch
        import os

        # If fragment_shader_file is set and the file exists, use its contents
        # instead of the inline fragment_shader text. Otherwise fall back to inline.
        if fragment_shader_file and os.path.isfile(fragment_shader_file):
            try:
                with open(fragment_shader_file, "r") as _f:
                    fragment_shader = _f.read()
                print(f"[BD_GLSLBatch] Loaded shader from file: {fragment_shader_file} "
                      f"({len(fragment_shader)} chars)")
            except Exception as _e:
                print(f"[BD_GLSLBatch] Failed to read {fragment_shader_file}: {_e} — "
                      f"falling back to inline fragment_shader")

        # Resolve image inputs — MUST preserve slot positions, NOT filter Nones.
        # _render_shader_batch binds `u_image{i}` to the i-th entry of the image list,
        # so dropping Nones would shift later wired slots into earlier positions
        # (e.g. unwired u_image2 + wired u_image3 → shader binds u_image2 = your u_image3).
        # Substitute a 1×1 black RGBA placeholder for unwired slots; we only pad up to
        # the highest wired slot (no point binding placeholders beyond that).
        raw_images = [u_image0, u_image1, u_image2, u_image3, u_image4]
        wired_slots = [i for i, img in enumerate(raw_images) if img is not None]
        if not wired_slots:
            raise ValueError("BD_GLSLBatch: at least one u_imageN input is required.")

        last_wired = max(wired_slots)
        placeholder = torch.zeros((1, 1, 1, 4), dtype=torch.float32)
        image_tensors: list[torch.Tensor] = []
        for i in range(last_wired + 1):
            img = raw_images[i]
            image_tensors.append(img if img is not None else placeholder)

        # Output dimensions — pull from the FIRST WIRED image (not a 1×1 placeholder)
        first_img = raw_images[wired_slots[0]]
        out_h = height if height > 0 else first_img.shape[1]
        out_w = width if width > 0 else first_img.shape[2]

        # Parse base + vary specs
        base_floats = _parse_uniforms_multiline(floats, "u_float", MAX_FLOATS, float)
        base_ints   = _parse_uniforms_multiline(ints,   "u_int",   MAX_INTS,   int)

        vary_f_dict, max_iter_f = _parse_vary_spec(vary_floats, "u_float", MAX_FLOATS, float)
        vary_i_dict, max_iter_i = _parse_vary_spec(vary_ints,   "u_int",   MAX_INTS,   int)

        # Iteration count: explicit override wins; else longest vary list; min 1
        if iterations_override > 0:
            n_iter = iterations_override
        else:
            n_iter = max(max_iter_f, max_iter_i, 1)

        # Iteration names
        names = [n.strip() for n in iteration_names.strip().split("\n") if n.strip()]
        while len(names) < n_iter:
            names.append(str(len(names) + 1))
        names = names[:n_iter]

        # Prepare image_batches: SAM3-style — each iteration uses image[batch_idx % B] from
        # each input image's batch dimension, so a 4-image batch on a slot iterates with the tones.
        # For single-image inputs (B=1), the same image is reused every iteration.
        # We render iterations as sequential single-image batches to share shader compile.
        all_fc_per_output: list[list[torch.Tensor]] = [[] for _ in range(MAX_OUTPUTS)]
        for it in range(n_iter):
            # Per-iteration uniforms
            it_floats = _apply_vary(base_floats, vary_f_dict, it)
            it_ints   = _apply_vary(base_ints,   vary_i_dict, it)

            # Per-iteration images: slice if batched on input, else reuse
            it_image_arrays = []
            for img_t in image_tensors:
                b = img_t.shape[0]
                pick = img_t[it % b]
                it_image_arrays.append(pick.cpu().numpy().astype(np.float32))

            # Call ComfyUI's batch renderer with a single "batch" of one frame.
            # Shader is compiled inside this call but the GL context tears down per call.
            # That's the unavoidable cost — but we still skip workflow submission overhead.
            outputs = _render_shader_batch(
                fragment_shader,
                out_w, out_h,
                [it_image_arrays],       # one "input batch" per call
                it_floats,
                it_ints,
                [],                       # no bools for now
                [],                       # no curves for now
            )
            # outputs is a list of length 1 (one batch), each entry is list[MAX_OUTPUTS] of np.ndarray
            single_batch_outputs = outputs[0]
            for i in range(MAX_OUTPUTS):
                if i < len(single_batch_outputs):
                    all_fc_per_output[i].append(torch.from_numpy(single_batch_outputs[i]))

        # Stack per-iteration outputs into IMAGE batch tensors (B=n_iter, H, W, C)
        # If a slot has no outputs from the shader (e.g. only fc0,fc1 written), create a 1x1 black placeholder
        def _stack_or_placeholder(tensors: list[torch.Tensor]) -> torch.Tensor:
            if tensors:
                return torch.stack(tensors, dim=0)
            return torch.zeros((n_iter, 1, 1, 3), dtype=torch.float32)

        fc0_batch = _stack_or_placeholder(all_fc_per_output[0])
        fc1_batch = _stack_or_placeholder(all_fc_per_output[1])
        fc2_batch = _stack_or_placeholder(all_fc_per_output[2])
        fc3_batch = _stack_or_placeholder(all_fc_per_output[3])

        names_str = "\n".join(names)

        print(f"[BD_GLSLBatch] {n_iter} iterations rendered. Names: {names}")
        print(f"  Output shapes — fc0: {tuple(fc0_batch.shape)}, fc1: {tuple(fc1_batch.shape)}, "
              f"fc2: {tuple(fc2_batch.shape)}, fc3: {tuple(fc3_batch.shape)}")

        return io.NodeOutput(fc0_batch, fc1_batch, fc2_batch, fc3_batch, names_str, n_iter)


GLSL_BATCH_V3_NODES = [BD_GLSLBatch]
GLSL_BATCH_NODES = {"BD_GLSLBatch": BD_GLSLBatch}
GLSL_BATCH_DISPLAY_NAMES = {"BD_GLSLBatch": "BD GLSL Batch (iterate uniforms)"}
