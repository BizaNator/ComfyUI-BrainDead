"""
BD Save Context — define template + variables once, downstream BD_SaveFile nodes
look it up by context_id and just specify their suffix per save point.

Pattern:
  BD_SaveContext(context_id="character", template="...", character="letti", version="03")
    → context_id (STRING handle)

  context_id → BD_SaveFile(image,  context_id, suffix="_albedo")  → file saved
             → BD_SaveFile(mask,   context_id, suffix="_skin_mask") → file saved
             → BD_SaveFile(latent, context_id, suffix="_latent")    → file saved
             ...

The context lives in a global dict keyed by context_id. BD_SaveFile (extended)
looks it up, resolves `template + suffix` with the variables, and saves to that
path. One save node handles all data types via AnyType detection.
"""

import os
import re
import time
from glob import glob

import folder_paths
from comfy_api.latest import io


_SAVE_CONTEXTS: dict[str, dict] = {}


def _resolve_template(template: str, vars_dict: dict, strict: bool = False) -> tuple[str, list]:
    unresolved = []

    def _repl(m):
        key = m.group(1)
        if key in vars_dict:
            return str(vars_dict[key])
        unresolved.append(key)
        return f"%{key}%"

    result = re.sub(r"%(\w+)%", _repl, template)
    if unresolved and strict:
        raise ValueError(
            f"Undefined template variables {sorted(set(unresolved))}. "
            f"Available: {sorted(vars_dict.keys())}"
        )
    return result, unresolved


def _parse_custom_vars(text: str) -> dict:
    out = {}
    for line in (text or "").strip().split("\n"):
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            if k:
                out[k] = v.strip()
    return out


def _next_increment(folder: str, base: str, ext: str) -> int:
    pattern = os.path.join(folder, f"{base}_*.{ext}")
    nums = []
    for p in glob(pattern):
        m = re.search(rf"{re.escape(base)}_(\d+)\.{re.escape(ext)}$", os.path.basename(p))
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1


def _resolve_save_path(template: str, vars_dict: dict, suffix: str,
                       ext: str, auto_increment: bool, increment_padding: int) -> tuple[str, str]:
    """Resolve template + suffix into (full_filepath, relative_subpath)."""
    full_template = template + suffix
    resolved, unresolved = _resolve_template(full_template, vars_dict)
    resolved = resolved.replace("\\", "/")

    output_base = folder_paths.get_output_directory()
    if "/" in resolved:
        subdir, base = resolved.rsplit("/", 1)
        folder = os.path.join(output_base, subdir)
    else:
        folder = output_base
        base = resolved

    os.makedirs(folder, exist_ok=True)

    if auto_increment:
        n = _next_increment(folder, base, ext)
        filename = f"{base}_{n:0{increment_padding}d}.{ext}"
    else:
        filename = f"{base}.{ext}"

    full_path = os.path.join(folder, filename)
    rel_path = os.path.relpath(full_path, output_base).replace("\\", "/")
    return full_path, rel_path


class BD_SaveContext(io.ComfyNode):
    """Define a save context: template + variables stored globally for downstream save nodes."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_SaveContext",
            display_name="BD Save Context",
            category="🧠BrainDead/Cache",
            is_output_node=True,
            description=(
                "Define a save-path template + variables ONCE, downstream BD_SaveImageWithContext "
                "and BD_SaveMaskWithContext nodes look it up by context_id. Each save node only "
                "needs to specify its `suffix` (e.g. '_albedo', '_head', '_skin_mask'). Add new "
                "save points anytime by wiring the same context_id — no concat-string spaghetti."
            ),
            inputs=[
                io.String.Input(
                    "context_id", default="default",
                    tooltip="Identifier for this save context. Save nodes match by this id. Use distinct ids "
                            "if you have multiple parallel contexts (e.g. 'character_v1' and 'mannequin_v1')."
                ),
                io.String.Input(
                    "template", default="characters/%character%/%name%_v%version%%suffix%",
                    tooltip="Base template with %varname% placeholders. The save node's `suffix` input "
                            "is appended to this template before resolution. Use %suffix% in the template "
                            "to control suffix placement (defaults to end if omitted)."
                ),
                io.String.Input("character", default="", optional=True),
                io.String.Input("name", default="", optional=True),
                io.String.Input("version", default="01", optional=True),
                io.String.Input("project", default="", optional=True),
                io.String.Input("tag", default="", optional=True,
                                tooltip="Generic tag — typical use: part name like 'face', 'topwear', 'arm-l'."),
                io.String.Input(
                    "custom_vars", multiline=True, default="", optional=True,
                    tooltip="Additional variables, one per line as key=value. e.g. resolution=4096, layer=topwear."
                ),
                io.Boolean.Input("auto_increment", default=True, optional=True,
                                 tooltip="Append _NNN to avoid overwriting. Applies to ALL save nodes using this context."),
                io.Int.Input("increment_padding", default=3, min=1, max=10, step=1, optional=True),
                io.Boolean.Input("strict", default=False, optional=True,
                                 tooltip="If True, save nodes raise an error when the template contains undefined "
                                         "%variables% at save time."),
            ],
            outputs=[
                io.String.Output(display_name="context_id"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> str:
        return f"ctx_{time.time()}"

    @classmethod
    def execute(cls, context_id, template, character="", name="", version="01",
                project="", tag="", custom_vars="",
                auto_increment=True, increment_padding=3, strict=False) -> io.NodeOutput:
        vars_dict = {
            "character": character, "name": name, "version": version,
            "project": project, "tag": tag,
        }
        vars_dict.update(_parse_custom_vars(custom_vars))

        _SAVE_CONTEXTS[context_id] = {
            "template": template,
            "vars": vars_dict,
            "auto_increment": auto_increment,
            "increment_padding": increment_padding,
            "strict": strict,
        }

        preview, unresolved = _resolve_template(template, vars_dict)
        status = (
            f"context_id='{context_id}' registered\n"
            f"  template: {template}\n"
            f"  vars: {vars_dict}\n"
            f"  preview (no suffix): {preview}"
        )
        if unresolved:
            status += f"\n  WARNING: undefined in template: {sorted(set(unresolved))}"
        print(f"[BD SaveContext] {status}", flush=True)
        return io.NodeOutput(context_id, status)


def get_context(context_id: str) -> dict | None:
    """Public accessor for registered save contexts. Returns None if not found."""
    return _SAVE_CONTEXTS.get(context_id)


def auto_pick_context() -> str | None:
    """If exactly one context is registered, return its id. Else None."""
    if len(_SAVE_CONTEXTS) == 1:
        return next(iter(_SAVE_CONTEXTS))
    return None


def resolve_context_path(context_id: str, suffix: str, ext: str,
                        node_filename: str = "", node_name_prefix: str = "") -> tuple[str, str]:
    """Resolve a context_id + suffix + node-level overrides into (full_path, relative_path).

    Variables exposed to template:
      - All context vars (character, name, version, project, tag, plus custom_vars)
      - %suffix% — node's suffix (always set, even empty)
      - %filename% — node's filename (overrides %name% when non-default)
      - %name_prefix% — node's name_prefix
    """
    ctx = _SAVE_CONTEXTS.get(context_id)
    if ctx is None:
        available = sorted(_SAVE_CONTEXTS.keys())
        raise ValueError(
            f"context_id='{context_id}' not registered. "
            f"Add a BD_SaveContext node upstream. Available: {available}"
        )
    template = ctx["template"]
    vars_dict = dict(ctx["vars"])
    if "%suffix%" not in template:
        template = template + "%suffix%"
    vars_dict["suffix"] = suffix or ""
    if node_filename and node_filename != "saved_file":
        vars_dict["filename"] = node_filename
        if "%filename%" not in template:
            vars_dict["name"] = node_filename
    if node_name_prefix:
        vars_dict["name_prefix"] = node_name_prefix
    return _resolve_save_path(
        template, vars_dict, suffix="",
        ext=ext, auto_increment=ctx["auto_increment"],
        increment_padding=ctx["increment_padding"],
    )


class BD_GetContextPath(io.ComfyNode):
    """Resolve a save context's template into a STRING for ANY save node's filename input.

    Output `filename_prefix` is suitable for ComfyUI's built-in SaveImage `filename_prefix` field
    (no extension, no auto-increment — SaveImage adds those itself). Output `full_path` includes
    extension and increment for nodes that want a complete path.

    Auto-picks the single registered context when context_id is empty.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_GetContextPath",
            display_name="BD Get Context Path",
            category="🧠BrainDead/Cache",
            description=(
                "Fetch a resolved STRING path from a BD_SaveContext. Wire `filename_prefix` "
                "into ComfyUI's built-in SaveImage (or any third-party save node that takes a "
                "STRING filename). Auto-picks the context if exactly one is registered."
            ),
            inputs=[
                io.String.Input("context_id", default="", optional=True,
                                tooltip="Match a BD_SaveContext id. Empty + exactly one registered = auto-pick."),
                io.String.Input("suffix", default="", optional=True,
                                tooltip="Per-fetch suffix → %suffix% in the template (e.g. '_albedo')."),
                io.String.Input("filename_override", default="", optional=True,
                                tooltip="If set, overrides %name% / %filename% in the template for THIS fetch only. "
                                        "Useful when the context defines a base template but you want a different leaf "
                                        "for this particular save."),
                io.Boolean.Input("include_extension", default=False, optional=True,
                                 tooltip="When True: full_path includes the extension (.png). "
                                         "filename_prefix never includes extension regardless."),
                io.String.Input("extension", default="png", optional=True,
                                tooltip="Extension used when include_extension=True. PNG by default."),
                io.Boolean.Input("include_increment", default=False, optional=True,
                                 tooltip="When True: full_path includes auto-increment _NNN suffix. "
                                         "filename_prefix never includes increment (let SaveImage handle it)."),
            ],
            outputs=[
                io.String.Output(display_name="filename_prefix"),
                io.String.Output(display_name="full_path"),
                io.String.Output(display_name="directory"),
                io.String.Output(display_name="basename"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, context_id="", suffix="", filename_override="",
                include_extension=False, extension="png",
                include_increment=False) -> io.NodeOutput:
        effective_ctx_id = context_id
        if not effective_ctx_id:
            picked = auto_pick_context()
            if picked is not None:
                effective_ctx_id = picked

        ctx = get_context(effective_ctx_id) if effective_ctx_id else None
        if ctx is None:
            available = sorted(_SAVE_CONTEXTS.keys())
            return io.NodeOutput(
                "", "", "", "",
                f"BD_GetContextPath: no usable context. context_id='{context_id}' not found, "
                f"auto-pick failed (registered: {available}). Add a BD_SaveContext upstream."
            )

        template = ctx["template"]
        vars_dict = dict(ctx["vars"])
        if "%suffix%" not in template:
            template = template + "%suffix%"
        vars_dict["suffix"] = suffix or ""
        if filename_override:
            vars_dict["filename"] = filename_override
            if "%filename%" not in template:
                vars_dict["name"] = filename_override

        resolved, _ = _resolve_template(template, vars_dict)
        resolved = resolved.replace("\\", "/")

        output_base = folder_paths.get_output_directory()
        if "/" in resolved:
            subdir, base = resolved.rsplit("/", 1)
            directory = os.path.join(output_base, subdir)
        else:
            directory = output_base
            base = resolved

        ext = (extension or "png").lstrip(".")

        base_with_inc = base
        if include_increment and ctx.get("auto_increment", True):
            n = _next_increment(directory, base, ext)
            pad = ctx.get("increment_padding", 3)
            base_with_inc = f"{base}_{n:0{pad}d}"

        if include_extension:
            full_basename = f"{base_with_inc}.{ext}"
        else:
            full_basename = base_with_inc
        full_path = os.path.join(directory, full_basename)

        auto_str = " (auto-picked)" if not context_id else ""
        status = (
            f"context='{effective_ctx_id}'{auto_str} → filename_prefix='{resolved}'"
            + (f"  full_path='{full_path}'" if include_extension or include_increment else "")
        )
        print(f"[BD GetContextPath] {status}", flush=True)
        return io.NodeOutput(resolved, full_path, directory, base, status)


SAVE_CONTEXT_V3_NODES = [BD_SaveContext, BD_GetContextPath]
SAVE_CONTEXT_NODES = {n.__name__: n for n in SAVE_CONTEXT_V3_NODES}
SAVE_CONTEXT_DISPLAY_NAMES = {
    "BD_SaveContext": "BD Save Context",
    "BD_GetContextPath": "BD Get Context Path",
}
