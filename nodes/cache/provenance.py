"""BD_Provenance — the generation record, collected at run time and carried by the file.

A delivered image says nothing about how it was made. The workflow sidecar fixes
half of that: it carries the graph, so the run can be reopened and repeated. It
still does not say which seed actually ran, which checkpoint that loader resolved
to on this machine, which node pack at which commit, how long it took, who made
it, or where the reference came from.

Half of that is knowledge only the RUN has (seeds, resolved model paths, pack
commits, host, duration). The other half is knowledge only a PERSON has (creator,
studio, website, the reference a character came from, a licence). This node holds
both: the person's half as widgets the workflow saves like any other widget
state, the run's half collected when it executes.

WHERE IT GOES

    workflow.extra.bd_provenance

`extra` is a real LiteGraph field that ComfyUI serialises and round-trips
untouched, so a block placed there survives a load, an edit and a save in the
ComfyUI UI. That matters more than it sounds: it means the record is not a
parallel file that drifts, it IS the workflow, and every save node that writes a
sidecar or a PNG text chunk carries it without knowing it exists.

ORDERING — THE ONE THING THAT CAN GO WRONG

The block reaches a save node because both read the same `extra_pnginfo` dict for
the run, which this node mutates in place. So this node must EXECUTE FIRST. Wire
`anchor` into the chain ahead of the save (any type, passed straight through) and
execution order is guaranteed by the dependency. Unwired, the node still runs and
still displays the record, but ComfyUI may schedule it after the save and the
files will not carry it — so the status says so rather than letting it look fine.

MODEL LINKS

A filename cannot be turned into a download URL by guessing. What the run does
know is exactly which file was resolved, its size and (optionally) its hash.
Anything more has to be told to us: `config/model_sources.json` maps a model
filename to a URL, and every entry that matches gets its link filled in. No entry
means no link, not a made-up one.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time

from comfy_api.latest import io, ui

# filename -> url, loaded once from config/model_sources.json.
_MODEL_SOURCES: dict | None = None
# (path, size, mtime) -> sha256, so a repeated run does not re-read 8 GB.
_HASH_CACHE: dict = {}
# custom_nodes dir -> {"url": ..., "commit": ...}; git is slow, packs do not move.
_PACK_CACHE: dict = {}

SEED_INPUTS = ("seed", "noise_seed", "rand_seed", "random_seed")
MODEL_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf",
                  ".onnx", ".sft")


def _config_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "config")


def _model_sources() -> dict:
    global _MODEL_SOURCES
    if _MODEL_SOURCES is None:
        _MODEL_SOURCES = {}
        path = os.path.join(_config_dir(), "model_sources.json")
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                _MODEL_SOURCES = {str(k): v for k, v in data.items()}
        except FileNotFoundError:
            pass
        except Exception as e:
            print("[BD Provenance] model_sources.json unreadable: %s" % e, flush=True)
    return _MODEL_SOURCES


def collect_seeds(prompt) -> list:
    """Every seed the run actually used, with the node that used it.

    Read from the API prompt rather than the widget defaults, so a seed the
    frontend randomised before submitting is the one recorded.
    """
    out = []
    if not isinstance(prompt, dict):
        return out
    for nid, node in prompt.items():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict):
            continue
        for name, value in inputs.items():
            if name in SEED_INPUTS and isinstance(value, (int, float)):
                out.append({"node": str(nid),
                            "class_type": node.get("class_type"),
                            "input": name,
                            "value": int(value)})
    return sorted(out, key=lambda d: (d["class_type"] or "", d["node"]))


def _resolve_model(filename: str):
    """The file a loader widget actually resolved to, or None.

    ComfyUI keeps the search paths; ask it rather than guessing a directory.
    """
    try:
        import folder_paths
    except Exception:
        return None
    for kind in folder_paths.folder_names_and_paths:
        try:
            hit = folder_paths.get_full_path(kind, filename)
        except Exception:
            continue
        if hit:
            return hit, kind
    return None


def _sha256(path: str) -> str:
    try:
        st = os.stat(path)
    except OSError:
        return ""
    key = (path, st.st_size, int(st.st_mtime))
    if key in _HASH_CACHE:
        return _HASH_CACHE[key]
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
    except OSError:
        return ""
    _HASH_CACHE[key] = h.hexdigest()
    return _HASH_CACHE[key]


def collect_models(prompt, hash_models: bool = False) -> list:
    """Every model file the prompt named, resolved to a real path on this host.

    Detection is by VALUE, not by a list of loader class names: any string input
    ending in a model suffix is a model, which keeps working for node packs this
    one has never heard of.
    """
    seen, out = set(), []
    if not isinstance(prompt, dict):
        return out
    sources = _model_sources()
    for nid, node in sorted(prompt.items()):
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict):
            continue
        for name, value in inputs.items():
            if not isinstance(value, str) or not value.lower().endswith(MODEL_SUFFIXES):
                continue
            if value in seen:
                continue
            seen.add(value)
            rec = {"name": value, "node": str(nid),
                   "class_type": node.get("class_type"), "input": name}
            hit = _resolve_model(value)
            if hit:
                path, kind = hit
                rec["kind"] = kind
                rec["path"] = path
                try:
                    rec["bytes"] = os.path.getsize(path)
                except OSError:
                    pass
                if hash_models:
                    digest = _sha256(path)
                    if digest:
                        rec["sha256"] = digest
            url = sources.get(value) or sources.get(os.path.basename(value))
            if url:
                rec["url"] = url
            out.append(rec)
    return out


def _pack_info(directory: str) -> dict:
    """A node pack's repo URL and commit, read from its own git checkout."""
    if directory in _PACK_CACHE:
        return _PACK_CACHE[directory]
    info = {}
    for args, key in ((["config", "--get", "remote.origin.url"], "url"),
                      (["rev-parse", "--short", "HEAD"], "commit")):
        try:
            r = subprocess.run(["git", "-C", directory] + args,
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                info[key] = r.stdout.strip()
        except Exception:
            break
    if info.get("url", "").startswith("git@"):
        # git@github.com:Owner/Repo.git -> a link someone can actually click
        host, _, path = info["url"][4:].partition(":")
        info["url"] = "https://%s/%s" % (host, path[:-4] if path.endswith(".git") else path)
    _PACK_CACHE[directory] = info
    return info


def collect_addons(prompt) -> list:
    """The node packs the run actually used — not everything installed.

    A class_type is mapped back to the module that defined it, and the module
    path back to its custom_nodes directory. Stock ComfyUI nodes resolve outside
    custom_nodes and are reported once as `comfyui`.
    """
    if not isinstance(prompt, dict):
        return []
    used = {n.get("class_type") for n in prompt.values()
            if isinstance(n, dict) and n.get("class_type")}
    try:
        import nodes as comfy_nodes
        mappings = comfy_nodes.NODE_CLASS_MAPPINGS
    except Exception:
        return []

    by_dir, stock = {}, 0
    for class_type in used:
        cls = mappings.get(class_type)
        module = sys.modules.get(getattr(cls, "__module__", "") or "")
        file = getattr(module, "__file__", None)
        if not file:
            stock += 1
            continue
        parts = os.path.abspath(file).replace("\\", "/").split("/")
        if "custom_nodes" not in parts:
            stock += 1
            continue
        i = parts.index("custom_nodes")
        if i + 1 >= len(parts):
            stock += 1
            continue
        directory = "/".join(parts[:i + 2])
        by_dir.setdefault(directory, set()).add(class_type)

    out = []
    if stock:
        out.append({"name": "comfyui", "nodes_used": stock})
    for directory, classes in sorted(by_dir.items()):
        rec = {"name": os.path.basename(directory), "nodes_used": len(classes)}
        rec.update(_pack_info(directory))
        out.append(rec)
    return out


def _comfyui_version() -> str:
    try:
        from comfyui_version import __version__
        return str(__version__)
    except Exception:
        return ""


def build_provenance(prompt, fields: dict, include_models=True,
                     include_addons=True, hash_models=False) -> dict:
    """The block, run-side facts plus whatever the person filled in."""
    block = {
        "schema": "bd_provenance/1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "started_at": time.time(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "comfyui_version": _comfyui_version(),
        "seeds": collect_seeds(prompt),
    }
    if include_models:
        block["models"] = collect_models(prompt, hash_models)
    if include_addons:
        block["addons"] = collect_addons(prompt)
    for k, v in fields.items():
        if isinstance(v, str):
            v = v.strip()
        if v:
            block[k] = v
    return block


def _summary(block: dict) -> str:
    lines = ["%s  %s  %s" % (block.get("created", ""), block.get("host", ""),
                             block.get("run_id", ""))]
    for k in ("title", "creator", "studio", "website", "reference", "license"):
        if block.get(k):
            lines.append("%-9s %s" % (k + ":", block[k]))
    seeds = block.get("seeds") or []
    if seeds:
        lines.append("seeds:    " + ", ".join(
            "%s=%d" % (s["class_type"], s["value"]) for s in seeds[:6]))
        if len(seeds) > 6:
            lines[-1] += "  (+%d)" % (len(seeds) - 6)
    for key, label in (("models", "models"), ("addons", "addons")):
        items = block.get(key) or []
        if items:
            lines.append("%-9s %s" % (label + ":", ", ".join(
                i.get("name", "?") for i in items[:6])))
    if block.get("notes"):
        lines.append("notes:    " + block["notes"].replace("\n", " / ")[:200])
    return "\n".join(lines)


class BD_Provenance(io.ComfyNode):
    """Stamp the run's generation record into the workflow the files carry."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_Provenance",
            display_name="BD Provenance",
            category="🧠BrainDead/Cache",
            description=(
                "Collect the generation record — seeds, resolved models, node packs "
                "and their commits, host, date, run id — and write it into "
                "workflow.extra.bd_provenance, where ComfyUI round-trips it and every "
                "sidecar and PNG chunk carries it. Fill creator/studio/reference here; "
                "they are saved with the workflow like any other widget. Wire `anchor` "
                "ahead of the save node so this runs first."
            ),
            is_output_node=True,
            inputs=[
                io.AnyType.Input(
                    "anchor", optional=True,
                    tooltip="Pass anything through on its way to the save node. This is "
                            "what guarantees this node runs BEFORE the save, so the files "
                            "carry the record. Unwired, ordering is not guaranteed.",
                ),
                io.String.Input("title", default="", optional=True,
                                tooltip="What this run produced."),
                io.String.Input("creator", default="", optional=True),
                io.String.Input("studio", default="", optional=True),
                io.String.Input("website", default="", optional=True),
                io.String.Input("reference", default="", optional=True,
                                tooltip="Where the source or the likeness came from — a "
                                        "URL, a character name, a shoot, a ticket."),
                io.String.Input("license", default="", optional=True),
                io.String.Input("notes", default="", multiline=True, optional=True),
                io.Boolean.Input("include_models", default=True, optional=True),
                io.Boolean.Input("include_addons", default=True, optional=True),
                io.Boolean.Input(
                    "hash_models", default=False, optional=True,
                    tooltip="sha256 every resolved model file. Exact, and slow the first "
                            "time — results are cached per (path, size, mtime).",
                ),
            ],
            outputs=[
                io.AnyType.Output(display_name="anchor"),
                io.String.Output(display_name="provenance"),
            ],
            hidden=[io.Hidden.extra_pnginfo, io.Hidden.prompt,
                    io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, anchor=None, title="", creator="", studio="", website="",
                reference="", license="", notes="", include_models=True,
                include_addons=True, hash_models=False) -> io.NodeOutput:
        prompt = cls.hidden.prompt
        extra = cls.hidden.extra_pnginfo if isinstance(cls.hidden.extra_pnginfo, dict) else {}
        workflow = extra.get("workflow")

        block = build_provenance(
            prompt,
            {"title": title, "creator": creator, "studio": studio,
             "website": website, "reference": reference, "license": license,
             "notes": notes},
            include_models, include_addons, hash_models)

        try:
            from comfy.cli_args import args  # noqa: F401
            from server import PromptServer
            run_id = getattr(PromptServer.instance, "last_prompt_id", "")
            if run_id:
                block["run_id"] = str(run_id)
        except Exception:
            pass

        notes_out = []
        if isinstance(workflow, dict):
            # In place: the save nodes read this same dict later in the run.
            workflow.setdefault("extra", {})
            if isinstance(workflow["extra"], dict):
                workflow["extra"]["bd_provenance"] = block
                notes_out.append("stamped into workflow.extra")
            else:
                notes_out.append("workflow.extra is not an object — not stamped")
        else:
            notes_out.append(
                "no UI graph in extra_pnginfo, so there is nothing to stamp. ComfyUI "
                "fills it only for FRONTEND submissions; a headless caller must send "
                'extra_data={"extra_pnginfo": {"workflow": ui_graph}}')

        if anchor is None and isinstance(prompt, dict):
            savers = [n.get("class_type") for n in prompt.values()
                      if isinstance(n, dict)
                      and str(n.get("class_type", "")).startswith(("BD_Save", "SaveImage"))]
            if savers:
                notes_out.append(
                    "anchor is unwired and this graph saves files (%s) — execution "
                    "order is not guaranteed, so the record may be stamped after they "
                    "are written. Wire anchor through on the way to the save."
                    % ", ".join(sorted(set(savers))[:3]))

        summary = _summary(block)
        status = summary + "\n" + "\n".join("* " + n for n in notes_out)
        print("[BD Provenance] " + "; ".join(notes_out), flush=True)
        return io.NodeOutput(anchor, status, ui=ui.PreviewText(status))


PROVENANCE_V3_NODES = [BD_Provenance]
PROVENANCE_NODES = {"BD_Provenance": BD_Provenance}
PROVENANCE_DISPLAY_NAMES = {"BD_Provenance": "BD Provenance"}
