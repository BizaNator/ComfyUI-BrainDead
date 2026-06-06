#!/usr/bin/env python3
"""
audit_workflows.py — Validate (and optionally fix/rebuild) example_workflows/*.json
against the live ComfyUI /object_info schemas.

For each node in each workflow:
  1. Fetches the live schema from /object_info/<NodeType>
  2. Maps widgets_values to their input definitions
     (required inputs first, optional second; skipping wired and connection-type inputs)
  3. Handles both V1 COMBO (spec[0] is a list) and V3 COMBO (spec[0]=='COMBO', options in spec[1])
  4. Checks INT/FLOAT against min/max, COMBO against options list
  5. With --fix: replaces bad values with the schema default and rewrites the JSON
  6. With --rebuild: completely replaces all widget_values with schema defaults (safe for templates)

Usage:
    python3 tools/audit_workflows.py [--fix] [--rebuild] [--server http://127.0.0.1:8188] [<workflow.json> ...]

If no files are given, audits all example_workflows/*.json in the repo root.
"""

import json
import sys
import argparse
import urllib.request
from pathlib import Path

# Only these primitive types ever produce a widget in the ComfyUI UI.
# Everything else (IMAGE, MASK, TRIMESH, TRELLIS2_CONDITIONING, custom types, …)
# is a socket-only connection type — no widget even when unwired.
PRIMITIVE_WIDGET_TYPES = {"STRING", "INT", "FLOAT", "BOOLEAN", "IMAGEUPLOAD"}
# V1 COMBO: spec[0] is a list of options
# V3 COMBO: spec[0] == 'COMBO', options in spec[1]['options']
# Both are handled in widget_inputs_for_node.

# IMAGEUPLOAD widgets exist but have no meaningful validation
WIDGET_PASS_THROUGH = {"IMAGEUPLOAD"}


def fetch_schema(server: str, node_type: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{server}/object_info/{node_type}", timeout=5) as r:
            data = json.load(r)
            return data.get(node_type)
    except Exception:
        return None


CONTROL_AFTER_GENERATE_OPTIONS = ["fixed", "increment", "decrement", "randomize"]


def _normalize_spec(raw_type, opts: dict) -> tuple:
    """
    Normalize V1 and V3 COMBO specs to a unified (is_combo, itype) form.

    V1: raw_type is a list of options → is_combo=True, itype=raw_type
    V3: raw_type=='COMBO', options in opts['options'] → is_combo=True, itype=opts['options']
    Primitive/connection: everything else
    """
    if isinstance(raw_type, list):
        # V1 COMBO
        return True, raw_type
    if raw_type == "COMBO":
        # V3 COMBO — options live in opts
        return True, opts.get("options", [])
    # Primitive or socket-only type
    return False, raw_type


def widget_inputs_for_node(schema: dict, linked_names: set[str]) -> list[tuple[str, str | list, dict]]:
    """
    Return the ordered list of (name, type, opts) that map to widgets_values slots.
    Order: required inputs first, optional second.
    Skips: connection-type inputs and any input wired via a link.
    Handles V1 COMBO (list) and V3 COMBO ('COMBO' string with options in opts).
    """
    result = []
    for phase in ("required", "optional"):
        for name, spec in schema["input"].get(phase, {}).items():
            raw_type = spec[0]
            opts = spec[1] if len(spec) > 1 else {}

            is_combo, itype = _normalize_spec(raw_type, opts)

            # Only primitive types and COMBOs produce widgets.
            # All non-primitive string types (IMAGE, MASK, TRIMESH, PIXAL3D_INPUT, …)
            # are socket-only — no widget even when unwired.
            if not is_combo and itype not in PRIMITIVE_WIDGET_TYPES:
                continue

            # Skip if wired via a link
            if name in linked_names:
                continue

            result.append((name, itype, opts))

            # ComfyUI frontend injects control_after_generate after every seed-type INT
            if not is_combo and itype == "INT" and "seed" in name.lower():
                result.append(("control_after_generate", CONTROL_AFTER_GENERATE_OPTIONS, {}))
    return result


def _is_dynamic_combo(opts: dict, options: list) -> bool:
    """True when COMBO options are dynamically populated (file lists, model lists).
    We detect this by checking if the options look like file paths or are very long."""
    if len(options) > 30:
        return True
    if any("." in str(o) and "/" not in str(o) and len(str(o)) > 4 for o in options[:3]):
        # Looks like filenames (has dot, not a path, not trivially short like "v1.0")
        return True
    return False


def check_value(name: str, itype, opts: dict, val) -> list[str]:
    issues = []
    if isinstance(itype, str) and itype in WIDGET_PASS_THROUGH:
        return []  # no validation for upload/custom widgets
    if isinstance(itype, list):  # COMBO (V1 or V3 normalized to list)
        if _is_dynamic_combo(opts, itype):
            return []  # file/model lists are machine-specific; skip validation
        if val not in itype:
            issues.append(f"'{val}' not in options {itype[:5]}{'...' if len(itype)>5 else ''}")
    elif itype == "FLOAT":
        mn, mx = opts.get("min"), opts.get("max")
        if not isinstance(val, (int, float)):
            issues.append(f"expected float, got {type(val).__name__}")
        else:
            if mn is not None and val < mn:
                issues.append(f"{val} < min {mn}")
            if mx is not None and val > mx:
                issues.append(f"{val} > max {mx}")
    elif itype == "INT":
        mn, mx = opts.get("min"), opts.get("max")
        if not isinstance(val, (int, float)):
            issues.append(f"expected int, got {type(val).__name__}")
        else:
            if mn is not None and val < mn:
                issues.append(f"{val} < min {mn}")
            if mx is not None and val > mx:
                issues.append(f"{val} > max {mx}")
    elif itype == "BOOLEAN":
        if not isinstance(val, bool):
            issues.append(f"expected bool, got {type(val).__name__} ({val!r})")
    return issues


def default_for(itype, opts: dict):
    """Return the schema default, or a safe fallback."""
    if "default" in opts:
        return opts["default"]
    if isinstance(itype, list) and itype == CONTROL_AFTER_GENERATE_OPTIONS:
        return "fixed"
    if isinstance(itype, list):
        return itype[0]
    if itype == "FLOAT":
        mn, mx = opts.get("min"), opts.get("max")
        if mn is not None:
            return max(mn, 0.0)
        return 0.0
    if itype == "INT":
        mn = opts.get("min")
        if mn is not None:
            return max(mn, 0)
        return 0
    if itype == "BOOLEAN":
        return False
    if itype == "STRING":
        return opts.get("default", "")
    return None


def audit_workflow(path: Path, server: str, fix: bool, rebuild: bool) -> int:
    wf = json.loads(path.read_text())
    nodes = wf.get("nodes", [])

    # Build a map of links: dst_node_id → set of dst input names that are wired
    linked: dict[int, set[str]] = {}
    for link in wf.get("links", []):
        # link = [link_id, src_node, src_slot, dst_node, dst_slot, type]
        if len(link) >= 4:
            dst_node_id = link[3]
            linked.setdefault(dst_node_id, set())

    # Populate linked input names from node.inputs
    for node in nodes:
        nid = node["id"]
        for inp in node.get("inputs", []):
            if inp.get("link") is not None:
                linked.setdefault(nid, set()).add(inp["name"])

    total_issues = 0
    modified = False

    for node in nodes:
        ntype = node.get("type", "")
        if ntype in ("MarkdownNote", "Note", "PrimitiveNode"):
            continue

        schema = fetch_schema(server, ntype)
        if schema is None:
            print(f"  [{ntype}] (id={node['id']}) — not in /object_info, skipping")
            continue

        wired = linked.get(node["id"], set())
        widget_order = widget_inputs_for_node(schema, wired)
        vals = list(node.get("widgets_values", []))

        node_issues = []

        # --rebuild: completely replace widget_values with schema defaults
        if rebuild and widget_order:
            new_vals = [default_for(itype, opts) for _, itype, opts in widget_order]
            if new_vals != vals:
                node_issues.append(f"  🔄 Rebuilding {len(vals)} → {len(new_vals)} values from schema defaults")
                vals = new_vals
                node["widgets_values"] = vals
                modified = True
        else:
            # Fewer values than schema expects = real problem (widgets will shift)
            # More values than schema = usually hidden widgets (IMAGEUPLOAD etc); warn only
            if len(vals) < len(widget_order):
                node_issues.append(
                    f"  ⚠  widget count mismatch: JSON has {len(vals)} values, schema expects {len(widget_order)} — padding with defaults"
                )
                if fix:
                    for i in range(len(vals), len(widget_order)):
                        _, itype, opts = widget_order[i]
                        vals.append(default_for(itype, opts))
                    node["widgets_values"] = vals
                    modified = True
            elif len(vals) > len(widget_order):
                # Extra widgets (e.g. IMAGEUPLOAD hidden buttons) — warn but don't fix
                node_issues.append(
                    f"  ℹ  {len(vals) - len(widget_order)} extra widget value(s) beyond schema (likely hidden upload/custom widgets — ok)"
                )

            for i, (name, itype, opts) in enumerate(widget_order):
                if i >= len(vals):
                    break
                val = vals[i]
                issues = check_value(name, itype, opts, val)
                if issues:
                    dflt = default_for(itype, opts)
                    node_issues.append(f"  ❌ [{i}] {name}={val!r} → {', '.join(issues)}  (default={dflt!r})")
                    if fix:
                        vals[i] = dflt
                        modified = True
                    total_issues += 1

        if node_issues:
            print(f"\n  [{ntype}] id={node['id']}")
            for msg in node_issues:
                print(msg)

    if (fix or rebuild) and modified:
        path.write_text(json.dumps(wf, indent=2, ensure_ascii=False))
        print(f"\n  ✅ Fixed and saved: {path.name}")

    return total_issues


def main():
    parser = argparse.ArgumentParser(description="Audit (and optionally fix) BrainDead workflow templates.")
    parser.add_argument("--fix", action="store_true", help="Replace bad values with schema defaults")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild ALL widget_values from schema defaults (safe for templates)")
    parser.add_argument("--server", default="http://127.0.0.1:8188", help="ComfyUI server URL")
    parser.add_argument("files", nargs="*", help="Specific .json files to audit (default: all example_workflows/*.json)")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        paths = sorted((repo_root / "example_workflows").glob("*.json"))

    mode = "[REBUILD MODE]" if args.rebuild else ("[FIX MODE]" if args.fix else "[DRY RUN]")
    print(f"{mode} Auditing {len(paths)} workflow(s) against {args.server}\n")

    grand_total = 0
    for path in paths:
        print(f"{'='*60}\n{path.name}")
        issues = audit_workflow(path, args.server, args.fix, args.rebuild)
        if issues == 0:
            print("  ✓ No issues")
        grand_total += issues

    print(f"\n{'='*60}")
    print(f"Total issues: {grand_total}")
    if grand_total and not args.fix and not args.rebuild:
        print("Run with --fix to auto-correct using schema defaults.")
        print("Run with --rebuild to fully reset all widget values to schema defaults.")


if __name__ == "__main__":
    main()
