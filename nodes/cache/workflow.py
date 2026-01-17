"""
V3 API workflow versioning nodes for automatic backup and recovery.

BD_WorkflowVersionCache - Auto-save workflow versions on change
BD_WorkflowVersionList - List saved versions with metadata
BD_WorkflowVersionRestore - Restore a specific version
BD_WorkflowVersionClear - Delete old versions
"""

import os
import time
import json

from comfy_api.latest import io

from ...utils.shared import (
    OUTPUT_DIR,
    hash_workflow_structure,
    hash_workflow_full,
    auto_workflow_id,
    list_workflow_versions,
    save_workflow_version,
    load_workflow_version,
    compare_workflow_versions,
    clear_workflow_versions,
)

# Track last known workflow hash per workflow_id to detect changes
_WORKFLOW_HASH_CACHE = {}


class BD_WorkflowVersionCache(io.ComfyNode):
    """
    Automatically save workflow versions when changes are detected.

    USAGE:
    1. Add this node anywhere in your workflow
    2. Set a workflow_id or leave empty for auto-generated ID based on structure
    3. Configure max_versions to control how many versions are kept
    4. Workflow is auto-saved on first run and whenever changes are detected
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_WorkflowVersionCache",
            display_name="BD Workflow Version Cache",
            category="🧠BrainDead/Cache",
            description="Automatically save workflow versions for backup and recovery. Set a manual workflow_id to track evolution over time.",
            is_output_node=True,
            inputs=[
                io.String.Input("workflow_id", default="", tooltip="Name for this workflow (leave empty for auto-detect)"),
                io.Int.Input("max_versions", default=50, min=0, max=999, tooltip="How many versions to keep (0 = unlimited)"),
                io.Boolean.Input("save_on_any_change", default=True),
                io.Boolean.Input("enabled", default=True, tooltip="Toggle versioning on/off"),
                io.AnyType.Input("trigger", optional=True, tooltip="Optional input to control execution order"),
                io.String.Input("description", default="", optional=True, tooltip="Note to attach to saved version"),
            ],
            hidden=[io.Hidden.extra_pnginfo, io.Hidden.prompt],
            outputs=[
                io.String.Output(display_name="status"),
                io.String.Output(display_name="workflow_id"),
                io.Int.Output(display_name="version_count"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> str:
        """Always re-run to check for workflow changes."""
        return str(time.time())

    @classmethod
    def execute(cls, workflow_id: str, max_versions: int, save_on_any_change: bool,
                enabled: bool, trigger=None, description: str = "") -> io.NodeOutput:
        global _WORKFLOW_HASH_CACHE

        workflow_data = None
        if cls.hidden.extra_pnginfo and isinstance(cls.hidden.extra_pnginfo, dict):
            workflow_data = cls.hidden.extra_pnginfo.get('workflow')

        if not workflow_data:
            return io.NodeOutput("No workflow data available", workflow_id or "unknown", 0)

        effective_id = workflow_id.strip() if workflow_id else auto_workflow_id(workflow_data)

        if not enabled:
            versions = list_workflow_versions(effective_id)
            return io.NodeOutput("Disabled", effective_id, len(versions))

        if save_on_any_change:
            current_hash = hash_workflow_full(workflow_data)
        else:
            current_hash = hash_workflow_structure(workflow_data)

        cache_key = f"{effective_id}_{'full' if save_on_any_change else 'struct'}"
        last_hash = _WORKFLOW_HASH_CACHE.get(cache_key)

        existing_versions = list_workflow_versions(effective_id)

        if last_hash == current_hash and existing_versions:
            status = f"No changes (v{existing_versions[0]['version']} current)"
            return io.NodeOutput(status, effective_id, len(existing_versions))

        if existing_versions:
            latest_struct_hash = existing_versions[0].get('structure_hash', '')
            current_struct_hash = hash_workflow_structure(workflow_data)
            if latest_struct_hash == current_struct_hash[:8]:
                _WORKFLOW_HASH_CACHE[cache_key] = current_hash
                status = f"No changes (v{existing_versions[0]['version']} current)"
                return io.NodeOutput(status, effective_id, len(existing_versions))

        desc = description if description else "auto-saved"
        success, message, version_num = save_workflow_version(
            effective_id, workflow_data, desc, max_versions
        )

        if success:
            _WORKFLOW_HASH_CACHE[cache_key] = current_hash
            status = f"Saved v{version_num}"
            print(f"[BD Workflow Version] {status} for '{effective_id}'")
        else:
            status = f"Error: {message}"
            print(f"[BD Workflow Version] {status}")

        updated_versions = list_workflow_versions(effective_id)
        return io.NodeOutput(status, effective_id, len(updated_versions))


class BD_WorkflowVersionList(io.ComfyNode):
    """List all saved versions for a workflow with metadata."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_WorkflowVersionList",
            display_name="BD Workflow Version List",
            category="🧠BrainDead/Cache",
            description="List all saved versions for a workflow. Connect workflow_id from BD Workflow Version Cache.",
            inputs=[
                io.String.Input("workflow_id", default=""),
                io.Boolean.Input("show_hashes", default=True),
                io.Int.Input("max_display", default=20, min=1, max=100),
                io.Int.Input("compare_version_a", default=0, min=0, max=9999, optional=True),
                io.Int.Input("compare_version_b", default=0, min=0, max=9999, optional=True),
            ],
            outputs=[
                io.String.Output(display_name="version_list"),
                io.String.Output(display_name="diff_result"),
                io.Int.Output(display_name="total_versions"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> str:
        """Always re-run to show current versions."""
        return str(time.time())

    @classmethod
    def execute(cls, workflow_id: str, show_hashes: bool = True, max_display: int = 20,
                compare_version_a: int = 0, compare_version_b: int = 0) -> io.NodeOutput:
        if not workflow_id or not workflow_id.strip():
            return io.NodeOutput("Error: workflow_id is required.", "", 0)

        workflow_id = workflow_id.strip()
        versions = list_workflow_versions(workflow_id)

        if not versions:
            return io.NodeOutput(f"No versions found for '{workflow_id}'", "", 0)

        lines = [f"Workflow: {workflow_id}", f"Total versions: {len(versions)}", ""]
        lines.append("Version | Timestamp           | Nodes | Hash")
        lines.append("-" * 50)

        for v in versions[:max_display]:
            timestamp = v['timestamp'][:19] if v['timestamp'] else 'Unknown'
            hash_str = f" | {v['workflow_hash']}" if show_hashes else ""
            desc_str = f" ({v['description']})" if v['description'] and v['description'] != 'auto-saved' else ""
            lines.append(f"v{v['version']:4d}   | {timestamp} | {v['node_count']:5d}{hash_str}{desc_str}")

        if len(versions) > max_display:
            lines.append(f"... and {len(versions) - max_display} more versions")

        version_list = "\n".join(lines)

        diff_result = ""
        if compare_version_a > 0 and compare_version_b > 0:
            diff = compare_workflow_versions(workflow_id, compare_version_a, compare_version_b)
            if 'error' in diff:
                diff_result = diff['error']
            else:
                diff_result = f"Comparing v{compare_version_a} -> v{compare_version_b}:\n{diff['summary']}"

        return io.NodeOutput(version_list, diff_result, len(versions))


class BD_WorkflowVersionRestore(io.ComfyNode):
    """Restore a specific workflow version and output as JSON."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_WorkflowVersionRestore",
            display_name="BD Workflow Version Restore",
            category="🧠BrainDead/Cache",
            description="Restore a saved workflow version. Drag exported JSON into ComfyUI to restore.",
            is_output_node=True,
            inputs=[
                io.String.Input("workflow_id", default=""),
                io.Int.Input("version_number", default=0, min=0, max=9999, tooltip="Which version to restore (0 = latest)"),
                io.Boolean.Input("save_to_file", default=True, tooltip="Export to output/ folder for download"),
                io.String.Input("output_filename", default="", optional=True, tooltip="Custom filename (optional)"),
            ],
            outputs=[
                io.String.Output(display_name="workflow_json"),
                io.String.Output(display_name="file_path"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> str:
        """Always re-run to allow restoration."""
        return str(time.time())

    @classmethod
    def execute(cls, workflow_id: str, version_number: int, save_to_file: bool,
                output_filename: str = "") -> io.NodeOutput:
        if not workflow_id or not workflow_id.strip():
            return io.NodeOutput("", "", "Error: workflow_id is required.")

        workflow_id = workflow_id.strip()
        versions = list_workflow_versions(workflow_id)

        if not versions:
            return io.NodeOutput("", "", f"No versions found for '{workflow_id}'")

        if version_number == 0:
            target_version = versions[0]['version']
        else:
            target_version = version_number

        workflow_data, metadata = load_workflow_version(workflow_id, target_version)

        if workflow_data is None:
            return io.NodeOutput("", "", f"Error: {metadata}")

        workflow_json = json.dumps(workflow_data, indent=2)

        file_path = ""
        if save_to_file:
            if output_filename:
                filename = output_filename
                if not filename.endswith('.json'):
                    filename += '.json'
            else:
                filename = f"{workflow_id}_v{target_version}.json"

            file_path = os.path.join(OUTPUT_DIR, filename)

            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(workflow_json)
            except Exception as e:
                return io.NodeOutput(workflow_json, "", f"Restored v{target_version} but failed to save file: {e}")

        timestamp = metadata.get('timestamp', 'Unknown')[:19]
        status = f"Restored v{target_version} ({timestamp}, {metadata.get('node_count', 0)} nodes)"

        if file_path:
            status += f" - Saved to {os.path.basename(file_path)}"

        return io.NodeOutput(workflow_json, file_path, status)


class BD_WorkflowVersionClear(io.ComfyNode):
    """Clear saved workflow versions."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_WorkflowVersionClear",
            display_name="BD Workflow Version Clear",
            category="🧠BrainDead/Cache",
            description="Delete saved workflow versions to free disk space.",
            is_output_node=True,
            inputs=[
                io.String.Input("workflow_id", default=""),
                io.Int.Input("keep_latest", default=0, min=0, max=999, tooltip="Keep N most recent versions (0 = delete all)"),
                io.Boolean.Input("confirm_clear", default=False, tooltip="Must be True to actually delete"),
            ],
            outputs=[
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, workflow_id: str, keep_latest: int, confirm_clear: bool) -> io.NodeOutput:
        global _WORKFLOW_HASH_CACHE

        if not workflow_id or not workflow_id.strip():
            return io.NodeOutput("Error: workflow_id is required.")

        workflow_id = workflow_id.strip()
        versions = list_workflow_versions(workflow_id)

        if not versions:
            return io.NodeOutput(f"No versions found for '{workflow_id}'")

        if not confirm_clear:
            to_delete = len(versions) - keep_latest if keep_latest > 0 else len(versions)
            to_delete = max(0, to_delete)
            return io.NodeOutput(f"Would delete {to_delete} of {len(versions)} versions. Set confirm_clear=True to proceed.")

        deleted, message = clear_workflow_versions(workflow_id, keep_latest)

        for key in list(_WORKFLOW_HASH_CACHE.keys()):
            if key.startswith(workflow_id):
                del _WORKFLOW_HASH_CACHE[key]

        return io.NodeOutput(message)


# V3 node list for extension
WORKFLOW_V3_NODES = [
    BD_WorkflowVersionCache,
    BD_WorkflowVersionList,
    BD_WorkflowVersionRestore,
    BD_WorkflowVersionClear,
]

# V1 compatibility - NODE_CLASS_MAPPINGS dict
WORKFLOW_NODES = {
    "BD_WorkflowVersionCache": BD_WorkflowVersionCache,
    "BD_WorkflowVersionList": BD_WorkflowVersionList,
    "BD_WorkflowVersionRestore": BD_WorkflowVersionRestore,
    "BD_WorkflowVersionClear": BD_WorkflowVersionClear,
}

WORKFLOW_DISPLAY_NAMES = {
    "BD_WorkflowVersionCache": "BD Workflow Version Cache",
    "BD_WorkflowVersionList": "BD Workflow Version List",
    "BD_WorkflowVersionRestore": "BD Workflow Version Restore",
    "BD_WorkflowVersionClear": "BD Workflow Version Clear",
}
