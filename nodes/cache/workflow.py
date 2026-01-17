"""
Workflow versioning nodes for automatic backup and recovery.

BD_WorkflowVersionCache - Auto-save workflow versions on change
BD_WorkflowVersionList - List saved versions with metadata
BD_WorkflowVersionRestore - Restore a specific version
BD_WorkflowVersionClear - Delete old versions
"""

import os
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


class BD_WorkflowVersionCache:
    """
    Automatically save workflow versions when changes are detected.

    USAGE:
    1. Add this node anywhere in your workflow
    2. Set a workflow_id or leave empty for auto-generated ID based on structure
    3. Configure max_versions to control how many versions are kept
    4. Workflow is auto-saved on first run and whenever changes are detected
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_id": ("STRING", {"default": "", "placeholder": "Leave empty for auto-detect"}),
                "max_versions": ("INT", {"default": 50, "min": 0, "max": 999}),
                "save_on_any_change": ("BOOLEAN", {"default": True}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "trigger": ("*",),
                "description": ("STRING", {"default": "", "multiline": False}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT",
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("status", "workflow_id", "version_count")
    FUNCTION = "cache_workflow_version"
    CATEGORY = "BrainDead/Cache"
    OUTPUT_NODE = True
    DESCRIPTION = """
Automatically save workflow versions for backup and recovery.

RECOMMENDED: Set a manual workflow_id (e.g., "my_project") to track
your workflow's evolution over time.

Inputs:
- workflow_id: Name for this workflow (leave empty for auto-detect)
- max_versions: How many versions to keep (0 = unlimited)
- save_on_any_change: Currently uses structure detection
- enabled: Toggle versioning on/off
- trigger: Optional input to control execution order
- description: Note to attach to saved version

Storage: output/BrainDead_Cache/workflow_versions/
"""

    @classmethod
    def IS_CHANGED(cls, workflow_id, max_versions, save_on_any_change, enabled,
                   trigger=None, description="", extra_pnginfo=None, prompt=None):
        import time
        return time.time()

    def cache_workflow_version(self, workflow_id, max_versions, save_on_any_change, enabled,
                                trigger=None, description="", extra_pnginfo=None, prompt=None):
        workflow_data = None
        if extra_pnginfo and isinstance(extra_pnginfo, dict):
            workflow_data = extra_pnginfo.get('workflow')

        if not workflow_data:
            return ("No workflow data available", workflow_id or "unknown", 0)

        effective_id = workflow_id.strip() if workflow_id else auto_workflow_id(workflow_data)

        if not enabled:
            versions = list_workflow_versions(effective_id)
            return ("Disabled", effective_id, len(versions))

        if save_on_any_change:
            current_hash = hash_workflow_full(workflow_data)
        else:
            current_hash = hash_workflow_structure(workflow_data)

        cache_key = f"{effective_id}_{'full' if save_on_any_change else 'struct'}"
        last_hash = _WORKFLOW_HASH_CACHE.get(cache_key)

        existing_versions = list_workflow_versions(effective_id)

        if last_hash == current_hash and existing_versions:
            status = f"No changes (v{existing_versions[0]['version']} current)"
            return (status, effective_id, len(existing_versions))

        if existing_versions:
            latest_struct_hash = existing_versions[0].get('structure_hash', '')
            current_struct_hash = hash_workflow_structure(workflow_data)
            if latest_struct_hash == current_struct_hash[:8]:
                _WORKFLOW_HASH_CACHE[cache_key] = current_hash
                status = f"No changes (v{existing_versions[0]['version']} current)"
                return (status, effective_id, len(existing_versions))

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
        return (status, effective_id, len(updated_versions))


class BD_WorkflowVersionList:
    """List all saved versions for a workflow with metadata."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_id": ("STRING", {"default": ""}),
                "show_hashes": ("BOOLEAN", {"default": True}),
                "max_display": ("INT", {"default": 20, "min": 1, "max": 100}),
            },
            "optional": {
                "compare_version_a": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "compare_version_b": ("INT", {"default": 0, "min": 0, "max": 9999}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("version_list", "diff_result", "total_versions")
    FUNCTION = "list_versions"
    CATEGORY = "BrainDead/Cache"
    DESCRIPTION = """
List all saved versions for a workflow.

Inputs:
- workflow_id: Connect from BD Workflow Version Cache output
- show_hashes: Include hash in output table
- max_display: Limit number of versions shown
- compare_version_a/b: Compare two versions (shows diff)
"""

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import time
        return time.time()

    def list_versions(self, workflow_id, show_hashes=True, max_display=20,
                      compare_version_a=0, compare_version_b=0):
        if not workflow_id or not workflow_id.strip():
            return ("Error: workflow_id is required.", "", 0)

        workflow_id = workflow_id.strip()
        versions = list_workflow_versions(workflow_id)

        if not versions:
            return (f"No versions found for '{workflow_id}'", "", 0)

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

        return (version_list, diff_result, len(versions))


class BD_WorkflowVersionRestore:
    """Restore a specific workflow version and output as JSON."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_id": ("STRING", {"default": ""}),
                "version_number": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "save_to_file": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "output_filename": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("workflow_json", "file_path", "status")
    FUNCTION = "restore_version"
    CATEGORY = "BrainDead/Cache"
    OUTPUT_NODE = True
    DESCRIPTION = """
Restore a saved workflow version.

Inputs:
- workflow_id: Connect from BD Workflow Version Cache output
- version_number: Which version to restore (0 = latest)
- save_to_file: Export to output/ folder for download
- output_filename: Custom filename (optional)

To restore: Drag the exported JSON file into ComfyUI.
"""

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import time
        return time.time()

    def restore_version(self, workflow_id, version_number, save_to_file, output_filename=""):
        import json

        if not workflow_id or not workflow_id.strip():
            return ("", "", "Error: workflow_id is required.")

        workflow_id = workflow_id.strip()
        versions = list_workflow_versions(workflow_id)

        if not versions:
            return ("", "", f"No versions found for '{workflow_id}'")

        if version_number == 0:
            target_version = versions[0]['version']
        else:
            target_version = version_number

        workflow_data, metadata = load_workflow_version(workflow_id, target_version)

        if workflow_data is None:
            return ("", "", f"Error: {metadata}")

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
                return (workflow_json, "", f"Restored v{target_version} but failed to save file: {e}")

        timestamp = metadata.get('timestamp', 'Unknown')[:19]
        status = f"Restored v{target_version} ({timestamp}, {metadata.get('node_count', 0)} nodes)"

        if file_path:
            status += f" - Saved to {os.path.basename(file_path)}"

        return (workflow_json, file_path, status)


class BD_WorkflowVersionClear:
    """Clear saved workflow versions."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_id": ("STRING", {"default": ""}),
                "keep_latest": ("INT", {"default": 0, "min": 0, "max": 999}),
                "confirm_clear": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "clear_versions"
    CATEGORY = "BrainDead/Cache"
    OUTPUT_NODE = True
    DESCRIPTION = """
Delete saved workflow versions to free disk space.

Inputs:
- workflow_id: Connect from BD Workflow Version Cache output
- keep_latest: Keep N most recent versions (0 = delete all)
- confirm_clear: Must be True to actually delete
"""

    def clear_versions(self, workflow_id, keep_latest, confirm_clear):
        if not workflow_id or not workflow_id.strip():
            return ("Error: workflow_id is required.",)

        workflow_id = workflow_id.strip()
        versions = list_workflow_versions(workflow_id)

        if not versions:
            return (f"No versions found for '{workflow_id}'",)

        if not confirm_clear:
            to_delete = len(versions) - keep_latest if keep_latest > 0 else len(versions)
            to_delete = max(0, to_delete)
            return (f"Would delete {to_delete} of {len(versions)} versions. Set confirm_clear=True to proceed.",)

        deleted, message = clear_workflow_versions(workflow_id, keep_latest)

        for key in list(_WORKFLOW_HASH_CACHE.keys()):
            if key.startswith(workflow_id):
                del _WORKFLOW_HASH_CACHE[key]

        return (message,)


# Node exports
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
