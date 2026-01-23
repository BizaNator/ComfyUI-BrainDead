/**
 * BrainDead Notifications - Toast notifications for mesh processing nodes
 */

import { app } from "../../scripts/app.js";

const BD_NODES = [
    "BD_UVUnwrap",
    "BD_CuMeshSimplify",
    "BD_BlenderDecimate",
    "BD_BlenderMergePlanes",
    "BD_SampleVoxelgridColors",
    "BD_ApplyColorField",
];

// Track warnings we've already shown to avoid spam
const shownWarnings = new Set();

app.registerExtension({
    name: "BrainDead.Notifications",

    async setup() {
        // Listen for execution results
        const originalHandleExecuted = app.api.dispatchEvent?.bind(app.api) || (() => {});

        // Hook into api events
        app.api.addEventListener("executed", (event) => {
            const { node, output } = event.detail || {};
            if (!node || !output) return;

            // Check if this is a BrainDead node
            const nodeType = app.graph?.getNodeById(node)?.type;
            if (!nodeType || !BD_NODES.some(n => nodeType.includes(n))) return;

            // Look for status output (usually last string output)
            const status = output.status?.[0] || output.text?.[0];
            if (!status || typeof status !== 'string') return;

            // Check for fallback message
            if (status.includes("falling back to Blender") || status.includes("fallback")) {
                showToast("info", "UV Unwrap Fallback",
                    "xatlas failed on non-manifold mesh. Using Blender Smart UV instead.", 5000);
            }

            // Check for errors
            if (status.startsWith("ERROR:")) {
                showToast("error", "BrainDead Error", status, 8000);
            }
        });

        // Hook into node execution start for large mesh warnings
        app.api.addEventListener("executing", (event) => {
            const nodeId = event.detail?.node;
            if (!nodeId) return;

            const node = app.graph?.getNodeById(nodeId);
            if (!node) return;

            // Check if this is a UV unwrap or decimate node
            if (node.type === "BD_UVUnwrap" || node.type === "BD_BlenderDecimate") {
                // Try to get input mesh info from connected node
                checkLargeMeshWarning(node);
            }
        });

        console.log("[BrainDead] Notifications extension loaded");
    },

    async nodeCreated(node) {
        // Add warning tooltip for certain nodes
        if (node.type === "BD_UVUnwrap") {
            const origOnExecute = node.onExecute;
            node.onExecute = function() {
                // Check method setting
                const methodWidget = this.widgets?.find(w => w.name === "method");
                if (methodWidget?.value === "xatlas_gpu") {
                    // Could show a warning here if we detect face-split mesh input
                }
                if (origOnExecute) origOnExecute.apply(this, arguments);
            };
        }
    }
});

function showToast(severity, title, message, duration = 5000) {
    // Create unique key to avoid duplicate toasts
    const key = `${title}:${message}`;
    if (shownWarnings.has(key)) return;
    shownWarnings.add(key);

    // Clear after a while to allow showing again later
    setTimeout(() => shownWarnings.delete(key), 30000);

    // Use ComfyUI's toast API if available
    if (app.extensionManager?.toast?.add) {
        app.extensionManager.toast.add({
            severity: severity, // "info", "warn", "error", "success"
            summary: title,
            detail: message,
            life: duration,
        });
    } else if (app.ui?.dialog?.show) {
        // Fallback to dialog
        console.log(`[BrainDead ${severity}] ${title}: ${message}`);
    } else {
        // Console fallback
        const logFn = severity === "error" ? console.error :
                      severity === "warn" ? console.warn : console.info;
        logFn(`[BrainDead] ${title}: ${message}`);
    }
}

function checkLargeMeshWarning(node) {
    // Try to find connected mesh input
    const meshInput = node.inputs?.find(i => i.name === "mesh");
    if (!meshInput || !meshInput.link) return;

    const link = app.graph?.links?.[meshInput.link];
    if (!link) return;

    const sourceNode = app.graph?.getNodeById(link.origin_id);
    if (!sourceNode) return;

    // Check if source node has face count info in title or widgets
    const titleMatch = sourceNode.title?.match(/(\d+(?:,\d+)*)\s*faces/i);
    if (titleMatch) {
        const faceCount = parseInt(titleMatch[1].replace(/,/g, ''));
        if (faceCount > 1000000) {
            showToast("warn", "Large Mesh Warning",
                `Processing ${faceCount.toLocaleString()} faces may take a very long time. Consider decimating first.`,
                10000);
        }
    }
}
