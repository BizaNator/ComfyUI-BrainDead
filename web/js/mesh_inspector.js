/**
 * BD Mesh Inspector - ComfyUI Extension
 * Registers the interactive Three.js mesh viewer widget for BD_MeshInspector node.
 */

import { app } from "../../../scripts/app.js";

// Auto-detect extension folder name from current script URL
const EXTENSION_FOLDER = (() => {
    const url = import.meta.url;
    const match = url.match(/\/extensions\/([^/]+)\//);
    return match ? match[1] : "ComfyUI-BrainDead";
})();

app.registerExtension({
    name: "braindead.meshinspector",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "BD_MeshInspector") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // Create iframe for 3D viewer
            const iframe = document.createElement("iframe");
            iframe.style.width = "100%";
            iframe.style.height = "100%";
            iframe.style.border = "none";
            iframe.style.backgroundColor = "#1a1a1a";
            iframe.style.borderRadius = "4px";
            iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer_inspector.html?v=${Date.now()}`;

            // Add DOM widget
            const widget = this.addDOMWidget("inspector", "MESH_INSPECTOR", iframe, {
                getValue() { return ""; },
                setValue(v) { },
            });

            // Widget sizing - 4:3 aspect ratio
            widget.computeSize = function (width) {
                const w = width || 512;
                return [w, Math.round(w * 0.75)];
            };

            widget.element = iframe;
            this.meshInspectorIframe = iframe;
            this.setSize([520, 460]);

            // Handle execution results
            const onExecuted = this.onExecuted;
            this.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                if (!message?.mesh_file || !message.mesh_file[0]) return;

                const filename = message.mesh_file[0];
                const filepath = `/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=`;

                // Wait for iframe to be ready, then send data
                const sendData = () => {
                    if (!iframe.contentWindow) return;
                    iframe.contentWindow.postMessage({
                        type: "LOAD_INSPECTOR_MESH",
                        filepath: filepath,
                        initial_mode: message.initial_mode?.[0] || "full_material",
                        metallic_json: message.metallic_json?.[0] || "",
                        roughness_json: message.roughness_json?.[0] || "",
                        metallic_map_b64: message.metallic_map_b64?.[0] || "",
                        roughness_map_b64: message.roughness_map_b64?.[0] || "",
                        normal_map_b64: message.normal_map_b64?.[0] || "",
                        emissive_map_b64: message.emissive_map_b64?.[0] || "",
                        alpha_map_b64: message.alpha_map_b64?.[0] || "",
                        diffuse_map_b64: message.diffuse_map_b64?.[0] || "",
                        vertex_colors_b64: message.vertex_colors_b64?.[0] || "",
                        has_uvs: message.has_uvs?.[0] || false,
                        has_colors: message.has_colors?.[0] || false,
                        timestamp: Date.now(),
                    }, "*");
                };

                // Retry sending until iframe is loaded
                setTimeout(sendData, 200);
            };

            return r;
        };
    },
});
