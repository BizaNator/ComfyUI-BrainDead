/**
 * BD Preview 3D - ComfyUI Extension
 * Reuses the BD_MeshInspector three.js iframe viewer for the BD_Preview3D node.
 */

import { app } from "../../../scripts/app.js";

const EXTENSION_FOLDER = (() => {
    const url = import.meta.url;
    const match = url.match(/\/extensions\/([^/]+)\//);
    return match ? match[1] : "ComfyUI-BrainDead";
})();

app.registerExtension({
    name: "braindead.preview3d",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "BD_Preview3D") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const iframe = document.createElement("iframe");
            iframe.style.width = "100%";
            iframe.style.height = "100%";
            iframe.style.border = "none";
            iframe.style.backgroundColor = "#1a1a1a";
            iframe.style.borderRadius = "4px";
            iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer_inspector.html?v=${Date.now()}`;

            const widget = this.addDOMWidget("preview3d", "BD_PREVIEW3D", iframe, {
                getValue() { return ""; },
                setValue(v) { },
            });

            widget.element = iframe;
            this.bdPreview3dIframe = iframe;
            this.setSize([520, 460]);

            widget.computeSize = function (width) {
                const w = width || 512;
                return [w, Math.round(w * 0.75)];
            };

            const node = this;
            const origOnResize = node.onResize;
            node.onResize = function (size) {
                origOnResize?.apply(this, arguments);
                if (!iframe.parentElement) return;
                const widgetY = widget.last_y || 0;
                if (widgetY <= 0) return;
                const targetH = Math.max(200, size[1] - widgetY - 6);
                iframe.parentElement.style.height = targetH + "px";
                iframe.parentElement.style.overflow = "hidden";
            };

            const onExecuted = this.onExecuted;
            this.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                if (!message?.mesh_file || !message.mesh_file[0]) return;

                const filename = message.mesh_file[0];
                const viewType = message.view_type?.[0] || "temp";
                const subfolder = message.subfolder?.[0] || "";
                const filepath = `/view?filename=${encodeURIComponent(filename)}&type=${viewType}&subfolder=${encodeURIComponent(subfolder)}`;

                const sendData = () => {
                    if (!iframe.contentWindow) return;
                    iframe.contentWindow.postMessage({
                        type: "LOAD_INSPECTOR_MESH",
                        filepath: filepath,
                        initial_mode: message.initial_mode?.[0] || "full_material",
                        has_uvs: message.has_uvs?.[0] || false,
                        has_colors: message.has_colors?.[0] || false,
                        timestamp: Date.now(),
                    }, "*");
                };
                setTimeout(sendData, 200);
            };

            return r;
        };
    },
});
