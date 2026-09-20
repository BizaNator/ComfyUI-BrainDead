// Faithful UI-graph -> API-format export.
//
// Drives a real ComfyUI page and calls the ACTUAL window.app.graphToPrompt()
// export path (the same code "Save (API Format)" uses) instead of
// reimplementing widget-order/subgraph/reroute/bypass resolution in a second
// language. graphToPrompt only exists client-side (bundled in
// comfyui_frontend_package's JS, never ported to a standalone Python
// function), so this is the only way to get a byte-faithful conversion —
// see B:\Brains\Skills\ComfyUI\comfyui_api_runner.md for the existing
// run_workflow.py path, which covers the non-subgraph case and strips
// Reroute but has no subgraph handling (can't convert e.g. FaceMaker's 17
// subgraph definitions). This tool exists for that gap.
//
// OPTIONAL, not a pipeline dependency: requires node + the `playwright`
// npm package (`npm install playwright` — browser binaries auto-download or
// reuse an existing ~/.cache/ms-playwright install). Nothing in the BD node
// pack or the FaceMaker graph depends on this being present.
//
// Usage: node export_workflow_api.js <comfyui_base_url> <input_workflow.json> <output_api.json>
//
// Also usable as an upstream background-image source for
// BD_SaveWorkflowImage's `image` input (a real rendered canvas export,
// rather than the built-in drawn schematic) — that's a separate concern
// from this script, which only produces the API JSON.

const fs = require('fs');
const { chromium } = require('playwright');

async function main() {
  const [, , baseUrl, inputPath, outputPath] = process.argv;
  if (!baseUrl || !inputPath || !outputPath) {
    console.error('usage: node export_workflow_api.js <base_url> <input_workflow.json> <output_api.json>');
    process.exit(2);
  }

  const graphData = JSON.parse(fs.readFileSync(inputPath, 'utf8'));

  const browser = await chromium.launch();
  try {
    const page = await browser.newPage();
    const consoleErrors = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });
    page.on('pageerror', (err) => consoleErrors.push(String(err)));

    await page.goto(baseUrl, { waitUntil: 'networkidle', timeout: 60000 });

    // Wait for the app singleton + the exact method we need.
    await page.waitForFunction(
      () => typeof window.app !== 'undefined' && typeof window.app.graphToPrompt === 'function',
      { timeout: 60000 }
    );

    const result = await page.evaluate(async (graph) => {
      // clean=true, restore_view=true — matches how the frontend loads a workflow file.
      await window.app.loadGraphData(graph, true, true);
      // Let the graph settle (widget bindings, subgraph instancing) before
      // serializing — large graphs with many subgraph definitions need a
      // few seconds, not milliseconds. Verified deterministic across
      // repeated runs at 500ms and 4s settle: byte-identical output.
      await new Promise((r) => setTimeout(r, 4000));
      const converted = await window.app.graphToPrompt();
      return {
        output: converted.output,
        nodeCount: Object.keys(converted.output || {}).length,
      };
    }, graphData);

    if (!result || !result.output) {
      console.error('graphToPrompt() returned no output. Console errors:', consoleErrors.join('\n'));
      process.exit(1);
    }

    fs.writeFileSync(outputPath, JSON.stringify(result.output, null, 2));
    console.log(JSON.stringify({
      status: 'ok',
      api_node_count: result.nodeCount,
      ui_node_count: (graphData.nodes || []).length,
      console_errors: consoleErrors,
      output_path: outputPath,
    }));
  } finally {
    await browser.close();
  }
}

main().catch((e) => {
  console.error('FATAL:', e);
  process.exit(1);
});
