"""SBAI-11046: a sidecar has to open in ComfyUI, and say so when it cannot carry the graph.

The sidecar was `{"workflow": ..., "prompt": ...}`. ComfyUI`s loader accepts a UI
graph, or an API prompt, at the TOP level -- a wrapper object is neither, so
every sidecar written this way opened as an empty canvas.

Worse, `workflow` was usually null. ComfyUI populates extra_pnginfo.workflow only
for FRONTEND submissions; a headless caller has to attach the UI graph itself.
1,156 sidecars from the first corpus pass carry nothing at all.

No node can conjure a graph the server never received, so the contract is:

  * graph present -> write it at top level, prompt carried under `extra`
  * graph absent  -> write the API prompt at top level (opens as an API import)
                     and warn once, naming the cause

The module imports `comfy_api.latest`, which needs the live ComfyUI env, so the
function is AST-extracted -- same method as test_workflow_sidecar_paths.
"""
import ast
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "nodes" / "cache" / "workflow_sidecar.py"


def _extract(name):
    tree = ast.parse(MODULE.read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"os": os, "json": json}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(MODULE), "exec"), ns)
    return ns[name]


UI_GRAPH = {"id": "abc", "nodes": [{"id": 1, "type": "LoadImage"}], "links": [],
            "groups": [], "last_node_id": 1}
PROMPT = {"1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}}}


class SidecarPayload(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.png = os.path.join(self.dir, "out.png")
        self.write = _extract("write_workflow_sidecar")

    def _written(self, extra_pnginfo, prompt):
        buf = io.StringIO()
        with redirect_stdout(buf):
            path = self.write(self.png, extra_pnginfo, prompt)
        with open(path) as f:
            return json.load(f), buf.getvalue()

    def test_the_ui_graph_is_the_document(self):
        got, _ = self._written({"workflow": UI_GRAPH}, PROMPT)

        # Top level, not nested under a "workflow" key -- this is the whole point.
        self.assertEqual(got["nodes"], UI_GRAPH["nodes"])
        self.assertEqual(got["links"], UI_GRAPH["links"])
        self.assertNotIn("workflow", got)

    def test_the_api_prompt_rides_along_without_breaking_the_load(self):
        got, _ = self._written({"workflow": UI_GRAPH}, PROMPT)

        # ComfyUI ignores unknown keys under "extra" and round-trips them.
        self.assertEqual(got["extra"]["bd_api_prompt"], PROMPT)

    def test_an_empty_node_list_is_still_a_graph(self):
        # A real but empty canvas is not the headless-caller fault, and must not
        # be reported as one.
        got, warned = self._written({"workflow": {"nodes": [], "links": []}}, PROMPT)

        self.assertEqual(got["nodes"], [])
        self.assertEqual(warned, "")

    def test_a_missing_graph_falls_back_to_the_api_prompt(self):
        got, warned = self._written({}, PROMPT)

        # Complete and runnable, just not laid out -- better than an empty canvas.
        self.assertEqual(got, PROMPT)
        self.assertIn("extra_pnginfo", warned)

    def test_the_warning_names_the_fix_once(self):
        _, first = self._written({}, PROMPT)
        _, second = self._written({}, PROMPT)

        self.assertIn("extra_data", first)
        self.assertEqual(second, "")


if __name__ == "__main__":
    unittest.main()
