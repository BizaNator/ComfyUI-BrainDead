"""SBAI-11046: a workflow sidecar must never land on the file it belongs to.

`write_workflow_sidecar` derives `<filepath minus ext>.json`. Every caller until
BD_PartsExport passed an image, so that was always a new name. BD_PartsExport
passes its `_manifest.json`, and `.json` minus its extension plus `.json` is the
manifest itself — the sidecar was written and then immediately overwritten by
the manifest write that follows it. Silent: the export looked fine and no
sidecar ever appeared. With the writes ordered the other way it would have
destroyed the manifest instead.

The module imports `comfy_api.latest`, which needs the live ComfyUI env, so the
function is AST-extracted — same method as test_mouth_parts_mask_shapes.
"""
import ast
import json
import os
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "nodes" / "cache" / "workflow_sidecar.py"


def _extract(name):
    tree = ast.parse(MODULE.read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"os": os, "json": json}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(MODULE), "exec"), ns)
    return ns[name]


write_workflow_sidecar = _extract("write_workflow_sidecar")
GRAPH = {"workflow": {"nodes": [], "links": []}}
PROMPT = {"1": {"class_type": "X", "inputs": {}}}


class SidecarPath(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_a_json_caller_does_not_overwrite_its_own_file(self):
        manifest = os.path.join(self.dir, "_manifest.json")
        with open(manifest, "w") as f:
            json.dump({"parts": ["nose", "lips"]}, f)

        got = write_workflow_sidecar(manifest, GRAPH, PROMPT)

        self.assertNotEqual(os.path.abspath(got), os.path.abspath(manifest))
        self.assertEqual(os.path.basename(got), "_manifest_workflow.json")
        with open(manifest) as f:
            self.assertEqual(json.load(f), {"parts": ["nose", "lips"]})
        with open(got) as f:
            self.assertEqual(json.load(f)["nodes"], GRAPH["workflow"]["nodes"])

    def test_an_image_caller_is_unchanged(self):
        png = os.path.join(self.dir, "nose_nose.png")
        with open(png, "wb"):
            pass
        got = write_workflow_sidecar(png, GRAPH, PROMPT)
        self.assertEqual(os.path.basename(got), "nose_nose.json")

    def test_extensionless_caller_gains_json(self):
        p = os.path.join(self.dir, "bundle")
        got = write_workflow_sidecar(p, GRAPH, PROMPT)
        self.assertEqual(os.path.basename(got), "bundle.json")

    def test_nothing_to_write_returns_empty(self):
        self.assertEqual(write_workflow_sidecar(
            os.path.join(self.dir, "x.png"), None, None), "")
        self.assertEqual(write_workflow_sidecar("", GRAPH, PROMPT), "")


if __name__ == "__main__":
    unittest.main()
