"""BD_Provenance: what the run knows has to be collected, not typed in.

The point of the block is that nobody maintains it by hand, so the collectors
carry the whole weight. Each one has a way of being quietly wrong:

  * seeds read from the WIDGET would record the value before the frontend
    randomised it -- the API prompt is the only place the seed that actually ran
    appears.
  * models detected by a list of known loader class names stop working the day
    someone installs a pack this one has never heard of. Detection is by value.
  * a model filename cannot be turned into a download URL by guessing, so an
    unlisted model gets no link rather than a plausible wrong one.
  * empty widgets must not land in the block, or every record carries
    `"studio": ""` and the file looks filled in.

The module imports `comfy_api.latest`, which needs the live ComfyUI env, so the
module body is AST-extracted minus that import and the node class -- the same
method as test_workflow_sidecar_payload.
"""
import ast
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "nodes" / "cache" / "provenance.py"


def _module():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    drop = ("PROVENANCE_V3_NODES", "PROVENANCE_NODES", "PROVENANCE_DISPLAY_NAMES")

    def keep(n):
        if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("comfy_api"):
            return False
        if isinstance(n, ast.ClassDef):
            return False
        # the export table names the class, which is exactly what was dropped
        if isinstance(n, ast.Assign):
            names = [t.id for t in n.targets if isinstance(t, ast.Name)]
            if any(x in drop for x in names):
                return False
        return True

    body = [n for n in tree.body if keep(n)]
    ns = {"__file__": str(MODULE)}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(MODULE), "exec"), ns)
    return ns


M = _module()

PROMPT = {
    "3": {"class_type": "KSampler",
          "inputs": {"seed": 12345, "steps": 20, "cfg": 7.5}},
    "4": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
    "7": {"class_type": "SamplerCustom",
          "inputs": {"noise_seed": 999, "denoise": 1.0}},
    "9": {"class_type": "LoraLoader",
          "inputs": {"lora_name": "styles/filmgrain.safetensors", "strength": 0.6}},
    "11": {"class_type": "CLIPTextEncode",
           "inputs": {"text": "a seed of doubt", "clip": ["4", 1]}},
}


class Seeds(unittest.TestCase):

    def test_both_spellings_are_collected(self):
        got = {(s["node"], s["value"]) for s in M["collect_seeds"](PROMPT)}

        self.assertEqual(got, {("3", 12345), ("7", 999)})

    def test_a_word_that_merely_contains_seed_is_not_one(self):
        # "a seed of doubt" is a prompt, and `steps` is not a seed either.
        values = [s["input"] for s in M["collect_seeds"](PROMPT)]

        self.assertEqual(sorted(values), ["noise_seed", "seed"])

    def test_junk_does_not_raise(self):
        self.assertEqual(M["collect_seeds"](None), [])
        self.assertEqual(M["collect_seeds"]({"1": "not a node"}), [])


class Models(unittest.TestCase):

    def test_detection_is_by_value_not_by_loader_name(self):
        names = [m["name"] for m in M["collect_models"](PROMPT)]

        # LoraLoader is not in any hardcoded list -- the .safetensors is.
        self.assertIn("sd_xl_base_1.0.safetensors", names)
        self.assertIn("styles/filmgrain.safetensors", names)

    def test_the_prompt_text_is_not_a_model(self):
        names = [m["name"] for m in M["collect_models"](PROMPT)]

        self.assertNotIn("a seed of doubt", names)

    def test_the_same_file_twice_is_recorded_once(self):
        twice = dict(PROMPT)
        twice["12"] = {"class_type": "CheckpointLoaderSimple",
                       "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}}

        names = [m["name"] for m in M["collect_models"](twice)]

        self.assertEqual(names.count("sd_xl_base_1.0.safetensors"), 1)

    def test_an_unlisted_model_gets_no_invented_url(self):
        for m in M["collect_models"](PROMPT):
            self.assertNotIn("url", m)


class Block(unittest.TestCase):

    def test_an_empty_widget_does_not_land_in_the_record(self):
        block = M["build_provenance"](PROMPT, {"creator": "", "studio": "  ",
                                               "reference": "Bill Page ref sheet"},
                                      include_models=False, include_addons=False)

        self.assertNotIn("creator", block)
        self.assertNotIn("studio", block)
        self.assertEqual(block["reference"], "Bill Page ref sheet")

    def test_the_block_carries_a_start_for_the_save_to_close(self):
        block = M["build_provenance"](PROMPT, {}, include_models=False,
                                      include_addons=False)

        self.assertEqual(block["schema"], "bd_provenance/1")
        self.assertGreater(block["started_at"], 0)
        # duration is the SAVE's to fill -- the node runs first by design.
        self.assertNotIn("duration_sec", block)

    def test_the_summary_survives_a_nearly_empty_block(self):
        text = M["_summary"]({"created": "x", "host": "y"})

        self.assertIn("x", text)


if __name__ == "__main__":
    unittest.main()
