"""SBAI-11046: the UI-graph -> API converter wired links and widgets by position.

Two independent positional assumptions, both wrong, both silent:

1. A link's `target_slot` is the slot the link had AT SAVE TIME. A node that
   hides, collapses or reorders optional inputs no longer has that index in its
   `inputs` array. COB_PartsPrep_v01 saved BD_PartsBuilder's `mask_labels` link
   at slot 11 against a four-entry array: the write was dropped, `masks` never
   got wired, and `depth_image` received the IMAGE meant for `mask_labels`.
   BD_PartsExport's `base_image` link was dropped the same way, so the widget
   queue filled it with the boolean `True`.

2. `widgets_values` is positional, and a widget converted to an input KEEPS its
   slot. Consuming the queue only for unlinked inputs read every later widget one
   slot early; consuming it for every linked input instead over-reads, because
   BD_SaveFile's `data` is declared `["*"]` -- a real socket that no string test
   recognises as one.

Neither failure raises. The prompt is accepted and `status` comes back `success`
while the affected nodes error into the server log, so these have to be asserted
on the converted prompt rather than on a run.

The converter only needs stdlib, so it is imported directly; `object_info` is a
fixture here, not a live server.
"""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "workflow_to_api.py"


def _load():
    spec = importlib.util.spec_from_file_location("bd_workflow_to_api", TOOL)
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [str(TOOL)]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


sg = _load()


def _node(nid, ntype, inputs, widgets=None, outputs=None):
    return {
        "id": nid,
        "type": ntype,
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs or [{"name": "out", "type": "IMAGE", "links": []}],
        "widgets_values": widgets if widgets is not None else [],
    }


def _sock(name, type_, link=None):
    return {"name": name, "type": type_, "link": link}


def _widget(name, type_, link=None):
    return {"name": name, "type": type_, "widget": {"name": name}, "link": link}


def _convert(workflow, object_info):
    flat = sg.FlatGraph()
    sg.flatten(workflow, (workflow.get("definitions") or {}).get("subgraphs", []),
               flat, prefix="")
    return sg.flat_to_api(flat, object_info)


class StaleTargetSlot(unittest.TestCase):
    """A link recorded against a slot index the node no longer has."""

    OBJECT_INFO = {
        "Source": {"input": {"required": {}}, "output": ["IMAGE"]},
        "Labels": {"input": {"required": {}}, "output": ["IMAGE", "STRING"]},
        "Masker": {"input": {"required": {}}, "output": ["MASK"]},
        "Builder": {"input": {"required": {
            "image": ["IMAGE", {}],
            "masks": ["MASK", {}],
            "depth_image": ["IMAGE", {}],
            "mask_labels": ["STRING", {}],
        }}, "output": ["PARTS"]},
    }

    def _graph(self):
        # Builder's array is four entries; the STRING link was saved at slot 11,
        # which is how COB_PartsPrep_v01 stores it.
        return {
            "nodes": [
                _node(1, "Source", []),
                _node(2, "Labels", [], outputs=[
                    {"name": "img", "type": "IMAGE", "links": []},
                    {"name": "labels", "type": "STRING", "links": []}]),
                _node(3, "Masker", []),
                _node(4, "Builder", [
                    _sock("image", "IMAGE", 10),
                    _sock("masks", "MASK", 13),
                    _sock("depth_image", "IMAGE", 11),
                    _sock("mask_labels", "STRING", 12),
                ]),
            ],
            "links": [
                [10, 1, 0, 4, 0, "IMAGE"],
                [11, 2, 0, 4, 3, "IMAGE"],    # recorded slot 3, really depth_image
                [12, 2, 1, 4, 11, "STRING"],  # recorded slot 11, out of range
                [13, 3, 0, 4, 2, "MASK"],     # recorded slot 2, really masks
            ],
        }

    def test_every_input_is_wired_by_name(self):
        api = _convert(self._graph(), self.OBJECT_INFO)
        self.assertEqual(api["4"]["inputs"], {
            "image": ["1", 0],
            "masks": ["3", 0],
            "depth_image": ["2", 0],
            "mask_labels": ["2", 1],
        })

    def test_no_input_is_dropped(self):
        api = _convert(self._graph(), self.OBJECT_INFO)
        # The out-of-range slot used to be silently discarded, leaving the node
        # to fail validation at queue time with the prompt reporting success.
        self.assertIn("mask_labels", api["4"]["inputs"])

    def test_types_agree_after_wiring(self):
        api = _convert(self._graph(), self.OBJECT_INFO)
        outs = {"1": ["IMAGE"], "2": ["IMAGE", "STRING"], "3": ["MASK"]}
        want = {"image": "IMAGE", "masks": "MASK",
                "depth_image": "IMAGE", "mask_labels": "STRING"}
        for name, expect in want.items():
            src, slot = api["4"]["inputs"][name]
            self.assertEqual(outs[src][slot], expect, name)


class WidgetConvertedToInput(unittest.TestCase):
    """widgets_values keeps a slot for a widget that has been linked."""

    OBJECT_INFO = {
        "Text": {"input": {"required": {}}, "output": ["STRING"]},
        "Seg": {"input": {
            "required": {"image": ["IMAGE", {}], "prompts": ["STRING", {}]},
            "optional": {"negative": ["STRING", {}], "mode": [["union", "isect"], {}],
                         "threshold": ["FLOAT", {}]},
        }, "output": ["MASK"]},
    }

    def _graph(self, prompts_link):
        return {
            "nodes": [
                _node(1, "Text", [], outputs=[{"name": "s", "type": "STRING", "links": []}]),
                _node(2, "Seg", [
                    _sock("image", "IMAGE", None),
                    _widget("prompts", "STRING", prompts_link),
                    _widget("negative", "STRING"),
                    _widget("mode", "COMBO"),
                    _widget("threshold", "FLOAT"),
                ], widgets=["person", "", "union", 0.5]),
            ],
            "links": ([[20, 1, 0, 2, 1, "STRING"]] if prompts_link else []),
        }

    def test_later_widgets_are_not_shifted(self):
        api = _convert(self._graph(prompts_link=20), self.OBJECT_INFO)
        self.assertEqual(api["2"]["inputs"]["prompts"], ["1", 0])
        self.assertEqual(api["2"]["inputs"]["negative"], "")
        self.assertEqual(api["2"]["inputs"]["mode"], "union")
        self.assertEqual(api["2"]["inputs"]["threshold"], 0.5)

    def test_unlinked_case_is_unchanged(self):
        api = _convert(self._graph(prompts_link=None), self.OBJECT_INFO)
        self.assertEqual(api["2"]["inputs"]["prompts"], "person")
        self.assertEqual(api["2"]["inputs"]["mode"], "union")


class WildcardSocketKeepsNoWidgetSlot(unittest.TestCase):
    """A linked `["*"]` input is a socket and never consumed a widget slot."""

    OBJECT_INFO = {
        "Any": {"input": {"required": {}}, "output": ["IMAGE"]},
        "Save": {"input": {
            "required": {"data": ["*", {}], "filename": ["STRING", {}],
                         "skip_if_exists": ["BOOLEAN", {}]},
            "optional": {"name_prefix": ["STRING", {}], "suffix": ["STRING", {}]},
        }, "output": []},
    }

    def test_first_widget_is_not_eaten_by_the_socket(self):
        wf = {
            "nodes": [
                _node(1, "Any", []),
                _node(2, "Save", [
                    _sock("data", "*", 30),
                    _widget("filename", "STRING"),
                    _widget("skip_if_exists", "BOOLEAN"),
                    _widget("name_prefix", "STRING"),
                    _widget("suffix", "STRING"),
                ], widgets=["", True, "overlay", "whitelines"]),
            ],
            "links": [[30, 1, 0, 2, 0, "*"]],
        }
        api = _convert(wf, self.OBJECT_INFO)
        self.assertEqual(api["2"]["inputs"]["data"], ["1", 0])
        self.assertEqual(api["2"]["inputs"]["filename"], "")
        self.assertEqual(api["2"]["inputs"]["skip_if_exists"], True)
        self.assertEqual(api["2"]["inputs"]["name_prefix"], "overlay")
        self.assertEqual(api["2"]["inputs"]["suffix"], "whitelines")


class PartialInputArray(unittest.TestCase):
    """`inputs` lists only wired slots; widgets_values still holds every widget.

    BD-parts_pointclick saves BD_PartsExport with four entries -- three wired
    sockets and `filename`, marked `widget` because it is linked -- against
    sixteen widget values. Taking the widget order from that array finds one
    widget and drops fifteen values, so the order has to come from the schema.
    """

    OBJECT_INFO = {
        "Src": {"input": {"required": {}}, "output": ["PARTS", "STRING"]},
        "Export": {"input": {
            "required": {"parts": ["PARTS", {}], "filename": ["STRING", {}]},
            "optional": {"name_prefix": ["STRING", {}],
                         "auto_increment": ["BOOLEAN", {}],
                         "context_id": ["STRING", {}],
                         "save_depth": ["BOOLEAN", {}],
                         "composite_size": ["INT", {}],
                         "base_image": ["IMAGE", {}]},
        }, "output": []},
    }

    def test_every_widget_value_lands(self):
        wf = {
            "nodes": [
                _node(1, "Src", [], outputs=[
                    {"name": "p", "type": "PARTS", "links": []},
                    {"name": "s", "type": "STRING", "links": []}]),
                _node(2, "Export", [
                    _sock("parts", "PARTS", 40),
                    _sock("context_id", "STRING", 41),
                    _widget("filename", "STRING", 42),
                ], widgets=["parts", "", True, "", True, 6144]),
            ],
            "links": [[40, 1, 0, 2, 0, "PARTS"],
                      [41, 1, 1, 2, 1, "STRING"],
                      [42, 1, 1, 2, 2, "STRING"]],
        }
        api = _convert(wf, self.OBJECT_INFO)
        self.assertEqual(api["2"]["inputs"], {
            "parts": ["1", 0],
            "filename": ["1", 1],      # linked: its "parts" slot is discarded
            "name_prefix": "",
            "auto_increment": True,
            "context_id": ["1", 1],    # linked: its "" slot is discarded
            "save_depth": True,
            "composite_size": 6144,
        })


class SeedControlWidget(unittest.TestCase):
    """control_after_generate has no input entry but does occupy a value slot."""

    OBJECT_INFO = {
        "K": {"input": {"required": {
            "seed": ["INT", {}], "steps": ["INT", {}], "cfg": ["FLOAT", {}],
        }}, "output": []},
    }

    def test_control_value_is_discarded(self):
        wf = {"nodes": [_node(1, "K", [
            _widget("seed", "INT"),
            _widget("steps", "INT"),
            _widget("cfg", "FLOAT"),
        ], widgets=[12345, "randomize", 4, 1.0])], "links": []}
        api = _convert(wf, self.OBJECT_INFO)
        self.assertEqual(api["1"]["inputs"],
                         {"seed": 12345, "steps": 4, "cfg": 1.0})

    def test_absent_control_value_is_not_invented(self):
        wf = {"nodes": [_node(1, "K", [
            _widget("seed", "INT"),
            _widget("steps", "INT"),
            _widget("cfg", "FLOAT"),
        ], widgets=[12345, 4, 1.0])], "links": []}
        api = _convert(wf, self.OBJECT_INFO)
        self.assertEqual(api["1"]["inputs"],
                         {"seed": 12345, "steps": 4, "cfg": 1.0})


class DynamicComboSubWidget(unittest.TestCase):
    """A dynamic combo expands into sub-inputs the schema never declares."""

    OBJECT_INFO = {
        "R": {"input": {"required": {
            "input": ["COMFY_MATCHTYPE_V3", {}],
            "resize_type": ["COMFY_DYNAMICCOMBO_V3", {}],
            "scale_method": [["area", "nearest-exact"], {}],
        }}, "output": ["IMAGE"]},
    }

    def test_sub_widget_is_forwarded_and_siblings_stay_put(self):
        wf = {"nodes": [_node(1, "R", [
            _sock("input", "COMFY_MATCHTYPE_V3"),
            _widget("resize_type", "COMFY_DYNAMICCOMBO_V3"),
            _sock("resize_type.match", "IMAGE"),
            _widget("resize_type.crop", "COMBO"),
            _widget("scale_method", "COMBO"),
        ], widgets=["match size", "center", "area"])], "links": []}
        api = _convert(wf, self.OBJECT_INFO)
        self.assertEqual(api["1"]["inputs"]["resize_type"], "match size")
        self.assertEqual(api["1"]["inputs"]["resize_type.crop"], "center")
        self.assertEqual(api["1"]["inputs"]["scale_method"], "area")

    def test_frontend_only_widget_is_not_forwarded(self):
        # LoadImage's `upload` button has no schema entry and no dotted parent;
        # ComfyUI's own export leaves it out.
        oi = {"L": {"input": {"required": {"image": [["a.png"], {}]}}, "output": ["IMAGE"]}}
        wf = {"nodes": [_node(1, "L", [
            _widget("image", "COMBO"),
            _widget("upload", "IMAGEUPLOAD"),
        ], widgets=["a.png", "image"])], "links": []}
        api = _convert(wf, oi)
        self.assertEqual(api["1"]["inputs"], {"image": "a.png"})


class BDSAM3MultiPromptWidgetOrder(unittest.TestCase):
    """BD_SAM3MultiPrompt node 13: link 16 on `prompts` while widgets_values[0]
    is still the prompt text. Consuming the queue only for unlinked inputs read
    every later widget one slot early -- the concrete node from the SBAI-11046
    ticket report, named so a future regression here greps straight to it."""

    OBJECT_INFO = {
        "Src": {"input": {"required": {}}, "output": ["STRING"]},
        "BD_SAM3MultiPrompt": {"input": {
            "required": {"prompts": ["STRING", {}]},
            "optional": {"negative_prompts": ["STRING", {}],
                         "combine_mode": [["union", "isect"], {}],
                         "vote_threshold": ["FLOAT", {}]},
        }, "output": ["MASK"]},
    }

    def test_bd_sam3_multi_prompt_widget_order(self):
        wf = {
            "nodes": [
                _node(1, "Src", [], outputs=[{"name": "s", "type": "STRING", "links": []}]),
                _node(13, "BD_SAM3MultiPrompt", [
                    _widget("prompts", "STRING", link=16),
                    _widget("negative_prompts", "STRING"),
                    _widget("combine_mode", "COMBO"),
                    _widget("vote_threshold", "FLOAT"),
                # `prompts` keeps its widgets_values slot even though it's linked
                # (widget_values_by_name reads-and-discards it); the ticket's bug
                # was treating linked-but-widget-typed fields as consuming no slot,
                # which shifts every value after it by one.
                ], widgets=["discarded_prompts_text", "negative_text", "union", 0.7]),
            ],
            "links": [[16, 1, 0, 13, 0, "STRING"]],
        }
        api = _convert(wf, self.OBJECT_INFO)
        self.assertEqual(api["13"]["inputs"], {
            "prompts": ["1", 0],
            "negative_prompts": "negative_text",
            "combine_mode": "union",
            "vote_threshold": 0.7,
        })


class BDPartsExportBaseImageWidgetFallback(unittest.TestCase):
    """BD_PartsExport.base_image arriving as the boolean True: its link was
    dropped, so the widget queue filled the input instead. Same cause as
    StaleTargetSlot/PartialInputArray above, asserted directly on the literal
    value the ticket reported rather than on wiring shape."""

    OBJECT_INFO = {
        "Src": {"input": {"required": {}}, "output": ["PARTS_BUNDLE"]},
        "BD_PartsExport": {"input": {
            # Declaration order matters: widget_values_by_name walks the merged
            # required-then-optional schema, not the node's `inputs` array.
            "required": {"parts_bundle": ["PARTS_BUNDLE", {}],
                         "base_image": ["IMAGE", {}],
                         "filename": ["STRING", {}]},
        }, "output": []},
    }

    def test_bd_parts_export_base_image_widget_fallback(self):
        wf = {
            "nodes": [
                _node(20, "Src", [], outputs=[{"name": "b", "type": "PARTS_BUNDLE", "links": []}]),
                _node(30, "BD_PartsExport", [
                    _sock("parts_bundle", "PARTS_BUNDLE", link=40),
                    # IMAGE is a connection type, so it's only in the widget queue
                    # at all because the graph marks it `widget` here -- an IMAGE
                    # input the frontend still exposes as a widget when unwired.
                    # No `link`: this is the dropped-link case from the ticket.
                    _widget("base_image", "IMAGE"),
                    _widget("filename", "STRING"),
                ], widgets=[True, "my_output.png"]),
            ],
            "links": [[40, 20, 0, 30, 0, "PARTS_BUNDLE"]],
        }
        api = _convert(wf, self.OBJECT_INFO)
        self.assertEqual(api["30"]["inputs"]["parts_bundle"], ["20", 0])
        self.assertEqual(api["30"]["inputs"]["base_image"], True)
        self.assertEqual(api["30"]["inputs"]["filename"], "my_output.png")


class UuidTypedNodeWithoutExpansion(unittest.TestCase):
    """A subgraph-instance node (type is the definition's UUID) with no
    matching `definitions.subgraphs` entry can't be expanded or scheduled --
    flat_to_api's object_info lookup already drops any node with no schema
    match, which is what actually protects this case, not UUID-specific
    detection. This locks that fallback in for the subgraph-instance shape
    specifically, since a change there would silently start emitting
    unschedulable nodes instead of skipping them."""

    def test_uuid_typed_nodes_skipped_without_expansion(self):
        wf = {
            "nodes": [_node("uuid-1234-5678", "aebc74c3-5e6b-4dc3-91f7-81ee51938f19", [
                _sock("input", "IMAGE", link=1),
            ])],
            "links": [[1, 10, 0, "uuid-1234-5678", 0, "IMAGE"]],
        }
        # No `definitions.subgraphs` entry for that UUID, and object_info has
        # no schema for it either -- both true of a saved-but-orphaned instance.
        api = _convert(wf, object_info={})
        self.assertNotIn("uuid-1234-5678", api)


class PartsBuilderRealWorkflowRegression(unittest.TestCase):
    """Convert the real BD-parts_builder.json template with the live server's
    own /object_info and compare shape against BD-parts_builder.api.json --
    ComfyUI's own frontend export, kept as ground truth. Hermetic tests above
    cover the wiring rules in isolation; this is the end-to-end check that a
    real 354-node production graph with real subgraphs converts without
    dropping or inventing edges."""

    UI_PATH = Path(__file__).resolve().parents[1] / "example_workflows" / "BD-parts_builder.json"
    API_PATH = Path(__file__).resolve().parents[1] / "api" / "BD-parts_builder.api.json"
    SERVER = "http://127.0.0.1:8188"

    @classmethod
    def setUpClass(cls):
        cls.ui = cls.api_oracle = cls.object_info = None
        if cls.UI_PATH.exists():
            with open(cls.UI_PATH) as f:
                cls.ui = json.load(f)
        if cls.API_PATH.exists():
            with open(cls.API_PATH) as f:
                cls.api_oracle = json.load(f)
        try:
            cls.object_info = sg.api_get(f"{cls.SERVER}/object_info")
        except Exception:
            cls.object_info = None

    def _require_fixtures(self):
        if not self.ui or not self.api_oracle or not self.object_info:
            self.skipTest("BD-parts_builder.json/.api.json or a live ComfyUI server is not available")

    def test_parts_builder_node_count(self):
        self._require_fixtures()
        converted = sg.workflow_to_api(self.ui, self.object_info)
        self.assertEqual(len(converted), len(self.api_oracle),
                          "converted node count should match the frontend-exported ground truth")

    def test_parts_builder_link_integrity(self):
        self._require_fixtures()
        converted = sg.workflow_to_api(self.ui, self.object_info)
        emitted_ids = set(converted.keys())
        dangling = [
            (nid, inp, val[0])
            for nid, node in converted.items()
            for inp, val in node.get("inputs", {}).items()
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str)
            and val[0] not in emitted_ids
        ]
        self.assertEqual(dangling, [], f"dangling link refs: {dangling[:5]}")


if __name__ == "__main__":
    unittest.main()
