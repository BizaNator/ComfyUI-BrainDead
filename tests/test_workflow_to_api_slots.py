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


if __name__ == "__main__":
    unittest.main()
