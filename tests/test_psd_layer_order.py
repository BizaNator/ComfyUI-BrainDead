"""pytoshop layer order (UEFN-430): nested_layers_to_psd takes its list TOP first - the contract parts_export relies on
when it reverses its back-to-front list. If pytoshop ever changes this, exported PSDs flip again; this test says so."""
import io

import numpy as np
import pytest

pytoshop = pytest.importorskip("pytoshop")
from pytoshop import enums  # noqa: E402
from pytoshop.user import nested_layers as nl  # noqa: E402


def _layer(name, v):
    a = np.full((4, 4), v, np.uint8)
    return nl.Image(name=name, top=0, left=0, bottom=4, right=4, channels={-1: np.full((4, 4), 255, np.uint8), 0: a, 1: a, 2: a})


def test_nested_layers_to_psd_list_is_top_first():
    psd = nl.nested_layers_to_psd([_layer("top", 200), _layer("bottom", 10)], color_mode=3, size=(4, 4),
                                  compression=enums.Compression.raw)
    buf = io.BytesIO()
    psd.write(buf)
    buf.seek(0)
    recs = pytoshop.read(buf).layer_and_mask_info.layer_info.layer_records
    # PSD layer records are stored bottom-up: the first record is the bottom layer
    assert [r.name for r in recs] == ["bottom", "top"]


def test_parts_export_reverses_its_back_to_front_list():
    src = open(__file__.replace("tests/test_psd_layer_order.py", "nodes/segmentation/parts_export.py")).read()
    assert "layers[::-1], color_mode=3" in src
