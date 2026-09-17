"""SBAI-10643: BD_MediaPipeSAM3FaceSegment must emit batch-explicit (1,H,W) masks.

comfy core 0.27.0's JoinImageWithAlpha computes batch_size = max(len(image),
len(alpha)); len() on a 2D (H,W) mask returns its HEIGHT, so a 1536x1536
silhouette mask fed to the alpha input explodes the join to batch 1536
(run-r01 host OOM at 222 GB RSS; 648 GiB RGB repeat predicted at 6144x6144).
Evidence: head-profile-workflow-20260913-v01/run-r03/mask-compatibility-report.json.

face_mp_sam3.py needs the live comfy env to import (comfy_api.io, mediapipe,
relative face_mp_shared), so _m is AST-extracted — the same method the
mask-compatibility-report evidence used against the installed core functions.
"""
import ast
import math
import textwrap
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
NODE_FILE = ROOT / 'nodes' / 'segmentation' / 'face_mp_sam3.py'


def _extract_fn(name):
    tree = ast.parse(NODE_FILE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return textwrap.dedent(ast.unparse(node))
    raise AssertionError(f'{name} not found in {NODE_FILE}')


def _load_m():
    namespace = {'torch': torch, 'np': np}
    exec(compile(_extract_fn('_m'), str(NODE_FILE), 'exec'), namespace)
    return namespace['_m']


# ── Faithful replicas of comfy 0.27.0 join math (nodes_compositing.py sha
#    e001e296, comfy/utils.py sha 8b8805ca) — kept inline so this test runs
#    without a ComfyUI install on sys.path. ─────────────────────────────────
def _resize_mask(mask, shape):
    return F.interpolate(
        mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
        size=(shape[0], shape[1]), mode='bilinear').squeeze(1)


def _repeat_to_batch_size(tensor, batch_size, dim=0):
    if tensor.shape[dim] > batch_size:
        return tensor.narrow(dim, 0, batch_size)
    elif tensor.shape[dim] < batch_size:
        return tensor.repeat(
            dim * [1] + [math.ceil(batch_size / tensor.shape[dim])] +
            [1] * (len(tensor.shape) - 1 - dim)).narrow(dim, 0, batch_size)
    return tensor


def _join_like_core(image, alpha):
    batch_size = max(len(image), len(alpha))
    alpha = 1.0 - _resize_mask(alpha.to(image), image.shape[1:])
    alpha = _repeat_to_batch_size(alpha, batch_size)
    image = _repeat_to_batch_size(image, batch_size)
    return torch.cat((image[..., :3], alpha.unsqueeze(-1)), dim=-1)


class MaskShapeContractTests(unittest.TestCase):
    def test_m_emits_batch_explicit_3d_masks(self):
        m = _load_m()
        out = m(np.full((1536, 1536), 255, np.uint8))
        self.assertEqual(out.ndim, 3)
        self.assertEqual(tuple(out.shape), (1, 1536, 1536))

    def test_m_preserves_values(self):
        m = _load_m()
        src = np.array([[0, 128], [255, 255]], np.uint8)
        out = m(src)
        self.assertTrue(torch.allclose(out,
                                       torch.tensor([[[0.0, 128 / 255], [1.0, 1.0]]])))


class JoinShapeSafetyTests(unittest.TestCase):
    """The exact scenario from the FaceMaker v5 workflow: node 6377 InvertMask
    silhouette (originally 2D) wired into node 6374 JoinImageWithAlpha.alpha."""

    def setUp(self):
        torch.manual_seed(0)
        self.tiny_image = torch.rand(1, 4, 4, 3)

    def test_fixed_mask_keeps_batch_1(self):
        silhouette = _load_m()(np.full((1536, 1536), 255, np.uint8))
        inverted = 1.0 - silhouette  # InvertMask is shape-preserving
        joined = _join_like_core(self.tiny_image, inverted)
        self.assertEqual(tuple(joined.shape), (1, 4, 4, 4))

    def test_2d_mask_is_the_documented_hazard(self):
        # The pre-fix behaviour this suite guards against: len((1536,1536))
        # == 1536 is read as batch size and the join explodes.
        silhouette_2d = torch.from_numpy(
            np.full((1536, 1536), 255, np.float32) / 255.0)
        self.assertEqual(len(silhouette_2d), 1536)
        joined = _join_like_core(self.tiny_image, 1.0 - silhouette_2d)
        self.assertEqual(tuple(joined.shape), (1536, 4, 4, 4))


if __name__ == '__main__':
    unittest.main()
