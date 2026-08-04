"""Regression coverage for the bounded Trellis-to-FBX polling contract."""

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "run_unreal_fbx.py"
SPEC = importlib.util.spec_from_file_location("run_unreal_fbx", MODULE_PATH)
run_unreal_fbx = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_unreal_fbx)


class RunUnrealFbxTimeoutTests(unittest.TestCase):
    def test_default_timeout_allows_cpu_offload_stage(self):
        args = run_unreal_fbx.build_parser().parse_args(
            ["--image", "input.png", "--name", "Example"]
        )

        self.assertEqual(args.timeout, 7200)
        self.assertEqual(args.timeout, run_unreal_fbx.DEFAULT_TIMEOUT_SECONDS)


if __name__ == "__main__":
    unittest.main()
