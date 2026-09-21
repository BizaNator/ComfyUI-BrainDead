"""The shipped character category table must catch what Qwen-VL actually writes.

The table is matched against free-form VLM tags, not a controlled vocabulary, so
a pattern that reads correctly can still never fire. Two did, on billpage:

  * `~sunglasses` never matched `black_sunglass_frame` — the tag has the
    singular stem — so the part fell through uncategorised and landed in the
    character root with a doubled filename.
  * `~earring` never matched `earings`, the spelling the VLM produces. That tag
    DOES contain "ring", so every pair of earrings was filed as a finger ring.

Longest-match-wins is what keeps the added stems from stealing other tags, so
`gold_ring` is asserted here too.

parts_types imports the live ComfyUI env, so the pieces are AST-extracted —
same method as test_mouth_parts_mask_shapes.
"""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "nodes" / "segmentation" / "parts_types.py"
TABLE = ROOT / "config" / "parts_categories_character.txt"
WANT = ("CategoryTable", "parse_category_table", "_fuzzy_match")


def _table():
    tree = ast.parse(MODULE.read_text())
    keep = [n for n in tree.body
            if getattr(n, "name", "") in WANT
            or isinstance(n, (ast.Import, ast.ImportFrom))]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(MODULE), "exec"), ns)
    return ns["parse_category_table"](TABLE.read_text())


class CharacterCategoryTable(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.t = _table()

    def test_vlm_spellings_of_the_removable_accessories(self):
        for tag, want in [
            ("black_sunglass_frame", ("sunglasses", "accessories")),
            ("earings", ("earring", "accessories")),
            ("earing", ("earring", "accessories")),
            ("blue_baseball_cap", ("cap", "accessories")),
            ("purple_headband", ("headband", "accessories")),
        ]:
            with self.subTest(tag=tag):
                self.assertEqual(self.t.get(tag), want)

    def test_longest_match_still_protects_neighbours(self):
        # "ring" is a substring of the earring spellings; the finger ring must
        # not be captured by them, nor they by it.
        self.assertEqual(self.t.get("gold_ring"), ("ring", "accessories"))

    def test_anatomy_and_clothing_are_untouched(self):
        for tag, want in [
            ("face", ("face", "head")),
            ("left_eye", ("left_eye", "head")),
            ("hair", ("hair", "head")),
            ("chest", ("chest", "body")),
            ("blue_polo_shirt", ("shirt", "clothing")),
        ]:
            with self.subTest(tag=tag):
                self.assertEqual(self.t.get(tag), want)


if __name__ == "__main__":
    unittest.main()
