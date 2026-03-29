import importlib.util
import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "resume-review-annotator-v2.py"


def load_annotator_module():
    spec = importlib.util.spec_from_file_location("snapshot_annotator", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AnnotatorLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.annotator = load_annotator_module()

    def test_build_note_body_parts_is_compact(self):
        issue = {
            "focus_text": "“街道卫生站工作者”",
            "comment": "简历方向和岗位方向不一致，投递会被直接筛掉。",
            "rewrite_tip": "求职意向改成活动策划执行，并删掉无关定位。",
            "rewrite_example": "把求职意向改成“活动策划执行专员”。",
        }

        parts = self.annotator.build_note_body_parts(issue)

        self.assertEqual(
            parts,
            [
                "看这里：“街道卫生站工作者”",
                "问题：简历方向和岗位方向不一致，投递会被直接筛掉",
                "改法：求职意向改成活动策划执行，并删掉无关定位",
            ],
        )

    def test_normalize_note_text_removes_single_quotes(self):
        text = "把'街道卫生站工作者'改成'活动策划执行专员'"

        normalized = self.annotator.normalize_note_text(text)

        self.assertNotIn("'", normalized)
        self.assertIn("“街道卫生站工作者”", normalized)
        self.assertIn("“活动策划执行专员”", normalized)

    def test_layout_note_title_wraps_long_title(self):
        image = Image.new("RGB", (600, 400), "white")
        draw = ImageDraw.Draw(image)
        title_font = self.annotator.load_font(26, family="sans")

        lines, height = self.annotator.layout_note_title(
            draw,
            "求职意向与目标岗位完全脱节",
            title_font,
            170,
        )

        self.assertGreater(len(lines), 1)
        self.assertGreater(height, 30)


if __name__ == "__main__":
    unittest.main()
