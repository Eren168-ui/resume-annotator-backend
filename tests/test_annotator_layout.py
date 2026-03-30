import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def test_load_font_supports_linux_noto_cjk_path(self):
        linux_font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

        with (
            mock.patch.object(self.annotator.os.path, "exists", side_effect=lambda path: path == linux_font),
            mock.patch.object(self.annotator.ImageFont, "truetype", return_value="linux-font") as truetype,
            mock.patch.object(self.annotator.ImageFont, "load_default", return_value="default-font"),
        ):
            font = self.annotator.load_font(24, family="sans")

        self.assertEqual(font, "linux-font")
        truetype.assert_called_once_with(linux_font, size=24, index=2)

    def test_build_tail_page_blocks_compacts_to_single_page_summary(self):
        review = {
            "summary": "候选人基础较好，但与目标岗位的电商活动运营场景仍有明显错位，需要把经历改成贴岗版本。",
            "jd_keywords": ["天猫", "活动策划", "活动执行", "曝光", "转化率", "数据分析", "资源分配"],
            "jd_hard_skills": ["活动执行", "数据分析", "资源分配"],
            "jd_soft_skills": ["沟通协调", "团队协作"],
            "jd_responsibilities": ["负责店铺日常活动策划与执行，提升曝光与转化率"],
            "match_assessment": {
                "keyword_coverage": "medium",
                "professionalism": "medium",
                "clarity": "medium",
                "fit": "medium",
            },
            "strengths": ["有项目统筹能力", "有活动执行经验", "逻辑清晰"],
            "weaknesses": ["电商场景经验不足", "成果量化不够", "关键词不够贴岗"],
            "issues": [
                {"title": "求职意向不贴岗", "comment": "目前岗位定位过泛。", "rewrite_tip": "直接写活动策划执行岗。"},
                {"title": "成果缺少数字", "comment": "难判断真实影响力。", "rewrite_tip": "补人数、场次、转化或反馈。"},
                {"title": "电商关键词不足", "comment": "容易被系统忽略。", "rewrite_tip": "补曝光、转化、资源位等词。"},
                {"title": "项目排序靠后", "comment": "核心经历没有顶上来。", "rewrite_tip": "把最贴岗经历前置。"},
            ],
        }

        blocks = self.annotator.build_tail_page_blocks(review)
        titles = [title for title, _lines in blocks]

        self.assertEqual(
            titles,
            ["总评", "JD 关键信号", "匹配结论", "优先修改方向", "下一轮重点"],
        )
        self.assertLessEqual(len(blocks), 5)
        block_map = {title: lines for title, lines in blocks}
        self.assertGreaterEqual(len(block_map["总评"]), 2)
        self.assertTrue(any(line.startswith("优势：") for line in block_map["总评"]))
        self.assertGreaterEqual(len(block_map["匹配结论"]), 2)
        self.assertTrue(any(line.startswith("当前判断：") for line in block_map["匹配结论"]))
        self.assertEqual(len(block_map["优先修改方向"]), 4)
        self.assertTrue(any(line.startswith("素材：") for line in block_map["下一轮重点"]))

    def test_render_tail_pages_prefers_single_page_summary(self):
        review = {
            "summary": "候选人基础不错，但需要把目标岗位版本、关键词和成果表达统一起来。",
            "jd_keywords": ["天猫", "活动策划", "活动执行", "转化率", "曝光", "资源位", "数据分析"],
            "jd_hard_skills": ["活动执行", "数据分析", "资源分配"],
            "jd_soft_skills": ["沟通协调", "团队协作"],
            "jd_responsibilities": ["负责店铺活动策划与执行，提升曝光与转化率"],
            "match_assessment": {
                "keyword_coverage": "medium",
                "professionalism": "medium",
                "clarity": "medium",
                "fit": "medium",
            },
            "strengths": ["项目统筹不错", "执行经验完整"],
            "weaknesses": ["场景不贴岗", "成果缺数字"],
            "issues": [
                {"title": "求职意向不贴岗", "comment": "岗位方向偏了。", "rewrite_tip": "改成活动策划执行岗。"},
                {"title": "成果缺少数字", "comment": "结果不够可验证。", "rewrite_tip": "补场次、人数、反馈。"},
                {"title": "关键词不足", "comment": "贴岗词太少。", "rewrite_tip": "补曝光、转化、资源位。"},
                {"title": "项目排序靠后", "comment": "核心经历没有顶上来。", "rewrite_tip": "把最贴岗项目提前。"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            pages = self.annotator.render_tail_pages(
                review,
                Path(tmp_dir),
                page_width=1600,
                page_height=2200,
            )

        self.assertEqual(len(pages), 1)


if __name__ == "__main__":
    unittest.main()
