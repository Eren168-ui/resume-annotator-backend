import unittest

from services.adapter import adapt_result


class AdaptResultConsultationTests(unittest.TestCase):
    def test_empty_consultation_guide_falls_back_to_low_fit_rewrite_recommendation(self):
        result = adapt_result(
            {
                "summary": "当前简历与岗位存在明显错位。",
                "match_assessment": {
                    "keyword_coverage": "low",
                    "professionalism": "medium",
                    "clarity": "medium",
                    "fit": "low",
                },
                "issues": [
                    {"title": "求职意向错位", "severity": "high", "category": "job_target", "comment": "方向不一致", "rewrite_tip": "改成目标岗位"},
                    {"title": "经历不贴岗", "severity": "high", "category": "relevance", "comment": "经历偏弱", "rewrite_tip": "重排经历"},
                ],
                "consultation_guide": {},
            },
            "task-low-fit",
        )

        self.assertTrue(result["consultation"]["recommend"])
        self.assertIn("深度改稿", result["consultation"]["headline"])
        self.assertIn("还没到可直接投递", " ".join(result["consultation"]["priorities"] + [result["consultation"]["reason"]]))

    def test_complete_consultation_guide_keeps_backend_headline(self):
        result = adapt_result(
            {
                "summary": "可先修改一轮。",
                "match_assessment": {
                    "keyword_coverage": "medium",
                    "professionalism": "medium",
                    "clarity": "medium",
                    "fit": "medium",
                },
                "consultation_guide": {
                    "recommended": False,
                    "headline": "可先按当前建议先自行修改一轮",
                    "summary": "先自己改完再复查。",
                    "reasons": ["基础尚可"],
                    "session_focus": ["先改重点经历"],
                    "prep_items": ["补充量化结果"],
                    "cta": "改完再看。",
                },
            },
            "task-medium-fit",
        )

        self.assertFalse(result["consultation"]["recommend"])
        self.assertEqual(result["consultation"]["headline"], "可先按当前建议先自行修改一轮")


if __name__ == "__main__":
    unittest.main()
