import json
import unittest

from services import task_service


class TaskDisplayRepairTests(unittest.TestCase):
    def test_row_to_dict_repairs_candidate_name_from_resume_file(self):
        row = {
            "id": "task-1",
            "created_at": "2026-03-29T00:00:00+00:00",
            "status": "completed",
            "status_message": None,
            "candidate_name": "第12题_Only.🌟@微信_20251230_吴泓磊简历_1",
            "jd_title": "未知岗位",
            "jd_file": "jd.jpg",
            "resume_file": "第12题_Only.🌟@微信_20251230_吴泓磊简历_1.pdf",
            "resume_pages": 1,
            "fail_reason": None,
            "result_json": json.dumps(
                {
                    "summary": "与“活动策划执行专员”岗位存在一定匹配。",
                    "jdKeywords": {
                        "coreResponsibilities": [
                            "主导农贸市场活动的创意策划与实施落地"
                        ]
                    },
                    "consultation": {"reason": "与“活动策划执行专员”岗位存在一定匹配。"},
                },
                ensure_ascii=False,
            ),
        }

        task = task_service._row_to_dict(row)

        self.assertEqual(task["candidateName"], "吴泓磊")
        self.assertEqual(task["jdTitle"], "活动策划执行专员")

    def test_row_to_dict_keeps_existing_clean_values(self):
        row = {
            "id": "task-2",
            "created_at": "2026-03-29T00:00:00+00:00",
            "status": "completed",
            "status_message": "刚刚 API 调用太多了，现在正在延迟重试，已进入排队状态。",
            "candidate_name": "吴泓磊",
            "jd_title": "活动策划执行专员",
            "jd_file": "jd.jpg",
            "resume_file": "吴泓磊简历.pdf",
            "resume_pages": 1,
            "fail_reason": None,
            "result_json": None,
        }

        task = task_service._row_to_dict(row)

        self.assertEqual(task["candidateName"], "吴泓磊")
        self.assertEqual(task["jdTitle"], "活动策划执行专员")
        self.assertEqual(task["statusMessage"], "刚刚 API 调用太多了，现在正在延迟重试，已进入排队状态。")


if __name__ == "__main__":
    unittest.main()
