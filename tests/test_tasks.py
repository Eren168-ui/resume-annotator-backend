import asyncio
import unittest
from unittest import mock

from api import tasks


class TaskErrorMessageTests(unittest.TestCase):
    def test_unauthorized_error_keeps_detailed_http_body(self):
        message = tasks._build_task_error_message(
            "API 请求失败 HTTP 401 (https://oneapi.hk/v1/messages)\n\nResponse body:\ninvalid_api_key"
        )
        self.assertIn("API 认证失败 (401)", message)
        self.assertIn("invalid_api_key", message)

    def test_rate_limit_error_keeps_detailed_http_body(self):
        message = tasks._build_task_error_message(
            "API 请求失败 HTTP 429 (https://oneapi.hk/v1/messages)\n\nResponse body:\nrate_limit"
        )
        self.assertIn("API 请求频率超限 (429)", message)
        self.assertIn("rate_limit", message)


class TaskWorkerConfigTests(unittest.TestCase):
    def test_task_max_workers_reads_env(self):
        with mock.patch.dict("os.environ", {"TASK_MAX_WORKERS": "6"}, clear=False):
            self.assertEqual(tasks._task_max_workers(), 6)

    def test_task_max_workers_falls_back_on_invalid_value(self):
        with mock.patch.dict("os.environ", {"TASK_MAX_WORKERS": "0"}, clear=False):
            self.assertEqual(tasks._task_max_workers(), 3)


class DeleteTaskRouteTests(unittest.TestCase):
    def test_delete_task_rejects_processing_task(self):
        with mock.patch.object(
            tasks.task_service,
            "get_task",
            return_value={"id": "task-1", "status": "processing"},
        ):
            with self.assertRaises(Exception) as ctx:
                asyncio.run(tasks.delete_task("task-1", {"sub": "u1"}))

        self.assertEqual(getattr(ctx.exception, "status_code", None), 409)

    def test_delete_task_removes_completed_task_and_files(self):
        with (
            mock.patch.object(
                tasks.task_service,
                "get_task",
                return_value={"id": "task-1", "status": "completed"},
            ),
            mock.patch.object(tasks.task_service, "delete_task", return_value=True) as delete_task,
            mock.patch.object(tasks.storage, "delete_task_files") as delete_task_files,
        ):
            response = asyncio.run(tasks.delete_task("task-1", {"sub": "u1"}))

        self.assertEqual(response.status_code, 204)
        delete_task.assert_called_once_with("task-1")
        delete_task_files.assert_called_once_with("task-1")


class TaskResultRouteTests(unittest.TestCase):
    def test_get_task_result_rebuilds_download_manifest_from_existing_files(self):
        manifest = {
            "annotatedPdf": "annotated_task-1.pdf",
            "markdownReport": "report_task-1.md",
            "report": "report_task-1.html",
            "jsonResult": "result_task-1.json",
            "pageImages": [],
        }

        with (
            mock.patch.object(
                tasks.task_service,
                "get_task",
                return_value={"id": "task-1", "status": "completed"},
            ),
            mock.patch.object(
                tasks.task_service,
                "get_task_result",
                return_value={"summary": "ok", "downloads": {"annotatedPdf": "old.pdf"}},
            ),
            mock.patch.object(tasks.storage, "build_download_manifest", return_value=manifest),
        ):
            result = asyncio.run(tasks.get_task_result("task-1", {"sub": "u1"}))

        self.assertEqual(result["downloads"], manifest)


if __name__ == "__main__":
    unittest.main()
