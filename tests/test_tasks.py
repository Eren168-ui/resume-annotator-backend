import unittest

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


if __name__ == "__main__":
    unittest.main()
