import importlib
import io
import os
import sys
import tempfile
import unittest
import http.client
from unittest import mock

from PIL import Image


MODULE_NAME = "services.providers"


def reload_providers(env: dict[str, str]):
    with mock.patch.dict(os.environ, env, clear=True):
        sys.modules.pop(MODULE_NAME, None)
        return importlib.import_module(MODULE_NAME)


class ClaudeProviderConfigTests(unittest.TestCase):
    def test_openai_model_candidates_keep_primary_then_fallbacks(self):
        providers = reload_providers(
            {
                "AI_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-dashscope",
                "OPENAI_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode",
                "OPENAI_MODEL": "qwen-vl-max-latest",
                "OPENAI_FALLBACK_MODELS": "qwen-vl-max-2025-08-13,qwen-vl-plus-latest,qwen-vl-plus",
            }
        )

        self.assertEqual(
            providers._get_openai_model_candidates(),
            [
                "qwen-vl-max-latest",
                "qwen-vl-max-2025-08-13",
                "qwen-vl-plus-latest",
                "qwen-vl-plus",
            ],
        )

    def test_openai_provider_fails_over_to_next_model_on_rate_limit(self):
        providers = reload_providers(
            {
                "AI_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-dashscope",
                "OPENAI_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode",
                "OPENAI_MODEL": "qwen-vl-max-latest",
                "OPENAI_FALLBACK_MODELS": "qwen-vl-max-2025-08-13,qwen-vl-plus-latest",
            }
        )
        provider = providers.OpenAIProvider()

        with mock.patch.object(
            providers,
            "_http_post",
            side_effect=[
                RuntimeError("API 请求失败 HTTP 429 (https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions)"),
                {"choices": [{"message": {"content": "{\"summary\":\"ok\"}"}}]},
            ],
        ) as http_post:
            result = provider._call_api([{"role": "user", "content": "ping"}], "task-fallback")

        self.assertEqual(result, {"summary": "ok"})
        self.assertEqual(http_post.call_count, 2)
        first_payload = http_post.call_args_list[0].args[2]
        second_payload = http_post.call_args_list[1].args[2]
        self.assertEqual(first_payload["model"], "qwen-vl-max-latest")
        self.assertEqual(second_payload["model"], "qwen-vl-max-2025-08-13")

    def test_dashscope_omits_max_tokens_for_json_schema(self):
        providers = reload_providers(
            {
                "AI_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-dashscope",
                "OPENAI_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode",
                "OPENAI_MODEL": "qwen-vl-max-latest",
            }
        )
        provider = providers.OpenAIProvider()

        with mock.patch.object(
            providers,
            "_http_post",
            return_value={"choices": [{"message": {"content": "{\"summary\":\"ok\"}"}}]},
        ) as http_post:
            provider._call_api([{"role": "user", "content": "ping"}], "task-dashscope")

        payload = http_post.call_args.args[2]
        self.assertNotIn("max_tokens", payload)

    def test_non_dashscope_keeps_max_tokens_for_json_schema(self):
        providers = reload_providers(
            {
                "AI_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-openai",
                "OPENAI_BASE_URL": "https://api.openai.com",
                "OPENAI_MODEL": "gpt-4o-mini",
            }
        )
        provider = providers.OpenAIProvider()

        with mock.patch.object(
            providers,
            "_http_post",
            return_value={"choices": [{"message": {"content": "{\"summary\":\"ok\"}"}}]},
        ) as http_post:
            provider._call_api([{"role": "user", "content": "ping"}], "task-openai")

        payload = http_post.call_args.args[2]
        self.assertEqual(payload["max_tokens"], 4096)

    def test_claude_proxy_uses_openai_base_and_bearer_auth(self):
        providers = reload_providers(
            {
                "AI_PROVIDER": "claude",
                "ANTHROPIC_API_KEY": "sk-proxy",
                "OPENAI_BASE_URL": "https://oneapi.hk/",
                "OPENAI_MODEL": "claude-4.6-opus-medium",
            }
        )

        self.assertEqual(providers.ANTHROPIC_BASE_URL, "https://oneapi.hk")
        self.assertEqual(providers.ANTHROPIC_MODEL, "claude-4.6-opus-medium")
        self.assertEqual(providers._get_claude_endpoint(), "https://oneapi.hk/v1/messages")
        self.assertEqual(
            providers._get_claude_headers(),
            {
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-proxy",
            },
        )

    def test_official_claude_api_keeps_x_api_key_headers(self):
        providers = reload_providers(
            {
                "AI_PROVIDER": "claude",
                "ANTHROPIC_API_KEY": "sk-ant",
                "ANTHROPIC_BASE_URL": "https://api.anthropic.com/",
                "ANTHROPIC_MODEL": "claude-opus-4-6",
            }
        )

        self.assertEqual(providers.ANTHROPIC_BASE_URL, "https://api.anthropic.com")
        self.assertEqual(providers._get_claude_endpoint(), "https://api.anthropic.com/v1/messages")
        self.assertEqual(
            providers._get_claude_headers(),
            {
                "Content-Type": "application/json",
                "x-api-key": "sk-ant",
                "anthropic-version": "2023-06-01",
            },
        )

    def test_claude_api_key_falls_back_to_openai_api_key(self):
        providers = reload_providers(
            {
                "AI_PROVIDER": "claude",
                "OPENAI_API_KEY": "sk-openai-fallback",
                "OPENAI_BASE_URL": "https://oneapi.hk/",
                "OPENAI_MODEL": "claude-4.6-opus-medium",
            }
        )

        self.assertEqual(providers.ANTHROPIC_API_KEY, "sk-openai-fallback")
        self.assertEqual(
            providers._get_claude_headers(),
            {
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-openai-fallback",
            },
        )

    def test_debug_headers_are_redacted(self):
        providers = reload_providers(
            {
                "AI_PROVIDER": "claude",
                "ANTHROPIC_API_KEY": "sk-secret-token",
                "OPENAI_BASE_URL": "https://oneapi.hk/",
            }
        )

        self.assertEqual(
            providers._redact_headers_for_log(
                {
                    "Authorization": "Bearer sk-secret-token",
                    "x-api-key": "sk-secret-token",
                    "Content-Type": "application/json",
                }
            ),
            {
                "Authorization": "Bearer sk-s...oken",
                "x-api-key": "sk-s...oken",
                "Content-Type": "application/json",
            },
        )

    def test_note_text_is_sanitized_to_chinese_quotes(self):
        providers = reload_providers({"AI_PROVIDER": "claude"})

        self.assertEqual(
            providers._sanitize_note_text("把'街道卫生站工作者'改成'活动策划执行专员'"),
            "把“街道卫生站工作者”改成“活动策划执行专员”",
        )

    def test_extract_result_accepts_fenced_json(self):
        providers = reload_providers({"AI_PROVIDER": "openai"})

        result = providers.OpenAIProvider._extract_result(
            {
                "choices": [
                    {
                        "message": {
                            "content": "```json\n{\"summary\":\"ok\"}\n```"
                        }
                    }
                ]
            },
            "task-json",
        )

        self.assertEqual(result, {"summary": "ok"})

    def test_large_png_is_downscaled_for_api_upload(self):
        providers = reload_providers({"AI_PROVIDER": "claude"})

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "large.png")
            image = Image.new("RGB", (1600, 2200), color="white")
            image.save(path, format="PNG")
            original_size = os.path.getsize(path)

            mime, encoded = providers._encode_image(providers.Path(path))

            self.assertEqual(mime, "image/jpeg")
            decoded = providers.base64.b64decode(encoded)
            self.assertLess(len(decoded), original_size)

            loaded = Image.open(io.BytesIO(decoded))
            self.assertLessEqual(max(loaded.size), 1200)

    def test_claude_falls_back_to_compat_mode_on_disconnect(self):
        providers = reload_providers(
            {
                "AI_PROVIDER": "claude",
                "ANTHROPIC_API_KEY": "sk-proxy",
                "OPENAI_BASE_URL": "https://oneapi.hk/",
                "OPENAI_MODEL": "claude-4.6-opus-medium",
            }
        )
        provider = providers.ClaudeProvider()
        expected = {
            "summary": "ok",
            "jd_keywords": [],
            "jd_hard_skills": [],
            "jd_soft_skills": [],
            "jd_responsibilities": [],
            "match_assessment": {
                "keyword_coverage": "low",
                "professionalism": "low",
                "clarity": "low",
                "fit": "low",
            },
            "strengths": [],
            "weaknesses": [],
            "issues": [],
        }

        with mock.patch.object(
            provider,
            "_call_api_with_tools",
            side_effect=RuntimeError("中转服务提前断开了连接，通常是多模态请求兼容性或图片过大导致。"),
        ), mock.patch.object(
            provider,
            "_call_api_via_compat",
            return_value=expected,
        ) as compat_call:
            result = provider._call_api(
                content=[{"type": "text", "text": "demo"}],
                endpoint="https://oneapi.hk/v1/messages",
                jd_text="jd",
                resume_text="resume",
                task_id="task-1",
            )

        compat_call.assert_called_once()
        self.assertEqual(result, expected)

    def test_http_post_retries_after_remote_disconnect(self):
        providers = reload_providers({"AI_PROVIDER": "claude"})
        response_mock = mock.MagicMock()
        response_mock.__enter__.return_value.read.return_value = b'{"ok": true}'

        with mock.patch.object(
            providers.urllib.request,
            "urlopen",
            side_effect=[http.client.RemoteDisconnected("boom"), response_mock],
        ) as urlopen, mock.patch.object(providers.time, "sleep") as sleep:
            result = providers._http_post(
                "https://oneapi.hk/v1/chat/completions",
                {"Authorization": "Bearer sk-demo", "Content-Type": "application/json"},
                {"model": "demo", "messages": [{"role": "user", "content": "ping"}]},
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once()

    def test_compat_mode_retries_with_lite_prompt_after_disconnect(self):
        providers = reload_providers(
            {
                "AI_PROVIDER": "claude",
                "ANTHROPIC_API_KEY": "sk-proxy",
                "OPENAI_BASE_URL": "https://oneapi.hk/",
                "OPENAI_MODEL": "claude-4.6-opus-medium",
            }
        )
        provider = providers.ClaudeProvider()
        response = {
            "choices": [
                {
                    "message": {
                        "content": '{"summary":"ok","jd_keywords":[],"jd_hard_skills":[],"jd_soft_skills":[],"jd_responsibilities":[],"match_assessment":{"keyword_coverage":"low","professionalism":"low","clarity":"low","fit":"low"},"strengths":[],"weaknesses":[],"issues":[]}'
                    }
                }
            ]
        }

        with mock.patch.object(
            providers,
            "_http_post",
            side_effect=[
                RuntimeError("中转服务提前断开了连接，通常是多模态请求兼容性或图片过大导致。"),
                response,
            ],
        ) as http_post:
            result = provider._call_api_via_compat("jd text" * 200, "resume text" * 400, "task-2")

        self.assertEqual(result["summary"], "ok")
        self.assertEqual(http_post.call_count, 2)
        first_payload = http_post.call_args_list[0].args[2]
        second_payload = http_post.call_args_list[1].args[2]
        self.assertGreater(len(first_payload["messages"][1]["content"]), len(second_payload["messages"][1]["content"]))
        self.assertEqual(second_payload["max_tokens"], providers.COMPAT_MAX_TOKENS_LITE)

    def test_http_post_retries_after_rate_limit_and_updates_message(self):
        providers = reload_providers({"AI_PROVIDER": "claude"})
        response_mock = mock.MagicMock()
        response_mock.__enter__.return_value.read.return_value = b'{"ok": true}'
        rate_limit_error = providers.urllib.error.HTTPError(
            url="https://oneapi.hk/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(
                b'{"error":{"message":"1\xe5\x88\x86\xe9\x92\x9f\xe5\x86\x85\xe6\x9c\x80\xe5\xa4\x9a\xe8\xaf\xb7\xe6\xb1\x8210\xe6\xac\xa1"}}'
            ),
        )
        seen_messages = []

        with mock.patch.object(
            providers.urllib.request,
            "urlopen",
            side_effect=[rate_limit_error, response_mock],
        ) as urlopen, mock.patch.object(providers.time, "sleep") as sleep:
            result = providers._http_post(
                "https://oneapi.hk/v1/chat/completions",
                {"Authorization": "Bearer sk-demo", "Content-Type": "application/json"},
                {"model": "demo", "messages": [{"role": "user", "content": "ping"}]},
                on_rate_limit_retry=lambda message: seen_messages.append(message),
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(providers.HTTP_429_RETRY_DELAY)
        self.assertIn("延迟重试", seen_messages[0])
        self.assertIsNone(seen_messages[-1])


if __name__ == "__main__":
    unittest.main()
