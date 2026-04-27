import importlib
import os
import re
import sys
import unittest
from unittest import mock

MODULE_NAME = "main"


def load_app(env: dict[str, str]):
    sys.modules.pop(MODULE_NAME, None)
    with mock.patch.dict(os.environ, env, clear=True):
        module = importlib.import_module(MODULE_NAME)
    return module


class CorsConfigTests(unittest.TestCase):
    def _cors_options(self):
        module = load_app(
            {
                "FRONTEND_ORIGIN": "https://resume.erenlab.cn",
                "CLOUDFLARE_PAGES_PROJECT": "resume-annotator-b2b",
            }
        )
        return next(
            middleware.kwargs
            for middleware in module.app.user_middleware
            if middleware.cls.__name__ == "CORSMiddleware"
        )

    @staticmethod
    def _is_allowed(options: dict, origin: str) -> bool:
        if origin in options.get("allow_origins", []):
            return True
        regex = options.get("allow_origin_regex")
        return bool(regex and re.match(regex, origin))

    def test_allows_configured_production_origin(self):
        options = self._cors_options()

        self.assertTrue(self._is_allowed(options, "https://resume.erenlab.cn"))

    def test_allows_cloudflare_pages_production_domain(self):
        options = self._cors_options()

        self.assertTrue(self._is_allowed(options, "https://resume-annotator-b2b.pages.dev"))

    def test_allows_cloudflare_pages_preview_subdomain(self):
        options = self._cors_options()

        self.assertTrue(self._is_allowed(options, "https://feature-login.resume-annotator-b2b.pages.dev"))

    def test_rejects_unrelated_origin(self):
        options = self._cors_options()

        self.assertFalse(self._is_allowed(options, "https://evil.example.com"))


if __name__ == "__main__":
    unittest.main()
