"""
AI provider abstraction for the resume review step.

Supported providers:
  - openai  : OpenAI Chat Completions API (/v1/chat/completions)
              Works with OpenAI official, OneAPI, LiteLLM, and any OpenAI-compatible proxy.
              DOES NOT use the annotator script for the review step — calls the API directly.
  - claude  : Anthropic Messages API (/v1/messages) with tool_use for structured output.

Switch via env var:  AI_PROVIDER=openai | claude

Base URL config:
  OPENAI_BASE_URL=https://api.openai.com        (default, direct OpenAI)
  OPENAI_BASE_URL=https://your-oneapi.com       (OneAPI / any compatible proxy)
  OPENAI_BASE_URL=https://your-oneapi.com/v1    (also accepted — /v1 is normalized)
"""

from __future__ import annotations

import abc
import base64
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import time
import http.client
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# ── Config from env ────────────────────────────────────────────────────────────

AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").lower()


def _normalize_base_url(raw_value: str, default: str) -> str:
    return (raw_value or default).rstrip("/").removesuffix("/v1")

# OpenAI / OneAPI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv("OPENAI_FALLBACK_MODELS", "").split(",")
    if model.strip()
]
# Accept both OPENAI_BASE_URL (new) and OPENAI_API_BASE (legacy), strip trailing /v1
_raw_openai_base = (
    os.getenv("OPENAI_BASE_URL")
    or os.getenv("OPENAI_API_BASE")
    or "https://api.openai.com"
)
OPENAI_BASE_URL = _normalize_base_url(_raw_openai_base, "https://api.openai.com")
OPENAI_JSON_SCHEMA_OMIT_MAX_TOKENS = os.getenv(
    "OPENAI_JSON_SCHEMA_OMIT_MAX_TOKENS", "auto"
).lower()

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL") or OPENAI_MODEL or "claude-opus-4-6"
_raw_anthropic_base = (
    os.getenv("ANTHROPIC_BASE_URL")
    or os.getenv("ANTHROPIC_API_BASE")
    or os.getenv("OPENAI_BASE_URL")
    or os.getenv("OPENAI_API_BASE")
    or "https://api.anthropic.com"
)
ANTHROPIC_BASE_URL = _normalize_base_url(_raw_anthropic_base, "https://api.anthropic.com")

# Annotator script (only needed for render-pdf / report-md — local steps, no AI call)
ANNOTATOR_SCRIPT = Path(
    os.getenv(
        "ANNOTATOR_SCRIPT",
        str(
            Path(__file__).parent.parent
            / "tools"
            / "resume-review-annotator-v2.py"
        ),
    )
)

MAX_PAGES = int(os.getenv("MAX_RESUME_PAGES", "2"))
RENDER_DPI = int(os.getenv("RENDER_DPI", "180"))
API_IMAGE_MAX_DIM = int(os.getenv("API_IMAGE_MAX_DIM", "1200"))
API_IMAGE_MAX_BYTES = int(os.getenv("API_IMAGE_MAX_BYTES", "300000"))
API_IMAGE_JPEG_QUALITY = int(os.getenv("API_IMAGE_JPEG_QUALITY", "70"))
COMPAT_JD_TEXT_LIMIT = int(os.getenv("COMPAT_JD_TEXT_LIMIT", "400"))
COMPAT_RESUME_TEXT_LIMIT = int(os.getenv("COMPAT_RESUME_TEXT_LIMIT", "900"))
COMPAT_MAX_TOKENS = int(os.getenv("COMPAT_MAX_TOKENS", "1200"))
COMPAT_JD_TEXT_LIMIT_LITE = int(os.getenv("COMPAT_JD_TEXT_LIMIT_LITE", "220"))
COMPAT_RESUME_TEXT_LIMIT_LITE = int(os.getenv("COMPAT_RESUME_TEXT_LIMIT_LITE", "520"))
COMPAT_MAX_TOKENS_LITE = int(os.getenv("COMPAT_MAX_TOKENS_LITE", "900"))
HTTP_POST_RETRIES = int(os.getenv("HTTP_POST_RETRIES", "2"))
HTTP_POST_RETRY_DELAY = float(os.getenv("HTTP_POST_RETRY_DELAY", "1.2"))
HTTP_429_RETRIES = int(os.getenv("HTTP_429_RETRIES", "2"))
HTTP_429_RETRY_DELAY = float(os.getenv("HTTP_429_RETRY_DELAY", "70"))

# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
你是一名资深招聘顾问、简历教练和真实做过大学生/留学生求职辅导的老师。
你的任务不是泛泛点评，而是结合岗位 JD 和候选人简历，输出一份"岗位定向简历批注结果"。

输入会包含：
1. 岗位 JD 截图
2. JD 的 OCR 文本
3. 候选人简历 PDF 提取文本
4. 候选人简历每一页的图片

请遵守以下要求：
1. 只返回符合 schema 的 JSON。
2. 批注语言必须使用简体中文。
3. 批注口吻要像真人改简历，直接、自然、短，不要像系统说明书。
4. 必须先理解 JD，再判断简历匹配度，不允许脱离岗位空谈。
5. 从 JD 中提取 10-15 个最重要的关键词，并拆分为：硬技能、软技能、职责要求。
6. 对简历做 4 个维度评级（keyword_coverage/professionalism/clarity/fit），每项只能是 high/medium/low。
7. issues 最多返回 8 条，优先保留最影响通过率的问题。
8. 每条 issue 必须绑定到简历中的具体页码和具体区域，给出可落地的改法。
9. focus_text 必须尽量引用被批注区域中的原始文本。
10. anchor 使用 0-1000 归一化坐标系，字段为 x、y、w、h。
11. page 从 1 开始计数。
12. 不要编造简历里没有的经历、数据或结果。
13. 从 JD 中提取简明的岗位名称（如"运营专员"、"产品经理"、"Java 后端工程师"），填入 jd_title 字段，不超过 20 个字。
14. 从简历中提取候选人的真实姓名（中文或英文），填入 candidate_name 字段；无法确定时填 null。
comment 和 rewrite_tip 要短、准、像老师直接说的话。
每条 issue 保留“看这里 / 问题 / 改法”的真人老师口吻，但不要写成小作文。
comment 建议 18-36 个汉字，rewrite_tip 建议 16-36 个汉字，要有判断，也要有改法。
title 建议 4-10 个汉字，避免过长。
不要使用英文单引号，涉及引用时统一用中文双引号。
""".strip()

# ── JSON Schema ────────────────────────────────────────────────────────────────

MATCH_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "jd_title": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "candidate_name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "jd_keywords": {"type": "array", "items": {"type": "string"}},
        "jd_hard_skills": {"type": "array", "items": {"type": "string"}},
        "jd_soft_skills": {"type": "array", "items": {"type": "string"}},
        "jd_responsibilities": {"type": "array", "items": {"type": "string"}},
        "match_assessment": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "keyword_coverage": {"type": "string", "enum": ["high", "medium", "low"]},
                "professionalism": {"type": "string", "enum": ["high", "medium", "low"]},
                "clarity": {"type": "string", "enum": ["high", "medium", "low"]},
                "fit": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["keyword_coverage", "professionalism", "clarity", "fit"],
        },
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "consultation_guide": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "recommended": {"type": "boolean"},
                "headline": {"type": "string"},
                "summary": {"type": "string"},
                "reasons": {"type": "array", "items": {"type": "string"}},
                "session_focus": {"type": "array", "items": {"type": "string"}},
                "prep_items": {"type": "array", "items": {"type": "string"}},
                "cta": {"type": "string"},
            },
            "required": ["recommended", "headline", "summary", "reasons",
                         "session_focus", "prep_items", "cta"],
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "page": {"type": "integer"},
                    "title": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "category": {"type": "string"},
                    "star_gap": {"type": "string", "enum": ["S", "T", "A", "R", "STAR"]},
                    "comment": {"type": "string"},
                    "rewrite_tip": {"type": "string"},
                    "rewrite_example": {"type": "string"},
                    "focus_text": {"type": "string"},
                    "padding": {"type": "number"},
                    "ocr_max_lines": {"type": "number"},
                    "ocr_enabled": {"type": "boolean"},
                    "anchor": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "w": {"type": "number"},
                            "h": {"type": "number"},
                        },
                        "required": ["x", "y", "w", "h"],
                    },
                },
                "required": ["id", "page", "title", "severity", "category", "star_gap",
                             "comment", "rewrite_tip", "focus_text", "anchor"],
            },
        },
    },
    "required": [
        "summary", "jd_title", "candidate_name",
        "jd_keywords", "jd_hard_skills", "jd_soft_skills",
        "jd_responsibilities", "match_assessment", "strengths", "weaknesses", "issues",
    ],
}

# ── Shared preprocessing helpers ──────────────────────────────────────────────

def _require(binary: str) -> str:
    p = shutil.which(binary)
    if not p:
        raise RuntimeError(
            f"Required binary not found in PATH: '{binary}'. "
            f"macOS: brew install poppler tesseract tesseract-lang"
        )
    return p


def _encode_image(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(ext)
    if not mime:
        raise ValueError(f"Unsupported image format: {ext}")

    raw = path.read_bytes()
    if _should_compress_for_api(path, raw):
        mime, raw = _compress_image_for_api(path)

    return mime, base64.b64encode(raw).decode()


def _should_compress_for_api(path: Path, raw: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
    except Exception:
        return False

    return (
        len(raw) > API_IMAGE_MAX_BYTES
        or width > API_IMAGE_MAX_DIM
        or height > API_IMAGE_MAX_DIM
        or path.suffix.lower() == ".png"
    )


def _compress_image_for_api(path: Path) -> tuple[str, bytes]:
    with Image.open(path) as image:
        normalized = image.convert("RGB")
        normalized.thumbnail((API_IMAGE_MAX_DIM, API_IMAGE_MAX_DIM))
        output = io.BytesIO()
        normalized.save(
            output,
            format="JPEG",
            quality=API_IMAGE_JPEG_QUALITY,
            optimize=True,
        )
        return "image/jpeg", output.getvalue()


def _pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int, max_pages: int) -> list[Path]:
    _require("pdftoppm")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "page"
    cmd = ["pdftoppm", "-png", "-r", str(dpi), "-l", str(max_pages), str(pdf_path), str(prefix)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        err = r.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"pdftoppm failed (exit {r.returncode}): {err}")
    images = sorted(output_dir.glob("page-*.png"))
    if not images:
        raise RuntimeError(
            "pdftoppm produced no images — is the PDF valid, non-encrypted, and non-empty?"
        )
    return images


def _pdf_to_text(pdf_path: Path) -> str:
    _require("pdftotext")
    r = subprocess.run(["pdftotext", str(pdf_path), "-"], capture_output=True)
    return r.stdout.decode("utf-8", errors="replace")


def _ocr_image(image_path: Path) -> str:
    if not shutil.which("tesseract"):
        logger.warning("tesseract not found — JD OCR skipped. Install: brew install tesseract tesseract-lang")
        return ""
    r = subprocess.run(
        ["tesseract", str(image_path), "stdout", "-l", "chi_sim+eng", "--psm", "6"],
        capture_output=True,
    )
    return r.stdout.decode("utf-8", errors="replace").strip()


def _http_post(
    url: str,
    headers: dict,
    payload: dict,
    timeout: int = 180,
    on_rate_limit_retry: Any = None,
) -> dict:
    """
    Shared HTTP POST helper. Raises RuntimeError with FULL response body on failure.
    """
    logger.debug(
        "Requesting API at %s with headers=%s payload_meta=%s",
        url,
        _redact_headers_for_log(headers),
        {
            "model": payload.get("model"),
            "message_count": len(payload.get("messages", [])),
            "has_tools": bool(payload.get("tools")),
        },
    )
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        max_attempts = max(HTTP_POST_RETRIES, HTTP_429_RETRIES)
        for attempt in range(1, max_attempts + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if on_rate_limit_retry:
                        on_rate_limit_retry(None)
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="replace")
                logger.error(
                    "HTTP %s from %s\nResponse body:\n%s",
                    exc.code, url, raw
                )
                if exc.code == 429 and attempt < HTTP_429_RETRIES:
                    if on_rate_limit_retry:
                        on_rate_limit_retry(
                            "刚刚 API 调用太多了，现在正在延迟重试，已进入排队状态。预计约 70 秒后自动重试。"
                        )
                    logger.warning(
                        "Retrying %s after rate limit (%s/%s)",
                        url,
                        attempt,
                        HTTP_429_RETRIES,
                    )
                    time.sleep(HTTP_429_RETRY_DELAY)
                    continue
                raise RuntimeError(
                    f"API 请求失败 HTTP {exc.code} ({url})\n\n"
                    f"Response body:\n{raw}"
                ) from exc
            except http.client.RemoteDisconnected as exc:
                logger.error("Remote disconnected calling %s: %s", url, exc)
                if attempt < HTTP_POST_RETRIES:
                    logger.warning(
                        "Retrying %s after remote disconnect (%s/%s)",
                        url,
                        attempt,
                        HTTP_POST_RETRIES,
                    )
                    time.sleep(HTTP_POST_RETRY_DELAY)
                    continue
                raise RuntimeError(
                    "中转服务提前断开了连接，通常是多模态请求兼容性或图片过大导致。"
                    "请重试；如果仍然失败，请降低图片尺寸后再试。"
                ) from exc
    except urllib.error.URLError as exc:
        logger.error("URL error calling %s: %s", url, exc)
        raise RuntimeError(
            f"网络连接失败: {exc}\n"
            f"请检查 OPENAI_BASE_URL 配置是否正确，当前值: {url}"
        ) from exc


def _use_claude_proxy_auth(base_url: str) -> bool:
    return _normalize_base_url(base_url, "https://api.anthropic.com") != "https://api.anthropic.com"


def _is_dashscope_openai_compatible(base_url: str) -> bool:
    return _normalize_base_url(base_url, "https://api.openai.com") == "https://dashscope.aliyuncs.com/compatible-mode"


def _should_omit_max_tokens_for_json_schema(base_url: str) -> bool:
    if OPENAI_JSON_SCHEMA_OMIT_MAX_TOKENS in {"1", "true", "yes", "on"}:
        return True
    if OPENAI_JSON_SCHEMA_OMIT_MAX_TOKENS in {"0", "false", "no", "off"}:
        return False
    return _is_dashscope_openai_compatible(base_url)


def _get_openai_model_candidates() -> list[str]:
    candidates: list[str] = []
    for model in [OPENAI_MODEL, *OPENAI_FALLBACK_MODELS]:
        if model and model not in candidates:
            candidates.append(model)
    return candidates


def _should_failover_openai_model(exc: Exception) -> bool:
    message = str(exc).lower()
    matchers = [
        "http 429",
        "rate limit",
        "请求频率超限",
        "请求次数太多",
        "too many requests",
        "quota",
        "insufficient_quota",
        "余额",
        "额度",
        "billing",
        "余额不足",
        "timeout",
        "timed out",
        "read operation timed out",
        "connection reset",
        "connection aborted",
        "remote end closed connection without response",
        "remote disconnected",
        "model unavailable",
        "model_not_found",
        "does not exist",
        "暂不可用",
        "不可用",
    ]
    return any(token in message for token in matchers)


def _mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def _redact_headers_for_log(headers: dict[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in headers.items():
        lower_key = key.lower()
        if lower_key == "authorization" and value.startswith("Bearer "):
            token = value.removeprefix("Bearer ").strip()
            redacted[key] = f"Bearer {_mask_secret(token)}"
        elif lower_key in {"x-api-key", "authorization"}:
            redacted[key] = _mask_secret(value)
        else:
            redacted[key] = value
    return redacted


def _get_claude_endpoint() -> str:
    return f"{ANTHROPIC_BASE_URL}/v1/messages"


def _get_claude_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if _use_claude_proxy_auth(ANTHROPIC_BASE_URL):
        headers["Authorization"] = f"Bearer {ANTHROPIC_API_KEY}"
        return headers

    headers["x-api-key"] = ANTHROPIC_API_KEY
    headers["anthropic-version"] = "2023-06-01"
    return headers


# ── Provider interface ────────────────────────────────────────────────────────

class AIProvider(abc.ABC):
    @abc.abstractmethod
    def review_match(
        self,
        jd_image_path: Path,
        resume_pdf_path: Path,
        result_path: Path,
        task_id: str,
    ) -> None:
        """Perform AI review and save result JSON to result_path."""


# ── OpenAI provider ───────────────────────────────────────────────────────────

class OpenAIProvider(AIProvider):
    """
    Calls OpenAI Chat Completions API (/v1/chat/completions) directly.

    Uses OPENAI_BASE_URL so it works with:
      - OpenAI official   (OPENAI_BASE_URL=https://api.openai.com)
      - OneAPI proxy      (OPENAI_BASE_URL=https://your-oneapi.com)
      - LiteLLM / any OpenAI-compatible endpoint

    Does NOT use the annotator script for the review step.
    render-pdf and report-md still use the annotator script (local, no AI).
    """

    def review_match(
        self,
        jd_image_path: Path,
        resume_pdf_path: Path,
        result_path: Path,
        task_id: str,
    ) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "AI_PROVIDER=openai 但未设置 OPENAI_API_KEY，请在 .env 中配置"
            )

        endpoint = f"{OPENAI_BASE_URL}/v1/chat/completions"
        logger.info(
            "[%s] OpenAI provider: endpoint=%s model=%s",
            task_id, endpoint, OPENAI_MODEL,
        )

        with tempfile.TemporaryDirectory(prefix="openai-review-") as tmp:
            tmp_path = Path(tmp)
            page_images = _pdf_to_images(resume_pdf_path, tmp_path / "pages", RENDER_DPI, MAX_PAGES)
            resume_text = _pdf_to_text(resume_pdf_path)
            jd_text = _ocr_image(jd_image_path)

            messages = self._build_messages(jd_image_path, jd_text, resume_text, page_images)

            logger.info("[%s] OpenAI provider: calling API (%d pages)", task_id, len(page_images))
            result = self._call_api(messages, task_id)

        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("[%s] OpenAI provider: saved result to %s", task_id, result_path)

    def _build_messages(
        self,
        jd_image_path: Path,
        jd_text: str,
        resume_text: str,
        page_images: list[Path],
    ) -> list[dict]:
        jd_mime, jd_b64 = _encode_image(jd_image_path)

        user_content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "请结合岗位 JD 与候选人简历，输出岗位定向简历批注结果。\n\n"
                    "以下 OCR/提取文本可能存在少量误差，请以图片和 PDF 页面内容为准。\n\n"
                    f"【岗位 JD OCR 文本】\n{jd_text}\n\n"
                    f"【候选人简历提取文本】\n{resume_text}"
                ),
            },
            {"type": "text", "text": "下面是岗位 JD 截图："},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{jd_mime};base64,{jd_b64}", "detail": "high"},
            },
        ]

        for i, page_path in enumerate(page_images, start=1):
            page_mime, page_b64 = _encode_image(page_path)
            user_content.append({"type": "text", "text": f"下面是候选人简历第 {i} 页图片："})
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{page_mime};base64,{page_b64}",
                    "detail": "high",
                },
            })

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _call_api(self, messages: list[dict], task_id: str) -> dict:
        url = f"{OPENAI_BASE_URL}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        models = _get_openai_model_candidates()
        last_error: Exception | None = None

        for index, model in enumerate(models, start=1):
            payload = {
                "model": model,
                "messages": messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "resume_job_match_review",
                        "strict": True,
                        "schema": MATCH_RESPONSE_SCHEMA,
                    },
                },
            }
            if not _should_omit_max_tokens_for_json_schema(OPENAI_BASE_URL):
                payload["max_tokens"] = 4096
            else:
                logger.info(
                    "[%s] OpenAI provider: omit max_tokens for json_schema compatibility (base=%s)",
                    task_id,
                    OPENAI_BASE_URL,
                )

            try:
                logger.info(
                    "[%s] OpenAI provider: trying model %s (%s/%s)",
                    task_id,
                    model,
                    index,
                    len(models),
                )
                data = _http_post(url, headers, payload)
                return self._extract_result(data, task_id)
            except RuntimeError as exc:
                last_error = exc
                if index < len(models) and _should_failover_openai_model(exc):
                    logger.warning(
                        "[%s] OpenAI provider: model %s failed, switch to %s: %s",
                        task_id,
                        model,
                        models[index],
                        exc,
                    )
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("OpenAI provider 未找到可用模型")

    @staticmethod
    def _extract_result(data: dict, task_id: str) -> dict:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            logger.error("[%s] Unexpected response shape: %s", task_id, json.dumps(data)[:1000])
            raise RuntimeError(
                f"API 返回格式异常，无法提取 choices[0].message.content\n"
                f"完整响应: {json.dumps(data, ensure_ascii=False)[:2000]}"
            ) from exc

        if content is None:
            # Some models return content=null when finish_reason=content_filter
            finish_reason = data.get("choices", [{}])[0].get("finish_reason", "unknown")
            raise RuntimeError(
                f"API 返回 content=null (finish_reason={finish_reason})。"
                "可能触发内容过滤，或模型不支持 json_schema response_format。"
            )

        if isinstance(content, dict):
            return content  # Some providers return pre-parsed JSON

        content = _coerce_json_text(content)
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error("[%s] Non-JSON content: %s", task_id, content[:500])
            raise RuntimeError(
                f"API 返回的不是合法 JSON:\n{content[:1000]}"
            ) from exc


# ── Claude provider ───────────────────────────────────────────────────────────

class ClaudeProvider(AIProvider):
    """
    Calls Anthropic Messages API with tool_use for structured JSON output.
    Uses ANTHROPIC_API_KEY + ANTHROPIC_MODEL + ANTHROPIC_BASE_URL env vars.
    If OPENAI_BASE_URL is set without ANTHROPIC_BASE_URL, treat it as the relay base.
    """

    def review_match(
        self,
        jd_image_path: Path,
        resume_pdf_path: Path,
        result_path: Path,
        task_id: str,
    ) -> None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("AI_PROVIDER=claude 但未设置 ANTHROPIC_API_KEY，请在 .env 中配置")

        logger.info("[%s] Claude provider: preprocessing files", task_id)

        with tempfile.TemporaryDirectory(prefix="claude-review-") as tmp:
            tmp_path = Path(tmp)
            page_images = _pdf_to_images(resume_pdf_path, tmp_path / "pages", RENDER_DPI, MAX_PAGES)
            resume_text = _pdf_to_text(resume_pdf_path)
            jd_text = _ocr_image(jd_image_path)
            content = self._build_content(jd_image_path, jd_text, resume_text, page_images)

            endpoint = _get_claude_endpoint()
            logger.info("[%s] Claude provider: endpoint=%s model=%s", task_id, endpoint, ANTHROPIC_MODEL)
            result = self._call_api(content, endpoint, jd_text, resume_text, task_id)

        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("[%s] Claude provider: saved result to %s", task_id, result_path)

    def _build_content(
        self,
        jd_image_path: Path,
        jd_text: str,
        resume_text: str,
        page_images: list[Path],
    ) -> list[dict]:
        jd_mime, jd_b64 = _encode_image(jd_image_path)
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "请结合岗位 JD 与候选人简历，输出岗位定向简历批注结果。\n\n"
                    f"【岗位 JD OCR 文本】\n{jd_text}\n\n"
                    f"【候选人简历提取文本】\n{resume_text}"
                ),
            },
            {"type": "text", "text": "下面是岗位 JD 截图："},
            {"type": "image", "source": {"type": "base64", "media_type": jd_mime, "data": jd_b64}},
        ]
        for i, page_path in enumerate(page_images, start=1):
            page_mime, page_b64 = _encode_image(page_path)
            content.append({"type": "text", "text": f"下面是候选人简历第 {i} 页图片："})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": page_mime, "data": page_b64},
            })
        return content

    def _call_api(
        self,
        content: list[dict],
        endpoint: str,
        jd_text: str,
        resume_text: str,
        task_id: str,
    ) -> dict:
        try:
            return self._call_api_with_tools(content, endpoint, task_id)
        except RuntimeError as exc:
            if not _should_fallback_to_compat(exc):
                raise
            logger.warning(
                "[%s] Claude tool mode failed on relay, falling back to compat mode: %s",
                task_id,
                exc,
            )
            return self._call_api_via_compat(jd_text, resume_text, task_id)

    def _call_api_with_tools(self, content: list[dict], endpoint: str, task_id: str) -> dict:
        from services import task_service

        payload = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 4096,
            "system": _SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": content}],
            "tools": [{
                "name": "submit_resume_review",
                "description": "提交岗位定向简历批注结果，必须覆盖所有 required 字段。",
                "input_schema": MATCH_RESPONSE_SCHEMA,
            }],
            "tool_choice": {"type": "tool", "name": "submit_resume_review"},
        }
        headers = _get_claude_headers()
        data = _http_post(
            endpoint,
            headers,
            payload,
            on_rate_limit_retry=lambda message: task_service.set_status_message(task_id, message),
        )
        return self._extract_tool_result(data)

    def _call_api_via_compat(self, jd_text: str, resume_text: str, task_id: str) -> dict:
        from services import task_service

        headers = {
            "Authorization": f"Bearer {ANTHROPIC_API_KEY}",
            "Content-Type": "application/json",
        }
        for lite in (False, True):
            prompt = _build_compat_prompt(
                jd_text=jd_text,
                resume_text=resume_text,
                lite=lite,
            )
            payload = {
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": COMPAT_MAX_TOKENS_LITE if lite else COMPAT_MAX_TOKENS,
            }
            try:
                data = _http_post(
                    f"{OPENAI_BASE_URL}/v1/chat/completions",
                    headers,
                    payload,
                    on_rate_limit_retry=lambda message: task_service.set_status_message(task_id, message),
                )
                result = OpenAIProvider._extract_result(data, task_id)
                return _normalize_review_result(result)
            except RuntimeError as exc:
                if lite or not _should_fallback_to_compat(exc):
                    raise
                logger.warning(
                    "[%s] Compat mode failed on relay, retrying with lite prompt: %s",
                    task_id,
                    exc,
                )
        raise RuntimeError("兼容模式调用失败")

    @staticmethod
    def _extract_tool_result(response: dict) -> dict:
        for block in response.get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == "submit_resume_review":
                result = block.get("input", {})
                if not isinstance(result, dict):
                    raise ValueError(f"Tool input is not a dict: {type(result)}")
                return result
        raise ValueError(
            f"No tool_use block in Anthropic response. "
            f"stop_reason={response.get('stop_reason')}, "
            f"content_types={[b.get('type') for b in response.get('content', [])]}\n"
            f"Full response: {json.dumps(response, ensure_ascii=False)[:2000]}"
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def get_provider() -> AIProvider:
    """Return the active AI provider based on AI_PROVIDER env var."""
    if AI_PROVIDER == "claude":
        logger.info("Using Claude provider (model=%s, base=%s)", ANTHROPIC_MODEL, ANTHROPIC_BASE_URL)
        return ClaudeProvider()
    elif AI_PROVIDER == "openai":
        logger.info(
            "Using OpenAI provider (model=%s, base=%s)",
            OPENAI_MODEL, OPENAI_BASE_URL,
        )
        return OpenAIProvider()
    else:
        raise ValueError(
            f"Unknown AI_PROVIDER={AI_PROVIDER!r}. Valid values: openai, claude"
        )


def _should_fallback_to_compat(exc: RuntimeError) -> bool:
    msg = str(exc)
    return (
        "中转服务提前断开了连接" in msg
        or "Remote end closed connection without response" in msg
    )


def _build_compat_prompt(jd_text: str, resume_text: str, lite: bool = False) -> str:
    jd_limit = COMPAT_JD_TEXT_LIMIT_LITE if lite else COMPAT_JD_TEXT_LIMIT
    resume_limit = COMPAT_RESUME_TEXT_LIMIT_LITE if lite else COMPAT_RESUME_TEXT_LIMIT
    jd_excerpt = jd_text.strip()[:jd_limit]
    resume_excerpt = resume_text.strip()[:resume_limit]
    return (
        "你是一名简历顾问。只返回合法 JSON，不要解释，不要 markdown。\n"
        "JSON 必须包含以下字段：summary, jd_keywords, jd_hard_skills, jd_soft_skills, "
        "jd_responsibilities, match_assessment, strengths, weaknesses, issues。\n"
        "match_assessment 必须包含 keyword_coverage, professionalism, clarity, fit，取值只能是 high/medium/low。\n"
        "issues 最多 5 条。每条 issue 必须包含 id, page, title, severity, category, star_gap, comment, "
        "rewrite_tip, focus_text, anchor。anchor 必须包含 x,y,w,h，使用 0-1000 坐标。"
        "如果无法精确定位，请使用 page=1，anchor={x:80,y:120,w:840,h:120} 一类默认值。\n\n"
        "title 控制在 4-10 个汉字。comment 每条尽量 18-36 个汉字，rewrite_tip 尽量 16-36 个汉字。"
        "风格参考真人老师改简历：有判断、有改法，但不要长篇解释。不要使用英文单引号。\n\n"
        f"【岗位JD】\n{jd_excerpt}\n\n"
        f"【简历文本】\n{resume_excerpt}"
    )


def _sanitize_note_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    while "'" in text:
        updated = text.replace("'", '"')
        if updated == text:
            break
        text = updated
    result_chars: list[str] = []
    quote_open = True
    for char in text:
        if char == '"':
            result_chars.append("“" if quote_open else "”")
            quote_open = not quote_open
        else:
            result_chars.append(char)
    return " ".join("".join(result_chars).split())


def _coerce_json_text(content: str) -> str:
    text = str(content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _normalize_review_result(raw: dict[str, Any]) -> dict[str, Any]:
    assessment = raw.get("match_assessment")
    if not isinstance(assessment, dict):
        assessment = {}

    normalized_assessment = {}
    for key in ["keyword_coverage", "professionalism", "clarity", "fit"]:
        level = str(assessment.get(key, "low")).lower()
        normalized_assessment[key] = level if level in {"high", "medium", "low"} else "low"

    issues = raw.get("issues")
    if not isinstance(issues, list):
        issues = []

    normalized_issues = []
    for index, issue in enumerate(issues[:8], start=1):
        if not isinstance(issue, dict):
            continue
        anchor = issue.get("anchor")
        if not isinstance(anchor, dict):
            anchor = {}
        normalized_issues.append(
            {
                "id": str(issue.get("id") or f"ISSUE-{index}"),
                "page": max(1, int(issue.get("page", 1) or 1)),
                "title": _sanitize_note_text(issue.get("title") or f"问题 {index}"),
                "severity": _normalize_severity(issue.get("severity")),
                "category": str(issue.get("category") or "general"),
                "star_gap": _normalize_star_gap(issue.get("star_gap")),
                "comment": _sanitize_note_text(issue.get("comment") or "需要进一步改写。"),
                "rewrite_tip": _sanitize_note_text(issue.get("rewrite_tip") or issue.get("rewrite_example") or "改成更贴近岗位的表达。"),
                "rewrite_example": _sanitize_note_text(issue.get("rewrite_example") or ""),
                "focus_text": _sanitize_note_text(issue.get("focus_text") or ""),
                "padding": float(issue.get("padding", 18) or 18),
                "ocr_max_lines": float(issue.get("ocr_max_lines", 3) or 3),
                "ocr_enabled": bool(issue.get("ocr_enabled", True)),
                "anchor": {
                    "x": _clamp_anchor_value(anchor.get("x", 80)),
                    "y": _clamp_anchor_value(anchor.get("y", 120)),
                    "w": _clamp_anchor_value(anchor.get("w", 840)),
                    "h": _clamp_anchor_value(anchor.get("h", 120)),
                },
            }
        )

    return {
        "summary": str(raw.get("summary") or "已生成简历与岗位匹配分析。"),
        "jd_keywords": _ensure_string_list(raw.get("jd_keywords")),
        "jd_hard_skills": _ensure_string_list(raw.get("jd_hard_skills")),
        "jd_soft_skills": _ensure_string_list(raw.get("jd_soft_skills")),
        "jd_responsibilities": _ensure_string_list(raw.get("jd_responsibilities")),
        "match_assessment": normalized_assessment,
        "strengths": _ensure_string_list(raw.get("strengths")),
        "weaknesses": _ensure_string_list(raw.get("weaknesses")),
        "consultation_guide": raw.get("consultation_guide", {}),
        "issues": normalized_issues,
    }


def _ensure_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_severity(value: Any) -> str:
    normalized = str(value or "medium").lower()
    return normalized if normalized in {"low", "medium", "high"} else "medium"


def _normalize_star_gap(value: Any) -> str:
    normalized = str(value or "STAR").upper()
    return normalized if normalized in {"S", "T", "A", "R", "STAR"} else "STAR"


def _clamp_anchor_value(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1000.0, numeric))
