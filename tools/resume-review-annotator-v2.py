#!/usr/bin/env python3
"""
Resume review annotator MVP.

Two modes:
1. review: call the OpenAI Responses API with one resume image and get JSON issues
2. render: draw red circles and note boxes from a JSON file onto the image

This script avoids third-party API SDK dependencies. It only requires Pillow for rendering.
"""

from __future__ import annotations

import argparse
import base64
import csv
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

from PIL import Image, ImageDraw, ImageFont


# Base URL is configurable via OPENAI_BASE_URL (or legacy OPENAI_API_BASE).
# Strip trailing /v1 so the path can be appended cleanly.
_OPENAI_BASE = (
    os.getenv("OPENAI_BASE_URL")
    or os.getenv("OPENAI_API_BASE")
    or "https://api.openai.com"
).rstrip("/")
if _OPENAI_BASE.endswith("/v1"):
    _OPENAI_BASE = _OPENAI_BASE[:-3]
API_URL = f"{_OPENAI_BASE}/v1/responses"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
JSON_SCHEMA_NAME = "resume_review"
SUPPORTED_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
OCR_CACHE: dict[tuple[str, int, str], list["OCRLine"]] = {}
OCR_WORD_CACHE: dict[tuple[str, int, str], list["OCRWord"]] = {}
PDF_TEXT_CACHE: dict[str, tuple[dict[int, tuple[float, float]], dict[int, list["PDFLine"]]]] = {}


REVIEW_PROMPT = """
你是一名资深招聘顾问和简历教练。
请审阅用户上传的简历图片，从招聘筛选、表达清晰度、量化成果、STAR 完整性四个角度进行批注。

要求：
1. 只返回符合 schema 的 JSON。
2. 最多返回 8 条问题，优先挑最影响通过率的问题。
3. 每条问题必须绑定到图片中的具体区域。
3.1 区域必须尽量精确，优先框具体一句话、一个 bullet、一个小标题或一个字段，不要一口气框整大段。
4. comment 和 rewrite_tip 使用简体中文。
5. star_gap 只能是 S、T、A、R 或 STAR。
6. 不要编造图片里不存在的内容。
7. 优先指出：
   - 没有结果和量化指标
   - 只有职责，没有动作
   - 术语堆砌但业务价值不清楚
   - 格式拥挤或可读性差
8. anchor 使用 0-1000 归一化坐标系，字段为 x、y、w、h。
9. 批注语言像真人改简历，直接、自然、短，不要像系统提示词。
10. 每条 comment 和 rewrite_tip 尽量控制在 8-20 个汉字左右，短而直接。

summary 给整体判断。
strengths 给 2-4 条优点。
weaknesses 给 2-4 条缺点。
issues 里每条问题要短、准、可修改。
""".strip()


MATCH_REVIEW_PROMPT = """
你是一名资深招聘顾问、简历教练和真实做过大学生/留学生求职辅导的老师。
你的任务不是泛泛点评，而是结合岗位 JD 和候选人简历，输出一份“岗位定向简历批注结果”。

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
5. 从 JD 中提取 10-15 个最重要的关键词，并拆分为：
   - 硬技能
   - 软技能
   - 职责要求
6. 对简历做 4 个维度评级：
   - keyword_coverage
   - professionalism
   - clarity
   - fit
   每项只能是 high / medium / low。
7. issues 最多返回 8 条，但优先保留最影响通过率的问题。
8. 每条 issue 必须：
   - 绑定到简历中的具体页码和具体区域
   - 优先框一句话、一个 bullet、一个字段，不要大面积圈整块
   - 给出具体问题
   - 给出可落地的改法
   - 给出一句可直接参考的改写方向
9. focus_text 必须尽量引用被批注区域中的原始文本，方便后续 OCR 精准定位。
10. anchor 使用 0-1000 归一化坐标系，字段为 x、y、w、h。
11. page 从 1 开始计数。
12. 不要编造简历里没有的经历、数据或结果。

请优先指出这些问题：
- 与 JD 关键词不对齐
- 只有职责，没有动作
- 只有动作，没有结果
- 没有量化指标
- 项目/实习价值说不清
- 顶部定位不聚焦
- 教育/课程/技能堆砌但无岗位价值
- 模板或版式影响筛选效率

行业/专业批注偏好：
- 商科/运营/市场/咨询方向：优先看业务结果、分析方法、数据洞察、岗位相关性。
- 理工科/算法/开发/工程方向：优先看技术动作、工具链、性能指标、项目难度与落地结果。
- 大学生/留学生简历：不要用社招口径要求“完整业务 ownership”，但要尽量把校园项目、实习、课程项目写得像真实产出。

comment 和 rewrite_tip 要短、准、像老师直接说的话。
title 建议 4-8 个汉字。

除上述批注外，请额外补一段适合放在报告尾部的 `consultation_guide`，用于引导后续 1v1 深度咨询：
- 要说明为什么这份简历值得继续深改
- 要明确 1v1 最值得优先解决的 3-5 个方向
- 要告诉候选人下一轮需要补什么素材
- 口吻仍然像真人老师，不要像营销文案
""".strip()


RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
        },
        "weaknesses": {
            "type": "array",
            "items": {"type": "string"},
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "category": {"type": "string"},
                    "star_gap": {
                        "type": "string",
                        "enum": ["S", "T", "A", "R", "STAR"],
                    },
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
                "required": [
                    "id",
                    "title",
                    "severity",
                    "category",
                    "star_gap",
                    "comment",
                    "rewrite_tip",
                    "focus_text",
                    "anchor",
                ],
            },
        },
    },
    "required": ["summary", "strengths", "weaknesses", "issues"],
}


MATCH_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
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
            "required": [
                "recommended",
                "headline",
                "summary",
                "reasons",
                "session_focus",
                "prep_items",
                "cta",
            ],
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "page": {"type": "integer", "minimum": 1},
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
                "required": [
                    "id",
                    "page",
                    "title",
                    "severity",
                    "category",
                    "star_gap",
                    "comment",
                    "rewrite_tip",
                    "focus_text",
                    "anchor",
                ],
            },
        },
    },
    "required": [
        "summary",
        "jd_keywords",
        "jd_hard_skills",
        "jd_soft_skills",
        "jd_responsibilities",
        "match_assessment",
        "strengths",
        "weaknesses",
        "issues",
    ],
}


@dataclass
class Anchor:
    x: float
    y: float
    w: float
    h: float


@dataclass
class OCRLine:
    text: str
    left: int
    top: int
    width: int
    height: int


@dataclass
class OCRWord:
    text: str
    left: int
    top: int
    width: int
    height: int
    line_key: tuple[str, str, str]


@dataclass
class PDFLine:
    text: str
    left: float
    top: float
    width: float
    height: float


@dataclass
class Rect:
    left: int
    top: int
    right: int
    bottom: int


def load_font(size: int, family: str = "sans") -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    if family == "serif":
        candidates = [
            ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", 2),
            ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 2),
            ("/usr/share/fonts/truetype/noto/NotoSerifCJK-Regular.ttc", 2),
            ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 2),
            ("/System/Library/Fonts/Supplemental/Songti.ttc", None),
            ("/System/Library/Fonts/Songti.ttc", None),
            ("/System/Library/Fonts/Supplemental/Times New Roman.ttf", None),
            ("/System/Library/Fonts/Supplemental/Georgia.ttf", None),
            ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", None),
        ]
    else:
        candidates = [
            ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 2),
            ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", 2),
            ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 2),
            ("/usr/share/fonts/truetype/noto/NotoSerifCJK-Regular.ttc", 2),
            ("/System/Library/Fonts/PingFang.ttc", None),
            ("/System/Library/Fonts/Supplemental/PingFang.ttc", None),
            ("/System/Library/Fonts/Supplemental/Avenir Next.ttc", None),
            ("/System/Library/Fonts/Supplemental/HelveticaNeue.ttc", None),
            ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", None),
            ("/Library/Fonts/Arial Unicode.ttf", None),
        ]
    for path, index in candidates:
        if os.path.exists(path):
            try:
                kwargs = {"size": size}
                if index is not None:
                    kwargs["index"] = index
                return ImageFont.truetype(path, **kwargs)
            except OSError:
                pass
    return ImageFont.load_default()


def sample_panel_background(image: Image.Image) -> tuple[int, int, int]:
    width, height = image.size
    x_start = max(0, int(width * 0.72))
    x_end = max(x_start + 1, width - 10)
    y_start = max(0, int(height * 0.08))
    y_end = max(y_start + 1, int(height * 0.30))

    pixels: list[tuple[int, int, int]] = []
    step_x = max(1, (x_end - x_start) // 24)
    step_y = max(1, (y_end - y_start) // 18)

    for x in range(x_start, x_end, step_x):
        for y in range(y_start, y_end, step_y):
            r, g, b = image.getpixel((x, y))
            # Prefer bright neutral colors from blank resume areas.
            if min(r, g, b) >= 220 and max(r, g, b) - min(r, g, b) <= 25:
                pixels.append((r, g, b))

    if not pixels:
        return (255, 255, 255)

    avg_r = sum(pixel[0] for pixel in pixels) // len(pixels)
    avg_g = sum(pixel[1] for pixel in pixels) // len(pixels)
    avg_b = sum(pixel[2] for pixel in pixels) // len(pixels)
    return (avg_r, avg_g, avg_b)


def encode_image(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    media_type = SUPPORTED_IMAGE_TYPES.get(suffix)
    if not media_type:
        raise ValueError(f"Unsupported image type: {path.suffix}")
    data = path.read_bytes()
    return media_type, base64.b64encode(data).decode("utf-8")


def build_review_payload(image_path: Path, model: str) -> dict[str, Any]:
    media_type, encoded = encode_image(image_path)
    return {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": REVIEW_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "请分析这张简历图片，并输出结构化批注结果。",
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{encoded}",
                    },
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": JSON_SCHEMA_NAME,
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            }
        },
    }


def build_match_review_payload(
    jd_image_path: Path,
    jd_text: str,
    resume_text: str,
    resume_image_paths: list[Path],
    model: str,
) -> dict[str, Any]:
    jd_media_type, jd_encoded = encode_image(jd_image_path)
    user_content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "请结合岗位 JD 与候选人简历，输出岗位定向简历批注结果。\n\n"
                "以下 OCR/提取文本可能存在少量误差，请以图片和 PDF 页面内容为准。\n\n"
                f"【岗位 JD OCR 文本】\n{jd_text}\n\n"
                f"【候选人简历提取文本】\n{resume_text}"
            ),
        },
        {"type": "input_text", "text": "下面是岗位 JD 截图："},
        {"type": "input_image", "image_url": f"data:{jd_media_type};base64,{jd_encoded}"},
    ]

    for page_index, page_path in enumerate(resume_image_paths, start=1):
        page_media_type, page_encoded = encode_image(page_path)
        user_content.append({"type": "input_text", "text": f"下面是候选人简历第 {page_index} 页图片："})
        user_content.append({"type": "input_image", "image_url": f"data:{page_media_type};base64,{page_encoded}"})

    return {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": MATCH_REVIEW_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "resume_job_match_review",
                "strict": True,
                "schema": MATCH_RESPONSE_SCHEMA,
            }
        },
    }


def call_responses_api(payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI API request failed: {exc}") from exc


def extract_output_json(response_data: dict[str, Any]) -> dict[str, Any]:
    for item in response_data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return json.loads(content["text"])
            if content.get("type") == "output_json" and content.get("json"):
                return content["json"]
    raise ValueError("No JSON output found in Responses API result.")


def collapse_whitespace(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r", "\n").splitlines()]
    compact: list[str] = []
    blank_pending = False
    for line in lines:
        if not line:
            if compact and not blank_pending:
                compact.append("")
            blank_pending = True
            continue
        compact.append(line)
        blank_pending = False
    return "\n".join(compact).strip()


def require_binary(binary_name: str) -> str:
    resolved = shutil.which(binary_name)
    if not resolved:
        raise RuntimeError(f"Required binary not found in PATH: {binary_name}")
    return resolved


def ocr_plain_text(image_path: Path, psm: int = 6, languages: str = "chi_sim+eng") -> str:
    require_binary("tesseract")
    try:
        raw = subprocess.check_output(
            [
                "tesseract",
                str(image_path),
                "stdout",
                "--psm",
                str(psm),
                "-l",
                languages,
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Tesseract OCR failed for {image_path}: {exc}") from exc
    return collapse_whitespace(raw)


def convert_pdf_to_images(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 180,
    max_pages: int | None = None,
) -> list[Path]:
    require_binary("pdftoppm")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "resume_page"
    cmd = ["pdftoppm", "-png", "-r", str(dpi)]
    if max_pages and max_pages > 0:
        cmd += ["-f", "1", "-l", str(max_pages)]
    cmd += [str(pdf_path), str(prefix)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"pdftoppm failed for {pdf_path}: {exc}") from exc
    images = sorted(output_dir.glob("resume_page-*.png"))
    if not images:
        raise RuntimeError(f"No page images were generated from PDF: {pdf_path}")
    return images


def extract_text_from_pdf(pdf_path: Path) -> str:
    require_binary("pdftotext")
    try:
        raw = subprocess.check_output(
            ["pdftotext", str(pdf_path), "-"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"pdftotext failed for {pdf_path}: {exc}") from exc
    return collapse_whitespace(raw)


def extract_resume_text_with_fallback(pdf_path: Path, page_images: list[Path]) -> tuple[str, str]:
    pdf_text = extract_text_from_pdf(pdf_path)
    if len(normalize_for_match(pdf_text)) >= 120:
        return pdf_text, "pdftotext"

    ocr_chunks = []
    for image_path in page_images:
        text = ocr_plain_text(image_path, psm=6)
        if text:
            ocr_chunks.append(text)
    ocr_text = collapse_whitespace("\n\n".join(ocr_chunks))
    if len(normalize_for_match(ocr_text)) > len(normalize_for_match(pdf_text)):
        return ocr_text, "ocr_fallback"
    return pdf_text, "pdftotext_weak"


def summarize_input_meta(
    jd_image_path: Path,
    resume_pdf_path: Path,
    page_images: list[Path],
    jd_text_source: str,
    resume_text_source: str,
) -> dict[str, Any]:
    return {
        "mode": "resume_job_match_v2",
        "jd_image": str(jd_image_path),
        "resume_pdf": str(resume_pdf_path),
        "resume_page_count": len(page_images),
        "jd_text_source": jd_text_source,
        "resume_text_source": resume_text_source,
    }


def coerce_match_review_pages(review: dict[str, Any], page_count: int) -> dict[str, Any]:
    issues = review.get("issues", [])
    coerced: list[dict[str, Any]] = []
    for index, issue in enumerate(issues, start=1):
        fixed = dict(issue)
        page = int(fixed.get("page", 1) or 1)
        if page < 1:
            page = 1
        if page_count > 0:
            page = min(page, page_count)
        fixed["page"] = page
        fixed.setdefault("id", f"issue_{index}")
        fixed.setdefault("focus_text", "")
        coerced.append(fixed)
    review["issues"] = coerced
    return review


def translate_assessment_label(value: str) -> str:
    return {"high": "高", "medium": "中", "low": "低"}.get(value, value or "未评估")


def build_consultation_guide(review: dict[str, Any]) -> dict[str, Any]:
    existing = review.get("consultation_guide")
    if isinstance(existing, dict):
        reasons = [str(item).strip() for item in existing.get("reasons", []) if str(item).strip()]
        session_focus = [str(item).strip() for item in existing.get("session_focus", []) if str(item).strip()]
        prep_items = [str(item).strip() for item in existing.get("prep_items", []) if str(item).strip()]
        headline = str(existing.get("headline", "")).strip()
        summary = str(existing.get("summary", "")).strip()
        cta = str(existing.get("cta", "")).strip()
        if headline and summary and cta and reasons and session_focus and prep_items:
            return {
                "recommended": bool(existing.get("recommended", True)),
                "headline": headline,
                "summary": summary,
                "reasons": reasons[:4],
                "session_focus": session_focus[:5],
                "prep_items": prep_items[:5],
                "cta": cta,
            }

    assessment = review.get("match_assessment", {})
    issues = review.get("issues", [])
    top_issues = issues[:5]
    high_issue_count = sum(1 for issue in issues if str(issue.get("severity", "")).lower() == "high")
    low_dimensions = sum(
        1
        for key in ("keyword_coverage", "professionalism", "clarity", "fit")
        if str(assessment.get(key, "")).lower() == "low"
    )
    recommended = low_dimensions >= 2 or high_issue_count >= 2 or len(issues) >= 4

    reasons: list[str] = []
    summary = str(review.get("summary", "")).strip()
    if summary:
        reasons.append(summary)

    fit_label = translate_assessment_label(str(assessment.get("fit", "")).lower())
    coverage_label = translate_assessment_label(str(assessment.get("keyword_coverage", "")).lower())
    if assessment:
        reasons.append(f"当前岗位匹配度为{fit_label}，关键词覆盖率为{coverage_label}，还没到可直接投递的状态。")

    for issue in top_issues[:2]:
        title = str(issue.get("title", "")).strip()
        comment = str(issue.get("comment", "")).strip()
        if title and comment:
            reasons.append(f"{title}：{comment}")
    reasons = reasons[:4] or ["这份简历还有明显可提升空间，建议先深改再投递。"]

    category_focus = {
        "job_target": "重写顶部求职定位与摘要，让岗位入口先对齐 JD。",
        "relevance": "筛出最贴岗的经历，按 JD 关键词重排并重写表述。",
        "impact": "把职责改成结果导向表达，补齐量化指标和业务结果。",
        "summary": "把空泛自评改成岗位标签 + 可验证能力，不再堆形容词。",
        "content_priority": "调整内容优先级，把最能打动岗位的项目和经历前置。",
        "layout": "压缩弱相关信息，给核心经历和成果留出版面。",
        "wording": "替换弱动词和空泛措辞，让动作和贡献更清楚。",
        "experience": "把项目/实习经历改成任务、动作、结果更完整的写法。",
        "skills": "技能区只保留对目标岗真正有用的内容，并补足使用场景。",
    }
    session_focus: list[str] = []
    for issue in top_issues:
        category = str(issue.get("category", "")).strip()
        rewrite_tip = str(issue.get("rewrite_tip", "")).strip()
        if category in category_focus:
            session_focus.append(category_focus[category])
        elif rewrite_tip:
            session_focus.append(rewrite_tip)
    if not session_focus:
        session_focus = [
            "先统一目标岗位版本，避免一份简历同时服务多个方向。",
            "逐段补齐动作、结果和岗位关键词，至少把高风险问题先修掉。",
            "把最贴岗的经历、项目和成果重新排序，提升首屏通过率。",
        ]
    dedup_focus: list[str] = []
    for item in session_focus:
        if item not in dedup_focus:
            dedup_focus.append(item)
    session_focus = dedup_focus[:5]

    jd_keywords = [str(item).strip() for item in review.get("jd_keywords", []) if str(item).strip()]
    prep_items = [
        "准备目标 JD 原文，并标出你最想主攻的 1 个岗位版本。",
        "把每段经历能补的数字先列出来，例如人数、场次、效率、结果、覆盖范围。",
        "补充 2-3 段最想保留经历的真实背景：你具体做了什么、难点是什么、最终结果是什么。",
    ]
    if jd_keywords:
        prep_items.append(f"优先确认这些关键词里你真实具备哪些：{'、'.join(jd_keywords[:6])}。")
    prep_items = prep_items[:4]

    headline = "建议继续做 1v1 深度改稿" if recommended else "可按当前建议先自行修改一轮"
    guide_summary = (
        "这份简历的主要问题不是单句润色，而是岗位定位、经历取舍和成果表达需要一起重构。"
        if recommended
        else "这份简历已经有基础，可先按批注自行修改，再决定是否做 1v1 深挖。"
    )
    cta = (
        "如果继续做 1v1，建议下一轮直接围绕目标岗位版本、3段核心经历重写和量化结果补齐来做。"
        if recommended
        else "建议先按本次批注改完一版，再用新版本复查是否还存在明显错位。"
    )
    return {
        "recommended": recommended,
        "headline": headline,
        "summary": guide_summary,
        "reasons": reasons,
        "session_focus": session_focus,
        "prep_items": prep_items,
        "cta": cta,
    }


def enrich_match_review(review: dict[str, Any], page_count: int) -> dict[str, Any]:
    review = coerce_match_review_pages(review, page_count)
    review["consultation_guide"] = build_consultation_guide(review)
    return review


def build_report_markdown(review: dict[str, Any]) -> str:
    assessment = review.get("match_assessment", {})
    lines = [
        "# 岗位定向简历批注报告",
        "",
        "## 总评",
        review.get("summary", "").strip() or "暂无总评。",
        "",
    ]

    if review.get("jd_keywords"):
        lines += [
            "## JD 关键词",
            f"- 关键词：{'、'.join(review.get('jd_keywords', []))}",
            f"- 硬技能：{'、'.join(review.get('jd_hard_skills', [])) or '暂无'}",
            f"- 软技能：{'、'.join(review.get('jd_soft_skills', [])) or '暂无'}",
            f"- 核心职责：{'、'.join(review.get('jd_responsibilities', [])) or '暂无'}",
            "",
        ]

    if assessment:
        lines += [
            "## 匹配度评估",
            f"- 关键词覆盖率：{translate_assessment_label(assessment.get('keyword_coverage', ''))}",
            f"- 专业性：{translate_assessment_label(assessment.get('professionalism', ''))}",
            f"- 清晰度：{translate_assessment_label(assessment.get('clarity', ''))}",
            f"- 匹配度：{translate_assessment_label(assessment.get('fit', ''))}",
            "",
        ]

    strengths = review.get("strengths", [])
    if strengths:
        lines += ["## 优点", *[f"- {item}" for item in strengths], ""]

    weaknesses = review.get("weaknesses", [])
    if weaknesses:
        lines += ["## 短板", *[f"- {item}" for item in weaknesses], ""]

    issues = review.get("issues", [])
    if issues:
        lines += ["## 逐条批注", ""]
        for index, issue in enumerate(issues, start=1):
            lines += [
                f"### {index}. {issue.get('title', '未命名问题')}",
                f"- 页码：第 {issue.get('page', 1)} 页",
                f"- 看这里：{issue.get('focus_text', '').strip() or '未提供'}",
                f"- 问题：{issue.get('comment', '').strip() or '未提供'}",
                f"- 修改原则：{issue.get('rewrite_tip', '').strip() or '未提供'}",
                f"- 参考改写：{issue.get('rewrite_example', '').strip() or '未提供'}",
                "",
            ]

    guide = build_consultation_guide(review)
    if guide:
        lines += [
            "## 1v1 深度咨询建议",
            guide.get("headline", "").strip() or "建议继续深改",
            "",
            guide.get("summary", "").strip() or "",
            "",
            "### 为什么建议继续",
            *[f"- {item}" for item in guide.get("reasons", [])],
            "",
            "### 1v1 优先处理",
            *[f"- {item}" for item in guide.get("session_focus", [])],
            "",
            "### 你下一轮先补这些素材",
            *[f"- {item}" for item in guide.get("prep_items", [])],
            "",
            f"### 下一步\n{guide.get('cta', '').strip() or '建议按本次批注继续优化。'}",
            "",
        ]

    meta = review.get("_meta")
    if isinstance(meta, dict):
        lines += [
            "## 生成信息",
            f"- JD 文件：`{meta.get('jd_image', '')}`",
            f"- 简历文件：`{meta.get('resume_pdf', '')}`",
            f"- 简历页数：{meta.get('resume_page_count', '')}",
            f"- JD 文本来源：`{meta.get('jd_text_source', '')}`",
            f"- 简历文本来源：`{meta.get('resume_text_source', '')}`",
            "",
        ]

    return "\n".join(lines).strip() + "\n"


def build_report_text(review: dict[str, Any]) -> str:
    assessment = review.get("match_assessment", {})
    lines = [
        "岗位定向简历批注报告",
        "",
        "总评",
        review.get("summary", "").strip() or "暂无总评。",
        "",
    ]

    if review.get("jd_keywords"):
        lines += [
            "JD 关键词",
            f"关键词：{'、'.join(review.get('jd_keywords', []))}",
            f"硬技能：{'、'.join(review.get('jd_hard_skills', [])) or '暂无'}",
            f"软技能：{'、'.join(review.get('jd_soft_skills', [])) or '暂无'}",
            f"核心职责：{'、'.join(review.get('jd_responsibilities', [])) or '暂无'}",
            "",
        ]

    if assessment:
        lines += [
            "匹配度评估",
            f"关键词覆盖率：{translate_assessment_label(assessment.get('keyword_coverage', ''))}",
            f"专业性：{translate_assessment_label(assessment.get('professionalism', ''))}",
            f"清晰度：{translate_assessment_label(assessment.get('clarity', ''))}",
            f"匹配度：{translate_assessment_label(assessment.get('fit', ''))}",
            "",
        ]

    strengths = review.get("strengths", [])
    if strengths:
        lines += ["优点", *[f"- {item}" for item in strengths], ""]

    weaknesses = review.get("weaknesses", [])
    if weaknesses:
        lines += ["短板", *[f"- {item}" for item in weaknesses], ""]

    issues = review.get("issues", [])
    if issues:
        lines += ["逐条批注", ""]
        for index, issue in enumerate(issues, start=1):
            lines += [
                f"{index}. {issue.get('title', '未命名问题')}",
                f"页码：第 {issue.get('page', 1)} 页",
                f"看这里：{issue.get('focus_text', '').strip() or '未提供'}",
                f"问题：{issue.get('comment', '').strip() or '未提供'}",
                f"修改原则：{issue.get('rewrite_tip', '').strip() or '未提供'}",
                f"参考改写：{issue.get('rewrite_example', '').strip() or '未提供'}",
                "",
            ]

    guide = build_consultation_guide(review)
    if guide:
        lines += [
            "1v1 深度咨询建议",
            guide.get("headline", "").strip() or "建议继续深改",
            "",
            guide.get("summary", "").strip() or "",
            "",
            "为什么建议继续",
            *[f"- {item}" for item in guide.get("reasons", [])],
            "",
            "1v1 优先处理",
            *[f"- {item}" for item in guide.get("session_focus", [])],
            "",
            "你下一轮先补这些素材",
            *[f"- {item}" for item in guide.get("prep_items", [])],
            "",
            "下一步",
            guide.get("cta", "").strip() or "建议按本次批注继续优化。",
            "",
        ]

    meta = review.get("_meta")
    if isinstance(meta, dict):
        lines += [
            "生成信息",
            f"JD 文件：{meta.get('jd_image', '')}",
            f"简历文件：{meta.get('resume_pdf', '')}",
            f"简历页数：{meta.get('resume_page_count', '')}",
            f"JD 文本来源：{meta.get('jd_text_source', '')}",
            f"简历文本来源：{meta.get('resume_text_source', '')}",
            "",
        ]

    return "\n".join(lines).strip() + "\n"


def write_simple_docx(text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paragraphs = text.splitlines()
    xml_paragraphs: list[str] = []
    for paragraph in paragraphs:
        if paragraph:
            xml_paragraphs.append(
                "<w:p><w:r><w:t xml:space=\"preserve\">"
                + xml_escape(paragraph)
                + "</w:t></w:r></w:p>"
            )
        else:
            xml_paragraphs.append("<w:p/>")

    document_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">"
        "<w:body>"
        + "".join(xml_paragraphs)
        + (
            "<w:sectPr>"
            "<w:pgSz w:w=\"11906\" w:h=\"16838\"/>"
            "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" "
            "w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
            "</w:sectPr>"
        )
        + "</w:body></w:document>"
    )

    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""

    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", rels_xml)
        archive.writestr("word/document.xml", document_xml)


def normalized_to_pixels(anchor: dict[str, Any], width: int, height: int) -> Anchor:
    return Anchor(
        x=(float(anchor["x"]) / 1000.0) * width,
        y=(float(anchor["y"]) / 1000.0) * height,
        w=(float(anchor["w"]) / 1000.0) * width,
        h=(float(anchor["h"]) / 1000.0) * height,
    )


def normalize_for_match(text: str) -> str:
    lowered = text.lower()
    filtered = []
    for ch in lowered:
        if ch.isalnum() or "\u4e00" <= ch <= "\u9fff":
            filtered.append(ch)
    return "".join(filtered)


def parse_pdf_text_layout(pdf_path: Path) -> tuple[dict[int, tuple[float, float]], dict[int, list[PDFLine]]]:
    cache_key = str(pdf_path)
    if cache_key in PDF_TEXT_CACHE:
        return PDF_TEXT_CACHE[cache_key]

    require_binary("pdftotext")
    raw = subprocess.check_output(
        ["pdftotext", "-tsv", str(pdf_path), "-"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    rows = list(csv.DictReader(StringIO(raw), delimiter="\t"))

    page_sizes: dict[int, tuple[float, float]] = {}
    line_words: dict[tuple[int, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        level = row.get("level")
        if level == "1" and row.get("text") == "###PAGE###":
            page_num = int(row["page_num"])
            page_sizes[page_num] = (float(row["width"]), float(row["height"]))
            continue
        if level != "5":
            continue
        text = (row.get("text") or "").strip()
        if not text:
            continue
        key = (int(row["page_num"]), row["block_num"], row["par_num"], row["line_num"])
        line_words.setdefault(key, []).append(row)

    page_lines: dict[int, list[PDFLine]] = {}
    for (page_num, _block, _par, _line), words in line_words.items():
        words = sorted(words, key=lambda item: (float(item["left"]), float(item["top"])))
        left = min(float(item["left"]) for item in words)
        top = min(float(item["top"]) for item in words)
        right = max(float(item["left"]) + float(item["width"]) for item in words)
        bottom = max(float(item["top"]) + float(item["height"]) for item in words)
        page_lines.setdefault(page_num, []).append(
            PDFLine(
                text="".join(item["text"] for item in words),
                left=left,
                top=top,
                width=right - left,
                height=bottom - top,
            )
        )

    for page_num, lines in page_lines.items():
        lines.sort(key=lambda item: (item.top, item.left))
        page_lines[page_num] = lines

    PDF_TEXT_CACHE[cache_key] = (page_sizes, page_lines)
    return page_sizes, page_lines


def run_tesseract_lines(image_path: Path, psm: int = 6, languages: str = "chi_sim+eng") -> list[OCRLine]:
    cache_key = (str(image_path), psm, languages)
    if cache_key in OCR_CACHE:
        return OCR_CACHE[cache_key]
    raw = subprocess.check_output(
        [
            "tesseract",
            str(image_path),
            "stdout",
            "--psm",
            str(psm),
            "-l",
            languages,
            "tsv",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    rows = list(csv.DictReader(StringIO(raw), delimiter="\t"))
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("level") != "5":
            continue
        if not row.get("text", "").strip():
            continue
        key = (row["block_num"], row["par_num"], row["line_num"])
        grouped.setdefault(key, []).append(row)

    lines: list[OCRLine] = []
    words_cache: list[OCRWord] = []
    for words in grouped.values():
        words = sorted(words, key=lambda item: int(item["left"]))
        left = min(int(item["left"]) for item in words)
        top = min(int(item["top"]) for item in words)
        right = max(int(item["left"]) + int(item["width"]) for item in words)
        bottom = max(int(item["top"]) + int(item["height"]) for item in words)
        line_key = (words[0]["block_num"], words[0]["par_num"], words[0]["line_num"])
        for item in words:
            words_cache.append(
                OCRWord(
                    text=item["text"],
                    left=int(item["left"]),
                    top=int(item["top"]),
                    width=int(item["width"]),
                    height=int(item["height"]),
                    line_key=line_key,
                )
            )
        lines.append(
            OCRLine(
                text="".join(item["text"] for item in words),
                left=left,
                top=top,
                width=right - left,
                height=bottom - top,
            )
        )
    sorted_lines = sorted(lines, key=lambda item: (item.top, item.left))
    OCR_CACHE[cache_key] = sorted_lines
    OCR_WORD_CACHE[cache_key] = sorted(words_cache, key=lambda item: (item.top, item.left))
    return sorted_lines


def merge_ocr_lines(lines: list[OCRLine]) -> OCRLine:
    left = min(line.left for line in lines)
    top = min(line.top for line in lines)
    right = max(line.left + line.width for line in lines)
    bottom = max(line.top + line.height for line in lines)
    text = "".join(line.text for line in lines)
    return OCRLine(text=text, left=left, top=top, width=right - left, height=bottom - top)


def merge_ocr_words(words: list[OCRWord]) -> OCRLine:
    left = min(word.left for word in words)
    top = min(word.top for word in words)
    right = max(word.left + word.width for word in words)
    bottom = max(word.top + word.height for word in words)
    text = "".join(word.text for word in words)
    return OCRLine(text=text, left=left, top=top, width=right - left, height=bottom - top)


def merge_pdf_lines(lines: list[PDFLine]) -> PDFLine:
    left = min(line.left for line in lines)
    top = min(line.top for line in lines)
    right = max(line.left + line.width for line in lines)
    bottom = max(line.top + line.height for line in lines)
    text = "".join(line.text for line in lines)
    return PDFLine(text=text, left=left, top=top, width=right - left, height=bottom - top)


def rect_from_anchor(anchor: Anchor) -> Rect:
    return Rect(
        left=int(anchor.x),
        top=int(anchor.y),
        right=int(anchor.x + anchor.w),
        bottom=int(anchor.y + anchor.h),
    )


def rect_from_word(word: OCRWord) -> Rect:
    return Rect(
        left=word.left,
        top=word.top,
        right=word.left + word.width,
        bottom=word.top + word.height,
    )


def rect_intersection_area(a: Rect, b: Rect) -> int:
    left = max(a.left, b.left)
    top = max(a.top, b.top)
    right = min(a.right, b.right)
    bottom = min(a.bottom, b.bottom)
    if right <= left or bottom <= top:
        return 0
    return (right - left) * (bottom - top)


def rect_center_inside(a: Rect, b: Rect) -> bool:
    cx = (a.left + a.right) / 2
    cy = (a.top + a.bottom) / 2
    return b.left <= cx <= b.right and b.top <= cy <= b.bottom


def vertical_overlap(a: Rect, b: Rect) -> int:
    return max(0, min(a.bottom, b.bottom) - max(a.top, b.top))


def horizontal_overlap(a: Rect, b: Rect) -> int:
    return max(0, min(a.right, b.right) - max(a.left, b.left))


def expand_box_safely(
    image_path: Path,
    base_anchor: Anchor,
    width: int,
    height: int,
    desired_padding: int,
) -> tuple[int, int, int, int]:
    cache_key = (str(image_path), 6, "chi_sim+eng")
    run_tesseract_lines(image_path)
    words = OCR_WORD_CACHE.get(cache_key, [])

    base_rect = rect_from_anchor(base_anchor)
    target_words: list[OCRWord] = []
    for word in words:
        word_rect = rect_from_word(word)
        inter = rect_intersection_area(word_rect, base_rect)
        word_area = max(1, word.width * word.height)
        if inter / word_area >= 0.2 or rect_center_inside(word_rect, base_rect):
            target_words.append(word)

    if target_words:
        target_rect = Rect(
            left=min(word.left for word in target_words),
            top=min(word.top for word in target_words),
            right=max(word.left + word.width for word in target_words),
            bottom=max(word.top + word.height for word in target_words),
        )
    else:
        target_rect = base_rect

    side_gap = max(0, desired_padding)
    min_clearance = 3

    max_left_expand = side_gap
    max_right_expand = side_gap
    max_top_expand = side_gap
    max_bottom_expand = side_gap

    for word in words:
        word_rect = rect_from_word(word)
        if rect_intersection_area(word_rect, target_rect) > 0 or rect_center_inside(word_rect, target_rect):
            continue

        if vertical_overlap(word_rect, target_rect) >= max(6, min(word.height, target_rect.bottom - target_rect.top) // 3):
            if word_rect.right <= target_rect.left:
                max_left_expand = min(max_left_expand, max(0, target_rect.left - word_rect.right - min_clearance))
            if word_rect.left >= target_rect.right:
                max_right_expand = min(max_right_expand, max(0, word_rect.left - target_rect.right - min_clearance))

        if horizontal_overlap(word_rect, target_rect) >= max(12, min(word.width, target_rect.right - target_rect.left) // 4):
            if word_rect.bottom <= target_rect.top:
                max_top_expand = min(max_top_expand, max(0, target_rect.top - word_rect.bottom - min_clearance))
            if word_rect.top >= target_rect.bottom:
                max_bottom_expand = min(max_bottom_expand, max(0, word_rect.top - target_rect.bottom - min_clearance))

    x0 = max(0, target_rect.left - max_left_expand)
    y0 = max(0, target_rect.top - max_top_expand)
    x1 = min(width - 1, target_rect.right + max_right_expand)
    y1 = min(height - 1, target_rect.bottom + max_bottom_expand)

    x1 = max(x1, x0 + 2)
    y1 = max(y1, y0 + 2)
    return (x0, y0, x1, y1)


def expand_manual_box(
    anchor: Anchor,
    width: int,
    height: int,
    padding: int,
) -> tuple[int, int, int, int]:
    x0 = max(0, int(anchor.x - padding))
    y0 = max(0, int(anchor.y - padding))
    x1 = min(width - 1, int(anchor.x + anchor.w + padding))
    y1 = min(height - 1, int(anchor.y + anchor.h + padding))
    x1 = max(x1, x0 + 2)
    y1 = max(y1, y0 + 2)
    return (x0, y0, x1, y1)


def find_best_ocr_anchor(
    image_path: Path,
    focus_text: str,
    width: int,
    height: int,
    max_lines: int = 2,
) -> dict[str, float] | None:
    target = normalize_for_match(focus_text)
    if not target:
        return None

    ocr_lines = run_tesseract_lines(image_path)
    if not ocr_lines:
        return None
    cache_key = (str(image_path), 6, "chi_sim+eng")
    ocr_words = OCR_WORD_CACHE.get(cache_key, [])

    best_score = 0.0
    best_line: OCRLine | None = None
    limit = max(1, int(max_lines))
    max_words = 24

    for start in range(len(ocr_words)):
        line_keys: set[tuple[str, str, str]] = set()
        for span in range(1, max_words + 1):
            end = start + span
            if end > len(ocr_words):
                break
            chunk = ocr_words[start:end]
            line_keys = {word.line_key for word in chunk}
            if len(line_keys) > limit:
                break
            merged = merge_ocr_words(chunk)
            candidate = normalize_for_match(merged.text)
            if not candidate:
                continue
            score = difflib.SequenceMatcher(None, target, candidate).ratio()
            if target in candidate or candidate in target:
                score += 0.45
            overlap = len(set(target) & set(candidate)) / max(1, len(set(target)))
            score += overlap * 0.30
            score -= max(0, len(candidate) - len(target)) * 0.003
            if score > best_score:
                best_score = score
                best_line = merged

    for start in range(len(ocr_lines)):
        for span in range(1, limit + 1):
            end = start + span
            if end > len(ocr_lines):
                break
            merged = merge_ocr_lines(ocr_lines[start:end])
            candidate = normalize_for_match(merged.text)
            if not candidate:
                continue
            score = difflib.SequenceMatcher(None, target, candidate).ratio()
            if target in candidate or candidate in target:
                score += 0.35
            if len(candidate) > 0:
                overlap = len(set(target) & set(candidate)) / max(1, len(set(target)))
                score += overlap * 0.25
            score -= max(0, len(candidate) - len(target)) * 0.003
            if score > best_score:
                best_score = score
                best_line = merged

    if best_line is None or best_score < 0.45:
        return None

    return {
        "x": (best_line.left / width) * 1000.0,
        "y": (best_line.top / height) * 1000.0,
        "w": (best_line.width / width) * 1000.0,
        "h": (best_line.height / height) * 1000.0,
    }


def find_best_pdf_anchor(
    pdf_path: Path,
    page_number: int,
    focus_text: str,
    max_lines: int = 2,
) -> dict[str, float] | None:
    target = normalize_for_match(focus_text)
    if not target:
        return None

    page_sizes, page_lines = parse_pdf_text_layout(pdf_path)
    page_width, page_height = page_sizes.get(page_number, (0.0, 0.0))
    lines = page_lines.get(page_number, [])
    if not lines or page_width <= 0 or page_height <= 0:
        return None

    best_score = 0.0
    best_line: PDFLine | None = None
    limit = max(1, int(max_lines))

    for start in range(len(lines)):
        for span in range(1, limit + 1):
            end = start + span
            if end > len(lines):
                break
            merged = merge_pdf_lines(lines[start:end])
            candidate = normalize_for_match(merged.text)
            if not candidate:
                continue
            score = difflib.SequenceMatcher(None, target, candidate).ratio()
            if target in candidate or candidate in target:
                score += 0.55
            overlap = len(set(target) & set(candidate)) / max(1, len(set(target)))
            score += overlap * 0.30
            score -= max(0, len(candidate) - len(target)) * 0.003
            if score > best_score:
                best_score = score
                best_line = merged

    if best_line is None or best_score < 0.55:
        return None

    return {
        "x": (best_line.left / page_width) * 1000.0,
        "y": (best_line.top / page_height) * 1000.0,
        "w": (best_line.width / page_width) * 1000.0,
        "h": (best_line.height / page_height) * 1000.0,
    }


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    if not text:
        return [""]
    trailing_punctuation = set("，。！？；：、】【）》」』〕〉.,!?;:%)]}>\"'、")
    leading_punctuation = set("（【《「『〔〈([{")
    result: list[str] = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph:
            result.append("")
            continue
        current = ""
        for char in paragraph:
            candidate = current + char
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= max_width or not current:
                current = candidate
                continue

            if char in trailing_punctuation:
                current = candidate
                continue

            if char in leading_punctuation and current:
                result.append(current)
                current = char
                continue

            result.append(current)
            current = char
        if current:
            result.append(current)

    cleaned: list[str] = []
    for line in result:
        if line and len(line) == 1 and line in trailing_punctuation and cleaned:
            cleaned[-1] += line
            continue
        cleaned.append(line)
    return cleaned


def measure_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    line_height: int,
) -> tuple[list[str], int]:
    lines = wrap_text(draw, text, font, max_width)
    return lines, len(lines) * line_height


def normalize_note_text(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    replacements = {
        "｜": "|",
        "→": "->",
        "①": "1.",
        "②": "2.",
        "③": "3.",
        "④": "4.",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
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
    text = "".join(result_chars)
    return " ".join(text.split())


def compact_note_text(text: str, max_length: int = 36, prefer_first_clause: bool = False) -> str:
    text = normalize_note_text(text)
    if not text:
        return ""
    text = re.split(r"[。！？\n]", text, maxsplit=1)[0].strip()
    text = text.rstrip("，；：、,. ")
    segments = [segment.strip() for segment in re.split(r"[，；：]", text) if segment.strip()]
    if prefer_first_clause and segments:
        text = segments[0]
        if len(text) > max_length:
            text = text[: max_length - 1].rstrip("，；：、,. ") + "…"
        return text
    compact = ""
    for segment in segments:
        candidate = f"{compact}，{segment}" if compact else segment
        if len(candidate) <= max_length:
            compact = candidate
        else:
            break
    if compact:
        text = compact
    if len(text) > max_length:
        text = text[: max_length - 1].rstrip("，；：、,. ") + "…"
    return text


def compact_tail_text(text: str, max_length: int = 78, keep_two_clauses: bool = False) -> str:
    text = normalize_note_text(text)
    if not text:
        return ""
    segments = [segment.strip() for segment in re.split(r"[。！？]", text) if segment.strip()]
    if segments:
        text = "。".join(segments[:2] if keep_two_clauses else segments[:1])
    clauses = [segment.strip() for segment in re.split(r"[，；：]", text) if segment.strip()]
    if clauses:
        picked = clauses[:2] if keep_two_clauses else clauses[:1]
        text = "，".join(picked)
    text = " ".join(text.split()).strip("，；：、,. ")
    if len(text) > max_length:
        text = text[: max_length - 1].rstrip("，；：、,. ") + "…"
    return text


def compact_tail_issue_line(issue: dict[str, Any]) -> str:
    title = compact_tail_text(issue.get("title", "") or "未命名问题", max_length=18)
    comment = compact_tail_text(issue.get("comment", ""), max_length=34)
    rewrite_tip = compact_tail_text(
        issue.get("rewrite_tip") or issue.get("rewrite_example") or "",
        max_length=30,
    )
    parts = [f"{title}：{comment or '需要直接改写'}"]
    if rewrite_tip:
        parts.append(f"改：{rewrite_tip}")
    return " ".join(part for part in parts if part).strip()


def build_note_body_parts(issue: dict[str, Any]) -> list[str]:
    focus_text = compact_note_text(issue.get("focus_text", ""), max_length=28, prefer_first_clause=True)
    comment = compact_note_text(issue.get("comment", ""), max_length=38)
    rewrite_tip = compact_note_text(
        issue.get("rewrite_tip") or issue.get("rewrite_example") or "",
        max_length=40,
    )
    parts: list[str] = []
    if focus_text:
        parts.append(f"看这里：{focus_text}")
    if comment:
        parts.append(f"问题：{comment}")
    if rewrite_tip:
        parts.append(f"改法：{rewrite_tip}")
    return parts


def layout_note_title(
    draw: ImageDraw.ImageDraw,
    title: str,
    title_font: ImageFont.ImageFont,
    max_width: int,
) -> tuple[list[str], int]:
    title = normalize_note_text(title)
    return measure_wrapped_text(draw, title, title_font, max_width, 30)


def draw_note_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    index: int,
    issue: dict[str, Any],
    title_font: ImageFont.ImageFont,
    meta_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> None:
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    shadow_far = (x1 + 4, y1 + 8, x2 + 4, y2 + 8)
    shadow_near = (x1 + 2, y1 + 3, x2 + 2, y2 + 3)
    draw.rounded_rectangle(shadow_far, radius=20, fill=(239, 239, 239))
    draw.rounded_rectangle(shadow_near, radius=20, fill=(246, 246, 246))
    draw.rounded_rectangle(box, radius=20, outline=(226, 232, 240), width=2, fill=(255, 255, 255))
    cursor_x = x1 + 66
    title_width = max(120, box_width - 82)
    title_lines, title_height = layout_note_title(draw, issue["title"], title_font, title_width)
    header_height = max(60, 18 + title_height + 14)
    header_box = (x1 + 1, y1 + 1, x2 - 1, y1 + header_height)
    draw.rounded_rectangle(header_box, radius=20, fill=(250, 251, 253))
    draw.rectangle((x1 + 1, y1 + 42, x2 - 1, y1 + header_height), fill=(250, 251, 253))
    badge_box = (x1 + 16, y1 + 14, x1 + 52, y1 + 50)
    draw.ellipse(badge_box, fill=(239, 68, 68))
    draw.text((x1 + 27, y1 + 18), str(index), font=meta_font, fill="white")

    cursor_y = y1 + 14
    for line in title_lines:
        draw.text((cursor_x, cursor_y), line, font=title_font, fill=(17, 24, 39))
        cursor_y += 30

    cursor_y = y1 + header_height + 14
    body_width = box_width - 32
    body_parts = build_note_body_parts(issue)
    body_text = "\n".join(body_parts)
    for line in wrap_text(draw, body_text, body_font, body_width):
        draw.text((x1 + 16, cursor_y), line, font=body_font, fill=(55, 65, 81))
        cursor_y += 28


def estimate_box_height(
    draw: ImageDraw.ImageDraw,
    issue: dict[str, Any],
    title_font: ImageFont.ImageFont,
    meta_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    box_width: int,
) -> int:
    del meta_font
    title_width = max(120, box_width - 82)
    _title_lines, title_height = layout_note_title(draw, issue.get("title", ""), title_font, title_width)
    header_height = max(60, 18 + title_height + 14)
    body_width = box_width - 32
    body_parts = build_note_body_parts(issue)
    body_text = "\n".join(body_parts)
    line_count = len(wrap_text(draw, body_text, body_font, body_width))
    return header_height + 26 + line_count * 28 + 18


def infer_review_page_count(review: dict[str, Any]) -> int:
    meta = review.get("_meta")
    if isinstance(meta, dict):
        page_count = meta.get("resume_page_count")
        if isinstance(page_count, int) and page_count > 0:
            return page_count
    pages = [int(issue.get("page", 1) or 1) for issue in review.get("issues", [])]
    return max(pages, default=1)


def build_tail_page_blocks(review: dict[str, Any]) -> list[tuple[str, list[str]]]:
    assessment = review.get("match_assessment", {})
    guide = build_consultation_guide(review)
    blocks: list[tuple[str, list[str]]] = []
    strengths = [compact_tail_text(item, max_length=26) for item in review.get("strengths", []) if str(item).strip()]
    weaknesses = [compact_tail_text(item, max_length=26) for item in review.get("weaknesses", []) if str(item).strip()]

    summary = str(review.get("summary", "")).strip()
    if summary:
        summary_lines = [compact_tail_text(summary, max_length=126, keep_two_clauses=True)]
        if strengths:
            summary_lines.append(f"优势：{'；'.join(strengths[:2])}")
        if weaknesses:
            summary_lines.append(f"短板：{'；'.join(weaknesses[:2])}")
        blocks.append(("总评", summary_lines))

    jd_keywords = [str(item).strip() for item in review.get("jd_keywords", []) if str(item).strip()]
    if jd_keywords:
        signal_lines = [f"关键词：{'、'.join(jd_keywords[:8])}"]
        responsibilities = [str(item).strip() for item in review.get("jd_responsibilities", []) if str(item).strip()]
        if responsibilities:
            signal_lines.append(f"核心职责：{compact_tail_text(responsibilities[0], max_length=58, keep_two_clauses=True)}")
        hard_skills = [str(item).strip() for item in review.get("jd_hard_skills", []) if str(item).strip()]
        soft_skills = [str(item).strip() for item in review.get("jd_soft_skills", []) if str(item).strip()]
        if hard_skills:
            signal_lines.append(f"硬技能：{'、'.join(hard_skills[:5])}")
        if soft_skills:
            signal_lines.append(f"软技能：{'、'.join(soft_skills[:4])}")
        blocks.append(
            (
                "JD 关键信号",
                signal_lines,
            )
        )

    if assessment:
        assessment_labels = {
            "keyword_coverage": translate_assessment_label(assessment.get("keyword_coverage", "")),
            "professionalism": translate_assessment_label(assessment.get("professionalism", "")),
            "clarity": translate_assessment_label(assessment.get("clarity", "")),
            "fit": translate_assessment_label(assessment.get("fit", "")),
        }
        weakest_dimension = min(
            ("keyword_coverage", "professionalism", "clarity", "fit"),
            key=lambda key: {"低": 0, "中": 1, "高": 2}.get(assessment_labels[key], 1),
        )
        weakest_label = {
            "keyword_coverage": "覆盖",
            "professionalism": "专业",
            "clarity": "清晰",
            "fit": "匹配",
        }[weakest_dimension]
        blocks.append(
            (
                "匹配结论",
                [
                    (
                        "覆盖 / 专业 / 清晰 / 匹配："
                        f"{assessment_labels['keyword_coverage']} / "
                        f"{assessment_labels['professionalism']} / "
                        f"{assessment_labels['clarity']} / "
                        f"{assessment_labels['fit']}"
                    ),
                    f"当前判断：{compact_tail_text(guide.get('headline', '') or guide.get('summary', ''), max_length=34, keep_two_clauses=True)}",
                    f"最弱项：先补{weakest_label}表达，再提整体匹配度。",
                ],
            )
        )

    issues = review.get("issues", [])
    if issues:
        blocks.append(("优先修改方向", [compact_tail_issue_line(issue) for issue in issues[:4]]))

    next_lines: list[str] = []
    if strengths:
        next_lines.append(f"保留：{'；'.join(strengths[:2])}")
    if weaknesses:
        next_lines.append(f"补齐：{'；'.join(weaknesses[:2])}")
    focus_items = guide.get("session_focus") or guide.get("reasons") or []
    next_lines.extend(f"- {compact_tail_text(item, max_length=46)}" for item in focus_items[:2])
    if guide.get("prep_items"):
        next_lines.append(
            "素材："
            + "；".join(compact_tail_text(item, max_length=24) for item in guide.get("prep_items", [])[:2])
        )
    if len(next_lines) < 5 and len(focus_items) > 2:
        next_lines.append(f"- {compact_tail_text(focus_items[2], max_length=46)}")
    if next_lines:
        blocks.append(("下一轮重点", next_lines[:5]))
    return blocks


def layout_tail_block(
    draw: ImageDraw.ImageDraw,
    title: str,
    lines: list[str],
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    max_width: int,
) -> tuple[list[str], list[str], int]:
    title_lines, title_height = measure_wrapped_text(draw, title, title_font, max_width, 30)
    wrapped_body_lines: list[str] = []
    body_height = 0
    for line in lines:
        wrapped_lines, wrapped_height = measure_wrapped_text(draw, line, body_font, max_width, 22)
        wrapped_body_lines.extend(wrapped_lines)
        wrapped_body_lines.append("")
        body_height += wrapped_height + 4
    if wrapped_body_lines and not wrapped_body_lines[-1]:
        wrapped_body_lines.pop()
        body_height -= 4
    total_height = 18 + title_height + 6 + body_height + 18
    return title_lines, wrapped_body_lines, total_height


def estimate_tail_block_height(
    draw: ImageDraw.ImageDraw,
    title: str,
    lines: list[str],
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    max_width: int,
) -> int:
    _title_lines, _body_lines, total_height = layout_tail_block(
        draw=draw,
        title=title,
        lines=lines,
        title_font=title_font,
        body_font=body_font,
        max_width=max_width,
    )
    return total_height


def draw_tail_block(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    lines: list[str],
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle((x1 + 3, y1 + 6, x2 + 3, y2 + 6), radius=22, fill=(230, 228, 224))
    draw.rounded_rectangle(box, radius=22, outline=(228, 221, 214), width=2, fill=(255, 252, 248))
    draw.rounded_rectangle((x1 + 18, y1 + 18, x1 + 28, y1 + 28), radius=5, fill=(185, 28, 28))

    content_x = x1 + 42
    content_y = y1 + 12
    content_width = (x2 - x1) - 64
    title_lines, body_lines, _total_height = layout_tail_block(
        draw=draw,
        title=title,
        lines=lines,
        title_font=title_font,
        body_font=body_font,
        max_width=content_width,
    )
    for line in title_lines:
        draw.text((content_x, content_y), line, font=title_font, fill=(35, 31, 32))
        content_y += 28
    content_y += 6
    for line in body_lines:
        if line:
            draw.text((content_x, content_y), line, font=body_font, fill=(72, 64, 58))
            content_y += 22
        else:
            content_y += 4


def render_tail_pages(
    review: dict[str, Any],
    output_dir: Path,
    page_width: int,
    page_height: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = build_tail_page_blocks(review)
    if not blocks:
        return []

    title_font = load_font(30, family="serif")
    section_font = load_font(21, family="sans")
    body_font = load_font(17, family="sans")
    small_font = load_font(14, family="sans")

    scratch = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    margin_x = 44
    top_margin = 112
    bottom_margin = 38
    block_spacing = 12
    content_width = page_width - margin_x * 2

    pages: list[Path] = []
    page_index = 1
    canvas = Image.new("RGB", (page_width, page_height), (245, 239, 232))
    draw = ImageDraw.Draw(canvas)
    current_y = top_margin

    def draw_header(page_no: int) -> None:
        header_title = "岗位定向简历批注总结" if page_no == 1 else "岗位定向简历批注总结（续）"
        draw.text((margin_x, 34), header_title, font=title_font, fill=(46, 39, 36))
        draw.text((margin_x, 76), "用于交付总结和后续 1v1 引导", font=small_font, fill=(120, 111, 104))
        draw.line((margin_x, 104, page_width - margin_x, 104), fill=(221, 213, 206), width=2)

    def save_page(page_no: int) -> None:
        path = output_dir / f"tail-page-{page_no:02d}.png"
        canvas.save(path)
        pages.append(path)

    draw_header(page_index)
    for title, lines in blocks:
        block_height = estimate_tail_block_height(scratch, title, lines, section_font, body_font, content_width - 64)
        if current_y + block_height > page_height - bottom_margin:
            save_page(page_index)
            page_index += 1
            canvas = Image.new("RGB", (page_width, page_height), (245, 239, 232))
            draw = ImageDraw.Draw(canvas)
            draw_header(page_index)
            current_y = top_margin
        box = (margin_x, current_y, page_width - margin_x, current_y + block_height)
        draw_tail_block(draw, box, title, lines, section_font, body_font)
        current_y = box[3] + block_spacing

    save_page(page_index)
    return pages


def build_connector_elbows(width: int, note_x1: int, issue_count: int) -> list[int]:
    lane_start = width + 10
    lane_end = max(lane_start, note_x1 - 10)
    lane_step = 7
    lanes = list(range(lane_start, lane_end + 1, lane_step)) or [lane_start]
    if issue_count <= len(lanes):
        return lanes[:issue_count]
    return [lanes[index % len(lanes)] for index in range(issue_count)]


def render_annotations(
    image_path: Path,
    review: dict[str, Any],
    output_path: Path,
    margin_width: int = 420,
    pdf_path: Path | None = None,
    page_number: int = 1,
) -> None:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    panel_background = sample_panel_background(image)

    title_font = load_font(26, family="sans")
    meta_font = load_font(18, family="sans")
    body_font = load_font(19, family="sans")
    red = (239, 68, 68)

    note_x1 = width + 64
    note_x2 = width + margin_width - 20
    note_y = 24
    note_spacing = 18

    issues = review.get("issues", [])
    note_boxes: list[tuple[int, int, int, int]] = []
    connector_elbows = build_connector_elbows(width, note_x1, len(issues))
    scratch = ImageDraw.Draw(Image.new("RGB", (10, 10)))

    for issue in issues:
        box_height = estimate_box_height(scratch, issue, title_font, meta_font, body_font, note_x2 - note_x1)
        box = (note_x1, note_y, note_x2, note_y + box_height)
        note_boxes.append(box)
        note_y = box[3] + note_spacing

    canvas_height = max(height, note_y + 20)
    canvas = Image.new("RGB", (width + margin_width, canvas_height), "white")
    right_panel = Image.new("RGB", (margin_width, canvas_height), panel_background)
    canvas.paste(image, (0, 0))
    canvas.paste(right_panel, (width, 0))
    draw = ImageDraw.Draw(canvas)

    draw.line([(width, 0), (width, canvas_height)], fill=(235, 235, 235), width=2)

    for index, (issue, box) in enumerate(zip(issues, note_boxes), start=1):
        refined_anchor = None
        anchor_mode = "raw"
        focus_text = issue.get("focus_text", "").strip()
        use_ocr = issue.get("ocr_enabled", True)
        if focus_text and pdf_path is not None:
            refined_anchor = find_best_pdf_anchor(
                pdf_path=pdf_path,
                page_number=page_number,
                focus_text=focus_text,
                max_lines=int(issue.get("ocr_max_lines", 2)),
            )
            if refined_anchor:
                anchor_mode = "pdf"
        if focus_text and use_ocr and refined_anchor is None:
            refined_anchor = find_best_ocr_anchor(
                image_path=image_path,
                focus_text=focus_text,
                width=width,
                height=height,
                max_lines=int(issue.get("ocr_max_lines", 2)),
            )
            if refined_anchor:
                anchor_mode = "ocr"
        anchor_source = refined_anchor or issue["anchor"]
        anchor = normalized_to_pixels(anchor_source, width, height)
        padding = max(6, int(issue.get("padding", 2)))
        if anchor_mode == "pdf":
            focus_box = expand_manual_box(
                anchor=anchor,
                width=width,
                height=height,
                padding=min(8, padding),
            )
        elif use_ocr and anchor_mode == "ocr":
            # Expand only into actual whitespace so the outline does not touch nearby lines.
            focus_box = expand_box_safely(
                image_path=image_path,
                base_anchor=anchor,
                width=width,
                height=height,
                desired_padding=padding,
            )
        else:
            focus_box = expand_manual_box(
                anchor=anchor,
                width=width,
                height=height,
                padding=padding,
            )
        draw.rounded_rectangle(focus_box, radius=10, outline=red, width=3)

        marker_center = (focus_box[2], int((focus_box[1] + focus_box[3]) / 2))
        box_center_y = int((box[1] + box[3]) / 2)
        elbow_x = connector_elbows[index - 1]
        draw.line([marker_center, (elbow_x, marker_center[1]), (elbow_x, box_center_y), (box[0], box_center_y)], fill=red, width=3)
        draw.ellipse((marker_center[0] - 4, marker_center[1] - 4, marker_center[0] + 4, marker_center[1] + 4), fill=red)
        draw_note_box(draw, box, index, issue, title_font, meta_font, body_font)

    canvas.save(output_path)


def write_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cmd_review(args: argparse.Namespace) -> int:
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing API key. Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    image_path = Path(args.image)
    output_path = Path(args.output)
    payload = build_review_payload(image_path, args.model)
    response_data = call_responses_api(payload, api_key)
    review = extract_output_json(response_data)
    write_json(review, output_path)
    print(f"Saved review JSON to {output_path}")
    return 0


def cmd_review_match(args: argparse.Namespace) -> int:
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing API key. Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    jd_image_path = Path(args.jd_image)
    resume_pdf_path = Path(args.resume_pdf)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="resume-review-v2-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        resume_pages_dir = tmp_path / "resume_pages"
        page_images = convert_pdf_to_images(
            pdf_path=resume_pdf_path,
            output_dir=resume_pages_dir,
            dpi=args.dpi,
            max_pages=args.max_pages,
        )
        jd_text = ocr_plain_text(jd_image_path, psm=6)
        resume_text, resume_text_source = extract_resume_text_with_fallback(resume_pdf_path, page_images)
        payload = build_match_review_payload(
            jd_image_path=jd_image_path,
            jd_text=jd_text,
            resume_text=resume_text,
            resume_image_paths=page_images,
            model=args.model,
        )
        response_data = call_responses_api(payload, api_key)

    review = extract_output_json(response_data)
    review = enrich_match_review(review, len(page_images))
    review["_meta"] = summarize_input_meta(
        jd_image_path=jd_image_path,
        resume_pdf_path=resume_pdf_path,
        page_images=page_images,
        jd_text_source="ocr_tesseract",
        resume_text_source=resume_text_source,
    )
    write_json(review, output_path)
    print(f"Saved match review JSON to {output_path}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    image_path = Path(args.image)
    review_path = Path(args.review)
    output_path = Path(args.output)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    render_annotations(image_path, review, output_path, margin_width=args.margin)
    print(f"Saved annotated image to {output_path}")
    return 0


def cmd_render_pdf(args: argparse.Namespace) -> int:
    resume_pdf_path = Path(args.resume_pdf)
    review_path = Path(args.review)
    output_path = Path(args.output)
    review = json.loads(review_path.read_text(encoding="utf-8"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_dir = Path(args.png_dir) if args.png_dir else None
    if png_dir:
        png_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="resume-render-v2-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        source_pages_dir = tmp_path / "source_pages"
        rendered_pages_dir = tmp_path / "rendered_pages"
        rendered_pages_dir.mkdir(parents=True, exist_ok=True)

        page_images = convert_pdf_to_images(
            pdf_path=resume_pdf_path,
            output_dir=source_pages_dir,
            dpi=args.dpi,
            max_pages=args.max_pages,
        )
        review = enrich_match_review(review, len(page_images))
        issues_by_page: dict[int, list[dict[str, Any]]] = {}
        for issue in review.get("issues", []):
            page = int(issue.get("page", 1) or 1)
            issues_by_page.setdefault(page, []).append(issue)

        rendered_paths: list[Path] = []
        for page_index, page_image in enumerate(page_images, start=1):
            page_issues = issues_by_page.get(page_index, [])
            rendered_path = rendered_pages_dir / f"annotated-page-{page_index:02d}.png"
            if page_issues:
                page_review = {"issues": page_issues}
                render_annotations(
                    page_image,
                    page_review,
                    rendered_path,
                    margin_width=args.margin,
                    pdf_path=resume_pdf_path,
                    page_number=page_index,
                )
            else:
                with Image.open(page_image) as page:
                    page.convert("RGB").save(rendered_path)
            rendered_paths.append(rendered_path)
            if png_dir:
                shutil.copy2(rendered_path, png_dir / rendered_path.name)

        if not args.no_tail_pages and rendered_paths:
            with Image.open(rendered_paths[0]) as first_rendered:
                tail_page_width, tail_page_height = first_rendered.size
            tail_pages = render_tail_pages(
                review=review,
                output_dir=rendered_pages_dir / "tail_pages",
                page_width=tail_page_width,
                page_height=tail_page_height,
            )
            for tail_page in tail_pages:
                rendered_paths.append(tail_page)
                if png_dir:
                    shutil.copy2(tail_page, png_dir / tail_page.name)

        if not rendered_paths:
            print("No rendered pages were produced.", file=sys.stderr)
            return 1

        pdf_images = [Image.open(path).convert("RGB") for path in rendered_paths]
        try:
            first, rest = pdf_images[0], pdf_images[1:]
            first.save(output_path, save_all=True, append_images=rest)
        finally:
            for image in pdf_images:
                image.close()

    print(f"Saved annotated PDF to {output_path}")
    return 0


def cmd_report_markdown(args: argparse.Namespace) -> int:
    review_path = Path(args.review)
    output_path = Path(args.output)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review = enrich_match_review(review, infer_review_page_count(review))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report_markdown(review), encoding="utf-8")
    print(f"Saved markdown report to {output_path}")
    return 0


def cmd_print_schema(_: argparse.Namespace) -> int:
    print(json.dumps(RESPONSE_SCHEMA, ensure_ascii=False, indent=2))
    return 0


def cmd_print_match_schema(_: argparse.Namespace) -> int:
    print(json.dumps(MATCH_RESPONSE_SCHEMA, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resume review annotation tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    review_parser = subparsers.add_parser("review", help="Call the model and save review JSON")
    review_parser.add_argument("--image", required=True, help="Path to the resume image")
    review_parser.add_argument("--output", required=True, help="Path to save the review JSON")
    review_parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to use, default: {DEFAULT_MODEL}")
    review_parser.add_argument("--api-key", help="OpenAI API key, optional if OPENAI_API_KEY is set")
    review_parser.set_defaults(func=cmd_review)

    render_parser = subparsers.add_parser("render", help="Render an annotated image from review JSON")
    render_parser.add_argument("--image", required=True, help="Path to the resume image")
    render_parser.add_argument("--review", required=True, help="Path to the review JSON")
    render_parser.add_argument("--output", required=True, help="Path to save the annotated image")
    render_parser.add_argument("--margin", type=int, default=420, help="Right-side note margin width")
    render_parser.set_defaults(func=cmd_render)

    schema_parser = subparsers.add_parser("schema", help="Print the JSON schema used for model output")
    schema_parser.set_defaults(func=cmd_print_schema)

    review_match_parser = subparsers.add_parser(
        "review-match",
        help="Call the model with a JD image and resume PDF, then save match review JSON",
    )
    review_match_parser.add_argument("--jd-image", required=True, help="Path to the JD screenshot image")
    review_match_parser.add_argument("--resume-pdf", required=True, help="Path to the resume PDF")
    review_match_parser.add_argument("--output", required=True, help="Path to save the match review JSON")
    review_match_parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to use, default: {DEFAULT_MODEL}")
    review_match_parser.add_argument("--max-pages", type=int, default=2, help="Maximum resume PDF pages to process")
    review_match_parser.add_argument("--dpi", type=int, default=180, help="PDF render DPI")
    review_match_parser.add_argument("--api-key", help="OpenAI API key, optional if OPENAI_API_KEY is set")
    review_match_parser.set_defaults(func=cmd_review_match)

    render_pdf_parser = subparsers.add_parser("render-pdf", help="Render an annotated PDF from match review JSON")
    render_pdf_parser.add_argument("--resume-pdf", required=True, help="Path to the resume PDF")
    render_pdf_parser.add_argument("--review", required=True, help="Path to the match review JSON")
    render_pdf_parser.add_argument("--output", required=True, help="Path to save the annotated PDF")
    render_pdf_parser.add_argument("--png-dir", help="Optional directory to save annotated page PNGs")
    render_pdf_parser.add_argument("--margin", type=int, default=420, help="Right-side note margin width")
    render_pdf_parser.add_argument("--max-pages", type=int, default=2, help="Maximum resume PDF pages to process")
    render_pdf_parser.add_argument("--dpi", type=int, default=180, help="PDF render DPI")
    render_pdf_parser.add_argument("--no-tail-pages", action="store_true", help="Do not append summary/1v1 tail pages")
    render_pdf_parser.set_defaults(func=cmd_render_pdf)

    report_md_parser = subparsers.add_parser("report-md", help="Convert match review JSON into a markdown report")
    report_md_parser.add_argument("--review", required=True, help="Path to the match review JSON")
    report_md_parser.add_argument("--output", required=True, help="Path to save the markdown report")
    report_md_parser.set_defaults(func=cmd_report_markdown)

    schema_match_parser = subparsers.add_parser("schema-match", help="Print the match review JSON schema")
    schema_match_parser.set_defaults(func=cmd_print_match_schema)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
