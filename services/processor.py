"""
Orchestrates the full annotation pipeline for a task.

Steps:
  1. AI review  → result.json       (via active AI provider: openai or claude)
  2. render-pdf → annotated.pdf     (annotator script subprocess — local, no AI)
  3. report-md  → report.md         (annotator script subprocess — local, no AI)
  4. md → HTML  → report.html       (via markdown2)

The AI call in step 1 is provider-agnostic via get_provider().
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

try:
    import markdown2
    HAS_MARKDOWN2 = True
except ImportError:
    HAS_MARKDOWN2 = False

from services.storage import (
    task_upload_dir,
    task_output_dir,
    result_json_path,
    annotated_pdf_path,
    report_md_path,
    report_html_path,
)
from services.adapter import adapt_result
from services.providers import get_provider, ANNOTATOR_SCRIPT

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _run_local(cmd: list[str], task_id: str, step: str) -> None:
    """Run a local (non-AI) annotator subprocess command."""
    logger.info("[%s] %s: %s", task_id, step, " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    if r.returncode != 0:
        stderr = r.stderr.strip()
        stdout = r.stdout.strip()
        detail = stderr or stdout or "(no output)"
        logger.error("[%s] %s failed (exit %d)\nstderr:\n%s\nstdout:\n%s",
                     task_id, step, r.returncode, stderr, stdout)
        raise RuntimeError(f"{step} 失败 (exit {r.returncode}):\n{detail}")


def _md_to_html(md_path: Path, html_path: Path) -> None:
    md_text = md_path.read_text(encoding="utf-8")
    if HAS_MARKDOWN2:
        body = markdown2.markdown(md_text, extras=["tables", "fenced-code-blocks", "header-ids"])
    else:
        paras = md_text.split("\n\n")
        body = "\n".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paras)

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>简历批注报告</title>
  <style>
    body {{ font-family: -apple-system, 'PingFang SC', sans-serif; max-width: 800px;
           margin: 40px auto; padding: 0 24px; color: #1a1a1a; line-height: 1.7; }}
    h1, h2, h3 {{ color: #111; }}
    h1 {{ font-size: 1.6em; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; }}
    h2 {{ font-size: 1.25em; margin-top: 32px; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; }}
    h3 {{ font-size: 1.05em; margin-top: 20px; color: #374151; }}
    ul {{ padding-left: 1.4em; }}
    li {{ margin-bottom: 4px; }}
    p {{ margin: 8px 0; }}
    code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
    pre {{ background: #f3f4f6; padding: 16px; border-radius: 8px; overflow-x: auto; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""
    html_path.write_text(html, encoding="utf-8")


def process_task(task_id: str, jd_filename: str, resume_filename: str) -> dict:
    """
    Run the full annotation pipeline for a task.
    Returns adapted result dict ready for DB storage.
    Raises RuntimeError with a user-friendly message on failure.
    """
    upload_dir = task_upload_dir(task_id)
    output_dir = task_output_dir(task_id)

    jd_path = upload_dir / jd_filename
    resume_path = upload_dir / resume_filename
    result_path = result_json_path(task_id)
    pdf_out = annotated_pdf_path(task_id)
    md_out = report_md_path(task_id)
    html_out = report_html_path(task_id)

    if not jd_path.exists():
        raise RuntimeError(f"JD 图片文件不存在: {jd_filename}")
    if not resume_path.exists():
        raise RuntimeError(f"简历 PDF 文件不存在: {resume_filename}")

    # Step 1: AI review (provider-agnostic)
    logger.info("[%s] Step 1: AI review (provider=%s)", task_id, os.getenv("AI_PROVIDER", "openai"))
    provider = get_provider()
    provider.review_match(jd_path, resume_path, result_path, task_id)

    if not result_path.exists():
        raise RuntimeError("AI review 步骤未生成 result.json")

    # Step 2: Render annotated PDF (local — no AI call)
    logger.info("[%s] Step 2: render-pdf", task_id)
    if ANNOTATOR_SCRIPT.exists():
        try:
            _run_local([
                sys.executable, str(ANNOTATOR_SCRIPT),
                "render-pdf",
                "--resume-pdf", str(resume_path),
                "--review", str(result_path),
                "--output", str(pdf_out),
            ], task_id, "render-pdf")
        except RuntimeError as e:
            logger.warning("[%s] render-pdf failed (non-fatal, continuing): %s", task_id, e)
    else:
        logger.warning("[%s] Annotator script not found, skipping render-pdf", task_id)

    # Step 3: Generate markdown report (local — no AI call)
    logger.info("[%s] Step 3: report-md", task_id)
    if ANNOTATOR_SCRIPT.exists():
        try:
            _run_local([
                sys.executable, str(ANNOTATOR_SCRIPT),
                "report-md",
                "--review", str(result_path),
                "--output", str(md_out),
            ], task_id, "report-md")
        except RuntimeError as e:
            logger.warning("[%s] report-md failed (non-fatal): %s", task_id, e)

    # Step 4: Convert markdown → HTML
    logger.info("[%s] Step 4: md → html", task_id)
    if md_out.exists():
        _md_to_html(md_out, html_out)

    # Step 5: Adapt result for frontend
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    adapted = adapt_result(raw, task_id)

    return {
        "adapted_result": adapted,
        "candidate_name": _extract_candidate_name(raw, resume_filename),
        "jd_title": _extract_jd_title(raw),
    }


def _extract_candidate_name(raw: dict, resume_filename: str) -> str:
    name = raw.get("candidate_name") or ""
    if name and name not in ("null", "NULL", "None"):
        return name.strip()
    # Fallback: extract from filename
    stem = Path(resume_filename).stem
    for suffix in ["_resume", "_简历", "-resume", "-简历", "_Resume", "_CV", "-CV"]:
        stem = stem.replace(suffix, "")
    # Remove common patterns like date prefixes/suffixes
    import re as _re
    stem = _re.sub(r'[\d_\-]{4,}', '', stem).strip('_- ')
    return stem or "未知"


def _extract_jd_title(raw: dict) -> str:
    title = raw.get("jd_title") or raw.get("job_title") or ""
    if title and title not in ("null", "NULL", "None"):
        return title.strip()
    # Fallback 1: try to infer from hard skills (e.g. "Python" → guess role)
    responsibilities = raw.get("jd_responsibilities", [])
    if responsibilities:
        first = responsibilities[0]
        # Take at most first 20 chars before punctuation as title
        import re as _re
        m = _re.match(r'^[^，,。.；;！!？?]{2,20}', first)
        if m:
            return m.group(0).strip()
    # Fallback 2: first hard skill as partial title
    hard_skills = raw.get("jd_hard_skills", [])
    if hard_skills:
        return hard_skills[0][:15]
    return "未知岗位"
