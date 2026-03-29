"""
Adapt the raw JSON output from resume-review-annotator-v2.py
to the frontend-expected TaskResult shape.
"""

from __future__ import annotations
import re
from typing import Any


# Map "high"/"medium"/"low" → numeric score (midpoint of bucket)
_SCORE_MAP = {"high": 78, "medium": 55, "low": 28}
_LABEL_MAP = {
    "keyword_coverage": "关键词覆盖率",
    "professionalism": "专业性",
    "clarity": "清晰度",
    "fit": "整体匹配度",
}
_LEVEL_CN = {"high": "高", "medium": "中", "low": "低"}


def _level_to_score(level: str) -> dict:
    level = (level or "low").lower()
    score = _SCORE_MAP.get(level, 28)
    return {"score": score, "level": level}


def _build_consultation(raw: dict[str, Any]) -> dict[str, Any]:
    guide = raw.get("consultation_guide", {}) or {}
    reasons = [str(item).strip() for item in guide.get("reasons", []) if str(item).strip()]
    session_focus = [str(item).strip() for item in guide.get("session_focus", []) if str(item).strip()]
    prep_items = [str(item).strip() for item in guide.get("prep_items", []) if str(item).strip()]
    headline = str(guide.get("headline", "")).strip()
    summary = str(guide.get("summary", "")).strip()
    cta = str(guide.get("cta", "")).strip()
    if headline and summary and reasons and session_focus and prep_items:
        return {
            "recommend": bool(guide.get("recommended", True)),
            "headline": headline,
            "reason": summary,
            "priorities": session_focus[:5],
            "prepItems": prep_items[:5],
            "cta": cta,
        }

    assessment = raw.get("match_assessment", {}) or {}
    issues = raw.get("issues", []) or []
    low_dimensions = sum(
        1
        for key in ("keyword_coverage", "professionalism", "clarity", "fit")
        if str(assessment.get(key, "")).lower() == "low"
    )
    high_issue_count = sum(1 for issue in issues if str(issue.get("severity", "")).lower() == "high")
    recommend = low_dimensions >= 2 or high_issue_count >= 2 or len(issues) >= 4

    fit_cn = _LEVEL_CN.get(str(assessment.get("fit", "low")).lower(), "低")
    coverage_cn = _LEVEL_CN.get(str(assessment.get("keyword_coverage", "low")).lower(), "低")
    summary = str(raw.get("summary", "")).strip()
    reason = (
        summary
        or f"当前岗位匹配度为{fit_cn}，关键词覆盖率为{coverage_cn}，还没到可直接投递的状态。"
    )
    if recommend and "还没到可直接投递" not in reason:
        reason = f"{reason.rstrip('。')}，还没到可直接投递的状态。"

    priorities: list[str] = []
    category_focus = {
        "job_target": "先把求职方向改成目标岗位，别让岗位入口继续错位。",
        "relevance": "把最贴岗的经历前置，弱相关内容往后放。",
        "impact": "把职责改成结果表达，补人数、场次、效果等量化信息。",
        "summary": "自我评价不要空话，改成岗位标签加可验证能力。",
    }
    for issue in issues[:3]:
        category = str(issue.get("category", "")).strip()
        rewrite_tip = str(issue.get("rewrite_tip", "")).strip()
        if category in category_focus:
            priorities.append(category_focus[category])
        elif rewrite_tip:
            priorities.append(rewrite_tip)
    if not priorities:
        priorities = [
            "先把高优先级问题改完，再看是否还需要继续深改。",
            "优先补齐最贴岗经历的动作、结果和关键词。",
        ]

    prep_items = [
        "准备目标 JD 原文，明确只保留一个主投岗位版本。",
        "补 2-3 段核心经历的量化结果，如人数、场次、产出和效果。",
    ]
    headline = "建议继续做 1v1 深度改稿" if recommend else "可按当前建议先自行修改一轮"
    cta = (
        "建议先重写核心经历和求职方向，再决定是否继续深改。"
        if recommend
        else "先按当前批注改完一版，再回来复查。"
    )
    return {
        "recommend": recommend,
        "headline": headline,
        "reason": reason,
        "priorities": priorities[:5],
        "prepItems": prep_items[:5],
        "cta": cta,
    }


def adapt_result(raw: dict[str, Any], task_id: str) -> dict[str, Any]:
    """Convert raw annotator JSON → frontend TaskResult dict."""

    assessment = raw.get("match_assessment", {})

    match_scores: dict[str, Any] = {}
    for key, label in _LABEL_MAP.items():
        level_str = str(assessment.get(key, "low")).lower()
        match_scores[key] = {
            "label": label,
            **_level_to_score(level_str),
        }

    # JD keywords
    jd_keywords = {
        "hardSkills": raw.get("jd_hard_skills", []),
        "softSkills": raw.get("jd_soft_skills", []),
        "coreResponsibilities": raw.get("jd_responsibilities", []),
    }
    # Fallback: split flat jd_keywords list if structured fields are empty
    if not any(jd_keywords.values()):
        flat = raw.get("jd_keywords", [])
        jd_keywords["hardSkills"] = flat[:5]
        jd_keywords["softSkills"] = flat[5:10]
        jd_keywords["coreResponsibilities"] = flat[10:]

    # Issues
    issues = []
    for item in raw.get("issues", []):
        issues.append({
            "id": item.get("id", f"issue_{len(issues)+1}"),
            "title": item.get("title", ""),
            "severity": item.get("severity", "medium"),
            "category": item.get("category", ""),
            "starGap": item.get("star_gap") or None,
            "page": int(item.get("page", 1)),
            "location": item.get("focus_text", "").strip() or item.get("location", ""),
            "comment": item.get("comment", ""),
            "rewriteTip": item.get("rewrite_tip", "") or item.get("rewrite_example", ""),
        })

    # Consultation
    consultation = _build_consultation(raw)

    return {
        "summary": raw.get("summary", ""),
        "jdKeywords": jd_keywords,
        "matchScores": match_scores,
        "strengths": raw.get("strengths", []),
        "weaknesses": raw.get("weaknesses", []),
        "issues": issues,
        "consultation": consultation,
        "downloads": {
            "annotatedPdf": f"annotated_{task_id}.pdf",
            "report": f"report_{task_id}.html",
        },
    }
