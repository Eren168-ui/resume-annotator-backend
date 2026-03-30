"""Task lifecycle management — create, update status, query."""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from models.db import get_connection
from models.schemas import TaskStatus

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row) -> dict:
    if row is None:
        return None
    d = dict(row)
    candidate_name = _derive_candidate_name(d.get("candidate_name"), d.get("resume_file"))
    jd_title = _derive_jd_title(d.get("jd_title"), d.get("result_json"))
    return {
        "id": d["id"],
        "createdAt": d["created_at"],
        "status": d["status"],
        "statusMessage": d.get("status_message"),
        "candidateName": candidate_name,
        "jdTitle": jd_title,
        "jdFile": d.get("jd_file"),
        "resumeFile": d.get("resume_file"),
        "resumePages": d.get("resume_pages"),
        "failReason": d.get("fail_reason"),
    }


def _looks_like_dirty_candidate_name(value: Optional[str]) -> bool:
    text = (value or "").strip()
    if not text:
        return True
    if text in {"未知", "未知候选人", "null", "NULL", "None"}:
        return True
    return any(token in text for token in [".pdf", "_", "@", "微信", "Only", "第"]) or bool(re.search(r"\d{4,}", text))


def _extract_candidate_name_from_resume_file(resume_file: Optional[str]) -> Optional[str]:
    stem = Path(resume_file or "").stem
    if not stem:
        return None
    match = re.search(r"([\u4e00-\u9fa5]{2,4})(?=简历)", stem)
    if match:
        return match.group(1)
    candidates = [
        token for token in re.findall(r"[\u4e00-\u9fa5]{2,4}", stem)
        if token not in {"简历", "微信"}
    ]
    return candidates[-1] if candidates else None


def _derive_candidate_name(candidate_name: Optional[str], resume_file: Optional[str]) -> Optional[str]:
    if candidate_name and not _looks_like_dirty_candidate_name(candidate_name):
        return candidate_name.strip()
    extracted = _extract_candidate_name_from_resume_file(resume_file)
    return extracted or (candidate_name.strip() if candidate_name else None)


def _looks_like_invalid_jd_title(value: Optional[str]) -> bool:
    text = (value or "").strip()
    if not text or text == "未知岗位":
        return True
    if len(text) > 14:
        return True
    return text.startswith(("主导", "负责", "组织", "确保", "推进", "对活动"))


def _infer_jd_title_from_result(result_json: Optional[str]) -> Optional[str]:
    if not result_json:
        return None
    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        return None

    texts = [
        str(data.get("summary", "")),
        str((data.get("consultation") or {}).get("reason", "")),
    ]
    jd_keywords = data.get("jdKeywords") or {}
    texts.extend(jd_keywords.get("coreResponsibilities") or [])
    texts.extend(jd_keywords.get("hardSkills") or [])
    joined = "\n".join(t for t in texts if t)

    quoted = re.search(r'[“"]([^”"\n]{2,20})[”"]岗位', joined)
    if quoted:
        return quoted.group(1).strip()

    keyword_map = [
        ("活动策划执行专员", ("活动", "策划", "执行")),
        ("产品运营专员", ("产品", "运营")),
        ("市场推广", ("市场", "推广")),
        ("校园招聘专员", ("校园招聘",)),
    ]
    for title, keywords in keyword_map:
        if all(keyword in joined for keyword in keywords):
            return title
    return None


def _derive_jd_title(jd_title: Optional[str], result_json: Optional[str]) -> Optional[str]:
    if jd_title and not _looks_like_invalid_jd_title(jd_title):
        return jd_title.strip()
    inferred = _infer_jd_title_from_result(result_json)
    return inferred or (jd_title.strip() if jd_title else None)


def create_task(
    jd_file: str,
    resume_file: str,
    note: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    task_id = str(uuid.uuid4())[:12]
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO tasks (id, created_at, status, jd_file, resume_file, note, user_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (task_id, _now_iso(), TaskStatus.PENDING, jd_file, resume_file, note, user_id),
        )
        conn.commit()
    logger.info("Created task %s", task_id)
    return task_id


def set_processing(task_id: str, status_message: Optional[str] = None) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE tasks SET status = ?, status_message = ? WHERE id = ?",
            (TaskStatus.PROCESSING, status_message, task_id),
        )
        conn.commit()


def set_status_message(task_id: str, status_message: Optional[str]) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE tasks SET status_message = ? WHERE id = ?",
            (status_message, task_id),
        )
        conn.commit()


def set_completed(
    task_id: str,
    adapted_result: dict,
    candidate_name: str,
    jd_title: str,
    resume_pages: Optional[int] = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """UPDATE tasks SET status = ?, result_json = ?, candidate_name = ?,
               jd_title = ?, resume_pages = ?, status_message = NULL WHERE id = ?""",
            (
                TaskStatus.COMPLETED,
                json.dumps(adapted_result, ensure_ascii=False),
                candidate_name,
                jd_title,
                resume_pages,
                task_id,
            ),
        )
        conn.commit()
    logger.info("Task %s completed", task_id)


def set_failed(task_id: str, reason: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE tasks SET status = ?, fail_reason = ?, status_message = NULL WHERE id = ?",
            (TaskStatus.FAILED, reason, task_id),
        )
        conn.commit()
    logger.warning("Task %s failed: %s", task_id, reason)


def get_task(task_id: str) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return _row_to_dict(row)


def list_tasks(page: int = 1, limit: int = 20) -> tuple[list[dict], int]:
    offset = (page - 1) * limit
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [_row_to_dict(r) for r in rows], total


def get_task_result(task_id: str) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT status, result_json FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
    if row is None:
        return None
    if row["status"] != TaskStatus.COMPLETED:
        return None
    if not row["result_json"]:
        return None
    return json.loads(row["result_json"])


def delete_task(task_id: str) -> bool:
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        conn.commit()
    return cursor.rowcount > 0
