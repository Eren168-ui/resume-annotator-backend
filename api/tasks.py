"""Task API endpoints."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Response
from fastapi.responses import FileResponse

from api.auth import require_auth
from models.schemas import CreateTaskOut, TaskListOut, TaskOut, TaskStatus
from services import task_service, storage
from services.processor import process_task

logger = logging.getLogger(__name__)
router = APIRouter()


def _task_max_workers() -> int:
    try:
        value = int(os.getenv("TASK_MAX_WORKERS", "3"))
    except ValueError:
        return 3
    return value if value >= 1 else 3


# Thread pool for concurrent annotation jobs
_pool = ThreadPoolExecutor(max_workers=_task_max_workers())

ALLOWED_RESUME_TYPES = {"application/pdf", "application/octet-stream"}
ALLOWED_JD_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


def _build_task_error_message(full_msg: str) -> str:
    if "pdftotext" in full_msg or "pdftoppm" in full_msg:
        return "简历 PDF 解析失败，请确认文件为可文字版 PDF（非扫描件）。"
    if "OPENAI_API_KEY" in full_msg or "ANTHROPIC_API_KEY" in full_msg or "API key" in full_msg.lower():
        return "服务端 API 密钥未配置，请联系管理员。"
    if "HTTP 401" in full_msg:
        return f"API 认证失败 (401)，请检查 API Key 是否正确。详情: {full_msg[:300]}"
    if "HTTP 429" in full_msg:
        return f"API 请求频率超限 (429)，请稍后重试。详情: {full_msg[:300]}"
    if "HTTP 4" in full_msg or "HTTP 5" in full_msg:
        return full_msg
    return full_msg


# ── Task list ──────────────────────────────────────────────────────────────────

@router.get("/tasks", response_model=TaskListOut)
async def list_tasks(
    page: int = 1,
    limit: int = 20,
    _user: dict = Depends(require_auth),
):
    tasks, total = task_service.list_tasks(page=page, limit=limit)
    return {"tasks": tasks, "total": total}


# ── Single task ────────────────────────────────────────────────────────────────

@router.get("/tasks/{task_id}", response_model=TaskOut)
async def get_task(task_id: str, _user: dict = Depends(require_auth)):
    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail={"message": f"任务 {task_id} 不存在"})
    return task


@router.delete("/tasks/{task_id}", status_code=204)
async def delete_task(task_id: str, _user: dict = Depends(require_auth)):
    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail={"message": f"任务 {task_id} 不存在"})

    if task["status"] in {TaskStatus.PENDING, TaskStatus.PROCESSING}:
        raise HTTPException(
            status_code=409,
            detail={"message": "处理中或排队中的任务暂不支持删除，请等待任务结束后再删除。"},
        )

    deleted = task_service.delete_task(task_id)
    if not deleted:
        raise HTTPException(status_code=404, detail={"message": f"任务 {task_id} 不存在"})

    storage.delete_task_files(task_id)
    return Response(status_code=204)


# ── Create task ────────────────────────────────────────────────────────────────

@router.post("/tasks", response_model=CreateTaskOut, status_code=201)
async def create_task(
    jd_image: UploadFile = File(...),
    resume_pdf: UploadFile = File(...),
    note: Optional[str] = Form(None),
    _user: dict = Depends(require_auth),
):
    # Validate JD image
    jd_content_type = jd_image.content_type or ""
    if not jd_content_type.startswith("image/"):
        raise HTTPException(
            status_code=422,
            detail={"message": "JD 文件必须是图片（PNG / JPG / JPEG / WEBP）"},
        )

    # Validate resume PDF
    resume_content_type = resume_pdf.content_type or ""
    if "pdf" not in resume_content_type and "octet-stream" not in resume_content_type:
        # Allow by extension as fallback
        if not resume_pdf.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=422,
                detail={"message": "简历必须是 PDF 格式"},
            )

    # Read files
    jd_data = await jd_image.read()
    resume_data = await resume_pdf.read()

    if len(jd_data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail={"message": "JD 图片超过 20MB 限制"})
    if len(resume_data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail={"message": "简历 PDF 超过 20MB 限制"})
    if len(resume_data) < 100:
        raise HTTPException(status_code=422, detail={"message": "简历 PDF 文件为空或损坏"})

    # Save files
    jd_ext = Path(jd_image.filename or "jd.png").suffix or ".png"
    jd_filename = f"jd{jd_ext}"
    resume_filename = resume_pdf.filename or "resume.pdf"

    task_id = task_service.create_task(
        jd_file=jd_image.filename or jd_filename,
        resume_file=resume_filename,
        note=note,
        user_id=_user.get("sub"),
    )

    storage.save_upload(task_id, jd_filename, jd_data)
    storage.save_upload(task_id, resume_filename, resume_data)

    # Submit to background thread pool
    _pool.submit(_run_processing, task_id, jd_filename, resume_filename)

    return {"id": task_id, "status": "pending"}


def _run_processing(task_id: str, jd_filename: str, resume_filename: str) -> None:
    """Runs in a background thread."""
    task_service.set_processing(task_id)
    try:
        result = process_task(task_id, jd_filename, resume_filename)
        task_service.set_completed(
            task_id,
            adapted_result=result["adapted_result"],
            candidate_name=result["candidate_name"],
            jd_title=result["jd_title"],
        )
    except Exception as exc:
        # Log full traceback so operator can see exact error
        logger.exception("[%s] Task failed: %s", task_id, exc)

        # Build a user-visible message: specific diagnosis first, full msg as fallback
        full_msg = str(exc)
        task_service.set_failed(task_id, _build_task_error_message(full_msg))


# ── Task result ────────────────────────────────────────────────────────────────

@router.get("/tasks/{task_id}/result")
async def get_task_result(task_id: str, _user: dict = Depends(require_auth)):
    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail={"message": f"任务 {task_id} 不存在"})
    if task["status"] != "completed":
        raise HTTPException(
            status_code=404,
            detail={"message": f"任务尚未完成，当前状态: {task['status']}"},
        )
    result = task_service.get_task_result(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail={"message": "结果数据不存在"})
    return result


# ── Download ───────────────────────────────────────────────────────────────────

_DOWNLOAD_META = {
    "annotated_pdf": ("annotated.pdf", "application/pdf"),
    "report": ("report.html", "text/html; charset=utf-8"),
}


@router.get("/tasks/{task_id}/download/{file_key}")
async def download_file(
    task_id: str,
    file_key: str,
    _user: dict = Depends(require_auth),
):
    if file_key not in _DOWNLOAD_META:
        raise HTTPException(
            status_code=400,
            detail={"message": f"不支持的文件类型: {file_key}。支持: annotated_pdf, report"},
        )

    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail={"message": "任务不存在"})
    if task["status"] != "completed":
        raise HTTPException(status_code=404, detail={"message": "任务尚未完成"})

    path = storage.get_download_path(task_id, file_key)
    if path is None:
        raise HTTPException(status_code=404, detail={"message": "文件尚未生成，请稍后重试"})

    _, media_type = _DOWNLOAD_META[file_key]

    # Build a human-readable filename from candidate name
    candidate = (task.get("candidateName") or "").strip()
    import re as _re
    safe = _re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', candidate)[:20]
    if not safe:
        safe = task_id[:8]

    if file_key == "annotated_pdf":
        readable_filename = f"{safe}简历批注.pdf"
    else:
        readable_filename = f"{safe}分析报告.html"

    return FileResponse(
        path=str(path),
        media_type=media_type,
        filename=readable_filename,
    )
