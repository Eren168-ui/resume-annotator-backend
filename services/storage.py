"""File storage abstraction — local disk, ready to swap for S3/R2."""

import os
import shutil
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"


# ── Directory helpers ─────────────────────────────────────────────────────────

def task_upload_dir(task_id: str) -> Path:
    p = UPLOADS_DIR / task_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def task_output_dir(task_id: str) -> Path:
    p = OUTPUTS_DIR / task_id
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Save uploaded file ─────────────────────────────────────────────────────────

def save_upload(task_id: str, filename: str, data: bytes) -> Path:
    dest = task_upload_dir(task_id) / filename
    dest.write_bytes(data)
    return dest


def delete_task_files(task_id: str) -> None:
    for directory in (UPLOADS_DIR / task_id, OUTPUTS_DIR / task_id):
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)


# ── Paths for known output artifacts ─────────────────────────────────────────

def result_json_path(task_id: str) -> Path:
    return task_output_dir(task_id) / "result.json"


def annotated_pdf_path(task_id: str) -> Path:
    return task_output_dir(task_id) / "annotated.pdf"


def report_md_path(task_id: str) -> Path:
    return task_output_dir(task_id) / "report.md"


def report_html_path(task_id: str) -> Path:
    return task_output_dir(task_id) / "report.html"


def build_download_manifest(task_id: str) -> dict:
    manifest = {"pageImages": []}
    file_map = {
        "annotatedPdf": (annotated_pdf_path(task_id), f"annotated_{task_id}.pdf"),
        "markdownReport": (report_md_path(task_id), f"report_{task_id}.md"),
        "report": (report_html_path(task_id), f"report_{task_id}.html"),
        "jsonResult": (result_json_path(task_id), f"result_{task_id}.json"),
    }

    for key, (path, filename) in file_map.items():
        if path.exists():
            manifest[key] = filename

    return manifest


# ── File existence check ──────────────────────────────────────────────────────

def get_download_path(task_id: str, file_key: str) -> Path | None:
    """Return the local path for a download file key, or None if not found."""
    mapping = {
        "annotated_pdf": annotated_pdf_path(task_id),
        "markdown_report": report_md_path(task_id),
        "report": report_html_path(task_id),
        "json_result": result_json_path(task_id),
    }
    path = mapping.get(file_key)
    if path and path.exists():
        return path
    return None
