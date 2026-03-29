"""SQLite database initialization and connection management."""

import sqlite3
import os
from pathlib import Path

DB_PATH = Path(os.getenv("DATA_DIR", "./data")) / "tasks.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id          TEXT PRIMARY KEY,
                created_at  TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                user_id     TEXT,
                jd_file     TEXT,
                resume_file TEXT,
                resume_pages INTEGER,
                note        TEXT,
                candidate_name TEXT,
                jd_title    TEXT,
                status_message TEXT,
                fail_reason TEXT,
                result_json TEXT
            )
        """)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
        if "status_message" not in columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN status_message TEXT")
        conn.commit()
