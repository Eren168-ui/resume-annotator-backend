"""FastAPI application entry point."""

import logging
import os
import re

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models.db import init_db
from api.auth import router as auth_router
from api.tasks import router as tasks_router

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="简历批注系统 API",
    version="1.0.0",
    docs_url="/docs",
)


def _split_origins(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


def _build_cors_settings() -> tuple[list[str], str | None]:
    allow_origins = {
        "http://localhost:5173",
        "http://localhost:4173",
    }
    allow_origins.update(_split_origins(os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")))
    allow_origins.update(_split_origins(os.getenv("FRONTEND_ORIGINS")))

    regex_parts: list[str] = []
    pages_project = (os.getenv("CLOUDFLARE_PAGES_PROJECT") or "resume-annotator-b2b").strip()
    if pages_project:
        allow_origins.add(f"https://{pages_project}.pages.dev")
        regex_parts.append(rf"^https://([a-z0-9-]+\.)?{re.escape(pages_project)}\.pages\.dev$")

    extra_origin_regex = (os.getenv("FRONTEND_ORIGIN_REGEX") or "").strip()
    if extra_origin_regex:
        regex_parts.append(extra_origin_regex)

    allow_origin_regex = "|".join(f"(?:{pattern})" for pattern in regex_parts) or None
    return sorted(allow_origins), allow_origin_regex


# CORS — allow production frontend + Cloudflare Pages preview domains.
ALLOW_ORIGINS, ALLOW_ORIGIN_REGEX = _build_cors_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(auth_router, prefix="/api")
app.include_router(tasks_router, prefix="/api")


# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"message": "服务器内部错误，请稍后重试"},
    )


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info("Initializing database...")
    ai_provider = os.getenv("AI_PROVIDER", "openai").lower()
    if ai_provider == "claude":
        base_url = (
            os.getenv("ANTHROPIC_BASE_URL")
            or os.getenv("ANTHROPIC_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "https://api.anthropic.com"
        )
        model = (
            os.getenv("ANTHROPIC_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "claude-opus-4-6"
        )
        logger.info("Using Claude API with base URL: %s", base_url)
        logger.info("Using Claude provider (model=%s, base=%s)", model, base_url)
    elif ai_provider == "openai":
        base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "https://api.openai.com"
        )
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        logger.info("Using OpenAI provider (model=%s, base=%s)", model, base_url)
    init_db()
    logger.info("Backend ready.")


@app.get("/health")
async def health():
    return {"status": "ok"}
