"""FastAPI application entry point."""

import logging
import os

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

# CORS — allow the frontend origin(s)
LOCAL_FRONTEND_ORIGINS = [
    "http://localhost:4173",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:5176",
    "http://127.0.0.1:4173",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
    "http://127.0.0.1:5176",
]


def build_allowed_origins(
    frontend_origin: str | None = None,
    frontend_origins: str | None = None,
    local_frontend_origins: list[str] | None = None,
) -> list[str]:
    """Build a stable deduped CORS allowlist.

    Supports either a single FRONTEND_ORIGIN or a comma-separated FRONTEND_ORIGINS
    so production can allow both apex and www domains without code changes.
    """
    origins: list[str] = []

    if frontend_origin:
        origins.append(frontend_origin.strip())

    if frontend_origins:
        origins.extend(item.strip() for item in frontend_origins.split(","))

    if local_frontend_origins:
        origins.extend(local_frontend_origins)

    return list(dict.fromkeys(origin for origin in origins if origin))


ALLOWED_ORIGINS = build_allowed_origins(
    frontend_origin=os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"),
    frontend_origins=os.getenv("FRONTEND_ORIGINS", ""),
    local_frontend_origins=LOCAL_FRONTEND_ORIGINS,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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
