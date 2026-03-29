"""Pydantic schemas for request/response validation."""

from __future__ import annotations
from typing import Optional, List, Any
from pydantic import BaseModel


# ── Auth ──────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    token: str
    user: dict


# ── Task ──────────────────────────────────────────────────────────────────────

class TaskStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskOut(BaseModel):
    id: str
    createdAt: str
    status: str
    statusMessage: Optional[str] = None
    candidateName: Optional[str] = None
    jdTitle: Optional[str] = None
    jdFile: Optional[str] = None
    resumeFile: Optional[str] = None
    resumePages: Optional[int] = None
    failReason: Optional[str] = None


class TaskListOut(BaseModel):
    tasks: List[TaskOut]
    total: int


class CreateTaskOut(BaseModel):
    id: str
    status: str


# ── Result ────────────────────────────────────────────────────────────────────

class MatchScore(BaseModel):
    score: int
    label: str
    level: str


class MatchScores(BaseModel):
    keyword_coverage: MatchScore
    professionalism: MatchScore
    clarity: MatchScore
    fit: MatchScore


class JdKeywords(BaseModel):
    hardSkills: List[str]
    softSkills: List[str]
    coreResponsibilities: List[str]


class Issue(BaseModel):
    id: str
    title: str
    severity: str
    category: str
    starGap: Optional[str] = None
    page: int
    location: str
    comment: str
    rewriteTip: str


class Consultation(BaseModel):
    recommend: bool
    headline: Optional[str] = None
    reason: str
    priorities: List[str]
    prepItems: List[str]
    cta: Optional[str] = None


class Downloads(BaseModel):
    annotatedPdf: str
    report: str


class TaskResult(BaseModel):
    summary: str
    jdKeywords: JdKeywords
    matchScores: MatchScores
    strengths: List[str]
    weaknesses: List[str]
    issues: List[Issue]
    consultation: Consultation
    downloads: Downloads


# ── Download ──────────────────────────────────────────────────────────────────

class DownloadUrlOut(BaseModel):
    url: str


# ── Error ────────────────────────────────────────────────────────────────────

class ErrorOut(BaseModel):
    message: str
