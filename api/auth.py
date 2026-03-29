"""JWT authentication — MVP with fixed credentials from env."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import jwt
from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from models.schemas import LoginRequest, LoginResponse, ErrorOut

router = APIRouter()
_bearer = HTTPBearer(auto_error=False)

JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_SECONDS = int(os.getenv("JWT_EXPIRE_SECONDS", str(60 * 60 * 24)))  # 24h


def _is_dev_auth_bypass_enabled() -> bool:
    return os.getenv("DEV_AUTH_BYPASS", "").strip().lower() == "true"


def _load_users_from_env() -> dict[str, dict[str, str]]:
    raw = os.getenv("AUTH_USERS_JSON", "").strip()
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("AUTH_USERS_JSON 不是合法 JSON") from exc

    users: dict[str, dict[str, str]] = {}
    for index, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            raise RuntimeError("AUTH_USERS_JSON 中的每个账号都必须是对象")

        email = str(item.get("email", "")).strip()
        password = str(item.get("password", "")).strip()
        if not email or not password:
            raise RuntimeError("AUTH_USERS_JSON 中的账号必须包含 email 和 password")

        users[email] = {
            "id": str(item.get("id") or f"u{index}"),
            "name": str(item.get("name") or f"用户 {index}"),
            "password": password,
        }
    return users

def _get_users() -> dict[str, dict[str, str]]:
    users = _load_users_from_env()
    if users:
        return users

    admin_email = os.getenv("ADMIN_EMAIL", "admin@example.com").strip()
    admin_password = os.getenv("ADMIN_PASSWORD", "changeme").strip()
    return {
        admin_email: {
            "id": "u1",
            "name": "管理员",
            "password": admin_password,
        }
    }


def _make_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRE_SECONDS,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail={"message": "Token 已过期，请重新登录"})
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail={"message": "无效 Token"})


def require_auth(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
) -> dict:
    if _is_dev_auth_bypass_enabled():
        return {"sub": "u1", "email": os.getenv("ADMIN_EMAIL", "admin@erenlab.cn")}
    if not credentials:
        raise HTTPException(status_code=401, detail={"message": "未登录"})
    return _decode_token(credentials.credentials)


@router.post("/auth/login", response_model=LoginResponse)
async def login(body: LoginRequest):
    if _is_dev_auth_bypass_enabled():
        token = _make_token("u1", body.email or os.getenv("ADMIN_EMAIL", "admin@erenlab.cn"))
        return {
            "token": token,
            "user": {"id": "u1", "email": body.email or os.getenv("ADMIN_EMAIL", "admin@erenlab.cn"), "name": "管理员"},
        }
    user = _get_users().get(body.email)
    if not user or user["password"] != body.password:
        raise HTTPException(
            status_code=401,
            detail={"message": "邮箱或密码错误"},
        )
    token = _make_token(user["id"], body.email)
    return {
        "token": token,
        "user": {"id": user["id"], "email": body.email, "name": user["name"]},
    }
