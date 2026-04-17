"""
JWT auth with local SQLite user store (doctor / admin roles).
Set AUTH_DISABLED=1 for tests or local stack without login.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "medassist_auth.db"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)

JWT_SECRET = os.environ.get("JWT_SECRET", "dev-insecure-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "1440"))
DEFAULT_ORG_ID = os.environ.get("DEFAULT_ORG_ID", "default")
AUTH_DISABLED = os.environ.get("AUTH_DISABLED", "").strip().lower() in ("1", "true", "yes")
AUTH_DISABLED_ROLE = os.environ.get("AUTH_DISABLED_ROLE", "admin").strip().lower()
ALLOW_OPEN_REGISTRATION = os.environ.get("ALLOW_OPEN_REGISTRATION", "").strip().lower() in ("1", "true", "yes")


@dataclass
class TokenUser:
    sub: str
    username: str
    role: str
    org_id: str


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    _ensure_data_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_user_db() -> None:
    _ensure_data_dir()
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                org_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
    _bootstrap_users_if_empty()


def _bootstrap_users_if_empty() -> None:
    conn = _connect()
    try:
        n = conn.execute("SELECT COUNT(*) FROM app_users").fetchone()[0]
        if n > 0:
            return
        pairs = [
            (
                os.environ.get("MEDASSIST_BOOTSTRAP_DOCTOR_USER"),
                os.environ.get("MEDASSIST_BOOTSTRAP_DOCTOR_PASSWORD"),
                "doctor",
            ),
            (
                os.environ.get("MEDASSIST_BOOTSTRAP_ADMIN_USER"),
                os.environ.get("MEDASSIST_BOOTSTRAP_ADMIN_PASSWORD"),
                "admin",
            ),
        ]
        raw = os.environ.get("MEDASSIST_BOOTSTRAP_JSON", "").strip()
        if raw:
            try:
                for row in json.loads(raw):
                    u = row.get("username")
                    p = row.get("password")
                    r = (row.get("role") or "doctor").lower()
                    o = row.get("org_id") or DEFAULT_ORG_ID
                    if u and p and r in ("doctor", "admin"):
                        _insert_user(conn, str(u), str(p), r, str(o))
            except (json.JSONDecodeError, TypeError):
                pass
        for username, password, role in pairs:
            if username and password:
                _insert_user(conn, username.strip(), password, role, DEFAULT_ORG_ID)
        conn.commit()
    finally:
        conn.close()


def _insert_user(conn: sqlite3.Connection, username: str, password: str, role: str, org_id: str) -> None:
    uid = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO app_users (id, username, password_hash, role, org_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (uid, username.lower(), pwd_context.hash(password), role, org_id, datetime.now(timezone.utc).isoformat()),
    )


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def get_user_by_username(username: str) -> dict[str, Any] | None:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, username, password_hash, role, org_id FROM app_users WHERE username = ?",
            (username.lower().strip(),),
        ).fetchone()
        if not row:
            return None
        return dict(row)
    finally:
        conn.close()


def create_user(username: str, password: str, role: str, org_id: str) -> dict[str, Any]:
    if role not in ("doctor", "admin"):
        raise ValueError("role must be doctor or admin")
    uid = str(uuid.uuid4())
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO app_users (id, username, password_hash, role, org_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (uid, username.lower().strip(), pwd_context.hash(password), role, org_id, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    except sqlite3.IntegrityError as e:
        raise ValueError("username already exists") from e
    finally:
        conn.close()
    return {"id": uid, "username": username.lower().strip(), "role": role, "org_id": org_id}


def create_access_token(*, sub: str, username: str, role: str, org_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": sub,
        "username": username,
        "role": role,
        "org_id": org_id,
        "iat": now,
        "exp": now + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


def _token_user_from_claims(data: dict[str, Any]) -> TokenUser:
    return TokenUser(
        sub=str(data.get("sub", "")),
        username=str(data.get("username", "")),
        role=str(data.get("role", "doctor")).lower(),
        org_id=str(data.get("org_id") or DEFAULT_ORG_ID),
    )


def auth_disabled_user() -> TokenUser:
    return TokenUser(
        sub="auth-disabled",
        username="auth-disabled",
        role=AUTH_DISABLED_ROLE if AUTH_DISABLED_ROLE in ("doctor", "admin") else "admin",
        org_id=DEFAULT_ORG_ID,
    )


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
) -> TokenUser:
    if AUTH_DISABLED:
        return auth_disabled_user()
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        data = decode_token(credentials.credentials)
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _token_user_from_claims(data)


def require_roles(*allowed: str):
    allowed_l = {a.lower() for a in allowed}

    async def _dep(user: Annotated[TokenUser, Depends(get_current_user)]) -> TokenUser:
        if user.role not in allowed_l:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user

    return _dep


def user_can_access_encounter(user: TokenUser, encounter: dict[str, Any]) -> bool:
    """Doctors: same org and (same owner or legacy row without owner). Admins: same org or global flag."""
    enc_org = (encounter.get("org_id") or "").strip() or DEFAULT_ORG_ID
    if user.org_id and enc_org != user.org_id:
        if user.role != "admin":
            return False
        if os.environ.get("MEDASSIST_ADMIN_CROSS_ORG", "").strip().lower() not in ("1", "true", "yes"):
            return False
    owner = (encounter.get("created_by_user_id") or "").strip()
    if user.role == "admin":
        return True
    if user.role == "doctor":
        if not owner:
            return True
        return owner == user.sub
    return False


CurrentUser = Annotated[TokenUser, Depends(get_current_user)]
AdminUser = Annotated[TokenUser, Depends(require_roles("admin"))]
