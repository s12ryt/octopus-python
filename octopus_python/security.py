from __future__ import annotations

import secrets
import string
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

API_KEY_PREFIX = "sk-octopus-"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


def token_expire_from_minutes(expire: int) -> datetime:
    if expire == -1:
        delta = timedelta(days=30)
    elif expire > 0:
        delta = timedelta(minutes=expire)
    else:
        delta = timedelta(minutes=15)
    return datetime.now(timezone.utc) + delta


def generate_jwt(username: str, password_hash: str, expire: int) -> tuple[str, datetime]:
    expire_at = token_expire_from_minutes(expire)
    payload: dict[str, Any] = {
        "sub": username,
        "iss": "octopus",
        "exp": expire_at,
        "iat": datetime.now(timezone.utc),
    }
    token = jwt.encode(payload, username + password_hash, algorithm="HS256")
    return token, expire_at


def verify_jwt(token: str, username: str, password_hash: str) -> dict[str, Any] | None:
    try:
        return jwt.decode(token, username + password_hash, algorithms=["HS256"], issuer="octopus")
    except Exception:
        return None


def generate_api_key() -> str:
    alphabet = string.ascii_letters + string.digits
    return API_KEY_PREFIX + "".join(secrets.choice(alphabet) for _ in range(48))


def generate_stream_token() -> str:
    return secrets.token_hex(32)
