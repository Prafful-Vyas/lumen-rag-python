from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from lumen.composition import Container

bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthClaims:
    sub: str
    workspace_id: str
    role: Literal["admin", "member"]


def get_container(request: Request) -> Container:
    return request.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        return False


def sign_token(container: Container, claims: AuthClaims) -> str:
    payload = {
        "sub": claims.sub,
        "workspaceId": claims.workspace_id,
        "role": claims.role,
        "exp": datetime.now(UTC) + timedelta(days=container.settings.jwt_ttl_days),
    }
    return jwt.encode(payload, container.settings.jwt_secret, algorithm="HS256")


def require_auth(
    container: ContainerDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    token: Annotated[str | None, Query()] = None,
) -> AuthClaims:
    """The only place workspaceId enters the system for an authenticated request: it is read from
    the verified JWT and nowhere else. No handler reads a workspace id from the body or query
    string, so a client cannot even express a cross-tenant request. Plain link navigation (opening
    a cited file) cannot set a header, so a `token` query param carries the same verified JWT."""
    raw = credentials.credentials if credentials else token
    if not raw:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    try:
        payload = jwt.decode(raw, container.settings.jwt_secret, algorithms=["HS256"])
        return AuthClaims(sub=payload["sub"], workspace_id=payload["workspaceId"], role=payload["role"])
    except (jwt.PyJWTError, KeyError) as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token") from exc


AuthDep = Annotated[AuthClaims, Depends(require_auth)]


def require_admin(auth: AuthDep) -> AuthClaims:
    if auth.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin role required")
    return auth


AdminDep = Annotated[AuthClaims, Depends(require_admin)]


def rate_limit(window_seconds: int, max_requests: int | None, key_prefix: str):
    """Fixed-window Redis limiter per workspace. Coarse, but enough to blunt accidental hammering."""

    async def dependency(container: ContainerDep, auth: AuthDep) -> None:
        limit = max_requests if max_requests is not None else container.settings.query_rate_limit_per_minute
        window = int(time.time() // window_seconds)
        key = f"ratelimit:{key_prefix}:{auth.workspace_id}:{window}"
        count = await container.redis.incr(key)
        if count == 1:
            await container.redis.expire(key, window_seconds)
        if count > limit:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many requests, slow down")

    return dependency
