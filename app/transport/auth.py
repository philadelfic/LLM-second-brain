"""Bearer-аутентификация (REQUIREMENTS NFR-2, §7).

Миддлварь на всё, кроме `/health`: `Authorization: Bearer <MCP_AUTH_TOKEN>`.
Нет или неверный токен → 401.

Реализация — чистый ASGI-миддлварь (без BaseHTTPMiddleware): не вмешивается
в стриминг ответов, что критично для MCP Streamable HTTP (SSE-потоки).
Сравнение токена — `secrets.compare_digest` (timing-safe).
"""

from __future__ import annotations

import secrets
from typing import Any

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

UNAUTHORIZED_DETAIL = "Unauthorized"


class BearerAuthMiddleware:
    """Пропускает запросы с верным Bearer-токеном; остальным отвечает 401."""

    def __init__(self, app: ASGIApp, token: str, exempt_paths: frozenset[str]) -> None:
        self.app = app
        self.token = token
        self.exempt_paths = exempt_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            # Не-HTTP протоколы (например, lifespan) авторизации не требуют.
            await self.app(scope, receive, send)
            return

        if scope["path"] in self.exempt_paths:
            await self.app(scope, receive, send)
            return

        authorization = Headers(scope=scope).get("Authorization", "")
        scheme, _, value = authorization.partition(" ")
        # Схема нечувствительна к регистру (RFC 7235); значение — чувствительно.
        if scheme.lower() != "bearer" or not secrets.compare_digest(value, self.token):
            await self._unauthorized(scope, receive, send)
            return

        await self.app(scope, receive, send)

    @staticmethod
    async def _unauthorized(scope: Scope, receive: Receive, send: Send) -> None:
        response: Any = JSONResponse(
            {"detail": UNAUTHORIZED_DETAIL},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
        await response(scope, receive, send)