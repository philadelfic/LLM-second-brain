"""Bearer authentication (REQUIREMENTS NFR-2, §7).

The middleware covers everything except `/health`:
`Authorization: Bearer <MCP_AUTH_TOKEN>`. A missing or wrong token → 401.

The implementation is a pure ASGI middleware (no BaseHTTPMiddleware): it does
not interfere with response streaming, which is critical for MCP Streamable
HTTP (SSE streams). The token comparison — `secrets.compare_digest`
(timing-safe) over bytes: this way a non-ASCII value in the header (e.g.
`Bearer ñ`) is handled regularly (a mismatch → 401) instead of crashing the
middleware with a TypeError.
"""

from __future__ import annotations

import secrets
from typing import Any

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

UNAUTHORIZED_DETAIL = "Unauthorized"


class BearerAuthMiddleware:
    """Lets through requests with a correct Bearer token; answers 401 to the rest."""

    def __init__(self, app: ASGIApp, token: str, exempt_paths: frozenset[str]) -> None:
        self.app = app
        self.token = token
        self.exempt_paths = exempt_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            # Non-HTTP protocols (e.g. lifespan) need no authorization.
            await self.app(scope, receive, send)
            return

        if scope["path"] in self.exempt_paths:
            await self.app(scope, receive, send)
            return

        authorization = Headers(scope=scope).get("Authorization", "")
        scheme, _, value = authorization.partition(" ")
        # The scheme is case-insensitive (RFC 7235); the value is case-sensitive.
        if scheme.lower() != "bearer" or not secrets.compare_digest(
            value.encode("utf-8"), self.token.encode("utf-8")
        ):
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