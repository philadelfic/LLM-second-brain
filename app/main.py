"""FastAPI-приложение LLM Second Brain.

Собирает каркас Фазы 1: REST-ручки (сейчас `/health`), Bearer-миддлварь
(NFR-2). MCP-сервер (`/mcp`, инструменты `memory_*`) монтируется Шагом 3.

Запуск: `python -m app` (uvicorn) — порт и уровень логов из окружения.
"""

from __future__ import annotations

import sys

from fastapi import FastAPI

from app import __version__
from app.config import ConfigError, Settings, get_settings
from app.transport.auth import BearerAuthMiddleware
from app.transport.rest import rest_router

# Пути, доступные без Bearer-токена (NFR-2: только /health).
OPEN_PATHS = frozenset({"/health"})


def create_app(settings: Settings | None = None) -> FastAPI:
    """Собрать приложение; настройки — из окружения, если не переданы явно."""
    if settings is None:
        try:
            settings = get_settings()
        except ConfigError as exc:
            # Фатальная ошибка конфигурации: понятное сообщение вместо traceback.
            print(f"FATAL: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc

    app = FastAPI(
        title="LLM Second Brain",
        version=__version__,
        # REST — внутренняя поверхность (REQUIREMENTS §3): OpenAPI-документацию
        # не публикуем, меньше лишних ручек за миддлварью.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.include_router(rest_router)
    app.add_middleware(
        BearerAuthMiddleware,
        token=settings.mcp_auth_token,
        exempt_paths=OPEN_PATHS,
    )
    return app


app = create_app()