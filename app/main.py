"""FastAPI-приложение LLM Second Brain.

Собирает каркас Фазы 1: REST-ручки (`/health`), MCP-сервер (`/mcp`, 6
инструментов `memory_*`), Bearer-миддлварь на всё, кроме `/health` (NFR-2).
Фаза 2: при старте инициализируется хранилище (SQLite-схема, ARCH §3.3).

Запуск: `python -m app` (uvicorn) — порт и уровень логов из окружения.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.config import ConfigError, Settings, get_settings
from app.storage.db import StorageError, init_db
from app.transport.auth import BearerAuthMiddleware
from app.transport.mcp import build_mcp
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

    mcp = build_mcp(settings)
    # Внутренний маршрут MCP-сервера — ровно MCP_PATH. host="0.0.0.0" — не
    # localhost, поэтому SDK не включает DNS-rebinding protection (сервис
    # живёт в LAN за Bearer-токеном; Open WebUI ходит с не-localhost Host).
    mcp_app = mcp.streamable_http_app(
        streamable_http_path=settings.mcp_path,
        host="0.0.0.0",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Инициализация хранилища при старте (критерий приёмки Фазы 2):
        # схема создаётся/сверяется до первого запроса; ошибка — фатальна.
        try:
            init_db(settings)
        except StorageError as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        # Жизненный цикл MCP-сессий Streamable HTTP = жизни процесса.
        async with mcp.session_manager.run():
            yield

    app = FastAPI(
        title="LLM Second Brain",
        version=__version__,
        # REST — внутренняя поверхность (REQUIREMENTS §3): OpenAPI-документацию
        # не публикуем, меньше лишних ручек за миддлварью.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.include_router(rest_router)
    # Монтирование в корень: MCP_PATH задаёт точный путь MCP-эндпоинта;
    # маршруты FastAPI (/health и будущие REST) матчатся раньше mount.
    app.mount("/", mcp_app)
    app.add_middleware(
        BearerAuthMiddleware,
        token=settings.mcp_auth_token,
        exempt_paths=OPEN_PATHS,
    )
    return app


app = create_app()