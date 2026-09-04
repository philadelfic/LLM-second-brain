"""FastAPI-приложение LLM Second Brain.

Каркас Фазы 1: REST, MCP-сервер (`/mcp`, 6 инструментов `memory_*`),
Bearer-миддлварь на всё, кроме `/health` (NFR-2). Фаза 2: при старте
инициализируется хранилище (SQLite, ARCH §3.3); MCP и REST работают
над общим service-слоем (ARCH §1) через `app.state.services`.
Фаза 3–4: в lifespan поднимается фоновый воркер (ARCH §3.4) — две
независимые очереди pending_vector (до-векторизация) и pending_summary
(до-суммаризация, режим «Б» — единственный путь генерации summary);
очереди живут в БД и переживают рестарт; при останове — graceful отмена.

Запуск: `python -m app` (uvicorn) — порт и уровень логов из окружения.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI

from app import __version__
from app.config import ConfigError, Settings, get_settings
from app.observability import setup_logging
from app.services import build_services
from app.services.worker import BackgroundWorker
from app.storage.db import StorageError, init_db
from app.transport.auth import BearerAuthMiddleware
from app.transport.mcp import build_mcp
from app.transport.rest import build_rest_router

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

    services = build_services(settings)  # один Settings-снимок на процесс
    # JSON-логи stdout (NFR-4): включаются при сборке приложения (импорт
    # app.main / тесты / uvicorn — один и тот же идемпотентный setup).
    setup_logging(settings.log_level)
    mcp = build_mcp(settings, services)
    # §3.4: две очереди — векторы (Фаза 3) и summary (Фаза 4, режим «Б»);
    # суммаризатор из DI — тот же экземпляр, что в /health.summarizer_ok.
    # Судья дедупа (Фаза 8, Этап 3.1) — тоже DI из services, один экземпляр
    # на процесс (вердикты по кандидатам — Задача 3.2).
    worker = BackgroundWorker(
        settings,
        services.embedding,
        services.summary,
        judge=services.judge,
        classifier=services.classifier,
        promoter=services.promotion,
    )
    # Суммаризация стартует сразу при save/update: NoteService сигналит
    # воркеру, тот немедленно догоняет pending_summary (не ждёт back-off).
    services.notes.set_summary_notifier(worker.notify_summary_pending)
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
        logging.getLogger("app").info(
            "service started",
            extra={
                "event": "startup",
                "version": __version__,
                "port": settings.port,
                "mcp_path": settings.mcp_path,
                "db_path": settings.db_path,
            },
        )
        # Жизненный цикл MCP-сессий Streamable HTTP = жизни процесса.
        async with mcp.session_manager.run():
            # Фоновый воркер: очереди pending живут в БД — после рестарта
            # дорезюмируются/довекторизуются сами (ARCH §3.4); при останове —
            # отмена таски.
            worker_task = asyncio.create_task(worker.run(), name="pending-backlog")
            # BackupService (Фаза 5, NFR-3): снапшот сразу после старта,
            # далее — раз в BACKUP_INTERVAL_SEC; отказы — в лог, петля живёт.
            backup_task = asyncio.create_task(
                services.backup.run(), name="backup-snapshots"
            )
            try:
                yield
            finally:
                worker.stop()  # мягкий флаг: не начинать новую партию
                worker_task.cancel()  # и прервать ожидание, если спит
                with suppress(asyncio.CancelledError):
                    await worker_task
                services.backup.stop()
                backup_task.cancel()
                with suppress(asyncio.CancelledError):
                    await backup_task

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
    app.include_router(build_rest_router(settings))
    app.state.services = services  # REST-ручки достают сервисы отсюда
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