"""FastAPI-приложение LLM Second Brain.

Каркас Фазы 1: REST, MCP-сервер (`/mcp`, 6 инструментов `memory_*`),
Bearer-миддлварь на всё, кроме `/health` (NFR-2). Фаза 2: при старте
инициализируется хранилище (SQLite, ARCH §3.3); MCP и REST работают
над общим service-слоем (ARCH §1) через `app.state.services`.
Фаза 3–4: в lifespan поднимается фоновый воркер (ARCH §3.4) — две
независимые очереди pending_vector (до-векторизация) и pending_summary
(до-суммаризация, режим «Б» — единственный путь генерации summary);
очереди живут в БД и переживают рестарт; при останове — graceful отмена.
Фаза 11: перед стартом воркера — стартовая проверка трёх LLM-слотов
(решение №5): лёгкий GET без генерации, параллельно; отказ авторизации
(401/403) фатален (SystemExit 2 + hint про {SLOT}_API_KEY), недоступность
— WARN и штатный старт (NFR-3).

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
from app.services import Services, build_services
from app.services.llm_client import CHECK_AUTH_FAILED, CHECK_OK, LLMClient
from app.services.worker import BackgroundWorker
from app.storage.db import StorageError, init_db
from app.transport.auth import BearerAuthMiddleware
from app.transport.mcp import build_mcp
from app.transport.rest import build_rest_router

# Пути, доступные без Bearer-токена (NFR-2: только /health).
OPEN_PATHS = frozenset({"/health"})


def _startup_slots(services: Services) -> list[tuple[str, LLMClient, object]]:
    """Три LLM-слота стартовой проверки (решение №5): (имя, клиент, сервис).

    Сервис — носитель `last_attempt_ok` (/health): результат проверки
    инициализирует его до первого реального вызова. DI-сборки тестов без
    клиентов слотов (llm_* = None, сервисы подменены фейками) пропускаются:
    у фейков стартовая проверка — заглушка ok (бриф §3).
    """
    return [
        (slot, client, service)
        for slot, client, service in (
            ("embedding", services.llm_embedding, services.embedding),
            ("summary", services.llm_summary, services.summary),
            ("judge", services.llm_judge, services.judge),
        )
        if client is not None
    ]


async def _provider_startup_check(services: Services) -> None:
    """Стартовая проверка трёх LLM-слотов (решение №5) — в lifespan.

    После init_db, ДО старта воркера: лёгкий GET к каждому слоту (без
    генерации — не грузит модели в память, не тратит токены), три слота
    параллельно, по 1 попытке (~5 с на слот). Исходы:
    - ok → INFO + `last_attempt_ok=True` — /health честен ещё до первого
      реального вызова (без ложного None);
    - auth_failed (401/403) → фатальный отказ старта: SystemExit(2) с
      hint про {SLOT}_API_KEY — нерабочая конфигурация не должна
      подниматься и молча копить pending;
    - unreachable/model_missing (сеть/таймаут/5xx/404/модель не в списке)
      → WARN, старт продолжается (NFR-3: транзиентная недоступность не
      роняет рестарты — ребут хоста, Ollama ещё поднимается; воркер
      догонит по pending-очередям).
    Лог-события: provider_check_started / provider_check_result
    (слот, провайдер, исход — аудит в стиле Фазы 10).
    """
    logger = logging.getLogger("app")
    slots = _startup_slots(services)
    if not slots:
        # DI-сборка без клиентов слотов (фейки вместо сервисов): проверка —
        # заглушка ok (бриф §3), сервис стартует без сетевых обращений.
        return
    for slot, client, _service in slots:
        logger.info(
            "стартовая проверка LLM-слота запущена",
            extra={
                "event": "provider_check_started",
                "slot": slot,
                "provider": client.spec.provider,
            },
        )
    outcomes = await asyncio.gather(
        *[asyncio.to_thread(client.check) for _slot, client, _service in slots]
    )
    for (slot, client, service), outcome in zip(slots, outcomes):
        provider = client.spec.provider
        extra = {
            "event": "provider_check_result",
            "slot": slot,
            "provider": provider,
            "outcome": outcome,
            "model": client.spec.model,
            "api_key": bool(client.spec.api_key.strip()),
        }
        if outcome == CHECK_OK:
            service.last_attempt_ok = True
            logger.info("стартовая проверка LLM-слота пройдена", extra=extra)
        elif outcome == CHECK_AUTH_FAILED:
            logger.error(
                "стартовая проверка LLM-слота: отказ авторизации", extra=extra
            )
            message = (
                f"стартовая проверка слота {slot}: провайдер {provider} "
                f"отклонил авторизацию (401/403) — задай {slot.upper()}_API_KEY "
                "в docker-compose и перезапусти"
            )
            print(f"FATAL: {message}", file=sys.stderr)
            raise SystemExit(2)
        else:
            # unreachable/model_missing → WARN + деградация (NFR-3). Приёмка
            # пула 4 (решение координатора, уточнение брифа §1.5):
            # last_attempt_ok остаётся None — семантика NFR-4 «исход последней
            # РЕАЛЬНОЙ попытки»: стартовая проверка — GET живости, не векторизация
            # (тесты Фазы 8: save не кодирует синхронно → None). Деградация
            # видна в WARN-логах старта; False поставит первый реальный отказ.
            logger.warning(
                "стартовая проверка LLM-слота не пройдена — старт продолжается "
                "(деградация, воркер догонит по pending-очередям)",
                extra=extra,
            )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Собрать приложение; настройки — из окружения, если не переданы явно."""
    if settings is None:
        try:
            settings = get_settings()
        except ConfigError as exc:
            # Фатальная ошибка конфигурации: понятное сообщение вместо traceback.
            print(f"FATAL: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc

    try:
        services = build_services(settings)  # один Settings-снимок на процесс
    except ConfigError as exc:
        # Фатальная конфигурация (например, judge_system из файла без
        # маркеров «ДУБЛЬ»/«НЕ ДУБЛЬ» — решение №7): понятное сообщение
        # вместо traceback, старт прерывается.
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
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
        # Стартовая проверка трёх LLM-слотов (решение №5): после init_db,
        # ДО старта воркера — лёгкий GET без генерации, параллельно;
        # отказ авторизации фатален, недоступность — WARN + деградация.
        await _provider_startup_check(services)
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