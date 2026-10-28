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
import sqlite3
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI

from app import __version__
from app.config import ConfigError, Settings, get_settings
from app.observability import setup_logging
from app.services import Services, WarningsService, build_services
from app.services.decision import DecisionService
from app.services.llm_client import (
    CHECK_AUTH_FAILED,
    CHECK_OK,
    DECISION_CHECK_OK,
    LLMClient,
    SlotSpec,
)
from app.services.operations import OPERATIONS, unknown_operation_warnings
from app.services.backup import BackupService
from app.services.slot_budget import resolve_output_budget
from app.services.update_check import UpdateChecker
from app.services.worker import BackgroundWorker
from app.storage.db import StorageError, init_db
from app.storage.migration import (
    MigrationError,
    ensure_startup_schema,
    run_startup_migration,
    startup_plan,
)
# LSBN-F-22 (M3): union-контекст чтения (holder — services.migration).
from app.services.migration import refresh_migration_context
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
    ] + _decision_slot(services)


def _decision_slot(services: Services) -> list[tuple[str, LLMClient, DecisionService]]:
    """Пара decision-слота стартовой проверки — только когда она собрана.

    LSBN-F-12: при выключенном слоте пары нет (Services.decision is None) —
    стартовая проверка его не видит вообще.
    """
    if services.llm_decision is not None and services.decision is not None:
        return [("decision", services.llm_decision, services.decision)]
    return []


def _handle_decision_check(
    service: DecisionService,
    client: LLMClient,
    outcome: str,
    logger: logging.Logger,
) -> None:
    """Исход проверки decision-слота (LSBN-F-12): WARN + гашение, старт жив.

    Осознанное отступление от фатальности auth генеративных слотов: у decision
    есть штатный генеративный путь, и ронять обязательный слот из-за
    необязательного нельзя (§13.6). Событие — `decision_check` (slot/model/result);
    `ok` слот не трогает, иной исход — `disable(outcome)`: `unsupported` даёт
    причину переключения «модель не умеет», `unreachable`/`auth_failed` —
    «слота сейчас нет» (§13.6, f12-04).
    """
    extra = {
        "event": "decision_check",
        "slot": "decision",
        "model": client.spec.model,
        "result": outcome,
    }
    if outcome == DECISION_CHECK_OK:
        logger.info("стартовая проверка decision-слота пройдена", extra=extra)
        return
    service.disable(outcome=outcome)
    logger.warning(
        f"decision-слот не подтверждён (result={outcome}) — слот погашен, "
        "решения идут генеративным путём; старт продолжается",
        extra=extra,
    )


def _log_operation_config_warning(
    logger: logging.Logger,
    *,
    operation: str,
    slot: str,
    code: str,
    text: str,
    spec: SlotSpec,
) -> None:
    """Одно предупреждение конфигурации ОПЕРАЦИИ (LSBN-F-34, arch §13.5).

    Тот же контракт события `slot_config_warning`, что у слотов (main.py
    пишет, сервис стартует): `slot` — слот-группа операции, `operation` —
    имя операции из реестра. Эффективный бюджет — тем же
    `resolve_output_budget`, что и в вызове.
    """
    budget, _source = resolve_output_budget(
        operation=spec.name,
        prompt_tokens=0,
        explicit=spec.num_predict or None,
        context_tokens=spec.context_tokens,
        reserve_tokens=spec.prompt_reserve_tokens,
        defensive_cap=spec.max_output_tokens,
    )
    logger.warning(
        text,
        extra={
            "event": "slot_config_warning",
            "slot": slot,
            "operation": operation,
            "code": code,
            "model_type": spec.model_type,
            "budget": budget,
            "context_tokens": spec.context_tokens,
            "timeout_sec": int(spec.read_timeout),
            "text": text,
        },
    )


def _operations_config_warnings(logger: logging.Logger, settings: Settings) -> None:
    """Предупреждения конфигурации операций (LSBN-F-34; решение №4 брифа).

    Те же правила `slot_config_warnings` (таймаут/бюджет/контекст), что у
    слотов, но по эффективному spec ПЕРЕОПРЕДЕЛЁННОЙ операции — операция без
    плоского env уже покрыта предупреждениями своего слота, дублей не
    создаём. Плюс WARN об env с неизвестным именем операции (опечатка вида
    `MERG_MODEL`) — не фатально, сервис стартует. Вызывается при старте
    (lifespan), после проверки слотов.
    """
    for operation in OPERATIONS:
        spec, sources = SlotSpec.for_operation(operation, settings)
        if "op" not in sources.values():
            continue
        for code, text in LLMClient.slot_config_warnings(spec):
            _log_operation_config_warning(
                logger,
                operation=operation,
                slot=spec.name,
                code=code,
                text=text,
                spec=spec,
            )
    # Неизвестное имя операции — отдельное событие БЕЗ слота: контракт
    # `slot_config_warning` (всегда со слотом) не ломается (см.
    # test_main_startup_warnings).
    for _code, text in unknown_operation_warnings():
        logger.warning(
            text,
            extra={
                "event": "unknown_operation_warning",
                "code": "unknown_operation",
                "text": text,
            },
        )


def _run_startup_migration(
    settings: Settings, plan: dict, backup: BackupService
) -> None:
    """LSBN-F-22 (M1), фазы 1–3: журнал → снапшот → перелив и swap.

    Вызывается из lifespan, когда `startup_plan` увидел расхождение env с
    живым CHECK (фаза 0). Порядок фазы 1 (постановка: «старт, видимость,
    снапшот»): ПЕРВАЯ строка журнала джоб с причиной и объёмом, затем
    снапшот прежней базы — без него перелив не начинается (решение M1:
    отказ снапшота — WARN, сервис остаётся на прежней базе и переживёт
    рестарт с тем же расхождением). Сбой перелива/swap тоже не роняет
    старт: активная база цела (`run_startup_migration` гарантирует
    восстановление указателя), расхождение остаётся видимым в журнале.
    """
    logger = logging.getLogger("app")
    logger.info(
        f"миграция: env={plan['target']}, схема={plan['schema_limit']}, "
        f"цель {plan['target']}, к обработке {plan['pending']}",
        extra={
            "event": "migration_started",
            "job": "migration",
            "mode": plan["mode"],
            "env_limit": plan["target"],
            "schema_limit": plan["schema_limit"],
            "target": plan["target"],
            "pending": plan["pending"],
            "reason": plan["reason"],
        },
    )
    try:
        backup.snapshot()
    except (sqlite3.Error, OSError) as exc:
        logger.error(
            "снапшот прежней базы не удался — миграция отложена",
            extra={
                "event": "migration_snapshot_failed",
                "job": "migration",
                "error": str(exc),
            },
        )
        return
    try:
        counters = run_startup_migration(settings, plan)
    except (MigrationError, StorageError, OSError) as exc:
        logger.error(
            "перелив/swap не состоялись — сервис остаётся на прежней базе",
            extra={
                "event": "migration_failed",
                "job": "migration",
                "mode": plan["mode"],
                "error": str(exc),
            },
        )
        return
    logger.info(
        "миграция: перелив и swap выполнены — сервис на новой базе",
        extra={
            "event": "migration_swapped",
            "job": "migration",
            "mode": plan["mode"],
            **counters,
        },
    )


async def _provider_startup_check(
    services: Services, settings: Settings | None = None
) -> None:
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
    if settings is None:
        settings = get_settings()
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
        *[
            asyncio.to_thread(
                client.check_decision if slot == "decision" else client.check
            )
            for slot, client, _service in slots
        ]
    )
    for (slot, client, service), outcome in zip(slots, outcomes):
        if slot == "decision":
            _handle_decision_check(service, client, outcome, logger)
            continue
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
        # №49 (arch §13.5): предупреждения конфигурации слота — WARN и старт
        # продолжается (фатален только auth_failed, см. выше); пишутся и при
        # unreachable/model_missing — они про конфигурацию, не про сеть. У
        # слота векторизации нет бюджета вывода и *_CONTEXT_TOKENS, а у
        # decision нет ни бюджета, ни *_CONTEXT_TOKENS (LSBN-F-12).
        if slot == "embedding":
            continue
        warnings = LLMClient.slot_config_warnings(client.spec)
        if warnings:
            budget, _budget_source = resolve_output_budget(
                operation=slot,
                prompt_tokens=0,
                explicit=client.spec.num_predict or None,
                context_tokens=client.spec.context_tokens,
                reserve_tokens=client.spec.prompt_reserve_tokens,
                defensive_cap=client.spec.max_output_tokens,
            )
            for code, text in warnings:
                logger.warning(
                    text,
                    extra={
                        "event": "slot_config_warning",
                        "slot": slot,
                        "code": code,
                        "model_type": client.spec.model_type,
                        "budget": budget,
                        "context_tokens": client.spec.context_tokens,
                        "timeout_sec": int(client.spec.read_timeout),
                        "text": text,
                    },
                )
    # LSBN-F-34 (решение №4 брифа): предупреждения конфигурации операций —
    # переопределённые плоским env (те же правила, что у слотов) + WARN о
    # неизвестном имени операции. Сервис стартует (не фатально).
    _operations_config_warnings(logger, settings)


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
    # §3.4: две очереди — векторы (Фаза 3) и summary (Фаза 4, режим «Б»);
    # суммаризатор из DI — тот же экземпляр, что в /health.summarizer_ok.
    # Судья дедупа (Фаза 8, Этап 3.1) — тоже DI из services, один экземпляр
    # на процесс (вердикты по кандидатам — Задача 3.2).
    # №12 (3.2.0): один сервис предупреждений на процесс — им воркер пишет
    # «слияние отложено» (F-23 п. 6), а NoteService инвалидирует
    # предупреждения при правке (ниже). Снятию классификатор не нужен
    # (только SQL) — узкая инъекция без циклических импортов.
    warnings_service = WarningsService(
        settings, classifier=None, namespaces=services.namespaces
    )
    worker = BackgroundWorker(
        settings,
        services.embedding,
        services.summary,
        judge=services.judge,
        classifier=services.classifier,
        promoter=services.promotion,
        terms=services.terms,
        decision=services.decision,
        warnings=warnings_service,
    )
    # Суммаризация стартует сразу при save/update: NoteService сигналит
    # воркеру, тот немедленно догоняет pending_summary (не ждёт back-off).
    services.notes.set_summary_notifier(worker.notify_summary_pending)
    # Векторизация стартует сразу при save/update: та же запись заметки
    # (vector_status='pending') сигналит embedding-петле — свежая заметка
    # кодируется немедленно, а не после выросшего back-off (до 15 мин).
    # Событие только ускоряет — задание живёт в статусе заметки (gate
    # 2026-09-19, прецедент summary-нотификатора).
    services.notes.set_vector_notifier(worker.notify_embedding_pending)
    # №12 (FR-4.4): правка заметки инвалидирует её предупреждение — в той же
    # транзакции.
    services.notes.set_warnings_resolver(warnings_service.resolve_for_subject)
    # Область навыков (3.0.0): save/правка сигналит петле areas — вектора
    # записи догоняются сразу, а не по выросшему back-off (прецедент
    # summary-нотификатора). None — DI-сборка тестов без области навыков.
    if services.skills is not None:
        services.skills.set_areas_notifier(worker.notify_areas_pending)
    # Область «user» (3.0.0, lsb-0009): запись/правка/удаление факта будит
    # петлю areas тем же сигналом — вектора записи догоняются сразу, а не по
    # выросшему back-off (прецедент области навыков). Проверка на None — как
    # у области навыков: поле может быть пустым в DI-сборках тестов (№40).
    if services.user_facts is not None:
        services.user_facts.set_areas_notifier(worker.notify_areas_pending)
    # Область terms (3.0.0, lsb-0008): запись/правка/удаление термина будит
    # петлю areas тем же сигналом — вектора записи догоняются сразу, а не по
    # выросшему back-off (прецедент областей навыков и «user»). Проверка на
    # None — симметрично прочим областям (№40).
    if services.terms is not None:
        services.terms.set_areas_notifier(worker.notify_areas_pending)
    # Область логов (3.2.0, LSBN-F-10, f10-02): запись/правка/удаление лога будит ту
    # же петлю areas — вектор записи (непустые action + reason + outcome, §5.3)
    # догоняется сразу, а не по выросшему back-off. Проверка на None — как у
    # прочих областей (№40).
    if services.logs is not None:
        services.logs.set_areas_notifier(worker.notify_areas_pending)
    # Внутренний маршрут MCP-сервера — ровно MCP_PATH. host="0.0.0.0" — не
    # localhost, поэтому SDK не включает DNS-rebinding protection (сервис
    # живёт в LAN за Bearer-токеном; Open WebUI ходит с не-localhost Host).
    # MCP собирается после воркера (№39, FR-4.4): миддлварь хинта о сломанном
    # слоте берёт снимок разрывов из воркера (`slot_states`). №45 (FR-45.8):
    # чекер версии создаётся здесь и передаётся в MCP — миддлварь `update_hint`
    # читает только его кеш; фоновая петля стартует в lifespan (если включено).
    update_checker = UpdateChecker(settings)
    mcp = build_mcp(settings, services, worker=worker, update_checker=update_checker)
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
            # LSBN-F-22: таблица migration_notes + штамп schema.note_chars
            # на каждой старте (идемпотентно; наблюдаемость — design §3).
            ensure_startup_schema(settings)
        except StorageError as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        # LSBN-F-22 (M1), фазы 0–3: env разошёлся с живым CHECK — сервис сам
        # строит новую базу, переливает влезающее 1:1 и переключается файловым
        # swap (автостарт без предохранителя — вариант C постановки; без
        # простоя: старт продолжается на уже новой базе). Сбой не роняет старт:
        # активная база цела, причина — в журнале джоб (первая строка миграции).
        plan = startup_plan(settings)
        if plan["mode"] != "none":
            _run_startup_migration(settings, plan, services.backup)
        # LSBN-F-22 (M3): union-контекст чтения (search/notes/dedup) — по
        # состоянию миграции активной базы: phase pour/split и живой
        # prev_file → вторая сессия по прежней базе (в т.ч. возобновление —
        # рестарт середины миграции); done/aborted/нет — None (нулевой
        # оверхед, пути прежние).
        refresh_migration_context(settings)
        # Стартовая проверка трёх LLM-слотов (решение №5): после init_db,
        # ДО старта воркера — лёгкий GET без генерации, параллельно;
        # отказ авторизации фатален, недоступность — WARN + деградация.
        # LSBN-F-34: settings переданы явно — предупреждения операций считаются
        # по тому же снимку настроек, что собрал процесс.
        await _provider_startup_check(services, settings)
        logging.getLogger("app").info(
            "service started",
            extra={
                "event": "startup",
                "version": __version__,
                "port": settings.port,
                "mcp_path": settings.mcp_path,
                "db_path": settings.db_path,
                # Разделитель слота нарезки (3.4.1, LSBN-T-27): по проде
                # видно, сколько заметок берётся за прогон и на сколько
                # джоба миграции уходит в паузу.
                "migration_batch": settings.migration_batch,
                "migration_pause_sec": settings.migration_pause_sec,
            },
        )
        # №40 (FR-4.3): состав включённых областей — из журнала старта видно,
        # какие области активны (диагностика не гадает). Флаг гейтит только
        # поверхность/фон; данные и REST остаются доступны и у выключенной.
        logging.getLogger("app").info(
            "areas enabled",
            extra={
                "event": "areas_enabled",
                "areas": {
                    "skills": settings.area_skills_enabled,
                    "terms": settings.area_terms_enabled,
                    "user_facts": settings.area_user_facts_enabled,
                    # LSBN-F-10 (f10-02): область логов активна тем же реестром;
                    # ключ добавлен минимально, реестровый рефакторинг события —
                    # отдельная позиция (f10-04).
                    "logs": settings.area_logs_enabled,
                },
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
            # №45 (FR-45.3): фоновая проверка версии — только когда включена;
            # первый опрос сразу после старта, далее по интервалу.
            update_task: asyncio.Task[None] | None = None
            if settings.update_check_enabled:
                update_task = asyncio.create_task(
                    update_checker.run(), name="update-check"
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
                if update_task is not None:
                    update_checker.stop()
                    update_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await update_task

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
    # Воркер нужен REST-ручке /health: снимки очередей (queues, FR-2.2)
    # собираются по реестру джоб — только SQL, без обращений к моделям.
    app.state.worker = worker
    # №45 (FR-4.4): снимок версии для /health.update — из памяти чекера.
    app.state.update_checker = update_checker
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