"""The FastAPI application of LLM Second Brain.

Phase 1 framework: REST, the MCP server (`/mcp`, 6 `memory_*` tools), a
Bearer middleware over everything except `/health` (NFR-2). Phase 2: the
storage is initialized at startup (SQLite, ARCH §3.3); MCP and REST work
over the shared service layer (ARCH §1) via `app.state.services`.
Phases 3–4: the lifespan brings up the background worker (ARCH §3.4) — two
independent queues, pending_vector (re-vectorization) and pending_summary
(re-summarization, mode "B" — the only path that generates a summary);
the queues live in the DB and survive restarts; shutdown cancels them
gracefully. Phase 11: before the worker starts, the three LLM slots get a
startup check: a light GET without generation, in parallel; an auth
failure (401/403) is fatal (SystemExit 2 + the {SLOT}_API_KEY hint),
unreachability — WARN and a normal start (NFR-3).

Run: `python -m app` (uvicorn) — the port and the log level come from the
environment.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

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
    backfill_origin_note_ids,
    ensure_startup_schema,
    run_startup_migration,
    startup_plan,
)
# LSBN-F-22 (M3): the union read context (the holder lives in services.migration).
from app.services.migration import refresh_migration_context
from app.shutdown import bounded_shutdown
from app.transport.auth import BearerAuthMiddleware
from app.transport.mcp import build_mcp
from app.transport.rest import build_rest_router

# Paths reachable without a Bearer token (NFR-2: /health only).
OPEN_PATHS = frozenset({"/health"})


def _startup_slots(services: Services) -> list[tuple[str, LLMClient, object]]:
    """The three LLM slots of the startup check: (name, client, service).

    The service carries `last_attempt_ok` (/health): the check result
    initializes it before the first real call. Test DI builds without slot
    clients (llm_* = None, services replaced by fakes) are skipped: for fakes
    the startup check is a stubbed ok (brief §3).
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
    """The decision-slot pair of the startup check — only when it is built.

    LSBN-F-12: with the slot off there is no pair (Services.decision is None) —
    the startup check does not see it at all.
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
    """The decision-slot check outcome (LSBN-F-12): WARN + shutoff, the start lives.

    A deliberate deviation from the auth fatality of generative slots: decision
    has its own generative fallback, and an optional slot must not kill the
    mandatory ones (§13.6). The event is `decision_check` (slot/model/result);
    `ok` leaves the slot alone, any other outcome — `disable(outcome)`:
    `unsupported` gives the "model cannot" fallback reason,
    `unreachable`/`auth_failed` — "no slot right now" (§13.6, f12-04).
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
    """One configuration warning of an OPERATION (LSBN-F-34, arch §13.5).

    The same `slot_config_warning` event contract as the slots have (main.py
    writes, the service still starts): `slot` — the operation's slot group,
    `operation` — the registry name of the operation. The effective budget is
    resolved by the same `resolve_output_budget` as in a call.
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
    """Operation configuration warnings (LSBN-F-34; brief decision No4).

    The same `slot_config_warnings` rules (timeout/budget/context) as the
    slots, but over the effective spec of an OVERRIDDEN operation — an
    operation without flat env is already covered by its slot's warnings, no
    duplicates. Plus the WARN about env with an unknown operation name (a typo
    like `MERG_MODEL`) — not fatal, the service starts. Called at startup
    (lifespan), after the slot checks.
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
    # An unknown operation name — a separate event WITHOUT a slot: the
    # `slot_config_warning` contract (always with a slot) stays intact (see
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
    """LSBN-F-22 (M1), phases 1–3: journal → snapshot → pour and swap.

    Called from the lifespan when `startup_plan` sees the env diverge from the
    live CHECK (phase 0). Phase 1 order (the design: "start, visibility,
    snapshot"): the FIRST job-journal line with the reason and the volume,
    then the snapshot of the previous base — without it the pour never starts
    (M1: a snapshot failure is a WARN, the service stays on the previous base
    and survives a restart with the same divergence). A pour/swap failure does
    not kill the start either: the active base is intact
    (`run_startup_migration` guarantees the pointer is restored) and the
    divergence stays visible in the journal.
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
    """The startup check of the three LLM slots — inside the lifespan.

    After init_db, BEFORE the worker starts: a light GET per slot (no
    generation — models stay out of memory, no tokens spent), three slots in
    parallel, one attempt each (~5 s per slot). Outcomes:
    - ok → INFO + `last_attempt_ok=True` — /health is honest before the first
      real call (no misleading None);
    - auth_failed (401/403) → a fatal start refusal: SystemExit(2) with the
      {SLOT}_API_KEY hint — a broken configuration must not come up and
      silently accumulate pending;
    - unreachable/model_missing (network/timeout/5xx/404/model not listed)
      → WARN, the start continues (NFR-3: a transient outage must not kill
      restarts — a host reboot with Ollama still coming up; the worker
      catches up over the pending queues).
    Log events: provider_check_started / provider_check_result
    (slot, provider, outcome — Phase-10-style audit).
    """
    logger = logging.getLogger("app")
    if settings is None:
        settings = get_settings()
    slots = _startup_slots(services)
    if not slots:
        # A DI build without slot clients (fakes instead of services): the
        # check is a stubbed ok (brief §3), the service starts with no network
        # calls.
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
            # unreachable/model_missing → WARN + degradation (NFR-3). The pool-4
            # acceptance (brief §1.5): last_attempt_ok stays None — the NFR-4
            # semantics "the outcome of the last REAL attempt": the startup
            # check is a liveness GET, not vectorization (Phase 8 tests: save
            # does not encode synchronously → None). The degradation is visible
            # in the startup WARN logs; the first real failure sets False.
            logger.warning(
                "стартовая проверка LLM-слота не пройдена — старт продолжается "
                "(деградация, воркер догонит по pending-очередям)",
                extra=extra,
            )
        # No49 (arch §13.5): slot configuration warnings — WARN and the start
        # continues (only auth_failed is fatal, see above); they are written
        # for unreachable/model_missing too — they are about configuration, not
        # the network. The embedding slot has no output budget and no
        # *_CONTEXT_TOKENS; decision has neither (LSBN-F-12).
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
    # LSBN-F-34 (brief decision No4): operation configuration warnings —
    # operations overridden by flat env (the same rules as slots) + the WARN
    # about an unknown operation name. The service starts (not fatal).
    _operations_config_warnings(logger, settings)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application; settings come from the environment unless passed."""
    if settings is None:
        try:
            settings = get_settings()
        except ConfigError as exc:
            # Fatal configuration error: a readable message instead of a traceback.
            print(f"FATAL: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc

    try:
        services = build_services(settings)  # one Settings snapshot per process
    except ConfigError as exc:
        # A fatal configuration (for example, a judge_system file without the
        # "DUPLICATE"/"NOT DUPLICATE" markers): a readable message instead of a
        # traceback, the start aborts.
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    # JSON logs to stdout (NFR-4): enabled at application build (importing
    # app.main / tests / uvicorn — the same idempotent setup).
    setup_logging(
        settings.log_level, settings.log_file or None, settings.log_max_bytes, settings.log_backups
    )
    # §3.4: two queues — vectors (Phase 3) and summary (Phase 4, mode "B");
    # the DI summarizer is the same instance /health.summarizer_ok reports.
    # The dedup judge (Phase 8, step 3.1) is DI from services too, one
    # instance per process (candidate verdicts — Task 3.2).
    # No12 (3.2.0): one warnings service per process — the worker writes
    # "merge deferred" with it (F-23 p. 6), and NoteService invalidates
    # warnings on edit (below). The removal needs no classifier (SQL only) —
    # a narrow injection without circular imports.
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
    # Summarization starts immediately on save/update: NoteService signals
    # the worker, which catches up pending_summary right away (no back-off
    # wait).
    services.notes.set_summary_notifier(worker.notify_summary_pending)
    # Vectorization also starts immediately on save/update: the same note
    # write (vector_status='pending') signals the embedding loop — a fresh note
    # is encoded right away, not after the grown back-off (up to 15 min).
    # The event only accelerates — the work lives in the note status (the
    # 2026-09-19 gate, the summary-notifier precedent).
    services.notes.set_vector_notifier(worker.notify_embedding_pending)
    # No12 (FR-4.4): editing a note invalidates its warning — in the same
    # transaction.
    services.notes.set_warnings_resolver(warnings_service.resolve_for_subject)
    # The skills area (3.0.0): save/edit signals the areas loop — the record's
    # vectors are caught up immediately, not after the grown back-off (the
    # summary-notifier precedent). None — a test DI build without the skills
    # area.
    if services.skills is not None:
        services.skills.set_areas_notifier(worker.notify_areas_pending)
    # The "user" area (3.0.0, lsb-0009): fact save/edit/delete wakes the areas
    # loop with the same signal — the record's vectors are caught up
    # immediately, not after the grown back-off (the skills-area precedent).
    # The None check mirrors the skills area: the field may be empty in test
    # DI builds (No40).
    if services.user_facts is not None:
        services.user_facts.set_areas_notifier(worker.notify_areas_pending)
    # The terms area (3.0.0, lsb-0008): term save/edit/delete wakes the areas
    # loop with the same signal — the record's vectors are caught up
    # immediately, not after the grown back-off (the skills and "user" area
    # precedents). The None check mirrors the other areas (No40).
    if services.terms is not None:
        services.terms.set_areas_notifier(worker.notify_areas_pending)
    # The logs area (3.2.0, LSBN-F-10, f10-02): log save/edit/delete wakes the
    # same areas loop — the record vector (non-empty action + reason + outcome,
    # §5.3) is caught up immediately, not after the grown back-off. The None
    # check mirrors the other areas (No40).
    if services.logs is not None:
        services.logs.set_areas_notifier(worker.notify_areas_pending)
    # The internal MCP route is exactly MCP_PATH. host="0.0.0.0" — not
    # localhost, so the SDK does not enable DNS-rebinding protection (the
    # service lives in a LAN behind a Bearer token; Open WebUI sends a
    # non-localhost Host). MCP is built after the worker (No39, FR-4.4): the
    # broken-slot hint middleware takes the break snapshot from the worker
    # (`slot_states`). No45 (FR-45.8): the version checker is created here and
    # handed to MCP — the `update_hint` middleware reads only its cache; the
    # background loop starts in the lifespan (when enabled).
    update_checker = UpdateChecker(settings)
    mcp = build_mcp(settings, services, worker=worker, update_checker=update_checker)
    mcp_app = mcp.streamable_http_app(
        streamable_http_path=settings.mcp_path,
        host="0.0.0.0",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Storage initialization at startup (the Phase 2 acceptance test):
        # the schema is created/verified before the first request; an error is
        # fatal.
        try:
            init_db(settings)
            # LSBN-F-22: the migration_notes table + the schema.note_chars
            # stamp on every start (idempotent; observability — design §3).
            ensure_startup_schema(settings)
            # 3.6.2: the parts of the notes split before carry the id of their
            # source note (origin_note_id from migration_notes, idempotent).
            # BEFORE the startup plan — a deferred migration then pours the
            # filled values into the new base together with the notes.
            backfill_origin_note_ids(settings)
        except StorageError as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        # LSBN-F-22 (M1), phases 0–3: the env diverged from the live CHECK —
        # the service builds a new base itself, pours what fits 1:1 and
        # switches by a file swap (an automatic start without a safety catch —
        # variant C of the design; zero downtime: the start continues on the
        # already-new base). A failure does not kill the start: the active base
        # is intact, the reason is in the job journal (the first migration
        # line).
        plan = startup_plan(settings)
        if plan["mode"] != "none":
            _run_startup_migration(settings, plan, services.backup)
        # LSBN-F-22 (M3): the union read context (search/notes/dedup) — by the
        # active base migration state: phase pour/split with a live prev_file →
        # a second session over the previous base (including a resume — a
        # restart in the middle of a migration); done/aborted/none — None (zero
        # overhead, the old paths).
        refresh_migration_context(settings)
        # The startup check of the three LLM slots: after init_db, BEFORE the
        # worker starts — a light GET without generation, in parallel; an auth
        # failure is fatal, unreachability — WARN + degradation.
        # LSBN-F-34: settings are passed explicitly — operation warnings are
        # computed over the same settings snapshot the process was built with.
        await _provider_startup_check(services, settings)
        logging.getLogger("app").info(
            "service started",
            extra={
                "event": "startup",
                "version": __version__,
                "port": settings.port,
                "mcp_path": settings.mcp_path,
                "db_path": settings.db_path,
                # The migration slot cadence (3.4.1, LSBN-T-27): the startup
                # record shows how many notes a run takes and how long the
                # migration job pauses.
                "migration_batch": settings.migration_batch,
                "migration_pause_sec": settings.migration_pause_sec,
            },
        )
        # No40 (FR-4.3): the set of enabled areas in the startup log —
        # diagnostics does not guess. The flag gates only the surface/loop;
        # data and REST stay reachable for a disabled area too.
        logging.getLogger("app").info(
            "areas enabled",
            extra={
                "event": "areas_enabled",
                "areas": {
                    "skills": settings.area_skills_enabled,
                    "terms": settings.area_terms_enabled,
                    "user_facts": settings.area_user_facts_enabled,
                    # LSBN-F-10 (f10-02): the logs area is live in the same
                    # registry; the key was added minimally, the event's registry
                    # refactor is a separate item (f10-04).
                    "logs": settings.area_logs_enabled,
                },
            },
        )
        # The MCP Streamable HTTP session lifecycle = the process lifetime.
        async with mcp.session_manager.run():
            # Background worker: the pending queues live in the DB — after a
            # restart they resume/re-encode on their own (ARCH §3.4); on
            # shutdown the task is cancelled.
            worker_task = asyncio.create_task(worker.run(), name="pending-backlog")
            # BackupService (Phase 5, NFR-3): a snapshot right after the start,
            # then once per BACKUP_INTERVAL_SEC; failures go to the log, the
            # loop keeps living.
            backup_task = asyncio.create_task(
                services.backup.run(), name="backup-snapshots"
            )
            # No45 (FR-45.3): the background version check — only when enabled;
            # the first poll right after the start, then by the interval.
            update_task: asyncio.Task[None] | None = None
            if settings.update_check_enabled:
                update_task = asyncio.create_task(
                    update_checker.run(), name="update-check"
                )
            # LSBN-D-16: the background handles for diagnostics and tests
            # (after a stop nothing must keep running — the shutdown test
            # asserts every task here is done).
            app.state.background_tasks = (worker_task, backup_task, update_task)
            try:
                yield
            finally:
                # LSBN-D-16: the loops are cancelled BEFORE uvicorn logs
                # "Application shutdown complete", and the wait is bounded by
                # SHUTDOWN_TIMEOUT_SEC: a worker thread blocked on an LLM
                # call must not hold the container half-dead (see
                # app/shutdown.py) — a soft flag, a cancel, a bounded wait.
                worker.stop()  # soft flag: do not start a new batch
                worker_task.cancel()  # and interrupt the wait if sleeping
                services.backup.stop()
                backup_task.cancel()
                if update_task is not None:
                    update_checker.stop()
                    update_task.cancel()
                await bounded_shutdown(
                    [
                        task
                        for task in (worker_task, backup_task, update_task)
                        if task is not None
                    ],
                    timeout_sec=settings.shutdown_timeout_sec,
                )

    app = FastAPI(
        title="LLM Second Brain",
        version=__version__,
        # REST is an internal surface (REQUIREMENTS §3): no OpenAPI docs —
        # fewer extra handles behind the middleware.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.include_router(build_rest_router(settings))
    app.state.services = services  # REST handles fetch the services here
    # The worker is needed by the /health REST handle: queue snapshots
    # (queues, FR-2.2) are built over the job registry — SQL only, no model
    # calls.
    app.state.worker = worker
    # No45 (FR-4.4): the version snapshot for /health.update — from the
    # checker's memory.
    app.state.update_checker = update_checker
    # Mounted at the root: MCP_PATH pins the exact MCP endpoint path;
    # FastAPI routes (/health and future REST) match before the mount.
    app.mount("/", mcp_app)
    app.add_middleware(
        BearerAuthMiddleware,
        token=settings.mcp_auth_token,
        exempt_paths=OPEN_PATHS,
    )
    return app


app = create_app()