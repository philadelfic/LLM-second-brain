"""REST handles (ARCHITECTURE §3.1): the internal operator/diagnostics surface.

    Thin wrappers over the same service layer as the MCP tools (ARCH §1): one
    service codebase, identical behavior. Domain validation — in the service
    (`NoteValidationError`/`SearchValidationError` → 422); pagination constraints
    — fixed FR contracts (limit 1..50, offset ≥ 0).

    Phase 2: /notes CRUD + /search + /health counters from the DB (NFR-4).
    Phase 10 (Step 6): the operator's structure handles — GET/POST/PATCH/DELETE
    /namespaces + merge (all with note moving; nothing is lost, §5.7);
    the structure handles are NOT in MCP (client models do not govern structure).
    Errors: 422 — path/description validation; 404 — node not found; 409 —
    protection/conflict (default, a merge into itself, a root with children, a
    taken path).

    Phase 11 (decision #9, follow-up 5b): `title` in the POST/PUT /notes body
    (passed — validated by the service, invalid → 422 "set a title ≤5 words";
    in PUT not passed — the old one stays) and in the get/search/list outputs (to
    the operator; null — a titleless migration note). The "new notes always have
    a title" contract is one for both surfaces: POST /notes without a title or
    with an invalid one → 422 fail+hint, the note is NOT created. The
    sentinel-legacy NoteService.save(text) without a title — the migration/script
    path at the service layer, unreachable from the transports.

    Release 3.0.0 (lsb-0007-05): the skills area's REST mirrors — /skills
    (creation/listing/search/read/edit/soft delete), the global
    `/skills/instruction-template` and the version-copies archive
    (GET /skills/{id}/versions; REST only, not output to MCP). The same Bearer
    and the same service layer; full outputs — without the MCP whitelist trim.
    Codes: 201 — creation; 200 — read/edit/delete; 422 — shape validation and
    soft failures (the service's text = a hint); 404 — the skill not found
    (including a deleted one).

    Release 3.0.0 (lsb-0009-03): the "user" area's REST mirrors — /user-facts
    (creation with a dedup hint, search, read, edit, soft delete). The same
    Bearer and the same service layer as MCP (`user_save`/`user_search`/
    `user_get`/`user_update`/`user_delete`); full outputs — without the MCP trim.
    No listing (the mirrors mirror the area's handles). Codes: 201 — creation;
    200 — read/edit/delete; 422 — shape validation and the service's soft
    failures (the text = the verbatim lsb-0009 §3.7 canon hint, including the
    dedup's strong match); 404 — the fact not found (including a deleted one).

    Release 3.0.0 (lsb-0008-03): the "terms" area's REST mirrors — /terms
    (search, a write by key, read, an edit by id, soft delete). The same Bearer
    and the same service layer as MCP (`terms_save`/`terms_search`/`terms_get`);
    full outputs — without the MCP trim. No listing: the only road to a
    definition is the search. Codes: 201 — creation; 200 — an update by key, a
    read, an edit, a deletion; 422 — shape validation and the service's soft
    failures (the text = the verbatim lsb-0008 §3.7 canon hint, including a close
    context); 404 — the record not found (including a deleted one); 409 — the
    edit's new key taken by another active record (substrate §3.6).

    Release 3.1.0 (lsb-0013-02): the operator's listings — the same page contract
    as MCP (arch lsb-0013 §3.4): `GET /notes` and `GET /skills` pass the surface
    ceiling to the service (`list_max_limit_rest`, 50) and carry
    `total`/`has_more`/`next_offset`/`next_cursor` in the answer. There is no
    "+N more" text hint on REST — that is the "there is more" marker for the
    model in the MCP output; the service's own hint ("memory is empty") stays as
    it was.

    Release 3.1.0 (techdebt-0036): `/health` carries `version` — the application
    version (`app.__version__`), consistent with the release tag; the `queues`
    field is preserved.

    Release 3.1.0 (lsb-0010-05): `GET /notes/{id}` carries the note's links —
    `links` from OTHER namespaces (an item {id, title, namespace, chars}, level 1
    with a fallback to level 0, arch §3.5). No new handles: a separate links
    viewer is not introduced.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel

from app import __version__
from app.config import Settings
from app.services import Services
from app.services import history
from app.services.backup import BackupService
from app.services.freshness import FreshnessValidationError
# LSBN-F-22 (M3): the /health.migration block (phase/reason/progress/invariant)
# and the operator's migration abort (POST /migration/abort).
from app.services.migration import abort_migration, migration_health
from app.services.namespaces import NamespaceError, NamespaceValidationError
from app.services.notes import (
    _UNSET_COMMENT,
    _UNSET_TEXT,
    NoteValidationError,
)
# LSBN-F-34: the /health.operations snapshot (model/provider/source per operation).
from app.services.operations import operations_health
from app.services.search import SearchValidationError
from app.storage.db import StorageError
from app.storage.migration import MigrationError
from app.services.skills import (
    HINT_NOT_FOUND as HINT_SKILL_NOT_FOUND,
    SkillValidationError,
    SkillsService,
)
from app.services.terms import (
    TermKeyConflictError,
    TermValidationError,
    TermsService,
)
from app.services.logs import LogsService, LogValidationError
# "Not passed = keep" in PUT /logs/{id}: the same sentinel as MCP's
# (`_LOG_UNSET`) — REST introduces no empty-field semantics of its own.
from app.services.logs import _UNSET as _LOG_UNSET
from app.services.user_facts import UserFactValidationError, UserFactsService
from app.services.warnings import WarningsService, WarningsValidationError

# Entities with history (§5.3): ALL are available to the operator, including the global
# skills template — no area flags on this surface (§13.10: the operator's REST
# stays available even with an area disabled).
_HISTORY_ENTITIES = frozenset(
    {
        history.ENTITY_NOTE,
        history.ENTITY_SKILL,
        history.ENTITY_SKILL_TEMPLATE,
        history.ENTITY_TERM,
        history.ENTITY_USER_FACT,
    }
)


def _check_history_entity(entity: str) -> None:
    """An unknown history entity — 404 (the layer's exception does not go outward)."""
    if entity not in _HISTORY_ENTITIES:
        raise HTTPException(
            status_code=404,
            detail="unknown history entity: "
            + entity
            + " (known: "
            + ", ".join(sorted(_HISTORY_ENTITIES))
            + ")",
        )


class NoteCreate(BaseModel):
    """The POST /notes body. Author — if the operator knows the source model.

            Phase 11 (decision #9, follow-up 5b): `title` is mandatory — absent (not
            passed) or invalid (empty/longer than TITLE_MAX_WORDS words) → 422 "set a
            title ≤5 words"; the note is NOT created; the contract is one with MCP
            memory_save (the sentinel-legacy save(text) — the service layer only).
            no24-01 (3.2.0): the optional `comment` (≤250) — a note known already at
            write time.
    """

    text: str
    title: str | None = None
    author: str | None = None
    comment: str | None = None  # no24-01 (3.2.0): the note's comment (≤250)


class NamespaceCreate(BaseModel):
    """The POST /namespaces body: the operator creates a confirmed node (Step 6).

            The automation creates provisional ones itself (the Step-5 trigger); the
            handle is only for the starter set and roots (there are no auto-roots,
            §5.7).
    """

    path: str
    description: str


class NamespacePatch(BaseModel):
    """The PATCH /namespaces/{path} body: the description, the status and/or a rename.

            `path` in the body — the node's new path (a rename with note moving,
            §5.7); omitted — the node stays in place.
    """

    path: str | None = None
    description: str | None = None
    status: str | None = None


class NamespaceMerge(BaseModel):
    """The POST /namespaces/{path}/merge body: the target canonical node."""

    into: str


class NoteUpdate(BaseModel):
    """The PUT /notes/{id} body: a whole-note overwrite (FR-5).

            Phase 11 (decision #9): `title` is optional — passed and valid →
            overwrite; not passed → the old one stays (the merge path does not
            overwrite it).
            no24-01 (3.2.0): `comment` — the note's comment (≤250); the field's
            presence is determined by `model_fields_set` (omitted → the
            `_UNSET_COMMENT` sentinel = untouched; `null`/`""` → a reset). `text` is
            optional too: omitted or `null` → the text is not overwritten (a
            comment-only edit).
    """

    text: str | None = None
    title: str | None = None
    comment: str | None = None


class SkillCreate(BaseModel):
    """The POST /skills body: the skill shape (arch lsb-0007 §3.1).

            The mandatory shape fields — `name`/`description`/`steps`/`text`;
            `example` and `extra` (the skill's class fields) are optional. The limits
            and the creation anti-synonymy are checked by the service: a violation →
            422 with the verbatim §3.8 canon hint.
    """

    name: str
    description: str
    steps: str
    text: str
    example: str | None = None
    extra: dict[str, str] | None = None


class SkillUpdate(SkillCreate):
    """The PUT /skills/{id} body: the same full shape; the former version — into history.

            There is no separate partial update (§3.4): the shape is overwritten
            whole; the version delta goes into the shared history (`record_versions`,
            #41).
    """


class InstructionTemplate(BaseModel):
    """The PUT /skills/instruction-template body: the global template (§3.1).

            One template per area ("how to execute the steps", ≤2000 characters); an
            empty or overly long value → 422 with the service's hint.
    """

    instruction_template: str


class UserFactCreate(BaseModel):
    """The POST /user-facts body: one atomic fact (arch lsb-0009 §3.1).

            `name` — the fact's title, ≤5 words (the notes' `title` contract); `body`
            — the body, ≤1200 characters. The limits and empty values are checked by
            the SERVICE (not the schema — like the notes' `title` and the skill
            shape): a violation → `UserFactValidationError` → 422 with the verbatim
            §3.7 canon hint; the fact is NOT saved.
    """

    name: str
    body: str


class UserFactUpdate(BaseModel):
    """The PUT /user-facts/{id} body: editing `name` and/or `body` (arch §3.3).

            The service's "not passed = keep" semantics is propagated by the exact set
            of passed fields (`model_fields_set`): an omitted field reaches the
            service as the `_UNSET_*` sentinel default (the value stays); `null` in a
            mandatory field reaches as None and gets a soft 422 failure with the "not
            passed = keep" hint (mandatory fields are never cleared).
    """

    name: str | None = None
    body: str | None = None


class TermCreate(BaseModel):
    """The POST /terms body: the term shape with a mandatory context (§3.1).

            The area key — (`term` + `context`); a repeated key = an update (201/200).
            The limits (term ≤100, context ≤40, definition ≤350) and the context's
            obligatoriness are checked by the SERVICE (not the schema — like the
            skill shape): a violation → `TermValidationError` → 422 with the verbatim
            §3.7 canon hint; no record is written.
    """

    term: str
    context: str
    definition: str


class TermUpdate(TermCreate):
    """The PUT /terms/{id} body: the same full key shape (`term` + `context`).

            There is no separate partial update (the skill-shape precedent): the edit
            addresses the record by id and reassembles the key whole; the new key
            taken by another active record → 409 (`TermKeyConflictError`) — the
            foreign sense is not overwritten; the old key is freed only by deletion.
    """


class LogCreate(BaseModel):
    """The POST /logs body: a logs-area record (arch §7.2, f10-03).

            The fields — like `logs_save`; `action`/`reason` are mandatory (§7.3),
            but the SERVICE checks their non-emptiness and limits (not the schema —
            like the term shape): a violation → `LogValidationError` → 422 with the
            verbatim §7.3 canon hint; no record. `entity`/links and
            `source`/`session` are validated there too; not-passed
            `source`/`session` are taken by the `bind_origin` middleware.
    """

    action: str
    reason: str
    entity: str | None = None
    entity_id: int | None = None
    entity_version: int | None = None
    entity_ref: str | None = None
    outcome: str | None = None
    verified: str | None = None
    source: str | None = None
    session: str | None = None


class LogUpdate(BaseModel):
    """The PUT /logs/{id} body: a partial edit — "not passed = keep" (§7.2).

            All fields are optional: a not-passed one (None) goes as the "keep as is"
            sentinel (the same `_LOG_UNSET` as MCP's `logs_update`). The service will
            return `updated: False` + a hint when nothing was passed or the record is
            missing; the object links are checked by effective values (as in
            `logs_update`).
    """

    action: str | None = None
    reason: str | None = None
    entity: str | None = None
    entity_id: int | None = None
    entity_version: int | None = None
    entity_ref: str | None = None
    outcome: str | None = None
    verified: str | None = None
    source: str | None = None
    session: str | None = None


class HistoryRollback(BaseModel):
    """The POST /history/{entity}/{id}/rollback body: the target version number (§7.2).

            Rollback — an operator handle (FR-5.4): the model returns the record to
            its former state by a NEW version without rewriting the history (§13.6).
    """

    version: int


class QueueStat(BaseModel):
    """One queue's snapshot for /health.queues (FR-2.2, #39 arch §7.2).

            `pending` — how many tasks are takeable NOW; `oldest_pending_sec` — the
            age of the oldest task waiting to be done (pending or waiting; 3.3.1 —
            the age is visible during a slot breaker too; null — neither category
            exists); `waiting` — deferred because of the slot/readiness (#39);
            `attention` — the `waiting` subset waiting for a human (no39-02).
    """

    pending: int
    oldest_pending_sec: int | None
    waiting: int = 0
    attention: int = 0


class SlotStat(BaseModel):
    """One slot's snapshot for /health.slots (#39, arch §7.2).

            `ok` — the slot's last attempt outcome (the `embedding_ok`/
            `summarizer_ok`/`judge_ok` precedent; None — no attempts yet);
            `last_outcome` — the last failure's outcome (the `SlotOutcome`
            dictionary); `paused_until` — ISO-8601 UTC, while the breaker is open
            (null — the slot works); `attention` — the slot's tasks waiting for a
            human (no39-02). #50 (lsbdef-0008):
            `last_finish_reason`/`last_budget`/`last_budget_source`/`last_had_thinking`
            — the last call's diagnostics (None — no call) — the operator sees the
            reason (trimmed by the budget / reasoning ate the output) without the
            journal.
            LSBN-T-33: `last_answer_budget`/`last_reasoning_budget`/`last_sent_budget`
            — the last call's separate budgets; "eaten by reasoning" is visible via
            `last_outcome=truncated` + `last_had_thinking=true` and the budget split.
            Assembled by the worker; SQL/memory only — no models called.
    """

    ok: bool | None
    last_outcome: str | None
    paused_until: str | None
    attention: int
    last_finish_reason: str | None = None
    last_budget: int | None = None
    last_budget_source: str | None = None
    last_had_thinking: bool | None = None
    last_answer_budget: int | None = None
    last_reasoning_budget: int | None = None
    last_sent_budget: int | None = None


class BackupStat(BaseModel):
    """The backups observability for /health (#14, arch §7.2).

            Assembled by scanning the BACKUP_DIR directory, WITHOUT opening the DB:
            the snapshots' count/age/volume, the count of marked ones (`.bad`) and the
            latest verify/ship outcomes from the sidecar index (None — none yet).
    """

    count: int
    newest_age_sec: int | None
    oldest_age_sec: int | None
    total_bytes: int
    bad: int
    last_verify: dict | None
    last_ship: dict | None


class BackupVerifyRequest(BaseModel):
    """The `POST /backups/verify` body (#14, arch §7.2): a file or the newest.

            No/empty `file` → the newest snapshot; otherwise — a file name in
            BACKUP_DIR.
    """

    file: str | None = None


class HealthResponse(BaseModel):
    """The /health contract (NFR-4): for the docker healthcheck and the operator.

            `version` — the application version (FR-4.2, techdebt-0036): exactly one
            source — `app.__version__` (the canon's §7.1); no duplicate field in
            settings. The operator compares it with the release tag: "exactly this
            release is up on the contour".
    """

    status: str
    version: str  # techdebt-0036 (FR-4.2): the application version
    embedding_ok: bool | None  # Phase 3
    summarizer_ok: bool | None  # Phase 4
    judge_ok: bool | None  # Phase 11: the dedup/structure judge (the judge slot)
    # LSBN-F-12 (arch §7.1): whether the decision slot can do it (None — disabled or
    # no check yet; False — the startup check/call suppressed the slot).
    decision_ok: bool | None
    notes_count: int
    # LSBN-S-1: the note text's soft limit (the save/update/merge validation).
    # 3.4.1 (LSBN-T-26): from the build settings — the same number as the new databases' CHECK
    # and the self-migration target (one number — one place: MAX_NOTE_CHARS).
    note_chars_limit: int
    # LSBN-D-14 (3.6.2): the operator brake of the background dedup — false: the
    # `judge` job does not start (its pending backlog stays visible in `queues`).
    dedup_enabled: bool
    pending_vector: int
    pending_summary: int
    queues: dict[str, QueueStat]  # lsb-0014-03: the wait per queue
    # #39 (arch §7.2): per slot {ok, last_outcome, paused_until, attention} —
    # the slot breaker's reason and pause, the tasks in attention. SQL/memory only.
    slots: dict[str, SlotStat]
    # LSBN-F-34 (the brief's decision #4): per operation {model, provider, source} —
    # the effective model/provider and the configuration source (op|slot|default).
    # LSBN-T-33: alongside, the operation's separate budgets (answer/reasoning/sent +
    # their sources) — the configuration ceiling at an empty prompt. NO addresses
    # or keys: /health is exposed outward (NFR-2); the address lives only
    # in the journal (`slot_call`) and in the job's event on error.
    operations: dict[str, dict[str, Any]]
    # #40 (FR-4.3): per area {enabled, pending} — over ALL areas,
    # including disabled ones (their accumulation is visible). Assembled by the worker; SQL only.
    areas: dict[str, dict]
    # #12 (no12-02, §7.1/§7.2): the open warnings' aggregates — {by_kind,
    # total} over the shared `warnings` table (memory and terms). SQL only.
    warnings: dict[str, Any]
    # LSBN-F-22 (M3, design §4 phase 7): the migration block — {phase, target,
    # reason, counters{old_total, poured, split_notes, parts, quarantine,
    # expired, waiting}, invariant_ok} — the reason and progress are visible to the operator
    # (the brief p. 2); the counters — NOTES (the pilot's decision). SQL only.
    migration: dict[str, Any]
    # LSBN-F-12 (arch §7.1): the decision slot's aggregates — {enabled, ok, answers,
    # fallbacks, by_reason:{unavailable,unsupported,low_confidence}} from process
    # memory (probabilities are never served outward — the journal only).
    decision: dict[str, Any]
    # #45 (arch §7.2): the version observability — {enabled, current, latest,
    # checked_at, hint_shown_at}; assembled from the `UpdateChecker` memory (no I/O).
    update: dict[str, Any]
    # #14 (arch §7.2): the backups observability — a BACKUP_DIR directory scan without
    # opening the DB + the latest verify/ship outcomes from the sidecar index.
    backups: BackupStat


def _services(request: Request) -> Services:
    return request.app.state.services  # type: ignore[no-any-return]


def _decision_health(services: Services) -> tuple[bool | None, dict[str, Any]]:
    """The decision slot's snapshot for /health (LSBN-F-12, arch §7.1/§7.3).

            `decision_ok` — whether the slot can do it (None — disabled/not built);
            `decision` — the `{enabled, ok, answers, fallbacks, by_reason}` aggregates
            from the service's memory (probabilities are never served outward — the
            journal only). No models called: only counter reads
            (`DecisionService.snapshot()`); an empty slot — the "disabled" shape with
            zero counters.
    """
    service = services.decision
    if service is None:
        return None, {
            "enabled": False,
            "ok": None,
            "answers": 0,
            "fallbacks": 0,
            "by_reason": {
                "unavailable": 0,
                "unsupported": 0,
                "low_confidence": 0,
            },
        }
    snapshot = service.snapshot()
    return snapshot["ok"], {
        "enabled": snapshot["enabled"],
        "ok": snapshot["ok"],
        "answers": snapshot["answers"],
        "fallbacks": snapshot["fallbacks"],
        "by_reason": snapshot["by_reason"],
    }


def _skills_service(request: Request) -> SkillsService:
    """The skills area service — shared by all /skills REST handles.

            `Services.skills` is optional only for legacy DI builds; the application
            (`create_app`) always assembles the area — its unavailability would mean
            a configuration error, so we answer 503 instead of crashing with a
            traceback.
    """
    service = _services(request).skills
    if service is None:
        raise HTTPException(status_code=503, detail="skills area is not available")
    return service


def _user_facts_service(request: Request) -> UserFactsService:
    """The "user" area service — shared by all /user-facts REST handles.

            The same service layer as the `user_*` MCP tools (substrate §3.4): the
            area's behavior is identical on both surfaces; only the output differs
            (REST — full field contracts).
    """
    return _services(request).user_facts


def _warnings_service(request: Request) -> WarningsService:
    """The shared warnings surface service (#12, no12-02) — for /warnings.

            `Services.warnings` is optional only for legacy DI builds; the application
            (`create_app`/`build_services`) always provides it — its unavailability
            would mean an assembly error; we answer 503 instead of crashing with a
            traceback.
    """
    service = _services(request).warnings
    if service is None:
        raise HTTPException(
            status_code=503, detail="warnings surface is not available"
        )
    return service


def _backup_service(request: Request) -> BackupService:
    """The backups service — for the operator's `GET /backups` and `POST /backups/verify`.

            `Services.backup` is typed optional only for legacy test DI builds (there
            it is None); the application (`create_app`/`build_services`) always
            provides it — we answer 503 to its unavailability instead of crashing
            with a traceback (like `_skills_service`/`_warnings_service`).
    """
    service = _services(request).backup
    if service is None:
        raise HTTPException(status_code=503, detail="backup service is not available")
    return service


def _terms_service(request: Request) -> TermsService:
    """The "terms" area service — shared by all /terms REST handles.

            The same service layer as the `terms_*` MCP tools (substrate §3.4): the
            area's behavior is identical on both surfaces; only the output differs
            (REST — full field contracts, no whitelist trim).
    """
    return _services(request).terms


def _logs_service(request: Request) -> LogsService:
    """The logs area service — shared by all /logs REST handles (f10-03).

            `Services.logs` is optional only for legacy DI builds; the application
            (`create_app`/`build_services`) assembles the area ALWAYS — the operator's
            access survives even a disabled flag (§13.11), so 503 is only for builds
            without the area (like `_skills_service`/`_warnings_service`).
    """
    service = _services(request).logs
    if service is None:
        raise HTTPException(status_code=503, detail="logs area is not available")
    return service


def _unprocessable(exc: ValueError) -> HTTPException:
    """Domain violations → 422 with the service's text (no internals)."""
    return HTTPException(status_code=422, detail=str(exc))


def _log_field(value: Any) -> Any:
    """A PUT /logs/{id} field: not passed (None) means "keep as is".

            The bridge between the REST schema (optional fields default to None) and
            the service, where "not passed" is the `_LOG_UNSET` sentinel (one
            codebase with MCP's `logs_update`); this way an empty string stays
            meaningful (a field clearing).
    """
    return _LOG_UNSET if value is None else value


def _conflict(exc: NamespaceError) -> HTTPException:
    """A structure-handles conflict (Step 6): the default protection, a duplicate
            path, a merge into itself — 409 with the service's text (not found —
            checked in advance).
    """
    return HTTPException(status_code=409, detail=str(exc))


async def _bind_rest_origin(request: Request) -> AsyncIterator[None]:
    """Wrap a REST request with the versions source (§13.4, #41).

            `source='rest'`; the session — the `X-LSB-Session` header (not passed →
            NULL: a human-like identifier is never invented). The router dependency
            holds the contextvar while the request is processed; the services'
            `asyncio.to_thread` propagates the context — the operator's edit versions
            carry the `rest` source.
    """
    with history.bind_origin("rest", session=request.headers.get("X-LSB-Session")):
        yield


def build_rest_router(settings: Settings) -> APIRouter:
    """Assemble the router; the pagination defaults — from the environment settings."""
    rest_router = APIRouter(dependencies=[Depends(_bind_rest_origin)])

    @rest_router.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        """The process's liveness. Answers without a token (the NFR-2 exception).

                    `embedding_ok` — the last vectorization attempt's outcome (None — no
                    attempts; updated by EmbeddingService — the single point of all
                    encodings);
                    `summarizer_ok` — the last summary generation's outcome (None — no
                    attempts; updated by SummaryService — all generations come from the
                    worker);
                    `judge_ok` — the last judge call's outcome (Phase 11: the startup
                    check at ok gives True; at unavailability it stays None — False will
                    be set by the first real failure, see app/main.py).
                    The counters — from the DB: active notes (trash is not served).
                    `queues` — the job queues' snapshots (FR-2.2, lsb-0014-03): per queue,
                    the waiting task count and the oldest's age; assembled by the worker
                    over the registry, SQL only — no model calls.
                    The legacy fields (`pending_vector`/`pending_summary`) are preserved.
                    `version` — the application version (FR-4.2, techdebt-0036): the
                    operator compares it with the release tag at acceptance.
                    `migration` — LSBN-F-22 (M3): the self-migration's phase/reason/
                    progress ({"phase": None} — the migration never touched the base).
        """
        services = _services(request)
        worker = request.app.state.worker  # type: ignore[attr-defined]
        counts = await asyncio.to_thread(services.notes.health_counts)
        queues = await asyncio.to_thread(worker.queues_health)
        # #39 (arch §7.2): the slots snapshot — the breaker's reason and pause (SQL/
        # memory only, no models called).
        slots = await asyncio.to_thread(worker.slots_health)
        areas = await asyncio.to_thread(worker.areas_health)
        # #12 (no12-02): the open warnings' aggregates — SQL only.
        if services.warnings is None:
            warnings: dict[str, Any] = {"by_kind": {}, "total": 0}
        else:
            warnings = await asyncio.to_thread(services.warnings.counts)
        # LSBN-F-12 (arch §7.1): `decision_ok` and the `decision` block — from the service's
        # memory (§7.3): how many decisions were made, how many switches and for
        # which reasons. No models called (a counter snapshot only).
        decision_ok, decision = _decision_health(services)
        # #45 (arch §7.2): the version snapshot — from the checker's memory, no I/O.
        checker = getattr(request.app.state, "update_checker", None)
        update = (
            checker.health()
            if checker is not None
            else {
                "enabled": False,
                "current": __version__,
                "latest": None,
                "checked_at": None,
                "hint_shown_at": None,
            }
        )
        # #14 (arch §7.2): the backups observability — a directory scan without opening
        # the DB (`BackupService.status`); a DI build without the service — an empty shape.
        backup = services.backup
        backups = (
            BackupStat(
                count=0,
                newest_age_sec=None,
                oldest_age_sec=None,
                total_bytes=0,
                bad=0,
                last_verify=None,
                last_ship=None,
            )
            if backup is None
            else BackupStat(**await asyncio.to_thread(backup.status))
        )
        # LSBN-F-22 (M3): the migration's phase/reason/progress — SQL only
        # (the brief p. 2: "the reason and progress are visible in /health").
        migration = await asyncio.to_thread(migration_health, settings)
        return HealthResponse(
            status="ok",
            version=__version__,
            embedding_ok=services.embedding.last_attempt_ok,
            summarizer_ok=services.summary.last_attempt_ok,
            judge_ok=services.judge.last_attempt_ok,
            decision_ok=decision_ok,
            notes_count=counts["notes_count"],
            # LSBN-S-1: the note's soft limit is visible to the operator (from settings).
            note_chars_limit=settings.max_note_chars,
            # LSBN-D-14: the dedup brake is visible to the operator (from settings).
            dedup_enabled=settings.dedup_enabled,
            pending_vector=counts["pending_vector"],
            pending_summary=counts["pending_summary"],
            queues=queues,
            slots=slots,
            # LSBN-F-34: the operations block — from the registry and settings, no I/O
            # and no model calls (model/provider/source per operation).
            operations=operations_health(settings),
            areas=areas,
            warnings=warnings,
            decision=decision,
            update=update,
            backups=backups,
            migration=migration,
        )

    @rest_router.post("/notes", status_code=201)
    async def create_note(payload: NoteCreate, request: Request) -> dict:
        """Create a note (memory_save FR-4: vectorization + dedup).

                    An environment without Ollama → a degradation (pending + a warning; a
                    text-based dedup).
                    Decision #9 (follow-up 5b): the title is mandatory — without one/invalid →
                    TitleValidationError → 422 fail+hint (the contract is one with MCP).
        """
        try:
            # title=None from the transport ("the client did not name the note") — a service
            # failure (TitleValidationError); the sentinel legacy is unreachable
            # via REST: title is always passed explicitly.
            return await asyncio.to_thread(
                _services(request).notes.save,
                payload.text,
                payload.author,
                title=payload.title,
                comment=payload.comment,
            )
        except NoteValidationError as exc:
            raise _unprocessable(exc) from exc

    @rest_router.get("/notes")
    async def list_notes(
        request: Request,
        limit: int | None = Query(default=None, ge=1, le=50),
        offset: int = Query(default=0, ge=0),
        updated_after: str | None = Query(default=None),
    ) -> dict:
        """The memory overview: a page of brief summaries + the pagination fields (FR-2).

                    lsb-0013-02 (arch §3.4): the surface ceiling (`list_max_limit_rest`,
                    50) is checked by the service; the answer carries the same page fields
                    as MCP (`total`/`has_more`/`next_offset`/`next_cursor`); there is no
                    "+N more" hint on REST — the operator's full contract. no29-01:
                    `updated_after` (ISO-8601 UTC, an inclusive bound) filters the page
                    and `total`; an invalid value → 422 with a hint text.
        """
        try:
            return await asyncio.to_thread(
                _services(request).notes.list,
                limit,
                offset,
                max_limit=settings.list_max_limit_rest,
                updated_after=updated_after,
            )
        except (NoteValidationError, FreshnessValidationError) as exc:
            raise _unprocessable(exc) from exc

    @rest_router.get("/notes/comments")
    async def search_comments(
        request: Request,
        q: str = Query(..., min_length=1, max_length=settings.max_query_chars),
        top_k: int | None = Query(default=None, ge=1, le=20),
        namespace: str | None = Query(default=None),
        namespace_exact: bool = Query(default=False),
        updated_after: str | None = Query(default=None),
    ) -> dict:
        """A separate COMMENT search (no24-02, §13.6): over comment only.

                    Declared BEFORE `GET /notes/{note_id}` (like `/skills/search` before
                    `/skills/{skill_id}`) — otherwise "comments" would parse as note_id
                    and return 422. The contract is the same as MCP's (`{results, hint?}`);
                    the results are not mixed with the semantic search. Invalid
                    `q`/`top_k`/`updated_after` and an unregistered node → 422 with a
                    hint text.
        """
        try:
            return await asyncio.to_thread(
                _services(request).search.comment_search,
                q, top_k, namespace, namespace_exact, updated_after,
            )
        except (SearchValidationError, FreshnessValidationError) as exc:
            raise _unprocessable(exc) from exc
        except NamespaceError as exc:
            raise _unprocessable(exc) from exc

    @rest_router.get("/notes/{note_id}")
    async def get_note(note_id: int, request: Request) -> dict:
        """One note's full text + links from OTHER namespaces.

                    The single-id alias of batch memory_get: the service contract is full
                    (`chars` — the text volume); lsb-0010-05 adds `links` — an item {id,
                    title, namespace, chars} (the assembly — at the transport level,
                    arch §3.5). There is NO separate operator handle for viewing links:
                    links are visible in the note read; the table is available in the DB.
        """
        services = _services(request)
        result = await asyncio.to_thread(services.notes.get, [note_id])
        if not result["notes"]:
            raise HTTPException(
                status_code=404,
                detail=result.get("hint", "заметка не найдена"),
            )
        note = dict(result["notes"][0])
        # An empty list — a normal answer: no `hint` and no pointers.
        # `Services.links` can be None in test DI builds (no links) —
        # we degrade to an empty list, like the skills announce without the area (lsb-0007).
        links = services.links
        note["links"] = (
            list(await asyncio.to_thread(links.related, note_id)) if links else []
        )
        return note

    @rest_router.put("/notes/{note_id}")
    async def update_note(note_id: int, payload: NoteUpdate, request: Request) -> dict:
        """Overwrite the note whole (FR-5).

                    no24-01 (3.2.0): `comment` — by `model_fields_set` (omitted → the
                    `_UNSET_COMMENT` sentinel = untouched; `null`/`""` → a reset); `text`
                    not passed (or `null`) → the `_UNSET_TEXT` sentinel — a comment-only
                    edit, the text is not overwritten (the lsb-0004-01 service contract).
                    ``CommentValidationError`` → 422 by the existing
                    `except NoteValidationError`.
        """
        try:
            result = await asyncio.to_thread(
                _services(request).notes.update,
                note_id,
                payload.text if payload.text is not None else _UNSET_TEXT,
                title=payload.title,
                comment=(
                    payload.comment
                    if "comment" in payload.model_fields_set
                    else _UNSET_COMMENT
                ),
            )
        except NoteValidationError as exc:
            raise _unprocessable(exc) from exc
        if not result["updated"]:
            raise HTTPException(
                status_code=404,
                detail=result.get("hint", "заметка не найдена"),
            )
        return result

    @rest_router.delete("/notes/{note_id}")
    async def delete_note(note_id: int, request: Request) -> dict:
        """Soft delete (FR-6): physically the note stays in trash."""
        result = await asyncio.to_thread(_services(request).notes.delete, note_id)
        if not result["deleted"]:
            raise HTTPException(
                status_code=404,
                detail=result.get("hint", "заметка не найдена"),
            )
        return result

    @rest_router.get("/search")
    async def search_notes(
        request: Request,
        q: str = Query(..., min_length=1, max_length=settings.max_query_chars),
        top_k: int | None = Query(default=None, ge=1, le=20),
        updated_after: str | None = Query(default=None),
    ) -> dict:
        """Search (Phase 3 — the vec0+FTS hybrid; the FR-1 output; offline → FTS-only).

                    no29-01: `updated_after` (ISO-8601 UTC, an inclusive bound) narrows
                    the hybrid to fresh notes; an invalid value → 422 with a hint text.
        """
        try:
            return await asyncio.to_thread(
                _services(request).search.search, q, top_k, None, False,
                updated_after,
            )
        except (SearchValidationError, FreshnessValidationError) as exc:
            raise _unprocessable(exc) from exc

    # --- namespaces: the operator's structure handles (Phase 10, Step 6) ------
    # §5.7: the structure handles — the operator's REST, NOT in MCP (client models do
    # not govern structure). One service codebase; the existing REST contracts
    # are untouched (the Phase-9 precedent — new routes do not change old ones).

    @rest_router.get("/namespaces")
    async def list_namespaces(request: Request) -> dict:
        """The node registry with counters + promotion_candidates (US-9).

                    candidates — the trigger's live SQL aggregation (growing hint groups
                    of default notes, not yet run through the structure judge).
        """
        services = _services(request)
        result = await asyncio.to_thread(services.namespaces.list_all)
        try:
            candidates = await asyncio.to_thread(services.promotion.candidates)
        except Exception:
            # candidates — an auxiliary layer: an aggregation failure does not break
            # the registry (a degradation, as in MCP memory_namespaces).
            candidates = []
        return {
            "namespaces": result["namespaces"],
            "promotion_candidates": candidates,
        }

    @rest_router.post("/namespaces", status_code=201)
    async def create_namespace(payload: NamespaceCreate, request: Request) -> dict:
        """Register a node (confirmed): the starter set/new roots/leaves."""
        try:
            return await asyncio.to_thread(
                _services(request).namespaces.create, payload.path, payload.description
            )
        except NamespaceValidationError as exc:
            raise _unprocessable(exc) from exc
        except NamespaceError as exc:
            raise _conflict(exc) from exc

    @rest_router.patch("/namespaces/{path:path}")
    async def patch_namespace(path: str, payload: NamespacePatch, request: Request) -> dict:
        """Edit a node: the description, the status (confirm) and/or a rename.

                    A rename — moving notes/markup/verdicts (§5.7: nothing is lost); the
                    status — the provisional → confirmed audit (and back, if the operator
                    changed their mind). No service fields passed — the node unchanged.
        """
        services = _services(request)
        node = await asyncio.to_thread(services.namespaces.get, path)
        if node is None:
            raise HTTPException(status_code=404, detail=f"узел «{path}» не зарегистрирован")
        try:
            if payload.path is not None:
                node = await asyncio.to_thread(
                    services.namespaces.rename, node["path"], payload.path
                )
            if payload.description is not None:
                node = await asyncio.to_thread(
                    services.namespaces.update_description, node["path"], payload.description
                )
            if payload.status is not None:
                node = await asyncio.to_thread(
                    services.namespaces.set_status, node["path"], payload.status
                )
        except NamespaceValidationError as exc:
            raise _unprocessable(exc) from exc
        except NamespaceError as exc:
            raise _conflict(exc) from exc
        return node

    @rest_router.post("/namespaces/{path:path}/merge")
    async def merge_namespace(path: str, payload: NamespaceMerge, request: Request) -> dict:
        """Merge a leaf into the canonical node: the notes moved, the node is gone (US-11)."""
        services = _services(request)
        node = await asyncio.to_thread(services.namespaces.get, path)
        if node is None:
            raise HTTPException(status_code=404, detail=f"узел «{path}» не зарегистрирован")
        try:
            return await asyncio.to_thread(
                services.namespaces.merge_node, node["path"], payload.into
            )
        except NamespaceValidationError as exc:
            raise _unprocessable(exc) from exc
        except NamespaceError as exc:
            raise _conflict(exc) from exc

    @rest_router.delete("/namespaces/{path:path}")
    async def delete_namespace(path: str, request: Request) -> dict:
        """Delete a node with note moving (§5.7: nothing is lost)."""
        services = _services(request)
        node = await asyncio.to_thread(services.namespaces.get, path)
        if node is None:
            raise HTTPException(status_code=404, detail=f"узел «{path}» не зарегистрирован")
        try:
            return await asyncio.to_thread(services.namespaces.delete_node, node["path"])
        except NamespaceValidationError as exc:
            raise _unprocessable(exc) from exc
        except NamespaceError as exc:
            raise _conflict(exc) from exc

    # --- the skills area: REST mirrors (release 3.0.0, lsb-0007-05) ----------
    # Arch lsb-0007 §3.7 + substrate §3.6: the skills operator surface —
    # the same Bearer and the same service layer as MCP; full outputs (the whitelist
    # trimming lives only in the tools). Codes: 201 — creation; 200 —
    # read/edit/delete; 422 — shape validation and the service's soft failures
    # (the text = a hint); 404 — the skill not found/deleted. THE ROUTE ORDER:
    # the static ones (`/skills/search`, `/skills/instruction-template`) are declared
    # BEFORE `/skills/{skill_id}` — otherwise "search" would fall into the integer path.

    @rest_router.post("/skills", status_code=201)
    async def create_skill(payload: SkillCreate, request: Request) -> dict:
        """Create a skill: the shape validation + the anti-synonymy as in MCP (§3.4).

                    The service's soft failures → 422 with the verbatim §3.8 canon hint:
                    a shape limit violation (`SkillValidationError`) and a candidate "too
                    similar" to an active skill (`created: False` + a hint with the
                    existing one's id/name).
        """
        try:
            result = await asyncio.to_thread(
                _skills_service(request).save,
                name=payload.name,
                description=payload.description,
                steps=payload.steps,
                text=payload.text,
                example=payload.example,
                extra=payload.extra,
            )
        except SkillValidationError as exc:
            raise _unprocessable(exc) from exc
        if not result.get("created"):
            raise HTTPException(status_code=422, detail=result.get("hint", ""))
        return result

    @rest_router.get("/skills")
    async def list_skills(
        request: Request,
        limit: int | None = Query(default=None, ge=1, le=50),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        """A listing of active skills by FULL records (pagination like /notes).

                    MCP's `skills_list` serves a compact cut (id/name/description) — the
                    operator needs the full record, so per active skill the same §3.1
                    composite as `GET /skills/{id}`'s is assembled (including the global
                    `instruction_template` section). The version archive and deleted rows
                    do not enter the listing.
                    lsb-0013-02 (arch §3.4): the surface ceiling (`list_max_limit_rest`)
                    goes into the service; the answer carries the same page fields as MCP;
                    there is no "+N more" hint on REST.
        """
        skills = _skills_service(request)

        def _full_listing() -> dict:
            """Service calls in one thread: the listing + a composite per item."""
            listing = skills.list(
                limit, offset, max_limit=settings.list_max_limit_rest
            )
            return {
                "items": [skills.get(item["id"]) for item in listing["items"]],
                "total": listing["total"],
                "has_more": listing["has_more"],
                "next_offset": listing["next_offset"],
                "next_cursor": listing["next_cursor"],
            }

        try:
            return await asyncio.to_thread(_full_listing)
        except SkillValidationError as exc:
            raise _unprocessable(exc) from exc

    @rest_router.get("/skills/search")
    async def search_skills(
        request: Request,
        q: str = Query(..., min_length=1, max_length=settings.max_query_chars),
        top_k: int | None = Query(default=None, ge=1, le=20),
    ) -> dict:
        """The skill hybrid search (vec0 + FTS → RRF) — the full output.

                    The difference from MCP's `skills_search` — no output cut: the
                    degradation `warning` (FTS-only, NFR-3) is visible to the operator; an
                    empty result — a soft answer with the verbatim §3.8 canon hint (not an
                    error).
        """
        try:
            return await asyncio.to_thread(
                _skills_service(request).search, q, top_k
            )
        except SkillValidationError as exc:
            raise _unprocessable(exc) from exc

    @rest_router.get("/skills/instruction-template")
    async def get_instruction_template(request: Request) -> dict:
        """The global `instruction_template` ("how to execute the steps", §3.1)."""
        return await asyncio.to_thread(
            _skills_service(request).instruction_template
        )

    @rest_router.put("/skills/instruction-template")
    async def put_instruction_template(
        payload: InstructionTemplate, request: Request
    ) -> dict:
        """Edit the global template: empty/longer than 2000 chars → 422 + a hint."""
        try:
            return await asyncio.to_thread(
                _skills_service(request).set_instruction_template,
                payload.instruction_template,
            )
        except SkillValidationError as exc:
            raise _unprocessable(exc) from exc

    @rest_router.get("/skills/{skill_id}")
    async def get_skill(skill_id: int, request: Request) -> dict:
        """The skill's full record + the composite (`instruction_template`, `extra`).

                    A deleted/nonexistent skill is indistinguishable from "no row" →
                    404 with the §3.8 canon hint (the service restoration — the
                    operator).
        """
        record = await asyncio.to_thread(_skills_service(request).get, skill_id)
        if "name" not in record:  # the service's soft answer: no row / deleted
            raise HTTPException(
                status_code=404, detail=record.get("hint", "навык не найден")
            )
        return record

    @rest_router.put("/skills/{skill_id}")
    async def update_skill(
        skill_id: int, payload: SkillUpdate, request: Request
    ) -> dict:
        """Edit the skill shape: the former version goes into the archive (§3.4).

                    The shape validation — as at creation (422 + a hint); the service's
                    `updated: False` (no active row) → 404 with the canon hint.
        """
        try:
            result = await asyncio.to_thread(
                _skills_service(request).save,
                id=skill_id,
                name=payload.name,
                description=payload.description,
                steps=payload.steps,
                text=payload.text,
                example=payload.example,
                extra=payload.extra,
            )
        except SkillValidationError as exc:
            raise _unprocessable(exc) from exc
        if not result.get("updated"):
            raise HTTPException(
                status_code=404, detail=result.get("hint", "навык не найден")
            )
        return result

    @rest_router.delete("/skills/{skill_id}")
    async def delete_skill(skill_id: int, request: Request) -> dict:
        """Soft-delete a skill (§3.4): the row/indexes stay alive; the outputs do not see it.

                    A repeated/nonexistent deletion — 404 with the §3.8 canon hint.
        """
        result = await asyncio.to_thread(
            _skills_service(request).delete, skill_id
        )
        if not result.get("deleted"):
            raise HTTPException(
                status_code=404, detail=result.get("hint", "навык не найден")
            )
        return result

    @rest_router.get("/skills/{skill_id}/versions")
    async def get_skill_versions(skill_id: int, request: Request) -> dict:
        """The skill's version archive — a REST alias over the SHARED history (#41).

                    The second mechanic (`skill_versions`) was removed in no41-03 (§13.7):
                    the handle reads the same `record_versions` (`read_versions`/
                    `read_version` over the `skill` entity) and assembles the FORMER
                    answer shape — the operator reference's contract does not change. The
                    skill not found/deleted → 404 with the §3.8 canon hint (as before).
        """
        live = await asyncio.to_thread(
            history.entity_fields, history.ENTITY_SKILL, skill_id, settings
        )
        if live is None or live[1] == "deleted":
            raise HTTPException(status_code=404, detail=HINT_SKILL_NOT_FOUND)

        def _archive() -> dict:
            """Assemble the former archive shape from the shared history's versions."""
            listing = history.read_versions(history.ENTITY_SKILL, skill_id, settings)
            versions = []
            for meta in listing["versions"]:
                detail = history.read_version(
                    history.ENTITY_SKILL, skill_id, meta["version"], settings
                )
                fields = detail.get("fields") or {}
                versions.append(
                    {
                        "version": meta["version"],
                        "name": fields.get("name"),
                        "description": fields.get("description"),
                        "example": fields.get("example"),
                        "steps": fields.get("steps"),
                        "text": fields.get("text"),
                        "extra": fields.get("extra"),
                        "created_at": meta["created_at"],
                    }
                )
            return {"id": skill_id, "versions": versions}

        return await asyncio.to_thread(_archive)

    # --- the user area: REST mirrors (release 3.0.0, lsb-0009-03) -------------
    # Arch lsb-0009 §3.6 + substrate §3.6: the operator surface of facts about
    # the user — the same Bearer and the same service layer as MCP
    # (`user_save`/`user_search`/`user_get`/`user_update`/`user_delete`);
    # full outputs (the whitelist trimming lives only in the tools). Codes:
    # 201 — creation; 200 — read/edit/delete; 422 — shape validation and
    # the service's soft failures (the text = the verbatim §3.7 canon hint, including
    # the dedup's strong match `stored: False`); 404 — the fact not found
    # (including a deleted one — soft delete). No listing: the mirrors mirror
    # the area's MCP handles (arch §3.6). THE ROUTE ORDER: the static
    # `/user-facts/search` is declared BEFORE `/user-facts/{fact_id}` — otherwise
    # "search" would fall into the integer path.

    @rest_router.post("/user-facts", status_code=201)
    async def create_user_fact(payload: UserFactCreate, request: Request) -> dict:
        """Create a fact: the shape validation + the dedup hint as in MCP (§3.4).

                    The order and texts are the service's: a limit violation (`name` >5
                    words, `body` >1200, an empty mandatory field) →
                    `UserFactValidationError` → 422 with the verbatim hint; a strong match
                    with an active fact → the service's soft failure (`stored: False`) →
                    422 with the hint leading to `user_update`; the middle zone → 201,
                    with `related` (similar facts' id/name) and the hint in the answer.
                    Success always carries the permanent atomicity hint (FR-7.2).
        """
        try:
            result = await asyncio.to_thread(
                _user_facts_service(request).save,
                name=payload.name,
                body=payload.body,
            )
        except UserFactValidationError as exc:
            raise _unprocessable(exc) from exc
        if not result.get("stored"):
            # A service soft failure (a strong match) — 422 with its hint.
            raise HTTPException(status_code=422, detail=result.get("hint", ""))
        return result

    @rest_router.get("/user-facts/search")
    async def search_user_facts(
        request: Request,
        q: str = Query(..., min_length=1, max_length=settings.max_query_chars),
        top_k: int | None = Query(default=None, ge=1, le=20),
    ) -> dict:
        """The area's hybrid search — a "probe" (arch §3.3) with the FULL output.

                    The difference from MCP's `user_search` — no output cut: the
                    degradation `warning` (FTS-only, NFR-3) is visible to the operator; an
                    empty result — a soft answer with the verbatim §3.7 canon hint (not an
                    error). The output — an `excerpt` (≤300 characters); only
                    `GET /user-facts/{id}` serves the full body.
        """
        try:
            return await asyncio.to_thread(
                _user_facts_service(request).search, q, top_k
            )
        except UserFactValidationError as exc:
            raise _unprocessable(exc) from exc

    @rest_router.get("/user-facts/{fact_id}")
    async def get_user_fact(fact_id: int, request: Request) -> dict:
        """The full fact (`id`, `name`, `body`) — only this handle serves the full body.

                    A nonexistent/deleted fact is indistinguishable from "no row" → 404
                    with the §3.7 canon hint (the service restoration — the operator).
        """
        result = await asyncio.to_thread(_user_facts_service(request).get, fact_id)
        if "name" not in result:  # the service's soft answer: no row / deleted
            raise HTTPException(
                status_code=404, detail=result.get("hint", "user fact not found")
            )
        return result

    @rest_router.put("/user-facts/{fact_id}")
    async def update_user_fact(
        fact_id: int, payload: UserFactUpdate, request: Request
    ) -> dict:
        """Edit a fact: "not passed" = keep; `null` in a mandatory one — 422.

                    Only the really passed fields go into the service
                    (`model_fields_set`): an omitted field stays behind the `_UNSET_*`
                    sentinel, so the former value is preserved; `null` reaches as None and
                    the service answers a soft failure with the "not passed = keep" hint
                    (422). The validation of passed values — as at creation (422 + the
                    verbatim hint); the service's `hint` (no active row) → 404 with the
                    canon hint.
        """
        updates: dict[str, str | None] = {}
        if "name" in payload.model_fields_set:
            updates["name"] = payload.name
        if "body" in payload.model_fields_set:
            updates["body"] = payload.body
        try:
            result = await asyncio.to_thread(
                _user_facts_service(request).update, fact_id, **updates
            )
        except UserFactValidationError as exc:
            raise _unprocessable(exc) from exc
        if "hint" in result:  # not found/deleted — 404 with the §3.7 canon hint
            raise HTTPException(status_code=404, detail=result["hint"])
        return result

    @rest_router.delete("/user-facts/{fact_id}")
    async def delete_user_fact(fact_id: int, request: Request) -> dict:
        """Soft-delete a fact (§3.4): the row/indexes stay alive; the outputs do not see it.

                    A repeated/nonexistent deletion — 404 with the §3.7 canon hint.
        """
        result = await asyncio.to_thread(
            _user_facts_service(request).delete, fact_id
        )
        if not result.get("deleted"):
            raise HTTPException(
                status_code=404, detail=result.get("hint", "user fact not found")
            )
        return result

    # --- the terms area: REST mirrors (release 3.0.0, lsb-0008-03) ------------
    # Arch lsb-0008 §3.6 + substrate §3.6: the terminology area's operator surface —
    # the same Bearer and the same service as MCP (`terms_save`/
    # `terms_search`/`terms_get`); full outputs (the whitelist trimming
    # lives only in the tools). Codes: 201 — creation; 200 — a key update, a read,
    # an edit, a deletion; 422 — shape validation and the service's soft failures
    # (the text = the verbatim §3.7 canon hint, including the close context
    # `created: False`); 404 — the record not found (including a deleted one);
    # 409 — the edit's new key is taken by another active record (substrate §3.6).
    # There is NO listing: the only road to a definition is the search;
    # the mirrors mirror the area's MCP handles. THE ROUTE ORDER: the static
    # `/terms/search` is declared BEFORE `/terms/{term_id}` — otherwise "search" would fall
    # into the integer path.

    @rest_router.post("/terms", status_code=201)
    async def create_term(
        payload: TermCreate, request: Request, response: Response
    ) -> dict:
        """Write a term by key: 201 — created; 200 — updated (§3.5).

                    The FR-3.1 mechanics and the texts are the service's: the key matched →
                    an edit of the existing sense (a dynamic 200 status); a close context
                    (`created: False` + a hint with the existing record's id) and a
                    limit/empty-context violation (`TermValidationError`) → 422 with the
                    verbatim §3.7 canon hint. Otherwise — a new sense → 201.
        """
        try:
            result = await asyncio.to_thread(
                _terms_service(request).save,
                term=payload.term,
                context=payload.context,
                definition=payload.definition,
            )
        except TermValidationError as exc:
            raise _unprocessable(exc) from exc
        if result.get("created"):
            return result
        if result.get("updated"):
            response.status_code = 200  # an edit by key — a dynamic status
            return result
        # A service soft failure (a close context) — 422 with its hint.
        raise HTTPException(status_code=422, detail=result.get("hint", ""))

    @rest_router.get("/terms/search")
    async def search_terms(
        request: Request,
        q: str = Query(..., min_length=1, max_length=settings.max_query_chars),
        top_k: int | None = Query(default=None, ge=1, le=20),
    ) -> dict:
        """The term's two-step search (§3.4) with the FULL output.

                    Step 1 — an exact normalized lookup: ALL the term's senses with
                    contexts (`senses`/`exact: true`); step 2 — the area hybrid
                    (`senses`/`exact: false` + the "not an exact match" hint); step 3 —
                    empty + the §3.7 canon hint. The difference from MCP's `terms_search`
                    — no output cut: the degradation `warning` (FTS-only, NFR-3) is
                    visible to the operator.
        """
        try:
            return await asyncio.to_thread(
                _terms_service(request).search, q, top_k
            )
        except TermValidationError as exc:
            raise _unprocessable(exc) from exc

    @rest_router.get("/terms/{term_id}")
    async def get_term(term_id: int, request: Request) -> dict:
        """Read the record (`id`, `term`, `context`, `definition`).

                    A nonexistent/deleted record is indistinguishable from "no row" →
                    404 with the §3.7 canon hint (the service restoration — the
                    operator).
        """
        record = await asyncio.to_thread(_terms_service(request).get, term_id)
        if "term" not in record:  # the service's soft answer: no row / deleted
            raise HTTPException(
                status_code=404, detail=record.get("hint", "term not found")
            )
        return record

    @rest_router.put("/terms/{term_id}")
    async def update_term(
        term_id: int, payload: TermUpdate, request: Request
    ) -> dict:
        """An edit by id: the whole shape; a move to a taken key → 409 (§3.6).

                    The shape validation — as at creation (422 + the verbatim canon
                    hint); the new key taken by ANOTHER active record →
                    `TermKeyConflictError` → 409 with a hint (the foreign sense is not
                    overwritten; the operator edits it or changes the key); the service's
                    `updated: False` (no active row) → 404.
        """
        try:
            result = await asyncio.to_thread(
                _terms_service(request).update,
                term_id,
                payload.term,
                payload.context,
                payload.definition,
            )
        except TermKeyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TermValidationError as exc:
            raise _unprocessable(exc) from exc
        if not result.get("updated"):
            raise HTTPException(
                status_code=404, detail=result.get("hint", "term not found")
            )
        return result

    @rest_router.delete("/terms/{term_id}")
    async def delete_term(term_id: int, request: Request) -> dict:
        """Soft delete (§3.3): the row/indexes stay alive; the outputs do not see it.

                    The deleted key is freed (the partial UNIQUE over active records):
                    the same key can be created anew. A repeated/nonexistent deletion →
                    404 with the §3.7 canon hint.
        """
        result = await asyncio.to_thread(
            _terms_service(request).delete, term_id
        )
        if not result.get("deleted"):
            raise HTTPException(
                status_code=404, detail=result.get("hint", "term not found")
            )
        return result

    # --- the logs area: the operator surface (LSBN-F-10, f10-03) --------------
    # §7.2: FULL outputs (the body is not trimmed — the operator needs the full text;
    # MCP trims by whitelists); they work at ANY value of the area flag
    # (§13.11). Order: no static subpaths — every `/logs/{log_id}`
    # is integer; "search" does not take the REST road here (§7.2).

    @rest_router.post("/logs", status_code=201)
    async def create_log(payload: LogCreate, request: Request) -> dict:
        """Create a record: 201 — the FULL record; 422 — the shape (§7.2).

                    The shape is checked by the service BEFORE the DB (a violation → the
                    verbatim §7.3 hint); after creation we serve the record whole
                    (`get`), not the save envelope.
        """
        service = _logs_service(request)
        try:
            result = await asyncio.to_thread(
                service.save,
                action=payload.action,
                reason=payload.reason,
                entity=payload.entity,
                entity_id=payload.entity_id,
                entity_version=payload.entity_version,
                entity_ref=payload.entity_ref,
                outcome=payload.outcome,
                verified=payload.verified,
                source=payload.source,
                session=payload.session,
            )
        except LogValidationError as exc:
            raise _unprocessable(exc) from exc
        return await asyncio.to_thread(service.get, result["id"])

    @rest_router.get("/logs")
    async def list_logs(
        request: Request,
        entity: str | None = Query(default=None),
        entity_id: int | None = Query(default=None),
        entity_version: int | None = Query(default=None),
        session: str | None = Query(default=None),
        source: str | None = Query(default=None),
        date_from: str | None = Query(default=None),
        date_to: str | None = Query(default=None),
        limit: int | None = Query(default=None),
        offset: int = Query(default=0),
    ) -> dict:
        """A listing with slices and pagination (§7.2), the `list_max_limit_rest` ceiling.

                    The reverse question "why did the object change" — the SAME route with
                    `entity`+`entity_id`[+`entity_version`] (§8/§13.4); no separate
                    handle. The limit ceiling and the offset sign are checked by the
                    service → a soft 422 with a hint.
        """
        try:
            return await asyncio.to_thread(
                _logs_service(request).list,
                entity=entity,
                entity_id=entity_id,
                entity_version=entity_version,
                session=session,
                source=source,
                date_from=date_from,
                date_to=date_to,
                limit=limit,
                offset=offset,
            )
        except LogValidationError as exc:
            raise _unprocessable(exc) from exc

    @rest_router.get("/logs/{log_id}")
    async def get_log(log_id: int, request: Request) -> dict:
        """Read the record whole (the body is not trimmed); none/deleted → 404 (§7.2).

                    A nonexistent/deleted record is indistinguishable from "no row" → 404
                    with the §7.3 canon hint (the service restoration — the operator).
        """
        record = await asyncio.to_thread(_logs_service(request).get, log_id)
        if "action" not in record:  # the service's soft answer: no row / deleted
            raise HTTPException(
                status_code=404,
                detail=record.get("hint", "log entry not found"),
            )
        return record

    @rest_router.put("/logs/{log_id}")
    async def update_log(
        log_id: int, payload: LogUpdate, request: Request
    ) -> dict:
        """A partial edit: 200 `{updated: true}`; 404 — no record; 422 — the shape.

                    "Not passed = keep" (sentinels); every actual edit is versioned by the
                    shared history layer (#41, §13.5). No record or nothing passed → 404
                    with the §7.3 canon hint.
        """
        try:
            result = await asyncio.to_thread(
                _logs_service(request).update,
                log_id,
                action=_log_field(payload.action),
                reason=_log_field(payload.reason),
                entity=_log_field(payload.entity),
                entity_id=_log_field(payload.entity_id),
                entity_version=_log_field(payload.entity_version),
                entity_ref=_log_field(payload.entity_ref),
                outcome=_log_field(payload.outcome),
                verified=_log_field(payload.verified),
                source=_log_field(payload.source),
                session=_log_field(payload.session),
            )
        except LogValidationError as exc:
            raise _unprocessable(exc) from exc
        if not result.get("updated"):
            raise HTTPException(
                status_code=404,
                detail=result.get("hint", "log entry not found"),
            )
        return result

    @rest_router.delete("/logs/{log_id}")
    async def delete_log(log_id: int, request: Request) -> dict:
        """Soft-delete a record: 200 `{id, deleted: true}` / 404 (§7.2).

                    The row and indexes stay alive; the agent outputs do not see the
                    record (§11); a repeated/nonexistent deletion → 404 with the §7.3
                    canon hint (the restoration — by the operator).
        """
        result = await asyncio.to_thread(
            _logs_service(request).delete, log_id
        )
        if not result.get("deleted"):
            raise HTTPException(
                status_code=404,
                detail=result.get("hint", "log entry not found"),
            )
        return result

    # --- version history: the operator surface (#41, release 3.2.0) -----------
    # §7.2: the version list, reading a version, rollback and statistics. Rollback — REST
    # only (the model is not given it, §13.6); with `HISTORY_ENABLED=false` a read serves
    # empty with a hint (§8), not an error. THE ROUTE ORDER: the static
    # `/history/stats` is declared BEFORE `/history/{entity}/{entity_id}` — otherwise
    # "stats" would fall into {entity}.

    @rest_router.get("/history/stats")
    async def get_history_stats(request: Request) -> dict:
        """The history's size: version counters per entity + the total (FR-10.3)."""
        return await asyncio.to_thread(history.history_stats, settings)

    @rest_router.get("/history/{entity}/{entity_id}")
    async def list_history(
        entity: str,
        entity_id: int,
        request: Request,
        limit: int | None = Query(default=None, ge=1, le=50),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        """A record's version list without bodies: metadata + "what changed" (FR-5.1).

                    The answer shape: `{entity, id, versions: [{version, created_at,
                    event, source, session, changed}], total, has_more, next_offset}`;
                    the page — per the lsb-0013 contract. An unknown entity → 404; an
                    empty history — a valid answer (including with the layer off — with
                    the §8 hint).
        """
        _check_history_entity(entity)
        return await asyncio.to_thread(
            history.read_versions,
            entity,
            entity_id,
            settings,
            limit=limit,
            offset=offset,
        )

    @rest_router.get("/history/{entity}/{entity_id}/{version}")
    async def read_history_version(
        entity: str, entity_id: int, version: int, request: Request
    ) -> dict:
        """A concrete version's content: metadata + reconstructed fields.

                    The answer shape: `{entity, id, version, created_at, event, source,
                    session, changed, fields: {…}}`; no such version/record → 404
                    (FR-5.2); the layer throws no exceptions. With the history layer off —
                    empty + the "history is off" hint (§8), not an error.
        """
        _check_history_entity(entity)
        result = await asyncio.to_thread(
            history.read_version, entity, entity_id, version, settings
        )
        if "fields" not in result and result.get("hint") != history.HINT_HISTORY_OFF:
            raise HTTPException(
                status_code=404, detail=result.get("hint", "version not found")
            )
        return result

    @rest_router.post("/history/{entity}/{entity_id}/rollback")
    async def rollback_history(
        entity: str, entity_id: int, payload: HistoryRollback, request: Request
    ) -> dict:
        """Roll back to a former version through the regular edit path (§7.3, §13.6).

                    The target version is read by `read_version` (none → 404); its fields
                    are applied by the entity's regular edit under
                    `bind_origin("rest", …, rollback_to=N)` — so the notes' FTS triggers
                    fire by themselves, and re-vectorization goes through the same code
                    (`vector_status='pending'` + a vector drop). The rollback writes a NEW
                    version (`event='rollback'`, `restore_from=N`) and does not rewrite
                    the history. `text` is passed to a note whole — the guarantee of
                    re-vectorization on a `namespace` rollback. The services' soft
                    failures → 404/422.
        """
        _check_history_entity(entity)
        target = await asyncio.to_thread(
            history.read_version, entity, entity_id, payload.version, settings
        )
        if "fields" not in target:
            raise HTTPException(
                status_code=404, detail=target.get("hint", "version not found")
            )
        live = await asyncio.to_thread(
            history.entity_fields, entity, entity_id, settings
        )
        if live is None:
            raise HTTPException(status_code=404, detail="record not found")
        fields = target["fields"]
        services = _services(request)
        session_id = request.headers.get("X-LSB-Session")

        def _apply() -> None:
            """The entity's regular edit under the rollback origin (in a worker thread)."""
            with history.bind_origin(
                "rest", session=session_id, rollback_to=payload.version
            ):
                if entity == history.ENTITY_NOTE:
                    outcome = services.notes.update(
                        entity_id,
                        text=fields["text"],
                        title=fields["title"],
                        namespace=fields["namespace"],
                        expires_at=fields["expires_at"],
                    )
                    if not outcome.get("updated"):
                        raise HTTPException(
                            status_code=404,
                            detail=outcome.get("hint", "note not found"),
                        )
                    return
                if entity == history.ENTITY_SKILL:
                    outcome = _skills_service(request).save(
                        id=entity_id,
                        name=fields["name"],
                        description=fields["description"],
                        steps=fields["steps"],
                        text=fields["text"],
                        example=fields["example"],
                        extra=fields["extra"],
                    )
                    if not outcome.get("updated"):
                        raise HTTPException(
                            status_code=404,
                            detail=outcome.get("hint", "skill not found"),
                        )
                    return
                if entity == history.ENTITY_TERM:
                    outcome = _terms_service(request).update(
                        entity_id,
                        fields["term"],
                        fields["context"],
                        fields["definition"],
                    )
                    if not outcome.get("updated"):
                        raise HTTPException(
                            status_code=404,
                            detail=outcome.get("hint", "term not found"),
                        )
                    return
                if entity == history.ENTITY_USER_FACT:
                    outcome = _user_facts_service(request).update(
                        entity_id, name=fields["name"], body=fields["body"]
                    )
                    if not outcome.get("changed"):
                        raise HTTPException(
                            status_code=404,
                            detail=outcome.get("hint", "user fact not found"),
                        )
                    return
                # skill_template — the global template (id 0)
                _skills_service(request).set_instruction_template(
                    fields["instruction_template"]
                )

        try:
            await asyncio.to_thread(_apply)
        except (
            NoteValidationError,
            NamespaceError,
            NamespaceValidationError,
            SkillValidationError,
            TermValidationError,
            UserFactValidationError,
        ) as exc:
            raise _unprocessable(exc) from exc
        head = await asyncio.to_thread(
            history.read_versions, entity, entity_id, settings, limit=1
        )
        versions = head.get("versions") or []
        return {
            "entity": entity,
            "id": entity_id,
            "version": versions[0]["version"] if versions else payload.version,
            "restored_from": payload.version,
            "revectorized": True,
        }

    # --- warnings: the unified surface (#12, no12-02, §7.1) -------------------
    # One reading point per system: the same filters as MCP's `warnings_list`,
    # but the FULL contract (no MCP trim) and the `WARNINGS_MAX_LIMIT_REST` ceiling.
    # Reads — SQL only, no model (§13.4); clearing — by a human.

    @rest_router.get("/warnings")
    async def list_warnings(
        request: Request,
        kind: str | None = Query(default=None),
        subject_type: str | None = Query(default=None),
        subject_id: int | None = Query(default=None),
        status: str = Query(default="open"),
        limit: int | None = Query(
            default=None, ge=1, le=settings.warnings_max_limit_rest
        ),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        """The operator's warnings listing: the full contract + the page fields.

                    The surface ceiling is checked by both the schema (`le`) and the
                    service (a single failure text; the lsb-0013 precedent); the default
                    limit — `DEFAULT_LIST_LIMIT`.
        """
        try:
            return await asyncio.to_thread(
                _warnings_service(request).list,
                kind=kind,
                subject_type=subject_type,
                subject_id=subject_id,
                status=status,
                limit=(
                    settings.default_list_limit if limit is None else limit
                ),
                offset=offset,
                max_limit=settings.warnings_max_limit_rest,
            )
        except WarningsValidationError as exc:
            raise _unprocessable(exc) from exc

    @rest_router.post("/warnings/{warning_id}/dismiss")
    async def dismiss_warning(warning_id: int, request: Request) -> dict:
        """Clear a warning by a human: `{id, status, resolved_at}`.

                    No such record / already cleared — 404 (the service's soft answer
                    leads to it).
        """
        result = await asyncio.to_thread(
            _warnings_service(request).dismiss, warning_id
        )
        if not result.get("dismissed"):
            raise HTTPException(
                status_code=404,
                detail=result.get("hint", "warning not found"),
            )
        return {
            "id": result["id"],
            "status": result["status"],
            "resolved_at": result["resolved_at"],
        }

    # --- the job queue: the "needs attention" state (#39, no39-02) ------------
    # §6.8/§7.4: tasks frozen by a pair's repeating non-transient failure
    # are visible to the operator and return to work without data edits. SQL
    # only (no models called); task attention — the worker's zone, not memory
    # content (we write nothing into `warnings` or the warnings surface).

    @rest_router.get("/jobs/attention")
    async def list_attention_jobs(request: Request) -> dict:
        """The list of tasks waiting for a human: `{jobs: [...], total: N}`.

                    The record shape — `{job_id, job, slot, note_id, older_id, outcome,
                    attempts, since}` (arch §7.4); assembled by the worker; SQL only.
        """
        worker = request.app.state.worker  # type: ignore[attr-defined]
        jobs = await asyncio.to_thread(worker.attention_jobs)
        return {"jobs": jobs, "total": len(jobs)}

    @rest_router.post("/jobs/attention/{job_id}/retry")
    async def retry_attention_job(job_id: int, request: Request) -> dict:
        """The operator's return of a task to work: `{job_id, released, attempts}`.

                    It clears attention and resets the pair counter (the work stays
                    `pending` and is taken by the loop from the next run). No such task /
                    it is not in attention → 404.
        """
        worker = request.app.state.worker  # type: ignore[attr-defined]
        result = await asyncio.to_thread(worker.release_attention, job_id)
        if not result.get("released"):
            raise HTTPException(
                status_code=404, detail="job is not waiting for attention"
            )
        return result

    # --- self-migration: aborting (LSBN-F-22, M3, design §4) ------------------
    # The operator handle "the migration can be aborted without losses" (the brief
    # p. 2): the service stays on the previous database; nothing is deleted. Not in MCP —
    # structure/migration is governed by the operator (§13.6, like /namespaces).

    @rest_router.post("/migration/abort")
    async def abort_migration_route(request: Request) -> dict:
        """Abort the self-migration: stop the splitting job, reverse-pour the delta
                    into the previous database (parts excluded), swap back, phase=aborted.

                    `{aborted: True, phase, prev_file, counters}` — aborted by this call;
                    `{aborted: False, hint}` — a soft failure (no migration / already
                    aborted / no splitting queue — a return "up" completes at startup).
                    A pour/swap failure — 500: the active database is intact (the
                    MigrationError contract); the call can be repeated (reverse_pour is
                    idempotent; phase=aborted is already set — the job stands down).
        """
        try:
            return await asyncio.to_thread(abort_migration, settings)
        except (MigrationError, StorageError, OSError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"migration abort failed — service keeps running on "
                f"the current base: {exc}",
            ) from exc

    # --- backups: observability and restore verification (#14, arch §7.2) -----
    # Operator handles: the snapshot list and on-demand verification. Reading
    # the directory/snapshot — without writing to the battle DB (arch §5.1); the outcomes
    # themselves live in the sidecar index next to the copies.

    @rest_router.get("/backups")
    async def list_backups(request: Request) -> dict:
        """The snapshot list for the operator: `{items: [...], total}` (newest first).

                    An item — `{file, size_bytes, created_at, verified, bad, shipped}`;
                    `verified` ∈ ok/failed/unknown (§7.2). A directory scan; we do not
                    open the DB.
        """
        items = await asyncio.to_thread(_backup_service(request).list_snapshots)
        return {"items": items, "total": len(items)}

    @rest_router.post("/backups/verify")
    async def verify_backup(payload: BackupVerifyRequest, request: Request) -> dict:
        """Verify a snapshot on demand (FR-2.2): the `verify()` result as is.

                    No/empty `file` → the newest; no such file → 404 "no such snapshot".
        """
        service = _backup_service(request)
        if payload.file:
            target = await asyncio.to_thread(service.resolve, payload.file)
        else:
            target = await asyncio.to_thread(service.newest_snapshot)
        if target is None:
            raise HTTPException(status_code=404, detail="no such snapshot")
        return await asyncio.to_thread(service.verify, target)

    return rest_router
