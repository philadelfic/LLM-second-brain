"""The `warnings` job and its state (#12, release 3.2.0; arch §5.1/§6.5/§13.5).

    The background `warnings` job walks over **placed** notes (a title + a ready
    summary + structure), compares the actual `namespace` with the classifier's
    markup and **highlights** the divergence — WITHOUT an automatic move
    (the 2026-10-08 decision, FR-2.2/§13.5). The result is a row of the shared
    `warnings` surface (`app/storage/db.py`), readable without model calls (the
    surface — no12-02).

    The state (the `warnings` table) is served by `WarningsService`:

    * `open` — an upsert by the open key `(subject_type, subject_id, kind)`: no
      more than one OPEN warning (a partial unique index, §5.1);
      a repeated run moves `message`/`payload`/`updated_at` instead of multiplying
      rows (idempotency, #12 FR-4.4/§13.7);
    * `resolve` — the reason is gone (auto): `resolved_at=now`, `status='open'`
      stays;
    * `resolve_for_subject` — clearing all of a subject's warnings on a note edit
      (#12 FR-4.4); called in THE SAME transaction through the narrow injection
      `NoteService.set_warnings_resolver`;
    * `dismiss` — cleared by a human (`status='dismissed'`, REST — no12-02);
    * `list`/`counts` — SQL-only reads, no models (FR-3.2, §13.4);
    * `check_note`/`run` — the job's run with a classifier call budget.

    The job registers on the lsb-0014 framework (a builder + a line in
    `JOB_BUILDERS`; the import is one-way — warnings → jobs, no cycle) and
    declares the `summary` slot (the same model as the `nodes` job); the slot
    breaker gates it with the framework's gate (3.4.1, LSBN-T-29). A fresh
    `default` is left to the `nodes` job (§13.5): two jobs do not argue over one
    node.

    Release 3.9.0 (LSBN-S-9): the job works the `classify` operation — in the
    `operation` mode its breaker key is the OPERATION (`classify`), not the
    `summary` slot group, and the failure of the classifier goes to that
    breaker. The `legacy` mode keeps the slot group and the journal-only
    failure (as before).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from app.config import Settings
from app.services import jobs
from app.services.classifier import ClassificationError
from app.services.jobs import JobSpec, queue_snapshot
from app.services.listing import page_fields
from app.services.slot_outcome import SlotOutcome, outcome_of
from app.storage.db import session, transaction

if TYPE_CHECKING:  # the service has no runtime dependencies (DI)
    from app.services.namespaces import NamespaceService
    from app.services.worker import BackgroundWorker

# The job's name in the journal and the queue for `/health.queues`.
WARNINGS_JOB = "warnings"

# The warning kinds dictionary — one for all surfaces (#12 + #32).
# The last two are introduced by #32, but the values are fixed here.
KIND_NOTE_MISPLACED = "note_misplaced"
KIND_NOTE_NO_NODE = "note_no_node"
KIND_TERM_CONTEXTS_EXCESS = "term_contexts_excess"
KIND_TERM_CONTEXT_CLUSTER = "term_context_cluster"

# The message texts are impersonal, English (no people's names and no word
# "operator"): assembled at open time, read without a join to the texts.
MESSAGE_MISPLACED = (
    "note {id} is in '{namespace}' but the classifier places it in "
    "'{hint_path}' (confidence {confidence})"
)
MESSAGE_NO_NODE = (
    "note {id}: the classifier suggests node '{hint_path}', which does not "
    "exist yet"
)

# The candidate selection (SQL, no model, arch §6.5): active notes with a ready
# non-empty summary, PLACED (`namespace != 'default'` OR already parsed by
# `nodes`), not yet checked after the last edit, newest first.
_CANDIDATE_WHERE = (
    "deleted_at IS NULL "
    "AND summary_status = 'ok' AND summary IS NOT NULL AND summary != '' "
    "AND (namespace != 'default' OR node_order_at IS NOT NULL) "
    "AND (warnings_at IS NULL OR warnings_at < updated_at)"
)
_CANDIDATE_SELECT = (
    "SELECT id, updated_at FROM notes WHERE " + _CANDIDATE_WHERE
    + " ORDER BY updated_at DESC, id DESC LIMIT ?"
)
_CANDIDATE_STAT_SELECT = (
    "SELECT COUNT(*) AS pending, "
    "CAST(strftime('%s','now') - strftime('%s', MIN(updated_at)) AS INTEGER) "
    "AS oldest_pending_sec FROM notes WHERE " + _CANDIDATE_WHERE
)

# The listing columns (`GET /warnings`, §7.1) — the answer shape and pagination in no12-02.
_LIST_COLUMNS = (
    "id, kind, subject_type, subject_id, message, payload, status, "
    "created_at, updated_at, resolved_at"
)


class WarningsValidationError(ValueError):
    """The warnings listing shape violated (`limit`/`offset`).

            A soft failure: the transport answers with a `hint` (MCP) or 422 with the
            text (REST); the precedent — lsb-0013, shared with the memory listings.
    """


class WarningsService:
    """The warnings state and the `warnings` job's run (arch §6.5).

            The dependencies are a narrow injection (`classifier`/`namespaces`): in
            unit tests the classifier is a fake; in a run — the same model of the
            `summary` slot as the `nodes` job's. `conn_factory` — the connection
            assembly point (`session` by default); with a ready `conn` passed the
            service works in a foreign transaction (a note edit, arch §6.5).
    """

    def __init__(
        self,
        settings: Settings,
        *,
        classifier: Any = None,
        namespaces: "NamespaceService | None" = None,
        conn_factory: Callable[[Settings], Any] = session,
    ) -> None:
        self._settings = settings
        self._classifier = classifier
        self._namespaces = namespaces
        self._conn = conn_factory
        # The last `check_note`'s outcome: none | opened | updated | resolved |
        # failed. The run needs it to count `warnings_run` without changing
        # the `check_note` return contract (str | None).
        self._last_check = "none"
        # Whether the last `open` was an insert (not an update) — the counter.
        self._last_open_created = False

    # --- the warnings state ----------------------------------------------------

    @contextmanager
    def _connection(self, conn: sqlite3.Connection | None) -> Iterator[Any]:
        """One's own connection with a transaction, or an already-open one (a foreign transaction)."""
        if conn is not None:
            yield conn
            return
        with self._conn(self._settings) as own, transaction(own):
            yield own

    def open(
        self,
        kind: str,
        subject_type: str,
        subject_id: int,
        message: str,
        payload: dict,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        """Open/update a warning (an upsert by the open key).

                    The `warning_opened` event is written ONLY on insert: a repeated run
                    updates the row and does not duplicate the journal (idempotency,
                    §13.7).
        """
        payload_json = json.dumps(payload or {}, ensure_ascii=False, sort_keys=True)
        with self._connection(conn) as active:
            row = active.execute(
                "SELECT id FROM warnings WHERE subject_type = ? "
                "AND subject_id = ? AND kind = ? AND status = 'open' "
                "AND resolved_at IS NULL",
                (subject_type, subject_id, kind),
            ).fetchone()
            if row is not None:
                warning_id = int(row["id"])
                active.execute(
                    "UPDATE warnings SET message = ?, payload = ?, "
                    "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                    "WHERE id = ?",
                    (message, payload_json, warning_id),
                )
                self._last_open_created = False
                return warning_id
            cursor = active.execute(
                "INSERT INTO warnings "
                "(kind, subject_type, subject_id, message, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (kind, subject_type, subject_id, message, payload_json),
            )
        warning_id = int(cursor.lastrowid)
        self._last_open_created = True
        self._log(
            "warnings: opened warning",
            "warning_opened",
            warning_id=warning_id,
            kind=kind,
            subject_type=subject_type,
            subject_id=subject_id,
        )
        return warning_id

    def resolve(self, kind: str, subject_type: str, subject_id: int) -> int:
        """Clear a warning — the reason is gone (auto); the number of cleared rows.

                    `status='open'` stays, `resolved_at=now`: a cleared one does not return
                    to the open sample and is not deleted (an audit, the `promotions`
                    precedent).
        """
        with self._connection(None) as conn:
            count = self._resolve_in(conn, subject_type, subject_id, kind=kind)
        if count:
            self._log(
                "warnings: resolved warning",
                "warning_resolved",
                kind=kind,
                subject_type=subject_type,
                subject_id=subject_id,
                resolved=count,
            )
        return count

    def resolve_for_subject(
        self,
        subject_type: str,
        subject_id: int,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        """Clear ALL of a subject's open warnings (a note edit, FR-4.4).

                    Called in THE SAME transaction as the edit (a ready `conn`).
        """
        with self._connection(conn) as active:
            count = self._resolve_in(active, subject_type, subject_id)
        if count:
            self._log(
                "warnings: resolved warnings on subject edit",
                "warning_resolved",
                subject_type=subject_type,
                subject_id=subject_id,
                resolved=count,
            )
        return count

    def _resolve_in(
        self,
        conn: sqlite3.Connection,
        subject_type: str,
        subject_id: int,
        *,
        kind: str | None = None,
    ) -> int:
        sql = (
            "UPDATE warnings SET resolved_at = "
            "strftime('%Y-%m-%dT%H:%M:%SZ','now'), updated_at = "
            "strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE subject_type = ? "
            "AND subject_id = ? AND status = 'open' AND resolved_at IS NULL"
        )
        params: list[object] = [subject_type, subject_id]
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        return int(conn.execute(sql, params).rowcount)

    def dismiss(self, warning_id: int) -> dict:
        """Clear a warning by a human; none/already cleared — a soft answer."""
        with self._connection(None) as conn:
            cursor = conn.execute(
                "UPDATE warnings SET status = 'dismissed', resolved_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now'), updated_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ? "
                "AND status = 'open' AND resolved_at IS NULL",
                (warning_id,),
            )
            if cursor.rowcount == 0:
                return {
                    "id": warning_id,
                    "dismissed": False,
                    "hint": "warning not found or already resolved/dismissed",
                }
            row = conn.execute(
                "SELECT resolved_at FROM warnings WHERE id = ?", (warning_id,)
            ).fetchone()
        self._log(
            "warnings: dismissed warning",
            "warning_dismissed",
            warning_id=warning_id,
        )
        return {
            "id": warning_id,
            "dismissed": True,
            "status": "dismissed",
            "resolved_at": row["resolved_at"],
        }

    def list(
        self,
        *,
        kind: str | None = None,
        subject_type: str | None = None,
        subject_id: int | None = None,
        status: str = "open",
        limit: int = 20,
        offset: int = 0,
        max_limit: int = 50,
    ) -> dict:
        """The warnings listing — SQL only, no models (FR-3.2, §13.4).

                    The shape is shared with the memory listings: the page fields are
                    computed by the shared `page_fields` helper (the `NoteService.list`
                    precedent). A `limit` outside 1..`max_limit` (and a negative `offset`)
                    — `WarningsValidationError`: the transport answers softly (the
                    lsb-0013 precedent) instead of silently trimming.
        """
        limit = int(limit)
        offset = int(offset)
        if not 1 <= limit <= max_limit:
            raise WarningsValidationError(
                f"limit: expected 1..{max_limit}, got {limit}"
            )
        if offset < 0:
            raise WarningsValidationError(f"offset: expected ≥ 0, got {offset}")
        where: list[str] = []
        params: list[object] = []
        for column, value in (
            ("kind", kind),
            ("subject_type", subject_type),
            ("subject_id", subject_id),
            ("status", status),
        ):
            if value is not None:
                where.append(f"{column} = ?")
                params.append(value)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        with self._connection(None) as conn:
            total = int(
                conn.execute(
                    "SELECT COUNT(*) AS n FROM warnings" + clause, params
                ).fetchone()["n"]
            )
            rows = conn.execute(
                f"SELECT {_LIST_COLUMNS} FROM warnings" + clause
                + " ORDER BY id DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        warnings = [self._list_dict(row) for row in rows]
        return {"warnings": warnings, **page_fields(total, offset, len(warnings))}

    def counts(self) -> dict:
        """The aggregates for `/health.warnings`: open per kind + total (§7.1)."""
        with self._connection(None) as conn:
            rows = conn.execute(
                "SELECT kind, COUNT(*) AS n FROM warnings WHERE status = 'open' "
                "AND resolved_at IS NULL GROUP BY kind"
            ).fetchall()
        by_kind = {row["kind"]: int(row["n"]) for row in rows}
        return {"by_kind": by_kind, "total": sum(by_kind.values())}

    @staticmethod
    def _list_dict(row: sqlite3.Row) -> dict:
        """A `warnings` row — into the answer shape (the payload parsed from JSON)."""
        data = {column: row[column] for column in _LIST_COLUMNS.split(", ")}
        try:
            data["payload"] = json.loads(data["payload"] or "{}")
        except (TypeError, ValueError):
            data["payload"] = {}
        return data

    # --- the job run -----------------------------------------------------------

    def check_note(self, note_id: int) -> str | None:
        """One run of the note check; the warning kind or None.

                    Reads `title`, `summary`, `namespace`; calls the classifier
                    (`title` + `summary`, the registry's known nodes) and decides per §6.5.
                    A classifier failure (`ClassificationError`) does NOT mark the note
                    (no marker; a retry by the next run — the #39 contract):
                    a `warnings_classifier_failed` event with the `outcome` is written.
        """
        self._last_check = "none"
        with self._conn(self._settings) as conn:
            row = conn.execute(
                "SELECT title, summary, namespace, summary_status FROM notes "
                "WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            ).fetchone()
        if row is None:
            return None
        summary = row["summary"] or ""
        if row["summary_status"] != "ok" or not summary.strip():
            return None
        if self._classifier is None:
            return None
        known = self._namespaces.list_all()["namespaces"]
        title = row["title"] or ""
        text = f"{title}\n{summary}" if title else summary
        try:
            result = self._classifier.classify(text, known)
        except ClassificationError as exc:
            self._last_check = "failed"
            self._log(
                "warnings: classifier failed — note stays unchecked, retry later",
                "warnings_classifier_failed",
                outcome=outcome_of(exc).value,
                note_id=note_id,
                detail=str(exc)[:160],
            )
            return None
        return self._decide(note_id, row["namespace"], result, known)

    def _decide(self, note_id: int, namespace: str, result: Any, known: list) -> str | None:
        """The §6.5 decision: highlight "not in place"/"no node" without a move."""
        hint = result.hint_path
        confidence = result.confidence
        if confidence < self._settings.warnings_min_confidence:
            return None  # low confidence — small noise is not highlighted
        if not hint:
            return None  # a "generic" markup does not argue with a concrete placement
        if namespace == hint:
            # The markup matched the node — the reason is gone: clear the open one.
            resolved = self.resolve(KIND_NOTE_MISPLACED, "note", note_id)
            resolved += self.resolve(KIND_NOTE_NO_NODE, "note", note_id)
            self._last_check = "resolved" if resolved else "none"
            return KIND_NOTE_MISPLACED if resolved else None
        existing = {node["path"] for node in known}
        if hint not in existing:
            self.open(
                KIND_NOTE_NO_NODE,
                "note",
                note_id,
                MESSAGE_NO_NODE.format(id=note_id, hint_path=hint),
                {"hint_path": hint, "confidence": confidence},
            )
            self._last_check = "opened" if self._last_open_created else "updated"
            return KIND_NOTE_NO_NODE
        if hint.split("/")[0] != namespace.split("/")[0]:
            self.open(
                KIND_NOTE_MISPLACED,
                "note",
                note_id,
                MESSAGE_MISPLACED.format(
                    id=note_id,
                    namespace=namespace,
                    hint_path=hint,
                    confidence=confidence,
                ),
                {
                    "namespace": namespace,
                    "hint_path": hint,
                    "confidence": confidence,
                },
            )
            self._last_check = "opened" if self._last_open_created else "updated"
            return KIND_NOTE_MISPLACED
        # A shared root: a "parent/child" divergence — a warning only
        # at the strict level (FR-3.3; the default — small noise).
        if self._settings.warnings_level_strict:
            self.open(
                KIND_NOTE_MISPLACED,
                "note",
                note_id,
                MESSAGE_MISPLACED.format(
                    id=note_id,
                    namespace=namespace,
                    hint_path=hint,
                    confidence=confidence,
                ),
                {
                    "namespace": namespace,
                    "hint_path": hint,
                    "confidence": confidence,
                },
            )
            self._last_check = "opened" if self._last_open_created else "updated"
            return KIND_NOTE_MISPLACED
        return None

    def run(self, batch: int | None = None) -> int:
        """The job's run: a batch of candidates within the classifier's budget; their count.

                    The `JOB_WARNINGS_CLASSIFIER_BUDGET` budget — model calls per run:
                    notes beyond the budget get no marker and land in the next run
                    (FR-4.2). A parsed note is marked `notes.warnings_at=now`
                    (anti-looping). There is no note moving (§13.5).
        """
        limit = self._settings.job_warnings_batch if batch is None else int(batch)
        budget = int(self._settings.job_warnings_classifier_budget)
        with self._conn(self._settings) as conn:
            rows = conn.execute(_CANDIDATE_SELECT, (limit,)).fetchall()
        checked = 0
        opened = 0
        resolved = 0
        used = 0
        for row in rows:
            if used >= budget:
                break
            kind = self.check_note(int(row["id"]))
            used += 1
            if self._last_check == "failed":
                continue  # no marker — the note is a candidate again
            self._mark_checked(int(row["id"]))
            checked += 1
            if self._last_check == "opened":
                opened += 1
            elif self._last_check == "resolved":
                resolved += 1
        self._log(
            "warnings: run finished",
            "warnings_run",
            checked=checked,
            opened=opened,
            resolved=resolved,
            budget_used=used,
        )
        return checked

    def _mark_checked(self, note_id: int) -> None:
        """Set the parsed marker: the note does not eat the budget twice."""
        with self._conn(self._settings) as conn:
            conn.execute(
                "UPDATE notes SET warnings_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ?",
                (note_id,),
            )

    def queue_stat(self) -> dict:
        """The `warnings` queue snapshot for `/health.queues` — SQL only, no model."""
        with self._conn(self._settings) as conn:
            row = conn.execute(_CANDIDATE_STAT_SELECT).fetchone()
        return queue_snapshot(row["pending"], row["oldest_pending_sec"])

    def queue_empty(self) -> bool:
        """Is the `warnings` queue empty (a cheap queue recheck, pool 6)."""
        return int(self.queue_stat()["pending"] or 0) == 0

    @staticmethod
    def _log(message: str, event: str, **fields: Any) -> None:
        """The job's event in the shared journal: `event` + the mandatory `job` field."""
        logging.getLogger("app").info(
            message, extra={"event": event, "job": WARNINGS_JOB, **fields}
        )


class _ReportingClassifier:
    """The classifier of the `warnings` job: the failure goes to the breaker.

            Release 3.9.0 (LSBN-S-9): the `warnings` job works the `classify`
            operation — the failure of the classifier is recorded at the
            breaker of the operation (its own breaker in the `operation` mode).
            The narrow point of the reporting is the call itself: the service
            sees the same classifier contract (the failure is re-raised, the
            `warnings_classifier_failed` event is written as before), and the
            outcome goes to the worker's breaker.

            `report` — the worker's `_note_classify_failure`: in the `legacy`
            mode it does nothing (the job never touched the breaker).
    """

    def __init__(
        self, classifier: Any, report: Callable[[SlotOutcome], None]
    ) -> None:
        self._classifier = classifier
        self._report = report

    def classify(self, text: str, known_nodes: list) -> Any:
        """The classifier's call — the failure is reported and re-raised."""
        try:
            return self._classifier.classify(text, known_nodes)
        except ClassificationError as exc:
            self._report(outcome_of(exc))
            raise


def build_warnings_job(
    worker: "BackgroundWorker", settings: Settings
) -> JobSpec | None:
    """The `warnings` job builder (FR-1.1): without a classifier — not applicable.

            The "by interval" form (no event, no queue recheck): the run goes into
            `to_thread`, the queue is derived from notes. The slot — `summary`
            (the same model as `nodes`); the slot gate — like the others (3.4.1,
            LSBN-T-29): `gate` — the worker's `_slot_gate` (with a probe knock),
            `is_gated` — `_slot_blocked` (queue snapshots without models). Before, the
            job declared a slot but kept poking the dead slot on a breaker — now the
            framework does not take the batch.

            Release 3.9.0 (LSBN-S-9): the job works the `classify` operation —
            the breaker key is the OPERATION in the `operation` mode (the
            `classify` breaker), the `summary` slot group in the `legacy` mode
            (as before). The gate and the blocked check read the same key, and
            the failure of the classifier goes to that breaker (the sink).
    """
    if worker._classifier is None:  # the Phase-3 test mode — there was no loop
        return None
    # The breaker key of the job: the `classify` operation in the `operation`
    # mode, the `summary` slot group in the `legacy` mode.
    key = worker._operation_slot("classify")
    service = WarningsService(
        settings,
        # The classifier with the reporting of the failure (the breaker of the
        # `classify` operation — the operation owns the failure).
        classifier=_ReportingClassifier(
            worker._classifier,
            lambda outcome: worker._note_classify_failure(WARNINGS_JOB, outcome),
        ),
        namespaces=worker._namespaces,
    )
    batch = settings.job_warnings_batch

    async def process() -> int:
        return await asyncio.to_thread(service.run, batch)

    return JobSpec(
        name=WARNINGS_JOB,
        queue=WARNINGS_JOB,
        interval_sec=settings.job_warnings_interval_sec,
        batch=batch,
        enabled=settings.job_warnings_enabled,
        process=process,
        queue_empty=service.queue_empty,
        wait_event=None,
        idle_hook=None,
        queue_stat=service.queue_stat,
        slot=key,
        gate=lambda: worker._slot_gate(key, WARNINGS_JOB),
        is_gated=lambda: worker._slot_blocked(key),
    )


# Registering the job in the framework registry (FR-1.1/FR-1.6): one's own job — in one's own
# module (one line in `JOB_BUILDERS`); the import is one-way — warnings → jobs.
jobs.JOB_BUILDERS += (build_warnings_job,)
