"""Schema self-migration, phase 4 — the splitting job (LSBN-F-22, M2) and M3:
    the union read context, abort, the /health reconciliation.

    Oversized notes (migration_notes 'waiting', M1) are split by the model into
    self-sufficient parts as they are processed — "processed, became available"
    (the F-22 brief, p. 1). The source texts live in the PREVIOUS database (the
    union read — M3): the job reads them over the legacy path from
    `migration.state.prev_file`, checks the TTL and calls `SummaryService.split`
    (the `split` operation — SPLIT_* env, a separate splitting slot, the default is
    local) and `NoteService.save_migration_parts` — one transaction per note:
    parts + history source="migration_split" + the 'split' status + new_ids.

    M3 (design §4): a process-level `MigrationContext` (legacy_path|None, the
    holder lives here, the default None) — search/notes/dedup read the previous
    database over a second connection while phase ∈ {pour, split};
    `abort_migration` — stop the job (phase=aborted in the active database),
    reverse-pour the delta into the previous one (parts are excluded), swap back —
    the service stays on the previous database; phase 7: zero the splitting queue →
    phase=done; `migration_health` — reconciliation counters (old_total/poured —
    NOTES, the pilot's decision) for /health.

    Statuses (design F-22 §3/§4):
    - TTL expired (or the source vanished from the previous database) → 'expired' —
      the model is not called;
    - parts saved → 'split' (set by save_migration_parts);
    - `MergeIncompleteError` (the model loses content; a retry would give the same
      two rounds) → 'quarantine' + a warnings record, a retry on back-off;
    - transient (`SummaryError`) — the status unchanged, the batch is released: the
      model is unavailable for everyone else, a retry on the loop's back-off
      (NFR-3).

    The retry economy of 2026-11-12 (the live failure: the judge's budget bug
    sent every verification to OFF_CONTRACT and the queue re-split the same
    notes every 10 minutes for days, burning the generation slot):
    - the split result is CACHED in migration_notes.parts_json (keyed by the
      source's sha1): a retry after a transient judge failure verifies the SAME
      parts — the split model is not called again;
    - the #39 semantics: a transient failure (unavailable/timeout/auth) does NOT
      eat the attempt budget — waiting is normal, the job's back-off paces it; a
      non-transient one (empty/truncated/off_contract) does;
    - the attempt ceiling = SLOT_MAX_ATTEMPTS: exhausted, the note moves to the
      ATTENTION state (migration_notes.attention=1, the status 'карантин') — the
      job stops taking it (the queue stays open: phase=done does NOT fire while
      attentioned notes remain), the operator decides in warnings; nothing is
      lost — the source stays whole in the previous database;
    - the mechanical garbage floor (SPLIT_FACTS_FLOOR): below it the parts are
      a loss, not a retelling — the note is quarantined WITHOUT burning the
      judge's call.

    The job registers on the lsb-0014 framework (a builder + a line in
    `jobs.JOB_BUILDERS`, the warnings/terms_normalize precedent): the "by
    interval" form, the summary slot — the splitting slot's breaker is visible in
    /health and gates the job like the others (the gate in the framework, 3.4.1,
    LSBN-T-29: on a breaker the batch is not taken). No summarizer (test DI
    builds) — the job is not applicable. No migration in progress (no
    'waiting'/'quarantine') — the run is empty and cheap (SQL only), the queue is
    visible in /health.queues.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.config import Settings
from app.services import jobs
from app.services.judge import JudgeError
from app.services.slot_outcome import SlotOutcome, outcome_of
from app.services.summary import (
    MERGE_MAX_ROUNDS,
    MergeIncompleteError,
    SPLIT_FACTS_FLOOR,
    SummaryError,
    facts_coverage,
)
from app.storage.db import StorageError, note_chars_limit, session, transaction
from app.storage.migration import (
    MIGRATION_NOTES_TABLE,
    SOURCE_MIGRATION_SPLIT,
    MigrationError,
    ensure_migration_schema,
    free_away_name,
    read_state,
    reverse_pour,
    set_state_phase,
    sibling_db_path,
    swap,
)

#: Phases with the union read enabled (design §4: phases 3–7; after
#: zeroing — done; after an abort — aborted — both disable the union).
_UNION_PHASES = ("pour", "split")

if TYPE_CHECKING:  # annotations only — the module stays clean
    from app.services.notes import NoteService
    from app.services.summary import MergeResult, Summarizer
    from app.services.warnings import WarningsService
    from app.services.worker import BackgroundWorker

MIGRATION_JOB = "migration"

#: The splitting quarantine (design F-22 §4): an open kind=`split_deferred` record
#: on the source (upsert — re-deferring moves updated_at, does not
#: multiply rows; the merge_deferred precedent of F-23). Visible to the operator in
#: warnings; with the splitting model unavailable the service works (the brief p. 4).
SPLIT_DEFERRED_WARNING = "split_deferred"

#: Statuses the job takes into work: 'waiting' — primary, 'quarantine'
#: — a retry on back-off (design §4: "quarantine … a retry on back-off").
_TAKEN_STATUSES = ("ожидает", "карантин")

#: The #39 outcome classes (mirrors worker._NON_TRANSIENT_OUTCOMES): a transient
#: failure (unavailable/timeout/auth) means "wait" — it does NOT eat the attempt
#: budget, the job's back-off paces the retries; a non-transient one
#: (empty/truncated/off_contract) means "the note is hopeless as it is" — it
#: counts, and on the SLOT_MAX_ATTEMPTS ceiling the note moves to attention.
_NON_TRANSIENT_OUTCOMES = frozenset(
    {SlotOutcome.EMPTY, SlotOutcome.TRUNCATED, SlotOutcome.OFF_CONTRACT}
)

logger = logging.getLogger("app")


# --- union read context (M3, design §4) -------------------------------------


@dataclass(frozen=True)
class MigrationContext:
    """The process-level union context: where the previous database is and the migration phase.

            The holder — the `_CONTEXT` variable below, the default None: without a
            migration the consumers (search/notes/dedup) work exactly over the old
            paths — zero overhead (no second connection, no extra SQL). The context is
            set by `refresh_migration_context` (lifespan after phases 0–3, the job at
            phase=done, abort after the swap) — the consumers only read.
    """

    legacy_path: str  # the previous database's absolute path (migration.state.prev_file)
    phase: str  # pour|split — the read surfaces do not distinguish by it
    # The id range reserve (design §8.1): the lower bound of NEW ids
    # (= MAX(id) of the previous database + 1, migration.state.id_floor). NoteService._insert
    # issues new notes/parts ids no lower — no collisions with the old_ids
    # of waiting sources (shadowing in the union, losses on abort). 0 — a state
    # without the key (a migration started before id_floor existed): max+1 as before.
    id_floor: int = 0


_CONTEXT: MigrationContext | None = None


def migration_context() -> MigrationContext | None:
    """The process's current union context; None — no migration / disabled."""
    return _CONTEXT


def refresh_migration_context(settings: Settings) -> None:
    """Sync the context with the active database's migration state.

            Enables the union (phase ∈ {pour, split} and prev_file exists) or disables
            it (no state, done/aborted, the file vanished). Called from lifespan after
            phases 0–3 (including on resume — a restart in the middle of a migration
            raises the union without a new pour), by the job at phase=done, and by
            abort after the swap. Idempotent; a read failure — the union is off.
            No database file — the union is disabled WITHOUT opening a connection:
            sqlite3.connect creates an empty file, a side effect a read must not have
            (a fresh database before init_db).
    """
    global _CONTEXT
    _CONTEXT = None
    if not Path(settings.db_path).is_file():
        return
    try:
        with session(settings) as conn:
            state = read_state(conn)
    except (sqlite3.Error, StorageError):
        return
    if not state or str(state.get("phase")) not in _UNION_PHASES:
        return
    prev_file = str(state.get("prev_file") or "")
    if not prev_file:
        return
    candidate = Path(settings.db_path).parent / prev_file
    if not candidate.is_file():
        return
    _CONTEXT = MigrationContext(
        legacy_path=str(candidate),
        phase=str(state["phase"]),
        id_floor=int(state.get("id_floor") or 0),
    )


def legacy_visible_ids(settings: Settings) -> set[int]:
    """Ids of notes whose only live copy is the previous database.

            The 'waiting'/'quarantine' statuses (the text is still in the previous
            database); 'split' and 'expired' are NOT included — the source was replaced
            by parts or closed, the union does not show it (design §4: "excluding the
            ids of 'split'/'expired'"). Shadowed ids are excluded: the active database
            issues new notes max(id)+1, which may coincide with a waiting source's id —
            such a note in the active database is ITS OWN, not a source (otherwise the
            union would stitch different notes by one id). No migration_notes table
            (no migration happened) — an empty set.
    """
    try:
        with session(settings) as conn:
            rows = conn.execute(
                f"SELECT old_id FROM {MIGRATION_NOTES_TABLE} "
                "WHERE status IN ('ожидает', 'карантин') "
                "AND old_id NOT IN (SELECT id FROM notes)"
            ).fetchall()
    except (sqlite3.Error, StorageError):
        return set()
    return {int(row[0]) for row in rows}


def _now() -> str:
    """The current time in the DDL format ('%Y-%m-%dT%H:%M:%SZ') — the TTL comparison.

            The format matches notes.expires_at (lsb-0004-02) — a lexicographic string
            comparison is correct for one format.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class MigrationService:
    """Phase 4 of self-migration: splitting 'waiting' notes into parts ≤ the target limit."""

    def __init__(
        self,
        settings: Settings,
        *,
        summarizer: "Summarizer",
        notes: "NoteService",
        warnings: "WarningsService | None" = None,
        judge: "Judge | None" = None,
    ) -> None:
        self._settings = settings
        self._summarizer = summarizer
        self._notes = notes
        # The warnings surface (the quarantine is visible to the operator, the brief
        # p. 4); None — test DI builds: only the journal event remains
        # (the _defer_merge_pair precedent).
        self._warnings = warnings
        # The split verdict (Oleg's decision 11.11.2026): the source and ALL new
        # fact-notes go to the judge; without a verdict the split is not accepted.
        # None - test DI builds (the check is skipped, the unit tests cover it).
        self._judge = judge

    # --- queue (for /health.queues and the loop) -------------------------------

    def queue_stat(self) -> dict[str, int | None]:
        """The splitting queue snapshot (SQL only): pending + the oldest's age.

                    The age — over `updated_at` of the taken status (the stamp is moved by
                    splitting and quarantine alike) — the same "age over pending+waiting"
                    semantics (3.3.1). No migration_notes table (no migration happened) —
                    the queue is empty. `attention` — the notes frozen after
                    SLOT_MAX_ATTEMPTS non-transient attempts: not taken, not pending,
                    visible (the #39 attention precedent).
        """
        try:
            with session(self._settings) as conn:
                ensure_migration_schema(conn)
                row = conn.execute(
                    f"SELECT COUNT(*) AS pending, "
                    f"CAST(strftime('%s','now') AS INTEGER) - "
                    f"CAST(strftime('%s', MIN(updated_at)) AS INTEGER) AS age, "
                    f"COALESCE(SUM(attention), 0) AS attention "
                    f"FROM {MIGRATION_NOTES_TABLE} WHERE status IN (?, ?) "
                    f"AND attention = 0",
                    _TAKEN_STATUSES,
                ).fetchone()
        except (sqlite3.Error, StorageError):
            return {"pending": 0, "oldest_pending_sec": None, "attention": 0}
        return {
            "pending": int(row["pending"]),
            "oldest_pending_sec": row["age"],
            "attention": int(row["attention"] or 0),
        }

    def queue_empty(self) -> bool:
        """Is the splitting queue empty (a cheap queue recheck, pool 6)?

                    Attentioned notes COUNT as work: the queue is not "empty" while a
                    frozen note waits for the operator — otherwise phase=done would
                    fire and the union read would hide the note's source from the
                    service (the note is not in the active base yet).
        """
        stat = self.queue_stat()
        return (
            int(stat["pending"] or 0) == 0 and int(stat["attention"] or 0) == 0
        )

    # --- phase 4: the splitting batch -----------------------------------------

    def process_batch(self, limit: int | None = None) -> int:
        """Split a batch of 'waiting' notes (lowest id first); the processed count.

                    Processed = a changed status ('split'/'expired'/'quarantine'); a
                    transient failure changes no status and releases the batch (return what
                    is already done; an empty run grows the loop's back-off — the model is
                    unavailable, the brief p. 4: the service works, splitting waits).
        """
        # 1) Is the migration active and where is the previous database (design §3: state).
        # phase=done/aborted — the job stands down (M3): after an abort and after
        # zeroing there is nothing to split and it must not be done (the service is already on the previous database).
        state = self._read_state()
        if state is None or str(state.get("phase")) not in _UNION_PHASES:
            return 0
        legacy_path = self._legacy_path(state)
        if legacy_path is None:
            return 0
        # The 12.11.2026 columns (parts_json/attention) appear on a base that
        # started migrating before them — idempotent ALTER before the selects.
        with session(self._settings) as conn:
            ensure_migration_schema(conn)
        rows = self._taken_rows(limit)
        if not rows:
            # Phase 7 (M3): the splitting queue is empty — the reconciliation is zeroed,
            # phase=done (the union is disabled; the state stays — migration_notes
            # is the rollback map). Attentioned notes are NOT "empty": the phase
            # stays open while they wait for the operator. Idempotent: the next
            # run is an idle one.
            if self._attention_count() == 0:
                self._close_if_drained(state)
            else:
                logger.info(
                    "migration: split queue holds attentioned notes — "
                    "phase stays open for the operator",
                    extra={
                        "event": "migration_attention_pending",
                        "job": MIGRATION_JOB,
                        "attention": self._attention_count(),
                    },
                )
            return 0
        # DEF-7a: splitting has started — the phase is visible in state and /health (design §3:
        # pour — the startup pour only, split — splitting; done/aborted — M3).
        self._mark_split_phase(state)
        done = 0
        legacy_settings = self._settings.model_copy(update={"db_path": legacy_path})
        with session(self._settings) as active, session(legacy_settings) as legacy:
            for row in rows:
                old_id = int(row["old_id"])
                note = legacy.execute(
                    "SELECT id, text, namespace, expires_at, deleted_at FROM notes "
                    "WHERE id = ?",
                    (old_id,),
                ).fetchone()
                if note is None or note["text"] is None:
                    # No source (the operator deleted it in the previous database): nothing
                    # to split — 'expired' closes the note without the model.
                    self._mark(
                        active, old_id, "expired", reason="исходника нет в прежней базе"
                    )
                    done += 1
                    continue
                if note["deleted_at"] is not None:
                    # A note from the trash (live case 2026-11-11: the queue carried
                    # five trashed notes and one of them was split into 75 fragments).
                    # Splitting the trash spends the slot and pollutes nothing useful:
                    # the note is closed, the union will not show it anyway.
                    self._mark(
                        active, old_id, "expired", reason="заметка в корзине"
                    )
                    done += 1
                    continue
                if note["expires_at"] is not None and str(
                    note["expires_at"]
                ) <= _now():
                    # TTL expired (design §4): for 'expired' the physical deletion
                    # is done by the regular cleanup job in the previous database (FR-6).
                    self._mark(active, old_id, "expired", reason="истёк TTL")
                    done += 1
                    continue
                outcome = self._split_one(
                    active,
                    old_id,
                    str(note["namespace"] or "default"),
                    str(note["text"]),
                    attempts=int(row["attempts"]),
                    cached=row["parts_json"],
                )
                if outcome == "retry-later":
                    break  # transient: the model is down — release the batch
                done += 1
        return done

    # --- internal --------------------------------------------------------------

    def _close_if_drained(self, state: dict) -> None:
        """Phase 7: the queue is empty → phase=done and the union is off (M3).

                    Called on an empty run during an active phase: 'waiting' and
                    'quarantine' are exhausted — every note is either poured 1:1, split, or
                    expired. The state is not removed (design §3/§7): migration_notes stays
                    the rollback map, the counters — in /health.
        """
        try:
            with session(self._settings) as conn, transaction(conn):
                set_state_phase(conn, "done")
        except (sqlite3.Error, StorageError):
            return  # the reconciliation survives a restart — done will be set by the next run
        refresh_migration_context(self._settings)
        logger.info(
            "migration: split queue drained — counters reconciled, phase=done",
            extra={
                "event": "migration_done",
                "job": MIGRATION_JOB,
                "target": state.get("target"),
                "counters": state.get("counters"),
            },
        )

    def _mark_split_phase(self, state: dict) -> None:
        """The splitting phase start: phase pour → split in state (DEF-7a, design §3).

                    Written once — further runs do not rewrite the phase; done (a zeroed
                    queue) and aborted (an abort, M3) — as they were. A write failure does
                    not break the batch: the next run sets the phase (the
                    _close_if_drained precedent).
        """
        if str(state.get("phase")) == "split":
            return
        try:
            with session(self._settings) as conn, transaction(conn):
                set_state_phase(conn, "split")
        except (sqlite3.Error, StorageError):
            return
        refresh_migration_context(self._settings)

    def _split_one(
        self,
        active: sqlite3.Connection,
        old_id: int,
        namespace: str,
        text: str,
        *,
        attempts: int,
        cached: str | None,
    ) -> str:
        """Split one note → the judge's verdict → save; the status outcome.

                    Returns: "done" (the status changed), "quarantined" (deferred /
                    attention — the note stays whole in the previous base),
                    "retry-later" (transient — the status unchanged, release the
                    batch).

                    The retry economy (2026-11-12): a cached split (parts_json,
                    keyed by the source's sha1) is verified as is — the split model
                    is not called again after a transient judge failure; the
                    garbage floor refuses to burn the judge on hopeless parts; the
                    attempt ceiling (SLOT_MAX_ATTEMPTS, non-transient failures
                    only — the #39 semantics) freezes the note in attention.
        """
        parts = self._cached_parts(cached, text)
        result: "MergeResult | None" = None
        if parts is None:
            try:
                result = self._summarizer.split(text)
            except MergeIncompleteError as exc:
                # F-22 design §4: splitting did not converge within MERGE_MAX_ROUNDS.
                # The cached parts (if any) belong to the failed split — clear them,
                # a retry re-splits from scratch.
                self._clear_cached(active, old_id)
                return self._defer(
                    active,
                    old_id,
                    str(exc),
                    attempts=attempts,
                    detail=str(exc)[:300],
                    rounds=int(getattr(exc, "rounds", 0)) or None,
                    meta=getattr(exc, "meta", None),
                    event="migration_split_deferred",
                )
            except SummaryError as exc:
                return self._slot_failure(active, old_id, exc, attempts)
            parts = [(part.title, part.text) for part in result.parts]
            self._store_cached(active, old_id, text, parts)
        # The mechanical garbage floor: below it the parts are a loss, not a
        # retelling — quarantine WITHOUT burning the judge's call.
        coverage = facts_coverage(text, [part_text for _title, part_text in parts])
        if coverage < SPLIT_FACTS_FLOOR:
            self._clear_cached(active, old_id)
            return self._defer(
                active,
                old_id,
                f"покрытие фактов {coverage:.2f} ниже порога {SPLIT_FACTS_FLOOR:.2f} "
                "— части потеряли факты источника",
                attempts=attempts,
                detail=f"facts coverage {coverage:.2f} < {SPLIT_FACTS_FLOOR:.2f}",
                event="migration_split_garbage",
            )
        # The judge's verdict (Oleg's decision 11.11.2026): the source and the new
        # notes go to the judge - "was everything decomposed, nothing missed"
        # (batched segment by segment since 12.11.2026). A refusal is a quarantine;
        # an unavailable judge is a transient failure — the cached parts stay,
        # the retry verifies them WITHOUT a new split.
        if self._judge is not None:
            try:
                verdict = self._judge.verify_split(
                    text, [part_text for _t, part_text in parts]
                )
            except JudgeError as exc:
                return self._slot_failure(active, old_id, exc, attempts)
            if verdict is not None:
                self._clear_cached(active, old_id)
                return self._defer(
                    active,
                    old_id,
                    f"судья: {verdict}",
                    attempts=attempts,
                    detail=verdict[:300],
                    parts=len(parts),
                    event="migration_split_rejected",
                )
            self._log(
                "migration: judge confirmed the split",
                "migration_split_verified",
                old_id=old_id,
                parts=len(parts),
            )
        # Observability (DEF-6): a slot_call for EVERY splitting call — like
        # merge/dedup (worker); the 2nd-round follow-up is visible as a separate
        # line. A cached split reuses the saved parts — the original call was
        # journaled by the run that produced it.
        if result is not None:
            self._log_slot_call(old_id, result)
        outcome = self._notes.save_migration_parts(old_id, namespace, parts)
        self._log(
            "migration: note split into self-contained parts",
            "migration_split",
            old_id=old_id,
            rounds=result.rounds if result is not None else 1,
            model=result.meta.model if result is not None and result.meta else None,
            parts=len(outcome["parts"]),
            part_ids=[int(part["id"]) for part in outcome["parts"]],
        )
        return "done"

    # --- the retry economy (2026-11-12) ---------------------------------------

    def _defer(
        self,
        active: sqlite3.Connection,
        old_id: int,
        reason: str,
        *,
        attempts: int,
        event: str,
        **log_fields: Any,
    ) -> str:
        """A non-transient failure: quarantine — or attention at the ceiling.

                    `attempts` — the count BEFORE this failure (the row's value at
                    the take). On the SLOT_MAX_ATTEMPTS ceiling the note moves to
                    attention: the status stays 'карантин', the attention flag
                    takes it out of the taken set, the warnings record tells the
                    operator. Nothing is lost: the source is whole in the previous
                    base.
        """
        if attempts + 1 >= self._settings.slot_max_attempts:
            self._to_attention(active, old_id, reason)
            return "quarantined"
        self._mark(
            active, old_id, "карантин", reason=reason[:500], bump_attempts=True
        )
        self._log(
            "migration: split deferred — note quarantined",
            event,
            old_id=old_id,
            **log_fields,
        )
        self._open_warning(old_id, reason)
        return "quarantined"

    def _slot_failure(
        self,
        active: sqlite3.Connection,
        old_id: int,
        exc: SummaryError,
        attempts: int,
    ) -> str:
        """A split/verdict call failure, classified by the #39 outcome.

                    Transient (unavailable/timeout/auth): the status and the
                    attempt budget stay — waiting is normal, the loop's back-off
                    paces the retry; only the queue's age stamp moves. Non-transient
                    (empty/truncated/off_contract): the note eats the attempt —
                    the `_defer` quarantine/attention mechanics.
        """
        outcome = outcome_of(exc)
        if outcome in _NON_TRANSIENT_OUTCOMES:
            return self._defer(
                active,
                old_id,
                str(exc),
                attempts=attempts,
                detail=str(exc)[:300],
                meta=getattr(exc, "meta", None),
                event="migration_split_failed",
            )
        self._touch(active, old_id)
        self._log(
            "migration: split failed — note stays pending, retry by back-off",
            "migration_split_failed",
            old_id=old_id,
            outcome=outcome.value,
            detail=str(exc)[:300],
            meta=getattr(exc, "meta", None),
        )
        return "retry-later"

    def _to_attention(
        self, active: sqlite3.Connection, old_id: int, reason: str
    ) -> None:
        """Freeze the note: the SLOT_MAX_ATTEMPTS ceiling is reached (#39).

                    The attention flag takes the note out of the taken set (the job
                    stops burning the slots on it), the status stays 'карантин',
                    the cached parts are dropped (a human will re-decide), the
                    warnings record carries the reason.
        """
        active.execute(
            f"UPDATE {MIGRATION_NOTES_TABLE} "
            f"SET attention = 1, status = 'карантин', reason = ?, "
            f"attempts = attempts + 1, parts_json = NULL, "
            f"updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
            f"WHERE old_id = ?",
            (reason[:500], old_id),
        )
        self._log(
            "migration: split attempts exhausted — note moved to attention",
            "migration_split_attention",
            old_id=old_id,
            attempts=self._settings.slot_max_attempts,
            detail=reason[:300],
        )
        self._open_warning(old_id, f"требует внимания: {reason}")

    @staticmethod
    def _source_hash(text: str) -> str:
        """The cache key: the source's sha1 (a source edit invalidates the cache)."""
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _cached_parts(
        self, cached: str | None, text: str
    ) -> list[tuple[str, str]] | None:
        """The cached split parts, None when absent/stale/malformed."""
        if not cached:
            return None
        try:
            data = json.loads(cached)
        except (TypeError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        if data.get("source_sha1") != self._source_hash(text):
            return None
        raw_parts = data.get("parts")
        if not isinstance(raw_parts, list) or not raw_parts:
            return None
        parts: list[tuple[str, str]] = []
        for item in raw_parts:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not all(isinstance(value, str) for value in item)
            ):
                return None
            parts.append((item[0], item[1]))
        return parts

    @staticmethod
    def _store_cached(
        active: sqlite3.Connection,
        old_id: int,
        text: str,
        parts: list[tuple[str, str]],
    ) -> None:
        """Remember the split result: a judge retry verifies THESE parts."""
        payload = {
            "source_sha1": MigrationService._source_hash(text),
            "parts": [[title, part_text] for title, part_text in parts],
        }
        active.execute(
            f"UPDATE {MIGRATION_NOTES_TABLE} "
            f"SET parts_json = ?, "
            f"updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
            f"WHERE old_id = ?",
            (json.dumps(payload, ensure_ascii=False), old_id),
        )

    @staticmethod
    def _clear_cached(active: sqlite3.Connection, old_id: int) -> None:
        """Drop the cached parts (a reject/attention: the next try re-splits)."""
        active.execute(
            f"UPDATE {MIGRATION_NOTES_TABLE} SET parts_json = NULL WHERE old_id = ?",
            (old_id,),
        )

    @staticmethod
    def _touch(active: sqlite3.Connection, old_id: int) -> None:
        """Move the queue age stamp WITHOUT eating the attempt budget (#39)."""
        active.execute(
            f"UPDATE {MIGRATION_NOTES_TABLE} "
            f"SET updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
            f"WHERE old_id = ?",
            (old_id,),
        )

    def _attention_count(self) -> int:
        """The frozen notes (SQL only); no table — 0."""
        try:
            with session(self._settings) as conn:
                ensure_migration_schema(conn)
                row = conn.execute(
                    f"SELECT COUNT(*) FROM {MIGRATION_NOTES_TABLE} "
                    f"WHERE attention = 1"
                ).fetchone()
        except (sqlite3.Error, StorageError):
            return 0
        return int(row[0])

    def _read_state(self) -> dict | None:
        """The active database's migration state; None — nothing to split."""
        try:
            with session(self._settings) as conn:
                return read_state(conn)
        except (sqlite3.Error, StorageError):
            return None

    def _legacy_path(self, state: dict) -> str | None:
        """The previous database's path from state (prev_file); none — a warning and an idle run.

                    After the state is removed (phase 7, M3) or for a never-migrated
                    database prev_file is absent — that is normal (the job is simply not
                    needed).
        """
        prev_file = state.get("prev_file")
        if not prev_file:
            return None
        candidate = Path(self._settings.db_path).parent / str(prev_file)
        if not candidate.is_file():
            logger.warning(
                "migration: legacy file from state is missing — split paused",
                extra={
                    "event": "migration_legacy_missing",
                    "job": MIGRATION_JOB,
                    "file": str(prev_file),
                },
            )
            return None
        return str(candidate)

    def _taken_rows(self, limit: int | None) -> list[Any]:
        """Minimal ids of the takeable statuses ('waiting' + quarantine on
                    back-off); the attentioned notes are NOT taken — the operator's."""
        query = (
            f"SELECT old_id, status, attempts, parts_json "
            f"FROM {MIGRATION_NOTES_TABLE} "
            f"WHERE status IN (?, ?) AND attention = 0 ORDER BY old_id"
        )
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        try:
            with session(self._settings) as conn:
                return conn.execute(query, _TAKEN_STATUSES).fetchall()
        except (sqlite3.Error, StorageError):
            return []  # no migration_notes table — no migration happened

    @staticmethod
    def _mark(
        conn: sqlite3.Connection,
        old_id: int,
        status: str,
        *,
        reason: str | None = None,
        bump_attempts: bool = False,
    ) -> None:
        """Change a migration record's status (inside the caller's session)."""
        conn.execute(
            f"UPDATE {MIGRATION_NOTES_TABLE} "
            f"SET status = ?, reason = ?, attempts = attempts + ?, "
            f"updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
            f"WHERE old_id = ?",
            (status, reason, 1 if bump_attempts else 0, old_id),
        )

    def _open_warning(self, old_id: int, reason: str) -> None:
        """The quarantine is visible to the operator (the brief p. 4); None — journal only."""
        if self._warnings is None:
            return
        try:
            self._warnings.open(
                SPLIT_DEFERRED_WARNING,
                "note",
                old_id,
                "migration split incomplete after "
                f"{MERGE_MAX_ROUNDS} rounds — note quarantined, kept in "
                "the legacy base",
                {"reason": reason[:500]},
            )
        except Exception:  # noqa: BLE001 — a warning does not break the batch
            logger.warning(
                "migration: failed to record split_deferred warning",
                extra={
                    "event": "warning_open_failed",
                    "job": MIGRATION_JOB,
                    "old_id": old_id,
                },
            )

    def _log(self, message: str, event: str, **fields: Any) -> None:
        """The job's event in the shared journal: the mandatory `job` field (FR-1.4).

                    The call metadata (#49/#50) goes as flat `meta.model`/`meta.provider`
                    fields — like the worker's slot_call; rounds/parts/part_ids — splitting
                    observability (the brief: "split by which model" — in the journal and
                    the report).
        """
        meta = fields.pop("meta", None)
        payload: dict[str, Any] = {"event": event, "job": MIGRATION_JOB}
        for key, value in fields.items():
            if value is not None:
                payload[key] = value
        if meta is not None:
            for key in ("model", "provider", "source"):
                value = getattr(meta, key, None)
                if value is not None:
                    payload[key] = value
        logger.info(message, extra=payload)

    def _log_slot_call(self, old_id: int, result: MergeResult) -> None:
        """A slot_call per splitting call (DEF-6; the merge precedent — worker).

                    The same successful-call fields as the worker's (`_call_log_fields`):
                    the operation/model/budget/duration — "how long it took" is visible per
                    line per call; without metadata (a test double) — silent. A local
                    import: the worker is heavy and drags half the system — migration must
                    not import it at the top (the lazy-import precedent of notes).
        """
        from app.services.worker import SLOT_SUMMARY, _call_log_fields

        for meta in result.metas or [result.meta]:
            if meta is None:
                continue
            payload: dict[str, Any] = {
                "event": "slot_call",
                "slot": SLOT_SUMMARY,
                "job": MIGRATION_JOB,
                "note_id": old_id,
                **_call_log_fields(meta),
            }
            logger.info("slot_call", extra=payload)


# --- abort (M3, design §4 "Aborting") ---------------------------------------


#: Soft failures of POST /migration/abort: nothing to abort / already aborted.
ABORT_HINT_NONE = (
    "no migration to abort: the base is not mid-migration "
    "(phase is done/aborted or there is no migration state)"
)
ABORT_HINT_ALREADY = (
    "migration is already aborted — the service is on the previous base"
)
ABORT_HINT_NOTHING_TO_SPLIT = (
    "nothing to abort: the split queue is empty (up-migrations complete "
    "at startup, nothing waits for the model)"
)


def abort_migration(settings: Settings) -> dict:
    """Abort the migration without losses: the service stays on the previous database.

            The order (design §4): 1) "stop the job" — phase=aborted in the active
            database, the next splitting batch takes zero notes (the running batch only
            manages to write parts into the outgoing database — they are excluded from
            the pour, risk design 5: short sessions + the phase-1 snapshot);
            2) reverse-pour the active database's delta into the previous one — an
            upsert of UPDATE-else-INSERT (REPLACE breaks the external content's FTS),
            parts (record_versions source='migration_split') and their dependencies
            are excluded — "ids outside the parts mapping"; 3) swap: the active (new)
            database moves under its CHECK name, the previous one returns to db_path —
            NOT deleted; 4) phase=aborted in the restored database (set by
            reverse_pour) and the union context is off. A repeated call after a failure
            between (1) and (3) resumes the abort: reverse_pour is idempotent (upsert).

            Returns: {"aborted": True, ...} — aborted by this call; a soft failure
            {"aborted": False, "hint": ...} — nothing to abort / already aborted.
            Aborting applies to a "downward" migration (there is a splitting queue): a
            return "upward" completes synchronously at startup — nothing to abort.
    """
    with session(settings) as conn:
        state = read_state(conn)
        live = note_chars_limit(conn)
    if state is None or str(state.get("phase")) not in (*_UNION_PHASES, "aborted"):
        return {"aborted": False, "hint": ABORT_HINT_NONE}
    prev_name = str(state.get("prev_file") or "")
    if not prev_name or prev_name == Path(settings.db_path).name:
        # Already on the previous database (the swap after abort happened): prev_file
        # points at the outgoing database, i.e. at db_path itself.
        return {"aborted": False, "hint": ABORT_HINT_ALREADY}
    legacy_path = str(Path(settings.db_path).parent / prev_name)
    if not Path(legacy_path).is_file():
        raise MigrationError(f"файл прежней базы не найден: {prev_name}")
    if str(state.get("phase")) in _UNION_PHASES:
        pending = _pending_count(settings)
        if pending == 0:
            # A return "upward"/a zeroed queue: splitting is not needed, there is nothing
            # to pour back (everything is already in both databases) — leave it alone.
            return {"aborted": False, "hint": ABORT_HINT_NOTHING_TO_SPLIT}
        # 1) Stop the job: phase=aborted BEFORE the pour (the job sees it on the next
        # batch; the union read still works — the previous database is intact until the swap).
        with session(settings) as conn, transaction(conn):
            set_state_phase(conn, "aborted")
    # 2) The active database's delta → the previous one (parts excluded); target = the
    # receiver's CHECK (prev_limit): the schema.note_chars stamp stays its own.
    # exclude — all the migration's old_ids: the active database may issue a new note
    # the source's id (max+1); an upsert would overwrite the previous database's original;
    # such a note remains only in the outgoing file (it is not deleted).
    counters = reverse_pour(
        settings,
        source_path=legacy_path,
        away_file=str(settings.db_path),
        target=int(state.get("prev_limit") or live or 0),
        reason="миграция прервана оператором",
        phase="aborted",
        exclude_note_ids=_migration_note_ids(settings),
    )
    # 3) Swap back: the new database moves under its CHECK name (not deleted),
    # the previous one returns to db_path — the following session() calls open it.
    # The name is decided BEFORE reverse_pour (LSBN-D-12): it goes into state and
    # the receiver's chain; if taken by a foreign file — the .away suffix.
    swap(
        settings,
        away_file=free_away_name(
            settings, sibling_db_path(settings.db_path, int(live or 0))
        ),
        incoming_file=legacy_path,
    )
    # 4) The restored database's state — phase=aborted (reverse_pour);
    # the union read is disabled.
    refresh_migration_context(settings)
    logger.info(
        "migration aborted — service stays on the previous base",
        extra={
            "event": "migration_aborted",
            "job": MIGRATION_JOB,
            "prev_file": prev_name,
            **counters,
        },
    )
    return {
        "aborted": True,
        "phase": "aborted",
        "prev_file": prev_name,
        "counters": counters,
    }


def _pending_count(settings: Settings) -> int:
    """The number of notes in the splitting queue ('waiting' + 'quarantine')."""
    try:
        with session(settings) as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM {MIGRATION_NOTES_TABLE} "
                "WHERE status IN ('ожидает', 'карантин')"
            ).fetchone()
    except (sqlite3.Error, StorageError):
        return 0
    return int(row[0])


def _migration_note_ids(settings: Settings) -> set[int]:
    """All old_ids of the migration (any status) — no upsert into the previous database.

            For the abort's reverse-pour: the source id in the previous database is
            taken by its own row, and the active database may have issued it to a new
            note (sqlite: max(id)+1) — an upsert by such an id would overwrite the
            original.
    """
    try:
        with session(settings) as conn:
            rows = conn.execute(
                f"SELECT old_id FROM {MIGRATION_NOTES_TABLE}"
            ).fetchall()
    except (sqlite3.Error, StorageError):
        return set()
    return {int(row[0]) for row in rows}


# --- phase 7: the /health reconciliation (M3, design §4/§7) ------------------


def migration_health(settings: Settings) -> dict:
    """The `migration` block for /health: phase, direction, reason, invariant.

            The counters are NOTES (the pilot's decision): old_total — the previous
            database's notes (a live COUNT — the reconciliation database), poured —
            poured 1:1 at the start (recorded in state.counters by the pour; after an
            abort — everything that came back whole), split_notes/parts/quarantine/
            expired/waiting — the live migration_notes statuses and the parts history.
            The reconciliation invariant:
            old_total = poured + split_notes + waiting + quarantine + expired
            (a mismatch = the previous database was edited bypassing the migration —
            the operator deleted a source); after an abort parts=0 and the queue is
            empty. `slot` — the splitting slot delimiter (3.4.1, LSBN-T-27): the batch
            and the pause from the process settings — the operator sees how much is
            taken per run and how long the job sleeps. Without migration state —
            {"phase": None} (slot stays: the values are configuration properties, not
            state).

            The direction (DEF-7b, LSBN-D-11): `direction: forward|reverse`. The
            reverse-pour state (a return upward — phase=done; an abort — aborted) is
            written by `reverse_pour` with the `kept_parts` caption in counters; in
            this mode the forward pour's `poured` counter does not exist —
            `poured_rows` (rows of ALL the delta's tables) and `kept_parts` (parts left
            unpoured in the outgoing database) are shown instead, and the forward
            invariant is not computed (`invariant_ok=None`): row and note numbers are
            incomparable, the old output read as a migration error. For an abort the
            old view is preserved ("everything came back whole").
    """
    slot = {
        "batch": int(settings.migration_batch),
        "pause_sec": int(settings.migration_pause_sec),
    }
    try:
        with session(settings) as conn:
            state = read_state(conn)
            statuses = _status_counts(conn)
            parts = _part_count(conn)
    except (sqlite3.Error, StorageError):
        return {"phase": None, "slot": slot}
    if state is None:
        return {"phase": None, "slot": slot}
    phase = str(state.get("phase"))
    counters = state.get("counters", {})
    waiting = statuses.get("ожидает", 0)
    split_notes = statuses.get("нарезана", 0)
    # The frozen notes are counted APART from the quarantine (the invariant
    # closes over both): quarantine retries on back-off, attention waits for
    # the operator.
    attention = _attention_count_rows(conn)
    quarantine = statuses.get("карантин", 0) - attention
    expired = statuses.get("expired", 0)
    old_total = _legacy_total(settings, state, phase)
    # reverse-pour is the only writer of `kept_parts`; the forward pour does not
    # own this field (pour/split/done downward — old_total/pending).
    direction = "reverse" if "kept_parts" in counters else "forward"
    if old_total is None:
        poured = None
        invariant_ok = None
    elif phase == "aborted":
        # After the abort everything came back whole: the previous database is full again,
        # state.counters from reverse_pour (poured rows) — not about notes.
        poured = old_total
        invariant_ok = (
            old_total
            == poured + split_notes + waiting + quarantine + expired + attention
        )
    elif direction == "reverse":
        # A return upward (phase=done): the forward pour's poured was never written,
        # the forward invariant does not apply to these numbers (DEF-7b).
        poured = None
        invariant_ok = None
    else:
        poured = int(counters.get("poured", 0))
        invariant_ok = (
            old_total
            == poured + split_notes + waiting + quarantine + expired + attention
        )
    block: dict = {
        "phase": phase,
        "target": state.get("target"),
        "reason": state.get("reason"),
        "slot": slot,
        "counters": {
            "old_total": old_total,
            "poured": poured,
            "split_notes": split_notes,
            "parts": parts,
            "quarantine": quarantine,
            "expired": expired,
            "waiting": waiting,
            "attention": attention,
        },
        "invariant_ok": invariant_ok,
    }
    if direction == "reverse":
        # Reverse-mode fields: the delta's rows (all tables) and the unpoured
        # parts; the forward poured does not exist in this state.
        block["direction"] = "reverse"
        block["counters"]["poured_rows"] = counters.get("poured")
        block["counters"]["kept_parts"] = counters.get("kept_parts")
    else:
        block["direction"] = "forward"
    return block


def _status_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """migration_notes statuses → {status: count}; no table — empty."""
    try:
        rows = conn.execute(
            f"SELECT status, COUNT(*) AS n FROM {MIGRATION_NOTES_TABLE} "
            "GROUP BY status"
        ).fetchall()
    except sqlite3.Error:
        return {}
    return {str(row["status"]): int(row["n"]) for row in rows}


def _attention_count_rows(conn: sqlite3.Connection) -> int:
    """The frozen notes (migration_notes.attention=1); no table/column — 0."""
    try:
        row = conn.execute(
            f"SELECT COUNT(*) FROM {MIGRATION_NOTES_TABLE} WHERE attention = 1"
        ).fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0])


def _part_count(conn: sqlite3.Connection) -> int:
    """The number of split parts in the active database (the record_versions source)."""
    try:
        row = conn.execute(
            "SELECT COUNT(DISTINCT entity_id) FROM record_versions "
            "WHERE entity = 'note' AND source = ?",
            (SOURCE_MIGRATION_SPLIT,),
        ).fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0])


def _legacy_total(settings: Settings, state: dict, phase: str) -> int | None:
    """The live note count of the reconciliation's source database; no file — None.

            During the migration (pour/split/done) the previous database sits nearby
            under the state.prev_file name; after an abort it IS the active one
            (prev_file points at the outgoing database) — count db_path.
    """
    prev_file = str(state.get("prev_file") or "")
    if phase == "aborted":
        target_path = settings.db_path
    elif prev_file and prev_file != Path(settings.db_path).name:
        candidate = Path(settings.db_path).parent / prev_file
        if not candidate.is_file():
            return None
        target_path = str(candidate)
    else:
        return None
    try:
        with session(
            settings.model_copy(update={"db_path": target_path})
        ) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0])
    except (sqlite3.Error, StorageError):
        return None


def build_migration_job(
    worker: "BackgroundWorker", settings: Settings
) -> jobs.JobSpec | None:
    """The `migration` job builder: without a summarizer — not applicable.

            The "by interval" form (no event: 'waiting' notes appear only at the very
            start of a migration — M1); the run goes into `to_thread`. The slot —
            `summary` (splitting inherits its slot when SPLIT_* are unset). The slot
            gate — like the others (3.4.1, LSBN-T-29): `gate` — the worker's
            `_slot_gate` (with a probe knock after the pause expires), `is_gated` —
            `_slot_blocked` (queue snapshots without models); before, the job declared
            a slot but kept poking the dead slot on a breaker — now the framework does
            not take the batch.

            The splitting slot delimiter (3.4.1, LSBN-T-27):
            a batch is `MIGRATION_BATCH` notes, after a run the loop sleeps
            `MIGRATION_PAUSE_SEC` (pause_after_run), then takes the next one —
            splitting does not monopolize the generation server. Own numbers, NOT
            foreign meanings: `embedding_batch_size` — the chunks of one /api/embed
            vectorization request, `pending_retry_sec` — the starting retry interval
            of pre-processing. Both values are visible in /health.migration.slot and in
            the startup log (`startup`); tuned by an override without a release.
    """
    if worker._summarizer is None:  # a test mode without a summarizer
        return None
    service = MigrationService(
        settings,
        summarizer=worker._summarizer,
        notes=worker._notes,
        warnings=worker._warnings,
        judge=worker._judge,
    )

    async def process() -> int:
        # Exactly the delimiter's batch per run (LSBN-T-27): not all 'waiting'
        # at once — the next run comes after the slot's pause.
        return await asyncio.to_thread(
            service.process_batch, settings.migration_batch
        )

    return jobs.JobSpec(
        name=MIGRATION_JOB,
        queue=MIGRATION_JOB,
        interval_sec=settings.migration_pause_sec,
        batch=settings.migration_batch,
        enabled=True,
        process=process,
        queue_empty=service.queue_empty,
        wait_event=None,
        idle_hook=None,
        queue_stat=service.queue_stat,
        slot="summary",
        pause_after_run=True,
        gate=lambda: worker._slot_gate("summary", MIGRATION_JOB),
        is_gated=lambda: worker._slot_blocked("summary"),
    )


# Registering the job in the framework registry (FR-1.1/FR-1.6): one's own job — in one's own
# module (one line in `JOB_BUILDERS`); the import is one-way — migration → jobs.
jobs.JOB_BUILDERS += (build_migration_job,)
