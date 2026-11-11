"""LogsService — the "Assistant logs" area service (LSBN-F-10, release 3.2.0).

    The record shape, the object dictionary and the id/version/ref links — arch
    §7.3; the projection and versioning — §5.4/§11; the service contract — §6.1.

    Key properties
    --------------
    * **A log entry is not knowledge** (FR-2.4/§4): the service writes only on an
      explicit call by the agent (`logs_save`) or the operator (REST, 03); there is
      no autologging, slots do not write their own calls into the area.
    * **Mandatory fields — `action` + `reason`** (§7.3 p.1): "what was done" and
      "why"; the change object, the outcome and "what was verified" are optional
      but structured. The shape is checked **before** the DB: each failure has its
      own verbatim hint (`LogValidationError`, the `TermValidationError`
      precedent), no record is written.
    * **The object dictionary is closed** (§5.4): versioned entities (`note`,
      `skill`, `term`, `user_fact`, `assistant_log`) require `entity_id` (and allow
      `entity_version`), external ones (`file`, `commit`, `tracker_task`, `config`,
      `contour`) — a non-empty `entity_ref` (a path, a commit hash, a task key, a
      config name).
    * **`source`/`session` are never invented by the model** (§7.1/§13.3): by
      default they come from `history.current_origin()` (the `bind_origin`
      middleware), explicit caller values win. In the log TABLE `source`/`session`
      are content fields ("which channel/session produced the action"), separate
      from `record_versions.source/session` (the origin of the version itself).
    * **Versions — by the same history layer** (#41, §11/§13.5): `record_initial`
      at creation, `record_change` on an actual edit, `record_event('delete')` on a
      soft delete — with the same connection and in the same transaction. The
      service opens no connections/transactions of its own for history: the caller
      holds atomicity.
    * **Embedding is NOT called on the write path**: the vector (non-empty `action`
      + `reason` + `outcome`) is caught up by the worker's `areas` loop over
      `vector_status='pending'`, and after a successful save/update/delete the
      service wakes the loop (`set_areas_notifier`).
    * **Area isolation** (§13.2/§13.4): the service's SQL addresses only
      `assistant_logs`; the change object's state (`entity_state`) is computed by
      the history layer (04), not by the logs service.
    * **Search and listing — the area's own SQL** (§13.2, f10-03): a hybrid of
      vec0-KNN + FTS5/BM25 → RRF with slices by object/session/source/absolute
      dates; `entity_state`/`resolve_entity` — 04.

    Response contracts (the MCP layer trims service fields by a whitelist — f10-02):
    - save   → {created: True, id, version: 1} | {created: False, hint}
    - get    → {id, action, reason, entity, entity_id, entity_version, entity_ref,
                outcome, verified, source, session, created_at, updated_at};
               missing/deleted → {id, hint}
    - update → {id, updated: True} | {id, updated: False, hint}
    - delete → {id, deleted: True} | {id, deleted: False, hint}
    - search → {results: [{id, action, entity, entity_id, created_at, score}],
                warning?, hint?} — `action` trimmed to an excerpt (f10-03)
    - list   → {items: [{id, action, entity, entity_id, created_at}], total,
                has_more, next_offset, next_cursor} — `action` in full (f10-03);
               only the MCP output trims it, the operator's REST gets the full text
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app import __version__
from app.config import Settings
from app.services import history
# The area's empty outcomes and the FTS-only degradation — shared constants (arch §7.3/§8:
# "reused like other areas'"), no own texts are introduced.
from app.services.areas import HINT_NO_RESULTS, HINT_SHORT_QUERY
from app.services.embedding import Embedder, EmbeddingError
# Absolute slice dates — the block-4 parser (#29, arch §8/§13.12): no own parser.
from app.services.freshness import (
    FreshnessValidationError,
    parse_updated_after,
)
from app.services.listing import page_fields
from app.services.ranking import fuse_rrf, match_expression
from app.services.search import CANDIDATE_LIMIT, MAX_TOP_K, WARNING_FTS_ONLY
from app.storage import area_vectors
from app.storage.db import (
    ASSISTANT_LOGS_FTS_COLUMNS,
    ASSISTANT_LOGS_FTS_TABLE,
    ASSISTANT_LOGS_TABLE,
)
# The area connection: `session` is imported under an alias — the name `session`
# is taken by the log's content parameter (`save`/`update`, arch §5.4/§7.1).
from app.storage.db import session as db_session
from app.storage.db import transaction

# --- Change object dictionary (arch §5.4: "a constant in the service") -------

# Versioned memory entities — addressed by id (and, if known, by version).
VERSIONED_ENTITY_KINDS = frozenset(
    {"note", "skill", "term", "user_fact", "assistant_log"}
)
# Objects outside memory — they have no id, the reference is text-only.
EXTERNAL_ENTITY_KINDS = frozenset(
    {"file", "commit", "tracker_task", "config", "contour"}
)
# The full closed `entity` dictionary.
ENTITY_KINDS = VERSIONED_ENTITY_KINDS | EXTERNAL_ENTITY_KINDS

# Record sources (§5.1): the channel the record arrived into the area by. Background
# jobs write `job:<name>` (the set is open — a prefix, not a dictionary element).
LOG_SOURCES = frozenset({"mcp", "rest", "merge", "migration"})
JOB_SOURCE_PREFIX = "job:"

# --- Soft-failure hints (arch §7.3: "verbatim constants in code") -----------

HINT_ACTION_REQUIRED = (
    "not saved: action is required — say what was done (≤500 characters)"
)
HINT_REASON_REQUIRED = (
    "not saved: reason is required — say why it was done (≤500 characters)"
)
HINT_ACTION_LIMIT = "not saved: action limit is 500 characters — shorten it"
HINT_REASON_LIMIT = "not saved: reason limit is 500 characters — shorten it"
HINT_ENTITY_UNKNOWN = (
    "not saved: unknown entity '{given}' — one of: note, skill, term, user_fact, "
    "assistant_log, file, commit, tracker_task, config, contour"
)
HINT_ENTITY_ID_REQUIRED = (
    "not saved: entity '{given}' is versioned — pass entity_id, and "
    "entity_version when the change was made to a known version"
)
HINT_ENTITY_REF_REQUIRED = (
    "not saved: entity '{given}' has no id in this memory — pass entity_ref (a "
    "path, commit hash, task key or config name)"
)
HINT_ENTITY_VERSION_NEEDS_ID = "not saved: entity_version requires entity_id"
HINT_UPDATE_EMPTY = (
    "not updated: nothing passed — pass at least one field to change"
)
HINT_NOT_FOUND = (
    "log entry not found — there is no such entry in this area; save it via "
    "logs_save(action, reason) if it is worth keeping"
)
# Date slices — absolute only (§7.3, §8/§13.12): a relative
# wording is rejected in the same soft-failure style.
HINT_DATE_ABSOLUTE = (
    "not filtered: dates are absolute only (YYYY-MM-DD); relative dates "
    "are not supported"
)
# The §7.3 canon does not describe a reversed range (it has only HINT_DATE_ABSOLUTE):
# the text follows the "not filtered: …" style and leads to fixing the bounds.
# The deviation is recorded in the f10-03 report.
HINT_DATE_RANGE = (
    "not filtered: date_from is later than date_to — swap the bounds"
)
# A journal export without period bounds makes no sense (LSBN-F-26): the agent
# can export "the whole journal" anyway — by widening the period; without at least one date
# the handle refuses, so a random call does not spawn a file with the whole history.
HINT_EXPORT_PERIOD_REQUIRED = (
    "not exported: pass at least one bound — date_from or date_to "
    "(YYYY-MM-DD, UTC)"
)

# §7.3 sets text field limits only for `action`/`reason`; for
# `entity_ref`/`outcome`/`verified` (the §6.1 contract: "(5) limits …") the texts
# follow the canonical template "not saved: <field> limit is N characters —
# shorten it". The deviation is recorded in the f10-02 report.
HINT_ENTITY_REF_LIMIT = (
    "not saved: entity_ref limit is 200 characters — shorten it"
)
HINT_OUTCOME_LIMIT = "not saved: outcome limit is 500 characters — shorten it"
HINT_VERIFIED_LIMIT = "not saved: verified limit is 200 characters — shorten it"

# An unknown record source: the §7.3 table sets no text (the source dictionary
# is described in §5.1), so the text follows the same template "not saved: …
# — one of: …" and leads to the valid values. The deviation is in the f10-02 report.
HINT_SOURCE_UNKNOWN = (
    "not saved: unknown source '{given}' — one of: mcp, rest, merge, migration, "
    "or job:<name>"
)

# The "not passed = keep" sentinel (the `memory_update`/`user_update` precedent):
# it distinguishes "the field untouched" from "the field cleared" (an empty string).
_UNSET: Any = object()


class LogValidationError(ValueError):
    """A violation of the log record shape (dictionary, links, limits, source).

            The exception text is a verbatim hint (§7.3): the MCP transport returns it
            as fail + hint, REST (03) — as 422. The record is NOT written: the
            precedents are `TermValidationError`, `SkillValidationError`,
            `UserFactValidationError`.
    """


class LogsService:
    """CRUD of the "Assistant logs" area: save, read, update, soft delete.

            DI: `settings` (the shape limits, the area flag), `embedding` (the encoder
            for search — f10-03; the write path does NOT call it). The service keeps no
            `AreaSearch` of its own (§13.2: an area's search is its own SQL); search,
            listing and slices (f10-03) stand on the same primitives
            (`match_expression`, `fuse_rrf`, `area_vectors.knn`, the threshold,
            `WARNING_FTS_ONLY`) but address only their own tables (§6.1/§8).
    """

    def __init__(
        self, settings: Settings, embedding: Embedder | None = None
    ) -> None:
        self._settings = settings
        # The embedder DI: search (03) encodes the query with the same slot as notes.
        self._embedding: Embedder | None = embedding
        # A signal to the worker (main.py): wake the areas loop as soon as an area's
        # pending record appears, instead of waiting out the grown back-off.
        self._areas_notifier: Callable[[], None] | None = None

    def set_areas_notifier(self, notifier: Callable[[], None] | None) -> None:
        """Wire the wakeup signal of the worker's `areas` loop (main.py).

                    Modeled on `SkillsService`/`TermsService`/`UserFactsService`: the
                    service is assembled before the worker, the notifier arrives after.
        """
        self._areas_notifier = notifier

    # --- writes (arch §6.1, §7.3) --------------------------------------------

    def save(
        self,
        action: str,
        reason: str,
        *,
        entity: str | None = None,
        entity_id: int | None = None,
        entity_version: int | None = None,
        entity_ref: str | None = None,
        outcome: str | None = None,
        verified: str | None = None,
        source: str | None = None,
        session: str | None = None,
    ) -> dict[str, Any]:
        """Save a log entry: "what was done, why, the object, the outcome, what was verified".

                    The order is mandatory: the shape is checked BEFORE any DB access
                    (§7.3). A violation — `LogValidationError` with a verbatim hint, no
                    record. Text fields are taken from the caller as is, but with
                    `strip()`.

                    `source`/`session` not passed → taken from `history.current_origin()`
                    (the `bind_origin` middleware, §7.1): the model does not invent the
                    source.

                    The write is atomic: INSERT (`vector_status='pending'`) and the
                    baseline version (`record_initial`) — in one transaction; after the
                    commit the service wakes the `areas` loop (the vector is caught up by
                    the worker). Embedding is not called.
        """
        form = self._checked_form(
            action=action,
            reason=reason,
            entity=entity,
            entity_id=entity_id,
            entity_version=entity_version,
            entity_ref=entity_ref,
            outcome=outcome,
            verified=verified,
        )
        resolved_source, resolved_session = self._resolved_origin(source, session)
        with db_session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "INSERT INTO assistant_logs (action, reason, entity, entity_id, "
                "entity_version, entity_ref, outcome, verified, source, session, "
                "vector_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
                (
                    form["action"],
                    form["reason"],
                    form["entity"],
                    form["entity_id"],
                    form["entity_version"],
                    form["entity_ref"],
                    form["outcome"],
                    form["verified"],
                    resolved_source,
                    resolved_session,
                ),
            )
            log_id = int(cursor.lastrowid or 0)
            # #41 (§13.5): the record's baseline — in the same transaction, the §5.4 projection.
            live = self._projection_row(conn, log_id)
            history.record_initial(
                conn,
                history.ENTITY_ASSISTANT_LOG,
                log_id,
                history.assistant_log_projection(live),
            )
        self._notify_areas_pending()
        return {"created": True, "id": log_id, "version": 1}

    def update(
        self,
        id: int,
        *,
        action: str | None = _UNSET,
        reason: str | None = _UNSET,
        entity: str | None = _UNSET,
        entity_id: int | None = _UNSET,
        entity_version: int | None = _UNSET,
        entity_ref: str | None = _UNSET,
        outcome: str | None = _UNSET,
        verified: str | None = _UNSET,
        source: str | None = _UNSET,
        session: str | None = _UNSET,
    ) -> dict[str, Any]:
        """Partially edit a record by id: "not passed — keep" (sentinels).

                    Outcomes:

                    * no field passed → `{id, updated: False, hint}` with
                      `HINT_UPDATE_EMPTY`, no versions added, the DB untouched;
                    * the record is missing / soft deleted → `HINT_NOT_FOUND` (§7.3: a
                      deleted record is not "someone else's" — it is absent for the agent);
                    * otherwise — an update of the passed fields (an empty string at the
                      object = clearing), `vector_status='pending'`, `updated_at` = now and
                      a version delta (`record_change`) in the same transaction; the object
                      links are checked against the effective (current + passed) values.

                    After a successful edit the service wakes the `areas` loop.
        """
        if all(
            value is _UNSET
            for value in (
                action,
                reason,
                entity,
                entity_id,
                entity_version,
                entity_ref,
                outcome,
                verified,
                source,
                session,
            )
        ):
            return {"id": id, "updated": False, "hint": HINT_UPDATE_EMPTY}
        changes: dict[str, Any] = {}
        if action is not _UNSET:
            changes["action"] = self._checked_action(action)
        if reason is not _UNSET:
            changes["reason"] = self._checked_reason(reason)
        if entity is not _UNSET:
            changes["entity"] = self._checked_entity(entity)
        if entity_ref is not _UNSET:
            changes["entity_ref"] = self._checked_entity_ref(entity_ref)
        if outcome is not _UNSET:
            changes["outcome"] = self._checked_outcome(outcome)
        if verified is not _UNSET:
            changes["verified"] = self._checked_verified(verified)
        if entity_id is not _UNSET:
            changes["entity_id"] = entity_id
        if entity_version is not _UNSET:
            changes["entity_version"] = entity_version
        if source is not _UNSET:
            # An empty `source` = "not passed": the column is NOT NULL, the source
            # of the call context was already written at creation.
            checked_source = self._checked_source(source)
            if checked_source is not None:
                changes["source"] = checked_source
        if session is not _UNSET:
            changes["session"] = self._text(session)
        if not changes:
            # All passed fields turned out empty (= "not passed").
            return {"id": id, "updated": False, "hint": HINT_UPDATE_EMPTY}
        with db_session(self._settings) as conn, transaction(conn):
            row = conn.execute(
                "SELECT entity, entity_id, entity_version, entity_ref "
                "FROM assistant_logs WHERE id = ? AND deleted_at IS NULL",
                (id,),
            ).fetchone()
            if row is None:
                return {"id": id, "updated": False, "hint": HINT_NOT_FOUND}
            if any(
                field is not _UNSET
                for field in (entity, entity_id, entity_version, entity_ref)
            ):
                # Object links — by effective values: an edit of one field
                # must not leave the row in an inconsistent state.
                self._checked_entity_links(
                    changes.get("entity", row["entity"]),
                    changes.get("entity_id", row["entity_id"]),
                    changes.get("entity_version", row["entity_version"]),
                    changes.get("entity_ref", row["entity_ref"]),
                )
            assignments = ", ".join(f"{column} = ?" for column in changes)
            conn.execute(
                f"UPDATE assistant_logs SET {assignments}, vector_status = "
                "'pending', updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (*changes.values(), id),
            )
            # #41 (§13.5): the edit — a version delta in the same transaction.
            live = self._projection_row(conn, id)
            history.record_change(
                conn,
                history.ENTITY_ASSISTANT_LOG,
                id,
                history.assistant_log_projection(live),
            )
        self._notify_areas_pending()
        return {"id": id, "updated": True}

    # --- reads (arch §6.1) ----------------------------------------------------

    def get(self, id: int) -> dict[str, Any]:
        """Read a record by id — the full §7.1 contract (the body is not trimmed).

                    No record or soft deleted → `{id, hint}` of the verbatim §7.3 canon.
                    `entity_state` is NOT computed here: the object state is the history
                    layer (04, §13.4); the area service does not walk foreign tables.
        """
        with db_session(self._settings) as conn:
            row = conn.execute(
                "SELECT id, action, reason, entity, entity_id, entity_version, "
                "entity_ref, outcome, verified, source, session, created_at, "
                "updated_at FROM assistant_logs "
                "WHERE id = ? AND deleted_at IS NULL",
                (id,),
            ).fetchone()
        if row is None:
            return {"id": id, "hint": HINT_NOT_FOUND}
        record: dict[str, Any] = {
            "id": int(row["id"]),
            "action": row["action"],
            "reason": row["reason"],
            "entity": row["entity"],
            "entity_id": row["entity_id"],
            "entity_version": row["entity_version"],
            "entity_ref": row["entity_ref"],
            "outcome": row["outcome"],
            "verified": row["verified"],
            "source": row["source"],
            "session": row["session"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        if row["entity"] is not None:
            # The referenced object's state (f10-04, §11): the field appears ONLY
            # when `entity` is set — a record without a reference has no state.
            record.update(self._entity_state_field(row["entity"], row["entity_id"]))
        return record

    # --- referenced object state (f10-04, arch §11/§13.4) ---------------------

    def _entity_state(
        self, entity: str | None, entity_id: int | None
    ) -> str | None:
        """The referenced object's state — computed by the history layer (arch §11, §13.4).

                    `"active"`/`"deleted"`/`"unavailable"` — versioned memory entities;
                    `"external"` — non-versioned and unknown (including no reference at
                    all). `None` — the state is not completed: the history layer is off
                    (`HISTORY_ENABLED=false`) or the resolve failed; a resolve failure does
                    NOT crash the output (§11). The area service does no raw SQL over
                    foreign tables — only the history layer's API (§4, area isolation).
        """
        try:
            resolved = history.resolve_entity(entity, entity_id or 0, self._settings)
        except Exception:  # noqa: BLE001 — a resolve failure does not crash the output (§11)
            return None
        if resolved is not None:
            return resolved["state"]
        # A non-versioned/unknown entity, or the history layer is off.
        return "external" if self._settings.history_enabled else None

    def _entity_state_field(
        self, entity: str | None, entity_id: int | None
    ) -> dict[str, str]:
        """The `entity_state` field for an output item; `{}` — when not completed."""
        state = self._entity_state(entity, entity_id)
        return {} if state is None else {"entity_state": state}

    # --- search and slices (arch §6.1/§8, f10-03) -----------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        entity: str | None = None,
        entity_id: int | None = None,
        entity_version: int | None = None,
        session: str | None = None,
        source: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        """Hybrid search over the area's records with slices (arch §6.1/§8).

                    Vector side — KNN over `assistant_logs_vec` (top `CANDIDATE_LIMIT`, the
                    `settings.score_threshold` gate); FTS side — BM25 over
                    `assistant_logs_fts`; merge — `fuse_rrf` (the shared `settings.rrf_k`),
                    id deduplication. **The area's own SQL** (§13.2): `AreaSearch` cannot
                    slice, and extending the shared helper would touch foreign areas.

                    Slices (§8) — by object (`entity`/`entity_id`/`entity_version`),
                    session, source and absolute dates; the same predicates go into both
                    `WHERE`s (FTS — directly in SQL, vector — post-filtering in
                    `_filter_rows`). An empty query — a soft `HINT_SHORT_QUERY`, an empty
                    result — `HINT_NO_RESULTS`; an encoder failure does not break the
                    search: FTS-only + `WARNING_FTS_ONLY` (NFR-3). `action` in the output
                    is trimmed to `logs_search_excerpt_chars` (FR-2.3): the body is read
                    by id.
        """
        top_k = self._settings.default_top_k if top_k is None else top_k
        if not 1 <= top_k <= MAX_TOP_K:
            raise LogValidationError(
                f"top_k: expected 1..{MAX_TOP_K}, got {top_k}"
            )
        where_sql, params = self._slice_where(
            entity=entity,
            entity_id=entity_id,
            entity_version=entity_version,
            session=session,
            source=source,
            date_from=date_from,
            date_to=date_to,
        )
        stripped = query.strip() if isinstance(query, str) else ""
        if not stripped:
            # An empty query: nothing to search — a soft answer, no embedder.
            return {"results": [], "warning": None, "hint": HINT_SHORT_QUERY}
        query_vector = self._query_vector(stripped)
        expression = match_expression(stripped)
        with db_session(self._settings) as conn:
            rankings: list[list[int]] = []
            if query_vector is not None:
                hits = area_vectors.knn(
                    conn,
                    area_vectors.ASSISTANT_LOGS_VEC_TABLE,
                    area_vectors.ASSISTANT_LOGS_VEC_ID_COLUMN,
                    query_vector,
                    CANDIDATE_LIMIT,
                )
                # The cosine threshold gate (the notes/areas precedent): without it
                # "empty" would become unreachable on any non-empty table.
                hits = [
                    (row_id, cosine)
                    for row_id, cosine in hits
                    if cosine >= self._settings.score_threshold
                ]
                rankings.append([row_id for row_id, _cosine in hits])
            if expression:
                rankings.append(
                    [
                        int(row["id"])
                        for row in self._fts_candidates(
                            conn, expression, where_sql, params
                        )
                    ]
                )
            scores = fuse_rrf(rankings, self._settings.rrf_k)
            # A soft read + the same slice: deleted records vanish, foreign slices
            # are cut off on the vector side too (vec0 knows no filters).
            rows = self._filter_rows(conn, list(scores), where_sql, params)
        results = self._merged_search_results(scores, rows)[:top_k]
        warning = None if query_vector is not None else WARNING_FTS_ONLY
        if not results:
            hint = (
                HINT_SHORT_QUERY
                if query_vector is None and expression is None
                else HINT_NO_RESULTS
            )
            return {"results": [], "warning": warning, "hint": hint}
        return {"results": results, "warning": warning}

    def list(
        self,
        *,
        entity: str | None = None,
        entity_id: int | None = None,
        entity_version: int | None = None,
        session: str | None = None,
        source: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        max_limit: int | None = None,
    ) -> dict[str, Any]:
        """Listing with slices: newest first, a page per the lsb-0013 contract.

                    Order — `created_at DESC, id DESC` (the
                    `idx_assistant_logs_entity_active` index); `total` — the number of
                    active records with THE SAME slice; page fields — `listing.page_fields`
                    (`next_cursor` is reserved for a future keyset). The `limit` ceiling —
                    `max_limit` (None → `settings.list_max_limit_rest`: MCP passes its own,
                    smaller ceiling); outside `1..max_limit` and `offset < 0` — a soft
                    `LogValidationError` failure (the `SkillsService.list` precedent). The
                    `action` body is not trimmed: the operator needs the full text, MCP
                    trims on its side.
        """
        ceiling = (
            self._settings.list_max_limit_rest if max_limit is None
            else max_limit
        )
        limit = self._settings.default_list_limit if limit is None else limit
        if not 1 <= limit <= ceiling:
            raise LogValidationError(
                f"limit: expected 1..{ceiling}, got {limit}"
            )
        if offset < 0:
            raise LogValidationError(
                f"offset: expected 0 or greater, got {offset}"
            )
        where_sql, params = self._slice_where(
            entity=entity,
            entity_id=entity_id,
            entity_version=entity_version,
            session=session,
            source=source,
            date_from=date_from,
            date_to=date_to,
        )
        with db_session(self._settings) as conn:
            total = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM assistant_logs t WHERE {where_sql}",
                    params,
                ).fetchone()[0]
            )
            rows = conn.execute(
                f"SELECT t.id, t.action, t.entity, t.entity_id, t.created_at "
                f"FROM assistant_logs t WHERE {where_sql} "
                "ORDER BY t.created_at DESC, t.id DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        items = [
            {
                "id": int(row["id"]),
                "action": row["action"],
                "entity": row["entity"],
                "entity_id": row["entity_id"],
                "created_at": row["created_at"],
                **self._entity_state_field(row["entity"], row["entity_id"]),
            }
            for row in rows
        ]
        return {"items": items, **page_fields(total, offset, len(items))}

    # --- period export (LSBN-F-26, release 3.5.0) -----------------------------

    def export(
        self,
        *,
        entity: str | None = None,
        entity_id: int | None = None,
        entity_version: int | None = None,
        session: str | None = None,
        source: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        collected_by: str = "logs_export",
    ) -> dict[str, Any]:
        """A ready-made journal digest for a period as a FILE into `<db dir>/exports/`.

                    The release gate assembles the "what, when, verified how" digest with
                    one handle instead of manual assembly from `logs_search`. The slices
                    are the same as `search`/`list`; at least one of `date_from`/`date_to`
                    is required (without a period a random call would produce a file with
                    the whole history). Order — oldest to newest: the journal is read
                    chronologically. Full field texts without excerpt trimming (that is the
                    export's purpose); grouping by days, empty fields — "—". Read-only:
                    only SELECT and a file write — memory, vectors and the journal itself
                    are untouched, no models are called. The timezone is UTC, as in
                    `logs_get`.
        """
        if date_from is None and date_to is None:
            raise LogValidationError(HINT_EXPORT_PERIOD_REQUIRED)
        where_sql, params = self._slice_where(
            entity=entity,
            entity_id=entity_id,
            entity_version=entity_version,
            session=session,
            source=source,
            date_from=date_from,
            date_to=date_to,
        )
        rows = self._fetch_projection(where_sql, params)
        slice_map: dict[str, Any] = {
            "date_from": self._checked_date(date_from),
            "date_to": self._checked_date(date_to),
            "entity": self._checked_entity(entity),
            "entity_id": entity_id,
            "entity_version": entity_version,
            "session": self._text(session),
            "source": self._text(source),
        }
        path = self._write_export_file(
            rows, slice_map=slice_map, collected_by=collected_by
        )
        return {"file": path.name, "path": str(path), "records": len(rows), **slice_map}

    def _fetch_projection(
        self, where_sql: str, params: list[Any]
    ) -> list[dict[str, Any]]:
        """The full projection of a slice in one query, oldest to newest.

                    A separate read layer (not N+1 from `logs_get`): the MCP output trims
                    the text, the gate digest needs it in full (§7.1). Rows come out as
                    projection dicts right away — without `entity_state` (the object state
                    is the history layer; the gate digest does not need it).
        """
        with db_session(self._settings) as conn:
            cursor = conn.execute(
                "SELECT id, action, reason, entity, entity_id, entity_version, "
                "entity_ref, outcome, verified, source, session, created_at "
                f"FROM assistant_logs t WHERE {where_sql} "
                "ORDER BY t.created_at ASC, t.id ASC",
                params,
            )
            return [
                {
                    "id": int(row["id"]),
                    "action": row["action"],
                    "reason": row["reason"],
                    "entity": row["entity"],
                    "entity_id": row["entity_id"],
                    "entity_version": row["entity_version"],
                    "entity_ref": row["entity_ref"],
                    "outcome": row["outcome"],
                    "verified": row["verified"],
                    "source": row["source"],
                    "session": row["session"],
                    "created_at": row["created_at"],
                }
                for row in cursor.fetchall()
            ]

    def _write_export_file(
        self,
        rows: list[dict[str, Any]],
        *,
        slice_map: dict[str, Any],
        collected_by: str,
    ) -> Path:
        """Assemble the markdown digest and write it into `<db_dir>/exports/`.

                    The file name is `logs-<from>_<to>.md` (a missing side — `all`): the
                    per-period file is visible on the host through the `./data:/data`
                    volume. The directory is created at the first export. The header
                    carries the period, the slice, the record count, the LSB version and
                    the build moment — the gate checks the coverage at one glance.
        """
        from_label = slice_map["date_from"] or "all"
        to_label = slice_map["date_to"] or "all"
        exports_dir = Path(self._settings.db_path).parent / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        path = exports_dir / f"logs-{from_label}_{to_label}.md"
        lines = self._render_export(
            rows, slice_map=slice_map, collected_by=collected_by
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _render_export(
        self,
        rows: list[dict[str, Any]],
        *,
        slice_map: dict[str, Any],
        collected_by: str,
    ) -> list[str]:
        """Lines of the markdown digest: header → grouping by days → records.

                    The format is a ready digest, not a raw list: per record — id, time,
                    the object (`entity:entity_id@version` or `entity_ref`) and the §7.1
                    fields; empty fields — "—". The timezone is UTC (§7.3), as in
                    `logs_get`.
        """
        from_label = slice_map["date_from"] or "all"
        to_label = slice_map["date_to"] or "all"
        slice_text = ", ".join(
            f"{name}={value}"
            for name, value in slice_map.items()
            if value is not None
        ) or "все записи"
        lines = [
            "# Журнал работ — выгрузка (logs_export)",
            "",
            f"- Период: {from_label}..{to_label} (UTC, включительно)",
            f"- Срез: {slice_text}",
            f"- Записей: {len(rows)}",
            f"- Версия LSB: {__version__}",
            f"- Собрано: {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} ({collected_by})",
            "",
        ]
        if not rows:
            lines.append("Записей за период нет — пустая выжимка с шапкой.")
            return lines
        current_day: str | None = None
        for row in rows:
            day = str(row["created_at"])[:10]
            if day != current_day:
                current_day = day
                lines.extend(["", f"## {day}", ""])
            lines.extend(self._render_entry(row))
        return lines

    def _render_entry(self, row: dict[str, Any]) -> list[str]:
        """One record block of the digest (no secrets — the §7.1 fields have none anyway)."""
        missing = "—"
        if row["entity"] is not None:
            object_label = f"{row['entity']}:{row['entity_id']}"
            if row["entity_version"] is not None:
                object_label += f"@{row['entity_version']}"
        else:
            object_label = str(row["entity_ref"]) if row["entity_ref"] else missing
        created = str(row["created_at"])
        lines = [
            f"### #{row['id']} · {created[11:19]} · {object_label}",
            "",
            f"- action: {row['action'] or missing}",
            f"- reason: {row['reason'] or missing}",
            f"- outcome: {row['outcome'] or missing}",
            f"- verified: {row['verified'] or missing}",
            f"- source/session: {row['source'] or missing} / {row['session'] or missing}",
            "",
        ]
        return lines

    # --- internal: slices and candidate sources (arch §8) ---------------------

    def _slice_where(
        self,
        *,
        entity: str | None,
        entity_id: int | None,
        entity_version: int | None,
        session: str | None,
        source: str | None,
        date_from: str | None,
        date_to: str | None,
    ) -> tuple[str, list[Any]]:
        """One SQL predicate of slices for `search` and `list` (arch §8).

                    Returns `("t.… AND t.…", [parameters])` — a ready `WHERE` part for the
                    `t` alias of the area table: a soft read (`deleted_at IS NULL`) plus
                    the passed slices. Filters are validated BEFORE the DB (the `entity`
                    dictionary, the `entity_version`↔`entity_id` link, absolute dates): a
                    violation — `LogValidationError` with the canon's hint.
        """
        checked_entity = self._checked_entity(entity)
        if entity_version is not None and entity_id is None:
            raise LogValidationError(HINT_ENTITY_VERSION_NEEDS_ID)
        start = self._checked_date(date_from)
        end = self._checked_date(date_to)
        if start is not None and end is not None and start > end:
            raise LogValidationError(HINT_DATE_RANGE)
        clauses = ["t.deleted_at IS NULL"]
        params: list[Any] = []
        if checked_entity is not None:
            clauses.append("t.entity = ?")
            params.append(checked_entity)
        if entity_id is not None:
            clauses.append("t.entity_id = ?")
            params.append(int(entity_id))
        if entity_version is not None:
            clauses.append("t.entity_version = ?")
            params.append(int(entity_version))
        session_value = self._text(session)
        if session_value is not None:
            clauses.append("t.session = ?")
            params.append(session_value)
        source_value = self._text(source)
        if source_value is not None:
            clauses.append("t.source = ?")
            params.append(source_value)
        if start is not None:
            # An absolute UTC day, inclusive on both sides (§8/§13.12).
            clauses.append("substr(t.created_at, 1, 10) >= ?")
            params.append(start)
        if end is not None:
            clauses.append("substr(t.created_at, 1, 10) <= ?")
            params.append(end)
        return " AND ".join(clauses), params

    def _checked_date(self, value: str | None) -> str | None:
        """`date_from`/`date_to`: only an absolute `YYYY-MM-DD` date (§8).

                    Parsing — by the block-4 parser (`freshness.parse_updated_after`, #29):
                    ISO-8601 UTC is accepted, but the slice keeps only the day; a relative
                    wording → `HINT_DATE_ABSOLUTE` (a soft failure, §7.3/§13.12).
        """
        text = self._text(value)
        if text is None:
            return None
        try:
            return parse_updated_after(text)[:10]
        except FreshnessValidationError as exc:
            raise LogValidationError(HINT_DATE_ABSOLUTE) from exc

    def _fts_candidates(
        self,
        conn: sqlite3.Connection,
        expression: str,
        where_sql: str,
        params: list[Any],
    ) -> list[sqlite3.Row]:
        """Top-`CANDIDATE_LIMIT` FTS5/BM25 over active records with the slice.

                    Mirrors `AreaSearch._fts_candidates`, but over the area's own tables
                    and with the shared slice predicate; rank from 1 for RRF.
        """
        weights = tuple(1.0 for _ in ASSISTANT_LOGS_FTS_COLUMNS)
        weight_placeholders = ", ".join("?" * len(weights))
        return conn.execute(
            f"SELECT t.id, t.created_at, "
            f"       bm25({ASSISTANT_LOGS_FTS_TABLE}, {weight_placeholders}) "
            "AS badness "
            f"FROM {ASSISTANT_LOGS_FTS_TABLE} JOIN {ASSISTANT_LOGS_TABLE} t "
            f"ON t.id = {ASSISTANT_LOGS_FTS_TABLE}.rowid "
            f"WHERE {ASSISTANT_LOGS_FTS_TABLE} MATCH ? AND {where_sql} "
            "ORDER BY badness, t.created_at DESC, t.id DESC LIMIT ?",
            (*weights, expression, *params, CANDIDATE_LIMIT),
        ).fetchall()

    def _filter_rows(
        self,
        conn: sqlite3.Connection,
        ids: list[int],
        where_sql: str,
        params: list[Any],
    ) -> dict[int, sqlite3.Row]:
        """Active candidates that passed THE SAME slice as the FTS side.

                    The vector side (vec0) knows no filters — slices are applied here, by
                    one predicate (§8: "filters apply to both sides").
        """
        if not ids:
            return {}
        placeholders = ", ".join("?" * len(ids))
        rows = conn.execute(
            "SELECT t.id, t.action, t.entity, t.entity_id, t.created_at "
            f"FROM assistant_logs t WHERE {where_sql} "
            f"AND t.id IN ({placeholders})",
            (*params, *ids),
        ).fetchall()
        return {int(row["id"]): row for row in rows}

    def _merged_search_results(
        self, scores: dict[int, float], rows: dict[int, sqlite3.Row]
    ) -> list[dict[str, Any]]:
        """Sort score DESC → created_at DESC → id DESC and trim action."""
        excerpt = self._settings.logs_search_excerpt_chars
        candidates = [
            (scores[row_id], row)
            for row_id, row in rows.items()
            if row_id in scores
        ]
        candidates.sort(
            key=lambda pair: (pair[0], pair[1]["created_at"], pair[1]["id"]),
            reverse=True,
        )
        return [
            {
                "id": int(row["id"]),
                "action": self._excerpt(row["action"], excerpt),
                "entity": row["entity"],
                "entity_id": row["entity_id"],
                "created_at": row["created_at"],
                "score": score,
                **self._entity_state_field(row["entity"], row["entity_id"]),
            }
            for score, row in candidates
        ]

    def _query_vector(self, query: str) -> list[float] | None:
        """Encode the query; an embedder failure — degrade to FTS (§8, NFR-3).

                    No embedder configured (an DI empty service) — FTS-only as well: the
                    search stays alive without the external dependency.
        """
        if self._embedding is None:
            return None
        try:
            return self._embedding.embed(query)
        except EmbeddingError:
            return None

    @staticmethod
    def _excerpt(text: str, limit: int) -> str:
        """Trim the body for the search output (the full text — `logs_get`)."""
        return text if len(text) <= limit else text[:limit]

    # --- deletion (soft delete, arch §11) -------------------------------------

    def delete(self, id: int) -> dict[str, Any]:
        """Soft-delete a record: `deleted_at` = now, the row and indexes stay alive.

                    Idempotent in meaning: a repeated/nonexistent id → a soft
                    `deleted: False` answer + the §7.3 canon hint (restoration — by the
                    operator). The deletion is a history fact (`record_event('delete')`) in
                    the same transaction: the log record does not vanish; the operator and
                    the history can read it (§11).
        """
        with db_session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE assistant_logs SET deleted_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (id,),
            )
            deleted = bool(cursor.rowcount)
            if deleted:
                history.record_event(
                    conn, history.ENTITY_ASSISTANT_LOG, id, "delete"
                )
        if not deleted:
            return {"id": id, "deleted": False, "hint": HINT_NOT_FOUND}
        # The signal to the areas loop is one for all area changes: a deletion adds no
        # pending, but the signal is harmless and keeps a single wakeup point.
        self._notify_areas_pending()
        return {"id": id, "deleted": True}

    # --- internal: shape (§7.3, the order is mandatory) -----------------------

    def _checked_form(
        self,
        *,
        action: Any,
        reason: Any,
        entity: Any,
        entity_id: int | None,
        entity_version: int | None,
        entity_ref: Any,
        outcome: Any,
        verified: Any,
    ) -> dict[str, Any]:
        """Check the new record's shape BEFORE the DB and return it normalized.

                    The §7.3 order: (1) `action`/`reason` — non-empty and within limits;
                    (2) `entity` from the dictionary (empty — allowed); (3) the object
                    links (an id for versioned, a text reference for external,
                    `entity_version` without an id — a failure); (4) the limits of
                    `entity_ref`/`outcome`/`verified`. Each failure — its own hint.
        """
        checked_action = self._checked_action(action)
        checked_reason = self._checked_reason(reason)
        checked_entity = self._checked_entity(entity)
        self._checked_entity_links(
            checked_entity, entity_id, entity_version, self._text(entity_ref)
        )
        return {
            "action": checked_action,
            "reason": checked_reason,
            "entity": checked_entity,
            "entity_id": None if entity_id is None else int(entity_id),
            "entity_version": (
                None if entity_version is None else int(entity_version)
            ),
            "entity_ref": self._checked_entity_ref(entity_ref),
            "outcome": self._checked_outcome(outcome),
            "verified": self._checked_verified(verified),
        }

    def _checked_entity_links(
        self,
        entity: str | None,
        entity_id: int | None,
        entity_version: int | None,
        entity_ref: str | None,
    ) -> None:
        """The change object's links: id / version / text reference (§7.3 p.3)."""
        if entity in VERSIONED_ENTITY_KINDS and entity_id is None:
            raise LogValidationError(
                HINT_ENTITY_ID_REQUIRED.format(given=entity)
            )
        if entity_version is not None and entity_id is None:
            raise LogValidationError(HINT_ENTITY_VERSION_NEEDS_ID)
        if entity in EXTERNAL_ENTITY_KINDS and not entity_ref:
            raise LogValidationError(
                HINT_ENTITY_REF_REQUIRED.format(given=entity)
            )

    def _checked_action(self, action: str | None) -> str:
        """The mandatory `action`: non-empty and ≤ `logs_action_max_chars`."""
        value = self._text(action)
        if not value:
            raise LogValidationError(HINT_ACTION_REQUIRED)
        if len(value) > self._settings.logs_action_max_chars:
            raise LogValidationError(HINT_ACTION_LIMIT)
        return value

    def _checked_reason(self, reason: str | None) -> str:
        """The mandatory `reason`: non-empty and ≤ `logs_reason_max_chars`."""
        value = self._text(reason)
        if not value:
            raise LogValidationError(HINT_REASON_REQUIRED)
        if len(value) > self._settings.logs_reason_max_chars:
            raise LogValidationError(HINT_REASON_LIMIT)
        return value

    def _checked_entity(self, entity: str | None) -> str | None:
        """`entity` — the change object's type: empty or from the closed dictionary.

                    An unknown type → a soft failure listing the dictionary (the model does
                    not guess the object type: it goes into the reverse question "why did
                    it change").
        """
        value = self._text(entity)
        if value is None:
            return None
        if value not in ENTITY_KINDS:
            raise LogValidationError(HINT_ENTITY_UNKNOWN.format(given=value))
        return value

    def _checked_entity_ref(self, entity_ref: str | None) -> str | None:
        """`entity_ref` — a text reference to an object outside memory (≤ 200)."""
        value = self._text(entity_ref)
        if value is not None and (
            len(value) > self._settings.logs_entity_ref_max_chars
        ):
            raise LogValidationError(HINT_ENTITY_REF_LIMIT)
        return value

    def _checked_outcome(self, outcome: str | None) -> str | None:
        """`outcome` — the outcome of the work (≤ `logs_outcome_max_chars`)."""
        value = self._text(outcome)
        if value is not None and len(value) > self._settings.logs_outcome_max_chars:
            raise LogValidationError(HINT_OUTCOME_LIMIT)
        return value

    def _checked_verified(self, verified: str | None) -> str | None:
        """`verified` — what was verified (≤ `logs_verified_max_chars`)."""
        value = self._text(verified)
        if value is not None and (
            len(value) > self._settings.logs_verified_max_chars
        ):
            raise LogValidationError(HINT_VERIFIED_LIMIT)
        return value

    def _checked_source(self, source: str | None) -> str | None:
        """An explicitly passed `source` — from the §5.1 dictionary or `job:<name>`.

                    The source is never invented by the model (§7.1): an unknown value — a
                    soft failure in the same hint style. An empty string = "not passed".
        """
        value = self._text(source)
        if value is None:
            return None
        if value not in LOG_SOURCES and not value.startswith(JOB_SOURCE_PREFIX):
            raise LogValidationError(HINT_SOURCE_UNKNOWN.format(given=value))
        return value

    def _resolved_origin(
        self, source: str | None, session: str | None
    ) -> tuple[str | None, str | None]:
        """The record's source/session: explicit arguments beat the current `bind_origin`.

                    Both fields are independent: a not-passed one comes from
                    `history.current_origin()` (the call middleware), a passed one is
                    validated (source) and cleaned (`strip`).
        """
        current_source, current_session = history.current_origin()
        if source is None:
            resolved_source: str | None = current_source
        else:
            # An empty string = "not passed": the call context's source is used.
            resolved_source = self._checked_source(source) or current_source
        resolved_session = (
            current_session if session is None else self._text(session)
        )
        return resolved_source, resolved_session

    # --- internal: utilities --------------------------------------------------

    @staticmethod
    def _text(value: Any) -> str | None:
        """The field text "as is" but without edge spaces; empty → `None`."""
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        return stripped or None

    @staticmethod
    def _projection_row(conn: sqlite3.Connection, log_id: int) -> sqlite3.Row:
        """The log's projection row after a write — one shape for baseline/delta."""
        return conn.execute(
            "SELECT action, reason, entity, entity_id, entity_version, "
            "entity_ref, outcome, verified, source, session "
            "FROM assistant_logs WHERE id = ?",
            (log_id,),
        ).fetchone()

    def _notify_areas_pending(self) -> None:
        """A signal to the worker: the area changed (wake the `areas` loop at once)."""
        if self._areas_notifier is not None:
            self._areas_notifier()
