"""TermsService — the "terms" area's service layer (lsb-0008, release 3.0.0).

    The record shape and limits — arch lsb-0008 §3.1; the key and normalization —
    §3.2; the data — §3.3; the "all senses with contexts" search — §3.4; the write
    mechanics and hints — §3.5; the canonical texts (soft-failure hints) — §3.7.

    Key properties
    --------------
    * **The area key** — `(term_norm, context_norm)` (§3.2): normalization via the
      substrate's `normalize_key` (lower → trim → collapsed spaces → yo→ye);
      original forms are stored too. Uniqueness — a partial UNIQUE over ACTIVE
      rows (`WHERE deleted_at IS NULL`): the key does not hold a soft-deleted row,
      and the same key can be created anew (§3.3).
    * **The context is ALWAYS mandatory** (§3.1) — empty → a failure with the
      verbatim §3.7 canon hint. Limits: `term` ≤ `term_max_chars` (100), `context` ≤
      `term_context_max_chars` (40), `definition` ≤ `term_definition_max_chars`
      (350); a violation → `TermValidationError`, the record is NOT written.
    * **Polysemy is the norm** (§3.5): the key matched → an UPDATE (the definition
      and original forms); the key did not match and the context is too close to an
      already-used context of THE SAME term (trigram similarity ≥
      `term_context_similarity` = 0.75) → a soft failure BEFORE the write (there is
      none) + a hint with the existing context and its id; otherwise → an INSERT of
      a new sense. The existing record is not overwritten: "MGU" for students is
      not overwritten by "MGU" in accounting.
    * **Hints in the successful answer** (FR-3.2): `senses` — this term's senses
      `[{id, context}]` (active only), `contexts` — the area's global list of used
      contexts (top `term_contexts_hint_limit` = 30 by frequency, no normalization
      duplicates).
    * **Embedding is NOT called at write time** (FR-3.5, substrate §3.3): context
      closeness is trigram-based (arithmetic); the write's vectors are caught up by
      the worker's `areas` loop over `vector_status='pending'`, and after a
      successful save/edit/delete the service wakes it (`set_areas_notifier`).
    * **The search is two-step** (§3.4): step 1 — an exact normalized lookup over
      `term_norm` → ALL the term's senses with contexts (`exact: true`); step 2 —
      no exact one → the area hybrid (`AreaSearch` over `TERMS_AREA`, the vector =
      `term + context + definition`) → the nearest senses `exact: false` + the hint
      "this is not an exact match"; step 3 — nothing → empty + the §3.7 canon
      hint.
    * **Deletion is soft**: a repeated/nonexistent id → a soft answer with a hint
      (restoration — by the operator, the notes precedent).
    * **There is no listing** (§3.1 of the requirements): the only road to a
      definition is the search; the notes pipeline is untouched by the service
      (isolation: the SQL addresses only the area's tables).

    Response contracts (the MCP layer trims service fields by a whitelist —
    lsb-0008-02; the REST mirror — lsb-0008-03):
    - save (a new sense) → {created: True, id, senses, contexts}
    - save (the key matched) → {updated: True, id, senses, contexts}
    - save (a close context) → {created: False, hint, existing: {id, context}}
    - search (exact)  → {senses: [{id, context, definition}], exact: True}
    - search (close) → {senses: [{id, term, context, definition, score}],
                        exact: False, hint}
    - search (empty)   → {senses: [], exact: False, hint}
    - update           → {id, updated: True} | {id, updated: False, hint};
                         a taken key → TermKeyConflictError (the transport → 409)
    - get              → {id, term, context, definition}; not found → {id, hint}
    - delete           → {id, deleted: True} | {id, deleted: False, hint}
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

from app.config import Settings
from app.services import history
from app.services.areas import (
    TERMS_AREA,
    AreaSearch,
    AreaSearchValidationError,
    normalize_key,
    trigram_similarity,
)
from app.services.embedding import Embedder, EmbeddingService
from app.services.search import MAX_TOP_K
from app.storage.db import session, transaction

# --- Hints (canon §3.7: "verbatim constants in code"; English texts) --------

# No exact term: the nearest senses are served but NOT as an exact match
# (§3.4) — otherwise the model would silently substitute a wrong definition.
HINT_NO_EXACT = (
    "no exact term — the senses above are the closest by meaning, not an "
    "exact match: check them against the context of the conversation"
)

# No term at all (search step 3, `get`, `delete`) — a soft answer without fail.
HINT_NOT_FOUND = (
    "term not found — there is no such term in this memory; save it via "
    "terms_save(term, context, definition) if it is worth keeping"
)

# A close context (a soft failure BEFORE the write, §3.5): the template substitutes the given
# context, the existing one and its id — the model reuses the wording.
HINT_CONTEXT_CLOSE = (
    "not saved: context '{given}' is too close to the existing one "
    "'{existing}' (id={id}) — reuse that context wording, or make the new "
    "context clearly different"
)

HINT_TERM_LIMIT = "not saved: term limit is 100 characters — shorten it"
HINT_CONTEXT_LIMIT = "not saved: context limit is 40 characters — shorten it"
HINT_DEFINITION_LIMIT = (
    "not saved: definition limit is 350 characters — shorten it"
)
HINT_CONTEXT_REQUIRED = (
    "not saved: context is required — specify the context in which the term "
    "is used (≤40 characters)"
)

# A hint the §3.7 canon table does not set: an empty `term` (the canon describes
# only the limits, the context obligatoriness and the close context). An empty term
# breaks the area key — a failure; the style and language — like the canonical ones.
HINT_TERM_REQUIRED = "not saved: term is required"

# A key conflict on an edit BY ID (PUT /terms/{id}, substrate §3.6: 409). The canon
# §3.7 sets the soft write-failure texts but not the key conflict: the transport
# must tell it apart from validation, so the text follows the canonical style
# (English, leading to the existing record and a safe exit).
HINT_KEY_CONFLICT = (
    "not saved: the key (term + context) is already used by another active "
    "record (id={id}: '{term}' / '{context}') — edit that record instead, or "
    "choose a different context or term"
)


class TermValidationError(ValueError):
    """A term's domain constraint violation (shape limits, search query).

            The exception text is a hint for the model (the MCP/REST transport gives it
            to the client as fail + hint); the record is NOT written: the precedents
            are `TitleValidationError`/`TITLE_HINT`, `SkillValidationError`/
            `UserFactValidationError`.
    """


class TermKeyConflictError(ValueError):
    """The edit's new key is taken by ANOTHER active record of the area (substrate
            §3.6: a key conflict → 409; a separate type so the transport does not
            confuse it with shape violations and soft failures served as 422).

            The exception text — the verbatim hint constant `HINT_KEY_CONFLICT` with
            the existing record's id: no write happens; the operator edits it or
            changes the key.
    """


class TermsService:
    """The "terms" area CRUD: the key (term + context), senses, hints, search.

            DI: `settings` (the shape limits, the context closeness threshold, the
            contexts list limit), `embedding` (encoding the hybrid search query). The
            write path does NOT call the encoder (§3.5, substrate §3.3): context
            closeness is trigram-based, the vectors are caught up by the worker's
            `areas` loop. The service has no listing — the public handles are only
            `save`/`update`/`search`/`get`/`delete`.
    """

    def __init__(
        self, settings: Settings, embedding: Embedder | None = None
    ) -> None:
        self._settings = settings
        # DI for tests: a HashEmbedder/fake with a call log instead of the network.
        self._embedding: Embedder = (
            embedding if embedding is not None else EmbeddingService(settings)
        )
        # The area's hybrid search (substrate §3.5): vec0-KNN over `terms_vec`
        # (the vector = `term + context + definition`) + FTS5/BM25 over `terms_fts`
        # → RRF. The helper's SQL addresses only the area's tables (isolation).
        self._area_search = AreaSearch(settings, TERMS_AREA, self._embedding)
        # A signal to the worker (main.py): wake the areas loop at once when
        # an area pending record appears, instead of waiting out the grown back-off.
        self._areas_notifier: Callable[[], None] | None = None

    def set_areas_notifier(self, notifier: Callable[[], None] | None) -> None:
        """Wire the wakeup signal of the worker's `areas` loop (main.py).

                    Modeled on `SkillsService.set_areas_notifier`/`UserFactsService`: the
                    service is assembled before the worker, the notifier arrives after.
        """
        self._areas_notifier = notifier

    # --- writes and edits by key (arch §3.5) ----------------------------------

    def save(self, term: str, context: str, definition: str) -> dict[str, Any]:
        """Save a term with the key (`term` + `context`) — the three §3.5 outcomes.

                    The order is mandatory: the shape is checked BEFORE any DB access
                    (a limit violation → `TermValidationError` with the verbatim §3.7
                    hint).

                    * the key matched (the normalization `term_norm` + `context_norm`) →
                      an UPDATE of the definition and original forms →
                      `{updated: True, id, ...}`;
                    * the key did not match, but the context is too close (the trigram
                      similarity of the normalized strings ≥ `term_context_similarity`) to
                      a context of THE SAME term → a soft failure, no write, the hint
                      names the existing context and its id;
                    * otherwise → an INSERT of a new sense → `{created: True, id, ...}`.

                    In both successful outcomes the answer carries `senses` (this term's
                    senses after the write) and `contexts` (the area's used contexts). No
                    embedding is called: the context similarity is arithmetic (§3.5).
        """
        checked_term = self._checked_term(term)
        checked_context = self._checked_context(context)
        checked_definition = self._checked_definition(definition)
        term_norm = normalize_key(checked_term)
        context_norm = normalize_key(checked_context)
        with session(self._settings) as conn, transaction(conn):
            key_row = conn.execute(
                "SELECT id FROM terms WHERE term_norm = ? "
                "AND context_norm = ? AND deleted_at IS NULL",
                (term_norm, context_norm),
            ).fetchone()
            if key_row is not None:
                # The key matched: this is an edit of the existing sense, not a duplicate
                # (the original forms may change; the key does not).
                term_id = int(key_row["id"])
                conn.execute(
                    "UPDATE terms SET term = ?, term_norm = ?, context = ?, "
                    "context_norm = ?, definition = ?, "
                    "vector_status = 'pending', "
                    "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                    "WHERE id = ? AND deleted_at IS NULL",
                    (
                        checked_term,
                        term_norm,
                        checked_context,
                        context_norm,
                        checked_definition,
                        term_id,
                    ),
                )
                # #41 (FR-3.3, §13.5): an edit by the matched key — a version
                # delta in the shared history, the same transaction; the service
                # `term_norm`/`context_norm` are not in the projection (FR-4.2).
                live = self._projection_row(conn, term_id)
                history.record_change(
                    conn, history.ENTITY_TERM, term_id, history.term_projection(live)
                )
                answer: dict[str, Any] = {"updated": True, "id": term_id}
            else:
                near = self._nearest_context(
                    conn, term_norm, checked_context
                )
                if (
                    near is not None
                    and near[2] >= self._settings.term_context_similarity
                ):
                    # A soft failure BEFORE the write: a close
                    # context wording creates no new sense — the hint
                    # leads to the existing context and its id.
                    return {
                        "created": False,
                        "hint": HINT_CONTEXT_CLOSE.format(
                            given=checked_context,
                            existing=near[1],
                            id=near[0],
                        ),
                        "existing": {"id": near[0], "context": near[1]},
                    }
                cursor = conn.execute(
                    "INSERT INTO terms (term, term_norm, context, "
                    "context_norm, definition, vector_status) "
                    "VALUES (?, ?, ?, ?, ?, 'pending')",
                    (
                        checked_term,
                        term_norm,
                        checked_context,
                        context_norm,
                        checked_definition,
                    ),
                )
                term_id = int(cursor.lastrowid or 0)
                # #41 (§13.5): the new sense's baseline — in the same transaction.
                live = self._projection_row(conn, term_id)
                history.record_initial(
                    conn, history.ENTITY_TERM, term_id, history.term_projection(live)
                )
                answer = {"created": True, "id": term_id}
            # The answer's reference part (FR-3.2): the term's senses after the write and
            # the area's used contexts — both success branches.
            answer["senses"] = self._senses(conn, term_norm)
            answer["contexts"] = self._contexts(conn)
        self._notify_areas_pending()
        return answer

    def update(
        self, id: int, term: str, context: str, definition: str
    ) -> dict[str, Any]:
        """Edit a record BY ID (PUT /terms/{id}; arch §3.6, substrate §3.6).

                    It differs from `save` by addressing the row by id, not by the key:
                    the purpose is rare operator cleanup/editing of terminology, including
                    a move to another key (`term`/`context`). The shape is checked BEFORE
                    the DB (a limit violation or an empty context →
                    `TermValidationError` → 422 with the verbatim §3.7 hint); the key is
                    recomputed (`term_norm`/`context_norm`), the vector goes to `pending`,
                    `updated_at` is refreshed (the `areas` loop catches up).

                    Outcomes:

                    * the new key is taken by ANOTHER active record →
                      `TermKeyConflictError` (`HINT_KEY_CONFLICT`, the transport → 409):
                      the area key is unique, and a move must not overwrite a foreign
                      sense;
                    * the same key (the record itself) — not a conflict: the edit goes
                      normally;
                    * no record / soft deleted → a soft answer `{id, updated: False,
                      hint}` of canon §3.7 (the transport → 404).

                    The "close context" hint does NOT apply here: that is the protection of
                    writing a NEW sense (§3.5), while an edit addresses an existing row
                    deliberately.
        """
        checked_term = self._checked_term(term)
        checked_context = self._checked_context(context)
        checked_definition = self._checked_definition(definition)
        term_norm = normalize_key(checked_term)
        context_norm = normalize_key(checked_context)
        with session(self._settings) as conn, transaction(conn):
            row = conn.execute(
                "SELECT id FROM terms WHERE id = ? AND deleted_at IS NULL",
                (id,),
            ).fetchone()
            if row is None:
                return {"id": id, "updated": False, "hint": HINT_NOT_FOUND}
            occupied = conn.execute(
                "SELECT id, term, context FROM terms WHERE term_norm = ? "
                "AND context_norm = ? AND deleted_at IS NULL AND id != ?",
                (term_norm, context_norm, id),
            ).fetchone()
            if occupied is not None:
                raise TermKeyConflictError(
                    HINT_KEY_CONFLICT.format(
                        id=int(occupied["id"]),
                        term=str(occupied["term"]),
                        context=str(occupied["context"]),
                    )
                )
            conn.execute(
                "UPDATE terms SET term = ?, term_norm = ?, context = ?, "
                "context_norm = ?, definition = ?, "
                "vector_status = 'pending', "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (
                    checked_term,
                    term_norm,
                    checked_context,
                    context_norm,
                    checked_definition,
                    id,
                ),
            )
            # #41 (FR-3.3, §13.5): an edit by id — a version delta in the same
            # transaction; `term_norm`/`context_norm`/`vector_status` are outside the projection.
            live = self._projection_row(conn, id)
            history.record_change(
                conn, history.ENTITY_TERM, id, history.term_projection(live)
            )
        self._notify_areas_pending()
        return {"id": id, "updated": True}

    # --- reads (arch §3.4, §3.5) ----------------------------------------------

    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """Find a term: an exact normalized lookup → the area hybrid → empty.

                    Step 1: an exact `term_norm` match (normalization removes case, "yo",
                    extra spaces) → ALL the term's active senses `[{id, context,
                    definition}]`, `exact: true`; one sense is not picked for the model
                    (§3.4).
                    Step 2: no exact one → the `TERMS_AREA` hybrid (vec0-KNN over
                    `term + context + definition` + FTS5/BM25 → RRF) → the nearest senses
                    `[{id, term, context, definition, score}]`, `exact: false` + the
                    verbatim "not an exact match" hint. Step 3: nothing →
                    `{senses: [], exact: false, hint}`.

                    `top_k` — the bounds like the notes search (`1..MAX_TOP_K`), a
                    violation — `TermValidationError` (a soft transport failure). An
                    embedder failure does not break the search: FTS-only + `warning`
                    (NFR-3). Notes, skills and user facts do not enter the output: the
                    helper's SQL addresses only the area's tables.
        """
        if top_k is not None and not 1 <= top_k <= MAX_TOP_K:
            raise TermValidationError(
                f"top_k: expected 1..{MAX_TOP_K}, got {top_k}"
            )
        query_norm = normalize_key(query)
        if query_norm:
            with session(self._settings) as conn:
                exact_rows = conn.execute(
                    "SELECT id, context, definition FROM terms "
                    "WHERE term_norm = ? AND deleted_at IS NULL ORDER BY id",
                    (query_norm,),
                ).fetchall()
            if exact_rows:
                return {
                    "senses": [
                        {
                            "id": int(row["id"]),
                            "context": row["context"],
                            "definition": row["definition"],
                        }
                        for row in exact_rows
                    ],
                    "exact": True,
                }
        try:
            found = self._area_search.search(query, top_k)
        except AreaSearchValidationError as exc:  # a query outside the area's domain
            raise TermValidationError(str(exc)) from exc
        answer: dict[str, Any] = {
            "senses": [
                {
                    "id": int(item["id"]),
                    "term": item["term"],
                    "context": item["context"],
                    "definition": item["definition"],
                    "score": item["score"],
                }
                for item in found["results"]
            ],
            "exact": False,
            "warning": found.get("warning"),
        }
        # The §3.7 canon hint: no exact one — the nearest; nothing — no term.
        answer["hint"] = HINT_NO_EXACT if answer["senses"] else HINT_NOT_FOUND
        return answer

    def get(self, id: int) -> dict[str, Any]:
        """Read a record by id: `{id, term, context, definition}`; none — a hint."""
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT id, term, context, definition FROM terms "
                "WHERE id = ? AND deleted_at IS NULL",
                (id,),
            ).fetchone()
        if row is None:
            return {"id": id, "hint": HINT_NOT_FOUND}
        return {
            "id": int(row["id"]),
            "term": row["term"],
            "context": row["context"],
            "definition": row["definition"],
        }

    # --- deletion (soft delete, arch §3.3) ------------------------------------

    def delete(self, id: int) -> dict[str, Any]:
        """Soft-delete a record: `deleted_at` = now; the row/indexes stay alive.

                    Idempotent in meaning: a repeated/nonexistent id → a soft
                    `deleted: False` answer + the §3.7 canon hint (restoration — by the
                    operator). The deleted key is freed: the partial UNIQUE — over active
                    records only, the same key can be created anew.
        """
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE terms SET deleted_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (id,),
            )
            deleted = bool(cursor.rowcount)
            if deleted:
                # #41 (FR-6.1, §13.5): the soft delete — a version event.
                history.record_event(conn, history.ENTITY_TERM, id, "delete")
        if not deleted:
            return {"id": id, "deleted": False, "hint": HINT_NOT_FOUND}
        # The signal to the areas loop is one for all successful area changes: a deletion adds no
        # pending, but the signal is harmless and keeps a single
        # worker wakeup point (the lsb-0008-01 brief).
        self._notify_areas_pending()
        return {"id": id, "deleted": True}

    # --- internal: hints and context closeness --------------------------------

    def group(self, term_norm: str) -> list[dict[str, Any]]:
        """The active rows of a term's group — the `terms-normalize` job's read helper.

                    Returns `[{id, term, context, definition, context_norm, updated_at}]`
                    over active rows only (`deleted_at IS NULL`), the order — by id.
                    SQL only: the job writes via the existing `update`/`delete`; their
                    behavior and contracts do not change (#32 FR-9.1). The area gains no
                    listing — the handle lives only inside the background job.
        """
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, term, context, definition, context_norm, updated_at "
                "FROM terms WHERE term_norm = ? AND deleted_at IS NULL "
                "ORDER BY id",
                (term_norm,),
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "term": str(row["term"]),
                "context": str(row["context"]),
                "definition": str(row["definition"]),
                "context_norm": str(row["context_norm"]),
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    @staticmethod
    def _projection_row(conn: sqlite3.Connection, term_id: int) -> sqlite3.Row:
        """The term's projection row after a write — one shape for baseline/delta."""
        return conn.execute(
            "SELECT term, context, definition FROM terms WHERE id = ?",
            (term_id,),
        ).fetchone()

    def _senses(
        self, conn: sqlite3.Connection, term_norm: str
    ) -> list[dict[str, Any]]:
        """This term's active senses — `[{id, context}]` by id ascending."""
        rows = conn.execute(
            "SELECT id, context FROM terms WHERE term_norm = ? "
            "AND deleted_at IS NULL ORDER BY id",
            (term_norm,),
        ).fetchall()
        return [
            {"id": int(row["id"]), "context": row["context"]} for row in rows
        ]

    def _contexts(self, conn: sqlite3.Connection) -> list[str]:
        """The area's used contexts: top by frequency, no duplicates.

                    The frequency — the number of the area's active records with this
                    `context_norm` (GROUP BY `context_norm`), the limit —
                    `term_contexts_hint_limit` (30). No normalization duplicates. Which
                    original form is returned when normalized forms match is decided so:
                    the form of the LATEST record of that context (MAX(id)) — "the current
                    wording"; at equal frequency such contexts go first (the order:
                    frequency DESC, `last_id` DESC). The choice is deterministic and does
                    not depend on the rows' order in the table.
        """
        rows = conn.execute(
            "SELECT t.context AS context FROM ("
            "  SELECT context_norm, MAX(id) AS last_id, COUNT(*) AS uses "
            "  FROM terms WHERE deleted_at IS NULL GROUP BY context_norm"
            ") c JOIN terms t ON t.id = c.last_id "
            "ORDER BY c.uses DESC, c.last_id DESC LIMIT ?",
            (self._settings.term_contexts_hint_limit,),
        ).fetchall()
        return [str(row["context"]) for row in rows]

    def _nearest_context(
        self, conn: sqlite3.Connection, term_norm: str, context: str
    ) -> tuple[int, str, float] | None:
        """The nearest used context of this term — `(id, form, similarity)`.

                    The similarity — trigram-based (the substrate's `trigram_similarity`,
                    the overlap coefficient over normalized strings); no embedding is
                    called (§3.5). At equal similarity the smaller id wins — the result is
                    deterministic. No active senses of the term → None (the write goes
                    without a hint).
        """
        rows = conn.execute(
            "SELECT id, context FROM terms WHERE term_norm = ? "
            "AND deleted_at IS NULL ORDER BY id",
            (term_norm,),
        ).fetchall()
        nearest: tuple[int, str, float] | None = None
        for row in rows:
            similarity = trigram_similarity(context, str(row["context"]))
            if nearest is None or similarity > nearest[2]:
                nearest = (int(row["id"]), str(row["context"]), similarity)
        return nearest

    # --- internal: shape validation -------------------------------------------

    def _checked_term(self, term: str | None) -> str:
        """The mandatory `term`: non-empty and ≤ `term_max_chars` characters.

                    An empty term breaks the area key → a failure with a hint (the §3.7
                    canon sets only the limit); over the limit → the verbatim §3.7 canon
                    hint. Returns the original form without edge spaces (original forms are
                    stored next to the normalized ones, §3.2).
        """
        normalized = term.strip() if isinstance(term, str) else ""
        if not normalized:
            raise TermValidationError(HINT_TERM_REQUIRED)
        if len(normalized) > self._settings.term_max_chars:
            raise TermValidationError(HINT_TERM_LIMIT)
        return normalized

    def _checked_context(self, context: str | None) -> str:
        """The mandatory `context`: non-empty and ≤ `term_context_max_chars`.

                    The context is ALWAYS mandatory, even for a single sense (§3.1):
                    empty → a failure with the verbatim §3.7 canon hint; over the limit —
                    the same.
        """
        normalized = context.strip() if isinstance(context, str) else ""
        if not normalized:
            raise TermValidationError(HINT_CONTEXT_REQUIRED)
        if len(normalized) > self._settings.term_context_max_chars:
            raise TermValidationError(HINT_CONTEXT_LIMIT)
        return normalized

    def _checked_definition(self, definition: str | None) -> str:
        """`definition` ≤ `term_definition_max_chars` characters.

                    Canon §3.1 marks only the context mandatory (an empty definition is not
                    a limit violation), so only the upper bound is checked: over it → the
                    verbatim §3.7 canon hint.
        """
        normalized = definition.strip() if isinstance(definition, str) else ""
        if len(normalized) > self._settings.term_definition_max_chars:
            raise TermValidationError(HINT_DEFINITION_LIMIT)
        return normalized

    # --- internal: the signal --------------------------------------------------

    def _notify_areas_pending(self) -> None:
        """A signal to the worker: the area changed (wake the `areas` loop at once)."""
        if self._areas_notifier is not None:
            self._areas_notifier()
