"""UserFactsService — the "user" area's service layer (lsb-0009, release 3.0.0).

    The fact shape and limits — arch lsb-0009 §3.1; the data — §3.2; the handles
    and contracts — §3.3; the "is there something similar" dedup without embedding
    at write time — §3.4; the canonical texts (the permanent atomicity hint, the
    soft-failure hints) — §3.7.

    Key properties
    --------------
    * **Atomicity** — one fact = one record: the `name` shape (≤5 words, the
      notes' `title` contract) + `body` (≤1200 characters); a violation →
      `UserFactValidationError` with the verbatim §3.7 canon hint (the fact is NOT
      saved).
    * **The dedup hint BEFORE the write** (§3.4): candidates — FTS5 `trigram` over
      `name + body` (top-3, active only); the similarity — trigram-based
      (the substrate's `trigram_similarity`) over the normalized text. A strong
      match (≥ `USER_SIMILAR_STRONG`) → a soft failure + `similar` (no write);
      the middle zone (≥ `USER_SIMILAR_WEAK`) → the write happens + a reference
      `related` list; below the thresholds — the write without a hint. Embedding
      is NOT called at write time (FR-3.5): candidates and similarity are
      arithmetic.
    * **The permanent atomicity hint** (FR-7.2) — in EVERY successful `save`
      answer.
    * **The "not passed" sentinels = keep** (`_UNSET_NAME`/`_UNSET_BODY`,
      JSON-serializable strings — the lsb-0004/`notes.py` precedent); `null` in a
      mandatory field → a soft failure (mandatory fields are never cleared).
    * **An edit invalidates the vector** (`vector_status='pending'`, `updated_at`
      refreshed) — the worker's `areas` loop catches up the vectors (substrate
      §3.3); after a successful save/edit/delete the service wakes it
      (`set_areas_notifier`).
    * **Deletion is soft** (`deleted_at`): the fact vanishes from all the area's
      outputs; a repeated/nonexistent id → a soft answer with a hint (restoration
      — by the operator, the notes precedent).
    * **The search is an area probe** (§3.3, substrate §3.5): the `AreaSearch`
      hybrid over `USER_FACTS_AREA` (`user_facts_vec` KNN, the vector =
      `name + body` + `user_facts_fts` BM25 → RRF); the output has no bodies —
      an `excerpt` ≤ `USER_SEARCH_EXCERPT_CHARS` (the full body — `get`); empty →
      a soft answer with the §3.7 canon hint. The helper's SQL addresses only the
      area's tables (isolation).

    Response contracts (full; the MCP layer trims service fields by a whitelist —
    lsb-0009-02; the REST mirror — lsb-0009-03):
    - save (a write)    → {id, stored: True, hint: <the permanent atomicity hint>}
                          (+ the middle zone's related/hint)
    - save (a strong match) → {stored: False, hint, similar: {id, name}}
    - update           → {id, changed: True} | {id, changed: False, hint}
    - get              → {id, name, body}; not found → {id, hint}
    - search           → {results: [{id, name, excerpt}], warning?};
                         empty → {results: [], warning?, hint}
    - delete           → {id, deleted: True} | {id, deleted: False, hint}
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

from app.config import Settings
from app.services import history
from app.services.areas import (
    USER_FACTS_AREA,
    AreaSearch,
    AreaSearchValidationError,
    trigram_similarity,
)
from app.services.embedding import Embedder, EmbeddingService
from app.services.ranking import match_expression
from app.services.search import MAX_TOP_K
from app.storage.db import session, transaction

# The dedup candidates — the area's top-3 active records (arch §3.4): we check
# only the FTS-nearest ones; there is no full registry scan.
DEDUP_CANDIDATES = 3

# --- Hints (canon §3.7: "verbatim constants in code"; English texts) --------
# The permanent hint of a successful save (FR-7.2): the atomicity rule is anchored at
# the moment of action — several facts = several separate calls.
HINT_ATOMIC = "one fact = one record — several facts mean several separate calls"

# Dedup, a strong match (FR-7.3): no write; the hint leads to editing
# the existing fact; the template substitutes the nearest record's id and name.
HINT_SIMILAR_FACT = (
    "similar fact already exists: {id} — {name}; the same fact? update it via "
    "user_update(id={id}); a new fact — save it as a separate record"
)

# Dedup, the middle zone: the write happens; similar facts are listed for reference.
HINT_RELATED_FACTS = (
    "possibly related facts: {related}; if this is the same fact, refine it via "
    "user_update instead of creating a duplicate"
)

# A soft failure at `body` > 1200 (FR-7.4): "looks like several facts".
HINT_BODY_LIMIT = (
    "not saved: looks like several facts in one record — split them and save "
    "one fact per call (body limit is 1200 characters)"
)

# The `name` limit (canon §3.7): the note title contract (≤5 words).
HINT_NAME_LIMIT = "not saved: name must be ≤5 words (like a note title)"

# Not found (`get`/`update`/`delete`) — a soft answer with the way to the area's search.
HINT_NOT_FOUND = (
    "fact not found (possibly deleted); search the user area via user_search"
)

# An empty area search: "no fact" is a valid outcome (a probe); the model does not enter
# the area without a reason.
HINT_SEARCH_EMPTY = (
    "nothing found in the user area — no fact matching this request is stored"
)

# Hints the §3.7 canon table does not set (the canon describes only the limits
# and the soft failures above): an empty mandatory shape field and `null` in a mandatory
# edit field. The style and language — like the canonical ones.
HINT_NAME_REQUIRED = "not saved: name is required"
HINT_BODY_REQUIRED = "not saved: body is required"
HINT_REQUIRED_UNSET = (
    "not updated: {field} is mandatory — omitting the argument keeps the "
    "current value, null cannot reset it"
)

# The "not passed" sentinels (the lsb-0004 precedent): JSON-serializable strings —
# the defaults of the edit MCP tool's parameters; pydantic does not complain about a
# non-serializable default (NFR-4). The comparison in the service — by identity (`is`),
# so a collision with the fact's real content is impossible even with an equal
# string (like the notes' `_UNSET_SUMMARY`).
_UNSET_NAME: str = "__LSB_UNSET_USER_NAME__"
_UNSET_BODY: str = "__LSB_UNSET_USER_BODY__"


class UserFactValidationError(ValueError):
    """A fact's domain constraint violation (shape limits, search query).

            The exception text is a hint for the model (the MCP/REST transport gives it
            to the client as fail + hint); the fact is NOT saved: the precedents are
            `TitleValidationError`/`TITLE_HINT` and `SkillValidationError`.
    """


def _fact_text(name: str, body: str) -> str:
    """The fact's text for similarity and vectorization — `name + body`.

            The same format as the area's vector (`AreaSpec.embed_text`, substrate
            §3.3: user — `name + body`, non-empty fields joined by newline).
    """
    return "\n".join(part for part in (name, body) if part)


def _related_text(related: list[dict[str, Any]]) -> str:
    """The reference list of similar facts for the middle zone's hint — `id — name`."""
    return "; ".join(f"{item['id']} — {item['name']}" for item in related)


class UserFactsService:
    """The "user" area CRUD: atomic facts, the dedup hint, an isolated search.

            DI: `settings` (the shape limits, the similarity thresholds, the output
            trim), `embedding` (encoding the hybrid search query). The write path does
            NOT call the encoder (FR-3.5, substrate §3.3): the write's vectors are
            caught up by the worker's `areas` loop over `vector_status='pending'`.
            The skill-creator seed (lsb-0007-04) lives in the `skills` table and has
            nothing to do with this area — the service neither reads nor changes it.
    """

    def __init__(
        self, settings: Settings, embedding: Embedder | None = None
    ) -> None:
        self._settings = settings
        # DI for tests: a HashEmbedder/fake with a call log instead of the network.
        # The fact write never calls the encoder (dedup — FTS + trigrams);
        # the encoder is needed only by the area's hybrid search.
        self._embedding: Embedder = (
            embedding if embedding is not None else EmbeddingService(settings)
        )
        # The area's hybrid search (substrate §3.5): vec0-KNN over `user_facts_vec`
        # (the vector = `name + body`) + FTS5/BM25 over `user_facts_fts` → RRF.
        # The helper's SQL addresses only the area's tables (isolation).
        self._area_search = AreaSearch(settings, USER_FACTS_AREA, self._embedding)
        # A signal to the worker (main.py): wake the areas loop at once when
        # an area pending record appears, instead of waiting out the grown back-off.
        self._areas_notifier: Callable[[], None] | None = None

    def set_areas_notifier(self, notifier: Callable[[], None] | None) -> None:
        """Wire the wakeup signal of the worker's `areas` loop (main.py).

                    Modeled on `SkillsService.set_areas_notifier`: the service is assembled
                    before the worker, the notifier arrives after its creation.
        """
        self._areas_notifier = notifier

    # --- writes: creation (arch §3.4) -----------------------------------------

    def save(self, *, name: str, body: str) -> dict[str, Any]:
        """Save ONE atomic fact (validation → dedup → write).

                    The order is mandatory: the shape is checked BEFORE the dedup (a limit
                    violation → `UserFactValidationError` with the verbatim §3.7 hint);
                    the dedup scan — BEFORE the write and without embedding (§3.4). A
                    strong match → a soft failure `{stored: False, hint, similar}` without
                    a write; the middle zone → the write + `related` + a reference hint;
                    below the thresholds → the write. Every successful answer carries the
                    permanent atomicity hint (FR-7.2): several facts = several separate
                    calls.
        """
        checked_name = self._checked_name(name)
        checked_body = self._checked_body(body)
        with session(self._settings) as conn:
            candidates = self._dedup_candidates(conn, checked_name, checked_body)
        if candidates:
            nearest, similarity = candidates[0]
            if similarity >= self._settings.user_similar_strong:
                # A soft failure: no write; the hint
                # leads to editing the existing fact or to a separate record.
                return {
                    "stored": False,
                    "hint": HINT_SIMILAR_FACT.format(
                        id=nearest["id"], name=nearest["name"]
                    ),
                    "similar": nearest,
                }
        related = [
            item
            for item, similarity in candidates
            if similarity >= self._settings.user_similar_weak
        ]
        fact_id = self._create(checked_name, checked_body)
        answer: dict[str, Any] = {
            "id": fact_id,
            "stored": True,
            "hint": HINT_ATOMIC,
        }
        if related:
            # The middle zone: the write happened; similar ones are listed for reference.
            # Both canonical texts stay verbatim in the answer: the permanent
            # atomicity hint — always; the middle zone's hint — only here.
            answer["related"] = related
            answer["hint"] = (
                f"{HINT_ATOMIC}; "
                + HINT_RELATED_FACTS.format(related=_related_text(related))
            )
        self._notify_areas_pending()
        return answer

    def _create(self, name: str, body: str) -> int:
        """INSERT a fact with `vector_status='pending'` (the vectors — the `areas` loop).

                    #41 (§13.5): the baseline version is written by THE SAME transaction
                    as the INSERT — the state "the fact exists, no history" never happens.
        """
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "INSERT INTO user_facts (name, body, vector_status) "
                "VALUES (?, ?, 'pending')",
                (name, body),
            )
            fact_id = int(cursor.lastrowid or 0)
            row = conn.execute(
                "SELECT name, body FROM user_facts WHERE id = ?", (fact_id,)
            ).fetchone()
            history.record_initial(
                conn,
                history.ENTITY_USER_FACT,
                fact_id,
                history.user_fact_projection(row),
            )
            return fact_id

    # --- edit (arch §3.3: "not passed" = keep) --------------------------------

    def update(
        self,
        id: int,
        *,
        name: str | None = _UNSET_NAME,
        body: str | None = _UNSET_BODY,
    ) -> dict[str, Any]:
        """Edit a fact by id: `name` and/or `body`.

                    "Not passed" (the sentinel, the parameter default) = keep the current
                    value; `null` in a mandatory field — a soft failure (mandatory fields
                    are not cleared; the hint carries the "not passed = keep" semantics).
                    A passed value is validated as at creation. The edit returns the
                    record's vector to `pending` and refreshes `updated_at` (the `areas`
                    loop catches up the vectors); not found/deleted → a soft answer with
                    the §3.7 canon hint. Both parameters not passed — no edit (neither
                    `updated_at` nor the vector is touched).
        """
        name_passed = name is not _UNSET_NAME
        body_passed = body is not _UNSET_BODY
        if name_passed and name is None:
            raise UserFactValidationError(
                HINT_REQUIRED_UNSET.format(field="name")
            )
        if body_passed and body is None:
            raise UserFactValidationError(
                HINT_REQUIRED_UNSET.format(field="body")
            )
        checked_name = self._checked_name(name) if name_passed else None
        checked_body = self._checked_body(body) if body_passed else None
        if checked_name is None and checked_body is None:
            # Both parameters "not passed": the content is not changed (the
            # lsb-0004 precedent — "not passed" neither clears nor rewrites
            # the record), but we still check the row's existence: a PUT over a deleted
            # or nonexistent id must give "not found" (the transport → 404),
            # otherwise an empty PUT would mask the fact's absence.
            with session(self._settings) as conn:
                exists = conn.execute(
                    "SELECT 1 FROM user_facts WHERE id = ? AND deleted_at IS NULL",
                    (id,),
                ).fetchone()
            if exists is None:
                return {"id": id, "changed": False, "hint": HINT_NOT_FOUND}
            return {"id": id, "changed": False}
        with session(self._settings) as conn, transaction(conn):
            row = conn.execute(
                "SELECT id, name, body FROM user_facts "
                "WHERE id = ? AND deleted_at IS NULL",
                (id,),
            ).fetchone()
            if row is None:
                return {"id": id, "changed": False, "hint": HINT_NOT_FOUND}
            conn.execute(
                "UPDATE user_facts SET name = ?, body = ?, "
                "vector_status = 'pending', "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (
                    checked_name if checked_name is not None else row["name"],
                    checked_body if checked_body is not None else row["body"],
                    id,
                ),
            )
            # #41 (FR-3.3, §13.5): an actual edit — a version delta in the same
            # transaction; an empty PUT never reaches here (it returned earlier above),
            # and an edit without content changes creates no version (an empty delta).
            live = conn.execute(
                "SELECT name, body FROM user_facts WHERE id = ?", (id,)
            ).fetchone()
            history.record_change(
                conn,
                history.ENTITY_USER_FACT,
                id,
                history.user_fact_projection(live),
            )
        self._notify_areas_pending()
        return {"id": id, "changed": True}

    # --- reads (arch §3.3) ----------------------------------------------------

    def get(self, id: int) -> dict[str, Any]:
        """Read a fact: `{id, name, body}`; not found → a soft answer with a hint.

                    Only this handle serves the full body (the search — a trimmed
                    `excerpt`).
        """
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT id, name, body FROM user_facts "
                "WHERE id = ? AND deleted_at IS NULL",
                (id,),
            ).fetchone()
        if row is None:
            return {"id": id, "hint": HINT_NOT_FOUND}
        return {"id": int(row["id"]), "name": row["name"], "body": row["body"]}

    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """The area's hybrid search — a "probe" (arch §3.3, substrate §3.5).

                    An `AreaSearch` build for the user area: vec0-KNN over `user_facts_vec`
                    (the vector = `name + body`) + FTS5/BM25 over `user_facts_fts`
                    (`name`/`body`) → RRF, the `deleted_at IS NULL` filter. The output has
                    no bodies: `{results: [{id, name, excerpt}], warning?}`, the `excerpt`
                    trimmed to `user_search_excerpt_chars` (the full body — `get`). Empty —
                    a soft answer with the verbatim §3.7 canon hint (no fact — a valid
                    outcome). An embedder failure does not break the search: FTS-only +
                    `warning` (NFR-3). Isolation: the helper's SQL addresses only the
                    area's tables — notes, skills and terms are not read. `top_k` — the
                    bounds like the notes search (`1..MAX_TOP_K`), a violation —
                    `UserFactValidationError` (a soft transport failure).
        """
        if top_k is not None and not 1 <= top_k <= MAX_TOP_K:
            raise UserFactValidationError(
                f"top_k: expected 1..{MAX_TOP_K}, got {top_k}"
            )
        try:
            found = self._area_search.search(query, top_k)
        except AreaSearchValidationError as exc:  # a query outside the area's domain
            raise UserFactValidationError(str(exc)) from exc
        answer: dict[str, Any] = {
            "results": [
                {
                    "id": int(item["id"]),
                    "name": item["name"],
                    "excerpt": self._excerpt(item["body"]),
                }
                for item in found["results"]
            ],
            "warning": found.get("warning"),
        }
        if not answer["results"]:
            answer["hint"] = HINT_SEARCH_EMPTY
        return answer

    # --- deletion (soft delete, arch §3.4) ------------------------------------

    def delete(self, id: int) -> dict[str, Any]:
        """Soft-delete a fact: `deleted_at` = now; the row/indexes stay alive.

                    Idempotent in meaning: a repeated/nonexistent id → a soft
                    `deleted: False` answer + the §3.7 canon hint (restoration — by the
                    operator, the notes precedent). A deleted fact vanishes from all the
                    area's outputs.
        """
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE user_facts SET deleted_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (id,),
            )
            deleted = bool(cursor.rowcount)
            if deleted:
                # #41 (FR-6.1, §13.5): the soft delete — a version event.
                history.record_event(
                    conn, history.ENTITY_USER_FACT, id, "delete"
                )
        if not deleted:
            return {"id": id, "deleted": False, "hint": HINT_NOT_FOUND}
        # The signal to the areas loop is one for all successful area changes
        # (the lsb-0009-01 brief): a deletion adds no pending, but
        # the signal is harmless and keeps a single worker wakeup point.
        self._notify_areas_pending()
        return {"id": id, "deleted": True}

    # --- internal: dedup -------------------------------------------------------

    def _dedup_candidates(
        self, conn: sqlite3.Connection, name: str, body: str
    ) -> list[tuple[dict[str, Any], float]]:
        """Dedup candidates with trigram similarity, by closeness descending.

                    Candidates — FTS5 `trigram` over `name + body` (the top-3 active
                    records of the ISOLATED area, arch §3.4): the SQL addresses only
                    `user_facts`/`user_facts_fts`. The similarity is computed arithmetically
                    over the normalized text (the substrate's `trigram_similarity`: lower →
                    trim → spaces → yo→ye) — no embedding at write time (FR-3.5). Deleted
                    records are never candidates. No words ≥3 characters — the trigram
                    index cannot search: no candidates (the write goes without a hint).
        """
        expression = match_expression(f"{name} {body}")
        if not expression:
            return []
        weights = ", ".join("1.0" for _ in USER_FACTS_AREA.fts_columns)
        rows = conn.execute(
            f"SELECT t.id, t.name, t.body "
            f"FROM {USER_FACTS_AREA.fts_table} JOIN {USER_FACTS_AREA.table} t "
            f"ON t.id = {USER_FACTS_AREA.fts_table}.rowid "
            f"WHERE {USER_FACTS_AREA.fts_table} MATCH ? AND t.deleted_at IS NULL "
            f"ORDER BY bm25({USER_FACTS_AREA.fts_table}, {weights}), "
            "t.updated_at DESC, t.id DESC LIMIT ?",
            (expression, DEDUP_CANDIDATES),
        ).fetchall()
        target = _fact_text(name, body)
        scored = [
            (
                {"id": int(row["id"]), "name": row["name"]},
                trigram_similarity(target, _fact_text(row["name"], row["body"])),
            )
            for row in rows
        ]
        scored.sort(key=lambda pair: (pair[1], pair[0]["id"]), reverse=True)
        return scored

    # --- internal: shape validation -------------------------------------------

    def _checked_name(self, name: str | None) -> str:
        """The mandatory `name`: non-empty and ≤ `user_name_max_words` words.

                    The mechanism is the same as the notes' `is_valid_title` (the `title`
                    contract): words = `len(name.split())`; the limit — from settings
                    (the canon — `TITLE_MAX_WORDS=5`). A violation →
                    `UserFactValidationError` with the verbatim §3.7 canon hint; returns
                    the normalized `name` without edge spaces.
        """
        normalized = name.strip() if isinstance(name, str) else ""
        if not normalized:
            raise UserFactValidationError(HINT_NAME_REQUIRED)
        if len(normalized.split()) > self._settings.user_name_max_words:
            raise UserFactValidationError(HINT_NAME_LIMIT)
        return normalized

    def _checked_body(self, body: str | None) -> str:
        """The mandatory `body`: non-empty and ≤ `user_body_max_chars` characters.

                    Over the limit — the verbatim §3.7 canon hint ("looks like several
                    facts in one record — split it"): the record is NOT saved.
        """
        normalized = body.strip() if isinstance(body, str) else ""
        if not normalized:
            raise UserFactValidationError(HINT_BODY_REQUIRED)
        if len(normalized) > self._settings.user_body_max_chars:
            raise UserFactValidationError(HINT_BODY_LIMIT)
        return normalized

    # --- internal: output and the signal ---------------------------------------

    def _excerpt(self, body: str) -> str:
        """The body trim for the search output (`excerpt` ≤ the settings limit).

                    The search never serves the full body (arch §3.3): details — via
                    `get`.
        """
        limit = self._settings.user_search_excerpt_chars
        return body if len(body) <= limit else body[:limit]

    def _notify_areas_pending(self) -> None:
        """A signal to the worker: the area changed (wake the `areas` loop at once)."""
        if self._areas_notifier is not None:
            self._areas_notifier()
