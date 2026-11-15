"""NoteService — note CRUD (REQUIREMENTS FR-2…FR-6, ARCHITECTURE §4.1–§4.6;
    Phase 8, Stage 1: save is instant — vectorization and summarization moved to
    the background).

    One service codebase for MCP and REST (ARCH §1). Since Phase 8 save/update do
    NOT encode the text synchronously: the note is written right away with
    vector_status='pending' (the text + chunks in one transaction), the vectors of
    the note and its chunks are caught up by the background worker (§3.4).
    Cosine dedup at write time becomes impossible — it moves to background dedup
    (Stage 2), where recognized duplicates are merged (Stage 2.2): the summarizer
    joins both texts of the pair with the merge prompt, the earlier duplicate is
    updated by the regular update(), the later one goes to trash (soft delete).
    The synchronous path keeps the instant verbatim dedup by text (SQL/FTS, no
    Ollama): it does not catch paraphrases — that is now the background dedup's
    zone. There is no summarization in the synchronous path (mode "B", Phase 4):
    summary is always the fallback truncation, generation is the background
    worker's job (the notifier wakes it right after the write); the embedding loop
    is woken the same way (the vectorization notifier) — a fresh note is encoded
    not after the back-off but immediately (gate 2026-09-19).

    Service-layer response contracts (full; REST returns them as is;
    the MCP layer trims service fields — see Phase 9):
    - save (success)  → {id, stored: True, summary_pending: True} — **without** a
      warning: vectorization is now always background, not "deferred due to a
      failure" (Phase 8);
      Phase 10 (Step 5, US-8): +an optional "similar exists in <ns>" hint when
      another node already holds a verbatim duplicate (cross-node duplicates are
      legitimate — the write is not blocked; the hint is an orientation layer, not
      a degradation)
    - save (duplicate)  → {duplicated: True, id, text, hint} (not created)
    - save (title failure) → TitleValidationError (decision #9): new notes without
      a title or longer than TITLE_MAX_WORDS words are not created — the transport
      gives the client fail+hint (MCP {stored: False, hint}, REST 422); only
      migration notes remain titleless (a direct save call without a title — the
      legacy path, the worker generates it later)
    - get    → {notes: [...]} (an array even for one id; missing/deleted ids are
      skipped; an empty result — a soft answer with a hint; every note carries
      title, Phase 11 — MCP memory_get trims it)
      lsb-0010-04: +chars — the full text's volume in characters (FR-4.1/FR-4.2)
    - list   → {items: [...], total, has_more, next_offset, next_cursor} (without
      full texts; every item carries title, Phase 11, and chars —
      lsb-0010-04) (+a hint when empty) — lsb-0013 page fields
    - update → {id, updated: True, summary_pending: True} | a soft updated: False
      answer
    - delete → {id, deleted: True} | a soft deleted: False answer (soft delete)

    About update **without a warning**: the FR-5 contract provides none — the model
    learns only from save/search answers (§5.3), the worker's pre-vectorization is
    transparent; the same holds for save (Phase 8): pending is the regular state of
    any new note, not a degradation.

    Note titles (Phase 11, decision #9): the client model names the note at write
    time. save — `title` is mandatory: absent (the transport passed None), empty or
    longer than TITLE_MAX_WORDS words → TitleValidationError, the note is NOT
    created (the transport gives fail+hint "set a title ≤5 words"). A direct save
    call without a title (the _UNSET_TITLE sentinel) — the legacy path of
    migration/scripts: the note is written with title=NULL. update — `title` is
    optional: passed and valid → overwrite, not passed → the old one stays; the
    worker's merge path calls update without a title — the earlier note's title is
    not overwritten.

    Pagination/sorting: `ORDER BY updated_at DESC, id DESC` — freshness beats age
    (FR-2); timestamps live with second precision (the ARCH §3.3 DDL format), so
    within one second the id decides "fresher" (the later record is bigger) — a
    deterministic order without sleeps.

    Chunks (Phase 7): the note is stored whole; chunks exist only for vectors.
    In save/update chunks are laid out by the pure token splitter (no Ollama) and
    written to notes_chunks in the same transaction **without vectors** (Phase 8)
    — the background worker builds them (pending is derived anti-join, step 5),
    including reuse: the full-text vector is copied into a single chunk ≤
    CHUNK_SIZE if the notes queue vectorized the note before the chunk queue.
    Dedup — by the full text only (notes_vec): chunk vectors take no part in dedup.
    Soft delete does not touch chunks (trash); the physical chunk cleanup —
    replacement on update, cascade + orphan self-healing on physical deletion
    (step 2).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable
from typing import Any

from app.config import TITLE_MAX_WORDS, Settings
from app.services import history
from app.services.dedup import DeduplicationService, duplicate_response
from app.services.embedding import Embedder, EmbeddingService
from app.services.emit import summary_of
from app.services.freshness import parse_updated_after
from app.services.jobs import queue_snapshot
from app.services.listing import page_fields
from app.services.namespaces import NamespaceService
from app.services.splitter import split_text
from app.storage import chunks, expirations, vectors
from app.storage.db import (
    DEFAULT_NAMESPACE,
    note_chars_limit,
    session,
    transaction,
)
from app.storage.migration import (
    MIGRATION_NOTES_TABLE,
    SOURCE_MIGRATION_SPLIT,
    ensure_migration_schema,
)

# Maximum number of chunks per one memory_get call (lsb-0003, decision of O.,
# 2026-09-08). Listing ceilings (lsb-0013) are env parameters of the surfaces
# (list_max_limit_mcp / list_max_limit_rest), not module constants.
MAX_READ_CHUNKS = 3

# The note title (Phase 11, decision #9): the client model names the note at
# write time; absent/invalid — a write failure with this hint (§5.3).
TITLE_HINT = "set a title ≤5 words"
# no24-01 (3.2.0): the canonical comment-failure text (the 250 domain limit).
COMMENT_HINT = "comment: up to 250 characters"

# The note text soft limit (LSBN-S-1; 3.4.1, LSBN-T-26): the target length
# of a note — settings.max_note_chars (env MAX_NOTE_CHARS, default 2000);
# more — several self-sufficient notes. The number LIVES IN THE CONFIG
# (app.config.Settings — the single place): save/edit validation,
# merging (merge/split), hints, handle descriptions and prompts take it from
# the settings at build time. It equals the schema target: env = the live CHECK
# of the active database (the reconciliation and self-migration F-22 keep parity) — changing the limit
# is an override edit and a restart; the database is carried over by self-migration.
# The NOTE_TEXT_LIMIT literal constant was removed (3.4.1).

# The canonical soft-failure text "the text is longer than the limit": names the limit,
# the actual length and the splitting rule (MCP — fail + hint, REST — 422).
HINT_NOTE_TOO_LONG = (
    "text: {actual} characters — over the {limit}-character note limit; "
    "split it into several self-contained notes, each carrying its own "
    "names, numbers and dates"
)

# LSBN-F-22 (M3): a soft failure of editing/deleting a SOURCE after splitting —
# the content lives in the parts; editing the source would diverge it from them
# (design §4: "after 'split' — a soft failure with a hint"). 'expired' — the same
# failure: the note was closed by the migration, the regular cleanup will remove it.
HINT_SOURCE_SPLIT = (
    "this note was split into self-contained parts during migration; the "
    "source is read-only — edit or delete the parts instead (find them via "
    "memory_search)"
)

# The "title not passed" sentinel: a direct NoteService.save call without a transport
# (migration, scripts, tests) — the legacy path; the note is written with title=NULL
# (only migration notes remain titleless; the title is generated later
# by the worker). The transports (MCP/REST) always pass title explicitly — including
# None when the client model did not name the note: that is a failure (fail+hint).
_UNSET_TITLE: Any = object()

# The "not passed" sentinels for update (lsb-0004-01):
# the contract: "not passed" = keep, null = clear. For text and summary
# we must distinguish "the parameter was not passed" from an explicit None (for summary None =
# regenerate from the current text). The transport (MCP) passes the same
# sentinels when the client did not specify the parameter.
_UNSET_TEXT: Any = object()
# The summary sentinel — a JSON-serializable string: used as the default of the
# memory_update MCP tool parameter; pydantic must not complain about a non-serializable
# default (otherwise PydanticJsonSchemaWarning breaks the JSON logs, NFR-4). The comparison
# in the service — by identity (is), so a collision with a real summary
# is impossible even with an equal string.
_UNSET_SUMMARY: str = "__LSB_UNSET_SUMMARY__"
# The expires_at sentinel (lsb-0004-02): "not passed" =
# keep (update) / a permanent note (save); null = clear the TTL (clear).
# A JSON-serializable string (like _UNSET_SUMMARY): the default of an MCP tool parameter;
# pydantic does not complain about a non-serializable default (NFR-4). The comparison in the service —
# by identity (is); a collision with a real TTL is impossible.
_UNSET_EXPIRES_AT: str = "__LSB_UNSET_EXPIRES_AT__"
# The `comment` sentinel (no24-01, 3.2.0): "not passed" = keep (update) /
# empty (save); null or an empty string = clear. A JSON-serializable string (like
# _UNSET_SUMMARY/_UNSET_EXPIRES_AT): the default of an MCP tool parameter; pydantic
# does not complain about a non-serializable default (NFR-4). The comparison in the service — by
# identity (is); a collision with a real comment is impossible.
_UNSET_COMMENT: str = "__LSB_UNSET_COMMENT__"


def is_valid_title(title: str | None) -> bool:
    """A valid title (decision #9): non-empty and ≤ TITLE_MAX_WORDS words.

            Words = len(title.split()): whitespace variations at the edges and inside
            do not count as words; "empty" — an empty or whitespace-only string.
    """
    if title is None:
        return False
    return bool(title.strip()) and len(title.split()) <= TITLE_MAX_WORDS


class NoteValidationError(ValueError):
    """A violation of domain constraints (text length, batch size, pagination).

            A backstop behind the transport's pydantic schema: an MCP client sending
            garbage is filtered by the tool schema, but the service protects itself
            too.
    """


class TitleValidationError(NoteValidationError):
    """A title-contract violation (Phase 11, decision #9): title absent,
            empty or longer than TITLE_MAX_WORDS words.

            The regular save failure mechanism (like NamespaceError for an
            unregistered node): REST catches the parent NoteValidationError → 422, the
            MCP transport catches TitleValidationError → fail + hint with TITLE_HINT.
            The note is NOT created.
    """


class CommentValidationError(NoteValidationError):
    """A comment-contract violation (no24-01, 3.2.0): the comment is longer
            than COMMENT_MAX_CHARS (250) characters.

            A soft failure (like TitleValidationError): REST catches the parent
            NoteValidationError → 422, the MCP transport → fail + hint. The note
            (including the old comment) is NOT changed — no partial write (FR-5.8).
    """


class NoteService:
    """Note bank CRUD; save/update — an instant write (Phase 8),
            vectorization/summarization — background; delete — soft.
    """

    def __init__(
        self,
        settings: Settings,
        embedding: Embedder | None = None,
        dedup: DeduplicationService | None = None,
        summary_notifier: Callable[[], None] | None = None,
        vector_notifier: Callable[[], None] | None = None,
    ) -> None:
        self._settings = settings
        # DI for tests: a HashEmbedder/fake instead of live Ollama. Since Phase 8
        # the synchronous path does not call the encoder (vectors are the background worker's),
        # the service stays in the constructor as the assembly point for search/health.
        self._embedding: Embedder = (
            embedding if embedding is not None else EmbeddingService(settings)
        )
        self._dedup = dedup if dedup is not None else DeduplicationService(settings)
        # Phase 10: the namespace registry (node validation, subtrees for list).
        self._namespaces = NamespaceService(settings)
        # A signal to the summarization worker (main.py): wake the loop at once when
        # a pending summary appears, instead of waiting out the grown back-off.
        self._summary_notifier = summary_notifier
        # A signal to the vectorization worker (main.py): wake the embedding loop at once
        # when a pending vector appears (save/update/merge_pair), without waiting
        # for the grown back-off (up to 15 min) — a fresh note is encoded and
        # found by search immediately. The event only accelerates: the task lives
        # in the note's status itself (`vector_status='pending'`); the reliability
        # contract does not change.
        self._vector_notifier = vector_notifier
        # #12 (3.2.0): clearing the note's warnings on edit (FR-4.4) —
        # a narrow injection (`WarningsService.resolve_for_subject`), without
        # import cycles; may be absent in unit tests.
        self._warnings_resolver: Callable[..., int] | None = None
        # #30 (3.2.0): the "similar" hint at save (arch §13.4) — a narrow
        # injection of `SearchService.similar_by_text` (FTS+Dice, no model),
        # no search import into notes (no cycle); in unit tests it may
        # be absent — then the `similar` field is not in the answer.
        self._similar_finder: Callable[..., list[dict[str, Any]]] | None = None

    def set_summary_notifier(self, notifier: Callable[[], None]) -> None:
        """Wire the summarization worker's wakeup signal (main.py)."""
        self._summary_notifier = notifier

    def set_vector_notifier(self, notifier: Callable[[], None]) -> None:
        """Wire the worker's embedding-loop wakeup signal (main.py)."""
        self._vector_notifier = notifier

    def set_warnings_resolver(self, resolver: Callable[..., int]) -> None:
        """Wire the note's warning clearing on edit (#12, main.py).

                    `resolver(subject_type, subject_id, *, conn)` — usually
                    `WarningsService.resolve_for_subject`: an edit invalidates the note's
                    open warning in THE SAME transaction.
        """
        self._warnings_resolver = resolver

    def set_similar_finder(
        self, finder: Callable[..., list[dict[str, Any]]] | None
    ) -> None:
        """Wire the similar-search at save (#30, build_services).

                    `finder(text, title, *, exclude_ids, limit, pool, threshold, text_chars)`
                    — usually `SearchService.similar_by_text`: FTS candidates + Dice over
                    significant words, WITHOUT a model call (§13.4). `None` (the default) —
                    the hint is off.
        """
        self._similar_finder = finder

    def _migration_write_route(self, note_id: int) -> dict[str, Any] | None:
        """The note-write route during a migration (LSBN-F-22, M3).

                    Edits of notes waiting for splitting ('waiting'/'quarantine') go into
                    the PREVIOUS database — their text lives there (design §4/§6.3: "edits
                    go into the old database"; the "edit ↔ splitting" race is defused by
                    statuses). The answer:
                    - None — the regular path (no migration / the note is not a migration
                      one / it is already absent in the previous database — a regular
                      not-found);
                    - {"refusal": hint} — a 'split'/'expired' source: a soft failure (the
                      content moved into parts; an edit would diverge from them);
                    - {"settings", "limit"} — write by legacy settings, the text soft limit
                      = the previous database's live CHECK (the active one has the code
                      constant).
        """
        # A lazy import: services.migration pulls summary → notes (a cycle);
        # the precedent — notes.py ↔ ttl.py.
        from app.services.migration import migration_context

        context = migration_context()
        if context is None:
            return None
        with session(self._settings) as conn:
            row = conn.execute(
                f"SELECT status FROM {MIGRATION_NOTES_TABLE} WHERE old_id = ?",
                (note_id,),
            ).fetchone()
        if row is None:
            return None
        if str(row["status"]) in ("нарезана", "expired"):
            return {"refusal": HINT_SOURCE_SPLIT}
        legacy_settings = self._settings.model_copy(
            update={"db_path": context.legacy_path}
        )
        with session(legacy_settings) as legacy_conn:
            live = legacy_conn.execute(
                "SELECT id FROM notes WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            ).fetchone()
            limit = note_chars_limit(legacy_conn)
        if live is None:
            return None  # no source in the previous database — a regular not-found
        return {
            "settings": legacy_settings,
            "limit": int(limit)
            if limit is not None
            else self._settings.max_note_chars,
        }

    # --- FR-4 memory_save (ARCH §4.1) --------------------------------------

    def save(
        self,
        text: str,
        author: str | None = None,
        namespace: str | None = None,
        title: str | None = _UNSET_TITLE,
        expires_at: str | None = _UNSET_EXPIRES_AT,
        comment: str | None = _UNSET_COMMENT,
    ) -> dict[str, Any]:
        """Validation (text, title) → verbatim dedup → INSERT in a transaction.

                    Phase 8 (Stage 1): Ollama is not called in the synchronous path — no
                    vector is built, vectorization moved to the background (the worker's
                    pending queue); cosine dedup at write time is impossible, it moves to
                    background dedup (Stage 2). Only the verbatim duplicate is cut off
                    synchronously (SQL/FTS). The answer is instant and without a warning:
                    vectorization is not "deferred due to a failure" — it is always
                    background.

                    Phase 11 (decision #9): `title` — the mandatory title of a new note
                    (meaningful, ≤ TITLE_MAX_WORDS words): absent at the client model (the
                    transport passed None) or invalid → TitleValidationError — the note is
                    NOT created (the transport gives fail+hint). A direct call without a
                    title (the _UNSET_TITLE sentinel) — the legacy path of
                    migration/scripts: the note is written with title=NULL, the worker
                    generates the title later. The parameter was added LAST: former
                    positional calls (save(text, author), save(text, None, ns)) keep their
                    meaning.

                    Phase 10 (§5.7): `namespace` — the write's target node (only a
                    registered one; not given — `default`); an unregistered node —
                    NamespaceError (the Step-3 transport wraps it into fail + hint).
                    #28 (3.2.0, arch §13.1/§13.2): `None` = "not passed" — the `default`
                    node, and after a successful insert the answer carries a short
                    `namespace_hint` (by the node map and the note's words, WITHOUT a model
                    call). The hint does not move the note and does not block the write; an
                    explicit `namespace` yields no hints. Dedup at save — within the same
                    namespace (cross-node duplicates are legitimate): a verbatim duplicate
                    in one's own node blocks the write (duplicated); a duplicate in a
                    FOREIGN node does not block — the answer gets the hint "similar exists
                    in <ns>" (US-8, the orientation layer).
        """
        self._validate_text(text)
        note_title = self._validated_save_title(title)
        # no24-01 (3.2.0): the comment is optional:
        # a value of 1..COMMENT_MAX_CHARS → write; the sentinel/null/empty → NULL.
        comment_value = (
            None if comment is _UNSET_COMMENT else self._checked_comment(comment)
        )
        ns = self._namespaces.validate_placement(
            namespace if namespace is not None else DEFAULT_NAMESPACE
        )
        # expires_at (lsb-0004-02): passed → parse into an absolute ISO-8601
        # UTC (TTLValidationError — a soft failure BEFORE the write); not passed
        # (the sentinel) → None — a permanent note, nothing into note_expirations.
        expires_value = self._resolve_save_expires(expires_at)
        # Chunks are computed by the pure splitter (~milliseconds, no Ollama) BEFORE
        # the transaction — the transaction itself stays short.
        chunks_data = self._chunks_of(text)
        # The verbatim dedup (one's own ns) and the foreign-hint scan (a foreign ns) — INSIDE
        # BEGIN IMMEDIATE, BEFORE the INSERT: the check and the write in one transaction
        # exclude the TOCTOU race of two concurrent saves of one text (the second
        # writer waits for the first one's COMMIT via busy_timeout and sees the
        # already-inserted row — a verbatim duplicate is impossible). A recognized
        # duplicate: no insert, the transaction commits empty (allowed),
        # the duplicate_response answer after it closes.
        duplicate = None
        foreign_hint = None
        foreign_id: int | None = None
        with session(self._settings) as conn, transaction(conn):
            duplicate = self._dedup.find_by_text(text, namespace=ns, conn=conn)
            if duplicate is None:
                # The foreign-node dedup hint (Phase 10, US-8): a close duplicate in
                # another node is legitimate (cross-node duplicates are not forbidden,
                # §5.7); the write is NOT blocked, but the model learns to
                # orient itself: a hint in the answer.
                foreign = self._dedup.find_by_text(text, namespace=None, conn=conn)
                if foreign is not None:
                    foreign_id = int(foreign["id"])
                    foreign_hint = (
                        f"a similar one exists in «{foreign['namespace']}»; writing here is not "
                        "blocked — cross-node duplicates are legitimate"
                    )
                note_id = self._insert(
                    conn, text, author, vector_status="pending", namespace=ns,
                    title=note_title, expires_at=expires_value, comment=comment_value,
                )
                if expires_value is not None:
                    # A TTL is set: sync the expiration queue (one row
                    # per note, PK note_id).
                    expirations.upsert(conn, note_id, expires_value)
                self._store_chunks(conn, note_id, chunks_data, None)
                # #41 (§13.5): the baseline version is written by THE SAME transaction as the
                # INSERT — the state "the note exists, no history" never happens.
                row = conn.execute(
                    "SELECT text, title, namespace, expires_at, comment FROM notes "
                    "WHERE id = ?",
                    (note_id,),
                ).fetchone()
                history.record_initial(
                    conn, history.ENTITY_NOTE, note_id, history.note_projection(row)
                )
        if duplicate is not None:
            return duplicate_response(duplicate)
        self._notify_summary_pending()
        # The note was written with vector_status='pending' — wake the embedding loop:
        # the vector is built at once, not after the grown back-off.
        self._notify_vector_pending()
        result: dict[str, Any] = {
            "id": note_id,
            "stored": True,
            "summary_pending": True,
        }
        if foreign_hint is not None:
            result["hint"] = foreign_hint  # an orientation layer, not a warning
        # #28: a write without `namespace` — a short node hint (only when
        # the node is not given and the switch is on). A hint failure does not
        # crash the write: the field is simply not added, the event goes to the log (§13.2).
        if namespace is None and self._settings.memory_node_hint_enabled:
            try:
                result["namespace_hint"] = self._namespace_hint(note_title, text)
            except Exception:  # noqa: BLE001 — the hint does not crash the write
                logging.getLogger("app").warning(
                    "save hint failed",
                    extra={"event": "save_hint_failed", "hint": "namespace"},
                )
        # #30 (3.2.0): the "similar" hint — 1–2 textually closest notes
        # (FTS+Dice, WITHOUT a model) for the "extend or create" decision (arch
        # §13.4). Computed ONLY on a successful write (not on duplicated);
        # the fresh note is already in the index — we exclude its id and the id of the foreign
        # verbatim duplicate (it is already given by a `hint` — no signal duplication).
        # A search failure does NOT crash the write: the field is not added, an event goes to the log.
        if self._settings.memory_similar_enabled and self._similar_finder is not None:
            try:
                similar = self._similar_finder(
                    text,
                    note_title,
                    exclude_ids=frozenset(
                        candidate_id
                        for candidate_id in (note_id, foreign_id)
                        if candidate_id is not None
                    ),
                    limit=self._settings.memory_similar_limit,
                    pool=self._settings.memory_similar_pool,
                    threshold=self._settings.memory_similar_threshold,
                    text_chars=self._settings.memory_similar_text_chars,
                )
            except Exception:  # noqa: BLE001 — the hint does not crash the write
                logging.getLogger("app").warning(
                    "save hint failed",
                    extra={"event": "save_hint_failed", "hint": "similar"},
                )
            else:
                if similar:
                    result["similar"] = similar
        return result

    def _namespace_hint(self, title: str | None, text: str) -> str:
        """The node hint for a write without `namespace` (#28, arch §13.1/§13.2).

                    We compare the title's words and the first `memory_node_hint_text_chars`
                    characters of the text with the node map (`NamespaceService.suggest`);
                    with a non-empty list — "saved to `default`; looks like "a" / "b"",
                    otherwise — an explicit "no node matched — see memory_namespaces"
                    (FR-1.4). The string is short (the F-11 budget, ≤ ~120 characters), it
                    does not move the note.
        """
        terms = " ".join(
            filter(
                None,
                [title, text[: self._settings.memory_node_hint_text_chars]],
            )
        )
        candidates = self._namespaces.suggest(
            terms,
            limit=self._settings.memory_node_hint_max_nodes,
            min_score=self._settings.memory_node_hint_min_score,
        )
        if candidates:
            paths = " / ".join(f"«{candidate['path']}»" for candidate in candidates)
            return f"saved to `default`; looks like {paths}"
        return "saved to `default`; no node matched — see memory_namespaces"

    def _insert(
        self,
        conn: sqlite3.Connection,
        text: str,
        author: str | None,
        vector_status: str = "pending",
        namespace: str = "default",
        title: str | None = None,
        expires_at: str | None = None,
        comment: str | None = None,
        origin_note_id: int | None = None,
    ) -> int:
        """INSERT the note row (inside an open transaction).

                    title (decision #9) — validated/normalized by the caller
                    (_validated_save_title); None — the legacy migration path (the worker
                    generates the title).
                    expires_at (lsb-0004-02) — an absolute ISO-8601 UTC (parsed by the
                    caller); None — a permanent note (no TTL).
                    comment (no24-01) — validated/normalized by the caller
                    (_checked_comment); None — no comment.
                    origin_note_id (3.6.2) — the origin group of a split part (the
                    id of the source note, no FK); None — an ordinary note.
        """
        from app.services.migration import migration_context

        values = (
            text,
            title,
            comment,
            author if author else self._settings.author_default,
            vector_status,
            namespace,
            expires_at,
            origin_note_id,
        )
        columns = (
            "text, title, comment, author, vector_status, namespace, expires_at, "
            "origin_note_id"
        )
        ctx = migration_context()
        if ctx is None:  # no migration — the default rowid (max+1), no overhead
            cursor = conn.execute(
                f"INSERT INTO notes ({columns}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
            return int(cursor.lastrowid or 0)
        # A live migration context (design §8.1): the id explicitly, no lower than id_floor.
        # The active database's default max+1 may coincide with the old_id of a waiting
        # source (long tails were not poured) — shadowing in the union,
        # a get/update divergence, a note lost on abort.
        note_id = int(
            conn.execute(
                "SELECT MAX(COALESCE(MAX(id), 0) + 1, ?) FROM notes",
                (ctx.id_floor,),
            ).fetchone()[0]
        )
        cursor = conn.execute(
            f"INSERT INTO notes (id, {columns}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (note_id, *values),
        )
        return int(cursor.lastrowid or 0)

    # --- FR-3 memory_get (batch; the id alias is normalized by the transport) ---

    def get(self, ids: list[int]) -> dict[str, Any]:
        """A direct read of active notes; the order — as in the query.

                    Missing/deleted ids are skipped (FR-3); a repeated id in the query
                    returns the note once. Empty → a soft answer with a hint (§5.3).
                    LSBN-F-22 (M3): during a migration — an id fallback into the previous
                    database (union, design §4): 'waiting'/'quarantine' exist only there;
                    'split'/'expired' are not served — the source was replaced by parts.
                    No context — exactly the old path.
        """
        if not 1 <= len(ids) <= self._settings.max_get_batch:
            raise NoteValidationError(
                f"ids: expected 1..{self._settings.max_get_batch} id, "
                f"got {len(ids)}"
            )
        wanted = list(dict.fromkeys(ids))  # the query order, no duplicates
        placeholders = ",".join("?" * len(wanted))
        with session(self._settings) as conn:
            rows = conn.execute(
                f"SELECT * FROM notes WHERE deleted_at IS NULL "
                f"AND id IN ({placeholders})",
                wanted,
            ).fetchall()
        by_id = {row["id"]: row for row in rows}
        # A lazy import: services.migration pulls summary → notes (a cycle);
        # the precedent — notes.py ↔ ttl.py.
        from app.services.migration import legacy_visible_ids, migration_context

        context = migration_context()
        if context is not None:
            visible = legacy_visible_ids(self._settings)
            legacy_missing = [
                note_id for note_id in wanted if note_id not in by_id
                and note_id in visible
            ]
            if legacy_missing:
                legacy_marks = ",".join("?" * len(legacy_missing))
                with session(
                    self._settings.model_copy(
                        update={"db_path": context.legacy_path}
                    )
                ) as legacy_conn:
                    legacy_rows = legacy_conn.execute(
                        f"SELECT * FROM notes WHERE deleted_at IS NULL "
                        f"AND id IN ({legacy_marks})",
                        legacy_missing,
                    ).fetchall()
                by_id.update({row["id"]: row for row in legacy_rows})
        notes = [
            self._full_note(by_id[note_id]) for note_id in wanted if note_id in by_id
        ]
        if not notes:
            return {
                "notes": [],
                "hint": "none of the requested notes were found "
                "(possibly deleted); browse — memory_list",
            }
        return {"notes": notes}

    def get_chunk(
        self,
        note_id: int,
        query: str | None = None,
        chunk: int | None = None,
        limit: int = 1,
    ) -> dict[str, Any]:
        """Read a note by chunk (lsb-0003, #16): by meaning (`query`) or by
                    number (`chunk`) with pagination (`limit`), 3 chunks in a row at most.

                    Soft failures (a hint — the model corrects itself): an invalid
                    `limit`; `query`+`chunk` together; `chunk` outside [0, total_chunks);
                    the note not found; with `query` — not a single vectorized chunk. The
                    answer is a `chunks` list (not a concatenation): the overlap of
                    neighboring chunks is not duplicated. The total_chunks >= 1 guarantee
                    (FR-6): no chunks — the whole text as chunk 0.
        """
        if not 1 <= limit <= MAX_READ_CHUNKS:
            raise NoteValidationError(
                f"limit: expected 1..{MAX_READ_CHUNKS}, got {limit}"
            )
        if query is not None and chunk is not None:
            raise NoteValidationError(
                "pass either query (by meaning) or chunk (by number) — not both"
            )
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT * FROM notes WHERE deleted_at IS NULL AND id = ?",
                (note_id,),
            ).fetchone()
        if row is None:
            return {
                "chunks": [],
                "hint": "note not found (possibly deleted); browse — memory_list",
            }
        text = row["text"]
        with session(self._settings) as conn:
            chunk_rows = chunks.get_note_chunks(conn, note_id)
        # (idx, text) in text order; total_chunks >= 1 (the FR-6 guarantee).
        ordered = [(idx, ctext) for _cid, idx, ctext, _tok in chunk_rows]
        if not ordered:
            ordered = [(0, text)]
        total_chunks = len(ordered)
        text_by_idx = dict(ordered)

        if query is not None:
            qv = self._embedding.embed(query)
            with session(self._settings) as conn:
                ranked = chunks.rank_chunks(conn, note_id, qv, limit)
            if not ranked:
                return {
                    "chunks": [],
                    "total_chunks": total_chunks,
                    "hint": "no vectorized chunks — try later or read the note "
                    "in full (without query/chunk)",
                }
            items = [
                {"chunk_index": idx, "text": text_by_idx[idx]} for idx in ranked
            ]
        else:  # chunk mode (chunk is not None: query+chunk were already rejected)
            assert chunk is not None
            if not 0 <= chunk < total_chunks:
                return {
                    "chunks": [],
                    "total_chunks": total_chunks,
                    "hint": f"chunk out of range: 0..{total_chunks - 1}; "
                    "to read further use the next chunk number",
                }
            selected = sorted(
                idx for idx in range(chunk, min(chunk + limit, total_chunks))
            )
            items = [
                {"chunk_index": idx, "text": text_by_idx[idx]} for idx in selected
            ]
        chars = sum(len(item["text"]) for item in items)
        return {
            "chunks": items,
            "total_chunks": total_chunks,
            "chars": chars,
            "hint": "",
        }

    # --- FR-2 memory_list -------------------------------------------------

    def list(
        self,
        limit: int | None = None,
        offset: int = 0,
        namespace: str | None = None,
        namespace_exact: bool = False,
        max_limit: int | None = None,
        updated_after: str | None = None,
    ) -> dict[str, Any]:
        """The memory overview: brief summaries by freshness + total (FR-2).

                    Phase 10: namespace — a node/subtree filter (None — global, as before);
                    every item carries its namespace.

                    lsb-0013: `max_limit` is the listing ceiling of the calling surface
                    (None → `list_max_limit_rest`, so existing callers keep working; the
                    MCP surface passes `list_max_limit_mcp`), and the limit error text is
                    the single one shared by all listings. Page fields come from the shared
                    `page_fields` helper and are present in every branch of the answer;
                    ordering, filters and `total` are unchanged.

                    no29-01: `updated_after` (ISO-8601 UTC, an inclusive bound) narrows both
                    the page and `total` by the same `updated_at >= ?` condition; the order
                    `updated_at DESC, id DESC` does not change. An invalid value —
                    FreshnessValidationError (the transport gives 422 / a soft failure with
                    a hint).
                    LSBN-F-22 (M3): during a migration — a merge with the previous database
                    (union): the ones waiting for splitting ('waiting'/'quarantine') exist
                    only there, a common sort by updated_at/id, `total` — the sum of both
                    databases; 'split'/'expired' are not shown (the source was replaced by
                    parts). No context — exactly the old path.
        """
        max_limit = (
            self._settings.list_max_limit_rest if max_limit is None else max_limit
        )
        limit = self._settings.default_list_limit if limit is None else limit
        if not 1 <= limit <= max_limit:
            raise NoteValidationError(
                f"limit: expected 1..{max_limit}, got {limit}"
            )
        if offset < 0:
            raise NoteValidationError(f"offset: expected ≥ 0, got {offset}")
        ns_nodes = self._namespaces.filter_nodes(namespace, namespace_exact)
        if ns_nodes is not None:
            ns_ph = ",".join("?" * len(ns_nodes))
            ns_clause = f" AND namespace IN ({ns_ph})"
            ns_params: list[object] = list(ns_nodes)
        else:
            ns_clause = ""
            ns_params = []
        if updated_after is not None:
            fresh_clause = " AND updated_at >= ?"
            fresh_params: list[object] = [parse_updated_after(updated_after)]
        else:
            fresh_clause = ""
            fresh_params = []
        with session(self._settings) as conn:
            # The active page is taken FROM THE START (LIMIT offset+limit OFFSET 0):
            # the union merge cuts the common page by (updated_at, id) —
            # the top of each database suffices; without a migration this is exactly the old
            # page (a slice of one list equals OFFSET).
            rows = conn.execute(
                "SELECT id, title, namespace, summary, summary_status, author, "
                "created_at, updated_at, expires_at, text "
                "FROM notes WHERE deleted_at IS NULL"
                f"{ns_clause}{fresh_clause} "
                "ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?",
                [*ns_params, *fresh_params, offset + limit, 0],
            ).fetchall()
            total = conn.execute(
                f"SELECT COUNT(*) FROM notes WHERE deleted_at IS NULL"
                f"{ns_clause}{fresh_clause}",
                [*ns_params, *fresh_params],
            ).fetchone()[0]
        # LSBN-F-22 (M3): the union page. From the previous database only the
        # splitting-waiting ones are taken (the others' ids exist in the active one — it serves them);
        # the top (offset+limit) of each database suffices for the common top-N merge.
        # A lazy import: services.migration pulls summary → notes (a cycle);
        # the precedent — notes.py ↔ ttl.py.
        from app.services.migration import legacy_visible_ids, migration_context

        context = migration_context()
        legacy_rows: list = []
        if context is not None:
            visible = legacy_visible_ids(self._settings)
            if visible:
                visible_marks = ",".join("?" * len(visible))
                visible_clause = f" AND id IN ({visible_marks})"
                with session(
                    self._settings.model_copy(
                        update={"db_path": context.legacy_path}
                    )
                ) as legacy_conn:
                    legacy_rows = legacy_conn.execute(
                        "SELECT id, title, namespace, summary, summary_status, "
                        "author, created_at, updated_at, expires_at, text "
                        "FROM notes WHERE deleted_at IS NULL"
                        f"{ns_clause}{fresh_clause}{visible_clause} "
                        "ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?",
                        [
                            *ns_params,
                            *fresh_params,
                            *visible,
                            offset + limit,
                            0,
                        ],
                    ).fetchall()
                    total += legacy_conn.execute(
                        f"SELECT COUNT(*) FROM notes WHERE deleted_at IS NULL"
                        f"{ns_clause}{fresh_clause}{visible_clause}",
                        [*ns_params, *fresh_params, *visible],
                    ).fetchone()[0]
        rows = sorted(
            [*rows, *legacy_rows],
            key=lambda row: (row["updated_at"], row["id"]),
            reverse=True,
        )[offset : offset + limit]
        items = [
            {
                "id": row["id"],
                "title": row["title"],  # Phase 11 (decision #9): may be None (migration)
                "summary": summary_of(row, self._settings),
                # lsb-0010-04 (FR-4.1…FR-4.3): the FULL text's volume in characters —
                # the model decides whether memory_get suffices or a chunk is needed. The text is
                # already selected for the summary fallback — no new DB reads.
                "chars": len(row["text"]),
                "summary_status": row["summary_status"],
                "author": row["author"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "namespace": row["namespace"],
                "expires_at": row["expires_at"],
            }
            for row in rows
        ]
        fields = page_fields(total, offset, len(items))
        if not items and offset == 0:
            return {"items": [], **fields, "hint": "memory is empty"}
        if not items:
            return {
                "items": [],
                **fields,
                "hint": "page beyond the memory: offset ≥ total; reduce offset",
            }
        return {"items": items, **fields}

    # --- FR-5 memory_update (a full rewrite; vectorization — background) ------

    def update(
        self,
        note_id: int,
        text: str | None = _UNSET_TEXT,
        namespace: str | None = None,
        title: str | None = None,
        summary: str | None = _UNSET_SUMMARY,
        expires_at: str | None = _UNSET_EXPIRES_AT,
        comment: str | None = _UNSET_COMMENT,
    ) -> dict[str, Any]:
        """UPDATE a note per the lsb-0004-01 contract: "not passed" = keep,
                    null = clear. text is optional — title/summary/namespace can be edited
                    WITHOUT rewriting the text.

                    Parameter semantics:
                    - `text`: passed → replace the text (text_changed=True); not passed
                      (the _UNSET_TEXT sentinel) → the text is untouched.
                    - `title`: passed and valid → overwrite; not passed (None) → the old
                      one stays (decision #9; the worker's merge path does not overwrite
                      the earlier note's title). Invalid → TitleValidationError.
                    - `summary`: passed (a value) → use as is, do NOT regenerate
                      (summary_status='ok'); not passed (the sentinel) → if text_changed →
                      regenerate, else keep; None → regenerate from the current text.
                    - `namespace`: passed → a move; not passed → keep.

                    expires_at (lsb-0004-02): passed (a value) → set — parse into an
                    absolute ISO-8601 UTC, update notes.expires_at + upsert into
                    note_expirations; not passed (the _UNSET_EXPIRES_AT sentinel) → keep —
                    untouched; None → clear — notes.expires_at=NULL + remove from
                    note_expirations (the note becomes permanent).

                    Summary regeneration rules: a text update without an explicit summary →
                    regenerate (summary='' + 'pending'); a summary update → do not generate
                    (only vectorize); editing summary and text together → do not regenerate
                    (use the passed summary).

                    If text is NOT passed (an edit of title/summary/namespace only):
                    vector_status is untouched, the vector is not dropped, chunks are not
                    recomputed, the grooming markup is not reset — only the requested fields
                    change (plus the links-marker reset on a title edit — lsb-0010-02
                    below). If text is passed — the full regular set of resets as before:
                    vector_status='pending', chunk replacement, dropping the stale vector,
                    the grooming-markup reset (v2.1.1, the 2026-09-05 audit).

                    lsb-0010-02: the links marker `links_at` is reset when `text` OR `title`
                    is edited (the title changes both the vector and the entities/mention
                    rules) — the note re-enters the links-computation queue.

                    Backward compatibility: update(note_id, text) without
                    title/summary/namespace works as before — the text is replaced, the
                    summary is regenerated (summary_pending=True).

                    LSBN-F-22 (M3): during a migration, edits of notes waiting for
                    splitting ('waiting'/'quarantine') go into the PREVIOUS database where
                    their text lives (the text soft limit — that database's live CHECK);
                    after 'split'/'expired' — a soft failure with a hint (see
                    `_migration_write_route`).
        """
        text_changed = text is not _UNSET_TEXT
        route = self._migration_write_route(note_id)
        if route is not None and "refusal" in route:
            return {"id": note_id, "updated": False, "hint": route["refusal"]}
        write_settings = (
            route["settings"] if route is not None else self._settings
        )
        if text_changed:
            self._validate_text(
                text,
                limit=route["limit"]
                if route is not None
                else self._settings.max_note_chars,
            )
        note_title = None if title is None else self._checked_title(title)
        # no24-01 (3.2.0): the comment is edited by a SEPARATE path (sentinel semantics
        # like summary/expires_at): not passed → keep (the column untouched);
        # a value of 1..COMMENT_MAX_CHARS → write; None/empty/whitespace → clear
        # (NULL). The check — BEFORE any write: a soft failure without a partial write.
        comment_touched = comment is not _UNSET_COMMENT
        comment_value = self._checked_comment(comment) if comment_touched else None
        # A quick check before the write: a nonexistent id is untouched.
        with session(write_settings) as conn:
            row = conn.execute(
                "SELECT id, namespace FROM notes "
                "WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            ).fetchone()
        if row is None:
            return self._not_found(note_id)
        ns = self._namespaces.validate_placement(namespace) if namespace is not None \
            else row["namespace"]

        # --- summary: regeneration rules (lsb-0004-01) ---
        # summary not passed (the sentinel): text_changed → regenerate,
        # else keep (summary untouched). summary=None → regenerate
        # from the current text. summary=a value → use it, do not
        # regenerate (summary_status='ok' — the worker will not touch it).
        if summary is not _UNSET_SUMMARY:
            summary_touched = True
            if summary is None:
                summary_value, summary_status = "", "pending"
            else:
                summary_value, summary_status = summary, "ok"
        else:
            summary_touched = text_changed
            if text_changed:
                summary_value, summary_status = "", "pending"
            else:
                summary_value, summary_status = None, None

        # --- expires_at: set/keep/clear (lsb-0004-02) ---
        # passed (a value) → set (parse); not passed (the sentinel) → keep
        # (untouched); None → clear (remove the TTL). expires_value — the target
        # value for notes.expires_at (None = NULL); expires_touched — whether
        # to touch the column and sync note_expirations at all.
        expires_touched = expires_at is not _UNSET_EXPIRES_AT
        if expires_touched:
            # A local import: ttl.py imports NoteValidationError from
            # notes.py (a circular dependency) — impossible at the module level.
            from app.services.ttl import parse_ttl

            expires_value = None if expires_at is None else parse_ttl(expires_at)
        else:
            expires_value = None  # keep — untouched

        # A dynamic UPDATE: only the requested fields are touched.
        sets: list[str] = []
        params: list[object] = []
        # lsb-0010-02: an edit of text or title returns the note to the links
        # computation queue (the links_at marker). The title changes both the vector and the
        # entities/mention rules, so it is reset on a title-only edit too
        # (the grooming markup — as before, only on text).
        # lsb-0011-01: the default-traversal marker (`node_order_at`) is reset there too —
        # a note parsed by the `nodes` job becomes a traversal
        # candidate again on a text/title change.
        if text_changed or note_title is not None:
            sets.append("links_at = NULL")
            sets.append("node_order_at = NULL")
        if text_changed:
            sets.append("text = ?")
            params.append(text)
            sets.append("vector_status = 'pending'")
            sets.append(
                "classified_at = NULL, hint_path = NULL, confidence = NULL"
            )
        if note_title is not None:
            sets.append("title = ?")
            params.append(note_title)
        if comment_touched:
            # Only the comment (and updated_at): text/title/summary/
            # summary_status/vector_status/markup/links_at/node_order_at,
            # chunks and notes_vec are NOT touched (no24-01, FR-5.3).
            sets.append("comment = ?")
            params.append(comment_value)
        sets.append("namespace = ?")
        params.append(ns)
        if summary_touched:
            sets.append("summary = ?")
            params.append(summary_value)
            sets.append("summary_status = ?")
            params.append(summary_status)
        if expires_touched:
            sets.append("expires_at = ?")
            params.append(expires_value)
        sets.append("updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')")
        sql = (
            "UPDATE notes SET " + ", ".join(sets)
            + " WHERE id = ? AND deleted_at IS NULL"
        )
        params.append(note_id)
        chunks_data = self._chunks_of(text) if text_changed else None
        with session(write_settings) as conn, transaction(conn):
            cursor = conn.execute(sql, params)
            updated = cursor.rowcount  # 0 = no such active note
            if updated and expires_touched:
                # The expiration queue sync (lsb-0004-02): set → upsert,
                # clear → delete the row. Only when the UPDATE matched — otherwise
                # we do not write a row for a nonexistent note.
                if expires_value is None:
                    expirations.delete(conn, note_id)
                else:
                    expirations.upsert(conn, note_id, expires_value)
            if text_changed:
                # Phase 7: old chunks (and their vectors) are replaced by new ones in one
                # transaction; Phase 8: the vectors are built by the background worker (pending).
                self._store_chunks(conn, note_id, chunks_data, None)
                # v2.1.1 (pool 3): also reset the STALE full note vector —
                # notes_vec is empty until the worker catches up: in the pending window the note
                # takes part in FTS search only (ARCH §3.3); the old vector over the
                # old text feeds neither the vector search nor cosine dedup.
                vectors.drop(conn, note_id)
            if updated:
                # #41 (§13.5): the edit version — THE SAME transaction; the projection
                # is read after the UPDATE. `summary` is included only on an EXPLICIT
                # pass (§13.3): auto-regeneration creates no versions.
                row = conn.execute(
                    "SELECT text, title, namespace, expires_at, comment, summary "
                    "FROM notes WHERE id = ?",
                    (note_id,),
                ).fetchone()
                history.record_change(
                    conn,
                    history.ENTITY_NOTE,
                    note_id,
                    history.note_projection(
                        row, explicit_summary=summary is not _UNSET_SUMMARY
                    ),
                )
                # #12 (FR-4.4): a note edit invalidates its warnings
                # — in THE SAME transaction (a warning would not survive a rollback).
                if self._warnings_resolver is not None:
                    self._warnings_resolver("note", note_id, conn=conn)
        if not updated:
            return self._not_found(note_id)
        if text_changed:
            # The text edit returned the note to the vectorization queue — wake
            # the embedding loop at once (the task lives in vector_status='pending').
            self._notify_vector_pending()
        summary_pending = summary_touched and summary_status == "pending"
        if summary_pending:
            self._notify_summary_pending()
        return {"id": note_id, "updated": True, "summary_pending": summary_pending}

    # --- FR-6 memory_delete (soft delete) ----------------------------------

    def delete(self, note_id: int) -> dict[str, Any]:
        """Soft delete: `deleted_at` = now; physically the row/index/vector live.

                    lsb-0004-02 (stage 5): on a successful soft delete the row is removed
                    from note_expirations (expirations.delete) in the same transaction — so
                    a deleted note does not "hang" in the queue until the background
                    cleanup. If the note is not found (deleted=False) — the row is
                    untouched.
                    LSBN-F-22 (M3): deleting a note waiting for splitting goes into the
                    previous database (its row lives there); after 'split'/'expired' — a
                    soft failure with a hint.
        """
        route = self._migration_write_route(note_id)
        if route is not None and "refusal" in route:
            return {"id": note_id, "deleted": False, "hint": route["refusal"]}
        write_settings = (
            route["settings"] if route is not None else self._settings
        )
        with session(write_settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE notes SET deleted_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            )
            deleted = cursor.rowcount
            if deleted:
                # Only when the soft delete matched: the TTL row is removed in the same
                # transaction (idempotent — a missing row is fine).
                expirations.delete(conn, note_id)
                # #41 (FR-6.1): the soft delete — a version event in the same
                # transaction; a repeated/nonexistent deletion gives no version.
                history.record_event(
                    conn, history.ENTITY_NOTE, note_id, "delete"
                )
        if not deleted:
            return {
                "id": note_id,
                "deleted": False,
                "hint": "note not found (possibly already deleted)",
            }
        return {"id": note_id, "deleted": True}

    def merge_pair(
        self,
        older_id: int,
        merged_text: str,
        newer_id: int,
    ) -> dict[str, Any]:
        """Merge duplicates in ONE transaction (pool 6): update the earlier note
                    and soft-delete the later one.

                    Before pool 6 the worker called update(earlier) and delete(later)
                    separately — a failure between them (a DB error) left the job pending →
                    a repeat would merge the already-updated earlier note again. Here — a
                    single transaction: either both operations or neither (rollback), no
                    half-state.

                    The regular update-reset set (as in update without a title): the text,
                    chunk replacement, vector_status='pending', the summary reset, the
                    grooming markup, the links marker links_at and the default-traversal
                    marker node_order_at (lsb-0011-01: merging makes the note a traversal
                    candidate anew); updated_at; **the title is not touched** (decision
                    #9), the namespace is preserved
                    (the earlier note stays in its node). The `deleted_at IS NULL` guard on
                    both notes: the operator's soft delete is not overridden.

                    Returns ``{"older_id", "merged", "newer_id", "deleted"}``:
                    ``merged=False`` when there is no active earlier note (the UPDATE did
                    not match) — then nothing is written and the later one is NOT deleted.
                    Notify after the transaction. An internal method (the worker's merge
                    path) — the REST/MCP contracts do not change.
        """
        self._validate_text(merged_text)
        chunks_data = self._chunks_of(merged_text)
        with session(self._settings) as conn, transaction(conn):
            # The earlier one: the full regular update set (the title untouched, the namespace
            # preserved). The deleted_at IS NULL guard — the operator's delete.
            cursor = conn.execute(
                "UPDATE notes SET text = ?, "
                "vector_status = 'pending', "
                "summary = '', summary_status = 'pending', "
                "classified_at = NULL, hint_path = NULL, confidence = NULL, "
                "links_at = NULL, "
                "node_order_at = NULL, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (merged_text, older_id),
            )
            merged = cursor.rowcount > 0
            deleted = False
            if merged:
                # Only if the earlier one is alive: we replace its chunks and drop the stale
                # full vector (like update) — and only then soft-delete the later one.
                self._store_chunks(conn, older_id, chunks_data, None)
                vectors.drop(conn, older_id)
                cursor = conn.execute(
                    "UPDATE notes SET deleted_at = "
                    "strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                    "WHERE id = ? AND deleted_at IS NULL",
                    (newer_id,),
                )
                deleted = cursor.rowcount > 0
                # #41 (FR-7.1, §13.5): both records — in THE SAME transaction.
                # The survivor: a content version with the `merge` source; the absorbed one:
                # a delete version-event (the history is readable even after a soft delete).
                older_row = conn.execute(
                    "SELECT text, title, namespace, expires_at, comment FROM notes "
                    "WHERE id = ?",
                    (older_id,),
                ).fetchone()
                history.record_change(
                    conn,
                    history.ENTITY_NOTE,
                    older_id,
                    history.note_projection(older_row),
                    source="merge",
                )
                if deleted:
                    history.record_event(
                        conn,
                        history.ENTITY_NOTE,
                        newer_id,
                        "delete",
                        source="merge",
                    )
                # #12 (FR-4.4): merging invalidates BOTH notes'
                # warnings — in THE SAME transaction (a pair edit).
                if self._warnings_resolver is not None:
                    self._warnings_resolver("note", older_id, conn=conn)
                    self._warnings_resolver("note", newer_id, conn=conn)
        if merged:
            self._notify_summary_pending()
            # The merge rewrote the earlier note's text and returned it to the
            # vectorization queue — we wake the embedding loop too.
            self._notify_vector_pending()
        return {
            "older_id": older_id,
            "merged": merged,
            "newer_id": newer_id,
            "deleted": deleted,
        }

    def merge_into_parts(
        self,
        older_id: int,
        newer_id: int,
        parts: list[tuple[str, str]],
    ) -> dict[str, Any]:
        """Replace a duplicate pair with a list of parts in ONE transaction (F-23, p. 5).

                    The LSBN-F-23 merge contract: the result is NOT an updated earlier text
                    but several self-sufficient note-parts. In one transaction (pool 6):
                    1) each part is INSERTed as a new note (the part's title, the earlier
                    note's namespace — the duplicate pair lives in one node; vector/summary
                    — pending, by their queues; no TTL — a note with a timer never reaches
                    here, no46-03); the part's content version goes into history at once
                    (source="merge");
                    2) the parts INHERIT the links of BOTH originals (every links row with
                    a side = an original is rewritten onto each part, kind/score preserved)
                    and REFERENCE the originals by an explicit mention link — the part's
                    origin is visible in the environment map;
                    3) BOTH originals go to trash by the regular soft delete (the
                    `deleted_at IS NULL` guard — the operator's delete is not overridden)
                    and stay in history: a delete version-event (source="merge") on each;
                    the TTL row is removed in the same transaction (the delete precedent,
                    lsb-0004-02);
                    4) both originals' warnings are invalidated (#12, in the same
                    transaction).

                    The DB schema does not change (that is F-22): parts are regular notes,
                    trash and history are regular mechanisms. The transaction rollback is
                    the goal: the half-state "a part created, the originals alive" is
                    excluded. The migration_split parts (save_migration_parts) carry
                    origin_note_id — the id of their source note; the parts of THIS
                    merge get no origin group (out of F-23 scope).

                    Returns ``{"older_id", "newer_id", "merged", "parts":
                    [{"id", "title"}, ...], "deleted": {"older", "newer"}}``;
                    ``merged=False`` — no active earlier note (the operator deleted it
                    between the scan and the merge), nothing is written. Notify after the
                    transaction. An internal method (the worker's merge path) — the
                    REST/MCP contracts do not change.
        """
        if not parts:
            raise NoteValidationError("merge_into_parts: пустой список частей")
        prepared: list[tuple[str, str, list[tuple[str, int]]]] = []
        for title, text in parts:
            self._validate_text(text)
            checked_title = self._checked_title(title)
            prepared.append((checked_title or "", text, self._chunks_of(text)))
        with session(self._settings) as conn, transaction(conn):
            older = conn.execute(
                "SELECT namespace FROM notes WHERE id = ? AND deleted_at IS NULL",
                (older_id,),
            ).fetchone()
            if older is None:
                # No active earlier note — the merge did not happen, nothing is written
                # (the merge_pair precedent: the later one is untouched).
                return {
                    "older_id": older_id,
                    "newer_id": newer_id,
                    "merged": False,
                    "parts": [],
                    "deleted": {"older": False, "newer": False},
                }
            source_ids = (older_id, newer_id)
            # The originals' inheritable links — a snapshot BEFORE the parts are inserted.
            inherited = conn.execute(
                "SELECT note_a, note_b, kind, score FROM links "
                "WHERE note_a IN (?, ?) OR note_b IN (?, ?)",
                (*source_ids, *source_ids),
            ).fetchall()
            # 1) The parts — new notes (vector/summary — pending, their own queues).
            created: list[dict[str, Any]] = []
            for title, text, chunks_data in prepared:
                part_id = self._insert(
                    conn,
                    text,
                    None,
                    vector_status="pending",
                    namespace=older["namespace"],
                    title=title,
                )
                self._store_chunks(conn, part_id, chunks_data, None)
                part_row = conn.execute(
                    "SELECT text, title, namespace, expires_at, comment "
                    "FROM notes WHERE id = ?",
                    (part_id,),
                ).fetchone()
                history.record_change(
                    conn,
                    history.ENTITY_NOTE,
                    part_id,
                    history.note_projection(part_row),
                    source="merge",
                )
                created.append({"id": part_id, "title": title})
            # 2) Inheriting the links of both originals + an explicit mention link
            # "part → source" (the part's origin in the environment map).
            # links keeps ONE row per pair (PK note_a<note_b) — the canon
            # "the highest priority mention > entities > cosine": a pair conflict
            # is resolved by a priority upsert (as in LinksService), not IGNORE.
            _kind_upsert = (
                "INSERT INTO links (note_a, note_b, kind, score) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(note_a, note_b) DO UPDATE SET "
                "kind = excluded.kind, score = excluded.score "
                "WHERE (CASE excluded.kind WHEN 'mention' THEN 3 "
                "WHEN 'entities' THEN 2 ELSE 1 END) "
                "> (CASE links.kind WHEN 'mention' THEN 3 "
                "WHEN 'entities' THEN 2 ELSE 1 END)"
            )
            for part in created:
                part_id = int(part["id"])
                for row in inherited:
                    other = (
                        row["note_b"]
                        if row["note_a"] in source_ids
                        else row["note_a"]
                    )
                    if other == part_id:
                        continue
                    conn.execute(
                        _kind_upsert,
                        (
                            min(part_id, other),
                            max(part_id, other),
                            row["kind"],
                            row["score"],
                        ),
                    )
                for source_id in source_ids:
                    conn.execute(
                        _kind_upsert,
                        (min(part_id, source_id), max(part_id, source_id), "mention", 1.0),
                    )
            # 3) Both originals — to trash; the TTL rows too; the delete version-event
            # — into history (readable even after a soft delete, #41).
            deleted = {"older": False, "newer": False}
            for key, source_id in (("older", older_id), ("newer", newer_id)):
                cursor = conn.execute(
                    "UPDATE notes SET deleted_at = "
                    "strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                    "WHERE id = ? AND deleted_at IS NULL",
                    (source_id,),
                )
                deleted[key] = cursor.rowcount > 0
                if deleted[key]:
                    expirations.delete(conn, source_id)
                    history.record_event(
                        conn,
                        history.ENTITY_NOTE,
                        source_id,
                        "delete",
                        source="merge",
                    )
            # 4) #12: the merge invalidates BOTH originals' warnings.
            if self._warnings_resolver is not None:
                self._warnings_resolver("note", older_id, conn=conn)
                self._warnings_resolver("note", newer_id, conn=conn)
        self._notify_summary_pending()
        # The parts wait for vectorization — wake the embedding loop without waiting for the back-off.
        self._notify_vector_pending()
        return {
            "older_id": older_id,
            "newer_id": newer_id,
            "merged": True,
            "parts": created,
            "deleted": deleted,
        }

    def save_migration_parts(
        self,
        old_id: int,
        namespace: str,
        parts: list[tuple[str, str]],
    ) -> dict[str, Any]:
        """Save the splitting parts in one transaction (LSBN-F-22, M2).

                    The migration of a long note (design F-22 §4, phase 4): parts are NEW
                    notes of the active database (modeled on merge_into_parts F-23: the
                    part's title, the source's namespace — passed by the job from the
                    previous database; vector/summary — pending, by their queues; no TTL —
                    no46-03), each part's content version goes into history at once with
                    source="migration_split" — BY THIS source the reverse pour
                    (storage.migration.reverse_pour, M1) recognizes the parts and does not
                    drag them on a rollback. The source is NOT touched: the text lives in
                    the previous database; the note's migration status — the same
                    transaction: migration_notes → 'split' + new_ids (JSON) — resumability
                    and the rollback map (design §3).

                    Returns ``{"old_id", "split": True, "parts": [{"id", "title"}]}``.
                    A part validation failure (`NoteValidationError`) rolls the transaction
                    back whole: the half-state "a part created, the status old" is
                    excluded. Notify after the transaction. An internal method (the
                    migration job) — the REST/MCP contracts do not change.
        """
        if not parts:
            raise NoteValidationError("save_migration_parts: пустой список частей")
        prepared: list[tuple[str | None, str, list[tuple[str, int]]]] = []
        for title, text in parts:
            self._validate_text(text)
            checked_title = self._migration_part_title(title)
            prepared.append((checked_title, text, self._chunks_of(text)))
        new_ids: list[int] = []
        with session(self._settings) as conn, transaction(conn):
            # The migration table may not exist in test DI builds
            # (an active database without a migration) — idempotent, the stamp does not hurt.
            ensure_migration_schema(conn)
            created: list[dict[str, Any]] = []
            for title, text, chunks_data in prepared:
                part_id = self._insert(
                    conn,
                    text,
                    None,
                    vector_status="pending",
                    namespace=namespace,
                    title=title,
                    origin_note_id=old_id,
                )
                self._store_chunks(conn, part_id, chunks_data, None)
                part_row = conn.execute(
                    "SELECT text, title, namespace, expires_at, comment "
                    "FROM notes WHERE id = ?",
                    (part_id,),
                ).fetchone()
                history.record_change(
                    conn,
                    history.ENTITY_NOTE,
                    part_id,
                    history.note_projection(part_row),
                    source=SOURCE_MIGRATION_SPLIT,
                )
                created.append({"id": part_id, "title": title or ""})
                new_ids.append(part_id)
            # The source's status — in the same transaction (design §3): 'split' +
            # new_ids; an upsert — a direct call without a passed job is not lost.
            # parts_json (the split cache) is cleared: the parts are saved, the
            # cache has no consumer left.
            conn.execute(
                f"INSERT INTO {MIGRATION_NOTES_TABLE} "
                "(old_id, status, new_ids, reason, parts_json, updated_at) "
                "VALUES (?, 'нарезана', ?, NULL, NULL, "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now')) "
                "ON CONFLICT(old_id) DO UPDATE SET "
                "status = 'нарезана', new_ids = excluded.new_ids, "
                "reason = NULL, parts_json = NULL, updated_at = excluded.updated_at",
                (old_id, json.dumps(new_ids)),
            )
        self._notify_summary_pending()
        # The parts wait for vectorization — wake the embedding loop without waiting for the back-off.
        self._notify_vector_pending()
        return {"old_id": old_id, "split": True, "parts": created}

    # --- NFR-4 /health ------------------------------------------------------
    def health_counts(self) -> dict[str, int]:
        """Counters for /health: active notes and pending statuses.

                    trash is not counted: there is no background generation for deleted
                    notes (REQUIREMENTS FR-6); the operator's undo returns the note to
                    active — and it counts as pending again until generation. Since Phase 8
                    the vectorization pending is the regular state of every new/updated
                    note.
        """
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS notes_count, "
                "COALESCE(SUM(vector_status = 'pending'), 0) AS pending_vector, "
                "COALESCE(SUM(summary_status = 'pending'), 0) AS pending_summary "
                "FROM notes WHERE deleted_at IS NULL"
            ).fetchone()
        return {
            "notes_count": row["notes_count"],
            "pending_vector": row["pending_vector"],
            "pending_summary": row["pending_summary"],
        }

    def vector_queue_stat(self) -> dict[str, int | None]:
        """The note-vectorization queue snapshot for `/health.queues` (FR-2.2).

                    The same predicate as the legacy `pending_vector` counter (the numbers
                    must match): the pending status of active notes, trash not counted.
        """
        return self._pending_queue_stat("vector_status")

    def summary_queue_stat(self) -> dict[str, int | None]:
        """The note-summarization queue snapshot for `/health.queues` (FR-2.2).

                    The same predicate as the legacy `pending_summary` counter.
        """
        return self._pending_queue_stat("summary_status")

    def _pending_queue_stat(self, status_column: str) -> dict[str, int | None]:
        """The note queue aggregate: the pending count and the oldest's age (s).

                    `status_column` — internal constants only ('vector_status' /
                    'summary_status'), not user input. The age is computed in SQL by the
                    DB clock (`now - updated_at`) — with unavailable models pending does
                    not decrease and the age grows (FR-2.2/FR-2.4).
        """
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS pending, "
                "MAX(CAST(strftime('%s','now') AS INTEGER) - "
                "CAST(strftime('%s', updated_at) AS INTEGER)) "
                "AS oldest_pending_sec FROM notes "
                f"WHERE deleted_at IS NULL AND {status_column} = 'pending'"
            ).fetchone()
        return queue_snapshot(row["pending"], row["oldest_pending_sec"])

    # --- internal -------------------------------------------------------------

    # --- Phase 7: chunk indexing (brief §6) -----------------------------------

    def _chunks_of(self, text: str) -> list[tuple[str, int]]:
        """A note's chunks by the pure token splitter (no external calls): text
                    + the token size — the contents of notes_chunks. The order = idx.
        """
        splits = split_text(
            text,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            chunk_min_target=self._settings.chunk_min_target,
        )
        return [(chunk.text, chunk.tokens) for chunk in splits]

    def _store_chunks(
        self,
        conn: sqlite3.Connection,
        note_id: int,
        chunks_data: list[tuple[str, int]],
        note_vector: list[float] | None,
    ) -> None:
        """Write the note's chunks (in an open transaction); on update — a full
                    replacement: old chunks and their vectors go away with the rows.

                    Phase 8: the synchronous path always passes note_vector=None — the full
                    vector no longer exists at write time, the chunk vectors are built by
                    the background worker (including the reuse of a single chunk from
                    notes_vec). The parameter is kept as an extension point (e.g. for the
                    Phase-8 background paths).
        """
        chunk_ids = chunks.replace_note_chunks(conn, note_id, chunks_data)
        if (
            note_vector is not None
            and len(chunk_ids) == 1
            and chunks_data[0][1] <= self._settings.chunk_size
        ):
            chunks.upsert_vector(conn, chunk_ids[0], note_vector)

    def _notify_summary_pending(self) -> None:
        """A signal to the worker: a note with a pending summary appeared (wake at once)."""
        if self._summary_notifier is not None:
            self._summary_notifier()

    def _notify_vector_pending(self) -> None:
        """A signal to the worker: the note returned to the vectorization queue (pending)."""
        if self._vector_notifier is not None:
            self._vector_notifier()

    @staticmethod
    def _not_found(note_id: int) -> dict[str, Any]:
        return {
            "id": note_id,
            "updated": False,
            "hint": "note not found (possibly deleted)",
        }

    def _validated_save_title(self, title: str | None) -> str | None:
        """The title for save (decision #9): the sentinel → None (the legacy path of a
                    direct call without a transport); passed (including None from the
                    transport — the client did not name the note) → must be valid, else
                    TitleValidationError. Returns the normalized title (edge spaces
                    stripped).
        """
        if title is _UNSET_TITLE:
            return None
        return self._checked_title(title)

    @staticmethod
    def _resolve_save_expires(expires_at: str | None) -> str | None:
        """expires_at for save (lsb-0004-02): the sentinel → None (a permanent
                    note, no TTL); passed → parse into an absolute ISO-8601 UTC
                    (TTLValidationError — a soft failure, the transport gives fail+hint).
        """
        if expires_at is _UNSET_EXPIRES_AT:
            return None
        # A local import: ttl.py imports NoteValidationError from notes.py
        # (a circular dependency) — an import at the module level is impossible.
        from app.services.ttl import parse_ttl

        return parse_ttl(expires_at)

    @staticmethod
    def _checked_title(title: str | None) -> str:
        """A valid title is mandatory (decision #9): empty/longer than
                    TITLE_MAX_WORDS words → TitleValidationError (the transport gives
                    fail+hint). Returns the normalized title (edge spaces stripped).
        """
        if not is_valid_title(title):
            raise TitleValidationError(TITLE_HINT)
        assert title is not None  # is_valid_title already rejected None — for typing
        return title.strip()

    @staticmethod
    def _migration_part_title(title: str | None) -> str | None:
        """The title of a splitting part (release 3.9.0, LSBN-S-11).

                    A valid title — written as is. An EMPTY title — "the title
                    is not generated yet" (the `title` operation answered
                    nothing): the part is saved with a NULL title, so the note
                    goes to the title queue (the notes with `title IS NULL`) and
                    the title is generated there. A title longer than
                    TITLE_MAX_WORDS — the validation error (as before).
        """
        if title is None or not title.strip():
            return None
        return NoteService._checked_title(title)

    def _checked_comment(self, comment: str | None) -> str | None:
        """The note's comment (no24-01): None/empty/whitespace → None (a reset),
                    otherwise the normalized value (edge spaces stripped). The length after
                    strip — 1..COMMENT_MAX_CHARS (250); else CommentValidationError — a
                    soft failure, the note is not changed.
        """
        if comment is None:
            return None
        stripped = comment.strip()
        if not stripped:
            return None
        if not 1 <= len(stripped) <= self._settings.comment_max_chars:
            raise CommentValidationError(
                f"comment: up to {self._settings.comment_max_chars} characters, "
                f"got {len(stripped)}"
            )
        return stripped

    def _validate_text(self, text: str, limit: int | None = None) -> None:
        """1..limit — the note's soft limit (LSBN-S-1; the REQUIREMENTS FR-4/FR-5
                    domain rule).

                    The default upper bound — settings.max_note_chars (env MAX_NOTE_CHARS,
                    3.4.1/LSBN-T-26): the same number as the new databases' CHECK and the
                    self-migration target — one number in one place. Longer than the limit
                    — a failure with the rule "split into several self-sufficient notes";
                    the active database's live CHECK keeps parity (the reconciliation +
                    self-migration F-22). LSBN-F-22 (M3): an edit of a note waiting for
                    splitting goes into the previous database — its live CHECK is the
                    ceiling (passed by `_migration_write_route`). `limit=None` reads the
                    SETTING at call time — services assembled with another Settings
                    snapshot validate by their own value.
        """
        if limit is None:
            limit = self._settings.max_note_chars
        if len(text) > limit:
            raise NoteValidationError(
                HINT_NOTE_TOO_LONG.format(actual=len(text), limit=limit)
            )
        if len(text) < 1:
            raise NoteValidationError(
                f"text: length must be 1..{limit} characters, "
                f"got {len(text)}"
            )

    def _full_note(self, row: sqlite3.Row) -> dict[str, Any]:
        """The memory_get output format (FR-3): the full text + metadata.
                    Phase 10: +namespace (the orientation layer: the model sees where it
                    lies).
                    Phase 11 (decision #9): +title (the REST output to the operator; MCP
                    memory_get trims it by a whitelist — context economy, the full text is
                    there).
                    lsb-0004-02: +expires_at (an absolute ISO-8601 UTC or None — permanent).
                    lsb-0010-04 (FR-4.2): +chars — the FULL text's volume in characters
                    (not bytes/tokens); in chunkless mode it equals the value in
                    memory_search/memory_list. In chunk mode the top-level `chars` of the
                    answer is still the sum of the served chunks (the lsb-0003 contract).
        """
        return {
            "id": row["id"],
            "title": row["title"],
            "text": row["text"],
            "chars": len(row["text"]),  # lsb-0010-04: the full text's volume
            "summary": summary_of(row, self._settings),
            "summary_status": row["summary_status"],
            "author": row["author"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "namespace": row["namespace"],
            "expires_at": row["expires_at"],
            "comment": row["comment"],  # no24-01 (3.2.0): the note's comment
        }