"""LinksService - the note links (lsb-0010, 3.1.0).

    Level 0 - the "lazy graph": the candidates are the KNN over the note's own
    vector (`SearchService.similar_notes`, global, no namespace filter); no new
    tables (FR-1, arch §3.1).

    Level 1 - the `links` table: stored links of three kinds
    (`mention` > `entities` > `cosine`), computed mechanically and idempotently,
    with no model calls (FR-2, arch §3.2-3.4): the cosine - KNN over the ready
    vector; the significant words and the title mentions - an FTS pool + exact
    rule checks in code. A pair's kind is deterministic and NOT "the last
    recomputation" (the gate 2A decision): a pair carries one row with the highest
    kind priority; a note's candidate merges with the stored kind (no
    downgrades), and a pair the note itself no longer finds is confirmed by the
    other side with its own rules - otherwise it stays while the neighbor awaits
    its recomputation (`links_at IS NULL`). So the pair set and their kinds do not
    depend on the queue walk order.
    The `notes.links_at` marker - the computation queue (backfill and increment
    under one selection rule in the `links` job, design 8): `recompute_batch`
    drains a queue batch, `queue_stat` describes it for `/health`, and
    `purge_orphans` is the idle-branch hygiene. The job registers in the framework
    registry (`build_links_job`); the shape is "by interval + the `links` event"
    (the gate 1c decision): the `JOB_LINKS_INTERVAL_SEC` interval stays the
    safety net/backfill, and a note that got a ready vector wakes the loop with
    the `worker._links_event` (notify_links_pending) - level 1 appears
    immediately, not at the next interval. Before waiting, the loop rechecks the
    queue after `clear()` (`queue_empty` - the same selection as
    `recompute_batch`): a signal arriving in the window before `clear()` is not
    lost, and a note awaiting its vector (`vector_status='pending'`) is not in the
    selection - the loop sleeps by back-off, with no busy loop (pool 6, the lost
    wakeup).

    The link output (design 10, arch §3.5): `related` serves level 1 (the `links`
    table) with the kind priority and falls back to level 0 only when the level-1
    cutoffs left no links at all. Level 1 is served only for a note with the
    marker set: `links_at IS NULL` - the links are not computed yet (or were
    reset by a `text`/`title` edit), so the stored rows are stale - an immediate
    level-0 fallback (the gate 4A decision). The transport (`mcp.py`) shapes the
    compact model-facing output; the full contract is `rest.py`.

    The environment map (LSBN-F-16, 3.4.0) - a compressor of the same graph:
    `neighborhood` serves one row per neighbor (`{id, title, namespace, kind}` +
    `direction` for the `mention` kind), with the source level
    (`auto`/`1`/`0`), a page (`limit`/`offset`) and the `total` of the same
    filter; the source is reported as `level_used`, and level 0 is completed with
    `"kind": "cosine"`.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from typing import TYPE_CHECKING, Any

from app.config import Settings
from app.services import jobs
from app.services.jobs import JobSpec, queue_snapshot
from app.services.search import SearchService
from app.storage import vectors
from app.storage.db import session, transaction

if TYPE_CHECKING:  # annotations only: no runtime dependency on the worker
    from app.services.worker import BackgroundWorker

# The link-kind priority: a note pair stores ONE row - the kind with the
# highest priority wins (arch §3.2: mention > entities > cosine).
_KIND_PRIORITY = {"mention": 3, "entities": 2, "cosine": 1}

# Significant words (arch §3.3): lowercase, the tokens [a-zа-яё0-9-]; the FTS
# query takes up to the _FTS_MAX_WORDS longest words - a query cost cap
# (a code constant, not env).
_TOKEN_RE = re.compile(r"[a-zа-яё0-9-]+")
_FTS_MAX_WORDS = 8

# The trigram limit: FTS5/trigram does not search strings under 3 characters -
# a shorter title never becomes an `mention` candidate (arch §3.3).
_TRIGRAM_MIN_CHARS = 3

# The link computation job's name in the journal and the framework registry
# `/health.queues` - the same name.
LINKS_JOB = "links"

# The single selection rule of the link computation queue (arch §3.4): backfill
# and increment drain one selection - active notes with a ready vector and
# an empty `links_at` marker, freshest first. Notes with
# `vector_status='pending'` are not selected (they wait in the `vector` queue); the marker is not lost.
_PENDING_SELECT = (
    "SELECT id FROM notes "
    "WHERE deleted_at IS NULL AND vector_status = 'ok' AND links_at IS NULL "
    "ORDER BY updated_at DESC, id DESC LIMIT ?"
)

# The queue snapshot for `/health.queues` (FR-2.2): the same predicate as the selection;
# the oldest's age - `now - MIN(updated_at)` (via MAX of the differences - otherwise).
_QUEUE_STAT_SQL = (
    "SELECT COUNT(*) AS pending, "
    "MAX(CAST(strftime('%s','now') AS INTEGER) - "
    "CAST(strftime('%s', updated_at) AS INTEGER)) AS oldest_pending_sec "
    "FROM notes "
    "WHERE deleted_at IS NULL AND vector_status = 'ok' AND links_at IS NULL"
)

# The level-1 link output (arch §3.5): a link is read both ways (`note_a`
# OR `note_b`) with a join to `notes`; the cutoffs - the own namespace (links
# mean "between sections"), soft-deleted, and notes of deleted (unknown) nodes.
# The order: the kind priority (mention -> entities -> cosine; a pair stores one
# row with the highest priority) -> `score` DESC -> freshness -> id DESC; the
# ceiling - the LIMIT parameter. The query runs only when the note has the
# `links_at` marker set (the gate 4A decision): without it the stored rows are stale.
# LSBN-F-16: `l.kind AS kind` - the link kind in an environment-map row (the
# `related` list ignores the field); OFFSET - the map page (`map_offset`).
_RELATED_SQL = (
    "SELECT n.id AS id, n.title AS title, n.namespace AS namespace, "
    "l.kind AS kind, n.text AS text FROM links l JOIN notes n ON n.id = "
    "CASE WHEN l.note_a = ? THEN l.note_b ELSE l.note_a END "
    "WHERE (l.note_a = ? OR l.note_b = ?) AND n.deleted_at IS NULL "
    "AND n.namespace != ? "
    "AND n.namespace IN (SELECT path FROM namespaces) "
    "ORDER BY CASE l.kind WHEN 'mention' THEN 3 WHEN 'entities' THEN 2 "
    "ELSE 1 END DESC, l.score DESC, n.updated_at DESC, n.id DESC "
    "LIMIT ? OFFSET ?"
)

# The same filter as `_RELATED_SQL`, without the order and the page - the
# environment map's `total` (LSBN-F-16): the `map_limit`/`map_offset`
# pagination counts "how many in total", not just the served page size.
_RELATED_COUNT_SQL = (
    "SELECT COUNT(*) AS total FROM links l JOIN notes n ON n.id = "
    "CASE WHEN l.note_a = ? THEN l.note_b ELSE l.note_a END "
    "WHERE (l.note_a = ? OR l.note_b = ?) AND n.deleted_at IS NULL "
    "AND n.namespace != ? "
    "AND n.namespace IN (SELECT path FROM namespaces)"
)

# The significant words' stop-word list (ru+en) - a code constant, not env (arch §3.3).
_STOP_WORDS = frozenset(
    {
        # ru
        "чтобы", "только", "можно", "нельзя", "будет", "будут", "были", "было",
        "быть", "бывает", "этого", "этому", "этой", "этом", "этих", "этот",
        "эти", "которая", "который", "которые", "которого", "которой",
        "которым", "которых", "такой", "такая", "такие", "также", "потому",
        "поэтому", "когда", "тогда", "если", "даже", "после", "перед", "между",
        "через", "более", "менее", "больше", "меньше", "всегда", "иногда",
        "сейчас", "теперь", "потом", "почти", "совсем", "очень", "хорошо",
        "лучше", "конечно", "наверное", "возможно", "может", "могут", "другой",
        "другая", "другие", "себя", "себе", "меня", "тебя", "него", "неё",
        "нее", "них", "нами", "вами", "всего", "всё", "все", "ещё", "еще",
        "либо", "какой", "какая", "какие", "каких", "зачем", "куда", "здесь",
        "тоже", "впрочем", "итак", "однако", "хотя", "например", "кроме",
        "вместо", "около", "благодаря", "внутри", "среди", "наконец",
        # en
        "about", "above", "after", "again", "against", "almost", "along",
        "already", "although", "always", "among", "another", "anything",
        "around", "because", "became", "become", "before", "began", "behind",
        "being", "below", "beside", "better", "between", "beyond", "cannot",
        "could", "doing", "during", "either", "enough", "even", "every",
        "everything", "except", "first", "from", "further", "given", "going",
        "have", "having", "here", "herself", "himself", "however", "into",
        "itself", "just", "known", "last", "later", "least", "less", "likely",
        "little", "made", "make", "making", "many", "maybe", "mean", "means",
        "might", "more", "most", "much", "must", "myself", "near", "need",
        "never", "next", "none", "nothing", "often", "once", "only", "other",
        "others", "ourselves", "over", "perhaps", "quite", "rather", "really",
        "right", "same", "seem", "seems", "several", "shall", "should", "since",
        "some", "somebody", "someone", "something", "sometimes", "still",
        "such", "than", "that", "their", "theirs", "them", "themselves",
        "then", "there", "therefore", "these", "they", "thing", "things",
        "think", "this", "those", "though", "through", "thus", "together",
        "toward", "towards", "under", "until", "upon", "used", "using", "very",
        "want", "well", "were", "what", "whatever", "when", "where", "whether",
        "which", "while", "whom", "whose", "will", "with", "within", "without",
        "would", "your", "yours", "yourself",
    }
)


def _prefer(
    found: dict[int, tuple[str, float | None]],
    other_id: int,
    kind: str,
    score: float | None,
) -> None:
    """Store the link kind if it outranks the found one (one row per pair)."""
    current = found.get(other_id)
    if current is None or _KIND_PRIORITY[kind] > _KIND_PRIORITY[current[0]]:
        found[other_id] = (kind, score)


def _merge_kind(
    stored: tuple[str, float | None] | None,
    candidate: tuple[str, float | None],
) -> tuple[str, float | None]:
    """A pair's kind by priority (arch §3.2, the gate 2A decision).

            A pair carries one row, and its kind is NEVER downgraded by a
            recomputation: the union of the existing and the found link is the maximum
            by priority (`mention` > `entities` > `cosine`). The `score` comes from
            the winning kind (NULL for `mention`); at equal priority (the same kind)
            the fresh candidate wins.

    """
    if stored is None or _KIND_PRIORITY[candidate[0]] >= _KIND_PRIORITY[stored[0]]:
        return candidate
    return stored


def _cosine(first: list[float], second: list[float]) -> float:
    """The cosine of two vectors (0.0 - a zero norm; no normalization assumed)."""
    dot = sum(a * b for a, b in zip(first, second))
    norm_first = sum(a * a for a in first) ** 0.5
    norm_second = sum(b * b for b in second) ** 0.5
    if norm_first == 0.0 or norm_second == 0.0:
        return 0.0
    return dot / (norm_first * norm_second)


def _pair_cosine(
    conn: sqlite3.Connection, note_a: int, note_b: int
) -> float | None:
    """A pair's cosine over the ready vectors (`app.storage.vectors`); None - no vector.

            The `vectors` layer is the project's only cosine source (the same metric as
            KNN); the thresholds themselves live in the config
            (`LINK_COSINE_THRESHOLD`).

    """
    first = vectors.get_vector(conn, note_a)
    second = vectors.get_vector(conn, note_b)
    if first is None or second is None:
        return None
    return _cosine(first, second)


class LinksService:
    """The note links: level 0 (the "lazy graph") and level 1 (the `links` table)."""

    def __init__(
        self, settings: Settings, search: SearchService | None = None
    ) -> None:
        self._settings = settings
        # DI for tests and the shared instance from build_services; otherwise - its own.
        self._search = search if search is not None else SearchService(settings)

    #--- the link output: level 1 with a level-0 fallback (arch §3.5) -------

    def related(self, note_id: int, limit: int | None = None) -> list[dict[str, Any]]:
        """Related notes from OTHER namespaces; `[]` is a normal answer.

                    The priority - level 1 (arch §3.5): the stored `links` rows are read
                    both ways, with the output cutoffs (own namespace, soft-deleted, notes
                    of deleted nodes) and the order "kind priority -> `score` DESC ->
                    freshness -> id DESC". Level 1 is served only for a note with the
                    `links_at` marker set: without the marker the links are not computed
                    yet (or were reset by a `text`/`title` edit, the vector pending) - the
                    stored rows are stale, so level 1 is not served at all (the gate 4A
                    decision). The level-0 fallback (the note's vector KNN, §3.1) applies
                    when the level-1 cutoffs left no links or the marker is absent. The
                    ceiling of both branches - `LINK_TOP`; `limit` may only lower it
                    (FR-1.1).

                    No active note with such an id, an empty links table, or a note without
                    a ready vector - an empty list without an error: the absence of links
                    is neither an error nor a reason for a `hint` (FR-3.3).

        """
        top = (
            self._settings.link_top
            if limit is None
            else min(limit, self._settings.link_top)
        )
        if top < 1:
            return []
        own = self._note_state(note_id)
        if own is None:
            return []  # no active note (trash / an unknown id)
        own_namespace, links_at = own
        if links_at is not None:
            stored = self._related_level1(note_id, own_namespace, top)
            if stored:
                return stored
        return self._related_level0(note_id, own_namespace, top)

    def _note_state(self, note_id: int) -> tuple[str, str | None] | None:
        """The active note's state for the output: `(namespace, links_at)`.

                    `None` - no active note (trash/unknown id). The `links_at` marker is
                    read by the same query as the namespace (no extra read per output):
                    `None` - the note's links are not computed yet - level 1 is not served
                    (the gate 4A decision, arch §3.4).

        """
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT namespace, links_at FROM notes "
                "WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            ).fetchone()
        if row is None:
            return None
        return str(row["namespace"]), row["links_at"]

    def _related_level1(
        self, note_id: int, own_namespace: str, top: int
    ) -> list[dict[str, Any]]:
        """The level-1 links (the `links` table, arch §3.5) - the cutoffs and the order in SQL.

                    Reading both ways (`note_a`/`note_b`) with a join to `notes`; the
                    output cutoffs: the own namespace (links mean "between sections"),
                    soft-deleted, notes of deleted nodes. The order and the ceiling are
                    `_RELATED_SQL` (LIMIT): a pair stores one row with the highest kind
                    priority, so `mention` outranks `entities` and `entities` outranks
                    `cosine`. Called only for a note with the `links_at` marker set (see
                    `_note_state`): without it the stored rows are stale. Empty - a normal
                    answer (the table is empty or everything was cut): the caller decides
                    (the level-0 fallback).

        """
        with session(self._settings) as conn:
            rows = conn.execute(
                _RELATED_SQL,
                (note_id, note_id, note_id, own_namespace, top, 0),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "namespace": row["namespace"],
                "chars": len(row["text"]),
            }
            for row in rows
        ]

    def _related_level0(
        self, note_id: int, own_namespace: str, top: int
    ) -> list[dict[str, Any]]:
        """Level 0 (arch §3.1): the full pool after the cutoffs, cut to `top`
                    (the `LINK_TOP` ceiling). The order - by descending closeness.
        """
        if top < 1:
            return []
        return self._related_level0_pool(note_id, own_namespace)[:top]

    def _related_level0_pool(
        self, note_id: int, own_namespace: str
    ) -> list[dict[str, Any]]:
        """The full cutoff level-0 pool (arch §3.1) - the base of `related` (cut to
                    `LINK_TOP`) and of the environment map (cut by the `map_offset` page, F-16).

                    The `LINK_POOL` pool from the KNN over the note's full vector (no
                    namespace filter) -> the cutoffs (the own namespace, soft-deleted,
                    notes of deleted nodes). The order - by descending closeness (the
                    `similar_notes` order).
        """
        candidates = self._search.similar_notes(
            note_id,
            self._settings.link_pool,
            self._settings.link_lazy_threshold,
        )
        if not candidates:
            return []
        with session(self._settings) as conn:
            placeholders = ",".join("?" * len(candidates))
            rows = {
                row["id"]: row
                for row in conn.execute(
                    "SELECT id, title, namespace, text FROM notes "
                    f"WHERE deleted_at IS NULL AND id IN ({placeholders})",
                    [hit_id for hit_id, _ in candidates],
                )
            }
            # The registry nodes: a note of a deleted (unknown) node never enters a link.
            known = {row[0] for row in conn.execute("SELECT path FROM namespaces")}
        result: list[dict[str, Any]] = []
        for hit_id, _cosine in candidates:
            row = rows.get(hit_id)
            if row is None:
                continue  # a soft read: trash vanishes
            if row["namespace"] == own_namespace:
                continue  # links mean "between sections" (FR-1.4b)
            if row["namespace"] not in known:
                continue  # a note of a deleted node
            result.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "namespace": row["namespace"],
                    "chars": len(row["text"]),
                }
            )
        return result

    #--- the note environment map (LSBN-F-16, 3.4.0) --------------------

    def neighborhood(
        self, note_id: int, level: str, limit: int, offset: int
    ) -> dict[str, Any]:
        """The note's environment map: one row per neighbor, with the link kind.

                    The read compressor (LSBN-F-16): instead of the `related` list (each
                    item's volume, "open and look") - a row map `{id, title, namespace,
                    kind}` over which the agent expands 1-2 notes by point reads. `level`
                    sets the source: `"auto"` - like `related` (level 1 with the
                    `links_at` marker set, otherwise with the fallback - level 0); `"1"` -
                    the stored links only (no fallback; also only with the marker - the
                    gate 4A decision: without it the rows are stale); `"0"` - the KNN pool
                    only. The served source is visible in `level_used` (0|1); at level 0 a
                    neighbor has no kind - completed with `"kind": "cosine"` (that is
                    level 0's very method). For `"1"` the source is always 1: both when
                    there are no links and when the `links_at` marker is reset (stale rows
                    are not served, no fallback). The `mention` kind rows get `direction`
                    (`_mention_directions`): "out" - the neighbor's title occurs in this
                    note's text, "in" - the reverse; the symmetric kinds
                    (`entities`/`cosine`) have no direction. The page - `limit`/`offset`;
                    `total` - the size of the SAME filter without the page (the map's
                    pagination). The `limit`/`offset` ranges and the `level` value are
                    checked by the transport (the handle's soft refusals). No links - a
                    normal answer: `neighbors: []` without an error. No active note with
                    such an id - an empty map, `level_used: 0`.

        """
        if level not in ("auto", "1", "0"):
            raise ValueError(f"unknown neighborhood level: {level!r}")
        if limit < 1 or offset < 0:
            return {"neighbors": [], "level_used": 0, "total": 0}
        own = self._note_state(note_id)
        if own is None:
            return {"neighbors": [], "level_used": 0, "total": 0}
        own_namespace, links_at = own
        rows: list[dict[str, Any]] = []
        total = 0
        level_used = 0
        if level == "1":
            level_used = 1  # the source - the stored links, even when they served empty
        if level in ("1", "auto") and links_at is not None:
            rows, total = self._neighborhood_level1(
                note_id, own_namespace, limit, offset
            )
            level_used = 1
        if level != "1" and not rows:
            pool = self._related_level0_pool(note_id, own_namespace)
            total = len(pool)
            rows = pool[offset : offset + limit]
            level_used = 0
        neighbors = [
            {
                "id": row["id"],
                "title": row["title"],
                "namespace": row["namespace"],
                "kind": row["kind"] if level_used == 1 else "cosine",
            }
            for row in rows
        ]
        if level_used == 1:
            self._mention_directions(note_id, neighbors)
        return {"neighbors": neighbors, "level_used": level_used, "total": total}

    def _neighborhood_level1(
        self, note_id: int, own_namespace: str, limit: int, offset: int
    ) -> tuple[list[dict[str, Any]], int]:
        """A level-1 links page + the `total` of the same filter (LSBN-F-16).

                    The same query as `_related_level1` (`_RELATED_SQL`: the output
                    cutoffs, the order "kind priority -> score -> freshness -> id"), with
                    a LIMIT/OFFSET page; a separate `COUNT(*)` of the same filter without
                    the page gives the map's `total`. Called only for a note with the
                    `links_at` marker (the gate 4A decision, see `neighborhood`).

        """
        with session(self._settings) as conn:
            rows = conn.execute(
                _RELATED_SQL,
                (note_id, note_id, note_id, own_namespace, limit, offset),
            ).fetchall()
            total = int(
                conn.execute(
                    _RELATED_COUNT_SQL,
                    (note_id, note_id, note_id, own_namespace),
                ).fetchone()["total"]
            )
        return (
            [
                {
                    "id": row["id"],
                    "title": row["title"],
                    "namespace": row["namespace"],
                    "kind": row["kind"],
                }
                for row in rows
            ],
            total,
        )

    def _mention_directions(
        self, note_id: int, neighbors: list[dict[str, Any]]
    ) -> None:
        """Complete the `direction` of the `mention`-kind rows (LSBN-F-16, in place).

                    "out" - the neighbor's title occurs in this note's text (this note
                    mentions the neighbor); "in" - the reverse (the other side remembers
                    the link: its recomputation found this note's title in the neighbor's
                    text). Cheap: one note-text select per call, then casefold substrings
                    over the titles; the symmetric kinds (`entities`/`cosine`) get no
                    direction - inventing an entity would be dishonest (the question-3
                    decision).

        """
        mention = [row for row in neighbors if row["kind"] == "mention"]
        if not mention:
            return
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT text FROM notes WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            ).fetchone()
        haystack = row["text"].casefold() if row is not None else ""
        for item in mention:
            title = (item["title"] or "").strip()
            item["direction"] = (
                "out" if title and title.casefold() in haystack else "in"
            )

    #--- level 1: the computation and hygiene ---------------------------------------

    def compute_for_note(self, note_id: int) -> int:
        """Recompute one note's level-1 links; return the row count.

                    One transaction (arch §3.3, idempotent): collect the note's candidates
                    -> merge them with the ALREADY stored pairs (the kind by priority; a
                    lost link is unacceptable - the gate 2A decision) -> write in the
                    canonical order `note_a < note_b` -> stamp `notes.links_at`. A repeated
                    computation spawns no rows: a pair is the primary key and the write is
                    complete (not an increment).

                    The link kinds (`kind`; a pair carries the highest priority of
                    mention > entities > cosine): `cosine` - the KNN pool above
                    `LINK_COSINE_THRESHOLD`, `score` - the cosine; `entities` - the shared
                    significant words of `title`+`summary` (>=
                    `LINK_ENTITIES_MIN_COMMON`), `score` - the shared-word share;
                    `mention` - another note's title (>= 3 characters) occurs in this
                    note's `title`/`text`, `score` - NULL. Not a single model call.

                    A pair's kind is NOT "the last recomputation" (the gate 2A decision):
                    the note's candidate merges with the stored kind by priority (no
                    downgrades), and a pair the note itself no longer finds is confirmed by
                    the other side with its own rules (`_neighbour_kind`); while the
                    neighbor awaits its recomputation (`links_at IS NULL`) or lies in
                    trash - the row is not lost. A pair confirmed by nobody whose cosine
                    can be checked is deleted. This way the pair set and their kinds do
                    not depend on the queue walk order.

                    The computation cutoffs: the note itself and the soft-deleted
                    (arch §3.3); the "own namespace" cutoff is NOT done here - it belongs
                    to the output (arch §3.5); moving a note between nodes needs no
                    recomputation.

                    No active note with such an id (trash included) - 0 without changes:
                    a trash note's link rows are untouched (trash stays restorable,
                    arch §3.2). The return - the number of written link rows.

        """
        cosine_pool = self._search.similar_notes(
            note_id,
            self._settings.link_pool,
            self._settings.link_cosine_threshold,
        )
        with session(self._settings) as conn, transaction(conn):
            source = conn.execute(
                "SELECT id, title, text, summary, vector_status FROM notes "
                "WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            ).fetchone()
            if source is None:
                return 0
            found = self._candidate_kinds(conn, source, cosine_pool)
            existing = self._existing_pairs(conn, note_id)
            kept = self._resolve_pairs(conn, source, found, existing)
            conn.execute(
                "DELETE FROM links WHERE note_a = ? OR note_b = ?",
                (note_id, note_id),
            )
            for other_id, (kind, score) in sorted(kept.items()):
                note_a, note_b = sorted((note_id, other_id))
                conn.execute(
                    "INSERT INTO links (note_a, note_b, kind, score) "
                    "VALUES (?, ?, ?, ?)",
                    (note_a, note_b, kind, score),
                )
            conn.execute(
                "UPDATE notes SET links_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ?",
                (note_id,),
            )
        return len(kept)

    def _existing_pairs(
        self, conn: sqlite3.Connection, note_id: int
    ) -> dict[int, tuple[str, float | None]]:
        """A note's stored pairs: `{neighbor: (kind, score)}` (read both ways)."""
        rows = conn.execute(
            "SELECT note_a, note_b, kind, score FROM links "
            "WHERE note_a = ? OR note_b = ?",
            (note_id, note_id),
        ).fetchall()
        return {
            (row["note_b"] if row["note_a"] == note_id else row["note_a"]): (
                row["kind"],
                row["score"],
            )
            for row in rows
        }

    def _resolve_pairs(
        self,
        conn: sqlite3.Connection,
        source: sqlite3.Row,
        found: dict[int, tuple[str, float | None]],
        existing: dict[int, tuple[str, float | None]],
    ) -> dict[int, tuple[str, float | None]]:
        """The final pair set of a note: the kind by priority; a loss is unacceptable.

                    A pair's kind is the union by priority (arch §3.2, the gate 2A
                    decision), not "the last recomputation": the note's candidate merges
                    with the stored kind (`_merge_kind`), no downgrades. A pair the note
                    itself no longer finds is confirmed by the other side with its own
                    rules (`_neighbour_kind`); while the neighbor awaits its recomputation
                    (`links_at IS NULL`) or lies in trash - the row is kept (its own
                    recomputation will decide, arch §3.2/§3.4). A pair confirmed by nobody
                    is deleted, but only when the pair's cosine is checkable at all:
                    without a ready vector the link is not lost - it waits for the vector
                    (arch §3.3).

        """
        kept: dict[int, tuple[str, float | None]] = {}
        # (a) the pairs the note finds itself - merged with the stored kind.
        for other_id, candidate in found.items():
            kept[other_id] = _merge_kind(existing.get(other_id), candidate)
        # (b) the pairs it no longer finds itself - the other side's vote.
        for other_id, stored in existing.items():
            if other_id in kept:
                continue
            neighbour = self._neighbour_state(conn, other_id)
            if neighbour is None:
                continue  # the neighbor is physically deleted: the hygiene will remove the orphaned row
            if neighbour["links_at"] is None or neighbour["deleted_at"] is not None:
                kept[other_id] = stored  # the neighbor awaits recomputation / is in trash - we keep it
                continue
            voice = self._neighbour_kind(conn, source, neighbour)
            if voice is not None:
                kept[other_id] = _merge_kind(stored, voice)
            elif _pair_cosine(conn, source["id"], other_id) is None:
                kept[other_id] = stored  # the vector is not ready - the cosine cannot be checked
            # otherwise: neither side confirms - the pair is deleted
        return kept

    def _neighbour_state(
        self, conn: sqlite3.Connection, other_id: int
    ) -> sqlite3.Row | None:
        """The neighbor's state for checking its rules; None - physically deleted."""
        return conn.execute(
            "SELECT id, title, text, summary, vector_status, links_at, deleted_at "
            "FROM notes WHERE id = ?",
            (other_id,),
        ).fetchone()

    def _neighbour_kind(
        self,
        conn: sqlite3.Connection,
        source: sqlite3.Row,
        neighbour: sqlite3.Row,
    ) -> tuple[str, float | None] | None:
        """The kind by which the neighbor confirms a pair with the note (the same three rules).

                    The neighbor's rules are checked (the gate 2A decision): `mention` -
                    THIS note's title (>= 3 characters) occurs in the neighbor's
                    `title`/`text`; `entities` - >= `LINK_ENTITIES_MIN_COMMON` shared
                    significant words of `title`+`summary` (the rule is symmetric);
                    `cosine` - both vectors are ready and the cosine is not below
                    `LINK_COSINE_THRESHOLD`. None - the neighbor does not confirm the pair.

        """
        source_title = (source["title"] or "").strip()
        if len(source_title) >= _TRIGRAM_MIN_CHARS:
            haystack = f"{neighbour['title'] or ''}\n{neighbour['text']}".casefold()
            if source_title.casefold() in haystack:
                return ("mention", None)
        source_words = self._significant_words(source["title"], source["summary"])
        neighbour_words = self._significant_words(
            neighbour["title"], neighbour["summary"]
        )
        common = source_words & neighbour_words
        if len(common) >= self._settings.link_entities_min_common:
            score = len(common) / max(len(source_words), len(neighbour_words))
            return ("entities", score)
        source_ready = source["vector_status"] == "ok"
        if source_ready and neighbour["vector_status"] == "ok":
            cosine = _pair_cosine(conn, source["id"], neighbour["id"])
            if cosine is not None and cosine >= self._settings.link_cosine_threshold:
                return ("cosine", cosine)
        return None

    def purge_orphans(self) -> int:
        """Tear down the link rows of physically deleted notes; return the row count.

                    Hygiene (arch §3.4) - modeled after the existing cleanups
                    (`worker_jobs`): cheap and rare, in the idle branch of the `links` job
                    (design 8). A soft delete does not remove a note: the row lives while
                    the note is restorable (§3.2).

        """
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "DELETE FROM links WHERE note_a NOT IN (SELECT id FROM notes) "
                "OR note_b NOT IN (SELECT id FROM notes)"
            )
            return cursor.rowcount

    #--- the computation queue: the batch and the snapshot (lsb-0010-03, FR-2.2) ---------------

    def recompute_batch(self, limit: int) -> int:
        """Drain a batch of notes from the computation queue; return the processed count.

                    The single selection rule for backfill and increment (arch §3.4):
                    active notes with a ready vector (`vector_status='ok'`) and an empty
                    marker `links_at IS NULL`, freshest first
                    (`updated_at DESC, id DESC`), at most `limit` per run. Each selected
                    note is recomputed by `compute_for_note` - the `links_at` marker is
                    set in the same transaction, so the next selection will not return it.
                    The first run naturally drains the accumulated base in batches; then
                    the selection is empty - the job sleeps.

                    Notes with `vector_status='pending'` are not in the selection: they
                    are visible in the `vector` queue, no work is lost - the marker waits
                    for a ready vector. The computation calls no models (FR-2.2). The
                    return - the drained batch size.

        """
        with session(self._settings) as conn:
            rows = conn.execute(_PENDING_SELECT, (limit,)).fetchall()
        for row in rows:
            self.compute_for_note(int(row["id"]))
        return len(rows)

    def queue_empty(self) -> bool:
        """Whether the link computation queue is empty (a cheap check; pool 6, the lost wakeup).

                    Before `clear()+wait` the loop rechecks the queue itself, not only the
                    event: a notify could arrive between an empty run and `clear()` -
                    without this check the signal would be erased and the note would wait
                    for the interval/back-off (up to 15 min). The selection is exactly the
                    one `recompute_batch` drains: it comes from `queue_stat` (the same
                    `_QUEUE_STAT_SQL`; the predicate is not duplicated). A note awaiting
                    its vector does not make the queue non-empty - the loop goes to sleep
                    instead of spinning. SQL only, no models (FR-2.2).

        """
        return int(self.queue_stat()["pending"] or 0) == 0

    def queue_stat(self) -> dict[str, int | None]:
        """The link computation queue snapshot for `/health.queues` (FR-2.2).

                    `pending` - active notes with a ready vector and an empty marker
                    (`vector_status='ok' AND links_at IS NULL`) - exactly the selection
                    `recompute_batch` drains; `oldest_pending_sec` - the oldest note's age
                    (`now - MIN(updated_at)`), `null` - the queue is empty. SQL only, no
                    model calls (the shared shape - `jobs.queue_snapshot`).

        """
        with session(self._settings) as conn:
            row = conn.execute(_QUEUE_STAT_SQL).fetchone()
        return queue_snapshot(row["pending"], row["oldest_pending_sec"])

    #--- level 1: the candidates and the rules ------------------------------------

    def _candidate_kinds(
        self,
        conn: sqlite3.Connection,
        source: sqlite3.Row,
        cosine_pool: list[tuple[int, float]],
    ) -> dict[int, tuple[str, float | None]]:
        """The link kinds of the note's candidates: `{id: (kind, score)}`.

                    The candidate pool - the union of the three rules' pools
                    (arch §3.3); the base is not enumerated whole: the KNN pool
                    (`cosine`, also the `entities` pool) UNION the FTS over the
                    significant words of `title`+`summary` (`entities`) UNION the FTS over
                    the words of `title`+`text` (`mention`). Only active notes except the
                    note itself enter the pool; the rules below are checked exactly, so an
                    extra pool candidate never becomes a link.

        """
        source_words = self._significant_words(source["title"], source["summary"])
        candidate_ids = {other_id for other_id, _ in cosine_pool}
        candidate_ids.update(
            self._fts_ids(
                conn,
                source["id"],
                self._fts_expression(
                    self._longest(list(source_words), _TRIGRAM_MIN_CHARS)
                ),
            )
        )
        candidate_ids.update(self._mention_ids(conn, source))
        rows = self._candidate_rows(conn, candidate_ids)
        found: dict[int, tuple[str, float | None]] = {}
        # cosine - the KNN pool's cosine above the threshold (the threshold itself - in similar_notes).
        for other_id, cosine in cosine_pool:
            if other_id in rows:
                found[other_id] = ("cosine", cosine)
        # entities - an exact intersection of the significant-word sets.
        for other_id, row in rows.items():
            words = self._significant_words(row["title"], row["summary"])
            common = source_words & words
            if len(common) < self._settings.link_entities_min_common:
                continue
            score = len(common) / max(len(source_words), len(words))
            _prefer(found, other_id, "entities", score)
        # mention - another note's title occurs in this note's title/text.
        # Mentions by `id` are not caught by the rule: an id match is not a title.
        haystack = f"{source['title'] or ''}\n{source['text']}".casefold()
        for other_id, row in rows.items():
            title = (row["title"] or "").strip()
            if len(title) < _TRIGRAM_MIN_CHARS:
                continue
            if title.casefold() in haystack:
                _prefer(found, other_id, "mention", None)
        return found

    def _mention_ids(
        self, conn: sqlite3.Connection, source: sqlite3.Row
    ) -> list[int]:
        """The `mention` candidate pool: FTS over the words of `title`+`text` (arch §3.3).

                    The trigram searches substrings of 3+ characters, so only such words
                    go into the query (up to the `_FTS_MAX_WORDS` longest) - this is only
                    the pool; the exact check "another note's title occurs in this note's
                    title/text" belongs to `_candidate_kinds`. Mentions by `id` in the text
                    are not caught by the rule: an id match is not a title, and digits
                    under 3 characters are outside the trigram anyway.

        """
        words = self._longest(
            _TOKEN_RE.findall(f"{source['title'] or ''} {source['text']}".lower()),
            _TRIGRAM_MIN_CHARS,
        )
        return self._fts_ids(conn, source["id"], self._fts_expression(words))

    def _fts_ids(
        self, conn: sqlite3.Connection, note_id: int, expression: str | None
    ) -> list[int]:
        """The note pool by an FTS expression: `LINK_POOL`, without the note itself and trash.

                    `CROSS JOIN` (as in dedup) forces FTS into an outer loop - otherwise
                    the planner scans `notes` and runs MATCH on every row. The order -
                    BM25 with an id tie-break: the pool is deterministic when `LINK_POOL`
                    overflows.

        """
        if expression is None:
            return []
        rows = conn.execute(
            "SELECT n.id FROM notes_fts CROSS JOIN notes n ON n.id = notes_fts.rowid "
            "WHERE notes_fts MATCH ? AND n.deleted_at IS NULL AND n.id != ? "
            "ORDER BY bm25(notes_fts), n.id LIMIT ?",
            (expression, note_id, self._settings.link_pool),
        ).fetchall()
        return [row[0] for row in rows]

    def _candidate_rows(
        self, conn: sqlite3.Connection, ids: set[int]
    ) -> dict[int, sqlite3.Row]:
        """The candidates' metadata (a soft read: trash and the deleted vanish)."""
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            "SELECT id, title, summary FROM notes "
            f"WHERE deleted_at IS NULL AND id IN ({placeholders})",
            sorted(ids),
        ).fetchall()
        return {row["id"]: row for row in rows}

    def _significant_words(self, title: str | None, summary: str | None) -> set[str]:
        """The significant words of `title`+`summary` (arch §3.3): lowercase, tokens
                    >= `LINK_ENTITIES_MIN_WORD_CHARS`, without the ru+en stop words.
        """
        tokens = _TOKEN_RE.findall(f"{title or ''} {summary or ''}".lower())
        return {
            token
            for token in tokens
            if len(token) >= self._settings.link_entities_min_word_chars
            and token not in _STOP_WORDS
        }

    @staticmethod
    def _longest(tokens: list[str], min_chars: int) -> list[str]:
        """Up to the `_FTS_MAX_WORDS` longest unique tokens (length >= min_chars).

                    The FTS query cost cap: only the length top enters the expression -
                    deterministic (`-len`, then the token itself).

        """
        unique = {token for token in tokens if len(token) >= min_chars}
        return sorted(unique, key=lambda token: (-len(token), token))[:_FTS_MAX_WORDS]

    @staticmethod
    def _fts_expression(words: list[str]) -> str | None:
        """An FTS5 expression: quoted substrings joined by OR (the trigram contract).

                    None - an empty word list: the trigram has nothing to search. Quotes
                    are replaced with doubles (inside an FTS5 phrase that is the escape),
                    as in search.

        """
        unique = list(dict.fromkeys(words))
        if not unique:
            return None
        return " OR ".join(
            f'"{word.replace(chr(34), chr(34) * 2)}"' for word in unique
        )


#--- the link computation job builder (the lsb-0014 framework) --------------------------


def build_links_job(worker: BackgroundWorker, settings: Settings) -> JobSpec:
    """The `links` job: the background level-1 computation; the queue - the `links_at` marker.

            The shape "by interval + the `links` event" (the gate 1c decision): a run
            drains a `JOB_LINKS_BATCH` batch from the queue (freshest first), progress
            resets the back-off, an empty selection - the `purge_orphans` hygiene
            (idle_hook) and the wait (`JOB_LINKS_INTERVAL_SEC` with back-off), but the
            `notify_links_pending` signal (a note got its ready vector -
            process_pending) wakes the loop immediately. The queue is the note's state
            (`links_at IS NULL`), so the event only accelerates: no work is lost.
            `JOB_LINKS_ENABLED=false` does not start the job, but the queue stays
            visible in `/health` (the registry keeps it). The link computation calls
            no models (FR-2.2): the cosine - a ready vector; entities/mention - FTS.

            The queue recheck (`queue_empty`) is set - the shape closes the narrow
            lost-wakeup race (pool 6): an empty run -> `idle_hook` -> the event's
            `clear()`, and a signal about a fresh vector arriving in that window would
            otherwise be erased (`clear()` does not remember it). `queue_empty` looks
            at exactly the `recompute_batch` selection
            (`vector_status='ok' AND links_at IS NULL`): a non-empty queue - the loop
            continues the run immediately (no sleep); an empty one - it sleeps until
            an event/timeout. A note awaiting its vector
            (`vector_status='pending'`) is not in the selection - the queue is empty
            and the loop goes to sleep by back-off: no busy loop; the work arrives
            with the vectorization event (`notify_links_pending`).

            The service is built over the worker's shared embedder (no separate
            client): the link computation calls no encoding, but `LinksService`
            requires `SearchService` for the `cosine` KNN pool.

    """
    links = LinksService(settings, search=SearchService(settings, worker._embedding))
    batch = settings.job_links_batch

    async def process() -> int:
        # The synchronous SQL/computation goes to a thread - the event loop is not
        # occupied (like the worker loops: process_* are synchronous).
        return await asyncio.to_thread(links.recompute_batch, batch)

    return JobSpec(
        name=LINKS_JOB,
        queue=LINKS_JOB,
        interval_sec=settings.job_links_interval_sec,
        batch=batch,
        enabled=settings.job_links_enabled,
        process=process,
        queue_empty=links.queue_empty,
        wait_event=worker._links_event,
        idle_hook=links.purge_orphans,
        queue_stat=links.queue_stat,
    )


# The job registration in the framework registry (FR-1.1): a job lives in its
# own module (one line in `JOB_BUILDERS`); the import is one-way - links -> jobs, no cycle.
jobs.JOB_BUILDERS += (build_links_job,)
