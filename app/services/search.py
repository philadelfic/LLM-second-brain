"""SearchService — hybrid search (Phase 3, ARCHITECTURE §4.2; Phase 7 — vectors
    are built per chunk).

    Two candidate sources, a Reciprocal Rank Fusion merge, one top_k:
    - vector (Phase 7): KNN over chunks (top-50) → aggregation to notes: a note's
      best chunk sets the closeness (cosine = the best chunk's, comparable with
      SCORE_THRESHOLD); notes without READY chunks (inherited before Phase 7 and
      pending at the worker) — a fallback to the full-text vector (notes_vec) as in
      Phase 3. The full text and summary never affected search quality before —
      they do not now: vectors do not care, and summary is output-only;
    - full text: FTS5/BM25 top-50 (trigram: Russian word forms, exact tokens) —
      catches identifiers/dates; not subject to the threshold (REQUIREMENTS FR-1).
      Query words — OR of substrings (BUG-001: AND lost a note missing one word).
    - RRF is robust to incomparable scales (cosine vs BM25): score(d) =
      Σ_sources 1/(RRF_K + rank_source), rank from 1. The vector side after
      aggregation is one list, ranked as in Phase 3.

    Degradation (NFR-3): a query-encoding failure → FTS-only search + `warning`
    (the §5.3 teaching text); NOT a hard error — search does not break on an
    external dependency. Candidates whose best vector hit is below the threshold
    fly out before the merge.

    The FR-1 output: summary/snippet, without the full text (memory_get addresses
    it). Phase 11 (decision #9): +title — the real notes.title value (None remains
    only for titleless migration notes).
    lsb-0010-04 (FR-4.1/FR-4.2): +chars — the note's FULL text volume in characters
    (not snippet/summary); the value for one note matches memory_list/memory_get
    without chunks.
    The snippet — from the BEST chunk (the chunk's first SNIPPET_CHARS characters):
    the model sees a relevant fragment of a long note, not its beginning; for
    fallback candidates — from the full text, as in Phase 3. Ties (an equal
    rrf_score) — by updated_at DESC, id DESC: fresher is more useful (in the spirit
    of FR-2), deterministic without randomness.
    The output is the full service-layer contract (REST returns it as is);
    the MCP layer trims service fields — see Phase 9.

    lsb-0001-01 (Step 3): the hybrid's FTS side — with bm25 weights: a substring
    match in the title ranks above the same match in the text (FR-1.2;
    the TITLE_FTS_WEIGHT weight — the R3 gate decision). Plus the separate
    title mode SearchService.search_title: a strict search over titles
    (the D3 decision — an occurrence in the text does not match), a compact output
    and score steps (the D2 decision — FR-2.1/FR-2.2).
"""

from __future__ import annotations

import sqlite3
from contextlib import ExitStack
from typing import Any

from app.config import Settings
from app.services.embedding import Embedder, EmbeddingError, EmbeddingService
from app.services.emit import snippet, summary_of
from app.services.freshness import parse_updated_after
from app.services.namespaces import NamespaceService
from app.services.ranking import (
    apply_boosts,
    fuse_rrf,
    match_expression,
    significant_tokens,
    utc_now_iso,
)
from app.storage import chunks, vectors
from app.storage.db import session

# The contract's upper bound (REQUIREMENTS FR-1: top_k 1..20); env sets
# only the default — DEFAULT_TOP_K, so MAX_TOP_K is not configurable.
MAX_TOP_K = 20

# Candidates from each source before the merge (ARCH §4.2, REQUIREMENTS §5.4).
CANDIDATE_LIMIT = 50

# Splitting the query's compound tokens (BUG-001): "open-webui" → "open" +
# "webui" — FTS catches texts where the spelling differs ("Open WebUI").
# The query's FTS expression — the shared app.services.ranking.match_expression
# primitive (since release 3.0.0 the in-area search uses it too).

# A query-encoding failure: the search degrades to FTS-only + a warning (§5.3).
WARNING_FTS_ONLY = (
    "search without semantics: query encoding failed (vectorization "
    "unavailable), output is full-text only — paraphrases may be missed"
)

HINT_NO_RESULTS = (
    "nothing found for the query; rephrase more broadly "
    "(searches by substrings from 3 characters) or browse via memory_list"
)

HINT_SHORT_QUERY = (
    "each query word is shorter than 3 characters — trigram does not search them; "
    "add meaningful words"
)

# The bm25 weight of the notes_fts title column (lsb-0001-01, FR-1.2): a match
# of a substring in the title yields a LOWER badness → higher in ORDER BY badness,
# than the same match in the text (the notes_fts columns: title, text; the text weight —
# 1.0). The value is tuned by tests (the R3 gate decision); there is no env variable.
TITLE_FTS_WEIGHT = 2.0

# An empty title mode (FR-2.2): a compact output; the regular semantic search
# searches wider — the hint teaches the model to pick the right tool.
HINT_NO_RESULTS_TITLE = (
    "nothing found by title; check the title substring or browse "
    "via memory_list; regular semantic search also searches by text"
)

# An empty separate COMMENT search (no24-02, arch §13.6): comments are searched only
# here — the semantic search does not see them; the hint teaches the model to tell the handles apart.
HINT_NO_COMMENTS = (
    "no notes with such a comment; comments are searched separately — "
    "regular search looks at note text only"
)


class SearchValidationError(ValueError):
    """A violation of the query's domain constraints (length, top_k)."""


class SearchService:
    """The hybrid of vec0 + chunk-vec0 (aggregation) + FTS5 → RRF; the embedder is DI."""

    def __init__(
        self, settings: Settings, embedding: Embedder | None = None
    ) -> None:
        self._settings = settings
        # DI for tests: a deterministic HashEmbedder instead of the network.
        self._embedding: Embedder = embedding if embedding is not None else EmbeddingService(settings)
        # Phase 10: the namespace registry (node validation, subtrees).
        self._namespaces = NamespaceService(settings)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        namespace: str | None = None,
        namespace_exact: bool = False,
        updated_after: str | None = None,
    ) -> dict[str, Any]:
        """Hybrid search (ARCH §4.2 + Phase 7); the FR-1 output without the full text.

                    Phase 10: `namespace` — a hierarchy node; a filter over its subtree
                    (the node + registered leaves; `namespace_exact` — the node itself
                    only). None — a global search (RRF as before). An unregistered node —
                    NamespaceError (the Step-3 transport wraps it into fail + hint).
                    Every result carries its `namespace` — the model sees where it was
                    found (orientation layer #3).
                    no29-01: `updated_after` — a freshness filter (ISO-8601 UTC, an
                    inclusive bound): only notes with `updated_at >= value` enter the
                    output; it applies to both hybrid sides, the sort does not change.
                    An invalid value — FreshnessValidationError (a soft transport failure).
        """
        query = self._validate_query(query)
        top_k = self._default_top_k() if top_k is None else top_k
        if not 1 <= top_k <= MAX_TOP_K:
            raise SearchValidationError(
                f"top_k: expected 1..{MAX_TOP_K}, got {top_k}"
            )
        query_vector = self._query_vector(query)
        expression = self._match_expression(query)
        ns_nodes = self._namespaces.filter_nodes(namespace, namespace_exact)
        fresh = (
            parse_updated_after(updated_after)
            if updated_after is not None
            else None
        )
        # LSBN-F-22 (M3): the union read — during a migration candidates are taken
        # from the PREVIOUS database too ('waiting'/'quarantine': their only copy lives there;
        # 'split'/'expired' — excluded, the source was replaced by parts).
        # A lazy import: services.migration pulls summary → notes (a cycle),
        # the precedent — notes.py ↔ ttl.py. A None context — the old paths,
        # zero overhead: no second connection, no extra SQL.
        from app.services.migration import legacy_visible_ids, migration_context

        context = migration_context()

        with session(self._settings) as conn, ExitStack() as stack:
            vector_hits = (
                self._vector_candidates(conn, query_vector, ns_nodes, fresh)
                if query_vector is not None
                else []
            )
            fts_rows = (
                self._fts_candidates(conn, expression, ns_nodes, fresh)
                if expression
                else []
            )
            legacy_conn = None
            if context is not None:
                visible = legacy_visible_ids(self._settings)
                if visible:
                    legacy_conn = stack.enter_context(
                        session(
                            self._settings.model_copy(
                                update={"db_path": context.legacy_path}
                            )
                        )
                    )
                    if query_vector is not None:
                        vector_hits += [
                            hit
                            for hit in self._vector_candidates(
                                legacy_conn, query_vector, ns_nodes, fresh
                            )
                            if hit[0] in visible
                        ]
                    if expression:
                        fts_rows += [
                            row
                            for row in self._fts_candidates(
                                legacy_conn, expression, ns_nodes, fresh
                            )
                            if row["id"] in visible
                        ]

            # --- the RRF merge: score(d) = Σ 1/(RRF_K + rank) ----------------
            # The vector source — an already-aggregated list of notes
            # (the best chunk set the cosine and the snippet); Phase 3 — the full vector.
            # The merge — the shared app.services.ranking.fuse_rrf primitive (the same
            # math as the in-area search, substrate 3.0.0); during a
            # migration the ranked lists of both databases are merged by it too.
            cosine_by_id: dict[int, float] = {}
            snippet_source: dict[int, str | None] = {}
            for note_id, cosine, chunk_text in vector_hits:
                cosine_by_id[note_id] = cosine
                snippet_source[note_id] = chunk_text
            scores = fuse_rrf(
                (
                    [note_id for note_id, _cosine, _text in vector_hits],
                    [row["id"] for row in fts_rows],
                ),
                self._settings.rrf_k,
            )

            rows = self._fetch_rows(conn, list(scores))
            if legacy_conn is not None:
                # 'Waiting' notes exist only in the previous database: the metadata
                # are fetched from it (the active copy for shared ids — above).
                missing = [note_id for note_id in scores if note_id not in rows]
                if missing:
                    rows.update(self._fetch_rows(legacy_conn, missing))
            # --- ranking boosts (release 3.4.0, LSBN-F-14) -------------------
            # Only the OUTPUT ORDER: the sort key in _merge is the boosted
            # score; the item's rrf_score field stays the raw RRF (the contract
            # §4.2.6 / the F-13 headers do not change). The switch —
            # RANKING_BOOSTS_ENABLED; when false, no extra SQL over links runs.
            # LINKED_BOOST — one pass over the candidate list BEFORE the sort
            # and the top cut. The title mode and comment_search are not touched by boosts.
            order: dict[int, float] = scores
            if scores and self._settings.ranking_boosts_enabled:
                linked_ids = self._linked_ids(conn, list(rows))
                if legacy_conn is not None:
                    # The 'waiting' notes' links stayed in the previous database.
                    linked_ids |= self._linked_ids(legacy_conn, list(rows))
                order = apply_boosts(
                    scores,
                    {
                        note_id: {
                            "query": query,
                            "title": row["title"],
                            "updated_at": row["updated_at"],
                            "has_links": note_id in linked_ids,
                        }
                        for note_id, row in rows.items()
                    },
                    now=utc_now_iso(),
                    enabled=True,
                )
        results = self._merge(scores, cosine_by_id, snippet_source, rows, order)[
            :top_k
        ]
        warning = None if query_vector is not None else WARNING_FTS_ONLY
        if not results:
            hint = (
                HINT_SHORT_QUERY
                if query_vector is None and expression is None
                else HINT_NO_RESULTS
            )
            return {"results": [], "warning": warning, "hint": hint}
        return {"results": results, "warning": warning}

    def search_title(
        self,
        query: str,
        top_k: int | None = None,
        namespace: str | None = None,
        namespace_exact: bool = False,
        updated_after: str | None = None,
    ) -> dict[str, Any]:
        """Title search of memory_search (lsb-0001-01): FR-2.1/FR-2.2 + the owner's
                    D2/D3 decisions.

                    D3 — strictly over titles: a substring matches ONLY the title;
                    an occurrence in the text does not match here (the regular hybrid
                    search finds such text). The matching — on the Python side
                    (`needle in title.lower()`), not LIKE/FTS: SQLite's lower() is
                    case-insensitive only for ASCII, Cyrillic is not covered (and
                    trigram FTS does not search it). The scale — the same argument as for
                    brute-force vec0 (NFR-5): up to 50,000 notes — one scan and a linear
                    pass over the titles only.

                    Ranking (D2, steps): 1.0 — the query matched the title WHOLLY
                    (case-insensitively), 0.5 — a substring. Order: exact ones higher;
                    among partial ones — an occurrence closer to the title's start is
                    higher; a tie — updated_at DESC, then id DESC (in the spirit of
                    _merge). The output is compact (FR-2.2): id, title, namespace,
                    summary, score — without snippet/rrf_score/the full text (the summary —
                    via summary_of: pending → the §5.5 fallback truncation, ready — as
                    is).

                    Validations like semantic: the query length 1..max_query_chars, top_k
                    1..MAX_TOP_K (the default — default_top_k). The trigram rule "words
                    ≥3 chars" does NOT apply here — this is not FTS. `namespace` — a
                    hierarchy node: the subtree (`namespace_exact` — the node itself
                    only), as in search(); an unregistered node — NamespaceError,
                    propagated as is (the MCP transport will hook in at step 03 of release
                    2.2 and wrap it into fail + hint, like search). A warning never happens
                    in this mode — no embedding is used. no29-01: `updated_after` — a
                    freshness filter (the same inclusive bound as in search); ranking and
                    output do not change. LSBN-F-22 (M3): during a migration the titles of
                    notes waiting for splitting are searched in the previous database too
                    (union, design §4).
        """
        query = self._validate_query(query)
        top_k = self._default_top_k() if top_k is None else top_k
        if not 1 <= top_k <= MAX_TOP_K:
            raise SearchValidationError(
                f"top_k: expected 1..{MAX_TOP_K}, got {top_k}"
            )
        ns_nodes = self._namespaces.filter_nodes(namespace, namespace_exact)
        ns_clause = ""
        ns_params: list[object] = []
        if ns_nodes:
            ns_clause = " AND namespace IN (" + ",".join("?" * len(ns_nodes)) + ")"
            ns_params = list(ns_nodes)
        fresh_clause = ""
        fresh_params: list[object] = []
        if updated_after is not None:
            # no29-01: the same inclusive freshness bound as in search().
            fresh_clause = " AND updated_at >= ?"
            fresh_params = [parse_updated_after(updated_after)]
        with session(self._settings) as conn:
            # text — the summary_of input only (the §5.5 pending fallback); the full text
            # does not go into the output (the compact FR-2.2 contract).
            rows = conn.execute(
                "SELECT id, title, namespace, text, summary, summary_status, "
                "       created_at, updated_at "
                "FROM notes WHERE deleted_at IS NULL AND title IS NOT NULL"
                f"{ns_clause}{fresh_clause}",
                [*ns_params, *fresh_params],
            ).fetchall()
        # LSBN-F-22 (M3): union — the titles of notes waiting for splitting live in the
        # previous database ('split'/'expired' are not shown — the source
        # was replaced by parts). A lazy import — a cycle (see search()).
        from app.services.migration import legacy_visible_ids, migration_context

        context = migration_context()
        if context is not None:
            visible = legacy_visible_ids(self._settings)
            if visible:
                marks = ",".join("?" * len(visible))
                with session(
                    self._settings.model_copy(
                        update={"db_path": context.legacy_path}
                    )
                ) as legacy_conn:
                    rows += legacy_conn.execute(
                        "SELECT id, title, namespace, text, summary, "
                        "       summary_status, created_at, updated_at "
                        "FROM notes WHERE deleted_at IS NULL "
                        f"AND title IS NOT NULL AND id IN ({marks})"
                        f"{ns_clause}{fresh_clause}",
                        [*visible, *ns_params, *fresh_params],
                    ).fetchall()
        needle = query.lower()
        ranked: list[tuple[float, int, sqlite3.Row]] = []
        for row in rows:
            lowered = row["title"].lower()
            if needle not in lowered:
                continue
            ranked.append(
                (1.0 if lowered == needle else 0.5, lowered.find(needle), row)
            )
        # The order (decision D2): score DESC → the occurrence position ASC →
        # updated_at DESC → id DESC. Stable passes: the last key is the
        # main one; updated_at/id are not negated — DESC via a reverse=True pass.
        ranked.sort(key=lambda hit: hit[2]["id"], reverse=True)
        ranked.sort(key=lambda hit: hit[2]["updated_at"], reverse=True)
        ranked.sort(key=lambda hit: hit[1])
        ranked.sort(key=lambda hit: hit[0], reverse=True)
        results = [
            {
                "id": row["id"],
                "title": row["title"],
                "namespace": row["namespace"],
                "summary": summary_of(row, self._settings),
                "score": score,
            }
            for score, _position, row in ranked[:top_k]
        ]
        if not results:
            return {"results": [], "hint": HINT_NO_RESULTS_TITLE}
        return {"results": results}

    def comment_search(
        self,
        query: str,
        top_k: int | None = None,
        namespace: str | None = None,
        namespace_exact: bool = False,
        updated_after: str | None = None,
    ) -> dict[str, Any]:
        """A separate COMMENT search (no24-02, arch §13.6): over comment only.

                    It searches a SEPARATE FTS table `notes_comment_fts` (trigram/BM25) —
                    the results are NOT mixed with the semantic `search` (that one looks
                    only at title/text). Validation like `search`: `query`
                    1..max_query_chars, `top_k` 1..MAX_TOP_K (the default default_top_k).
                    `namespace` — a hierarchy node (the subtree; `namespace_exact` — the
                    node itself only); an unregistered node — NamespaceError (the transport
                    gives a hint). no29-01: `updated_after` — an inclusive freshness bound
                    (n.updated_at). No embedding is used — no `warning` (FTS/BM25). The
                    output — {id, title, namespace, comment, updated_at} (≤ top_k);
                    empty → a hint.
        """
        query = self._validate_query(query)
        top_k = self._default_top_k() if top_k is None else top_k
        if not 1 <= top_k <= MAX_TOP_K:
            raise SearchValidationError(
                f"top_k: expected 1..{MAX_TOP_K}, got {top_k}"
            )
        expression = match_expression(query)
        ns_nodes = self._namespaces.filter_nodes(namespace, namespace_exact)
        fresh = (
            parse_updated_after(updated_after)
            if updated_after is not None
            else None
        )
        if expression is None:  # all words shorter than 3 characters — trigram is blind
            return {"results": [], "hint": HINT_NO_COMMENTS}
        params: list[object] = [expression]
        ns_clause = ""
        if ns_nodes:
            ns_ph = ",".join("?" * len(ns_nodes))
            ns_clause = f" AND n.namespace IN ({ns_ph})"
            params.extend(ns_nodes)
        fresh_clause = ""
        if fresh is not None:
            fresh_clause = " AND n.updated_at >= ?"
            params.append(fresh)
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT n.id, n.title, n.namespace, n.comment, n.updated_at, "
                "       bm25(notes_comment_fts, 1.0) AS badness "
                "FROM notes_comment_fts "
                "JOIN notes n ON n.id = notes_comment_fts.rowid "
                "WHERE notes_comment_fts MATCH ? AND n.deleted_at IS NULL "
                "AND n.comment IS NOT NULL"
                f"{ns_clause}{fresh_clause} "
                "ORDER BY badness, n.updated_at DESC, n.id DESC LIMIT ?",
                (*params, CANDIDATE_LIMIT),
            ).fetchall()
        results = [
            {
                "id": row["id"],
                "title": row["title"],
                "namespace": row["namespace"],
                "comment": row["comment"],
                "updated_at": row["updated_at"],
            }
            for row in rows[:top_k]
        ]
        if not results:
            return {"results": [], "hint": HINT_NO_COMMENTS}
        return {"results": results}

    def similar_notes(
        self, note_id: int, pool: int, threshold: float
    ) -> list[tuple[int, float]]:
        """KNN over the note's FULL vector (notes_vec) — the §3.1 link candidates.

                    The difference from `search`: the input is not a query text but the
                    note's own vector; there is NO namespace filter (`ns_filter=None`) —
                    links are searched over the whole database, and the "own section"
                    cutoff is done by the output (LinksService, arch §3.5).
                    The cutoffs here: the note itself, soft-deleted, `cosine < threshold`
                    (an inclusive threshold: a candidate exactly at the threshold stays).

                    A note without a ready vector (`vector_status != 'ok'` or no notes_vec
                    row) yields an empty list, without an error (FR-1.5): there are simply
                    no links. Returns [(note_id, cosine)] by closeness descending, at most
                    `pool`.
        """
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT vector_status, deleted_at FROM notes WHERE id = ?",
                (note_id,),
            ).fetchone()
            if (
                row is None
                or row["deleted_at"] is not None
                or row["vector_status"] != "ok"
            ):
                return []
            vector = vectors.get_vector(conn, note_id)
            if vector is None:
                return []
            # We widen the KNN window by trash vectors (vec0 does not know them — the same
            # post-cutoff as in search) and by the note itself: after the filter
            # we return up to `pool` active candidates.
            trash = conn.execute(
                "SELECT COUNT(*) FROM notes_vec WHERE note_id IN "
                "(SELECT id FROM notes WHERE deleted_at IS NOT NULL)"
            ).fetchone()[0]
            hits = vectors.knn(conn, vector, pool + trash + 1)
            if not hits:
                return []
            placeholders = ",".join("?" * len(hits))
            active = {
                hit[0]
                for hit in conn.execute(
                    "SELECT id FROM notes WHERE deleted_at IS NULL "
                    f"AND id IN ({placeholders})",
                    [hit_id for hit_id, _ in hits],
                )
            }
        results = [
            (hit_id, cosine)
            for hit_id, cosine in hits
            if hit_id != note_id and hit_id in active and cosine >= threshold
        ]
        return results[:pool]

    def similar_by_text(
        self,
        text: str,
        title: str | None,
        *,
        exclude_ids: frozenset[int] = frozenset(),
        limit: int,
        pool: int,
        threshold: float,
        text_chars: int,
    ) -> list[dict[str, Any]]:
        """The "similar" hint at write time (#30, arch §6.4/§8/§13.4).

                    WITHOUT embedding or model calls: the fresh note's vector is not built
                    yet (a vector on the write path is forbidden). Candidates — from the
                    already-existing FTS index `notes_fts` (as in `DedupService`,
                    the `CROSS JOIN` forces the outer loop over FTS); closeness — **Dice**
                    over the sets of significant words (`ranking.significant_tokens`) of
                    the title and the first `text_chars` characters of the text; those with
                    `score >= threshold` stay (an inclusive threshold), the order —
                    score DESC, the tie-break id DESC, at most `limit`. Returns
                    `[{"id", "title", "namespace"}]` (an empty list — a normal answer: no
                    similar ones). `exclude_ids` cuts off the fresh note itself and the ids
                    of foreign verbatim duplicates; a search failure is the caller's
                    concern (save wraps it in try/except); errors are not silenced here.
        """
        head = " ".join(filter(None, [title, text[:text_chars]]))
        expression = match_expression(head)
        if expression is None:  # all words shorter than 3 characters — trigram is blind
            return []
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT n.id, n.title, n.namespace, n.text "
                "FROM notes_fts "
                "CROSS JOIN notes n ON n.id = notes_fts.rowid "
                "WHERE notes_fts MATCH ? AND n.deleted_at IS NULL "
                "ORDER BY bm25(notes_fts, ?, 1.0), n.updated_at DESC, n.id DESC "
                "LIMIT ?",
                (expression, TITLE_FTS_WEIGHT, pool),
            ).fetchall()
        head_tokens = set(significant_tokens(head))
        scored: list[tuple[float, int, str | None, str]] = []
        for row in rows:
            if row["id"] in exclude_ids:
                continue
            candidate = " ".join(
                filter(None, [row["title"], (row["text"] or "")[:text_chars]])
            )
            candidate_tokens = set(significant_tokens(candidate))
            if not head_tokens or not candidate_tokens:
                continue
            dice = 2 * len(head_tokens & candidate_tokens) / (
                len(head_tokens) + len(candidate_tokens)
            )
            if dice >= threshold:
                scored.append((dice, row["id"], row["title"], row["namespace"]))
        scored.sort(key=lambda item: (-item[0], -item[1]))
        return [
            {"id": note_id, "title": cand_title, "namespace": namespace}
            for _, note_id, cand_title, namespace in scored[:limit]
        ]

    # --- candidate sources ----------------------------------------------------

    def _vector_candidates(
        self,
        conn: sqlite3.Connection,
        query_vector: list[float],
        ns_nodes: list[str] | None,
        updated_after: str | None = None,
    ) -> list[tuple[int, float, str | None]]:
        """Vector candidates: chunks → aggregation to notes + a fallback.

                    Every note enters the source ONCE: its best chunk sets the closeness;
                    notes without ready chunks (no notes_chunks rows at all, or no chunk
                    vectors) get the full vector's cosine (notes_vec).
                    Phase 10: the search over the subtree nodes' partitions (ns_nodes) — or
                    globally. The result is already sorted by cosine (for the RRF rank) and
                    filtered by activity and threshold.
        """
        by_chunk_source = self._chunk_candidates(conn, query_vector, ns_nodes, updated_after)
        # The Phase-7 fallback: notes without ready chunk vectors
        # (legacy/pending) are searched by the full text, as in Phase 3.
        for note_id, cosine in self._full_text_candidates(
            conn, query_vector, ns_nodes, updated_after
        ):
            if note_id not in by_chunk_source:
                by_chunk_source[note_id] = (cosine, None)
        ranked = sorted(
            by_chunk_source.items(), key=lambda pair: pair[1][0], reverse=True
        )
        return [
            (note_id, cosine, chunk_text)
            for note_id, (cosine, chunk_text) in ranked
        ]

    def _chunk_candidates(
        self,
        conn: sqlite3.Connection,
        query_vector: list[float],
        ns_nodes: list[str] | None,
        updated_after: str | None = None,
    ) -> dict[int, tuple[float, str]]:
        """KNN over notes_chunks_vec (top-50) → one record per note.

                    The note's first chunk in the closeness-sorted KNN output is its best
                    chunk: it sets the cosine and the snippet text. The threshold is
                    compared with the best chunk's cosine (brief §6). Chunks of trash notes
                    are dropped post-filter (vec0 knows nothing of notes). Phase 10: KNN
                    scans the subtree nodes' partitions (ns_nodes) — or globally; the
                    trash window is counted in the same partitions. no29-01:
                    `updated_after` — an inclusive freshness bound in the active candidate
                    set (the `active` query).
        """
        if ns_nodes:
            ns_ph = ",".join("?" * len(ns_nodes))
            ns_clause = f" AND n.namespace IN ({ns_ph})"
            ns_params: list[object] = list(ns_nodes)
        else:
            ns_clause = ""
            ns_params = []
        fresh_clause = ""
        fresh_params: list[object] = []
        if updated_after is not None:
            fresh_clause = " AND n.updated_at >= ?"
            fresh_params = [updated_after]
        trash = conn.execute(
            "SELECT COUNT(*) FROM notes_chunks_vec WHERE chunk_id IN "
            "(SELECT c.id FROM notes_chunks c JOIN notes n ON n.id = c.note_id "
            f" WHERE n.deleted_at IS NOT NULL{ns_clause})",
            ns_params,
        ).fetchone()[0]
        hits = chunks.knn(conn, query_vector, CANDIDATE_LIMIT + trash, ns_filter=ns_nodes)
        hits = hits[:CANDIDATE_LIMIT]
        if not hits:
            return {}
        placeholders = ",".join("?" * len(hits))
        # one pass: the chunk's note_id and text + the note's activity (in nodes)
        chunk_meta = {
            row["id"]: (row["note_id"], row["text"])
            for row in conn.execute(
                "SELECT c.id, c.note_id, c.text FROM notes_chunks c "
                "JOIN notes n ON n.id = c.note_id "
                f"WHERE n.deleted_at IS NULL AND c.id IN ({placeholders})"
                f"{ns_clause}{fresh_clause}",
                [chunk_id for chunk_id, _ in hits] + ns_params + fresh_params,
            )
        }
        by_note: dict[int, tuple[float, str]] = {}
        threshold = self._settings.score_threshold
        for chunk_id, cosine_value in hits:  # the KNN order: the best chunk first
            meta = chunk_meta.get(chunk_id)
            if meta is None:
                continue  # a chunk of a trash note
            note_id, text = meta
            if note_id in by_note:
                continue  # the note's best chunk is already counted
            if cosine_value < threshold:
                continue  # the best chunk is below the threshold — the rest the more so
            by_note[note_id] = (cosine_value, text)
        return by_note

    def _full_text_candidates(
        self,
        conn: sqlite3.Connection,
        query_vector: list[float],
        ns_nodes: list[str] | None,
        updated_after: str | None = None,
    ) -> list[tuple[int, float]]:
        """Top-50 active by the full text's cosine + the threshold gate (FR-1).

                    The KNN window is widened by the trash-vector count; the post-filter
                    keeps only active ones. Notes already in the chunk aggregation are
                    filtered out here (they are not fallback). Phase 10: KNN — in the
                    subtree nodes' partitions, active — only from the same nodes. no29-01:
                    `updated_after` — an inclusive freshness bound in the `active` query.
        """
        if ns_nodes:
            ns_ph = ",".join("?" * len(ns_nodes))
            ns_clause = f" AND namespace IN ({ns_ph})"
            ns_params: list[object] = list(ns_nodes)
        else:
            ns_clause = ""
            ns_params = []
        fresh_clause = ""
        fresh_params: list[object] = []
        if updated_after is not None:
            fresh_clause = " AND updated_at >= ?"
            fresh_params = [updated_after]
        trash = conn.execute(
            "SELECT COUNT(*) FROM notes_vec WHERE note_id IN "
            "(SELECT id FROM notes WHERE deleted_at IS NOT NULL"
            f"{ns_clause})",
            ns_params,
        ).fetchone()[0]
        hits = vectors.knn(conn, query_vector, CANDIDATE_LIMIT + trash, ns_filter=ns_nodes)
        if not hits:
            return []
        placeholders = ",".join("?" * len(hits))
        active = {
            row[0]
            for row in conn.execute(
                f"SELECT id FROM notes WHERE deleted_at IS NULL "
                f"{ns_clause}{fresh_clause} AND id IN ({placeholders})",
                ns_params + fresh_params + [note_id for note_id, _ in hits],
            )
        }
        threshold = self._settings.score_threshold
        return [
            (note_id, cosine)
            for note_id, cosine in hits[:CANDIDATE_LIMIT]
            if note_id in active and cosine >= threshold
        ]

    def _fts_candidates(
        self,
        conn: sqlite3.Connection,
        expression: str,
        ns_nodes: list[str] | None,
        updated_after: str | None = None,
    ) -> list[sqlite3.Row]:
        """Top-50 FTS5/BM25 over active notes in the subtree nodes; rank from 1 —
                    for RRF. Phase 10: the namespace filter — by a JOIN on notes (the FTS
                    index is not rebuilt, the Step-1 brief). lsb-0001-01 (FR-1.2): the bm25
                    weights (TITLE_FTS_WEIGHT, 1.0) — a title match above a text match.
                    no29-01: `updated_after` — an inclusive freshness bound (`n.updated_at`).
        """
        if ns_nodes:
            ns_ph = ",".join("?" * len(ns_nodes))
            ns_clause = f" AND n.namespace IN ({ns_ph})"
            # The bm25 weight parameters go BEFORE the others: the argument
            # call order — as in SQL (SELECT bm25(..., ?, ?) earlier than WHERE ?).
            params: list[object] = [TITLE_FTS_WEIGHT, 1.0, expression, *ns_nodes]
        else:
            ns_clause = ""
            params = [TITLE_FTS_WEIGHT, 1.0, expression]
        fresh_clause = ""
        if updated_after is not None:
            fresh_clause = " AND n.updated_at >= ?"
            params = [*params, updated_after]
        return conn.execute(
            "SELECT n.id, n.text, n.summary, n.summary_status, n.author, "
            "       n.namespace, n.created_at, n.updated_at, "
            "       bm25(notes_fts, ?, ?) AS badness "
            "FROM notes_fts JOIN notes n ON n.id = notes_fts.rowid "
            f"WHERE notes_fts MATCH ?{ns_clause} AND n.deleted_at IS NULL"
            f"{fresh_clause} "
            "ORDER BY badness, n.updated_at DESC, n.id DESC LIMIT ?",
            (*params, CANDIDATE_LIMIT),
        ).fetchall()

    # --- merge and output -------------------------------------------------------

    @staticmethod
    def _linked_ids(conn: sqlite3.Connection, ids: list[int]) -> set[int]:
        """Ids of candidates that have a row in links (either side).

                    LSBN-F-14: one SELECT per search call, over the candidate id list
                    (before the sort and the top cut). The result may contain ids outside
                    `ids` (a pair's partner) — harmless: the boost is looked up only for
                    candidates.
        """
        if not ids:
            return set()
        placeholders = ",".join("?" * len(ids))
        found = conn.execute(
            f"SELECT note_a, note_b FROM links "
            f"WHERE note_a IN ({placeholders}) OR note_b IN ({placeholders})",
            [*ids, *ids],
        ).fetchall()
        return {row["note_a"] for row in found} | {row["note_b"] for row in found}

    def _fetch_rows(
        self, conn: sqlite3.Connection, ids: list[int]
    ) -> dict[int, sqlite3.Row]:
        """Candidates' metadata (a soft read: deleted ones vanish)."""
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT id, title, namespace, text, summary, summary_status, author, "
            f"       created_at, updated_at "
            f"FROM notes WHERE deleted_at IS NULL AND id IN ({placeholders})",
            ids,
        ).fetchall()
        return {row["id"]: row for row in rows}

    def _merge(
        self,
        scores: dict[int, float],
        cosine_by_id: dict[int, float],
        snippet_source: dict[int, str | None],
        rows: dict[int, sqlite3.Row],
        order: dict[int, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Sort: score DESC → updated_at DESC → id DESC; the FR-1 assembly.

                    LSBN-F-14: `order` — the boosted scores (apply_boosts) set ONLY the
                    order; the old tie-breaks (updated_at DESC, id DESC) are preserved.
                    The item's rrf_score field — the raw RRF from `scores`; the output
                    contract (§4.2.6, the F-13 headers) does not change. `order is None` —
                    a sort by `scores` (the pre-boost behavior).
        """
        keys = scores if order is None else order
        candidates = [
            (keys[note_id], scores[note_id], row)
            for note_id, row in rows.items()
            if note_id in scores
        ]
        candidates.sort(
            key=lambda triple: (
                triple[0],
                triple[2]["updated_at"],
                triple[2]["id"],
            ),
            reverse=True,
        )
        # Compatibility: rrf_score by the final merge; cosine — only for
        # candidates with a valid vector hit (≥ SCORE_THRESHOLD); the snippet —
        # from the best chunk when it exists, else from the full text (fallback).
        return [
            self._result(
                row,
                score,
                cosine_by_id.get(row["id"]),
                snippet_source.get(row["id"]),
            )
            for _key, score, row in candidates
        ]

    def _result(
        self,
        row: sqlite3.Row,
        rrf_score: float,
        cosine: float | None,
        snippet_source: str | None,
    ) -> dict[str, Any]:
        """The FR-1 item format: without the note's text (memory_get addresses it).

                    Phase 11 (decision #9, follow-up 5b): +title — the real notes.title
                    value (None only for titleless migration notes);
                    REST /search returns the output as is, the MCP layer takes the same
                    field. lsb-0010-04 (FR-4.2): +chars — the note's FULL text volume in
                    characters (not snippet/summary); the text is already in the sample
                    (the snippet fallback) — no new reads. The same value as in
                    memory_list/memory_get.
        """
        return {
            "id": row["id"],
            "title": row["title"],
            "chars": len(row["text"]),  # lsb-0010-04: the full text's volume
            "summary": summary_of(row, self._settings),
            "snippet": snippet(snippet_source or row["text"], self._settings),
            "summary_status": row["summary_status"],
            "rrf_score": rrf_score,
            "cosine": cosine,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "author": row["author"],
            "namespace": row["namespace"],
        }

    # --- internal -------------------------------------------------------------

    def _query_vector(self, query: str) -> list[float] | None:
        """Encode the query; failure — degrade to FTS, not an exception."""
        try:
            return self._embedding.embed(query)
        except EmbeddingError:
            return None

    def _validate_query(self, query: str) -> str:
        """1..MAX_QUERY_CHARS — the FR-1 domain rule (a schema backstop)."""
        if not 1 <= len(query) <= self._settings.max_query_chars:
            raise SearchValidationError(
                f"query: length must be 1..{self._settings.max_query_chars} "
                f"characters, got {len(query)}"
            )
        return query

    def _default_top_k(self) -> int:
        return self._settings.default_top_k

    @staticmethod
    def _match_expression(query: str) -> str | None:
        """The query's FTS expression (the shared app.services.ranking primitive).

                    Words ≥3 characters as quoted substrings joined by OR (trigram);
                    None — not a single word for trigram to search by.
        """
        return match_expression(query)