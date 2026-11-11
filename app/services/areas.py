"""Area plumbing (substrate 3.0.0): normalization, similarity, hybrid search.

    One reusable helper for all areas (skills / terms / user / logs):
    each area is described by an `AreaSpec` — its own tables, indexed fields
    and output projection; `AreaSearch` runs the hybrid
    vec0-KNN + FTS5(BM25) → RRF (the same `settings.rrf_k`, the
    `app.services.ranking.fuse_rrf` primitive), filters `deleted_at IS NULL`
    and deduplicates ids.

    Degradation (NFR-3): a query-encoding failure → FTS-only search + `warning`
    (the notes precedent `WARNING_FTS_ONLY`) — search does not break on an external
    dependency. Isolation is an architectural invariant: the helper's SQL addresses
    only its area's tables — no `notes*`, no other areas.

    `normalize_key` (the terms key and context/fact comparison) and
    `trigram_similarity` (context closeness and dedup hints without embedding)
    are shared by the area services (lsb-0008-01, lsb-0009-01).
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.services.embedding import Embedder, EmbeddingError, EmbeddingService
from app.services.ranking import fuse_rrf, match_expression
from app.services.search import (
    CANDIDATE_LIMIT,
    HINT_SHORT_QUERY,
    MAX_TOP_K,
    WARNING_FTS_ONLY,
)
from app.storage import area_vectors
from app.storage.db import (
    ASSISTANT_LOGS_FTS_COLUMNS,
    ASSISTANT_LOGS_FTS_TABLE,
    ASSISTANT_LOGS_TABLE,
    SKILLS_FTS_COLUMNS,
    SKILLS_FTS_TABLE,
    SKILLS_TABLE,
    TERMS_FTS_COLUMNS,
    TERMS_FTS_TABLE,
    TERMS_TABLE,
    USER_FACTS_FTS_COLUMNS,
    USER_FACTS_FTS_TABLE,
    USER_FACTS_TABLE,
    session,
)

# Empty area result — a soft answer with a hint (substrate architecture §3.5:
# "nothing found" is a valid outcome; the model does not enter an area without a reason).
HINT_NO_RESULTS = (
    "nothing found in this area; rephrase more broadly "
    "(searches by substrings from 3 characters)"
)

# Trigram size (character-based): same as the FTS5 trigram and the tests' HashEmbedder.
TRIGRAM_SIZE = 3

_WHITESPACE_RE = re.compile(r"\s+")


# --- normalization and trigram similarity ---------------------------------


def normalize_key(value: str) -> str:
    """Normalize a key string: lower → trim → collapse inner spaces → yo→ye.

            Area canon (lsb-0008 §3.2, lsb-0009 §3.4): one mechanic for the term key
            (`term_norm` / `context_norm`) and for fact comparison. Case, trailing
            whitespace and inconsistent inner spacing do not create duplicates;
            "yo" and "ye" are the same letter.
    """
    lowered = value.lower().strip()
    collapsed = _WHITESPACE_RE.sub(" ", lowered)
    return collapsed.replace("ё", "е")


def _trigrams(text: str) -> set[str]:
    """The set of character trigrams of a text (a sliding window of 3)."""
    if len(text) < TRIGRAM_SIZE:
        return set()
    return {text[index : index + TRIGRAM_SIZE] for index in range(len(text) - 2)}


def trigram_similarity(left: str, right: str) -> float:
    """Trigram similarity of normalized strings — 0..1, symmetric.

            Overlap coefficient (Szymkiewicz–Simpson): |A ∩ B| / min(|A|, |B|) over
            the trigram sets of the normalized strings. Chosen deliberately: term
            contexts are short ("MGU" vs "MGU students" — close contexts, decision
            lsb-0008 §3.5), while Dice/Jaccard underestimate such pairs (1/(1+N)).
            Boundaries: 1.0 — the strings matched, or one's trigrams are contained in
            the other; 0.0 — normalization is empty, there are no trigrams, or the
            sets do not intersect. Similarity is computed arithmetically, without
            embedding (writes stay instant).
    """
    normalized_left = normalize_key(left)
    normalized_right = normalize_key(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0
    left_trigrams = _trigrams(normalized_left)
    right_trigrams = _trigrams(normalized_right)
    if not left_trigrams or not right_trigrams:
        return 0.0
    return len(left_trigrams & right_trigrams) / min(
        len(left_trigrams), len(right_trigrams)
    )


# --- area registration -----------------------------------------------------


@dataclass(frozen=True)
class AreaSpec:
    """Area description for `AreaSearch`: its tables, fields, output projection.

            `fts_columns` — FTS5 index columns (for BM25); `select_columns` — output
            projection fields WITHOUT `id` (id is always added); `embed_fields` —
            fields concatenated into the record's vectorization text (order =
            concatenation order); `fts_weights` — bm25 column weights (empty — all 1.0).

            `settings_flag` — the name of the `Settings` bool field enabling the area
            (FR-1.1/FR-5.1, #40): a disabled area serves no MCP handles, its section
            and announcement are not baked in, and the `areas` loop skips it. A future
            narrowing to handle groups (LSBN-F-11) is added as a new flag level, not
            by renaming.
    """

    name: str
    settings_flag: str
    table: str
    fts_table: str
    vec_table: str
    vec_id_column: str
    fts_columns: tuple[str, ...]
    select_columns: tuple[str, ...]
    embed_fields: tuple[str, ...]
    fts_weights: tuple[float, ...] = ()

    def embed_text(self, row: sqlite3.Row) -> str:
        """The record's vectorization text: non-empty fields joined by newline.

                    Formats (substrate architecture §3.3): skills — `name + description`;
                    terms — `term + context + definition`; user — `name + body`.
        """
        return "\n".join(
            str(row[column]) for column in self.embed_fields if row[column]
        )


SKILLS_AREA = AreaSpec(
    name="skills",
    settings_flag="area_skills_enabled",
    table=SKILLS_TABLE,
    fts_table=SKILLS_FTS_TABLE,
    vec_table=area_vectors.SKILLS_VEC_TABLE,
    vec_id_column=area_vectors.SKILLS_VEC_ID_COLUMN,
    fts_columns=SKILLS_FTS_COLUMNS,
    select_columns=("name", "description"),
    embed_fields=("name", "description"),
)
TERMS_AREA = AreaSpec(
    name="terms",
    settings_flag="area_terms_enabled",
    table=TERMS_TABLE,
    fts_table=TERMS_FTS_TABLE,
    vec_table=area_vectors.TERMS_VEC_TABLE,
    vec_id_column=area_vectors.TERMS_VEC_ID_COLUMN,
    fts_columns=TERMS_FTS_COLUMNS,
    select_columns=("term", "context", "definition"),
    embed_fields=("term", "context", "definition"),
)
USER_FACTS_AREA = AreaSpec(
    name="user",
    settings_flag="area_user_facts_enabled",
    table=USER_FACTS_TABLE,
    fts_table=USER_FACTS_FTS_TABLE,
    vec_table=area_vectors.USER_FACTS_VEC_TABLE,
    vec_id_column=area_vectors.USER_FACTS_VEC_ID_COLUMN,
    fts_columns=USER_FACTS_FTS_COLUMNS,
    select_columns=("name", "body"),
    embed_fields=("name", "body"),
)
# The "Assistant logs" area (LSBN-F-10, arch §5.3/§6.2): vectorization text —
# non-empty `action` + `reason` + `outcome` joined by newline; output —
# the change object reference and time (full read — f10-02/03).
LOGS_AREA = AreaSpec(
    name="logs",
    settings_flag="area_logs_enabled",
    table=ASSISTANT_LOGS_TABLE,
    fts_table=ASSISTANT_LOGS_FTS_TABLE,
    vec_table=area_vectors.ASSISTANT_LOGS_VEC_TABLE,
    vec_id_column=area_vectors.ASSISTANT_LOGS_VEC_ID_COLUMN,
    fts_columns=ASSISTANT_LOGS_FTS_COLUMNS,
    select_columns=("action", "entity", "entity_id", "entity_version", "created_at"),
    embed_fields=("action", "reason", "outcome"),
)

# Area registry (the worker's `areas` loop and the isolation tests walk it).
ALL_AREAS: tuple[AreaSpec, ...] = (
    SKILLS_AREA,
    TERMS_AREA,
    USER_FACTS_AREA,
    LOGS_AREA,
)


def enabled_areas(settings: Settings) -> tuple[AreaSpec, ...]:
    """Registry of ENABLED areas — a filter of `ALL_AREAS` by flags (FR-1.1, #40).

            The single point of flag reads: the MCP surface, the instruction/announcement
            and the background `areas` loop walk this registry, not `ALL_AREAS`.
            A disabled area does not disappear from data or observability (FR-3.1;
            `/health.areas` is built over `ALL_AREAS`), but takes no part in the
            surface or background processing. The F-11 extension point: narrowing the
            set is an edit of the registry, not of flag names (FR-5.1).
    """
    return tuple(
        spec for spec in ALL_AREAS if getattr(settings, spec.settings_flag)
    )


# --- hybrid search within an area -----------------------------------------


class AreaSearchValidationError(ValueError):
    """A violation of the area query's domain constraints (length, top_k)."""


class AreaSearch:
    """Hybrid vec0-KNN + FTS5(BM25) → RRF within one area; the embedder is DI.

            The area is given by `AreaSpec`: one helper for all areas, and each area's
            SQL addresses only its own tables. Output — the area projection
            (`select_columns`) + `id` and `score` (RRF merge) — exactly the fields the
            feature contracts describe. An empty result is a soft answer with a hint;
            an embedder failure — FTS-only + warning (WARNING_FTS_ONLY).
    """

    def __init__(
        self,
        settings: Settings,
        spec: AreaSpec,
        embedding: Embedder | None = None,
    ) -> None:
        self._settings = settings
        self._spec = spec
        # DI for tests: a deterministic HashEmbedder instead of the network.
        self._embedding: Embedder = (
            embedding if embedding is not None else EmbeddingService(settings)
        )

    @property
    def spec(self) -> AreaSpec:
        """The area description the helper was built for (diagnostics, tests)."""
        return self._spec

    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """Hybrid search within an area (substrate architecture §3.5).

                    Vector side — KNN over the area's vec0; FTS side — BM25 over the
                    area's FTS5 index; merge — RRF (the same `settings.rrf_k`), the
                    `deleted_at IS NULL` filter and id deduplication. A query-encoding
                    failure does not break the search: FTS-only + `warning` (NFR-3).
        """
        query = self._validate_query(query)
        top_k = self._settings.default_top_k if top_k is None else top_k
        if not 1 <= top_k <= MAX_TOP_K:
            raise AreaSearchValidationError(
                f"top_k: expected 1..{MAX_TOP_K}, got {top_k}"
            )
        query_vector = self._query_vector(query)
        expression = match_expression(query)
        with session(self._settings) as conn:
            rankings: list[list[int]] = []
            if query_vector is not None:
                hits = area_vectors.knn(
                    conn,
                    self._spec.vec_table,
                    self._spec.vec_id_column,
                    query_vector,
                    CANDIDATE_LIMIT,
                )
                # Cosine threshold gate (the notes precedent, SCORE_THRESHOLD):
                # vec0-KNN ALWAYS returns the k nearest — without a threshold the "probe"
                # of an area ("empty = no such record") would become unreachable
                # on any non-empty table, and the soft empty-search hint
                # (substrate §3.5) would never be emitted. The threshold is a shared
                # encoder calibration, not a per-area one.
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
                        for row in self._fts_candidates(conn, expression)
                    ]
                )
            scores = fuse_rrf(rankings, self._settings.rrf_k)
            # Soft read: deleted (trash) records vanish from the output; their ids in
            # FTS/vec physically remain (substrate architecture §3.2).
            rows = self._fetch_rows(conn, list(scores))
        results = self._merge(scores, rows)[:top_k]
        warning = None if query_vector is not None else WARNING_FTS_ONLY
        if not results:
            hint = (
                HINT_SHORT_QUERY
                if query_vector is None and expression is None
                else HINT_NO_RESULTS
            )
            return {"results": [], "warning": warning, "hint": hint}
        return {"results": results, "warning": warning}

    # --- candidate sources --------------------------------------------------

    def _fts_candidates(
        self, conn: sqlite3.Connection, expression: str
    ) -> list[sqlite3.Row]:
        """Top-50 FTS5/BM25 over the area's active records; rank from 1 — for RRF."""
        spec = self._spec
        weights = spec.fts_weights or tuple(1.0 for _ in spec.fts_columns)
        weight_placeholders = ", ".join("?" * len(weights))
        return conn.execute(
            f"SELECT t.id, t.updated_at, "
            f"       bm25({spec.fts_table}, {weight_placeholders}) AS badness "
            f"FROM {spec.fts_table} JOIN {spec.table} t "
            f"ON t.id = {spec.fts_table}.rowid "
            f"WHERE {spec.fts_table} MATCH ? AND t.deleted_at IS NULL "
            "ORDER BY badness, t.updated_at DESC, t.id DESC LIMIT ?",
            (*weights, expression, CANDIDATE_LIMIT),
        ).fetchall()

    def _fetch_rows(
        self, conn: sqlite3.Connection, ids: list[int]
    ) -> dict[int, sqlite3.Row]:
        """Active candidate records with projection fields (soft read)."""
        if not ids:
            return {}
        spec = self._spec
        placeholders = ", ".join("?" * len(ids))
        columns = ", ".join(("id", *spec.select_columns, "updated_at"))
        rows = conn.execute(
            f"SELECT {columns} FROM {spec.table} "
            f"WHERE deleted_at IS NULL AND id IN ({placeholders})",
            ids,
        ).fetchall()
        return {int(row["id"]): row for row in rows}

    # --- merge and output ----------------------------------------------------

    def _merge(
        self,
        scores: dict[int, float],
        rows: dict[int, sqlite3.Row],
    ) -> list[dict[str, Any]]:
        """Sort: score DESC → updated_at DESC → id DESC; projection assembly."""
        spec = self._spec
        candidates = [
            (scores[row_id], row)
            for row_id, row in rows.items()
            if row_id in scores
        ]
        candidates.sort(
            key=lambda pair: (pair[0], pair[1]["updated_at"], pair[1]["id"]),
            reverse=True,
        )
        return [
            {
                "id": int(row["id"]),
                **{column: row[column] for column in spec.select_columns},
                "score": score,
            }
            for score, row in candidates
        ]

    # --- internal -------------------------------------------------------------

    def _query_vector(self, query: str) -> list[float] | None:
        """Encode the query; failure — degrade to FTS, not an exception."""
        try:
            return self._embedding.embed(query)
        except EmbeddingError:
            return None

    def _validate_query(self, query: str) -> str:
        """1..MAX_QUERY_CHARS — the area query's domain rule."""
        if not 1 <= len(query) <= self._settings.max_query_chars:
            raise AreaSearchValidationError(
                f"query: length must be 1..{self._settings.max_query_chars} "
                f"characters, got {len(query)}"
            )
        return query
