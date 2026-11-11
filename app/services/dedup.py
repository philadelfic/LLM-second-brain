"""DeduplicationService (ARCHITECTURE §3.2, REQUIREMENTS FR-4; Phase 8, step 2).

    Dedup is split into two paths:

    - **Synchronous, at write time (save/update)** - only a verbatim text
      duplicate (`find_by_text`): an exact match (SQL) and an "almost exact" one
      over the normalized text (case/whitespace) through FTS5 words. Instant, no
      Ollama; it does not catch paraphrases - since Phase 8 (step 1) that is by
      design: the vector is not built at write time, so a cosine comparison is
      impossible on the synchronous path.
    - **Background, after re-vectorization (steps 2-3)**: the worker looks up the
      top-N cosine candidates (`find_candidates`, the lower threshold
      DEDUP_CANDIDATE_SIMILARITY ~0.80 - deliberately wider than the "duplicate"
      threshold to catch the 0.80-0.85 paraphrases). The "duplicate" verdict
      belongs to the LLM judge (step 3.2, JudgeService - the DEDUP_JUDGE_MODEL
      model, think:false): every candidate is asked over the pair of texts; the
      cosine is only a pre-filter; without a judge (DI None, the test mode) the
      worker merges by cosine >= DEDUP_SIMILARITY (the step 2.2 fallback).

    Trash (a soft delete) takes part in neither path: a deleted note is not a
    duplicate candidate (the operator's undo returns it as a separate note,
    REQUIREMENTS FR-6); trash vectors are alive (ARCH §3.3) - dedup reads only
    the active ones.

    A note with a timer (a TTL, `expires_at IS NOT NULL`, no46-03, 3.2.0) is also
    never a merge candidate in either path: such a note must not be merged with
    another (the owner's rule). The selection excludes it on the SQL side (both
    the KNN branch and the text fallback), with no Python post-filter; the
    `links` are allowed - a TTL note may be linked to others.
"""

from __future__ import annotations

import sqlite3

from app.config import Settings
from app.storage import vectors
from app.storage.db import session

# The teaching hint (§5.3) - the shared channel of the two dedup paths.
DEDUP_HINT = (
    "a nearly identical note already exists; to refine it — call memory_update "
    "(run memory_get first so you don't lose details)"
)

# The FTS-fallback candidates among which we look for a normalized duplicate.
FALLBACK_SCAN = 20


def duplicate_response(row: sqlite3.Row) -> dict[str, object]:
    """The dedup contract (FR-4): the existing note is returned whole."""
    return {
        "duplicated": True,
        "id": row["id"],
        "text": row["text"],
        "hint": DEDUP_HINT,
    }


def normalize_text(text: str) -> str:
    """The normalization of the "verbatim" comparison: case and whitespace wobble."""
    return " ".join(text.split()).casefold()


class DeduplicationService:
    """The verbatim text dedup + the top-N cosine candidates (the step 2 pre-filter)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    #--- background dedup: the candidates (Phase 8, step 2.1) -------------------------

    def find_candidates(
        self,
        vector: list[float],
        exclude_id: int | None = None,
        namespace: str | None = None,
    ) -> list[tuple[int, float]]:
        """The top-N active candidates by cosine (the step 2 candidate pre-filter).

                    The lower threshold DEDUP_CANDIDATE_SIMILARITY (~0.80) is deliberately
                    wider than the fallback threshold DEDUP_SIMILARITY: a candidate is a
                    "possible paraphrase" (the typical 0.80-0.85 paraphrases the 0.92
                    threshold never sees). The final word belongs to the judge (step 3.2):
                    the worker asks JudgeService per candidate (think:false) and merges
                    the first "DUPLICATE" (_merge_duplicates: merge by the summarizer,
                    the earlier one is updated, the later one - trash); without a judge
                    (DI None, the test mode) - the step 2.2 cosine fallback over
                    DEDUP_SIMILARITY.

                    exclude_id - the freshly vectorized note: its vector is already in
                    notes_vec and would take a top-N slot with cosine 1.0 without the
                    exclusion. Trash is not a candidate: the KNN window widens by the
                    trash-vector count, and the post-filter keeps only active ones (as in
                    SearchService - vec0 knows nothing of notes). A note with a TTL
                    (`expires_at IS NOT NULL`, no46-03) is also never a merge candidate:
                    excluded in SQL (a condition next to `deleted_at IS NULL`); its vector
                    takes a KNN window slot too - we widen for it as well.
                    Phase 10 (§5.7): `namespace` - dedup within a namespace (the KNN scans
                    the node's partition); None - global (legacy compatibility).
                    Cross-namespace close content is not a duplicate: notes in different
                    nodes are legitimate (the orienting hint - Step 5).
                    The return - [(note_id, cosine)] by descending closeness, at most the
                    top-N.

        """
        top_n = self._settings.dedup_candidate_top_n
        threshold = self._settings.dedup_candidate_similarity
        with session(self._settings) as conn:
            if namespace:
                ns_clause = " AND namespace = ?"
                ns_params: list[object] = [namespace]
            else:
                ns_clause = ""
                ns_params = []
            # Non-candidates (trash and TTL notes, no46-03) keep their vectors in
            # notes_vec anyway - we widen the KNN window by their count,
            # otherwise such a neighbor would push an active candidate out of the top-N.
            skipped = conn.execute(
                "SELECT COUNT(*) FROM notes_vec WHERE note_id IN "
                "(SELECT id FROM notes WHERE "
                "(deleted_at IS NOT NULL OR expires_at IS NOT NULL)"
                f"{ns_clause})",
                ns_params,
            ).fetchone()[0]
            window = top_n + skipped + (1 if exclude_id is not None else 0)
            hits = vectors.knn(
                conn,
                vector,
                window,
                ns_filter=[namespace] if namespace else None,
            )
            if not hits:
                return []
            placeholders = ",".join("?" * len(hits))
            active = {
                row[0]
                for row in conn.execute(
                    f"SELECT id FROM notes WHERE deleted_at IS NULL "
                    f"AND expires_at IS NULL "
                    f"{ns_clause} AND id IN ({placeholders})",
                    ns_params + [note_id for note_id, _ in hits],
                )
            }
            result: list[tuple[int, float]] = []
            for candidate_id, cosine_value in hits:
                if cosine_value < threshold:
                    break  # the KNN output is sorted - everything further is only lower
                if candidate_id == exclude_id or candidate_id not in active:
                    continue  # the fresh note itself / trash / TTL - not candidates
                result.append((candidate_id, cosine_value))
                if len(result) == top_n:
                    break
            return result

    #--- the fallback --------------------------------------------------------

    def find_by_text(
        self,
        text: str,
        namespace: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> sqlite3.Row | None:
        """A verbatim duplicate without vectorization (ARCH §4.1 "dedup by FTS").

                    The optional `conn` - work on the PASSED connection (without an own
                    `session()`): save runs the check and the INSERT in one BEGIN
                    IMMEDIATE transaction, which rules out the TOCTOU race of two
                    concurrent saves of the same text (the writers serialize through
                    busy_timeout; the second one sees the already-inserted row - a
                    verbatim duplicate is impossible). Without `conn` - as before: an own
                    `session()` (backward compatibility).

                    LSBN-F-22 (M3): during a migration - a second attempt over the
                    PREVIOUS base (the union, design §4): the 'waiting'/'quarantine'
                    notes exist only there, and a duplicate over them must block the save
                    ("no duplicates at save" - design §6.4). There is no race: the
                    previous base is read-only during the migration (the waiting notes'
                    writes go there through NoteService, but a verbatim INSERT of new
                    notes does not). No context - the old paths.

        """
        if namespace:
            ns_clause = " AND namespace = ?"
            ns_fts_clause = " AND n.namespace = ?"
            ns_params: list[object] = [namespace]
        else:
            ns_clause = ""
            ns_fts_clause = ""
            ns_params = []
        if conn is not None:
            found = self._literal_duplicate(
                conn, text, ns_clause, ns_fts_clause, ns_params
            )
        else:
            with session(self._settings) as own_conn:
                found = self._literal_duplicate(
                    own_conn, text, ns_clause, ns_fts_clause, ns_params
                )
        if found is not None:
            return found
        # A lazy import: services.migration pulls summary -> notes (a cycle),
        # the precedent - notes.py <-> ttl.py.
        from app.services.migration import migration_context

        context = migration_context()
        if context is None:
            return None
        legacy_settings = self._settings.model_copy(
            update={"db_path": context.legacy_path}
        )
        with session(legacy_settings) as legacy_conn:
            return self._literal_duplicate(
                legacy_conn, text, ns_clause, ns_fts_clause, ns_params
            )

    def _literal_duplicate(
        self,
        conn: sqlite3.Connection,
        text: str,
        ns_clause: str,
        ns_fts_clause: str,
        ns_params: list[object],
    ) -> sqlite3.Row | None:
        """The verbatim dedup body on an open connection (SQL + the FTS fallback).

                    A note with a TTL (`expires_at IS NOT NULL`, no46-03) is not a
                    candidate here either: the `expires_at IS NULL` condition is added to
                    both SQLs (the exact scan and the FTS fallback) - such a merge is
                    forbidden by the owner's rule.

                    Plan hints (pool 13; a 50k benchmark: find_by_text with an ns filter -
                    132 s):
                    - `NOT INDEXED` on the exact scan: without it, with
                      `AND namespace = ?` the planner takes idx_notes_namespace (a
                      low-selective one - all the node's rows) as the outer loop and
                      fetches the text row by row (random pages instead of an 18-ms
                      sequential scan);
                    - `CROSS JOIN` in the FTS query: forcing notes_fts as the outer loop -
                      without it the same index put notes first and ran MATCH on every
                      row of the node. A `CROSS JOIN` in SQLite forbids the optimizer
                      from reordering the tables.

        """
        exact = conn.execute(
            "SELECT id, text, namespace FROM notes NOT INDEXED "
            "WHERE deleted_at IS NULL AND expires_at IS NULL "
            f"AND text = ?{ns_clause} ORDER BY id LIMIT 1",
            (text, *ns_params),
        ).fetchone()
        if exact is not None:
            return exact
        norm = normalize_text(text)
        words = [word for word in norm.split() if len(word) >= 3]
        if not words:  # only words shorter than 3 characters - the trigram is blind
            return None
        expression = " AND ".join(
            f'"{word.replace(chr(34), chr(34) * 2)}"'
            for word in dict.fromkeys(words)
        )
        candidates = conn.execute(
            "SELECT n.id, n.text, n.namespace FROM notes_fts "
            "CROSS JOIN notes n ON n.id = notes_fts.rowid "
            f"WHERE notes_fts MATCH ? AND n.deleted_at IS NULL "
            f"AND n.expires_at IS NULL{ns_fts_clause} LIMIT ?",
            (expression, *ns_params, FALLBACK_SCAN),
        ).fetchall()
        for candidate in candidates:
            if normalize_text(candidate["text"]) == norm:
                return candidate
        return None