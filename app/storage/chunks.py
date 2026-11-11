"""The chunk store — notes_chunks + notes_chunks_vec, vec0 (Phase 7, step 2).

A layer without domain rules (like `app.storage.vectors` for the full text):
tables, rows, serialization and KNN. Domain rules (when to split a note into
chunks, single-vector reuse, the worker queue) live in the service layers
(steps 3–5).

Key decisions (brief §5):
- `notes_chunks` — a plain table with an FK `note_id -> notes(id) ON DELETE
  CASCADE`: a hard note deletion by the operator cascades the chunks away; a
  soft delete does not touch rows — trash chunks stay (the brief decision).
  `PRAGMA foreign_keys=ON` is set in `db.session` on every connection (SQLite
  ships with FKs off; the operator's CLI may not enable them — orphans are
  cleaned at startup, see `clean_orphans`).
- `UNIQUE(note_id, idx)` — a deterministic chunk order and a duplicate guard;
  the same (note_id, idx) index serves the per-note select/delete.
- `notes_chunks_vec` — vec0 (`chunk_id` PK), the same cosine metric and the
  same dimension gate as `notes_vec`. vec0 has no FK support, so orphans
  (vectors of deleted chunks) are cleaned explicitly: service code deletes
  vectors before deleting chunks, `init_db.clean_orphans` self-heals at
  startup.
- "A chunk's vector is ready" is the very fact of a row in notes_chunks_vec
  (anti-join) — notes_chunks carries no separate status field (brief §5):
  on a model/dimension change the vector table is dropped — every chunk
  automatically becomes "pending" and the worker catches up.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence

from app.storage.vectors import pack, unpack

CHUNKS_TABLE = "notes_chunks"
CHUNKS_VEC_TABLE = "notes_chunks_vec"

# sqlite_master stores the original CREATE VIRTUAL TABLE text — the dimension
# is extracted from there (symmetric to app.storage.vectors).
_VEC_DIM_RE = re.compile(r"float\[(\d+)\]")

# The namespace partition (Phase 10): the `+ns` column is the sqlite-vec 0.1.6.
_PARTITION_RE = re.compile(r"\+\s*ns\s+TEXT")


class ChunkError(RuntimeError):
    """A chunk-layer error: a broken schema or a wrong-dimension vector."""


# --- schema ---------------------------------------------------------------

_CHUNKS_DDL = """
CREATE TABLE IF NOT EXISTS notes_chunks (
  id      INTEGER PRIMARY KEY,
  note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
  idx     INTEGER NOT NULL,
  text    TEXT    NOT NULL,
  tokens  INTEGER NOT NULL,
  UNIQUE(note_id, idx)
)
"""


def create_table(conn) -> None:
    """Create notes_chunks (idempotent, called from db.init_db)."""
    conn.execute(_CHUNKS_DDL)


def create_vec_table(conn, dim: int) -> None:
    """Create notes_chunks_vec with the pinned dimension (like notes_vec).

    Phase 10: the `+ns` column is the partition key over the owning note's
    namespace: a subtree-filtered KNN scans only the subtree's partitions.
    The ns value on insert comes from the owner's notes.namespace (the
    worker).
    """
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS notes_chunks_vec USING vec0("
        "  chunk_id   INTEGER PRIMARY KEY,"
        "  +ns        TEXT,"
        f"  embedding  float[{dim}] distance_metric=cosine"
        ")"
    )


def has_partition(conn) -> bool:
    """Whether notes_chunks_vec carries the `+ns` partition (the Phase 10 migration); no table — False."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='notes_chunks_vec'"
    ).fetchone()
    if row is None:
        return False
    return _PARTITION_RE.search(row[0] or "") is not None


def existing_vec_dim(conn) -> int | None:
    """The notes_chunks_vec dimension from the sqlite_master DDL; None — no table."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='notes_chunks_vec'"
    ).fetchone()
    if row is None:
        return None
    match = _VEC_DIM_RE.search(row[0] or "")
    if not match:
        raise ChunkError("схема notes_chunks_vec повреждена: размерность не читается")
    return int(match.group(1))


def clean_orphans(conn) -> tuple[int, int]:
    """Physically remove orphans after the operator's direct DB edits.

    The layer trusts the operator, not the code: a sqlite3 CLI without
    PRAGMA foreign_keys=ON does NOT cascade a note deletion into chunks.
    Called in init_db at startup (like the FTS-integrity-check): (chunks
    without a note, vectors without a chunk).
    """
    dead_chunks = conn.execute(
        "DELETE FROM notes_chunks WHERE note_id NOT IN (SELECT id FROM notes)"
    ).rowcount
    dead_vectors = conn.execute(
        "DELETE FROM notes_chunks_vec WHERE chunk_id NOT IN "
        "(SELECT id FROM notes_chunks)"
    ).rowcount
    return dead_chunks, dead_vectors


# --- notes_chunks rows --------------------------------------------------


def replace_note_chunks(
    conn, note_id: int, chunks_data: Sequence[tuple[str, int]]
) -> list[int]:
    """Fully replace a note's chunks (INSERT on save, replacement on update).

    Old chunk vectors are removed together with the chunks (vec0 has no FK —
    an explicit DELETE). Returns the inserted chunk ids in idx order — the
    worker (step 5) fills them during re-vectorization. Called inside a
    transaction by the caller.
    """
    conn.execute(
        f"DELETE FROM {CHUNKS_VEC_TABLE} WHERE chunk_id IN "
        f"(SELECT id FROM {CHUNKS_TABLE} WHERE note_id = ?)",
        (note_id,),
    )
    conn.execute(f"DELETE FROM {CHUNKS_TABLE} WHERE note_id = ?", (note_id,))
    ids: list[int] = []
    for idx, (text, tokens) in enumerate(chunks_data):
        cursor = conn.execute(
            f"INSERT INTO {CHUNKS_TABLE} (note_id, idx, text, tokens) VALUES (?, ?, ?, ?)",
            (note_id, idx, text, tokens),
        )
        ids.append(int(cursor.lastrowid))
    return ids


def drop_note_chunks(conn, note_id: int) -> None:
    """Hard-remove a note's chunks and their vectors (hard deletion,
    re-chunking). A soft delete does NOT call this — trash chunks stay."""
    replace_note_chunks(conn, note_id, [])


def get_note_chunks(conn, note_id: int) -> list[tuple[int, int, str, int]]:
    """A note's chunks: [(chunk_id, idx, text, tokens)] by ascending idx."""
    return [
        (int(row[0]), int(row[1]), str(row[2]), int(row[3]))
        for row in conn.execute(
            f"SELECT id, idx, text, tokens FROM {CHUNKS_TABLE} "
            "WHERE note_id = ? ORDER BY idx",
            (note_id,),
        )
    ]


def count_chunks(conn) -> int:
    """The total chunk count (diagnostics, the Phase 7 /health)."""
    return int(conn.execute(f"SELECT COUNT(*) FROM {CHUNKS_TABLE}").fetchone()[0])


# --- the pending-chunk queue -----------------------------------------------


def pending_chunk_rows(conn, limit: int) -> list:
    """Read a pending batch for the worker (step 5): id, text, tokens, note_id, ns.

    Anti-join, "oldest first" order and `limit`; three extra columns beyond
    (id, text): the worker needs note_id — for single-chunk reuse (checked
    against the note); text/tokens — the write guard against a race with
    update (ARCH §4.5); Phase 10: ns — the owning note's namespace partition
    for the vector write.
    """
    return conn.execute(
        f"SELECT c.id, c.text, c.tokens, c.note_id, n.namespace FROM {CHUNKS_TABLE} c "
        "JOIN notes n ON n.id = c.note_id "
        f"LEFT JOIN {CHUNKS_VEC_TABLE} v ON v.chunk_id = c.id "
        "WHERE v.chunk_id IS NULL ORDER BY c.id LIMIT ?",
        (limit,),
    ).fetchall()


def count_pending(conn) -> int:
    """The number of chunks without vectors (the Phase 7 /health, worker metrics)."""
    return int(
        conn.execute(
            f"SELECT COUNT(*) FROM {CHUNKS_TABLE} c "
            f"LEFT JOIN {CHUNKS_VEC_TABLE} v ON v.chunk_id = c.id "
            "WHERE v.chunk_id IS NULL"
        ).fetchone()[0]
    )


# --- notes_chunks_vec vectors ----------------------------------------------


def upsert_vector(conn, chunk_id: int, vector: list[float], ns: str = "default") -> None:
    """Write/overwrite a chunk vector (DELETE+INSERT — as in vectors).

    Phase 10: `ns` is the owning note's namespace partition (the default keeps
    backward compatibility: tests and legacy paths write to default).
    """
    conn.execute(f"DELETE FROM {CHUNKS_VEC_TABLE} WHERE chunk_id = ?", (chunk_id,))
    conn.execute(
        f"INSERT INTO {CHUNKS_VEC_TABLE} (chunk_id, ns, embedding) VALUES (?, ?, ?)",
        (chunk_id, ns, pack(vector)),
    )


def upsert_vector_if_exists(
    conn, chunk_id: int, vector: list[float], text: str, tokens: int, ns: str = "default"
) -> bool:
    """Write a chunk vector only if the chunk is unchanged since the read.

    A race with memory_update (ARCH §4.5, the `AND text = ?` analogue of the
    summary): between reading the queue and the end of encoding, an update
    could replace the chunks (DELETE+INSERT — ids are reused on re-insert),
    and the old text's vector must not land on the new chunk. The check is
    (id, text, tokens); False — the chunk is gone/replaced: the vector is not
    written (no orphan is born), the fresh chunks are caught up by the next
    queue batch.
    Phase 10: `ns` is the owning note's namespace partition.
    """
    conn.execute(f"DELETE FROM {CHUNKS_VEC_TABLE} WHERE chunk_id = ?", (chunk_id,))
    cursor = conn.execute(
        f"INSERT INTO {CHUNKS_VEC_TABLE} (chunk_id, ns, embedding) "
        f"SELECT ?, ?, ? WHERE EXISTS (SELECT 1 FROM {CHUNKS_TABLE} "
        "WHERE id = ? AND text = ? AND tokens = ?)",
        (chunk_id, ns, pack(vector), chunk_id, text, tokens),
    )
    return cursor.rowcount > 0


def get_vector(conn, chunk_id: int) -> list[float] | None:
    """The chunk vector, or None (still pending)."""
    row = conn.execute(
        f"SELECT embedding FROM {CHUNKS_VEC_TABLE} WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    return None if row is None else unpack(row[0])


def drop_vector(conn, chunk_id: int) -> None:
    """Remove a chunk vector (re-chunking/cleanup)."""
    conn.execute(f"DELETE FROM {CHUNKS_VEC_TABLE} WHERE chunk_id = ?", (chunk_id,))


def count_vectors(conn) -> int:
    """The number of chunk vectors in the index."""
    return int(conn.execute(f"SELECT COUNT(*) FROM {CHUNKS_VEC_TABLE}").fetchone()[0])


def knn(
    conn,
    query_vector: list[float],
    k: int,
    ns_filter: Sequence[str] | None = None,
) -> list[tuple[int, float]]:
    """Top-k chunks by cosine closeness (KNN brute-force vec0).

    Returns [(chunk_id, cosine)]. Note-activity filtering, aggregation up to
    notes and the thresholds live above vec0, in SearchService (step 4): the
    vec0 table knows nothing about notes (including trash). Phase 10:
    `ns_filter` is the list of subtree node paths (the KNN scans only their
    partitions); None/empty — a global search.
    """
    if ns_filter:
        placeholders = ",".join("?" * len(ns_filter))
        cursor = conn.execute(
            f"SELECT chunk_id, distance FROM {CHUNKS_VEC_TABLE} "
            f"WHERE +ns IN ({placeholders}) "
            "AND embedding MATCH ? AND k = ? ORDER BY distance",
            (*ns_filter, pack(query_vector), k),
        )
    else:
        cursor = conn.execute(
            f"SELECT chunk_id, distance FROM {CHUNKS_VEC_TABLE} "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (pack(query_vector), k),
        )
    return [(int(row[0]), 1.0 - row[1]) for row in cursor]


def _cosine(a: list[float], b: list[float]) -> float:
    """The cosine closeness of two vectors (norm-insensitive)."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def rank_chunks(
    conn, note_id: int, query_vector: list[float], limit: int
) -> list[int]:
    """The indices (idx) of one note's chunks by cosine closeness to a query.

    lsb-0003: reading a note by the chunk matching a semantic query. Skips
    chunks without a vector (pending — not caught up yet) and returns up to
    `limit` indices by descending closeness (the most relevant first).
    """
    rows = get_note_chunks(conn, note_id)  # (chunk_id, idx, text, tokens)
    scored: list[tuple[int, float]] = []
    for chunk_id, idx, _text, _tokens in rows:
        vec = get_vector(conn, chunk_id)
        if vec is None:
            continue  # pending — not ranked
        scored.append((idx, _cosine(query_vector, vec)))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [idx for idx, _score in scored[:limit]]