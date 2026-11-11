"""The vector store — notes_vec, vec0 (ARCHITECTURE §3.3, Phase 3).

A layer without domain rules: vector serialization (float32 little-endian),
write/read and the KNN query against the vec0 table. Domain rules (when to
encode text, closeness thresholds) live in the service layers (`embedding`,
`search`, `dedup`).

Key decisions:
- `distance_metric=cosine` (sqlite-vec 0.1.6): in consistent units —
  `distance = 1 - cosine`, hence `cosine = 1 - distance` (cosine ignores the
  norm — the Ollama embedding need not be normalized); the scale is fixed by
  REQUIREMENTS §8 (the SCORE_THRESHOLD/DEDUP_SIMILAR thresholds are given as
  cosine closeness), not as a Euclidean distance.
- Up to 50 000 notes a brute-force vec0 scan is acceptable (NFR-5) — no
  separate ANN index, one DB file.
- The dimension is `float[{dim}]` from env at database creation; the DB↔env
  check lives in `db.init_db` (a mismatch → startup refusal; reindex —
  scripts/reindex.py).
"""

from __future__ import annotations

import re
import sqlite3
import struct
from collections.abc import Sequence

# sqlite_master stores the original CREATE VIRTUAL TABLE text — the dimension
# is extracted from there (vec0 shadow tables are private version details).
_VEC_DIM_RE = re.compile(r"float\[(\d+)\]")

# The namespace partition (Phase 10): the `+ns` column in the DDL is the
# sqlite-vec 0.1.6 partition key: a filtered KNN scans only its partition.
_PARTITION_RE = re.compile(r"\+\s*ns\s+TEXT")

VEC_TABLE = "notes_vec"


class VectorError(RuntimeError):
    """A vector-layer error: a broken schema or a wrong-dimension vector."""


def existing_vec_dim(conn: sqlite3.Connection) -> int | None:
    """The dimension from the sqlite_master DDL; None — no table yet."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='notes_vec'"
    ).fetchone()
    if row is None:
        return None
    match = _VEC_DIM_RE.search(row[0] or "")
    if not match:
        raise VectorError("схема notes_vec повреждена: размерность не читается")
    return int(match.group(1))


def create_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    """Create notes_vec with the pinned dimension (called once per database).

    Phase 10: the `+ns` column is the partition key over the note namespace
    (sqlite-vec 0.1.6): a KNN with the `+ns IN (...)` filter scans only the
    subtree's partitions.
    """
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS notes_vec USING vec0("
        "  note_id    INTEGER PRIMARY KEY,"
        "  +ns        TEXT,"
        f"  embedding  float[{dim}] distance_metric=cosine"
        ")"
    )


def has_partition(conn: sqlite3.Connection) -> bool:
    """Whether notes_vec carries the `+ns` partition (the Phase 10 migration); no table — False."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='notes_vec'"
    ).fetchone()
    if row is None:
        return False
    return _PARTITION_RE.search(row[0] or "") is not None


# --- serialization ---------------------------------------------------------


def pack(vector: list[float]) -> bytes:
    """list[float] → a vec0 BLOB: float32 little-endian (platform-independent)."""
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack(blob: bytes) -> list[float]:
    """BLOB → list[float] (float32); the byte length must be a multiple of 4."""
    size = len(blob) // struct.calcsize("<f")
    return list(struct.unpack(f"<{size}f", blob))


# --- operations -----------------------------------------------------------


def upsert(
    conn: sqlite3.Connection,
    note_id: int,
    vector: list[float],
    ns: str = "default",
) -> None:
    """Write/overwrite a note vector (save/update/re-vectorization).

    DELETE + INSERT, not INSERT OR REPLACE: a vec0 virtual table does not
    guarantee REPLACE/ON CONFLICT support — DELETE+INSERT goes through the
    public xUpdate protocol and survives extension version changes. A
    dimension error surfaces as sqlite3.Error (the session() bag).
    Phase 10: `ns` is the note-namespace partition (the default keeps backward
    compatibility: tests and legacy paths write to default).
    """
    drop(conn, note_id)
    conn.execute(
        "INSERT INTO notes_vec(note_id, ns, embedding) VALUES (?, ?, ?)",
        (note_id, ns, pack(vector)),
    )


def drop(conn: sqlite3.Connection, note_id: int) -> None:
    """Hard-remove a vector (reindex, physical trash cleanup — not a soft delete)."""
    conn.execute("DELETE FROM notes_vec WHERE note_id = ?", (note_id,))


def get_vector(conn: sqlite3.Connection, note_id: int) -> list[float] | None:
    """The note vector, or None (no vector yet — pending)."""
    row = conn.execute(
        "SELECT embedding FROM notes_vec WHERE note_id = ?", (note_id,)
    ).fetchone()
    return None if row is None else unpack(row[0])


def knn(
    conn: sqlite3.Connection,
    query_vector: list[float],
    k: int,
    ns_filter: Sequence[str] | None = None,
) -> list[tuple[int, float]]:
    """Top-k notes by cosine closeness to the query (KNN brute-force vec0).

    Some soft-deleted notes (trash) keep their vectors (ARCH §3.3) — deleted
    rows are filtered above vec0, in SearchService (per-item cutoff).
    Phase 10: `ns_filter` is the list of subtree node paths (the KNN scans
    only their partitions); None/empty — a global search over the whole table.
    Returns [(note_id, cosine)] by descending closeness, cosine = 1 - distance.
    """
    if ns_filter:
        placeholders = ",".join("?" * len(ns_filter))
        cursor = conn.execute(
            "SELECT note_id, distance FROM notes_vec "
            f"WHERE +ns IN ({placeholders}) "
            "AND embedding MATCH ? AND k = ? ORDER BY distance",
            (*ns_filter, pack(query_vector), k),
        )
    else:
        cursor = conn.execute(
            "SELECT note_id, distance FROM notes_vec "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (pack(query_vector), k),
        )
    return [(row[0], 1.0 - row[1]) for row in cursor]


def count(conn: sqlite3.Connection) -> int:
    """The number of vectors in the index (/health diagnostics, operator scripts)."""
    return int(conn.execute("SELECT COUNT(*) FROM notes_vec").fetchone()[0])