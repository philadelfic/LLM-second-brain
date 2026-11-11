"""Area vector indexes (the 3.0.0 substrate): skills_vec / terms_vec /
user_facts_vec / assistant_logs_vec — vec0.

A layer without domain rules (like `app.storage.vectors` for notes_vec):
creation, drop, vector write/read and KNN. The domain rules (when to encode
text, which fields to concatenate, the thresholds) live in the service layers
(`app.services.areas`, `app.services.worker`).

Key decisions:
- One vec0 table per area, the dimension = the installation's `embedding_dim`
  (shared across everything, like notes_vec); the metric is `cosine`, so
  `cosine = 1 - distance` (substrate architecture §3.2).
- A model/dimension change (the existing auto-reindex over `meta`, see
  `app.storage.db._sync_embedding_meta`) drops the area vectors too: every
  area record → `vector_status='pending'`, the worker's `areas` loop catches
  up.
- The vec0 key column and table names live here, in one place: the service
  area registrations (`app.services.areas`) assemble their AreaSpec from
  these constants, and db.py creates/drops the same tables.
"""

from __future__ import annotations

import sqlite3

from app.storage.vectors import pack, unpack

SKILLS_VEC_TABLE = "skills_vec"
SKILLS_VEC_ID_COLUMN = "skill_id"
TERMS_VEC_TABLE = "terms_vec"
TERMS_VEC_ID_COLUMN = "term_id"
USER_FACTS_VEC_TABLE = "user_facts_vec"
USER_FACTS_VEC_ID_COLUMN = "fact_id"
# The assistant-logs area (LSBN-F-10, arch §5.3): one vec0 per area, the key —
# the id of an `assistant_logs` row; dimension/metric are shared
# (`create_vec_tables`).
ASSISTANT_LOGS_VEC_TABLE = "assistant_logs_vec"
ASSISTANT_LOGS_VEC_ID_COLUMN = "log_id"

# (vec0 table, key column) — the registry for creating/dropping all area vecs.
VEC_TABLES: tuple[tuple[str, str], ...] = (
    (SKILLS_VEC_TABLE, SKILLS_VEC_ID_COLUMN),
    (TERMS_VEC_TABLE, TERMS_VEC_ID_COLUMN),
    (USER_FACTS_VEC_TABLE, USER_FACTS_VEC_ID_COLUMN),
    (ASSISTANT_LOGS_VEC_TABLE, ASSISTANT_LOGS_VEC_ID_COLUMN),
)


def create_vec_tables(conn: sqlite3.Connection, dim: int) -> None:
    """Create the missing area vecs with the pinned dimension (idempotent)."""
    for table, id_column in VEC_TABLES:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0("
            f"  {id_column} INTEGER PRIMARY KEY,"
            f"  embedding float[{dim}] distance_metric=cosine"
            ")"
        )


def drop_vec_tables(conn: sqlite3.Connection) -> None:
    """Drop every area vec (auto-reindex on a model/dimension change)."""
    for table, _id_column in VEC_TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {table}")


def vec_tables_exist(conn: sqlite3.Connection) -> bool:
    """All area vecs present (a live DB that survived an upgrade — not)."""
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    return all(table in names for table, _id_column in VEC_TABLES)


# --- operations ------------------------------------------------------------


def upsert(
    conn: sqlite3.Connection,
    table: str,
    id_column: str,
    row_id: int,
    vector: list[float],
) -> None:
    """Write/overwrite the vector of an area record.

    DELETE + INSERT, not INSERT OR REPLACE (the notes_vec precedent): a vec0
    virtual table does not guarantee REPLACE/ON CONFLICT support. A dimension
    error surfaces as sqlite3.Error (the session() bag).
    """
    conn.execute(f"DELETE FROM {table} WHERE {id_column} = ?", (row_id,))
    conn.execute(
        f"INSERT INTO {table}({id_column}, embedding) VALUES (?, ?)",
        (row_id, pack(vector)),
    )


def get_vector(
    conn: sqlite3.Connection, table: str, id_column: str, row_id: int
) -> list[float] | None:
    """The vector of an area record, or None (no vector yet — pending)."""
    row = conn.execute(
        f"SELECT embedding FROM {table} WHERE {id_column} = ?", (row_id,)
    ).fetchone()
    return None if row is None else unpack(row[0])


def knn(
    conn: sqlite3.Connection,
    table: str,
    id_column: str,
    query_vector: list[float],
    k: int,
) -> list[tuple[int, float]]:
    """Top-k area records by cosine closeness (brute-force vec0).

    Some soft-deleted records (trash) keep their vector (architecture §3.2) —
    deleted rows are filtered above vec0, in `AreaSearch` (per-item cutoff).
    Returns [(id, cosine)] by descending closeness, `cosine = 1 - distance`.
    """
    cursor = conn.execute(
        f"SELECT {id_column}, distance FROM {table} "
        "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (pack(query_vector), k),
    )
    return [(int(row[0]), 1.0 - row[1]) for row in cursor]


def count(conn: sqlite3.Connection, table: str) -> int:
    """The number of vectors in an area vec (diagnostics, tests, operator scripts)."""
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
