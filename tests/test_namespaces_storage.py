"""The namespaces storage layer tests (Phase 10, Step 1): migration, registry, partitions.

    Checked: the zero migration of notes.namespace/classified_at over a live DB;
    the `+ns` partitioning of vec0 tables (sqlite-vec 0.1.6 partition keys); KNN
    with a subtree filter; the 'default' node in the registry; pending_chunk_rows
    with the owner note's namespace. The tests' DB dimension — 4 (fast vectors).
"""

from __future__ import annotations

import sqlite3

import pytest

from app.config import get_settings
from app.storage import chunks, vectors
from app.storage.db import init_db, session


@pytest.fixture(autouse=True)
def _small_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    """A small dimension: packing/inserting 4-dimensional literals in tests."""
    monkeypatch.setenv("EMBEDDING_DIM", "4")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _vec(*components: float) -> bytes:
    from app.storage.vectors import pack

    return pack(list(components))


def _legacy_schema(conn: sqlite3.Connection) -> None:
    """The DB schema "before Phase 10": notes without namespace/classified_at, vec0
            without a partition (as at HEAD f57aeb6 — the migration base for US-12).
    """
    conn.execute(
        """
        CREATE TABLE notes (
          id             INTEGER PRIMARY KEY,
          text           TEXT    NOT NULL CHECK(length(text) BETWEEN 1 AND 35000),
          summary        TEXT    NOT NULL DEFAULT '',
          author         TEXT    NOT NULL DEFAULT 'unknown',
          vector_status  TEXT    NOT NULL DEFAULT 'pending',
          summary_status TEXT    NOT NULL DEFAULT 'pending',
          created_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
          updated_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
          deleted_at     TEXT    NULL
        )
        """
    )
    conn.execute(
        "CREATE VIRTUAL TABLE notes_vec USING vec0("
        "  note_id INTEGER PRIMARY KEY,"
        "  embedding float[4] distance_metric=cosine)"
    )
    conn.execute(
        """
        CREATE TABLE notes_chunks (
          id      INTEGER PRIMARY KEY,
          note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
          idx     INTEGER NOT NULL,
          text    TEXT    NOT NULL,
          tokens  INTEGER NOT NULL,
          UNIQUE(note_id, idx)
        )
        """
    )
    conn.execute(
        "CREATE VIRTUAL TABLE notes_chunks_vec USING vec0("
        "  chunk_id INTEGER PRIMARY KEY,"
        "  embedding float[4] distance_metric=cosine)"
    )


@pytest.fixture
def legacy_db() -> None:
    """A live old-schema DB with data: 2 notes, a vector, a chunk."""
    init_db(get_settings())  # create a fresh one (take the tmp path from test_env)
    with session(get_settings()) as conn:
        conn.execute("DROP TABLE notes_vec")
        conn.execute("DROP TABLE notes_chunks_vec")
        conn.execute("DROP TABLE notes_chunks")
        conn.execute("DROP TABLE notes")
        _legacy_schema(conn)
        conn.execute("INSERT INTO notes (id, text) VALUES (1, 'заметка про РФД')")
        conn.execute("INSERT INTO notes (id, text) VALUES (2, 'заметка про resume')")
        conn.execute(
            "INSERT INTO notes_vec (note_id, embedding) VALUES (1, ?)",
            (_vec(1.0, 0.0, 0.0, 0.0),),
        )
        conn.execute(
            "INSERT INTO notes_chunks (note_id, idx, text, tokens) "
            "VALUES (1, 0, 'заметка про РФД', 8)"
        )
        conn.execute(
            "INSERT INTO notes_chunks_vec (chunk_id, embedding) VALUES (1, ?)",
            (_vec(1.0, 0.0, 0.0, 0.0),),
        )


# --- a fresh DB's schema ----------------------------------------------------


class TestFreshSchema:
    def test_notes_has_namespace_columns(self) -> None:
        init_db(get_settings())
        with session(get_settings()) as conn:
            columns = {
                row["name"]: row for row in conn.execute("PRAGMA table_info(notes)")
            }
        assert set(columns) == {
            "id", "text", "title", "summary", "author", "vector_status",
            "summary_status", "created_at", "updated_at", "deleted_at",
            "namespace", "classified_at", "hint_path", "confidence",
            "expires_at", "links_at", "node_order_at", "warnings_at",
            "comment",  # no24-01 (3.2.0): the note's comment (NULL = none)
        }
        assert columns["namespace"]["dflt_value"] == "'default'"
        assert columns["classified_at"]["notnull"] == 0
        assert columns["expires_at"]["notnull"] == 0  # lsb-0004-02: NULL = permanent

    def test_namespaces_registry_with_default_node(self) -> None:
        """The registry is created; the 'default' node exists (confirmed)."""
        init_db(get_settings())
        with session(get_settings()) as conn:
            row = conn.execute(
                "SELECT description, status FROM namespaces WHERE path = 'default'"
            ).fetchone()
        assert row is not None
        assert row["status"] == "confirmed"

    def test_vec_tables_have_namespace_partition(self) -> None:
        init_db(get_settings())
        with session(get_settings()) as conn:
            assert vectors.has_partition(conn)
            assert chunks.has_partition(conn)
            ddl = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='notes_vec'"
            ).fetchone()[0]
            assert "+ns" in ddl


# --- migrating a live old-schema DB (US-12) ---------------------------------


class TestLegacyMigration:
    def test_columns_added_data_preserved_vectors_pending(self, legacy_db: None) -> None:
        init_db(get_settings())
        with session(get_settings()) as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
            assert {"namespace", "classified_at"} <= columns
            notes = conn.execute(
                "SELECT id, namespace, classified_at, vector_status FROM notes ORDER BY id"
            ).fetchall()
            # Existing notes → 'default', no classification; the vectors —
            # into the re-encoding queue (the partition).
            assert [row["namespace"] for row in notes] == ["default", "default"]
            assert all(row["classified_at"] is None for row in notes)
            assert all(row["vector_status"] == "pending" for row in notes)
            # The vec tables are recreated with the partition and empty (the vectors invalid);
            # the texts and chunks are alive (the Phase-7 trash semantics)
            assert vectors.has_partition(conn)
            assert chunks.has_partition(conn)
            assert vectors.count(conn) == 0
            assert chunks.count_vectors(conn) == 0
            assert chunks.count_chunks(conn) == 1
            assert (
                conn.execute(
                    "SELECT 1 FROM namespaces WHERE path = 'default'"
                ).fetchone()
                is not None
            )

    def test_migration_is_idempotent(self, legacy_db: None) -> None:
        init_db(get_settings())
        init_db(get_settings())
        with session(get_settings()) as conn:
            assert vectors.has_partition(conn)
            assert chunks.has_partition(conn)
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM namespaces WHERE path = 'default'"
                ).fetchone()[0]
                == 1
            )


# --- partitions and KNN filters ----------------------------------------------


class TestPartitionKnn:
    @pytest.fixture
    def seeded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDING_DIM", "4")
        get_settings.cache_clear()
        init_db(get_settings())
        with session(get_settings()) as conn:
            vectors.upsert(conn, 1, [1.0, 0.0, 0.0, 0.0], ns="work")
            vectors.upsert(conn, 2, [0.0, 1.0, 0.0, 0.0], ns="projects")
            vectors.upsert(conn, 3, [0.9, 0.1, 0.0, 0.0], ns="work/sbos2020")

    def test_knn_filters_by_single_node(self, seeded: None) -> None:
        with session(get_settings()) as conn:
            hits = vectors.knn(conn, [1.0, 0.0, 0.0, 0.0], k=10, ns_filter=["work"])
        assert [note_id for note_id, _ in hits] == [1]

    def test_knn_filters_by_subtree_node_list(self, seeded: None) -> None:
        """The work + work/sbos2020 subtree — two nodes, one IN filter over partitions."""
        with session(get_settings()) as conn:
            hits = vectors.knn(
                conn, [1.0, 0.0, 0.0, 0.0], k=10, ns_filter=["work", "work/sbos2020"]
            )
        assert [note_id for note_id, _ in hits] == [1, 3]

    def test_knn_without_filter_is_global(self, seeded: None) -> None:
        with session(get_settings()) as conn:
            hits = vectors.knn(conn, [1.0, 0.0, 0.0, 0.0], k=10)
        assert {note_id for note_id, _ in hits} == {1, 2, 3}

    def test_partition_value_can_be_updated(self, seeded: None) -> None:
        """A partition UPDATE (a note move) — sqlite-vec 0.1.6 supports it."""
        with session(get_settings()) as conn:
            conn.execute("UPDATE notes_vec SET ns = 'projects' WHERE note_id = 1")
            hits = vectors.knn(conn, [1.0, 0.0, 0.0, 0.0], k=10, ns_filter=["work"])
        assert hits == []

    def test_upsert_default_partition(self, seeded: None) -> None:
        """A call without ns (legacy compatibility) — writes into the default partition."""
        with session(get_settings()) as conn:
            vectors.upsert(conn, 4, [0.0, 0.0, 1.0, 0.0])
            rows = conn.execute(
                "SELECT ns FROM notes_vec WHERE note_id = 4"
            ).fetchall()
        assert [(row[0] for row in rows)] and rows[0][0] == "default"


class TestChunksPartition:
    def test_upsert_and_knn_with_ns(self) -> None:
        init_db(get_settings())
        with session(get_settings()) as conn:
            conn.execute(
                "INSERT INTO notes (id, text, namespace) VALUES (1, 'т', 'work')"
            )
            conn.execute(
                "INSERT INTO notes (id, text, namespace) VALUES (2, 'т', 'projects')"
            )
            chunks.replace_note_chunks(conn, 1, [("текст один", 8)])
            chunks.replace_note_chunks(conn, 2, [("текст два", 8)])
            chunks.upsert_vector(conn, 1, [1.0, 0.0, 0.0, 0.0], ns="work")
            chunks.upsert_vector(conn, 2, [0.0, 1.0, 0.0, 0.0], ns="projects")
            hits = chunks.knn(conn, [1.0, 0.0, 0.0, 0.0], k=10, ns_filter=["work"])
        assert [chunk_id for chunk_id, _ in hits] == [1]

    def test_upsert_vector_if_exists_writes_partition(self) -> None:
        init_db(get_settings())
        with session(get_settings()) as conn:
            conn.execute(
                "INSERT INTO notes (id, text, namespace) VALUES (1, 'т', 'work/sbos2020')"
            )
            chunks.replace_note_chunks(conn, 1, [("текст один", 8)])
            written = chunks.upsert_vector_if_exists(
                conn, 1, [1.0, 0.0, 0.0, 0.0], "текст один", 8, ns="work/sbos2020"
            )
            row = conn.execute(
                "SELECT ns FROM notes_chunks_vec WHERE chunk_id = 1"
            ).fetchone()
        assert written is True
        assert row[0] == "work/sbos2020"

    def test_pending_chunk_rows_expose_owner_namespace(self) -> None:
        """The worker receives the owner note's namespace (the partition at insert)."""
        init_db(get_settings())
        with session(get_settings()) as conn:
            conn.execute(
                "INSERT INTO notes (id, text, namespace) VALUES (7, 'текст', 'work')"
            )
            chunks.replace_note_chunks(conn, 7, [("текст", 8)])
            rows = chunks.pending_chunk_rows(conn, 10)
        assert [(row["id"], row["namespace"]) for row in rows] == [(1, "work")]