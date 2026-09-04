"""Тесты слоя хранения неймспейсов (Фаза 10, Шаг 1): миграция, реестр, партиции.

Проверяем: нулевую миграцию колонок notes.namespace/classified_at поверх живой
БД; партицию `+ns` vec0-таблиц (sqlite-vec 0.1.6 partition keys); KNN с
фильтром поддерева; дефолт-узел 'default' в реестре; pending_chunk_rows с
namespace заметки-владельца. Размерность БД в тестах — 4 (быстрые вектора).
"""

from __future__ import annotations

import sqlite3

import pytest

from app.config import get_settings
from app.storage import chunks, vectors
from app.storage.db import init_db, session


@pytest.fixture(autouse=True)
def _small_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Маленькая размерность: pack/insert 4-мерных литералов в тестах."""
    monkeypatch.setenv("EMBEDDING_DIM", "4")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _vec(*components: float) -> bytes:
    from app.storage.vectors import pack

    return pack(list(components))


def _legacy_schema(conn: sqlite3.Connection) -> None:
    """Схема БД «до Фазы 10»: notes без namespace/classified_at, vec0 без
    партиции (как HEAD f57aeb6 — миграционная база для US-12)."""
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
    """Живая БД старой схемы с данными: 2 заметки, вектор, чанк."""
    init_db(get_settings())  # создать свежую (забрать tmp-путь из test_env)
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


# --- схема свежей БД ---------------------------------------------------------


class TestFreshSchema:
    def test_notes_has_namespace_columns(self) -> None:
        init_db(get_settings())
        with session(get_settings()) as conn:
            columns = {
                row["name"]: row for row in conn.execute("PRAGMA table_info(notes)")
            }
        assert set(columns) == {
            "id", "text", "summary", "author", "vector_status",
            "summary_status", "created_at", "updated_at", "deleted_at",
            "namespace", "classified_at", "domain_hint", "subdomain_hint",
            "confidence",
        }
        assert columns["namespace"]["dflt_value"] == "'default'"
        assert columns["classified_at"]["notnull"] == 0

    def test_namespaces_registry_with_default_node(self) -> None:
        """Реестр создан; узел 'default' существует (confirmed)."""
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


# --- миграция живой БД старой схемы (US-12) ----------------------------------


class TestLegacyMigration:
    def test_columns_added_data_preserved_vectors_pending(self, legacy_db: None) -> None:
        init_db(get_settings())
        with session(get_settings()) as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
            assert {"namespace", "classified_at"} <= columns
            notes = conn.execute(
                "SELECT id, namespace, classified_at, vector_status FROM notes ORDER BY id"
            ).fetchall()
            # Существующие заметки → 'default', классификации нет, вектора —
            # в очередь пере-кодирования (партиция).
            assert [row["namespace"] for row in notes] == ["default", "default"]
            assert all(row["classified_at"] is None for row in notes)
            assert all(row["vector_status"] == "pending" for row in notes)
            # vec-таблицы пересозданы с партицией и пусты (вектора невалидны),
            # тексты и чанки живы (trash-семантика Фазы 7)
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


# --- партиции и KNN-фильтры ---------------------------------------------------


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
        """Поддерево work + work/sbos2020 — два узла, один IN-фильтр партиций."""
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
        """UPDATE партиции (переезд заметки) — sqlite-vec 0.1.6 поддерживает."""
        with session(get_settings()) as conn:
            conn.execute("UPDATE notes_vec SET ns = 'projects' WHERE note_id = 1")
            hits = vectors.knn(conn, [1.0, 0.0, 0.0, 0.0], k=10, ns_filter=["work"])
        assert hits == []

    def test_upsert_default_partition(self, seeded: None) -> None:
        """Вызов без ns (legacy-совместимость) — пишет в партицию default."""
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
        """Воркер получает namespace заметки-владельца (партиция при вставке)."""
        init_db(get_settings())
        with session(get_settings()) as conn:
            conn.execute(
                "INSERT INTO notes (id, text, namespace) VALUES (7, 'текст', 'work')"
            )
            chunks.replace_note_chunks(conn, 7, [("текст", 8)])
            rows = chunks.pending_chunk_rows(conn, 10)
        assert [(row["id"], row["namespace"]) for row in rows] == [(1, "work")]