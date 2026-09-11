"""Схема областей (субстрат 3.0.0, постановка 00): DDL, апгрейд живой БД, идемпотентность.

ARCH substrate §3.1–3.2: отдельные таблицы/FTS5(trigram)/vec0 по областям
skills/terms/user в общей БД notes.db; апгрейд v2.2.1 → 3.0.0 — идемпотентный
старт без ручных миграций; существующие объекты заметок не пересоздаются и не
теряют данные; повторный init_db — no-op.
"""

from __future__ import annotations

import logging

import pytest
from fakes import HashEmbedder, clear_seeded_skills

from app.config import get_settings
from app.storage import area_vectors, vectors
from app.storage.db import init_db, session, transaction

# Объекты субстрата: таблицы записей, FTS-индексы, vec0, архив/мета скиллов.
AREA_TABLES = {
    "skills",
    "skills_fts",
    "skills_vec",
    "skill_versions",
    "skills_meta",
    "terms",
    "terms_fts",
    "terms_vec",
    "user_facts",
    "user_facts_fts",
    "user_facts_vec",
}
AREA_TRIGGERS = {
    "skills_fts_ai",
    "skills_fts_au",
    "terms_fts_ai",
    "terms_fts_au",
    "user_facts_fts_ai",
    "user_facts_fts_au",
}
AREA_RECORDS = ("skills", "terms", "user_facts")
# Колонки, общие для всех записей областей (контракт постановки 00).
COMMON_COLUMNS = {"id", "vector_status", "created_at", "updated_at", "deleted_at"}


@pytest.fixture
def dim8(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """БД размерности 8 (вектора — литеральные, без внешних сервисов).

    Сид skill-создателя (lsb-0007-04) снимаем: субстрат проверяет схему
    областей на пустых таблицах (сид — контракт фичи lsb-0007, его проверяет
    tests/test_skills_announce.py).
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)
    return settings


def _objects(conn) -> set[str]:
    """Имена всех объектов схемы (таблицы/индексы/триггеры)."""
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}


def _drop_area_objects(conn) -> None:
    """Эмуляция живой БД v2.2.1: объектов областей ещё нет (новизна 3.0.0)."""
    for trigger in AREA_TRIGGERS:
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    conn.execute("DROP INDEX IF EXISTS idx_terms_key_active")
    for table in (
        "skills_fts",
        "terms_fts",
        "user_facts_fts",
        "skills_vec",
        "terms_vec",
        "user_facts_vec",
        "skill_versions",
        "skills_meta",
        "skills",
        "terms",
        "user_facts",
    ):
        conn.execute(f"DROP TABLE IF EXISTS {table}")


def _seed_note(conn, note_id: int, text: str, title: str) -> None:
    """Заметка образца v2.2.1: вектор готов (notes_vec), статус ok."""
    embedder = HashEmbedder(8)
    conn.execute(
        "INSERT INTO notes (id, text, title, vector_status) VALUES (?, ?, ?, 'ok')",
        (note_id, text, title),
    )
    vectors.upsert(conn, note_id, embedder.embed(text))


class TestAreaSchema:
    """DDL субстрата: таблицы, FTS-индексы, триггеры, vec0, частичный UNIQUE."""

    def test_area_tables_and_triggers_created(self, dim8) -> None:
        with session(dim8) as conn:
            names = _objects(conn)
        assert AREA_TABLES <= names
        assert AREA_TRIGGERS <= names
        assert "idx_terms_key_active" in names

    def test_common_columns_and_vector_status_default(self, dim8) -> None:
        """Общие колонки записей + умолчание vector_status='pending'."""
        with session(dim8) as conn:
            for table in AREA_RECORDS:
                columns = {
                    row["name"]: row
                    for row in conn.execute(f"PRAGMA table_info({table})")
                }
                assert COMMON_COLUMNS <= set(columns), table
                assert columns["vector_status"]["dflt_value"] == "'pending'", table
                assert columns["id"]["pk"] == 1, table
                assert columns["deleted_at"]["notnull"] == 0, table
                conn.execute(f"INSERT INTO {table} DEFAULT VALUES")
                row = conn.execute(
                    f"SELECT vector_status, created_at, updated_at FROM {table}"
                ).fetchone()
                assert row["vector_status"] == "pending"
                assert row["created_at"].endswith("Z")
                assert row["updated_at"] == row["created_at"]

    def test_area_fts_external_content_trigram(self, dim8) -> None:
        """FTS-индексы областей — внешний контент с trigram-токенизатором."""
        with session(dim8) as conn:
            for table, fts in (
                ("skills", "skills_fts"),
                ("terms", "terms_fts"),
                ("user_facts", "user_facts_fts"),
            ):
                sql = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE name = ?", (fts,)
                ).fetchone()[0]
                assert f"content='{table}'" in sql
                assert "content_rowid='id'" in sql
                assert "tokenize='trigram'" in sql

    def test_terms_partial_unique_indexes_active_only(self, dim8) -> None:
        """UNIQUE(term_norm, context_norm) — только по активным записям (lsb-0008)."""
        with session(dim8) as conn, transaction(conn):
            conn.execute(
                "INSERT INTO terms (term, term_norm, context, context_norm, definition) "
                "VALUES ('ГЗ', 'гз', 'студенты МГУ', 'студенты мгу', 'госэкзамен')"
            )
            with pytest.raises(Exception):  # sqlite3.IntegrityError из session()
                conn.execute(
                    "INSERT INTO terms (term, term_norm, context, context_norm, definition) "
                    "VALUES ('гз', 'гз', 'студенты мгу', 'студенты мгу', 'дубль')"
                )
        # Soft delete освобождает ключ: частичный индекс фильтрует активные.
        with session(dim8) as conn, transaction(conn):
            conn.execute(
                "UPDATE terms SET deleted_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = 1"
            )
            conn.execute(
                "INSERT INTO terms (term, term_norm, context, context_norm, definition) "
                "VALUES ('гз', 'гз', 'студенты мгу', 'студенты мгу', 'новый смысл')"
            )
            assert conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0] == 2

    def test_fts_triggers_sync_insert_and_update(self, dim8) -> None:
        """Триггеры областей: INSERT индексирует, UPDATE переключает подстроки."""
        with session(dim8) as conn, transaction(conn):
            conn.execute(
                "INSERT INTO skills (name, description, steps, text) "
                "VALUES ('Деплой релиза', 'как катить', 'шаги', 'текст')"
            )
            assert [
                row[0]
                for row in conn.execute(
                    "SELECT rowid FROM skills_fts WHERE skills_fts MATCH ?",
                    ('"деплой"',),
                )
            ] == [1]
            conn.execute(
                "UPDATE skills SET name = 'Откат релиза' WHERE id = 1"
            )
            assert (
                conn.execute(
                    "SELECT rowid FROM skills_fts WHERE skills_fts MATCH ?",
                    ('"деплой"',),
                ).fetchall()
                == []
            )
            assert [
                row[0]
                for row in conn.execute(
                    "SELECT rowid FROM skills_fts WHERE skills_fts MATCH ?",
                    ('"откат"',),
                )
            ] == [1]

    def test_area_sql_does_not_reference_notes(self, dim8) -> None:
        """Изоляция на уровне таблиц: DDL области не адресует notes*."""
        with session(dim8) as conn:
            rows = conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE name IN "
                "('skills','skills_fts','skills_vec','skill_versions','skills_meta',"
                "'terms','terms_fts','terms_vec','user_facts','user_facts_fts',"
                "'user_facts_vec','idx_terms_key_active')"
            ).fetchall()
        assert rows  # все объекты областей на месте
        for row in rows:
            assert "notes" not in (row["sql"] or "").lower()


class TestUpgradeFromV221:
    """Апгрейд живой БД v2.2.1 → 3.0.0 без ручных миграций."""

    def test_live_db_upgrades_notes_intact(self, dim8, caplog) -> None:
        """Объекты областей создаются, заметки и их индексы целы."""
        with session(dim8) as conn, transaction(conn):
            _seed_note(conn, 1, "живая заметка про паровозостроение", "Сводка")
            _drop_area_objects(conn)  # БД «до 3.0.0»: областей нет
        with session(dim8) as conn:
            assert "skills" not in _objects(conn)

        init_db(dim8)  # апгрейд стартом

        with session(dim8) as conn:
            assert AREA_TABLES <= _objects(conn)
            assert AREA_TRIGGERS <= _objects(conn)
            # заметки целы: строка, статус, вектор, FTS-индекс
            row = conn.execute(
                "SELECT text, vector_status FROM notes WHERE id = 1"
            ).fetchone()
            assert row["text"] == "живая заметка про паровозостроение"
            assert row["vector_status"] == "ok"
            assert vectors.get_vector(conn, 1) is not None
            assert [
                hit[0]
                for hit in conn.execute(
                    "SELECT rowid FROM notes_fts WHERE notes_fts MATCH ?",
                    ('"паровозостроение"',),
                )
            ] == [1]
            # vec0 областей созданы под текущую размерность
            for table, _column in area_vectors.VEC_TABLES:
                sql = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE name = ?", (table,)
                ).fetchone()[0]
                assert "float[8]" in sql

    def test_repeat_init_db_is_noop(self, dim8, caplog) -> None:
        """Повторный init_db: объекты те же, данные целы, перестройки нет."""
        with session(dim8) as conn, transaction(conn):
            _seed_note(conn, 1, "заметка для повторного старта", "Сводка")
            _drop_area_objects(conn)
        init_db(dim8)  # апгрейд
        with session(dim8) as conn:
            before = _objects(conn)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="app"):
            init_db(dim8)  # повторный старт — no-op
        with session(dim8) as conn:
            assert _objects(conn) == before
            assert vectors.get_vector(conn, 1) is not None  # вектора не сброшены
            assert (
                conn.execute("SELECT vector_status FROM notes WHERE id = 1").fetchone()[0]
                == "ok"
            )
        assert not [
            record
            for record in caplog.records
            if getattr(record, "event", None) in ("reindex_started", "reindex_done")
        ]

    def test_model_change_resets_area_vectors(self, dim8, monkeypatch, caplog) -> None:
        """Смена модели/размерности: area-vec дропнуты, записи областей → pending."""
        embedder = HashEmbedder(8)
        with session(dim8) as conn, transaction(conn):
            conn.execute(
                "INSERT INTO skills (name, description, steps, text, vector_status) "
                "VALUES ('Навык', 'описание', 'шаги', 'текст', 'ok')"
            )
            conn.execute(
                "INSERT INTO terms (term, term_norm, context, context_norm, definition, "
                "vector_status) VALUES ('ГЗ', 'гз', 'МГУ', 'мгу', 'экзамен', 'ok')"
            )
            conn.execute(
                "INSERT INTO user_facts (name, body, vector_status) "
                "VALUES ('Пункт', 'тело', 'ok')"
            )
            area_vectors.upsert(
                conn, area_vectors.SKILLS_VEC_TABLE,
                area_vectors.SKILLS_VEC_ID_COLUMN, 1, embedder.embed("Навык\nописание"),
            )
            area_vectors.upsert(
                conn, area_vectors.TERMS_VEC_TABLE,
                area_vectors.TERMS_VEC_ID_COLUMN, 1, embedder.embed("ГЗ\nМГУ\nэкзамен"),
            )
            area_vectors.upsert(
                conn, area_vectors.USER_FACTS_VEC_TABLE,
                area_vectors.USER_FACTS_VEC_ID_COLUMN, 1, embedder.embed("Пункт\nтело"),
            )
        monkeypatch.setenv("EMBEDDING_MODEL", "another-embedding-model")
        monkeypatch.setenv("EMBEDDING_DIM", "16")
        get_settings.cache_clear()
        with caplog.at_level(logging.INFO, logger="app"):
            init_db(get_settings())
        with session(get_settings()) as conn:
            for table in AREA_RECORDS:
                statuses = [
                    row[0]
                    for row in conn.execute(
                        f"SELECT vector_status FROM {table}"
                    )
                ]
                assert statuses == ["pending"], table
            for table, _column in area_vectors.VEC_TABLES:
                assert area_vectors.count(conn, table) == 0
                sql = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE name = ?", (table,)
                ).fetchone()[0]
                assert "float[16]" in sql  # пересозданы под новую размерность
        assert [
            record
            for record in caplog.records
            if getattr(record, "event", None) == "reindex_started"
        ]
