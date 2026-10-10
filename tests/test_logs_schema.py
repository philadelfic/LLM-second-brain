"""Схема области «Логи ассистента» (LSBN-F-10, f10-01): таблица, FTS, vec0, индексы.

Область — таблица `assistant_logs` + FTS5 внешнего контента (`tokenize='trigram'`)
+ vec0 (`log_id`, размерность инсталляции) + два частичных индекса обратного
запроса (arch §5.1/§5.2/§5.3). Миграция идемпотентна и применима с любой
предыдущей версии: повторный `init_db` набор объектов и число строк не меняет.
"""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings
from app.storage.db import init_db, session, transaction

DIM = 8


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Схема в тестовой БД; размерность 8 — модели не участвуют."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def _objects(settings: Settings) -> set[str]:
    with session(settings) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master").fetchall()
    return {row["name"] for row in rows}


def _sql(settings: Settings, name: str) -> str:
    with session(settings) as conn:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?", (name,)
        ).fetchone()
    assert row is not None and row["sql"] is not None
    return " ".join(str(row["sql"]).split())


def _log_count(settings: Settings) -> int:
    with session(settings) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM assistant_logs").fetchone()[0])


def _fts_hits(settings: Settings, term: str) -> int:
    with session(settings) as conn:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM assistant_logs_fts "
                "WHERE assistant_logs_fts MATCH ?",
                (term,),
            ).fetchone()[0]
        )


def test_logs_tables_created(settings: Settings) -> None:
    """Таблицы, FTS (trigram, внешний контент) и vec0 (log_id, dim) созданы."""
    objects = _objects(settings)
    assert {"assistant_logs", "assistant_logs_fts", "assistant_logs_vec"} <= objects

    fts = _sql(settings, "assistant_logs_fts")
    assert "content='assistant_logs'" in fts
    assert "content_rowid='id'" in fts
    assert "tokenize='trigram'" in fts

    vec = _sql(settings, "assistant_logs_vec")
    assert "log_id INTEGER PRIMARY KEY" in vec
    assert f"embedding float[{DIM}]" in vec
    assert "distance_metric=cosine" in vec

    # DDL §5.1: колонки записи, дефолты source/vector_status — точно как в арх-доке.
    with session(settings) as conn:
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(assistant_logs)")
        }
    assert columns == {
        "id",
        "action",
        "reason",
        "entity",
        "entity_id",
        "entity_version",
        "entity_ref",
        "outcome",
        "verified",
        "source",
        "session",
        "vector_status",
        "created_at",
        "updated_at",
        "deleted_at",
    }


def test_init_idempotent(settings: Settings) -> None:
    """Повторный `init_db`: набор объектов и число строк логов не меняются."""
    before = _objects(settings)
    init_db(settings)  # рестарт без записей — no-op
    assert _objects(settings) == before

    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO assistant_logs (action, reason) VALUES ('deploy', 'out')"
        )
    init_db(settings)  # рестарт с записью — no-op
    assert _objects(settings) == before
    assert _log_count(settings) == 1


def test_logs_indexes_present(settings: Settings) -> None:
    """Оба частичных индекса обратного запроса есть; оба — по активным записям."""
    objects = _objects(settings)
    assert {"idx_assistant_logs_entity_active", "idx_assistant_logs_created_active"} <= (
        objects
    )

    created = _sql(settings, "idx_assistant_logs_created_active")
    assert "created_at DESC" in created
    assert "WHERE deleted_at IS NULL" in created

    entity = _sql(settings, "idx_assistant_logs_entity_active")
    assert "entity, entity_id, created_at DESC" in entity
    assert "WHERE deleted_at IS NULL" in entity


def test_logs_fts_triggers(settings: Settings) -> None:
    """INSERT и UPDATE синхронизируют FTS: поиск по `reason` находит строку."""
    objects = _objects(settings)
    assert {"assistant_logs_fts_ai", "assistant_logs_fts_au"} <= objects

    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO assistant_logs (action, reason) "
            "VALUES ('deploy', 'rolled out the release')"
        )
    assert _fts_hits(settings, "release") == 1  # AFTER INSERT

    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE assistant_logs SET reason = 'reverted the patch' WHERE id = 1"
        )
    assert _fts_hits(settings, "release") == 0  # старый текст ушёл (delete + insert)
    assert _fts_hits(settings, "reverted") == 1  # новый текст проиндексирован
