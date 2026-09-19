"""Миграция lsb-0010-02: таблица `links`, индекс и маркер `notes.links_at`.

Апгрейд живой БД предыдущей версии (v3.0.0: колонки `links_at` нет, таблицы
`links` нет) — схема создаётся при старте идемпотентно, данные не трогаются,
у существующих заметок `links_at = NULL`: первый прогон джобы `links`
естественно делает backfill (arch §3.4). Повторный `init_db` — no-op. Смена
модели/размерности эмбеддинга (автореиндексация) дополнительно очищает связи
и маркеры: вектора другой модели делают их невалидными.
"""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings
from app.storage.db import init_db, session


def _notes_columns() -> set[str]:
    with session(get_settings()) as conn:
        return {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}


def _links_ddl() -> str | None:
    with session(get_settings()) as conn:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'links'"
        ).fetchone()
        return None if row is None else row[0]


def _has_index(name: str) -> bool:
    with session(get_settings()) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND name = ?",
            (name,),
        ).fetchone()[0] == 1


def _links_rows() -> list[tuple[int, int, str]]:
    with session(get_settings()) as conn:
        return [
            (row["note_a"], row["note_b"], row["kind"])
            for row in conn.execute("SELECT * FROM links ORDER BY note_a, note_b")
        ]


def _links_at() -> dict[int, str | None]:
    with session(get_settings()) as conn:
        return {
            row["id"]: row["links_at"]
            for row in conn.execute("SELECT id, links_at FROM notes ORDER BY id")
        }


def _downgrade_to_v300(settings: Settings) -> None:
    """Привести свежую БД к состоянию v3.0.0: без таблицы `links` и маркера.

    Схема строится текущим `init_db` (все прочие таблицы, meta, FTS и вектора
    — как у живой БД предыдущей версии с тем же окружением), после чего
    снимаются ровно объекты этой постановки: таблица `links` и колонка
    `notes.links_at` (SQLite 3.35+). Так апгрейд проверяется на живой БД,
    а не на пустом файле.
    """
    with session(settings) as conn:
        conn.execute(
            "INSERT INTO notes (id, title, text, vector_status) "
            "VALUES (1, 'Старая заметка', 'текст первой заметки', 'ok')"
        )
        conn.execute(
            "INSERT INTO notes (id, title, text, vector_status) "
            "VALUES (2, 'Вторая заметка', 'текст второй заметки', 'pending')"
        )
        conn.execute("DROP TABLE IF EXISTS links")
        conn.execute("ALTER TABLE notes DROP COLUMN links_at")


@pytest.fixture
def legacy(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Живая БД предыдущей версии (v3.0.0) с двумя заметками."""
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    _downgrade_to_v300(settings)
    # Предусловие: объектов этой постановки на «старой» БД ещё нет.
    assert "links_at" not in _notes_columns()
    assert _links_ddl() is None
    return settings


def test_upgrade_creates_schema_and_preserves_notes(legacy: Settings) -> None:
    """init_db поверх v3.0.0: links + idx_links_b + links_at, данные целы."""
    init_db(get_settings())

    ddl = _links_ddl()
    assert ddl is not None
    # Канонический порядок пары — первичный ключ, а не отдельная колонка.
    assert "PRIMARY KEY (note_a, note_b)" in " ".join(ddl.split())
    assert _has_index("idx_links_b")
    assert "links_at" in _notes_columns()
    # Существующие заметки на месте, маркер пуст — backfill сделает джоба.
    assert _links_at() == {1: None, 2: None}
    with session(get_settings()) as conn:
        assert [
            (row["id"], row["title"], row["text"])
            for row in conn.execute("SELECT id, title, text FROM notes ORDER BY id")
        ] == [
            (1, "Старая заметка", "текст первой заметки"),
            (2, "Вторая заметка", "текст второй заметки"),
        ]


def test_migration_is_idempotent(legacy: Settings) -> None:
    """Повторный init_db ничего не пересоздаёт и не затирает."""
    init_db(get_settings())
    with session(get_settings()) as conn:
        conn.execute(
            "INSERT INTO links (note_a, note_b, kind, score) "
            "VALUES (1, 2, 'cosine', 0.9)"
        )
        conn.execute(
            "UPDATE notes SET links_at = '2026-01-01T00:00:00Z' WHERE id = 1"
        )

    init_db(get_settings())  # рестарт сервиса

    assert _links_rows() == [(1, 2, "cosine")]
    assert _links_at() == {1: "2026-01-01T00:00:00Z", 2: None}
    assert "links_at" in _notes_columns()
    assert "PRIMARY KEY (note_a, note_b)" in " ".join((_links_ddl() or "").split())


def test_model_change_clears_links_and_markers(
    legacy: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Смена размерности/модели (автореиндексация) чистит связи и маркеры."""
    init_db(get_settings())
    with session(get_settings()) as conn:
        conn.execute(
            "INSERT INTO links (note_a, note_b, kind, score) "
            "VALUES (1, 2, 'entities', 0.5)"
        )
        conn.execute(
            "UPDATE notes SET links_at = '2026-01-01T00:00:00Z'"
        )

    monkeypatch.setenv("EMBEDDING_DIM", "4")
    get_settings.cache_clear()
    init_db(get_settings())

    assert _links_rows() == []
    assert _links_at() == {1: None, 2: None}
    with session(get_settings()) as conn:  # вектора невалидны — все в очередь
        statuses = [
            row[0]
            for row in conn.execute("SELECT vector_status FROM notes ORDER BY id")
        ]
        assert statuses == ["pending", "pending"]
