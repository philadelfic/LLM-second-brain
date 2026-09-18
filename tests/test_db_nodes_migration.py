"""Миграция lsb-0011-01: служебный маркер `notes.node_order_at`.

Апгрейд живой БД предыдущей версии (v3.1.0 до постановки 11: колонки
`node_order_at` нет) — колонка создаётся при старте идемпотентно, данные не
трогаются, у существующих заметок `node_order_at = NULL`: первый прогон джобы
`nodes` естественно разбирает накопленный `default` (ретро-прогон). Повторный
`init_db` — no-op.
"""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings
from app.storage.db import init_db, session


def _notes_columns() -> set[str]:
    with session(get_settings()) as conn:
        return {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}


def _node_order() -> dict[int, str | None]:
    with session(get_settings()) as conn:
        return {
            row["id"]: row["node_order_at"]
            for row in conn.execute("SELECT id, node_order_at FROM notes ORDER BY id")
        }


def _downgrade_to_prev(settings: Settings) -> None:
    """Привести свежую БД к состоянию до постановки 11: без маркера обхода.

    Схема строится текущим `init_db` (все прочие таблицы и колонки — как у
    живой БД), после чего снимается ровно объект этой постановки: колонка
    `notes.node_order_at` (SQLite 3.35+). Так апгрейд проверяется на живой БД,
    а не на пустом файле.
    """
    with session(settings) as conn:
        conn.execute(
            "INSERT INTO notes (id, title, text, namespace, hint_path, "
            "confidence, vector_status) VALUES "
            "(1, 'Старая заметка', 'текст первой заметки', 'default', "
            "'work', 0.9, 'ok')"
        )
        conn.execute(
            "INSERT INTO notes (id, title, text, vector_status) "
            "VALUES (2, 'Вторая заметка', 'текст второй заметки', 'pending')"
        )
        conn.execute("ALTER TABLE notes DROP COLUMN node_order_at")


@pytest.fixture
def legacy(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Живая БД предыдущей версии (до постановки 11) с двумя заметками."""
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    _downgrade_to_prev(settings)
    # Предусловие: объекта этой постановки на «старой» БД ещё нет.
    assert "node_order_at" not in _notes_columns()
    return settings


def test_upgrade_adds_marker_and_preserves_notes(legacy: Settings) -> None:
    """init_db поверх предыдущей версии: колонка node_order_at, данные целы."""
    init_db(get_settings())

    assert "node_order_at" in _notes_columns()
    # У существующих заметок маркер пуст — обход разберёт их первым прогоном.
    assert _node_order() == {1: None, 2: None}
    with session(get_settings()) as conn:
        rows = [
            (row["id"], row["title"], row["text"], row["hint_path"], row["confidence"])
            for row in conn.execute(
                "SELECT id, title, text, hint_path, confidence FROM notes ORDER BY id"
            )
        ]
    assert rows == [
        (1, "Старая заметка", "текст первой заметки", "work", 0.9),
        (2, "Вторая заметка", "текст второй заметки", None, None),
    ]


def test_migration_is_idempotent(legacy: Settings) -> None:
    """Повторный init_db ничего не пересоздаёт и не затирает маркер."""
    init_db(get_settings())
    with session(get_settings()) as conn:
        conn.execute(
            "UPDATE notes SET node_order_at = '2026-01-01T00:00:00Z' WHERE id = 1"
        )

    init_db(get_settings())  # рестарт сервиса

    assert "node_order_at" in _notes_columns()
    assert _node_order() == {1: "2026-01-01T00:00:00Z", 2: None}
