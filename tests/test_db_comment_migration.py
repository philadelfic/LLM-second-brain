"""Колонка `notes.comment` (no24-01, релиз 3.2.0): схема и идемпотентная миграция.

Унаследованная БД предыдущей версии (колонки `comment` нет) → `init_db`
добавляет её PRAGMA-guarded `ALTER` (по образцу `title`), данные целы; повторный
старт — no-op. Свежие БД получают колонку из `_NOTES_DDL`. Лимит 250 в схеме
НЕ CHECK — валидация доменная (как у `title`), поэтому проверяем только
наличие колонки и `NULL` по умолчанию («комментария нет»).
"""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings
from app.storage.db import init_db, session


def _notes_columns() -> set[str]:
    with session(get_settings()) as conn:
        return {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}


def _comments() -> dict[int, str | None]:
    with session(get_settings()) as conn:
        return {
            row["id"]: row["comment"]
            for row in conn.execute("SELECT id, comment FROM notes ORDER BY id")
        }


def _downgrade_to_prev(settings: Settings) -> None:
    """Свежая БД → состояние до постановки 24: снять колонку `comment`.

    Схема строится текущим `init_db` (все прочие таблицы и колонки — как у
    живой БД), после чего снимается ровно объект этой постановки (SQLite
    3.35+ поддерживает `DROP COLUMN`): апгрейд проверяется на живой БД с
    данными, а не на пустом файле.
    """
    with session(settings) as conn:
        conn.execute(
            "INSERT INTO notes (id, text, title, namespace) "
            "VALUES (1, 'текст первой заметки', 'Старая заметка', 'default')"
        )
        conn.execute(
            "INSERT INTO notes (id, text, title) "
            "VALUES (2, 'текст второй заметки', 'Вторая заметка')"
        )
        # no24-02: снимаем и объекты FTS пометок (признак версии «до №24») —
        # иначе DROP COLUMN comment падает: триггер ссылается на new.comment.
        conn.execute("DROP TRIGGER IF EXISTS notes_comment_fts_ai")
        conn.execute("DROP TRIGGER IF EXISTS notes_comment_fts_au")
        conn.execute("DROP TABLE IF EXISTS notes_comment_fts")
        conn.execute("ALTER TABLE notes DROP COLUMN comment")


@pytest.fixture
def legacy(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Живая БД предыдущей версии (до постановки 24) с двумя заметками."""
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    _downgrade_to_prev(settings)
    assert "comment" not in _notes_columns()  # предусловие: БД «до №24»
    return settings


def test_migrate_adds_comment_column(legacy: Settings) -> None:
    """БД без `comment` → init_db добавляет колонку; повторный старт — no-op."""
    init_db(get_settings())
    assert "comment" in _notes_columns()
    assert _comments() == {1: None, 2: None}  # данные целы, комментариев нет

    init_db(get_settings())  # идемпотентность: второй прогон ничего не меняет
    assert "comment" in _notes_columns()
    assert _comments() == {1: None, 2: None}


def test_fresh_db_gets_comment_from_ddl() -> None:
    """Свежая БД получает колонку сразу из `_NOTES_DDL` (миграция — no-op)."""
    init_db(get_settings())
    with session(get_settings()) as conn:
        columns = {
            row["name"]: row for row in conn.execute("PRAGMA table_info(notes)")
        }
    assert "comment" in columns
    assert columns["comment"]["notnull"] == 0  # NULL = комментария нет
    assert columns["comment"]["dflt_value"] is None  # лимит — доменный, не CHECK
