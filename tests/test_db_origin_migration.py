"""The `notes.origin_note_id` column (3.6.2): the schema and the idempotent migration.

    An inherited DB of the previous version (no `origin_note_id` column) →
    `init_db` adds it with a PRAGMA-guarded `ALTER` (modeled on `title` and
    `comment`); the data intact, the old rows — NULL (an ordinary note, not a
    part of a group); a repeated start — a no-op. Fresh DBs get the column from
    `_NOTES_DDL`. No FK and no index: the value is a group label of the parts
    of one origin, not a reference (the backfill of the already split ones
    lives in app.storage.migration).
"""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings
from app.storage.db import init_db, session


def _notes_columns() -> set[str]:
    with session(get_settings()) as conn:
        return {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}


def _origins() -> dict[int, int | None]:
    with session(get_settings()) as conn:
        return {
            row["id"]: row["origin_note_id"]
            for row in conn.execute("SELECT id, origin_note_id FROM notes ORDER BY id")
        }


def _downgrade_to_prev(settings: Settings) -> None:
    """A fresh DB → the pre-3.6.2 state: remove the `origin_note_id` column.

                The schema is built by the current `init_db` (all other tables and
                columns — as in a live DB), after which exactly this brief's object
                is removed (SQLite 3.35+ supports `DROP COLUMN`): the upgrade is
                checked on a live DB with data, not on an empty file.
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
        # No trigger reads origin_note_id — DROP COLUMN is safe as is.
        conn.execute("ALTER TABLE notes DROP COLUMN origin_note_id")


@pytest.fixture
def legacy(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """A live DB of the previous version (before 3.6.2) with two notes."""
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    _downgrade_to_prev(settings)
    assert "origin_note_id" not in _notes_columns()  # the precondition: a "pre-3.6.2" DB
    return settings


def test_migrate_adds_origin_note_column(legacy: Settings) -> None:
    """A DB without `origin_note_id` → init_db adds it; a repeated start — a no-op."""
    init_db(get_settings())
    assert "origin_note_id" in _notes_columns()
    assert _origins() == {1: None, 2: None}  # the data intact; the old rows are no parts

    init_db(get_settings())  # idempotency: a second run changes nothing
    assert "origin_note_id" in _notes_columns()
    assert _origins() == {1: None, 2: None}


def test_fresh_db_gets_origin_note_from_ddl() -> None:
    """A fresh DB gets the column straight from `_NOTES_DDL` (the migration — a no-op)."""
    init_db(get_settings())
    with session(get_settings()) as conn:
        columns = {
            row["name"]: row for row in conn.execute("PRAGMA table_info(notes)")
        }
    assert "origin_note_id" in columns
    assert columns["origin_note_id"]["notnull"] == 0  # NULL = an ordinary note
    assert columns["origin_note_id"]["dflt_value"] is None  # no default, no FK, no index
