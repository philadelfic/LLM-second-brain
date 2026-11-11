"""The `notes.comment` column (no24-01, release 3.2.0): the schema and the idempotent migration.

    An inherited DB of the previous version (no `comment` column) → `init_db`
    adds it with a PRAGMA-guarded `ALTER` (modeled on `title`); the data intact;
    a repeated start — a no-op. Fresh DBs get the column from `_NOTES_DDL`. The
    250 limit in the schema is NOT a CHECK — the validation is domain-level
    (like `title`), so we only check the column's presence and the `NULL` default
    ("no comment").
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
    """A fresh DB → the pre-brief-24 state: remove the `comment` column.

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
        # no24-02: we also drop the comments' FTS objects (the mark of a "pre-24" version) —
        # otherwise DROP COLUMN comment fails: the trigger references new.comment.
        conn.execute("DROP TRIGGER IF EXISTS notes_comment_fts_ai")
        conn.execute("DROP TRIGGER IF EXISTS notes_comment_fts_au")
        conn.execute("DROP TABLE IF EXISTS notes_comment_fts")
        conn.execute("ALTER TABLE notes DROP COLUMN comment")


@pytest.fixture
def legacy(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """A live DB of the previous version (before brief 24) with two notes."""
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    _downgrade_to_prev(settings)
    assert "comment" not in _notes_columns()  # the precondition: a "pre-24" DB
    return settings


def test_migrate_adds_comment_column(legacy: Settings) -> None:
    """A DB without `comment` → init_db adds the column; a repeated start — a no-op."""
    init_db(get_settings())
    assert "comment" in _notes_columns()
    assert _comments() == {1: None, 2: None}  # the data intact; no comments

    init_db(get_settings())  # idempotency: a second run changes nothing
    assert "comment" in _notes_columns()
    assert _comments() == {1: None, 2: None}


def test_fresh_db_gets_comment_from_ddl() -> None:
    """A fresh DB gets the column straight from `_NOTES_DDL` (the migration — a no-op)."""
    init_db(get_settings())
    with session(get_settings()) as conn:
        columns = {
            row["name"]: row for row in conn.execute("PRAGMA table_info(notes)")
        }
    assert "comment" in columns
    assert columns["comment"]["notnull"] == 0  # NULL = no comment
    assert columns["comment"]["dflt_value"] is None  # the limit — domain-level, not a CHECK
