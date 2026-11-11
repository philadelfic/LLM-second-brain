"""The lsb-0010-02 migration: the `links` table, the index and the `notes.links_at` marker.

    Upgrading a live DB of the previous version (v3.0.0: no `links_at` column, no
    `links` table) — the schema is created idempotently at startup, the data is
    untouched; existing notes have `links_at = NULL`: the `links` job's first run
    naturally does the backfill (arch §3.4). A repeated `init_db` — a no-op. An
    embedding model/dimension change (auto-reindex) additionally clears the links
    and markers: another model's vectors make them invalid.
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
    """Bring a fresh DB to the v3.0.0 state: without the `links` table and the marker.

                The schema is built by the current `init_db` (all other tables, meta,
                FTS and vectors — as in a live DB of the previous version with the
                same environment), after which exactly this brief's objects are
                removed: the `links` table and the `notes.links_at` column
                (SQLite 3.35+). This way the upgrade is checked on a live DB, not on
                an empty file.
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
    """A live DB of the previous version (v3.0.0) with two notes."""
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    _downgrade_to_v300(settings)
    # The precondition: this brief's objects are not on the "old" DB yet.
    assert "links_at" not in _notes_columns()
    assert _links_ddl() is None
    return settings


def test_upgrade_creates_schema_and_preserves_notes(legacy: Settings) -> None:
    """init_db over v3.0.0: links + idx_links_b + links_at; the data intact."""
    init_db(get_settings())

    ddl = _links_ddl()
    assert ddl is not None
    # The pair's canonical order — the primary key, not a separate column.
    assert "PRIMARY KEY (note_a, note_b)" in " ".join(ddl.split())
    assert _has_index("idx_links_b")
    assert "links_at" in _notes_columns()
    # Existing notes in place, the marker empty — the job will do the backfill.
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
    """A repeated init_db recreates and overwrites nothing."""
    init_db(get_settings())
    with session(get_settings()) as conn:
        conn.execute(
            "INSERT INTO links (note_a, note_b, kind, score) "
            "VALUES (1, 2, 'cosine', 0.9)"
        )
        conn.execute(
            "UPDATE notes SET links_at = '2026-01-01T00:00:00Z' WHERE id = 1"
        )

    init_db(get_settings())  # a service restart

    assert _links_rows() == [(1, 2, "cosine")]
    assert _links_at() == {1: "2026-01-01T00:00:00Z", 2: None}
    assert "links_at" in _notes_columns()
    assert "PRIMARY KEY (note_a, note_b)" in " ".join((_links_ddl() or "").split())


def test_model_change_clears_links_and_markers(
    legacy: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dimension/model change (auto-reindex) clears the links and markers."""
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
    with session(get_settings()) as conn:  # the vectors invalid — all into the queue
        statuses = [
            row[0]
            for row in conn.execute("SELECT vector_status FROM notes ORDER BY id")
        ]
        assert statuses == ["pending", "pending"]
