"""The "Assistant logs" area's schema (LSBN-F-10, f10-01): the table, FTS, vec0, indexes.

    The area — the `assistant_logs` table + FTS5 of external content
    (`tokenize='trigram'`) + vec0 (`log_id`, the installation's dimension) + two
    partial reverse-query indexes (arch §5.1/§5.2/§5.3). The migration is
    idempotent and applicable from any previous version: a repeated `init_db`
    changes neither the object set nor the row count.
"""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings
from app.storage.db import init_db, session, transaction

DIM = 8


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """The schema in a test DB; dimension 8 — no models involved."""
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
    """The tables, the FTS (trigram, external content) and the vec0 (log_id, dim) created."""
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

    # The §5.1 DDL: the record columns; the source/vector_status defaults — exactly as in the arch doc.
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
    """A repeated `init_db`: the object set and the logs' row count unchanged."""
    before = _objects(settings)
    init_db(settings)  # a restart without records — a no-op
    assert _objects(settings) == before

    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO assistant_logs (action, reason) VALUES ('deploy', 'out')"
        )
    init_db(settings)  # a restart with a record — a no-op
    assert _objects(settings) == before
    assert _log_count(settings) == 1


def test_logs_indexes_present(settings: Settings) -> None:
    """Both partial reverse-query indexes exist; both over active records."""
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
    """INSERT and UPDATE sync the FTS: a search by `reason` finds the row."""
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
    assert _fts_hits(settings, "release") == 0  # the old text gone (delete + insert)
    assert _fts_hits(settings, "reverted") == 1  # the new text indexed
