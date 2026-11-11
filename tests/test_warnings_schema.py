"""#12 (3.2.0): the warnings schema — the `warnings` table, indexes, the marker.

    The shared surface of memory and term warnings (#12 + #32, arch §5.1): the
    `warnings` table, the open key's partial unique index and the listing's
    working index, plus the `notes.warnings_at` service marker (§5.3/§5.4). The
    state write is idempotent: a repeated `init_db` (a restart) — a no-op.
"""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings
from app.services.warnings import KIND_NOTE_MISPLACED, WarningsService
from app.storage.db import init_db, session


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """The schema in a test DB; dimension 8 — no models involved."""
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def _objects(settings: Settings) -> set[str]:
    with session(settings) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','index')"
        ).fetchall()
    return {row["name"] for row in rows}


def _notes_columns(settings: Settings) -> set[str]:
    with session(settings) as conn:
        return {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}


def _warning_count(settings: Settings) -> int:
    with session(settings) as conn:
        return int(conn.execute("SELECT COUNT(*) AS n FROM warnings").fetchone()["n"])


def test_schema_and_marker_idempotent(settings: Settings) -> None:
    """The marker and the table/indexes exist; a repeated `init_db` does not grow the rows."""
    assert "warnings_at" in _notes_columns(settings)
    objects = _objects(settings)
    assert "warnings" in objects
    assert "idx_warnings_open_unique" in objects
    assert "idx_warnings_status_kind" in objects

    WarningsService(settings, classifier=None, namespaces=None).open(
        KIND_NOTE_MISPLACED, "note", 1, "m", {"v": 1}
    )
    before = _warning_count(settings)
    init_db(settings)  # a service restart — a no-op
    assert _warning_count(settings) == before == 1
    assert "warnings_at" in _notes_columns(settings)


def test_open_unique_partial(settings: Settings) -> None:
    """A second `open` of the same key updates the row; after a `resolve` — a new one."""
    service = WarningsService(settings, classifier=None, namespaces=None)
    first = service.open(KIND_NOTE_MISPLACED, "note", 1, "m1", {"v": 1})
    second = service.open(KIND_NOTE_MISPLACED, "note", 1, "m2", {"v": 2})
    assert first == second  # an upsert, not a second row
    with session(settings) as conn:
        rows = conn.execute(
            "SELECT message FROM warnings WHERE id = ?", (first,)
        ).fetchall()
    assert len(rows) == 1 and rows[0]["message"] == "m2"

    assert service.resolve(KIND_NOTE_MISPLACED, "note", 1) == 1
    third = service.open(KIND_NOTE_MISPLACED, "note", 1, "m3", {})
    assert third != first  # the reason returned — a new row opens
    assert _warning_count(settings) == 2
