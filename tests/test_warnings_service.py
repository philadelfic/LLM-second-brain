"""#12 (3.2.0): the warnings service — upsert, clearing, aggregates (arch §6.5).

    `WarningsService` — the shared `warnings` surface's state: opening/updating
    by the open key, clearing by the system and by a human, the listing and the
    counters. SQL only, no model calls (§13.4).
"""

from __future__ import annotations

import json
import logging

import pytest

from app.config import Settings, get_settings
from app.services.warnings import (
    KIND_NOTE_MISPLACED,
    KIND_NOTE_NO_NODE,
    KIND_TERM_CONTEXTS_EXCESS,
    WarningsService,
)
from app.storage.db import init_db, session, transaction


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def _service(settings: Settings) -> WarningsService:
    return WarningsService(settings, classifier=None, namespaces=None)


def _row(settings: Settings, warning_id: int):
    with session(settings) as conn:
        return conn.execute(
            "SELECT status, message, payload, created_at, updated_at, resolved_at "
            "FROM warnings WHERE id = ?",
            (warning_id,),
        ).fetchone()


def _events(caplog, event: str) -> list:
    return [r for r in caplog.records if getattr(r, "event", None) == event]


def test_open_upsert_updates_payload(settings: Settings, caplog) -> None:
    """The upsert moves message/payload/updated_at; `warning_opened` — once."""
    service = _service(settings)
    with caplog.at_level(logging.INFO, logger="app"):
        warning_id = service.open(KIND_NOTE_MISPLACED, "note", 1, "m1", {"v": 1})
        # We push updated_at into the past: the upsert moving it becomes visible.
        with session(settings) as conn, transaction(conn):
            conn.execute(
                "UPDATE warnings SET updated_at = '2000-01-01T00:00:00Z' "
                "WHERE id = ?",
                (warning_id,),
            )
        service.open(KIND_NOTE_MISPLACED, "note", 1, "m2", {"v": 2})

    row = _row(settings, warning_id)
    assert row["message"] == "m2"
    assert json.loads(row["payload"]) == {"v": 2}
    assert row["updated_at"] > "2000-01-01T00:00:00Z"
    assert len(_events(caplog, "warning_opened")) == 1  # the update does not multiply


def test_dismiss_and_idempotent(settings: Settings, caplog) -> None:
    """`dismiss` → dismissed/resolved_at; a repeat — a soft answer with a hint."""
    service = _service(settings)
    warning_id = service.open(KIND_NOTE_MISPLACED, "note", 1, "m", {})
    with caplog.at_level(logging.INFO, logger="app"):
        first = service.dismiss(warning_id)
    assert first["dismissed"] is True
    assert first["status"] == "dismissed"
    assert first["resolved_at"] is not None
    row = _row(settings, warning_id)
    assert row["status"] == "dismissed" and row["resolved_at"] is not None

    second = service.dismiss(warning_id)
    assert second["dismissed"] is False
    assert "hint" in second
    assert len(_events(caplog, "warning_dismissed")) == 1


def test_resolve_for_subject_clears_all(settings: Settings) -> None:
    """ALL of a subject's open warnings are cleared; foreign ones — not."""
    service = _service(settings)
    service.open(KIND_NOTE_MISPLACED, "note", 1, "m1", {})
    service.open(KIND_NOTE_NO_NODE, "note", 1, "m2", {})
    service.open(KIND_NOTE_MISPLACED, "note", 2, "m3", {})

    assert service.resolve_for_subject("note", 1) == 2
    with session(settings) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM warnings WHERE subject_id = 1 "
            "AND status = 'open' AND resolved_at IS NULL"
        ).fetchone()
        other = conn.execute(
            "SELECT resolved_at FROM warnings WHERE subject_id = 2"
        ).fetchone()
    assert row["n"] == 0
    assert other["resolved_at"] is None  # a foreign note is untouched


def test_counts_by_kind(settings: Settings) -> None:
    """`counts()` groups the open ones by kind; cleared/dismissed not counted."""
    service = _service(settings)
    service.open(KIND_NOTE_MISPLACED, "note", 1, "m1", {})
    service.open(KIND_NOTE_MISPLACED, "note", 2, "m2", {})
    service.open(KIND_NOTE_NO_NODE, "note", 3, "m3", {})
    dismissed = service.open(KIND_TERM_CONTEXTS_EXCESS, "term", 5, "m4", {})
    service.dismiss(dismissed)
    service.resolve(KIND_NOTE_MISPLACED, "note", 1)

    counts = service.counts()
    assert counts["by_kind"][KIND_NOTE_MISPLACED] == 1
    assert counts["by_kind"][KIND_NOTE_NO_NODE] == 1
    assert KIND_TERM_CONTEXTS_EXCESS not in counts["by_kind"]
    assert counts["total"] == 2
