"""The origin backfill of the split parts (3.6.2): notes.origin_note_id from migration_notes.

    The source of truth is the `migration_notes` row of a split source
    ('нарезана' + the new_ids JSON of its parts), not a meta stamp — meta is
    poured between bases. Idempotency lives in the data
    (`WHERE origin_note_id IS NULL`): a second start fills nothing and an
    already filled part is never overwritten. A base without the migration
    table is a zero answer, a broken new_ids JSON is skipped with a warning.
"""

from __future__ import annotations

import json
import logging

import pytest

from app.config import get_settings
from app.storage.db import init_db, session, transaction
from app.storage.migration import (
    MIGRATION_NOTES_TABLE,
    backfill_origin_note_ids,
    ensure_startup_schema,
)


def _seed(notes: list[int], sources: list[tuple[int, str | None]]) -> None:
    """Notes with the given ids + migration_notes rows (new_ids — a raw string)."""
    with session(get_settings()) as conn, transaction(conn):
        for note_id in notes:
            conn.execute(
                "INSERT INTO notes (id, text) VALUES (?, ?)",
                (note_id, f"заметка {note_id}"),
            )
        for old_id, new_ids in sources:
            conn.execute(
                f"INSERT INTO {MIGRATION_NOTES_TABLE} (old_id, status, new_ids) "
                "VALUES (?, 'нарезана', ?)",
                (old_id, new_ids),
            )


def _origins() -> dict[int, int | None]:
    with session(get_settings()) as conn:
        return {
            row["id"]: row["origin_note_id"]
            for row in conn.execute("SELECT id, origin_note_id FROM notes ORDER BY id")
        }


def test_backfill_fills_every_part_of_every_source() -> None:
    """Two sources × N parts → every part carries the id of its own source."""
    init_db(get_settings())
    ensure_startup_schema(get_settings())
    _seed(
        notes=[11, 12, 13, 21, 22],
        sources=[(1, json.dumps([11, 12])), (2, json.dumps([21, 22, 13]))],
    )
    counters = backfill_origin_note_ids(get_settings())
    assert counters == {"sources": 2, "parts": 5, "filled": 5}
    assert _origins() == {11: 1, 12: 1, 13: 2, 21: 2, 22: 2}


def test_existing_origin_is_not_overwritten_and_second_run_is_noop() -> None:
    """The idempotency: a filled value stays, a repeated run fills zero."""
    init_db(get_settings())
    ensure_startup_schema(get_settings())
    _seed(notes=[11, 12], sources=[(1, json.dumps([11, 12]))])
    with session(get_settings()) as conn, transaction(conn):
        conn.execute("UPDATE notes SET origin_note_id = 99 WHERE id = 11")

    first = backfill_origin_note_ids(get_settings())
    assert first == {"sources": 1, "parts": 2, "filled": 1}
    assert _origins() == {11: 99, 12: 1}  # the known origin is not overwritten

    second = backfill_origin_note_ids(get_settings())
    assert second == {"sources": 1, "parts": 2, "filled": 0}
    assert _origins() == {11: 99, 12: 1}


def test_missing_migration_notes_is_noop() -> None:
    """A base that never migrated into has no migration_notes — zero, no error."""
    init_db(get_settings())
    with session(get_settings()) as conn:
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (MIGRATION_NOTES_TABLE,),
        ).fetchone() is None
    assert backfill_origin_note_ids(get_settings()) == {
        "sources": 0,
        "parts": 0,
        "filled": 0,
    }


def test_empty_migration_notes_is_noop() -> None:
    """The prod order (init_db + ensure_startup_schema): an empty table — zero."""
    init_db(get_settings())
    ensure_startup_schema(get_settings())
    assert backfill_origin_note_ids(get_settings()) == {
        "sources": 0,
        "parts": 0,
        "filled": 0,
    }


def test_broken_json_is_skipped_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A broken new_ids JSON — the source is skipped, the whole run is not."""
    init_db(get_settings())
    ensure_startup_schema(get_settings())
    _seed(
        notes=[11, 21],
        sources=[(1, json.dumps([11])), (2, "{не json")],
    )
    with caplog.at_level(logging.WARNING, logger="app"):
        counters = backfill_origin_note_ids(get_settings())
    assert counters == {"sources": 2, "parts": 1, "filled": 1}
    assert _origins() == {11: 1, 21: None}
    events = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "origin_backfill_broken"
    ]
    assert [record.old_id for record in events] == [2]


def test_counters_are_logged_only_when_something_is_filled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The counters go to the log once and only when filled > 0."""
    init_db(get_settings())
    ensure_startup_schema(get_settings())
    _seed(notes=[11, 12], sources=[(1, json.dumps([11, 12]))])
    with caplog.at_level(logging.INFO, logger="app"):
        backfill_origin_note_ids(get_settings())
    events = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "origin_backfill"
    ]
    assert len(events) == 1
    assert (events[0].sources, events[0].parts, events[0].filled) == (1, 2, 2)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="app"):
        backfill_origin_note_ids(get_settings())
    assert [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "origin_backfill"
    ] == []
