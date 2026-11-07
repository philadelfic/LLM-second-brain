"""The summary limit migration (the 3.1.0 gate decision, 2026-09-19): 150 characters.

    Upgrading a live DB to 3.1.0: active notes with a READY summary longer than
    `MAX_SUMMARY_CHARS` are marked for regeneration (`summary_status='pending'`);
    the summary text is NOT spoiled by a rude trim — the background `summary` job
    lays the new digest (its answer is trimmed by `summary.cap_summary`). Notes
    already in the queue (`summary_status != 'ok'`) are untouched; trash (which
    the job does not serve) stays as is. Idempotency: after regeneration the
    summaries are ≤ the limit — a repeated service start finds nothing (a no-op,
    no log event); the mechanics needs no separate stamp in `meta`.
"""

from __future__ import annotations

import logging

import pytest

from app.config import Settings, get_settings
from app.storage.db import init_db, session, transaction

# Deliberately longer than any sane limit: "d" — simple characters without spaces;
# all that matters is the summary being longer than the limit.
LONG_SUMMARY = "д" * 300
SHORT_SUMMARY = "Короткое саммари заметки."


def _seed(settings: Settings) -> None:
    """A live DB before the upgrade: four notes with different summary states.

            1 — a ready long one (the migration must set pending, the text untouched);
            2 — a ready short one (within the limit — untouched);
            3 — already pending with a long text (in the queue anyway — the status not
            rewritten);
            4 — trash with a ready long one (the `summary` job does not serve trash).
    """
    with session(settings) as conn, transaction(conn):
        seeded = (
            (1, LONG_SUMMARY, "ok", None),
            (2, SHORT_SUMMARY, "ok", None),
            (3, LONG_SUMMARY, "pending", None),
            (4, LONG_SUMMARY, "ok", "2026-09-19T00:00:00Z"),
        )
        for note_id, summary, summary_status, deleted_at in seeded:
            conn.execute(
                "INSERT INTO notes (id, text, title, summary, summary_status, "
                "deleted_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    note_id,
                    f"текст заметки номер {note_id} про релиз и деплой",
                    f"Заметка {note_id}",
                    summary,
                    summary_status,
                    deleted_at,
                ),
            )


@pytest.fixture
def upgraded(test_env: dict[str, str]) -> Settings:
    """A fresh DB (schema + seeds) and the data before the summary limit migration."""
    settings = get_settings()
    init_db(settings)
    _seed(settings)
    return settings


def _summaries(settings: Settings) -> dict[int, tuple[str, str]]:
    """A snapshot of all notes' summary fields: id → (text, status)."""
    with session(settings) as conn:
        return {
            row["id"]: (row["summary"], row["summary_status"])
            for row in conn.execute(
                "SELECT id, summary, summary_status FROM notes ORDER BY id"
            )
        }


def _regen_events(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "summary_regen_queued"
    ]


def test_upgrade_queues_only_long_ready_summaries(
    upgraded: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """The migration at start: pending — only ready summaries longer than the limit."""
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="app"):
        init_db(get_settings())

    summaries = _summaries(get_settings())
    # 1 — a long ready one → regeneration; the summary TEXT is not spoiled by trimming.
    assert summaries[1] == (LONG_SUMMARY, "pending")
    # 2 — a short ready one within the limit → untouched.
    assert summaries[2] == (SHORT_SUMMARY, "ok")
    # 3 — already pending: in the queue anyway; the status and text as they were.
    assert summaries[3] == (LONG_SUMMARY, "pending")
    # 4 — trash: the job does not serve it; we never create an eternal pending.
    assert summaries[4] == (LONG_SUMMARY, "ok")

    events = _regen_events(caplog)
    assert len(events) == 1
    assert events[0].levelno == logging.WARNING
    assert events[0].notes == 1  # exactly one note went to regeneration
    assert events[0].max_summary_chars == get_settings().max_summary_chars


def test_migration_is_idempotent(
    upgraded: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """The regeneration done → a repeated start — a no-op (without a log event)."""
    init_db(get_settings())  # the migration: note 1 → pending
    # The background `summary` job redid the summary (cap_summary holds the limit).
    with session(get_settings()) as conn, transaction(conn):
        conn.execute(
            "UPDATE notes SET summary = ?, summary_status = 'ok' WHERE id = 1",
            (SHORT_SUMMARY,),
        )

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="app"):
        init_db(get_settings())  # a repeated service start
        init_db(get_settings())  # one more — the migration does not resurrect

    assert _regen_events(caplog) == []  # nothing left to migrate
    assert _summaries(get_settings()) == {
        1: (SHORT_SUMMARY, "ok"),
        2: (SHORT_SUMMARY, "ok"),
        3: (LONG_SUMMARY, "pending"),  # still waiting for the job (its own scenario)
        4: (LONG_SUMMARY, "ok"),  # trash was never migrated
    }


def test_limit_comes_from_settings(
    upgraded: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The migration threshold — exactly `settings.max_summary_chars`, not a baked-in 150."""
    monkeypatch.setenv("MAX_SUMMARY_CHARS", "50")
    get_settings.cache_clear()
    try:
        init_db(get_settings())
        summaries = _summaries(get_settings())
    finally:
        get_settings.cache_clear()
    # A short summary (< 50 chars) stays 'ok'; a long one (300) — into the queue.
    assert summaries[2] == (SHORT_SUMMARY, "ok")
    assert summaries[1] == (LONG_SUMMARY, "pending")
