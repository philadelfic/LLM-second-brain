"""Releasing "needs attention" on a slot configuration change (lsbdef0008-02/-03).

    The worker's start compares the current slot configuration fingerprint
    (`slot_budget.slot_config_fingerprint`) with the one recorded in `meta`
    (block 2, `SLOT_CONFIG_META_KEY`): no fingerprint or a different one — "the
    reason is gone"; attention is cleared for the tasks of ALL slots
    (`attention_at=NULL`, `attempts=0`) with a `job_attention_released` event of
    `reason="startup"`; the same fingerprint — nothing is cleared.
    `SLOT_ATTENTION_RESET_ON_START=false` disables the step entirely (the
    operator's will). The fingerprint includes "think / not think" (`*_THINK`)
    and the client timeout (`*_TIMEOUT_SEC`) — changing either clears the
    attention by itself (the 2026-10-12 acceptance defect: the `JUDGE_THINK=false`
    workaround + a small budget did not unfreeze pairs). The attempt counter, the
    back-off and the queue observability — block 2 (no39-02), not duplicated
    here.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from fakes import FailingEmbedder

from app.config import Settings, get_settings
from app.services.slot_budget import (
    FINGERPRINT_DEFAULTS,
    FINGERPRINT_FIELDS,
    slot_config_fingerprint,
)
from app.services.worker import SLOT_CONFIG_META_KEY, BackgroundWorker
from app.storage.db import init_db, session, transaction


def _settings(tmp_path, monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    resolved = get_settings()
    init_db(resolved)
    return resolved


def _worker(settings) -> BackgroundWorker:
    worker = BackgroundWorker(settings, FailingEmbedder())
    worker._ensure_job_table()
    return worker


def _seed_job(
    settings,
    *,
    slot: str = "judge",
    kind: str = "dedup",
    note_id: int = 1,
) -> int:
    _worker(settings)
    with session(settings) as conn, transaction(conn):
        cursor = conn.execute(
            "INSERT INTO worker_jobs (slot, kind, note_id, payload) "
            "VALUES (?, ?, ?, ?)",
            (slot, kind, note_id, None),
        )
        return int(cursor.lastrowid or 0)


def _frozen_job(settings, job_id: int) -> None:
    """A "needs attention" task: attempts exhausted, attention_at set."""
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE worker_jobs SET attempts = 8, last_outcome = 'truncated', "
            "attention_at = '2026-01-01T00:00:00Z' WHERE id = ?",
            (job_id,),
        )


def _write_fingerprint(settings, value: str) -> None:
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (SLOT_CONFIG_META_KEY, value),
        )


def _stored_fingerprint(settings) -> str | None:
    with session(settings) as conn:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (SLOT_CONFIG_META_KEY,)
        ).fetchone()
    return row["value"] if row is not None else None


def _fingerprint_with(settings, **overrides: object) -> str:
    """A fingerprint of the "old" slot configuration: the fields from `Settings`, some replaced.

                The carrier is built over `FINGERPRINT_FIELDS` and reads the values
                from `Settings` (the defaults — as in battle), so the test needs no
                second settings instance or DB: it puts the "before the change"
                configuration fingerprint into `meta`, while the worker starts
                "after".
    """
    values = {
        name: getattr(settings, name, FINGERPRINT_DEFAULTS.get(name, ""))
        for name in FINGERPRINT_FIELDS
    }
    values.update(overrides)
    return slot_config_fingerprint(SimpleNamespace(**values))


def _job_row(settings, job_id: int) -> dict:
    with session(settings) as conn:
        row = conn.execute(
            "SELECT attempts, last_outcome, attention_at, status "
            "FROM worker_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    return dict(row) if row is not None else {}


def _events(caplog, event: str) -> list:
    return [
        record for record in caplog.records if getattr(record, "event", None) == event
    ]


def test_changed_fingerprint_releases_attention(tmp_path, monkeypatch, caplog) -> None:
    """The fingerprint changed → attention cleared, the counter reset, a `startup` event."""
    settings = _settings(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO, logger="app")
    worker = _worker(settings)
    job_id = _seed_job(settings, note_id=1)
    _frozen_job(settings, job_id)
    _write_fingerprint(settings, "old-fingerprint")

    released = worker._startup_attention_reset()

    assert released == 1
    row = _job_row(settings, job_id)
    assert row["attention_at"] is None
    assert row["attempts"] == 0
    assert row["status"] == "pending"  # the task is alive, not "dead"
    events = _events(caplog, "job_attention_released")
    assert events and getattr(events[-1], "reason", None) == "startup"
    assert _stored_fingerprint(settings) == slot_config_fingerprint(settings)


def test_missing_fingerprint_releases_attention(tmp_path, monkeypatch) -> None:
    """No fingerprint in `meta` (the first start with the mechanism) → attention cleared."""
    settings = _settings(tmp_path, monkeypatch)
    worker = _worker(settings)
    job_id = _seed_job(settings, note_id=1)
    _frozen_job(settings, job_id)

    assert worker._startup_attention_reset() == 1

    assert _job_row(settings, job_id)["attention_at"] is None
    assert _stored_fingerprint(settings) == slot_config_fingerprint(settings)


def test_same_fingerprint_keeps_attention(tmp_path, monkeypatch) -> None:
    """The same fingerprint → nothing is cleared (the fingerprint does not "jerk")."""
    settings = _settings(tmp_path, monkeypatch)
    worker = _worker(settings)
    job_id = _seed_job(settings, note_id=1)
    _frozen_job(settings, job_id)
    _write_fingerprint(settings, slot_config_fingerprint(settings))

    assert worker._startup_attention_reset() == 0

    row = _job_row(settings, job_id)
    assert row["attention_at"] is not None
    assert row["attempts"] == 8


def test_think_change_releases_attention(tmp_path, monkeypatch, caplog) -> None:
    """A "think / not think" change at startup unfreezes the pairs (the 2026-10-12 defect).

                The slots were on the temporary `SUMMARY_THINK=false` +
                `JUDGE_THINK=true` workaround; after returning to regular reasoning
                the attention clears without manual actions (FR-2.3); the fingerprint
                in `meta` is rewritten with the current one.
    """
    settings = _settings(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO, logger="app")
    worker = _worker(settings)
    job_id = _seed_job(settings, note_id=1)
    _frozen_job(settings, job_id)
    # the "before the change" configuration: reasoning off at summary, on at judge
    _write_fingerprint(
        settings, _fingerprint_with(settings, summary_think=False, judge_think=True)
    )

    assert worker._startup_attention_reset() == 1

    row = _job_row(settings, job_id)
    assert row["attention_at"] is None
    assert row["attempts"] == 0
    assert row["status"] == "pending"
    events = _events(caplog, "job_attention_released")
    assert events and getattr(events[-1], "reason", None) == "startup"
    assert _stored_fingerprint(settings) == slot_config_fingerprint(settings)


def test_same_think_keeps_attention(tmp_path, monkeypatch) -> None:
    """The same `THINK` → attention is not cleared (no false resets)."""
    settings = _settings(
        tmp_path, monkeypatch, SUMMARY_THINK="true", JUDGE_THINK="true"
    )
    worker = _worker(settings)
    job_id = _seed_job(settings, note_id=1)
    _frozen_job(settings, job_id)
    _write_fingerprint(settings, slot_config_fingerprint(settings))

    assert worker._startup_attention_reset() == 0

    row = _job_row(settings, job_id)
    assert row["attention_at"] is not None
    assert row["attempts"] == 8
    # the field really takes part in the comparison: otherwise the "same THINK" test means nothing
    assert _fingerprint_with(settings, judge_think=False) != _stored_fingerprint(
        settings
    )


@pytest.mark.parametrize(
    "override",
    [{"summary_timeout_sec": 120}, {"judge_timeout_sec": 60}],
    ids=["summary", "judge"],
)
def test_timeout_change_releases_attention(
    tmp_path, monkeypatch, caplog, override
) -> None:
    """Changing only `*_TIMEOUT_SEC` (without the budget) also clears the attention.

                The timeout changes the call's outcome (`timeout` instead of an
                answer) — part of "the reason is gone", so a timeout edit unfreezes
                the pair (arch §13.4).
    """
    settings = _settings(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO, logger="app")
    worker = _worker(settings)
    job_id = _seed_job(settings, note_id=1)
    _frozen_job(settings, job_id)
    _write_fingerprint(settings, _fingerprint_with(settings, **override))

    assert worker._startup_attention_reset() == 1

    row = _job_row(settings, job_id)
    assert row["attention_at"] is None
    assert row["attempts"] == 0
    events = _events(caplog, "job_attention_released")
    assert events and getattr(events[-1], "reason", None) == "startup"
    assert _stored_fingerprint(settings) == slot_config_fingerprint(settings)


def test_reset_disabled_by_flag(tmp_path, monkeypatch) -> None:
    """`SLOT_ATTENTION_RESET_ON_START=false` disables the step entirely."""
    settings = _settings(
        tmp_path, monkeypatch, SLOT_ATTENTION_RESET_ON_START="false"
    )
    worker = _worker(settings)
    job_id = _seed_job(settings, note_id=1)
    _frozen_job(settings, job_id)
    _write_fingerprint(settings, "old-fingerprint")

    assert worker._startup_attention_reset() == 0

    assert _job_row(settings, job_id)["attention_at"] is not None
    assert _stored_fingerprint(settings) == "old-fingerprint"


def test_release_then_queue_drains(tmp_path, monkeypatch) -> None:
    """The block-2 boundary: after the attention is cleared the judge queue is taken again."""
    settings = _settings(tmp_path, monkeypatch)
    worker = _worker(settings)
    job_id = _seed_job(settings, slot="judge", kind="dedup", note_id=1)
    _frozen_job(settings, job_id)
    _write_fingerprint(settings, "old-fingerprint")
    # a frozen task does not drive the loop (no busy-loop)
    assert worker._judge_queue_empty() is True

    assert worker._startup_attention_reset() == 1

    assert worker._judge_queue_empty() is False
    taken = [int(row["id"]) for row in worker._pending_jobs("judge", "dedup", 10)]
    assert job_id in taken
