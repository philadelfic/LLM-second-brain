"""Событие `queue_waiting` при отложенной работе (№39, 3.2.0; arch §6.9/§7.4).

FR-2.3/№39: «работа есть, но она не выполняется» видно в журнале не только по
`pending`, но и по `waiting` — ожидание отложенного слота/зависимости остаётся
видимым, хотя петлю в busy-loop оно не гонит. Совсем пустая очередь молчит.
"""

from __future__ import annotations

import logging

import pytest

from app.services.jobs import JobSpec, _log_queue_waiting


async def _noop() -> int:
    return 0


def _spec(queue_stat) -> JobSpec:
    """Минимальная джоба с очередью: важна только `queue_stat` для события."""
    return JobSpec(
        name="probe",
        queue="probe",
        interval_sec=30,
        batch=None,
        enabled=True,
        process=_noop,
        queue_empty=None,
        wait_event=None,
        idle_hook=None,
        queue_stat=queue_stat,
    )


def _waiting_records(caplog) -> list:
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "queue_waiting"
    ]


@pytest.mark.asyncio
async def test_event_on_waiting_only(caplog) -> None:
    """pending=0, но waiting>0 → событие `queue_waiting` несёт `waiting`."""
    caplog.clear()
    spec = _spec(lambda: {
        "pending": 0, "oldest_pending_sec": None, "waiting": 2, "attention": 1,
    })
    with caplog.at_level(logging.INFO, logger="app"):
        await _log_queue_waiting(spec)
    records = _waiting_records(caplog)
    assert len(records) == 1
    assert records[0].queue == "probe"
    assert records[0].pending == 0
    assert records[0].waiting == 2


@pytest.mark.asyncio
async def test_event_silent_on_fully_empty(caplog) -> None:
    """Совсем пустая очередь (pending=0, waiting=0) — события нет."""
    caplog.clear()
    empty = _spec(lambda: {
        "pending": 0, "oldest_pending_sec": None, "waiting": 0, "attention": 0,
    })
    with caplog.at_level(logging.INFO, logger="app"):
        await _log_queue_waiting(empty)
    assert _waiting_records(caplog) == []
