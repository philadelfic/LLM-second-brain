"""Событие `queue_waiting` при отложенной работе (№39, 3.2.0; arch §6.9/§7.4).

FR-2.3/№39: «работа есть, но она не выполняется» видно в журнале не только по
`pending`, но и по `waiting` — ожидание отложенного слота/зависимости остаётся
видимым, хотя петлю в busy-loop оно не гонит. Совсем пустая очередь молчит.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from app.services.jobs import BackoffState, JobSpec, _log_queue_waiting, run_loop


def patch_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Мгновенный `asyncio.sleep` с записью запрошенных пауз (back-off не ждём)."""
    real_sleep = asyncio.sleep
    delays: list[float] = []

    async def spy(delay: float, *args: object, **kwargs: object) -> None:
        delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", spy)
    return delays


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


@pytest.mark.asyncio
async def test_event_once_per_iteration_on_idle_run(monkeypatch, caplog) -> None:
    """Холостой прогон: `queue_waiting` — ровно один раз на итерацию петли (№39).

    Задание, которому слот отказывает, для петли не работа (no39-05): после
    одного шанса на lost wakeup петля уходит в ожидание и пишет событие ОДИН раз
    — иначе шквал `queue_waiting` был бы зеркалом холостого шторма 12.10.2026
    (событие на каждый прогон без сна).
    """
    caplog.clear()
    delays = patch_sleep(monkeypatch)
    calls = {"runs": 0}

    async def process() -> int:
        calls["runs"] += 1
        return 0

    spec = JobSpec(
        name="probe",
        queue="probe",
        interval_sec=30,
        batch=None,
        enabled=True,
        process=process,
        queue_empty=lambda: False,
        wait_event=asyncio.Event(),
        idle_hook=None,
        queue_stat=lambda: {
            "pending": 1, "oldest_pending_sec": 0, "waiting": 0, "attention": 0,
        },
    )
    with caplog.at_level(logging.INFO, logger="app"):
        await asyncio.wait_for(
            run_loop(spec, lambda: len(delays) >= 2, BackoffState(30)), timeout=1.0
        )
    # Две итерации ожидания: по одному событию каждая (прогоны — 2 × lost wakeup).
    assert calls["runs"] == 4
    assert len(_waiting_records(caplog)) == 2
