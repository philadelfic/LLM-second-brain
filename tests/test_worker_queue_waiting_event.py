"""The `queue_waiting` event at deferred work (#39, 3.2.0; arch §6.9/§7.4).

    FR-2.3/#39: "there is work, but it is not being done" is visible in the journal
    not only by `pending` but by `waiting` too — waiting for a deferred
    slot/dependency stays visible although it does not drive the loop into a
    busy-loop. A fully empty queue stays silent.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from app.services.jobs import BackoffState, JobSpec, _log_queue_waiting, run_loop


def patch_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """An instant `asyncio.sleep` recording the requested pauses (no back-off waiting)."""
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
    """A minimal job with a queue: only `queue_stat` matters for the event."""
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
    """pending=0 but waiting>0 → the `queue_waiting` event carries `waiting`."""
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
    """A fully empty queue (pending=0, waiting=0) — no event."""
    caplog.clear()
    empty = _spec(lambda: {
        "pending": 0, "oldest_pending_sec": None, "waiting": 0, "attention": 0,
    })
    with caplog.at_level(logging.INFO, logger="app"):
        await _log_queue_waiting(empty)
    assert _waiting_records(caplog) == []


@pytest.mark.asyncio
async def test_event_once_per_iteration_on_idle_run(monkeypatch, caplog) -> None:
    """An idle run: `queue_waiting` — exactly once per loop iteration (#39).

                A task the slot refuses is not work for the loop (no39-05): after one
                lost-wakeup chance the loop goes into a wait and writes the event ONCE
                — otherwise a hail of `queue_waiting` would mirror the idle storm of
                2026-10-12 (an event per run without sleep).
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
    # Two wait iterations: one event each (the runs — 2 × lost wakeup).
    assert calls["runs"] == 4
    assert len(_waiting_records(caplog)) == 2
