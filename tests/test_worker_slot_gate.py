"""The slot gate in the job framework (#39, release 3.2.0; LSBN-T-29; arch §6.2, §6.7).

    A unit without the network: with the breaker open no batch is taken (no model
    calls burn) — since 3.4.1 the check is done by the FRAMEWORK (`JobSpec.gate`
    in `run_loop`), not a manual `if` in a job body; when the probe knock is due,
    one `probe()` lifts the breaker (`ok`) or reschedules the pause
    (`auth_failed`); a slot failure lands in the journal as an event with the
    outcome kind, and the work stays pending. An end-to-end loop run with closed
    gates (0 — not progress, no39-05) — in tests/test_jobs.py; external jobs'
    gates — in tests/test_jobs_slot_gate.py.
"""

from __future__ import annotations

import logging

import pytest
from fakes import (
    FailingEmbedder,
    FixedSummarizer,
    HashEmbedder,
    RecordingDedup,
    ScriptedJudge,
    fake_meta,
)

from app.config import get_settings
from app.services.judge import JudgeResult
from app.services.notes import NoteService
from app.services.slot_outcome import SlotOutcome
from app.services.worker import BackgroundWorker, build_judge_job
from app.storage.db import init_db


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    resolved = get_settings()
    init_db(resolved)
    return resolved


class ProbeJudge:
    """A fake judge with a controllable `probe()` — for the judge slot's gate."""

    def __init__(self, result: str = "ok", verdicts: list[bool] | None = None) -> None:
        self._probe_result = result
        self.verdicts = list(verdicts or [])
        self.judge_calls: list[tuple[str, str]] = []
        self.probe_calls = 0
        self.last_attempt_ok: bool | None = None

    def probe(self) -> str:
        self.probe_calls += 1
        return self._probe_result

    def judge(self, text_new: str, text_candidate: str) -> JudgeResult:
        self.judge_calls.append((text_new, text_candidate))
        self.last_attempt_ok = True
        verdict = self.verdicts.pop(0) if self.verdicts else False
        return JudgeResult(verdict=verdict, meta=fake_meta("judge", "judge"))

    def close(self) -> None:  # interface-compatible with JudgeService
        return None


def _freeze(monkeypatch, start: float = 1000.0) -> dict:
    """Freeze the breaker module's clock (deterministic pauses)."""
    clock = {"t": start}
    monkeypatch.setattr("app.services.slot_breaker.time.time", lambda: clock["t"])
    return clock


def _worker_with_dedup_jobs(settings, judge) -> BackgroundWorker:
    """Two notes with ready vectors → two pending dedup jobs into the queue."""
    notes = NoteService(settings, FailingEmbedder())
    notes.save("первая заметка ворот слота")
    notes.save("вторая заметка ворот слота")
    worker = BackgroundWorker(
        settings,
        HashEmbedder(8),
        FixedSummarizer("Ф."),
        RecordingDedup(candidates=[(1, 0.94)]),
        judge=judge,
    )
    assert worker.process_pending() == 2  # creates dedup jobs (the judge queue)
    assert worker._pending_jobs("judge", "dedup", 10)
    return worker


@pytest.mark.asyncio
async def test_gate_skips_batch_when_paused(settings, monkeypatch) -> None:
    """The breaker open, the knock not due → the gates closed, no model calls.

                The gate is now at the framework (`JobSpec.gate`, LSBN-T-29): closed
                gates give 0 BEFORE `process()` — the job body takes no batch, the
                judge is not called.
    """
    _freeze(monkeypatch)
    judge = ProbeJudge(result="ok")
    worker = _worker_with_dedup_jobs(settings, judge)
    worker._breakers["judge"].note_failure(SlotOutcome.AUTH)
    assert worker._breakers["judge"].is_paused() is True
    spec = build_judge_job(worker, settings)
    assert spec.gate is not None and spec.gate() is False
    assert judge.judge_calls == []  # no batch was taken
    assert judge.probe_calls == 0  # the knock is not due yet
    assert worker._pending_jobs("judge", "dedup", 10)  # the work stayed pending


@pytest.mark.asyncio
async def test_gate_probes_then_runs(settings, monkeypatch) -> None:
    """The knock is due, probe() returned ok → the breaker lifted, the batch goes."""
    clock = _freeze(monkeypatch)
    judge = ProbeJudge(result="ok", verdicts=[False])
    worker = _worker_with_dedup_jobs(settings, judge)
    worker._breakers["judge"].note_failure(SlotOutcome.AUTH)
    clock["t"] += settings.slot_pause_sec + 1  # the pause expired — time to knock
    spec = build_judge_job(worker, settings)
    assert spec.gate() is True  # one knock lifted the breaker — the batch goes now
    assert judge.probe_calls == 1
    # a note with no earlier candidates is released at once; the second one — by the judge's verdict
    assert await spec.process() == 2
    assert len(judge.judge_calls) == 1
    assert worker._breakers["judge"].is_paused() is False


@pytest.mark.asyncio
async def test_probe_auth_rearms_pause(settings, monkeypatch) -> None:
    """probe() returned auth_failed → the pause rescheduled, no batch."""
    clock = _freeze(monkeypatch)
    judge = ProbeJudge(result="auth_failed")
    worker = _worker_with_dedup_jobs(settings, judge)
    worker._breakers["judge"].note_failure(SlotOutcome.AUTH)
    clock["t"] += settings.slot_pause_sec + 1
    spec = build_judge_job(worker, settings)
    assert spec.gate() is False
    assert judge.probe_calls == 1
    assert judge.judge_calls == []
    assert worker._breakers["judge"].is_paused() is True
    assert worker._breakers["judge"].probe_due() is False


def test_failure_logged_with_outcome(settings, caplog) -> None:
    """A judge failure writes an event with job, slot, outcome, detail; the work stays pending."""
    notes = NoteService(settings, FailingEmbedder())
    notes.save("первая заметка отказа судьи")
    notes.save("вторая заметка отказа судьи")
    worker = BackgroundWorker(
        settings,
        HashEmbedder(8),
        FixedSummarizer("Ф."),
        RecordingDedup(candidates=[(1, 0.94)]),
        judge=ScriptedJudge([True], fail=True),
    )
    assert worker.process_pending() == 2
    with caplog.at_level(logging.WARNING, logger="app"):
        # The earlier note (no candidates) is released at once — 1; the failed
        # judge job stays pending (the judge queue's back-off).
        assert worker.process_judge_pending() == 1
    events = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "dedup_judge_failed"
    ]
    assert len(events) == 1
    event = events[0]
    assert event.job == "judge"
    assert event.slot == "judge"
    assert event.outcome == "off_contract"
    assert event.detail  # the reason fragment is not lost
    assert worker._pending_jobs("judge", "dedup", 10)  # the task lives in the queue
