"""Ворота слота в воркере (№39, релиз 3.2.0; arch §6.2, §6.7).

Юнит без сети: при открытом разрыве партия не берётся (вызовы модели не
жгутся); при наступившем пробном стуке один `probe()` снимает разрыв
(`ok`) или переставляет паузу (`auth_failed`); отказ слота попадает в
журнал событием с типом исхода, а работа остаётся pending.
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
)

from app.config import get_settings
from app.services.notes import NoteService
from app.services.slot_outcome import SlotOutcome
from app.services.worker import BackgroundWorker
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
    """Судья-фейк с управляемым `probe()` — для ворот слота judge."""

    def __init__(self, result: str = "ok", verdicts: list[bool] | None = None) -> None:
        self._probe_result = result
        self.verdicts = list(verdicts or [])
        self.judge_calls: list[tuple[str, str]] = []
        self.probe_calls = 0
        self.last_attempt_ok: bool | None = None

    def probe(self) -> str:
        self.probe_calls += 1
        return self._probe_result

    def judge(self, text_new: str, text_candidate: str) -> bool:
        self.judge_calls.append((text_new, text_candidate))
        self.last_attempt_ok = True
        return self.verdicts.pop(0) if self.verdicts else False

    def close(self) -> None:  # интерфейс-совместимость с JudgeService
        return None


def _freeze(monkeypatch, start: float = 1000.0) -> dict:
    """Заморозить часы модуля разрыва (детерминированные паузы)."""
    clock = {"t": start}
    monkeypatch.setattr("app.services.slot_breaker.time.time", lambda: clock["t"])
    return clock


def _worker_with_dedup_jobs(settings, judge) -> BackgroundWorker:
    """Две заметки с готовыми векторами → две pending dedup-работы в очередь."""
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
    assert worker.process_pending() == 2  # создаёт dedup-работы (очередь judge)
    assert worker._pending_jobs("judge", "dedup", 10)
    return worker


@pytest.mark.asyncio
async def test_gate_skips_batch_when_paused(settings, monkeypatch) -> None:
    """Разрыв открыт, стук не наступил → process_* = 0 и вызовов модели нет."""
    _freeze(monkeypatch)
    judge = ProbeJudge(result="ok")
    worker = _worker_with_dedup_jobs(settings, judge)
    worker._breakers["judge"].note_failure(SlotOutcome.AUTH)
    assert worker._breakers["judge"].is_paused() is True
    assert await worker._process_judge() == 0
    assert judge.judge_calls == []  # партия не бралась
    assert judge.probe_calls == 0  # стук ещё не наступил
    assert worker._pending_jobs("judge", "dedup", 10)  # работа осталась pending


@pytest.mark.asyncio
async def test_gate_probes_then_runs(settings, monkeypatch) -> None:
    """Стук наступил, probe() вернул ok → разрыв снят, партия идёт."""
    clock = _freeze(monkeypatch)
    judge = ProbeJudge(result="ok", verdicts=[False])
    worker = _worker_with_dedup_jobs(settings, judge)
    worker._breakers["judge"].note_failure(SlotOutcome.AUTH)
    clock["t"] += 901  # пауза истекла — пора стучать
    # заметка без ранних кандидатов снимается сразу, вторая — по вердикту судьи
    assert await worker._process_judge() == 2
    assert judge.probe_calls == 1
    assert len(judge.judge_calls) == 1
    assert worker._breakers["judge"].is_paused() is False


@pytest.mark.asyncio
async def test_probe_auth_rearms_pause(settings, monkeypatch) -> None:
    """probe() вернул auth_failed → пауза переставлена, партии нет."""
    clock = _freeze(monkeypatch)
    judge = ProbeJudge(result="auth_failed")
    worker = _worker_with_dedup_jobs(settings, judge)
    worker._breakers["judge"].note_failure(SlotOutcome.AUTH)
    clock["t"] += 901
    assert await worker._process_judge() == 0
    assert judge.probe_calls == 1
    assert judge.judge_calls == []
    assert worker._breakers["judge"].is_paused() is True
    assert worker._breakers["judge"].probe_due() is False


def test_failure_logged_with_outcome(settings, caplog) -> None:
    """Отказ судьи пишет событие с job, slot, outcome, detail; работа pending."""
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
        # Ранняя заметка (кандидатов нет) снимается сразу — 1; отказавшая
        # judge-работа остаётся pending (back-off очереди judge).
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
    assert event.detail  # фрагмент причины не теряется
    assert worker._pending_jobs("judge", "dedup", 10)  # задание живёт в очереди
