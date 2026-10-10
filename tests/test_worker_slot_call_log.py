"""Журнал вызовов слота в воркере (№49, релиз 3.2.0; arch §7.2).

Юнит без сети: успешный вызов пишет `slot_call` с бюджетом/источником/
токенами/длительностью; отказ слота, у которого вызов состоялся, — ОДНО
событие блока 2 (№39), расширенное полями вызова.
"""

from __future__ import annotations

import logging

import pytest
from fakes import (
    FailingEmbedder,
    FixedSummarizer,
    HashEmbedder,
    RecordingDedup,
    fake_meta,
)

from app.config import get_settings
from app.services.judge import JudgeError
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


def records(caplog, event: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if getattr(r, "event", None) == event]


def test_slot_call_on_success(settings, caplog) -> None:
    notes = NoteService(settings, FailingEmbedder())
    notes.save("заметка для журнала вызова слота")
    worker = BackgroundWorker(settings, HashEmbedder(8), FixedSummarizer("Суммари."))
    with caplog.at_level(logging.INFO, logger="app"):
        assert worker.process_summary_pending() == 1
    calls = records(caplog, "slot_call")
    assert len(calls) == 1
    record = calls[0]
    assert record.job == "summary"
    assert record.slot == "summary"
    assert record.operation == "summary"
    assert record.budget == 1000
    assert record.budget_source == "legacy"
    assert record.completion_tokens == 5
    assert record.duration_ms == 1


class MetaFailingJudge:
    """Судья-фейк: вызов состоялся, вердикт не распознан — отказ с метаданными."""

    def __init__(self) -> None:
        self.last_attempt_ok: bool | None = None

    def probe(self) -> str:
        return "ok"

    def judge(self, text_new: str, text_candidate: str):
        self.last_attempt_ok = False
        raise JudgeError(
            "вердикт не распознан",
            outcome=SlotOutcome.OFF_CONTRACT,
            meta=fake_meta("judge", "judge"),
        )

    def close(self) -> None:
        return None


def test_failure_event_carries_call_meta(settings, caplog) -> None:
    notes = NoteService(settings, FailingEmbedder())
    notes.save("первая заметка отказа судьи")
    notes.save("вторая заметка отказа судьи")
    worker = BackgroundWorker(
        settings,
        HashEmbedder(8),
        FixedSummarizer("Ф."),
        RecordingDedup(candidates=[(1, 0.94)]),
        judge=MetaFailingJudge(),
    )
    assert worker.process_pending() == 2  # создаёт dedup-работы (очередь judge)
    with caplog.at_level(logging.INFO, logger="app"):
        worker.process_judge_pending()
    failures = records(caplog, "dedup_judge_failed")
    assert len(failures) == 1  # второй записи нет
    record = failures[0]
    assert record.outcome == "off_contract"
    assert record.budget == 1000
    assert record.budget_source == "legacy"
    assert record.operation == "judge"
    assert records(caplog, "slot_call") == []
