"""Причина дефекта lsbdef-0008 видна в журнале (релиз 3.2.0; arch §6.3/§7.2).

Юнит без сети: слот отдаёт пустой `content` при `done_reason=length` и поле
`thinking` (reasoning-модель съела бюджет вывода). Событие отказа называет
причину `outcome=truncated` и несёт диагностику последнего вызова
(`had_thinking`, `finish_reason`, бюджет); работа остаётся `pending` — повтор
по back-off, очередь не зацикливается на «пустом ответе». Ответ без маркера
вердикта — `off_contract` с фрагментом `detail` (≤160 симв.).
"""

from __future__ import annotations

import logging

import httpx
import pytest
from fakes import (
    FailingEmbedder,
    FixedSummarizer,
    HashEmbedder,
    RecordingDedup,
)

from app.config import get_settings
from app.services.judge import JudgeService
from app.services.notes import NoteService
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("JUDGE_PROVIDER", "ollama")
    get_settings.cache_clear()
    resolved = get_settings()
    init_db(resolved)
    return resolved


def records(caplog, event: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if getattr(r, "event", None) == event]


def judge_body(content: str, **over: object) -> dict:
    body = {
        "model": "m",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }
    body.update(over)
    return body


def judge_service(settings, body: dict) -> JudgeService:
    return JudgeService(
        settings,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)),
    )


def build_worker(settings, judge) -> BackgroundWorker:
    worker = BackgroundWorker(
        settings,
        HashEmbedder(8),
        FixedSummarizer("Ф."),
        RecordingDedup(candidates=[(1, 0.94)]),
        judge=judge,
    )
    worker._ensure_job_table()
    return worker


def pending_judge_jobs(settings) -> int:
    with session(settings) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM worker_jobs "
            "WHERE slot = 'judge' AND status = 'pending'"
        ).fetchone()
    return int(row["c"])


def test_reasoning_budget_eaten_is_truncated(settings, caplog) -> None:
    """Пустой content + length + thinking → truncated с диагностикой, работа pending."""
    notes = NoteService(settings, FailingEmbedder())
    notes.save("первая заметка обрезки бюджета")
    notes.save("вторая заметка обрезки бюджета")
    body = judge_body("", done_reason="length")
    body["message"]["thinking"] = "длинные рассуждения съели бюджет вывода"
    worker = build_worker(settings, judge_service(settings, body))
    assert worker.process_pending() == 2  # создаёт dedup-работы (очередь judge)
    with caplog.at_level(logging.INFO, logger="app"):
        worker.process_judge_pending()
    failures = records(caplog, "dedup_judge_failed")
    assert len(failures) == 1
    record = failures[0]
    assert record.outcome == "truncated"
    assert record.had_thinking is True
    assert record.finish_reason == "length"
    assert isinstance(record.budget, int) and record.budget > 0
    # Причина названа, очередь не зациклится: работа ждёт back-off, не «done».
    assert pending_judge_jobs(settings) == 1


def test_off_contract_detail_present(settings, caplog) -> None:
    """Ответ без маркера вердикта → off_contract с фрагментом detail (≤160)."""
    notes = NoteService(settings, FailingEmbedder())
    notes.save("первая заметка без вердикта")
    notes.save("вторая заметка без вердикта")
    worker = build_worker(settings, judge_service(settings, judge_body("нет вердикта тут")))
    assert worker.process_pending() == 2
    with caplog.at_level(logging.INFO, logger="app"):
        worker.process_judge_pending()
    failures = records(caplog, "dedup_judge_failed")
    assert len(failures) == 1
    record = failures[0]
    assert record.outcome == "off_contract"
    assert record.detail
    assert len(record.detail) <= 160
