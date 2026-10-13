"""Очереди при недоступных моделях: pending/waiting/attention (№39, 3.2.0).

Архитектура §6.9/§13.9: `pending` — только работа, которую петля может взять
СЕЙЧАС; `waiting` — отложено слотом/готовностью; `attention` — ждёт человека.
При открытом разрыве слота вся текущая работа уезжает в `waiting`
(`pending=0`), а очередь для петли становится пустой — иначе отложенное
задание гонит back-off-петлю в busy-loop (lsbdef-0008). Юнит без сети.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fakes import FailingEmbedder, FixedSummarizer, HashEmbedder

from app.config import get_settings
from app.services.notes import NoteService
from app.services.slot_outcome import SlotOutcome
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session, transaction

DIM = 8


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    resolved = get_settings()
    init_db(resolved)
    return resolved


def _ts(seconds_ago: int) -> str:
    moment = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def make_worker(settings, embedding=None) -> BackgroundWorker:
    return BackgroundWorker(
        settings,
        embedding if embedding is not None else HashEmbedder(DIM),
        FixedSummarizer("Фикс-суммари."),
    )


def _judge_job_ids(settings) -> list[int]:
    with session(settings) as conn:
        return [int(row["id"]) for row in conn.execute(
            "SELECT id FROM worker_jobs WHERE slot = 'judge' ORDER BY id"
        ).fetchall()]


def _freeze(monkeypatch, start: float = 1000.0) -> dict:
    clock = {"t": start}
    monkeypatch.setattr("app.services.slot_breaker.time.time", lambda: clock["t"])
    return clock


def test_judge_waiting_and_attention(settings) -> None:
    """pending-работа без внимания — в pending; с вниманием — в attention/waiting."""
    worker = make_worker(settings)
    worker._create_job("judge", "dedup", 1)
    worker._create_job("judge", "dedup", 2)
    attention_id, pending_id = _judge_job_ids(settings)
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE worker_jobs SET attention_at = ? WHERE id = ?",
            (_ts(0), attention_id),
        )

    stat = worker._judge_queue_stat()
    assert stat["pending"] == 1          # берётся только работа без внимания
    assert stat["attention"] == 1        # замороженная ждёт человека
    assert stat["waiting"] == 1          # у судьи отложено именно замороженное
    assert stat["oldest_pending_sec"] is not None
    assert pending_id != attention_id


def test_queue_empty_ignores_attention(settings) -> None:
    """Единственное attention-задание: очередь судьи для петли пуста (нет busy-loop)."""
    worker = make_worker(settings)
    worker._create_job("judge", "dedup", 1)
    (job_id,) = _judge_job_ids(settings)
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE worker_jobs SET attention_at = ? WHERE id = ?",
            (_ts(0), job_id),
        )

    assert worker._judge_queue_stat()["pending"] == 0
    assert worker._judge_queue_empty() is True


def test_paused_slot_moves_pending_to_waiting(settings, monkeypatch) -> None:
    """Разрыв открыт: pending слота уходит в waiting, петля очередь считает пустой."""
    _freeze(monkeypatch)
    worker = make_worker(settings)
    worker._create_job("judge", "dedup", 1)
    worker._create_job("judge", "dedup", 2)
    assert worker._judge_queue_stat()["pending"] == 2

    worker._breakers["judge"].note_failure(SlotOutcome.AUTH)  # разрыв открыт
    stat = worker._judge_queue_stat()
    assert stat["pending"] == 0
    assert stat["waiting"] == 2  # вся текущая работа — в ожидании слота
    assert worker._judge_queue_empty() is True  # петля не крутится вхолостую


def test_vector_queue_waits_on_embedding_break(settings, monkeypatch) -> None:
    """Разрыв embedding: векторная очередь ждёт слот (pending=0, waiting=1)."""
    _freeze(monkeypatch)
    NoteService(settings, FailingEmbedder()).save("заметка ждёт кодировщик")
    worker = make_worker(settings, FailingEmbedder())

    assert worker._vector_queue_stat()["pending"] == 1
    worker._breakers["embedding"].note_failure(SlotOutcome.AUTH)
    stat = worker._vector_queue_stat()
    assert stat["pending"] == 0
    assert stat["waiting"] == 1
