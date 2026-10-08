"""Счётчик попыток пары и «требует вмешательства» (№39, no39-02; arch §6.8/§13.3).

Юнит на хосте: счётчик растёт только на нетранзиентных исходах пары, при
исчерпании `SLOT_MAX_ATTEMPTS` задание замирает (`attention_at`), но остаётся
`pending` и не берётся петлёй (вызовы модели не жгутся); возвращается в работу
по каждому из протоколов сброса — правка пары, смена лучшего кандидата,
успешный стук слота, старт со сменой отпечатка конфигурации. Транзиентные
исходы счётчик не замораживают.
"""

from __future__ import annotations

import json
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

from app.config import Settings, get_settings
from app.services.judge import JudgeResult
from app.services.notes import NoteService
from app.services.slot_budget import slot_config_fingerprint
from app.services.slot_outcome import SlotOutcome
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


def _worker(settings, judge=None) -> BackgroundWorker:
    worker = BackgroundWorker(settings, FailingEmbedder(), judge=judge)
    worker._ensure_job_table()
    return worker


def _seed_job(
    settings,
    *,
    slot: str = "judge",
    kind: str = "dedup",
    note_id: int = 1,
    payload: str | None = None,
) -> int:
    _worker(settings)
    with session(settings) as conn, transaction(conn):
        cursor = conn.execute(
            "INSERT INTO worker_jobs (slot, kind, note_id, payload) "
            "VALUES (?, ?, ?, ?)",
            (slot, kind, note_id, payload),
        )
        return int(cursor.lastrowid or 0)


def _set_job(settings, job_id: int, **fields: object) -> None:
    assignments = ", ".join(f"{name} = ?" for name in fields)
    with session(settings) as conn, transaction(conn):
        conn.execute(
            f"UPDATE worker_jobs SET {assignments} WHERE id = ?",
            (*fields.values(), job_id),
        )


def _job_row(settings, job_id: int) -> dict:
    with session(settings) as conn:
        row = conn.execute(
            "SELECT attempts, last_outcome, last_error, attention_at, status "
            "FROM worker_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    return dict(row) if row is not None else {}


def _judge_job_id(settings, note_id: int) -> int:
    with session(settings) as conn:
        row = conn.execute(
            "SELECT id FROM worker_jobs WHERE slot = 'judge' AND kind = 'dedup' "
            "AND note_id = ?",
            (note_id,),
        ).fetchone()
    assert row is not None
    return int(row["id"])


def _only_job(settings, keep_id: int) -> None:
    """Оставить в очереди одну работу (прочие — done): чистая выборка."""
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE worker_jobs SET status = 'done' "
            "WHERE slot = 'judge' AND kind = 'dedup' AND id != ?",
            (keep_id,),
        )


def _worker_with_dedup_jobs(settings, judge, candidates, count: int = 3):
    notes = NoteService(settings, FailingEmbedder())
    for index in range(count):
        notes.save(f"заметка внимания номер {index}")
    worker = BackgroundWorker(
        settings,
        HashEmbedder(8),
        FixedSummarizer("Ф."),
        RecordingDedup(candidates=list(candidates)),
        judge=judge,
    )
    assert worker.process_pending() == count  # векторизация + dedup-работы
    return worker


def _attention_events(caplog, event: str) -> list:
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) == event
    ]


def _freeze(monkeypatch, start: float = 1000.0) -> dict:
    """Заморозить часы (общий модуль `time`) — детерминированные паузы."""
    clock = {"t": start}
    monkeypatch.setattr("app.services.slot_breaker.time.time", lambda: clock["t"])
    return clock


class ProbeJudge:
    """Судья-фейк с управляемым дешёвым `probe()`."""

    def __init__(self, result: str = "ok") -> None:
        self._result = result
        self.probe_calls = 0
        self.judge_calls: list = []
        self.last_attempt_ok: bool | None = None

    def probe(self) -> str:
        self.probe_calls += 1
        return self._result

    def judge(self, text_new: str, text_candidate: str) -> JudgeResult:
        self.judge_calls.append((text_new, text_candidate))
        self.last_attempt_ok = True
        return JudgeResult(verdict=False, meta=fake_meta("judge", "judge"))

    def close(self) -> None:
        return None


# --- счётчик попыток ---------------------------------------------------------


def test_attempts_grow_on_off_contract(tmp_path, monkeypatch) -> None:
    """Три нетранзиентных отказа → attempts=3, last_outcome='off_contract'."""
    settings = _settings(tmp_path, monkeypatch)
    worker = _worker(settings)
    job_id = _seed_job(settings, note_id=1)
    for _ in range(3):
        worker._note_attempt(job_id, SlotOutcome.OFF_CONTRACT, detail="bad marker")
    row = _job_row(settings, job_id)
    assert row["attempts"] == 3
    assert row["last_outcome"] == "off_contract"
    assert row["attention_at"] is None
    assert row["last_error"] == "bad marker"


def test_transient_does_not_freeze(tmp_path, monkeypatch) -> None:
    """unavailable/timeout/auth не растят счётчик и не ставят attention_at."""
    settings = _settings(tmp_path, monkeypatch, SLOT_MAX_ATTEMPTS="1")
    worker = _worker(settings)
    job_id = _seed_job(settings, note_id=1)
    for outcome in (
        SlotOutcome.UNAVAILABLE,
        SlotOutcome.TIMEOUT,
        SlotOutcome.AUTH,
    ):
        worker._note_attempt(job_id, outcome, detail="transient failure")
    row = _job_row(settings, job_id)
    assert row["attempts"] == 0
    assert row["attention_at"] is None
    assert row["last_outcome"] == "auth"


def test_attention_after_max_attempts(tmp_path, monkeypatch, caplog) -> None:
    """При slot_max_attempts=3 третий отказ ставит attention_at и job_attention."""
    settings = _settings(tmp_path, monkeypatch, SLOT_MAX_ATTEMPTS="3")
    caplog.set_level(logging.INFO, logger="app")
    worker = _worker(settings)
    job_id = _seed_job(settings, note_id=1)
    for _ in range(3):
        worker._note_attempt(job_id, SlotOutcome.EMPTY, detail="empty answer")
    row = _job_row(settings, job_id)
    assert row["attention_at"] is not None
    events = _attention_events(caplog, "job_attention")
    assert events and events[-1].attempts == 3
    assert events[-1].outcome == "empty"
    assert events[-1].job == "judge"


# --- задание не берётся, но живо --------------------------------------------


def test_attention_job_not_taken(tmp_path, monkeypatch) -> None:
    """Задание с attention_at не попадает в выборку и не зовёт судью."""
    settings = _settings(tmp_path, monkeypatch, SLOT_MAX_ATTEMPTS="1")
    judge = ScriptedJudge(default=False)
    worker = _worker_with_dedup_jobs(settings, judge, [(1, 0.94)], count=2)
    job_id = _judge_job_id(settings, 2)
    worker._note_attempt(job_id, SlotOutcome.OFF_CONTRACT, detail="off contract")
    assert _job_row(settings, job_id)["attention_at"] is not None
    _only_job(settings, job_id)
    assert job_id not in [int(row["id"]) for row in worker._pending_jobs("judge", "dedup", 10)]
    assert worker.process_judge_pending() == 0
    assert judge.judge_calls == []  # вызовов модели нет


def test_attention_stays_pending(tmp_path, monkeypatch) -> None:
    """Замороженное задание — не «мёртвое»: строка остаётся status='pending'."""
    settings = _settings(tmp_path, monkeypatch, SLOT_MAX_ATTEMPTS="1")
    judge = ScriptedJudge(default=False)
    worker = _worker_with_dedup_jobs(settings, judge, [(1, 0.94)], count=2)
    job_id = _judge_job_id(settings, 2)
    worker._note_attempt(job_id, SlotOutcome.TRUNCATED, detail="truncated")
    row = _job_row(settings, job_id)
    assert row["attention_at"] is not None
    assert row["status"] == "pending"


# --- протоколы сброса --------------------------------------------------------


def test_candidate_change_resets(tmp_path, monkeypatch) -> None:
    """Новый лучший кандидат → другая пара: attempts=0, attention_at=NULL."""
    settings = _settings(tmp_path, monkeypatch)
    judge = ScriptedJudge(default=False)
    worker = _worker_with_dedup_jobs(settings, judge, [(1, 0.94)], count=3)
    job_id = _judge_job_id(settings, 3)
    _set_job(
        settings,
        job_id,
        attempts=5,
        attention_at="2026-01-01T00:00:00Z",
        payload=json.dumps({"candidate_id": 1}),
    )
    worker._dedup.candidates = [(2, 0.9)]  # лучший кандидат сменился
    worker.process_judge_pending()
    row = _job_row(settings, job_id)
    assert row["attempts"] == 0
    assert row["attention_at"] is None


def test_note_edit_resets(tmp_path, monkeypatch, caplog) -> None:
    """Правка updated_at заметки → _release_attention(..., 'edit') + событие."""
    settings = _settings(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO, logger="app")
    judge = ScriptedJudge(default=False)
    worker = _worker_with_dedup_jobs(settings, judge, [(1, 0.94)], count=3)
    job_id = _judge_job_id(settings, 3)
    _set_job(
        settings,
        job_id,
        attempts=5,
        attention_at="2026-01-01T00:00:00Z",
        payload=json.dumps(
            {"candidate_id": 1, "note_updated_at": "2026-01-01T00:00:00Z"}
        ),
    )
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE notes SET updated_at = '2026-02-02T00:00:00Z' WHERE id = 3"
        )
    worker.process_judge_pending()
    row = _job_row(settings, job_id)
    assert row["attempts"] == 0
    assert row["attention_at"] is None
    events = _attention_events(caplog, "job_attention_released")
    assert any(getattr(record, "reason", None) == "edit" for record in events)


def test_probe_success_releases_slot(tmp_path, monkeypatch) -> None:
    """CHECK_OK на стуке снимает внимание у заданий слота."""
    settings = _settings(tmp_path, monkeypatch)
    job_id = _seed_job(settings, slot="judge", kind="dedup", note_id=1)
    _set_job(
        settings,
        job_id,
        attempts=8,
        last_outcome="off_contract",
        attention_at="2026-01-01T00:00:00Z",
    )
    clock = _freeze(monkeypatch)
    judge = ProbeJudge(result="ok")
    worker = _worker(settings, judge=judge)
    worker._breakers["judge"].note_failure(SlotOutcome.AUTH)
    assert worker._slot_gate("judge", "judge") is False  # стук ещё не наступил
    clock["t"] += settings.slot_pause_sec + 1
    assert worker._slot_gate("judge", "judge") is True
    assert judge.probe_calls == 1
    row = _job_row(settings, job_id)
    assert row["attention_at"] is None
    assert row["attempts"] == 0


def test_startup_resets_on_config_change(tmp_path, monkeypatch) -> None:
    """Смена отпечатка конфигурации → внимание снято, отпечаток перезаписан."""
    settings = _settings(tmp_path, monkeypatch)
    job_id = _seed_job(settings, note_id=1)
    _set_job(
        settings,
        job_id,
        attempts=8,
        last_outcome="off_contract",
        attention_at="2026-01-01T00:00:00Z",
    )
    worker = _worker(settings)
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (SLOT_CONFIG_META_KEY, "old-fingerprint"),
        )
    released = worker._startup_attention_reset()
    assert released == 1
    assert _job_row(settings, job_id)["attention_at"] is None
    with session(settings) as conn:
        stored = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (SLOT_CONFIG_META_KEY,)
        ).fetchone()
    assert stored["value"] == slot_config_fingerprint(settings)
