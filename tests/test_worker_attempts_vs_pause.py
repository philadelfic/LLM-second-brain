"""Счёт попыток пары независимо от паузы слота (№39, no39-04; arch §6.2/§6.8).

Дефект приёмки 3.2.0 (lsbdef-0008, решение О. 2026-10-12): открытая пауза слота
останавливала партию, а успешный стук обнулял счёт пары — порог
`SLOT_MAX_ATTEMPTS` был недостижим, и состояние «требует вмешательства» не
наступало никогда. Здесь закреплена разведённая граница: **пауза — состояние
СЛОТА, счёт — свойство ПАРЫ**. Нетранзиентные отказы считаются и при открытом
разрыве; успешный стук снимает разрыв, но не счёт; в фазе разрыва работа видна
как `waiting`, а не `attention`; транзиентные отказы счёт по-прежнему не
замораживают.
"""

from __future__ import annotations

import logging

import pytest
from fakes import FailingEmbedder

from app.config import Settings, get_settings
from app.services.slot_outcome import SlotOutcome
from app.services.worker import BackgroundWorker
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
) -> int:
    _worker(settings)
    with session(settings) as conn, transaction(conn):
        cursor = conn.execute(
            "INSERT INTO worker_jobs (slot, kind, note_id, payload) "
            "VALUES (?, ?, ?, ?)",
            (slot, kind, note_id, None),
        )
        return int(cursor.lastrowid or 0)


def _job_row(settings, job_id: int) -> dict:
    with session(settings) as conn:
        row = conn.execute(
            "SELECT attempts, last_outcome, last_error, attention_at, status "
            "FROM worker_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    return dict(row) if row is not None else {}


def _events(caplog, event: str) -> list:
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
    """Судья-фейк с управляемым дешёвым `probe()` (стук слота)."""

    def __init__(self, result: str = "ok") -> None:
        self._result = result
        self.probe_calls = 0
        self.last_attempt_ok: bool | None = None

    def probe(self) -> str:
        self.probe_calls += 1
        return self._result

    def close(self) -> None:
        return None


# --- достижимость порога внимания -------------------------------------------


def test_counter_reaches_max_across_pause_cycles(
    tmp_path, monkeypatch, caplog
) -> None:
    """8 отказов через циклы «пауза → стук → снова отказ» → attention.

    `slot_pause_after=3`, `slot_max_attempts=8`: разрыв слота открывается
    после третьего отказа ОДНОГО исхода, но счёт ПАРЫ его не замечает — после
    снятия паузы стуком отсчёт продолжается с прежнего значения (4, 5, … 8), а
    не с нуля. Восьмой отказ ставит `attention_at` и событие `job_attention`
    с `attempts=8`.
    """
    settings = _settings(
        tmp_path,
        monkeypatch,
        SLOT_PAUSE_AFTER="3",
        SLOT_MAX_ATTEMPTS="8",
    )
    caplog.set_level(logging.INFO, logger="app")
    clock = _freeze(monkeypatch)
    judge = ProbeJudge(result="ok")
    worker = _worker(settings, judge=judge)
    job_id = _seed_job(settings, slot="judge", kind="dedup", note_id=1)
    breaker = worker._breakers["judge"]

    for _ in range(8):
        # нетранзиентный отказ: он же отказ слота
        worker._note_slot_outcome("judge", "judge", SlotOutcome.EMPTY)
        worker._note_attempt(job_id, SlotOutcome.EMPTY, detail="empty answer")
        if breaker.is_paused():
            # партия замерла до стука: пауза истекла, probe ok снимает разрыв,
            # и следующий отказ копится ПОВЕРХ прежнего счёта пары
            clock["t"] += settings.slot_pause_sec + 1
            assert worker._slot_gate("judge", "judge") is True
            assert breaker.is_paused() is False

    row = _job_row(settings, job_id)
    assert row["attempts"] == 8
    assert row["attention_at"] is not None
    events = _events(caplog, "job_attention")
    assert events and events[-1].attempts == 8


def test_probe_success_does_not_reset_attempts(tmp_path, monkeypatch, caplog) -> None:
    """Стук снимает разрыв слота, но НЕ обнуляет счёт пары.

    «Слот ожил» — не «причина пары ушла»: после трёх отказов и успешного стука
    `attempts` остаётся 3, `attention_at` — NULL, событие `slot_resumed` есть,
    а `job_attention_released` (и снятие внимания по причине `probe`) — нет.
    """
    settings = _settings(tmp_path, monkeypatch, SLOT_PAUSE_AFTER="3")
    caplog.set_level(logging.INFO, logger="app")
    clock = _freeze(monkeypatch)
    judge = ProbeJudge(result="ok")
    worker = _worker(settings, judge=judge)
    job_id = _seed_job(settings, slot="judge", kind="dedup", note_id=1)
    for _ in range(3):
        worker._note_attempt(job_id, SlotOutcome.OFF_CONTRACT, detail="off contract")
    assert _job_row(settings, job_id)["attempts"] == 3

    worker._breakers["judge"].note_failure(SlotOutcome.AUTH)
    assert worker._slot_gate("judge", "judge") is False  # стук ещё не наступил
    clock["t"] += settings.slot_pause_sec + 1
    assert worker._slot_gate("judge", "judge") is True  # стук снял разрыв
    assert judge.probe_calls == 1

    row = _job_row(settings, job_id)
    assert row["attempts"] == 3
    assert row["attention_at"] is None
    assert _events(caplog, "slot_resumed")
    assert not _events(caplog, "job_attention_released")
    # крючок: стук не снимал работу с внимания (и не пытался)
    assert worker._attention_resets.get("probe", 0) == 0


def test_attention_set_while_gap_open(tmp_path, monkeypatch) -> None:
    """Отказ пары при ОТКРЫТОМ разрыве слота всё равно ставит attention.

    Счёт ведёт пара: её безнадёжность не зависит от того, что слот сейчас в
    паузе. Разрыв здесь не читается вовсе.
    """
    settings = _settings(tmp_path, monkeypatch, SLOT_MAX_ATTEMPTS="8")
    worker = _worker(settings)
    job_id = _seed_job(settings, slot="judge", kind="dedup", note_id=1)
    # разрыв слота открыт (auth открывает сразу) — пара о нём не знает
    worker._breakers["judge"].note_failure(SlotOutcome.AUTH)
    assert worker._breakers["judge"].is_paused() is True
    for _ in range(8):
        worker._note_attempt(job_id, SlotOutcome.EMPTY, detail="empty answer")
    row = _job_row(settings, job_id)
    assert row["attempts"] == 8
    assert row["attention_at"] is not None


def test_transient_does_not_freeze(tmp_path, monkeypatch) -> None:
    """unavailable/timeout/auth не растят счёт пары и не ставят attention_at."""
    settings = _settings(tmp_path, monkeypatch, SLOT_MAX_ATTEMPTS="1")
    worker = _worker(settings)
    job_id = _seed_job(settings, slot="judge", kind="dedup", note_id=1)
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


# --- наблюдаемость: waiting ≠ attention --------------------------------------


def test_gap_shows_waiting_not_attention(tmp_path, monkeypatch) -> None:
    """При открытом разрыве работа видна в `waiting`, а не в `attention`."""
    settings = _settings(tmp_path, monkeypatch)
    judge = ProbeJudge(result="ok")
    worker = _worker(settings, judge=judge)
    _seed_job(settings, slot="judge", kind="dedup", note_id=1)
    worker._breakers["judge"].note_failure(SlotOutcome.AUTH)
    assert worker._breakers["judge"].is_paused() is True

    judge_queue = worker.queues_health()["judge"]
    assert judge_queue["waiting"] >= 1
    assert judge_queue["attention"] == 0
    assert judge_queue["pending"] == 0  # петля партию не берёт

    assert worker.slots_health()["judge"]["paused_until"] is not None


# --- конфигурация: софт-WARN -------------------------------------------------


def test_config_warns_when_max_attempts_not_above_pause(
    tmp_path, monkeypatch, caplog
) -> None:
    """SLOT_MAX_ATTEMPTS ≤ SLOT_PAUSE_AFTER → WARN с текстом действия.

    Фатальным не делаем: оба значения — воля оператора, старт не падает.
    """
    monkeypatch.setenv("SLOT_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("SLOT_PAUSE_AFTER", "3")
    with caplog.at_level(logging.WARNING, logger="app"):
        settings = _settings(tmp_path, monkeypatch)

    assert settings.slot_max_attempts == 3
    assert settings.slot_pause_after == 3
    warnings = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "slot_attempts_below_pause"
    ]
    assert warnings
    text = warnings[-1].getMessage()
    assert "SLOT_MAX_ATTEMPTS" in text
    assert "SLOT_PAUSE_AFTER" in text
