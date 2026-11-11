"""The pair attempt count independently of the slot pause (#39, no39-04; arch §6.2/§6.8).

    The 3.2.0 acceptance defect (lsbdef-0008): an open slot pause stopped the
    batch, and a successful knock reset the pair's counter — the
    `SLOT_MAX_ATTEMPTS` threshold was unreachable and the "needs attention" state
    never came. This pins the separated boundary: **the pause is the SLOT's
    state, the count is the PAIR's property**. Non-transient failures count even
    with the breaker open; a successful knock lifts the breaker but not the
    count; in the breaker phase the work shows as `waiting`, not `attention`;
    transient failures still do not freeze the count.
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
    """Freeze the clock (the shared `time` module) — deterministic pauses."""
    clock = {"t": start}
    monkeypatch.setattr("app.services.slot_breaker.time.time", lambda: clock["t"])
    return clock


class ProbeJudge:
    """A fake judge with a controllable cheap `probe()` (a slot knock)."""

    def __init__(self, result: str = "ok") -> None:
        self._result = result
        self.probe_calls = 0
        self.last_attempt_ok: bool | None = None

    def probe(self) -> str:
        self.probe_calls += 1
        return self._result

    def close(self) -> None:
        return None


# --- the attention threshold's reachability --------------------------------


def test_counter_reaches_max_across_pause_cycles(
    tmp_path, monkeypatch, caplog
) -> None:
    """8 failures through "pause → knock → failure again" cycles → attention.

                `slot_pause_after=3`, `slot_max_attempts=8`: the slot breaker opens
                after the third failure of ONE outcome, but the PAIR's count does not
                notice it — after the pause is lifted by a knock the count continues
                from its former value (4, 5, … 8), not from zero. The eighth failure
                sets `attention_at` and a `job_attention` event with `attempts=8`.
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
        # a non-transient failure: it is a slot failure too
        worker._note_slot_outcome("judge", "judge", SlotOutcome.EMPTY)
        worker._note_attempt(job_id, SlotOutcome.EMPTY, detail="empty answer")
        if breaker.is_paused():
            # the batch froze before the knock: the pause expired, a probe ok lifts the breaker,
            # and the next failure counts ON TOP of the pair's former count
            clock["t"] += settings.slot_pause_sec + 1
            assert worker._slot_gate("judge", "judge") is True
            assert breaker.is_paused() is False

    row = _job_row(settings, job_id)
    assert row["attempts"] == 8
    assert row["attention_at"] is not None
    events = _events(caplog, "job_attention")
    assert events and events[-1].attempts == 8


def test_probe_success_does_not_reset_attempts(tmp_path, monkeypatch, caplog) -> None:
    """A knock lifts the slot breaker but does NOT reset the pair's count.

                "The slot revived" is not "the pair's reason is gone": after three
                failures and a successful knock `attempts` stays 3, `attention_at` is
                NULL, the `slot_resumed` event exists, and `job_attention_released`
                (and the attention release by the `probe` reason) does not.
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
    assert worker._slot_gate("judge", "judge") is False  # the knock is not due yet
    clock["t"] += settings.slot_pause_sec + 1
    assert worker._slot_gate("judge", "judge") is True  # the knock lifted the breaker
    assert judge.probe_calls == 1

    row = _job_row(settings, job_id)
    assert row["attempts"] == 3
    assert row["attention_at"] is None
    assert _events(caplog, "slot_resumed")
    assert not _events(caplog, "job_attention_released")
    # the hook: the knock neither released the work from attention nor tried to
    assert worker._attention_resets.get("probe", 0) == 0


def test_attention_set_while_gap_open(tmp_path, monkeypatch) -> None:
    """A pair failure at an OPEN slot breaker still sets attention.

                The pair keeps the count: its hopelessness does not depend on the slot
                being paused now. The breaker is not read here at all.
    """
    settings = _settings(tmp_path, monkeypatch, SLOT_MAX_ATTEMPTS="8")
    worker = _worker(settings)
    job_id = _seed_job(settings, slot="judge", kind="dedup", note_id=1)
    # the slot breaker is open (auth opens it at once) — the pair does not know about it
    worker._breakers["judge"].note_failure(SlotOutcome.AUTH)
    assert worker._breakers["judge"].is_paused() is True
    for _ in range(8):
        worker._note_attempt(job_id, SlotOutcome.EMPTY, detail="empty answer")
    row = _job_row(settings, job_id)
    assert row["attempts"] == 8
    assert row["attention_at"] is not None


def test_transient_does_not_freeze(tmp_path, monkeypatch) -> None:
    """unavailable/timeout/auth do not grow the pair's count and do not set attention_at."""
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


# --- observability: waiting ≠ attention -------------------------------------


def test_gap_shows_waiting_not_attention(tmp_path, monkeypatch) -> None:
    """With the breaker open the work shows in `waiting`, not in `attention`."""
    settings = _settings(tmp_path, monkeypatch)
    judge = ProbeJudge(result="ok")
    worker = _worker(settings, judge=judge)
    _seed_job(settings, slot="judge", kind="dedup", note_id=1)
    worker._breakers["judge"].note_failure(SlotOutcome.AUTH)
    assert worker._breakers["judge"].is_paused() is True

    judge_queue = worker.queues_health()["judge"]
    assert judge_queue["waiting"] >= 1
    assert judge_queue["attention"] == 0
    assert judge_queue["pending"] == 0  # the loop takes no batch

    assert worker.slots_health()["judge"]["paused_until"] is not None


# --- configuration: a soft WARN ----------------------------------------------


def test_config_warns_when_max_attempts_not_above_pause(
    tmp_path, monkeypatch, caplog
) -> None:
    """SLOT_MAX_ATTEMPTS ≤ SLOT_PAUSE_AFTER → a WARN with the action text.

                Not fatal: both values are the operator's will; the startup does not
                crash.
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
