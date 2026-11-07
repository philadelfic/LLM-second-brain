"""The slot breaker — a pause with a probe knock (#39, release 3.2.0; arch §6.2).

    A unit without the network and the DB. `SlotBreaker` — the SLOT's state (not
    the job's): `AUTH` opens the breaker at once; any other non-transient outcome
    — after `open_after` identical ones in a row; `UNAVAILABLE`/`TIMEOUT` do not
    open it. The pause — `pause_sec` ahead of the failure; `probe_due` comes at
    its expiry; success lifts the breaker.
"""

from __future__ import annotations

from app.services.slot_breaker import SlotBreaker
from app.services.slot_outcome import SlotOutcome


def _freeze(monkeypatch, start: float = 1000.0) -> dict:
    """Freeze the breaker module's clock: the time tests are deterministic."""
    clock = {"t": start}
    monkeypatch.setattr("app.services.slot_breaker.time.time", lambda: clock["t"])
    return clock


def test_auth_opens_immediately() -> None:
    """The first access failure opens the breaker (open_after does not apply)."""
    breaker = SlotBreaker("judge", 900, open_after=3)
    assert breaker.note_failure(SlotOutcome.AUTH) is True
    assert breaker.is_paused() is True


def test_opens_after_threshold() -> None:
    """Three (open_after) identical non-transient failures in a row — the pause on the third."""
    breaker = SlotBreaker("summary", 900, open_after=3)
    assert breaker.note_failure(SlotOutcome.OFF_CONTRACT) is False
    assert breaker.is_paused() is False
    assert breaker.note_failure(SlotOutcome.OFF_CONTRACT) is False
    assert breaker.is_paused() is False
    assert breaker.note_failure(SlotOutcome.OFF_CONTRACT) is True
    assert breaker.is_paused() is True


def test_transient_does_not_pause() -> None:
    """Unavailability/timeout do not open the breaker — waiting is normal (back-off)."""
    breaker = SlotBreaker("embedding", 900, open_after=3)
    for _ in range(5):
        assert breaker.note_failure(SlotOutcome.UNAVAILABLE) is False
    for _ in range(5):
        assert breaker.note_failure(SlotOutcome.TIMEOUT) is False
    assert breaker.is_paused() is False


def test_streak_broken_by_other_outcome() -> None:
    """"Identical in a row": an outcome change resets the non-transient counter."""
    breaker = SlotBreaker("summary", 900, open_after=3)
    assert breaker.note_failure(SlotOutcome.OFF_CONTRACT) is False
    assert breaker.note_failure(SlotOutcome.EMPTY) is False
    assert breaker.note_failure(SlotOutcome.OFF_CONTRACT) is False
    assert breaker.note_failure(SlotOutcome.EMPTY) is False
    assert breaker.is_paused() is False  # no outcome reached three
    assert breaker.note_failure(SlotOutcome.EMPTY) is True  # the third EMPTY in a row


def test_probe_due_after_pause(monkeypatch) -> None:
    """Before pause_sec expires no knock is due; after — it is."""
    clock = _freeze(monkeypatch)
    breaker = SlotBreaker("judge", 900, open_after=3)
    breaker.note_failure(SlotOutcome.AUTH)
    assert breaker.probe_due() is False
    clock["t"] += 899
    assert breaker.probe_due() is False
    clock["t"] += 2
    assert breaker.probe_due() is True
    assert breaker.is_paused() is False  # the pause expired — time to knock


def test_success_closes(monkeypatch) -> None:
    """A successful knock lifts the breaker; last_outcome kept in the snapshot."""
    _freeze(monkeypatch)
    breaker = SlotBreaker("summary", 900, open_after=3)
    breaker.note_failure(SlotOutcome.AUTH)
    breaker.note_success()
    assert breaker.is_paused() is False
    assert breaker.probe_due() is False
    snapshot = breaker.snapshot()
    assert snapshot["state"] == "closed"
    assert snapshot["last_outcome"] == "auth"
    assert snapshot["paused_until"] is None
    assert snapshot["failures"] == 0


def test_failed_knock_rearms_pause(monkeypatch) -> None:
    """A knock failure reschedules the pause — the slot sleeps pause_sec again."""
    clock = _freeze(monkeypatch)
    breaker = SlotBreaker("judge", 900, open_after=3)
    breaker.note_failure(SlotOutcome.AUTH)
    clock["t"] += 901
    assert breaker.probe_due() is True
    assert breaker.note_failure(SlotOutcome.AUTH) is False  # a reschedule, not an "opening"
    assert breaker.is_paused() is True
    assert breaker.probe_due() is False


def test_snapshot_shape(monkeypatch) -> None:
    """The snapshot carries slot/state/last_outcome/paused_until/failures (for /health)."""
    _freeze(monkeypatch)
    breaker = SlotBreaker("embedding", 900, open_after=3)
    snapshot = breaker.snapshot()
    assert set(snapshot) == {
        "slot",
        "state",
        "last_outcome",
        "paused_until",
        "failures",
    }
    assert snapshot["slot"] == "embedding"
    assert snapshot["state"] == "closed"
    assert snapshot["last_outcome"] is None
    assert snapshot["paused_until"] is None
    breaker.note_failure(SlotOutcome.AUTH)
    paused = breaker.snapshot()
    assert paused["state"] == "paused"
    assert paused["last_outcome"] == "auth"
    assert paused["paused_until"].endswith("Z")
