"""Разрыв слота — пауза с пробным стуком (№39, релиз 3.2.0; arch §6.2).

Юнит без сети и БД. `SlotBreaker` — состояние СЛОТА (не джобы): `AUTH`
открывает разрыв сразу, иной нетранзиентный исход — после `open_after`
подряд одинаковых, `UNAVAILABLE`/`TIMEOUT` разрыв не открывают. Пауза —
`pause_sec` вперёд от отказа; `probe_due` наступает по её истечении, успех
снимает разрыв.
"""

from __future__ import annotations

from app.services.slot_breaker import SlotBreaker
from app.services.slot_outcome import SlotOutcome


def _freeze(monkeypatch, start: float = 1000.0) -> dict:
    """Заморозить часы модуля разрыва: тесты времени детерминированы."""
    clock = {"t": start}
    monkeypatch.setattr("app.services.slot_breaker.time.time", lambda: clock["t"])
    return clock


def test_auth_opens_immediately() -> None:
    """Первый же отказ доступа открывает разрыв (open_after не применяется)."""
    breaker = SlotBreaker("judge", 900, open_after=3)
    assert breaker.note_failure(SlotOutcome.AUTH) is True
    assert breaker.is_paused() is True


def test_opens_after_threshold() -> None:
    """Три (open_after) подряд одинаковых нетранзиентных отказа — пауза на третьем."""
    breaker = SlotBreaker("summary", 900, open_after=3)
    assert breaker.note_failure(SlotOutcome.OFF_CONTRACT) is False
    assert breaker.is_paused() is False
    assert breaker.note_failure(SlotOutcome.OFF_CONTRACT) is False
    assert breaker.is_paused() is False
    assert breaker.note_failure(SlotOutcome.OFF_CONTRACT) is True
    assert breaker.is_paused() is True


def test_transient_does_not_pause() -> None:
    """Недоступность/таймаут разрыв не открывают — ждать нормально (back-off)."""
    breaker = SlotBreaker("embedding", 900, open_after=3)
    for _ in range(5):
        assert breaker.note_failure(SlotOutcome.UNAVAILABLE) is False
    for _ in range(5):
        assert breaker.note_failure(SlotOutcome.TIMEOUT) is False
    assert breaker.is_paused() is False


def test_streak_broken_by_other_outcome() -> None:
    """«Подряд одинаковых»: смена исхода сбрасывает счётчик нетранзиентных."""
    breaker = SlotBreaker("summary", 900, open_after=3)
    assert breaker.note_failure(SlotOutcome.OFF_CONTRACT) is False
    assert breaker.note_failure(SlotOutcome.EMPTY) is False
    assert breaker.note_failure(SlotOutcome.OFF_CONTRACT) is False
    assert breaker.note_failure(SlotOutcome.EMPTY) is False
    assert breaker.is_paused() is False  # ни одного исхода не набралось трижды
    assert breaker.note_failure(SlotOutcome.EMPTY) is True  # третий EMPTY подряд


def test_probe_due_after_pause(monkeypatch) -> None:
    """До истечения pause_sec стук не наступает, после — наступает."""
    clock = _freeze(monkeypatch)
    breaker = SlotBreaker("judge", 900, open_after=3)
    breaker.note_failure(SlotOutcome.AUTH)
    assert breaker.probe_due() is False
    clock["t"] += 899
    assert breaker.probe_due() is False
    clock["t"] += 2
    assert breaker.probe_due() is True
    assert breaker.is_paused() is False  # пауза истекла — пора стучать


def test_success_closes(monkeypatch) -> None:
    """Успешный стук снимает разрыв; last_outcome сохранён в снимке."""
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
    """Отказ стука переставляет паузу — слот снова спит на pause_sec."""
    clock = _freeze(monkeypatch)
    breaker = SlotBreaker("judge", 900, open_after=3)
    breaker.note_failure(SlotOutcome.AUTH)
    clock["t"] += 901
    assert breaker.probe_due() is True
    assert breaker.note_failure(SlotOutcome.AUTH) is False  # переставление, не «открытие»
    assert breaker.is_paused() is True
    assert breaker.probe_due() is False


def test_snapshot_shape(monkeypatch) -> None:
    """Снимок несёт slot/state/last_outcome/paused_until/failures (для /health)."""
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
