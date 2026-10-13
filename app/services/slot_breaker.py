"""Разрыв слота — пауза с пробным стуком (№39, релиз 3.2.0; arch §6.2, §13.2).

`SlotBreaker` — состояние одного СЛОТА (не джобы): слот `summary` делят
суммаризация, слияние, классификация, описания и обход `nodes`; слот `judge`
— dedup-судья и судья структуры. Разрыв живёт в одной точке на слот, иначе
«судья структуры не знает, что dedup-судья уже упал по auth». Состояние — в
памяти процесса (переживания рестарта не требует: на старте конфигурация
слота переоценивается стартовой проверкой).

Правила (arch §6.2, §13.2): отказ доступа (`AUTH`) открывает разрыв сразу;
иной нетранзиентный исход (`EMPTY`/`TRUNCATED`/`OFF_CONTRACT`) — после
`open_after` отказов ОДНОГО исхода; `UNAVAILABLE`/`TIMEOUT` разрыв не открывают
(ждать — нормальное состояние, им занимается back-off петли). Пауза —
`pause_sec` вперёд от отказа; после её истечения слот делает ОДИН дешёвый
пробный вызов (`probe()`, без генерации): успех снимает разрыв
(`note_success`), отказ переставляет паузу (`note_failure`).

Модуль ничего не логирует и не пишет в БД: журналит вызывающий (воркер) —
один формат события на все слоты (`slot_paused`/`slot_resumed`/`slot_probe`).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from app.services.slot_outcome import SlotOutcome

# Исходы, которые НЕ открывают разрыв: ждать их — нормальное состояние,
# которым занимается back-off петли (arch §13.2).
_TRANSIENT_OUTCOMES = frozenset({SlotOutcome.UNAVAILABLE, SlotOutcome.TIMEOUT})


class SlotBreaker:
    """Разрыв одного слота: пауза до потолка back-off + пробный стук.

    `open_after` — сколько отказов ОДНОГО нетранзиентного исхода открывают
    разрыв (`SLOT_PAUSE_AFTER`); для `AUTH` не применяется — разрыв сразу.
    """

    def __init__(self, slot: str, pause_sec: int, *, open_after: int = 3) -> None:
        self._slot = slot
        self._pause_sec = int(pause_sec)
        self._open_after = max(int(open_after), 1)
        self._last_outcome: SlotOutcome | None = None
        # Серия отказов по КАЖДОМУ нетранзиентному исходу (arch §6.2/§13.2:
        # «после open_after отказов одного исхода»). Чередование разных исходов
        # серию не сбрасывает, поэтому ни один исход не копит чужую серию.
        self._failures: dict[SlotOutcome, int] = {}
        self._paused_until: float | None = None

    def note_failure(self, outcome: SlotOutcome) -> bool:
        """Зафиксировать отказ; True — разрыв открылся ЭТИМ вызовом.

        Если разрыв уже открыт (пробный стук отказал) — пауза переставляется
        на `pause_sec`, но новым открытием это не считается (False): событие
        `slot_paused` пишется только на первом открытии.
        """
        now = time.time()
        self._last_outcome = outcome
        if self._paused_until is not None:
            self._paused_until = now + self._pause_sec
            return False
        if outcome is SlotOutcome.AUTH:
            # Отказ доступа: ждать бессмысленно — разрыв сразу (FR-4.3).
            self._failures[outcome] = self._failures.get(outcome, 0) + 1
            self._paused_until = now + self._pause_sec
            return True
        if outcome in _TRANSIENT_OUTCOMES:
            # Недоступность/таймаут: разрыв не открывают, серии сбрасывают.
            self._failures.clear()
            return False
        count = self._failures.get(outcome, 0) + 1
        self._failures[outcome] = count
        if count >= self._open_after:
            self._paused_until = now + self._pause_sec
            return True
        return False

    def note_success(self) -> None:
        """Снять разрыв (успешный пробный стук); `last_outcome` сохраняется."""
        self._paused_until = None
        self._failures.clear()

    def is_paused(self, now: float | None = None) -> bool:
        """Открыт ли разрыв прямо сейчас (пауза ещё не истекла)."""
        if self._paused_until is None:
            return False
        return (time.time() if now is None else now) < self._paused_until

    def probe_due(self, now: float | None = None) -> bool:
        """Наступил ли пробный стук (пауза истекла, разрыв ещё не снят)."""
        if self._paused_until is None:
            return False
        return (time.time() if now is None else now) >= self._paused_until

    def snapshot(self) -> dict:
        """Снимок состояния словарём — для наблюдаемости и журнала (arch §6.2).

        `state` — `paused` (пауза активна) или `closed`; `paused_until` —
        ISO-8601 UTC или None (закрыто/истекло); `failures` — счётчик серии
        последнего исхода.
        """
        paused = self.is_paused()
        return {
            "slot": self._slot,
            "state": "paused" if paused else "closed",
            "last_outcome": (
                self._last_outcome.value
                if isinstance(self._last_outcome, SlotOutcome)
                else self._last_outcome
            ),
            "paused_until": (
                self._iso_utc(self._paused_until) if paused else None
            ),
            "failures": self._failures.get(self._last_outcome, 0),
        }

    @staticmethod
    def _iso_utc(timestamp: float | None) -> str | None:
        """Unix-время → ISO-8601 UTC с суффиксом Z (формат остальных меток)."""
        if timestamp is None:
            return None
        return datetime.fromtimestamp(timestamp, timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
