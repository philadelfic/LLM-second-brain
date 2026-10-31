"""The slot break - a pause with a probe knock (No39, 3.2.0; arch SS6.2, SS13.2).

    `SlotBreaker` is the state of one SLOT (not a job): the `summary` slot is
    shared by summarization, merging, classification, descriptions and the `nodes`
    sweep; the `judge` slot - by the dedup judge and the structure judge. The
    break lives in one place per slot - otherwise "the structure judge does not
    know the dedup judge already died by auth". The state is in process memory (no
    restart persistence needed: at startup the slot configuration is re-evaluated
    by the startup check).

    The rules (arch SS6.2, SS13.2): an access failure (`AUTH`) opens the break
    immediately; another non-transient outcome (`EMPTY`/`TRUNCATED`/
    `OFF_CONTRACT`) - after `open_after` failures of ONE outcome;
    `UNAVAILABLE`/`TIMEOUT` never open a break (waiting is a normal state, handled
    by the loop's back-off). The pause is `pause_sec` ahead of the failure; when
    it expires the slot makes ONE cheap probe call (`probe()`, no generation):
    success clears the break (`note_success`), a failure re-arms the pause
    (`note_failure`).

    The module neither logs nor writes to the DB: the caller (the worker)
    journals - one event format across all slots
    (`slot_paused`/`slot_resumed`/`slot_probe`).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from app.services.slot_outcome import SlotOutcome

# The outcomes that do NOT open a break: waiting for them is a normal state,
# handled by the loop's back-off (arch §13.2).
_TRANSIENT_OUTCOMES = frozenset({SlotOutcome.UNAVAILABLE, SlotOutcome.TIMEOUT})


class SlotBreaker:
    """One slot's break: a pause up to the back-off ceiling + a probe knock.

            `open_after` - how many failures of ONE non-transient outcome open the
            break (`SLOT_PAUSE_AFTER`); not applied to `AUTH` - the break opens
            immediately.

    """

    def __init__(self, slot: str, pause_sec: int, *, open_after: int = 3) -> None:
        self._slot = slot
        self._pause_sec = int(pause_sec)
        self._open_after = max(int(open_after), 1)
        self._last_outcome: SlotOutcome | None = None
        # A failure series per EACH non-transient outcome (arch §6.2/§13.2:
        #"after open_after failures of one outcome"). Alternating different outcomes
        #does not reset the series, so no outcome accumulates a foreign one.
        self._failures: dict[SlotOutcome, int] = {}
        self._paused_until: float | None = None

    def note_failure(self, outcome: SlotOutcome) -> bool:
        """Record a failure; True - THIS call opened the break.

                    When the break is already open (a probe knock failed) - the pause is
                    re-armed by `pause_sec`, but it does not count as a new opening
                    (False): the `slot_paused` event is written on the first opening only.

        """
        now = time.time()
        self._last_outcome = outcome
        if self._paused_until is not None:
            self._paused_until = now + self._pause_sec
            return False
        if outcome is SlotOutcome.AUTH:
            # An access failure: waiting is pointless - the break opens immediately (FR-4.3).
            self._failures[outcome] = self._failures.get(outcome, 0) + 1
            self._paused_until = now + self._pause_sec
            return True
        if outcome in _TRANSIENT_OUTCOMES:
            # Unavailability/timeout: they never open a break and reset the series.
            self._failures.clear()
            return False
        count = self._failures.get(outcome, 0) + 1
        self._failures[outcome] = count
        if count >= self._open_after:
            self._paused_until = now + self._pause_sec
            return True
        return False

    def note_success(self) -> None:
        """Clear the break (a successful probe knock); `last_outcome` is kept."""
        self._paused_until = None
        self._failures.clear()

    def is_paused(self, now: float | None = None) -> bool:
        """Whether the break is open right now (the pause has not expired)."""
        if self._paused_until is None:
            return False
        return (time.time() if now is None else now) < self._paused_until

    def probe_due(self, now: float | None = None) -> bool:
        """Whether the probe knock is due (the pause expired, the break is still on)."""
        if self._paused_until is None:
            return False
        return (time.time() if now is None else now) >= self._paused_until

    def snapshot(self) -> dict:
        """The state snapshot as a dict - for observability and the journal (arch SS6.2).

                    `state` - `paused` (the pause is active) or `closed`; `paused_until` -
                    ISO-8601 UTC or None (closed/expired); `failures` - the counter of the
                    last outcome's series.

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
        """Unix time -> ISO-8601 UTC with a Z suffix (the format of the other stamps)."""
        if timestamp is None:
            return None
        return datetime.fromtimestamp(timestamp, timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
