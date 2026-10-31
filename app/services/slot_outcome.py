"""The single outcome of a slot call (No39, 3.2.0; arch SS6.1, SS13.1).

    A slot failure stops being anonymous: a call now has an outcome type
    `SlotOutcome` - shared by the slot block No49 / lsbdef-0008 / LSBN-F-12 (a
    common dictionary). The module neither logs nor writes to the DB - a pure
    "client error -> outcome" mapping; the caller (the worker) journals, giving
    one event format across all slots.

    `classify_llm_error` is the ONLY place mapping a client error to an outcome;
    the check order is mandatory (`LLMAuthError`/`exc.auth` -> `exc.timeout` ->
    `exc.empty` -> `exc.truncated` -> `exc.transient` -> `OFF_CONTRACT`) - the
    `transient=True` of a timeout would otherwise eat `TIMEOUT`. The outcome is
    NEVER derived from the message text - the client sets explicit flags
    (`app/services/llm_client.py`).

    The module imports neither the domain nor the transport: `LLMError`/
    `LLMAuthError` is the slot-client boundary; nobody above sees the outcome
    beforehand (`SlotCallError` is an internal type).
"""

from __future__ import annotations

from enum import Enum

from app.services.llm_client import LLMAuthError, LLMError


class SlotOutcome(str, Enum):
    """The slot call outcome dictionary: exactly six values (arch SS6.1).

            * `UNAVAILABLE` - network/5xx/429: the slot does not answer (wait, back-off);
            * `AUTH` - 401/403: waiting is pointless (a pause + a probe knock);
            * `EMPTY` - there is a response but the content is empty (NOT unavailability);
            * `TRUNCATED` - cut by the output limit (finish_reason=length, block No49);
            * `TIMEOUT` - did not answer in time (not "empty", not "unavailable");
            * `OFF_CONTRACT` - there is a response but the marker/structure is missing.

    """

    UNAVAILABLE = "unavailable"
    AUTH = "auth"
    EMPTY = "empty"
    TRUNCATED = "truncated"
    TIMEOUT = "timeout"
    OFF_CONTRACT = "off_contract"


def classify_llm_error(exc: LLMError) -> SlotOutcome:
    """Map an LLM client error to a slot call outcome (arch SS6.1, SS13.1).

            The check order is strictly fixed: AUTH -> TIMEOUT -> EMPTY -> TRUNCATED
            -> UNAVAILABLE -> OFF_CONTRACT. The `timeout`/`empty`/`truncated`/`auth`
            flags are set by the client itself (`LLMError`); `truncated` was
            unreachable before block No49, but the classifier already reads it via
            `getattr`.

            `auth` (arch SS6.1: AUTH = "401/403 - waiting is pointless") arrives by
            TWO paths: `LLMAuthError` - 401/403 at an openai slot (the key unset or
            rejected); the `auth` flag - 401/403 at an ollama slot where Bearer never
            goes (the 3.2.0 acceptance, the live run: a summary-provider 401 used to
            classify as `off_contract`, the break opened after three failures with a
            foreign cause).

    """
    if isinstance(exc, LLMAuthError) or getattr(exc, "auth", False):
        return SlotOutcome.AUTH
    if getattr(exc, "timeout", False):
        return SlotOutcome.TIMEOUT
    if getattr(exc, "empty", False):
        return SlotOutcome.EMPTY
    if getattr(exc, "truncated", False):
        return SlotOutcome.TRUNCATED
    if getattr(exc, "transient", False):
        return SlotOutcome.UNAVAILABLE
    return SlotOutcome.OFF_CONTRACT


def outcome_of(exc: BaseException) -> SlotOutcome:
    """Read the outcome from a domain error (`exc.outcome`), else OFF_CONTRACT.

            The slot domain errors (`SummaryError`/`JudgeError`/`ClassificationError`/
            `EmbeddingError`) carry the `outcome` field; a foreign exception without
            it - a safe `OFF_CONTRACT` (an unknown cause is never presented as a known
            one).

    """
    outcome = getattr(exc, "outcome", None)
    if isinstance(outcome, SlotOutcome):
        return outcome
    try:
        return SlotOutcome(outcome)
    except (TypeError, ValueError):
        return SlotOutcome.OFF_CONTRACT


class SlotCallError(RuntimeError):
    """A slot failure with an outcome (internal; the domains and transport never see it).

            `slot` - the slot name (`embedding`/`summary`/`judge`), `outcome` - the
            outcome kind, `detail` - a cause fragment without secrets (for the
            journal).

    """

    def __init__(
        self,
        message: str,
        *,
        slot: str,
        outcome: SlotOutcome,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.slot = slot
        self.outcome = outcome
        self.detail = detail
