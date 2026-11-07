"""The single slot call outcome contract (#39, release 3.2.0; arch §6.1).

    A unit without the network: `classify_llm_error` — the single
    "client error → outcome" mapping point; the check order is mandatory
    (AUTH → TIMEOUT → EMPTY → TRUNCATED → UNAVAILABLE → OFF_CONTRACT). `outcome_of`
    reads the outcome from a domain error; a foreign exception — OFF_CONTRACT.
    `SlotCallError` — the internal outcome carrier (domains/the transport never
    see it).
"""

from __future__ import annotations

import httpx

from app.config import get_settings
from app.services.judge import JudgeError
from app.services.llm_client import LLMAuthError, LLMClient, LLMError, SlotSpec
from app.services.slot_outcome import (
    SlotCallError,
    SlotOutcome,
    classify_llm_error,
    outcome_of,
)


def test_auth_maps_to_auth() -> None:
    """401/403 (LLMAuthError) — the AUTH outcome (waiting is pointless)."""
    exc = LLMAuthError("ключ отклонён", status=401)
    assert classify_llm_error(exc) is SlotOutcome.AUTH


def test_auth_flag_maps_to_auth() -> None:
    """401 at an ollama slot (the `auth` flag without LLMAuthError) — the AUTH outcome.

                The 3.2.0 acceptance, the live run of 2026-10-13: a 401 from the
                summary slot's provider went as `off_contract`, the breaker opened
                only after three failures and with a foreign cause — scenario 3's
                auth branch could not go green.
    """
    exc = LLMError(
        "HTTP 401 от /api/chat: denied", status=401, transient=False, auth=True
    )
    assert classify_llm_error(exc) is SlotOutcome.AUTH


def test_timeout_wins_over_transient() -> None:
    """A timeout marked both transient and timeout — the TIMEOUT outcome, not UNAVAILABLE."""
    exc = LLMError("read timeout", transient=True, timeout=True)
    assert classify_llm_error(exc) is SlotOutcome.TIMEOUT


def test_empty_not_unavailable() -> None:
    """Empty content — EMPTY, not "unavailability" (FR-3.3)."""
    exc = LLMError("пустой content", empty=True)
    assert classify_llm_error(exc) is SlotOutcome.EMPTY


def test_truncated_when_flag() -> None:
    """A trim by the output limit (the block-#49 field) — TRUNCATED."""
    exc = LLMError("обрезано", truncated=True)
    assert classify_llm_error(exc) is SlotOutcome.TRUNCATED


def test_truncated_reachable() -> None:
    """The lsbdef-0008 regression: the truncated flag is reachable and gives SlotOutcome.TRUNCATED."""
    assert classify_llm_error(LLMError("обрезано", truncated=True)) is (
        SlotOutcome.TRUNCATED
    )


def test_transient_maps_to_unavailable() -> None:
    """Network/5xx/429 (transient) — UNAVAILABLE."""
    assert classify_llm_error(LLMError("HTTP 503", transient=True)) is (
        SlotOutcome.UNAVAILABLE
    )


def test_other_is_off_contract() -> None:
    """Any other client failure (HTTP 400, non-JSON) — OFF_CONTRACT."""
    assert classify_llm_error(LLMError("HTTP 400 от /api/chat")) is (
        SlotOutcome.OFF_CONTRACT
    )


def test_outcome_of_domain_error() -> None:
    """A domain error serves its own outcome; a foreign exception — OFF_CONTRACT."""
    assert outcome_of(JudgeError("нет вердикта", outcome=SlotOutcome.OFF_CONTRACT)) is (
        SlotOutcome.OFF_CONTRACT
    )
    assert outcome_of(RuntimeError("чужое")) is SlotOutcome.OFF_CONTRACT


def test_client_timeout_is_classified_as_timeout() -> None:
    """A client transport timeout carries the timeout flag → TIMEOUT (the §6.3 boundary)."""
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise httpx.ReadTimeout("timed out", request=request)

    client = LLMClient(
        SlotSpec.for_summary(get_settings()), transport=httpx.MockTransport(handler)
    )
    try:
        client.chat("system", "user", num_predict=10)
    except LLMError as exc:
        assert exc.timeout is True
        assert exc.transient is True
        assert classify_llm_error(exc) is SlotOutcome.TIMEOUT
    else:  # pragma: no cover — chat must raise
        raise AssertionError("ожидался LLMError")


def test_slot_call_error_carries_fields() -> None:
    """SlotCallError — the internal outcome carrier with the slot and the detail."""
    exc = SlotCallError(
        "отказ", slot="judge", outcome=SlotOutcome.AUTH, detail="ключ отклонён"
    )
    assert isinstance(exc, RuntimeError)
    assert exc.slot == "judge"
    assert exc.outcome is SlotOutcome.AUTH
    assert exc.detail == "ключ отклонён"
