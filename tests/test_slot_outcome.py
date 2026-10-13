"""Единый контракт исхода вызова слота (№39, релиз 3.2.0; arch §6.1).

Юнит без сети: `classify_llm_error` — единственная точка сопоставления
«ошибка клиента → исход», порядок проверок обязателен (AUTH → TIMEOUT →
EMPTY → TRUNCATED → UNAVAILABLE → OFF_CONTRACT). `outcome_of` читает исход
у доменной ошибки, у чужого исключения — OFF_CONTRACT. `SlotCallError` —
внутренний носитель исхода (домены/транспорт его не видят).
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
    """401/403 (LLMAuthError) — исход AUTH (ждать бессмысленно)."""
    exc = LLMAuthError("ключ отклонён", status=401)
    assert classify_llm_error(exc) is SlotOutcome.AUTH


def test_auth_flag_maps_to_auth() -> None:
    """401 у ollama-слота (признак `auth` без LLMAuthError) — исход AUTH.

    Приёмка 3.2.0, живой прогон 13.10.2026: 401 от провайдера слота summary
    шёл как `off_contract`, разрыв открывался лишь после трёх отказов и с
    чужой причиной — ветвь auth сценария 3 не могла стать зелёной.
    """
    exc = LLMError(
        "HTTP 401 от /api/chat: denied", status=401, transient=False, auth=True
    )
    assert classify_llm_error(exc) is SlotOutcome.AUTH


def test_timeout_wins_over_transient() -> None:
    """Таймаут помечен и transient, и timeout — исход TIMEOUT, не UNAVAILABLE."""
    exc = LLMError("read timeout", transient=True, timeout=True)
    assert classify_llm_error(exc) is SlotOutcome.TIMEOUT


def test_empty_not_unavailable() -> None:
    """Пустой content — EMPTY, а не «недоступность» (FR-3.3)."""
    exc = LLMError("пустой content", empty=True)
    assert classify_llm_error(exc) is SlotOutcome.EMPTY


def test_truncated_when_flag() -> None:
    """Обрезка по лимиту вывода (поле блока №49) — TRUNCATED."""
    exc = LLMError("обрезано", truncated=True)
    assert classify_llm_error(exc) is SlotOutcome.TRUNCATED


def test_truncated_reachable() -> None:
    """Регресс lsbdef-0008: признак truncated достижим и даёт SlotOutcome.TRUNCATED."""
    assert classify_llm_error(LLMError("обрезано", truncated=True)) is (
        SlotOutcome.TRUNCATED
    )


def test_transient_maps_to_unavailable() -> None:
    """Сеть/5xx/429 (transient) — UNAVAILABLE."""
    assert classify_llm_error(LLMError("HTTP 503", transient=True)) is (
        SlotOutcome.UNAVAILABLE
    )


def test_other_is_off_contract() -> None:
    """Прочий отказ клиента (HTTP 400, не-JSON) — OFF_CONTRACT."""
    assert classify_llm_error(LLMError("HTTP 400 от /api/chat")) is (
        SlotOutcome.OFF_CONTRACT
    )


def test_outcome_of_domain_error() -> None:
    """Доменная ошибка отдаёт свой исход; чужое исключение — OFF_CONTRACT."""
    assert outcome_of(JudgeError("нет вердикта", outcome=SlotOutcome.OFF_CONTRACT)) is (
        SlotOutcome.OFF_CONTRACT
    )
    assert outcome_of(RuntimeError("чужое")) is SlotOutcome.OFF_CONTRACT


def test_client_timeout_is_classified_as_timeout() -> None:
    """Транспортный таймаут клиента несёт timeout-флаг → TIMEOUT (граница §6.3)."""
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
    else:  # pragma: no cover — chat обязан бросить
        raise AssertionError("ожидался LLMError")


def test_slot_call_error_carries_fields() -> None:
    """SlotCallError — внутренний носитель исхода со слотом и деталью."""
    exc = SlotCallError(
        "отказ", slot="judge", outcome=SlotOutcome.AUTH, detail="ключ отклонён"
    )
    assert isinstance(exc, RuntimeError)
    assert exc.slot == "judge"
    assert exc.outcome is SlotOutcome.AUTH
    assert exc.detail == "ключ отклонён"
