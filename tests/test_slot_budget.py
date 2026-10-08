"""Политика бюджета вывода слота (№49, релиз 3.2.0; arch §6.1, §13.2).

Чистые функции без сети и БД: явный бюджет оператора, «сколько влезет» из
заявленного контекста и деградация к прежним потолкам при неизвестном.
"""

from __future__ import annotations

from app.services.slot_budget import LEGACY_OUTPUT_BUDGET, resolve_output_budget

CAP = 32768


def resolve(**kwargs) -> tuple[int, str]:
    params = {
        "operation": "summary",
        "prompt_tokens": 0,
        "explicit": None,
        "context_tokens": 0,
        "reserve_tokens": 512,
        "defensive_cap": CAP,
    }
    params.update(kwargs)
    return resolve_output_budget(**params)


def test_explicit_wins() -> None:
    """Явный бюджет оператора не зажимается даже при заданном контексте."""
    assert resolve(explicit=1024, context_tokens=8192, prompt_tokens=1000) == (
        1024,
        "explicit",
    )


def test_context_minus_prompt() -> None:
    """«Сколько влезет» = контекст − промпт − запас."""
    assert resolve(context_tokens=8192, prompt_tokens=1000) == (6680, "context")


def test_capped_by_defensive_cap() -> None:
    """Защитный потолок применяется к авто-бюджету."""
    assert resolve(context_tokens=100000, prompt_tokens=10) == (CAP, "context")


def test_unknown_context_legacy() -> None:
    """Неизвестный контекст — прежние потолки операций (деградация)."""
    assert resolve(operation="judge", context_tokens=0) == (256, "legacy")
    assert resolve(operation="summary", context_tokens=0) == (35000, "legacy")
    assert LEGACY_OUTPUT_BUDGET["classify"] == 768


def test_never_below_one() -> None:
    """Промпт + запас длиннее контекста — бюджет не отрицательный, а 1."""
    assert resolve(context_tokens=100, prompt_tokens=500) == (1, "context")
    assert resolve(context_tokens=100, prompt_tokens=100, reserve_tokens=512) == (
        1,
        "context",
    )
