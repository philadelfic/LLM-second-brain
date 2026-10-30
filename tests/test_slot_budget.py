"""Политика бюджета вывода слота (№49, релиз 3.2.0; arch §6.1, §13.2).

Чистые функции без сети и БД: явный бюджет оператора, «сколько влезет» из
заявленного контекста и деградация к прежним потолкам при неизвестном.
LSBN-T-33 (релиз 3.5.0): раздельные бюджеты ответа и рассуждения
(`resolve_budgets`) — рассуждение занимает остаток окна контекста и не
отнимает бюджет у ответа.
"""

from __future__ import annotations

from app.services.slot_budget import (
    LEGACY_OUTPUT_BUDGET,
    resolve_budgets,
    resolve_output_budget,
)

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


def split(**kwargs):
    """Вызов resolve_budgets с дефолтами (контекст/потолок/запас — общие)."""
    params = {
        "operation": "summary",
        "prompt_tokens": 0,
        "explicit_answer": None,
        "explicit_reasoning": None,
        "context_tokens": 0,
        "reserve_tokens": 512,
        "defensive_cap": CAP,
        "think_enabled": True,
    }
    params.update(kwargs)
    return resolve_budgets(**params)


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


# --- раздельные бюджеты (LSBN-T-33, ТкР 4.1) ----------------------------------


class TestResolveBudgets:
    def test_explicit_answer_and_reasoning_sum(self) -> None:
        """Явные бюджеты складываются: sent = 8000 + 32000 (ТкР 4.1.1)."""
        budgets = split(
            explicit_answer=8000,
            explicit_reasoning=32000,
            context_tokens=100000,
            defensive_cap=100000,
        )
        assert budgets.answer_budget == 8000
        assert budgets.answer_source == "explicit"
        assert budgets.reasoning_budget == 32000
        assert budgets.reasoning_source == "explicit"
        assert budgets.sent_budget == 40000
        # Дефолтный защитный потолок 32768 зажимает сумму.
        clamped = split(explicit_answer=8000, explicit_reasoning=32000, context_tokens=100000)
        assert clamped.sent_budget == 32768

    def test_think_off_sends_answer_only(self) -> None:
        """think=false: рассуждение не резервируется, sent = ответ (ТкР 4.1.2)."""
        budgets = split(
            explicit_answer=8000,
            explicit_reasoning=32000,
            context_tokens=100000,
            think_enabled=False,
        )
        assert budgets.sent_budget == 8000
        # Поле рассуждения считено, но в вызов не идёт.

    def test_sent_capped_by_context_window(self) -> None:
        """Известный контекст: sent = min(ответ+рассуждение, окно) (ТкР 4.1.3)."""
        budgets = split(
            explicit_answer=20000,
            explicit_reasoning=32000,
            context_tokens=32768,
            prompt_tokens=1000,
        )
        assert budgets.sent_budget == 32768 - 1000 - 512
        assert budgets.reasoning_budget == 32000

    def test_context_derived_reasoning_fills_the_window(self) -> None:
        """Явный ответ + авто-рассуждение: рассуждение — остаток окна."""
        budgets = split(
            explicit_answer=12000, context_tokens=32768, prompt_tokens=1000
        )
        window = 32768 - 1000 - 512
        assert budgets.answer_budget == 12000
        assert budgets.reasoning_budget == window - 12000
        assert budgets.sent_budget == window

    def test_unknown_context_degrades_like_341(self) -> None:
        """Неизвестный контекст — деградация: ответ по прежнему потолку,
        рассуждение 0, sent = ответ (ТкР 4.1.4, поведение 3.4.1)."""
        budgets = split(operation="merge", explicit_answer=None, context_tokens=0)
        assert budgets.answer_budget == 35000
        assert budgets.answer_source == "legacy"
        assert budgets.reasoning_budget == 0
        assert budgets.reasoning_source == "legacy"
        assert budgets.sent_budget == 35000

    def test_long_prompt_never_negative(self) -> None:
        """Промпт длиннее контекста: ответ не ниже 1, рассуждение не ниже 0
        (ТкР 4.1.5)."""
        budgets = split(context_tokens=100, prompt_tokens=500)
        assert budgets.answer_budget == 1
        assert budgets.reasoning_budget == 0
        assert budgets.sent_budget == 1

    def test_explicit_reasoning_zero_is_not_set(self) -> None:
        """0 у явного рассуждения = «не задан» (авто), как у слот-уровня."""
        budgets = split(
            explicit_answer=12000, explicit_reasoning=0, context_tokens=32768
        )
        assert budgets.reasoning_source == "context"

    def test_reasoning_capped_by_defensive_cap(self) -> None:
        """Защитный потолок зажимает и рассуждение."""
        budgets = split(
            explicit_answer=100,
            context_tokens=1000000,
            defensive_cap=32768,
        )
        assert budgets.reasoning_budget == 32768
        assert budgets.sent_budget == 32768
