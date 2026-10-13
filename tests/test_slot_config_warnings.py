"""Предупреждения конфигурации слота (№49, релиз 3.2.0; arch §7.2, §13.5).

Чистая функция `LLMClient.slot_config_warnings(spec)` — без сети и старта
приложения: пять кодов (`reasoning_tight_budget`, `reasoning_think_off`,
`budget_gt_context`, `context_unknown`, `timeout_short`) срабатывают по делу и
молчат на чистой конфигурации. Тип модели решает, какие защиты применимы:
`generation` + `think=false` — штатный запрет, не предупреждение.
"""

from __future__ import annotations

from app.services.llm_client import LLMClient, SlotSpec


def make_spec(**over: object) -> SlotSpec:
    base: dict[str, object] = {
        "name": "summary",
        "provider": "ollama",
        "base_url": "http://slot.test",
        "model": "m",
        "api_key": "",
        "read_timeout": 900.0,
        "context_tokens": 32768,
    }
    base.update(over)
    return SlotSpec(**base)  # type: ignore[arg-type]


def codes(spec: SlotSpec) -> list[str]:
    return [code for code, _text in LLMClient.slot_config_warnings(spec)]


def test_reasoning_tight_budget() -> None:
    """reasoning + контекст 8192 + явный бюджет 256 → мало места размышлению."""
    spec = make_spec(model_type="reasoning", context_tokens=8192, num_predict=256)
    assert "reasoning_tight_budget" in codes(spec)


def test_reasoning_tight_budget_ok_at_default() -> None:
    """reasoning + контекст 32768 + авто-бюджет → предупреждения нет."""
    spec = make_spec(model_type="reasoning", context_tokens=32768, num_predict=0)
    assert codes(spec) == []


def test_reasoning_think_off() -> None:
    """reasoning + think=False → размышление запрещено осознанно?"""
    spec = make_spec(model_type="reasoning", think=False)
    assert "reasoning_think_off" in codes(spec)


def test_think_off_without_reasoning_is_quiet() -> None:
    """generation + think=False → штатный запрет, предупреждения нет."""
    spec = make_spec(model_type="generation", think=False)
    assert codes(spec) == []


def test_budget_gt_context() -> None:
    """Явный бюджет 40000 при контексте 32768 → сервер обрежет/отклонит."""
    spec = make_spec(context_tokens=32768, num_predict=40000)
    assert "budget_gt_context" in codes(spec)


def test_context_unknown_warns() -> None:
    """Контекст 0 → деградация к прежним потолкам, ровно одно предупреждение."""
    spec = make_spec(context_tokens=0)
    assert codes(spec) == ["context_unknown"]


def test_timeout_short() -> None:
    """reasoning + таймаут 60 с → долгое размышление выглядит как таймаут."""
    spec = make_spec(model_type="reasoning", read_timeout=60.0)
    assert "timeout_short" in codes(spec)


def test_clean_config_no_warnings() -> None:
    """generation + контекст 32768 + таймаут 900 → список пуст."""
    spec = make_spec(model_type="generation", context_tokens=32768, read_timeout=900.0)
    assert codes(spec) == []
