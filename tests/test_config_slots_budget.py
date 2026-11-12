"""The slot budget's defaults and validation (#49, release 3.2.0; arch §8, §13.2).

    The budget defaults — auto ("as much as fits"): 0 in `*_NUM_PREDICT`, 0 in the
    contexts; the reserve and the protective ceiling are set. A reserve smaller
    than the declared context — a fatal configuration error.
"""

from __future__ import annotations

import pytest

from app.config import ConfigError, get_settings

REQUIRED_ENV: dict[str, str] = {
    "EMBEDDING_BASE_URL": "http://localhost:11434",
    "SUMMARY_BASE_URL": "http://localhost:11434",
    "SUMMARY_MODEL": "lfm2.5:latest",
    "JUDGE_BASE_URL": "http://localhost:11434",
    "JUDGE_MODEL": "lfm2.5:latest",
    "MCP_AUTH_TOKEN": "test-secret-token",
}


def load(monkeypatch: pytest.MonkeyPatch, **overrides: str):
    env = {name.lower(): value for name, value in REQUIRED_ENV.items()}
    env.update({name.lower(): value for name, value in overrides.items()})
    for name, value in env.items():
        monkeypatch.setenv(name.upper(), value)
    get_settings.cache_clear()
    try:
        return get_settings()
    finally:
        get_settings.cache_clear()


def test_auto_defaults(monkeypatch) -> None:
    monkeypatch.delenv("DB_PATH", raising=False)
    settings = load(monkeypatch)
    assert settings.summary_num_predict == 0
    assert settings.merge_num_predict == 0
    assert settings.judge_num_predict == 0
    assert settings.summary_context_tokens == 0
    assert settings.judge_context_tokens == 0
    assert settings.slot_prompt_reserve_tokens == 512
    assert settings.slot_max_output_tokens == 32768


def test_reserve_above_context_fatal(monkeypatch) -> None:
    monkeypatch.delenv("DB_PATH", raising=False)
    with pytest.raises(ConfigError, match="slot_prompt_reserve_tokens"):
        load(
            monkeypatch,
            SUMMARY_CONTEXT_TOKENS="4096",
            SLOT_PROMPT_RESERVE_TOKENS="8192",
        )


def test_thinking_judge_without_budget_is_fatal(monkeypatch) -> None:
    """Думающий судья с авто-бюджетом и неизвестным контекстом — фатальный старт.

    Иначе авто-бюджет деградирует к прошлой капе операции judge (256 токенов),
    рассуждение съедает ответ и вердикт не выходит (живой сбой 12.11.2026).
    """
    monkeypatch.delenv("DB_PATH", raising=False)
    with pytest.raises(ConfigError, match="JUDGE_NUM_PREDICT"):
        load(monkeypatch, JUDGE_THINK="true")


def test_thinking_judge_with_explicit_budget_ok(monkeypatch) -> None:
    """Явный бюджет или объявленный контекст думающего судьи — старт возможен."""
    monkeypatch.delenv("DB_PATH", raising=False)
    settings = load(monkeypatch, JUDGE_THINK="true", JUDGE_NUM_PREDICT="4096")
    assert settings.judge_num_predict == 4096
    settings = load(
        monkeypatch, JUDGE_THINK="true", JUDGE_CONTEXT_TOKENS="131072"
    )
    assert settings.judge_context_tokens == 131072


def test_thoughtless_judge_auto_budget_ok(monkeypatch) -> None:
    """Недумающий судья с авто-бюджетом — прежнее поведение, старт возможен."""
    monkeypatch.delenv("DB_PATH", raising=False)
    settings = load(monkeypatch, JUDGE_THINK="false")
    assert settings.judge_think is False
