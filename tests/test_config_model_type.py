"""The model kind per slot and the timeouts (#49, release 3.2.0; arch §8, §13.1, §13.4).

    The kind default — `generation` (honest: it does not attribute reasoning to
    the model); a value outside the set and `decision` at a generative slot are
    fatal (NFR-6). The new timeout defaults — 900 s (the back-off/pause ceiling,
    "think long"); the `SLOT_REASONING_MIN_OUTPUT_TOKENS` threshold cannot exceed
    the protective ceiling.
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


def test_model_types_default_generation(monkeypatch) -> None:
    monkeypatch.delenv("DB_PATH", raising=False)
    settings = load(monkeypatch)
    assert settings.summary_model_type == "generation"
    assert settings.judge_model_type == "generation"


def test_invalid_model_type_fatal(monkeypatch) -> None:
    monkeypatch.delenv("DB_PATH", raising=False)
    with pytest.raises(ConfigError, match="summary_model_type"):
        load(monkeypatch, SUMMARY_MODEL_TYPE="smart")


def test_decision_on_generative_slot_fatal(monkeypatch) -> None:
    monkeypatch.delenv("DB_PATH", raising=False)
    with pytest.raises(ConfigError, match="DECISION_ENABLED"):
        load(monkeypatch, JUDGE_MODEL_TYPE="decision")


def test_timeout_defaults(monkeypatch) -> None:
    monkeypatch.delenv("DB_PATH", raising=False)
    settings = load(monkeypatch)
    assert settings.summary_timeout_sec == 900
    assert settings.judge_timeout_sec == 900
    assert settings.slot_min_timeout_sec == 300


def test_reasoning_min_above_cap_fatal(monkeypatch) -> None:
    monkeypatch.delenv("DB_PATH", raising=False)
    with pytest.raises(ConfigError, match="slot_reasoning_min_output_tokens"):
        load(
            monkeypatch,
            SLOT_REASONING_MIN_OUTPUT_TOKENS="65536",
            SLOT_MAX_OUTPUT_TOKENS="32768",
        )
