"""Конфигурация decision-слота (LSBN-F-12, релиз 3.2.0; arch §8).

Дефолты (выключен, таймаут 300, порог 0.5, зона 0.05, переключение вкл,
потолки 64/40); при `DECISION_ENABLED=true` обязательны адрес (http(s)) и
модель; порог ± зона обязаны лежать внутри 0..1 — иначе фатальный `ConfigError`
(NFR-6).
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
    monkeypatch.delenv("DB_PATH", raising=False)
    env = {name.lower(): value for name, value in REQUIRED_ENV.items()}
    env.update({name.lower(): value for name, value in overrides.items()})
    for name, value in env.items():
        monkeypatch.setenv(name.upper(), value)
    get_settings.cache_clear()
    try:
        return get_settings()
    finally:
        get_settings.cache_clear()


def test_decision_defaults(monkeypatch) -> None:
    settings = load(monkeypatch)
    assert settings.decision_enabled is False
    assert settings.decision_timeout_sec == 300
    assert settings.decision_threshold == 0.5
    assert settings.decision_uncertainty_zone == 0.05
    assert settings.decision_fallback is True
    assert settings.decision_max_questions == 64
    assert settings.decision_max_labels == 40


def test_enabled_requires_url_and_model(monkeypatch) -> None:
    with pytest.raises(ConfigError, match="decision_base_url"):
        load(monkeypatch, DECISION_ENABLED="true", DECISION_MODEL="m")
    with pytest.raises(ConfigError, match="decision_model"):
        load(monkeypatch, DECISION_ENABLED="true", DECISION_BASE_URL="http://x", DECISION_MODEL="")


def test_disabled_does_not_require_url_and_model(monkeypatch) -> None:
    settings = load(monkeypatch)
    assert settings.decision_base_url == ""
    assert settings.decision_model == ""


def test_threshold_zone_out_of_range_fatal(monkeypatch) -> None:
    with pytest.raises(ConfigError, match="decision_threshold"):
        load(
            monkeypatch,
            DECISION_ENABLED="true",
            DECISION_BASE_URL="http://x",
            DECISION_MODEL="m",
            DECISION_THRESHOLD="0.97",
        )


def test_max_questions_out_of_range_fatal(monkeypatch) -> None:
    with pytest.raises(ConfigError, match="decision_max_questions"):
        load(monkeypatch, DECISION_MAX_QUESTIONS="65")
