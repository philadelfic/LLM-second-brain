"""The decision slot's configuration (LSBN-F-12, release 3.2.0; arch §8).

    Defaults (off, timeout 300, threshold 0.5, zone 0.05, fallback on,
    ceilings 64/40); at `DECISION_ENABLED=true` an address (http(s)) and a model
    are mandatory; threshold ± zone must lie within 0..1 — else a fatal
    `ConfigError` (NFR-6).
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


# --- LSBN-F-37: порог choice и длина заметки в state ---------------------------


def test_choice_threshold_and_note_chars_defaults(monkeypatch) -> None:
    """Умолчания: порог choice по confidence 0.6, заметка в state — 800 знаков."""
    settings = load(monkeypatch)
    assert settings.decision_choice_threshold == 0.6
    assert settings.decision_note_chars == 800
    # Отдельный порог choice не подменяет общий порог noul/score.
    assert settings.decision_threshold == 0.5


def test_choice_threshold_and_note_chars_overridable(monkeypatch) -> None:
    """Обе ручки читаются из env независимо друг от друга."""
    settings = load(
        monkeypatch,
        DECISION_CHOICE_THRESHOLD="0.55",
        DECISION_NOTE_CHARS="1200",
        DECISION_THRESHOLD="0.8",
    )
    assert settings.decision_choice_threshold == 0.55
    assert settings.decision_note_chars == 1200
    assert settings.decision_threshold == 0.8


def test_choice_threshold_out_of_range_fatal(monkeypatch) -> None:
    """Порог choice вне 0..1 — фатальная ошибка конфигурации (NFR-6)."""
    with pytest.raises(ConfigError, match="decision_choice_threshold"):
        load(monkeypatch, DECISION_CHOICE_THRESHOLD="1.5")


def test_note_chars_out_of_range_fatal(monkeypatch) -> None:
    """Длина заметки в state вне 100..20000 — фатальная ошибка конфигурации."""
    with pytest.raises(ConfigError, match="decision_note_chars"):
        load(monkeypatch, DECISION_NOTE_CHARS="50")
    with pytest.raises(ConfigError, match="decision_note_chars"):
        load(monkeypatch, DECISION_NOTE_CHARS="20001")
