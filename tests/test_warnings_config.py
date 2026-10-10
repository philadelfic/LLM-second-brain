"""№12 (3.2.0): конфигурация джобы `warnings` — дефолты и валидация (arch §8)."""

from __future__ import annotations

import pytest

from app.config import ConfigError, get_settings


def test_defaults(test_env: dict[str, str]) -> None:
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.job_warnings_enabled is True
    assert settings.job_warnings_interval_sec == 3600
    assert settings.job_warnings_batch == 20
    assert settings.job_warnings_classifier_budget == 10
    assert settings.warnings_min_confidence == 0.80
    assert settings.warnings_level_strict is False


def test_invalid_reports_all_violations(test_env, monkeypatch) -> None:
    """Бюджет > батча и порог вне 0..1 → `ConfigError` со ВСЕМИ нарушениями."""
    monkeypatch.setenv("JOB_WARNINGS_BATCH", "5")
    monkeypatch.setenv("JOB_WARNINGS_CLASSIFIER_BUDGET", "9")
    monkeypatch.setenv("WARNINGS_MIN_CONFIDENCE", "1.5")
    get_settings.cache_clear()
    with pytest.raises(ConfigError) as exc:
        get_settings()
    message = str(exc.value)
    assert "job_warnings_classifier_budget" in message
    assert "warnings_min_confidence" in message
