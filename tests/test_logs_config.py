"""Конфигурация области «Логи ассистента» (LSBN-F-10, f10-01).

FR-1.3/NFR-6: флаг `area_logs_enabled` (дефолт «включено») и лимиты формы
`logs_*` читаются `Settings`; невалидное значение (int < 1 или невалидный bool)
— фатальный старт `ConfigError` с перечислением ВСЕХ нарушений, а не молчаливое
выключение (arch §9).
"""

from __future__ import annotations

import pytest

from app.config import ConfigError, get_settings, load_settings

_LOGS_ENV = (
    "AREA_LOGS_ENABLED",
    "LOGS_ACTION_MAX_CHARS",
    "LOGS_REASON_MAX_CHARS",
    "LOGS_ENTITY_REF_MAX_CHARS",
    "LOGS_OUTCOME_MAX_CHARS",
    "LOGS_VERIFIED_MAX_CHARS",
    "LOGS_SEARCH_EXCERPT_CHARS",
)


def _settings(monkeypatch: pytest.MonkeyPatch):
    """Дефолты области логов (env снят — берём значения модели)."""
    for name in _LOGS_ENV:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    return get_settings()


def test_logs_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Дефолты §9: флаг включён, лимиты 500/500/200/500/200/300."""
    settings = _settings(monkeypatch)
    assert settings.area_logs_enabled is True
    assert settings.logs_action_max_chars == 500
    assert settings.logs_reason_max_chars == 500
    assert settings.logs_entity_ref_max_chars == 200
    assert settings.logs_outcome_max_chars == 500
    assert settings.logs_verified_max_chars == 200
    assert settings.logs_search_excerpt_chars == 300


def test_logs_invalid_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Невалидный лимит и невалидный bool — фатальный старт (NFR-6)."""
    # Лимит < 1: сообщение несёт имя поля-нарушителя (собираются ВСЕ диапазоны).
    get_settings.cache_clear()
    monkeypatch.setenv("LOGS_ACTION_MAX_CHARS", "0")
    with pytest.raises(ConfigError) as exc:
        load_settings()
    assert "logs_action_max_chars" in str(exc.value)

    # Невалидный bool: pydantic-bool отказывает до диапазонов — фатальный старт.
    get_settings.cache_clear()
    monkeypatch.setenv("AREA_LOGS_ENABLED", "maybe")
    with pytest.raises(ConfigError) as exc:
        load_settings()
    assert "area_logs_enabled" in str(exc.value)
