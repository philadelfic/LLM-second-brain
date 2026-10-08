"""Конфигурация слоя истории (№41, постановка no41-01): дефолты и валидация.

Дефолт = поведение v3.1.0 + история: слой включён, политика хранения «хранить
всё» (`0`/`0`), порог построчной дельты 2048 символов. Невалидные значения
(`HISTORY_DELTA_INLINE_MAX=0`, отрицательный потолок версий) — фатальная ошибка
конфигурации со ВСЕМИ нарушениями сразу (NFR-6), а не молчаливый дефолт.
"""

from __future__ import annotations

import pytest

from app.config import ConfigError, get_settings, load_settings


def test_history_defaults() -> None:
    """Дефолты: слой включён, 0/0 (хранить всё), порог дельты 2048."""
    settings = get_settings()

    assert settings.history_enabled is True
    assert settings.history_max_versions == 0
    assert settings.history_retention_days == 0
    assert settings.history_delta_inline_max == 2048


def test_history_invalid_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """`HISTORY_DELTA_INLINE_MAX=0` и отрицательный потолок — один ConfigError."""
    monkeypatch.setenv("HISTORY_DELTA_INLINE_MAX", "0")
    monkeypatch.setenv("HISTORY_MAX_VERSIONS", "-1")
    monkeypatch.setenv("HISTORY_RETENTION_DAYS", "-5")
    get_settings.cache_clear()

    with pytest.raises(ConfigError) as error:
        load_settings()

    message = str(error.value)
    # Собираем ВСЕ нарушения сразу (NFR-6).
    assert "history_delta_inline_max" in message
    assert "history_max_versions" in message
    assert "history_retention_days" in message
    get_settings.cache_clear()
