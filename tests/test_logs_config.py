"""The "Assistant logs" area's configuration (LSBN-F-10, f10-01).

    FR-1.3/NFR-6: the `area_logs_enabled` flag (the default "enabled") and the
    `logs_*` shape limits are read by `Settings`; an invalid value (an int < 1 or
    an invalid bool) — a fatal `ConfigError` startup listing ALL violations, not a
    silent disabling (arch §9).
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
    """The logs area's defaults (env off — the model's values are taken)."""
    for name in _LOGS_ENV:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    return get_settings()


def test_logs_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """The §9 defaults: the flag on; the limits 500/500/200/500/200/300."""
    settings = _settings(monkeypatch)
    assert settings.area_logs_enabled is True
    assert settings.logs_action_max_chars == 500
    assert settings.logs_reason_max_chars == 500
    assert settings.logs_entity_ref_max_chars == 200
    assert settings.logs_outcome_max_chars == 500
    assert settings.logs_verified_max_chars == 200
    assert settings.logs_search_excerpt_chars == 300


def test_logs_invalid_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """An invalid limit and an invalid bool — a fatal startup (NFR-6)."""
    # A limit < 1: the message names the violating field (ALL ranges collected).
    get_settings.cache_clear()
    monkeypatch.setenv("LOGS_ACTION_MAX_CHARS", "0")
    with pytest.raises(ConfigError) as exc:
        load_settings()
    assert "logs_action_max_chars" in str(exc.value)

    # An invalid bool: the pydantic bool refuses before the ranges — a fatal startup.
    get_settings.cache_clear()
    monkeypatch.setenv("AREA_LOGS_ENABLED", "maybe")
    with pytest.raises(ConfigError) as exc:
        load_settings()
    assert "area_logs_enabled" in str(exc.value)
