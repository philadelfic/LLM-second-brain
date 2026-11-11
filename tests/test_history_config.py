"""The history layer's configuration (#41, brief no41-01): defaults and validation.

    The default = the v3.1.0 behavior + history: the layer is on, the retention
    policy is "keep everything" (`0`/`0`), the per-row delta threshold is 2048
    characters. Invalid values (`HISTORY_DELTA_INLINE_MAX=0`, a negative versions
    ceiling) — a fatal configuration error with ALL violations at once (NFR-6),
    not a silent default.
"""

from __future__ import annotations

import pytest

from app.config import ConfigError, get_settings, load_settings


def test_history_defaults() -> None:
    """Defaults: the layer on, 0/0 (keep everything), the delta threshold 2048."""
    settings = get_settings()

    assert settings.history_enabled is True
    assert settings.history_max_versions == 0
    assert settings.history_retention_days == 0
    assert settings.history_delta_inline_max == 2048


def test_history_invalid_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """`HISTORY_DELTA_INLINE_MAX=0` and a negative ceiling — one ConfigError."""
    monkeypatch.setenv("HISTORY_DELTA_INLINE_MAX", "0")
    monkeypatch.setenv("HISTORY_MAX_VERSIONS", "-1")
    monkeypatch.setenv("HISTORY_RETENTION_DAYS", "-5")
    get_settings.cache_clear()

    with pytest.raises(ConfigError) as error:
        load_settings()

    message = str(error.value)
    # We collect ALL violations at once (NFR-6).
    assert "history_delta_inline_max" in message
    assert "history_max_versions" in message
    assert "history_retention_days" in message
    get_settings.cache_clear()
