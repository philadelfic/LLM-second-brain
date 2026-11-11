"""The backups verification configuration (#14, 3.2.0, arch §8).

    `BACKUP_VERIFY_EVERY` — int ≥ 0, default 1: every fresh snapshot is verified;
    0 — the auto-verification off (only the manual `POST /backups/verify`).
    A negative value — a fatal configuration error with ALL violations.

    `BACKUP_SHIP_CMD` — the verified copy's ship command (the default empty —
    shipping off; the destination/credentials live outside the repository);
    `BACKUP_SHIP_TIMEOUT_SEC` — its timeout (int ≥ 1, default 300).
"""

from __future__ import annotations

import pytest

from app.config import ConfigError, Settings, load_settings

REQUIRED_ENV: dict[str, str] = {
    "EMBEDDING_BASE_URL": "http://localhost:11434",
    "SUMMARY_BASE_URL": "http://localhost:11434",
    "SUMMARY_MODEL": "lfm2.5:latest",
    "JUDGE_BASE_URL": "http://localhost:11434",
    "JUDGE_MODEL": "lfm2.5:latest",
    "MCP_AUTH_TOKEN": "test-secret-token",
}


def load_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    env = {name.lower(): value for name, value in REQUIRED_ENV.items()}
    env.update({name.lower(): value for name, value in overrides.items()})
    for name, value in env.items():
        monkeypatch.setenv(name.upper(), value)
    return load_settings()


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DB_PATH", raising=False)  # the §8 default, as in test_config
    monkeypatch.delenv("BACKUP_VERIFY_EVERY", raising=False)
    monkeypatch.delenv("BACKUP_SHIP_CMD", raising=False)
    monkeypatch.delenv("BACKUP_SHIP_TIMEOUT_SEC", raising=False)


def test_verify_every_defaults_and_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default 1; a negative value — a ConfigError with all violations."""
    assert load_env(monkeypatch).backup_verify_every == 1
    with pytest.raises(ConfigError) as exc_info:
        load_env(monkeypatch, BACKUP_VERIFY_EVERY="-1", BACKUP_KEEP="0")
    message = str(exc_info.value)
    assert "backup_verify_every" in message
    assert "backup_keep" in message


def test_verify_every_zero_is_manual_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """0 — a valid value: the auto-verification off (manual)."""
    assert load_env(monkeypatch, BACKUP_VERIFY_EVERY="0").backup_verify_every == 0


def test_ship_defaults_and_invalid_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ship defaults: an empty command (shipping off); the timeout 300 s."""
    settings = load_env(monkeypatch)
    assert settings.backup_ship_cmd == ""
    assert settings.backup_ship_timeout_sec == 300
    # The command is not a secret and is read as is (the destination/credentials — outside the repo).
    assert load_env(monkeypatch, BACKUP_SHIP_CMD="true").backup_ship_cmd == "true"


def test_ship_timeout_invalid_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """`BACKUP_SHIP_TIMEOUT_SEC=0` — a fatal configuration error."""
    with pytest.raises(ConfigError) as exc_info:
        load_env(monkeypatch, BACKUP_SHIP_TIMEOUT_SEC="0")
    assert 'backup_ship_timeout_sec' in str(exc_info.value)
