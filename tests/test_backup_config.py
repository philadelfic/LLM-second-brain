"""Конфигурация проверки бэкапов (№14, 3.2.0, arch §8).

`BACKUP_VERIFY_EVERY` — int ≥ 0, дефолт 1: проверяем каждый свежий снапшот;
0 — авто-проверка выключена (только ручная `POST /backups/verify`).
Отрицательное значение — фатальная ошибка конфигурации со ВСЕМИ нарушениями.
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
    monkeypatch.delenv("DB_PATH", raising=False)  # дефолт §8, как в test_config
    monkeypatch.delenv("BACKUP_VERIFY_EVERY", raising=False)


def test_verify_every_defaults_and_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Дефолт 1; отрицательное значение — ConfigError со всеми нарушениями."""
    assert load_env(monkeypatch).backup_verify_every == 1
    with pytest.raises(ConfigError) as exc_info:
        load_env(monkeypatch, BACKUP_VERIFY_EVERY="-1", BACKUP_KEEP="0")
    message = str(exc_info.value)
    assert "backup_verify_every" in message
    assert "backup_keep" in message


def test_verify_every_zero_is_manual_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """0 — валидное значение: авто-проверка выключена (вручную)."""
    assert load_env(monkeypatch, BACKUP_VERIFY_EVERY="0").backup_verify_every == 0
