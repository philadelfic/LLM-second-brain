"""Проверка «свежий релиз»: источник, деградация, троттлинг (№45, 3.2.0).

Юниты без сети и без БД: чистые функции и `UpdateChecker` над временным
файлом-источником. Источник — локальный JSON (`UPDATE_SOURCE_FILE`), HTTP —
опция; отсутствие/мусор/недоступность деградируют тихо (одно событие
`update_check_failed`, без исключений наружу). Хинт — не чаще периода
(`UPDATE_HINT_EVERY_SEC`; 0 — один раз на процесс).
"""

from __future__ import annotations

import json
import logging

import pytest

from app.config import ConfigError, Settings, load_settings
from app.services.update_check import UpdateChecker, parse_version

REQUIRED_ENV: dict[str, str] = {
    "EMBEDDING_BASE_URL": "http://localhost:11434",
    "SUMMARY_BASE_URL": "http://localhost:11434",
    "SUMMARY_MODEL": "lfm2.5:latest",
    "JUDGE_BASE_URL": "http://localhost:11434",
    "JUDGE_MODEL": "lfm2.5:latest",
    "MCP_AUTH_TOKEN": "test-secret-token",
}


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    env = {name.lower(): value for name, value in REQUIRED_ENV.items()}
    env.update({name.lower(): value for name, value in overrides.items()})
    for name, value in env.items():
        monkeypatch.setenv(name.upper(), value)
    return load_settings()


def _write_source(tmp_path, payload: dict) -> str:
    path = tmp_path / "release.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_parse_version() -> None:
    """`3.2.0` / `v3.2.0` → тройка; мусор → None (arch §13.9)."""
    assert parse_version("3.2.0") == (3, 2, 0)
    assert parse_version("v3.2.0") == (3, 2, 0)
    assert parse_version("  3.10.1 ") == (3, 10, 1)
    assert parse_version("nonsense") is None
    assert parse_version("3.2") is None
    assert parse_version("v3.2.0-beta") is None
    assert parse_version("") is None


def test_newer_release_detected(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Файл-источник с 3.2.0 при __version__=3.1.0 → latest.version == 3.2.0."""
    source = _write_source(tmp_path, {"version": "3.2.0", "notes": "short what's new"})
    settings = _settings(
        monkeypatch, UPDATE_CHECK_ENABLED="true", UPDATE_SOURCE_FILE=source
    )
    checker = UpdateChecker(settings)
    checker.refresh()
    latest = checker.latest()
    assert latest is not None
    assert latest["version"] == "3.2.0"
    assert latest["notes"] == "short what's new"
    assert latest["checked_at"]


def test_same_or_older_no_hint(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Версия равна/старше текущей (3.1.0) → хинта нет."""
    for version in ("3.1.0", "2.9.9", "v0.1.0"):
        source = _write_source(tmp_path, {"version": version})
        settings = _settings(
            monkeypatch, UPDATE_CHECK_ENABLED="true", UPDATE_SOURCE_FILE=source
        )
        checker = UpdateChecker(settings)
        checker.refresh()
        assert checker.hint() is None


def test_source_missing_degrades(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Нет файла-источника → latest None, hint None, update_check_failed, без сбоя."""
    missing = str(tmp_path / "absent.json")
    settings = _settings(
        monkeypatch, UPDATE_CHECK_ENABLED="true", UPDATE_SOURCE_FILE=missing
    )
    checker = UpdateChecker(settings)
    with caplog.at_level(logging.WARNING, logger="app"):
        checker.refresh()  # исключений наружу нет
    assert checker.latest() is None
    assert checker.hint() is None
    events = [getattr(record, "event", None) for record in caplog.records]
    assert "update_check_failed" in events


def test_malformed_json_degrades(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Битый JSON → деградация, update_check_failed, без исключений."""
    path = tmp_path / "release.json"
    path.write_text("{not json", encoding="utf-8")
    settings = _settings(
        monkeypatch, UPDATE_CHECK_ENABLED="true", UPDATE_SOURCE_FILE=str(path)
    )
    checker = UpdateChecker(settings)
    with caplog.at_level(logging.WARNING, logger="app"):
        checker.refresh()
    assert checker.latest() is None
    events = [getattr(record, "event", None) for record in caplog.records]
    assert "update_check_failed" in events


def test_hint_throttled_per_period(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Первый вызов даёт текст, второй (в окне периода) — None."""
    source = _write_source(tmp_path, {"version": "9.9.9", "notes": "big new release"})
    settings = _settings(
        monkeypatch,
        UPDATE_CHECK_ENABLED="true",
        UPDATE_SOURCE_FILE=source,
        UPDATE_HINT_EVERY_SEC="86400",
    )
    checker = UpdateChecker(settings)
    checker.refresh()
    with caplog.at_level(logging.INFO, logger="app"):
        first = checker.hint()
        second = checker.hint()
    assert first is not None
    assert "v9.9.9" in first
    assert "big new release" in first
    assert "ask your human to update the image." in first
    assert second is None
    events = [getattr(record, "event", None) for record in caplog.records]
    assert events.count("update_hint_shown") == 1


def test_hint_once_per_process_when_every_zero(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UPDATE_HINT_EVERY_SEC=0 — ровно один хинт на процесс."""
    source = _write_source(tmp_path, {"version": "9.9.9", "notes": "n"})
    settings = _settings(
        monkeypatch,
        UPDATE_CHECK_ENABLED="true",
        UPDATE_SOURCE_FILE=source,
        UPDATE_HINT_EVERY_SEC="0",
    )
    checker = UpdateChecker(settings)
    checker.refresh()
    assert checker.hint() is not None
    assert checker.hint() is None


def test_hint_respects_max_chars(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Текст хинта не длиннее UPDATE_HINT_MAX_CHARS (ограничение §8)."""
    source = _write_source(tmp_path, {"version": "9.9.9", "notes": "x" * 500})
    settings = _settings(
        monkeypatch,
        UPDATE_CHECK_ENABLED="true",
        UPDATE_SOURCE_FILE=source,
        UPDATE_HINT_MAX_CHARS="80",
    )
    checker = UpdateChecker(settings)
    checker.refresh()
    hint = checker.hint()
    assert hint is not None
    assert len(hint) <= 80


@pytest.mark.asyncio
async def test_disabled_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """UPDATE_CHECK_ENABLED=false → run не стартует, источник не читается."""
    calls: list[int] = []

    def fetcher() -> dict | None:
        calls.append(1)
        raise AssertionError("источник прочитан при выключенной проверке")

    settings = _settings(
        monkeypatch,
        UPDATE_CHECK_ENABLED="false",
        UPDATE_SOURCE_FILE="/tmp/should-not-be-read.json",
    )
    checker = UpdateChecker(settings, fetcher=fetcher)
    await checker.run()  # завершается сразу
    assert calls == []
    assert checker.latest() is None
    assert checker.hint() is None


def test_config_invalid_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Включено без источника / UPDATE_HINT_MAX_CHARS=0 → ConfigError (NFR-6)."""
    with pytest.raises(ConfigError, match="update_check_enabled"):
        _settings(monkeypatch, UPDATE_CHECK_ENABLED="true")
    with pytest.raises(ConfigError, match="update_hint_max_chars"):
        _settings(
            monkeypatch,
            UPDATE_SOURCE_FILE="/tmp/release.json",
            UPDATE_HINT_MAX_CHARS="0",
        )
