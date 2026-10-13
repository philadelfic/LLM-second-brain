"""Наблюдаемость области логов (LSBN-F-10, f10-04, arch §10).

`/health.areas.logs = {enabled, pending}` собирается из реестра `ALL_AREAS`
(№40) — накопление pending видно и у выключенной области (флаг гейтит только
агентскую поверхность и фон, §13.11); стартовое событие `areas_enabled` несёт
текущее значение флага; петля `areas` выключенную область не забирает — pending
лога не уменьшается, а при включении запись обрабатывается.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app
from app.services.logs import LogsService
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session
from fakes import HashEmbedder, clear_seeded_skills

DIM = 8

ACTION = "Записал замер эмбеддингов"
REASON = "нужно было решить про слот"


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    """Настройки теста: флаг области логов снят, заданные — выставлены."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    monkeypatch.delenv("AREA_LOGS_ENABLED", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)
    return settings


def _save_log(settings: Settings) -> int:
    """Запись лога своим сервисом (собирается и у выключенной области, §13.11)."""
    result = LogsService(settings, embedding=HashEmbedder(DIM)).save(ACTION, REASON)
    assert result["created"] is True, result
    return int(result["id"])


def _status(settings: Settings, log_id: int) -> str:
    with session(settings) as conn:
        return conn.execute(
            "SELECT vector_status FROM assistant_logs WHERE id = ?", (log_id,)
        ).fetchone()[0]


def _areas(settings: Settings) -> dict:
    """Снимок `/health.areas` как его видит оператор."""
    with TestClient(create_app(settings)) as client:
        return client.get("/health").json()["areas"]


def _areas_enabled_events(caplog: pytest.LogCaptureFixture) -> list:
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "areas_enabled"
    ]


def test_health_areas_includes_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """`/health.areas.logs` = {enabled, pending}; pending виден и под off-флагом."""
    settings = _settings(monkeypatch)
    log_id = _save_log(settings)
    assert _status(settings, log_id) == "pending"
    assert _areas(settings)["logs"] == {"enabled": True, "pending": 1}

    off = _settings(monkeypatch, AREA_LOGS_ENABLED="false")
    assert _areas(off)["logs"] == {"enabled": False, "pending": 1}


def test_startup_areas_event_includes_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Событие старта `areas_enabled` несёт `logs` с текущим значением флага."""
    on = _settings(monkeypatch)
    with caplog.at_level(logging.INFO, logger="app"):
        with TestClient(create_app(on)):
            pass
    events = _areas_enabled_events(caplog)
    assert events, [record.getMessage() for record in caplog.records]
    assert events[0].areas["logs"] is True

    off = _settings(monkeypatch, AREA_LOGS_ENABLED="false")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="app"):
        with TestClient(create_app(off)):
            pass
    off_events = _areas_enabled_events(caplog)
    assert off_events, [record.getMessage() for record in caplog.records]
    assert off_events[0].areas["logs"] is False


def test_area_loop_skips_disabled_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Петля `areas`: выключенная область pending лога не забирает (§10)."""
    settings = _settings(monkeypatch, AREA_LOGS_ENABLED="false")
    log_id = _save_log(settings)
    assert BackgroundWorker(settings, HashEmbedder(DIM)).process_pending_areas() == 0
    assert _status(settings, log_id) == "pending"

    on = _settings(monkeypatch, AREA_LOGS_ENABLED="true")
    assert BackgroundWorker(on, HashEmbedder(DIM)).process_pending_areas() == 1
    assert _status(settings, log_id) == "ok"
