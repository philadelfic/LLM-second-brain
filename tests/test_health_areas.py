"""The logs area's observability (LSBN-F-10, f10-04, arch §10).

    `/health.areas.logs = {enabled, pending}` is assembled from the `ALL_AREAS`
    registry (#40) — pending accumulation is visible for a disabled area too
    (the flag gates only the agent surface and the background, §13.11); the
    startup `areas_enabled` event carries the flag's current value; the `areas`
    loop does not drain a disabled area — the log's pending does not decrease,
    and on enabling the record is processed.
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
    """The test's settings: the logs area flag off; the given ones — set."""
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
    """A log record by its own service (assembled even with the area disabled, §13.11)."""
    result = LogsService(settings, embedding=HashEmbedder(DIM)).save(ACTION, REASON)
    assert result["created"] is True, result
    return int(result["id"])


def _status(settings: Settings, log_id: int) -> str:
    with session(settings) as conn:
        return conn.execute(
            "SELECT vector_status FROM assistant_logs WHERE id = ?", (log_id,)
        ).fetchone()[0]


def _areas(settings: Settings) -> dict:
    """The `/health.areas` snapshot as the operator sees it."""
    with TestClient(create_app(settings)) as client:
        return client.get("/health").json()["areas"]


def _areas_enabled_events(caplog: pytest.LogCaptureFixture) -> list:
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "areas_enabled"
    ]


def test_health_areas_includes_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """`/health.areas.logs` = {enabled, pending}; pending visible under the off flag too."""
    settings = _settings(monkeypatch)
    log_id = _save_log(settings)
    assert _status(settings, log_id) == "pending"
    assert _areas(settings)["logs"] == {"enabled": True, "pending": 1}

    off = _settings(monkeypatch, AREA_LOGS_ENABLED="false")
    assert _areas(off)["logs"] == {"enabled": False, "pending": 1}


def test_startup_areas_event_includes_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The startup `areas_enabled` event carries `logs` with the flag's current value."""
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
    """The `areas` loop: a disabled area does not drain the log's pending (§10)."""
    settings = _settings(monkeypatch, AREA_LOGS_ENABLED="false")
    log_id = _save_log(settings)
    assert BackgroundWorker(settings, HashEmbedder(DIM)).process_pending_areas() == 0
    assert _status(settings, log_id) == "pending"

    on = _settings(monkeypatch, AREA_LOGS_ENABLED="true")
    assert BackgroundWorker(on, HashEmbedder(DIM)).process_pending_areas() == 1
    assert _status(settings, log_id) == "ok"
