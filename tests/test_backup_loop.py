"""Интеграция проверки в петлю бэкапов (№14, 3.2.0, arch §6.1/§13.2).

`run()` после успешного снапшота проверяет каждый N-й снапшот по
`BACKUP_VERIFY_EVERY` (0 — только вручную); отказ проверки петлю не роняет
(NFR-3), исход — событием журнала.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path
from typing import Any

import pytest

from app.config import get_settings
from app.services.backup import IMAGE_PREFIX, IMAGE_SUFFIX, BackupService
from app.storage.db import init_db


@pytest.fixture
def service(tmp_path: Path):
    settings = get_settings()
    init_db(settings)
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    def build(every: int) -> BackupService:
        return BackupService(
            settings.model_copy(
                update={
                    "backup_dir": str(backup_dir),
                    "backup_interval_sec": 0.05,
                    "backup_verify_every": every,
                }
            )
        )

    return build, backup_dir


def _with_event(caplog: pytest.LogCaptureFixture, event: str) -> list[Any]:
    return [
        record for record in caplog.records if getattr(record, "event", None) == event
    ]


async def _run_until(service: BackupService, predicate) -> None:
    task = asyncio.create_task(service.run())
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if predicate():
                break
            await asyncio.sleep(0.05)
    finally:
        service.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_loop_verifies_every_n(service, monkeypatch: pytest.MonkeyPatch) -> None:
    """При `backup_verify_every=1` после снапшота вызван verify."""
    build, _backup_dir = service
    instance = build(1)
    calls: list[Any] = []
    real = instance.verify

    def spy(path: Path | None = None) -> dict[str, Any]:
        calls.append(path)
        return real(path)

    monkeypatch.setattr(instance, "verify", spy)
    await _run_until(instance, lambda: bool(calls))
    assert calls


@pytest.mark.asyncio
async def test_loop_does_not_verify_when_zero(service) -> None:
    """При `backup_verify_every=0` авто-проверка не запускается."""
    build, backup_dir = service
    instance = build(0)
    calls: list[Any] = []

    def spy(path: Path | None = None) -> dict[str, Any]:  # pragma: no cover
        calls.append(path)
        return {}

    instance.verify = spy  # type: ignore[method-assign]
    await _run_until(
        instance,
        lambda: len(list(backup_dir.glob(f"{IMAGE_PREFIX}*{IMAGE_SUFFIX}"))) >= 2,
    )
    assert calls == []


@pytest.mark.asyncio
async def test_verify_failure_does_not_kill_loop(
    service, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Отказ verify → событие backup_verify_failed, петля продолжает снапшоты."""
    build, backup_dir = service
    instance = build(1)

    def boom(path: Path | None = None) -> dict[str, Any]:
        raise RuntimeError("verify crashed")

    monkeypatch.setattr(instance, "verify", boom)
    with caplog.at_level(logging.WARNING, logger="app"):
        await _run_until(
            instance,
            lambda: bool(_with_event(caplog, "backup_verify_failed")),
        )
    assert _with_event(caplog, "backup_verify_failed")
    # Петля жива: снапшоты продолжают сниматься.
    assert len(list(backup_dir.glob(f"{IMAGE_PREFIX}*{IMAGE_SUFFIX}"))) >= 1
