"""The verification integrated into the backups loop (#14, 3.2.0, arch §6.1/§13.2).

    `run()` after a successful snapshot verifies every N-th snapshot per
    `BACKUP_VERIFY_EVERY` (0 — manual only); a verify failure does not crash the
    loop (NFR-3); the outcome — as a journal event. After a successful
    verification and with `BACKUP_SHIP_CMD` set, the copy ships off the server
    (`ship`); a ship failure does not crash the loop or break the local rotation
    (FR-3.2).
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

    def build(every: int, **overrides: Any) -> BackupService:
        return BackupService(
            settings.model_copy(
                update={
                    "backup_dir": str(backup_dir),
                    "backup_interval_sec": 0.05,
                    "backup_verify_every": every,
                    **overrides,
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
    """At `backup_verify_every=1` verify is called after a snapshot."""
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
    """At `backup_verify_every=0` the auto-verification does not start."""
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
    """A verify failure → a backup_verify_failed event; the loop keeps snapshotting."""
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
    # The loop is alive: snapshots keep being taken.
    assert len(list(backup_dir.glob(f"{IMAGE_PREFIX}*{IMAGE_SUFFIX}"))) >= 1


@pytest.mark.asyncio
async def test_loop_ships_after_verify(
    service, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A command set and the verification passed — `ship` called with the fresh snapshot."""
    build, _backup_dir = service
    instance = build(1, backup_ship_cmd="true")
    shipped: list[Path] = []

    def ship_spy(path: Path) -> dict[str, Any]:
        shipped.append(path)
        return {"file": path.name, "ok": True}

    monkeypatch.setattr(instance, "ship", ship_spy)
    await _run_until(instance, lambda: bool(shipped))
    assert shipped
    assert shipped[0].name.startswith(IMAGE_PREFIX)
    assert shipped[0].exists()


@pytest.mark.asyncio
async def test_loop_does_not_ship_when_verify_failed(
    service, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`verify.ok=False` — the ship does not start (FR-3.2)."""
    build, _backup_dir = service
    instance = build(1, backup_ship_cmd="true")
    verified: list[Any] = []
    shipped: list[Any] = []

    def verify_spy(path: Path | None = None) -> dict[str, Any]:
        verified.append(path)
        return {
            "file": None,
            "ok": False,
            "reason": "not_openable",
            "checks": {},
            "latency_ms": 0.0,
        }

    def ship_spy(path: Path) -> dict[str, Any]:  # pragma: no cover
        shipped.append(path)
        return {}

    monkeypatch.setattr(instance, "verify", verify_spy)
    monkeypatch.setattr(instance, "ship", ship_spy)
    await _run_until(instance, lambda: len(verified) >= 2)
    assert len(verified) >= 2
    assert shipped == []


@pytest.mark.asyncio
async def test_loop_does_not_ship_when_disabled(
    service, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty `BACKUP_SHIP_CMD` — ship is never called, although verify runs."""
    build, _backup_dir = service
    instance = build(1)  # the default command is empty
    verified: list[Any] = []
    shipped: list[Any] = []
    real = instance.verify

    def verify_spy(path: Path | None = None) -> dict[str, Any]:
        verified.append(path)
        return real(path)

    def ship_spy(path: Path) -> dict[str, Any]:  # pragma: no cover
        shipped.append(path)
        return {}

    monkeypatch.setattr(instance, "verify", verify_spy)
    monkeypatch.setattr(instance, "ship", ship_spy)
    await _run_until(instance, lambda: len(verified) >= 2)
    assert len(verified) >= 2
    assert shipped == []
