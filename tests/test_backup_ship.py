"""Shipping a verified copy off the server (#14, 3.2.0, arch §6.1/§13.4).

    The `BackupService.ship` unit: an empty command — a silent no-op without an
    event or a sidecar-index record; success — only `exit 0` AND a `size=<N>`
    report equal to the file size; otherwise `reason` ∈ {command_failed,
    size_mismatch, timeout}. An unreachable destination does not break local
    snapshots or rotation. The test commands — harmless local ones only
    (`printf`/`stat`/`exit`/`sleep`); no network.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from app.config import get_settings
from app.services.backup import (
    IMAGE_PREFIX,
    IMAGE_SUFFIX,
    MIN_SHIP_REPORT,
    SHIP_REASONS,
    STATE_FILE_NAME,
    BackupService,
)
from app.storage.db import init_db

# A sample command: prints the contract report `size=<the LOCAL file's size>`.
SIZE_REPORT_CMD = f"printf '{MIN_SHIP_REPORT}%s' $(stat -c%s \"$LSB_BACKUP_FILE\")"


def _service(tmp_path: Path, **overrides: object) -> tuple[BackupService, Path]:
    """A BackupService over an initialized DB; the copies — in tmp/backups."""
    settings = get_settings()
    init_db(settings)
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    adjusted = settings.model_copy(update={"backup_dir": str(backup_dir), **overrides})
    return BackupService(adjusted), backup_dir


def _events(caplog: pytest.LogCaptureFixture, event: str) -> list[Any]:
    return [
        record for record in caplog.records if getattr(record, "event", None) == event
    ]


def _snapshots(backup_dir: Path) -> list[str]:
    return sorted(path.name for path in backup_dir.glob(f"{IMAGE_PREFIX}*{IMAGE_SUFFIX}"))


class TestShipDisabled:
    def test_ship_disabled_is_noop(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An empty command: ok=None, skipped=True, disabled, no event."""
        service, backup_dir = _service(tmp_path)  # the default command is empty
        target = service.snapshot()
        with caplog.at_level(logging.INFO, logger="app"):
            result = service.ship(target)
        assert result["ok"] is None
        assert result["skipped"] is True
        assert result["reason"] == "disabled"
        assert result["file"] == target.name
        assert _events(caplog, "backup_shipped") == []
        assert _events(caplog, "backup_ship_failed") == []
        assert service.status()["last_ship"] is None
        assert not (backup_dir / STATE_FILE_NAME).exists()  # the index untouched


class TestShipOutcome:
    def test_ship_success_reports_size(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        service, backup_dir = _service(tmp_path, backup_ship_cmd=SIZE_REPORT_CMD)
        target = service.snapshot()
        with caplog.at_level(logging.INFO, logger="app"):
            result = service.ship(target)
        assert result["ok"] is True
        assert result["size_bytes"] == target.stat().st_size
        assert "reason" not in result
        assert result["latency_ms"] >= 0
        assert len(_events(caplog, "backup_shipped")) == 1
        assert _events(caplog, "backup_ship_failed") == []
        state = json.loads((backup_dir / STATE_FILE_NAME).read_text(encoding="utf-8"))
        assert state["last_ship"]["file"] == target.name
        assert state["last_ship"]["result"] == "ok"
        assert state["shipped"][target.name]["result"] == "ok"
        assert service.status()["last_ship"]["result"] == "ok"

    def test_ship_size_mismatch_fails(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The command prints a foreign size — ok=false, reason=size_mismatch."""
        service, _ = _service(tmp_path, backup_ship_cmd=f"printf '{MIN_SHIP_REPORT}1'")
        target = service.snapshot()
        assert target.stat().st_size != 1  # a DB snapshot cannot be a byte
        with caplog.at_level(logging.INFO, logger="app"):
            result = service.ship(target)
        assert result["ok"] is False
        assert result["reason"] == "size_mismatch"
        assert result["size_bytes"] == target.stat().st_size  # the local size
        failed = _events(caplog, "backup_ship_failed")
        assert len(failed) == 1
        assert getattr(failed[0], "reason", None) == "size_mismatch"
        assert _events(caplog, "backup_shipped") == []
        assert service.status()["last_ship"]["result"] == "failed"

    def test_ship_command_failure_does_not_raise(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`exit 1` — ok=false, reason=command_failed, no exception."""
        service, _ = _service(tmp_path, backup_ship_cmd="exit 1")
        target = service.snapshot()
        with caplog.at_level(logging.INFO, logger="app"):
            result = service.ship(target)
        assert result["ok"] is False
        assert result["reason"] == "command_failed"
        assert len(_events(caplog, "backup_ship_failed")) == 1
        assert _events(caplog, "backup_shipped") == []

    def test_ship_missing_report_is_command_failed(self, tmp_path: Path) -> None:
        """Code 0 but no contract `size=` report — the ship is not confirmed."""
        service, _ = _service(tmp_path, backup_ship_cmd="true")
        result = service.ship(service.snapshot())
        assert result["ok"] is False
        assert result["reason"] == "command_failed"

    def test_ship_timeout(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A "sleeping" command at a small timeout — reason=timeout (not a crash)."""
        service, _ = _service(
            tmp_path, backup_ship_cmd="sleep 3", backup_ship_timeout_sec=1
        )
        target = service.snapshot()
        with caplog.at_level(logging.INFO, logger="app"):
            result = service.ship(target)
        assert result["ok"] is False
        assert result["reason"] == "timeout"
        assert len(_events(caplog, "backup_ship_failed")) == 1

    def test_ship_reason_is_closed_set(self, tmp_path: Path) -> None:
        """Any ship failure — from the agreed `SHIP_REASONS` set."""
        service, _ = _service(tmp_path, backup_ship_cmd="exit 1")
        assert service.ship(service.snapshot())["reason"] in SHIP_REASONS


class TestShipLocalSafety:
    def test_ship_target_missing_is_local_safe(self, tmp_path: Path) -> None:
        """An unreachable destination: the copy intact; rotation and verification work."""
        service, backup_dir = _service(
            tmp_path, backup_ship_cmd='cp "$LSB_BACKUP_FILE" /nonexistent-lsb-ship-dir/out.db'
        )
        target = service.snapshot()
        before = _snapshots(backup_dir)
        result = service.ship(target)
        assert result["ok"] is False
        assert result["reason"] == "command_failed"
        assert _snapshots(backup_dir) == before  # the local snapshot untouched
        assert target.exists()
        assert service.rotate() == []  # rotation works as before
        assert service.verify(target)["ok"] is True
        assert service.status()["count"] == 1
