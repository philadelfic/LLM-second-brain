"""Ship outcomes survive a restart (#14, 3.2.0, arch §5.1/§6.1).

    The `backups-index.json` sidecar index is the only store of ship outcomes: a
    new service instance reads `last_ship` from disk, and the `GET /backups` view
    (`list_snapshots`) marks a confirmed copy `shipped`.
"""

from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.services.backup import STATE_FILE_NAME, BackupService
from app.storage.db import init_db

SIZE_REPORT_CMD = "printf 'size=%s' $(stat -c%s \"$LSB_BACKUP_FILE\")"


def _service(backup_dir: Path, **overrides: object) -> BackupService:
    settings = get_settings()
    init_db(settings)
    adjusted = settings.model_copy(update={"backup_dir": str(backup_dir), **overrides})
    return BackupService(adjusted)


def _new_dir(tmp_path: Path) -> Path:
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def test_last_ship_persisted(tmp_path: Path) -> None:
    """`last_ship` survives a new service instance (the sidecar index)."""
    backup_dir = _new_dir(tmp_path)
    first = _service(backup_dir, backup_ship_cmd=SIZE_REPORT_CMD)
    target = first.snapshot()
    assert first.ship(target)["ok"] is True
    assert (backup_dir / STATE_FILE_NAME).is_file()

    # A new instance over the same directory: the ship outcome is read from disk.
    second = _service(backup_dir, backup_ship_cmd=SIZE_REPORT_CMD)
    last_ship = second.status()["last_ship"]
    assert last_ship["file"] == target.name
    assert last_ship["result"] == "ok"
    assert last_ship["at"]
    assert last_ship["size_bytes"] == target.stat().st_size

    items = {item["file"]: item for item in second.list_snapshots()}
    assert items[target.name]["shipped"] is True
    # Shipping writes no verify mark: nobody verified the snapshot — unknown.
    assert items[target.name]["verified"] == "unknown"


def test_last_ship_failure_persisted(tmp_path: Path) -> None:
    """A ship failure survives a restart too (`result=failed` and the `reason`)."""
    backup_dir = _new_dir(tmp_path)
    first = _service(backup_dir, backup_ship_cmd="exit 1")
    target = first.snapshot()
    assert first.ship(target)["ok"] is False

    second = _service(backup_dir, backup_ship_cmd="exit 1")
    last_ship = second.status()["last_ship"]
    assert last_ship["file"] == target.name
    assert last_ship["result"] == "failed"
    assert last_ship["reason"] == "command_failed"

    items = {item["file"]: item for item in second.list_snapshots()}
    assert items[target.name]["shipped"] is False
