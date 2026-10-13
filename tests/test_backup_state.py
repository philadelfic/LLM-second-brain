"""Исходы выноса переживают рестарт (№14, 3.2.0, arch §5.1/§6.1).

Сайдкар-индекс `backups-index.json` — единственное хранилище исходов выноса:
новый экземпляр сервиса читает `last_ship` с диска, а представление
`GET /backups` (`list_snapshots`) помечает подтверждённую копию `shipped`.
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
    """`last_ship` переживает новый экземпляр сервиса (сайдкар-индекс)."""
    backup_dir = _new_dir(tmp_path)
    first = _service(backup_dir, backup_ship_cmd=SIZE_REPORT_CMD)
    target = first.snapshot()
    assert first.ship(target)["ok"] is True
    assert (backup_dir / STATE_FILE_NAME).is_file()

    # Новый экземпляр поверх того же каталога: исход выноса читается с диска.
    second = _service(backup_dir, backup_ship_cmd=SIZE_REPORT_CMD)
    last_ship = second.status()["last_ship"]
    assert last_ship["file"] == target.name
    assert last_ship["result"] == "ok"
    assert last_ship["at"]
    assert last_ship["size_bytes"] == target.stat().st_size

    items = {item["file"]: item for item in second.list_snapshots()}
    assert items[target.name]["shipped"] is True
    # Вынос отметку проверки не пишет: снапшот никто не проверял — unknown.
    assert items[target.name]["verified"] == "unknown"


def test_last_ship_failure_persisted(tmp_path: Path) -> None:
    """Отказ выноса тоже переживает рестарт (`result=failed` и `reason`)."""
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
