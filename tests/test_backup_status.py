"""Backups observability by a directory scan (#14, 3.2.0, arch §5.1/§7.2).

    `BackupService.status()` — a directory snapshot WITHOUT opening the DB: the
    snapshots' count/age/volume, the marked count and the latest verify outcomes
    from the sidecar index. The index survives a restart (the `backups-index.json`
    file).
"""

from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.services.backup import (
    IMAGE_PREFIX,
    IMAGE_SUFFIX,
    STATE_FILE_NAME,
    VERIFY_MARKER_SUFFIX,
    BackupService,
)
from app.storage.db import init_db


def _service(tmp_path: Path) -> tuple[BackupService, Path]:
    settings = get_settings()
    init_db(settings)
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    adjusted = settings.model_copy(update={"backup_dir": str(backup_dir)})
    return BackupService(adjusted), backup_dir


def _snapshot(backup_dir: Path, stamp: str, size: int) -> Path:
    path = backup_dir / f"{IMAGE_PREFIX}{stamp}{IMAGE_SUFFIX}"
    path.write_bytes(b"x" * size)
    return path


class TestStatus:
    def test_status_empty_dir(self, tmp_path: Path) -> None:
        service, _ = _service(tmp_path)
        status = service.status()
        assert status == {
            "count": 0,
            "newest_age_sec": None,
            "oldest_age_sec": None,
            "total_bytes": 0,
            "bad": 0,
            "last_verify": None,
            "last_ship": None,
        }

    def test_status_counts_ages_bytes(self, tmp_path: Path) -> None:
        service, backup_dir = _service(tmp_path)
        first = _snapshot(backup_dir, "20260101T000000Z", 3)
        _snapshot(backup_dir, "20260102T000000Z", 5)
        first.with_name(first.name + VERIFY_MARKER_SUFFIX).write_text(
            "{}", encoding="utf-8"
        )
        # A foreign file and the index itself do not count as snapshots.
        (backup_dir / STATE_FILE_NAME).write_text("{}", encoding="utf-8")
        (backup_dir / "operator-notes.txt").write_text("полка", encoding="utf-8")
        status = service.status()
        assert status["count"] == 2
        assert status["total_bytes"] == 8
        assert status["bad"] == 1
        assert status["newest_age_sec"] is not None
        assert status["oldest_age_sec"] is not None
        assert status["newest_age_sec"] <= status["oldest_age_sec"]
        assert status["last_verify"] is None  # the index has just been created

    def test_status_reads_last_verify_from_index(self, tmp_path: Path) -> None:
        service, _ = _service(tmp_path)
        target = service.snapshot()
        service.verify(target)
        status = service.status()
        assert status["last_verify"]["file"] == target.name
        assert status["last_verify"]["result"] == "ok"
        assert status["last_verify"]["at"]
        assert status["count"] >= 1

    def test_status_survives_broken_index(self, tmp_path: Path) -> None:
        """Garbage in the sidecar index does not crash observability (last_verify None)."""
        service, backup_dir = _service(tmp_path)
        (backup_dir / STATE_FILE_NAME).write_text("not json", encoding="utf-8")
        status = service.status()
        assert status["last_verify"] is None
        assert status["last_ship"] is None
