"""Ротация с учётом пометок (№14, 3.2.0, arch §13.3).

Помеченный сайдкаром `.bad` снапшот исключается из кандидатов ротации: он не
вытесняет годные и не удаляется молча — решение о судьбе копии принимает
человек (FR-2.3).
"""

from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.services.backup import (
    IMAGE_PREFIX,
    IMAGE_SUFFIX,
    VERIFY_MARKER_SUFFIX,
    BackupService,
)
from app.storage.db import init_db


def _service(tmp_path: Path, keep: int) -> tuple[BackupService, Path]:
    settings = get_settings()
    init_db(settings)
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    adjusted = settings.model_copy(
        update={"backup_dir": str(backup_dir), "backup_keep": keep}
    )
    return BackupService(adjusted), backup_dir


def _snapshot(backup_dir: Path, stamp: str) -> Path:
    path = backup_dir / f"{IMAGE_PREFIX}{stamp}{IMAGE_SUFFIX}"
    path.write_bytes(b"x")
    return path


class TestRotateMarked:
    def test_rotate_skips_bad_marked(self, tmp_path: Path) -> None:
        """Помеченный переживает ротацию; годные ротируются как раньше."""
        service, backup_dir = _service(tmp_path, keep=2)
        oldest = _snapshot(backup_dir, "20260101T000000Z")
        second = _snapshot(backup_dir, "20260102T000000Z")
        third = _snapshot(backup_dir, "20260103T000000Z")
        newest = _snapshot(backup_dir, "20260104T000000Z")
        oldest.with_name(oldest.name + VERIFY_MARKER_SUFFIX).write_text(
            "{}", encoding="utf-8"
        )
        removed = service.rotate()
        # Кандидаты — только годные (second, third, newest): вытесняется second.
        assert removed == [second]
        assert oldest.exists()  # помеченный в живых (не удалён молча)
        assert oldest.with_name(oldest.name + VERIFY_MARKER_SUFFIX).exists()
        assert third.exists()
        assert newest.exists()

    def test_rotate_keeps_all_when_marked_exceed_keep(self, tmp_path: Path) -> None:
        """Если все сверх нормы помечены — не трогаем никого."""
        service, backup_dir = _service(tmp_path, keep=1)
        first = _snapshot(backup_dir, "20260101T000000Z")
        second = _snapshot(backup_dir, "20260102T000000Z")
        for path in (first, second):
            path.with_name(path.name + VERIFY_MARKER_SUFFIX).write_text(
                "{}", encoding="utf-8"
            )
        assert service.rotate() == []
        assert first.exists()
        assert second.exists()
