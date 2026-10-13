"""Проверка восстановления снапшота (№14, 3.2.0, arch §6.1/§13.1-13.3).

Юнит `BackupService.verify`: свежий снапшот открывается read-only с
загруженным sqlite-vec, integrity ok, обязательные таблицы и vec0 на месте;
битый файл ловится и **помечается** сайдкаром `.bad` (не удаляется); база без
vec0 отвергается; расхождение `meta.embedding_model` — замечание, не отказ;
удачная повторная проверка снимает маркер; сам снапшот проверкой не меняется.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.config import get_settings
from app.services.backup import (
    EMBEDDING_MISMATCH,
    VERIFY_MARKER_SUFFIX,
    BackupService,
)
from app.storage.db import init_db


def _service(tmp_path: Path, **overrides: object) -> tuple[BackupService, Path]:
    """BackupService поверх инициализированной БД; копии — в tmp/backups."""
    settings = get_settings()
    init_db(settings)
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    adjusted = settings.model_copy(update={"backup_dir": str(backup_dir), **overrides})
    return BackupService(adjusted), backup_dir


def _marker(path: Path) -> Path:
    return path.with_name(path.name + VERIFY_MARKER_SUFFIX)


class TestVerify:
    def test_verify_fresh_snapshot_ok(self, tmp_path: Path) -> None:
        service, _ = _service(tmp_path)
        target = service.snapshot()
        result = service.verify(target)
        assert result["ok"] is True
        assert result["file"] == target.name
        assert result["checks"]["integrity"] == "ok"
        assert result["checks"]["missing_tables"] == []
        assert result["checks"]["vec_unavailable"] == []
        assert "reason" not in result
        assert result["latency_ms"] >= 0
        assert not _marker(target).exists()

    def test_verify_newest_when_path_none(self, tmp_path: Path) -> None:
        """"`path=None` → свежайший по имени снапшот."""
        service, _ = _service(tmp_path)
        target = service.snapshot()
        assert service.verify()["file"] == target.name

    def test_verify_missing_file_is_not_openable(self, tmp_path: Path) -> None:
        service, backup_dir = _service(tmp_path)
        result = service.verify(backup_dir / "notes-20200101T000000Z.db")
        assert result["ok"] is False
        assert result["reason"] == "not_openable"

    def test_verify_corrupt_file_fails_and_marks(self, tmp_path: Path) -> None:
        """Обрезанный/мусорный файл: ok=false и сайдкар <file>.bad создан."""
        service, backup_dir = _service(tmp_path)
        broken = backup_dir / "notes-20200101T000000Z.db"
        broken.write_bytes(b"\x00\x01\x02not a sqlite database" * 8)
        result = service.verify(broken)
        assert result["ok"] is False
        assert result["reason"] in {"not_openable", "integrity_failed"}
        marker = _marker(broken)
        assert marker.exists()
        assert broken.exists()  # не удалён молча (FR-2.3)

    def test_verify_missing_vec_table_fails(self, tmp_path: Path) -> None:
        """База без обязательных таблиц (в т.ч. vec0) — missing_tables."""
        service, backup_dir = _service(tmp_path)
        plain = backup_dir / "notes-20200101T000000Z.db"
        conn = sqlite3.connect(plain)
        try:
            conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY)")
            conn.commit()
        finally:
            conn.close()
        result = service.verify(plain)
        assert result["ok"] is False
        assert result["reason"] in {"missing_tables", "vec_unavailable"}
        assert "notes_vec" in result["checks"]["missing_tables"]

    def test_verify_embedding_mismatch_is_note(self, tmp_path: Path) -> None:
        """Иной `meta.embedding_model` — замечание в checks, ok остаётся true."""
        service, _ = _service(tmp_path)
        target = service.snapshot()
        conn = sqlite3.connect(target)
        try:
            conn.execute(
                "UPDATE meta SET value = 'other-model:1b' WHERE key = 'embedding_model'"
            )
            conn.commit()
        finally:
            conn.close()
        result = service.verify(target)
        assert result["ok"] is True
        assert result["checks"][EMBEDDING_MISMATCH] is True

    def test_verify_success_removes_marker(self, tmp_path: Path) -> None:
        """После устранения причины (годный снапшот) маркер снят."""
        service, _ = _service(tmp_path)
        target = service.snapshot()
        marker = _marker(target)
        marker.write_text('{"reason": "integrity_failed"}', encoding="utf-8")
        result = service.verify(target)
        assert result["ok"] is True
        assert not marker.exists()

    def test_verify_does_not_modify_snapshot(self, tmp_path: Path) -> None:
        """Проверка read-only: mtime и размер снапшота не изменились."""
        service, _ = _service(tmp_path)
        target = service.snapshot()
        before = (target.stat().st_mtime_ns, target.stat().st_size)
        service.verify(target)
        after = (target.stat().st_mtime_ns, target.stat().st_size)
        assert after == before
