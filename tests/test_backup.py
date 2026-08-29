"""BackupService (Фаза 5, NFR-3): онлайн-снапшоты, ротация, петля в lifespan.

Юнит: копия консистентна (обычный sqlite3 без расширений читает `notes`),
имя сортируемо (UTC до секунды), каталог создаётся автоматически, ротация
держит BACKUP_KEEP свежайших и не трогает чужие файлы, событие в JSON-логе.
Асинхронная петля — снимает по интервалу до stop(). E2E: приложение
(TestClient) снапшотит сразу после старта и далее по BACKUP_INTERVAL_SEC,
ротация по BACKUP_KEEP; отказ BACKUP_DIR не убивает сервис (NFR-3) —
событие `backup_failed` в логе, /health жив.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app
from app.services.backup import (
    IMAGE_PREFIX,
    IMAGE_SUFFIX,
    BackupService,
)
from app.storage.db import init_db

SNAPSHOT_PATTERN = rf"^{IMAGE_PREFIX}\d{{8}}T\d{{6}}Z{IMAGE_SUFFIX}$"


def _seed_notes(settings: Settings, count: int) -> None:
    """Прямые INSERT в notes (без сервисов: чистый юнит копии БД)."""
    conn = sqlite3.connect(settings.db_path)
    try:
        for index in range(count):
            conn.execute(
                "INSERT INTO notes(text) VALUES (?)",
                (f"заметка backup №{index} — копия исходной БД",),
            )
        conn.commit()
    finally:
        conn.close()


def _count_active(path: Path) -> int:
    """Число активных заметок в произвольном SQLite-файле (без vec-расширения)."""
    conn = sqlite3.connect(path)
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM notes WHERE deleted_at IS NULL"
            ).fetchone()[0]
        )
    finally:
        conn.close()


def _with_event(caplog: pytest.LogCaptureFixture, event: str) -> list[Any]:
    """Записи caplog с данным event (чужие записи без extra не терпят AttributeError)."""
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) == event
    ]


@pytest.fixture
def backup_service(tmp_path: Path):
    """BackupService поверх инициализированной БД; копии — в tmp/backups (создан)."""
    settings = get_settings()
    init_db(settings)
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(parents=True)  # в юнитах ротации файлы пишутся ДО snapshot
    adjusted = settings.model_copy(update={"backup_dir": str(backup_dir)})
    return BackupService(adjusted), backup_dir


class TestSnapshot:
    def test_creates_named_utc_file(
        self, backup_service: tuple[BackupService, Path]
    ) -> None:
        service, backup_dir = backup_service
        target = service.snapshot()
        assert target.exists()
        assert re.match(SNAPSHOT_PATTERN, target.name)
        assert target.parent == backup_dir

    def test_copy_contains_active_notes(
        self, backup_service: tuple[BackupService, Path]
    ) -> None:
        service, _ = backup_service
        _seed_notes(get_settings(), 3)
        assert _count_active(service.snapshot()) == 3

    def test_copy_is_consistent_after_write(
        self, backup_service: tuple[BackupService, Path]
    ) -> None:
        """Запись ПОСЛЕ снапшота не меняет уже снятую копию."""
        service, _ = backup_service
        _seed_notes(get_settings(), 2)
        first = service.snapshot()
        _seed_notes(get_settings(), 1)
        assert _count_active(first) == 2

    def test_backup_dir_is_created_automatically(self, tmp_path: Path) -> None:
        settings = get_settings()
        deep = tmp_path / "deep" / "nested" / "backups"
        service = BackupService(settings.model_copy(update={"backup_dir": str(deep)}))
        assert service.snapshot().parent == deep
        assert deep.is_dir()

    def test_snapshot_event_is_logged(
        self,
        backup_service: tuple[BackupService, Path],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        service, _ = backup_service
        with caplog.at_level(logging.INFO, logger="app"):
            service.snapshot()
        record = _with_event(caplog, "backup_created")[-1]
        assert record.size_bytes > 0  # type: ignore[attr-defined]
        assert record.latency_ms >= 0  # type: ignore[attr-defined]
        assert record.removed >= 0  # type: ignore[attr-defined]


class TestRotation:
    def test_keeps_backup_keep_newest(
        self, backup_service: tuple[BackupService, Path]
    ) -> None:
        service, backup_dir = backup_service
        keep = service._settings.backup_keep
        # 10 «старых» файлов с именами раньше текущей даты: имя — UTC-таймштамп,
        # лексикография = хронология, ротация отсекает ровно старьё.
        for index in range(10):
            (backup_dir / f"{IMAGE_PREFIX}2026010{index}T000000Z{IMAGE_SUFFIX}").write_bytes(b"x")
        service.snapshot()
        files = sorted(backup_dir.glob(f"{IMAGE_PREFIX}*{IMAGE_SUFFIX}"))
        assert len(files) == keep  # 11 файлов → остаётся 7 (BACKUP_KEEP по умолчанию)
        # Свежайший — настоящий снапшот, не один из фейков.
        assert re.match(SNAPSHOT_PATTERN, files[-1].name)

    def test_ignores_foreign_files(
        self, backup_service: tuple[BackupService, Path]
    ) -> None:
        service, backup_dir = backup_service
        foreign = backup_dir / "operator-notes.txt"
        foreign.write_text("полка оператора", encoding="utf-8")
        service.snapshot()
        service.rotate()
        assert foreign.exists()


class TestLoop:
    @pytest.mark.asyncio
    async def test_run_takes_snapshots_until_stopped(
        self,
        backup_service: tuple[BackupService, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        service, backup_dir = backup_service
        patched = service._settings.model_copy(
            update={"backup_interval_sec": 0.05}
        )
        monkeypatch.setattr(service, "_settings", patched)
        task = asyncio.create_task(service.run())
        try:
            # Детерминированное ожидание: поллинг каталога, а не слепой sleep
            # (под нагрузкой полный прогона to_thread стартует не мгновенно).
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if len(list(backup_dir.glob(f"{IMAGE_PREFIX}*{IMAGE_SUFFIX}"))) >= 2:
                    break
                await asyncio.sleep(0.05)
        finally:
            service.stop()
            task.cancel()  # если петля спит на интервале — прерываем досрочно
            with contextlib.suppress(asyncio.CancelledError):
                await task
        snapshots = list(backup_dir.glob(f"{IMAGE_PREFIX}*{IMAGE_SUFFIX}"))
        assert len(snapshots) >= 2  # петля гарантированно сняла несколько


class TestAppIntegration:
    def test_app_snapshots_periodically_and_rotates(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Lifespan поднимает петлю: сразу снапшот + далее по интервалу."""
        backup_dir = tmp_path / "app-backups"
        monkeypatch.setenv("BACKUP_INTERVAL_SEC", "1")
        monkeypatch.setenv("BACKUP_KEEP", "2")
        monkeypatch.setenv("BACKUP_DIR", str(backup_dir))
        get_settings.cache_clear()
        try:
            app = create_app()
            with caplog.at_level(logging.INFO, logger="app"), TestClient(app):
                deadline = time.monotonic() + 8.0
                while time.monotonic() < deadline:
                    if len(list(backup_dir.glob(f"{IMAGE_PREFIX}*{IMAGE_SUFFIX}"))) >= 3:
                        break
                    time.sleep(0.2)
            files = list(backup_dir.glob(f"{IMAGE_PREFIX}*{IMAGE_SUFFIX}"))
            assert len(files) == 2  # ротация держит BACKUP_KEEP
            assert _with_event(caplog, "backup_created")
        finally:
            get_settings.cache_clear()

    def test_backup_failure_does_not_kill_app(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """BACKUP_DIR — файл вместо каталога: mkdir падает, сервис живёт."""
        blocked = tmp_path / "blocked-backups"
        blocked.write_text("не каталог", encoding="utf-8")
        monkeypatch.setenv("BACKUP_DIR", str(blocked))
        get_settings.cache_clear()
        try:
            with (
                caplog.at_level(logging.INFO, logger="app"),
                TestClient(create_app()) as client,
            ):
                response = client.get("/health")
            assert response.status_code == 200
            assert _with_event(caplog, "backup_failed")
        finally:
            get_settings.cache_clear()