"""BackupService (ARCHITECTURE §3.2, NFR-3): периодические снапшоты БД.

Зачем: единственные данные сервиса — один файл SQLite; сценарии питания и
повреждение WAL переживаются только свежей копией. Снапшот делается штатным
SQLite backup API (`Connection.backup`) — **онлайн**: читает согласованное
состояние БД по WAL, не останавливая бытовые чтения/записи (сокет заблокирует
на миллисекунды, длинных блокировок нет).

Параметры (REQUIREMENTS §8): `BACKUP_DIR` (каталог копий, создаётся при
снапшоте), `BACKUP_INTERVAL_SEC` (интервал петли; первый снапшот — сразу
после старта), `BACKUP_KEEP` (число хранимых копий; ротация после каждого
снапшота).

Файл снапшота: `notes-YYYYmmddTHHMMSSZ.db` (UTC — лексикографический порядок
имен совпадает с хронологией; vec0-таблица требует sqlite-vec на открытии —
обычный читатель видит `notes` без расширения).

Отказы: ошибка снапшота не убивает процесс — событие `backup_failed` в
логе (NFR-4), петля продолжит по интервалу. Петля запускается lifespan-ом
отдельной asyncio-таской; тяжёлая часть (I/O) — в `asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings

# Коллекция снапшотов внутри BACKUP_DIR (имя = фиксированный префикс + UTC).
IMAGE_PREFIX = "notes-"
IMAGE_SUFFIX = ".db"
SNAPSHOT_RE = re.compile(rf"^{IMAGE_PREFIX}\d{{8}}T\d{{6}}Z{IMAGE_SUFFIX}$")

_LATENCY = time.perf_counter


class BackupService:
    """Периодический снапшот + ротация; петля — asyncio-таска в lifespan."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._stopping = False

    def stop(self) -> None:
        """Мягкая остановка: петля не начнёт новый снапшот."""
        self._stopping = True

    # --- синхронная часть (зывается из to_thread) ----------------------------

    def snapshot(self) -> Path:
        """Онлайн-снапшот БД (SQLite backup API) + ротация; путь нового файла.

        Raises:
            sqlite3.Error / OSError: пробрасываются — интерпретация и лог
            на стороне петли (`run`) или теста.
        """
        started = _LATENCY()
        backup_dir = Path(self._settings.backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        target = backup_dir / _snapshot_name()
        self._copy(self._settings.db_path, target)
        size = target.stat().st_size
        removed = self.rotate()
        logging.getLogger("app").info(
            "backup snapshot created",
            extra={
                "event": "backup_created",
                "file": target.name,
                "size_bytes": size,
                "removed": len(removed),
                "latency_ms": round((_LATENCY() - started) * 1000, 1),
            },
        )
        return target

    @staticmethod
    def _copy(source_path: str, target_path: Path) -> None:
        """SQLite backup API: копия консистентного состояния без остановки."""
        source = sqlite3.connect(source_path)
        try:
            dest = sqlite3.connect(target_path)
            try:
                source.backup(dest)
            finally:
                dest.close()
        finally:
            source.close()

    def rotate(self) -> list[Path]:
        """Оставить BACKUP_KEEP свежайших снапшотов; вернуть удалённые.

        Порядок — по имени (UTC-таймштамп в имени, лексикография = время).
        Сторонние файлы (не `notes-*.db`) не трогаем — оператор мог положить
        в каталог что-то своё.
        """
        backups = sorted(self._backup_dir().glob(f"{IMAGE_PREFIX}*{IMAGE_SUFFIX}"))
        excess = backups[: max(len(backups) - self._settings.backup_keep, 0)]
        for path in excess:
            path.unlink()
        return excess

    def _backup_dir(self) -> Path:
        return Path(self._settings.backup_dir)

    # --- Петля ---------------------------------------------------------------

    async def run(self) -> None:
        """Снапшот сразу после старта, далее — раз в BACKUP_INTERVAL_SEC."""
        while not self._stopping:
            try:
                await asyncio.to_thread(self.snapshot)
            except (sqlite3.Error, OSError):
                # БД/каталог недоступны — процесс должен жить (NFR-3);
                # подробности — в JSON-лог, повтор по интервалу.
                logging.getLogger("app").exception(
                    "backup snapshot failed", extra={"event": "backup_failed"}
                )
            await asyncio.sleep(self._settings.backup_interval_sec)


def _snapshot_name(moment: datetime | None = None) -> str:
    """Имя файла снапшота: notes-<UTC до секунды>.db (сортируемо)."""
    moment = moment or datetime.now(UTC)
    return f"{IMAGE_PREFIX}{moment.strftime('%Y%m%dT%H%M%SZ')}{IMAGE_SUFFIX}"