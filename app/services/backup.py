"""BackupService (ARCHITECTURE §3.2, NFR-3): периодические снапшоты БД.

Зачем: единственные данные сервиса — один файл SQLite; сценарии питания и
повреждение WAL переживаются только свежей копией. Снапшот делается штатным
SQLite backup API (`Connection.backup`) — **онлайн**: читает согласованное
состояние БД по WAL, не останавливая бытовые чтения/записи (сокет заблокирует
на миллисекунды, длинных блокировок нет).

Параметры (REQUIREMENTS §8): `BACKUP_DIR` (каталог копий, создаётся при
снапшоте), `BACKUP_INTERVAL_SEC` (интервал петли; первый снапшот — сразу
после старта), `BACKUP_KEEP` (число хранимых копий; ротация после каждого
снапшота), `BACKUP_VERIFY_EVERY` (проверка каждого N-го снапшота; 0 — только
вручную, `POST /backups/verify`), `BACKUP_SHIP_CMD` (операторская команда
выноса проверенной копии за пределы сервера; пусто — вынос выключен) и
`BACKUP_SHIP_TIMEOUT_SEC` (таймаут команды выноса, с).

Файл снапшота: `notes-YYYYmmddTHHMMSSZ.db` (UTC — лексикографический порядок
имён совпадает с хронологией; vec0-таблица требует sqlite-vec на открытии —
обычный читатель видит `notes` без расширения).

№14 (релиз 3.2.0, arch §5.1/§6.1/§13.1-13.3): сервис умеет **проверять
восстановление** снапшота (`verify`): открывает копию read-only с загруженным
sqlite-vec, гоняет `PRAGMA integrity_check`, сверяет наличие обязательных
таблиц (список — из реестров `app.storage`, не хардкод) и доступность каждой
vec0 (`SELECT 1 ... LIMIT 1`); расхождение `meta.embedding_model` — замечание,
а не отказ. Негодный снапшот **помечается** сайдкаром `<file>.bad`, а не
удаляется молча (FR-2.3); при удачной повторной проверке маркер снимается.
`status()` — наблюдаемость сканом каталога (без открытия БД), исходы проверки
переживают рестарт в сайдкар-индексе `backups-index.json`.

Отказы: ошибка снапшота не убивает процесс — событие `backup_failed` в
логе (NFR-4), петля продолжит по интервалу. Отказ проверки петлю не роняет
(NFR-3). Петля запускается lifespan-ом отдельной asyncio-таской; тяжёлая
часть (I/O) — в `asyncio.to_thread`.

Вынос копии (`ship`, no14-02, arch §6.1/§13.4): после успешной проверки
операторская команда `BACKUP_SHIP_CMD` уносит копию за пределы сервера;
назначение и учётки живут ВНЕ репозитория и образа — сервис передаёт команде
только путь к файлу (`LSB_BACKUP_FILE`/`LSB_BACKUP_NAME`). Успех — только
`returncode == 0` И отчёт `size=<N>` в stdout, равный размеру файла; иначе
отказ с `reason` и событием. Команда и её вывод в логи НЕ попадают (никаких
секретов). Отказ выноса петлю не роняет и локальную ротацию не ломает
(FR-3.2).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings
from app.services.namespaces import NamespaceService
from app.storage import area_vectors, chunks, db, vectors

# Коллекция снапшотов внутри BACKUP_DIR (имя = фиксированный префикс + UTC).
IMAGE_PREFIX = "notes-"
IMAGE_SUFFIX = ".db"
SNAPSHOT_RE = re.compile(rf"^{IMAGE_PREFIX}\d{{8}}T\d{{6}}Z{IMAGE_SUFFIX}$")

# №14 (arch §5.1/§6.1): артефакты наблюдаемости рядом со снапшотами.
VERIFY_MARKER_SUFFIX = ".bad"  # сайдкар негодного снапшота
STATE_FILE_NAME = "backups-index.json"  # сайдкар-индекс последних исходов
# Значения `reason` для verify (дословно EN, arch §6.1).
VERIFY_REASONS = (
    "not_openable",
    "integrity_failed",
    "missing_tables",
    "vec_unavailable",
)
# Мягкое замечание (не делает ok=false): копия снята до смены модели
# эмбеддинга — валидная база, при открытии просто переиндексируется (§13.1).
EMBEDDING_MISMATCH = "embedding_mismatch"

# №14 (arch §6.1/§13.4): контракт выноса копии за пределы сервера.
MIN_SHIP_REPORT = "size="  # отчёт команды выноса: `size=<N>`, N = размер копии
# Значения `reason` для ship (дословно EN, arch §6.1).
SHIP_REASONS = ("disabled", "command_failed", "size_mismatch", "timeout")

_STAMP_FMT = "%Y%m%dT%H%M%SZ"
_LATENCY = time.perf_counter


def _required_tables() -> tuple[str, ...]:
    """Обязательные таблицы снапшота — из реестров `app.storage` (не хардкод).

    Ядро — `db.CORE_TABLES`; плюс оба векторных индекса заметок
    (`vectors.VEC_TABLE`, `chunks.CHUNKS_VEC_TABLE`) и area-vec (включая
    vec-таблицу логов) — из `area_vectors.VEC_TABLES`. Список собирается
    динамически: добавление области/индекса в реестр автоматически расширяет
    проверку (arch §6.1/§13.1).
    """
    return (
        db.CORE_TABLES
        + (vectors.VEC_TABLE, chunks.CHUNKS_VEC_TABLE)
        + tuple(table for table, _id_column in area_vectors.VEC_TABLES)
    )


def _vec_tables() -> tuple[str, ...]:
    """vec0-таблицы, доступность которых проверяется запросом (arch §13.1)."""
    return (vectors.VEC_TABLE, chunks.CHUNKS_VEC_TABLE) + tuple(
        table for table, _id_column in area_vectors.VEC_TABLES
    )


def _now_iso() -> str:
    """Метка времени артефактов (UTC, как в именах снапшотов)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _created_at(name: str) -> str | None:
    """Время создания из имени снапшота (сортируемо, без чтения файла)."""
    match = SNAPSHOT_RE.match(name)
    if match is None:
        return None
    moment = datetime.strptime(name[len(IMAGE_PREFIX):-len(IMAGE_SUFFIX)], _STAMP_FMT)
    return moment.replace(tzinfo=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ship_size(stdout: str) -> int | None:
    """Размер копии из отчёта команды выноса (`size=<N>`, первое вхождение)."""
    match = re.search(rf"{re.escape(MIN_SHIP_REPORT)}(\d+)", stdout)
    return int(match.group(1)) if match is not None else None


class BackupService:
    """Периодический снапшот + ротация; петля — asyncio-таска в lifespan.

    Фаза 10 (Шаг 6): после каждого снапшота — груминг реестра неймспейсов
    (`NamespaceService.groom`): пустые provisional-листы чистятся, мелкие
    и пустые confirmed — сигнал оператору в логах. Ритм — раз в сутки,
    штатный интервал backup-петли (бриф Ф10 §3 «груминг в backup-цикл»);
    отдельный сервис не нужен — структурная автоматика редка.
    """

    def __init__(self, settings: Settings, namespaces: NamespaceService | None = None) -> None:
        self._settings = settings
        # Реестр неймспейсов для груминга (Фаза 10): общий экземпляр из
        # build_services; None — создать свой (тесты/юниты).
        self._namespaces = namespaces if namespaces is not None else NamespaceService(settings)
        self._stopping = False
        self._verify_counter = 0  # №14: счётчик успешных снапшотов в петле

    def stop(self) -> None:
        """Мягкая остановка: петля не начнёт новый снапшот."""
        self._stopping = True

    # --- синхронная часть (зывается из to_thread) ----------------------------

    def snapshot(self) -> Path:
        """Онлайн-снапшот БД (SQLite backup API) + ротация; путь нового файла.

        При отказе копирования частичный target-файл убирается — в ротации
        остаются только целые снапшоты (битый файл не занимает слот).

        Raises:
            sqlite3.Error / OSError: пробрасываются — интерпретация и лог
            на стороне петли (`run`) или теста.
        """
        started = _LATENCY()
        backup_dir = Path(self._settings.backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        target = backup_dir / _snapshot_name()
        try:
            self._copy(self._settings.db_path, target)
        except (sqlite3.Error, OSError):
            target.unlink(missing_ok=True)
            raise
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
        в каталог что-то своё. №14 (FR-2.3): снапшоты с сайдкаром `<file>.bad`
        исключаются из кандидатов — негодный не вытесняет годные, решение о
        его удалении принимает человек.
        """
        backups = sorted(self._backup_dir().glob(f"{IMAGE_PREFIX}*{IMAGE_SUFFIX}"))
        candidates = [path for path in backups if not self._marker_path(path).exists()]
        excess = candidates[: max(len(candidates) - self._settings.backup_keep, 0)]
        for path in excess:
            path.unlink()
        return excess

    def _backup_dir(self) -> Path:
        return Path(self._settings.backup_dir)

    def _snapshots(self) -> list[Path]:
        """Снапшоты каталога по имени (сортировка = хронология); без открытия БД."""
        backup_dir = self._backup_dir()
        if not backup_dir.is_dir():
            return []
        return sorted(
            path
            for path in backup_dir.glob(f"{IMAGE_PREFIX}*{IMAGE_SUFFIX}")
            if SNAPSHOT_RE.match(path.name)
        )

    def newest_snapshot(self) -> Path | None:
        """Свежайший по имени снапшот или None (пустой каталог)."""
        snapshots = self._snapshots()
        return snapshots[-1] if snapshots else None

    def resolve(self, name: str) -> Path | None:
        """Снапшот каталога по имени файла (без выхода за BACKUP_DIR) или None.

        Имя приводится к базовому элементу (`Path(name).name`) — операторская
        ручка не даёт прочитать произвольный файл хоста.
        """
        candidate = self._backup_dir() / Path(name).name
        if SNAPSHOT_RE.match(candidate.name) and candidate.is_file():
            return candidate
        return None

    def groom(self) -> dict[str, Any]:
        """Груминг реестра после снапшота (Фаза 10, Шаг 6): NamespaceService.

        Отдельный метод — точка DI и теста: петля зовёт его после снапшота,
        отказ пробрасывается (интерпретация — на стороне run).
        """
        return self._namespaces.groom()

    # --- проверка восстановления (№14, arch §6.1) ---------------------------

    def verify(self, path: Path | None = None) -> dict[str, Any]:
        """Проверить снапшот (read-only) и пометить негодный (arch §6.1/§13.1).

        `path=None` → свежайший по имени снапшот. Открывает копию read-only с
        загруженным sqlite-vec, гоняет `PRAGMA integrity_check`, сверяет
        обязательные таблицы (реестры `app.storage`) и доступность каждой
        vec0; `meta.embedding_model` — замечание, не отказ. `ok=false` →
        сайдкар `<file>.bad` (не удаляем!), `ok=true` → маркер снимается.
        Отсутствие файла — `{"ok": False, "reason": "not_openable"}` без падения.

        Returns:
            `{"file", "ok", "reason"? , "checks": {...}, "latency_ms"}`.
        """
        started = _LATENCY()
        if path is None:
            path = self.newest_snapshot()
        if path is None or not Path(path).is_file():
            return {
                "file": None if path is None else Path(path).name,
                "ok": False,
                "reason": "not_openable",
                "checks": {},
                "latency_ms": round((_LATENCY() - started) * 1000, 1),
            }
        path = Path(path)
        result = self._inspect(path)
        latency_ms = round((_LATENCY() - started) * 1000, 1)
        result["latency_ms"] = latency_ms
        if result["ok"]:
            self._marker_path(path).unlink(missing_ok=True)
        else:
            self._write_marker(path, result["reason"], result["checks"])
        self._record_verify(path, result["ok"], result.get("reason"))
        extra: dict[str, Any] = {
            "event": "backup_verify_ok" if result["ok"] else "backup_verify_failed",
            "file": path.name,
            "checks": result["checks"],
            "latency_ms": latency_ms,
        }
        if not result["ok"]:
            extra["reason"] = result["reason"]
        logging.getLogger("app").log(
            logging.INFO if result["ok"] else logging.WARNING,
            "backup snapshot verified" if result["ok"] else "backup snapshot failed verification",
            extra=extra,
        )
        return result

    def _inspect(self, path: Path) -> dict[str, Any]:
        """Разобрать снапшот: integrity → таблицы → vec0 → meta (read-only)."""
        checks: dict[str, Any] = {}
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        except sqlite3.Error:
            return {"file": path.name, "ok": False, "reason": "not_openable", "checks": checks}
        try:
            conn.row_factory = sqlite3.Row
            try:
                db._load_vec_extension(conn)
            except db.StorageError:
                return {
                    "file": path.name,
                    "ok": False,
                    "reason": "not_openable",
                    "checks": checks,
                }
            try:
                row = conn.execute("PRAGMA integrity_check").fetchone()
                integrity = str(row[0]) if row is not None else "unknown"
            except sqlite3.DatabaseError:
                checks["integrity"] = "failed"
                return {
                    "file": path.name,
                    "ok": False,
                    "reason": "integrity_failed",
                    "checks": checks,
                }
            checks["integrity"] = integrity
            if integrity != "ok":
                return {
                    "file": path.name,
                    "ok": False,
                    "reason": "integrity_failed",
                    "checks": checks,
                }
            names = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                )
            }
            missing = [table for table in _required_tables() if table not in names]
            checks["missing_tables"] = missing
            if missing:
                return {
                    "file": path.name,
                    "ok": False,
                    "reason": "missing_tables",
                    "checks": checks,
                }
            unavailable: list[str] = []
            for table in _vec_tables():
                try:
                    conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
                except sqlite3.Error:
                    unavailable.append(table)
            checks["vec_unavailable"] = unavailable
            if unavailable:
                return {
                    "file": path.name,
                    "ok": False,
                    "reason": "vec_unavailable",
                    "checks": checks,
                }
            checks[EMBEDDING_MISMATCH] = self._embedding_mismatch(conn)
            return {"file": path.name, "ok": True, "checks": checks}
        finally:
            conn.close()

    def _embedding_mismatch(self, conn: sqlite3.Connection) -> bool:
        """`meta.embedding_model` разошёлся с env — замечание (не отказ, §13.1)."""
        try:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'embedding_model'"
            ).fetchone()
        except sqlite3.Error:
            return False
        return row is not None and str(row[0]) != self._settings.embedding_model

    def _marker_path(self, path: Path) -> Path:
        """Сайдкар негодного снапшота: `<file>.bad`."""
        return path.with_name(path.name + VERIFY_MARKER_SUFFIX)

    def _write_marker(self, path: Path, reason: str, checks: dict[str, Any]) -> None:
        """Записать сайдкар негодного снапшота (`reason`/`at`/`checks`)."""
        payload = {"reason": reason, "at": _now_iso(), "checks": checks}
        self._marker_path(path).write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )

    # --- вынос копии за пределы сервера (№14, arch §6.1/§13.4) ---------------

    def ship(self, path: Path) -> dict[str, Any]:
        """Вынести проверенную копию снапшота за пределы сервера (FR-3.1/FR-3.2).

        Команда — операторская (`BACKUP_SHIP_CMD`); назначение и учётки живут
        ВНЕ репозитория и образа, сервис передаёт команде только путь к файлу в
        `LSB_BACKUP_FILE` (+ имя в `LSB_BACKUP_NAME`) и запускает команду через
        `shell` с таймаутом `BACKUP_SHIP_TIMEOUT_SEC`. Пустая команда — тихий
        no-op: `{\"ok\": None, \"skipped\": True, \"reason\": \"disabled\"}`; событие НЕ
        пишется и сайдкар-индекс не трогается.

        Успех — только `returncode == 0` И отчёт `size=<N>` в stdout, равный
        размеру локального файла; иначе `reason` ∈ `SHIP_REASONS`
        (`command_failed` — ненулевой код, нет отчёта или файл исчез;
        `size_mismatch` — отчёт есть, размер другой; `timeout` — таймаут).
        Исход пишется в сайдкар-индекс (`last_ship`) и событием
        `backup_shipped`/`backup_ship_failed`. Значения команды и её вывода в
        логи НЕ попадают (никаких секретов); отказ выноса локальные снапшоты и
        ротацию не трогает (FR-3.2).

        Returns:
            `{\"file\", \"ok\", \"size_bytes\"?, \"reason\"?, \"latency_ms\"}`; для
            выключенного выноса — `{\"file\", \"ok\": None, \"skipped\": True,
            \"reason\": \"disabled\"}`.
        """
        started = _LATENCY()
        target = Path(path)
        command = self._settings.backup_ship_cmd
        if not command.strip():
            return {
                "file": target.name,
                "ok": None,
                "skipped": True,
                "reason": "disabled",
            }
        try:
            local_size = target.stat().st_size
        except OSError:
            # Локальной копии нет — подтвердить вынос нечем; команду не зовём.
            return self._finish_ship(
                target, ok=False, size_bytes=None, reason="command_failed", started=started
            )
        env = {
            **os.environ,
            "LSB_BACKUP_FILE": str(target),
            "LSB_BACKUP_NAME": target.name,
        }
        try:
            completed = subprocess.run(
                command,
                shell=True,
                env=env,
                capture_output=True,
                text=True,
                timeout=self._settings.backup_ship_timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return self._finish_ship(
                target, ok=False, size_bytes=local_size, reason="timeout", started=started
            )
        except OSError:
            # Команда не запустилась (нет оболочки/прав) — как отказ команды.
            return self._finish_ship(
                target, ok=False, size_bytes=local_size, reason="command_failed", started=started
            )
        if completed.returncode != 0:
            return self._finish_ship(
                target, ok=False, size_bytes=local_size, reason="command_failed", started=started
            )
        reported = _parse_ship_size(completed.stdout)
        if reported is None:
            # Код 0, но контрактного отчёта `size=<N>` нет — вынос не подтверждён.
            return self._finish_ship(
                target, ok=False, size_bytes=local_size, reason="command_failed", started=started
            )
        if reported != local_size:
            return self._finish_ship(
                target, ok=False, size_bytes=local_size, reason="size_mismatch", started=started
            )
        return self._finish_ship(
            target, ok=True, size_bytes=local_size, reason=None, started=started
        )

    def _finish_ship(
        self,
        path: Path,
        *,
        ok: bool,
        size_bytes: int | None,
        reason: str | None,
        started: float,
    ) -> dict[str, Any]:
        """Записать исход выноса в сайдкар-индекс и журнал (без секретов)."""
        latency_ms = round((_LATENCY() - started) * 1000, 1)
        result: dict[str, Any] = {"file": path.name, "ok": ok, "latency_ms": latency_ms}
        if size_bytes is not None:
            result["size_bytes"] = size_bytes
        if reason is not None:
            result["reason"] = reason
        self._record_ship(path, result)
        extra: dict[str, Any] = {
            "event": "backup_shipped" if ok else "backup_ship_failed",
            "file": path.name,
            "latency_ms": latency_ms,
        }
        if size_bytes is not None:
            extra["size_bytes"] = size_bytes
        if reason is not None:
            extra["reason"] = reason
        logging.getLogger("app").log(
            logging.INFO if ok else logging.WARNING,
            "backup snapshot shipped" if ok else "backup snapshot ship failed",
            extra=extra,
        )
        return result

    # --- сайдкар-индекс (наблюдаемость переживает рестарт, arch §5.1) --------

    def _state_path(self) -> Path:
        return self._backup_dir() / STATE_FILE_NAME

    def _read_state(self) -> dict[str, Any]:
        """Прочитать сайдкар-индекс; нет файла/мусор → пустой словарь."""
        try:
            raw = self._state_path().read_text(encoding="utf-8")
            state = json.loads(raw)
        except (OSError, ValueError):
            return {}
        return state if isinstance(state, dict) else {}

    def _write_state(self, state: dict[str, Any]) -> None:
        """Записать сайдкар-индекс (атомарность не критична: наблюдаемость)."""
        try:
            self._backup_dir().mkdir(parents=True, exist_ok=True)
            self._state_path().write_text(
                json.dumps(state, sort_keys=True), encoding="utf-8"
            )
        except OSError:
            logging.getLogger("app").exception(
                "backup state write failed", extra={"event": "backup_state_failed"}
            )

    def _record_verify(self, path: Path, ok: bool, reason: str | None) -> None:
        """Обновить исход проверки в сайдкар-индексе (last_verify + по файлам)."""
        record: dict[str, Any] = {
            "at": _now_iso(),
            "file": path.name,
            "result": "ok" if ok else "failed",
        }
        if not ok and reason:
            record["reason"] = reason
        state = self._read_state()
        state["last_verify"] = record
        verified = state.get("verified")
        if not isinstance(verified, dict):
            verified = {}
            state["verified"] = verified
        verified[path.name] = record
        self._write_state(state)

    def _record_ship(self, path: Path, result: dict[str, Any]) -> None:
        """Обновить исход выноса в сайдкар-индексе (`last_ship` + `shipped`).

        Метка `shipped` ставится только при подтверждённом выносе; неудача уже
        подтверждённую копию не разметает — файл снапшота неизменяем, а
        наблюдаемость выноса живёт отдельной записью.
        """
        record: dict[str, Any] = {
            "at": _now_iso(),
            "file": path.name,
            "result": "ok" if result["ok"] else "failed",
        }
        if result.get("reason"):
            record["reason"] = result["reason"]
        if "size_bytes" in result:
            record["size_bytes"] = result["size_bytes"]
        state = self._read_state()
        state["last_ship"] = record
        shipped = state.get("shipped")
        if not isinstance(shipped, dict):
            shipped = {}
            state["shipped"] = shipped
        if result["ok"]:
            shipped[path.name] = record
        self._write_state(state)

    # --- наблюдаемость ------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Снимок каталога бэкапов (скан, БЕЗ открытия БД) + сайдкар-индекс.

        Значения — как в `/health.backups` (arch §7.2): `count`,
        `newest_age_sec`, `oldest_age_sec`, `total_bytes`, `bad` (число
        помеченных) и последние исходы `last_verify`/`last_ship` из
        `backups-index.json` (нет файла → None).
        """
        snapshots = self._snapshots()
        now = time.time()
        sized: list[tuple[Path, float, int]] = []
        for path in snapshots:
            try:
                info = path.stat()
            except OSError:
                continue  # файл исчез между glob и stat (гонка с ротацией)
            sized.append((path, info.st_mtime, info.st_size))
        ages = sorted(now - mtime for _path, mtime, _size in sized)
        state = self._read_state()
        return {
            "count": len(sized),
            "newest_age_sec": int(min(ages)) if ages else None,
            "oldest_age_sec": int(max(ages)) if ages else None,
            "total_bytes": sum(size for _path, _mtime, size in sized),
            "bad": sum(1 for path, _mtime, _size in sized if self._marker_path(path).exists()),
            "last_verify": state.get("last_verify"),
            "last_ship": state.get("last_ship"),
        }

    def list_snapshots(self) -> list[dict[str, Any]]:
        """Детальный список снапшотов (операторский `GET /backups`, arch §7.2).

        Свежие первыми; `verified` ∈ ok/failed/unknown (маркер `.bad` — явный
        `failed`), `bad` — есть сайдкар, `shipped` — по сайдкар-индексу
        (подтверждённый вынос копии за пределы сервера).
        """
        state = self._read_state()
        verified_map = state.get("verified")
        verified_map = verified_map if isinstance(verified_map, dict) else {}
        shipped_map = state.get("shipped")
        shipped_map = shipped_map if isinstance(shipped_map, dict) else {}
        items: list[dict[str, Any]] = []
        for path in reversed(self._snapshots()):  # свежие первыми
            try:
                size = path.stat().st_size
            except OSError:
                continue  # файл исчез между glob и stat (гонка с ротацией)
            bad = self._marker_path(path).exists()
            if bad:
                verified_state = "failed"
            else:
                record = verified_map.get(path.name)
                verified_state = (
                    str(record.get("result", "unknown"))
                    if isinstance(record, dict)
                    else "unknown"
                )
            items.append(
                {
                    "file": path.name,
                    "size_bytes": size,
                    "created_at": _created_at(path.name),
                    "verified": verified_state,
                    "bad": bad,
                    "shipped": path.name in shipped_map,
                }
            )
        return items

    # --- Петля ---------------------------------------------------------------

    async def run(self) -> None:
        """Снапшот сразу после старта, далее — раз в BACKUP_INTERVAL_SEC.

        №14 (arch §6.1): после успешного снапшота, если `BACKUP_VERIFY_EVERY`
        > 0 и счётчик дошёл до кратного значения, — проверка восстановления
        (`verify`); её отказ петлю не роняет (NFR-3), исход — событием.

        №14 (no14-02, arch §6.1): если проверка прошла (`verify.ok=True`) и задана
        операторская команда выноса (`BACKUP_SHIP_CMD`) — копия уносится за
        пределы сервера (`ship`, FR-3.1); отказ выноса петлю не роняет и
        локальную ротацию не ломает (FR-3.2).

        Фаза 10: после каждого снапшота — груминг реестра неймспейсов;
        отказ груминга не убивает петлю (как отказ снапшота, NFR-3):
        событие groom_failed в логе, повтор — следующий интервал.
        """
        while not self._stopping:
            target: Path | None = None
            try:
                target = await asyncio.to_thread(self.snapshot)
            except (sqlite3.Error, OSError):
                # БД/каталог недоступны — процесс должен жить (NFR-3);
                # подробности — в JSON-лог, повтор по интервалу.
                logging.getLogger("app").exception(
                    "backup snapshot failed", extra={"event": "backup_failed"}
                )
            if target is not None:
                verified = await self._maybe_verify()
                if verified and self._settings.backup_ship_cmd.strip():
                    await self._maybe_ship(target)
            try:
                report = await asyncio.to_thread(self.groom)
                if any(report.values()):
                    logging.getLogger("app").info(
                        "backup: namespace groom finished",
                        extra={"event": "groom_run", "report": report},
                    )
            except Exception:
                logging.getLogger("app").exception(
                    "namespace groom failed", extra={"event": "groom_failed"}
                )
            await asyncio.sleep(self._settings.backup_interval_sec)

    async def _maybe_verify(self) -> bool:
        """Проверить снапшот по расписанию `BACKUP_VERIFY_EVERY` (0 — вручную).

        Отказ проверки не роняет петлю: событие `backup_verify_failed`
        (NFR-3). Счётчик — в памяти процесса (после рестарта начинается заново
        — проверка редкая и дешёвая, состояние на диск не выносим).

        Returns:
            True — проверка выполнена и снапшот годен (`ok=True`); иначе False
            (проверка выключена, не пришла очередь или отказ).
        """
        every = self._settings.backup_verify_every
        if every <= 0:
            return False
        self._verify_counter += 1
        if self._verify_counter % every != 0:
            return False
        try:
            result = await asyncio.to_thread(self.verify)
        except Exception:
            logging.getLogger("app").exception(
                "backup verify failed", extra={"event": "backup_verify_failed"}
            )
            return False
        return bool(result.get("ok"))

    async def _maybe_ship(self, target: Path) -> None:
        """Вынести копию после успешной проверки (FR-3.1/FR-3.2).

        Отказ выноса петлю не роняет (NFR-3): исход — событие
        `backup_ship_failed`, локальная ротация работает как прежде.
        """
        try:
            await asyncio.to_thread(self.ship, target)
        except Exception:
            logging.getLogger("app").exception(
                "backup ship failed", extra={"event": "backup_ship_failed"}
            )


def _snapshot_name(moment: datetime | None = None) -> str:
    """Имя файла снапшота: notes-<UTC до секунды>.db (сортируемо)."""
    moment = moment or datetime.now(UTC)
    return f"{IMAGE_PREFIX}{moment.strftime('%Y%m%dT%H%M%SZ')}{IMAGE_SUFFIX}"
