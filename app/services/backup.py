"""BackupService (ARCHITECTURE §3.2, NFR-3): periodic DB snapshots.

    Why: the service's only data is a single SQLite file; power scenarios and WAL
    corruption survive only via a fresh copy. A snapshot uses the standard SQLite
    backup API (`Connection.backup`) — **online**: it reads a consistent DB state
    over WAL without stopping routine reads/writes (the socket blocks for
    milliseconds, no long locks).

    Parameters (REQUIREMENTS §8): `BACKUP_DIR` (the copies directory, created at
    snapshot time), `BACKUP_INTERVAL_SEC` (the loop interval; the first snapshot
    comes right after the start), `BACKUP_KEEP` (the number of stored copies;
    rotation after each snapshot), `BACKUP_VERIFY_EVERY` (verify every N-th
    snapshot; 0 — manual only, `POST /backups/verify`), `BACKUP_SHIP_CMD` (the
    operator command that ships a verified copy off the server; empty — shipping
    off) and `BACKUP_SHIP_TIMEOUT_SEC` (the ship command timeout, s).

    Snapshot file: `notes-YYYYmmddTHHMMSSZ.db` (UTC — the lexicographic name order
    matches the chronology; the vec0 table requires sqlite-vec on open — a plain
    reader sees `notes` without the extension).

    #14 (release 3.2.0, arch §5.1/§6.1/§13.1-13.3): the service can **verify the
    restore** of a snapshot (`verify`): opens the copy read-only with sqlite-vec
    loaded, runs `PRAGMA integrity_check`, checks the required tables (the list —
    from the `app.storage` registries, not hardcoded) and the reachability of each
    vec0 (`SELECT 1 ... LIMIT 1`); a `meta.embedding_model` mismatch is a remark,
    not a failure. A bad snapshot is **marked** with a `<file>.bad` sidecar, not
    silently deleted (FR-2.3); a successful re-check removes the marker.
    `status()` — observability by scanning the directory (no DB open), check
    outcomes survive a restart in the `backups-index.json` sidecar index.

    Failures: a snapshot error does not kill the process — a `backup_failed` event
    in the log (NFR-4), the loop continues on schedule. A verify failure does not
    crash the loop (NFR-3). The loop is started by lifespan as a separate asyncio
    task; the heavy part (I/O) goes through `asyncio.to_thread`.

    Shipping a copy (`ship`, no14-02, arch §6.1/§13.4): after a successful verify
    the operator command `BACKUP_SHIP_CMD` carries the copy off the server;
    the destination and credentials live OUTSIDE the repository and image — the
    service passes only the file path to the command (`LSB_BACKUP_FILE`/
    `LSB_BACKUP_NAME`). Success — only `returncode == 0` AND a `size=<N>` report
    on stdout equal to the file size; otherwise a failure with a `reason` and an
    event. The command and its output never reach the logs (no secrets). A ship
    failure does not crash the loop and does not break local rotation
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

# The snapshot collection inside BACKUP_DIR (name = fixed prefix + UTC).
IMAGE_PREFIX = "notes-"
IMAGE_SUFFIX = ".db"
SNAPSHOT_RE = re.compile(rf"^{IMAGE_PREFIX}\d{{8}}T\d{{6}}Z{IMAGE_SUFFIX}$")

# #14 (arch §5.1/§6.1): observability artifacts next to the snapshots.
VERIFY_MARKER_SUFFIX = ".bad"  # the bad-snapshot sidecar
STATE_FILE_NAME = "backups-index.json"  # the sidecar index of latest outcomes
# `reason` values for verify (verbatim EN, arch §6.1).
VERIFY_REASONS = (
    "not_openable",
    "integrity_failed",
    "missing_tables",
    "vec_unavailable",
)
# A soft remark (does not make ok=false): the copy was taken before the
# embedding model change — a valid base, simply reindexed on open (§13.1).
EMBEDDING_MISMATCH = "embedding_mismatch"

# #14 (arch §6.1/§13.4): the contract of shipping a copy off the server.
MIN_SHIP_REPORT = "size="  # the ship command's report: `size=<N>`, N = the copy size
# `reason` values for ship (verbatim EN, arch §6.1).
SHIP_REASONS = ("disabled", "command_failed", "size_mismatch", "timeout")

_STAMP_FMT = "%Y%m%dT%H%M%SZ"
_LATENCY = time.perf_counter


def _required_tables() -> tuple[str, ...]:
    """Required snapshot tables — from the `app.storage` registries (not hardcoded).

            The core — `db.CORE_TABLES`; plus both note vector indexes
            (`vectors.VEC_TABLE`, `chunks.CHUNKS_VEC_TABLE`) and area-vecs (including
            the logs vec table) — from `area_vectors.VEC_TABLES`. The list is assembled
            dynamically: adding an area/index to the registry extends the check
            automatically (arch §6.1/§13.1).
    """
    return (
        db.CORE_TABLES
        + (vectors.VEC_TABLE, chunks.CHUNKS_VEC_TABLE)
        + tuple(table for table, _id_column in area_vectors.VEC_TABLES)
    )


def _vec_tables() -> tuple[str, ...]:
    """vec0 tables whose reachability is checked by a query (arch §13.1)."""
    return (vectors.VEC_TABLE, chunks.CHUNKS_VEC_TABLE) + tuple(
        table for table, _id_column in area_vectors.VEC_TABLES
    )


def _now_iso() -> str:
    """Artifact timestamp (UTC, as in snapshot names)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _created_at(name: str) -> str | None:
    """Creation time parsed from the snapshot name (sortable, no file read)."""
    match = SNAPSHOT_RE.match(name)
    if match is None:
        return None
    moment = datetime.strptime(name[len(IMAGE_PREFIX):-len(IMAGE_SUFFIX)], _STAMP_FMT)
    return moment.replace(tzinfo=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ship_size(stdout: str) -> int | None:
    """The copy size from the ship command's report (`size=<N>`, first occurrence)."""
    match = re.search(rf"{re.escape(MIN_SHIP_REPORT)}(\d+)", stdout)
    return int(match.group(1)) if match is not None else None


class BackupService:
    """Periodic snapshot + rotation; the loop is an asyncio task in lifespan.

            Phase 10 (Step 6): after each snapshot — grooming of the namespace registry
            (`NamespaceService.groom`): empty provisional lists are cleaned, small and
            empty confirmed ones — a signal to the operator in the logs. The rhythm —
            once a day, the regular backup-loop interval (Ph10 brief §3 "grooming into
            the backup cycle"); a separate service is unnecessary — structural
            automation is rare.
    """

    def __init__(self, settings: Settings, namespaces: NamespaceService | None = None) -> None:
        self._settings = settings
        # The namespace registry for grooming (Phase 10): the shared instance from
        # build_services; None — create one's own (tests/units).
        self._namespaces = namespaces if namespaces is not None else NamespaceService(settings)
        self._stopping = False
        self._verify_counter = 0  # #14: the counter of successful snapshots in the loop

    def stop(self) -> None:
        """Soft stop: the loop will not start a new snapshot."""
        self._stopping = True

    # --- synchronous part (called from to_thread) -----------------------------

    def snapshot(self) -> Path:
        """Online DB snapshot (SQLite backup API) + rotation; the new file path.

                    On a copy failure the partial target file is removed — only whole
                    snapshots stay in rotation (a broken file takes no slot).

                    Raises:
                        sqlite3.Error / OSError: propagated — interpretation and logging
                        belong to the loop (`run`) or the test.
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
        """SQLite backup API: a copy of a consistent state without stopping."""
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
        """Keep the BACKUP_KEEP newest snapshots; return the removed ones.

                    Order — by name (a UTC timestamp in the name; lexicography = time).
                    Foreign files (not `notes-*.db`) are untouched — the operator may keep
                    something of their own in the directory. #14 (FR-2.3): snapshots with a
                    `<file>.bad` sidecar are excluded from candidates — a bad one does not
                    push out good ones; the removal decision belongs to a human.
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
        """Directory snapshots by name (sort = chronology); no DB open."""
        backup_dir = self._backup_dir()
        if not backup_dir.is_dir():
            return []
        return sorted(
            path
            for path in backup_dir.glob(f"{IMAGE_PREFIX}*{IMAGE_SUFFIX}")
            if SNAPSHOT_RE.match(path.name)
        )

    def newest_snapshot(self) -> Path | None:
        """The newest-by-name snapshot or None (empty directory)."""
        snapshots = self._snapshots()
        return snapshots[-1] if snapshots else None

    def resolve(self, name: str) -> Path | None:
        """A directory snapshot by file name (no escape from BACKUP_DIR) or None.

                    The name is reduced to its base element (`Path(name).name`) — the
                    operator handle cannot read an arbitrary host file.
        """
        candidate = self._backup_dir() / Path(name).name
        if SNAPSHOT_RE.match(candidate.name) and candidate.is_file():
            return candidate
        return None

    def groom(self) -> dict[str, Any]:
        """Registry grooming after a snapshot (Phase 10, Step 6): NamespaceService.

                    A separate method — the DI and test point: the loop calls it after the
                    snapshot; a failure is propagated (interpretation belongs to run).
        """
        return self._namespaces.groom()

    # --- restore verification (#14, arch §6.1) --------------------------------

    def verify(self, path: Path | None = None) -> dict[str, Any]:
        """Verify a snapshot (read-only) and mark a bad one (arch §6.1/§13.1).

                    `path=None` → the newest-by-name snapshot. Opens the copy read-only
                    with sqlite-vec loaded, runs `PRAGMA integrity_check`, checks the
                    required tables (the `app.storage` registries) and the reachability of
                    each vec0; `meta.embedding_model` — a remark, not a failure.
                    `ok=false` → the `<file>.bad` sidecar (never deleted!), `ok=true` →
                    the marker is removed. A missing file —
                    `{"ok": False, "reason": "not_openable"}` without a crash.

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
        """Inspect a snapshot: integrity → tables → vec0 → meta (read-only)."""
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
        """`meta.embedding_model` diverged from the effective model — a remark
        (not a failure, §13.1).

        Release 3.9.0 (LSBN-S-18): the comparison is with the EFFECTIVE
        embedding model — the one the vectors are actually built with (the
        `embed` operation in the `operation` mode, the slot value in `legacy`).
        The meta holds that same effective value (app/storage/db.py), so the
        remark tells the truth about the snapshot; in the `legacy` mode nothing
        changes (the slot value IS the effective one there).
        """
        try:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'embedding_model'"
            ).fetchone()
        except sqlite3.Error:
            return False
        if row is None:
            return False
        effective_model, _provider = db._effective_embedding_values(self._settings)
        return str(row[0]) != effective_model

    def _marker_path(self, path: Path) -> Path:
        """The bad-snapshot sidecar: `<file>.bad`."""
        return path.with_name(path.name + VERIFY_MARKER_SUFFIX)

    def _write_marker(self, path: Path, reason: str, checks: dict[str, Any]) -> None:
        """Write the bad-snapshot sidecar (`reason`/`at`/`checks`)."""
        payload = {"reason": reason, "at": _now_iso(), "checks": checks}
        self._marker_path(path).write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )

    # --- shipping a copy off the server (#14, arch §6.1/§13.4) ----------------

    def ship(self, path: Path) -> dict[str, Any]:
        """Ship a verified snapshot copy off the server (FR-3.1/FR-3.2).

                    The command is operator-owned (`BACKUP_SHIP_CMD`); the destination and
                    credentials live OUTSIDE the repository and image — the service passes
                    only the file path in `LSB_BACKUP_FILE` (+ the name in
                    `LSB_BACKUP_NAME`) and runs the command via `shell` with the
                    `BACKUP_SHIP_TIMEOUT_SEC` timeout. An empty command — a silent no-op:
                    `{"ok": None, "skipped": True, "reason": "disabled"}`; NO event is
                    written and the sidecar index is untouched.

                    Success — only `returncode == 0` AND a `size=<N>` report on stdout
                    equal to the local file size; otherwise `reason` ∈ `SHIP_REASONS`
                    (`command_failed` — a non-zero code, no report or the file vanished;
                    `size_mismatch` — a report exists, the size differs; `timeout`).
                    The outcome is written to the sidecar index (`last_ship`) and to the
                    `backup_shipped`/`backup_ship_failed` event. The command value and its
                    output never reach the logs (no secrets); a ship failure does not touch
                    local snapshots or rotation (FR-3.2).

                    Returns:
                        `{"file", "ok", "size_bytes"?, "reason"?, "latency_ms"}`; for
                        shipping off — `{"file", "ok": None, "skipped": True,
                        "reason": "disabled"}`.
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
            # No local copy — nothing to confirm the ship with; the command is not called.
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
            # The command did not start (no shell/permissions) — treated as a command failure.
            return self._finish_ship(
                target, ok=False, size_bytes=local_size, reason="command_failed", started=started
            )
        if completed.returncode != 0:
            return self._finish_ship(
                target, ok=False, size_bytes=local_size, reason="command_failed", started=started
            )
        reported = _parse_ship_size(completed.stdout)
        if reported is None:
            # Code 0, but no contract report `size=<N>` — the ship is not confirmed.
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
        """Write the ship outcome to the sidecar index and the journal (no secrets)."""
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

    # --- sidecar index (observability survives a restart, arch §5.1) ----------

    def _state_path(self) -> Path:
        return self._backup_dir() / STATE_FILE_NAME

    def _read_state(self) -> dict[str, Any]:
        """Read the sidecar index; no file/garbage → an empty dict."""
        try:
            raw = self._state_path().read_text(encoding="utf-8")
            state = json.loads(raw)
        except (OSError, ValueError):
            return {}
        return state if isinstance(state, dict) else {}

    def _write_state(self, state: dict[str, Any]) -> None:
        """Write the sidecar index (atomicity is not critical: observability)."""
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
        """Update the verify outcome in the sidecar index (last_verify + per file)."""
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
        """Update the ship outcome in the sidecar index (`last_ship` + `shipped`).

                    The `shipped` mark is set only for a confirmed ship; a failure does not
                    smudge an already confirmed copy — the snapshot file is immutable, and
                    ship observability lives in a separate record.
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

    # --- observability --------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """A snapshot of the backups directory (a scan, NO DB open) + the sidecar index.

                    Values — as in `/health.backups` (arch §7.2): `count`,
                    `newest_age_sec`, `oldest_age_sec`, `total_bytes`, `bad` (the number of
                    marked ones) and the latest `last_verify`/`last_ship` outcomes from
                    `backups-index.json` (no file → None).
        """
        snapshots = self._snapshots()
        now = time.time()
        sized: list[tuple[Path, float, int]] = []
        for path in snapshots:
            try:
                info = path.stat()
            except OSError:
                continue  # the file vanished between glob and stat (a race with rotation)
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
        """A detailed snapshot list (operator `GET /backups`, arch §7.2).

                    Newest first; `verified` ∈ ok/failed/unknown (a `.bad` marker — an
                    explicit `failed`), `bad` — a sidecar exists, `shipped` — per the
                    sidecar index (a confirmed ship of the copy off the server).
        """
        state = self._read_state()
        verified_map = state.get("verified")
        verified_map = verified_map if isinstance(verified_map, dict) else {}
        shipped_map = state.get("shipped")
        shipped_map = shipped_map if isinstance(shipped_map, dict) else {}
        items: list[dict[str, Any]] = []
        for path in reversed(self._snapshots()):  # newest first
            try:
                size = path.stat().st_size
            except OSError:
                continue  # the file vanished between glob and stat (a race with rotation)
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

    # --- Loop ------------------------------------------------------------------

    async def run(self) -> None:
        """A snapshot right after the start, then once per BACKUP_INTERVAL_SEC.

                    #14 (arch §6.1): after a successful snapshot, when `BACKUP_VERIFY_EVERY`
                    > 0 and the counter reaches the multiple, — a restore verification
                    (`verify`); its failure does not crash the loop (NFR-3), the outcome —
                    as an event.

                    #14 (no14-02, arch §6.1): if the verification passed (`verify.ok=True`)
                    and the operator ship command is set (`BACKUP_SHIP_CMD`) — the copy is
                    shipped off the server (`ship`, FR-3.1); a ship failure does not crash
                    the loop and does not break local rotation (FR-3.2).

                    Phase 10: after each snapshot — namespace registry grooming; a grooming
                    failure does not kill the loop (like a snapshot failure, NFR-3):
                    a groom_failed event in the log, a retry — the next interval.
        """
        while not self._stopping:
            target: Path | None = None
            try:
                target = await asyncio.to_thread(self.snapshot)
            except (sqlite3.Error, OSError):
                # The DB/directory is unreachable — the process must live (NFR-3);
                # details — to the JSON log, a retry on the interval.
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
        """Verify a snapshot on the `BACKUP_VERIFY_EVERY` schedule (0 — manual).

                    A verify failure does not crash the loop: a `backup_verify_failed`
                    event (NFR-3). The counter is in-process (after a restart it starts
                    over — verification is rare and cheap; no state on disk).

                    Returns:
                        True — the verification ran and the snapshot is good (`ok=True`);
                        otherwise False (verification off, not its turn, or a failure).
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
        """Ship a copy after a successful verification (FR-3.1/FR-3.2).

                    A ship failure does not crash the loop (NFR-3): the outcome — a
                    `backup_ship_failed` event, local rotation works as before.
        """
        try:
            await asyncio.to_thread(self.ship, target)
        except Exception:
            logging.getLogger("app").exception(
                "backup ship failed", extra={"event": "backup_ship_failed"}
            )


def _snapshot_name(moment: datetime | None = None) -> str:
    """Snapshot file name: notes-<UTC to the second>.db (sortable)."""
    moment = moment or datetime.now(UTC)
    return f"{IMAGE_PREFIX}{moment.strftime('%Y%m%dT%H%M%SZ')}{IMAGE_SUFFIX}"
