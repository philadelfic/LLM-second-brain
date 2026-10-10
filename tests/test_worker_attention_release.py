"""Снятие «требует вмешательства» по смене конфигурации слота (lsbdef0008-02).

Старт воркера сравнивает текущий отпечаток конфигурации слотов
(`slot_budget.slot_config_fingerprint`) с записанным в `meta` (блок 2,
`SLOT_CONFIG_META_KEY`): отпечатка нет или он отличается — «причина ушла»,
внимание снимается у заданий ВСЕХ слотов (`attention_at=NULL`, `attempts=0`),
событие `job_attention_released` с `reason="startup"`; отпечаток тот же —
ничего не снимаем. `SLOT_ATTENTION_RESET_ON_START=false` отключает шаг целиком
(воля оператора). Счётчик попыток, back-off и наблюдаемость очередей — блок 2
(no39-02), здесь не дублируются.
"""

from __future__ import annotations

import logging

import pytest
from fakes import FailingEmbedder

from app.config import Settings, get_settings
from app.services.slot_budget import slot_config_fingerprint
from app.services.worker import SLOT_CONFIG_META_KEY, BackgroundWorker
from app.storage.db import init_db, session, transaction


def _settings(tmp_path, monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    resolved = get_settings()
    init_db(resolved)
    return resolved


def _worker(settings) -> BackgroundWorker:
    worker = BackgroundWorker(settings, FailingEmbedder())
    worker._ensure_job_table()
    return worker


def _seed_job(
    settings,
    *,
    slot: str = "judge",
    kind: str = "dedup",
    note_id: int = 1,
) -> int:
    _worker(settings)
    with session(settings) as conn, transaction(conn):
        cursor = conn.execute(
            "INSERT INTO worker_jobs (slot, kind, note_id, payload) "
            "VALUES (?, ?, ?, ?)",
            (slot, kind, note_id, None),
        )
        return int(cursor.lastrowid or 0)


def _frozen_job(settings, job_id: int) -> None:
    """Задание «требует вмешательства»: attempts исчерпан, attention_at стоит."""
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE worker_jobs SET attempts = 8, last_outcome = 'truncated', "
            "attention_at = '2026-01-01T00:00:00Z' WHERE id = ?",
            (job_id,),
        )


def _write_fingerprint(settings, value: str) -> None:
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (SLOT_CONFIG_META_KEY, value),
        )


def _stored_fingerprint(settings) -> str | None:
    with session(settings) as conn:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (SLOT_CONFIG_META_KEY,)
        ).fetchone()
    return row["value"] if row is not None else None


def _job_row(settings, job_id: int) -> dict:
    with session(settings) as conn:
        row = conn.execute(
            "SELECT attempts, last_outcome, attention_at, status "
            "FROM worker_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    return dict(row) if row is not None else {}


def _events(caplog, event: str) -> list:
    return [
        record for record in caplog.records if getattr(record, "event", None) == event
    ]


def test_changed_fingerprint_releases_attention(tmp_path, monkeypatch, caplog) -> None:
    """Сменился отпечаток → внимание снято, счётчик сброшен, событие `startup`."""
    settings = _settings(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO, logger="app")
    worker = _worker(settings)
    job_id = _seed_job(settings, note_id=1)
    _frozen_job(settings, job_id)
    _write_fingerprint(settings, "old-fingerprint")

    released = worker._startup_attention_reset()

    assert released == 1
    row = _job_row(settings, job_id)
    assert row["attention_at"] is None
    assert row["attempts"] == 0
    assert row["status"] == "pending"  # задание живо, не «мёртвое»
    events = _events(caplog, "job_attention_released")
    assert events and getattr(events[-1], "reason", None) == "startup"
    assert _stored_fingerprint(settings) == slot_config_fingerprint(settings)


def test_missing_fingerprint_releases_attention(tmp_path, monkeypatch) -> None:
    """Отпечатка в `meta` нет (первый старт с механизмом) → внимание снято."""
    settings = _settings(tmp_path, monkeypatch)
    worker = _worker(settings)
    job_id = _seed_job(settings, note_id=1)
    _frozen_job(settings, job_id)

    assert worker._startup_attention_reset() == 1

    assert _job_row(settings, job_id)["attention_at"] is None
    assert _stored_fingerprint(settings) == slot_config_fingerprint(settings)


def test_same_fingerprint_keeps_attention(tmp_path, monkeypatch) -> None:
    """Тот же отпечаток → ничего не снимаем (отпечаток не «дёргается»)."""
    settings = _settings(tmp_path, monkeypatch)
    worker = _worker(settings)
    job_id = _seed_job(settings, note_id=1)
    _frozen_job(settings, job_id)
    _write_fingerprint(settings, slot_config_fingerprint(settings))

    assert worker._startup_attention_reset() == 0

    row = _job_row(settings, job_id)
    assert row["attention_at"] is not None
    assert row["attempts"] == 8


def test_reset_disabled_by_flag(tmp_path, monkeypatch) -> None:
    """`SLOT_ATTENTION_RESET_ON_START=false` отключает шаг целиком."""
    settings = _settings(
        tmp_path, monkeypatch, SLOT_ATTENTION_RESET_ON_START="false"
    )
    worker = _worker(settings)
    job_id = _seed_job(settings, note_id=1)
    _frozen_job(settings, job_id)
    _write_fingerprint(settings, "old-fingerprint")

    assert worker._startup_attention_reset() == 0

    assert _job_row(settings, job_id)["attention_at"] is not None
    assert _stored_fingerprint(settings) == "old-fingerprint"


def test_release_then_queue_drains(tmp_path, monkeypatch) -> None:
    """Граница с блоком 2: после снятия внимания очередь судьи снова берётся."""
    settings = _settings(tmp_path, monkeypatch)
    worker = _worker(settings)
    job_id = _seed_job(settings, slot="judge", kind="dedup", note_id=1)
    _frozen_job(settings, job_id)
    _write_fingerprint(settings, "old-fingerprint")
    # замороженное задание петлю не гонит (busy-loop'а нет)
    assert worker._judge_queue_empty() is True

    assert worker._startup_attention_reset() == 1

    assert worker._judge_queue_empty() is False
    taken = [int(row["id"]) for row in worker._pending_jobs("judge", "dedup", 10)]
    assert job_id in taken
