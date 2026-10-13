"""Снятие «требует вмешательства» по смене конфигурации слота (lsbdef0008-02/-03).

Старт воркера сравнивает текущий отпечаток конфигурации слотов
(`slot_budget.slot_config_fingerprint`) с записанным в `meta` (блок 2,
`SLOT_CONFIG_META_KEY`): отпечатка нет или он отличается — «причина ушла»,
внимание снимается у заданий ВСЕХ слотов (`attention_at=NULL`, `attempts=0`),
событие `job_attention_released` с `reason="startup"`; отпечаток тот же —
ничего не снимаем. `SLOT_ATTENTION_RESET_ON_START=false` отключает шаг целиком
(воля оператора). Отпечаток включает «думать / не думать» (`*_THINK`) и
клиентский таймаут (`*_TIMEOUT_SEC`) — смена любого из них снимает внимание
сама (дефект приёмки 12.10.2026: обход `JUDGE_THINK=false` + малый бюджет не
размораживал пары). Счётчик попыток, back-off и наблюдаемость очередей —
блок 2 (no39-02), здесь не дублируются.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from fakes import FailingEmbedder

from app.config import Settings, get_settings
from app.services.slot_budget import (
    FINGERPRINT_DEFAULTS,
    FINGERPRINT_FIELDS,
    slot_config_fingerprint,
)
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


def _fingerprint_with(settings, **overrides: object) -> str:
    """Отпечаток «прежней» конфигурации слота: поля из `Settings`, часть — с заменой.

    Носитель строится по `FINGERPRINT_FIELDS` и читает значения из `Settings`
    (дефолты — как в бою), поэтому тесту не нужен второй экземпляр настроек и
    БД: он подаёт в `meta` отпечаток конфигурации «до правки», а воркер стартует
    уже «после».
    """
    values = {
        name: getattr(settings, name, FINGERPRINT_DEFAULTS.get(name, ""))
        for name in FINGERPRINT_FIELDS
    }
    values.update(overrides)
    return slot_config_fingerprint(SimpleNamespace(**values))


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


def test_think_change_releases_attention(tmp_path, monkeypatch, caplog) -> None:
    """Смена «думать / не думать» на старте размораживает пары (дефект 12.10.2026).

    Слоты стояли на временном обходе `SUMMARY_THINK=false` + `JUDGE_THINK=true`;
    после возврата к штатному размышлению внимание снимается без ручных действий
    (FR-2.3), отпечаток в `meta` перезаписывается текущим.
    """
    settings = _settings(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO, logger="app")
    worker = _worker(settings)
    job_id = _seed_job(settings, note_id=1)
    _frozen_job(settings, job_id)
    # конфигурация «до правки»: размышление выключено у summary и включено у judge
    _write_fingerprint(
        settings, _fingerprint_with(settings, summary_think=False, judge_think=True)
    )

    assert worker._startup_attention_reset() == 1

    row = _job_row(settings, job_id)
    assert row["attention_at"] is None
    assert row["attempts"] == 0
    assert row["status"] == "pending"
    events = _events(caplog, "job_attention_released")
    assert events and getattr(events[-1], "reason", None) == "startup"
    assert _stored_fingerprint(settings) == slot_config_fingerprint(settings)


def test_same_think_keeps_attention(tmp_path, monkeypatch) -> None:
    """Тот же `THINK` → внимание не снимается (нет ложных сбросов)."""
    settings = _settings(
        tmp_path, monkeypatch, SUMMARY_THINK="true", JUDGE_THINK="true"
    )
    worker = _worker(settings)
    job_id = _seed_job(settings, note_id=1)
    _frozen_job(settings, job_id)
    _write_fingerprint(settings, slot_config_fingerprint(settings))

    assert worker._startup_attention_reset() == 0

    row = _job_row(settings, job_id)
    assert row["attention_at"] is not None
    assert row["attempts"] == 8
    # поле реально участвует в сверке: иначе тест «того же THINK» ничего не значит
    assert _fingerprint_with(settings, judge_think=False) != _stored_fingerprint(
        settings
    )


@pytest.mark.parametrize(
    "override",
    [{"summary_timeout_sec": 120}, {"judge_timeout_sec": 60}],
    ids=["summary", "judge"],
)
def test_timeout_change_releases_attention(
    tmp_path, monkeypatch, caplog, override
) -> None:
    """Смена одного лишь `*_TIMEOUT_SEC` (без бюджета) тоже снимает внимание.

    Таймаут меняет исход вызова (`timeout` вместо ответа) — это часть «причины
    ушла», поэтому правка таймаута размораживает пару (arch §13.4).
    """
    settings = _settings(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO, logger="app")
    worker = _worker(settings)
    job_id = _seed_job(settings, note_id=1)
    _frozen_job(settings, job_id)
    _write_fingerprint(settings, _fingerprint_with(settings, **override))

    assert worker._startup_attention_reset() == 1

    row = _job_row(settings, job_id)
    assert row["attention_at"] is None
    assert row["attempts"] == 0
    events = _events(caplog, "job_attention_released")
    assert events and getattr(events[-1], "reason", None) == "startup"
    assert _stored_fingerprint(settings) == slot_config_fingerprint(settings)


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
