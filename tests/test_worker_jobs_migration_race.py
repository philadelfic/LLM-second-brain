"""Гонка ленивой миграции `worker_jobs` (lsbdef-0013, LSBN-D-6; arch §5.2/§6.8).

`_ensure_job_table` зовут из синхронного SQL петель джоб (через
`asyncio.to_thread`) и параллельно стартовое снятие внимания (`run`) —
check-then-act на голом флаге `_jobs_table_ready` пропускал два потока
в ALTER одновременно: второй падал на `duplicate column name` и ронял
итерацию петли (`job loop iteration failed`) и стартовую сверку
(`job_attention_reset_failed`). Прод-деплой 3.2.0, 2026-10-13.

Миграция под локом + терпимый ALTER: два потока на старой схеме (до 3.2.0,
без колонок внимания) — ни один не бросает исключений, схема полная,
стартовое снятие внимания отрабатывает.
"""

from __future__ import annotations

import threading

import pytest

from app.config import get_settings
from app.services.worker import SLOT_CONFIG_META_KEY, BackgroundWorker
from app.storage.db import init_db, session, transaction

# Сколько попыток поймать гонку на одной и той же старой схеме: окно
# «флаг проверен → ALTER исполнен» узкое (несколько sqlite-операций),
# каждый раунд пересоздаёт старую схему и даёт паре потоков новый шанс.
RACE_ROUNDS = 20


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    resolved = get_settings()
    init_db(resolved)
    return resolved


def _create_old_schema(settings) -> None:
    """Старая схема (до 3.2.0): worker_jobs без колонок внимания."""
    with session(settings) as conn, transaction(conn):
        conn.execute("DROP TABLE IF EXISTS worker_jobs")
        conn.execute(
            "CREATE TABLE worker_jobs ("
            "  id         INTEGER PRIMARY KEY,"
            "  slot       TEXT NOT NULL,"
            "  kind       TEXT NOT NULL,"
            "  note_id    INTEGER NOT NULL,"
            "  payload    TEXT,"
            "  status     TEXT NOT NULL DEFAULT 'pending',"
            "  created_at TEXT NOT NULL,"
            "  updated_at TEXT NOT NULL"
            ")"
        )


def _columns(settings) -> set[str]:
    with session(settings) as conn:
        return {row["name"] for row in conn.execute(
            "PRAGMA table_info(worker_jobs)"
        ).fetchall()}


def _indexes(settings) -> set[str]:
    with session(settings) as conn:
        return {row["name"] for row in conn.execute(
            "PRAGMA index_list(worker_jobs)"
        ).fetchall()}


def test_migration_race_two_threads(settings) -> None:
    """Два потока мигрируют старую схему одновременно — ноль исключений."""
    worker = BackgroundWorker(settings, _NoopEmbedder())
    errors: list[BaseException] = []

    for _ in range(RACE_ROUNDS):
        _create_old_schema(settings)
        worker._jobs_table_ready = False  # оба потока видят «миграции не было»

        barrier = threading.Barrier(2)

        def _migrate() -> None:
            try:
                barrier.wait()
                worker._ensure_job_table()
            except BaseException as exc:  # noqa: BLE001 — гонку собираем, не роняем
                errors.append(exc)

        threads = [
            threading.Thread(target=_migrate, name=f"migrator-{n}")
            for n in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert errors == []
    # схема полная: 4 колонки внимания (№39) + индекс выборки «требует
    # вмешательства»
    assert {
        "attempts",
        "last_outcome",
        "last_error",
        "attention_at",
    } <= _columns(settings)
    assert "idx_worker_jobs_attention" in _indexes(settings)


def test_startup_attention_reset_after_race(settings) -> None:
    """Стартовое снятие внимания на отмигрированной БД не падает.

    Свежая база: отпечатка конфигурации в `meta` нет («сброшен») — сверка
    снимает внимание (пусто → 0), пишет отпечаток и возвращает число.
    """
    worker = BackgroundWorker(settings, _NoopEmbedder())
    worker._ensure_job_table()
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "DELETE FROM meta WHERE key = ?", (SLOT_CONFIG_META_KEY,)
        )

    released = worker._startup_attention_reset()

    assert isinstance(released, int)
    assert released == 0
    with session(settings) as conn:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (SLOT_CONFIG_META_KEY,)
        ).fetchone()
    assert row is not None  # отпечаток перезаписан


class _NoopEmbedder:
    """Заглушка эмбеддера: миграция схемы кодировщика не касается."""

    def embed(self, text: str) -> list[float]:
        return []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return []

    def close(self) -> None:
        return None
