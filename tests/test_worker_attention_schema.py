"""Схема worker_jobs: колонки внимания и индекс (№39, no39-02; arch §5.2).

Миграция ленивая (зона воркера): `_ensure_job_table` расширяет DDL и
идемпотентно добавляет недостающие колонки по `PRAGMA table_info(worker_jobs)`
— старая база (до 3.2.0) получает `attempts`/`last_outcome`/`last_error`/
`attention_at` и индекс `idx_worker_jobs_attention`; повторный вызов — no-op.
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session, transaction


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    resolved = get_settings()
    init_db(resolved)
    return resolved


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


def test_columns_added_idempotent(settings) -> None:
    """Старая схема (без колонок внимания) мигрируется; повтор — no-op."""
    with session(settings) as conn, transaction(conn):
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
    worker = BackgroundWorker(settings, _NoopEmbedder())

    worker._ensure_job_table()
    columns = _columns(settings)
    assert {"attempts", "last_outcome", "last_error", "attention_at"} <= columns
    assert "idx_worker_jobs_attention" in _indexes(settings)

    # повторный вызов (и в памяти, и после сброса мемоизации) — no-op
    worker._ensure_job_table()
    worker._jobs_table_ready = False
    worker._ensure_job_table()
    assert _columns(settings) == columns


class _NoopEmbedder:
    """Заглушка эмбеддера: схема внимания кодировщика не касается."""

    def embed(self, text: str) -> list[float]:
        return []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return []

    def close(self) -> None:
        return None
