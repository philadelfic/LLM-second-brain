"""Фоновый воркер (Фаза 3, шаг 3.5): партия, back-off, цикл, graceful-stop.

ARCH §3.4: до-векторизация pending; back-off 30с → ×2 → max 15 мин;
успех сбрасывает интервал. В API-тестах цикл живёт внутри TestClient (offline);
здесь — юнит на партии/интервалах + asyncio-тесты цикла с PENDING_RETRY_SEC=0.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

import pytest
from fakes import FailingEmbedder, HashEmbedder

from app.config import get_settings
from app.services.notes import NoteService
from app.services.worker import MAX_INTERVAL_SEC, BackgroundWorker, next_interval
from app.storage import vectors
from app.storage.db import init_db, session


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def fast(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Как settings, но с PENDING_RETRY_SEC=0 — цикл без пауз."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("PENDING_RETRY_SEC", "0")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def make_worker(settings, embedding) -> BackgroundWorker:
    return BackgroundWorker(settings, embedding)


# --- партии ------------------------------------------------------------------


def test_process_pending_vectorizes_batch(settings) -> None:
    notes = NoteService(settings, FailingEmbedder())
    texts = ["первая отложенная заметка", "вторая отложенная заметка"]
    for text in texts:  # FailingEmbedder → pending без векторов
        notes.save(text)
    worker = make_worker(settings, HashEmbedder(8))
    assert worker.process_pending() == 2
    with session(settings) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM notes WHERE vector_status = 'ok'"
        ).fetchone()[0] == 2
        assert vectors.count(conn) == 2
        assert vectors.get_vector(conn, 1) == pytest.approx(
            HashEmbedder(8).embed(texts[0]), abs=1e-6
        )


def test_process_pending_empty_queue(settings) -> None:
    assert make_worker(settings, HashEmbedder(8)).process_pending() == 0


def test_process_failure_keeps_pending(settings) -> None:
    notes = NoteService(settings, FailingEmbedder())
    notes.save("заметка без сервера")
    worker = make_worker(settings, FailingEmbedder())
    assert worker.process_pending() == 0
    with session(settings) as conn:
        row = conn.execute(
            "SELECT vector_status FROM notes WHERE id = 1"
        ).fetchone()
        assert row["vector_status"] == "pending"
        assert vectors.count(conn) == 0


def test_process_skips_trash(settings) -> None:
    """Trash не до-векторизуется: очередь — только активные заметки."""
    notes = NoteService(settings, FailingEmbedder())
    notes.save("удалим меня до воркера")
    notes.delete(1)
    notes.save("остаюсь в очереди до самого воркера")
    worker = make_worker(settings, HashEmbedder(8))
    assert worker.process_pending() == 1
    with session(settings) as conn:
        assert vectors.get_vector(conn, 1) is None  # trash вектора не получает
        assert vectors.get_vector(conn, 2) is not None


def test_process_respects_limit(settings) -> None:
    notes = NoteService(settings, FailingEmbedder())
    for number in range(3):
        notes.save(f"заметка очереди {number} с индивидуальной темой")
    worker = make_worker(settings, HashEmbedder(8))
    assert worker.process_pending(limit=2) == 2
    assert worker.process_pending(limit=2) == 1
    assert worker.process_pending() == 0


# --- back-off ----------------------------------------------------------------


def test_next_interval_formula() -> None:
    assert next_interval(30.0, 30) == 60.0
    assert next_interval(480.0, 30) == 900.0  # 960 усечено потолком
    assert next_interval(900.0, 30) == 900.0
    assert next_interval(MAX_INTERVAL_SEC, 30) == MAX_INTERVAL_SEC


def test_worker_starts_at_configured_interval(settings) -> None:
    assert make_worker(settings, HashEmbedder(8)).interval == 30.0
    assert make_worker(settings, HashEmbedder(8)).interval != MAX_INTERVAL_SEC


# --- живой цикл (asyncio) ------------------------------------------------------


@pytest.mark.asyncio
async def test_run_catches_pending_and_resets_interval(fast) -> None:
    """Цикл воркера догоняет pending и возвращает интервал к старту."""
    notes = NoteService(fast, FailingEmbedder())
    notes.save("отложенный текст для цикла воркера")
    worker = make_worker(fast, HashEmbedder(8))
    task = asyncio.create_task(worker.run())
    status = None
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with session(fast) as conn:
            status = conn.execute(
                "SELECT vector_status FROM notes WHERE id = 1"
            ).fetchone()[0]
        if status == "ok":
            break
        await asyncio.sleep(0.01)
    worker.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert status == "ok"
    # успех сбросил интервал к стартовому (PENDING_RETRY_SEC=0)
    assert worker.interval == float(fast.pending_retry_sec)


@pytest.mark.asyncio
async def test_stop_terminates_idle_loop(fast) -> None:
    """Мягкий стоп: цикл с пустой очередью завершается сам, без CancelledError."""
    worker = make_worker(fast, HashEmbedder(8))
    task = asyncio.create_task(worker.run())
    worker.stop()
    await asyncio.wait_for(task, timeout=2.0)  # не падает, не висит
    assert task.done() and task.cancelled() is False


# --- жизненный цикл приложения ------------------------------------------


def test_app_lifespan_starts_and_stops_worker(client, test_env) -> None:
    """create_app: воркер живёт в lifespan и корректно гасится (TestClient)."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["embedding_ok"] is None  # попыток не было


def test_health_reflects_last_embedding_attempt(client, token) -> None:
    """/health.embedding_ok: False после неудачной попытки кодирования."""
    client.post(
        "/notes",
        json={"text": "заметка до health-опроса"},
        headers={"Authorization": f"Bearer {token}"},
    )
    body = client.get("/health").json()
    assert body["embedding_ok"] is False
    assert body["pending_vector"] == 1  # воркер ещё не векторизовал (offline)