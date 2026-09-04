"""Фоновый воркер (Фазы 3–4): партии, back-off, цикл, graceful-stop.

ARCH §3.4: две независимые очереди (pending_vector с Фазы 3, pending_summary
с Фазы 4); back-off 30с → ×2 → max 15 мин — отдельно по каждой очереди;
успех в очереди сбрасывает только её интервал. В API-тестах цикл живёт внутри
TestClient (offline); здесь — юнит на партии/интервалах + asyncio-тесты цикла
с PENDING_RETRY_SEC=0.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

import pytest
from fakes import FailingEmbedder, FailingSummarizer, FixedSummarizer, HashEmbedder

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


@pytest.fixture
def slow(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Как settings, но с PENDING_RETRY_SEC=1 — интервалы различимы во времени."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("PENDING_RETRY_SEC", "1")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def make_worker(settings, embedding, summarizer=None) -> BackgroundWorker:
    return BackgroundWorker(settings, embedding, summarizer)


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
    worker = make_worker(settings, HashEmbedder(8))
    assert worker.interval == 30.0
    assert worker.summary_interval == 30.0  # у суммаризационной очереди свой счётчик
    assert worker.interval != MAX_INTERVAL_SEC


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


def test_save_makes_no_embedding_attempt(client, token) -> None:
    """Фаза 8: save не кодирует синхронно — embedding_ok остаётся None
    (попыток кодирования не было), заметка стоит в очереди pending_vector."""
    client.post(
        "/notes",
        # Фаза 11, follow-up 5b: REST strict — POST без title → 422;
        # по клиентскому контракту передаём валидный title (≤5 слов).
        json={"text": "заметка до health-опроса", "title": "Заметка health"},
        headers={"Authorization": f"Bearer {token}"},
    )
    body = client.get("/health").json()
    assert body["embedding_ok"] is None  # попыток кодирования не было
    assert body["pending_vector"] == 1  # очередь вектора — воркеру

# --- суммаризационная очередь (Фаза 4, режим «Б») -----------------------------


class MidFlightUpdateSummarizer(FixedSummarizer):
    """Фейк-гонка: пока воркер генерирует, заметка обновляется (ARCH §4.5)."""

    def __init__(self, notes_service, note_id: int, new_text: str) -> None:
        super().__init__("Суммари старого текста.")
        self._notes = notes_service
        self._note_id = note_id
        self._new_text = new_text

    def summarize(self, text: str) -> str:
        result = super().summarize(text)
        self._notes.update(self._note_id, self._new_text)  # текст меняется здесь
        return result


def test_process_summary_fills_pending(settings) -> None:
    notes = NoteService(settings, FailingEmbedder())  # save → все статусы pending
    saved = notes.save("заметка, ждущая суммаризацию фоновым воркером")
    worker = make_worker(
        settings, HashEmbedder(8), FixedSummarizer("Суммари одним предложением.")
    )
    assert worker.process_summary_pending() == 1
    with session(settings) as conn:
        row = conn.execute(
            "SELECT summary, summary_status FROM notes WHERE id = ?",
            (saved["id"],),
        ).fetchone()
        assert row["summary"] == "Суммари одним предложением."
        assert row["summary_status"] == "ok"


def test_process_summary_empty_queue(settings) -> None:
    worker = make_worker(settings, HashEmbedder(8), FixedSummarizer())
    assert worker.process_summary_pending() == 0
    assert worker.summary_interval == float(settings.pending_retry_sec)


def test_process_summary_failure_keeps_pending(settings) -> None:
    """Отказ генерации: статус/подпись не тронуты, в выдаче — fallback-усечение."""
    notes = NoteService(settings, FailingEmbedder())
    saved = notes.save("заметка при отказе суммаризатора")
    fake = FailingSummarizer()
    worker = make_worker(settings, FailingEmbedder(), fake)
    assert worker.process_summary_pending() == 0
    assert fake.calls == ["заметка при отказе суммаризатора"]  # попытка была
    with session(settings) as conn:
        row = conn.execute(
            "SELECT summary, summary_status FROM notes WHERE id = ?",
            (saved["id"],),
        ).fetchone()
        assert row["summary"] == ""
        assert row["summary_status"] == "pending"
    # выдача до готовности: fallback-усечение + pending (FR-1/FR-3)
    fetched = notes.get([saved["id"]])["notes"][0]
    assert fetched["summary"] == "заметка при отказе суммаризатора"
    assert fetched["summary_status"] == "pending"


def test_process_summary_skips_trash(settings) -> None:
    """Trash не до-суммаризуется: очередь — только активные заметки."""
    notes = NoteService(settings, FailingEmbedder())
    notes.save("удалим до суммаризации")
    notes.delete(1)
    notes.save("остаюсь в суммаризационной очереди")
    worker = make_worker(settings, FailingEmbedder(), FixedSummarizer("Суммари живых."))
    assert worker.process_summary_pending() == 1
    with session(settings) as conn:
        statuses = {
            row["id"]: (row["summary"], row["summary_status"])
            for row in conn.execute(
                "SELECT id, summary, summary_status FROM notes ORDER BY id"
            ).fetchall()
        }
        assert statuses[1] == ("", "pending")  # trash: очередь его не коснулась
        assert statuses[2] == ("Суммари живых.", "ok")


def test_process_summary_respects_limit(settings) -> None:
    notes = NoteService(settings, FailingEmbedder())
    for number in range(3):
        notes.save(f"заметка суммаризации {number}, тема отдельная")
    worker = make_worker(settings, FailingEmbedder(), FixedSummarizer("С."))
    assert worker.process_summary_pending(limit=2) == 2
    assert worker.process_summary_pending(limit=2) == 1
    assert worker.process_summary_pending() == 0


def test_process_summary_race_with_update(settings) -> None:
    """Гонка с memory_update: суммари старого текста не затирает свежую заметку."""
    notes = NoteService(settings, FailingEmbedder())
    saved = notes.save("старый текст, который изменят во время генерации")
    racy = MidFlightUpdateSummarizer(
        notes, saved["id"], "новый текст, записанный прямо во время генерации"
    )
    worker = make_worker(settings, FailingEmbedder(), racy)
    assert worker.process_summary_pending() == 0  # текст изменился — записи нет
    with session(settings) as conn:
        row = conn.execute(
            "SELECT summary, summary_status FROM notes WHERE id = ?",
            (saved["id"],),
        ).fetchone()
        assert row["summary"] == ""
        assert row["summary_status"] == "pending"  # догонится как обычный pending
    # заметку позже всё же суммаризуют уже по новому тексту
    worker._summarizer = FixedSummarizer("Свежее суммари нового текста.")
    assert worker.process_summary_pending() == 1
    with session(settings) as conn:
        row = conn.execute(
            "SELECT summary, summary_status FROM notes WHERE id = ?",
            (saved["id"],),
        ).fetchone()
        assert row["summary"] == "Свежее суммари нового текста."
        assert row["summary_status"] == "ok"


def test_process_summary_without_summarizer(settings) -> None:
    """DI None (тестовый режим Фазы 3): очередь не обслуживается, не падает."""
    notes = NoteService(settings, FailingEmbedder())
    notes.save("заметка без суммаризатора в воркере")
    assert make_worker(settings, HashEmbedder(8)).process_summary_pending() == 0
    with session(settings) as conn:
        assert conn.execute(
            "SELECT summary_status FROM notes WHERE id = 1"
        ).fetchone()[0] == "pending"


@pytest.mark.asyncio
async def test_run_catches_pending_summary(fast) -> None:
    """Цикл воркера догоняет pending_summary до ok (режим «Б» работает)."""
    notes = NoteService(fast, FailingEmbedder())
    notes.save("отложенный текст для суммаризации циклом")
    worker = make_worker(fast, FailingEmbedder(), FixedSummarizer("Суммари цикла."))
    task = asyncio.create_task(worker.run())
    status = None
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with session(fast) as conn:
            row = conn.execute(
                "SELECT summary, summary_status FROM notes WHERE id = 1"
            ).fetchone()
            status = row["summary_status"]
        if status == "ok":
            break
        await asyncio.sleep(0.01)
    worker.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert status == "ok"
    assert row["summary"] == "Суммари цикла."
    # успех сбросил интервал к стартовому (PENDING_RETRY_SEC=0)
    assert worker.summary_interval == float(fast.pending_retry_sec)


@pytest.mark.asyncio
async def test_backoff_independent_per_queue(slow) -> None:
    """ARCH §3.4: back-off независим — отказ векторизации не мешает суммаризации."""
    notes = NoteService(slow, FailingEmbedder())
    notes.save("заметка для независимости очередей воркера")
    worker = make_worker(slow, FailingEmbedder(), FailingSummarizer())
    task = asyncio.create_task(worker.run())
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and worker.interval <= 1.0:
        await asyncio.sleep(0.05)
    assert worker.interval > 1.0  # векторная очередь в back-off
    # «чиним» суммаризатор на лету: summary-петля разгребает свою очередь
    worker._summarizer = FixedSummarizer("Позднее, но успешное суммари.")
    status = None
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline:
        with session(slow) as conn:
            status = conn.execute(
                "SELECT summary_status FROM notes WHERE id = 1"
            ).fetchone()[0]
        if status == "ok":
            break
        await asyncio.sleep(0.05)
    worker.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert status == "ok"
    # успех сбросил только интервального счётчика своей очереди
    assert worker.summary_interval == float(slow.pending_retry_sec)
    assert worker.interval > float(slow.pending_retry_sec)  # векторный всё ещё растёт
