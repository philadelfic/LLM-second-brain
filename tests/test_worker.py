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
import logging
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from fakes import (
    FailingEmbedder,
    FailingSummarizer,
    FixedClassifier,
    FixedSummarizer,
    HashEmbedder,
    RecordingDedup,
)

from app.config import get_settings
from app.services.classifier import Classification
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.services.worker import MAX_INTERVAL_SEC, BackgroundWorker, next_interval
from app.storage import vectors
from app.storage.db import init_db, session, transaction


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


class MidFlightUpdateEmbedder(HashEmbedder):
    """Фейк-гонка: пока воркер кодирует партию, одна заметка обновляется
    (пул 1, аналог MidFlightUpdateSummarizer для embedding-петли)."""

    def __init__(self, dim: int, notes_service, note_id: int, new_text: str) -> None:
        super().__init__(dim)
        self._notes = notes_service
        self._note_id = note_id
        self._new_text = new_text

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        result = super().embed_texts(texts)
        self._notes.update(self._note_id, self._new_text)  # текст меняется здесь
        return result


def test_process_pending_race_with_update(settings) -> None:
    """Гонка с memory_update: протухший вектор не пишется, judge-работа не
    создаётся; остальные заметки партии довекторизованы штатно (пул 1)."""
    notes = NoteService(settings, FailingEmbedder())
    notes.save("старый текст, который изменят во время кодирования")
    notes.save("вторая заметка партии, не тронутая гонкой")
    racy = MidFlightUpdateEmbedder(
        8, notes, 1, "новый текст, записанный прямо во время кодирования"
    )
    worker = make_worker(settings, racy)
    assert worker.process_pending() == 1  # только вторая фактически довекторизована
    with session(settings) as conn:
        row1 = conn.execute("SELECT * FROM notes WHERE id = 1").fetchone()
        assert row1["vector_status"] == "pending"  # гонка: осталась pending
        assert vectors.get_vector(conn, 1) is None  # протухшего вектора нет
        row2 = conn.execute("SELECT * FROM notes WHERE id = 2").fetchone()
        assert row2["vector_status"] == "ok"
        assert vectors.get_vector(conn, 2) is not None
        # judge-работа только для фактически довекторизованной (id=2)
        jobs = conn.execute(
            "SELECT note_id FROM worker_jobs "
            "WHERE slot='judge' AND kind='dedup' ORDER BY note_id"
        ).fetchall()
        assert [j["note_id"] for j in jobs] == [2]


def test_process_pending_race_retry_vectorizes_new_text(settings) -> None:
    """Повтор: следующая партия кодирует новый текст → ok, вектор нового
    текста, judge-работа создана (пул 1)."""
    notes = NoteService(settings, FailingEmbedder())
    notes.save("старый текст, который изменят во время кодирования")
    notes.save("вторая заметка партии, не тронутая гонкой")
    racy = MidFlightUpdateEmbedder(
        8, notes, 1, "новый текст, записанный прямо во время кодирования"
    )
    worker = make_worker(settings, racy)
    worker.process_pending()  # первая гонка: id=1 осталась pending
    # повтор: id=1 кодируется уже по новому тексту
    assert worker.process_pending() == 1
    with session(settings) as conn:
        row = conn.execute("SELECT * FROM notes WHERE id = 1").fetchone()
        assert row["vector_status"] == "ok"
        assert vectors.get_vector(conn, 1) == pytest.approx(
            HashEmbedder(8).embed(
                "новый текст, записанный прямо во время кодирования"
            ),
            abs=1e-6,
        )
        jobs = conn.execute(
            "SELECT note_id FROM worker_jobs "
            "WHERE slot='judge' AND kind='dedup' ORDER BY note_id"
        ).fetchall()
        assert [j["note_id"] for j in jobs] == [1, 2]


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
async def test_run_summary_survives_non_slug_domain_hint(fast) -> None:
    """Пул 2: мусорный вывод классификатора (не-слаг domain_hint →
    NamespaceValidationError из _auto_move_target) не убивает summary-петлю:
    петля продолжает работать, последующие заметки суммаризуются штатно."""
    notes = NoteService(fast, FailingEmbedder())
    notes.save("заметка с мусорной разметкой классификатора")
    classifier = FixedClassifier(Classification("Работа", None, 0.9))
    worker = BackgroundWorker(
        fast, FailingEmbedder(), FixedSummarizer("Суммари цикла."), classifier=classifier
    )
    task = asyncio.create_task(worker.run())
    # первая заметка: суммаризация доведена, классификация упала (сдержана)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with session(fast) as conn:
            status = conn.execute(
                "SELECT summary_status FROM notes WHERE id = 1"
            ).fetchone()[0]
        if status == "ok":
            break
        await asyncio.sleep(0.01)
    assert status == "ok"
    with session(fast) as conn:
        row = conn.execute(
            "SELECT namespace FROM notes WHERE id = 1"
        ).fetchone()
        assert row["namespace"] == "default"  # переезд не произошёл
    # петля жива: вторая заметка тоже суммаризуется
    notes.save("вторая заметка после сбоя классификатора")
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with session(fast) as conn:
            status2 = conn.execute(
                "SELECT summary_status FROM notes WHERE id = 2"
            ).fetchone()[0]
        if status2 == "ok":
            break
        await asyncio.sleep(0.01)
    worker.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert status2 == "ok"


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


# --- Фаза 11 (решение №10): петли по слотам, title-доген, зависимости ---------


class BlockingEmbedder(HashEmbedder):
    """Кодировщик, блокирующий embedding-петлю (проверка параллельности петель)."""

    def __init__(self, dim: int, delay: float = 0.5) -> None:
        super().__init__(dim)
        self.delay = delay

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        time.sleep(self.delay)
        return super().embed_texts(texts)


class ConcurrentTrackingSummarizer(FixedSummarizer):
    """Суммаризатор, трекающий максимальную одновременность вызовов.

    summary-петля обрабатывает заметки последовательно (одна задача за раз на
    эндпоинт слота) — max_active обязан остаться 1.
    """

    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def summarize(self, text: str) -> str:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            return super().summarize(text)
        finally:
            with self._lock:
                self.active -= 1


def test_judge_interval_starts_at_configured(settings) -> None:
    """Своя независимая петля = свой счётчик back-off (решение №10)."""
    worker = make_worker(settings, HashEmbedder(8))
    assert worker.judge_interval == float(settings.pending_retry_sec)


def test_title_dogen_generates_and_truncates(settings) -> None:
    """Title-догенерация (решение №9): миграционная заметка (title IS NULL) →
    думающий вызов слота summary → результат обрезан до TITLE_MAX_WORDS слов,
    запись title."""
    notes = NoteService(settings, FailingEmbedder())
    saved = notes.save("заметка без названия")  # легаси-путь → title=NULL
    worker = make_worker(
        settings,
        HashEmbedder(8),
        FixedSummarizer("один два три четыре пять шесть семь"),
    )
    assert worker.process_title_pending() == 1
    with session(settings) as conn:
        row = conn.execute(
            "SELECT title FROM notes WHERE id = ?", (saved["id"],)
        ).fetchone()
    assert row["title"] == "один два три четыре пять"  # ≤5 слов
    assert len(row["title"].split()) == 5


def test_title_dogen_skips_notes_with_title(settings) -> None:
    """Новые заметки всегда с title — догенерация их не трогает (только NULL)."""
    notes = NoteService(settings, FailingEmbedder())
    notes.save("заметка с названием", title="Есть название")
    worker = make_worker(settings, HashEmbedder(8), FixedSummarizer("генерация"))
    assert worker.process_title_pending() == 0
    with session(settings) as conn:
        row = conn.execute("SELECT title FROM notes WHERE id = 1").fetchone()
    assert row["title"] == "Есть название"


def test_title_dogen_failure_keeps_null(settings) -> None:
    """Отказ генерации (NFR-3): title остаётся NULL, повтор по back-off."""
    notes = NoteService(settings, FailingEmbedder())
    saved = notes.save("заметка без названия")
    worker = make_worker(settings, HashEmbedder(8), FailingSummarizer())
    assert worker.process_title_pending() == 0
    with session(settings) as conn:
        row = conn.execute(
            "SELECT title FROM notes WHERE id = ?", (saved["id"],)
        ).fetchone()
    assert row["title"] is None


def test_judge_job_created_after_vector(settings) -> None:
    """Диспетчер зависимостей (решение №10): judge-работа появляется ТОЛЬКО
    после готовности вектора заметки (embedding-петля)."""
    notes = NoteService(settings, FailingEmbedder())
    notes.save("заметка для проверки зависимости")
    worker = make_worker(settings, HashEmbedder(8))
    # до векторизации judge-работ нет
    assert worker.process_judge_pending() == 0
    worker.process_pending()  # векторизация → judge-работа
    assert worker.process_judge_pending() == 1  # judge-работа появилась


def test_merge_job_created_after_judge_verdict(settings) -> None:
    """Диспетчер зависимостей (решение №10): merge-работа создаётся ТОЛЬКО
    после вердикта судьи (judge-петля) и ходит в summary-слот."""
    notes = NoteService(settings, FailingEmbedder())
    notes.save("ранняя заметка про бэкапы")
    notes.save("поздняя заметка про деплои")
    worker = BackgroundWorker(
        settings,
        HashEmbedder(8),
        FixedSummarizer(merged="слитый текст"),
        RecordingDedup(candidates=[(1, 0.95)]),
    )
    worker.process_pending()
    # до вердикта судьи merge-работ нет
    assert worker.process_merge_pending() == 0
    worker.process_judge_pending()  # вердикт → merge-работа
    assert worker.process_merge_pending() == 1


def test_merge_preserves_earlier_title(settings) -> None:
    """Merge сохраняет title ранней (решение №9): update без title не затирает
    название ранней заметки."""
    notes = NoteService(settings, FailingEmbedder())
    first = notes.save("ранняя заметка про бэкапы", title="Ранняя заметка")
    notes.save("поздняя заметка про деплои", title="Поздняя заметка")
    worker = BackgroundWorker(
        settings,
        HashEmbedder(8),
        FixedSummarizer(merged="слитый текст"),
        RecordingDedup(candidates=[(first["id"], 0.95)]),
    )
    worker.process_pending()
    worker.process_judge_pending()
    worker.process_merge_pending()
    with session(settings) as conn:
        row = conn.execute(
            "SELECT title, text FROM notes WHERE id = ?", (first["id"],)
        ).fetchone()
    assert row["title"] == "Ранняя заметка"  # title ранней сохранён
    assert row["text"] == "слитый текст"


def test_summary_loop_serializes_jobs(settings) -> None:
    """Сериализация работ одного слота: summary-петля обрабатывает заметки
    по одной (одна задача за раз на эндпоинт слота)."""
    notes = NoteService(settings, FailingEmbedder())
    for number in range(3):
        notes.save(f"заметка суммаризации {number}, тема отдельная")
    summarizer = ConcurrentTrackingSummarizer()
    worker = make_worker(settings, FailingEmbedder(), summarizer)
    assert worker.process_summary_pending() == 3
    assert summarizer.max_active == 1  # никогда не больше одной задачи


@pytest.mark.asyncio
async def test_embedding_loop_does_not_block_summary(fast) -> None:
    """Параллельность разных эндпоинтов: embedding-петля (заблокирована) не
    ждёт summary-петлю — суммаризация догоняется независимо."""
    notes = NoteService(fast, FailingEmbedder())
    notes.save("заметка для параллельности петель")
    worker = make_worker(
        fast, BlockingEmbedder(8, delay=0.5), FixedSummarizer("Суммари параллельно.")
    )
    task = asyncio.create_task(worker.run())
    status = None
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        with session(fast) as conn:
            status = conn.execute(
                "SELECT summary_status FROM notes WHERE id = 1"
            ).fetchone()[0]
        if status == "ok":
            break
        await asyncio.sleep(0.01)
    worker.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert status == "ok"  # summary-петля не ждала embedding-петлю


def test_classification_after_summarization(settings) -> None:
    """Диспетчер зависимостей (решение №10): классификация — только после
    суммаризации той же заметки (summary-петля)."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    notes = NoteService(settings, FailingEmbedder())
    notes.save("заметка про рабочие процессы")
    classifier = FixedClassifier(Classification("work", None, 0.95))
    worker = BackgroundWorker(
        settings, HashEmbedder(8), FixedSummarizer(), classifier=classifier
    )
    assert classifier.calls == []  # до суммаризации классификации нет
    worker.process_summary_pending()  # суммаризация → классификация
    assert len(classifier.calls) == 1


# --- супервизор петель (пул 4) ----------------------------------------------


class FlakyEmbedder(HashEmbedder):
    """Кодировщик: первые два вызова — НЕПРЕДВИДЕННЫЙ RuntimeError, дальше
    работает.

    Пул 4: проверка супервизора петли — RuntimeError (не EmbeddingError,
    «громкий» сбой) в итерации embedding-петли не убивает корутину. Два
    отказа подряд дают устойчивое окно наблюдения выросшего back-off-интервала
    (после одного отказа интервал успевает сброситься успешным повтором).
    """

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.calls = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls <= 2:
            raise RuntimeError("непредвиденный сбой кодирования")
        return super().embed_texts(texts)


@pytest.mark.asyncio
async def test_embedding_loop_survives_unexpected_failure(slow) -> None:
    """Пул 4: RuntimeError в итерации embedding-петли не убивает петлю —
    warning, пауза по back-off (интервал вырос), повторный вызов эмбеддера,
    заметка в итоге довекторизована; stop() завершает петлю."""
    notes = NoteService(slow, FailingEmbedder())
    notes.save("заметка для проверки супервизора embedding-петли")
    embedder = FlakyEmbedder(8)
    worker = make_worker(slow, embedder)
    task = asyncio.create_task(worker.run())
    deadline = time.monotonic() + 6.0
    # первые вызовы упали RuntimeError → петля в back-off, интервал вырос
    while time.monotonic() < deadline and embedder.calls < 2:
        await asyncio.sleep(0.01)
    assert embedder.calls >= 2
    while (
        time.monotonic() < deadline
        and worker.interval <= float(slow.pending_retry_sec)
    ):
        await asyncio.sleep(0.01)
    assert worker.interval > float(slow.pending_retry_sec)  # интервал вырос
    # петля выжила: заметка в итоге довекторизована
    status = None
    while time.monotonic() < deadline:
        with session(slow) as conn:
            status = conn.execute(
                "SELECT vector_status FROM notes WHERE id = 1"
            ).fetchone()[0]
        if status == "ok":
            break
        await asyncio.sleep(0.01)
    assert status == "ok"
    worker.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


class FlakySummarizer(FixedSummarizer):
    """Суммаризатор: первый вызов — НЕПРЕДВИДЕННЫЙ RuntimeError, дальше работает.

    Пул 4: проверка супервизора summary-петли. Счётчик — `attempts` (не
    `calls`): `calls` у FixedSummarizer — журнал текстов, его не затираем.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def summarize(self, text: str) -> str:
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("непредвиденный сбой суммаризации")
        return super().summarize(text)


@pytest.mark.asyncio
async def test_summary_loop_survives_unexpected_failure(fast) -> None:
    """Пул 4: RuntimeError в итерации summary-петли не убивает петлю —
    warning, повторный вызов суммаризатора, заметка в итоге суммаризована;
    stop() завершает петлю."""
    notes = NoteService(fast, FailingEmbedder())
    notes.save("заметка для проверки супервизора summary-петли", title="Заметка")
    summarizer = FlakySummarizer()
    worker = make_worker(fast, FailingEmbedder(), summarizer)
    task = asyncio.create_task(worker.run())
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and summarizer.attempts < 2:
        await asyncio.sleep(0.01)
    assert summarizer.attempts >= 2  # петля выжила после сбоя
    status = None
    while time.monotonic() < deadline:
        with session(fast) as conn:
            status = conn.execute(
                "SELECT summary_status FROM notes WHERE id = 1"
            ).fetchone()[0]
        if status == "ok":
            break
        await asyncio.sleep(0.01)
    assert status == "ok"  # повторная итерация довела суммаризацию
    worker.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# --- пул 5: гигиена worker_jobs (индекс, ретеншн, единократное DDL) ----------


def _jobs_ts(days_ago: float) -> str:
    """Timestamp в формате worker_jobs.updated_at, сдвинутый на days_ago суток."""
    moment = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _seed_worker_job(
    settings, note_id: int, status: str, updated_at: str, kind: str = "merge"
) -> None:
    """Вставить строку worker_jobs с явным status/updated_at."""
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO worker_jobs (slot, kind, note_id, status, updated_at) "
            "VALUES ('summary', ?, ?, ?, ?)",
            (kind, note_id, status, updated_at),
        )


def test_ensure_job_table_creates_queue_index_once(settings) -> None:
    """Пул 5: после _ensure_job_table индекс worker_jobs присутствует
    в sqlite_master; повторный вызов не создаёт дублей."""
    worker = make_worker(settings, HashEmbedder(8))
    worker._ensure_job_table()
    with session(settings) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND name = 'idx_worker_jobs_queue'"
        ).fetchall()
        assert [row["name"] for row in rows] == ["idx_worker_jobs_queue"]
    worker._ensure_job_table()  # повторный вызов — IF NOT EXISTS, дублей нет
    with session(settings) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' "
            "AND name = 'idx_worker_jobs_queue'"
        ).fetchone()[0]
        assert count == 1


def test_purge_done_jobs_removes_only_stale_done(settings, caplog) -> None:
    """Пул 5: ретеншн — очищаются только done-работы старше 7 суток;
    свежие done и pending не трогаются; событие jobs_purged в логе."""
    worker = make_worker(settings, HashEmbedder(8))
    worker._ensure_job_table()
    _seed_worker_job(settings, 1, "done", _jobs_ts(8))      # старая done → удалить
    _seed_worker_job(settings, 2, "done", _jobs_ts(0))      # свежая done → оставить
    _seed_worker_job(settings, 3, "pending", _jobs_ts(8))   # старый pending → оставить
    with caplog.at_level(logging.INFO, logger="app"):
        worker._purge_done_jobs()
    with session(settings) as conn:
        remaining = conn.execute(
            "SELECT note_id, status FROM worker_jobs ORDER BY note_id"
        ).fetchall()
    assert [(row["note_id"], row["status"]) for row in remaining] == [
        (2, "done"),
        (3, "pending"),
    ]
    purged = [
        r for r in caplog.records if getattr(r, "event", None) == "jobs_purged"
    ]
    assert purged and purged[-1].count == 1  # удалена ровно одна (старая done)


def test_purge_done_jobs_noop_when_nothing_stale(settings, caplog) -> None:
    """Пул 5: нет старых done — jobs_purged не логируется, строки целы."""
    worker = make_worker(settings, HashEmbedder(8))
    worker._ensure_job_table()
    _seed_worker_job(settings, 1, "done", _jobs_ts(0))
    with caplog.at_level(logging.INFO, logger="app"):
        worker._purge_done_jobs()
    with session(settings) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM worker_jobs"
        ).fetchone()[0] == 1
    assert not [
        r for r in caplog.records if getattr(r, "event", None) == "jobs_purged"
    ]


class BatchRecordEmbedder(HashEmbedder):
    """Хэш-кодировщик, фиксирующий размеры подъёмок embed_texts."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__(dim)
        self.batch_sizes: list[int] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return super().embed_texts(texts)


def test_process_pending_respects_embedding_batch_size(
    settings, monkeypatch
) -> None:
    """Пул 5: notes-петля без явного limit режется по EMBEDDING_BATCH_SIZE
    (а не фиксированной константой): подъёмки ровно по настроенному batch."""
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "3")
    get_settings.cache_clear()
    small = get_settings()  # тот же DB_PATH, batch=3
    notes = NoteService(small, FailingEmbedder())
    for number in range(5):
        notes.save(f"заметка {number} для batch notes-петли")
    embedder = BatchRecordEmbedder(8)
    worker = make_worker(small, embedder)
    assert worker.process_pending() == 3  # первая подъёмка ровно по batch
    assert embedder.batch_sizes == [3]
    assert worker.process_pending() == 2  # остаток — следующая подъёмка
    assert embedder.batch_sizes == [3, 2]
