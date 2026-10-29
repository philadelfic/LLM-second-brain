"""Джоба `embedding` на каркасе lsb-0014: форма «по интервалу + событие».

Gate 2026-09-19 (продолжение решения гейта 1c по `links`): запись/правка
заметки (`memory_save`/`memory_update`), слияние дублей (`merge_pair`) и
догенерация названия возвращают заметку в `vector_status='pending'` и будят
embedding-петлю сигналом `notify_embedding_pending`. Это закрывает две проблемы:

* FR-2.3 (наблюдаемость очередей): раньше событие `queue_waiting` появлялось
  только когда петля случайно проснётся по выросшему back-off и войдёт в
  ожидание с непустой очередью — в окне приёмки (десятки секунд) его могло не
  быть; теперь свежая pending-заметка будит петлю, и уход в сон с непустой
  очередью пишется сразу (детерминированно);
* свежая заметка векторизуется немедленно, а не через выросший back-off
  (до 15 мин) — модель не видит её как `pending` дольше необходимого, поиск
  находит её сразу.

Контракт надёжности сохранён: задание живёт в самой заметке
(`vector_status='pending'`), событие только ускоряет; при недоступных моделях
(`EmbeddingError`) сигнала нет и петля честно ждёт по back-off — перепроверка
очереди (`queue_empty` → `_embedding_queue_empty`, LSBN-T-30) этого не ломает:
отказ — не прогресс петли (arch §3.2, no39-05), на интервал ровно два прогона
(обычный + немедленный lost-wakeup, пул 6), дальше сон. Перепроверка страхует
сигнал, стёртый `clear()` (lost wakeup), и здесь же проверяются её драйверы:
разрыв embedding-слота делает очередь пустой для петли, pending-чанк —
берущейся работой.

Юниты на хосте: HashEmbedder/фейки, сеть не нужна.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time

import pytest
from fakes import FailingEmbedder, HashEmbedder

from app.config import Settings, get_settings
from app.main import create_app
from app.services.embedding import EmbeddingError
from app.services.jobs import BackoffState, build_job_specs, run_loop
from app.services.notes import NoteService
from app.services.slot_outcome import SlotOutcome
from app.services.worker import EMBEDDING_JOB, BackgroundWorker
from app.storage.db import init_db, session, transaction

DIM = 8


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Схема в тестовой БД; размерность 8 — вектора маленькие, сеть не нужна."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def make_worker(settings: Settings, embedding=None) -> BackgroundWorker:
    """Воркер без суммаризатора: embedding-джоба от него не зависит."""
    return BackgroundWorker(settings, embedding or HashEmbedder(DIM))


def embedding_spec(worker: BackgroundWorker, settings: Settings):
    """Описание джобы `embedding` из реестра каркаса."""
    specs = {spec.name: spec for spec in build_job_specs(worker, settings)}
    return specs[EMBEDDING_JOB]


def _seed_pending(settings: Settings, note_id: int = 1) -> None:
    """Заметка с `vector_status='pending'` прямым SQL (очередь векторизации)."""
    with session(settings) as conn:
        conn.execute(
            "INSERT INTO notes (id, title, text, namespace, vector_status) "
            "VALUES (?, 'Заметка', 'текст про встречу', 'default', 'pending')",
            (note_id,),
        )


def _vector_pending(settings: Settings) -> int:
    """Число pending по очереди векторизации прямым SQL — эталон джобы."""
    with session(settings) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS pending FROM notes "
            "WHERE deleted_at IS NULL AND vector_status = 'pending'"
        ).fetchone()
    return int(row["pending"])


class TransientFailEmbedder(FailingEmbedder):
    """FailingEmbedder с транзиентным исходом UNAVAILABLE (arch §13.2).

    Транзиентный отказ разрыв слота embedding НЕ открывает
    (`SlotBreaker._TRANSIENT_OUTCOMES`) — ворота партию берут, и картина
    итерации петли чистая: два прогона, потом сон по back-off (сценарий
    разрыва слота — в test_worker_queue_unavailable_models.py).
    """

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        raise EmbeddingError(self.message, outcome=SlotOutcome.UNAVAILABLE)


def patch_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Мгновенный `asyncio.sleep` с записью запрошенных пауз.

    С перепроверкой очереди (LSBN-T-30) итерация с непустой очередью до
    `wait_for` не доходит — петля спит `asyncio.sleep(back-off)`: реальную
    паузу (30 с) тест не переживёт — сон подменяется мгновенным yield'ом,
    запрошенная пауза записана (как `patch_sleep` в test_jobs).
    """
    real_sleep = asyncio.sleep
    delays: list[float] = []

    async def spy(delay: float, *args: object, **kwargs: object) -> None:
        delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", spy)
    return delays


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    """Дождаться условия живого цикла (опрос с yield'ами)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("условие не наступило за отведённое время")


def _queue_waiting_records(caplog) -> list:
    """События `queue_waiting` из журнала (FR-2.3)."""
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "queue_waiting"
    ]


# --- регистрация: форма «по интервалу + событие» ------------------------------


def test_embedding_job_has_event_and_queue_recheck(settings: Settings) -> None:
    """Регистрация: интервал с back-off + событие `embedding` + `queue_empty`."""
    worker = make_worker(settings)
    spec = embedding_spec(worker, settings)
    assert spec.queue == "vector"
    assert spec.interval_sec == settings.pending_retry_sec
    # Событие берётся у воркера (владелец очереди векторизации), save/update
    # ставят его через NoteService.
    assert spec.wait_event is worker._embedding_event
    # Перепроверка очереди (LSBN-T-30) страхует сигнал, стёртый `clear()`
    # (пул 6, lost wakeup); холостого шторма при недоступном кодировщике она
    # не даёт — отказ не считается прогрессом (no39-05), после одного
    # немедленного прогона петля уходит в сон по back-off.
    assert spec.queue_empty == worker._embedding_queue_empty
    # Гигиена worker_jobs осталась в idle-ветке.
    assert spec.idle_hook == worker._purge_done_jobs
    assert spec.queue_stat is not None


def test_main_wires_vector_notifier(settings: Settings) -> None:
    """main.py подключает сигнал: запись заметки будит embedding-петлю воркера."""
    app = create_app()
    notes = app.state.services.notes
    assert notes._vector_notifier == app.state.worker.notify_embedding_pending


# --- сигнал: запись заметки будит петлю --------------------------------------


def test_save_signals_vector_pending(settings: Settings) -> None:
    """`save` сигналит ровно один раз; дословный дубль — сигнала нет."""
    signals: list[int] = []
    notes = NoteService(
        settings, FailingEmbedder(), vector_notifier=lambda: signals.append(1)
    )
    notes.save("текст новой заметки про встречу")
    assert signals == [1]  # заметка записана с vector_status='pending'
    notes.save("текст новой заметки про встречу")  # дословный дубль
    assert signals == [1]  # очередь не пополнилась — будить нечего


def test_update_and_merge_signal_vector_pending(settings: Settings) -> None:
    """Правка текста и слияние дублей сигналят; правка без текста — нет."""
    signals: list[int] = []
    notes = NoteService(
        settings, FailingEmbedder(), vector_notifier=lambda: signals.append(1)
    )
    older = notes.save("ранний текст заметки про встречу")["id"]
    newer = notes.save("поздний текст заметки про встречу")["id"]
    signals.clear()

    notes.update(older, text="исправленный текст заметки про встречу")
    assert signals == [1]  # текст заменён → vector_status='pending'

    signals.clear()
    notes.update(older, summary="готовая суммари")
    assert signals == []  # текст не тронут — очередь векторизации не менялась

    merged = notes.merge_pair(older, "слитый текст заметки про встречу", newer)
    assert merged["merged"] is True
    assert signals == [1]  # сшивание вернуло раннюю заметку в очередь вектора


@pytest.mark.asyncio
async def test_save_wakes_loop_and_logs_queue_waiting(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """Свежая pending-заметка будит петлю немедленно (интервал 30 с не ждём).

    Первый прогон (пустая очередь) уводит петлю в сон на `PENDING_RETRY_SEC`;
    `save` через NoteService ставит сигнал — петля просыпается сразу, прогон
    пуст (модель недоступна) и уход в сон с непустой очередью пишет
    `queue_waiting` — FR-2.3 виден детерминированно, без случайного пробуждения
    по back-off.
    """
    worker = make_worker(settings, FailingEmbedder())
    spec = embedding_spec(worker, settings)
    notes = NoteService(
        settings, FailingEmbedder(),
        vector_notifier=worker.notify_embedding_pending,
    )
    runs: list[int] = []

    def counting(self, limit=None) -> int:
        runs.append(1)
        return 0  # модель недоступна — статусы не тронуты

    monkeypatch.setattr(BackgroundWorker, "process_pending", counting)
    with caplog.at_level(logging.INFO, logger="app"):
        task = asyncio.create_task(
            run_loop(spec, lambda: False, BackoffState(spec.interval_sec))
        )
        try:
            await _wait_until(lambda: len(runs) >= 1)
            await asyncio.sleep(0.05)
            # Петля дошла до ожидания: очередь пуста — `queue_waiting` молчит.
            assert _queue_waiting_records(caplog) == []
            notes.save("свежая заметка про встречу", title="Свежая заметка")
            started = time.monotonic()
            await _wait_until(lambda: len(runs) >= 2, timeout=1.0)
            assert time.monotonic() - started < 1.0  # не 30 с back-off
            await _wait_until(
                lambda: _queue_waiting_records(caplog), timeout=1.0
            )
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    record = _queue_waiting_records(caplog)[0]
    assert (record.job, record.queue) == (EMBEDDING_JOB, "vector")
    assert record.pending == 1
    assert spec.queue_stat()["pending"] == _vector_pending(settings) == 1


@pytest.mark.asyncio
async def test_embedding_error_sends_no_signal_and_grows_backoff(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """Отказ кодирования: сигнала нет, петля ждёт по back-off (не busy-loop).

    Заметка pending (модели недоступны) — событие не выставляется; итерация —
    ровно ДВА прогона (обычный + немедленный lost-wakeup перепроверки,
    LSBN-T-30), дальше пауза с ростом интервала: 30 → 60 с. Холостого шторма
    нет: отказ — не прогресс петли (arch §3.2, no39-05).
    """
    _seed_pending(settings)
    worker = make_worker(settings, TransientFailEmbedder())
    spec = embedding_spec(worker, settings)
    real_wait_for = asyncio.wait_for
    sleeps = patch_sleep(monkeypatch)
    runs: list[int] = []

    def counting(self, limit=None) -> int:
        runs.append(1)
        return 0  # модель недоступна — статусы не тронуты

    monkeypatch.setattr(BackgroundWorker, "process_pending", counting)
    with caplog.at_level(logging.INFO, logger="app"):
        await real_wait_for(
            run_loop(spec, lambda: len(sleeps) >= 2, BackoffState(spec.interval_sec)),
            timeout=2.0,
        )
    assert runs == [1, 1, 1, 1]  # два прогона на итерацию, не десятки (нет шторма)
    assert sleeps == [30.0, 60.0]  # back-off растёт: петля спит, а не крутится
    assert worker._embedding_event.is_set() is False  # EmbeddingError — сигнала нет
    assert spec.queue_stat()["pending"] == _vector_pending(settings) == 1
    # «Работа есть, но она не выполняется» (FR-2.3) — видно в журнале.
    assert _queue_waiting_records(caplog)


# --- драйвер петли `_embedding_queue_empty` (LSBN-T-30) -----------------------


def test_slot_breaker_makes_queue_empty(settings: Settings) -> None:
    """Разрыв embedding-слота — очередь для петли пуста даже с pending-работой.

    Петля спит по back-off и модель не зовёт (no39-05): ворота партию не
    берут, считать непустую очередь работой — значило бы крутить busy-loop.
    """
    _seed_pending(settings)
    worker = make_worker(settings, HashEmbedder(DIM))
    assert worker._embedding_queue_empty() is False  # работа берущаяся
    # AUTH-отказ открывает разрыв сразу (SlotBreaker, FR-4.3).
    worker._breakers["embedding"].note_failure(SlotOutcome.AUTH)
    assert worker._breakers["embedding"].is_paused() is True
    assert worker._embedding_queue_empty() is True


def test_pending_chunk_counts_as_work(settings: Settings) -> None:
    """Pending-чанк без pending-заметок — берущаяся работа (queue_empty False).

    Чанковая очередь — та же embedding-джоба (решение №10), предикат pending —
    тот же, что у вычитки `_read_chunk_batch`: чанк без строки в
    `notes_chunks_vec` (анти-джойн). Без чанка и без pending-заметок очередь
    пуста.
    """
    worker = make_worker(settings, HashEmbedder(DIM))
    assert worker._embedding_queue_empty() is True  # пусто: нет ни заметок, ни чанков
    with session(settings) as conn, transaction(conn):
        # Заметка с готовым вектором (не работа), но чанк без вектора — работа.
        conn.execute(
            "INSERT INTO notes (id, title, text, namespace, vector_status) "
            "VALUES (1, 'Заметка', 'текст про встречу', 'default', 'ok')"
        )
        conn.execute(
            "INSERT INTO notes_chunks (note_id, idx, text, tokens) "
            "VALUES (1, 0, 'текст про встречу', 3)"
        )
    assert worker._embedding_queue_empty() is False
