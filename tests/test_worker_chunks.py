"""Фоновый воркер — чанковая очередь (Фаза 7, шаг 5).

Проверяются (brief §6): вычитывающая партия pending-чанков, подъёмки по
EMBEDDING_BATCH_SIZE (фейк-счётчик размеров партий), параллелизм до
EMBEDDING_CONCURRENT_REQUESTS (threading-барьер: три одновременных, потолок
не превышен), отказ подъёмки не портит остальных (NFR-3), reuse единственного
чанка ≤ CHUNK_SIZE из notes_vec без второго кодирования, защита от гонки с
memory_update (вектор старого чанка не пишется на заменённый — как
`AND text = ?` у суммари), чанковая петля в run (выгребает, сбрасывает
интервал, мягкий стоп).

Дедуп текстовых фикстур отключён (NoDedup): одинаковые длинные заметки
здесь — рабочий материал, дедуп-механика покрыта своими тестами (шаг 3).
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time

import pytest
from fakes import FailingEmbedder, HashEmbedder, vectorize_notes
from test_notes_chunks import CountingHashEmbedder, text_with_tokens

from app.config import get_settings
from app.services.embedding import EmbeddingError
from app.services.notes import NoteService
from app.services.splitter import count_tokens, token_windows
from app.services.worker import BackgroundWorker
from app.storage import chunks, vectors
from app.storage.db import init_db, session

DEFS = {"chunk_size": 1024, "chunk_overlap": 180, "chunk_min_target": 200}


class NoDedup:
    """Отключённый дедуп: тесты воркера не про дубликаты (шаг 3)."""

    def find_by_cosine(self, vector: list[float]):
        return None

    def find_by_text(self, text: str):
        return None


def make_notes(settings, embedder) -> NoteService:
    return NoteService(settings, embedder, NoDedup())


class BatchRecordingEmbedder(HashEmbedder):
    """HashEmbedder, помнящий размер каждой подъёмки (фейк-счётчик партий)."""

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.batch_sizes: list[int] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return super().embed_texts(texts)


class BarrierEmbedder(HashEmbedder):
    """Кодировщик-барьер: вызов ждёт, пока соберутся parties одновременных.

    Если параллелизм воркера меньше parties, барьер не соберётся: каждый
    вызов упрётся в таймаут и упадёт EmbeddingError (очередь не двигается —
    тест увидит нули). Активность трекается локом: потолок параллелизма
    тоже проверяется.
    """

    def __init__(self, dim: int, parties: int, timeout: float = 5.0) -> None:
        super().__init__(dim)
        self.barrier = threading.Barrier(parties)
        self.timeout = timeout
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self._lock = threading.Lock()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.barrier.wait(timeout=self.timeout)
        except threading.BrokenBarrierError as exc:  # не сошлись — штатный отказ
            raise EmbeddingError("подъёмки не собрались: параллелизма мало") from exc
        finally:
            with self._lock:
                self.active -= 1
        return super().embed_texts(texts)


class FailingAfterN(HashEmbedder):
    """Первые N вызовов embed_texts успешны, далее — EmbeddingError."""

    def __init__(self, dim: int, fail_after: int) -> None:
        super().__init__(dim)
        self._fail_after = fail_after
        self.calls = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls > self._fail_after:
            raise EmbeddingError("подъёмка не переживает этот прогон")
        return super().embed_texts(texts)


class MidFlightUpdateEmbedder(HashEmbedder):
    """Фейк-гонка: во время кодирования заметка обновляется (ARCH §4.5)."""

    def __init__(self, dim: int, notes, note_id: int, new_text: str) -> None:
        super().__init__(dim)
        self._notes = notes
        self._note_id = note_id
        self._new_text = new_text
        self.triggered = False

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.triggered:
            self.triggered = True
            self._notes.update(self._note_id, self._new_text)
        return super().embed_texts(texts)


# --- fixtures ------------------------------------------------------------------


def set_env(monkeypatch: pytest.MonkeyPatch, tmp_path, **overrides: str) -> None:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("MAX_NOTE_CHARS", "20000")
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Дефолты compose: batch 32, concurrent 3, interval 30."""
    set_env(monkeypatch, tmp_path)
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def batched(tmp_path, monkeypatch):
    """Подъёмки по 2 чанка, параллелизм 2 — вычитывающая партия 4."""
    set_env(
        monkeypatch,
        tmp_path,
        EMBEDDING_BATCH_SIZE="2",
        EMBEDDING_CONCURRENT_REQUESTS="2",
    )
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def single(tmp_path, monkeypatch):
    """Каждая подъёмка = 1 чанк, до 3 одновременных (юнит «3 одновременных»)."""
    set_env(
        monkeypatch,
        tmp_path,
        EMBEDDING_BATCH_SIZE="1",
        EMBEDDING_CONCURRENT_REQUESTS="3",
    )
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def chunk_fast(tmp_path, monkeypatch):
    """PENDING_RETRY_SEC=0 — цикл без пауз (asyncio-тесты петли)."""
    set_env(monkeypatch, tmp_path, PENDING_RETRY_SEC="0")
    settings = get_settings()
    init_db(settings)
    return settings


# --- партии и вычитывающая партия -----------------------------------------------


def test_long_note_chunks_vectorized_by_worker(settings) -> None:
    """Многочанковая заметка: воркер довекторизует все pending-чанки."""
    notes = make_notes(settings, HashEmbedder(8))
    text = text_with_tokens(3000)
    note_id = notes.save(text)["id"]
    expected = len(token_windows(3000, **DEFS))
    worker = BackgroundWorker(settings, HashEmbedder(8))
    assert asyncio.run(worker.process_pending_chunks()) == expected
    hash_ = HashEmbedder(8)
    with session(settings) as conn:
        rows = chunks.get_note_chunks(conn, note_id)
        assert len(rows) == expected
        assert chunks.count_pending(conn) == 0
        for chunk_id, _idx, chunk_text, _tokens in rows:
            assert chunks.get_vector(conn, chunk_id) == pytest.approx(
                hash_.embed(chunk_text), abs=1e-6
            )


def test_batches_of_batch_size_by_fake_counter(settings) -> None:
    """Юнит брифа «батчи по 32»: 36 pending-чанков → подъёмки [32, 4]."""
    notes = make_notes(settings, HashEmbedder(8))
    expected = 0
    for _number in range(9):  # 9 заметок × 4 чанка = 36 (дедуп отключён)
        expected += len(token_windows(3000, **DEFS))
        notes.save(text_with_tokens(3000))
    assert expected == 36
    recorder = BatchRecordingEmbedder(8)
    worker = BackgroundWorker(settings, recorder)
    assert asyncio.run(worker.process_pending_chunks()) == expected
    assert recorder.batch_sizes == [32, 4]
    with session(settings) as conn:
        assert chunks.count_pending(conn) == 0
        assert chunks.count_vectors(conn) == expected


def test_note_vector_queue_untouched_by_chunk_queue(settings) -> None:
    """Чанковая петля не трогает notes-статусы: полный вектор пишет только
    notes-очередь (Фаза 8: после save он ещё pending)."""
    notes = make_notes(settings, HashEmbedder(8))
    note_id = notes.save(text_with_tokens(3000))["id"]
    assert vectorize_notes(settings, HashEmbedder(8)) == 1  # Фаза 8: очередь notes
    worker = BackgroundWorker(settings, HashEmbedder(8))
    assert asyncio.run(worker.process_pending_chunks()) == 4
    with session(settings) as conn:
        row = conn.execute(
            "SELECT vector_status FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
        assert vectors.count(conn) == 1  # полный вектор на месте (от save)
        assert row["vector_status"] == "ok"  # чанковой петлёй не пере-писан


# --- параллелизм -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_three_concurrent_requests(single) -> None:
    """Юнит брифа «3 одновременных»: барьер на троих собирается, потолок 3."""
    notes = make_notes(single, HashEmbedder(8))
    for _ in range(3):
        notes.save(text_with_tokens(1500))  # 2 чанка каждая — 6 pending
    embedder = BarrierEmbedder(8, parties=3)
    worker = BackgroundWorker(single, embedder)
    # limit=6: дефолтная вычитывающая партия тут 3 (1×3) — берём всё сразу.
    assert await worker.process_pending_chunks(limit=6) == 6
    assert embedder.calls == 6
    assert embedder.max_active <= 3  # потолок EMBEDDING_CONCURRENT_REQUESTS
    with session(single) as conn:
        assert chunks.count_pending(conn) == 0
        assert chunks.count_vectors(conn) == 6


@pytest.mark.asyncio
async def test_barrier_breaks_without_enough_concurrency(batched) -> None:
    """Параллелизма мало двух барьеру на троих: обе подъёмки виснут на барьере,
    конвертируются в EmbeddingError, очередь остаётся pending."""
    notes = make_notes(batched, HashEmbedder(8))
    for _ in range(2):
        notes.save(text_with_tokens(1500))  # 2+2 чанка
    embedder = BarrierEmbedder(8, parties=3, timeout=0.5)
    worker = BackgroundWorker(batched, embedder)
    assert await worker.process_pending_chunks() == 0
    with session(batched) as conn:
        assert chunks.count_pending(conn) == 4
        assert chunks.count_vectors(conn) == 0


def test_drain_rounds_of_batch_times_concurrency(batched) -> None:
    """Вычитывающая партия = EMBEDDING_BATCH_SIZE × CONCURRENT_REQUESTS:
    6 pending-чанков выгребаются за 4 + 2, потом пусто."""
    notes = make_notes(batched, HashEmbedder(8))
    for _ in range(3):
        notes.save(text_with_tokens(1500))
    worker = BackgroundWorker(batched, HashEmbedder(8))
    assert asyncio.run(worker.process_pending_chunks()) == 4
    assert asyncio.run(worker.process_pending_chunks()) == 2
    assert asyncio.run(worker.process_pending_chunks()) == 0


# --- отказы, деградация ------------------------------------------------------------


def test_encoding_failure_keeps_chunk_queue_pending(settings) -> None:
    """Полный отказ кодирования — 0 записей, чанки остаются pending."""
    notes = NoteService(settings, FailingEmbedder())  # save: заметка+чанки pending
    notes.save(text_with_tokens(1500))
    worker = BackgroundWorker(settings, FailingEmbedder())
    assert asyncio.run(worker.process_pending_chunks()) == 0
    with session(settings) as conn:
        assert chunks.count_pending(conn) == 2
        assert chunks.count_vectors(conn) == 0


def test_partial_failure_writes_success_batches(batched) -> None:
    """Отказ подъёмки не отменяет остальных (NFR-3): успешные партия
    записывается, отказавшая догоняется следующим прогоном."""
    notes = make_notes(batched, HashEmbedder(8))
    for _ in range(2):
        notes.save(text_with_tokens(1500))  # 2+2 чанка
    worker = BackgroundWorker(batched, FailingAfterN(8, fail_after=1))
    assert asyncio.run(worker.process_pending_chunks()) == 2  # первая подъёмка готова
    with session(batched) as conn:
        assert chunks.count_vectors(conn) == 2
        assert chunks.count_pending(conn) == 2
    worker._embedding = HashEmbedder(8)
    assert asyncio.run(worker.process_pending_chunks()) == 2
    with session(batched) as conn:
        assert chunks.count_pending(conn) == 0
        assert chunks.count_vectors(conn) == 4


# --- reuse единичного чанка ---------------------------------------------------------


def test_reuse_full_vector_for_single_chunk(settings) -> None:
    """Brief §6: 1 чанк ≤ CHUNK_SIZE + notes_vec готов ПОЗЖЕ save (reuse
    шага 3 не сработал — вектора в момент записи не было) → вектор чанка
    копируется из полного, кодировщик не зовётся."""
    notes = NoteService(settings, FailingEmbedder())
    text = "Короткая заметка, векторизированная только в очереди полных векторов"
    note_id = notes.save(text)["id"]
    notes_queue = BackgroundWorker(settings, HashEmbedder(8))
    assert notes_queue.process_pending() == 1  # notes_vec стал ok уже после save
    chunker = CountingHashEmbedder(8)
    worker = BackgroundWorker(settings, chunker)
    assert asyncio.run(worker.process_pending_chunks()) == 1
    assert chunker.texts == []  # кодирование не звалось — копия из notes_vec
    with session(settings) as conn:
        rows = chunks.get_note_chunks(conn, note_id)
        assert chunks.count_pending(conn) == 0
        chunk_vector = chunks.get_vector(conn, rows[0][0])
        note_vector = vectors.get_vector(conn, note_id)
    assert chunk_vector == pytest.approx(note_vector, abs=1e-7)


def test_merged_single_chunk_above_chunk_size_is_encoded(settings) -> None:
    """1 чанк, но > CHUNK_SIZE (1025 токенов) — reuse НЕ применяется,
    чанк кодируется воркером как прочие (условие брифа «≤ CHUNK_SIZE»)."""
    notes = make_notes(settings, HashEmbedder(8))
    text = text_with_tokens(1025)
    note_id = notes.save(text)["id"]
    counter = CountingHashEmbedder(8)
    worker = BackgroundWorker(settings, counter)
    assert asyncio.run(worker.process_pending_chunks()) == 1
    with session(settings) as conn:
        rows = chunks.get_note_chunks(conn, note_id)
        assert len(rows) == 1
        assert chunks.count_pending(conn) == 0
    assert counter.texts == [rows[0][2]]  # кодировался чанк, не копия


# --- гонка с memory_update ------------------------------------------------------------


def test_race_replaced_chunks_receive_no_stale_vector(settings) -> None:
    """Гонка с memory_update (ARCH §4.5): update заменил чанки в полёте на
    другие (новые id/text/tokens) — вектор вычитанного текста не пишется
    (аналог `AND text = ?`), сироты не рождаются; новые чанки догоняются
    следующим прогоном."""
    notes = make_notes(settings, HashEmbedder(8))
    text = text_with_tokens(1500)
    note_id = notes.save(text)["id"]
    new_text = "Совершенно другой текст заметки для замены в фейк-гонке. " + text_with_tokens(1600)
    expected = len(token_windows(count_tokens(new_text), **DEFS))
    worker = BackgroundWorker(
        settings, MidFlightUpdateEmbedder(8, notes, note_id, new_text)
    )
    # чанки заменены во время кодирования — ни один вектор не записан
    assert asyncio.run(worker.process_pending_chunks()) == 0
    with session(settings) as conn:
        assert chunks.count_vectors(conn) == 0  # вектор старого текста не записан
        assert chunks.count_chunks(conn) == expected  # новые чанки на месте
        assert chunks.count_pending(conn) == expected
    worker._embedding = HashEmbedder(8)
    assert asyncio.run(worker.process_pending_chunks()) == expected
    with session(settings) as conn:
        assert chunks.count_vectors(conn) == expected
        assert chunks.count_pending(conn) == 0


# --- петля run() ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chunk_loop_drains_queue_and_resets_interval(chunk_fast) -> None:
    """Третья петля воркера собирает чанковую очередь и сбрасывает интервал."""
    notes = make_notes(chunk_fast, HashEmbedder(8))
    notes.save(text_with_tokens(1500))  # 2 pending-чанка
    worker = BackgroundWorker(chunk_fast, HashEmbedder(8))
    task = asyncio.create_task(worker.run())
    vectorized = 0
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        with session(chunk_fast) as conn:
            vectorized = chunks.count_vectors(conn)
        if vectorized == 2:
            break
        await asyncio.sleep(0.01)
    worker.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert vectorized == 2
    # успех сбросил интервал чанковой очереди к стартовому
    assert worker.chunk_interval == float(chunk_fast.pending_retry_sec)


def test_chunk_interval_starts_at_configured(settings) -> None:
    """Своя независимая очередь = свой счётчик back-off (ARCH §3.4)."""
    worker = BackgroundWorker(settings, HashEmbedder(8))
    assert worker.chunk_interval == float(settings.pending_retry_sec)