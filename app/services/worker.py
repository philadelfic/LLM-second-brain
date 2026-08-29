"""Фоновый воркер (ARCHITECTURE §3.4): до-векторизация + до-суммаризация.

Единственный воркер на процесс, **три независимые очереди** — состояния
pending-статусов в БД (переживают рестарт, догоняются при старте сервиса):
- `pending_vector`  → batch `embed_texts` (один вызов на партию) → вектора в
  notes_vec, vector_status='ok';
- `pending_summary` → по заметке `summarizer.summarize(text)` (режим «Б» —
  единственный путь генерации summary, §5.5) → summary + summary_status='ok';
- pending-чанки (Фаза 7, анти-джойн «нет строки в notes_chunks_vec») →
  вычитывающая партия режется на подъёмки EMBEDDING_BATCH_SIZE; в полёте —
  до EMBEDDING_CONCURRENT_REQUESTS подъёмок (Semaphore + asyncio.to_thread).

Garanties:
- отказ любого внешнего сервера не портит данные: статус остаётся pending
  (NFR-3); интервал опроса растёт по back-off: PENDING_RETRY_SEC (30 с) → ×2
  → max 15 минут (REQUIREMENTS §5.3) — **независимо по каждой очереди**
  (ARCH §3.4): недоступный векторизатор не останавливает суммаризацию и
  наоборот; успех в очереди сбрасывает только её интервал и продолжает
  выгребать её немедленно.
- конкурентный доступ — через busy_timeout/WAL (§3.3); каждая партия —
  короткие транзакции, параллель с запросами безопасна.
- `process_*` синхронные (выполняются в `asyncio.to_thread` — event loop не
  занимаем); чанковая очередь Фазы 7 — своя async-обработка: кодирование
  подъёмок размножается Semaphore'ом прямо в петле, транзакции записи —
  короткие в to_thread. `run` — asyncio-таска (все петли под gather),
  старт/стоп — в lifespan.

Запись суммари защищена от гонки с memory_update: между вычиткой текста и
записью воркер мог получить обновлённый текст — UPDATE ограничен условием
`AND text = ?` (тот же текст; иначе суммари протухшего текста затёрло бы
свежий). В векторизации (Фаза 3) ре-векторизация синхронна в update —
вектор пишется по актуальному на момент записи тексту; расхождение «текст
менялся между вычиткой и записью» чинится следующей партией.

Чанковая очередь (Фаза 7) закрывает обе проблемы:
- reuse единичного чанка (brief §6): у заметки с ровно одним чанком
  ≤ CHUNK_SIZE вектор чанка = вектор полного текста из notes_vec, без вызова
  кодировщика. Это случай, когда reuse шага 3 не успел примениться при
  save — в момент записи полного вектора ещё не было (отказ Ollama);
  кодировать тот же текст второй раз незачем.
- гонка с update (ARCH §4.5, аналог `AND text = ?`): вектор чанка пишется
  только если (id, text, tokens) не менялись с вычитки — update мог
  заменить чанки (DELETE+INSERT, id при повторной вставке переиспользуются,
  что дало бы вектор чужого текста на новом id).

Статусы внешних серверов (для `/health.*_ok`, NFR-4) ведут сами сервисы —
воркер не агрегирует: все кодирования идут через EmbeddingService
(`embedding_ok` обновляется в `embed_texts` — единой точке кодирования), все
генерации — через Summarizer.

Суммаризатор инъектируется DI (build_services): None — петля не запускается
(тестовый режим Фазы 3); в проде всегда передан SummaryService.
"""

from __future__ import annotations

import asyncio

from app.config import Settings
from app.services.embedding import Embedder, EmbeddingError
from app.services.summary import Summarizer, SummaryError
from app.storage import chunks, vectors
from app.storage.db import session, transaction

# Заметок за один прогон (embed_texts — batch; summary — по одной, модель
# суммаризации тяжёлая): очередь выгребается последовательными партиями.
PENDING_BATCH = 50

# Потолок back-off (REQUIREMENTS §5.3 «max 15 мин»), env не настраивается.
MAX_INTERVAL_SEC = 15 * 60


def next_interval(current: float, start: int) -> float:
    """Шаг back-off: интервал удваивается, потолок — 15 минут (§3.4)."""
    return min(max(current * 2.0, float(start)), float(MAX_INTERVAL_SEC))


class BackgroundWorker:
    """Единственный фоновый воркер; очереди — pending-статусы в БД."""

    def __init__(
        self,
        settings: Settings,
        embedding: Embedder,
        summarizer: Summarizer | None = None,
    ) -> None:
        self._settings = settings
        self._embedding = embedding
        self._summarizer = summarizer
        self._vector_interval = float(max(settings.pending_retry_sec, 0))
        self._summary_interval = float(max(settings.pending_retry_sec, 0))
        self._chunk_interval = float(max(settings.pending_retry_sec, 0))
        self._stopping = False

    @property
    def interval(self) -> float:
        """Текущий интервал векторной очереди (диагностика, тесты)."""
        return self._vector_interval

    @property
    def summary_interval(self) -> float:
        """Текущий интервал суммаризационной очереди (диагностика, тесты)."""
        return self._summary_interval

    @property
    def chunk_interval(self) -> float:
        """Текущий интервал чанковой очереди (диагностика, тесты, Фаза 7)."""
        return self._chunk_interval

    def stop(self) -> None:
        """Мягкая остановка: петли завершатся после разборки текущей итерации."""
        self._stopping = True

    async def run(self) -> None:
        """Все петли очередей (запускается asyncio-таской при старте).

        Обработанные партии идут одна за другой (очередь выгребаем сразу);
        пустой прогон — пауза на текущий интервал очереди с удвоением.
        Петли независимы: back-off и выгребание — раздельные.
        """
        await asyncio.gather(
            self._run_vector(), self._run_summary(), self._run_chunks()
        )

    async def _run_vector(self) -> None:
        while not self._stopping:
            processed = await asyncio.to_thread(self.process_pending)
            if processed:
                self._vector_interval = float(
                    self._settings.pending_retry_sec
                )  # успех — сброс
                continue
            await asyncio.sleep(self._vector_interval)
            self._vector_interval = next_interval(
                self._vector_interval, self._settings.pending_retry_sec
            )

    async def _run_summary(self) -> None:
        if self._summarizer is None:
            return  # тестовый режим без суммаризатора: петля не нужна
        while not self._stopping:
            processed = await asyncio.to_thread(self.process_summary_pending)
            if processed:
                self._summary_interval = float(self._settings.pending_retry_sec)
                continue
            await asyncio.sleep(self._summary_interval)
            self._summary_interval = next_interval(
                self._summary_interval, self._settings.pending_retry_sec
            )

    async def _run_chunks(self) -> None:
        while not self._stopping:
            processed = await self.process_pending_chunks()
            if processed:
                self._chunk_interval = float(self._settings.pending_retry_sec)
                continue
            await asyncio.sleep(self._chunk_interval)
            self._chunk_interval = next_interval(
                self._chunk_interval, self._settings.pending_retry_sec
            )

    # --- синхронная работа (выполняется в to_thread) --------------------------

    def process_pending(self, limit: int = PENDING_BATCH) -> int:
        """Векторизовать одну партию pending; возвращает число обработанных.

        Отказ кодирования — 0: статусы не тронуты, воркер выждет back-off.
        """
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, text FROM notes "
                "WHERE vector_status = 'pending' AND deleted_at IS NULL "
                "ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
        if not rows:
            return 0
        try:
            embeddings = self._embedding.embed_texts([row["text"] for row in rows])
        except EmbeddingError:
            return 0
        for row, vector in zip(rows, embeddings):
            with session(self._settings) as conn, transaction(conn):
                vectors.upsert(conn, row["id"], vector)
                conn.execute(
                    "UPDATE notes SET vector_status = 'ok' WHERE id = ?",
                    (row["id"],),
                )
        return len(rows)

    def process_summary_pending(self, limit: int = PENDING_BATCH) -> int:
        """Досуммировать одну партию pending; число до 'ok' доведённых.

        Режим «Б» (§5.5): генерация — только здесь, по заметкам из очереди.
        Отказ генерации одной заметки не отменяет остальных (NFR-3): статус
        остаётся pending, заметка догонится следующей партией. Trash
        (deleted_at IS NOT NULL) не обслуживается — как и в векторизации.

        Гонка с memory_update (ARCH §4.5): суммари пишется только если текст
        не менялся с момента вычитки (`AND text = ?`) — протухшая выжимка не
        затирает свежую заметку.
        """
        if self._summarizer is None:
            return 0
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, text FROM notes "
                "WHERE summary_status = 'pending' AND deleted_at IS NULL "
                "ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
        done = 0
        for row in rows:
            try:
                summary = self._summarizer.summarize(row["text"])
            except SummaryError:
                continue  # отказ: status pending остаётся, повтор по back-off
            with session(self._settings) as conn, transaction(conn):
                cursor = conn.execute(
                    "UPDATE notes SET summary = ?, summary_status = 'ok' "
                    "WHERE id = ? AND summary_status = 'pending' AND text = ?",
                    (summary, row["id"], row["text"]),
                )
            if cursor.rowcount:
                done += 1
        return done

    # --- чанковая очередь (Фаза 7) ---------------------------------------------

    async def process_pending_chunks(self, limit: int | None = None) -> int:
        """Векторизовать одну вычитывающую партию pending-чанков (Фаза 7).

        Вычитывающая партия — EMBEDDING_BATCH_SIZE ×
        EMBEDDING_CONCURRENT_REQUESTS чанков (анти-джойн, старые первыми);
        она режется на подъёмки по EMBEDDING_BATCH_SIZE, а кодирование
        подъёмок идёт параллельно — не больше EMBEDDING_CONCURRENT_REQUESTS
        одновременно (Semaphore + asyncio.to_thread: кодирование —
        блокирующий HTTP, event loop не занимаем).

        Отказ подъёмки (EmbeddingError) не портит остальных (NFR-3): вектора
        успешных подъёмок записываются короткими транзакциями, отказавшие
        чанки остаются pending — догонятся следующим прогоном.
        Полный отказ — 0, воркер ждёт back-off; `embedding_ok` ведёт сам
        EmbeddingService (единая точка кодирования — как и в других петлях).

        Reuse (brief §6): у заметки с ровно одним чанком ≤ CHUNK_SIZE вектор
        чанка = вектор полного текста из notes_vec, без вызова кодировщика.
        Гонка с memory_update (ARCH §4.5) — как у суммари: запись только при
        неизменных (id, text, tokens) чанка; заменённые в полёте пропускаются.

        Возвращает число фактически записанных векторов (reuse + успешные
        подъёмки; гонно заменённые чанки не считаются).
        """
        batch = self._settings.embedding_batch_size
        drain = (
            limit
            if limit is not None
            else batch * self._settings.embedding_concurrent_requests
        )
        rows = await asyncio.to_thread(self._read_chunk_batch, drain)
        if not rows:
            return 0
        reused = await asyncio.to_thread(self._reuse_single_chunk_vectors, rows)
        remaining = [row for row in rows if row["id"] not in reused]
        batches = [remaining[i : i + batch] for i in range(0, len(remaining), batch)]
        results = await self._encode_batches(batches)
        written = await asyncio.to_thread(self._store_chunk_vectors, batches, results)
        return len(reused) + written

    def _read_chunk_batch(self, limit: int) -> list:
        """Вычитка партии pending-чанков (старые первыми), короткое чтение."""
        with session(self._settings) as conn:
            return chunks.pending_chunk_rows(conn, limit)

    def _reuse_single_chunk_vectors(self, rows: list) -> set[int]:
        """Копировать полный вектор заметки в её единственный чанк ≤ CHUNK_SIZE.

        Шаг 3 применяет reuse только при готовом notes_vec в момент save;
        если полный вектор достроен позже (отказ при save — потом воркер
        notes-очереди), чанк остаётся pending. Повторно кодировать тот же
        текст незачем: у одного чанка текст = полный текст заметки, вектор
        идентичен — копируем без вызова Ollama.
        """
        by_note: dict[int, list] = {}
        for row in rows:
            by_note.setdefault(row["note_id"], []).append(row)
        reused: set[int] = set()
        with session(self._settings) as conn:
            for note_id, note_rows in by_note.items():
                if len(note_rows) != 1:
                    continue  # у заметки в партии несколько pending-чанков
                count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM notes_chunks WHERE note_id = ?",
                        (note_id,),
                    ).fetchone()[0]
                )
                if count != 1 or note_rows[0]["tokens"] > self._settings.chunk_size:
                    continue
                full = vectors.get_vector(conn, note_id)
                if full is None:
                    continue  # полного вектора нет — чанк кодируется наравне
                with transaction(conn):
                    if chunks.upsert_vector_if_exists(
                        conn,
                        note_rows[0]["id"],
                        full,
                        note_rows[0]["text"],
                        note_rows[0]["tokens"],
                    ):
                        reused.add(note_rows[0]["id"])
        return reused

    async def _encode_batches(self, batches: list[list]) -> list:
        """Параллельное кодирование подъёмок; каждый вызов — embed_texts batch.

        Не более EMBEDDING_CONCURRENT_REQUESTS подъёмок одновременно.
        Возврат выровнен по подъёмкам: список векторов или исключение
        (EmbeddingError — штатный отказ; остальные — непредвиденный сбой).
        """
        semaphore = asyncio.Semaphore(self._settings.embedding_concurrent_requests)

        async def encode(batch: list) -> list[list[float]]:
            async with semaphore:
                return await asyncio.to_thread(
                    self._embedding.embed_texts, [row["text"] for row in batch]
                )

        return list(await asyncio.gather(*map(encode, batches), return_exceptions=True))

    def _store_chunk_vectors(self, batches: list[list], results: list) -> int:
        """Записать вектора успешных подъёмок (партия — короткая транзакция)."""
        written = 0
        for batch, result in zip(batches, results):
            if isinstance(result, EmbeddingError):
                continue  # отказ кодирования: чанки остаются pending (NFR-3)
            if isinstance(result, BaseException):
                raise result  # непредвиденный сбой — громко, как в других петлях
            with session(self._settings) as conn, transaction(conn):
                for row, vector in zip(batch, result):
                    if chunks.upsert_vector_if_exists(
                        conn, row["id"], vector, row["text"], row["tokens"]
                    ):
                        written += 1
        return written