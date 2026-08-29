"""Фоновый воркер (ARCHITECTURE §3.4): до-векторизация + до-суммаризация.

Единственный воркер на процесс, **две независимые очереди** — состояния
pending-статусов в БД (переживают рестарт, догоняются при старте сервиса):
- `pending_vector`  → batch `embed_texts` (один вызов на партию) → вектора в
  notes_vec, vector_status='ok';
- `pending_summary` → по заметке `summarizer.summarize(text)` (режим «Б» —
  единственный путь генерации summary, §5.5) → summary + summary_status='ok'.

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
  занимаем); `run` — asyncio-таска (обе петли под gather), старт/стоп — в
  lifespan.

Запись суммари защищена от гонки с memory_update: между вычиткой текста и
записью воркер мог получить обновлённый текст — UPDATE ограничен условием
`AND text = ?` (тот же текст; иначе суммари протухшего текста затёрло бы
свежий). Гонка возможна и в векторизации (Фаза 3): там ре-векторизация
синхронна в update — вектор пишется по актуальному на момент записи тексту;
расхождение «текст менялся между вычиткой и записью» чинится следующей
партией (SELECT читает текущий текст).

Статусы внешних серверов (для `/health.*_ok`, NFR-4) ведут сами сервисы —
воркер не агрегирует: все кодирования идут через EmbeddingService, все
генерации — через Summarizer.

Суммаризатор инъектируется DI (build_services): None — петля не запускается
(тестовый режим Фазы 3); в проде всегда передан SummaryService.
"""

from __future__ import annotations

import asyncio

from app.config import Settings
from app.services.embedding import Embedder, EmbeddingError
from app.services.summary import Summarizer, SummaryError
from app.storage import vectors
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
        self._stopping = False

    @property
    def interval(self) -> float:
        """Текущий интервал векторной очереди (диагностика, тесты)."""
        return self._vector_interval

    @property
    def summary_interval(self) -> float:
        """Текущий интервал суммаризационной очереди (диагностика, тесты)."""
        return self._summary_interval

    def stop(self) -> None:
        """Мягкая остановка: петли завершатся после разборки текущей итерации."""
        self._stopping = True

    async def run(self) -> None:
        """Обе петли очередей (запускается asyncio-таской при старте).

        Обработанные партии идут одна за другой (очередь выгребаем сразу);
        пустой прогон — пауза на текущий интервал очереди с удвоением.
        Петли независимы: back-off и выгребание — раздельные.
        """
        await asyncio.gather(self._run_vector(), self._run_summary())

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