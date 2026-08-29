"""Фоновый воркер (ARCHITECTURE §3.4): до-векторизация vector_status=pending.

Единственный asyncio-воркер на процесс. Партия работы:
1) вычитать batch pending-заметок из БД (выборка активных, deleted_at IS NULL);
2) закодировать тексты ОДНИМ вызовом `embed_texts` (batch API /api/embed);
3) записать вектора в notes_vec и поднять vector_status до 'ok'
   (по заметке — отдельная короткая транзакция).

Garanties:
- отказ векторизации не портит данные: статус остаётся pending (NFR-3),
  интервал опроса растёт по back-off: PENDING_RETRY_SEC (30 с) → ×2 → max
  15 минут (REQUIREMENTS §5.3) — недоступный сервер не долбим; успех возвращает
  интервал к стартовому и продолжает выгребать очередь немедленно.
- очередь — состояние в БД, не в памяти: переживает рестарт, догоняется при
  старте сервиса; конкурентный доступ через busy_timeout/WAL (§3.3).
- `process_pending` синхронный (выполняется в `asyncio.to_thread` — event loop
  не занимаем); `run` — цикл как asyncio-таска, старт/стоп — в lifespan.

Статус внешнего сервера (для `/health.embedding_ok`, NFR-4) обновляет сам
EmbeddingService — воркер не агрегирует (все кодирования идут через него).

Суммаризационная очередь добавится в Фазе 4 как второй независимый поток
(ARCH §3.4: back-off по каждой очереди свой).
"""

from __future__ import annotations

import asyncio

from app.config import Settings
from app.services.embedding import Embedder, EmbeddingError
from app.storage import vectors
from app.storage.db import session, transaction

# Заметок за один прогон (embed_texts — batch); больше не нужно: очередь
# выгребается последовательными партиями.
PENDING_BATCH = 50

# Потолок back-off (REQUIREMENTS §5.3 «max 15 мин»), env не настраивается.
MAX_INTERVAL_SEC = 15 * 60


def next_interval(current: float, start: int) -> float:
    """Шаг back-off: интервал удваивается, потолок — 15 минут (§3.4)."""
    return min(max(current * 2.0, float(start)), float(MAX_INTERVAL_SEC))


class BackgroundWorker:
    """Единственный фоновый воркер; очередь — pending-статусы в БД."""

    def __init__(self, settings: Settings, embedding: Embedder) -> None:
        self._settings = settings
        self._embedding = embedding
        self._interval = float(max(settings.pending_retry_sec, 0))
        self._stopping = False

    @property
    def interval(self) -> float:
        """Текущий интервал опроса (диагностика, тесты)."""
        return self._interval

    def stop(self) -> None:
        """Мягкая остановка: цикл завершится после разборки текущей итерации."""
        self._stopping = True

    async def run(self) -> None:
        """Цикл воркера (запускается asyncio-таской при старте приложения).

        Обработанные партии идут одна за другой (очередь выгребаем сразу);
        пустой прогон — пауза self.interval с последующим удвоением.
        """
        while not self._stopping:
            processed = await asyncio.to_thread(self.process_pending)
            if processed:
                self._interval = float(self._settings.pending_retry_sec)  # успех — сброс
                continue
            await asyncio.sleep(self._interval)
            self._interval = next_interval(
                self._interval, self._settings.pending_retry_sec
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