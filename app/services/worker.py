"""Фоновый воркер (ARCHITECTURE §3.4): до-векторизация + до-суммаризация.

Единственный воркер на процесс, **три независимые петли по СЛОТАМ** (Фаза 11,
решение №10) — состояния pending-статусов в БД (переживают рестарт, догоняются
при старте сервиса):

- **embedding-петля** (`_run_embedding`): вектора заметок (`pending_vector` →
  batch `embed_texts` → notes_vec, vector_status='ok') + чанковая очередь
  (Фаза 7, анти-джоин «нет строки в notes_chunks_vec»). Объединены в одну
  петлю; Semaphore EMBEDDING_CONCURRENT_REQUESTS остаётся. После готовности
  вектора каждой заметки создаётся judge-работа (дедуп) — диспетчер
  зависимостей (решение №10): судья опрашивается только по довекторизованной
  заметке.
- **summary-петля** (`_run_summary`): title-догенерация (миграция, title IS
  NULL) → summarize → merge (слияние дублей) → классификация → описание узла.
  notify будит эту петлю (save/update).
- **judge-петля** (`_run_judge`): судья дедупа (по judge-работам, созданным
  embedding-петлёй) + судья структуры (внутри PromotionService, триггер после
  классификации).

Job-очереди в БД по слотам (`worker_jobs`): judge-работа (kind='dedup')
создаётся ТОЛЬКО после готовности вектора заметки; merge-работа (kind='merge')
создаётся после вердикта судьи и ходит в summary-слот. Порядок заметок в
очереди — по id. Back-off раздельный по петлям (30 с → ×2 → 15 мин, как
сейчас); notify будит summary-петлю; save векторизует синхронно — вне очередей.

Title-догенерация (решение №9): заметки с `title IS NULL` (только миграционные
— новые всегда с title) → думающий вызов слота summary (think по флагу, как
суммаризация), результат обрезается до TITLE_MAX_WORDS механикой, запись title;
очередь наполняется только миграцией и после прогонки опустеет. Промпт
догенерации — зашит в воркере (TITLE_PROMPT), файлом не создаётся.

Garanties:
- отказ любого внешнего сервера не портит данные: статус остаётся pending
  (NFR-3); интервал опроса растёт по back-off: PENDING_RETRY_SEC (30 с) → ×2
  → max 15 минут (REQUIREMENTS §5.3) — **независимо по каждой петле**
  (ARCH §3.4): недоступный векторизатор не останавливает суммаризацию и
  наоборот; успех в петле сбрасывает только её интервал и продолжает
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
import json
import logging

from app.config import TITLE_MAX_WORDS, Settings
from app.services.classifier import ClassificationError, Classifier
from app.services.dedup import DeduplicationService
from app.services.embedding import Embedder, EmbeddingError
from app.services.judge import Judge, JudgeError
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.services.promotion import PromotionService
from app.services.summary import Summarizer, SummaryError
from app.storage import chunks, vectors
from app.storage.db import session, transaction

# Заметок за один прогон (embed_texts — batch; summary — по одной, модель
# суммаризации тяжёлая): очередь выгребается последовательными партиями.
PENDING_BATCH = 50

# Потолок back-off (REQUIREMENTS §5.3 «max 15 мин»), env не настраивается.
MAX_INTERVAL_SEC = 15 * 60

# Промпт догенерации названия (решение №9): ЗАШИТ в SummaryService.title
# (follow-up 6b — протокол Summarizer получил метод title; здесь раньше был
# мёртвый дубль константы, генерация шла с промптом суммаризации).
# Думающий вызов слота summary, обрезка до TITLE_MAX_WORDS — механика воркера.


def next_interval(current: float, start: int) -> float:
    """Шаг back-off: интервал удваивается, потолок — 15 минут (§3.4)."""
    return min(max(current * 2.0, float(start)), float(MAX_INTERVAL_SEC))


class BackgroundWorker:
    """Единственный фоновый воркер; очереди — pending-статусы в БД + worker_jobs."""

    def __init__(
        self,
        settings: Settings,
        embedding: Embedder,
        summarizer: Summarizer | None = None,
        dedup: DeduplicationService | None = None,
        judge: Judge | None = None,
        classifier: Classifier | None = None,
        promoter: PromotionService | None = None,
    ) -> None:
        self._settings = settings
        self._embedding = embedding
        self._summarizer = summarizer
        # LLM-судья дедупа (Фаза 8, Этап 3.1, DI из build_services — один
        # экземпляр на процесс): вердикт «дубль/не дубль» по каждому
        # косинус-кандидату (judge-петля, Этап 3.2). None — тестовый режим:
        # воркер сводит по косинус-фоллбеку Этапа 2.2.
        self._judge = judge
        # Фоновый дедуп (Фаза 8, Этап 2): поиск косинус-кандидатов против
        # ранних заметок. DI для тестов.
        self._dedup = (
            dedup if dedup is not None else DeduplicationService(settings)
        )
        # Сведение дублей (Этап 2.2) идёт штатной NOTE-логикой: update
        # раннего (ре-векторизация/ре-суммаризация своими очередями) и
        # soft delete позднего. Сервис собирается из тех же deps, что и
        # воркер (embedding — DI-фейк в юнит-тестах); save-пути здесь не
        # используются, notifier не нужен — после слияния воркер будит
        # свою же суммаризационную петлю (notify_summary_pending).
        self._notes = NoteService(settings, embedding=embedding)
        # Причёска (Фаза 10, Шаг 4): классификатор default-заметок после
        # суммаризации; None — тестовый режим без классификации.
        self._classifier = classifier
        # Реестр неймспейсов: известные узлы для классификатора и проверка
        # целевого узла авто-переезда.
        self._namespaces = NamespaceService(settings)
        # Триггер домена (Фаза 10, Шаг 5): авто-создание листов из hint-групп
        # после классификации; None — тестовый режим без триггера (в проде
        # DI из build_services: describer + судья структуры).
        self._promoter = promoter
        self._vector_interval = float(max(settings.pending_retry_sec, 0))
        self._summary_interval = float(max(settings.pending_retry_sec, 0))
        self._judge_interval = float(max(settings.pending_retry_sec, 0))
        self._stopping = False
        # Сигнал «появилась заметка с pending summary» — будит петлю
        # суммаризации немедленно (save/update), минуя выросший back-off.
        self._summary_event = asyncio.Event()
        # Сигнал «появилась judge-работа» — будит judge-петлю (embedding-петля
        # создала дедуп-работу после довекторизации).
        self._judge_event = asyncio.Event()
        # Job-очереди по слотам (Фаза 11, решение №10): таблица создаётся
        # воркером лениво при первом обращении к очередям (схема — зона
        # воркера, не db.py). В __init__ не создаём: воркер собирается в
        # create_app() на импорте, когда БД ещё не инициализирована.

    @property
    def interval(self) -> float:
        """Текущий интервал embedding-петли (диагностика, тесты)."""
        return self._vector_interval

    @property
    def summary_interval(self) -> float:
        """Текущий интервал суммаризационной петли (диагностика, тесты)."""
        return self._summary_interval

    @property
    def chunk_interval(self) -> float:
        """Интервал чанковой очереди (диагностика, тесты, Фаза 7).

        Чанковая очередь объединена с векторной в embedding-петлю (решение
        №10) — интервал общий с `interval`.
        """
        return self._vector_interval

    @property
    def judge_interval(self) -> float:
        """Текущий интервал judge-петли (диагностика, тесты, Фаза 11)."""
        return self._judge_interval

    def stop(self) -> None:
        """Мягкая остановка: петли завершатся после разборки текущей итерации."""
        self._stopping = True

    def notify_summary_pending(self) -> None:
        """Разбудить петлю суммаризации: появилась заметка с pending summary.

        Вызывается из save/update (поток `asyncio.to_thread`) —
        `asyncio.Event.set()` потокобезопасен. Петля немедленно выходит из
        ожидания и догоняет очередь, не дожидаясь выросшего back-off.
        """
        self._summary_event.set()

    def notify_judge_pending(self) -> None:
        """Разбудить judge-петлю: появилась judge-работа (дедуп).

        Вызывается из embedding-петли после создания дедуп-работы —
        `asyncio.Event.set()` потокобезопасен.
        """
        self._judge_event.set()

    async def run(self) -> None:
        """Все петли очередей (запускается asyncio-таской при старте).

        Обработанные партии идут одна за другой (очередь выгребаем сразу);
        пустой прогон — пауза на текущий интервал петли с удвоением.
        Петли независимы: back-off и выгребание — раздельные.
        """
        await asyncio.gather(
            self._run_embedding(), self._run_summary(), self._run_judge()
        )

    async def _run_embedding(self) -> None:
        while not self._stopping:
            processed = 0
            processed += await asyncio.to_thread(self.process_pending)
            processed += await self.process_pending_chunks()
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
            processed = 0
            processed += await asyncio.to_thread(self.process_title_pending)
            processed += await asyncio.to_thread(self.process_summary_pending)
            processed += await asyncio.to_thread(self.process_merge_pending)
            if processed:
                self._summary_interval = float(self._settings.pending_retry_sec)
                continue
            # Пустой прогон: ждём сигнал «новая заметка» (save/update) или
            # таймаут back-off. Сигнал будит петлю немедленно — суммаризация
            # стартует сразу после записи, а не через выросший интервал.
            self._summary_event.clear()
            try:
                await asyncio.wait_for(
                    self._summary_event.wait(), timeout=self._summary_interval
                )
            except asyncio.TimeoutError:
                self._summary_interval = next_interval(
                    self._summary_interval, self._settings.pending_retry_sec
                )

    async def _run_judge(self) -> None:
        while not self._stopping:
            processed = await asyncio.to_thread(self.process_judge_pending)
            if processed:
                self._judge_interval = float(self._settings.pending_retry_sec)
                continue
            # Пустой прогон: ждём сигнал «появилась judge-работа» или таймаут
            # back-off. Сигнал будит петлю немедленно после довекторизации.
            self._judge_event.clear()
            try:
                await asyncio.wait_for(
                    self._judge_event.wait(), timeout=self._judge_interval
                )
            except asyncio.TimeoutError:
                self._judge_interval = next_interval(
                    self._judge_interval, self._settings.pending_retry_sec
                )

    # --- job-очереди по слотам (Фаза 11, решение №10) -------------------------

    def _ensure_job_table(self) -> None:
        """Создать таблицу job-очередей (идемпотентно).

        Схема — зона воркера (не db.py): job-очереди по слотам — внутренняя
        механика фонового конвейера, диспетчер зависимостей между петлями.
        """
        with session(self._settings) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS worker_jobs ("
                "  id         INTEGER PRIMARY KEY,"
                "  slot       TEXT NOT NULL,"
                "  kind       TEXT NOT NULL,"
                "  note_id    INTEGER NOT NULL,"
                "  payload    TEXT,"
                "  status     TEXT NOT NULL DEFAULT 'pending',"
                "  created_at TEXT NOT NULL DEFAULT "
                "    (strftime('%Y-%m-%dT%H:%M:%SZ','now')),"
                "  updated_at TEXT NOT NULL DEFAULT "
                "    (strftime('%Y-%m-%dT%H:%M:%SZ','now'))"
                ")"
            )

    def _create_job(
        self, slot: str, kind: str, note_id: int, payload: str | None = None
    ) -> None:
        """Поставить работу в очередь слота (диспетчер зависимостей)."""
        self._ensure_job_table()
        with session(self._settings) as conn, transaction(conn):
            conn.execute(
                "INSERT INTO worker_jobs (slot, kind, note_id, payload) "
                "VALUES (?, ?, ?, ?)",
                (slot, kind, note_id, payload),
            )

    def _pending_jobs(
        self, slot: str, kind: str, limit: int
    ) -> list:
        """Вычитать pending-работы слота (порядок по id)."""
        self._ensure_job_table()
        with session(self._settings) as conn:
            return conn.execute(
                "SELECT id, note_id, payload FROM worker_jobs "
                "WHERE slot = ? AND kind = ? AND status = 'pending' "
                "ORDER BY id LIMIT ?",
                (slot, kind, limit),
            ).fetchall()

    def _mark_job_done(self, job_id: int) -> None:
        """Пометить работу выполненной (снята с очереди)."""
        self._ensure_job_table()
        with session(self._settings) as conn, transaction(conn):
            conn.execute(
                "UPDATE worker_jobs SET status = 'done', "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ?",
                (job_id,),
            )

    # --- синхронная работа (выполняется в to_thread) --------------------------

    def process_pending(self, limit: int = PENDING_BATCH) -> int:
        """Векторизовать одну партию pending; возвращает число обработанных.

        Отказ кодирования — 0: статусы не тронуты, воркер выждет back-off.
        После каждой довекторизации создаётся judge-работа (дедуп) в очередь
        слота judge (решение №10): судья опрашивается только по готовому
        вектору — диспетчер зависимостей. Само сведение дублей — в
        judge-петле (process_judge_pending) и summary-петле
        (process_merge_pending).
        """
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, text, namespace FROM notes "
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
        processed = len(rows)
        for row, vector in zip(rows, embeddings):
            with session(self._settings) as conn, transaction(conn):
                # Фаза 10: вектор пишется в партицию неймспейса заметки.
                vectors.upsert(conn, row["id"], vector, row["namespace"])
                conn.execute(
                    "UPDATE notes SET vector_status = 'ok' WHERE id = ?",
                    (row["id"],),
                )
            # Фаза 11 (решение №10): вектор готов — judge-работа (дедуп)
            # в очередь слота judge; диспетчер зависимостей.
            self._create_job("judge", "dedup", int(row["id"]))
        self.notify_judge_pending()
        return processed

    # --- judge-петля: судья дедупа (Фаза 8, Этап 3.2; Фаза 11, решение №10) ---

    def process_judge_pending(self, limit: int = PENDING_BATCH) -> int:
        """Обработать партию judge-работ (дедуп); число обработанных.

        По каждой judge-работе (создана после довекторизации заметки):
        косинус-кандидаты против ранних заметок (_find_dedup_candidates,
        DEDUP_CANDIDATE_* — только предфильтр) и приговор «дубль» — LLM-судья
        (Этап 3.2, JudgeService; без судьи — косинус-фоллбек Этапа 2.2).
        Признанный дубль → merge-работа в очередь слота summary (решение
        №10: merge ходит в summary-слот). Отказ судьи (JudgeError) — работа
        остаётся pending, повтор по back-off judge-петли (NFR-3: обе заметки
        целы). Протухшая заметка (удалена/вектор не готов) — работа снимается.
        """
        jobs = self._pending_jobs("judge", "dedup", limit)
        if not jobs:
            return 0
        done = 0
        for job in jobs:
            note_id = int(job["note_id"])
            with session(self._settings) as conn:
                row = conn.execute(
                    "SELECT id, text, namespace, vector_status, deleted_at "
                    "FROM notes WHERE id = ?",
                    (note_id,),
                ).fetchone()
                vector = vectors.get_vector(conn, note_id) if row is not None else None
            if (
                row is None
                or row["deleted_at"] is not None
                or row["vector_status"] != "ok"
                or vector is None
            ):
                self._mark_job_done(job["id"])
                done += 1
                continue
            older = self._find_dedup_candidates(
                note_id, vector, row["namespace"]
            )
            if older:
                # Перечитать тексты свежей заметки и всех кандидатов (гонка с
                # memory_update/delete: протухшие кандидаты срезаются заранее —
                # о них не спрашивают ни судью, ни суммаризатор).
                ids = [note_id] + [candidate_id for candidate_id, _ in older]
                placeholders = ",".join("?" * len(ids))
                with session(self._settings) as conn:
                    rows = conn.execute(
                        f"SELECT id, text, deleted_at FROM notes WHERE id IN "
                        f"({placeholders})",
                        ids,
                    ).fetchall()
                by_id = {r["id"]: r for r in rows}
                newer = by_id.get(note_id)
                if newer is None or newer["deleted_at"] is not None:
                    self._mark_job_done(job["id"])
                    done += 1
                    continue
                alive = [
                    (candidate_id, cosine_value)
                    for candidate_id, cosine_value in older
                    if (r := by_id.get(candidate_id)) is not None
                    and r["deleted_at"] is None
                ]
                try:
                    best = self._pick_duplicate(note_id, alive, by_id)
                except JudgeError:
                    logging.getLogger("app").warning(
                        "dedup: judge undecidable — both notes kept, retry queued",
                        extra={
                            "event": "dedup_judge_failed",
                            "note_id": note_id,
                            "candidates": [
                                candidate_id for candidate_id, _ in alive
                            ],
                        },
                    )
                    continue  # работа остаётся pending — повтор по back-off
                if best is not None:
                    older_id, cosine_value = best
                    # Merge-работа в очередь слота summary (решение №10).
                    self._create_job(
                        "summary",
                        "merge",
                        note_id,
                        payload=json.dumps(
                            {"older_id": older_id, "cosine": cosine_value}
                        ),
                    )
            self._mark_job_done(job["id"])
            done += 1
        return done

    # --- summary-петля: merge-работа (слияние дублей, Этап 2.2) --------------

    def process_merge_pending(self, limit: int = PENDING_BATCH) -> int:
        """Обработать партию merge-работ (слияние дублей); число обработанных.

        Merge-работа создана judge-петлёй после вердикта судьи (решение №10:
        merge ходит в summary-слот). Процедура (вариант B — решение Олега):
        1) перечитать тексты ранней и поздней заметок (гонка с
           memory_update/delete: протухшая пара срезается);
        2) summarizer.merge(текст_ранней, текст_поздней) — объединить;
        3) NoteService.update ранней (текст = объединённый; ре-векторизация
           и ре-суммаризация — штатно, своими очередями; title ранней
           сохраняется — update без title, решение №9);
        4) NoteService.delete поздней (soft delete, trash).

        Отказ слияния (SummaryError, NFR-3) данные не портит: обе заметки
        остаются, работа остаётся pending — повтор по back-off summary-петли.
        """
        if self._summarizer is None:
            return 0  # тестовый режим без суммаризатора: слияние невозможно
        jobs = self._pending_jobs("summary", "merge", limit)
        if not jobs:
            return 0
        done = 0
        for job in jobs:
            note_id = int(job["note_id"])
            try:
                payload = json.loads(job["payload"] or "{}")
            except (TypeError, ValueError):
                payload = {}
            older_id = payload.get("older_id")
            if older_id is None:
                self._mark_job_done(job["id"])
                done += 1
                continue
            with session(self._settings) as conn:
                rows = conn.execute(
                    "SELECT id, text, deleted_at FROM notes WHERE id IN (?, ?)",
                    (older_id, note_id),
                ).fetchall()
            by_id = {r["id"]: r for r in rows}
            older = by_id.get(older_id)
            newer = by_id.get(note_id)
            if (
                older is None
                or newer is None
                or older["deleted_at"] is not None
                or newer["deleted_at"] is not None
            ):
                self._mark_job_done(job["id"])
                done += 1
                continue
            try:
                merged = self._summarizer.merge(older["text"], newer["text"])
                updated = self._notes.update(older_id, merged)
                if not updated.get("updated"):
                    self._mark_job_done(job["id"])
                    done += 1
                    continue
                self._notes.delete(note_id)  # поздний дубль — в trash
            except SummaryError:
                logging.getLogger("app").warning(
                    "dedup: summarizer merge failed — both notes kept, retry queued",
                    extra={
                        "event": "dedup_merge_failed",
                        "older_id": older_id,
                        "note_id": note_id,
                    },
                )
                continue  # работа остаётся pending — повтор по back-off
            logging.getLogger("app").info(
                "dedup: duplicate merged into earlier note",
                extra={
                    "event": "dedup_merged",
                    "older_id": older_id,
                    "note_id": note_id,
                    "cosine": payload.get("cosine"),
                },
            )
            self._mark_job_done(job["id"])
            done += 1
            # Ранняя заметка обновлена (summary pending) — будим свою же петлю
            # суммаризации, не дожидаясь back-off.
            self.notify_summary_pending()
        return done

    # --- фоновый дедуп (Фаза 8, Этапы 2–3) ------------------------------------

    def _find_dedup_candidates(
        self, note_id: int, vector: list[float], namespace: str = "default"
    ) -> list[tuple[int, float]]:
        """Косинус-кандидаты дедупа после довекторизации заметки.

        Кандидаты ищутся только против **ранних** заметок (id меньше
        текущей): пара «поздняя ↔ ранняя» обрабатывается один раз, из
        стороны поздней — сведение (Этап 2.2) обновляет ранний дубль и
        soft-delete поздний, а встречный прогон той же пары из стороны
        ранней заметки зациклил бы обработку. Фаза 10 (§5.7): только в
        пределах неймспейса заметки. Кандидаты логируются
        (наблюдаемость); приговор «дубль» принимает process_judge_pending:
        каждый кандидат опрашивается судьёй (Этап 3.2, JudgeService —
        косинус лишь предфильтр); без судьи — косинус-фоллбек
        DEDUP_SIMILARITY (Этап 2.2).

        Возвращает список [(candidate_id, cosine)] — вход сведение.
        """
        found = self._dedup.find_candidates(
            vector, exclude_id=note_id, namespace=namespace
        )
        older = [pair for pair in found if pair[0] < note_id]
        if older:
            logging.getLogger("app").info(
                "dedup: cosine candidates found for vectorized note",
                extra={
                    "event": "dedup_candidates",
                    "note_id": note_id,
                    "candidates": older,
                },
            )
        return older

    def _pick_duplicate(
        self,
        note_id: int,
        candidates: list[tuple[int, float]],
        by_id: dict,
    ) -> tuple[int, float] | None:
        """Выбрать кандидата для сведения: судья (Этап 3.2) или фоллбек.

        Судья (Judge): опрашивается по каждому живому кандидату в порядке
        убывания близости (порядок выдачи find_candidates); первый вердикт
        «ДУБЛЬ» — приговор (candidate_id, cosine). «НЕ ДУБЛЬ» по всем —
        None: слияния нет, заметка считается обработанной (повторять
        вопрос по неизменному тексту незачем: дедуп ждёт следующую
        векторизацию, а та случится только после изменения текста).
        Каждая пара логируется (event=dedup_judge) — наблюдаемость
        вердиктов предфильтра.

        Фоллбек без судьи (DI None, тестовый режим Этапа 2.2): первый
        кандидат с cosine ≥ DEDUP_SIMILARITY (выдача отсортирована).

        JudgeError пробрасывается наружу — process_judge_pending трактует
        отказ судьи как отказ слияния (работа остаётся pending, повтор по
        back-off, NFR-3): неопределённость не превращаем в «не дубль».

        by_id — свежепрочитанные строки notes (id → row): судья сравнивает
        тексты пары (свежая, кандидат) — порядок аргументов фиксирует
        JUDGE_USER_TEMPLATE (ТЕКСТ 1 — новая, ТЕКСТ 2 — кандидат).
        """
        if self._judge is None:
            # Фоллбек Этапа 2.2 (тестовый режим): выдача отсортирована.
            return next(
                (
                    pair
                    for pair in candidates
                    if pair[1] >= self._settings.dedup_similarity
                ),
                None,
            )
        for candidate_id, cosine_value in candidates:
            verdict = self._judge.judge(
                by_id[note_id]["text"], by_id[candidate_id]["text"]
            )
            logging.getLogger("app").info(
                "dedup: judge verdict for cosine candidate",
                extra={
                    "event": "dedup_judge",
                    "note_id": note_id,
                    "candidate_id": candidate_id,
                    "cosine": cosine_value,
                    "verdict": verdict,
                },
            )
            if verdict:
                return (candidate_id, cosine_value)
        return None

    # --- summary-петля: title-догенерация (решение №9) -----------------------

    def process_title_pending(self, limit: int = PENDING_BATCH) -> int:
        """Догенерировать названия миграционных заметок (title IS NULL).

        Только миграционные заметки (новые всегда с title — контракт решения
        №9); очередь наполняется только миграцией и после прогонки опустеет.
        Думающий вызов слота summary (SummaryService.title — промпт зашит
        TITLE_PROMPT, Фаза 11), результат обрезается до TITLE_MAX_WORDS слов
        механикой, запись title. Отказ генерации
        (SummaryError) — заметка остаётся без названия, повтор по back-off
        (NFR-3). Возвращает число записанных названий.
        """
        if self._summarizer is None:
            return 0
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, text FROM notes "
                "WHERE title IS NULL AND deleted_at IS NULL "
                "ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
        done = 0
        for row in rows:
            try:
                generated = self._summarizer.title(row["text"])
            except SummaryError:
                logging.getLogger("app").warning(
                    "title: generation failed — kept null, retry by back-off",
                    extra={"event": "title_failed", "note_id": row["id"]},
                )
                continue  # отказ: title остаётся NULL, повтор по back-off
            title = self._truncate_title(generated)
            if not title:
                continue
            with session(self._settings) as conn, transaction(conn):
                cursor = conn.execute(
                    "UPDATE notes SET title = ? "
                    "WHERE id = ? AND title IS NULL AND deleted_at IS NULL",
                    (title, row["id"]),
                )
            if cursor.rowcount:
                done += 1
                logging.getLogger("app").info(
                    "title: generated for migration note",
                    extra={"event": "title_generated", "note_id": row["id"]},
                )
        return done

    @staticmethod
    def _truncate_title(text: str) -> str:
        """Обрезать сгенерированное название до TITLE_MAX_WORDS слов (решение №9).

        Слова = len(title.split()) — как в контракте валидации (notes.py).
        """
        words = text.split()
        return " ".join(words[:TITLE_MAX_WORDS])

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
                "SELECT id, text, namespace, classified_at FROM notes "
                "WHERE summary_status = 'pending' AND deleted_at IS NULL "
                "ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
        done = 0
        for row in rows:
            try:
                summary = self._summarizer.summarize(row["text"])
            except SummaryError:
                logging.getLogger("app").warning(
                    "summary: generation failed — kept pending, retry by back-off",
                    extra={"event": "summary_failed", "note_id": row["id"]},
                )
                continue  # отказ: status pending остаётся, повтор по back-off
            with session(self._settings) as conn, transaction(conn):
                cursor = conn.execute(
                    "UPDATE notes SET summary = ?, summary_status = 'ok' "
                    "WHERE id = ? AND summary_status = 'pending' AND text = ?",
                    (summary, row["id"], row["text"]),
                )
            if cursor.rowcount:
                done += 1
                # Причёска (Фаза 10, Шаг 4): после суммаризации default-заметки
                # (ещё не классифицированной) — разметка и авто-переезд.
                if row["namespace"] == "default" and row["classified_at"] is None:
                    self._classify_default_note(int(row["id"]), row["text"])
        return done

    # --- причёска (Фаза 10, Шаг 4) -------------------------------------------

    def _classify_default_note(self, note_id: int, text: str) -> None:
        """Разметить default-заметку после суммаризации; авто-переезд.

        Только default-заметки (уложенные не перетряхиваются, §5.7) и только
        один проход (classified_at — анти-зацикливание; повтор — после
        memory_update, который сбрасывает summary в pending). Отказ
        классификатора (ClassificationError) данные не портит: заметка
        остаётся в default, classified_at не ставится — повтор при следующем
        обновлении.

        Авто-переезд в существующий узел — при confidence ≥
        NAMESPACE_AUTO_MOVE_MIN_CONFIDENCE: в существующий лист (если
        subdomain_hint совпал с зарегистрированным), иначе в корень домена;
        новый лист (subdomain_hint не зарегистрирован) остаётся в default —
        его создаст триггер (Шаг 5) и переложит ретро-перекладкой. Переезд
        ставит vector_status='pending' — воркер пере-кодирует вектор в
        партицию нового неймспейса (старый вектор уходит DELETE+INSERT).
        """
        if self._classifier is None:
            return  # тестовый режим без классификатора
        known = self._namespaces.list_all()["namespaces"]
        try:
            result = self._classifier.classify(text, known)
        except ClassificationError:
            logging.getLogger("app").warning(
                "classify: failed — note stays in default, retry on next update",
                extra={"event": "classify_failed", "note_id": note_id},
            )
            return
        with session(self._settings) as conn, transaction(conn):
            conn.execute(
                "UPDATE notes SET domain_hint = ?, subdomain_hint = ?, "
                "confidence = ?, "
                "classified_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (result.domain_hint, result.subdomain_hint, result.confidence, note_id),
            )
        target = self._auto_move_target(result)
        if (
            target is not None
            and result.confidence >= self._settings.namespace_auto_move_min_confidence
        ):
            with session(self._settings) as conn, transaction(conn):
                conn.execute(
                    "UPDATE notes SET namespace = ?, vector_status = 'pending' "
                    "WHERE id = ? AND deleted_at IS NULL",
                    (target, note_id),
                )
            logging.getLogger("app").info(
                "classify: default note auto-moved into existing node",
                extra={
                    "event": "classified_moved",
                    "note_id": note_id,
                    "namespace": target,
                    "confidence": result.confidence,
                },
            )
        # Триггер домена (Фаза 10, Шаг 5): разметка могла докинуть hint-группу
        # до порога — прогоняем конвейер промоции (авто-создание/слияние).
        self._run_promotion()

    def _run_promotion(self) -> None:
        """Триггер домена (Шаг 5) после классификации default-заметки.

        Сбои триггера не роняют воркер: это этап обогащения, а не конвейера
        данных — суммаризация/векторизация важнее структурной автоматики.
        Ожидаемые отказы describer/судьи обрабатываются внутри
        PromotionService (кандидат остаётся без вердикта, NFR-3); здесь
        ловится ВСЁ остальное (включая баги) — warning с traceback в логи,
        петли очередей живут. Повтор — следующая классификация default-
        заметки: группы не теряются, просто дотягивают до порога позже.
        """
        if self._promoter is None:
            return  # тестовый режим без триггера
        try:
            report = self._promoter.run()
        except Exception:
            logging.getLogger("app").warning(
                "promotion: run failed — trigger deferred to next classification",
                extra={"event": "promotion_failed", "reason": "run"},
                exc_info=True,
            )
            return
        if any(report.values()):
            # Сводка — одним ключом: 'created' конфликтует с атрибутом
            # LogRecord (время создания) — extra его не принимает.
            logging.getLogger("app").info(
                "promotion: trigger run finished",
                extra={"event": "promotion_run", "report": report},
            )

    def _auto_move_target(self, result) -> str | None:
        """Целевой узел авто-переезда (только существующие узлы, §5.7).

        domain_hint — корень из реестра; если он не зарегистрирован (модель
        предложила новый корень — корни только оператор) — не двигаем.
        subdomain_hint: зарегистрированный лист → в него; новый лист → None
        (остаётся в default, триггер Шага 5 создаст и переложит); null →
        в корень домена (общая для домена заметка).
        """
        if not result.domain_hint:
            return None
        if not self._namespaces.exists(result.domain_hint):
            return None
        if result.subdomain_hint:
            leaf = f"{result.domain_hint}/{result.subdomain_hint}"
            if self._namespaces.exists(leaf):
                return leaf
            return None  # новый лист — триггер (Шаг 5) создаст и переложит
        return result.domain_hint

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
                        ns=note_rows[0]["namespace"],
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
                        conn,
                        row["id"],
                        vector,
                        row["text"],
                        row["tokens"],
                        ns=row["namespace"],
                    ):
                        written += 1
        return written
