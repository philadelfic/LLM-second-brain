"""Фоновый воркер (ARCHITECTURE §3.4): до-векторизация + до-суммаризация.

Единственный воркер на процесс, **четыре независимые петли по СЛОТАМ**
(Фаза 11, решение №10; релиз 3.0.0 — петля областей) — состояния
pending-статусов в БД (переживают рестарт, догоняются при старте сервиса):

- **embedding-петля** (`_run_embedding`): вектора заметок (`pending_vector` →
  batch `embed_texts` → notes_vec, vector_status='ok'; полный вектор — по
  конкатенации title+text, `_embed_input`, lsb-0001 FR-1.1) + чанковая очередь
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
- **петля areas** (`_run_areas`, субстрат 3.0.0): вектора записей областей
  skills/terms/user (`vector_status='pending'` → батч `embed_texts` → vec0
  области, 'ok'). Без LLM в момент записи (архитектура субстрата §3.3):
  тексты областей — `AreaSpec.embed_text` (skills `name + description`,
  terms `term + context + definition`, user `name + body`); отказ — событие
  `area_embed_failed`, записи остаются pending, повтор по своему back-off.
  Смена модели/размерности дропает area-vec вместе с notes_vec
  (`db._sync_embedding_meta`) — все записи областей возвращаются в pending.

Job-очереди в БД по слотам (`worker_jobs`): judge-работа (kind='dedup')
создаётся ТОЛЬКО после готовности вектора заметки; merge-работа (kind='merge')
создаётся после вердикта судьи и ходит в summary-слот. Порядок заметок в
очереди — по id. Back-off раздельный по петлям (30 с → ×2 → 15 мин, как
сейчас); notify будит summary-петлю; save/update не векторизуют синхронно —
векторизация фоновая (embedding-петля).

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
свежий). Векторизация защищена так же (пул 1): notes-петля пишет вектор
только при неизменных с вычитки (id, text, title, namespace, vector_status) —
guard-UPDATE в той же транзакции; протухшая партия не пишется, повтор —
следующей партией (аналог `AND text = ?` у суммари и
`upsert_vector_if_exists` у чанков). Название входит в вектор (lsb-0001
FR-1.1): запись названия (title-доген) в той же транзакции возвращает
векторизацию в pending и сбрасывает notes_vec — старый вектор по чистому
text не кормит ни поиск, ни дедуп до перекодирования.

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
from app.services.areas import AreaSpec, ALL_AREAS
from app.services.classifier import ClassificationError, Classifier
from app.services.dedup import DeduplicationService
from app.services.embedding import Embedder, EmbeddingError
from app.services.judge import Judge, JudgeError
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.services.promotion import PromotionService
from app.services.summary import Summarizer, SummaryError
from app.storage import area_vectors, chunks, vectors
from app.storage.db import delete_note_physical, session, transaction

# Сколько хранить выполненные done-работы в worker_jobs (retention). Не env —
# по паттерну TITLE_MAX_WORDS: после этого срока работы вычищаются idle-веткой
# embedding-петли (_purge_done_jobs), очередь не растёт безгранично.
WORKER_JOBS_RETENTION_DAYS = 7

# Потолок back-off (REQUIREMENTS §5.3 «max 15 мин»), env не настраивается.
MAX_INTERVAL_SEC = 15 * 60

# Интервал джобы зачистки просроченных заметок (lsb-0004-02, этап 4):
# фиксированные 5 минут (решение О. 2026-09-09), без настройки в компоузе.
EXPIRATION_CLEANUP_INTERVAL_SEC = 5 * 60

# Промпт догенерации названия (решение №9): ЗАШИТ в SummaryService.title
# (follow-up 6b — протокол Summarizer получил метод title; здесь раньше был
# мёртвый дубль константы, генерация шла с промптом суммаризации).
# Думающий вызов слота summary, обрезка до TITLE_MAX_WORDS — механика воркера.


def next_interval(current: float, start: int) -> float:
    """Шаг back-off: интервал удваивается, потолок — 15 минут (§3.4)."""
    return min(max(current * 2.0, float(start)), float(MAX_INTERVAL_SEC))


def _embed_input(title: str | None, text: str) -> str:
    """Вход кодирования полного вектора заметки (lsb-0001 FR-1.1).

    Вектор заметки строится по конкатенации названия и текста:
    f"{title}\n{text}". title is NULL (миграционная заметка до догенерации
    названия) — кодируется чистый text без префикса. Чанковые вектора это
    не касается (решение D1): там кодируется чистый текст чанка без title.
    """
    return text if title is None else f"{title}\n{text}"


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
        self._areas_interval = float(max(settings.pending_retry_sec, 0))
        self._stopping = False
        # Сигнал «появилась заметка с pending summary» — будит петлю
        # суммаризации немедленно (save/update), минуя выросший back-off.
        self._summary_event = asyncio.Event()
        # Сигнал «появилась judge-работа» — будит judge-петлю (embedding-петля
        # создала дедуп-работу после довекторизации).
        self._judge_event = asyncio.Event()
        # Сигнал «появилась pending-запись области» — будит петлю areas
        # (save/update сервисов областей зовут notify_areas_pending): текст
        # записывается мгновенно, вектор догоняет фоном (архитектура
        # субстрата §3.3 — без LLM в момент записи).
        self._areas_event = asyncio.Event()
        # Мемоизация создания таблицы job-очередей (пул 5): DDL исполняется
        # один раз на экземпляр воркера, а не при каждом обращении к очередям
        # (_create_job/_pending_jobs/_mark_job_done звали _ensure_job_table
        # на каждый вызов — 3 раза на работу).
        self._jobs_table_ready = False
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

    @property
    def areas_interval(self) -> float:
        """Текущий интервал петли areas (диагностика, тесты, субстрат 3.0.0)."""
        return self._areas_interval

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

    def notify_areas_pending(self) -> None:
        """Разбудить петлю areas: появилась pending-запись области.

        Вызывается сервисами областей из save/update (поток
        `asyncio.to_thread`) — `asyncio.Event.set()` потокобезопасен. Петля
        немедленно выходит из ожидания и догоняет очередь векторов, не
        дожидаясь выросшего back-off.
        """
        self._areas_event.set()

    async def run(self) -> None:
        """Все петли очередей (запускается asyncio-таской при старте).

        Обработанные партии идут одна за другой (очередь выгребаем сразу);
        пустой прогон — пауза на текущий интервал петли с удвоением.
        Петли независимы: back-off и выгребание — раздельные.
        """
        await asyncio.gather(
            self._run_embedding(), self._run_summary(), self._run_judge(),
            self._run_areas(), self._run_expiration_cleanup(),
        )

    async def _run_embedding(self) -> None:
        while not self._stopping:
            try:
                processed = 0
                processed += await asyncio.to_thread(self.process_pending)
                processed += await self.process_pending_chunks()
                if processed:
                    self._vector_interval = float(
                        self._settings.pending_retry_sec
                    )  # успех — сброс
                    continue
                # Idle-ветка (пул 5): работ нет — гигиена worker_jobs
                # (done-старше retention вычищаются) перед сном.
                await asyncio.to_thread(self._purge_done_jobs)
                await asyncio.sleep(self._vector_interval)
                self._vector_interval = next_interval(
                    self._vector_interval, self._settings.pending_retry_sec
                )
            except asyncio.CancelledError:
                raise  # отмена петли (graceful stop) — не глотать
            except Exception:
                # Супервизор петли (пул 4): НЕПРЕДВИДЕННЫЙ сбой итерации
                # (StorageError при исчерпании busy_timeout, дефект кода,
                # «громкий» re-raise из _store_chunk_vectors) не убивает
                # корутину — warning с traceback, пауза и повтор по back-off.
                logging.getLogger("app").warning(
                    "worker loop iteration failed — loop continues",
                    extra={"event": "loop_iteration_failed", "loop": "embedding"},
                    exc_info=True,
                )
                await asyncio.sleep(self._vector_interval)
                self._vector_interval = next_interval(
                    self._vector_interval, self._settings.pending_retry_sec
                )

    async def _run_summary(self) -> None:
        if self._summarizer is None:
            return  # тестовый режим без суммаризатора: петля не нужна
        while not self._stopping:
            try:
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
                # Пул 6 (lost wakeup): перепроверяем очередь ПОСЛЕ clear() —
                # так окно гонки закрыто полностью: работа, появившаяся до
                # clear(), видна селекту (продолжаем без сна); появившаяся
                # после — будит event, который clear() уже не трогает.
                # (Перепроверка ДО clear оставляла микро-окно «notify между
                # селектом и clear» — сигнал стирался бы.)
                if not await asyncio.to_thread(self._summary_queue_empty):
                    continue
                self._summary_event.clear()
                try:
                    await asyncio.wait_for(
                        self._summary_event.wait(), timeout=self._summary_interval
                    )
                except asyncio.TimeoutError:
                    self._summary_interval = next_interval(
                        self._summary_interval, self._settings.pending_retry_sec
                    )
            except asyncio.CancelledError:
                raise  # отмена петли (graceful stop) — не глотать
            except Exception:
                # Супервизор петли (пул 4): непредвиденный сбой итерации не
                # убивает summary-петлю — warning с traceback, пауза, повтор.
                logging.getLogger("app").warning(
                    "worker loop iteration failed — loop continues",
                    extra={"event": "loop_iteration_failed", "loop": "summary"},
                    exc_info=True,
                )
                await asyncio.sleep(self._summary_interval)
                self._summary_interval = next_interval(
                    self._summary_interval, self._settings.pending_retry_sec
                )

    async def _run_judge(self) -> None:
        while not self._stopping:
            try:
                processed = await asyncio.to_thread(self.process_judge_pending)
                if processed:
                    self._judge_interval = float(self._settings.pending_retry_sec)
                    continue
                # Пустой прогон: ждём сигнал «появилась judge-работа» или таймаут
                # back-off. Сигнал будит петлю немедленно после довекторизации.
                # Пул 6 (lost wakeup): перепроверка ПОСЛЕ clear() — окно
                # «notify между селектом и clear» закрыто (см. комментарий
                # в _run_summary).
                if not await asyncio.to_thread(self._judge_queue_empty):
                    continue
                self._judge_event.clear()
                try:
                    await asyncio.wait_for(
                        self._judge_event.wait(), timeout=self._judge_interval
                    )
                except asyncio.TimeoutError:
                    self._judge_interval = next_interval(
                        self._judge_interval, self._settings.pending_retry_sec
                    )
            except asyncio.CancelledError:
                raise  # отмена петли (graceful stop) — не глотать
            except Exception:
                # Супервизор петли (пул 4): непредвиденный сбой итерации не
                # убивает judge-петлю — warning с traceback, пауза, повтор.
                logging.getLogger("app").warning(
                    "worker loop iteration failed — loop continues",
                    extra={"event": "loop_iteration_failed", "loop": "judge"},
                    exc_info=True,
                )
                await asyncio.sleep(self._judge_interval)
                self._judge_interval = next_interval(
                    self._judge_interval, self._settings.pending_retry_sec
                )

    # --- петля areas: вектора записей областей (субстрат 3.0.0) ---------------

    async def _run_areas(self) -> None:
        """Петля векторизации областей (субстрат 3.0.0, архитектура §3.3).

        Своя очередь, свой интервал/back-off и свой сигнал: отказ векторизации
        областей не мешает заметкам и наоборот (ARCH §3.4). Пустой прогон —
        проверка очереди ПОСЛЕ clear() (пул 6, lost wakeup, как в
        _run_summary/_run_judge) и ожидание сигнала notify_areas_pending() или
        таймаута интервала. Супервизор петли (пул 4): непредвиденный сбой
        итерации не убивает петлю — warning с traceback, пауза, повтор.
        """
        while not self._stopping:
            try:
                processed = await asyncio.to_thread(self.process_pending_areas)
                if processed:
                    self._areas_interval = float(self._settings.pending_retry_sec)
                    continue
                if not await asyncio.to_thread(self._areas_queue_empty):
                    continue
                self._areas_event.clear()
                try:
                    await asyncio.wait_for(
                        self._areas_event.wait(), timeout=self._areas_interval
                    )
                except asyncio.TimeoutError:
                    self._areas_interval = next_interval(
                        self._areas_interval, self._settings.pending_retry_sec
                    )
            except asyncio.CancelledError:
                raise  # отмена петли (graceful stop) — не глотать
            except Exception:
                logging.getLogger("app").warning(
                    "worker loop iteration failed — loop continues",
                    extra={"event": "loop_iteration_failed", "loop": "areas"},
                    exc_info=True,
                )
                await asyncio.sleep(self._areas_interval)
                self._areas_interval = next_interval(
                    self._areas_interval, self._settings.pending_retry_sec
                )

    # --- джоба зачистки просроченных заметок (lsb-0004-02, этап 4) -----------

    async def _run_expiration_cleanup(self) -> None:
        """Петля зачистки просроченных заметок (lsb-0004-02, этап 4).

        Раз в EXPIRATION_CLEANUP_INTERVAL_SEC (фиксированные 5 минут, решение
        О. 2026-09-09) выгребает note_expirations: заметки с
        `expires_at <= now()` удаляются ПОЛНОСТЬЮ (notes+chunks+вектора+fts)
        вместе со строкой из note_expirations. Интервал фиксированный — без
        back-off (в отличие от очередей pending): зачистка не зависит от
        внешних сервисов, сбой итерации не ускоряет/замедляет расписание.
        Супервизор петли (пул 4): непредвиденный сбой итерации не убивает
        петлю — warning с traceback, пауза на интервал, повтор.
        """
        while not self._stopping:
            try:
                deleted = await asyncio.to_thread(self.process_expired_notes)
                if deleted:
                    logging.getLogger("app").info(
                        "expiration cleanup: purged expired notes",
                        extra={"event": "expiration_cleanup", "count": deleted},
                    )
                await asyncio.sleep(EXPIRATION_CLEANUP_INTERVAL_SEC)
            except asyncio.CancelledError:
                raise  # отмена петли (graceful stop) — не глотать
            except Exception:
                logging.getLogger("app").warning(
                    "expiration cleanup loop iteration failed — loop continues",
                    extra={"event": "loop_iteration_failed", "loop": "expiration"},
                    exc_info=True,
                )
                await asyncio.sleep(EXPIRATION_CLEANUP_INTERVAL_SEC)

    def process_expired_notes(self) -> int:
        """Удалить просроченные заметки; возвращает число удалённых.

        SELECT note_id FROM note_expirations WHERE expires_at <= now() (now —
        в том же ISO-8601 UTC формате, что expires_at: strftime
        '%Y-%m-%dT%H:%M:%SZ','now'); для каждого id — полное физическое
        удаление (notes+chunks+вектора+fts) + удаление строки из
        note_expirations (delete_note_physical).

        Идемпотентно и безопасно: если заметка уже удалена (например,
        оператором) — delete_note_physical просто не матчит ничего, не падает;
        строка из note_expirations при этом всё равно снимается. Каждая
        заметка — короткая транзакция: сбой одной не откатывает остальных.
        """
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT note_id FROM note_expirations "
                "WHERE expires_at <= strftime('%Y-%m-%dT%H:%M:%SZ','now')"
            ).fetchall()
        if not rows:
            return 0
        deleted = 0
        for row in rows:
            note_id = int(row["note_id"])
            with session(self._settings) as conn, transaction(conn):
                delete_note_physical(conn, note_id)
                deleted += 1
        return deleted

    # --- job-очереди по слотам (Фаза 11, решение №10) -------------------------

    def _ensure_job_table(self) -> None:
        """Создать таблицу + индекс job-очередей (идемпотентно, один раз).

        Схема — зона воркера (не db.py): job-очереди по слотам — внутренняя
        механика фонового конвейера, диспетчер зависимостей между петлями.

        Пул 5, единократное создание: полный DDL исполняется только при
        первом обращении на экземпляр воркера (мемоизация на флаге
        `_jobs_table_ready`); повторные вызовы (3 штуки на каждую работу:
        `_create_job`, `_pending_jobs`, `_mark_job_done`) возвращаются сразу.
        Индекс `idx_worker_jobs_queue` на (slot, kind, status, id): выборка
        pending-очереди слота — по индексу, а не полным сканом таблицы.
        """
        if self._jobs_table_ready:
            return
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_worker_jobs_queue "
                "ON worker_jobs(slot, kind, status, id)"
            )
        self._jobs_table_ready = True

    def _purge_done_jobs(self) -> None:
        """Вычистить done-работы старше retention-окна (пул 5, гигиена).

        worker_jobs копит ≥1 строку на каждую заметку (при пере-векторизациях
        после update — больше); без чистки таблица и полные сканы очереди
        росли бы безгранично. DELETE ограничен `status = 'done'` и
        `updated_at` старше WORKER_JOBS_RETENTION_DAYS — pending-работы не
        трогаются. Вызывается в idle-ветке embedding-петли (когда работ нет,
        перед сном). Удалено > 0 → info-лог event `jobs_purged`.
        """
        self._ensure_job_table()
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "DELETE FROM worker_jobs WHERE status = 'done' AND updated_at < "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now',"
                f"'-{WORKER_JOBS_RETENTION_DAYS} days')"
            )
        if cursor.rowcount > 0:
            logging.getLogger("app").info(
                "worker_jobs: purged done jobs older than retention window",
                extra={
                    "event": "jobs_purged",
                    "count": cursor.rowcount,
                    "retention_days": WORKER_JOBS_RETENTION_DAYS,
                },
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

    # --- потерянный будильник: дешёвая перепроверка очереди перед сном (пул 6)

    def _summary_queue_empty(self) -> bool:
        """Пуста ли summary-очередь (дешёвый SELECT, пул 6, lost wakeup).

        Петля перед `clear()+wait` перепроверяет саму очередь, а не только
        event: notify мог прийти во время выгребания партии, и `clear()`
        после него стёр бы сигнал (пауза до интервала на непустой очереди).
        Очередь пуста — только если нет title-догена (title IS NULL),
        pending суммаризации и pending merge-работ (все в summary-слоте).
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT (SELECT COUNT(*) FROM notes WHERE deleted_at IS NULL "
                "  AND (title IS NULL OR summary_status = 'pending')) "
                "+ (SELECT COUNT(*) FROM worker_jobs "
                "  WHERE slot = 'summary' AND status = 'pending') AS c"
            ).fetchone()
        return int(row["c"]) == 0

    def _judge_queue_empty(self) -> bool:
        """Пуста ли judge-очередь (дешёвый SELECT, пул 6, lost wakeup).

        Считает pending dedup-работы judge-слота — ту же очередь, что
        выгребает process_judge_pending в этой петле.
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM worker_jobs "
                "WHERE slot = 'judge' AND kind = 'dedup' AND status = 'pending'"
            ).fetchone()
        return int(row[0]) == 0

    # --- синхронная работа (выполняется в to_thread) --------------------------

    def process_pending(self, limit: int | None = None) -> int:
        """Векторизовать одну партию pending; возвращает число обработанных.

        Размер партии по умолчанию — settings.embedding_batch_size (пул 5:
        notes-петля режется по EMBEDDING_BATCH_SIZE, как и чанковая
        process_pending_chunks); явный `limit` переопределяет.

        Полный вектор строится по title+text (_embed_input, lsb-0001
        FR-1.1); заметка без названия (title IS NULL — легаси-путь до
        догенерации) кодируется чистым text.

        Отказ кодирования — 0: статусы не тронуты, воркер выждет back-off.
        Guard (пул 1): вектор пишется только при неизменных с вычитки
        (id, text, title, namespace, vector_status='pending') — иначе
        memory_update/переезд/догенерация названия в полёте оставили бы
        протухший вектор со статусом 'ok'. Промахнувшиеся остаются pending →
        следующая партия перекодирует заново (новый текст или название).
        После каждой фактической довекторизации
        создаётся judge-работа (дедуп) в очередь слота judge (решение №10):
        судья опрашивается только по готовому вектору — диспетчер
        зависимостей. Само сведение дублей — в judge-петле
        (process_judge_pending) и summary-петле (process_merge_pending).
        """
        batch = (
            limit if limit is not None else self._settings.embedding_batch_size
        )
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, text, title, namespace FROM notes "
                "WHERE vector_status = 'pending' AND deleted_at IS NULL "
                "ORDER BY id LIMIT ?",
                (batch,),
            ).fetchall()
        if not rows:
            return 0
        try:
            embeddings = self._embedding.embed_texts(
                [_embed_input(row["title"], row["text"]) for row in rows]
            )
        except EmbeddingError:
            return 0
        processed = 0
        for row, vector in zip(rows, embeddings):
            with session(self._settings) as conn, transaction(conn):
                # Guard (пул 1): вектор — только если заметка не менялась с
                # вычитки (id, text, title, namespace, vector_status='pending')
                # — иначе memory_update/переезд/догенерация названия в полёте
                # оставили бы протухший вектор со статусом 'ok'. `title IS ?`
                # сравнивает и NULL (SQLite): легаси-заметка без названия
                # проходит guard только при прежнем NULL. Промахнувшиеся
                # остаются pending → следующая партия перекодирует заново
                # (новый текст или название).
                cursor = conn.execute(
                    "UPDATE notes SET vector_status = 'ok' "
                    "WHERE id = ? AND text = ? AND title IS ? AND namespace = ? "
                    "AND vector_status = 'pending'",
                    (row["id"], row["text"], row["title"], row["namespace"]),
                )
                if not cursor.rowcount:
                    continue
                # Фаза 10: вектор пишется в партицию неймспейса заметки.
                vectors.upsert(conn, row["id"], vector, row["namespace"])
                processed += 1
            # Фаза 11 (решение №10): вектор готов — judge-работа (дедуп)
            # в очередь слота judge; диспетчер зависимостей.
            self._create_job("judge", "dedup", int(row["id"]))
        self.notify_judge_pending()
        return processed

    # --- петля areas: вектора записей областей (субстрат 3.0.0) ---------------

    def process_pending_areas(self, limit: int | None = None) -> int:
        """Векторизовать партию pending записей областей; число обработанных.

        Каждая область (skills/terms/user) — своим батчем: pending записи
        (`vector_status='pending'`, активные) → `embed_texts` по тексту
        области (`AreaSpec.embed_text`) → upsert в её vec0 → `'ok'`. Размер
        батча — `embedding_batch_size` (как у notes-очереди). Guard: вектор
        пишется только если запись не менялась с вычиты (поля векторизации)
        — иначе update в полёте оставил бы протухший вектор со статусом 'ok'.
        Отказ кодировщика — записи остаются pending, событие
        `area_embed_failed` (NFR-3), повтор по back-off петли areas.
        """
        batch = (
            limit if limit is not None else self._settings.embedding_batch_size
        )
        processed = 0
        for spec in ALL_AREAS:
            processed += self._process_area_batch(spec, batch)
        return processed

    def _process_area_batch(self, spec: AreaSpec, batch: int) -> int:
        """Батч одной области: вычитка pending → кодирование → upsert + 'ok'."""
        columns = ", ".join(("id", *spec.embed_fields))
        with session(self._settings) as conn:
            rows = conn.execute(
                f"SELECT {columns} FROM {spec.table} "
                "WHERE vector_status = 'pending' AND deleted_at IS NULL "
                "ORDER BY id LIMIT ?",
                (batch,),
            ).fetchall()
        if not rows:
            return 0
        try:
            embeddings = self._embedding.embed_texts(
                [spec.embed_text(row) for row in rows]
            )
        except EmbeddingError:
            logging.getLogger("app").warning(
                "area vectorization failed — records stay pending",
                extra={
                    "event": "area_embed_failed",
                    "area": spec.name,
                    "count": len(rows),
                },
            )
            return 0
        processed = 0
        guard = " AND ".join(f"{column} IS ?" for column in spec.embed_fields)
        for row, vector in zip(rows, embeddings):
            with session(self._settings) as conn, transaction(conn):
                # Guard по полям векторизации: `IS ?` сравнивает и NULL.
                cursor = conn.execute(
                    f"UPDATE {spec.table} SET vector_status = 'ok' "
                    "WHERE id = ? AND vector_status = 'pending' "
                    f"AND deleted_at IS NULL AND {guard}",
                    (row["id"], *(row[column] for column in spec.embed_fields)),
                )
                if not cursor.rowcount:
                    continue
                area_vectors.upsert(
                    conn,
                    spec.vec_table,
                    spec.vec_id_column,
                    int(row["id"]),
                    vector,
                )
                processed += 1
        return processed

    def _areas_queue_empty(self) -> bool:
        """Пуста ли очередь областей (дешёвый SELECT, пул 6, lost wakeup).

        pending-записи считаются по всем областям реестра (ALL_AREAS) —
        та же очередь, что выгребает process_pending_areas.
        """
        pending = 0
        with session(self._settings) as conn:
            for spec in ALL_AREAS:
                pending += int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM {spec.table} "
                        "WHERE vector_status = 'pending' "
                        "AND deleted_at IS NULL"
                    ).fetchone()[0]
                )
        return pending == 0

    # --- judge-петля: судья дедупа (Фаза 8, Этап 3.2; Фаза 11, решение №10) ---

    def process_judge_pending(self, limit: int | None = None) -> int:
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
        jobs = self._pending_jobs(
            "judge",
            "dedup",
            limit if limit is not None else self._settings.embedding_batch_size,
        )
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

    def process_merge_pending(self, limit: int | None = None) -> int:
        """Обработать партию merge-работ (слияние дублей); число обработанных.

        Merge-работа создана judge-петлёй после вердикта судьи (решение №10:
        merge ходит в summary-слот). Процедура (вариант B — решение Олега):
        1) перечитать тексты ранней и поздней заметок (гонка с
           memory_update/delete: протухшая пара срезается);
        2) summarizer.merge(текст_ранней, текст_поздней) — объединить;
        3) NoteService.merge_pair — ОДНОЙ транзакцией обновить раннюю заметку
           (текст = объединённый; ре-векторизация и ре-суммаризация — штатно,
           своими очередями; title ранней не трогается) и soft-delete поздней
           (trash). Пул 6: единая транзакция исключает полусостояние
           «обновлена ранняя, поздняя жива» при отказе между операциями.

        Отказ слияния (SummaryError, NFR-3) данные не портит: обе заметки
        остаются, работа остаётся pending — повтор по back-off summary-петли.
        """
        if self._summarizer is None:
            return 0  # тестовый режим без суммаризатора: слияние невозможно
        jobs = self._pending_jobs(
            "summary",
            "merge",
            limit if limit is not None else self._settings.embedding_batch_size,
        )
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
                outcome = self._notes.merge_pair(older_id, merged, note_id)
                if not outcome["merged"]:
                    # Ранняя исчезла (оператор удалил) между вычиткой и
                    # сведением — слияние не состоялось, работа снимается;
                    # поздняя при этом не трогается (merge_pair ничего не
                    # пишет, если ранней активной заметки нет).
                    self._mark_job_done(job["id"])
                    done += 1
                    continue
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

    def process_title_pending(self, limit: int | None = None) -> int:
        """Догенерировать названия миграционных заметок (title IS NULL).

        Только миграционные заметки (новые всегда с title — контракт решения
        №9); очередь наполняется только миграцией и после прогонки опустеет.
        Думающий вызов слота summary (SummaryService.title — промпт зашит
        TITLE_PROMPT, Фаза 11), результат обрезается до TITLE_MAX_WORDS слов
        механикой, запись title. Отказ генерации
        (SummaryError) — заметка остаётся без названия, повтор по back-off
        (NFR-3). Возвращает число записанных названий.

        Запись названия инвалидирует полный вектор (lsb-0001 FR-1.1: title
        участвует в кодировании) — в той же транзакции векторизация
        возвращается в 'pending' и notes_vec сброшен; воркер notes-очереди
        перекодирует уже с названием (_embed_input). Чанковые вектора не
        тронуты (решение D1: title на них не влияет).
        """
        if self._summarizer is None:
            return 0
        batch = (
            limit if limit is not None else self._settings.embedding_batch_size
        )
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, text FROM notes "
                "WHERE title IS NULL AND deleted_at IS NULL "
                "ORDER BY id LIMIT ?",
                (batch,),
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
                    # lsb-0001 FR-1.1: title участвует в полном векторе —
                    # запись названия делает старый вектор (по чистому text)
                    # протухшим: статус в 'pending', notes_vec сброшен (тот
                    # же смысл, что vectors.drop в notes.update(): в окне
                    # pending заметка участвует только в FTS-поиске, старый
                    # вектор без названия не кормит ни поиск, ни дедуп).
                    # Чанковые вектора не тронуты (решение D1: кодируются по
                    # чистому тексту чанка).
                    conn.execute(
                        "UPDATE notes SET vector_status = 'pending' WHERE id = ?",
                        (row["id"],),
                    )
                    stale = vectors.get_vector(conn, row["id"]) is not None
                    vectors.drop(conn, row["id"])
            if cursor.rowcount:
                done += 1
                logging.getLogger("app").info(
                    "title: generated for migration note",
                    extra={
                        "event": "title_generated",
                        "note_id": row["id"],
                        "vector_invalidated": stale,
                    },
                )
        return done

    @staticmethod
    def _truncate_title(text: str) -> str:
        """Обрезать сгенерированное название до TITLE_MAX_WORDS слов (решение №9).

        Слова = len(title.split()) — как в контракте валидации (notes.py).
        """
        words = text.split()
        return " ".join(words[:TITLE_MAX_WORDS])

    def process_summary_pending(self, limit: int | None = None) -> int:
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
        batch = (
            limit if limit is not None else self._settings.embedding_batch_size
        )
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, text, namespace, classified_at FROM notes "
                "WHERE summary_status = 'pending' AND deleted_at IS NULL "
                "ORDER BY id LIMIT ?",
                (batch,),
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
                    try:
                        self._classify_default_note(int(row["id"]), row["text"])
                    except asyncio.CancelledError:
                        raise  # отмена петли — не глотать
                    except Exception:
                        # Непредвиденный сбой причёски (в т.ч. баг) не роняет
                        # summary-петлю: остальные заметки партии обрабатываются
                        # дальше, классификация этой — после следующего update.
                        logging.getLogger("app").warning(
                            "classify: internal error — enrichment deferred",
                            extra={"event": "classify_crashed", "note_id": row["id"]},
                            exc_info=True,
                        )
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

        Пул 6: разметка и переезд — ОДНА транзакция (один UPDATE, guard
        `namespace='default'`): гонка с клиентским memory_update больше не
        оставляет полусостояния «разметка записана, переезд по старой оценке»;
        операторский переезд в полёте фоном не перебивается (rowcount). Цель
        авто-переезда считаем до транзакции — отказ _auto_move_target (в т.ч.
        NamespaceValidationError на мусорном hint) ничего не пишет в БД
        (строгая семантика «отказ классификации = не размечено»).
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
        # Пул 6: целевой узел авто-переезда ДО транзакции — если он падает
        # (в т.ч. NamespaceValidationError), БД не пишем вовсе.
        target = self._auto_move_target(result)
        move = (
            target is not None
            and result.confidence >= self._settings.namespace_auto_move_min_confidence
        )
        with session(self._settings) as conn, transaction(conn):
            # Один UPDATE: разметка + (при переезде) namespace/vector_status.
            # Guard `namespace = 'default'`: оператор, уложивший заметку в полёте,
            # фоном не перекладывается (rowcount 0 — переезда не было, нет и
            # лога classified_moved).
            columns = [
                "hint_path = ?",
                "confidence = ?",
                "classified_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')",
            ]
            params: list[object] = [
                result.hint_path,
                result.confidence,
            ]
            if move:
                columns.append("namespace = ?")
                columns.append("vector_status = 'pending'")
                params.append(target)
            cursor = conn.execute(
                "UPDATE notes SET "
                + ", ".join(columns)
                + " WHERE id = ? AND namespace = 'default' AND deleted_at IS NULL",
                (*params, note_id),
            )
        # Лог — только при ФАКТИЧЕСКОМ переезде (rowcount+condition).
        if move and cursor.rowcount:
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

        hint_path — полный путь разметки (1..3 слага); если он не
        зарегистрирован (модель предложила новый узел — его создаст/
        переложит триггер Шага 5) — не двигаем. Null (общая) — не двигаем.
        Не-путь (мусор классификатора) валидируется в `exists` внутри —
        NamespaceValidationError, ничего в БД не пишется (§5.7).
        """
        hint = result.hint_path
        if not hint:
            return None
        if not self._namespaces.exists(hint):
            return None
        return hint

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
        идентичен — копируем без вызова Ollama. Копия остаётся консистентной
        с notes_vec, который теперь включает название (title+text — lsb-0001
        FR-1.1); чанковая кодировка чистым текстом (решение D1) это не меняет.
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
