"""Фоновый воркер (ARCHITECTURE §3.4): до-векторизация + до-суммаризация.

Единственный воркер на процесс, **четыре независимые петли по СЛОТАМ**
(Фаза 11, решение №10; релиз 3.0.0 — петля областей) — состояния
pending-статусов в БД (переживают рестарт, догоняются при старте сервиса):

- **embedding-джоба** (`build_embedding_job`): вектора заметок (`pending_vector` →
  batch `embed_texts` → notes_vec, vector_status='ok'; полный вектор — по
  конкатенации title+text, `_embed_input`, lsb-0001 FR-1.1) + чанковая очередь
  (Фаза 7, анти-джоин «нет строки в notes_chunks_vec»). Объединены в одну
  петлю; Semaphore EMBEDDING_CONCURRENT_REQUESTS остаётся. После готовности
  вектора каждой заметки создаётся judge-работа (дедуп) — диспетчер
  зависимостей (решение №10): судья опрашивается только по довекторизованной
  заметке. Тем же моментом будится джоба `links` (`notify_links_pending`,
  решение гейта 1c): уровень 1 связей считается сразу после довекторизации, а
  не через интервал/back-off.
- **summary-джоба** (`build_summary_job`): title-догенерация (миграция, title IS
  NULL) → summarize → merge (слияние дублей) → классификация → описание узла.
  notify будит эту петлю (save/update).
- **judge-джоба** (`build_judge_job`): судья дедупа (по judge-работам, созданным
  embedding-петлёй) + судья структуры (внутри PromotionService, триггер после
  классификации).
- **джоба areas** (`build_areas_job`, субстрат 3.0.0): вектора записей областей
  skills/terms/user (`vector_status='pending'` → батч `embed_texts` → vec0
  области, 'ok'). Без LLM в момент записи (архитектура субстрата §3.3):
  тексты областей — `AreaSpec.embed_text` (skills `name + description`,
  terms `term + context + definition`, user `name + body`); отказ — событие
  `area_embed_failed`, записи остаются pending, повтор по своему back-off.
  Смена модели/размерности дропает area-vec вместе с notes_vec
  (`db._sync_embedding_meta`) — все записи областей возвращаются в pending.

Петли описаны `JobSpec`-ами и обслуживаются каркасом джоб (lsb-0014-02,
`app/services/jobs.py`): тела цикла, back-off и супервизор итерации — общие,
воркер держит работу (`process_*`), перепроверку очереди, сигналы и гигиену.
Сборщики своих джоб — `build_*_job` в конце модуля (регистрация — строка в
реестре каркаса). События работ несут обязательное поле `job` (FR-1.4).
Снимки очередей для `/health.queues` собирает `queues_health` по реестру:
каждая джоба описывает свою очередь сама (`queue_stat`), `/health` при
добавлении джобы не правится (lsb-0014-03, FR-2.2).

Джоба `nodes` (lsb-0011) разбирает накопленный `default` и реклассифицирует
объединённые заметки: (1) промоция, (2) задания `reclass` после сшивания
(lsb-0012 — приоритетный источник), (3) быстрый пул обхода — переезд по
готовой разметке (`hint_path` + `confidence`) без вызова моделей,
(4) классификаторный пул обхода — вызов классификатора в жёстком бюджете
(`JOB_NODES_CLASSIFIER_BUDGET`) по `title` + готовой суммари. Механика переезда
одна на все источники и на причёску после суммаризации (`_apply_node_order`).
Маркер `node_order_at` — анти-зацикливание (ставится при любом исходе разбора).

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
  короткие в to_thread. `run` — asyncio-таска (все джобы реестра под
  каркасным `run_loop`), старт/стоп — в lifespan.

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
from app.services import jobs
from app.services.areas import AreaSpec, ALL_AREAS
from app.services.classifier import Classification, ClassificationError, Classifier
from app.services.dedup import DeduplicationService
from app.services.embedding import Embedder, EmbeddingError
# Back-off и его потолок перенесены в каркас джоб (lsb-0014, arch §3.3):
# каркас владеет контрактом надёжности, петли воркера переиспользуют имена.
# `MAX_INTERVAL_SEC as MAX_INTERVAL_SEC` — явный ре-экспорт: на него по-прежнему
# ссылаются существующие тесты (tests/test_worker.py).
from app.services.jobs import MAX_INTERVAL_SEC as MAX_INTERVAL_SEC
from app.services.jobs import (
    BackoffState,
    JobSpec,
    build_job_specs,
    next_interval,
    queue_snapshot,
    run_loop,
)
from app.services.judge import Judge, JudgeError
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.services.promotion import PromotionService
from app.services.summary import Summarizer, SummaryError, cap_summary
from app.storage import area_vectors, chunks, vectors
from app.storage.db import delete_note_physical, session, transaction

# Сколько хранить выполненные done-работы в worker_jobs (retention). Не env —
# по паттерну TITLE_MAX_WORDS: после этого срока работы вычищаются idle-веткой
# embedding-петли (_purge_done_jobs), очередь не растёт безгранично.
WORKER_JOBS_RETENTION_DAYS = 7

# Интервал джобы зачистки просроченных заметок (lsb-0004-02, этап 4):
# фиксированные 5 минут (решение О. 2026-09-09), без настройки в компоузе.
# Джоба без очереди и без события: её состояние back-off помечено
# `fixed=True` — каркас держит паузу 300 с, back-off не растёт (поведение
# дословно как было, FR-1.5).
EXPIRATION_CLEANUP_INTERVAL_SEC = 5 * 60

# Имена джоб воркера (каркас lsb-0014): `JobSpec.name`, ключ состояния back-off
# и обязательное поле `job` в журнале — одна строка на джобу.
EMBEDDING_JOB = "embedding"
SUMMARY_JOB = "summary"
JUDGE_JOB = "judge"
AREAS_JOB = "areas"
EXPIRATION_JOB = "expiration"
# Джоба «порядок в узлах» (lsb-0011-01): обход накопленного `default`.
NODES_JOB = "nodes"

# Имя работы слота `nodes` после сшивания (lsb-0012, arch §3.5): объединённая
# заметка может лежать в ЛЮБОМ узле — обход `default` её не найдёт, поэтому
# реклассификация ставится заданием.
NODES_RECLASS_KIND = "reclass"

# Единое правило выборки быстрого пула обхода `default` (lsb-0011-01, arch §3.2):
# активные default-заметки с готовой разметкой (`hint_path` + `confidence` не
# ниже порога авто-переезда), ещё не разобранные (`node_order_at IS NULL`),
# свежие первыми — переезд без вызова модели.
_NODES_SWEEP_SELECT = (
    "SELECT id, hint_path, confidence FROM notes "
    "WHERE namespace = 'default' AND deleted_at IS NULL "
    "AND node_order_at IS NULL AND hint_path IS NOT NULL AND confidence >= ? "
    "ORDER BY updated_at DESC, id DESC LIMIT ?"
)

# Классификаторный пул обхода `default` (lsb-0011-02, arch §3.2): активные
# default-заметки БЕЗ разметки (`classified_at IS NULL`), но с готовой непустой
# суммари, ещё не разобранные, свежие первыми — вызов классификатора по
# `title` + `summary` (arch §3.4). Неготовая суммари в пул не попадает: заметка
# ждёт, маркер не ставится (FR-2.4). Число вызовов ограничивает бюджет
# `JOB_NODES_CLASSIFIER_BUDGET`.
_NODES_CLASSIFY_SELECT = (
    "SELECT id, title, summary FROM notes "
    "WHERE namespace = 'default' AND deleted_at IS NULL "
    "AND node_order_at IS NULL AND classified_at IS NULL "
    "AND summary_status = 'ok' AND summary IS NOT NULL AND summary <> '' "
    "ORDER BY updated_at DESC, id DESC LIMIT ?"
)

# Снимок очереди `nodes` для `/health.queues` (FR-2.2): pending-задания
# `reclass` после сшивания (lsb-0012) + кандидаты обоих пулов обхода (быстрый
# и классификаторный); возраст старейшего — старейший из двух источников
# (задания — `created_at`, обход — `updated_at`), `null` — очередь пуста.
_NODES_QUEUE_STAT_SQL = (
    "SELECT "
    "(SELECT COUNT(*) FROM worker_jobs WHERE slot = 'nodes' "
    " AND kind = 'reclass' AND status = 'pending') "
    "+ (SELECT COUNT(*) FROM notes WHERE namespace = 'default' "
    " AND deleted_at IS NULL AND node_order_at IS NULL "
    " AND hint_path IS NOT NULL AND confidence >= ?) "
    "+ (SELECT COUNT(*) FROM notes WHERE namespace = 'default' "
    " AND deleted_at IS NULL AND node_order_at IS NULL "
    " AND classified_at IS NULL AND summary_status = 'ok' "
    " AND summary IS NOT NULL AND summary <> '') AS pending, "
    "MAX("
    "COALESCE((SELECT MAX(CAST(strftime('%s','now') AS INTEGER) - "
    " CAST(strftime('%s', created_at) AS INTEGER)) FROM worker_jobs "
    " WHERE slot = 'nodes' AND kind = 'reclass' AND status = 'pending'), 0), "
    "COALESCE((SELECT MAX(CAST(strftime('%s','now') AS INTEGER) - "
    " CAST(strftime('%s', updated_at) AS INTEGER)) FROM notes "
    " WHERE namespace = 'default' AND deleted_at IS NULL "
    " AND node_order_at IS NULL AND ((hint_path IS NOT NULL AND confidence >= ?) "
    " OR (classified_at IS NULL AND summary_status = 'ok' "
    " AND summary IS NOT NULL AND summary <> ''))), 0)"
    ") AS oldest_pending_sec"
)

# Промпт догенерации названия (решение №9): ЗАШИТ в SummaryService.title
# (follow-up 6b — протокол Summarizer получил метод title; здесь раньше был
# мёртвый дубль константы, генерация шла с промптом суммаризации).
# Думающий вызов слота summary, обрезка до TITLE_MAX_WORDS — механика воркера.


def _embed_input(title: str | None, text: str) -> str:
    """Вход кодирования полного вектора заметки (lsb-0001 FR-1.1).

    Вектор заметки строится по конкатенации названия и текста:
    f"{title}\n{text}". title is NULL (миграционная заметка до догенерации
    названия) — кодируется чистый text без префикса. Чанковые вектора это
    не касается (решение D1): там кодируется чистый текст чанка без title.
    """
    return text if title is None else f"{title}\n{text}"


class BackgroundWorker:
    """Единственный фоновый воркер; очереди — pending-статусы в БД + worker_jobs.

    Цикл, back-off и супервизор итерации — каркасные (app/services/jobs.py,
    lsb-0014-02): воркер описывает свои джобы `JobSpec`-ами (сборщики
    `build_*_job` в конце модуля) и регистрирует их в реестре. События работ
    несут обязательное поле `job` (имя джобы, которой принадлежит работа).
    """

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
        # Состояние back-off по джобам (каркас lsb-0014): один объект на джобу,
        # живёт в воркере — интервал переживает итерации цикла и виден в
        # диагностике (свойства interval/summary_interval/judge_interval/
        # areas_interval). Стартовые интервалы — как были: PENDING_RETRY_SEC у
        # очередей, фиксированные 300 с у зачистки просроченных заметок.
        # Зачистка — единственная джоба с ФИКСИРОВАННЫМ расписанием
        # (`fixed=True`): пауза всегда 300 с, back-off не растёт (FR-1.5).
        self._backoff: dict[str, BackoffState] = {
            EMBEDDING_JOB: BackoffState(settings.pending_retry_sec),
            SUMMARY_JOB: BackoffState(settings.pending_retry_sec),
            JUDGE_JOB: BackoffState(settings.pending_retry_sec),
            AREAS_JOB: BackoffState(settings.pending_retry_sec),
            EXPIRATION_JOB: BackoffState(
                EXPIRATION_CLEANUP_INTERVAL_SEC, fixed=True
            ),
        }
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
        # Сигнал «появилось задание после сшивания» (lsb-0012): будит петлю
        # `nodes` сразу после merge — не ждём часового интервала. Сигнал ставит
        # та же summary-петля (`process_merge_pending`), DI-нотификатор не нужен:
        # джоба живёт внутри воркера (arch §3.5).
        self._nodes_event = asyncio.Event()
        # Сигнал «заметка получила готовый вектор» (решение гейта 1c): будит
        # петлю `links` сразу после довекторизации — уровень 1 связей
        # появляется немедленно, а не ждёт интервала/back-off. Ставит его
        # embedding-петля (`process_pending`); джоба `links` описана в своём
        # модуле (`links.py`), но событие берёт у воркера (общий владелец
        # embedding-очереди; DI-нотификатор не нужен).
        self._links_event = asyncio.Event()
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
        """Текущий интервал embedding-джобы (диагностика, тесты)."""
        return self._backoff[EMBEDDING_JOB].interval

    @property
    def summary_interval(self) -> float:
        """Текущий интервал summary-джобы (диагностика, тесты)."""
        return self._backoff[SUMMARY_JOB].interval

    @property
    def chunk_interval(self) -> float:
        """Интервал чанковой очереди (диагностика, тесты, Фаза 7).

        Чанковая очередь объединена с векторной в одну джобу (решение №10) —
        интервал общий с `interval`.
        """
        return self._backoff[EMBEDDING_JOB].interval

    @property
    def judge_interval(self) -> float:
        """Текущий интервал judge-джобы (диагностика, тесты, Фаза 11)."""
        return self._backoff[JUDGE_JOB].interval

    @property
    def areas_interval(self) -> float:
        """Текущий интервал джобы areas (диагностика, тесты, субстрат 3.0.0)."""
        return self._backoff[AREAS_JOB].interval

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

    def notify_nodes_pending(self) -> None:
        """Разбудить петлю `nodes`: появилось задание после сшивания (lsb-0012).

        Вызывается из той же summary-петли после успешного merge
        (`process_merge_pending` — синхронный код в `asyncio.to_thread`),
        `asyncio.Event.set()` потокобезопасен. Петля немедленно выходит из
        ожидания и разбирает `reclass`, не дожидаясь часового интервала
        `JOB_NODES_INTERVAL_SEC`. DI-нотификатор не нужен: джоба живёт внутри
        воркера (arch §3.5)."""
        self._nodes_event.set()

    def notify_links_pending(self) -> None:
        """Разбудить петлю `links`: заметка получила готовый вектор (гейт 1c).

        Вызывается из embedding-петли после фактической довекторизации
        (`process_pending` — синхронный SQL/HTTP в `asyncio.to_thread`),
        `asyncio.Event.set()` потокобезопасен. Петля немедленно выходит из
        ожидания и разбирает очередь расчёта связей (`links_at IS NULL`), не
        дожидаясь `JOB_LINKS_INTERVAL_SEC`/выросшего back-off. Задание живёт
        маркером `links_at` в БД — событие только ускоряет (контракт
        надёжности не меняется); при недоступных моделях (`EmbeddingError`)
        сигнала нет, очередь `links` пуста и петля ждёт по back-off.
        """
        self._links_event.set()

    def _is_stopping(self) -> bool:
        """Предикат мягкой остановки для каркасного цикла (`run_loop`)."""
        return self._stopping

    def _job_backoff(self, spec: JobSpec) -> BackoffState:
        """Состояние back-off джобы (для незарегистрированной — её интервал)."""
        return self._backoff.setdefault(
            spec.name, BackoffState(spec.interval_sec)
        )

    async def run(self) -> None:
        """Все джобы реестра — единым каркасным циклом (asyncio-таска).

        Джобы собираются `build_job_specs` (реестр `JobSpec`-ов каркаса; свои
        джобы воркер регистрирует в конце модуля) и обслуживаются `run_loop`.
        Обработанные партии идут одна за другой (очередь выгребаем сразу);
        пустой прогон — ожидание сигнала `notify_*` (форма «по требованию») или
        пауза по back-off. Джобы независимы: интервал и выгребание —
        раздельные (свой `BackoffState` на джобу).
        """
        specs = build_job_specs(self, self._settings)
        await asyncio.gather(
            *(
                run_loop(spec, self._is_stopping, self._job_backoff(spec))
                for spec in specs
            )
        )

    # --- работа джоб (цикл, back-off и супервизор — в каркасе) -----------------

    async def _process_embedding(self) -> int:
        """Прогон embedding-джобы: notes-очередь (в потоке) + чанковая очередь.

        Обе очереди — одна джоба (решение №10): полные вектора заметок
        (`process_pending` — синхронный SQL/HTTP, уходит в `to_thread`) и
        чанковая очередь (`process_pending_chunks` — своя async-обработка с
        Semaphore внутри). Возвращает число фактически обработанных заданий:
        0 уводит каркасную петлю в ожидание с back-off (arch §3.2).
        """
        processed = await asyncio.to_thread(self.process_pending)
        processed += await self.process_pending_chunks()
        return processed

    async def _process_summary(self) -> int:
        """Прогон summary-джобы: title-доген → summarize → merge (три шага).

        Каждый шаг — синхронный SQL/LLM, уходит в `to_thread`; порядок шагов
        сохранён (как в исходной петле). Возвращает число обработанных
        заданий: 0 уводит каркасную петлю к ожиданию сигнала
        `notify_summary_pending` (форма «по требованию», пул 6) или таймаута.
        """
        processed = await asyncio.to_thread(self.process_title_pending)
        processed += await asyncio.to_thread(self.process_summary_pending)
        processed += await asyncio.to_thread(self.process_merge_pending)
        return processed

    async def _run_summary(self) -> None:
        """Обслуживать summary-джобу каркасным циклом (запуск одной джобы).

        Тело петли живёт в каркасе; метод оставлен для изолированного запуска
        одной джобы — так тесты пула 6 (lost wakeup) проверяют селект по
        очереди и `clear()` сигнала без остальных петель.
        """
        spec = build_summary_job(self, self._settings)
        if spec is None:
            return  # без суммаризатора джоба неприменима (тестовый режим)
        await run_loop(spec, self._is_stopping, self._backoff[SUMMARY_JOB])

    async def _process_judge(self) -> int:
        """Прогон judge-джобы: партия judge-работ (дедуп) в `to_thread`.

        0 уводит каркасную петлю к ожиданию сигнала `notify_judge_pending`
        (форма «по требованию», пул 6) или таймаута интервала.
        """
        return await asyncio.to_thread(self.process_judge_pending)

    async def _run_judge(self) -> None:
        """Обслуживать judge-джобу каркасным циклом (запуск одной джобы).

        См. `_run_summary`: изоляция одной джобы — для тестов пула 6
        (lost wakeup) по judge-очереди.
        """
        spec = build_judge_job(self, self._settings)
        await run_loop(spec, self._is_stopping, self._backoff[JUDGE_JOB])

    # --- джоба areas: вектора записей областей (субстрат 3.0.0) ---------------

    async def _process_areas(self) -> int:
        """Прогон джобы areas: партия pending-записей областей в `to_thread`.

        Своя очередь, свой back-off и свой сигнал: отказ векторизации областей
        не мешает заметкам и наоборот (ARCH §3.4). 0 уводит каркасную петлю к
        ожиданию `notify_areas_pending` (пул 6) или таймаута интервала.
        """
        return await asyncio.to_thread(self.process_pending_areas)

    # --- джоба зачистки просроченных заметок (lsb-0004-02, этап 4) -----------

    async def _process_expiration(self) -> int:
        """Прогон джобы expiration: зачистка просроченных заметок.

        Выгребает note_expirations: заметки с `expires_at <= now()` удаляются
        ПОЛНОСТЬЮ (notes+chunks+вектора+fts) вместе со строкой из
        note_expirations (`process_expired_notes` в `to_thread`); удалено > 0
        — событие `expiration_cleanup` с обязательным полем `job`. Джоба
        обслуживается по ФИКСИРОВАННОМУ расписанию (её состояние back-off —
        `fixed=True`): пауза всегда EXPIRATION_CLEANUP_INTERVAL_SEC
        (фиксированные 5 минут, решение О. 2026-09-09), back-off не растёт —
        зачистка не зависит от внешних сервисов, расписание не
        ускоряется/замедляется (FR-1.5). Возвращает число удалённых заметок
        (расписание ведёт интервал, а не прогресс — см. `run_loop`).
        """
        deleted = await asyncio.to_thread(self.process_expired_notes)
        if deleted:
            logging.getLogger("app").info(
                "expiration cleanup: purged expired notes",
                extra={
                    "event": "expiration_cleanup",
                    "job": EXPIRATION_JOB,
                    "count": deleted,
                },
            )
        return deleted

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
                    "job": EMBEDDING_JOB,
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

    def _ensure_job(
        self, slot: str, kind: str, note_id: int, payload: str | None = None
    ) -> None:
        """Идемпотентно поставить работу в очередь слота (arch §3.5, FR-2.5).

        `INSERT ... WHERE NOT EXISTS (pending с теми же slot/kind/note_id)`:
        частая правка одной заметки не плодит дубли заданий (нужно джобам,
        которые ставят работу по событию). Существующий `_create_job`
        (judge/merge) не трогается: там дедуп не требуется — пара
        «поздняя ↔ ранняя» и вердикт судьи дают свои инварианты.
        """
        self._ensure_job_table()
        with session(self._settings) as conn, transaction(conn):
            conn.execute(
                "INSERT INTO worker_jobs (slot, kind, note_id, payload) "
                "SELECT ?, ?, ?, ? WHERE NOT EXISTS ("
                "SELECT 1 FROM worker_jobs WHERE slot = ? AND kind = ? "
                "AND note_id = ? AND status = 'pending')",
                (slot, kind, note_id, payload, slot, kind, note_id),
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

    # --- наблюдаемость очередей (lsb-0014-03, FR-2.2) -------------------------

    def queues_health(self) -> dict[str, dict]:
        """Собрать `/health.queues` по реестру джоб (FR-2.2): только SQL.

        У каждой зарегистрированной джобы с очередью берётся её собственный
        снимок (`JobSpec.queue_stat`) — `/health` не правится при добавлении
        джобы. Джоба без очереди (`expiration`) в объект не попадает;
        обращений к моделям нет (только чтение pending-состояния).
        """
        queues: dict[str, dict] = {}
        for spec in build_job_specs(self, self._settings):
            if spec.queue is None or spec.queue_stat is None:
                continue
            queues[spec.queue] = spec.queue_stat()
        return queues

    def _vector_queue_stat(self) -> dict:
        """Снимок очереди векторизации заметок: pending и возраст старейшего.

        Источник — `notes.vector_status='pending'` (тот же предикат, что у
        легаси-счётчика `pending_vector`); SQL живёт рядом с `health_counts`
        в NoteService.
        """
        return self._notes.vector_queue_stat()

    def _summary_queue_stat(self) -> dict:
        """Снимок очереди суммаризации заметок (`notes.summary_status`)."""
        return self._notes.summary_queue_stat()

    def _judge_queue_stat(self) -> dict:
        """Снимок очереди judge: pending-работы слота в `worker_jobs`.

        Число — `worker_jobs(slot='judge', status='pending')`, возраст
        старейшего — `now - MIN(created_at)`. Таблица создаётся лениво
        (`_ensure_job_table`), моделей не зовём.
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS pending, "
                "MAX(CAST(strftime('%s','now') AS INTEGER) - "
                "CAST(strftime('%s', created_at) AS INTEGER)) "
                "AS oldest_pending_sec FROM worker_jobs "
                "WHERE slot = 'judge' AND status = 'pending'"
            ).fetchone()
        return queue_snapshot(row["pending"], row["oldest_pending_sec"])

    def _areas_queue_stat(self) -> dict:
        """Снимок очереди областей: сумма pending по `ALL_AREAS` (FR-2.2).

        Источник pending — тот же, что у `_areas_queue_empty`: pending-записи
        активных строк каждой области (skills/terms/user). Суммы складываются,
        возраст старейшего — максимум `now - updated_at` по областям; пусто —
        `null`.
        """
        pending = 0
        oldest: int | None = None
        with session(self._settings) as conn:
            for spec in ALL_AREAS:
                row = conn.execute(
                    "SELECT COUNT(*) AS pending, "
                    "MAX(CAST(strftime('%s','now') AS INTEGER) - "
                    "CAST(strftime('%s', updated_at) AS INTEGER)) "
                    "AS oldest_pending_sec FROM "
                    f"{spec.table} WHERE vector_status = 'pending' "
                    "AND deleted_at IS NULL"
                ).fetchone()
                pending += int(row["pending"])
                value = row["oldest_pending_sec"]
                if value is not None:
                    oldest = int(value) if oldest is None else max(oldest, int(value))
        return queue_snapshot(pending, oldest)

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
        Если хотя бы одна заметка батча реально получила вектор, будится и
        петля `links` (`notify_links_pending`, решение гейта 1c): уровень 1
        связей считается сразу, а не через интервал — но по тому же маркеру
        `links_at`, поэтому задание не теряется по построению. Одно событие на
        батч достаточно. Отказ кодирования (`EmbeddingError`) выходит раньше —
        события нет, задания ждут готового вектора (back-off).
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
        if processed:
            # Заметка батча реально получила вектор — будим `links` (гейт 1c).
            # Одно событие на батч достаточно; отказ кодирования сюда не
            # доходит (EmbeddingError выше) — «модели недоступны → задания
            # ждут» сохраняется.
            self.notify_links_pending()
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
                    "job": AREAS_JOB,
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
                            "job": JUDGE_JOB,
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
                        "job": SUMMARY_JOB,
                        "older_id": older_id,
                        "note_id": note_id,
                    },
                )
                continue  # работа остаётся pending — повтор по back-off
            logging.getLogger("app").info(
                "dedup: duplicate merged into earlier note",
                extra={
                    "event": "dedup_merged",
                    "job": SUMMARY_JOB,
                    "older_id": older_id,
                    "note_id": note_id,
                    "cosine": payload.get("cosine"),
                },
            )
            self._mark_job_done(job["id"])
            done += 1
            # Реклассификация объединённой заметки (lsb-0012, arch §3.5): узел
            # ранней мог не подойти новому содержанию — ставим задание слоту
            # `nodes` (объединённая заметка может лежать в любом узле — обход
            # `default` её не найдёт) и будим его петлю сразу, не дожидаясь
            # часового интервала. Неудачная постановка/обработка данные не
            # портит: заметка остаётся в текущем узле, `classified_at` — NULL.
            self._ensure_job(NODES_JOB, NODES_RECLASS_KIND, older_id)
            self.notify_nodes_pending()
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
                    "job": JUDGE_JOB,
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
                    "job": JUDGE_JOB,
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
                    extra={
                        "event": "title_failed",
                        "job": SUMMARY_JOB,
                        "note_id": row["id"],
                    },
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
                        "job": SUMMARY_JOB,
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

        Лимит длины (решение гейта 3.1.0, 2026-09-19): ответ модели сохраняется
        только после `cap_summary` — жёсткое усечение до MAX_SUMMARY_CHARS по
        границе слова с многоточием. Это единственная точка записи модельного
        саммари, поэтому лимит не зависит от дисциплины модели.
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
                    extra={
                        "event": "summary_failed",
                        "job": SUMMARY_JOB,
                        "note_id": row["id"],
                    },
                )
                continue  # отказ: status pending остаётся, повтор по back-off
            # Страховка длины (решение гейта 3.1.0, 2026-09-19): модель может
            # не послушать промпт («no more than 150 characters») — в БД уходит
            # усечённое по границе слова саммари, всегда ≤ MAX_SUMMARY_CHARS.
            summary = cap_summary(summary, self._settings.max_summary_chars)
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
                            extra={
                                "event": "classify_crashed",
                                "job": SUMMARY_JOB,
                                "note_id": row["id"],
                            },
                            exc_info=True,
                        )
        return done

    # --- причёска (Фаза 10, Шаг 4) -------------------------------------------

    def _apply_node_order(
        self,
        note_id: int,
        expected_namespace: str,
        result: Classification,
        *,
        current_namespace: str | None = None,
        no_hint_reason: str = "low_confidence",
    ) -> tuple[str, str, str | None]:
        """Единственная точка переезда заметки по разметке (lsb-0011-01, arch §3.3).

        Один UPDATE: разметка (`hint_path`, `confidence`, `classified_at`) +
        маркер разбора `node_order_at` + (при переезде) `namespace` и
        `vector_status='pending'` (пере-кодировка в партицию нового узла —
        существующий механизм). Guard `namespace = :expected_namespace` и
        `deleted_at IS NULL`: операторский/клиентский переезд в полёте фоном
        не перебивается — `rowcount = 0` даёт `kept`/`node_changed`.

        Целевой узел считается ДО транзакции: отказ валидации разметки
        (`NamespaceValidationError` из `_auto_move_target`, в т.ч. мусорный
        `hint_path`) не пишет в БД вовсе — строгая семантика «отказ = не
        размечено». Возврат `(outcome, reason, target)`: `moved` — переезд;
        `kept` — оставлена (`hint_unknown` — узла нет в реестре,
        `no_hint_reason` — модель узла не предложила, `low_confidence` —
        уверенность ниже порога, `same_node` — узел тот же, `node_changed` —
        узел сменён в полёте). Все reason — из словаря arch §3.7.

        Механика одна на все источники (arch §3.3): её переиспользует
        причёска после суммаризации (вход по полному тексту), обход `default`
        (быстрый путь и классификаторный пул) и задание после сшивания.

        `current_namespace` — узел заметки, прочитанный В МОМЕНТ РЕШЕНИЯ
        (только задание после сшивания, lsb-0012): тогда результат без узла
        даёт `no_hint_reason` (`result_default` — заметка НЕ понижается,
        FR-2.4), а уверенный выбор текущего узла — `kept`/`same_node` без
        лишней пере-векторизации. Для обхода и причёски (`None`) поведение
        прежнее: `expected_namespace` — `'default'`.
        """
        target = self._auto_move_target(result)
        confident = (
            result.confidence >= self._settings.namespace_auto_move_min_confidence
        )
        move = target is not None and confident
        if current_namespace is not None and not result.hint_path:
            # Результат «общая»: движение «узел → свалка» запрещено (FR-2.4) —
            # заметка остаётся в узле ранней.
            target = None
            move = False
        elif (
            current_namespace is not None
            and confident
            and target == current_namespace
        ):
            move = False  # узел тот же: переезда (и пере-векторизации) нет
        # Маркер разбора ставится в ТОЙ ЖЕ транзакции, что и разметка
        # (lsb-0011-01, §3.6): атомарно, анти-зацикливание не отстаёт от решения.
        columns = [
            "hint_path = ?",
            "confidence = ?",
            "classified_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')",
            "node_order_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')",
        ]
        params: list[object] = [result.hint_path, result.confidence]
        if move:
            columns.append("namespace = ?")
            columns.append("vector_status = 'pending'")
            params.append(target)
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE notes SET "
                + ", ".join(columns)
                + " WHERE id = ? AND namespace = ? AND deleted_at IS NULL",
                (*params, note_id, expected_namespace),
            )
        if not cursor.rowcount:
            return "kept", "node_changed", target
        if move:
            return "moved", "hint_exists", target
        if target is None:
            return (
                "kept",
                "hint_unknown" if result.hint_path else no_hint_reason,
                None,
            )
        if current_namespace is not None and confident and target == current_namespace:
            return "kept", "same_node", target
        return "kept", "low_confidence", target

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

        lsb-0011-01: сама механика переезда вынесена в общую точку
        `_apply_node_order` (её же переиспользует обход `default`) — здесь
        остаётся только вход по полному тексту и триггер промоции; поведение
        классификации после суммаризации не меняется.
        """
        if self._classifier is None:
            return  # тестовый режим без классификатора
        known = self._namespaces.list_all()["namespaces"]
        try:
            result = self._classifier.classify(text, known)
        except ClassificationError:
            logging.getLogger("app").warning(
                "classify: failed — note stays in default, retry on next update",
                extra={
                    "event": "classify_failed",
                    "job": SUMMARY_JOB,
                    "note_id": note_id,
                },
            )
            return
        # Пул 6: переезд — ОДНА общая точка `_apply_node_order` (разметка +
        # namespace/vector_status одним UPDATE, guard внутри); отказ
        # вычисления цели (в т.ч. NamespaceValidationError на мусорном hint)
        # ничего не пишет в БД.
        outcome, _reason, target = self._apply_node_order(note_id, "default", result)
        # Лог — только при ФАКТИЧЕСКОМ переезде (rowcount+condition).
        if outcome == "moved":
            logging.getLogger("app").info(
                "classify: default note auto-moved into existing node",
                extra={
                    "event": "classified_moved",
                    "job": SUMMARY_JOB,
                    "note_id": note_id,
                    "namespace": target,
                    "confidence": result.confidence,
                },
            )
        # Триггер домена (Фаза 10, Шаг 5): разметка могла докинуть hint-группу
        # до порога — прогоняем конвейер промоции (авто-создание/слияние).
        self._run_promotion()

    def _run_promotion(self, job: str = SUMMARY_JOB) -> None:
        """Триггер домена (Шаг 5) после классификации default-заметки.

        Вызывается причёской (слот summary, `job=summary`) и обходом `nodes`
        перед подметанием `default` (`job=nodes`, lsb-0011-01): промоция идёт
        первой, чтобы заметка не «переезжала» в узел, который появится в этом
        же прогоне. Имя джобы в журнале — параметр (события не переименовываем,
        меняется лишь значение обязательного поля `job`).

        Сбои триггера не роняют воркер: это этап обогащения, а не конвейера
        данных — суммаризация/векторизация важнее структурной автоматики.
        Ожидаемые отказы describer/судьи обрабатываются внутри
        PromotionService (кандидат остаётся без вердикта, NFR-3); здесь
        ловится ВСЁ остальное (включая баги) — warning с traceback в логи,
        петли очередей живут. Повтор — следующая классификация default-
        заметки или следующий прогон обхода: группы не теряются, просто
        дотягивают до порога позже.
        """
        if self._promoter is None:
            return  # тестовый режим без триггера
        try:
            report = self._promoter.run()
        except Exception:
            logging.getLogger("app").warning(
                "promotion: run failed — trigger deferred to next classification",
                extra={
                    "event": "promotion_failed",
                    "job": job,
                    "reason": "run",
                },
                exc_info=True,
            )
            return
        if any(report.values()):
            # Сводка — одним ключом: 'created' конфликтует с атрибутом
            # LogRecord (время создания) — extra его не принимает.
            logging.getLogger("app").info(
                "promotion: trigger run finished",
                extra={
                    "event": "promotion_run",
                    "job": job,
                    "report": report,
                },
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

    # --- джоба «порядок в узлах» (lsb-0011) -----------------------------------

    def process_nodes(self, budget: int | None = None) -> int:
        """Прогон джобы `nodes`: промоция → задания → обход `default`.

        Полный порядок прогона (FR-1.2/FR-2.3/FR-3.3, arch §3.2):
        (1) промоция — `PromotionService.run()` (создание/слияние листов,
        ретро-перекладка), отказ не роняет джобу;
        (2) задания `reclass` после сшивания (lsb-0012) — приоритетный
        источник: объединённая заметка может лежать в любом узле, обход
        `default` её не найдёт;
        (3) быстрый пул обхода — остатком бюджета, переезд БЕЗ вызова модели;
        (4) классификаторный пул обхода — остатком бюджета, но не более
        `JOB_NODES_CLASSIFIER_BUDGET` вызовов классификатора за прогон.
        Бюджет `JOB_NODES_BATCH` (дефолт 20) — ОБЩИЙ на оба источника за
        прогон: не более него обработок (`moved` + `kept`); повторные прогоны
        продолжают backlog. Заметка, ждущая модель или суммари, не
        обрабатывается и маркера не получает — она попадёт в следующий прогон.

        Быстрый пул (arch §3.2): default-заметки с готовой разметкой
        (`hint_path` + `confidence` не ниже порога авто-переезда), ещё не
        разобранные (`node_order_at IS NULL`), свежие первыми. Узел
        зарегистрирован → переезд (`moved`/`hint_exists`, разметка не
        меняется); узла нет в реестре → `kept`/`hint_unknown` (лист создаст
        промоция — узел здесь не создаём).

        Каждая разобранная заметка — событие `node_order` (`job='nodes'`,
        `source='sweep'`/`'after_merge'`, `outcome`, `reason`, `target`,
        `note_id`). Возврат — число разобранных заметок (0 уводит каркасную
        петлю к ожиданию интервала или сигнала `notify_nodes_pending`).
        """
        limit = self._settings.job_nodes_batch if budget is None else budget
        # (1) Промоция первой: заметка не «переезжает» в узел, который
        # появится в этом же прогоне (FR-3.1). Отказ не роняет джобу (FR-3.2).
        self._run_promotion(NODES_JOB)
        # (2) Задания после сшивания — приоритетный источник (FR-3.3).
        done = self._process_reclass_jobs(limit)
        # (3) Быстрый пул обхода, (4) классификаторный — остатком бюджета.
        done += self._sweep_fast_marks(limit - done)
        done += self._sweep_classifier(limit - done)
        return done

    def _process_reclass_jobs(self, limit: int) -> int:
        """Обработать партию заданий `reclass` (после сшивания); их число.

        Порядок — по `id` очереди (свежие события не ждут за старыми).
        Задание, ждущее готовой суммари, остаётся pending и прогрессом не
        считается: петля уходит в сон по back-off, а не крутится вхолостую.
        """
        if limit <= 0:
            return 0
        done = 0
        for job in self._pending_jobs(NODES_JOB, NODES_RECLASS_KIND, limit):
            if self._process_reclass_job(int(job["note_id"])):
                self._mark_job_done(job["id"])
                done += 1
        return done

    def _process_reclass_job(self, note_id: int) -> bool:
        """Разобрать одно задание после сшивания; True — задание снято с очереди.

        Читаем заметку и решаем (FR-2.2/FR-2.5 lsb-0012, arch §3.5):

        * заметки нет (удалена оператором) — задание снимаем: решать нечего;
        * суммари ещё не готова (`summary_status != 'ok'` или пустая) —
          задание ОСТАЁТСЯ pending и прогрессом не считается (ждёт по
          back-off), по fallback-усечению решение не принимается, маркер
          разбора не ставится — событие `node_order`/`summary_pending`;
        * по готовой суммари — вызов классификатора на `title` + `summary`
          (двухступенчатость с полным `text` не вводим — arch §3.4) и общая
          точка переезда `_apply_node_order` с guard-ом по УЗЛУ, ПРОЧИТАННОМУ
          при решении: операторский/клиентский переезд в полёте фоном не
          перебивается (`node_changed`). Результат `default` заметку не
          понижает (`result_default`), тот же узел — `same_node`.

        Без классификатора (тестовый режим) решение не принимается: задание
        остаётся pending, данные не портятся.
        """
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT title, summary, summary_status, namespace FROM notes "
                "WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            ).fetchone()
        if row is None:
            return True  # заметки нет — задание снимаем
        summary = row["summary"] or ""
        if row["summary_status"] != "ok" or not summary.strip():
            self._log_node_order(
                note_id, "kept", "summary_pending", None, source="after_merge"
            )
            return False  # ждём суммари: задание остаётся pending
        if self._classifier is None:
            return False  # тестовый режим без классификатора
        known = self._namespaces.list_all()["namespaces"]
        text = f"{row['title']}\n{summary}" if row["title"] else summary
        try:
            result = self._classifier.classify(text, known)
        except ClassificationError:
            # Отказ классификатора данные не портит (FR-3.1): заметка остаётся
            # в текущем узле, разметка не пишется, задание — pending.
            logging.getLogger("app").warning(
                "nodes: reclass failed — note stays in its node, retry by back-off",
                extra={
                    "event": "node_order_failed",
                    "job": NODES_JOB,
                    "source": "after_merge",
                    "note_id": note_id,
                },
            )
            return False
        outcome, reason, target = self._apply_node_order(
            note_id,
            row["namespace"],
            result,
            current_namespace=row["namespace"],
            no_hint_reason="result_default",
        )
        self._log_node_order(note_id, outcome, reason, target, source="after_merge")
        return True

    def _sweep_fast_marks(self, budget: int) -> int:
        """Быстрый пул обхода `default`: переезд по готовой разметке, без модели.

        Заметки с `hint_path` + `confidence` не ниже порога авто-переезда,
        ещё не разобранные (`node_order_at IS NULL`), свежие первыми. Маркер
        `node_order_at` ставится при любом исходе (анти-зацикливание, §3.6):
        оставленная заметка не выедает бюджет повторно до изменения
        текста/названия (сброс маркера — NoteService.update/merge_pair).
        """
        if budget <= 0:
            return 0
        threshold = self._settings.namespace_auto_move_min_confidence
        with session(self._settings) as conn:
            rows = conn.execute(_NODES_SWEEP_SELECT, (threshold, budget)).fetchall()
        done = 0
        for row in rows:
            note_id = int(row["id"])
            result = Classification(row["hint_path"], row["confidence"])
            outcome, reason, target = self._apply_node_order(
                note_id, "default", result
            )
            done += 1
            self._log_node_order(note_id, outcome, reason, target)
        return done

    def _sweep_classifier(self, budget: int) -> int:
        """Классификаторный пул обхода `default`: разметка через модель.

        Заметки без разметки (`classified_at IS NULL`) с ГОТОВОЙ непустой
        суммари, свежие первыми; вход классификатору — `title` + `summary`
        (arch §3.4, полный `text` не отдаём). Число вызовов классификатора за
        прогон ограничено бюджетом `JOB_NODES_CLASSIFIER_BUDGET` и остатком
        общего бюджета. Отказ классификатора заметку не помечает — она
        останется кандидатом следующего прогона (FR-3.2).
        """
        limit = min(budget, self._settings.job_nodes_classifier_budget)
        if limit <= 0 or self._classifier is None:
            return 0
        with session(self._settings) as conn:
            rows = conn.execute(_NODES_CLASSIFY_SELECT, (limit,)).fetchall()
        known = self._namespaces.list_all()["namespaces"]
        done = 0
        for row in rows:
            note_id = int(row["id"])
            summary = row["summary"] or ""
            text = f"{row['title']}\n{summary}" if row["title"] else summary
            try:
                result = self._classifier.classify(text, known)
            except ClassificationError:
                logging.getLogger("app").warning(
                    "nodes: classify failed — note stays in default, retry later",
                    extra={
                        "event": "classify_failed",
                        "job": NODES_JOB,
                        "note_id": note_id,
                    },
                )
                continue  # маркер не ставим: заметка снова кандидат обхода
            outcome, reason, target = self._apply_node_order(
                note_id,
                "default",
                result,
                no_hint_reason="no_hint_classified",
            )
            done += 1
            self._log_node_order(note_id, outcome, reason, target)
        return done

    def _log_node_order(
        self,
        note_id: int,
        outcome: str,
        reason: str,
        target: str | None,
        source: str = "sweep",
    ) -> None:
        """Событие `node_order` джобы `nodes` (FR-4.2): исход разбора заметки.

        `reason` — из словаря arch §3.7 (`hint_exists`, `hint_unknown`,
        `low_confidence`, `no_hint_classified`, `summary_pending`, `same_node`,
        `result_default`, `node_changed`), `source` — источник решения
        (`sweep` — обход `default`, `after_merge` — задание после сшивания).
        Имена событий и полей существующие — наблюдаемость джобы не меняется.
        """
        logging.getLogger("app").info(
            "nodes: decided note order",
            extra={
                "event": "node_order",
                "job": NODES_JOB,
                "source": source,
                "note_id": note_id,
                "outcome": outcome,
                "reason": reason,
                "target": target,
            },
        )

    def _nodes_queue_stat(self) -> dict:
        """Снимок очереди `nodes` для `/health.queues` (FR-2.2).

        `pending` — pending-задания `reclass` после сшивания + кандидаты обоих
        пулов обхода (быстрый и классификаторный); `oldest_pending_sec` —
        возраст старейшего из двух источников (задания — `created_at`,
        обход — `updated_at`), `null` — очередь пуста. Только SQL, без
        обращений к моделям; снимок работает и у выключенной джобы.
        """
        self._ensure_job_table()
        threshold = self._settings.namespace_auto_move_min_confidence
        with session(self._settings) as conn:
            row = conn.execute(
                _NODES_QUEUE_STAT_SQL, (threshold, threshold)
            ).fetchone()
        return queue_snapshot(row["pending"], row["oldest_pending_sec"])

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


# --- сборщики джоб воркера (каркас lsb-0014-02) -------------------------------
#
# Регистрация вместо копии цикла (FR-1.1): каждая джоба описывается `JobSpec`
# и добавляется в реестр каркаса; цикл, back-off и супервизор итерации — общие
# (`app/services/jobs.py`). Интервалы существующих петель — как были (FR-1.5):
# PENDING_RETRY_SEC у очередей и фиксированные 300 с у зачистки (новых env эта
# постановка не заводит). `queue_stat` — снимок своей очереди для
# `/health.queues` (lsb-0014-03, FR-2.2): у `expiration` очереди нет (`None`).


def build_embedding_job(worker: BackgroundWorker, settings: Settings) -> JobSpec:
    """Джоба `embedding`: notes-очередь + чанковая очередь (одна джоба).

    Форма «по интервалу»: сигнала `notify_*` у embedding-петли не было и
    раньше. Пустой прогон — гигиена `worker_jobs` (`idle_hook`, пул 5) и пауза
    PENDING_RETRY_SEC с back-off; перепроверка очереди не нужна (задание живёт
    в статусе заметки и будет выбрано следующим прогоном).
    """
    return JobSpec(
        name=EMBEDDING_JOB,
        queue="vector",
        interval_sec=settings.pending_retry_sec,
        batch=None,
        enabled=True,
        process=worker._process_embedding,
        queue_empty=None,
        wait_event=None,
        idle_hook=worker._purge_done_jobs,
        queue_stat=worker._vector_queue_stat,
    )


def build_summary_job(
    worker: BackgroundWorker, settings: Settings
) -> JobSpec | None:
    """Джоба `summary`: title-доген → summarize → merge, сигнал `summary`.

    Без суммаризатора (тестовый режим Фазы 3) джоба неприменима — сборщик
    возвращает None, в реестр она не попадает (петли не было и раньше).
    """
    if worker._summarizer is None:
        return None
    return JobSpec(
        name=SUMMARY_JOB,
        queue="summary",
        interval_sec=settings.pending_retry_sec,
        batch=None,
        enabled=True,
        process=worker._process_summary,
        queue_empty=worker._summary_queue_empty,
        wait_event=worker._summary_event,
        idle_hook=None,
        queue_stat=worker._summary_queue_stat,
    )


def build_judge_job(worker: BackgroundWorker, settings: Settings) -> JobSpec:
    """Джоба `judge`: партия judge-работ (дедуп), сигнал `judge`."""
    return JobSpec(
        name=JUDGE_JOB,
        queue="judge",
        interval_sec=settings.pending_retry_sec,
        batch=None,
        enabled=True,
        process=worker._process_judge,
        queue_empty=worker._judge_queue_empty,
        wait_event=worker._judge_event,
        idle_hook=None,
        queue_stat=worker._judge_queue_stat,
    )


def build_areas_job(worker: BackgroundWorker, settings: Settings) -> JobSpec:
    """Джоба `areas`: вектора записей областей (субстрат 3.0.0), сигнал `areas`."""
    return JobSpec(
        name=AREAS_JOB,
        queue="areas",
        interval_sec=settings.pending_retry_sec,
        batch=None,
        enabled=True,
        process=worker._process_areas,
        queue_empty=worker._areas_queue_empty,
        wait_event=worker._areas_event,
        idle_hook=None,
        queue_stat=worker._areas_queue_stat,
    )


def build_expiration_job(worker: BackgroundWorker, settings: Settings) -> JobSpec:
    """Джоба `expiration`: зачистка просроченных заметок (lsb-0004-02, этап 4).

    Очереди нет (зачистка — не pending-состояние) и сигнала нет: джоба
    описана формой «по интервалу», а её состояние back-off помечено
    `fixed=True` (создаётся в `__init__`) — пауза всегда 300 с, без back-off
    (решение О. 2026-09-09; поведение сохранено дословно, FR-1.5).
    """
    return JobSpec(
        name=EXPIRATION_JOB,
        queue=None,
        interval_sec=EXPIRATION_CLEANUP_INTERVAL_SEC,
        batch=None,
        enabled=True,
        process=worker._process_expiration,
        queue_empty=None,
        wait_event=None,
        idle_hook=None,
        queue_stat=None,
    )


def build_nodes_job(worker: BackgroundWorker, settings: Settings) -> JobSpec:
    """Джоба `nodes`: обход `default` и реклассификация после сшивания (lsb-0011).

    Форма «по интервалу + событие» (arch §3.1): пустой прогон — сон на
    `JOB_NODES_INTERVAL_SEC`, но сигнал `notify_nodes_pending` (задание после
    сшивания, lsb-0012) будит петлю сразу. Прогон — промоция, задания
    `reclass`, затем обход `default` (быстрый пул без модели + классификаторный
    в бюджете `JOB_NODES_CLASSIFIER_BUDGET`). `JOB_NODES_ENABLED=false`
    джобу не запускает, но очередь остаётся видна в `/health` (реестр её
    сохраняет). Синхронный SQL/механика уходят в поток — event loop не занимаем.

    Перепроверку очереди (`queue_empty`) не задаём: заметка, ждущая модель или
    суммари, — отложенное задание, и петля ДОЛЖНА уйти в сон по back-off, а не
    крутиться вхолостую (arch §3.2, «отложенное задание — не прогресс»).
    """
    batch = settings.job_nodes_batch

    async def process() -> int:
        return await asyncio.to_thread(worker.process_nodes, batch)

    return JobSpec(
        name=NODES_JOB,
        queue=NODES_JOB,
        interval_sec=settings.job_nodes_interval_sec,
        batch=batch,
        enabled=settings.job_nodes_enabled,
        process=process,
        queue_empty=None,
        wait_event=worker._nodes_event,
        idle_hook=None,
        queue_stat=worker._nodes_queue_stat,
    )


# Регистрация петель воркера в реестре каркаса (FR-1.1): каркас собирает джобы
# из `JOB_BUILDERS`; свой модуль дописывает свои сборщики после их определения
# (импорт односторонний — worker → jobs, круга нет). Порядок — как у петель
# раньше: embedding, summary, judge, areas, expiration, nodes.
jobs.JOB_BUILDERS += (
    build_embedding_job,
    build_summary_job,
    build_judge_job,
    build_areas_job,
    build_expiration_job,
    build_nodes_job,
)
