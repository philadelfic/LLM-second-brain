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
  не через интервал/back-off. Форма самой embedding-джобы — «по интервалу +
  событие `embedding`» (gate 2026-09-19): запись/правка заметки
  (`memory_save`/`memory_update`), слияние дублей и догенерация названия
  возвращают заметку в `vector_status='pending'` и будят петлю сигналом
  `notify_embedding_pending` — свежая заметка векторизуется сразу, а не после
  выросшего back-off (до 15 мин). Событие только ускоряет: задание живёт в
  статусе заметки; при недоступных моделях (`EmbeddingError`) сигнала нет и
  петля честно уходит в ожидание с back-off.
- **summary-джоба** (`build_summary_job`): title-догенерация (миграция, title IS
  NULL) → summarize → merge (слияние дублей) → классификация → описание узла.
  notify будит эту петлю (save/update). Готовая выжимка заметки будит и петлю
  `nodes` (`notify_nodes_pending`, lsb-0012, по образцу `notify_links_pending`
  после довекторизации): её задание `reclass` ставится в момент сшивания
  дубликатов, когда суммари объединённой заметки ещё пересчитывается, — и ждёт
  именно этого момента. Сигнал — только при фактической готовности
  (`summary_status='ok'`): при отказе суммаризации его нет, задание честно
  ждёт по back-off.
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
import time

from app.config import TITLE_MAX_WORDS, Settings
from app.services import history
from app.services import jobs
from app.services.areas import AreaSpec, ALL_AREAS, enabled_areas
from app.services.classifier import Classification, ClassificationError, Classifier
from app.services.decision import SLOT_DECISION, DecisionService
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
from app.services.llm_client import CHECK_AUTH_FAILED, CHECK_OK, ChatMeta
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.services.promotion import PromotionService
from app.services.slot_breaker import SlotBreaker
from app.services.slot_budget import slot_config_fingerprint
from app.services.slot_outcome import SlotOutcome, outcome_of
from app.services.summary import Summarizer, SummaryError, cap_summary
from app.services.terms import TermsService
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

# LLM-слоты джоб (№39, arch §6.2): разрыв живёт на СЛОТ, а не на джобу —
# слот summary делят суммаризация/слияние/классификация/описания и обход
# `nodes`; слот judge — dedup-судья и судья структуры.
SLOT_EMBEDDING = "embedding"
SLOT_SUMMARY = "summary"
SLOT_JUDGE = "judge"
SLOT_NAMES = (SLOT_EMBEDDING, SLOT_SUMMARY, SLOT_JUDGE)

# Имя работы слота `nodes` после сшивания (lsb-0012, arch §3.5): объединённая
# заметка может лежать в ЛЮБОМ узле — обход `default` её не найдёт, поэтому
# реклассификация ставится заданием.
NODES_RECLASS_KIND = "reclass"

# Счётчик попыток пары и состояние «требует вмешательства» (№39, arch §6.8):
# нетранзиентные исходы замораживают пару — задание перестаёт браться
# (вызовы модели не жгутся), но остаётся `pending`. `unavailable`/`timeout`/
# `auth` счётчик НЕ замораживают: ими занимаются back-off и разрыв слота.
_NON_TRANSIENT_OUTCOMES = frozenset(
    {SlotOutcome.EMPTY, SlotOutcome.TRUNCATED, SlotOutcome.OFF_CONTRACT}
)

# Слот → имя джобы-владельца очереди (для событий внимания и /jobs/attention).
_SLOT_JOB_NAMES = {
    SLOT_EMBEDDING: EMBEDDING_JOB,
    SLOT_SUMMARY: SUMMARY_JOB,
    SLOT_JUDGE: JUDGE_JOB,
    NODES_JOB: NODES_JOB,
}


def _call_meta_fields(meta: ChatMeta) -> dict[str, object]:
    """Поля вызова для события отказа (§7.2): цифры/признаки, без секретов."""
    return {
        "operation": meta.operation,
        "model_type": meta.model_type,
        "budget": meta.budget,
        "budget_source": meta.budget_source,
        "prompt_tokens": meta.prompt_tokens,
        "completion_tokens": meta.completion_tokens,
        "reasoning_tokens": meta.reasoning_tokens,
        "had_thinking": meta.had_thinking,
        "finish_reason": meta.finish_reason,
        "duration_ms": meta.duration_ms,
    }


def _call_log_fields(meta: ChatMeta) -> dict[str, object]:
    """Полный набор полей успешного вызова `slot_call` (№49, §7.2)."""
    fields = _call_meta_fields(meta)
    fields.update(
        {
            "provider": meta.provider,
            "model": meta.model,
            "thinking_chars": meta.thinking_chars,
            "truncated": meta.truncated,
        }
    )
    return fields

# meta-ключ отпечатка конфигурации слотов (№39, arch §6.8/§13.3): смена
# конфигурации слота — часть «причины» внимания, поэтому старт со сменой
# отпечатка снимает `attention_at` у всех заданий. Состав отпечатка живёт в
# `slot_budget.slot_config_fingerprint` (lsbdef0008-02): модель, тип, бюджет,
# контекст, запас, потолок и параметры decision — одной реализацией на систему.
SLOT_CONFIG_META_KEY = "slot_config_fingerprint"

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

# Предикат «задание `reclass` берётся в ЭТОМ прогоне» (lsb-0012, arch §3.5):
# `_process_reclass_job` снимает задание, если заметки нет или она в корзине
# (решать нечего), и решает его по ГОТОВОЙ непустой суммари; задание, ждущее
# суммари, остаётся pending и прогрессом не считается. Снимок очереди считает
# только берущиеся задания: ждущее суммари — не работа прогона (сама заметка
# видна в очереди `summary`), иначе счётчик показывает работу, которой у джобы
# нет (FAIL приёмки 3.1.0, решение гейта 2026-09-19: `pending=1` держался 7
# минут — столько объединённая заметка ждала суммари).
_NODES_RECLASS_TAKEABLE = (
    "(n.id IS NULL OR n.deleted_at IS NOT NULL "
    "OR (n.summary_status = 'ok' AND trim(coalesce(n.summary, '')) <> ''))"
)


def _nodes_queue_stat_sql() -> str:
    """SQL снимка очереди `nodes` под ФАКТИЧЕСКИЕ пулы прогона (FR-2.2).

    `pending` — ровно та работа, которую `process_nodes` может взять в этом
    прогоне, и ничего больше:
    (1) задания `reclass` после сшивания (lsb-0012), которые решаются сейчас
    (`_NODES_RECLASS_TAKEABLE`);
    (2) кандидаты быстрого пула обхода `default` — предикат тот же, что у
    `_NODES_SWEEP_SELECT` (переезд по готовой разметке, без модели);
    (3) кандидаты классификаторного пула — предикат тот же, что у
    `_NODES_CLASSIFY_SELECT` (готовая непустая суммари, разметки ещё нет).
    `oldest_pending_sec` — возраст старейшего ИЗ ПОСЧИТАННЫХ источников
    (задания — `created_at`, кандидаты — `updated_at`), `null` — очередь
    пуста. Только SQL, без обращений к моделям; снимок работает и у
    выключенной джобы. Параметр `:threshold` — порог авто-переезда
    (`NAMESPACE_AUTO_MOVE_MIN_CONFIDENCE`).
    """
    reclass_jobs = (
        "SELECT COUNT(*) FROM worker_jobs j "
        "LEFT JOIN notes n ON n.id = j.note_id "
        "WHERE j.slot = 'nodes' AND j.kind = 'reclass' "
        f"AND j.status = 'pending' AND {_NODES_RECLASS_TAKEABLE}"
    )
    reclass_age = (
        "SELECT MAX(CAST(strftime('%s','now') AS INTEGER) "
        "- CAST(strftime('%s', j.created_at) AS INTEGER)) "
        "FROM worker_jobs j LEFT JOIN notes n ON n.id = j.note_id "
        "WHERE j.slot = 'nodes' AND j.kind = 'reclass' "
        f"AND j.status = 'pending' AND {_NODES_RECLASS_TAKEABLE}"
    )
    fast_pool = (
        "SELECT COUNT(*) FROM notes WHERE namespace = 'default' "
        "AND deleted_at IS NULL AND node_order_at IS NULL "
        "AND hint_path IS NOT NULL AND confidence >= :threshold"
    )
    fast_age = (
        "SELECT MAX(CAST(strftime('%s','now') AS INTEGER) "
        "- CAST(strftime('%s', updated_at) AS INTEGER)) FROM notes "
        "WHERE namespace = 'default' AND deleted_at IS NULL "
        "AND node_order_at IS NULL AND hint_path IS NOT NULL "
        "AND confidence >= :threshold"
    )
    classifier_pool = (
        "SELECT COUNT(*) FROM notes WHERE namespace = 'default' "
        "AND deleted_at IS NULL AND node_order_at IS NULL "
        "AND classified_at IS NULL AND summary_status = 'ok' "
        "AND summary IS NOT NULL AND summary <> ''"
    )
    classifier_age = (
        "SELECT MAX(CAST(strftime('%s','now') AS INTEGER) "
        "- CAST(strftime('%s', updated_at) AS INTEGER)) FROM notes "
        "WHERE namespace = 'default' AND deleted_at IS NULL "
        "AND node_order_at IS NULL AND classified_at IS NULL "
        "AND summary_status = 'ok' AND summary IS NOT NULL "
        "AND summary <> ''"
    )
    return (
        "SELECT "
        f"({reclass_jobs}) + ({fast_pool}) + ({classifier_pool}) AS pending, "
        "MAX("
        f"COALESCE(({reclass_age}), 0), "
        f"COALESCE(({fast_age}), 0), "
        f"COALESCE(({classifier_age}), 0)"
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
        terms: TermsService | None = None,
        decision: DecisionService | None = None,
    ) -> None:
        self._settings = settings
        self._embedding = embedding
        self._summarizer = summarizer
        # LLM-судья дедупа (Фаза 8, Этап 3.1, DI из build_services — один
        # экземпляр на процесс): вердикт «дубль/не дубль» по каждому
        # косинус-кандидату (judge-петля, Этап 3.2). None — тестовый режим:
        # воркер сводит по косинус-фоллбеку Этапа 2.2.
        self._judge = judge
        # LSBN-F-12: decision-слот (None — выключен, поведение v3.1.0). Разрыв
        # decision живёт ВНУТРИ сервиса (§13.8), воркер его не ведёт — берём
        # только снимок для `/health.slots.decision`.
        self._decision = decision
        # Фоновый дедуп (Фаза 8, Этап 2): поиск косинус-кандидатов против
        # ранних заметок. DI для тестов.
        self._dedup = (
            dedup if dedup is not None else DeduplicationService(settings)
        )
        # Сведение дублей (Этап 2.2) идёт штатной NOTE-логикой: update
        # раннего (ре-векторизация/ре-суммаризация своими очередями) и
        # soft delete позднего. Сервис собирается из тех же deps, что и
        # воркер (embedding — DI-фейк в юнит-тестах); save-пути здесь не
        # используются, notifier суммаризации не нужен — после слияния воркер
        # будит свою же суммаризационную петлю (notify_summary_pending).
        # Вектор-нотификатор этому сервису подключается ниже, после создания
        # событий: `merge_pair` возвращает раннюю заметку в очередь
        # векторизации, и петлю будит её же сигнал (notify_embedding_pending).
        self._notes = NoteService(settings, embedding=embedding)
        # Причёска (Фаза 10, Шаг 4): классификатор default-заметок после
        # суммаризации; None — тестовый режим без классификации.
        self._classifier = classifier
        # Реестр неймспейсов: известные узлы для классификатора и проверка
        # целевого узла авто-переезда.
        self._namespaces = NamespaceService(settings)
        # Область terms (3.0.0, lsb-0008; 3.2.0 — джоба `terms-normalize`):
        # тот же экземпляр, что у транспорта/петли areas (DI из build_services).
        # None — DI-сборка без области: сборщик джобы не регистрируется (§6.6).
        self._terms = terms
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
        # Разрывы слотов (№39, arch §6.2): состояние СЛОТА, а не джобы —
        # слот делят разные петли (summary: суммаризация/слияние/
        # классификация/описания и обход `nodes`; judge: dedup и структура),
        # поэтому разрыв живёт в одной точке на слот. `_slot_gate` проверяет
        # разрыв в начале прогона, `_log_slot_failure` фиксирует исход отказа.
        # Пауза — SLOT_PAUSE_SEC (потолок back-off), порог открытия —
        # SLOT_PAUSE_AFTER (auth открывает сразу). Состояние — в памяти
        # процесса: переживания рестарта не требует (конфигурация слота
        # переоценивается стартовой проверкой, arch §13.2).
        self._breakers: dict[str, SlotBreaker] = {
            name: SlotBreaker(
                name,
                settings.slot_pause_sec,
                open_after=settings.slot_pause_after,
            )
            for name in SLOT_NAMES
        }
        self._stopping = False
        # Сигнал «появилась заметка с pending-вектором» — будит embedding-петлю
        # сразу после записи (save/update/merge/title-доген), минуя выросший
        # back-off: свежая заметка кодируется немедленно, а не через 15 мин
        # (gate 2026-09-19; прецедент summary-петли). Ставит его NoteService
        # (DI-нотификатор из main.py) и сам воркер — на путях, где он пишет
        # заметку (слияние, догенерация названия).
        self._embedding_event = asyncio.Event()
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
        # Собственный сервис заметок воркера будит его же embedding-петлю на
        # путях, где пишет заметку сам воркер (`merge_pair` возвращает раннюю
        # заметку в очередь векторизации) — DI-нотификатор API-слоя здесь
        # недоступен (сервис собирается раньше событий).
        self._notes.set_vector_notifier(self.notify_embedding_pending)
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

    def notify_embedding_pending(self) -> None:
        """Разбудить embedding-петлю: заметка вернулась в очередь векторизации.

        Вызывается из save/update/merge (NoteService, поток `asyncio.to_thread`)
        и из самого воркера (слияние дублей, догенерация названия) —
        `asyncio.Event.set()` потокобезопасен. Петля немедленно выходит из
        ожидания и кодирует pending-заметки, не дожидаясь выросшего back-off
        (до 15 мин). Событие только ускоряет: задание живёт в самом статусе
        заметки (`vector_status='pending'`) и будет выбрано следующим прогоном.
        Перепроверки очереди у этой джобы нет (см. `build_embedding_job`): при
        недоступных моделях (`EmbeddingError`) непустая очередь не гоняет петлю
        вхолостую — она честно уходит в ожидание по back-off.
        """
        self._embedding_event.set()

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
        """Разбудить петлю `nodes`: работа появилась (lsb-0012, два повода).

        Зовётся из summary-петли в двух моментах, оба — синхронный код в
        `asyncio.to_thread`, поэтому `asyncio.Event.set()` потокобезопасен:

        * после успешного сшивания (`process_merge_pending`) — задание
          `reclass` объединённой заметки поставлено, но её суммари ещё
          пересчитывается: задание законно ждёт;
        * после фактической готовности выжимки (`process_summary_pending`,
          `summary_status='ok'`) — задание, ждавшее этой выжимки, стало
          берущимся; при отказе суммаризации сигнала нет.

        Петля немедленно выходит из ожидания и разбирает `reclass`, не
        дожидаясь часового интервала `JOB_NODES_INTERVAL_SEC`/выросшего
        back-off. Задание живёт строкой `worker_jobs` — событие только ускоряет
        (контракт «задание не теряется» не меняется). DI-нотификатор не нужен:
        джоба живёт внутри воркера (arch §3.5)."""
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
            ),
            # Старт со сменой конфигурации слота — «причина ушла»: внимание
            # снимается, отпечаток перезаписывается (№39 §6.8/§13.3). Стартовая
            # сверка идёт ПАРАЛЛЕЛЬНО петлям (не задерживает первый прогон).
            self._startup_attention_reset_async(),
        )

    # --- разрывы слотов (№39, arch §6.2/§6.7) ----------------------------------

    def slot_states(self) -> dict[str, dict]:
        """Снимок разрывов слотов (для журнала и хинта человеку; №39).

        Только чтение памяти процесса — без обращений к моделям. Потребитель —
        MCP-миддлварь хинта «предупреди своего человека». Полная поверхность
        `/health.slots` и наблюдаемость очередей — постановка no39-03.
        """
        return {name: breaker.snapshot() for name, breaker in self._breakers.items()}

    def slots_health(self) -> dict[str, dict]:
        """Снимок слотов для `/health.slots` (№39, arch §7.2): только SQL/память.

        По каждому слоту (`embedding`/`summary`/`judge`) — снимок разрыва
        (`SlotBreaker.snapshot`: `last_outcome`, `paused_until` и др.), исход
        последней попытки (`ok` — прецедент `embedding_ok`/`summarizer_ok`/
        `judge_ok`) и число заданий слота в внимании (`attention`, no39-02).
        №50 (lsbdef-0008, §7.3): плюс диагностика последнего вызова из
        `services.*.last_meta` — `last_finish_reason`/`last_budget`/
        `last_budget_source`/`last_had_thinking` (нет метаданных — None):
        оператор читает причину (обрезано по бюджету / размышление съело
        вывод) без журнала.
        Модели НЕ вызываются: только состояние в памяти и SQL по `worker_jobs`
        (сбор `/health` не должен будить слоты).
        """
        attention: dict[str, int] = {}
        try:
            self._ensure_job_table()
            with session(self._settings) as conn:
                rows = conn.execute(
                    "SELECT slot, COUNT(*) AS c FROM worker_jobs "
                    "WHERE attention_at IS NOT NULL GROUP BY slot"
                ).fetchall()
            attention = {rows[i]["slot"]: int(rows[i]["c"]) for i in range(len(rows))}
        except Exception:  # noqa: BLE001 — наблюдаемость не роняет /health
            attention = {}
        services = {
            SLOT_EMBEDDING: self._embedding,
            SLOT_SUMMARY: self._summarizer,
            SLOT_JUDGE: self._judge,
        }
        health: dict[str, dict] = {}
        for name in SLOT_NAMES:
            snapshot = dict(self._breakers[name].snapshot())
            service = services.get(name)
            snapshot["ok"] = getattr(service, "last_attempt_ok", None)
            snapshot["attention"] = attention.get(name, 0)
            meta = getattr(service, "last_meta", None)
            snapshot["last_finish_reason"] = meta.finish_reason if meta else None
            snapshot["last_budget"] = meta.budget if meta else None
            snapshot["last_budget_source"] = meta.budget_source if meta else None
            snapshot["last_had_thinking"] = meta.had_thinking if meta else None
            health[name] = snapshot
        health[SLOT_DECISION] = self._decision_slot_health(attention)
        return health

    def _decision_slot_health(self, attention: dict[str, int]) -> dict:
        """Слот `decision` для `/health.slots` (LSBN-F-12, §7.1): только память.

        Состояние берётся из `services.decision.snapshot()` (разрыв живёт в
        сервисе, §13.8): `ok`, `last_outcome`, `paused_until`; `attention` — по
        таблице заданий (своей очереди у decision нет — 0); четыре поля
        диагностики последнего вызова — `None` (у decision нет `last_meta`).
        Слот выключен/не собран — форма «выключен» (§7.1).
        """
        service = self._decision
        if service is None:
            return self._decision_disabled_stat()
        snapshot = service.snapshot()
        if not snapshot["enabled"]:
            return self._decision_disabled_stat()
        breaker = snapshot["breaker"]
        return {
            "ok": snapshot["ok"],
            "last_outcome": breaker["last_outcome"],
            "paused_until": breaker["paused_until"],
            "attention": attention.get(SLOT_DECISION, 0),
            "enabled": True,
            "last_finish_reason": None,
            "last_budget": None,
            "last_budget_source": None,
            "last_had_thinking": None,
        }

    @staticmethod
    def _decision_disabled_stat() -> dict:
        """Форма выключенного decision-слота в `/health.slots` (§7.1)."""
        return {
            "ok": None,
            "last_outcome": None,
            "paused_until": None,
            "attention": 0,
            "enabled": False,
        }

    def _probe_slot(self, slot: str) -> str:
        """Один дешёвый пробный вызов слота (без генерации, №39 §6.2).

        По имени слота — `probe()` соответствующего сервиса (тонкая обёртка
        над `LLMClient.check()`). Отсутствующий сервис или фейк без `probe`
        (тестовый режим) — заглушка `CHECK_OK` (прецедент стартовой проверки).
        """
        service = {
            SLOT_EMBEDDING: self._embedding,
            SLOT_SUMMARY: self._summarizer,
            SLOT_JUDGE: self._judge,
        }.get(slot)
        probe = getattr(service, "probe", None)
        if probe is None:
            return CHECK_OK
        return probe()

    def _slot_gate(self, slot: str | None, job: str | None = None) -> bool:
        """Ворота слота: можно ли брать партию (№39, arch §6.2/§6.7).

        `None`/неизвестный слот — True (разрыва нет). Разрыв закрыт — True.
        Разрыв открыт и стук не наступил — False (партия не берётся, вызовы
        модели не жгутся; прогон возвращает 0 → back-off петли). Стук наступил
        — один `probe()`: `CHECK_OK` снимает разрыв (партия идёт в этом же
        прогоне), `auth_failed`/иное — пауза переставляется, партии нет.
        """
        if slot is None:
            return True
        breaker = self._breakers.get(slot)
        if breaker is None:
            return True
        effective_job = job or slot
        now = time.time()
        if breaker.probe_due(now):
            result = self._probe_slot(slot)
            self._log_slot_event("slot_probe", slot, effective_job, result=result)
            if result == CHECK_OK:
                breaker.note_success()
                self._reset_attention_for_slot(slot, "probe")
                self._log_slot_event(
                    "slot_resumed",
                    slot,
                    effective_job,
                    after_sec=self._settings.slot_pause_sec,
                )
                return True
            outcome = (
                SlotOutcome.AUTH
                if result == CHECK_AUTH_FAILED
                else SlotOutcome.UNAVAILABLE
            )
            self._note_slot_outcome(effective_job, slot, outcome)
            return False
        if breaker.is_paused(now):
            return False
        return True

    def _note_slot_outcome(self, job: str, slot: str, outcome: SlotOutcome) -> bool:
        """Зафиксировать исход у разрыва слота; True — разрыв открылся сейчас."""
        breaker = self._breakers.get(slot)
        if breaker is None:
            return False
        opened = breaker.note_failure(outcome)
        if opened:
            self._log_slot_event(
                "slot_paused",
                slot,
                job,
                level=logging.WARNING,
                outcome=outcome.value
                if isinstance(outcome, SlotOutcome)
                else str(outcome),
                paused_until=breaker.snapshot()["paused_until"],
            )
        return opened

    def _log_slot_failure(
        self,
        job: str,
        slot: str,
        outcome: SlotOutcome,
        *,
        event: str,
        message: str = "slot call failed",
        note_id: int | None = None,
        older_id: int | None = None,
        candidate_id: int | None = None,
        detail: str | None = None,
        meta: ChatMeta | None = None,
        **extra: object,
    ) -> None:
        """Единая точка отказа слота (№39, arch §6.7): событие + исход в журнал.

        Поля события: `event`, `job`, `slot`, `outcome` (значение строки),
        `detail` (фрагмент ≤160 симв.; секреты маскируются 4 симв. + `***`),
        `note_id`/`older_id`/`candidate_id` по факту. №49: если вызов модели
        состоялся (отказ после ответа — `meta` у ошибки), событие дополняется
        полями вызова (§7.2) — второй записи нет. Отказ фиксируется и у
        разрыва слота: открылся — пишется `slot_paused`. Вызывающий сохраняет
        прежнее поведение (работа остаётся pending, `continue`).
        """
        fields: dict[str, object] = {
            "event": event,
            "job": job,
            "slot": slot,
            "outcome": outcome.value
            if isinstance(outcome, SlotOutcome)
            else str(outcome),
        }
        for key, value in (
            ("note_id", note_id),
            ("older_id", older_id),
            ("candidate_id", candidate_id),
            ("detail", self._mask_detail(detail)),
        ):
            if value is not None:
                fields[key] = value
        if meta is not None:
            fields.update(_call_meta_fields(meta))
        fields.update(extra)
        logging.getLogger("app").warning(message, extra=fields)
        self._note_slot_outcome(job, slot, outcome)

    def _log_slot_event(
        self,
        event: str,
        slot: str,
        job: str,
        *,
        level: int = logging.INFO,
        **fields: object,
    ) -> None:
        """Событие разрыва/стука (№39 §7.4): `event`/`slot`/`job` + поля по факту."""
        payload: dict[str, object] = {"event": event, "slot": slot, "job": job}
        for key, value in fields.items():
            if value is not None:
                payload[key] = value
        logging.getLogger("app").log(level, event, extra=payload)

    def _log_slot_call(
        self,
        job: str,
        slot: str,
        meta: ChatMeta | None,
        *,
        note_id: int | None = None,
        older_id: int | None = None,
        candidate_id: int | None = None,
    ) -> None:
        """Успешный вызов чата слота — событие `slot_call` (№49, §7.2).

        Одно событие на вызов: бюджет и его источник, счётчики, факт и объём
        размышления, причина завершения, обрезка, длительность. Без метаданных
        (тестовый двойник/слот без `chat_result`) событие не пишется — живой
        сервис их всегда отдаёт (№49 FR-3.1).
        """
        if meta is None:
            return
        fields = _call_log_fields(meta)
        for key, value in (
            ("note_id", note_id),
            ("older_id", older_id),
            ("candidate_id", candidate_id),
        ):
            if value is not None:
                fields[key] = value
        self._log_slot_event("slot_call", slot, job, level=logging.INFO, **fields)

    def _mask_detail(self, detail: str | None) -> str | None:
        """Фрагмент причины ≤160 симв. с маскировкой секретов (4 симв. + ***).

        Секреты (API-ключи слотов, MCP-токен) могут попасть в текст отказа
        (URL/эхо) — в журнал они не уходят ни в каком виде.
        """
        if detail is None:
            return None
        fragment = " ".join(str(detail).split())[:160]
        secrets = (
            self._settings.embedding_api_key,
            self._settings.summary_api_key,
            self._settings.judge_api_key,
            self._settings.mcp_auth_token,
        )
        for secret in secrets:
            if secret and secret.strip() and secret.strip() in fragment:
                fragment = fragment.replace(secret.strip(), secret.strip()[:4] + "***")
        return fragment

    # --- счётчик попыток пары и «требует вмешательства» (№39, arch §6.8) --------

    @staticmethod
    def _load_payload(raw: str | None) -> dict:
        """Разобрать payload работы (терпимо к пустому/битому JSON)."""
        try:
            loaded = json.loads(raw or "{}")
        except (TypeError, ValueError):
            loaded = {}
        return loaded if isinstance(loaded, dict) else {}

    def _set_job_payload(self, job_id: int, payload: dict) -> None:
        """Перезаписать payload работы (идентичность пары и её штамп)."""
        self._ensure_job_table()
        with session(self._settings) as conn, transaction(conn):
            conn.execute(
                "UPDATE worker_jobs SET payload = ? WHERE id = ?",
                (json.dumps(payload, ensure_ascii=False), job_id),
            )

    def _job_for_slot(self, slot: str) -> str:
        """Имя джобы-владельца очереди слота (для событий и /jobs/attention)."""
        return _SLOT_JOB_NAMES.get(slot, slot)

    def _log_attention_event(self, event: str, **fields: object) -> None:
        """Событие внимания задания (arch §7.4): `event` + поля по факту."""
        payload: dict[str, object] = {"event": event}
        for key, value in fields.items():
            if value is not None:
                payload[key] = value
        logging.getLogger("app").info(event, extra=payload)

    def _note_attempt(
        self,
        job_id: int,
        outcome: SlotOutcome,
        *,
        detail: str | None = None,
        pair_stamp: str | None = None,
    ) -> dict:
        """Зафиксировать исход попытки пары (№39, arch §6.8).

        Транзиентный исход (`unavailable`/`timeout`/`auth`) обновляет только
        `last_outcome`/`last_error` — счётчик не растёт (ждать — нормальное
        состояние, им занимаются back-off и разрыв). Нетранзиентный
        (`empty`/`truncated`/`off_contract`) наращивает `attempts`, и по
        достижении `SLOT_MAX_ATTEMPTS` ставит `attention_at` — работа замирает,
        оставаясь `pending`. Всё — одной транзакцией (`UPDATE … WHERE id = ?`).
        `pair_stamp` — отпечаток пары (заметка/кандидат), обновляется в payload.
        """
        self._ensure_job_table()
        masked = self._mask_detail(detail)
        value = outcome.value if isinstance(outcome, SlotOutcome) else str(outcome)
        nontransient = outcome in _NON_TRANSIENT_OUTCOMES
        with session(self._settings) as conn, transaction(conn):
            row = conn.execute(
                "SELECT id, slot, note_id, payload, attempts, attention_at "
                "FROM worker_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return {"job_id": job_id, "attempts": 0, "attention": False}
            attempts = int(row["attempts"] or 0)
            attention = row["attention_at"] is not None
            pair = self._load_payload(row["payload"])
            if pair_stamp is not None:
                pair["updated_at"] = pair_stamp
                conn.execute(
                    "UPDATE worker_jobs SET payload = ? WHERE id = ?",
                    (json.dumps(pair, ensure_ascii=False), job_id),
                )
            if nontransient:
                attempts += 1
                if attempts >= self._settings.slot_max_attempts:
                    attention = True
                    conn.execute(
                        "UPDATE worker_jobs SET attempts = ?, last_outcome = ?, "
                        "last_error = ?, attention_at = COALESCE(attention_at, "
                        "strftime('%Y-%m-%dT%H:%M:%SZ','now')), updated_at = "
                        "strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ?",
                        (attempts, value, masked, job_id),
                    )
                else:
                    conn.execute(
                        "UPDATE worker_jobs SET attempts = ?, last_outcome = ?, "
                        "last_error = ?, updated_at = "
                        "strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ?",
                        (attempts, value, masked, job_id),
                    )
            else:
                conn.execute(
                    "UPDATE worker_jobs SET last_outcome = ?, last_error = ?, "
                    "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ?",
                    (value, masked, job_id),
                )
            slot = row["slot"]
            note_id = row["note_id"]
            older_id = pair.get("older_id")
        if nontransient and attention:
            self._log_attention_event(
                "job_attention",
                job=self._job_for_slot(slot),
                slot=slot,
                note_id=note_id,
                older_id=older_id,
                outcome=value,
                attempts=attempts,
            )
        return {"job_id": job_id, "attempts": attempts, "attention": attention}

    def _release_attention(self, job_id: int, reason: str) -> dict:
        """Сбросить счётчик пары и снять внимание (№39, arch §6.8/§13.3).

        `attempts = 0, attention_at = NULL, last_error = NULL` — «причина
        ушла»: работа идёт как новая. `released` — было ли задание во внимании
        (операторская ручка отдаёт 404 на задание вне внимания); `found` —
        есть ли строка вообще. Событие `job_attention_released` с причиной
        (`edit`/`probe`/`operator`/`startup`) пишется на найденную строку.
        """
        self._ensure_job_table()
        with session(self._settings) as conn, transaction(conn):
            row = conn.execute(
                "SELECT id, slot, note_id, payload, attention_at "
                "FROM worker_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return {"job_id": job_id, "released": False, "found": False}
            was_attention = row["attention_at"] is not None
            conn.execute(
                "UPDATE worker_jobs SET attempts = 0, attention_at = NULL, "
                "last_error = NULL, updated_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ?",
                (job_id,),
            )
            slot = row["slot"]
            note_id = row["note_id"]
            older_id = self._load_payload(row["payload"]).get("older_id")
        self._log_attention_event(
            "job_attention_released",
            job=self._job_for_slot(slot),
            slot=slot,
            note_id=note_id,
            older_id=older_id,
            reason=reason,
        )
        return {
            "job_id": job_id,
            "released": was_attention,
            "found": True,
            "attempts": 0,
        }

    def release_attention(self, job_id: int) -> dict:
        """Операторская переоткрытие задания из внимания (arch §6.7/§7.4).

        Только для задания во внимании: вне внимания (или без строки) —
        `released: false` без правок (REST-ручка отдаёт 404).
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT attention_at FROM worker_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None or row["attention_at"] is None:
            return {"job_id": job_id, "released": False}
        self._release_attention(job_id, "operator")
        return {"job_id": job_id, "released": True, "attempts": 0}

    def _reset_attention_for_slot(self, slot: str, reason: str) -> int:
        """Снять внимание у заданий слота (`*` — у всех); число снятых.

        Крючки «причина ушла» (arch §6.8/§13.3): успешный стук слота
        (`probe`), старт со сменой отпечатка конфигурации (`startup`).
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            if slot == "*":
                rows = conn.execute(
                    "SELECT id FROM worker_jobs WHERE attention_at IS NOT NULL"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id FROM worker_jobs WHERE attention_at IS NOT NULL "
                    "AND slot = ?",
                    (slot,),
                ).fetchall()
        released = 0
        for row in rows:
            if self._release_attention(int(row["id"]), reason)["released"]:
                released += 1
        return released

    def attention_jobs(self) -> list[dict]:
        """Задания, ждущие человека (arch §6.7/§7.4): только SQL.

        Форма записи: `{job_id, job, slot, note_id, older_id, outcome,
        attempts, since}` — `since` — момент `attention_at`. Потребители —
        `/health` и операторская ручка `GET /jobs/attention` (моделей не зовём).
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, slot, note_id, payload, attempts, last_outcome, "
                "attention_at FROM worker_jobs WHERE attention_at IS NOT NULL "
                "ORDER BY id"
            ).fetchall()
        jobs: list[dict] = []
        for row in rows:
            jobs.append(
                {
                    "job_id": int(row["id"]),
                    "job": self._job_for_slot(row["slot"]),
                    "slot": row["slot"],
                    "note_id": row["note_id"],
                    "older_id": self._load_payload(row["payload"]).get("older_id"),
                    "outcome": row["last_outcome"],
                    "attempts": int(row["attempts"] or 0),
                    "since": row["attention_at"],
                }
            )
        return jobs

    def attention_count(self) -> int:
        """Сколько заданий ждёт человека (агрегат по всем слотам, arch §6.7)."""
        self._ensure_job_table()
        with session(self._settings) as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM worker_jobs "
                    "WHERE attention_at IS NOT NULL"
                ).fetchone()[0]
            )

    def _review_attention_jobs(self, slot: str, kind: str) -> int:
        """Пере-проверить пару замороженных работ без вызова модели (arch §6.8).

        Протокол «правка пары» без ручных правок: правка заметки/кандидата или
        смена лучшего кандидата возвращает работу в очередь (`attempts=0`),
        решение о вердикте/слиянии принимает следующий прогон. Только SQL и
        чтение вектора — моделей не зовём. Возвращает число размороженных.
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, note_id, payload FROM worker_jobs "
                "WHERE slot = ? AND kind = ? AND status = 'pending' "
                "AND attention_at IS NOT NULL ORDER BY id",
                (slot, kind),
            ).fetchall()
        released = 0
        for row in rows:
            payload = self._load_payload(row["payload"])
            note_id = int(row["note_id"])
            with session(self._settings) as conn:
                note = conn.execute(
                    "SELECT id, namespace, vector_status, deleted_at, updated_at "
                    "FROM notes WHERE id = ?",
                    (note_id,),
                ).fetchone()
                vector = (
                    vectors.get_vector(conn, note_id) if note is not None else None
                )
            if note is None or note["deleted_at"] is not None:
                continue
            # правка стороны пары (заметки/ранней) — причина ушла
            if (
                payload.get("note_updated_at") is not None
                and payload["note_updated_at"] != note["updated_at"]
            ) or (
                payload.get("newer_updated_at") is not None
                and payload["newer_updated_at"] != note["updated_at"]
            ):
                if self._release_attention(int(row["id"]), "edit")["released"]:
                    released += 1
                continue
            older_id = payload.get("older_id")
            if older_id is not None:
                with session(self._settings) as conn:
                    other = conn.execute(
                        "SELECT deleted_at, updated_at FROM notes WHERE id = ?",
                        (int(older_id),),
                    ).fetchone()
                if other is None or other["deleted_at"] is not None:
                    continue
                if (
                    payload.get("older_updated_at") is not None
                    and payload["older_updated_at"] != other["updated_at"]
                ):
                    if self._release_attention(int(row["id"]), "edit")["released"]:
                        released += 1
                continue
            # дедуп: лучший кандидат пересчитывается по вектору; без вектора
            # пару не проверить — ждём перевекторизации
            if note["vector_status"] != "ok" or vector is None:
                continue
            candidates = self._find_dedup_candidates(
                note_id, vector, note["namespace"]
            )
            alive = [pair for pair in candidates if pair[0] < note_id]
            best_id = int(alive[0][0]) if alive else None
            stored_id = payload.get("candidate_id")
            if (
                stored_id is not None
                and best_id is not None
                and stored_id != best_id
            ):
                if self._release_attention(int(row["id"]), "edit")["released"]:
                    released += 1
                continue
            if best_id is not None:
                with session(self._settings) as conn:
                    candidate = conn.execute(
                        "SELECT updated_at FROM notes WHERE id = ?", (best_id,)
                    ).fetchone()
                if (
                    candidate is not None
                    and payload.get("candidate_updated_at") is not None
                    and payload["candidate_updated_at"] != candidate["updated_at"]
                ):
                    if self._release_attention(int(row["id"]), "edit")["released"]:
                        released += 1
        return released

    def _startup_attention_reset(self) -> int:
        """Стартовая сверка отпечатка конфигурации слотов (arch §6.8/§13.3).

        Сравниваем текущий отпечаток (`slot_budget.slot_config_fingerprint`,
        lsbdef0008-02: модель/тип/бюджет/контекст/запас/потолок + параметры
        decision) с записанным в `meta` (`SLOT_CONFIG_META_KEY`). Отпечатка нет
        или он отличается — «причина ушла»: внимание снимается у заданий всех
        слотов (событие блока 2 `job_attention_released`, `reason="startup"`),
        отпечаток перезаписывается. Тот же отпечаток — ничего не снимаем.

        `SLOT_ATTENTION_RESET_ON_START=false` отключает шаг целиком: оператор
        запретил снятие, читать и переписывать отпечаток не нужно.
        """
        if not self._settings.slot_attention_reset_on_start:
            return 0
        fingerprint = slot_config_fingerprint(self._settings)
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = ?",
                (SLOT_CONFIG_META_KEY,),
            ).fetchone()
        stored = row["value"] if row is not None else None
        released = 0
        if stored != fingerprint:
            released = self._reset_attention_for_slot("*", "startup")
        with session(self._settings) as conn, transaction(conn):
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (SLOT_CONFIG_META_KEY, fingerprint),
            )
        return released

    async def _startup_attention_reset_async(self) -> None:
        """Стартовая сверка внимания в рабочем потоке (не блокирует петли).

        Отказ сверки не роняет воркер: внимание само снимется успешным стуком
        слота или правкой пары, а сама сверка — оптимизация «причина ушла при
        смене конфигурации» (arch §6.8).
        """
        try:
            await asyncio.to_thread(self._startup_attention_reset)
        except Exception:  # noqa: BLE001 — фоновая оптимизация не роняет старт
            logging.getLogger("app").warning(
                "slot attention startup reset failed",
                extra={"event": "job_attention_reset_failed"},
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
        if not self._slot_gate(SLOT_EMBEDDING, EMBEDDING_JOB):
            return 0  # разрыв слота: партия не берётся (№39, arch §6.2)
        with history.bind_origin(f"job:{EMBEDDING_JOB}"):
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
        if not self._slot_gate(SLOT_SUMMARY, SUMMARY_JOB):
            return 0  # разрыв слота summary: партия не берётся (№39, §6.2)
        with history.bind_origin(f"job:{SUMMARY_JOB}"):
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
        if not self._slot_gate(SLOT_JUDGE, JUDGE_JOB):
            return 0  # разрыв слота judge: партия не берётся (№39, §6.2)
        with history.bind_origin(f"job:{JUDGE_JOB}"):
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
        with history.bind_origin(f"job:{AREAS_JOB}"):
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
        with history.bind_origin(f"job:{EXPIRATION_JOB}"):
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
        Колонки внимания (№39, arch §5.2/§6.8) — `attempts`/`last_outcome`/
        `last_error`/`attention_at` — в свежем DDL и идемпотентным ALTER'ом по
        `PRAGMA table_info(worker_jobs)` для живой базы предыдущей версии;
        индекс `idx_worker_jobs_attention` обслуживает выборку «требует
        вмешательства».
        """
        if self._jobs_table_ready:
            return
        with session(self._settings) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS worker_jobs ("
                "  id           INTEGER PRIMARY KEY,"
                "  slot         TEXT NOT NULL,"
                "  kind         TEXT NOT NULL,"
                "  note_id      INTEGER NOT NULL,"
                "  payload      TEXT,"
                "  status       TEXT NOT NULL DEFAULT 'pending',"
                "  created_at   TEXT NOT NULL DEFAULT "
                "    (strftime('%Y-%m-%dT%H:%M:%SZ','now')),"
                "  updated_at   TEXT NOT NULL DEFAULT "
                "    (strftime('%Y-%m-%dT%H:%M:%SZ','now')),"
                "  attempts     INTEGER NOT NULL DEFAULT 0,"
                "  last_outcome TEXT,"
                "  last_error   TEXT,"
                "  attention_at TEXT"
                ")"
            )
            existing = {
                row["name"]
                for row in conn.execute(
                    "PRAGMA table_info(worker_jobs)"
                ).fetchall()
            }
            for name, ddl in (
                ("attempts", "INTEGER NOT NULL DEFAULT 0"),
                ("last_outcome", "TEXT"),
                ("last_error", "TEXT"),
                ("attention_at", "TEXT"),
            ):
                if name not in existing:
                    conn.execute(
                        f"ALTER TABLE worker_jobs ADD COLUMN {name} {ddl}"
                    )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_worker_jobs_queue "
                "ON worker_jobs(slot, kind, status, id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_worker_jobs_attention "
                "ON worker_jobs(status, attention_at)"
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
        """Вычитать pending-работы слота (порядок по id).

        Замороженные работы (`attention_at IS NOT NULL`, №39 §6.8) не берутся:
        повторяющийся нетранзиентный отказ перестаёт жечь вызовы модели, пока
        причина не ушла (работа при этом остаётся `pending`).
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            return conn.execute(
                "SELECT id, note_id, payload FROM worker_jobs "
                "WHERE slot = ? AND kind = ? AND status = 'pending' "
                "AND attention_at IS NULL "
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

        №39 (arch §6.9/§13.9): считаем только БЕРУЩЕЕСЯ сейчас — merge-работы
        без внимания (`attention_at IS NULL`). Замороженное задание
        (`attention_at`, no39-02) петлю брать не может, а если бы очередь
        считала его непустой — петля крутилась бы вхолостую (busy-loop).
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT (SELECT COUNT(*) FROM notes WHERE deleted_at IS NULL "
                "  AND (title IS NULL OR summary_status = 'pending')) "
                "+ (SELECT COUNT(*) FROM worker_jobs "
                "  WHERE slot = 'summary' AND status = 'pending' "
                "  AND attention_at IS NULL) AS c"
            ).fetchone()
        # №39 (arch §6.9): при открытом разрыве слота петля партию не берёт —
        # очередь для неё пуста (иначе отложенное задание гонит busy-loop).
        return int(row["c"]) == 0 or self._breakers[SLOT_SUMMARY].is_paused()

    def _judge_queue_empty(self) -> bool:
        """Пуста ли judge-очередь (дешёвый SELECT, пул 6, lost wakeup).

        Считает pending dedup-работы judge-слота — ту же очередь, что
        выгребает process_judge_pending в этой петле. №39 (arch §6.9/§13.9):
        замороженное задание (`attention_at`, no39-02) не считается — петля
        его не берёт, и очередь для неё пуста (busy-loop'а нет).
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM worker_jobs "
                "WHERE slot = 'judge' AND kind = 'dedup' AND status = 'pending' "
                "AND attention_at IS NULL"
            ).fetchone()
        # №39 (arch §6.9): при открытом разрыве судьи партия не берётся —
        # очередь для петли пуста (busy-loop'а нет).
        return int(row[0]) == 0 or self._breakers[SLOT_JUDGE].is_paused()

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

    def _gated_queue_stat(self, slot: str, stat: dict) -> dict:
        """Снимок очереди с учётом разрыва слота (№39, arch §6.9).

        Разрыв слота открыт (`SlotBreaker.is_paused`) — петля партию не берёт,
        поэтому вся текущая работа из `pending` уезжает в `waiting`
        (`pending=0`), а `oldest_pending_sec`/`attention` сохраняются: ожидание
        видно в `/health.queues`, но петля не крутится вхолостую. Иначе —
        снимок нормализуется как есть (возраст и внимание не пересчитываются).
        """
        pending = int(stat.get("pending", 0) or 0)
        waiting = int(stat.get("waiting", 0) or 0)
        attention = int(stat.get("attention", 0) or 0)
        breaker = self._breakers.get(slot)
        if breaker is not None and breaker.is_paused():
            return queue_snapshot(
                0,
                stat.get("oldest_pending_sec"),
                waiting=waiting + pending,
                attention=attention,
            )
        return queue_snapshot(
            pending,
            stat.get("oldest_pending_sec"),
            waiting=waiting,
            attention=attention,
        )

    def _vector_queue_stat(self) -> dict:
        """Снимок очереди векторизации заметок: pending и возраст старейшего.

        Источник — `notes.vector_status='pending'` (тот же предикат, что у
        легаси-счётчика `pending_vector`); SQL живёт рядом с `health_counts`
        в NoteService. №39 (arch §6.9): при открытом разрыве embedding очередь
        ждёт слот — `_gated_queue_stat` (работа едет в `waiting`).
        """
        return self._gated_queue_stat(
            SLOT_EMBEDDING, self._notes.vector_queue_stat()
        )

    def _summary_queue_stat(self) -> dict:
        """Снимок очереди суммаризации (№39, arch §6.9).

        `pending` — работа, которую петля может взять СЕЙЧАС: заметки
        (title-доген + pending суммаризации) плюс pending merge-работы слота
        summary БЕЗ внимания. `attention` — pending merge-работы с вниманием
        (no39-02, ждут человека), `waiting = attention`: у слота summary
        отложено именно замороженное. `oldest_pending_sec` — максимум возраста
        по обоим источникам; при открытом разрыве слота — `_gated_queue_stat`.
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT "
                "(SELECT COUNT(*) FROM notes WHERE deleted_at IS NULL "
                "  AND (title IS NULL OR summary_status = 'pending')) "
                "+ (SELECT COUNT(*) FROM worker_jobs WHERE slot = 'summary' "
                "  AND status = 'pending' AND attention_at IS NULL) AS pending, "
                "(SELECT COUNT(*) FROM worker_jobs WHERE slot = 'summary' "
                "  AND status = 'pending' AND attention_at IS NOT NULL) "
                "AS attention, "
                "MAX("
                "COALESCE((SELECT MAX(CAST(strftime('%s','now') AS INTEGER) - "
                "  CAST(strftime('%s', updated_at) AS INTEGER)) FROM notes "
                "  WHERE deleted_at IS NULL AND (title IS NULL OR "
                "  summary_status = 'pending')), 0), "
                "COALESCE((SELECT MAX(CAST(strftime('%s','now') AS INTEGER) - "
                "  CAST(strftime('%s', created_at) AS INTEGER)) FROM worker_jobs "
                "  WHERE slot = 'summary' AND status = 'pending'), 0)"
                ") AS oldest_pending_sec"
            ).fetchone()
        attention = int(row["attention"])
        stat = queue_snapshot(
            row["pending"],
            row["oldest_pending_sec"],
            waiting=attention,
            attention=attention,
        )
        return self._gated_queue_stat(SLOT_SUMMARY, stat)

    def _judge_queue_stat(self) -> dict:
        """Снимок очереди judge (№39, arch §6.9).

        `pending` — dedup-работы слота judge БЕЗ внимания (берущееся сейчас,
        драйвер `_judge_queue_empty`); `attention` — с вниманием (no39-02),
        `waiting = attention`: у судьи отложено именно замороженное. Возраст —
        `now - MIN(created_at)` по берущемуся. При открытом разрыве judge —
        `_gated_queue_stat`. Таблица создаётся лениво (`_ensure_job_table`),
        моделей не зовём.
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT "
                "COALESCE(SUM(status = 'pending' AND attention_at IS NULL), 0) "
                "AS pending, "
                "COALESCE(SUM(status = 'pending' AND attention_at IS NOT NULL), 0) "
                "AS attention, "
                "MAX(CASE WHEN status = 'pending' AND attention_at IS NULL "
                "  THEN CAST(strftime('%s','now') AS INTEGER) - "
                "  CAST(strftime('%s', created_at) AS INTEGER) END) "
                "AS oldest_pending_sec "
                "FROM worker_jobs WHERE slot = 'judge'"
            ).fetchone()
        attention = int(row["attention"])
        stat = queue_snapshot(
            row["pending"],
            row["oldest_pending_sec"],
            waiting=attention,
            attention=attention,
        )
        return self._gated_queue_stat(SLOT_JUDGE, stat)

    def _areas_queue_stat(self) -> dict:
        """Снимок очереди областей: сумма pending по ВКЛЮЧЁННЫМ областям (FR-2.5).

        Источник pending — тот же, что у `_areas_queue_empty`: pending-записи
        активных строк включённых областей (`enabled_areas`, №40) — выключенная
        область фоном не забирается и в общий счётчик не входит. Суммы
        складываются, возраст старейшего — максимум `now - updated_at` по
        областям; пусто — `null`. Накопление выключенной области видно
        отдельно в `/health.areas` (`areas_health`).
        """
        pending = 0
        oldest: int | None = None
        with session(self._settings) as conn:
            for spec in enabled_areas(self._settings):
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
        stat = queue_snapshot(pending, oldest)
        # №39 (arch §6.9): петля `areas` кодирует тем же слотом embedding —
        # разрыв слота означает, что очередь ждёт его.
        return self._gated_queue_stat(SLOT_EMBEDDING, stat)

    def areas_health(self) -> dict[str, dict]:
        """`/health.areas` (FR-4.3, №40): по ВСЕМ областям `{enabled, pending}`.

        Отдаётся по `ALL_AREAS`, а не по `enabled_areas`: накопление у
        выключенной области должно быть видно (прецедент `JOB_*_ENABLED`:
        «очередь остаётся видна в /health»). `pending` — тот же SQL, что у
        `_areas_queue_stat`, но БЕЗ суммирования (по области); обращений к
        моделям нет — только чтение pending-состояния.
        """
        enabled = {spec.name for spec in enabled_areas(self._settings)}
        health: dict[str, dict] = {}
        with session(self._settings) as conn:
            for spec in ALL_AREAS:
                row = conn.execute(
                    f"SELECT COUNT(*) FROM {spec.table} "
                    "WHERE vector_status = 'pending' "
                    "AND deleted_at IS NULL"
                ).fetchone()
                health[spec.name] = {
                    "enabled": spec.name in enabled,
                    "pending": int(row[0]),
                }
        return health

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
        for spec in enabled_areas(self._settings):
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
        except EmbeddingError as exc:
            # №39: отказ слота embedding несёт тип исхода (записи — pending).
            self._log_slot_failure(
                AREAS_JOB,
                SLOT_EMBEDDING,
                outcome_of(exc),
                event="area_embed_failed",
                message="area vectorization failed — records stay pending",
                detail=str(exc),
                area=spec.name,
                count=len(rows),
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

        pending-записи считаются по включённым областям реестра (`enabled_areas`,
        №40) — та же очередь, что выгребает process_pending_areas.
        """
        pending = 0
        with session(self._settings) as conn:
            for spec in enabled_areas(self._settings):
                pending += int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM {spec.table} "
                        "WHERE vector_status = 'pending' "
                        "AND deleted_at IS NULL"
                    ).fetchone()[0]
                )
        # №39 (arch §6.9): области кодируются слотом embedding — при его
        # разрыве партия не берётся, очередь для петли пуста (busy-loop'а нет).
        return pending == 0 or self._breakers[SLOT_EMBEDDING].is_paused()

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
        Заметка с таймером (TTL, no46-03) — тоже снимается, но ДО судьи и без
        вызова моделей (событие `dedup_skipped`, reason="note has a lifetime").
        """
        # Протокол «правка пары»: замороженные работы перепроверяются без
        # вызова модели — ушедшая причина размораживает их (арх §6.8).
        self._review_attention_jobs("judge", "dedup")
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
                    "SELECT id, text, namespace, vector_status, deleted_at, "
                    "expires_at "
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
            if row["expires_at"] is not None:
                # no46-03 (3.2.0): заметка с таймером (TTL, `expires_at IS NOT
                # NULL`) НИКОГДА не сливается — ранний выход ДО судьи и до
                # поиска кандидатов: модели не дёргаем, работа снимается;
                # наблюдаемое событие dedup_skipped с причиной. Правило О.
                # 2026-10-10; связи у такой заметки разрешены (links не тут).
                logging.getLogger("app").info(
                    "dedup: skipped — note has a lifetime",
                    extra={
                        "event": "dedup_skipped",
                        "job": JUDGE_JOB,
                        "note_id": note_id,
                        "reason": "note has a lifetime",
                    },
                )
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
                        f"SELECT id, text, deleted_at, updated_at FROM notes "
                        f"WHERE id IN ({placeholders})",
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
                # Идентичность пары (№39 §6.8/§13.3): лучший кандидат и штампы
                # сторон («заметка», «кандидат»). Новый лучший кандидат или
                # правка любой стороны — ДРУГАЯ пара: счётчик и внимание
                # сбрасываются, работа идёт как новая; штампы сохраняются на
                # следующую сверку (в т.ч. у замороженной работы,
                # `_review_attention_jobs`).
                best_id = int(alive[0][0]) if alive else None
                stored = self._load_payload(job["payload"])
                if best_id is not None:
                    candidate_updated = by_id[best_id]["updated_at"]
                    pair_changed = (
                        stored.get("candidate_id") is not None
                        and stored["candidate_id"] != best_id
                    ) or (
                        stored.get("note_updated_at") is not None
                        and stored["note_updated_at"] != newer["updated_at"]
                    ) or (
                        stored.get("candidate_updated_at") is not None
                        and stored["candidate_updated_at"] != candidate_updated
                    )
                    if pair_changed:
                        self._release_attention(job["id"], "edit")
                    stored["candidate_id"] = best_id
                    stored["note_updated_at"] = newer["updated_at"]
                    stored["candidate_updated_at"] = candidate_updated
                    self._set_job_payload(job["id"], stored)
                try:
                    best = self._pick_duplicate(note_id, alive, by_id)
                except JudgeError as exc:
                    # №39 (§6.7): отказ слота несёт тип исхода и фрагмент причины.
                    self._log_slot_failure(
                        JUDGE_JOB,
                        SLOT_JUDGE,
                        outcome_of(exc),
                        event="dedup_judge_failed",
                        message="dedup: judge undecidable — both notes kept, retry queued",
                        note_id=note_id,
                        detail=str(exc),
                        candidates=[
                            candidate_id for candidate_id, _ in alive
                        ],
                        meta=getattr(exc, "meta", None),
                    )
                    self._note_attempt(
                        job["id"],
                        outcome_of(exc),
                        detail=str(exc),
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

        no46-03 (3.2.0), второй барьер: если у любой заметки пары есть таймер
        (`expires_at IS NOT NULL`) — слияния нет, работа снимается, событие
        `dedup_skipped` (reason="note has a lifetime"). Это защита в глубине:
        заметка с TTL не сливается и снаружи (нормальный путь сюда не доходит —
        judge-петля такую заметку не обрабатывает).

        Отказ слияния (SummaryError, NFR-3) данные не портит: обе заметки
        остаются, работа остаётся pending — повтор по back-off summary-петли.
        """
        if self._summarizer is None:
            return 0  # тестовый режим без суммаризатора: слияние невозможно
        self._review_attention_jobs("summary", "merge")
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
                    "SELECT id, text, deleted_at, updated_at, expires_at "
                    "FROM notes "
                    "WHERE id IN (?, ?)",
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
            if older["expires_at"] is not None or newer["expires_at"] is not None:
                # no46-03 (3.2.0), второй барьер (защита в глубине): если у
                # ЛЮБОЙ заметки пары есть таймер — слияния нет (судья/суммаризатор
                # не вызываются), работа снимается, событие того же вида.
                logging.getLogger("app").info(
                    "dedup: merge skipped — a note has a lifetime",
                    extra={
                        "event": "dedup_skipped",
                        "job": SUMMARY_JOB,
                        "older_id": older_id,
                        "note_id": note_id,
                        "reason": "note has a lifetime",
                    },
                )
                self._mark_job_done(job["id"])
                done += 1
                continue
            # Пара merge-работы — заметка (`older_id`); штампы сторон для
            # сверки правки пары (№39 §6.8/§13.3): изменились — счётчик и
            # внимание сбрасываются, работа идёт как новая.
            pair_changed = (
                payload.get("older_updated_at") is not None
                and payload["older_updated_at"] != older["updated_at"]
            ) or (
                payload.get("newer_updated_at") is not None
                and payload["newer_updated_at"] != newer["updated_at"]
            )
            if pair_changed:
                self._release_attention(job["id"], "edit")
            if (
                payload.get("older_updated_at") != older["updated_at"]
                or payload.get("newer_updated_at") != newer["updated_at"]
            ):
                payload["older_updated_at"] = older["updated_at"]
                payload["newer_updated_at"] = newer["updated_at"]
                self._set_job_payload(job["id"], payload)
            try:
                merge_result = self._summarizer.merge(older["text"], newer["text"])
                self._log_slot_call(
                    SUMMARY_JOB,
                    SLOT_SUMMARY,
                    merge_result.meta,
                    older_id=older_id,
                    note_id=note_id,
                )
                outcome = self._notes.merge_pair(older_id, merge_result.text, note_id)
                if not outcome["merged"]:
                    # Ранняя исчезла (оператор удалил) между вычиткой и
                    # сведением — слияние не состоялось, работа снимается;
                    # поздняя при этом не трогается (merge_pair ничего не
                    # пишет, если ранней активной заметки нет).
                    self._mark_job_done(job["id"])
                    done += 1
                    continue
            except SummaryError as exc:
                self._log_slot_failure(
                    SUMMARY_JOB,
                    SLOT_SUMMARY,
                    outcome_of(exc),
                    event="dedup_merge_failed",
                    message="dedup: summarizer merge failed — both notes kept, retry queued",
                    older_id=older_id,
                    note_id=note_id,
                    detail=str(exc),
                    meta=getattr(exc, "meta", None),
                )
                self._note_attempt(
                    job["id"],
                    outcome_of(exc),
                    detail=str(exc),
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
            result = self._judge.judge(
                by_id[note_id]["text"], by_id[candidate_id]["text"]
            )
            self._log_slot_call(
                JUDGE_JOB,
                SLOT_JUDGE,
                result.meta,
                note_id=note_id,
                candidate_id=candidate_id,
            )
            logging.getLogger("app").info(
                "dedup: judge verdict for cosine candidate",
                extra={
                    "event": "dedup_judge",
                    "job": JUDGE_JOB,
                    "note_id": note_id,
                    "candidate_id": candidate_id,
                    "cosine": cosine_value,
                    "verdict": result.verdict,
                },
            )
            if result.verdict:
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
            except SummaryError as exc:
                self._log_slot_failure(
                    SUMMARY_JOB,
                    SLOT_SUMMARY,
                    outcome_of(exc),
                    event="title_failed",
                    message="title: generation failed — kept null, retry by back-off",
                    note_id=row["id"],
                    detail=str(exc),
                    meta=getattr(exc, "meta", None),
                )
                continue  # отказ: title остаётся NULL, повтор по back-off
            self._log_slot_call(
                SUMMARY_JOB, SLOT_SUMMARY, generated.meta, note_id=row["id"]
            )
            title = self._truncate_title(generated.text)
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
                # Заметка вернулась в очередь векторизации (название входит в
                # полный вектор) — будим embedding-петлю сразу.
                self.notify_embedding_pending()
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

        Фактическая готовность выжимки — сигнал соседней петле (по образцу
        `notify_links_pending` после довекторизации, решение гейта 2026-09-20):
        если хоть одна заметка батча реально доведена до `summary_status='ok'`,
        будится петля `nodes` (`notify_nodes_pending`) — её задание `reclass`
        после сшивания дубликатов могло ждать именно этой выжимки и «спать» по
        back-off (в приёмочном контуре до 15 мин). Отказ генерации (`continue`
        выше — статус остаётся pending) и пустой прогон сигнала не дают: «модели
        недоступны → задание ждёт» сохраняется.
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
                summary_result = self._summarizer.summarize(row["text"])
            except SummaryError as exc:
                self._log_slot_failure(
                    SUMMARY_JOB,
                    SLOT_SUMMARY,
                    outcome_of(exc),
                    event="summary_failed",
                    message="summary: generation failed — kept pending, retry by back-off",
                    note_id=row["id"],
                    detail=str(exc),
                    meta=getattr(exc, "meta", None),
                )
                continue  # отказ: status pending остаётся, повтор по back-off
            self._log_slot_call(
                SUMMARY_JOB, SLOT_SUMMARY, summary_result.meta, note_id=row["id"]
            )
            # Страховка длины (решение гейта 3.1.0, 2026-09-19): модель может
            # не послушать промпт («no more than 150 characters») — в БД уходит
            # усечённое по границе слова саммари, всегда ≤ MAX_SUMMARY_CHARS.
            summary = cap_summary(
                summary_result.text, self._settings.max_summary_chars
            )
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
        if done:
            # Хоть одна заметка партии реально получила готовую суммари — будим
            # петлю `nodes` (lsb-0012, по образцу `notify_links_pending`).
            # Одно событие на батч достаточно; отказ генерации сюда не доходит
            # (выше `continue` — статус остаётся pending).
            self.notify_nodes_pending()
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
            if cursor.rowcount and move:
                # №41 (FR-3.1/FR-4.4, §13.5): фактический переезд `namespace` —
                # контентное изменение, версия пишется ТОЙ ЖЕ транзакцией.
                # Источник — из `bind_origin` тела джобы (`job:<name>`);
                # служебные `hint_path`/`confidence` в проекцию не входят.
                row = conn.execute(
                    "SELECT text, title, namespace, expires_at, comment FROM notes "
                    "WHERE id = ?",
                    (note_id,),
                ).fetchone()
                history.record_change(
                    conn, history.ENTITY_NOTE, note_id, history.note_projection(row)
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
        except ClassificationError as exc:
            self._log_slot_failure(
                SUMMARY_JOB,
                SLOT_SUMMARY,
                outcome_of(exc),
                event="classify_failed",
                message="classify: failed — note stays in default, retry on next update",
                note_id=note_id,
                detail=str(exc),
                meta=getattr(exc, "meta", None),
            )
            return
        self._log_slot_call(
            SUMMARY_JOB, SLOT_SUMMARY, result.meta, note_id=note_id
        )
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
        # Ворота слота summary (№39, §6.2): при разрыве классификатор не
        # зовём — прогон возвращает 0, петля уходит в back-off; задание и
        # очередь не теряются (разбудит сигнал/таймаут после снятия разрыва).
        if not self._slot_gate(SLOT_SUMMARY, NODES_JOB):
            return 0
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
        считается: петля уходит в сон по back-off — а когда выжимка станет
        готовой, петлю разбудит `notify_nodes_pending` из
        `process_summary_pending`.
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
          back-off: выжимку доведёт summary-петля и разбудит эту сигналом
          `notify_nodes_pending`), по fallback-усечению решение не
          принимается, маркер разбора не ставится — событие
          `node_order`/`summary_pending`;
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
        except ClassificationError as exc:
            # Отказ классификатора данные не портит (FR-3.1): заметка остаётся
            # в текущем узле, разметка не пишется, задание — pending. №39:
            # событие несёт тип исхода отказа слота summary.
            self._log_slot_failure(
                NODES_JOB,
                SLOT_SUMMARY,
                outcome_of(exc),
                event="node_order_failed",
                message="nodes: reclass failed — note stays in its node, retry by back-off",
                note_id=note_id,
                detail=str(exc),
                source="after_merge",
                meta=getattr(exc, "meta", None),
            )
            return False
        self._log_slot_call(
            NODES_JOB, SLOT_SUMMARY, result.meta, note_id=note_id
        )
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
            except ClassificationError as exc:
                self._log_slot_failure(
                    NODES_JOB,
                    SLOT_SUMMARY,
                    outcome_of(exc),
                    event="classify_failed",
                    message="nodes: classify failed — note stays in default, retry later",
                    note_id=note_id,
                    detail=str(exc),
                    meta=getattr(exc, "meta", None),
                )
                continue  # маркер не ставим: заметка снова кандидат обхода
            self._log_slot_call(
                NODES_JOB, SLOT_SUMMARY, result.meta, note_id=note_id
            )
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

        Считает РОВНО работу, которую прогон `process_nodes` может взять в этом
        прогоне: (1) решаемые задания `reclass` после сшивания — заметка с
        готовой суммари либо её нет в живых (такое задание снимается сразу);
        ждущее суммари задание НЕ считается — джоба его не берёт, а сама
        заметка видна в очереди `summary`; (2) кандидаты быстрого пула обхода
        `default`; (3) кандидаты классификаторного пула. Иначе счётчик
        показывал бы работу, которой у джобы нет (FAIL приёмки 3.1.0, решение
        гейта 2026-09-19).

        `oldest_pending_sec` — возраст старейшего ИЗ ПОСЧИТАННЫХ источников
        (задания — `created_at`, кандидаты — `updated_at`), `null` — очередь
        пуста. Только SQL, без обращений к моделям; снимок работает и у
        выключенной джобы (`SQL` собирается в `_nodes_queue_stat_sql`).
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            row = conn.execute(
                _nodes_queue_stat_sql(),
                {
                    "threshold":
                        self._settings.namespace_auto_move_min_confidence
                },
            ).fetchone()
        stat = queue_snapshot(row["pending"], row["oldest_pending_sec"])
        # №39 (arch §6.9): классификатор обхода `nodes` ходит в слот summary —
        # при его разрыве очередь ждёт слот (работа едет в `waiting`).
        return self._gated_queue_stat(SLOT_SUMMARY, stat)

    def _nodes_queue_empty(self) -> bool:
        """Пуста ли очередь `nodes` (дешёвая перепроверка, пул 6, lost wakeup).

        Петля перед `clear()+wait` перепроверяет саму очередь, а не только
        событие: сигнал `notify_nodes_pending` (задание после сшивания),
        пришедший между пустым прогоном и `clear()`, иначе был бы стёрт —
        берущееся задание ждало бы интервал/back-off (до 15 мин). Источник —
        ровно тот же, что у `queue_stat` (arch §3.7): берущиеся задания
        `reclass` + кандидаты обоих пулов обхода `default`. Заметка/задание,
        ждущее суммари, очередь непустой НЕ делает: джоба его не возьмёт, а
        сама заметка видна в очереди `summary` — иначе петля крутилась бы
        вхолостую (busy-loop) вместо back-off. Только SQL, без обращений к
        моделям (FR-2.2).
        """
        return int(self._nodes_queue_stat()["pending"] or 0) == 0

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
# (`app/services/jobs.py`). Формы: «по интервалу» (`expiration`), «по интервалу +
# событие» (`embedding` — свежая pending-заметка) и «по требованию» с
# перепроверкой очереди (summary, judge, areas, links и `nodes` — задание после
# сшивания; перепроверка `nodes` с 2026-09-20). Интервалы существующих петель —
# как были (FR-1.5):
# PENDING_RETRY_SEC у очередей и фиксированные 300 с у зачистки (новых env эта
# постановка не заводит). `queue_stat` — снимок своей очереди для
# `/health.queues` (lsb-0014-03, FR-2.2): у `expiration` очереди нет (`None`).


def build_embedding_job(worker: BackgroundWorker, settings: Settings) -> JobSpec:
    """Джоба `embedding`: notes-очередь + чанковая очередь (одна джоба).

    Форма «по интервалу + событие `embedding`» (gate 2026-09-19): интервал
    `PENDING_RETRY_SEC` с back-off остаётся страховкой, а сигнал
    `notify_embedding_pending` (заметка записана/обновлена/слита — вернулась в
    `vector_status='pending'`) будит петлю сразу: свежая заметка кодируется
    немедленно, а не после выросшего back-off (до 15 мин). Пустой прогон —
    гигиена `worker_jobs` (`idle_hook`, пул 5) и пауза PENDING_RETRY_SEC с
    back-off.

    Перепроверку очереди (`queue_empty`) НЕ задаём намеренно: очередь
    векторизации — это и есть ожидание модели (`vector_status='pending'`),
    поэтому непустая очередь означает «отложенное задание», а не прогресс
    (arch §3.2) — с ней петля крутилась бы вхолостую в busy-loop при
    недоступном кодировщике. Задание при этом не теряется: оно живёт в статусе
    заметки и будет выбрано следующим прогоном, а события `queue_waiting`
    уход в ожидание с непустой очередью видны в журнале (FR-2.3).
    """
    return JobSpec(
        name=EMBEDDING_JOB,
        queue="vector",
        interval_sec=settings.pending_retry_sec,
        batch=None,
        enabled=True,
        process=worker._process_embedding,
        queue_empty=None,
        wait_event=worker._embedding_event,
        idle_hook=worker._purge_done_jobs,
        queue_stat=worker._vector_queue_stat,
        slot=SLOT_EMBEDDING,
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
        slot=SLOT_SUMMARY,
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
        slot=SLOT_JUDGE,
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
        slot=None,
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
        slot=None,
    )


def build_nodes_job(worker: BackgroundWorker, settings: Settings) -> JobSpec:
    """Джоба `nodes`: обход `default` и реклассификация после сшивания (lsb-0011).

    Форма «по интервалу + событие `nodes` + перепроверка очереди» (arch §3.1):
    пустой прогон — сон на `JOB_NODES_INTERVAL_SEC`, сигнал `notify_nodes_pending`
    (задание после сшивания и готовая выжимка объединённой заметки, lsb-0012)
    будит петлю сразу, а `queue_empty`
    закрывает гонку lost wakeup (пул 6): перед `clear()+wait` очередь
    перепроверяется (см. `_nodes_queue_empty`). Прогон — промоция, задания
    `reclass`, затем обход `default` (быстрый пул без модели + классификаторный
    в бюджете `JOB_NODES_CLASSIFIER_BUDGET`). `JOB_NODES_ENABLED=false`
    джобу не запускает, но очередь остаётся видна в `/health` (реестр её
    сохраняет). Синхронный SQL/механика уходят в поток — event loop не занимаем.

    Перепроверка (`queue_empty`) считает работой ровно то, что джоба может взять
    сейчас (берущееся задание `reclass` либо кандидат пула) и заставляет петлю
    продолжить прогон без сна; заметка/задание, ждущее модель или суммари, — не
    работа, и петля уходит в сон по back-off (busy-loop'а нет, arch §3.2,
    «отложенное задание — не прогресс»).
    """
    batch = settings.job_nodes_batch

    async def process() -> int:
        # №41 (§13.4): тело джобы несёт источник `job:nodes` — переезд
        # `namespace` записывает версию с этой пометкой (FR-4.4).
        with history.bind_origin(f"job:{NODES_JOB}"):
            return await asyncio.to_thread(worker.process_nodes, batch)

    return JobSpec(
        name=NODES_JOB,
        queue=NODES_JOB,
        interval_sec=settings.job_nodes_interval_sec,
        batch=batch,
        enabled=settings.job_nodes_enabled,
        process=process,
        queue_empty=worker._nodes_queue_empty,
        wait_event=worker._nodes_event,
        idle_hook=None,
        queue_stat=worker._nodes_queue_stat,
        slot=SLOT_SUMMARY,
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
