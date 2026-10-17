"""Хранилище (ARCHITECTURE §3.3): SQLite-схема notes + FTS5 (trigram) + vec0.

Слой без доменных правил: только схема, соединения и транзакции. Семантика
CRUD (валидации, статусы, soft delete) — в `app.services.notes`; векторные
операции (сериализация vec0, KNN) — в `app.storage.vectors` (Фаза 3).

Ключевые решения:
- Схема создаётся идемпотентно при старте (`init_db`, `IF NOT EXISTS`).
- `notes_fts` — FTS5 внешнего контента (`content='notes'`,
  `content_rowid='id'`, `tokenize='trigram'`), индексирует НАЗВАНИЕ и ТЕКСТ
  заметки (колонки title и text — lsb-0001 FR-1.2/FR-1.3), синхронизируется
  триггерами AFTER INSERT / AFTER UPDATE OF text, title. DELETE-триггер не
  нужен: удаление — soft (`deleted_at`), строка и FTS-индекс физически
  остаются в trash.
- Title-миграция (lsb-0001-01, решение гейта R4 — миграция при старте, не
  джоба): у живых БД, созданных до индексации названия, notes_fts
  пересоздаётся с колонкой title и переливается из notes заново;
  notes_vec дропается — полные вектора считались по тексту без названия
  и невалидны (все заметки, включая trash, → vector_status='pending',
  вектора в этом шаге не кодируются — догоняет фоновый воркер);
  notes_chunks_vec НЕ трогается: вектора чанков строятся по текстам
  чанков, название на них не влияет. Идемпотентность — meta-ключ
  `title_index_version` (значение «2»).
- `notes_vec` — vec0-таблица, размерность фиксируется при создании БД
  (ARCH §3.3); при несовпадении конфигурации с зафиксированной в БД
  (EMBEDDING_DIM или смена EMBEDDING_MODEL, записанная в таблице meta)
  запускается ПОЛНАЯ автореиндексация при старте: индекс пересоздаётся,
  все заметки (включая trash) становятся vector_status='pending',
  догоняются фоновым воркером (решение 2026-08-29).
- `notes_chunks` + `notes_chunks_vec` (Фаза 7): вектора строятся по чанкам
  заметки; схема и операции — `app.storage.chunks`. FK note_id+CASCADE и
  PRAGMA foreign_keys=ON в session(); сироты после прямых правок оператора
  чинятся при старте. Смена чанк-параметров (meta) — пере-чанковка (шаг 6),
  смена модели/размерности — дроп ОБОИХ векторных индексов.
- WAL + busy_timeout (ARCH §3.3): чтения конкурентны, писатели сериализуются,
  спор за блокировку разрешается ожиданием до BUSY_TIMEOUT_MS, а не
  мгновенным «database is locked».
- Соединение — на одну операцию (для SQLite дёшево), поэтому оно всегда
  создаётся, используется и закрывается в одном потоке (совместимо с
  `asyncio.to_thread`). Транзакции — явно (`transaction()`, autocommit):
  BEGIN IMMEDIATE сериализует писателей через busy_timeout.
- `synchronous=NORMAL` — штатная практика для WAL: переживает падение
  процесса; сценарии питания — на периодический backup (Фаза 5).
- На старте FTS-индекс сверяется с `notes` (оператор имеет прямой доступ к
  файлу БД, REQUIREMENTS §4); рассинхрон лечится rebuild'ом, прочие ошибки
  целостности — фатальны (StorageError).
- Связи заметок (lsb-0010-02, релиз 3.1.0): таблица `links` (одна строка
  на пару, канонический порядок `note_a < note_b`) + маркер расчёта
  `notes.links_at` — идемпотентная миграция при старте; смена модели или
  размерности эмбеддинга (автореиндексация ниже) очищает и связи.
- Лимит длины саммари (гейт 3.1.0, 2026-09-19): активные заметки с готовым
  саммари длиннее MAX_SUMMARY_CHARS помечаются на перегенерацию
  (`summary_status='pending'`) — текст саммари НЕ портится усечением, его
  заменяет фоновая джоба `summary`; идемпотентно (после перегенерации
  повторный старт — no-op), заметки с `summary_status != 'ok'` уже в очереди.
- Области 3.0.0 (субстрат, ARCH substrate §3.1–3.2): отдельные таблицы,
  FTS5-индексы и vec0-индексы по областям skills/terms/user в той же БД.
  Ни один объект заметок не пересоздаётся; старые записи не мигрируются
  (данные областей — с нуля). FTS — внешнего контента с триггерами
  AFTER INSERT / AFTER UPDATE; `terms` несёт ЧАСТИЧНЫЙ UNIQUE по активным
  записям (`WHERE deleted_at IS NULL`). Area-vec дропаются вместе с
  notes_vec в ветке смены модели/размерности — все записи областей →
  pending, догоняет петля `areas` воркера.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec

from app.config import Settings
from app.storage import area_vectors, chunks, expirations, vectors

# Сколько ждать блокировку записи, прежде чем сдаться (как timeout sqlite3,
# так и PRAGMA busy_timeout).
BUSY_TIMEOUT_MS = 5000

# Как выцупоть лимит CHECK(length(text) BETWEEN 1 AND ?) из DDL живой таблицы
# notes (сверка с MAX_NOTE_CHARS при старте — см. init_db).
_CHECK_LIMIT_RE = re.compile(r"length\(\s*text\s*\)\s+BETWEEN\s+1\s+AND\s+(\d+)")

# Признак «notes_fts уже индексирует title» в DDL из sqlite_master: новая
# схема начинается с колонки title, легаси (v2.1.1) — с text (миграция
# lsb-0001-01 — см. _sync_fts_title_index; по образцу _PARTITION_RE в
# app.storage.vectors).
_FTS_TITLE_RE = re.compile(r"fts5\(\s*title")


class StorageError(RuntimeError):
    """Ошибка хранилища: БД недоступна или схема повреждена.

    Фатальна на старте; при обслуживании запросов доходит транспортным
    слоям как обычное исключение (формат ответа — зона Фазы 5).
    """


# --- схема --------------------------------------------------------------

# DDL-черновик ARCHITECTURE §3.3 (без vec0-таблицы — это Фаза 3). Лимит
# CHECK подставляется из env при первой инициализации БД (ARCH §3.3:
# «лимиты подставляются из env»); подставляется целое число — безопасно.
_NOTES_DDL = """
CREATE TABLE IF NOT EXISTS notes (
  id             INTEGER PRIMARY KEY,
  text           TEXT    NOT NULL CHECK(length(text) BETWEEN 1 AND {max_note_chars}),
  title          TEXT    NULL,
  comment        TEXT    NULL,
  summary        TEXT    NOT NULL DEFAULT '',
  author         TEXT    NOT NULL DEFAULT 'unknown',
  vector_status  TEXT    NOT NULL DEFAULT 'pending',
  summary_status TEXT    NOT NULL DEFAULT 'pending',
  created_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  deleted_at     TEXT    NULL,
  namespace      TEXT    NOT NULL DEFAULT 'default',
  classified_at  TEXT    NULL,
  hint_path      TEXT    NULL,
  confidence     REAL    NULL,
  expires_at     TEXT    NULL,
  links_at       TEXT    NULL,
  node_order_at  TEXT    NULL,
  warnings_at    TEXT    NULL
)
"""

# Реестр неймспейсов (Фаза 10, REQUIREMENTS §5.7): узлы иерархии (path —
# слэш-путь, максимум 2 уровня), description — обязательное (контракт
# ≤2 предложений валидируется сервисом), status — confirmed | provisional
# (авто-созданные узлы судьи структуры). Существующие заметки при миграции
# уходят в 'default' (дефолт колонки + INSERT узла ниже).
_NAMESPACES_DDL = """
CREATE TABLE IF NOT EXISTS namespaces (
  path        TEXT PRIMARY KEY,
  description TEXT NOT NULL,
  status      TEXT NOT NULL DEFAULT 'confirmed'
              CHECK(status IN ('confirmed', 'provisional')),
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
)
"""

# lsb-0001-01 (FR-1.2/FR-1.3): FTS индексирует и название (title), и текст.
_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
  title, text, content='notes', content_rowid='id', tokenize='trigram'
)
"""

# Вердикты судьи структуры (Фаза 10, Шаг 5): cooldown триггера. Группа
# hint_path'ов, по которой судья вынес вердикт (слияние/отклонение), не
# дёргает LLM повторно — иначе отклонённый кандидат зациклил бы вызовы
# судьи при каждом прогоне (заметки с отклонённым hint'ом остаются в
# default навсегда — честно-общие). Слияние хранит канонический узел;
# сброс вердикта — оператор (REST, Шаг 6). Созданный узел записей не
# требует: группа уходит из default ретро-перекладкой.
#
# lsb-0005-03: ключ — единый полный путь разметки hint_path (1..3 уровня),
# заменяет пару (domain, subdomain). Старая схема (ПК (domain, subdomain))
# мигрируется в _migrate_promotions_key — существующие вердикты получают
# hint_path = domain/subdomain.
_PROMOTIONS_DDL = """
CREATE TABLE IF NOT EXISTS promotions (
  hint_path      TEXT NOT NULL,
  status         TEXT NOT NULL CHECK(status IN ('merged', 'rejected')),
  canonical_path TEXT NULL,
  decided_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  PRIMARY KEY (hint_path)
)
"""

# lsb-0004-02 (FR-5…FR-8): очередь удаления просроченных заметок. note_id —
# PK (одна запись на заметку), expires_at — абсолютный ISO-8601 (UTC) срок
# жизни. Создаётся идемпотентно (IF NOT EXISTS); синхронизация с notes —
# на уровне сервиса (вставка при TTL, обновление при смене, удаление при
# снятии/удалении заметки).
_NOTE_EXPIRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS note_expirations (
  note_id    INTEGER PRIMARY KEY,
  expires_at TEXT    NOT NULL
)
"""

# Связи заметок, уровень 1 (lsb-0010-02, arch §3.2): одна строка на пару
# заметок; пара хранится в каноническом порядке `note_a < note_b` (связь
# симметрична, чтение — в обе стороны), `kind` — вид связи с приоритетом
# mention > entities > cosine, `score` — косинус/доля общих слов (NULL для
# mention). Строки пишет только расчёт (`LinksService.compute_for_note`) —
# одна транзакция на заметку: удалить пары → вставить заново.
_LINKS_DDL = """
CREATE TABLE IF NOT EXISTS links (
  note_a     INTEGER NOT NULL,          -- меньший id пары
  note_b     INTEGER NOT NULL,          -- больший id пары
  kind       TEXT    NOT NULL,          -- 'mention' | 'entities' | 'cosine'
  score      REAL,                      -- косинус / доля общих слов / NULL
  created_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  PRIMARY KEY (note_a, note_b)
)
"""

# Индекс обратной стороны пары: выдача и пересчёт ищут связи заметки в обе
# стороны (`note_a = ? OR note_b = ?`), PRIMARY KEY покрывает только note_a.
_LINKS_INDEX_DDL = "CREATE INDEX IF NOT EXISTS idx_links_b ON links(note_b)"

# Таблица предупреждений (№12 + №32, arch §5.1): ОДНА общая поверхность
# предупреждений памяти и терминов, читаемая без вызова модели. Субстрат
# (прецедент общей `meta`): доступ по ключу `(subject_type, subject_id)`,
# SQL изоляции областей её не адресует. Не более одного ОТКРЫТОГО
# предупреждения на (субъект, вид) — повторный прогон джобы обновляет
# `updated_at`/`payload`, а не плодит строки (§13.7).
_WARNINGS_DDL = """
CREATE TABLE IF NOT EXISTS warnings (
  id           INTEGER PRIMARY KEY,
  kind         TEXT    NOT NULL,
  subject_type TEXT    NOT NULL,
  subject_id   INTEGER NOT NULL,
  message      TEXT    NOT NULL,
  payload      TEXT    NOT NULL DEFAULT '{}',
  status       TEXT    NOT NULL DEFAULT 'open' CHECK(status IN ('open','dismissed')),
  created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  resolved_at  TEXT    NULL
)
"""

# Частичный уникальный индекс: не более одного ОТКРЫТОГО предупреждения на
# (субъект, вид) — идемпотентность upsert'а (§5.1, §13.7).
_WARNINGS_UNIQUE_INDEX_DDL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_warnings_open_unique "
    "ON warnings(subject_type, subject_id, kind) "
    "WHERE status = 'open' AND resolved_at IS NULL"
)

# Рабочий индекс листинга/агрегатов (/health.warnings, GET /warnings).
_WARNINGS_STATUS_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_warnings_status_kind "
    "ON warnings(status, kind, id)"
)

# Вектора (Фаза 3): физическая схема в `app.storage.vectors` (размерность и
# cosine-метрика — там же); в init_db — создание/сверка при старте. Модель
# эмбеддинга, на которой построен индекс, — в таблице meta (см. ниже):
# смена модели/размерности поверх живой БД = автоматическая переиндексация.
_META_DDL = """
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
)
"""

# Синхронизация FTS с notes. Для внешнего контента удаление из индекса —
# спец-команда 'delete' со СТАРЫМИ значениями индексируемых колонок.
# UPDATE OF text, title (lsb-0001-01): изменения прочих колонок (summary,
# статусы, deleted_at при soft delete) FTS не трогают.
_TRIGGERS = (
    """
    CREATE TRIGGER IF NOT EXISTS notes_fts_ai AFTER INSERT ON notes BEGIN
      INSERT INTO notes_fts(rowid, title, text) VALUES (new.id, new.title, new.text);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS notes_fts_au AFTER UPDATE OF text, title ON notes BEGIN
      INSERT INTO notes_fts(notes_fts, rowid, title, text) VALUES ('delete', old.id, old.title, old.text);
      INSERT INTO notes_fts(rowid, title, text) VALUES (new.id, new.title, new.text);
    END
    """,
)

# №24 (no24-02, 3.2.0): ОТДЕЛЬНЫЙ FTS-индекс пометок (arch §5.2). Смысловой
# `notes_fts` индексирует title/text и пометок НЕ видит: пометки ищутся
# отдельным поиском (MCP `memory_comment_search` / REST `GET /notes/comments`)
# — результаты не смешиваются (§13.6).
_COMMENT_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS notes_comment_fts USING fts5(
  comment, content='notes', content_rowid='id', tokenize='trigram'
)
"""

# Синхронизация индекса пометок с notes (по образцу _TRIGGERS): INSERT всегда
# (в т.ч. comment IS NULL — «пометки нет»), UPDATE OF comment — спец-команда
# 'delete' со СТАРЫМ значением + вставка нового. DELETE-триггера нет: заметки
# soft-delete, физического удаления в штатном пути нет (как у notes_fts).
_COMMENT_TRIGGERS = (
    """
    CREATE TRIGGER IF NOT EXISTS notes_comment_fts_ai AFTER INSERT ON notes BEGIN
      INSERT INTO notes_comment_fts(rowid, comment) VALUES (new.id, new.comment);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS notes_comment_fts_au AFTER UPDATE OF comment ON notes BEGIN
      INSERT INTO notes_comment_fts(notes_comment_fts, rowid, comment) VALUES ('delete', old.id, old.comment);
      INSERT INTO notes_comment_fts(rowid, comment) VALUES (new.id, new.comment);
    END
    """,
)


# --- области 3.0.0 (субстрат: skills / terms / user) ----------------------

# Один раз на релиз: изолированный слой областей в общей БД notes.db.
# Ни один SQL области не адресует notes* (изоляция — тесты в обе стороны).
SKILLS_TABLE = "skills"
SKILLS_FTS_TABLE = "skills_fts"
SKILLS_FTS_COLUMNS = ("name", "description", "steps", "text")
TERMS_TABLE = "terms"
TERMS_FTS_TABLE = "terms_fts"
TERMS_FTS_COLUMNS = ("term", "context", "definition")
USER_FACTS_TABLE = "user_facts"
USER_FACTS_FTS_TABLE = "user_facts_fts"
USER_FACTS_FTS_COLUMNS = ("name", "body")

# Область «Логи ассистента» (LSBN-F-10, релиз 3.2.0, arch §5.1/§5.2): одна
# таблица записей + FTS5-индекс тех же смысловых текстовых полей. Структурные
# фильтры (`entity*`, `source`, `session`, `created_at`) — обычные колонки и в
# FTS не входят.
ASSISTANT_LOGS_TABLE = "assistant_logs"
ASSISTANT_LOGS_FTS_TABLE = "assistant_logs_fts"
ASSISTANT_LOGS_FTS_COLUMNS = ("action", "reason", "entity_ref", "outcome", "verified")

# Записи областей (без vec-таблиц): общие колонки + поля конкретной области.
AREA_RECORD_TABLES = (
    SKILLS_TABLE,
    TERMS_TABLE,
    USER_FACTS_TABLE,
    ASSISTANT_LOGS_TABLE,
)

# (таблица записей, FTS-таблица, индексируемые колонки) — реестр FTS/триггеров:
# один DDL и один набор триггеров на каждую область (без копипасты по областям).
_AREA_FTS_SPECS = (
    (SKILLS_TABLE, SKILLS_FTS_TABLE, SKILLS_FTS_COLUMNS),
    (TERMS_TABLE, TERMS_FTS_TABLE, TERMS_FTS_COLUMNS),
    (USER_FACTS_TABLE, USER_FACTS_FTS_TABLE, USER_FACTS_FTS_COLUMNS),
    (ASSISTANT_LOGS_TABLE, ASSISTANT_LOGS_FTS_TABLE, ASSISTANT_LOGS_FTS_COLUMNS),
)

# Обязательные таблицы «ядра» для проверки снапшота (no14-01, релиз 3.2.0):
# имена всех не-векторных таблиц, которые `init_db` создаёт в боевой БД (ядро
# заметок + записи/FTS областей). Векторные (vec0) таблицы живут в своих
# реестрах (`vectors.VEC_TABLE`, `chunks.CHUNKS_VEC_TABLE`, `area_vectors.VEC_TABLES`)
# и здесь не перечисляются: `BackupService.verify` собирает обязательный
# набор из всех реестров `app.storage`, а не хардкодит его. Единый источник
# имён ядра — этот кортеж.
CORE_TABLES: tuple[str, ...] = (
    "notes",
    "notes_fts",
    "notes_comment_fts",
    "namespaces",
    "promotions",
    "note_expirations",
    "links",
    "warnings",
    "meta",
    chunks.CHUNKS_TABLE,
    "record_versions",
    SKILLS_TABLE,
    SKILLS_FTS_TABLE,
    "skill_versions",
    "skills_meta",
    TERMS_TABLE,
    TERMS_FTS_TABLE,
    USER_FACTS_TABLE,
    USER_FACTS_FTS_TABLE,
    ASSISTANT_LOGS_TABLE,
    ASSISTANT_LOGS_FTS_TABLE,
)

# Поля записей — по архитектурам фич lsb-0007/0008/0009; общее у всех —
# id, vector_status (default pending), created_at, updated_at, deleted_at.
# CHECK-констрейнты, как у заметок, не используем: лимиты формы — валидация
# сервиса (мягкий отказ с hint), DDL — только структура.
_AREA_DDL = (
    """
CREATE TABLE IF NOT EXISTS skills (
  id            INTEGER PRIMARY KEY,
  name          TEXT    NOT NULL DEFAULT '',
  description   TEXT    NOT NULL DEFAULT '',
  example       TEXT    NULL,
  steps         TEXT    NOT NULL DEFAULT '',
  text          TEXT    NOT NULL DEFAULT '',
  extra         TEXT    NULL,
  version       INTEGER NOT NULL DEFAULT 1,
  vector_status TEXT    NOT NULL DEFAULT 'pending',
  created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  deleted_at    TEXT    NULL
)
""",
    """
CREATE TABLE IF NOT EXISTS skill_versions (
  skill_id    INTEGER NOT NULL,
  version     INTEGER NOT NULL,
  name        TEXT    NOT NULL DEFAULT '',
  description TEXT    NOT NULL DEFAULT '',
  example     TEXT    NULL,
  steps       TEXT    NOT NULL DEFAULT '',
  text        TEXT    NOT NULL DEFAULT '',
  extra       TEXT    NULL,
  created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  PRIMARY KEY (skill_id, version)
)
""",
    """
CREATE TABLE IF NOT EXISTS skills_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
)
""",
    """
CREATE TABLE IF NOT EXISTS terms (
  id            INTEGER PRIMARY KEY,
  term          TEXT    NOT NULL DEFAULT '',
  term_norm     TEXT    NOT NULL DEFAULT '',
  context       TEXT    NOT NULL DEFAULT '',
  context_norm  TEXT    NOT NULL DEFAULT '',
  definition    TEXT    NOT NULL DEFAULT '',
  vector_status TEXT    NOT NULL DEFAULT 'pending',
  created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  deleted_at    TEXT    NULL
)
""",
    """
CREATE TABLE IF NOT EXISTS user_facts (
  id            INTEGER PRIMARY KEY,
  name          TEXT    NOT NULL DEFAULT '',
  body          TEXT    NOT NULL DEFAULT '',
  vector_status TEXT    NOT NULL DEFAULT 'pending',
  created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  deleted_at    TEXT    NULL
)
""",
    """
CREATE TABLE IF NOT EXISTS assistant_logs (
  id             INTEGER PRIMARY KEY,
  action         TEXT    NOT NULL DEFAULT '',   -- что сделано (обязательно, <= LOGS_ACTION_MAX_CHARS)
  reason         TEXT    NOT NULL DEFAULT '',   -- почему (обязательно, <= LOGS_REASON_MAX_CHARS)
  entity         TEXT    NULL,                  -- тип объекта изменения (закрытый словарь, §5.4)
  entity_id      INTEGER NULL,                  -- id объекта в своей области (версионные сущности)
  entity_version INTEGER NULL,                  -- версия объекта на момент записи (№41)
  entity_ref     TEXT    NULL,                  -- текстовая ссылка/путь (объекты без id; <= LOGS_ENTITY_REF_MAX_CHARS)
  outcome        TEXT    NULL,                  -- результат (что получилось; <= LOGS_OUTCOME_MAX_CHARS)
  verified       TEXT    NULL,                  -- что проверено (<= LOGS_VERIFIED_MAX_CHARS)
  source         TEXT    NOT NULL DEFAULT 'mcp',-- источник записи: mcp|rest|job:<name>|merge|migration
  session        TEXT    NULL,                  -- идентификатор вызова/агента, не имя человека
  vector_status  TEXT    NOT NULL DEFAULT 'pending',
  created_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  deleted_at     TEXT    NULL
)
""",
)

# Глобальный шаблон области навыков (lsb-0007 §3.1/§3.8): одна запись
# skills_meta на область — «как исполнять шаги» (не дублируется у навыков).
# Ключ и текст сида: значение создаётся только при отсутствии ключа (правка —
# оператором через REST, lsb-0007-05; сид её не затирает).
INSTRUCTION_TEMPLATE_KEY = "instruction_template"
INSTRUCTION_TEMPLATE_SEED = (
    "Execute the steps in order: the content of each step in `text` says what "
    "exactly to do (result, format, rule). Do not skip or reorder steps; if a "
    "step cannot be executed, stop and report what is missing instead of "
    "improvising."
)

# Сид skill-создателя (lsb-0007 §3.6/§3.8): запись `skills` — процедура «как
# создавать навыки», тексты дословно канон арх-доки §3.8 (в доке разбиты на
# строки по ~80 симв. — это вёрстка, строки склеены пробелами, как у
# INSTRUCTION_TEMPLATE_SEED). `extra` пуст, `vector_status='pending'` — вектор
# догоняет петля `areas` воркера (субстрат §3.3).
# Маркер сида в `skills_meta` пишется при первой попытке сида: пока он есть,
# init_db сид НЕ воскрешает — удаление навыка оператором или моделью остаётся
# осмысленным (маркер живёт в БД дольше мягко удалённой строки).
CREATOR_SKILL_SEED_KEY = "creator_skill_seed"
CREATOR_SKILL_SEED_VALUE = "seeded"
CREATOR_SKILL_NAME = "Create skills"
CREATOR_SKILL_DESCRIPTION = "How to add or update a skill in this memory"
CREATOR_SKILL_STEPS = (
    "1) search for a similar skill; 2) name and description; 3) steps and "
    "text; 4) save; 5) read back and check."
)
CREATOR_SKILL_TEXT = (
    "1) Call skills_search with the task wording: a similar skill exists → "
    "update it (skills_save with id), never create a duplicate. 2) name ≤65 "
    "characters (≤5 words recommended); description ≤400 — what the procedure "
    "does. 3) steps ≤1000 — the order (what after what); text ≤7500 — what "
    "exactly each step does (result/format/rule); keep the \"how to execute\" "
    "wording in the global instruction_template, never duplicate it per skill. "
    "4) Optional class fields (trigger, mode, preconditions, fallbacks, "
    "invariant, exceptions, guardrails, references, output_contract, "
    "behavior_contract) and example (≤2000) — only when this skill class needs "
    "them. 5) skills_save, then skills_get to check that the procedure reads "
    "as a ready-to-execute routine. 6) Self-improvement: when you are sure a "
    "skill can be improved, propose the exact edit (what and why) and ask the "
    "user for approval; apply it only after approval — the server keeps the "
    "previous version as a copy automatically."
)

# Частичный UNIQUE (lsb-0008 §3.3): ключ термина (term_norm + context_norm)
# уникален только среди АКТИВНЫХ записей — soft-deleted строку ключ не держит
# (удалённый ключ освобождается, «undo — оператором»).
_TERMS_ACTIVE_KEY_INDEX_DDL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_terms_key_active "
    "ON terms(term_norm, context_norm) WHERE deleted_at IS NULL"
)

# Индексы обратного запроса области логов (arch §5.1, FR-4.1/FR-5.2): оба
# частичные — только по активным записям (прецедент idx_terms_key_active).
# Первый — «почему изменился объект»; второй — срез по абсолютной дате.
# Отдельных индексов по session/source нет: домен узкий, объём мал (§4).
_ASSISTANT_LOGS_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_assistant_logs_entity_active "
    "ON assistant_logs(entity, entity_id, created_at DESC) WHERE deleted_at IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_assistant_logs_created_active "
    "ON assistant_logs(created_at DESC) WHERE deleted_at IS NULL",
)

# История версий (№41, релиз 3.2.0, substrate-versioning §5.1): одна общая
# таблица на все сущности, PK `(entity, entity_id, version)` — покрывает
# lookup по записи и обратный порядок (от новых к старым). Индексов сверх PK
# НЕ заводится (решение §5.1). `version=1` — baseline (`snapshot` заполнен,
# `delta` NULL); `version≥2` — дельта, кроме перебазированной после prune.
_RECORD_VERSIONS_DDL = """
CREATE TABLE IF NOT EXISTS record_versions (
  entity       TEXT    NOT NULL,                     -- 'note'|'skill'|'skill_template'|'term'|'user_fact'
  entity_id    INTEGER NOT NULL,                     -- id строки живой таблицы; для skill_template — 0
  version      INTEGER NOT NULL,                     -- 1..N, baseline = 1
  event        TEXT    NOT NULL DEFAULT 'content'
               CHECK(event IN ('baseline','content','delete','restore','rollback')),
  source       TEXT    NOT NULL,                     -- 'mcp'|'rest'|'job:<имя>'|'merge'|'migration'
  session      TEXT    NULL,                         -- идентификатор вызова/агента, не имя человека
  changed      TEXT    NOT NULL DEFAULT '[]',        -- JSON-массив имён изменившихся полей
  snapshot     TEXT    NULL,                         -- JSON-объект: поля целиком (baseline / после prune)
  delta        TEXT    NULL,                         -- JSON-объект: поле → значение/строки (§5.2)
  restore_from INTEGER NULL,                         -- откат: номер восстановленной версии
  created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  PRIMARY KEY (entity, entity_id, version)
)
"""


def _area_fts_ddl(table: str, fts_table: str, columns: tuple[str, ...]) -> str:
    """DDL FTS5-индекса области: внешний контент, trigram (как notes_fts)."""
    return (
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {fts_table} USING fts5("
        f"  {', '.join(columns)}, content='{table}', content_rowid='id', "
        "tokenize='trigram')"
    )


def _area_trigger_ddls(
    table: str, fts_table: str, columns: tuple[str, ...]
) -> tuple[str, ...]:
    """Триггеры синхронизации FTS области: AFTER INSERT и AFTER UPDATE.

    DELETE-триггер не нужен: удаление области — soft (`deleted_at`), строка и
    FTS-индекс физически остаются в trash (прецедент заметок). Для внешнего
    контента удаление из индекса — спец-команда 'delete' со СТАРЫМИ
    значениями индексируемых колонок (только в UPDATE).
    """
    joined = ", ".join(columns)
    new_values = ", ".join(f"new.{column}" for column in columns)
    old_values = ", ".join(f"old.{column}" for column in columns)
    return (
        f"""
CREATE TRIGGER IF NOT EXISTS {fts_table}_ai AFTER INSERT ON {table} BEGIN
  INSERT INTO {fts_table}(rowid, {joined}) VALUES (new.id, {new_values});
END
""",
        f"""
CREATE TRIGGER IF NOT EXISTS {fts_table}_au AFTER UPDATE ON {table} BEGIN
  INSERT INTO {fts_table}({fts_table}, rowid, {joined}) VALUES ('delete', old.id, {old_values});
  INSERT INTO {fts_table}(rowid, {joined}) VALUES (new.id, {new_values});
END
""",
    )


def _create_area_schema(conn: sqlite3.Connection, settings: Settings) -> None:
    """Создать схему областей при старте — идемпотентно (ARCH substrate §3.1).

    Таблицы записей + FTS5 внешнего контента + триггеры + частичный UNIQUE
    ключа terms + vec0-таблицы текущей размерности. Живые БД апгрейдятся без
    ручных миграций (всё `IF NOT EXISTS`); существующие объекты заметок не
    трогаются вовсе.
    """
    for ddl in _AREA_DDL:
        conn.execute(ddl)
    for table, fts_table, columns in _AREA_FTS_SPECS:
        conn.execute(_area_fts_ddl(table, fts_table, columns))
        for trigger in _area_trigger_ddls(table, fts_table, columns):
            conn.execute(trigger)
    conn.execute(_TERMS_ACTIVE_KEY_INDEX_DDL)
    for ddl in _ASSISTANT_LOGS_INDEX_DDL:
        conn.execute(ddl)
    area_vectors.create_vec_tables(conn, settings.embedding_dim)


def _create_area_vec_if_missing(conn: sqlite3.Connection, dim: int) -> None:
    """Создать отсутствующие area-vec (наследники до 3.0.0, ручные правки)."""
    if not area_vectors.vec_tables_exist(conn):
        area_vectors.create_vec_tables(conn, dim)


def _reset_area_vectors(conn: sqlite3.Connection, settings: Settings) -> None:
    """Смена модели/размерности: area-vec дропаются и пересоздаются.

    Вектора другой модели несовместимы: все записи областей (включая trash)
    → `vector_status='pending'`, догоняет петля `areas` воркера — по образцу
    notes_vec (NFR-3: данные не теряются, поиск деградирует к FTS).
    """
    area_vectors.drop_vec_tables(conn)
    area_vectors.create_vec_tables(conn, settings.embedding_dim)
    for table in AREA_RECORD_TABLES:
        conn.execute(f"UPDATE {table} SET vector_status = 'pending'")


def _count_area_records(conn: sqlite3.Connection) -> int:
    """Число записей областей (события автореиндексации, наблюдение)."""
    return sum(
        int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in AREA_RECORD_TABLES
    )


# --- соединения ---------------------------------------------------------

@contextmanager
def session(settings: Settings) -> Iterator[sqlite3.Connection]:
    """Соединение на одну операцию + WAL-прагмы (ARCH §3.3).

    Autocommit (`isolation_level=None`): каждый оператор атомарен сам по
    себе, а многооператорные изменения оборачиваются `transaction()`.
    """
    try:
        conn = sqlite3.connect(
            settings.db_path,
            timeout=BUSY_TIMEOUT_MS / 1000,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row  # доступ к ячейкам по имени колонки
    except sqlite3.Error as exc:  # например, путь недоступен
        raise StorageError(
            f"не удалось открыть БД {settings.db_path}: {exc}"
        ) from exc
    try:
        conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        # Фаза 7: FK ON — без него ON DELETE CASCADE у notes_chunks молча
        # не срабатывает (в SQLite внешние ключи выключены по умолчанию).
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _load_vec_extension(conn)
        yield conn
    except sqlite3.Error as exc:
        raise StorageError(f"ошибка БД ({settings.db_path}): {exc}") from exc
    finally:
        conn.close()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Многооператорная запись: BEGIN IMMEDIATE … COMMIT, при ошибке ROLLBACK.

    BEGIN IMMEDIATE берёт блокировку записи сразу — два конкурентных
    писателя сериализуются через busy_timeout без deadlock при upgrade.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


# --- расширение sqlite-vec ------------------------------------------------

def _load_vec_extension(conn: sqlite3.Connection) -> None:
    """Загрузить sqlite-vec в соединение (нужно на каждом соединении).

    Загрузка расширения — свойство соединения, а не файла БД; стоимость —
    микросекунды. Без расширения vec0-таблицы не открываются вовсе, поэтому
    ошибка загрузки — StorageError (фатально на старте).
    """
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (AttributeError, sqlite3.Error) as exc:
        raise StorageError(
            f"расширение sqlite-vec недоступно (vec0): {exc}"
        ) from exc


# --- инициализация ------------------------------------------------------


def _assert_note_chars_limit(conn: sqlite3.Connection, settings: Settings) -> None:
    """Сверить MAX_NOTE_CHARS с фактическим CHECK-лимитом живой таблицы.

    Для свежей таблицы совпадение гарантировано (DDL выше); сверка ловит
    смену env поверх существующей БД — несовпадение — фатальная ошибка
    конфигурации: лимит зафиксирован в схеме при первой инициализации.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'notes'"
    ).fetchone()
    if row is None or row[0] is None:
        return  # таблица не создана — init_db упал выше по-своему
    match = _CHECK_LIMIT_RE.search(row[0])
    if match is None:
        return  # без CHECK (рукотворная схема) — сверки нет, не виноваты
    stored = int(match.group(1))
    if stored != settings.max_note_chars:
        raise StorageError(
            f"MAX_NOTE_CHARS разошёлся с БД: таблица создана с лимитом "
            f"{stored}, окружение задаёт {settings.max_note_chars}. "
            f"CHECK не меняется на лету: верни MAX_NOTE_CHARS={stored} "
            f"или пересоздай БД (сохрани заметки: sqlite3 {settings.db_path} .dump)."
        )


def init_db(settings: Settings) -> None:
    """Создать схему при старте (критерий приёмки Фазы 2); идемпотентно.

    При смене EMBEDDING_MODEL или EMBEDDING_DIM поверх существующей БД
    автоматически перестраиваются ОБА векторных индекса (полный текст и
    чанки): вектора другой модели несовместимы, все заметки уходят в pending
    и догоняются воркером (NFR-3 — данные не теряются, поиск деградирует
    к FTS до готовности).

    Миграция title-индекса (lsb-0001-01, решение гейта R4 — при старте,
    не джоба): notes_fts живых БД перестраивается под индексацию названия
    (title + text, FR-1.2/FR-1.3), notes_vec дропается — полные вектора
    считались по тексту без названия и невалидны, все заметки (включая
    trash) уходят в pending и догоняются воркером; вектора чанков
    (notes_chunks_vec) остаются валидными — они построены по текстам чанков.

    Raises:
        StorageError: БД недоступна, нет FTS5/vec0, схема повреждена,
        env разошёлся с зафиксированной схемой (CHECK-лимит MAX_NOTE_CHARS).
    """
    path = Path(settings.db_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with session(settings) as conn:
            conn.execute(_NOTES_DDL.format(max_note_chars=settings.max_note_chars))
            # Лимит CHECK «запечён» в DDL при первом создании (ARCH §3.3:
            # «лимиты подставляются из env»); смена MAX_NOTE_CHARS поверх
            # готовой БД — разрыв конфигурации: CHECK не меняется на лету,
            # крупные заметки стали бы падать в рантайме с невнятным
            # IntegrityError (сообщение CHECK-а ничего не говорит об env).
            _assert_note_chars_limit(conn, settings)
            conn.execute(_FTS_DDL)
            for trigger in _TRIGGERS:
                conn.execute(trigger)
            # Названия заметок (Фаза 11, решение №9): notes.title TEXT (nullable).
            # ДО title-миграции FTS: к моменту _sync_fts_title_index колонка
            # title уже существует — перелив индекса читает notes.title.
            _migrate_title_column(conn)
            # Пометка заметки (№24, 3.2.0): notes.comment TEXT (nullable).
            # Колонка нужна до вставок/чтений сервиса; FTS пометок — no24-02.
            _migrate_comment_column(conn)
            # FTS пометок (№24, no24-02): отдельная таблица notes_comment_fts
            # + триггеры — сразу после колонки comment и ДО сверки эмбеддинга
            # (таблицу не адресует). Rebuild не нужен: комментариев ещё нет.
            _migrate_comment_fts(conn)
            # Неймспейсы (Фаза 10): колонки notes + реестр + дефолт-узел.
            _migrate_namespace_columns(conn)
            _migrate_classification_columns(conn)
            # lsb-0005-02: единый полный путь разметки hint_path (замена
            # паре domain_hint/subdomain_hint) — бэкфилл из старой подписи.
            _migrate_hint_path_columns(conn)
            conn.execute(_NAMESPACES_DDL)
            # Временное хранение (lsb-0004-02): колонка notes.expires_at +
            # очередь удаления note_expirations (идемпотентно).
            _migrate_expiration_columns(conn)
            # Связи заметок (lsb-0010-02): таблица links + индекс + маркер
            # notes.links_at — идемпотентно, до ветки автореиндексации
            # (та очищает связи при смене модели/размерности).
            _migrate_link_columns(conn)
            # Порядок в узлах (lsb-0011-01): служебный маркер notes.node_order_at
            # — идемпотентно; у существующих заметок NULL (первый прогон джобы
            # `nodes` разбирает накопленный default — ретро-прогон, arch §4).
            _migrate_node_order_columns(conn)
            # Предупреждения (№12, 3.2.0): служебный маркер notes.warnings_at
            # — идемпотентно; у существующих заметок NULL (первый прогон
            # джобы `warnings` проверяет накопленное).
            _migrate_warnings_columns(conn)
            # Лимит длины саммари (гейт 3.1.0, 2026-09-19): готовые саммари
            # длиннее MAX_SUMMARY_CHARS → pending (перегенерация фоном).
            _migrate_summary_length(conn, settings)
            # List-индексы (пул 15): старый idx_notes_namespace заменён
            # (prefix namespace, deleted_at покрыт новым ns-индексом).
            conn.execute("DROP INDEX IF EXISTS idx_notes_namespace")
            conn.execute(_INDEX_DELETED_UPDATED_DDL)
            conn.execute(_INDEX_NS_DELETED_UPDATED_DDL)
            _ensure_default_namespace(conn)
            # №34 (3.2.0): описание `default` — EN; существующие БД доводятся
            # одноразово и только если текст точно равен прежнему RU-дефолту
            # (правка оператора сохраняется). Штамп в `meta` — см. функцию.
            _migrate_default_namespace_description(conn)
            conn.execute(_PROMOTIONS_DDL)
            # lsb-0005-03: ключ promotions (domain, subdomain) → единый
            # hint_path; миграция старых вердиктов (идемпотентно).
            _migrate_promotions_key(conn)
            # Title-индекс (lsb-0001-01, Часть A): перестройка notes_fts под
            # индексацию названия на живых БД — ДО integrity-check: перелив
            # выше оставляет индекс консистентным.
            fts_migrated = _sync_fts_title_index(conn)
            _check_fts_integrity(conn)
            # Партиция namespace (Фаза 10): vec-таблицы живых БД без `+ns`
            # пересоздаются с партицией, все заметки — в pending. ДО сверки
            # модели/чанков: обе ветки создают таблицы уже с партицией.
            _sync_namespace_partition(conn, settings)
            # Чанки (Фаза 7): таблица текстов чанков — до векторной сверки,
            # при несовпадении конфигурации дропается и notes_chunks_vec.
            chunks.create_table(conn)
            # Области 3.0.0 (субстрат): таблицы/FTS/индексы/vec0 областей —
            # до сверки модели/размерности: её ветка дропает area-vec и
            # пересоздаёт их под текущую размерность (_reset_area_vectors).
            _create_area_schema(conn, settings)
            # Предупреждения (№12/№32): общая таблица `warnings` + индексы —
            # субстрат, ни на что не влияет, моделей не трогает; ДО сверки
            # эмбеддинга (arch §5.4).
            _migrate_warnings_schema(conn)
            # Сид глобального шаблона навыков (lsb-0007 §3.8): сразу после
            # создания skills_meta, идемпотентно, существующее не трогаем.
            _ensure_skills_meta(conn)
            # Сид skill-создателя (lsb-0007 §3.6/§3.8): сразу после шаблона,
            # идемпотентно, с маркером в skills_meta (удалённый — не воскреснет).
            _ensure_creator_skill(conn)
            # История версий (№41, 3.2.0, substrate-versioning §5.1/§5.4): таблица
            # record_versions, baseline существующих записей и перенос архива
            # skill_versions в общую форму — идемпотентно, ПОСЛЕ сидов областей
            # (сид навыка получает baseline) и ДО сверки эмбеддинга: история —
            # про контент, векторных таблиц не адресует вовсе.
            _migrate_history_schema(conn, settings)
            # Вектора (Фаза 3 + решение 2026-08-29; Фаза 7: + вектора чанков):
            # создание при первом старте; при несовпадении зафиксированной
            # конфигурации (модель/размерность) с env — полная автореиндексация
            # обоих индексов.
            _sync_embedding_meta(conn, settings)
            # Title-индекс (lsb-0001-01, Часть B): полные вектора заметок
            # невалидны (строились по тексту без названия) — notes_vec
            # дропается, все заметки в pending; штамп meta-ключа — здесь.
            _sync_title_vectors(conn, settings, fts_migrated)
            selfheal_chunk_orphans(conn)
    except (sqlite3.Error, OSError) as exc:
        raise StorageError(
            f"не удалось инициализировать БД {settings.db_path}: {exc}"
        ) from exc


# --- неймспейсы (Фаза 10) --------------------------------------------------

DEFAULT_NAMESPACE = "default"
# Описание системного узла — модельный текст, значит EN (№34, 3.2.0). Прежний
# RU-дефолт остаётся отдельной константой: по нему одноразовая миграция
# отличает «сид» от правки оператора (см. _migrate_default_namespace_description).
DEFAULT_NAMESPACE_DESCRIPTION = "general notes not tied to any domain"
DEFAULT_NAMESPACE_DESCRIPTION_LEGACY = "общие заметки, не привязанные к доменам"

# Штамп одноразового доведения описания `default` (№34, 3.2.0): ключ в `meta`.
DEFAULT_DESCRIPTION_META_KEY = "default_description_version"
DEFAULT_DESCRIPTION_VERSION = "en-v1"


def _migrate_title_column(conn: sqlite3.Connection) -> None:
    """Нулевая миграция Фазы 11 (решение №9): notes.title TEXT (nullable).

    Свежие БД получают колонку из _NOTES_DDL; унаследованные — ALTER TABLE
    ADD COLUMN (старые заметки остаются без названия — только они могут им
    быть; новые всегда с title, контракт валидируется сервисом).
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
    if "title" not in columns:
        conn.execute("ALTER TABLE notes ADD COLUMN title TEXT")


def _migrate_comment_column(conn: sqlite3.Connection) -> None:
    """Нулевая миграция №24 (3.2.0): notes.comment TEXT (nullable).

    Свежие БД получают колонку из _NOTES_DDL; унаследованные — ALTER TABLE
    ADD COLUMN (идемпотентно, применимо с любой предыдущей версии; NULL =
    комментария нет). Лимит 250 в схеме НЕ CHECK — валидация доменная
    (CommentValidationError), как у `title`.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
    if "comment" not in columns:
        conn.execute("ALTER TABLE notes ADD COLUMN comment TEXT")


def _migrate_comment_fts(conn: sqlite3.Connection) -> None:
    """FTS-индекс пометок (no24-02, 3.2.0): таблица + оба триггера.

    Идемпотентно (IF NOT EXISTS) — повторный старт no-op. Rebuild НЕ делается:
    на момент миграции комментариев нет (колонка только что создана
    _migrate_comment_column), а последующие установки пометки идут через
    `AFTER UPDATE OF comment`. Смысловой `notes_fts` не трогается.
    """
    conn.execute(_COMMENT_FTS_DDL)
    for trigger in _COMMENT_TRIGGERS:
        conn.execute(trigger)


# Индексы list-выдач (фаза 2, пул 15; замер 50k: LIST p95 204 мс):
# покрывают ORDER BY updated_at DESC, id DESC — SQLite идёт по индексу
# (COVERING, уже отсортирован) и останавливается после LIMIT, без полного
# скана и сортировки. idx_notes_ns_deleted_updated заменяет собой
# idx_notes_namespace (prefix namespace, deleted_at — те же lookup'ы дедупа
# и счётчиков), старый индекс сносится при старте.
_INDEX_DELETED_UPDATED_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_notes_deleted_updated "
    "ON notes(deleted_at, updated_at DESC, id DESC)"
)
_INDEX_NS_DELETED_UPDATED_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_notes_ns_deleted_updated "
    "ON notes(namespace, deleted_at, updated_at DESC, id DESC)"
)


def _migrate_namespace_columns(conn: sqlite3.Connection) -> None:
    """Нулевая миграция Фазы 10 поверх живых БД: колонки notes.namespace и
    notes.classified_at. Свежие БД получают их из _NOTES_DDL; унаследованные —
    ALTER TABLE ADD COLUMN (все существующие заметки → 'default',
    classified_at NULL — причёска разберёт их позже, бриф Фазы 10, US-12).
    SQLite допускает ADD COLUMN NOT NULL только с константным DEFAULT —
    'default' им и является.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
    if "namespace" not in columns:
        conn.execute(
            "ALTER TABLE notes ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'"
        )
    if "classified_at" not in columns:
        conn.execute("ALTER TABLE notes ADD COLUMN classified_at TEXT")


def _migrate_classification_columns(conn: sqlite3.Connection) -> None:
    """Нулевая миграция Фазы 10 (Шаг 4): колонки разметки причёски
    confidence. Свежие БД получают её из _NOTES_DDL; унаследованные —
    ALTER TABLE ADD COLUMN (NULL — причёска разберёт их позже). Параметры
    разметки — внутренние данные, НЕ в MCP-контрактах (§5.7).

    lsb-0005-04: колонки domain_hint/subdomain_hint сняты со схемы (груминг
    переведён на единый hint_path — пара больше нигде не читается/пишется),
    поэтому здесь остаётся только confidence. Бэкфилл старой пары для
    унаследованных БД живёт в _migrate_hint_path_columns."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
    if "confidence" not in columns:
        conn.execute("ALTER TABLE notes ADD COLUMN confidence REAL")


def _migrate_hint_path_columns(conn: sqlite3.Connection) -> None:
    """Нулевая миграция lsb-0005-02: единый полный путь разметки hint_path.

    Заменяет пару domain_hint+subdomain_hint на один полный путь. Свежие БД
    получают колонку из _NOTES_DDL; унаследованные — ALTER TABLE ADD COLUMN
    + бэкфилл из старой пары (`X/Y` → `hint_path='X/Y'`, `X`+NULL →
    `hint_path='X'`, оба NULL → NULL). Идемпотентно: бэкфилл пишется только
    там, где `hint_path` ещё NULL, поэтому повторный запуск не перезаписывает
    уже размеченные заметки.

    Колонки domain_hint/subdomain_hint сняты со схемы (lsb-0005-04: груминг
    переведён на hint_path — пара нигде в app/ больше не читается/пишется,
    чистая зачистка), поэтому бэкфилл запускается только для СТАРЫХ БД, где
    пара ещё физически есть (проверка `domain_hint` в структуре таблицы);
    свежие БД эти колонки не создают и бэкфилл пропускают.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
    if "hint_path" not in columns:
        conn.execute("ALTER TABLE notes ADD COLUMN hint_path TEXT")
    if "domain_hint" in columns:  # старая БД с парой — бэкфилл
        conn.execute(
            """
            UPDATE notes SET hint_path = (
              CASE
                WHEN domain_hint IS NOT NULL AND subdomain_hint IS NOT NULL
                  THEN domain_hint || '/' || subdomain_hint
                WHEN domain_hint IS NOT NULL THEN domain_hint
                ELSE NULL
              END
            )
            WHERE hint_path IS NULL
              AND (domain_hint IS NOT NULL OR subdomain_hint IS NOT NULL)
            """
        )


def _migrate_promotions_key(conn: sqlite3.Connection) -> None:
    """Миграция lsb-0005-03: ключ promotions (domain, subdomain) → единый
    hint_path (идемпотентно).

    Свежие БД получают схему с hint_path PRIMARY KEY прямо из _PROMOTIONS_DDL;
    унаследованные (ключ (domain, subdomain)) пересоздаются: существующие
    вердикты мигрируются `hint_path = domain || '/' || subdomain`. Колонки
    domain/subdomain не переносятся — вердикт полностью описывается
    полным путём. Идемпотентность — по признаку старой колонки domain:
    после пересоздания её нет, повторный запуск — no-op.
    """
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(promotions)")
    }
    if "domain" not in columns:
        return  # свежая схема (ключ hint_path) или таблицы ещё нет — no-op
    with transaction(conn):
        conn.execute("ALTER TABLE promotions RENAME TO promotions_legacy")
        conn.execute(_PROMOTIONS_DDL)
        conn.execute(
            "INSERT INTO promotions (hint_path, status, canonical_path, decided_at) "
            "SELECT domain || '/' || subdomain, status, canonical_path, decided_at "
            "FROM promotions_legacy"
        )
        conn.execute("DROP TABLE promotions_legacy")


def _migrate_expiration_columns(conn: sqlite3.Connection) -> None:
    """Нулевая миграция lsb-0004-02 поверх живых БД: колонка notes.expires_at
    и таблица note_expirations. Свежие БД получают колонку из _NOTES_DDL и
    таблицу из _NOTE_EXPIRATIONS_DDL; унаследованные — ALTER TABLE ADD
    COLUMN (существующие заметки без TTL → NULL, постоянные) + CREATE TABLE
    IF NOT EXISTS (очередь удаления). Идемпотентно: повторный запуск не
    ломает (колонка уже есть — ALTER пропускается, таблица — IF NOT EXISTS)."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
    if "expires_at" not in columns:
        conn.execute("ALTER TABLE notes ADD COLUMN expires_at TEXT")
    conn.execute(_NOTE_EXPIRATIONS_DDL)


def _migrate_node_order_columns(conn: sqlite3.Connection) -> None:
    """Нулевая миграция lsb-0011-01 поверх живых БД: маркер notes.node_order_at.

    Свежие БД получают колонку из _NOTES_DDL; унаследованные — ALTER TABLE
    ADD COLUMN (node_order_at NULL у всех заметок: первый прогон джобы `nodes`
    естественно разбирает накопленный `default` — ретро-прогон). Маркер
    служебный: в выдачи MCP/REST не выходит, warning'ом не является.
    Идемпотентно: повторный запуск (рестарт сервиса) — no-op.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
    if "node_order_at" not in columns:
        conn.execute("ALTER TABLE notes ADD COLUMN node_order_at TEXT")


def _migrate_warnings_columns(conn: sqlite3.Connection) -> None:
    """Нулевая миграция №12 (3.2.0) поверх живых БД: маркер notes.warnings_at.

    Свежие БД получают колонку из `_NOTES_DDL`; унаследованные — ALTER TABLE
    ADD COLUMN (warnings_at NULL у всех заметок: первый прогон джобы `warnings`
    естественно проверяет их). Маркер — служебный (планирование обхода):
    версий не создаёт, в выдачи MCP/REST не выходит, предупреждение в заметке
    НЕ хранится (только строка в `warnings`, arch §5.3). Идемпотентно:
    повторный запуск (рестарт сервиса) — no-op.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
    if "warnings_at" not in columns:
        conn.execute("ALTER TABLE notes ADD COLUMN warnings_at TEXT")


def _migrate_warnings_schema(conn: sqlite3.Connection) -> None:
    """Нулевая миграция №12 (3.2.0): таблица `warnings` + оба индекса (§5.1).

    Общая поверхность предупреждений памяти и терминов (№12 + №32).
    CREATE TABLE/INDEX IF NOT EXISTS — идемпотентно, повторный старт no-op;
    моделей и прочих таблиц не касается (arch §5.4).
    """
    conn.execute(_WARNINGS_DDL)
    conn.execute(_WARNINGS_UNIQUE_INDEX_DDL)
    conn.execute(_WARNINGS_STATUS_INDEX_DDL)


def _migrate_link_columns(conn: sqlite3.Connection) -> None:
    """Нулевая миграция lsb-0010-02 поверх живых БД: таблица links, индекс
    idx_links_b и маркер расчёта notes.links_at.

    Свежие БД получают колонку из _NOTES_DDL и таблицу из _LINKS_DDL;
    унаследованные — ALTER TABLE ADD COLUMN (links_at NULL у всех заметок:
    первый прогон джобы `links` естественно делает backfill, arch §3.4) +
    CREATE TABLE/INDEX IF NOT EXISTS. Идемпотентно: повторный запуск — no-op.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
    if "links_at" not in columns:
        conn.execute("ALTER TABLE notes ADD COLUMN links_at TEXT")
    conn.execute(_LINKS_DDL)
    conn.execute(_LINKS_INDEX_DDL)


def _migrate_summary_length(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    """Миграция гейта 3.1.0 (2026-09-19): саммари длиннее лимита — на перегенерацию.

    Лимит длины саммари снижен до MAX_SUMMARY_CHARS (150): у живой БД уже
    сохранённые более длинные саммари надо переделать. Хранимый текст здесь
    НЕ трогается (грубое `substr` запрещено: обрезанная посередине фразы
    выжимка хуже отсутствия): заметка помечается `summary_status='pending'`,
    а новую выжимку кладёт фоновая джоба `summary` (`worker.process_summary_pending`
    — `cap_summary` держит лимит). Пока саммари перегенерируется, выдачи
    отдают fallback-усечение текста заметки (§5.5) — оно уже ≤ лимита.

    Идемпотентность: после перегенерации саммари ≤ лимита — повторный старт
    ничего не находит (no-op), штамп в meta не нужен.

    Что не трогаем:
    - заметки с `summary_status != 'ok'` — они и так в очереди суммаризации
      (повторная пометка ничего не меняет, статус не переписывается);
    - trash (`deleted_at IS NOT NULL`) — джоба `summary` его не обслуживает,
      заметка осталась бы вечно pending; саммари мусорных строк не важно.

    Признака происхождения саммари в схеме нет (модель или явная передача),
    поэтому явно вписанные длинные саммари тоже уходят на перегенерацию —
    лимит один на всё поле `summary`.
    """
    cursor = conn.execute(
        "UPDATE notes SET summary_status = 'pending' "
        "WHERE summary_status = 'ok' AND deleted_at IS NULL "
        "AND length(summary) > ?",
        (settings.max_summary_chars,),
    )
    if cursor.rowcount:
        logging.getLogger("app").warning(
            "summaries above the length limit queued for regeneration",
            extra={
                "event": "summary_regen_queued",
                "notes": cursor.rowcount,
                "reason": "summary_length",
                "max_summary_chars": settings.max_summary_chars,
            },
        )


def _ensure_skills_meta(conn: sqlite3.Connection) -> None:
    """Сид глобального instruction_template области навыков (идемпотентно).

    По образцу _ensure_default_namespace: запись создаётся только при
    отсутствии ключа. Существующее значение НЕ перезаписывается — шаблон
    правит оператор через REST (lsb-0007-05), повторный init_db (рестарт
    сервиса) его правку не затирает.
    """
    conn.execute(
        "INSERT INTO skills_meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO NOTHING",
        (INSTRUCTION_TEMPLATE_KEY, INSTRUCTION_TEMPLATE_SEED),
    )


def _ensure_creator_skill(conn: sqlite3.Connection) -> None:
    """Сид skill-создателя области навыков (lsb-0007 §3.6/§3.8), идемпотентно.

    Сид — ОДНА попытка на жизнь БД (маркер `creator_skill_seed` в
    `skills_meta`): пока маркера нет и активной записи с именем
    `CREATOR_SKILL_NAME` нет — навык вставляется; маркер пишется в любом
    случае, когда сид проходит гейт. Так повторный init_db (рестарт сервиса)
    не плодит копий, а удалённый оператором или моделью сид НЕ воскресает
    (мягко удалённая строка остаётся в trash, маркер — в skills_meta).
    """
    seeded = conn.execute(
        "SELECT 1 FROM skills_meta WHERE key = ?", (CREATOR_SKILL_SEED_KEY,)
    ).fetchone()
    if seeded is not None:
        return
    exists = conn.execute(
        "SELECT 1 FROM skills WHERE name = ? AND deleted_at IS NULL",
        (CREATOR_SKILL_NAME,),
    ).fetchone()
    if exists is None:
        conn.execute(
            "INSERT INTO skills (name, description, steps, text, extra, "
            "version, vector_status) VALUES (?, ?, ?, ?, NULL, 1, 'pending')",
            (
                CREATOR_SKILL_NAME,
                CREATOR_SKILL_DESCRIPTION,
                CREATOR_SKILL_STEPS,
                CREATOR_SKILL_TEXT,
            ),
        )
    conn.execute(
        "INSERT INTO skills_meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO NOTHING",
        (CREATOR_SKILL_SEED_KEY, CREATOR_SKILL_SEED_VALUE),
    )


# --- история версий (№41, релиз 3.2.0) --------------------------------------

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """Существует ли объект-таблица в БД (идемпотентность шагов по наличию)."""
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        is not None
    )


def _history_projection(entity: str, row: sqlite3.Row, fields: tuple[str, ...]) -> dict:
    """Проекция записи для истории (§5.3): поля как есть, `extra` — объектом.

    `extra` навыка в живой таблице — JSON-текст полей класса; в версии он живёт
    распарсенным объектом (одна форма у baseline и дельт), пусто — `None`.
    """
    projection: dict = {}
    for field in fields:
        value = row[field]
        if entity == "skill" and field == "extra":
            value = None if not value else json.loads(value)
        projection[field] = value
    return projection


def _insert_baseline(
    conn: sqlite3.Connection, entity: str, entity_id: int, projection: dict
) -> None:
    """`INSERT … WHERE NOT EXISTS`: baseline только там, где истории ещё нет.

    Само-лечаще: строка, появившаяся после baseline-миграции, добирает baseline
    при следующем старте; повторный проход — no-op (§5.4).
    """
    conn.execute(
        "INSERT INTO record_versions (entity, entity_id, version, event, source, "
        "changed, snapshot) SELECT ?, ?, 1, 'baseline', 'migration', '[]', ? "
        "WHERE NOT EXISTS (SELECT 1 FROM record_versions WHERE entity = ? "
        "AND entity_id = ?)",
        (entity, entity_id, json.dumps(projection, ensure_ascii=False), entity, entity_id),
    )


def _backfill_history_baseline(conn: sqlite3.Connection) -> None:
    """Baseline всех существующих записей: `version=1` = текущее содержимое.

    Проекция собирается в Python (`json.dumps`) — без зависимости от SQLite
    json1; проход по каждой сущности отдельный, `WHERE NOT EXISTS` держит
    идемпотентность (FR-8.1/FR-8.2, §5.4). Записи в trash тоже получают
    baseline: история — операторская, мягкое удаление её не отменяет.
    """
    from app.services.history import (
        CONTENT_FIELDS,
        ENTITY_ASSISTANT_LOG,
        ENTITY_NOTE,
        ENTITY_SKILL,
        ENTITY_SKILL_TEMPLATE,
        ENTITY_TERM,
        ENTITY_USER_FACT,
    )

    entity_tables = {
        ENTITY_NOTE: "notes",
        ENTITY_SKILL: "skills",
        ENTITY_TERM: "terms",
        ENTITY_USER_FACT: "user_facts",
        ENTITY_ASSISTANT_LOG: ASSISTANT_LOGS_TABLE,
    }
    for entity, table in entity_tables.items():
        fields = CONTENT_FIELDS[entity]
        columns = ", ".join(("id", *fields))
        for row in conn.execute(f"SELECT {columns} FROM {table}").fetchall():
            _insert_baseline(
                conn,
                entity,
                int(row["id"]),
                _history_projection(entity, row, fields),
            )
    # Глобальный шаблон навыков — отдельная сущность истории (id 0, ключ
    # skills_meta.instruction_template, §5.3).
    template = conn.execute(
        "SELECT value FROM skills_meta WHERE key = ?", (INSTRUCTION_TEMPLATE_KEY,)
    ).fetchone()
    if template is not None:
        _insert_baseline(
            conn, ENTITY_SKILL_TEMPLATE, 0, {"instruction_template": template["value"]}
        )


def _merge_skill_versions(conn: sqlite3.Connection, settings: Settings) -> None:
    """Перенести архив `skill_versions` в общую форму истории (§5.4, §13.7).

    Цепочка `1..N`: v1 — самая ранняя архивная копия целиком (`snapshot`,
    `event='baseline'`, `source='migration'`), промежуточные — дельты,
    посчитанные из архивных копий на месте переноса, vN — текущая строка
    `skills` (дельта от предыдущей). Номера — из `skill_versions.version` и
    `skills.version`; недостающие звенья не выдумываются — дельта считается
    между соседними ИМЕЮЩИМИСЯ версиями. Перенос идёт одной транзакцией
    вызывающего; таблица `skill_versions` остаётся legacy-слепком (§13.7) и
    кодом больше не читается. Навык, у которого история уже есть, пропускается:
    шаг идемпотентен; таблицы нет — no-op.
    """
    from app.services.history import CONTENT_FIELDS, ENTITY_SKILL, encode_delta

    if not _table_exists(conn, "skill_versions"):
        return
    fields = CONTENT_FIELDS[ENTITY_SKILL]
    skill_ids = [
        int(row[0])
        for row in conn.execute("SELECT DISTINCT skill_id FROM skill_versions")
    ]
    for skill_id in skill_ids:
        has_history = conn.execute(
            "SELECT 1 FROM record_versions WHERE entity = ? AND entity_id = ? LIMIT 1",
            (ENTITY_SKILL, skill_id),
        ).fetchone()
        if has_history is not None:
            continue
        chain: dict[int, dict] = {}
        for row in conn.execute(
            "SELECT * FROM skill_versions WHERE skill_id = ? ORDER BY version",
            (skill_id,),
        ).fetchall():
            chain[int(row["version"])] = _history_projection(ENTITY_SKILL, row, fields)
        live = conn.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()
        if live is not None:
            chain[int(live["version"])] = _history_projection(ENTITY_SKILL, live, fields)
        if not chain:
            continue
        previous: dict | None = None
        for version in sorted(chain):
            current = chain[version]
            if previous is None:
                conn.execute(
                    "INSERT INTO record_versions (entity, entity_id, version, event, "
                    "source, changed, snapshot) VALUES (?, ?, ?, 'baseline', "
                    "'migration', '[]', ?)",
                    (
                        ENTITY_SKILL,
                        skill_id,
                        version,
                        json.dumps(current, ensure_ascii=False),
                    ),
                )
            else:
                delta, changed = encode_delta(
                    previous, current, inline_max=settings.history_delta_inline_max
                )
                conn.execute(
                    "INSERT INTO record_versions (entity, entity_id, version, event, "
                    "source, changed, delta) VALUES (?, ?, ?, 'content', "
                    "'migration', ?, ?)",
                    (
                        ENTITY_SKILL,
                        skill_id,
                        version,
                        json.dumps(changed, ensure_ascii=False),
                        json.dumps(delta, ensure_ascii=False),
                    ),
                )
            previous = current


def _migrate_history_schema(conn: sqlite3.Connection, settings: Settings) -> None:
    """Схема истории + baseline существующих записей + перенос skill_versions.

    Шаги постановки no41-01: (а) `CREATE TABLE IF NOT EXISTS record_versions`
    (§5.1 — точно как в архитектуре, без индексов сверх PK); (б) перенос архива
    навыков в общую форму; (в) baseline всех существующих записей. Порядок
    «перенос → бэкфилл» обязателен: у навыка с архивом baseline — самая ранняя
    архивная копия, а не текущая строка (иначе v1 был бы перезаписан текущим
    содержимым и цепочка архива стала бы недостижима). Всё одной транзакцией,
    идемпотентно и применимо с любой предыдущей версии.
    """
    with transaction(conn):
        conn.execute(_RECORD_VERSIONS_DDL)
        _merge_skill_versions(conn, settings)
        _backfill_history_baseline(conn)


def _ensure_default_namespace(conn: sqlite3.Connection) -> None:
    """Узел 'default' существует всегда (идемпотентно): описание — дефолт,
    оператор может править его через REST (Шаг 6)."""
    conn.execute(
        "INSERT INTO namespaces (path, description, status) VALUES (?, ?, 'confirmed') "
        "ON CONFLICT(path) DO NOTHING",
        (DEFAULT_NAMESPACE, DEFAULT_NAMESPACE_DESCRIPTION),
    )


def _sync_namespace_partition(conn: sqlite3.Connection, settings: Settings) -> None:
    """Миграция vec-индексов под партицию namespace (Фаза 10).

    У БД, созданных до Фазы 10, notes_vec/notes_chunks_vec без колонки `+ns`:
    ОБЕ таблицы дропаются и пересоздаются с partition key, все заметки
    (включая trash) уходят в vector_status='pending' — фоновый воркер
    пере-кодирует (NFR-3: данные не теряются, поиск деградирует к FTS).
    Свежие БД пропускаются: их таблицы создаются в _sync_embedding_meta уже
    с партицией. Вызывается ДО _sync_embedding_meta (см. init_db).
    """
    if vectors.has_partition(conn) and chunks.has_partition(conn):
        return
    has_tables = (
        vectors.existing_vec_dim(conn) is not None
        or chunks.existing_vec_dim(conn) is not None
    )
    if not has_tables:
        return  # свежая БД — таблицы создадутся ниже уже с партицией
    notes_count = int(conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0])
    logging.getLogger("app").warning(
        "namespace partition introduced: rebuilding vector indexes",
        extra={
            "event": "reindex_started",
            "notes": notes_count,
            "reason": "namespace_partition",
        },
    )
    conn.execute("DROP TABLE IF EXISTS notes_vec")
    conn.execute("DROP TABLE IF EXISTS notes_chunks_vec")
    vectors.create_vec_table(conn, settings.embedding_dim)
    chunks.create_vec_table(conn, settings.embedding_dim)
    conn.execute("UPDATE notes SET vector_status = 'pending'")
    logging.getLogger("app").info(
        "vector indexes rebuilt with namespace partition",
        extra={
            "event": "reindex_done",
            "notes_pending": notes_count,
            "reason": "namespace_partition",
        },
    )


def _get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row[0])


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _migrate_default_namespace_description(conn: sqlite3.Connection) -> None:
    """Довести описание узла `default` до EN (№34, 3.2.0): одноразово.

    Свежая БД получает EN-текст из `_ensure_default_namespace` — для неё
    миграция no-op (текст не равен прежнему RU-дефолту), но штамп всё равно
    ставится. Унаследованная БД правится ТОЛЬКО если описание точно равно
    прежнему русскому дефолту (DEFAULT_NAMESPACE_DESCRIPTION_LEGACY): любая
    иная, в том числе русская, правка оператора сохраняется — «сид» и
    «правка» различаются сравнением текста, новой колонки не заводится.

    Одноразовость даёт штамп DEFAULT_DESCRIPTION_META_KEY в `meta`: он
    ставится НЕЗАВИСИМО от исхода (обновлена строка или нет), поэтому
    повторный `init_db` ничего не ищет и ничего не меняет. Вызывается в
    `init_db` сразу после `_ensure_default_namespace`.
    """
    conn.execute(_META_DDL)  # meta может ещё не существовать у древних БД —
    # штатно таблица создаётся позже, в _sync_embedding_meta (см. init_db)
    if _get_meta(conn, DEFAULT_DESCRIPTION_META_KEY) is not None:
        return  # доведено ранее — идемпотентно
    with transaction(conn):
        cursor = conn.execute(
            "UPDATE namespaces SET description = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
            "WHERE path = ? AND description = ?",
            (
                DEFAULT_NAMESPACE_DESCRIPTION,
                DEFAULT_NAMESPACE,
                DEFAULT_NAMESPACE_DESCRIPTION_LEGACY,
            ),
        )
        if cursor.rowcount == 1:
            logging.getLogger("app").info(
                "default namespace description migrated to english",
                extra={
                    "event": "default_description_migrated",
                    "path": DEFAULT_NAMESPACE,
                    "old": DEFAULT_NAMESPACE_DESCRIPTION_LEGACY,
                    "new": DEFAULT_NAMESPACE_DESCRIPTION,
                },
            )
        _set_meta(conn, DEFAULT_DESCRIPTION_META_KEY, DEFAULT_DESCRIPTION_VERSION)


# --- title-индекс (lsb-0001-01) ---------------------------------------------

# Штамп миграции в meta: «2» — notes_fts индексирует title и text (FR-1.2/
# FR-1.3); отсутствие ключа = БД, не прошедшая title-миграцию (FTS только
# по text). Общий для Части A (FTS) и Части B (вектора).
_TITLE_INDEX_META_KEY = "title_index_version"
_TITLE_INDEX_VERSION = "2"


def _sync_fts_title_index(conn: sqlite3.Connection) -> bool:
    """Часть A миграции title-индекса (lsb-0001-01): notes_fts под (title, text).

    Живые БД до lsb-0001-01 имеют notes_fts с одной колонкой text: таблица
    дропается и пересоздаётся по _FTS_DDL, индекс переливается из notes
    (rowid, title, text — title может быть NULL, FTS5 трактует его как
    пустую строку), триггеры пересоздаются под новые колонки. Решение
    гейта R4 — миграция при старте, не джоба; идемпотентность — общий
    с Частью B meta-ключ _TITLE_INDEX_META_KEY (штамп ставит
    _sync_title_vectors в конце init_db).

    Условие запуска: ключа нет И DDL notes_fts в sqlite_master без title
    (регексп _FTS_TITLE_RE). Ключа нет, но DDL уже с title (свежая БД) —
    no-op. Вызывается после миграций колонок: к моменту вызова notes.title
    уже существует (перелив читает её) и ДО _check_fts_integrity — перелив
    оставляет индекс консистентным.

    Возвращает True, если в этом запуске БД была легаси и перестроена, —
    Часть B (_sync_title_vectors) инвалидирует полные вектора только вслед
    за фактической перестройкой FTS.
    """
    conn.execute(_META_DDL)  # meta может ещё не существовать у древних БД —
    # штатно таблица создаётся позже, в _sync_embedding_meta (см. init_db)
    if _get_meta(conn, _TITLE_INDEX_META_KEY) is not None:
        return False  # мигрировано ранее — штамп ставит _sync_title_vectors
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'notes_fts'"
    ).fetchone()
    if row is None or row[0] is None or _FTS_TITLE_RE.search(row[0]):
        return False  # свежая БД: notes_fts создана выше уже с колонкой title
    notes_count = int(conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0])
    logging.getLogger("app").warning(
        "note title added to FTS index: rebuilding notes_fts",
        extra={
            "event": "reindex_started",
            "notes": notes_count,
            "reason": "title_index",
        },
    )
    conn.execute("DROP TRIGGER IF EXISTS notes_fts_ai")
    conn.execute("DROP TRIGGER IF EXISTS notes_fts_au")
    conn.execute("DROP TABLE IF EXISTS notes_fts")
    conn.execute(_FTS_DDL)
    conn.execute(
        "INSERT INTO notes_fts(rowid, title, text) SELECT id, title, text FROM notes"
    )
    for trigger in _TRIGGERS:
        conn.execute(trigger)
    logging.getLogger("app").info(
        "notes_fts rebuilt with title column",
        extra={
            "event": "reindex_done",
            "notes": notes_count,
            "reason": "title_index",
        },
    )
    return True


def _sync_title_vectors(
    conn: sqlite3.Connection, settings: Settings, fts_migrated: bool
) -> None:
    """Часть B миграции title-индекса (lsb-0001-01): полные вектора.

    Полные вектора (notes_vec) считались по тексту БЕЗ названия — после
    ввода title в FTS (и в кодирование, шаг 2) они невалидны: notes_vec
    дропается, все заметки (включая trash) уходят в vector_status='pending',
    вектора в этом шаге не кодируются — догоняет фоновый воркер.
    notes_chunks_vec НЕ трогается (решение гейта R4): вектора чанков
    строятся по текстам чанков, название на них не влияет.

    Условие: ключа нет (общий с Частью A) И в этом же запуске была
    перестроена FTS (`fts_migrated` от _sync_fts_title_index): обе части
    миграции двигаются вместе — штамп в meta не спасает от ручной потери
    meta целиком, а дропать валидные вектора из-за одной только потери
    штампа не нужно (без истории в meta вектора считаются актуальными —
    как в нулевой миграции _sync_embedding_meta). Штамп ключа — в конце,
    всегда (в т.ч. свежей БД с нулём заметок — пере-кодировать нечего).

    Вызывается после _sync_embedding_meta: её ветки (смена модели/чанков)
    успевают пересоздать таблицы текущей конфигурации — дроп здесь их уже
    не ломает; ДО selfheal_chunk_orphans.
    """
    if _get_meta(conn, _TITLE_INDEX_META_KEY) is not None:
        return
    notes_count = int(conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0])
    if fts_migrated and notes_count:
        logging.getLogger("app").warning(
            "title index migration: full-text vectors invalidated",
            extra={
                "event": "reindex_started",
                "notes": notes_count,
                "reason": "title_index",
            },
        )
        conn.execute("DROP TABLE IF EXISTS notes_vec")
        vectors.create_vec_table(conn, settings.embedding_dim)
        # Полные вектора невалидны — все заметки (включая trash) в очередь.
        conn.execute("UPDATE notes SET vector_status = 'pending'")
        logging.getLogger("app").info(
            "notes_vec rebuilt; background worker will re-encode all notes",
            extra={
                "event": "reindex_done",
                "notes_pending": notes_count,
                "reason": "title_index",
            },
        )
    _set_meta(conn, _TITLE_INDEX_META_KEY, _TITLE_INDEX_VERSION)


# Описания чанк-параметров (Фаза 7, brief §4): фиксируются в meta —
# смена любого из них поверх живой БД означает пере-чанковку (шаг 6).
_CHUNK_META_KEYS = ("chunk_size", "chunk_overlap", "chunk_min_target")


def _set_chunk_meta_defaults(conn: sqlite3.Connection, settings: Settings) -> None:
    """Зафиксировать чанк-параметры в meta (нулевая миграция: отсутствующий
    ключ — из env, без пере-чанковки; сравнение с env и пере-чанковка при
    смене значения — шаг 6 Фазы 7)."""
    for key in _CHUNK_META_KEYS:
        if _get_meta(conn, key) is None:
            _set_meta(conn, key, str(getattr(settings, key)))


def _create_chunk_vec_if_missing(conn: sqlite3.Connection, dim: int) -> None:
    """notes_chunks_vec появилась в Фазе 7 поверх живых БД — создать, если
    её ещё нет (не реиндексируя заметки — их вектора и так текущие)."""
    if chunks.existing_vec_dim(conn) is None:
        chunks.create_vec_table(conn, dim)


def selfheal_chunk_orphans(conn: sqlite3.Connection) -> None:
    """Вычистить сироты чанков при старте (Фаза 7, как FTS-integrity).

    Операторский sqlite3-CLI без PRAGMA foreign_keys=ON не каскадирует
    физическое удаление заметки в чанки, а vec0 FK не поддерживает вовсе:
    чанки без заметки и вектора без чанка убираются самолечением, событие в
    лог — только если чистить было что."""
    dead_chunks, dead_vectors = chunks.clean_orphans(conn)
    if dead_chunks or dead_vectors:
        logging.getLogger("app").warning(
            "cleaned chunk orphans left by direct DB edits",
            extra={
                "event": "chunks_orphans_cleaned",
                "chunks": dead_chunks,
                "chunk_vectors": dead_vectors,
            },
        )


def _sync_embedding_meta(conn: sqlite3.Connection, settings: Settings) -> None:
    """Сверить конфигурацию векторизации/чанков с записями в meta; разошлись —
    автореиндексация при старте.

    Две разные причины, одна механика (b972386, решение О. 2026-08-29):
    - смена модели/размерности/провайдера (env vs meta embedding_model/
      embedding_dim/embedding_provider, Фаза 11 — провайдер): ОБА векторных
      индекса пересоздаются под текущую размерность, все заметки (включая
      trash — их вектора тоже невалидны) уходят в pending, воркер
      пере-кодирует полные вектора;
    - смена чанк-параметров (env vs meta chunk_*, Фаза 7, brief §6): вектора
      полного текста остаются валидными (текст не менялся) — дропается только
      notes_chunks_vec; тексты чанков ПЕРЕСЧИТЫВАЮТСЯ у всех заметок
      (включая trash) по текущим параметрам сплиттера, воркер до-векторизует.
    Оба изменения сразу — работает по совокупности: один reindex_started/done,
    пересоздание обоих индексов, все заметки pending + пере-чанковка.

    Reuse при пере-чанковке: у заметки с ровно одним чанком ≤ CHUNK_SIZE
    вектор чанка = вектор полного текста из notes_vec без кодирования —
    вектор не зависит от чанк-параметров, а при сменившейся модели notes_vec
    уже пуст (дропнут до пере-чанковки) — reuse сам не сработает там, где
    полный вектор невалиден.

    Наследие: у БД без meta (созданных до этой правки) запись создаётся из
    текущего env без реиндексации — нулевая миграция; их вектора и так
    были построены той же моделью (иначе оператор потратил бы reindex.py).
    Чанк-параметры в meta пишутся аналогично — только отсутствующие ключи
    (отсутствие = «не менялись», сравнение с env только по существующим).
    Провайдер эмбеддинга (Фаза 11) — тот же паттерн: у БД без ключа
    embedding_provider (до Фазы 11) запись из env без реиндексации; смена
    провайдера при том же имени модели — тоже реиндекс (вектора другой
    провайдерской модели несовместимы).
    """
    conn.execute(_META_DDL)
    stored = _get_meta(conn, "embedding_model")
    stored_dim_raw = _get_meta(conn, "embedding_dim")
    existing_dim = vectors.existing_vec_dim(conn)
    if stored is None or stored_dim_raw is None:
        # Свежая БД (нет таблицы) или унаследованная (не трогаем, см. docstring).
        if existing_dim is None:
            vectors.create_vec_table(conn, settings.embedding_dim)
        _create_chunk_vec_if_missing(conn, settings.embedding_dim)
        _create_area_vec_if_missing(conn, settings.embedding_dim)
        _set_chunk_meta_defaults(conn, settings)
        _set_meta(conn, "embedding_model", settings.embedding_model)
        _set_meta(conn, "embedding_dim", str(settings.embedding_dim))
        _set_meta(conn, "embedding_provider", settings.embedding_provider)
        return
    stored_dim = int(stored_dim_raw)
    # Провайдер эмбеддинга (Фаза 11): у БД без ключа (до Фазы 11) — нулевая
    # миграция по паттерну chunk-ключей: запись из env БЕЗ реиндексации.
    stored_provider = _get_meta(conn, "embedding_provider")
    if stored_provider is None:
        _set_meta(conn, "embedding_provider", settings.embedding_provider)
        stored_provider = settings.embedding_provider
    model_changed = (
        stored != settings.embedding_model
        or stored_dim != settings.embedding_dim
        or stored_provider != settings.embedding_provider
    )
    # Чанк-параметры: сравниваем ТОЛЬКО зафиксированные ранее ключи
    # (отсутствующий ключ — нулевая миграция, «не менялись», пишется ниже).
    chunk_changes = {
        key: (str(stored_value), str(getattr(settings, key)))
        for key in _CHUNK_META_KEYS
        if (stored_value := _get_meta(conn, key)) is not None
        and stored_value != str(getattr(settings, key))
    }
    if not model_changed and not chunk_changes:
        # Совпало: но notes_chunks_vec могла ещё не существовать на живой БД
        # (Фаза 7 поверх Фазы 5) — создать при отсутствии. Area-vec — та же
        # логика (наследники до 3.0.0: у них есть meta, но нет областей).
        _create_chunk_vec_if_missing(conn, settings.embedding_dim)
        _create_area_vec_if_missing(conn, settings.embedding_dim)
        _set_chunk_meta_defaults(conn, settings)
        return
    notes_count = int(
        conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    )
    started_extra: dict[str, object] = {
        "event": "reindex_started",
        "notes": notes_count,
    }
    message = "chunk parameters changed: re-chunking all notes"
    if model_changed:
        message = "embedding provider/model/dim changed: rebuilding vector index"
        started_extra.update(
            from_model=stored,
            to_model=settings.embedding_model,
            from_dim=stored_dim,
            to_dim=settings.embedding_dim,
            from_provider=stored_provider,
            to_provider=settings.embedding_provider,
        )
    for key, (old_value, new_value) in chunk_changes.items():
        started_extra[f"from_{key}"] = old_value
        started_extra[f"to_{key}"] = new_value
    logging.getLogger("app").warning(message, extra=started_extra)
    if model_changed:
        conn.execute("DROP TABLE IF EXISTS notes_vec")
        vectors.create_vec_table(conn, settings.embedding_dim)
        # Вектора полных текстов невалидны — все заметки (вкл. trash) в очередь.
        conn.execute("UPDATE notes SET vector_status = 'pending'")
        # Связи уровня 1 (lsb-0010-02, arch §3.4): косинус считается по этим
        # векторам, «значимые слова»/упоминания — по тексту, но связи другой
        # модели невалидны целиком — таблица очищается, маркеры сброшены
        # (расчёт догонит воркер после до-векторизации).
        conn.execute("UPDATE notes SET links_at = NULL")
        conn.execute("DELETE FROM links")
        # Области 3.0.0: вектора другой модели несовместимы — area-vec
        # дропаются и пересоздаются, все записи областей → pending.
        _reset_area_vectors(conn, settings)
    # Вектора чанков невалидны при любой из причин (другая модель — другая
    # векторизация; другие параметры — другие чанки, их вектора больше не
    # соответствуют их же новым текстам). Дроп: все чанки → pending
    # (анти-джойн), воркер пере-кодирует.
    conn.execute("DROP TABLE IF EXISTS notes_chunks_vec")
    chunks.create_vec_table(conn, settings.embedding_dim)
    # Пере-чанковка + записи meta — одним атомарным блоком: rollback вернёт
    # и чанки, и meta предыдущей конфигурации, если что-то пойдёт не так.
    with transaction(conn):
        rechunked, reused = _rechunk_all_notes(conn, settings)
        if model_changed:
            _set_meta(conn, "embedding_model", settings.embedding_model)
            _set_meta(conn, "embedding_dim", str(settings.embedding_dim))
            _set_meta(conn, "embedding_provider", settings.embedding_provider)
        for key in _CHUNK_META_KEYS:
            _set_meta(conn, key, str(getattr(settings, key)))
    pending_chunks = int(chunks.count_pending(conn))
    done_extra: dict[str, object] = {
        "event": "reindex_done",
        "notes_rechunked": rechunked,
        "chunk_vectors_reused": reused,
        "pending_chunks": pending_chunks,
    }
    if model_changed:
        done_extra["pending_vector"] = notes_count
        done_extra["pending_areas"] = _count_area_records(conn)
    message_done = (
        "vector index rebuilt; background worker will re-encode all notes"
        if model_changed
        else "notes re-chunked; chunk vectors pending for background worker"
    )
    logging.getLogger("app").info(message_done, extra=done_extra)


def _rechunk_all_notes(
    conn: sqlite3.Connection, settings: Settings
) -> tuple[int, int]:
    """Пере-чанковать все заметки (включая trash) по текущим параметрам.

    Миграционная операция (вызывается только из _sync_embedding_meta при
    сменившихся модели/размерности или чанк-параметрах — brief §6): тексты
    пересчитываются чистым сплиттером, чанки и их вектора заменяются
    (replace_note_chunks чистит vec-строки явно). Импорт сплиттера —
    локальный: storage не тянет домен на импорте, пере-чанковка здесь —
    как некогда реиндекс в init_db (b972386), one-shot на старте сервиса.

    Reuse: заметка с ровно одним чанком ≤ CHUNK_SIZE получает вектор чанка
    = вектор полного текста из notes_vec — без кодирования; при сменившейся
    модели notes_vec к этому моменту пуст — reuse сам не срабатывает.

    Возвращает (пере-чанковано заметок, reuse-векторов записано).
    """
    from app.services.splitter import split_text  # локально, см. докстринг

    rows = conn.execute("SELECT id, text FROM notes ORDER BY id").fetchall()
    rechunked = reused = 0
    for row in rows:
        spans = split_text(
            row["text"],
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            chunk_min_target=settings.chunk_min_target,
        )
        chunk_ids = chunks.replace_note_chunks(
            conn, row["id"], [(chunk.text, chunk.tokens) for chunk in spans]
        )
        rechunked += 1
        if len(chunk_ids) == 1 and spans[0].tokens <= settings.chunk_size:
            note_vector = vectors.get_vector(conn, row["id"])
            if note_vector is not None:
                chunks.upsert_vector(conn, chunk_ids[0], note_vector)
                reused += 1
    return rechunked, reused


def _check_fts_integrity(conn: sqlite3.Connection) -> None:
    """Сверить FTS-индекс с notes; рассинхрон (оператор правил `notes.text`
    напрямую, мимо триггеров) лечится rebuild'ом — индекс полностью
    выводим из (title, text) заметок, данные не теряются."""
    try:
        conn.execute("INSERT INTO notes_fts(notes_fts) VALUES('integrity-check')")
        return
    except sqlite3.DatabaseError:
        pass  # рассинхрон/повреждение — самолечение ниже
    # rebuild: обнулить индекс (спец-команда 'delete' со всеми текущими
    # значениями) и залить заново; нечувствительно к прошлому состоянию.
    conn.execute(
        "INSERT INTO notes_fts(notes_fts, rowid, title, text) "
        "SELECT 'delete', id, title, text FROM notes"
    )
    conn.execute(
        "INSERT INTO notes_fts(rowid, title, text) SELECT id, title, text FROM notes"
    )


def delete_note_physical(conn: sqlite3.Connection, note_id: int) -> None:
    """Физически удалить заметку из всех индексов (lsb-0004-02, джоба зачистки).

    Полное физическое удаление (НЕ soft delete): просроченная заметка должна
    исчезнуть из всех индексов — notes + notes_chunks + notes_chunks_vec +
    notes_vec + notes_fts + note_expirations + links. Композиция существующих
    примитивов слоя (chunks.drop_note_chunks, vectors.drop, expirations.delete)
    + явный DELETE из notes_fts (внешний контент: DELETE-триггера нет, индекс
    не чистится каскадом) и notes.

    Связи (3.2.0, no46-03): `links` заметки сносятся здесь же, в ТОЙ ЖЕ
    транзакции вызывающего (`DELETE FROM links WHERE note_a = ? OR note_b = ?` —
    обе стороны пары, идемпотентно). Фоновый свип `purge_orphans` (idle-ветка
    джобы `links`) остаётся страховкой, а не единственной чисталкой. Soft
    delete связи НЕ трогает: заметка восстанавливаема — её строки `links`
    живут (arch §3.2).

    История (lsbdef-0014): физическое удаление уносит и версии заметки из
    `record_versions` — `notes.id` без AUTOINCREMENT переиспользуется, и
    оставшийся baseline переиспользованного id ронял `record_initial` новой
    заметки на UNIQUE `(entity, entity_id, version)` → падал `memory_save`.
    Soft delete историю СОХРАНЯЕТ (заметка в trash восстановима, версии
    остаются) — не путать.

    Идемпотентно и безопасно: отсутствующей заметки нет — каждый DELETE
    просто не матчит ничего; повторный вызов не падает. Вызывается внутри
    открытой транзакции вызывающего (transaction()).
    """
    # Чанки и их вектора (notes_chunks_vec + notes_chunks) — до notes:
    # notes_chunks_vec без FK, каскад от notes его не тронет.
    chunks.drop_note_chunks(conn, note_id)
    vectors.drop(conn, note_id)  # notes_vec
    conn.execute("DELETE FROM notes_fts WHERE rowid = ?", (note_id,))
    conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    expirations.delete(conn, note_id)  # note_expirations
    # 3.2.0 (no46-03): связи заметки уходят вместе с ней — обе стороны пары,
    # в той же транзакции (идемпотентно: повторный вызов просто не матчит).
    conn.execute(
        "DELETE FROM links WHERE note_a = ? OR note_b = ?", (note_id, note_id)
    )
    # История заметки (lsbdef-0014): версии уходят вместе с заметкой, в той же
    # транзакции — иначе освобождённый `notes.id` после переиспользования
    # наследовал чужой baseline и `memory_save` падал на UNIQUE. Только
    # ФИЗИЧЕСКОЕ удаление чистит историю; soft delete её сохраняет
    # (заметка восстановима).
    conn.execute(
        "DELETE FROM record_versions WHERE entity = 'note' AND entity_id = ?",
        (note_id,),
    )