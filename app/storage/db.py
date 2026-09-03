"""Хранилище (ARCHITECTURE §3.3): SQLite-схема notes + FTS5 (trigram) + vec0.

Слой без доменных правил: только схема, соединения и транзакции. Семантика
CRUD (валидации, статусы, soft delete) — в `app.services.notes`; векторные
операции (сериализация vec0, KNN) — в `app.storage.vectors` (Фаза 3).

Ключевые решения:
- Схема создаётся идемпотентно при старте (`init_db`, `IF NOT EXISTS`).
- `notes_fts` — FTS5 внешнего контента (`content='notes'`,
  `content_rowid='id'`, `tokenize='trigram'`), синхронизируется триггерами
  AFTER INSERT / AFTER UPDATE OF text. DELETE-триггер не нужен: удаление —
  soft (`deleted_at`), строка и FTS-индекс физически остаются в trash.
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
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec

from app.config import Settings
from app.storage import chunks, vectors

# Сколько ждать блокировку записи, прежде чем сдаться (как timeout sqlite3,
# так и PRAGMA busy_timeout).
BUSY_TIMEOUT_MS = 5000

# Как выцупоть лимит CHECK(length(text) BETWEEN 1 AND ?) из DDL живой таблицы
# notes (сверка с MAX_NOTE_CHARS при старте — см. init_db).
_CHECK_LIMIT_RE = re.compile(r"length\(\s*text\s*\)\s+BETWEEN\s+1\s+AND\s+(\d+)")


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
  summary        TEXT    NOT NULL DEFAULT '',
  author         TEXT    NOT NULL DEFAULT 'unknown',
  vector_status  TEXT    NOT NULL DEFAULT 'pending',
  summary_status TEXT    NOT NULL DEFAULT 'pending',
  created_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  deleted_at     TEXT    NULL,
  namespace      TEXT    NOT NULL DEFAULT 'default',
  classified_at  TEXT    NULL
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

_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
  text, content='notes', content_rowid='id', tokenize='trigram'
)
"""

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
# UPDATE OF text: изменения прочих колонок (summary, статусы) FTS не трогают.
_TRIGGERS = (
    """
    CREATE TRIGGER IF NOT EXISTS notes_fts_ai AFTER INSERT ON notes BEGIN
      INSERT INTO notes_fts(rowid, text) VALUES (new.id, new.text);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS notes_fts_au AFTER UPDATE OF text ON notes BEGIN
      INSERT INTO notes_fts(notes_fts, rowid, text) VALUES ('delete', old.id, old.text);
      INSERT INTO notes_fts(rowid, text) VALUES (new.id, new.text);
    END
    """,
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
            _check_fts_integrity(conn)
            # Неймспейсы (Фаза 10): колонки notes + реестр + дефолт-узел.
            _migrate_namespace_columns(conn)
            conn.execute(_NAMESPACES_DDL)
            conn.execute(_INDEX_NAMESPACE_DDL)
            _ensure_default_namespace(conn)
            # Партиция namespace (Фаза 10): vec-таблицы живых БД без `+ns`
            # пересоздаются с партицией, все заметки — в pending. ДО сверки
            # модели/чанков: обе ветки создают таблицы уже с партицией.
            _sync_namespace_partition(conn, settings)
            # Чанки (Фаза 7): таблица текстов чанков — до векторной сверки,
            # при несовпадении конфигурации дропается и notes_chunks_vec.
            chunks.create_table(conn)
            # Вектора (Фаза 3 + решение 2026-08-29; Фаза 7: + вектора чанков):
            # создание при первом старте; при несовпадении зафиксированной
            # конфигурации (модель/размерность) с env — полная автореиндексация
            # обоих индексов.
            _sync_embedding_meta(conn, settings)
            selfheal_chunk_orphans(conn)
    except (sqlite3.Error, OSError) as exc:
        raise StorageError(
            f"не удалось инициализировать БД {settings.db_path}: {exc}"
        ) from exc


# --- неймспейсы (Фаза 10) --------------------------------------------------

DEFAULT_NAMESPACE = "default"
DEFAULT_NAMESPACE_DESCRIPTION = "общие заметки, не привязанные к доменам"

_INDEX_NAMESPACE_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_notes_namespace "
    "ON notes(namespace, deleted_at)"
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
    - смена модели/размерности (env vs meta embedding_model/embedding_dim):
      ОБА векторных индекса пересоздаются под текущую размерность, все заметки
      (включая trash — их вектора тоже невалидны) уходят в pending, воркер
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
        _set_chunk_meta_defaults(conn, settings)
        _set_meta(conn, "embedding_model", settings.embedding_model)
        _set_meta(conn, "embedding_dim", str(settings.embedding_dim))
        return
    stored_dim = int(stored_dim_raw)
    model_changed = (
        stored != settings.embedding_model or stored_dim != settings.embedding_dim
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
        # (Фаза 7 поверх Фазы 5) — создать при отсутствии.
        _create_chunk_vec_if_missing(conn, settings.embedding_dim)
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
        message = "embedding model/dim changed: rebuilding vector index"
        started_extra.update(
            from_model=stored,
            to_model=settings.embedding_model,
            from_dim=stored_dim,
            to_dim=settings.embedding_dim,
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
    выводим из text, данные не теряются."""
    try:
        conn.execute("INSERT INTO notes_fts(notes_fts) VALUES('integrity-check')")
        return
    except sqlite3.DatabaseError:
        pass  # рассинхрон/повреждение — самолечение ниже
    # rebuild: обнулить индекс (спец-команда 'delete' со всеми текущими
    # значениями) и залить заново; нечувствительно к прошлому состоянию.
    conn.execute(
        "INSERT INTO notes_fts(notes_fts, text, rowid) "
        "SELECT 'delete', text, id FROM notes"
    )
    conn.execute("INSERT INTO notes_fts(rowid, text) SELECT id, text FROM notes")