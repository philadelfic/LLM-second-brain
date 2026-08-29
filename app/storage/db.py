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
from app.storage import vectors

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
  deleted_at     TEXT    NULL
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
    автоматически перестраивается векторный индекс: солёные вектора другой
    модели несовместимы, все заметки уходят в pending и догоняются воркером
    (NFR-3 — данные не теряются, поиск деградирует к FTS до готовности).

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
            # Вектора (Фаза 3 + решение 2026-08-29): создание при первом
            # старте; при несовпадении зафиксированной конфигурации
            # (модель/размерность) с env — полная автореиндексация.
            _sync_embedding_meta(conn, settings)
    except (sqlite3.Error, OSError) as exc:
        raise StorageError(
            f"не удалось инициализировать БД {settings.db_path}: {exc}"
        ) from exc


def _get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row[0])


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _sync_embedding_meta(conn: sqlite3.Connection, settings: Settings) -> None:
    """Сверить (модель, размерность) эмбеддинга с записью в meta; разошлись —
    полная автореиндексация: notes_vec пересоздаётся под текущую размерность,
    все заметки (включая trash — их вектора тоже невалидны) уходят в pending.

    Наследие: у БД без meta (созданных до этой правки) запись создаётся из
    текущего env без реиндексации — нулевой миграцией; их вектора и так
    были построены той же моделью (иначе оператор потратил бы reindex.py).
    """
    conn.execute(_META_DDL)
    stored = _get_meta(conn, "embedding_model")
    stored_dim_raw = _get_meta(conn, "embedding_dim")
    existing_dim = vectors.existing_vec_dim(conn)
    if stored is None or stored_dim_raw is None:
        # Свежая БД (нет таблицы) или унаследованная (не трогаем, см. docstring).
        if existing_dim is None:
            vectors.create_vec_table(conn, settings.embedding_dim)
        _set_meta(conn, "embedding_model", settings.embedding_model)
        _set_meta(conn, "embedding_dim", str(settings.embedding_dim))
        return
    stored_dim = int(stored_dim_raw)
    if stored == settings.embedding_model and stored_dim == settings.embedding_dim:
        return
    notes_count = int(
        conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    )
    logging.getLogger("app").warning(
        "embedding model/dim changed: rebuilding vector index",
        extra={
            "event": "reindex_started",
            "from_model": stored,
            "to_model": settings.embedding_model,
            "from_dim": stored_dim,
            "to_dim": settings.embedding_dim,
            "notes": notes_count,
        },
    )
    conn.execute("DROP TABLE IF EXISTS notes_vec")
    vectors.create_vec_table(conn, settings.embedding_dim)
    conn.execute("UPDATE notes SET vector_status = 'pending'")
    _set_meta(conn, "embedding_model", settings.embedding_model)
    _set_meta(conn, "embedding_dim", str(settings.embedding_dim))
    logging.getLogger("app").info(
        "vector index rebuilt; background worker will re-encode all notes",
        extra={"event": "reindex_done", "pending_vector": notes_count},
    )


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