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
  (ARCH §3.3); несовпадение с EMBEDDING_DIM при старте — отказ запуска
  (переиндексация — scripts/reindex.py).
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
# cosine-метрика — там же); в init_db — только создание/сверка при старте.

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

def init_db(settings: Settings) -> None:
    """Создать схему при старте (критерий приёмки Фазы 2); идемпотентно.

    Raises:
        StorageError: БД недоступна, нет FTS5/vec0 или схема повреждена
        (в том числе несовпадение размерности vec0-таблицы с EMBEDDING_DIM).
    """
    path = Path(settings.db_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with session(settings) as conn:
            conn.execute(_NOTES_DDL.format(max_note_chars=settings.max_note_chars))
            conn.execute(_FTS_DDL)
            for trigger in _TRIGGERS:
                conn.execute(trigger)
            _check_fts_integrity(conn)
            # Вектора (Фаза 3): создание при первом старте; при последующих —
            # сверка размерности с env, несовпадение — понятный отказ запуска
            # вместо молчаливо невалидного индекса (REQUIREMENTS §8).
            try:
                vectors.ensure_vec_table(conn, settings.embedding_dim)
            except vectors.VectorError as exc:
                raise StorageError(str(exc)) from exc
    except (sqlite3.Error, OSError) as exc:
        raise StorageError(
            f"не удалось инициализировать БД {settings.db_path}: {exc}"
        ) from exc


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