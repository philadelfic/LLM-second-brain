"""Самомиграция схемы: перелив БД под новый лимит заметки (LSBN-F-22, M1).

Слой без доменных правил: состояние миграции (meta + `migration_notes`),
создание новой базы, перелив 1:1, файловый swap и обратный перелив. Решение
о запуске (фазы 0–1) — в `app.services.migration` (lifespan); нарезка
невлезающих — джоба M2; union-чтение и прерывание — M3.

Ключевые решения (design §2–§4, решения О. не пересматриваются):
- Указатель активной БД — файловый swap при старте: `session()` открывает БД
  по пути на каждую операцию, rename переключает весь трафик, ноль правок в
  точках записи. Прежняя база → notes.<старый-лимит>.db (имя = её CHECK —
  цепочка сама себя документирует).
- Цепочка баз в `meta` активной базы: `migration.chain` = JSON
  [{file, note_chars}] прежних баз (старые впереди, активная не входит).
  Обратное направление (env вверх) разрешается по ней: источник — база chain,
  чей CHECK равен целевому числу; возврат к лимиту из цепочки всегда
  переиспользует ту базу (дельта активной переливается в неё), новая база
  создаётся только для лимита, которого в цепочке не было.
- Перелив 1:1, одна транзакция: справочники → notes (id сохранён, оба FTS
  строятся триггерами при INSERT — rebuild не нужен) → links/expirations/
  chunks/record_versions → вектора ПОСТРОЧНО (vec0 при INSERT…SELECT не
  проверен — design §7) → области. Невлезающие → `migration_notes` 'ожидает'
  (текст остаётся в прежней базе).
- Обратный перелив (откат): дельта активной базы апсертится В прежнюю
  (UPDATE-else-INSERT — REPLACE ломает FTS внешнего контента); части
  (record_versions source='migration_split') из дельты исключаются —
  длинные заметки возвращаются целыми. Swap возвращает прежний файл на
  db_path; базы не удаляются.
- Swap под открытым соединением (design §7 риск 5): сессии пократковременны,
  swap идёт в паузе старта; перед rename — WAL-checkpoint(TRUNCATE) обеих
  баз и снос пустых -wal/-shm, чтобы сайдкары не пережили rename.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import suppress
from pathlib import Path

from app.config import Settings
from app.storage import area_vectors, chunks, vectors
from app.storage.db import (
    init_db,
    note_chars_limit,
    session,
    transaction,
)

# --- состояние миграции в meta (design §3) --------------------------------

# Лимит активной схемы: дублирует живой CHECK читаемым для человека ключом.
META_SCHEMA_NOTE_CHARS = "schema.note_chars"
# Текущее состояние: JSON {phase: pour|split|done|aborted, target, prev_file,
# prev_limit, reason, counters} (split:{provider,model} — фаза 4, M2).
META_MIGRATION_STATE = "migration.state"
# Цепочка прежних баз: JSON [{file, note_chars}], старые впереди; активная
# база в список не входит. Имена файлов — относительно каталога db_path.
META_MIGRATION_CHAIN = "migration.chain"

# Происхождение частей нарезки (фаза 4, M2): по этому source в record_versions
# обратный перелив узнаёт части и исключает их из дельты (design §3).
SOURCE_MIGRATION_SPLIT = "migration_split"

MIGRATION_NOTES_TABLE = "migration_notes"

# Статусы заметки в миграции (design §3): 'ожидает' — ждёт нарезки (текст в
# прежней базе); 'нарезана'/'карантин'/'expired' — ставит джоба M2.
_MIGRATION_NOTES_DDL = f"""
CREATE TABLE IF NOT EXISTS {MIGRATION_NOTES_TABLE} (
  old_id     INTEGER PRIMARY KEY,
  status     TEXT    NOT NULL CHECK(status IN ('ожидает','нарезана','карантин','expired')),
  new_ids    TEXT    NULL,                  -- JSON-массив id частей (M2)
  attempts   INTEGER NOT NULL DEFAULT 0,
  reason     TEXT    NULL,
  updated_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
)
"""

logger = logging.getLogger("app")


class MigrationError(RuntimeError):
    """Ошибка самомиграции: перелив/swap не состоялись; активная база цела."""


# --- meta/chain/state -------------------------------------------------------


def _get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row[0])


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def read_chain(conn: sqlite3.Connection) -> list[dict]:
    """Цепочка прежних баз [{file, note_chars}] из meta; битый JSON — пусто."""
    raw = _get_meta(conn, META_MIGRATION_CHAIN)
    if not raw:
        return []
    try:
        chain = json.loads(raw)
    except ValueError:
        logger.warning(
            "migration.chain не читается — считаем цепочку пустой",
            extra={"event": "migration_chain_unreadable"},
        )
        return []
    return [entry for entry in chain if isinstance(entry, dict)]


def write_chain(conn: sqlite3.Connection, chain: list[dict]) -> None:
    _set_meta(conn, META_MIGRATION_CHAIN, json.dumps(chain, ensure_ascii=False))


def read_state(conn: sqlite3.Connection) -> dict | None:
    """Состояние миграции из meta; None — базы миграция не касалась."""
    raw = _get_meta(conn, META_MIGRATION_STATE)
    if not raw:
        return None
    try:
        state = json.loads(raw)
    except ValueError:
        return None
    return state if isinstance(state, dict) else None


def _write_state(conn: sqlite3.Connection, state: dict) -> None:
    _set_meta(
        conn, META_MIGRATION_STATE, json.dumps(state, ensure_ascii=False)
    )


def set_state_phase(conn: sqlite3.Connection, phase: str) -> dict | None:
    """Сменить фазу состояния миграции (остальные поля не тронуты).

    Публичная точка для сервиса (M3): «стоп джобы» прерывания пишет
    phase=aborted ДО reverse-pour (следующая партия нарезки видит её и
    берёт ноль заметок), фаза 7 ставит done. None — состояния нет
    (база миграцию не касалась), писать нечего.
    """
    state = read_state(conn)
    if state is None:
        return None
    state["phase"] = phase
    _write_state(conn, state)
    return state


def ensure_migration_schema(conn: sqlite3.Connection) -> None:
    """Таблица migration_notes + штамп schema.note_chars (идемпотентно).

    Вызывается для новой базы после init_db и для возвращаемой при откате
    (та создана до LSBN-F-22 — таблицы/штампа в ней может не быть).
    """
    conn.execute(_MIGRATION_NOTES_DDL)
    limit = note_chars_limit(conn)
    if limit is not None:
        _set_meta(conn, META_SCHEMA_NOTE_CHARS, str(limit))


# --- базовые примитивы перелива ---------------------------------------------


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """Колонки таблицы в порядке DDL (PRAGMA table_info)."""
    return [str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")]


def _table_pk(conn: sqlite3.Connection, table: str) -> list[str]:
    """Колонки PRIMARY KEY в порядке объявления (для ON CONFLICT)."""
    rows = [row for row in conn.execute(f"PRAGMA table_info({table})") if row["pk"]]
    return [str(row["name"]) for row in sorted(rows, key=lambda r: r["pk"])]


def _copy_rows(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    table: str,
    where: str = "1=1",
    params: tuple = (),
) -> int:
    """Скопировать строки таблицы источник→приёмник с сохранением id.

    Апсерт по PRIMARY KEY (UPDATE-else-INSERT), не REPLACE: REPLACE удаляет
    строку и вставляет заново, а у таблиц с FTS внешнего контента DELETE без
    'delete'-команды оставляет в индексе призрак (design §4: «REPLACE ломает
    FTS»). UPDATE-ветка апсерта триггеры UPDATE OF … честно срабатывают.
    Колонки — пересечение схем в порядке источника (схемы строит один
    init_db, расхождение — признак рукотворной правки).
    """
    src_cols = _table_columns(src, table)
    dst_cols = set(_table_columns(dst, table))
    columns = [c for c in src_cols if c in dst_cols]
    rows = src.execute(
        f"SELECT {', '.join(columns)} FROM {table} WHERE {where}", params
    ).fetchall()
    if not rows:
        return 0
    placeholders = ", ".join("?" * len(columns))
    pk = [c for c in _table_pk(dst, table) if c in columns]
    if pk:
        updates = ", ".join(
            f"{c}=excluded.{c}" for c in columns if c not in pk
        )
        sql = (
            f"INSERT INTO {table} ({', '.join(columns)}) "
            f"VALUES ({placeholders}) ON CONFLICT({', '.join(pk)})"
            + (f" DO UPDATE SET {updates}" if updates else "")
        )
    else:  # без PK апсерт невозможен — таблиц реестра без PK нет
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    dst.executemany(sql, [tuple(row) for row in rows])
    return len(rows)


def _vec_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """Колонки vec0-таблицы (виртуальная таблица: PRAGMA table_info врёт)."""
    cursor = conn.execute(f"SELECT * FROM {table} LIMIT 0")
    return [description[0] for description in cursor.description]


def _copy_vec_rows(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    table: str,
    key_column: str,
    where: str = "1=1",
    params: tuple = (),
) -> int:
    """Перелить вектора ПОСТРОЧНО (vec0 при INSERT…SELECT не проверен).

    Каждая строка: DELETE+INSERT по ключевой колонке (прецедент
    vectors.upsert — публичный протокол xUpdate, устойчив к версии
    расширения). embedding переносится как есть (BLOB float32): перелив
    идёт в пределах одной модели/размерности (смена эмбеддера — вне
    скоупа F-22, design §7).
    """
    columns = [c for c in _vec_columns(src, table) if c != "rowid"]
    rows = src.execute(
        f"SELECT {', '.join(columns)} FROM {table} WHERE {where}", params
    ).fetchall()
    placeholders = ", ".join("?" * len(columns))
    for row in rows:
        dst.execute(
            f"DELETE FROM {table} WHERE {key_column} = ?", (row[0],)
        )
        dst.execute(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
            tuple(row),
        )
    return len(rows)


# Реестр перелива «вниз» (фаза 3, design §4). Порядок обязателен:
# справочники → notes (FTS строят триггеры INSERT) → зависимости заметок
# (FK notes_chunks REFERENCES notes) → области (skills перед skill_versions).
# Фильтры — по ИСТОЧНИКУ: влезающее = length(text) ≤ цели; зависимости
# невлезающих заметок остаются в прежней базе вместе с текстом.
_FITTING = "length(text) <= ?"
_POUR_SPECS: tuple[tuple[str, str], ...] = (
    ("namespaces", "1=1"),
    ("promotions", "1=1"),
    ("notes", _FITTING),
    (
        "links",
        f"note_a IN (SELECT id FROM notes WHERE {_FITTING}) "
        f"AND note_b IN (SELECT id FROM notes WHERE {_FITTING})",
    ),
    (
        "note_expirations",
        f"note_id IN (SELECT id FROM notes WHERE {_FITTING})",
    ),
    (
        "warnings",
        f"subject_type != 'note' OR subject_id IN "
        f"(SELECT id FROM notes WHERE {_FITTING})",
    ),
    (
        chunks.CHUNKS_TABLE,
        f"note_id IN (SELECT id FROM notes WHERE {_FITTING})",
    ),
    (
        "record_versions",
        f"entity != 'note' OR entity_id IN (SELECT id FROM notes WHERE {_FITTING})",
    ),
    ("skills", "1=1"),
    ("skill_versions", "1=1"),
    ("skills_meta", "1=1"),
    ("terms", "1=1"),
    ("user_facts", "1=1"),
    ("assistant_logs", "1=1"),
    ("meta", "1=1"),  # мета эмбеддинга/чанков; migration.* переставляются ниже
)

# vec-таблицы: (таблица, ключевая колонка, фильтр строк). Area-vec переливаются
# целиком (содержимое областей переливается 1:1), вектора заметок/чанков —
# только влезающих.
_VEC_POUR_SPECS: tuple[tuple[str, str, str | None], ...] = (
    (vectors.VEC_TABLE, "note_id", f"note_id IN (SELECT id FROM notes WHERE {_FITTING})"),
    (
        chunks.CHUNKS_VEC_TABLE,
        "chunk_id",
        "chunk_id IN (SELECT id FROM notes_chunks WHERE note_id IN "
        "(SELECT id FROM notes WHERE length(text) <= ?))",
    ),
    *((table, key, None) for table, key in area_vectors.VEC_TABLES),
)


# --- фаза 2–3: новая база, перелив, swap ------------------------------------


def sibling_db_path(db_path: str, limit: int) -> str:
    """Имя базы-родственника рядом: notes.db → notes.<limit>.db.

    Имя = лимит CHECK той базы (design §2: «цепочка сама себя
    документирует»).
    """
    path = Path(db_path)
    return str(path.with_name(f"{path.stem}.{limit}{path.suffix}"))


def _prune_chain(chain: list[dict], file_name: str) -> list[dict]:
    """Цепочка без записей об указанном файле (цикл вниз-вверх не плодит дублей)."""
    return [entry for entry in chain if entry.get("file") != file_name]


def create_new_base(
    settings: Settings,
    *,
    source_path: str,
    target: int,
    prev_file: str,
    reason: str,
) -> dict:
    """Фазы 2–3: новая база notes.<target><suffix>, перелив 1:1, состояние.

    Идемпотентно по отношению к обрыву (design §6): недолитый файл прошлой
    попытки стирается — рестарт до swap переливает заново, прежняя база
    не тронута. Одна транзакция на весь перелив + состояние (design §4,
    фаза 3). Направление «вверх» без базы-родственника идёт тем же путём:
    фильтр length(text) <= target вбирает всё, migration_notes остаётся
    пустой (обратный перелив без частей — тоже без модели).

    Returns:
        Счётчики {old_total, poured, pending} — для журнала и /health (M3).
    """
    temp_path = sibling_db_path(settings.db_path, target)
    for suffix in ("", "-wal", "-shm"):  # сайдкары не должны пережить пересоздание
        Path(temp_path + suffix).unlink(missing_ok=True)
    new_settings = settings.model_copy(
        update={"db_path": temp_path, "max_note_chars": target}
    )
    init_db(new_settings)  # полная схема + сиды + vec-таблицы (design §3)
    src_settings = settings.model_copy(update={"db_path": source_path})
    with session(src_settings) as src, session(new_settings) as dst:
        old_total = int(
            src.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        )
        with transaction(dst):
            # Таблица migration_notes + штамп CHECK: init_db о миграции не
            # знает (схема релиза), а 'ожидает' пишутся ниже в эту же
            # транзакцию. Идемпотентно.
            ensure_migration_schema(dst)
            # Строки: справочники → notes (FTS строят триггеры) → зависимости
            # → реестры; мета переливается как есть, миграционные ключи
            # переставляются ниже явно.
            for table, where in _POUR_SPECS:
                _copy_rows(src, dst, table, where, (target,) * where.count("?"))
            # Вектора и области — ПОСТРОЧНО (vec0 при INSERT…SELECT не
            # проверен, design §7); области — без фильтра, целиком.
            for table, key_column, where in _VEC_POUR_SPECS:
                if where is None:
                    _copy_vec_rows(src, dst, table, key_column)
                else:
                    _copy_vec_rows(
                        src, dst, table, key_column, where, (target,) * where.count("?")
                    )
            # Мета миграции поверх перелитой: ключи миграции прежней базы
            # (после прошлой миграции) не должны течь в новую как есть.
            prev_name = Path(prev_file).name
            _set_meta(dst, META_SCHEMA_NOTE_CHARS, str(target))
            write_chain(
                dst,
                [
                    *_prune_chain(read_chain(src), prev_name),
                    {"file": prev_name, "note_chars": _live_limit_or(src, target)},
                ],
            )
            # Невлезающие → 'ожидает' (текст остаётся в прежней базе): id
            # берутся из ИСТОЧНИКА — в новой базе их уже нет, длинные не
            # перелились. Кросс-коннект в одном INSERT невозможен — перенос.
            long_ids = [
                int(row[0])
                for row in src.execute(
                    "SELECT id FROM notes WHERE length(text) > ?", (target,)
                ).fetchall()
            ]
            dst.executemany(
                f"INSERT INTO {MIGRATION_NOTES_TABLE}(old_id, status, reason) "
                "VALUES (?, 'ожидает', 'длиннее цели миграции')",
                [(old_id,) for old_id in long_ids],
            )
            pending = len(long_ids)
            counters = {
                "old_total": old_total,
                "poured": int(
                    dst.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
                ),
                "pending": pending,
            }
            # Резерв диапазона id (design §8.1): max(id) новой базы после
            # перелива ниже прежнего (длинные хвосты не перелились), и
            # max+1 может совпасть с old_id ждущего исходника — затенение
            # в union, потеря заметки при abort. id_floor — нижняя граница
            # НОВЫХ id (= MAX(id) прежней базы + 1), живёт в state, пока
            # миграция активна; NoteService._insert не пускает новые id
            # ниже неё.
            id_floor = (
                int(src.execute("SELECT COALESCE(MAX(id), 0) FROM notes").fetchone()[0])
                + 1
            )
            _write_state(
                dst,
                {
                    "phase": "pour",
                    "target": target,
                    "prev_file": prev_name,
                    "prev_limit": note_chars_limit(src),
                    "id_floor": id_floor,
                    "reason": reason,
                    "counters": counters,
                },
            )
        return counters


def _live_limit_or(conn: sqlite3.Connection, fallback: int) -> int:
    limit = note_chars_limit(conn)
    return limit if limit is not None else fallback


def swap(settings: Settings, *, away_file: str, incoming_file: str) -> None:
    """Файловый swap указателя активной базы (design §2, фаза 3).

    WAL-checkpoint обеих баз, снос пустых -wal/-shm (сайдкары не должны
    пережить rename), затем rename: активная → away_file, входящая →
    db_path. Следующие session() открывают уже новую базу; открытых
    соединений нет — swap идёт в паузе старта до первого запроса.
    """
    active = Path(settings.db_path)
    away = Path(away_file)
    incoming = Path(incoming_file)
    if not incoming.is_file():
        raise MigrationError(f"нет файла базы для swap: {incoming}")
    if away.exists() and not _same_file(away, active):
        raise MigrationError(
            f"имя прежней базы занято посторонним файлом: {away}"
        )
    _checkpoint(str(active))
    _checkpoint(str(incoming))
    active.replace(away)
    try:
        incoming.replace(active)
    except OSError:
        # Второй rename не прошёл: вернуть указатель — активная база цела
        # (контракт MigrationError), CrashConsistency прежнего файла не тронут.
        with suppress(OSError):
            away.replace(active)
        raise


def _same_file(a: Path, b: Path) -> bool:
    return a.resolve() == b.resolve()


def _checkpoint(db_path: str) -> None:
    """WAL-checkpoint(TRUNCATE) на отдельном соединении + снос сайдкаров."""
    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    for suffix in ("-wal", "-shm"):
        Path(db_path + suffix).unlink(missing_ok=True)


# --- обратное направление (env вверх / откат) -------------------------------


def _migration_part_ids(conn: sqlite3.Connection) -> set[int]:
    """Id частей нарезки в активной базе (record_versions source).

    Базы до появления record_versions (гипотетический откат очень старой
    схемы) частей не имеют — таблицы нет, множество пусто.
    """
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='record_versions'"
    ).fetchone()
    if row is None:
        return set()
    return {
        int(row_id)
        for (row_id,) in conn.execute(
            "SELECT DISTINCT entity_id FROM record_versions "
            "WHERE entity = 'note' AND source = ?",
            (SOURCE_MIGRATION_SPLIT,),
        )
    }


def reverse_pour(
    settings: Settings,
    *,
    source_path: str,
    away_file: str,
    target: int,
    reason: str,
    phase: str = "done",
    exclude_note_ids: set[int] | None = None,
) -> dict:
    """Обратный перелив: дельта активной базы → прежняя (с CHECK = цели).

    Перелив идёт 1:1 без модели (design §4): части (record_versions
    source='migration_split') и все их зависимости из дельты исключаются —
    в прежней базе длинные заметки возвращаются целыми, частей там нет и
    быть не должно. Апсерт по PK (REPLACE ломает FTS), вектора — построчно.
    Одна транзакция на приёмнике: строки + вектора + мета (цепочка получает
    уходящую базу, state → phase=`phase`: done при возврате вверх — нарезка
    не нужна, всё вернулось целым; aborted при прерывании миграции вниз,
    M3 — части возвращаются в исходники, сервис остаётся на прежней базе).
    Swap — отдельным вызовом.

    `exclude_note_ids` (M3, abort): id, чьи строки приёмника НЕЛЬЗЯ
    трогать дельтой. Прерывание передаёт все old_id из migration_notes:
    активная база выдаёт новым заметкам max(id)+1, что может совпасть
    с id исходника — апсерт по такому id ЗАТИРАЛ бы оригинал прежней базы
    посторонней заметкой (потеря исходника). Затенённые заметки остаются
    только в уходящей базе (файл не удаляется). Зависимости (связи, чанки,
    TTL, история, вектора) исключаются тем же множеством.
    """
    with session(settings) as active, session(
        settings.model_copy(update={"db_path": source_path})
    ) as src:
        part_notes = _migration_part_ids(active)
        # kept_parts — семантика частей (нарезка), exclude в неё не мешается:
        # счётчик прерывания/отката показывает именно неперелитые части.
        kept_parts = len(part_notes)
        if exclude_note_ids:
            part_notes = part_notes | set(exclude_note_ids)
        part_chunks = (
            {
                int(chunk_id)
                for (chunk_id,) in active.execute(
                    f"SELECT id FROM {chunks.CHUNKS_TABLE} WHERE note_id "
                    f"IN ({', '.join('?' * len(part_notes))})",
                    tuple(part_notes),
                )
            }
            if part_notes
            else set()
        )
        live = note_chars_limit(active)

        def _where_not_in(*columns: str, ids: set[int]) -> tuple[str, tuple]:
            """`col NOT IN (…)` по каждому столбцу; пустое множество — без фильтра."""
            if not ids:
                return "1=1", ()
            ordered = tuple(sorted(ids))
            marks = ", ".join("?" * len(ordered))
            where = " AND ".join(f"{column} NOT IN ({marks})" for column in columns)
            return where, ordered * len(columns)

        notes_where, notes_params = _where_not_in("id", ids=part_notes)
        # Зависимости частей (связи-наследования F-23, чанки, TTL, история,
        # предупреждения о частях) остаются в уходящей базе вместе с ними.
        warnings_where, warnings_params = _where_not_in("subject_id", ids=part_notes)
        if part_notes:
            warnings_where = f"subject_type != 'note' OR {warnings_where}"
        versions_where, versions_params = _where_not_in("entity_id", ids=part_notes)
        if part_notes:
            versions_where = f"entity != 'note' OR {versions_where}"
        specs: tuple[tuple[str, str, tuple], ...] = (
            ("namespaces", "1=1", ()),
            ("promotions", "1=1", ()),
            ("notes", notes_where, notes_params),
            ("links", *_where_not_in("note_a", "note_b", ids=part_notes)),
            ("note_expirations", *_where_not_in("note_id", ids=part_notes)),
            ("warnings", warnings_where, warnings_params),
            (chunks.CHUNKS_TABLE, *_where_not_in("note_id", ids=part_notes)),
            ("record_versions", versions_where, versions_params),
            ("skills", "1=1", ()),
            ("skill_versions", "1=1", ()),
            ("skills_meta", "1=1", ()),
            ("terms", "1=1", ()),
            ("user_facts", "1=1", ()),
            ("assistant_logs", "1=1", ()),
            ("meta", "1=1", ()),  # миграционные ключи переставляются ниже
        )
        vec_specs: tuple[tuple[str, str, str, tuple], ...] = (
            # Ключ vec-таблицы — note_id (не notes.id!), фильтр свой.
            (vectors.VEC_TABLE, "note_id", *_where_not_in("note_id", ids=part_notes)),
            (
                chunks.CHUNKS_VEC_TABLE,
                "chunk_id",
                *_where_not_in("chunk_id", ids=part_chunks),
            ),
            *((table, key, "1=1", ()) for table, key in area_vectors.VEC_TABLES),
        )

        counters = {"poured": 0, "kept_parts": kept_parts}
        with transaction(src):
            # Приёмник мог быть создан до LSBN-F-22: таблицы migration_notes
            # и штампа в нём может не быть (идемпотентно).
            ensure_migration_schema(src)
            for table, where, params in specs:
                counters["poured"] += _copy_rows(active, src, table, where, params)
            for table, key_column, where, params in vec_specs:
                counters["poured"] += _copy_vec_rows(
                    active, src, table, key_column, where, params
                )
            # Уходящая база встаёт в конец цепочки; state → phase (done при
            # возврате вверх, aborted при прерывании — M3): migration_notes
            # приёмника не тронуты (после прямого перелива 'ожидает' писались
            # в уходящую базу, не в приёмник).
            away_name = Path(away_file).name
            _set_meta(src, META_SCHEMA_NOTE_CHARS, str(target))
            write_chain(
                src,
                [
                    *_prune_chain(read_chain(src), away_name),
                    {"file": away_name, "note_chars": _live_limit_or(active, target)},
                ],
            )
            _write_state(
                src,
                {
                    "phase": phase,
                    "target": target,
                    "prev_file": away_name,
                    "prev_limit": live,
                    "reason": reason,
                    "counters": counters,
                },
            )
        return counters


# --- фаза 0: план старта (решение о миграции принимает lifespan) ------------


def startup_plan(settings: Settings) -> dict:
    """Фаза 0: сверка env с живым CHECK → план старта (design §4).

    Режимы: `none` — миграция не нужна (файла нет — его только что создал
    init_db, CHECK нет, env совпал, либо миграция была прервана оператором —
    phase=aborted, автостарт не возобновляется); `down` — цель меньше живого
    CHECK, новая база и перелив влезающего; `up_chain` — цель больше, в
    цепочке есть база с таким CHECK — дельта возвращается в неё (откат 1:1,
    части исключаются); `up_new` — цель больше, родственной базы нет — новая
    база вбирает всё (тоже без модели).
    """
    path = Path(settings.db_path)
    target = settings.max_note_chars
    if not path.is_file():  # свежая база — init_db создал её под env
        return {"mode": "none", "target": target, "schema_limit": None, "pending": 0}
    with session(settings) as conn:
        live = note_chars_limit(conn)
        # Прерывание (M3, design §4): оператор остановил миграцию — автостарт
        # НЕ возобновляется при рестарте с тем же env («сервис остаётся на
        # прежней базе»); повторный запуск — сознательное действие (вопрос
        # «дожать миграцию» открыт, design §6).
        state = read_state(conn)
        if state is not None and str(state.get("phase")) == "aborted":
            return {
                "mode": "none",
                "target": target,
                "schema_limit": live,
                "pending": 0,
                "aborted": True,
            }
        if live is None or live == target:
            return {
                "mode": "none",
                "target": target,
                "schema_limit": live,
                "pending": 0,
            }
        if target > live:
            source = _chain_source(settings, conn, target)
            if source is not None:
                return {
                    "mode": "up_chain",
                    "target": target,
                    "schema_limit": live,
                    "pending": 0,
                    "source_path": source,
                    "prev_file": sibling_db_path(settings.db_path, live),
                    "reason": (
                        f"env MAX_NOTE_CHARS={target} поверх живого "
                        f"CHECK={live}: возврат к базе цепочки"
                    ),
                }
            return {
                "mode": "up_new",
                "target": target,
                "schema_limit": live,
                "pending": 0,
                "prev_file": sibling_db_path(settings.db_path, live),
                "reason": (
                    f"env MAX_NOTE_CHARS={target} поверх живого CHECK={live}: "
                    f"базы с CHECK={target} в цепочке нет — новая база"
                ),
            }
        return {
            "mode": "down",
            "target": target,
            "schema_limit": live,
            "pending": int(
                conn.execute(
                    "SELECT COUNT(*) FROM notes WHERE length(text) > ?", (target,)
                ).fetchone()[0]
            ),
            "prev_file": sibling_db_path(settings.db_path, live),
            "reason": (
                f"env MAX_NOTE_CHARS={target} поверх живого CHECK={live}: "
                f"перелив влезающего, длинные — в нарезку (M2)"
            ),
        }


def _chain_source(settings: Settings, conn: sqlite3.Connection, target: int) -> str | None:
    """Путь базы цепочки с CHECK = target (новейшая годная впереди)."""
    directory = Path(settings.db_path).parent
    for entry in reversed(read_chain(conn)):  # новые впереди
        file_name = str(entry.get("file", ""))
        if not file_name or entry.get("note_chars") != target:
            continue
        candidate = directory / file_name
        if not candidate.is_file():
            logger.warning(
                "база цепочки не найдена — источник миграции пропускается",
                extra={"event": "migration_chain_file_missing", "file": file_name},
            )
            continue
        with session(settings.model_copy(update={"db_path": str(candidate)})) as src:
            if note_chars_limit(src) == target:
                return str(candidate)
        logger.warning(
            "CHECK базы цепочки не совпадает с обещанным — пропуск",
            extra={"event": "migration_chain_limit_mismatch", "file": file_name},
        )
    return None


def run_startup_migration(settings: Settings, plan: dict) -> dict:
    """Фазы 2–3 по плану `startup_plan`: перелив + swap; счётчики — назад.

    При ошибке перелива (до swap) активная база цела — вызывающий продолжает
    работу на прежней базе; сбой swap восстанавливает указатель (`swap`).
    """
    mode = plan["mode"]
    if mode in ("down", "up_new"):
        counters = create_new_base(
            settings,
            source_path=settings.db_path,
            target=plan["target"],
            prev_file=plan["prev_file"],
            reason=plan["reason"],
        )
        swap(
            settings,
            away_file=plan["prev_file"],
            incoming_file=sibling_db_path(settings.db_path, plan["target"]),
        )
        return counters
    if mode == "up_chain":
        counters = reverse_pour(
            settings,
            source_path=plan["source_path"],
            away_file=plan["prev_file"],
            target=plan["target"],
            reason=plan["reason"],
        )
        swap(
            settings,
            away_file=plan["prev_file"],
            incoming_file=plan["source_path"],
        )
        return counters
    raise MigrationError(f"неизвестный режим миграции: {mode}")


def ensure_startup_schema(settings: Settings) -> None:
    """Таблица migration_notes + штамп schema.note_chars на каждом старте.

    Идемпотентно: база без миграции получает пустую таблицу и читаемый
    штамп живого CHECK (observability — design §3); миграционная — свою.
    """
    with session(settings) as conn, transaction(conn):
        ensure_migration_schema(conn)
