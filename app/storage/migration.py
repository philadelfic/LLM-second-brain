"""Self-migration of the schema: pouring the DB under a new note limit (LSBN-F-22, M1).

    A layer without domain rules: the migration state (meta + `migration_notes`),
    new-base creation, the 1:1 pour, the file swap and the reverse pour. The
    launch decision (phases 0-1) lives in `app.services.migration` (lifespan);
    splitting the non-fitting - the M2 job; union reads and the abort - M3.

    Key decisions (design SS2-SS4; the owner's rulings are not revisited):
    - The active-DB pointer - a file swap at startup: `session()` opens the DB by
      path on every operation, a rename switches the whole traffic, zero edits at
      the write sites. The previous base -> notes.<old-limit>.db (the name = its
      CHECK - the chain documents itself).
    - The chain of bases in the active base's `meta`: `migration.chain` = JSON
      [{file, note_chars}] of previous bases (oldest first, the active one not
      included). The reverse direction (env up) resolves over it: the source is
      the chain base whose CHECK equals the target number; a return to a limit
      from the chain always reuses that base (the active delta pours into it); a
      new base is created only for a limit the chain never had.
    - The 1:1 pour, one transaction: registries -> notes (the id is preserved, both
      FTS indexes are built by the INSERT triggers - no rebuild needed) ->
      links/expirations/chunks/record_versions -> vectors ROW BY ROW (vec0 under
      INSERT...SELECT is unverified - design SS7) -> areas. Non-fitting ->
      `migration_notes` 'ozhidaet' (the text stays in the previous base).
    - The reverse pour (the rollback): the active base's delta is upserted INTO
      the previous base (UPDATE-else-INSERT - REPLACE breaks external-content
      FTS); the parts (record_versions source='migration_split') are excluded
      from the delta - long notes come back whole. The swap returns the previous
      file to db_path; bases are never deleted.
    - A swap under an open connection (design SS7 risk 5): sessions are
      short-lived, the swap runs in the startup pause; before the rename - a
      WAL-checkpoint(TRUNCATE) of both bases and removing empty -wal/-shm so the
      sidecars do not survive the rename.
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

# --- migration state in meta (design §3) --------------------------------

# The active schema limit: mirrors the live CHECK under a human-readable key.
META_SCHEMA_NOTE_CHARS = "schema.note_chars"
# The current state: JSON {phase: pour|split|done|aborted, target, prev_file,
# prev_limit, reason, counters} (split:{provider,model} — phase 4, M2).
META_MIGRATION_STATE = "migration.state"
# The chain of previous bases: JSON [{file, note_chars}], oldest first; the
# active base is not in the list. File names — relative to the db_path dir.
META_MIGRATION_CHAIN = "migration.chain"

# The split-parts origin (phase 4, M2): by this record_versions source the
# reverse pour recognizes the parts and excludes them from the delta (design §3).
SOURCE_MIGRATION_SPLIT = "migration_split"

MIGRATION_NOTES_TABLE = "migration_notes"

# Note statuses in a migration (design §3): 'ожидает' — waiting to be split
# (the text stays in the previous base); 'нарезана'/'карантин'/'expired' — the M2 job.
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
    """Self-migration failed: the pour/swap did not happen; the active base is intact."""


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
    """The chain of previous bases [{file, note_chars}] from meta; broken JSON - empty."""
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
    """The migration state from meta; None - the migration never touched the base."""
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
    """Change the migration state's phase (the other fields untouched).

            The public entry point for the service (M3): the abort's "stop the jobs"
            writes phase=aborted BEFORE the reverse-pour (the next splitting batch
            sees it and takes zero notes), phase 7 sets done. None - there is no
            state (the base was never migrated), nothing to write.

    """
    state = read_state(conn)
    if state is None:
        return None
    state["phase"] = phase
    _write_state(conn, state)
    return state


def ensure_migration_schema(conn: sqlite3.Connection) -> None:
    """The migration_notes table + the schema.note_chars stamp (idempotent).

            Called for a new base after init_db and for the one returned by a rollback
            (it was created before LSBN-F-22 - the table/stamp may be missing).

    """
    conn.execute(_MIGRATION_NOTES_DDL)
    limit = note_chars_limit(conn)
    if limit is not None:
        _set_meta(conn, META_SCHEMA_NOTE_CHARS, str(limit))


# --- basic pour primitives ---------------------------------------------


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """The table's columns in DDL order (PRAGMA table_info)."""
    return [str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")]


def _table_pk(conn: sqlite3.Connection, table: str) -> list[str]:
    """The PRIMARY KEY columns in declaration order (for ON CONFLICT)."""
    rows = [row for row in conn.execute(f"PRAGMA table_info({table})") if row["pk"]]
    return [str(row["name"]) for row in sorted(rows, key=lambda r: r["pk"])]


def _copy_rows(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    table: str,
    where: str = "1=1",
    params: tuple = (),
) -> int:
    """Copy a table's rows source -> receiver preserving ids.

            An upsert over the PRIMARY KEY (UPDATE-else-INSERT), not REPLACE: REPLACE
            deletes the row and inserts it again, and for external-content FTS tables
            a DELETE without the 'delete' command leaves a ghost in the index
            (design SS4: "REPLACE breaks FTS"). The UPDATE branch of the upsert lets
            the UPDATE OF ... triggers fire honestly. The columns - the intersection
            of the schemas in source order (one init_db builds both schemas; a
            divergence signals a hand-made edit).

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
    else:  # no PK — no upsert; no registry tables exist without a PK
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    dst.executemany(sql, [tuple(row) for row in rows])
    return len(rows)


def _vec_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """The columns of a vec0 table (a virtual table: PRAGMA table_info lies)."""
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
    """Pour the vectors ROW BY ROW (vec0 under INSERT...SELECT is unverified).

            Each row: DELETE+INSERT over the key column (the vectors.upsert precedent
            - the public xUpdate protocol, robust to extension version changes). The
            embedding is carried as is (a float32 BLOB): the pour runs within one
            model/dimension (an embedding change is out of F-22's scope, design SS7).

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


# The "down" pour registry (phase 3, design §4). The order is mandatory:
# registries → notes (the INSERT triggers build FTS) → note dependencies
# (the notes_chunks REFERENCES notes FK) → areas (skills before skill_versions).
# Filters — by the SOURCE: what fits = length(text) <= the target; the
# dependencies of non-fitting notes stay in the previous base with the text.
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
    ("meta", "1=1"),  # the embedding/chunks meta; migration.* keys are re-ordered below
)

# vec tables: (table, key column, row filter). Area vecs are poured whole
# (area content pours 1:1); note/chunk vectors — only the fitting ones,
# the fitting ones.
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


# --- phases 2–3: the new base, the pour, the swap ------------------------------------


def sibling_db_path(db_path: str, limit: int) -> str:
    """The name of a sibling base next to this one: notes.db -> notes.<limit>.db.

            The name = that base's CHECK limit (design SS2: "the chain documents
            itself").

    """
    path = Path(db_path)
    return str(path.with_name(f"{path.stem}.{limit}{path.suffix}"))


def _prune_chain(chain: list[dict], file_name: str) -> list[dict]:
    """The chain without records of the given file (a down-up cycle spawns no duplicates)."""
    return [entry for entry in chain if entry.get("file") != file_name]


#: The receiver-file suffix for when the target name is taken by a live chain
#: base (DEF-5): notes.<target>.db.incoming — swap places it on db_path, and
#: the chain base stays untouched (design: "bases are never deleted").
INCOMING_SUFFIX = ".incoming"


def _pour_target(settings: Settings, conn: sqlite3.Connection, target: int) -> str:
    """The new base's receiver file: notes.<target>.db or ...db.incoming (DEF-5).

            An existing notes.<target>.db described in the ACTIVE base's chain is a
            live chain base (the only copy of the previous run's parts): the new base
            pours next to it under the ...db.incoming name. Not in the chain - an
            unfinished remainder of an interrupted attempt (a restart before the swap,
            design SS6): the name is free for recreation. Called both in
            `create_new_base` and in `run_startup_migration` - swap must receive the
            exact file the pour went into; the decision is deterministic over the disk
            state and the active base's meta, and the active chain does not change
            between the two calls.

    """
    path = sibling_db_path(settings.db_path, target)
    if not Path(path).is_file():
        return path
    chain_files = {
        Path(str(entry.get("file", ""))).name for entry in read_chain(conn)
    }
    if Path(path).name in chain_files:
        return path + INCOMING_SUFFIX
    return path


def create_new_base(
    settings: Settings,
    *,
    source_path: str,
    target: int,
    prev_file: str,
    reason: str,
) -> dict:
    """Phases 2-3: the new base notes.<target><suffix>, the 1:1 pour, the state.

            Idempotent against a break (design SS6): the unfinished file of a previous
            attempt (absent from the active base's chain) is wiped - a restart before
            the swap pours again, the previous base untouched. A live chain base under
            the target name (DEF-5: a repeated forward after a reverse) is NEVER wiped
            - the pour goes into notes.<target>.db.incoming and swap places it on
            db_path. One transaction for the whole pour + the state (design SS4,
            phase 3). The "up" direction without a sibling base takes the same path:
            the length(text) <= target filter takes everything, migration_notes stays
            empty (a reverse pour without parts - also model-free).

            Returns:
                The counters {old_total, poured, pending} - for the journal and
                /health (M3).

    """
    canonical = sibling_db_path(settings.db_path, target)
    with session(settings) as conn:
        temp_path = _pour_target(settings, conn, target)
    if temp_path != canonical:
        logger.warning(
            "файл цели занят базой цепочки — новая база льётся под временным "
            "именем (базы не удаляются)",
            extra={
                "event": "migration_chain_base_spared",
                "file": Path(canonical).name,
                "incoming": Path(temp_path).name,
            },
        )
    for suffix in ("", "-wal", "-shm"):  # sidecars must not survive the recreation
        Path(temp_path + suffix).unlink(missing_ok=True)
    new_settings = settings.model_copy(
        update={"db_path": temp_path, "max_note_chars": target}
    )
    init_db(new_settings)  # the full schema + seeds + vec tables (design §3)
    src_settings = settings.model_copy(update={"db_path": source_path})
    with session(src_settings) as src, session(new_settings) as dst:
        old_total = int(
            src.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        )
        with transaction(dst):
            # The migration_notes table + the CHECK stamp: init_db knows nothing about
            # the migration (the release schema), and the waiting rows are written below
            # into the same transaction. Idempotent.
            ensure_migration_schema(dst)
            # Rows: registries → notes (the triggers build FTS) → dependencies
            # → registries; the meta pours as is, the migration keys are
            # re-set below explicitly.
            for table, where in _POUR_SPECS:
                _copy_rows(src, dst, table, where, (target,) * where.count("?"))
            # Vectors and areas — ROW BY ROW (vec0 under INSERT…SELECT is
            # unverified, design §7); the areas — unfiltered, whole.
            for table, key_column, where in _VEC_POUR_SPECS:
                if where is None:
                    _copy_vec_rows(src, dst, table, key_column)
                else:
                    _copy_vec_rows(
                        src, dst, table, key_column, where, (target,) * where.count("?")
                    )
            # The migration meta over the poured one: the previous base's migration keys
            # (left by an earlier migration) must not flow into the new base as is.
            prev_name = Path(prev_file).name
            _set_meta(dst, META_SCHEMA_NOTE_CHARS, str(target))
            write_chain(
                dst,
                [
                    *_prune_chain(read_chain(src), prev_name),
                    {"file": prev_name, "note_chars": _live_limit_or(src, target)},
                ],
            )
            # Non-fitting → 'ожидает' (the text stays in the previous base): the ids are
            # taken from the SOURCE — the new base does not have them, the long ones did
            # not pour. A cross-connection in one INSERT is impossible — a transfer.
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
            # The id range reserve (design §8.1): the new base's max(id) after the pour
            # is below the previous one (the long tails did not pour), and max+1 may
            # collide with the old_id of a waiting source — shadowing in the union, a
            # lost note on abort. id_floor is the lower bound of NEW ids
            # (= MAX(id) of the previous base + 1); it lives in state while the migration
            # is active; NoteService._insert refuses new ids below it.
            # it.
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


def free_away_name(settings: Settings, away_file: str) -> str:
    """An away-base name not taken by a foreign file (LSBN-D-12, DEF-5b).

            The name is free or it IS the active base (swap will rename that very one)
            - take it as is. Taken by a foreign file (for example, a remainder of past
            down-up cycles with the same CHECK - bases are never deleted by design) -
            pick a suffix: `...away`, `...away2`, ... The name must be resolved BEFORE
            the pour: it fans out into state.prev_file and the receiver chain, and
            swap must receive exactly that file. The `migration_away_name_rebased`
            event - the operator sees the name the base left under.

    """
    active = Path(settings.db_path)
    candidate = Path(away_file)
    if not candidate.exists() or _same_file(candidate, active):
        return away_file
    for number in range(2, 1000):
        suffix = ".away" if number == 2 else f".away{number}"
        alternate = candidate.with_name(candidate.name + suffix)
        if not alternate.exists() or _same_file(alternate, active):
            logger.warning(
                "имя уходящей базы занято посторонним файлом — база уходит "
                "под свободным именем (базы не удаляются)",
                extra={
                    "event": "migration_away_name_rebased",
                    "requested": candidate.name,
                    "file": alternate.name,
                },
            )
            return str(alternate)
    raise MigrationError(f"нет свободного имени для уходящей базы: {away_file}")


def swap(settings: Settings, *, away_file: str, incoming_file: str) -> None:
    """The file swap of the active-base pointer (design SS2, phase 3).

            A WAL-checkpoint of both bases, removing empty -wal/-shm (sidecars must
            not survive the rename), then the rename: active -> away_file, incoming ->
            db_path. The next session() calls open the new base already; no open
            connections - the swap runs in the startup pause before the first request.
            The "the name is taken by a foreign file" refusal stays as the fail-closed
            guard (LSBN-D-12): normal flows pick the name via `free_away_name` BEFORE
            the pour, a foreign name never reaches here anymore.

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
        # The second rename failed: restore the pointer — the active base is intact
        # (the MigrationError contract), the old file's crash consistency untouched.
        with suppress(OSError):
            away.replace(active)
        raise


def _same_file(a: Path, b: Path) -> bool:
    return a.resolve() == b.resolve()


def _checkpoint(db_path: str) -> None:
    """A WAL-checkpoint(TRUNCATE) on a separate connection + sidecar removal."""
    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    for suffix in ("-wal", "-shm"):
        Path(db_path + suffix).unlink(missing_ok=True)


# --- the reverse direction (env up / the rollback) -------------------------------


def _migration_part_ids(conn: sqlite3.Connection) -> set[int]:
    """The ids of split parts in the active base (record_versions source).

            Databases predating record_versions (a hypothetical rollback of a very old
            schema) have no parts - no table, an empty set.

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
    """The reverse pour: the active base's delta -> the previous one (CHECK = the target).

            The pour is 1:1 without a model (design SS4): the parts
            (record_versions source='migration_split') and all their dependencies are
            excluded from the delta - in the previous base the long notes come back
            whole, and there are no parts there and must not be. An upsert over the PK
            (REPLACE breaks FTS), the vectors row by row. One transaction on the
            receiver: rows + vectors + meta (the chain gets the away base, state ->
            phase=`phase`: done on a return up - no splitting needed, everything came
            back whole; aborted on an interruption of a downward migration, M3 - the
            parts return into the sources, the service stays on the previous base).
            The swap is a separate call.

            `exclude_note_ids` (M3, abort): ids whose receiver rows must NOT be
            touched by the delta. The abort passes every old_id from migration_notes:
            the active base hands new notes max(id)+1, which may collide with a
            source's id - an upsert over such an id would OVERWRITE the previous
            base's original with a stranger note (a lost source). Shadowed notes stay
            only in the away base (the file is not deleted). The dependencies (links,
            chunks, TTLs, history, vectors) are excluded by the same set.

    """
    with session(settings) as active, session(
        settings.model_copy(update={"db_path": source_path})
    ) as src:
        part_notes = _migration_part_ids(active)
        # kept_parts — the parts semantics (splitting), exclude does not pollute it:
        # the abort/rollback counter shows exactly the unpoured parts.
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
            """`col NOT IN (...)` per column; an empty set - no filter."""
            if not ids:
                return "1=1", ()
            ordered = tuple(sorted(ids))
            marks = ", ".join("?" * len(ordered))
            where = " AND ".join(f"{column} NOT IN ({marks})" for column in columns)
            return where, ordered * len(columns)

        notes_where, notes_params = _where_not_in("id", ids=part_notes)
        # The parts' dependencies (the F-23 inheritance links, chunks, TTLs, history,
        # the part warnings) stay in the away base together with them.
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
            ("meta", "1=1", ()),  # migration keys are re-ordered below
        )
        vec_specs: tuple[tuple[str, str, str, tuple], ...] = (
            # The vec table's key is note_id (not notes.id!), its own filter.
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
            # The receiver may have been created before LSBN-F-22: it may lack the
            # migration_notes table and the stamp (idempotent).
            ensure_migration_schema(src)
            for table, where, params in specs:
                counters["poured"] += _copy_rows(active, src, table, where, params)
            for table, key_column, where, params in vec_specs:
                counters["poured"] += _copy_vec_rows(
                    active, src, table, key_column, where, params
                )
            # The away base joins the end of the chain; state → phase (done on a return
            # up, aborted on an interruption — M3): the receiver's migration_notes are
            # untouched (after a forward pour the 'ожидает' rows were written to the away
            # base, not the receiver).
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


# --- phase 0: the startup plan (the lifespan makes the migration decision) ------------


def startup_plan(settings: Settings) -> dict:
    """Phase 0: compare the env with the live CHECK -> the startup plan (design SS4).

            Modes: `none` - no migration is needed (no file - init_db just created it,
            no CHECK, the env matches, or the migration was aborted by the operator -
            phase=aborted, the autostart does not resume); `down` - the target is
            below the live CHECK, a new base and a pour of what fits; `up_chain` - the
            target is above and the chain has a base with that CHECK - the delta
            returns into it (a 1:1 rollback, the parts excluded); `up_new` - the
            target is above, no sibling base - a new base takes everything (also
            model-free).

    """
    path = Path(settings.db_path)
    target = settings.max_note_chars
    if not path.is_file():  # a fresh database — init_db created it under env
        return {"mode": "none", "target": target, "schema_limit": None, "pending": 0}
    with session(settings) as conn:
        live = note_chars_limit(conn)
        # An interruption (M3, design §4): the operator stopped the migration — the
        # autostart does NOT resume on a restart with the same env ("the service
        # stays on the previous base"); a repeated run is a deliberate action (the
        # "finish the migration" question is open, design §6).
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
    """The path of the chain base with CHECK = target (newest fitting first)."""
    directory = Path(settings.db_path).parent
    for entry in reversed(read_chain(conn)):  # newest first
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
    """Phases 2-3 over the `startup_plan`: the pour + the swap; the counters - back.

            On a pour error (before the swap) the active base is intact - the caller
            keeps working on the previous base; a swap failure restores the pointer
            (`swap`).

    """
    mode = plan["mode"]
    # The away-base name is resolved BEFORE the pour (LSBN-D-12): it fans out
    # into state.prev_file and the receiver chain, and swap gets the same name.
    away = free_away_name(settings, plan["prev_file"])
    if mode in ("down", "up_new"):
        # The pour receiver — before create_new_base (DEF-5): the chain base under
        # the target name is never wiped, the new base pours into …db.incoming;
        # swap places exactly that one on db_path.
        with session(settings) as conn:
            incoming = _pour_target(settings, conn, plan["target"])
        counters = create_new_base(
            settings,
            source_path=settings.db_path,
            target=plan["target"],
            prev_file=away,
            reason=plan["reason"],
        )
        swap(
            settings,
            away_file=away,
            incoming_file=incoming,
        )
        return counters
    if mode == "up_chain":
        counters = reverse_pour(
            settings,
            source_path=plan["source_path"],
            away_file=away,
            target=plan["target"],
            reason=plan["reason"],
        )
        swap(
            settings,
            away_file=away,
            incoming_file=plan["source_path"],
        )
        return counters
    raise MigrationError(f"неизвестный режим миграции: {mode}")


def ensure_startup_schema(settings: Settings) -> None:
    """The migration_notes table + the schema.note_chars stamp on every start.

            Idempotent: a non-migration database gets an empty table and a readable
            stamp of the live CHECK (observability - design SS3); a migration one - its
            own.

    """
    with session(settings) as conn, transaction(conn):
        ensure_migration_schema(conn)
