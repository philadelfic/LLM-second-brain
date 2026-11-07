"""The history migration (#41, brief no41-01): the baseline and the skill_versions merge.

    `init_db` creates `record_versions` idempotently (`CREATE TABLE IF NOT
    EXISTS`; no indexes beyond the PK — §5.1) and backfills the baseline for ALL
    existing records: `version=1` = the current content (FR-8.1); a repeated
    start — a no-op; a row that appeared after the brief gets its baseline at the
    next start (self-healing, §5.4). The skills archive (`skill_versions`) moves
    into the common form: v1 — the earliest archive copy whole; the intermediate
    ones — deltas; vN — the current `skills` row; the numbers come from the
    archive/the live row; no losses.
"""

from __future__ import annotations

import json
from typing import Any

from app.config import Settings, get_settings
from app.services.history import ENTITY_NOTE, reconstruct
from app.storage.db import init_db, session, transaction


def _versions_per_entity() -> dict[str, int]:
    with session(get_settings()) as conn:
        return {
            str(row["entity"]): int(row["count"])
            for row in conn.execute(
                "SELECT entity, COUNT(*) AS count FROM record_versions GROUP BY entity"
            )
        }


def _record_count(table: str) -> int:
    with session(get_settings()) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _table_exists(name: str) -> bool:
    with session(get_settings()) as conn:
        return (
            conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?",
                (name,),
            ).fetchone()[0]
            == 1
        )


def _fields(settings: Settings, entity: str, entity_id: int, version: int) -> Any:
    with session(settings) as conn:
        return reconstruct(
            conn,
            entity,
            entity_id,
            version,
            inline_max=settings.history_delta_inline_max,
        )


def _seed_live_records(settings: Settings) -> None:
    """The live database's records: notes, a term, a fact (the skill seed already exists)."""
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO notes (id, text, title, namespace) "
            "VALUES (1, 'старая заметка', 'первая', 'default')"
        )
        conn.execute(
            "INSERT INTO notes (id, text, title, namespace) "
            "VALUES (2, 'вторая заметка', NULL, 'work')"
        )
        conn.execute(
            "INSERT INTO terms (id, term, context, definition) "
            "VALUES (1, 'терм', 'контекст', 'определение')"
        )
        conn.execute(
            "INSERT INTO user_facts (id, name, body) VALUES (1, 'факт', 'тело')"
        )


def test_record_versions_schema() -> None:
    """The §5.1 DDL exactly as in the architecture: the PK over (entity, entity_id, version)."""
    settings = get_settings()
    init_db(settings)

    with session(settings) as conn:
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' "
            "AND name = 'record_versions'"
        ).fetchone()[0]
        indexes = [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' "
                "AND tbl_name = 'record_versions'"
            )
            if not str(row["name"]).startswith("sqlite_autoindex")
        ]
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(record_versions)")
        }

    normalized = " ".join(ddl.split())
    assert "PRIMARY KEY (entity, entity_id, version)" in normalized
    assert "CHECK(event IN ('baseline','content','delete','restore','rollback'))" in (
        normalized
    )
    assert columns == {
        "entity",
        "entity_id",
        "version",
        "event",
        "source",
        "session",
        "changed",
        "snapshot",
        "delta",
        "restore_from",
        "created_at",
    }
    # No indexes beyond the PK: the PK covers the lookup and the reverse order (§5.1).
    assert indexes == []  # the sqlite_autoindex PK — not a separate index


def test_baseline_idempotent() -> None:
    """A repeated `init_db`: every record has a baseline; the version count unchanged."""
    settings = get_settings()
    init_db(settings)
    _seed_live_records(settings)

    init_db(settings)  # the first pass with records
    first = _versions_per_entity()

    # Every record of every entity has a baseline — exactly one version each.
    assert first[ENTITY_NOTE] == _record_count("notes") == 2
    assert first["term"] == _record_count("terms") == 1
    assert first["user_fact"] == _record_count("user_facts") == 1
    assert first["skill"] == _record_count("skills")  # the skill-creator seed
    assert first["skill_template"] == 1  # the skills_meta.instruction_template key
    with session(settings) as conn:
        rows = conn.execute(
            "SELECT version, event, snapshot, delta FROM record_versions"
        ).fetchall()
    assert all(
        int(row["version"]) == 1
        and row["event"] == "baseline"
        and row["snapshot"] is not None
        and row["delta"] is None
        for row in rows
    )
    # The baseline's content = the record's current content (FR-8.1).
    assert _fields(settings, ENTITY_NOTE, 1, 1) == {
        "text": "старая заметка",
        "title": "первая",
        "namespace": "default",
        "expires_at": None,
        "comment": None,  # no24-01 (3.2.0)
    }

    init_db(settings)  # a service restart — a no-op

    assert _versions_per_entity() == first


def test_baseline_selfheal_missing_row() -> None:
    """A row without history gets its baseline at the next start."""
    settings = get_settings()
    init_db(settings)
    _seed_live_records(settings)
    init_db(settings)
    assert len(_versions_per_entity()) > 0

    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO notes (id, text, title, namespace) "
            "VALUES (3, 'заметка между стартами', 'третья', 'default')"
        )

    init_db(settings)

    assert _versions_per_entity()[ENTITY_NOTE] == 3
    assert _fields(settings, ENTITY_NOTE, 3, 1) == {
        "text": "заметка между стартами",
        "title": "третья",
        "namespace": "default",
        "expires_at": None,
        "comment": None,  # no24-01 (3.2.0)
    }


def test_skill_versions_merged_into_common_form() -> None:
    """The `skill_versions` archive moved into the common form: every copy restorable."""
    settings = get_settings()
    init_db(settings)

    # A live database of the former mechanics: the skill v3 + the v1/v2 copies archive; no history.
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO skills (id, name, description, steps, text, version) "
            "VALUES (5, 'C', 'd', 's', 'текст C', 3)"
        )
        for version, name in ((1, "A"), (2, "B")):
            conn.execute(
                "INSERT INTO skill_versions (skill_id, version, name, description, "
                "steps, text) VALUES (5, ?, ?, 'd', 's', ?)",
                (version, name, f"текст {name}"),
            )
        conn.execute("DELETE FROM record_versions WHERE entity = 'skill'")

    init_db(settings)

    def projection(name: str) -> dict[str, Any]:
        return {
            "name": name,
            "description": "d",
            "example": None,
            "steps": "s",
            "text": f"текст {name}",
            "extra": None,
        }

    # The 1..3 chain restores every archive copy and the current row.
    assert _fields(settings, "skill", 5, 1) == projection("A")
    assert _fields(settings, "skill", 5, 2) == projection("B")
    assert _fields(settings, "skill", 5, 3) == projection("C")

    with session(settings) as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                "SELECT version, event, source, snapshot, delta FROM record_versions "
                "WHERE entity = 'skill' AND entity_id = 5 ORDER BY version"
            )
        ]
    assert [int(row["version"]) for row in rows] == [1, 2, 3]
    assert rows[0]["event"] == "baseline"
    assert rows[0]["source"] == "migration"
    assert json.loads(rows[0]["snapshot"]) == projection("A")
    assert rows[1]["delta"] is not None and rows[1]["snapshot"] is None
    assert rows[2]["delta"] is not None
    # The live skill row untouched; the archive stayed a legacy cast (§13.7).
    with session(settings) as conn:
        assert int(
            conn.execute("SELECT version FROM skills WHERE id = 5").fetchone()[0]
        ) == 3
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM skill_versions WHERE skill_id = 5"
            ).fetchone()[0]
            == 2
        )
    assert _table_exists("skill_versions")
