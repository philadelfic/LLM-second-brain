"""Запись версий истории (№41, постановка no41-01): baseline, дельты, события.

Версия пишется ТОЛЬКО на соединении вызывающего, внутри его `transaction()`
(FR-2.4, §13.5) — слой сам соединений и транзакций на пути записи не открывает.
`version=1` — baseline (проекция целиком), дальше дельты от головной версии;
пустая правка версии не создаёт; событие (удаление) — версия без дельты.
Для навыка `skills.version` остаётся зеркалом головной версии (§13.7), источник
и сессия версии приходят из contextvar (`bind_origin`, §13.4).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from app.config import Settings, get_settings
from app.services.history import (
    DEFAULT_SOURCE,
    ENTITY_NOTE,
    ENTITY_SKILL,
    bind_origin,
    current_origin,
    current_rollback_to,
    reconstruct,
    record_change,
    record_event,
    record_initial,
)
from app.storage.db import init_db, session, transaction

NOTE_FIELDS: dict[str, Any] = {
    "text": "первый текст",
    "title": "заметка",
    "namespace": "default",
    "expires_at": None,
}
SKILL_FIELDS: dict[str, Any] = {
    "name": "Deploy",
    "description": "How to deploy",
    "example": None,
    "steps": "1) build",
    "text": "build and run",
    "extra": None,
}


def _settings() -> Settings:
    settings = get_settings()
    init_db(settings)
    return settings


def _versions(entity: str, entity_id: int) -> list[sqlite3.Row]:
    with session(get_settings()) as conn:
        return list(
            conn.execute(
                "SELECT * FROM record_versions WHERE entity = ? AND entity_id = ? "
                "ORDER BY version",
                (entity, entity_id),
            )
        )


def _fields(entity: str, entity_id: int, version: int) -> dict[str, Any] | None:
    with session(get_settings()) as conn:
        return reconstruct(
            conn,
            entity,
            entity_id,
            version,
            inline_max=get_settings().history_delta_inline_max,
        )


def _note_projection(row: sqlite3.Row) -> dict[str, Any]:
    """Проекция заметки по §5.3 (текст/название/узел/срок; авто-`summary` — нет)."""
    return {
        "text": row["text"],
        "title": row["title"],
        "namespace": row["namespace"],
        "expires_at": row["expires_at"],
    }


def test_record_initial_baseline() -> None:
    """Baseline: `version=1`, `event='baseline'`, snapshot заполнен, delta NULL."""
    settings = _settings()

    with session(settings) as conn, transaction(conn):
        version = record_initial(conn, ENTITY_NOTE, 1, NOTE_FIELDS)

    assert version == 1
    row = _versions(ENTITY_NOTE, 1)[0]
    assert int(row["version"]) == 1
    assert row["event"] == "baseline"
    assert json.loads(row["snapshot"]) == NOTE_FIELDS
    assert row["delta"] is None
    assert json.loads(row["changed"]) == []
    assert row["restore_from"] is None
    assert row["source"] == DEFAULT_SOURCE
    assert row["session"] is None


def test_record_change_increments_head() -> None:
    """Правка: новая версия = head+1, дельта заполнена, snapshot пуст."""
    settings = _settings()
    with session(settings) as conn, transaction(conn):
        record_initial(conn, ENTITY_NOTE, 1, NOTE_FIELDS)

    edited = {**NOTE_FIELDS, "text": "второй текст"}
    with session(settings) as conn, transaction(conn):
        version = record_change(conn, ENTITY_NOTE, 1, edited)

    assert version == 2
    rows = _versions(ENTITY_NOTE, 1)
    assert [int(row["version"]) for row in rows] == [1, 2]
    assert rows[1]["event"] == "content"
    assert json.loads(rows[1]["changed"]) == ["text"]
    assert rows[1]["snapshot"] is None
    assert rows[1]["delta"] is not None
    assert _fields(ENTITY_NOTE, 1, 2) == edited


def test_record_change_noop_returns_none() -> None:
    """Без изменений версия не пишется: результат `None`, число версий прежнее."""
    settings = _settings()
    with session(settings) as conn, transaction(conn):
        record_initial(conn, ENTITY_NOTE, 1, NOTE_FIELDS)

    with session(settings) as conn, transaction(conn):
        version = record_change(conn, ENTITY_NOTE, 1, dict(NOTE_FIELDS))

    assert version is None
    assert len(_versions(ENTITY_NOTE, 1)) == 1


def test_record_event_delete_has_no_delta() -> None:
    """Событие удаления: `changed='[]'`, дельта пуста (версия-событие, FR-6.1)."""
    settings = _settings()
    with session(settings) as conn, transaction(conn):
        record_initial(conn, ENTITY_NOTE, 1, NOTE_FIELDS)

    with session(settings) as conn, transaction(conn):
        version = record_event(conn, ENTITY_NOTE, 1, "delete")

    assert version == 2
    row = _versions(ENTITY_NOTE, 1)[1]
    assert row["event"] == "delete"
    assert row["delta"] is None
    assert row["snapshot"] is None
    assert json.loads(row["changed"]) == []


def test_skill_version_mirror() -> None:
    """Для навыка `skills.version` — зеркало головной версии истории."""
    settings = _settings()
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO skills (id, name, description, steps, text, version) "
            "VALUES (10, 'Deploy', 'How to deploy', '1) build', 'build', 1)"
        )
        record_initial(conn, ENTITY_SKILL, 10, SKILL_FIELDS)

    with session(settings) as conn, transaction(conn):
        version = record_change(
            conn, ENTITY_SKILL, 10, {**SKILL_FIELDS, "name": "Deploy v2"}
        )

    assert version == 2
    with session(settings) as conn:
        assert (
            int(conn.execute("SELECT version FROM skills WHERE id = 10").fetchone()[0])
            == 2
        )


def test_reconstruct_equals_live_projection() -> None:
    """Инвариант FR-1.3: `reconstruct(head)` = проекция живой строки (3 правки)."""
    settings = _settings()
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO notes (id, text, title, namespace) "
            "VALUES (1, 'первый текст', 'заметка', 'default')"
        )
        row = conn.execute(
            "SELECT text, title, namespace, expires_at FROM notes WHERE id = 1"
        ).fetchone()
        record_initial(conn, ENTITY_NOTE, 1, _note_projection(row))

    for text, namespace in (("второй текст", "default"), ("второй текст", "work")):
        with session(settings) as conn, transaction(conn):
            conn.execute(
                "UPDATE notes SET text = ?, namespace = ? WHERE id = 1",
                (text, namespace),
            )
            row = conn.execute(
                "SELECT text, title, namespace, expires_at FROM notes WHERE id = 1"
            ).fetchone()
            assert record_change(conn, ENTITY_NOTE, 1, _note_projection(row)) is not None

    head = int(_versions(ENTITY_NOTE, 1)[-1]["version"])
    assert head == 3
    with session(settings) as conn:
        live = conn.execute(
            "SELECT text, title, namespace, expires_at FROM notes WHERE id = 1"
        ).fetchone()
    assert _fields(ENTITY_NOTE, 1, head) == _note_projection(live)


def test_origin_from_contextvar() -> None:
    """Источник и сессия версии — из `bind_origin` (§13.4), вне него — дефолт."""
    settings = _settings()

    assert current_origin() == (DEFAULT_SOURCE, None)
    assert current_rollback_to() is None
    with bind_origin("mcp", "agent-7"):
        assert current_origin() == ("mcp", "agent-7")
        with session(settings) as conn, transaction(conn):
            record_initial(conn, ENTITY_NOTE, 2, NOTE_FIELDS)
    assert current_origin() == (DEFAULT_SOURCE, None)

    row = _versions(ENTITY_NOTE, 2)[0]
    assert row["source"] == "mcp"
    assert row["session"] == "agent-7"


def test_record_change_rollback_event() -> None:
    """Откат: версия `event='rollback'` с `restore_from`, история не переписана."""
    settings = _settings()
    with session(settings) as conn, transaction(conn):
        record_initial(conn, ENTITY_NOTE, 3, NOTE_FIELDS)
        record_change(conn, ENTITY_NOTE, 3, {**NOTE_FIELDS, "text": "второй"})

    with bind_origin("rest", "op-1", rollback_to=1):
        assert current_rollback_to() == 1
        with session(settings) as conn, transaction(conn):
            version = record_change(conn, ENTITY_NOTE, 3, dict(NOTE_FIELDS))

    assert version == 3
    row = _versions(ENTITY_NOTE, 3)[2]
    assert row["event"] == "rollback"
    assert int(row["restore_from"]) == 1
    assert row["source"] == "rest"
    assert json.loads(row["changed"]) == ["text"]
    # История не переписана: прежние версии на месте, восстановление совпадает.
    assert [int(item["version"]) for item in _versions(ENTITY_NOTE, 3)] == [1, 2, 3]
    assert _fields(ENTITY_NOTE, 3, 3) == NOTE_FIELDS
