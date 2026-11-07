"""The `assistant_log` history entity (LSBN-F-10, f10-01): the projection and the baseline.

    The history layer (#41) accepts the logs area as a versioned entity:
    `CONTENT_FIELDS` lists exactly the content fields of arch §5.4 (without
    `vector_status` and the timestamps), and the baseline backfill self-healing
    adds `version=1, event='baseline', source='migration'` where no versions
    exist yet (a repeated start — a no-op).
"""

from __future__ import annotations

import json

import pytest

from app.config import Settings, get_settings
from app.services.history import (
    CONTENT_FIELDS,
    ENTITY_ASSISTANT_LOG,
    assistant_log_projection,
)
from app.storage.db import init_db, session, transaction

DIM = 8

_CONTENT = (
    "action",
    "reason",
    "entity",
    "entity_id",
    "entity_version",
    "entity_ref",
    "outcome",
    "verified",
    "source",
    "session",
)


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def _logs_versions(settings: Settings, entity_id: int) -> list:
    with session(settings) as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT version, event, source, snapshot, delta FROM record_versions "
                "WHERE entity = ? AND entity_id = ? ORDER BY version",
                (ENTITY_ASSISTANT_LOG, entity_id),
            )
        ]


def test_assistant_log_projection(settings: Settings) -> None:
    """The projection carries exactly the §5.4 content fields, no service fields or timestamps."""
    assert ENTITY_ASSISTANT_LOG == "assistant_log"
    assert set(CONTENT_FIELDS[ENTITY_ASSISTANT_LOG]) == set(_CONTENT)

    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO assistant_logs (action, reason, entity, entity_id, "
            "outcome) VALUES ('deploy', 'why', 'note', 7, 'done')"
        )
        row = conn.execute("SELECT * FROM assistant_logs WHERE id = 1").fetchone()

    projection = assistant_log_projection(row)
    assert set(projection) == set(_CONTENT)
    assert projection["action"] == "deploy"
    assert projection["reason"] == "why"
    assert projection["entity"] == "note"
    assert projection["entity_id"] == 7
    assert projection["outcome"] == "done"
    assert projection["source"] == "mcp"  # the DDL default
    for service_field in ("vector_status", "created_at", "updated_at", "deleted_at"):
        assert service_field not in projection


def test_baseline_backfill_assistant_logs(settings: Settings) -> None:
    """A row without history gets its baseline at the next start; a repeat — a no-op."""
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO assistant_logs (action, reason, entity, entity_id) "
            "VALUES ('deploy', 'why', 'note', 7)"
        )

    init_db(settings)  # the first pass with a write — the baseline
    versions = _logs_versions(settings, 1)
    assert len(versions) == 1
    first = versions[0]
    assert int(first["version"]) == 1
    assert first["event"] == "baseline"
    assert first["source"] == "migration"
    assert first["delta"] is None
    assert json.loads(first["snapshot"]) == {
        "action": "deploy",
        "reason": "why",
        "entity": "note",
        "entity_id": 7,
        "entity_version": None,
        "entity_ref": None,
        "outcome": None,
        "verified": None,
        "source": "mcp",
        "session": None,
    }

    init_db(settings)  # a restart — no new versions multiplied
    assert len(_logs_versions(settings, 1)) == 1
