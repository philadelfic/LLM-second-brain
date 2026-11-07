"""The history retention policy (#41, brief no41-01): prune with re-base (§13.8).

    At `HISTORY_MAX_VERSIONS > 0` (and/or `HISTORY_RETENTION_DAYS > 0`) the prune
    removes the oldest versions while KEEPING `version=1` (the chain's anchor) and
    the newest ones; the earliest kept version is rewritten into a full `snapshot`
    — its delta would reference a deleted version. Numbers are never reused, and
    the head version's restore stays correct.
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.services.history import ENTITY_NOTE, reconstruct, record_change, record_initial
from app.storage.db import init_db, session, transaction

NOTE_FIELDS = {
    "text": "первый текст",
    "title": "заметка",
    "namespace": "default",
    "expires_at": None,
}


def _versions(entity: str, entity_id: int) -> list[int]:
    with session(get_settings()) as conn:
        return [
            int(row["version"])
            for row in conn.execute(
                "SELECT version FROM record_versions WHERE entity = ? "
                "AND entity_id = ? ORDER BY version",
                (entity, entity_id),
            )
        ]


def _row(entity: str, entity_id: int, version: int) -> dict:
    with session(get_settings()) as conn:
        return dict(
            conn.execute(
                "SELECT version, snapshot, delta, changed FROM record_versions "
                "WHERE entity = ? AND entity_id = ? AND version = ?",
                (entity, entity_id, version),
            ).fetchone()
        )


def _fields(entity: str, entity_id: int, version: int) -> dict | None:
    with session(get_settings()) as conn:
        return reconstruct(
            conn,
            entity,
            entity_id,
            version,
            inline_max=get_settings().history_delta_inline_max,
        )


def test_prune_rebase_keeps_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ceiling of 3: the oldest non-zero version goes, the next one — a re-base; the chain lives."""
    monkeypatch.setenv("HISTORY_MAX_VERSIONS", "3")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)

    with session(settings) as conn, transaction(conn):
        record_initial(conn, ENTITY_NOTE, 1, NOTE_FIELDS)
    for text in ("второй текст", "третий текст", "четвёртый текст"):
        with session(settings) as conn, transaction(conn):
            assert (
                record_change(
                    conn, ENTITY_NOTE, 1, {**NOTE_FIELDS, "text": text}
                )
                is not None
            )

    # Kept: the anchor (1) and the newest (3, 4); the oldest non-zero (2) removed.
    assert _versions(ENTITY_NOTE, 1) == [1, 3, 4]
    rebased = _row(ENTITY_NOTE, 1, 3)
    assert rebased["snapshot"] is not None
    assert rebased["delta"] is None
    # Restoring the head and the re-based version — correct.
    assert _fields(ENTITY_NOTE, 1, 4) == {**NOTE_FIELDS, "text": "четвёртый текст"}
    assert _fields(ENTITY_NOTE, 1, 3) == {**NOTE_FIELDS, "text": "третий текст"}
    # Numbers are not reused: 2 never reappeared.
    assert 2 not in _versions(ENTITY_NOTE, 1)
