"""Политика хранения истории (№41, постановка no41-01): prune с re-base (§13.8).

При `HISTORY_MAX_VERSIONS > 0` (и/или `HISTORY_RETENTION_DAYS > 0`) prune
вычищает старейшие версии, СОХРАНЯЯ `version=1` (якорь цепочки) и новейшие;
самая ранняя сохраняемая версия переписывается в полный `snapshot` — её дельта
ссылалась бы на удалённую версию. Номера не переиспользуются, восстановление
головной версии остаётся верным.
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
    """Потолок 3: старейшая ненулевая версия уходит, следующая — re-base, цепочка жива."""
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

    # Сохранились: якорь (1) и новейшие (3, 4); старейшая ненулевая (2) убрана.
    assert _versions(ENTITY_NOTE, 1) == [1, 3, 4]
    rebased = _row(ENTITY_NOTE, 1, 3)
    assert rebased["snapshot"] is not None
    assert rebased["delta"] is None
    # Восстановление головной и перебазированной версии — верное.
    assert _fields(ENTITY_NOTE, 1, 4) == {**NOTE_FIELDS, "text": "четвёртый текст"}
    assert _fields(ENTITY_NOTE, 1, 3) == {**NOTE_FIELDS, "text": "третий текст"}
    # Номера не переиспользуются: 2 не появилась заново.
    assert 2 not in _versions(ENTITY_NOTE, 1)
