"""Физическое удаление и история (lsbdef-0014, релиз 3.3.1, постановка LSBN-D-7).

`delete_note_physical` (джоба `expiration` воркера) уносит и версии заметки
из `record_versions`: `notes.id` без AUTOINCREMENT переиспользуется, и
оставшийся baseline переиспользованного id ронял `record_initial` новой
заметки на UNIQUE `(entity, entity_id, version)` → падал `memory_save`
(тест-контур 15.10.2026: 16/16). Soft delete (`NoteService.delete`) историю
СОХРАНЯЕТ — заметка в trash восстановима, её версии остаются. После чистки
полный путь `memory_save` (save → baseline, update → дельта) на
переиспользованном id работает, история удалённой заметки не воскресает.
"""

from __future__ import annotations

import sqlite3
import uuid

import pytest
from fakes import HashEmbedder

from app.config import Settings, get_settings
from app.services.history import ENTITY_NOTE, read_versions, reconstruct
from app.services.notes import NoteService
from app.storage.db import delete_note_physical, init_db, session, transaction


def unique(text: str) -> str:
    """Уникальный текст: дословный дедуп не должен блокировать запись."""
    return f"{text} [{uuid.uuid4().hex[:8]}]"


@pytest.fixture
def service() -> NoteService:
    settings = get_settings()
    init_db(settings)
    return NoteService(settings, HashEmbedder(settings.embedding_dim))


def _versions(entity: str, entity_id: int) -> list[sqlite3.Row]:
    with session(get_settings()) as conn:
        return list(
            conn.execute(
                "SELECT * FROM record_versions WHERE entity = ? AND entity_id = ? "
                "ORDER BY version",
                (entity, entity_id),
            )
        )


def _physical_delete(settings: Settings, note_id: int) -> None:
    """Физическое удаление как в воркере (process_expired_notes): заметка —
    короткая транзакция на `delete_note_physical` (worker.py)."""
    with session(settings) as conn, transaction(conn):
        delete_note_physical(conn, note_id)


def test_physical_delete_purges_history_soft_delete_keeps_it(
    service: NoteService,
) -> None:
    """lsbdef-0014: физическое удаление чистит `record_versions`, soft delete — нет.

    Приёмочный критерий №1: baseline удалённой физически заметки из истории
    уходит (иначе переиспользованный id падал бы на UNIQUE); история мягко
    удалённой заметки (baseline + версия-событие `delete`) остаётся на месте —
    защита от перелёта чистки на soft-путь.
    """
    settings = get_settings()
    gone_id = int(service.save(unique("Физ-удаление: уходим целиком"), title="Уход")["id"])
    assert [int(row["version"]) for row in _versions(ENTITY_NOTE, gone_id)] == [1]

    trashed_id = int(service.save(unique("Soft delete: остаёмся в trash"), title="Trash")["id"])
    assert service.delete(trashed_id)["deleted"] is True
    assert [int(row["version"]) for row in _versions(ENTITY_NOTE, trashed_id)] == [1, 2]
    assert _versions(ENTITY_NOTE, trashed_id)[1]["event"] == "delete"

    _physical_delete(settings, gone_id)

    # Физическое удаление: заметки и её истории нет вовсе.
    with session(settings) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM notes WHERE id = ?", (gone_id,)
        ).fetchone()[0] == 0
    assert _versions(ENTITY_NOTE, gone_id) == []
    # Soft delete: история сохранена (заметка восстановима).
    assert [int(row["version"]) for row in _versions(ENTITY_NOTE, trashed_id)] == [1, 2]


def test_physical_delete_purges_comment_fts_soft_delete_keeps_it(
    service: NoteService,
) -> None:
    """LSBN-D-8: физическое удаление чистит FTS пометок, soft delete — нет.

    `notes_comment_fts` — внешний контент без DELETE-триггера: до 3.4.0
    термы пометки удалённой заметки оставались в файле (замер 17.10.2026),
    нарушая обещание жёсткого удаления «данных нет» (TTL-заметки с
    чувствительной пометкой). Soft delete пометку и её индекс хранит —
    заметка восстановима вместе с пометкой.
    """
    settings = get_settings()
    marker = "ксилофонограмма"
    gone_id = int(
        service.save(unique("Пометка: физическое удаление"), title="Пометка")["id"]
    )
    assert (
        service.update(gone_id, comment=f"пометка уходит {marker}")["updated"] is True
    )
    trashed_id = int(
        service.save(unique("Пометка: soft delete"), title="Пометка две")["id"]
    )
    assert (
        service.update(trashed_id, comment=f"пометка остаётся {marker}")["updated"]
        is True
    )

    def fts_hits() -> int:
        with session(settings) as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM notes_comment_fts "
                    "WHERE notes_comment_fts MATCH ?",
                    (f'"{marker}"',),
                ).fetchone()[0]
            )

    assert fts_hits() == 2  # обе пометки индексированы

    _physical_delete(settings, gone_id)

    # Физическое удаление: термы пометки ушли из FTS (осталась только
    # мягко удалённая — её строка и индекс сохранены).
    assert fts_hits() == 1
    with session(settings) as conn:
        assert conn.execute(
            "SELECT comment FROM notes WHERE id = ?", (trashed_id,)
        ).fetchone()["comment"] == f"пометка остаётся {marker}"


def test_reused_id_full_save_update_history_path(service: NoteService) -> None:
    """lsbdef-0014: после физического удаления id переиспользуется — полный путь жив.

    Приёмочный критерий №2: создать → удалить физически → создать (id
    переиспользован — первая свободная rowid) → `NoteService.save` (полный путь
    `memory_save` с `record_initial`) не падает; `NoteService.update` пишет
    дельту (version ≥ 2); `history_list` (`read_versions`) по
    переиспользованному id показывает только версии новой заметки — история
    удалённой не воскресает.
    """
    settings = get_settings()
    old_text = unique("История: текст до физического удаления")
    first_id = int(service.save(old_text, title="Первая")["id"])
    assert [int(row["version"]) for row in _versions(ENTITY_NOTE, first_id)] == [1]

    _physical_delete(settings, first_id)

    new_text = unique("История: текст после физического удаления")
    reused_id = int(service.save(new_text, title="Вторая")["id"])
    assert reused_id == first_id  # id переиспользован (notes.id без AUTOINCREMENT)

    # Полный путь `memory_save`: baseline новой заметки записан (не упал на
    # UNIQUE об остаток истории прежней) и описывает именно новую заметку.
    rows = _versions(ENTITY_NOTE, reused_id)
    assert [int(row["version"]) for row in rows] == [1]
    assert rows[0]["event"] == "baseline"
    assert rows[0]["snapshot"] is not None
    assert new_text in str(rows[0]["snapshot"])
    assert old_text not in str(rows[0]["snapshot"])

    # Правка: дельта пишется, version ≥ 2.
    assert service.update(reused_id, text=unique("История: правка новой заметки"))[
        "updated"
    ] is True
    rows = _versions(ENTITY_NOTE, reused_id)
    assert [int(row["version"]) for row in rows] == [1, 2]
    assert rows[1]["event"] == "content"

    # history_list по переиспользованному id — только версии новой заметки:
    # обе от её baseline, текст удалённой не воскресает.
    listing = read_versions(ENTITY_NOTE, reused_id, settings)
    assert listing["total"] == 2
    assert [int(item["version"]) for item in listing["versions"]] == [2, 1]
    with session(settings) as conn:
        head = reconstruct(
            conn,
            ENTITY_NOTE,
            reused_id,
            2,
            inline_max=settings.history_delta_inline_max,
        )
    assert head is not None
    assert head["text"] == service.get([reused_id])["notes"][0]["text"]
    assert old_text not in head["text"]
