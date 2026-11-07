"""Physical deletion and history (lsbdef-0014, release 3.3.1, the LSBN-D-7 brief).

    `delete_note_physical` (the worker's `expiration` job) also carries the note's
    versions out of `record_versions`: `notes.id` has no AUTOINCREMENT and is
    reused, so a leftover baseline of the reused id crashed the new note's
    `record_initial` on the UNIQUE `(entity, entity_id, version)` → `memory_save`
    failed (the test contour, 2026-10-15: 16/16). A soft delete
    (`NoteService.delete`) KEEPS the history — a trash note is restorable, its
    versions stay. After the cleanup the full `memory_save` path (save → baseline,
    update → a delta) works on a reused id, and the deleted note's history does
    not resurrect.
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
    """A unique text: the verbatim dedup must not block the write."""
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
    """A physical deletion like the worker's (process_expired_notes): the note —
            a short transaction per `delete_note_physical` (worker.py).
    """
    with session(settings) as conn, transaction(conn):
        delete_note_physical(conn, note_id)


def test_physical_delete_purges_history_soft_delete_keeps_it(
    service: NoteService,
) -> None:
    """lsbdef-0014: a physical deletion clears `record_versions`; a soft delete — does not.

                Acceptance criterion #1: a physically deleted note's baseline leaves
                the history (otherwise a reused id would fail on UNIQUE); a softly
                deleted note's history (the baseline + the `delete` version-event)
                stays in place — protecting against the cleanup overshooting onto the
                soft path.
    """
    settings = get_settings()
    gone_id = int(service.save(unique("Физ-удаление: уходим целиком"), title="Уход")["id"])
    assert [int(row["version"]) for row in _versions(ENTITY_NOTE, gone_id)] == [1]

    trashed_id = int(service.save(unique("Soft delete: остаёмся в trash"), title="Trash")["id"])
    assert service.delete(trashed_id)["deleted"] is True
    assert [int(row["version"]) for row in _versions(ENTITY_NOTE, trashed_id)] == [1, 2]
    assert _versions(ENTITY_NOTE, trashed_id)[1]["event"] == "delete"

    _physical_delete(settings, gone_id)

    # Physical deletion: the note and its history are gone entirely.
    with session(settings) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM notes WHERE id = ?", (gone_id,)
        ).fetchone()[0] == 0
    assert _versions(ENTITY_NOTE, gone_id) == []
    # Soft delete: the history kept (the note is restorable).
    assert [int(row["version"]) for row in _versions(ENTITY_NOTE, trashed_id)] == [1, 2]


def test_physical_delete_purges_comment_fts_soft_delete_keeps_it(
    service: NoteService,
) -> None:
    """LSBN-D-8: a physical deletion clears the comments' FTS; a soft delete — does not.

                `notes_comment_fts` is external content without a DELETE trigger:
                before 3.4.0 a deleted note's comment terms stayed in the file (the
                2026-10-17 measurement), breaking the hard deletion's promise of "no
                data" (TTL notes with a sensitive comment). A soft delete keeps the
                comment and its index — the note is restorable together with the
                comment.
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

    assert fts_hits() == 2  # both comments indexed

    _physical_delete(settings, gone_id)

    # Physical deletion: the comment's terms left the FTS (only the softly
    # deleted one remains — its row and index kept).
    assert fts_hits() == 1
    with session(settings) as conn:
        assert conn.execute(
            "SELECT comment FROM notes WHERE id = ?", (trashed_id,)
        ).fetchone()["comment"] == f"пометка остаётся {marker}"


def test_reused_id_full_save_update_history_path(service: NoteService) -> None:
    """lsbdef-0014: after a physical deletion the id is reused — the full path lives.

                Acceptance criterion #2: create → delete physically → create (the id
                reused — the first free rowid) → `NoteService.save` (the full
                `memory_save` path with `record_initial`) does not fail;
                `NoteService.update` writes a delta (version ≥ 2); `history_list`
                (`read_versions`) over the reused id shows only the new note's
                versions — the deleted one's history does not resurrect.
    """
    settings = get_settings()
    old_text = unique("История: текст до физического удаления")
    first_id = int(service.save(old_text, title="Первая")["id"])
    assert [int(row["version"]) for row in _versions(ENTITY_NOTE, first_id)] == [1]

    _physical_delete(settings, first_id)

    new_text = unique("История: текст после физического удаления")
    reused_id = int(service.save(new_text, title="Вторая")["id"])
    assert reused_id == first_id  # the id reused (notes.id has no AUTOINCREMENT)

    # The full `memory_save` path: the new note's baseline written (did not fail on
    # the UNIQUE over the former history's leftover) and describes exactly the new note.
    rows = _versions(ENTITY_NOTE, reused_id)
    assert [int(row["version"]) for row in rows] == [1]
    assert rows[0]["event"] == "baseline"
    assert rows[0]["snapshot"] is not None
    assert new_text in str(rows[0]["snapshot"])
    assert old_text not in str(rows[0]["snapshot"])

    # An edit: a delta written; version ≥ 2.
    assert service.update(reused_id, text=unique("История: правка новой заметки"))[
        "updated"
    ] is True
    rows = _versions(ENTITY_NOTE, reused_id)
    assert [int(row["version"]) for row in rows] == [1, 2]
    assert rows[1]["event"] == "content"

    # history_list over the reused id — only the new note's versions:
    # both from its baseline; the deleted one's text does not resurrect.
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
