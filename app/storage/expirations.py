"""The expired-note deletion queue — note_expirations (lsb-0004-02).

A layer without domain rules (like `app.storage.chunks`/`app.storage.vectors`):
only rows of the `note_expirations(note_id PK, expires_at)` table. The domain
semantics (when to insert/update/delete, TTL parsing) live in the service
layer `app.services.notes`.

Synchronization with `notes.expires_at` is at the service level (the
lsb-0004-02 design, step 3): insert when a note with a TTL is created, upsert
when the TTL changes, delete when the TTL is cleared or the note is deleted.
`note_id` is the PK: one row per note, so an "update" is an upsert
(INSERT ... ON CONFLICT DO UPDATE).
"""

from __future__ import annotations

import sqlite3

# The table is created idempotently in init_db (app/storage/db.py, _NOTE_EXPIRATIONS_DDL).
_EXPIRATIONS_TABLE = "note_expirations"


def upsert(conn: sqlite3.Connection, note_id: int, expires_at: str) -> None:
    """Insert/update the note lifetime in the deletion queue.

    `note_id` is the PK, so one row per note: INSERT ... ON CONFLICT DO UPDATE
    overwrites expires_at when the TTL changes (set in memory_update). Called
    inside an open service transaction.
    """
    conn.execute(
        f"INSERT INTO {_EXPIRATIONS_TABLE} (note_id, expires_at) VALUES (?, ?) "
        "ON CONFLICT(note_id) DO UPDATE SET expires_at = excluded.expires_at",
        (note_id, expires_at),
    )


def delete(conn: sqlite3.Connection, note_id: int) -> None:
    """Remove a row from the deletion queue (TTL cleared / clear / note deleted).

    Idempotent: a missing row matches nothing — the DELETE simply no-ops.
    Called inside an open service transaction.
    """
    conn.execute(
        f"DELETE FROM {_EXPIRATIONS_TABLE} WHERE note_id = ?",
        (note_id,),
    )
