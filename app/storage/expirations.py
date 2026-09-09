"""Очередь удаления просроченных заметок — note_expirations (lsb-0004-02).

Слой без доменных правил (как `app.storage.chunks`/`app.storage.vectors`):
только строки таблицы `note_expirations(note_id PK, expires_at)`. Доменная
семантика (когда вставлять/обновлять/удалять, парсинг TTL) — в сервисном
слое `app.services.notes`.

Синхронизация с `notes.expires_at` — на уровне сервиса (постановка lsb-0004-02,
этап 3): вставка при создании заметки с TTL, upsert при смене TTL, удаление
при снятии TTL (clear) или удалении заметки. `note_id` — PK: одна запись на
заметку, поэтому «обновление» — это upsert (INSERT ... ON CONFLICT DO UPDATE).
"""

from __future__ import annotations

import sqlite3

# Таблица создаётся идемпотентно в init_db (app/storage/db.py, _NOTE_EXPIRATIONS_DDL).
_EXPIRATIONS_TABLE = "note_expirations"


def upsert(conn: sqlite3.Connection, note_id: int, expires_at: str) -> None:
    """Вставить/обновить срок жизни заметки в очереди удаления.

    `note_id` — PK, поэтому одна запись на заметку: INSERT ... ON CONFLICT
    DO UPDATE перезаписывает expires_at при смене TTL (set в memory_update).
    Вызывается внутри открытой транзакции сервиса.
    """
    conn.execute(
        f"INSERT INTO {_EXPIRATIONS_TABLE} (note_id, expires_at) VALUES (?, ?) "
        "ON CONFLICT(note_id) DO UPDATE SET expires_at = excluded.expires_at",
        (note_id, expires_at),
    )


def delete(conn: sqlite3.Connection, note_id: int) -> None:
    """Удалить строку из очереди удаления (снятие TTL / clear / удаление заметки).

    Идемпотентно: отсутствующей строки нет — DELETE просто не матчит ничего.
    Вызывается внутри открытой транзакции сервиса.
    """
    conn.execute(
        f"DELETE FROM {_EXPIRATIONS_TABLE} WHERE note_id = ?",
        (note_id,),
    )
