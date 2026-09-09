"""Тесты джобы зачистки просроченных заметок (lsb-0004-02, этап 4).

Фоновый поток раз в 5 минут: SELECT note_id FROM note_expirations
WHERE expires_at <= now(); для каждого id — полное физическое удаление
(notes + notes_chunks + notes_chunks_vec + notes_vec + notes_fts) +
удаление строки из note_expirations. Идемпотентно и безопасно: если заметка
уже удалена (оператором) — просто пропустить, не падать.

Проверяем process_expired_notes() (синхронное ядро петли) — как и другие
юниты воркера (test_worker.py тестируют process_* напрямую).
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder

from app.config import get_settings
from app.services.notes import NoteService
from app.services.worker import (
    EXPIRATION_CLEANUP_INTERVAL_SEC,
    BackgroundWorker,
)
from app.storage import chunks, vectors
from app.storage.db import init_db, session, transaction

# ISO-8601 UTC, как в БД (strftime('%Y-%m-%dT%H:%M:%SZ','now')).
_ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def make_worker(settings) -> BackgroundWorker:
    return BackgroundWorker(settings, HashEmbedder(settings.embedding_dim))


def _insert_note_with_ttl(settings, text: str, expires_at: str) -> int:
    """Создать заметку с чанками/вектором и строкой в note_expirations.

    Возвращает note_id. expires_at — абсолютный ISO-8601 UTC (прошлое для
    просроченной, будущее для непросроченной).
    """
    note_id = NoteService(settings, HashEmbedder(settings.embedding_dim)).save(
        text
    )["id"]
    with session(settings) as conn, transaction(conn):
        # Чанки + их вектора (как сделал бы воркер после save).
        chunk_ids = chunks.replace_note_chunks(
            conn, note_id, [(text, len(text))]
        )
        for cid in chunk_ids:
            chunks.upsert_vector(conn, cid, [0.1] * settings.embedding_dim)
        # Полный вектор заметки.
        vectors.upsert(conn, note_id, [0.2] * settings.embedding_dim, "default")
        # Строка в очереди удаления.
        conn.execute(
            "INSERT INTO note_expirations (note_id, expires_at) VALUES (?, ?)",
            (note_id, expires_at),
        )
    return note_id


def _counts(settings, note_id: int) -> dict[str, int]:
    """Число строк по заметке во всех таблицах (0 = физически удалена)."""
    with session(settings) as conn:
        return {
            "notes": conn.execute(
                "SELECT COUNT(*) FROM notes WHERE id = ?", (note_id,)
            ).fetchone()[0],
            "chunks": conn.execute(
                "SELECT COUNT(*) FROM notes_chunks WHERE note_id = ?",
                (note_id,),
            ).fetchone()[0],
            "chunks_vec": conn.execute(
                "SELECT COUNT(*) FROM notes_chunks_vec WHERE chunk_id IN "
                "(SELECT id FROM notes_chunks WHERE note_id = ?)",
                (note_id,),
            ).fetchone()[0],
            "vec": conn.execute(
                "SELECT COUNT(*) FROM notes_vec WHERE note_id = ?",
                (note_id,),
            ).fetchone()[0],
            "fts": conn.execute(
                "SELECT COUNT(*) FROM notes_fts WHERE rowid = ?", (note_id,)
            ).fetchone()[0],
            "expirations": conn.execute(
                "SELECT COUNT(*) FROM note_expirations WHERE note_id = ?",
                (note_id,),
            ).fetchone()[0],
        }


class TestExpirationCleanup:
    def test_expired_note_fully_deleted(self, settings) -> None:
        """Просроченная заметка удаляется из ВСЕХ таблиц + note_expirations."""
        note_id = _insert_note_with_ttl(settings, "просроченная заметка", "2000-01-01T00:00:00Z")
        # Санity: заметка и все индексы существуют до зачистки.
        assert _counts(settings, note_id) == {
            "notes": 1, "chunks": 1, "chunks_vec": 1, "vec": 1, "fts": 1,
            "expirations": 1,
        }
        worker = make_worker(settings)
        assert worker.process_expired_notes() == 1
        assert _counts(settings, note_id) == {
            "notes": 0, "chunks": 0, "chunks_vec": 0, "vec": 0, "fts": 0,
            "expirations": 0,
        }

    def test_non_expired_note_remains(self, settings) -> None:
        """Непросроченная заметка (expires_at в будущем) остаётся нетронутой."""
        note_id = _insert_note_with_ttl(settings, "живая заметка", "2999-01-01T00:00:00Z")
        worker = make_worker(settings)
        assert worker.process_expired_notes() == 0
        assert _counts(settings, note_id) == {
            "notes": 1, "chunks": 1, "chunks_vec": 1, "vec": 1, "fts": 1,
            "expirations": 1,
        }

    def test_expired_and_live_mixed(self, settings) -> None:
        """Смесь: удаляется только просроченная, живая остаётся."""
        expired = _insert_note_with_ttl(settings, "просроченная", "2000-01-01T00:00:00Z")
        live = _insert_note_with_ttl(settings, "живая", "2999-01-01T00:00:00Z")
        worker = make_worker(settings)
        assert worker.process_expired_notes() == 1
        assert _counts(settings, expired)["notes"] == 0
        assert _counts(settings, live)["notes"] == 1
        assert _counts(settings, live)["expirations"] == 1

    def test_expiration_row_deleted_even_if_note_gone(self, settings) -> None:
        """Идемпотентность: заметка уже удалена (оператором) — не падаем,
        строка из note_expirations всё равно снимается."""
        note_id = _insert_note_with_ttl(settings, "удалённая оператором", "2000-01-01T00:00:00Z")
        # Оператор физически удалил заметку, но строка в очереди осталась.
        with session(settings) as conn, transaction(conn):
            conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        worker = make_worker(settings)
        assert worker.process_expired_notes() == 1  # строка снята
        assert _counts(settings, note_id)["expirations"] == 0

    def test_idempotent_second_run(self, settings) -> None:
        """Повторный прогон после зачистки — не падает, ничего не удаляет."""
        note_id = _insert_note_with_ttl(settings, "просроченная", "2000-01-01T00:00:00Z")
        worker = make_worker(settings)
        assert worker.process_expired_notes() == 1
        assert worker.process_expired_notes() == 0  # очередь пуста
        assert _counts(settings, note_id)["notes"] == 0

    def test_empty_queue_returns_zero(self, settings) -> None:
        """Пустая очередь — 0, без ошибок."""
        worker = make_worker(settings)
        assert worker.process_expired_notes() == 0

    def test_interval_constant(self) -> None:
        """Интервал фиксированный — 5 минут (решение О. 2026-09-09)."""
        assert EXPIRATION_CLEANUP_INTERVAL_SEC == 5 * 60
