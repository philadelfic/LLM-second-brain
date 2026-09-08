"""Миграция title-индекса (lsb-0001-01, шаг 1): схема FTS + вектора.

Эмуляция живой БД v2.1.1 (notes_fts с одной колонкой text, полные вектора
заметок и вектора чанков на месте) → повторный init_db: FTS пересобирается
под (title, text) с переливом, notes_vec дропается (все заметки, включая
trash, → pending), notes_chunks_vec остаётся валидным; идемпотентность —
meta-ключ title_index_version (решение гейта R4 — миграция при старте,
не джоба). Триггеры после миграции синхронизируют и название
(NoteService.save/update), soft delete FTS не трогает (ARCH §4.6).
"""

from __future__ import annotations

import logging
import re
import sqlite3

import pytest
from fakes import FailingEmbedder, HashEmbedder

from app.config import get_settings
from app.services.notes import NoteService
from app.storage import chunks, vectors
from app.storage.db import init_db, session, transaction

# Уникальные подстроки (trigram ищет ≥3 симв.): одна живёт только в title,
# другая — только в text. Совпадения по TITLE_NEEDLE до миграции обязаны
# быть пустыми (индекс только по text) и появиться только после неё.
TITLE_NEEDLE = "квантовариум"
TEXT_NEEDLE = "паровозостроение"

NOTE_A_TEXT = "заметка про ежедневный бэкап сервера"
NOTE_B_TEXT = "план релиза и паровозостроение в приложении"
NOTE_C_TEXT = "легаси-заметка эпохи миграции без названия"


@pytest.fixture
def dim8(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """БД с размерностью 8 (вектора — литеральные, без внешних сервисов)."""
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def _match(conn: sqlite3.Connection, needle: str) -> list[int]:
    """rowid-ы заметок, содержащих подстроку needle (как в test_storage)."""
    rows = conn.execute(
        "SELECT rowid FROM notes_fts WHERE notes_fts MATCH ?",
        (f'"{needle}"',),
    ).fetchall()
    return [row[0] for row in rows]


def _seed_v211_db(settings) -> None:
    """Живая БД образца v2.1.1: 3 заметки ('ok' + полные вектора + чанки
    с векторами), третья — trash и без названия; FTS откачен на старую
    1-колоночную схему, штамп title-миграции стёрт (мета без ключа)."""
    embedder = HashEmbedder(8)
    seeded: list[tuple[int, str, str | None]] = [
        (1, NOTE_A_TEXT, f"Отчёт {TITLE_NEEDLE}"),  # подстрока только в title
        (2, NOTE_B_TEXT, "Сводка по релизам"),  # подстрока только в text
        (3, NOTE_C_TEXT, None),  # легаси: title NULL (миграционные заметки)
    ]
    with session(settings) as conn, transaction(conn):
        for note_id, text, title in seeded:
            conn.execute(
                "INSERT INTO notes (id, text, title, vector_status) "
                "VALUES (?, ?, ?, 'ok')",
                (note_id, text, title),
            )
            vectors.upsert(conn, note_id, embedder.embed(text))
            chunk_ids = chunks.replace_note_chunks(conn, note_id, [(text, 7)])
            chunks.upsert_vector(conn, chunk_ids[0], embedder.embed(text))
        # третья — trash: миграция обязана догнать и её (UPDATE без WHERE)
        conn.execute(
            "UPDATE notes SET deleted_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
            "WHERE id = 3"
        )
        # Ручной откат FTS на схему v2.1.1: одна колонка text, триггеры
        # text-only, перелив из notes без title.
        conn.execute("DROP TRIGGER IF EXISTS notes_fts_ai")
        conn.execute("DROP TRIGGER IF EXISTS notes_fts_au")
        conn.execute("DROP TABLE notes_fts")
        conn.execute(
            "CREATE VIRTUAL TABLE notes_fts USING fts5("
            "  text, content='notes', content_rowid='id', tokenize='trigram')"
        )
        conn.execute(
            "CREATE TRIGGER notes_fts_ai AFTER INSERT ON notes BEGIN "
            "INSERT INTO notes_fts(rowid, text) VALUES (new.id, new.text); END"
        )
        conn.execute(
            "CREATE TRIGGER notes_fts_au AFTER UPDATE OF text ON notes BEGIN "
            "INSERT INTO notes_fts(notes_fts, rowid, text) "
            "VALUES ('delete', old.id, old.text); "
            "INSERT INTO notes_fts(rowid, text) VALUES (new.id, new.text); END"
        )
        conn.execute(
            "INSERT INTO notes_fts(rowid, text) SELECT id, text FROM notes"
        )
        # до миграции штампа нет: БД «не знает» о title-индексе
        conn.execute("DELETE FROM meta WHERE key = 'title_index_version'")


def test_v211_db_migrates_fts_and_vectors(dim8, caplog) -> None:
    """Старая БД → init_db: FTS под (title, text), вектора сброшены в pending,
    чанковые вектора живы, штамп установлен, события миграции в логе."""
    _seed_v211_db(dim8)
    # предусловие: старый индекс ищет по text и НЕ ищет по title
    with session(dim8) as conn:
        assert _match(conn, TITLE_NEEDLE) == []
        assert _match(conn, TEXT_NEEDLE) == [2]

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="app"):
        init_db(get_settings())

    with session(get_settings()) as conn:
        # (a) DDL notes_fts теперь начинается с колонки title.
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='notes_fts'"
        ).fetchone()[0]
        assert re.search(r"fts5\(\s*title", ddl)
        # (b) подстрока из title находится (индекс названия заработал).
        assert _match(conn, TITLE_NEEDLE) == [1]
        # (c) подстрока из text по-прежнему находится.
        assert _match(conn, TEXT_NEEDLE) == [2]
        # (d) все заметки (включая trash id=3) — pending, notes_vec пуст,
        # notes_chunks_vec не тронут (вектор чанка на месте).
        statuses = [
            row[0]
            for row in conn.execute(
                "SELECT vector_status FROM notes ORDER BY id"
            )
        ]
        assert statuses == ["pending", "pending", "pending"]
        assert vectors.count(conn) == 0
        assert chunks.count_vectors(conn) == 3
        chunk_id = chunks.get_note_chunks(conn, 1)[0][0]
        assert chunks.get_vector(conn, chunk_id) == pytest.approx(
            HashEmbedder(8).embed(NOTE_A_TEXT), abs=1e-6
        )
        # (e) штамп миграции установлен и равен «2».
        stamp = conn.execute(
            "SELECT value FROM meta WHERE key = 'title_index_version'"
        ).fetchone()[0]
        assert stamp == "2"
        # триггеры пересозданы под новые колонки.
        triggers = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            )
        }
        assert {"notes_fts_ai", "notes_fts_au"} <= triggers
    # (f) события миграции в логе с reason="title_index" (Часть A + Часть B).
    started = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "reindex_started"
    ]
    done = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "reindex_done"
    ]
    assert started and done
    assert all(record.reason == "title_index" for record in started + done)


def test_migration_is_idempotent(dim8, caplog) -> None:
    """Повторный init_db: повторной перестройки нет, состояние не меняется."""
    _seed_v211_db(dim8)
    init_db(get_settings())  # сама миграция

    caplog.clear()
    init_db(get_settings())  # идемпотентный рестарт
    with session(get_settings()) as conn:
        statuses = [
            row[0]
            for row in conn.execute(
                "SELECT vector_status FROM notes ORDER BY id"
            )
        ]
        assert statuses == ["pending", "pending", "pending"]
        assert vectors.count(conn) == 0
        assert chunks.count_vectors(conn) == 3  # чанковый вектор на месте
        stamp = conn.execute(
            "SELECT value FROM meta WHERE key = 'title_index_version'"
        ).fetchone()[0]
        assert stamp == "2"
    started = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "reindex_started"
    ]
    assert not started  # новых событий миграции нет


def test_triggers_index_title_on_save_and_update(dim8) -> None:
    """Триггеры синхронизируют и название: save с title сразу ищется по нему,
    update с новым title переключает индекс (старое название уходит)."""
    notes = NoteService(dim8, FailingEmbedder())
    nid = notes.save(
        "текст новой заметки про кеш", title=f"Заметка {TITLE_NEEDLE}"
    )["id"]
    with session(dim8) as conn:
        assert _match(conn, TITLE_NEEDLE) == [nid]  # title попал в индекс

    notes.update(
        nid, "текст новой заметки про кеш", title=f"Переименование {TEXT_NEEDLE}"
    )
    with session(dim8) as conn:
        assert _match(conn, TITLE_NEEDLE) == []  # старое название ушло
        assert _match(conn, TEXT_NEEDLE) == [nid]  # новое — попало


def test_soft_delete_keeps_fts_row(dim8) -> None:
    """Soft delete не трогает FTS (ARCH §4.6): строка индекса физически жива."""
    notes = NoteService(dim8, FailingEmbedder())
    nid = notes.save(
        "заметка про индексацию памяти", title="Индексация памяти"
    )["id"]
    assert notes.delete(nid) == {"id": nid, "deleted": True}
    with session(dim8) as conn:
        row = conn.execute(
            "SELECT deleted_at FROM notes WHERE id = ?", (nid,)
        ).fetchone()
        assert row["deleted_at"] is not None
        assert _match(conn, "Индексация") == [nid]  # строка FTS остаётся