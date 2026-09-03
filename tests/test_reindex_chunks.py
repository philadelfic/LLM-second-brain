"""Пере-чанковка при автореиндексации (Фаза 7, шаг 6): brief §6.

Смена чанк-параметров или модели/размерности поверх живой БД пересчитывает
тексты чанков ВСЕХ заметок (включая trash), дропает notes_chunks_vec (все
чанки → pending, воркер догоняет) и обновляет meta. Вектора полного текста
при чисто чанковой смене остаются валидными (notes_vec не трогается), при
смене модели — дропаются как раньше (b972386). Reuse единичного чанка
≤ CHUNK_SIZE применяется прямо при пере-чанковке: вектор чанка копируется
из notes_vec без кодирования (при смене модели notes_vec пуст — reuse сам
не срабатывает). Согласованности между заметками: дедуп-полный текст —
не зона теста (уже покрыт шагом 3).
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from fakes import HashEmbedder, vectorize_notes
from test_notes_chunks import text_with_tokens

from app.config import get_settings
from app.services.notes import NoteService
from app.services.splitter import count_tokens, token_windows
from app.services.worker import BackgroundWorker
from app.storage import chunks, vectors
from app.storage.db import init_db, session

DEFS = {"chunk_size": 1024, "chunk_overlap": 180, "chunk_min_target": 200}


class NoDedup:
    """Отключённый дедуп: длинные тексты-фикстуры почти идентичны (дедуп
    реализуется и проверяется в шаге 3, воркеру/миграции он шумит)."""

    def find_by_cosine(self, vector: list[float]):
        return None

    def find_by_text(self, text: str, namespace: str | None = None):
        return None


def make_notes(settings, embedder: HashEmbedder = HashEmbedder(8)) -> NoteService:
    return NoteService(settings, embedder, NoDedup())


def set_env(monkeypatch: pytest.MonkeyPatch, tmp_path, **overrides: str) -> None:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("MAX_NOTE_CHARS", "35000")
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path, monkeypatch):
    set_env(monkeypatch, tmp_path)
    settings = get_settings()
    init_db(settings)
    return settings


def test_chunk_size_change_rechunks_all_notes(settings, monkeypatch, caplog) -> None:
    """Смена CHUNK_SIZE: пере-чанковка ВСЕХ (вкл. trash) по новым параметрам,
    вектора чанков сброшены, notes_vec и статусы заметок нетронуты, meta
    обновлена, события reindex_started/done в логе, воркер догоняет."""
    notes = make_notes(settings)
    short_text = text_with_tokens(500)  # 1 чанк и после смены (500 ≤ 512)
    long_text = text_with_tokens(2500)
    trash_text = text_with_tokens(1800)
    short_id = notes.save(short_text)["id"]
    long_id = notes.save(long_text)["id"]
    trash_id = notes.save(trash_text)["id"]
    # Фаза 8: полный вектор строит воркер — ДО удаления trash (очередь
    # обслуживает только активные), чтобы у всех трёх был notes_vec.
    assert vectorize_notes(settings, HashEmbedder(8)) == 3
    notes.delete(trash_id)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="app"):
        monkeypatch.setenv("CHUNK_SIZE", "512")
        get_settings.cache_clear()
        init_db(get_settings())

    new_defs = {
        "chunk_size": 512,
        "chunk_overlap": 180,
        "chunk_min_target": 200,
    }
    expected_long = len(token_windows(count_tokens(long_text), **new_defs))
    expected_trash = len(token_windows(count_tokens(trash_text), **new_defs))
    with session(get_settings()) as conn:
        assert len(chunks.get_note_chunks(conn, short_id)) == 1
        assert chunks.get_note_chunks(conn, short_id)[0][2] == short_text
        assert len(chunks.get_note_chunks(conn, long_id)) == expected_long
        assert len(chunks.get_note_chunks(conn, trash_id)) == expected_trash
        # reuse при пере-чанковке: единственный чанк ≤ CHUNK_SIZE получил
        # копию полного вектора из notes_vec, без вызова кодировщика
        assert chunks.count_vectors(conn) == 1
        assert chunks.get_vector(
            conn, chunks.get_note_chunks(conn, short_id)[0][0]
        ) == pytest.approx(HashEmbedder(8).embed(short_text), abs=1e-6)
        assert chunks.count_pending(conn) == expected_long + expected_trash
        # notes_vec и статусы заметок не тронуты (текст не менялся)
        # notes_vec и статусы активных заметок нетронуты (текст не менялся);
        # у trash статус тоже 'ok' (он таким и был: trash вектора живы)
        assert conn.execute(
            "SELECT COUNT(*) FROM notes WHERE vector_status = 'ok'",
        ).fetchone()[0] == 3
        assert vectors.count(conn) == 3  # полные вектора живы
        stored = {k: v for k, v in conn.execute("SELECT key, value FROM meta")}
        assert stored["chunk_size"] == "512"
    started = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == "reindex_started"
    ]
    done = [r for r in caplog.records if getattr(r, "event", None) == "reindex_done"]
    assert len(started) == len(done) == 1
    assert started[0].from_chunk_size == "1024"  # type: ignore[attr-defined]
    assert started[0].to_chunk_size == "512"  # type: ignore[attr-defined]
    assert done[0].notes_rechunked == 3  # type: ignore[attr-defined]

    # воркер догоняет: только чанки без reuse-векторов
    worker = BackgroundWorker(get_settings(), HashEmbedder(8))
    assert asyncio.run(worker.process_pending_chunks()) == expected_long + expected_trash
    with session(get_settings()) as conn:
        assert chunks.count_pending(conn) == 0


def test_overlap_change_shifts_windows(settings, monkeypatch) -> None:
    """Смена CHUNK_OVERLAP: окна сдвигаются, число чанков меняется."""
    notes = make_notes(settings)
    note_id = notes.save(text_with_tokens(2100))["id"]  # 3 чанка при overlap 180
    with session(settings) as conn:
        assert chunks.count_chunks(conn) == 3
    monkeypatch.setenv("CHUNK_OVERLAP", "0")
    get_settings.cache_clear()
    init_db(get_settings())
    new_defs = {
        "chunk_size": 1024,
        "chunk_overlap": 0,
        "chunk_min_target": 200,
    }
    with session(get_settings()) as conn:
        rows = chunks.get_note_chunks(conn, note_id)
        pending, vectorized = chunks.count_pending(conn), chunks.count_vectors(conn)
    assert len(rows) == len(token_windows(count_tokens(text_with_tokens(2100)), **new_defs)) == 2
    assert pending == 2
    assert vectorized == 0  # 2100 > CHUNK_SIZE — reuse не применим


def test_model_and_chunk_change_together(settings, monkeypatch, caplog) -> None:
    """Смена модели/размерности И чанк-параметров одним рестартом: один
    reindex_started/done, ОБА индекса пересозданы, все заметки pending,
    чанки пере-разбиты, reuse не применим (полных векторов нет)."""
    notes = make_notes(settings)
    note_id = notes.save(text_with_tokens(2500))["id"]
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="app"):
        monkeypatch.setenv("EMBEDDING_DIM", "16")
        monkeypatch.setenv("CHUNK_SIZE", "512")
        get_settings.cache_clear()
        init_db(get_settings())
    new_defs = {"chunk_size": 512, "chunk_overlap": 180, "chunk_min_target": 200}
    expected = len(token_windows(2500, **new_defs))
    with session(get_settings()) as conn:
        ddl = conn.execute("SELECT sql FROM sqlite_master WHERE name='notes_vec'").fetchone()[0]
        assert "float[16]" in ddl
        assert len(chunks.get_note_chunks(conn, note_id)) == expected
        assert chunks.count_vectors(conn) == 0  # reuse невозможен: notes_vec пуст
        assert chunks.count_pending(conn) == expected
        assert conn.execute(
            "SELECT vector_status FROM notes WHERE id = ?", (note_id,)
        ).fetchone()[0] == "pending"
        stored = {k: v for k, v in conn.execute("SELECT key, value FROM meta")}
        assert stored["chunk_size"] == "512"
        assert stored["embedding_dim"] == "16"
    assert [r for r in caplog.records if getattr(r, "event", None) == "reindex_started"]


def test_identity_restart_is_noop(settings, tmp_path, monkeypatch, caplog) -> None:
    """Идемпотентность: рестарт с теми же параметрами не пере-чанкует и
    не пишет reindex-события, догонённые вектора не сбрасываются."""
    notes = make_notes(settings)
    note_id = notes.save(text_with_tokens(2500))["id"]
    assert vectorize_notes(settings, HashEmbedder(8)) == 1  # notes_vec: воркер
    worker = BackgroundWorker(settings, HashEmbedder(8))
    assert asyncio.run(worker.process_pending_chunks()) == 3  # 2500 → 3 чанка
    with session(settings) as conn:
        before = (chunks.count_chunks(conn), chunks.count_vectors(conn))
        pk_vector = vectors.get_vector(conn, note_id)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="app"):
        init_db(settings)  # те же env — ранний выход «всё совпало»
    with session(settings) as conn:
        after = (chunks.count_chunks(conn), chunks.count_vectors(conn))
    assert before == after
    assert pk_vector is not None  # полный вектор не сброшен идемпотентным соком
    assert pk_vector == pytest.approx(HashEmbedder(8).embed(text_with_tokens(2500)), abs=1e-6)
    assert not [r for r in caplog.records if getattr(r, "event", None) == "reindex_started"]


def test_legacy_note_gets_chunks_on_chunk_param_change(
    tmp_path, monkeypatch
) -> None:
    """Заметка эпохи до Фазы 7 (без чанков, полный вектор есть) при смене
    чанк-параметров чанкуется впервые; для многочанковой это только тексты
    (вектора чанков — pending воркеру)."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("MAX_NOTE_CHARS", "35000")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    legacy_text = text_with_tokens(2500)  # 3 чанка при дефолтных параметрах
    with session(settings) as conn:
        conn.execute(
            "INSERT INTO notes (id, text, vector_status) VALUES (?, ?, 'ok')",
            (1, legacy_text),
        )
        vectors.upsert(conn, 1, HashEmbedder(8).embed(legacy_text))
        assert chunks.count_chunks(conn) == 0  # легаси без чанков
    # пере-чанковка по смене CHUNK_SIZE охватывает и легаси
    monkeypatch.setenv("CHUNK_SIZE", "512")
    get_settings.cache_clear()
    init_db(get_settings())
    new_defs = {"chunk_size": 512, "chunk_overlap": 180, "chunk_min_target": 200}
    expected = len(token_windows(2500, **new_defs))
    with session(get_settings()) as conn:
        rows = chunks.get_note_chunks(conn, 1)
        vectorized = chunks.count_vectors(conn)
        pending = chunks.count_pending(conn)
        note_vector = vectors.get_vector(conn, 1)
    assert len(rows) == expected  # чанкуются впервые, по новым параметрам
    assert rows[0][2].startswith(legacy_text[:40])  # тексты — от сплиттера
    assert vectorized == 0  # многочанковая: reuse не применим
    assert pending == expected  # чанковая очередь подхватит
    assert note_vector is not None  # полный вектор легаси не трогается
