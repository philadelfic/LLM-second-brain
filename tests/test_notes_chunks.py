"""NoteService и чанки (Фаза 7, шаг 3): save/update/delete поверх notes_chunks.

Проверяются (brief §6): чанковка в save/update (без Ollama для чанков),
reuse единичного чанка (на заметку ≤ CHUNK_SIZE — ровно ОДИН вызов
кодировщика: «1-чанная заметка ≠ дубликат чанка-векторов»), pending по
анти-джойну для многочанковых заметок, замена чанков при update (в т.ч. при
отказе ре-векторизации), сохранение чанков trash при soft delete, неприкосновенность
чанков при дедуп-отказе. Свечение текстов чанков против сплиттера — в тестах
размерности/схемы (test_chunks_storage), здесь — доменные правила.

MAX_NOTE_CHARS поднят до боевого значения compose (20000, решение О.
2026-08-29) — иначе длинные (>1024 токенов) заметки не влезают в §8-дефолт
2000 симв.; это же закрепляет production-профиль в тесте.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fakes import FailingEmbedder, HashEmbedder

from app.config import get_settings
from app.services.notes import NoteService
from app.services.splitter import count_tokens, encoding, split_text, token_windows
from app.storage import chunks, vectors
from app.storage.db import init_db, session

PROD_MAX_NOTE_CHARS = "20000"

# Дефолты brief §4.
DEFS = {"chunk_size": 1024, "chunk_overlap": 180, "chunk_min_target": 200}

RU_SENTENCES = (
    "LLM Second Brain — self-hosted MCP-сервер долговременной памяти. "
    "Сервис хранит заметки в SQLite и отдаёт их моделям через инструменты. "
    "Векторизация идёт через внешнюю Ollama с моделью qwen3-embedding:8b. "
    "Полнотекстовый поиск держит русские словоформы на токенизаторе trigram. "
    "Слияние источников делает Reciprocal Rank Fusion с константой 60. "
    "Суммаризация работает в фоне и не блокирует запись заметки. "
)

_SOURCE_TOKENS = encoding().encode(RU_SENTENCES * 400)


def text_with_tokens(n_tokens: int) -> str:
    """Текст ровно из n_tokens токенов повторённой русской фикстуры."""
    return encoding().decode(_SOURCE_TOKENS[:n_tokens])


class CountingHashEmbedder(HashEmbedder):
    """HashEmbedder со счётчиком кодировок: reuse чанка не должен звать его.

    Учёт только в embed_texts (единая точка кодирования в EmbeddingService):
    embed -> embed_texts, иначе один вызов учтётся дважды."""

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.texts: list[str] = []

    def embed(self, text: str) -> list[float]:
        # не дублируем учёт: запись здесь есть и в embed_texts
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return super().embed_texts(texts)


@pytest.fixture(autouse=True)
def _prod_note_limit(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """MAX_NOTE_CHARS=20000 (боевой compose): многочанковые заметки влезают."""
    monkeypatch.setenv("MAX_NOTE_CHARS", PROD_MAX_NOTE_CHARS)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def service() -> tuple[NoteService, CountingHashEmbedder]:
    settings = get_settings()
    init_db(settings)
    embedder = CountingHashEmbedder(settings.embedding_dim)
    return NoteService(settings, embedder), embedder


class TestSaveChunking:
    def test_small_note_single_chunk_reuses_note_vector(
        self, service: tuple[NoteService, CountingHashEmbedder]
    ) -> None:
        """Бриф §6: 1 чанк ≤ CHUNK_SIZE → вектор чанка = полный вектор заметки;
        кодировщик отработал ровно ОДИН раз (не дважды — не дубль-вызов)."""
        svc, embedder = service
        text = "Короткая заметка о чанковой механике"
        note_id = svc.save(text)["id"]
        with session(get_settings()) as conn:
            rows = chunks.get_note_chunks(conn, note_id)
            note_vector = vectors.get_vector(conn, note_id)
            chunk_vector = chunks.get_vector(conn, rows[0][0])
            pending = chunks.count_pending(conn)
        assert len(rows) == 1
        assert rows[0][2] == text  # чанк == полный текст
        assert rows[0][3] == count_tokens(text)
        assert pending == 0
        assert chunk_vector == pytest.approx(note_vector, abs=1e-7)
        assert embedder.texts == [text]  # только полный текст, чанк не кодировался

    def test_long_note_chunks_are_pending_without_extra_encoding(
        self, service
    ) -> None:
        """Многочанковая заметка: чанки в notes_chunks без векторов (pending
        — их полечит воркер, шаг 5); Ollama в save позвана один раз."""
        svc, embedder = service
        text = text_with_tokens(3000)
        note_id = svc.save(text)["id"]
        windows = token_windows(3000, **DEFS)
        with session(get_settings()) as conn:
            rows = chunks.get_note_chunks(conn, note_id)
            pending, vectorized = chunks.count_pending(conn), chunks.count_vectors(conn)
        assert len(rows) == len(windows) == 4
        assert [row[2] for row in rows] == [
            c.text
            for c in split_text(text, **DEFS)  # type: ignore[arg-type]
        ]
        assert [row[3] for row in rows] == [end - start for start, end in windows]
        assert pending == 4
        assert vectorized == 0  # вектора чанков — воркеру
        assert embedder.texts == [text]  # кодировщик: только полный текст

    def test_merged_single_chunk_above_chunk_size_is_not_reused(self, service) -> None:
        """1025 токенов → 1 чанк, но > CHUNK_SIZE — reuse не применяется
        (условие брифа: «1 чанк И текст ≤ CHUNK_SIZE»), чанк в pending."""
        svc, _embedder = service
        text = text_with_tokens(1025)
        note_id = svc.save(text)["id"]
        with session(get_settings()) as conn:
            rows = chunks.get_note_chunks(conn, note_id)
            assert len(rows) == 1
            assert rows[0][3] == 1025  # легитимно > CHUNK_SIZE
            assert chunks.count_pending(conn) == 1
            assert chunks.count_vectors(conn) == 0  # не реиспользовал полный

    def test_dedup_hit_writes_no_chunks(self, service) -> None:
        """Дедуп на полном тексте отбивает дубль ДО записи чанков."""
        svc, _embedder = service
        text = text_with_tokens(1200)  # ~2 чанка
        first = svc.save(text)
        second = svc.save(text)
        assert second["duplicated"] is True
        assert second["id"] == first["id"]
        with session(get_settings()) as conn:
            assert chunks.count_chunks(conn) == 2  # только от первой записи

    def test_vectorization_failure_keeps_chunks_pending(self, service) -> None:
        """NFR-3: отказ Ollama — заметка+чанки сохранены, всё в pending."""
        settings = get_settings()
        init_db(settings)
        failing = NoteService(settings, FailingEmbedder())
        text = text_with_tokens(1500)  # ~2 чанка
        result = failing.save(text)
        assert result["stored"] is True
        assert "warning" in result
        with session(settings) as conn:
            assert chunks.count_chunks(conn) == len(
                token_windows(count_tokens(text), **DEFS)
            )
            assert chunks.count_vectors(conn) == 0
            assert vectors.count(conn) == 0
            row = conn.execute(
                "SELECT vector_status FROM notes WHERE id = ?", (result["id"],)
            ).fetchone()
        assert row[0] == "pending"


class TestUpdateChunking:
    def test_update_long_to_long_replaces_chunks(self, service) -> None:
        """Update: старые чанки удалены, новые на их месте (idx с нуля)."""
        svc, _embedder = service
        text_old, text_new = text_with_tokens(3000), text_with_tokens(5000)
        note_id = svc.save(text_old)["id"]
        assert svc.update(note_id, text_new)["updated"] is True
        windows = token_windows(5000, **DEFS)
        with session(get_settings()) as conn:
            rows = chunks.get_note_chunks(conn, note_id)
            pending, vectorized = chunks.count_pending(conn), chunks.count_vectors(conn)
        assert len(rows) == len(windows)
        assert [row[2] for row in rows] == [
            c.text for c in split_text(text_new, **DEFS)  # type: ignore[arg-type]
        ]
        assert [row[3] for row in rows] == [end - start for start, end in windows]
        assert pending == len(rows)
        assert vectorized == 0  # старых reuse-векторов нет

    def test_update_to_small_note_reuses_new_vector(self, service) -> None:
        """Длинная → короткая: новый полный вектор переиспользуется чанком."""
        svc, embedder = service
        note_id = svc.save(text_with_tokens(3000))["id"]
        new_text = "Обновлённая короткая заметка"
        assert svc.update(note_id, new_text)["updated"] is True
        with session(get_settings()) as conn:
            rows = chunks.get_note_chunks(conn, note_id)
            note_vector = vectors.get_vector(conn, note_id)
            chunk_vector = chunks.get_vector(conn, rows[0][0])
            pending = chunks.count_pending(conn)
        assert len(rows) == 1 and rows[0][2] == new_text
        assert pending == 0
        assert chunk_vector == pytest.approx(note_vector, abs=1e-7)
        # кодировщик: полные тексты заметок — и никаких чанков
        assert embedder.texts == [text_with_tokens(3000), new_text]

    def test_update_with_failing_vectorization_replaces_chunks(
        self, service
    ) -> None:
        """Отказ ре-векторизации при update не откатывает замену чанков."""
        svc, _embedder = service
        note_id = svc.save("маленькая, будет заменена большой")["id"]
        settings = get_settings()
        failing = NoteService(settings, FailingEmbedder())
        text = text_with_tokens(1500)
        assert failing.update(note_id, text)["updated"] is True
        with session(settings) as conn:
            rows = chunks.get_note_chunks(conn, note_id)
            vectorized = chunks.count_vectors(conn)
            row = conn.execute(
                "SELECT vector_status FROM notes WHERE id = ?", (note_id,)
            ).fetchone()
        assert len(rows) == len(token_windows(1500, **DEFS))
        assert vectorized == 0
        assert row["vector_status"] == "pending"


class TestDeleteChunking:
    def test_soft_delete_keeps_chunks_and_vectors(self, service) -> None:
        """FR-6/trash: soft delete НЕ трогает чанки (undo вернёт их живыми)."""
        svc, _embedder = service
        text = text_with_tokens(3000)
        note_id = svc.save(text)["id"]
        assert svc.delete(note_id)["deleted"] is True
        with session(get_settings()) as conn:
            chunked = chunks.count_chunks(conn)
            vectorized = chunks.count_vectors(conn)
            note = conn.execute(
                "SELECT deleted_at FROM notes WHERE id = ?", (note_id,)
            ).fetchone()
        assert chunked == len(token_windows(3000, **DEFS))
        assert vectorized == 0  # многочанковая: только pending
        assert note[0] is not None

    def test_small_note_vector_survives_soft_delete(self, service) -> None:
        svc, _embedder = service
        text = "Маленькая заметка для удаления"
        note_id = svc.save(text)["id"]
        svc.delete(note_id)
        with session(get_settings()) as conn:
            rows = chunks.get_note_chunks(conn, note_id)
            vectors_after = chunks.count_vectors(conn)
        assert len(rows) == 1  # чанк trash жив
        assert vectors_after == 1  # reuse-вектор жив (trash сохраняет вектора)


def chunk_status(row) -> str:
    """Читаемая выборка статуса в тестах — устарела, читаем row напрямую."""
    return row["vector_status"]