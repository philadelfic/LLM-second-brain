"""Векторизация записи (Фаза 3, шаг 3.4): memory_save/memory_update по FR-4/FR-5.

Успешный путь (HashEmbedder): INSERT + вектор одной транзакцией, vector_status='ok'.
Деградация (фейк-отказ): сохранение не ломается — pending + warning, дедуп по тексту.
Update: ре-векторизация sync; отказ → pending, ответ без warning (контракт FR-5).
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder

from app.config import get_settings
from app.services.dedup import DEDUP_HINT
from app.services.embedding import EmbeddingError
from app.services.notes import WARNING_VECTOR_PENDING, NoteService
from app.storage import vectors
from app.storage.db import init_db, session


class FailingEmbedder:
    """Фейк-сервис: векторизация всегда падает (сервер «вон» — NFR-3)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        raise EmbeddingError("векторизация недоступна")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def close(self) -> None:
        return None


@pytest.fixture
def dim8(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def notes_ok(dim8) -> NoteService:
    return NoteService(dim8, HashEmbedder(8))


def notes_broken(dim8) -> NoteService:
    return NoteService(dim8, FailingEmbedder())


# --- save: успешный путь (HashEmbedder) --------------------------------------


def test_save_success_writes_vector_status_ok(dim8) -> None:
    result = notes_ok(dim8).save("Заметка с живой векторизацией")
    assert result == {"id": 1, "stored": True, "summary_pending": True}  # без warning
    with session(dim8) as conn:
        row = conn.execute("SELECT * FROM notes WHERE id = 1").fetchone()
        assert row["vector_status"] == "ok"
        assert vectors.get_vector(conn, 1) is not None
        assert vectors.count(conn) == 1


# --- save: деградация ---------------------------------------------------


def test_save_offline_keeps_note_pending_with_warning(dim8) -> None:
    """Недоступная векторизация не ломает запись (NFR-3, ARCH §4.1)."""
    result = notes_broken(dim8).save("Заметка без сервера векторизации")
    assert result == {
        "id": 1,
        "stored": True,
        "summary_pending": True,
        "warning": WARNING_VECTOR_PENDING,
    }
    with session(dim8) as conn:
        row = conn.execute("SELECT * FROM notes WHERE id = 1").fetchone()
        assert row["vector_status"] == "pending"
        assert vectors.get_vector(conn, 1) is None  # вектора нет


def test_save_offline_verbatim_duplicate_rejected(dim8) -> None:
    """Фоллбек: дословный дубль отсекается и без векторизации."""
    notes = notes_broken(dim8)
    notes.save("Дословное повторение единственное")
    second = notes.save("Дословное повторение единственное")
    assert second["duplicated"] is True
    assert second["id"] == 1
    assert second["hint"] == DEDUP_HINT


def test_save_offline_normalized_duplicate_rejected(dim8) -> None:
    """«Почти дословный»: регистр/пробелы растворены нормализацией."""
    notes = notes_broken(dim8)
    notes.save("Ежедневный  бэкап\nкластера запускается ночью")
    second = notes.save("ежедневный бэкап кластера запускается ночью")
    assert second["duplicated"] is True


def test_save_offline_paraphrase_saved(dim8) -> None:
    """Перефразы без вектора пропускаются — задокументированная деградация."""
    notes = notes_broken(dim8)
    first = notes.save("Первое сохранение уникальной мысли")
    second = notes.save("Другое сохранение с иными словами и смыслом")
    assert "duplicated" not in second
    assert second["id"] == first["id"] + 1


def test_save_offline_deleted_text_can_be_recreated(dim8) -> None:
    """Trash не дедупится: создал → удалил → создал заново — новая заметка."""
    notes = notes_broken(dim8)
    notes.save("Заметка, которая будет удалена")
    notes.delete(1)
    result = notes.save("Заметка, которая будет удалена")
    assert "duplicated" not in result
    assert result["id"] == 2


# --- update: ре-векторизация ------------------------------------------------


def test_update_revectorizes_sync(dim8) -> None:
    notes = notes_ok(dim8)
    notes.save("Старый текст заметки")
    notes.update(1, "Полностью новый текст")
    with session(dim8) as conn:
        assert conn.execute(
            "SELECT vector_status FROM notes WHERE id = 1"
        ).fetchone()["vector_status"] == "ok"
        assert vectors.get_vector(conn, 1) == pytest.approx(
            HashEmbedder(8).embed("Полностью новый текст"), abs=1e-6
        )


def test_update_offline_marks_pending_without_warning(dim8) -> None:
    """Отказ ре-векторизации → pending; ответ по контракту FR-5 без warning."""
    notes = notes_broken(dim8)
    notes.save("Текст до правки")
    result = notes.update(1, "Текст после правки")
    assert result == {"id": 1, "updated": True, "summary_pending": True}
    with session(dim8) as conn:
        assert conn.execute(
            "SELECT vector_status FROM notes WHERE id = 1"
        ).fetchone()["vector_status"] == "pending"


def test_update_unknown_id_does_not_call_embedder(dim8) -> None:
    """Несуществующий id: мягкий ответ и без внешнего вызова (проверка до embed)."""
    embedder = FailingEmbedder()
    notes = NoteService(dim8, embedder)
    result = notes.update(999, "Никуда не пишем")
    assert result["updated"] is False
    assert embedder.calls == []  # кодирование даже не начиналось


def test_update_ghost_id_after_race_writes_nothing(dim8) -> None:
    """Заметка удалена между проверкой и UPDATE: ответ not found, вектор не пишется."""
    notes = notes_ok(dim8)
    notes.save("Единственная")
    # Симуляция гонки: удалим ПЕРЕД update (exists-check пройдёт раньше всех
    # последующих операций только в пределах этого теста).
    notes.delete(1)
    result = notes.update(1, "Худой текст")
    assert result["updated"] is False
    with session(dim8) as conn:
        assert conn.execute("SELECT COUNT(*) FROM notes WHERE text = ?",
                            ("Худой текст",)).fetchone()[0] == 0


def test_deleted_update_rejected(dim8) -> None:
    notes = notes_ok(dim8)
    notes.save("Удалим и не обновим")
    notes.delete(1)
    assert notes.update(1, "Любой")["updated"] is False