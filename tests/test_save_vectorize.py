"""Запись заметок (Фаза 8, Этап 1): save/update мгновенные, векторизация — фон.

Фаза 3 → Фаза 8: save/update БОЛЬШЕ не кодируют текст синхронно. Строка
пишется сразу с vector_status='pending', а notes_vec догоняет фоновый воркер
(process_pending, очередь pending_vector). Фейк-эмбеддер в синхронном пути
не вызывается ни разу (проверяется счётчиком вызовов); косинус-дедуп в
момент записи исчез (переехал в фоновый дедуп Этапа 2) — близкие, но не
дословные тексты обе сохраняются. Синхронно отсекается только дословный
дубль (SQL/FTS) — работает и без Ollama. Ответ save — без warning:
векторизация всегда фоновая, а не «отложена из-за отказа».
"""

from __future__ import annotations

import asyncio
import contextlib
import time

import pytest
from fakes import FailingEmbedder, HashEmbedder, cosine, vectorize_notes

from app.config import get_settings
from app.services.dedup import DEDUP_HINT
from app.services.notes import NoteService
from app.services.worker import BackgroundWorker
from app.storage import vectors
from app.storage.db import init_db, session


@pytest.fixture
def dim8(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def notes_with(dim8, embedder) -> NoteService:
    return NoteService(dim8, embedder)


@pytest.fixture
def fast_dim8(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Как dim8, но PENDING_RETRY_SEC=0 — живой цикл воркера без пауз."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("PENDING_RETRY_SEC", "0")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


# --- save: мгновенная запись без векторизации (Фаза 8, Этап 1) ----------------


def test_save_returns_contract_without_embed(dim8) -> None:
    """Фаза 8: save пишет текст немедленно, кодировщик не зовётся вовсе."""
    embedder = FailingEmbedder()
    notes = notes_with(dim8, embedder)
    result = notes.save("Заметка о мгновенной записи")
    assert result == {"id": 1, "stored": True, "summary_pending": True}  # без warning
    assert embedder.calls == []  # синхронный embed исчез (критерий приёмки)
    with session(dim8) as conn:
        row = conn.execute("SELECT * FROM notes WHERE id = 1").fetchone()
        assert row["vector_status"] == "pending"
        assert vectors.get_vector(conn, 1) is None  # вектора нет
        assert vectors.count(conn) == 0


def test_save_pending_then_worker_vectorizes(dim8) -> None:
    """Очередь pending_vector живая: воркер доводит заметку до 'ok'."""
    notes = notes_with(dim8, HashEmbedder(8))
    text = "Заметка, которую довекторизует воркер"
    notes.save(text)
    assert BackgroundWorker(dim8, HashEmbedder(8)).process_pending() == 1
    with session(dim8) as conn:
        row = conn.execute("SELECT * FROM notes WHERE id = 1").fetchone()
        assert row["vector_status"] == "ok"
        assert vectors.get_vector(conn, 1) == pytest.approx(
            HashEmbedder(8).embed(text), abs=1e-6
        )


def test_update_marks_pending_without_embed(dim8) -> None:
    """Фаза 8: update не кодирует синхронно — pending, без вектора, фон догонит."""
    embedder = FailingEmbedder()
    notes = notes_with(dim8, embedder)
    notes.save("Текст до правки")
    embedder.calls.clear()
    assert notes.update(1, "Текст после правки") == {
        "id": 1,
        "updated": True,
        "summary_pending": True,
    }
    assert embedder.calls == []  # update кодировщик не зовёт
    with session(dim8) as conn:
        row = conn.execute("SELECT * FROM notes WHERE id = 1").fetchone()
        assert row["vector_status"] == "pending"
        assert vectors.get_vector(conn, 1) is None


def test_update_pending_then_worker_vectorizes(dim8) -> None:
    """Ре-векторизация после update — фон: воркер пишет вектор нового текста."""
    notes = notes_with(dim8, HashEmbedder(8))
    notes.save("Старый текст заметки")
    vectorize_notes(dim8, HashEmbedder(8))
    notes.update(1, "Полностью новый текст")
    assert vectorize_notes(dim8, HashEmbedder(8)) == 1
    with session(dim8) as conn:
        row = conn.execute("SELECT * FROM notes WHERE id = 1").fetchone()
        assert row["vector_status"] == "ok"
        assert vectors.get_vector(conn, 1) == pytest.approx(
            HashEmbedder(8).embed("Полностью новый текст"), abs=1e-6
        )


def test_embedding_availability_no_longer_changes_save(dim8) -> None:
    """Доступность Ollama больше не влияет на контракт записи: ответ тот же,
    что и с живым кодировщиком; заметка просто стоит в очереди (NFR-3)."""
    broken = notes_with(dim8, FailingEmbedder())
    result = broken.save("Заметка без сервера векторизации")
    assert result == {"id": 1, "stored": True, "summary_pending": True}  # без warning
    with session(dim8) as conn:
        row = conn.execute("SELECT * FROM notes WHERE id = 1").fetchone()
        assert row["vector_status"] == "pending"
        assert vectors.get_vector(conn, 1) is None


# --- дедуп в синхронном пути ---------------------------------------------------


def test_save_offline_verbatim_duplicate_rejected(dim8) -> None:
    """Дословный дубль отсекается синхронно — SQL/FTS без векторизации."""
    notes = notes_with(dim8, FailingEmbedder())
    notes.save("Дословное повторение единственное")
    second = notes.save("Дословное повторение единственное")
    assert second["duplicated"] is True
    assert second["id"] == 1
    assert second["hint"] == DEDUP_HINT


def test_save_offline_normalized_duplicate_rejected(dim8) -> None:
    """«Почти дословный»: регистр/пробелы растворены нормализацией."""
    notes = notes_with(dim8, FailingEmbedder())
    notes.save("Ежедневный  бэкап\nкластера запускается ночью")
    second = notes.save("ежедневный бэкап кластера запускается ночью")
    assert second["duplicated"] is True


def test_save_no_cosine_dedup_close_texts_both_saved(dim8) -> None:
    """Косинус-дедуп ушёл в фон (Этап 2): близкие, но не дословные тексты
    обе сохраняются. Предусловие: HashEmbedder дал бы паре cosine ≥
    DEDUP_SIMILARITY — прежний save вторую отсёк бы, новый — нет."""
    first_text = "Ежедневный бэкап кластера запускается ночью"
    second_text = "Ежедневный бэкап кластер запускается ночью"
    embedder = HashEmbedder(8)
    # предусловие: старый косинус-дедуп поймал бы (0.9487 ≥ 0.92)...
    assert cosine(embedder.embed(first_text), embedder.embed(second_text)) >= 0.92
    notes = notes_with(dim8, embedder)
    first = notes.save(first_text)
    second = notes.save(second_text)  # словесный дедуп не ловит («кластер»)
    assert "duplicated" not in second
    assert second["id"] == first["id"] + 1


def test_save_offline_deleted_text_can_be_recreated(dim8) -> None:
    """Trash не дедупится: создал → удалил → создал заново — новая заметка."""
    notes = notes_with(dim8, FailingEmbedder())
    notes.save("Заметка, которая будет удалена")
    notes.delete(1)
    result = notes.save("Заметка, которая будет удалена")
    assert "duplicated" not in result
    assert result["id"] == 2


# --- update: краевые случаи (не зависят от векторизации) -----------------------


def test_update_unknown_id_does_not_call_embedder(dim8) -> None:
    """Несуществующий id: мягкий ответ; кодировщик не зовётся (Фаза 8 — и
    в успешном бы не звался, но здесь это видно на грани отказа)."""
    embedder = FailingEmbedder()
    notes = NoteService(dim8, embedder)
    result = notes.update(999, "Никуда не пишем")
    assert result["updated"] is False
    assert embedder.calls == []  # кодирование даже не начиналось


def test_update_ghost_id_after_race_writes_nothing(dim8) -> None:
    """Заметка удалена между проверкой и UPDATE: ответ not found, ничего не пишется."""
    notes = notes_with(dim8, HashEmbedder(8))
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
    notes = notes_with(dim8, HashEmbedder(8))
    notes.save("Удалим и не обновим")
    notes.delete(1)
    assert notes.update(1, "Любой")["updated"] is False


# --- воркер догоняет очередь в живом цикле (для полноты Этапа 1) ----------------


@pytest.mark.asyncio
async def test_run_vectorizes_saved_note(fast_dim8) -> None:
    """Петля воркера догоняет pending_vector записей (save → фон → 'ok')."""
    notes = notes_with(fast_dim8, FailingEmbedder())
    notes.save("текст для живого цикла векторизации")
    worker = BackgroundWorker(fast_dim8, HashEmbedder(8))
    task = asyncio.create_task(worker.run())
    status = None
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with session(fast_dim8) as conn:
            status = conn.execute(
                "SELECT vector_status FROM notes WHERE id = 1"
            ).fetchone()[0]
        if status == "ok":
            break
        await asyncio.sleep(0.01)
    worker.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert status == "ok"

