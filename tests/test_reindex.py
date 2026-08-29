"""Скрипт переиндексации (Фаза 3): сброс vec0, новая размерность, очередь.

Юнит-тесты без сети: живая векторизация в тестовой среде недоступна — скрипт
останавливается на шаге «все в pending» (NFR-3-безопасный путь); догон живой
векторизации — в integration (test_integration_live.py, если Ollama доступна).
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder

from app.config import get_settings
from app.services.notes import NoteService
from app.storage import vectors
from app.storage.db import init_db, session
from scripts.reindex import main


@pytest.fixture
def dim4(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """БД, созданная под старую размерность 4, с двумя векторизованными заметками."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "4")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    notes = NoteService(settings, HashEmbedder(4))
    notes.save("заметка при старой размерности")
    notes.save("вторая заметка при старой размерности")
    return settings


def test_reindex_rebuilds_table_at_new_dim(dim4, monkeypatch) -> None:
    """БД 4-мерных векторов + EMBEDDING_DIM=8: reindex чинит гейт старта.

    После скрипта: notes_vec float[8], все заметки pending (вектора сброшены),
    init_db с новым dim проходит без отказа, заметки сохранены.
    """
    assert NoteService(dim4).get([1, 2])["notes"]  # данные до переиндексации

    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()  # теперь 8
    assert main(["--yes"]) == 0

    with session(settings) as conn:
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'notes_vec'"
        ).fetchone()[0]
        assert "float[8]" in ddl  # таблица пересоздана под новую размерность
        statuses = conn.execute("SELECT vector_status FROM notes").fetchall()
        assert all(row[0] == "pending" for row in statuses)
        assert vectors.count(conn) == 0  # старые вектора удалены
    init_db(settings)  # гейт старта теперь проходит
    assert NoteService(settings).get([1, 2])["notes"]  # данные не потеряны


def test_reindex_same_dim_rebuilds_index(dim4) -> None:
    """Повторный вызов при неизменной размерности — тоже безопасен."""
    assert main(["--yes"]) == 0
    init_db(dim4)  # согласовано, отказов нет


def test_reindex_missing_db(tmp_path, monkeypatch) -> None:
    """Нет файла БД — доходчивый отказ (код 2), а не traceback."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notexist.db"))
    get_settings.cache_clear()
    assert main(["--yes"]) == 2


def test_reindex_confirmation_aborts(dim4) -> None:
    """Без --yes и с отказным ответом в prompt — БД полностью нетронута."""
    import builtins

    builtins_input = builtins.input
    builtins.input = lambda _prompt: "n"
    try:
        assert main([]) == 1  # отменено оператором
    finally:
        builtins.input = builtins_input
    with session(dim4) as conn:
        # индекс цел: таблица на месте, вектора не сброшены
        assert vectors.count(conn) == 2
        assert NoteService(dim4).get([1])["notes"]


def test_reindex_offline_leaves_pending_queue(dim4, monkeypatch) -> None:
    """Живая Ollama недоступна: заметки остаются pending, код 0 (NFR-3)."""
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    assert main(["--yes"]) == 0
    with session(settings) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM notes WHERE vector_status = 'pending'"
        ).fetchone()[0] == 2