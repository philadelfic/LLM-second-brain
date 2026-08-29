"""DeduplicationService (Фаза 3): косинусный порог, FTS-фоллбек, trash.

REQUIREMENTS FR-4: близость ≥ DEDUP_SIMILARITY → дубликат; перефразы ниже
порога — сохраняются. Фоллбек без векторизации ловит дословные повторы
(нормализация регистра/пробелов), перефразы пропускаются.
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder

from app.config import get_settings
from app.services.dedup import (
    DEDUP_HINT,
    DeduplicationService,
    duplicate_response,
    normalize_text,
)
from app.services.notes import NoteService
from app.storage import vectors
from app.storage.db import init_db, session

E1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
E2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.fixture
def dim8(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def notes(dim8) -> NoteService:
    """NoteService с HashEmbedder — успешный путь векторизации."""
    return NoteService(dim8, HashEmbedder(8), DeduplicationService(dim8))


# --- нормализация --------------------------------------------------------


def test_normalize_case_and_spaces() -> None:
    """Свёртка пробелов и регистра — единственная нормализация дедупа."""
    assert normalize_text("  Ежедневный   бэкап\nкластера  ") == (
        "ежедневный бэкап кластера"
    )


def test_duplicate_response_shape() -> None:
    response = duplicate_response({"id": 7, "text": "текст"})
    assert response == {
        "duplicated": True,
        "id": 7,
        "text": "текст",
        "hint": DEDUP_HINT,
    }


# --- косинусный дедуп --------------------------------------------------------


def test_find_by_cosine_above_threshold(dim8, notes) -> None:
    notes.save("приветственная заметка про деплой", author="m1")
    dedup = DeduplicationService(dim8)
    found = dedup.find_by_cosine(HashEmbedder(8).embed("приветственная заметка про деплой"))
    assert found is not None
    assert found["id"] == 1


def test_find_by_cosine_orthogonal_is_not_dup(dim8, notes) -> None:
    notes.save("заметка совсем другого смысла")
    dedup = DeduplicationService(dim8)
    assert dedup.find_by_cosine(E2) is None  # ортогональный вектор далеко


def test_find_by_cosine_empty_bank(dim8) -> None:
    assert DeduplicationService(dim8).find_by_cosine(E1) is None


def test_trash_hit_is_not_duplicate(dim8, notes) -> None:
    """Вектор в trash жив (ARCH §3.3), но дедуп читает только активные."""
    notes.save("уникальная заметка для удаления")
    notes.delete(1)
    dedup = DeduplicationService(dim8)
    query = HashEmbedder(8).embed("уникальная заметка для удаления")
    assert dedup.find_by_cosine(query) is None
    # а FTS-фоллбек тоже не ловит удалённый текст
    assert dedup.find_by_text("уникальная заметка для удаления") is None


# --- FTS-фоллбек ------------------------------------------------------------


def test_find_by_text_exact(dim8, notes) -> None:
    notes.save("Ежедневный бэкап кластера в 03:00")
    found = DeduplicationService(dim8).find_by_text("Ежедневный бэкап кластера в 03:00")
    assert found is not None and found["id"] == 1


def test_find_by_text_normalized_spaces_and_case(dim8, notes) -> None:
    """Нормализация: регистр и пробельные колебания — не различие."""
    notes.save("Ежедневный  бэкап\nкластера в 03:00")
    found = DeduplicationService(dim8).find_by_text("ежедневный бэкап кластера в 03:00")
    assert found is not None and found["id"] == 1


def test_find_by_text_short_exact_sql(dim8, notes) -> None:
    """Текст короче 3 символов: trigram слеп — ловит SQL-равенство."""
    notes.save("он")
    assert DeduplicationService(dim8).find_by_text("он") is not None
    assert DeduplicationService(dim8).find_by_text("она") is None  # другой текст


def test_find_by_text_no_match(dim8, notes) -> None:
    notes.save("Тема А полностью уникальна тут")
    assert DeduplicationService(dim8).find_by_text("совсем другая формулировка") is None


# --- интеграция с save (успешный путь) ---------------------------------------


def test_save_duplicate_rejected_with_existing_info(dim8, notes) -> None:
    first = notes.save("Точный дубль сохранить дважды")
    second = notes.save("Точный дубль сохранить дважды")  # HashEmbedder: cosine 1.0
    assert second == {
        "duplicated": True,
        "id": first["id"],
        "text": "Точный дубль сохранить дважды",
        "hint": DEDUP_HINT,
    }
    with session(dim8) as conn:
        assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1
        assert vectors.count(conn) == 1  # дубликат вектор не записывает


def test_save_paraphrase_below_threshold_kept(dim8, notes) -> None:
    """Перефраз (косинус < 0.92) — новая заметка (REQUIREMENTS FR-4)."""
    first = notes.save("Первая формулировка мысли о деплое TaskFlow")
    second = notes.save("Совсем иное: кулинарный рецепт борща с пампушками")
    assert "duplicated" not in second
    assert second["id"] == first["id"] + 1