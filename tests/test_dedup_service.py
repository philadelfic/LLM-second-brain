"""DeduplicationService (Фаза 3/8): косинусный порог, FTS-фоллбек, trash.

Метод find_by_cosine (топ-1, порог DEDUP_SIMILARITY) с Этапа 1 вызывается
только фоновой обработкой (вектора строит воркер — тесты догоняют очередь
через vectorize_notes); в синхронном save остался дословный дедуп по тексту.
Перефразы ниже порога сохраняются — их ловит фоновый дедуп: Этап 2.1
находит косинус-кандидатов (find_candidates, порог DEDUP_CANDIDATE_*),
Этап 2.2 сводит дубли, Этап 3 добавит судью-LLM.
"""

from __future__ import annotations

import math

import pytest
from fakes import HashEmbedder, vectorize_notes

from app.config import get_settings
from app.services.dedup import (
    DEDUP_HINT,
    DeduplicationService,
    duplicate_response,
    normalize_text,
)
from app.services.notes import NoteService
from app.storage import vectors
from app.storage.db import init_db, session, transaction

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
    # Фаза 8: вектор строит воркер — для косинус-поиска догоняем очередь.
    assert vectorize_notes(dim8, HashEmbedder(8)) == 1
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
    assert vectorize_notes(dim8, HashEmbedder(8)) == 1  # вектор trash жив
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
    assert vectorize_notes(dim8, HashEmbedder(8)) == 1  # Фаза 8: воркер: вектор
    second = notes.save("Точный дубль сохранить дважды")
    assert second["duplicated"] is True  # дословный дубль — синхронно (SQL/FTS)
    assert second == {
        "duplicated": True,
        "id": first["id"],
        "text": "Точный дубль сохранить дважды",
        "hint": DEDUP_HINT,
    }
    with session(dim8) as conn:
        assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1
        assert vectors.count(conn) == 1  # дубликат вектор не пишет


def test_save_paraphrase_below_threshold_kept(dim8, notes) -> None:
    """Перефраз (косинус < 0.92) — новая заметка (REQUIREMENTS FR-4)."""
    first = notes.save("Первая формулировка мысли о деплое TaskFlow")
    second = notes.save("Совсем иное: кулинарный рецепт борща с пампушками")
    assert "duplicated" not in second
    assert second["id"] == first["id"] + 1
# --- фоновый дедуп: find_candidates (Фаза 8, Этап 2.1) ------------------------


def _vec_at_cos(value: float) -> list[float]:
    """dim8-вектор с cos(vector, E1) = value (две ненулевые L2-оси).

    Косинус vec0 нечувствителен к норме, точность компонент — до 1e-7 (float32
    в notes_vec): сравнения — через pytest.approx.
    """
    return [value, math.sqrt(1.0 - value * value)] + [0.0] * 6


def note_with_vector(
    settings, notes: NoteService, text: str, vector: list[float]
) -> int:
    """Активная заметка с контролируемым вектором (юнит слоя дедупа)."""
    saved = notes.save(text)
    with session(settings) as conn, transaction(conn):
        vectors.upsert(conn, saved["id"], vector)
    return saved["id"]


@pytest.fixture
def dim8n(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """dim8 с кандидат-параметрами Этапа 2: топ-2, порог 0.9 (env §8)."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("DEDUP_CANDIDATE_TOP_N", "2")
    monkeypatch.setenv("DEDUP_CANDIDATE_SIMILARITY", "0.9")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def dim8n_notes(dim8n) -> NoteService:
    """NoteService поверх dim8n — сохранение для note_with_vector."""
    return NoteService(dim8n, HashEmbedder(8), DeduplicationService(dim8n))


def test_find_candidates_threshold_and_order(dim8, notes) -> None:
    """Топ-N по убыванию косинуса ≥ DEDUP_CANDIDATE_SIMILARITY (0.80 дефолт)."""
    note_with_vector(dim8, notes, "заметка-эталон для кандидатов", E1)  # cos 1.0
    note_with_vector(dim8, notes, "кандидат у верхней границы", _vec_at_cos(0.9))
    note_with_vector(dim8, notes, "кандидат ниже порога", _vec_at_cos(0.7))
    found = DeduplicationService(dim8).find_candidates(E1)
    # Порог 0.80 (дефолт) отсекает cos 0.7; порядок — по убыванию близости.
    assert [(note_id, pytest.approx(c, abs=1e-6)) for note_id, c in found] == [
        (1, 1.0),
        (2, 0.9),
    ]
    # exclude чужого id — не меняет выдачу.
    assert DeduplicationService(dim8).find_candidates(E1, exclude_id=999) == found


def test_find_candidates_top_n_from_env(dim8n, dim8n_notes) -> None:
    """DEDUP_CANDIDATE_TOP_N режет выдачу; порог и топ-N — по env."""
    note_with_vector(dim8n, dim8n_notes, "первая заметка банка", E1)
    note_with_vector(dim8n, dim8n_notes, "близкий кандидат", _vec_at_cos(0.95))
    note_with_vector(dim8n, dim8n_notes, "на пороге кандидата", _vec_at_cos(0.9))
    note_with_vector(dim8n, dim8n_notes, "ниже порога кандидата", _vec_at_cos(0.85))
    found = DeduplicationService(dim8n).find_candidates(E1)
    # cos 0.90 ≥ порога 0.9, но топ-N=2: остаётся пара лучших; 0.85 < порога.
    assert [(note_id, pytest.approx(c, abs=1e-6)) for note_id, c in found] == [
        (1, 1.0),
        (2, 0.95),
    ]


def test_find_candidates_exclude_self(dim8, notes) -> None:
    """Свежая заметка не кандидат сама себе: её вектор уже в notes_vec."""
    note_with_vector(dim8, notes, "первая заметка пары", E1)
    dedup = DeduplicationService(dim8)
    assert dedup.find_candidates(E1) == [(1, pytest.approx(1.0, abs=1e-6))]
    assert dedup.find_candidates(E1, exclude_id=1) == []  # единственный хит — self
    note_with_vector(dim8, notes, "вторая заметка той же темы", _vec_at_cos(0.9))
    second = DeduplicationService(dim8).find_candidates(
        _vec_at_cos(0.9), exclude_id=2
    )
    assert [note_id for note_id, _ in second] == [1]


def test_find_candidates_skips_trash(dim8, notes) -> None:
    """Trash (soft delete) не кандидат, хотя его вектор жив (ARCH §3.3)."""
    keeper = note_with_vector(dim8, notes, "живая заметка с вектором E1", E1)
    trashed = note_with_vector(
        dim8, notes, "удалённая заметка с близким вектором", _vec_at_cos(0.9)
    )
    notes.delete(trashed)
    found = DeduplicationService(dim8).find_candidates(E1)
    assert found == [(keeper, pytest.approx(1.0, abs=1e-6))]


def test_find_candidates_trash_does_not_consume_top_n(dim8, notes) -> None:
    """Trash-вектора не вытесняют активных из топ-N: окно KNN расширяется."""
    for index in range(3):  # trash-заметки с векторами-эталонами в банке
        trash_id = note_with_vector(dim8, notes, f"мусорная тема {index}", E1)
        notes.delete(trash_id)
    active = note_with_vector(dim8, notes, "активная заметка-эталон", E1)
    found = DeduplicationService(dim8).find_candidates(E1)
    assert found == [(active, pytest.approx(1.0, abs=1e-6))]


def test_find_candidates_empty_bank(dim8) -> None:
    """Банк векторов пуст — кандидатов нет."""
    assert DeduplicationService(dim8).find_candidates(E1) == []
