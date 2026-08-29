"""Гибридный поиск (Фаза 3, шаг 3.3): RRF-слияние, порог, деградация, trash.

ARCH §4.2: vec0 топ-50 (косинус по полным текстам) + FTS5 топ-50 (BM25) →
RRF → отсечение векторных кандидатов порогом → топ top_k. Отказ кодирования
запроса — деградация FTS-only + warning. Вектора в тестах — сценарные
(ортогональные оси): точное управление косинусом вместо грубого HashEmbedder.
"""

from __future__ import annotations

from typing import Any

import pytest
from fakes import HashEmbedder

from app.config import Settings, get_settings
from app.services.embedding import EmbeddingError
from app.services.notes import NoteService
from app.services.search import (
    HINT_NO_RESULTS,
    HINT_SHORT_QUERY,
    WARNING_FTS_ONLY,
    SearchService,
)
from app.storage import vectors
from app.storage.db import init_db, session, transaction

DIM = 8

E1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
E2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class ScriptedEmbedder:
    """Сценарный эмбеддер: очередь векторов или режим отказа (деградация)."""

    def __init__(
        self, dim: int, vectors: list[list[float]] | None = None, fail: bool = False
    ) -> None:
        self.dim = dim
        self._vectors = list(vectors or [])
        self.fail = fail
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if self.fail:
            raise EmbeddingError("кодирование недоступно")
        if not self._vectors:
            return [0.0] * self.dim
        return (
            self._vectors.pop(0) if len(self._vectors) > 1 else self._vectors[0]
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def close(self) -> None:
        return None


@pytest.fixture
def dim8(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """БД с vec0 8-мерной; settings с этим окружением."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def save_with_vector(
    settings: Settings,
    notes: NoteService,
    text: str,
    vector: list[float],
    author: str | None = None,
) -> int:
    """Заметка + вектор вручную: управляемый сценарий близости."""
    note_id = notes.save(text, author)["id"]
    with session(settings) as conn, transaction(conn):
        vectors.upsert(conn, note_id, vector)
    return note_id


def make_searcher(settings: Settings, embedder: Any) -> SearchService:
    return SearchService(settings, embedder)


# --- векторный источник -----------------------------------------------------


def test_vector_finds_semantics_without_shared_terms(dim8) -> None:
    """Ключевое отличие от FTS: запрос не содержит слов заметки, но найден
    вектором (query закодирован тем же вектором, что и заметка)."""
    notes = NoteService(dim8)
    note_id = save_with_vector(
        dim8, notes, "оранжевый кот дремлет на подоконнике", E1
    )
    searcher = SearchService(dim8, ScriptedEmbedder(DIM, [E1]))
    result = searcher.search("штраф за парковку 2026")
    assert [hit["id"] for hit in result["results"]] == [note_id]
    hit = result["results"][0]
    assert hit["cosine"] == pytest.approx(1.0, abs=1e-6)
    assert hit["rrf_score"] == pytest.approx(1 / (dim8.rrf_k + 1))
    assert result["warning"] is None


def test_vector_hit_below_threshold_dropped(dim8) -> None:
    """Косинус < SCORE_THRESHOLD — кандидат не проходит (FR-1).

    Заметка найдена FTS (слово совпадает) — выдача есть, но `cosine` null:
    векторный hit валиден только ≥ порога.
    """
    notes = NoteService(dim8)
    note_id = notes.save("nginx конфигурация реверс-прокси")["id"]
    with session(dim8) as conn, transaction(conn):
        vectors.upsert(conn, note_id, E1)
    # запрос ортогонален вектору заметки (cosine = 0 < порог)
    searcher = SearchService(dim8, ScriptedEmbedder(DIM, [E2]))
    hit = searcher.search("nginx")["results"][0]
    assert hit["id"] == note_id
    assert hit["cosine"] is None
    assert hit["rrf_score"] == pytest.approx(1 / (dim8.rrf_k + 1))


# --- слияние RRF ------------------------------------------------------------


def test_both_sources_sum_ranks(dim8) -> None:
    """Двойной хит: score = 1/(K+1) + 1/(K+1) — больше любого одиночного."""
    notes = NoteService(dim8)
    note_id = save_with_vector(
        dim8, notes, "кластер kubernetes обновлён до 1.29", E1
    )
    # запрос: слова совпадают (FTS rank 1) + тот же вектор (vec rank 1)
    searcher = SearchService(dim8, ScriptedEmbedder(DIM, [E1]))
    hit = searcher.search("кластер kubernetes")["results"][0]
    assert hit["id"] == note_id
    assert hit["cosine"] == pytest.approx(1.0, abs=1e-6)
    assert hit["rrf_score"] == pytest.approx(2 / (dim8.rrf_k + 1))


def test_double_hit_beats_fts_only(dim8) -> None:
    """Заметка с двумя источниками выше заметки только со словом в запросе."""
    notes = NoteService(dim8)
    save_with_vector(dim8, notes, "кластер kubernetes обновлён до 1.29", E1)
    notes.save("кластер kubernetes на голом металле")  # только FTS-хит
    searcher = SearchService(dim8, ScriptedEmbedder(DIM, [E1]))
    hits = searcher.search("кластер kubernetes")["results"]
    assert [hit["id"] for hit in hits] == [1, 2]
    assert hits[0]["rrf_score"] > hits[1]["rrf_score"]


# --- деградация -------------------------------------------------------


def test_degraded_fts_only_with_warning(dim8) -> None:
    """Отказ кодирования запроса → FTS-only + обучающий warning (§5.3)."""
    notes = NoteService(dim8)
    notes.save("Задача: внедрить ежедневный backup кластера")
    searcher = SearchService(dim8, ScriptedEmbedder(DIM, fail=True))
    result = searcher.search("внедрить backup кластера")
    assert [hit["id"] for hit in result["results"]] == [1]
    assert result["warning"]
    assert "семантик" in result["warning"]
    assert all(hit["cosine"] is None for hit in result["results"])
    assert result["warning"] == WARNING_FTS_ONLY


def test_degraded_empty_results_hint(dim8) -> None:
    searcher = SearchService(dim8, ScriptedEmbedder(DIM, fail=True))
    result = searcher.search("нетакогослова")
    assert result["results"] == []
    assert result["hint"] == HINT_NO_RESULTS


def test_degraded_short_query_only_hint(dim8) -> None:
    """Слов <3 символов и векторизация упала — отдельный hint"""
    searcher = SearchService(dim8, ScriptedEmbedder(DIM, fail=True))
    result = searcher.search("и а б")
    assert result["results"] == []
    assert result["hint"] == HINT_SHORT_QUERY


def test_semantic_success_has_no_warning(dim8) -> None:
    notes = NoteService(dim8)
    save_with_vector(dim8, notes, "Привет, это заметка с вектором", E1)
    searcher = SearchService(dim8, ScriptedEmbedder(DIM, [E1]))
    result = searcher.search("любой запрос")  # scripted вернёт E1
    assert result["results"]
    assert result["warning"] is None


# --- trash ------------------------------------------------------------------


def test_soft_deleted_hidden_but_vector_kept(dim8) -> None:
    """Soft delete прячет из выдачи; вектор физически жив (ARCH §3.3)."""
    notes = NoteService(dim8)
    note_id = save_with_vector(dim8, notes, "Секретная заметка про котов", E1)
    notes.delete(note_id)
    searcher = SearchService(dim8, ScriptedEmbedder(DIM, [E1]))
    result = searcher.search("секретная заметка")
    assert result["results"] == []
    with session(dim8) as conn:
        assert vectors.get_vector(conn, note_id) is not None


# --- HashEmbedder: детерминизм между вызовами --------------------------------


def test_hash_embedder_deterministic_between_calls(dim8) -> None:
    """Два независимых поисковика с HashEmbedder дают тот же порядок."""
    notes = NoteService(dim8)
    text = "приборка в серверной по пятницам"
    save_with_vector(dim8, notes, text, HashEmbedder(DIM).embed(text))
    first = SearchService(dim8, HashEmbedder(DIM)).search("приборка в серверной")
    second = SearchService(dim8, HashEmbedder(DIM)).search("приборка в серверной")
    ids_first = [hit["id"] for hit in first["results"]]
    ids_second = [hit["id"] for hit in second["results"]]
    assert ids_first == ids_second == [1]