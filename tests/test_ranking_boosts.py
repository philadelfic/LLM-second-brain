"""Бусты ранжирования после RRF (релиз 3.4.0, LSBN-F-14).

Детерминированные множители поверх слияния (`app.services.ranking`):
title (точное > префиксное), свежесть (линейный спад за окно), связи
(строка в links). Бусты задают ТОЛЬКО порядок выдачи SearchService.search —
поле rrf_score элемента остаётся сырым RRF (контракт §4.2.6 / шапки F-13).
Выключатель RANKING_BOOSTS_ENABLED=false возвращает «золотой» порядок
без бустов. fuse_rrf и поиск областей не затронуты.
"""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings
from app.services.embedding import EmbeddingError
from app.services.notes import NoteService
from app.services.ranking import (
    FRESHNESS_BOOST_MAX,
    FRESHNESS_WINDOW_DAYS,
    LINKED_BOOST,
    TITLE_EXACT_BOOST,
    TITLE_PREFIX_BOOST,
    apply_boosts,
    freshness_boost,
    title_boost,
    utc_now_iso,
)
from app.services.search import SearchService
from app.storage.db import init_db, session, transaction

DIM = 8
NOW = "2026-10-26T12:00:00Z"


class FailingEmbedder:
    """Эмбеддер-отказ: поиск уходит в FTS-only детерминированно."""

    dim = DIM

    def embed(self, text: str) -> list[float]:
        raise EmbeddingError("кодирование недоступно")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("кодирование недоступно")

    def close(self) -> None:
        return None


def _db_settings(
    tmp_path, monkeypatch: pytest.MonkeyPatch, boosts: bool = True
) -> Settings:
    """Изолированная БД + (опционально) выключенный буст-переключатель."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    if not boosts:
        monkeypatch.setenv("RANKING_BOOSTS_ENABLED", "false")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def dim8(tmp_path, monkeypatch: pytest.MonkeyPatch):
    yield _db_settings(tmp_path, monkeypatch)
    get_settings.cache_clear()  # не оставлять закэшированные настройки соседям


@pytest.fixture
def dim8_no_boosts(tmp_path, monkeypatch: pytest.MonkeyPatch):
    yield _db_settings(tmp_path, monkeypatch, boosts=False)
    get_settings.cache_clear()


# --- юнит: title ------------------------------------------------------------


def test_title_exact_and_prefix_steps() -> None:
    """Ступени буста заголовка: точное совпадение сильнее префикса."""
    assert title_boost("Docker COMPOSE", "  docker compose ") == TITLE_EXACT_BOOST
    assert title_boost("docker", "Docker Compose в бою") == TITLE_PREFIX_BOOST


def test_title_no_match_and_no_title() -> None:
    assert title_boost("docker", "Миграция кластера") == 0.0
    assert title_boost("docker", None) == 0.0
    assert title_boost("docker", "") == 0.0


# --- юнит: свежесть ---------------------------------------------------------


def test_freshness_today_is_maximum() -> None:
    assert freshness_boost(NOW, NOW) == pytest.approx(FRESHNESS_BOOST_MAX)


def test_freshness_declines_linearly() -> None:
    """Половина окна (15 дней из 30) — половина максимума."""
    assert freshness_boost("2026-10-11T12:00:00Z", NOW) == pytest.approx(
        FRESHNESS_BOOST_MAX / 2
    )


def test_freshness_zero_at_and_beyond_window() -> None:
    edge = "2026-09-26T12:00:00Z"  # ровно FRESHNESS_WINDOW_DAYS назад
    assert freshness_boost(edge, NOW) == 0.0
    assert freshness_boost("2026-09-11T12:00:00Z", NOW) == 0.0


def test_freshness_bad_or_missing_stamp_is_zero() -> None:
    assert freshness_boost(None, NOW) == 0.0
    assert freshness_boost("мусор", NOW) == 0.0


def test_freshness_window_constant_fixed_in_canon() -> None:
    """Окно свежести — константа кода (фиксация в каноне, не env)."""
    assert FRESHNESS_WINDOW_DAYS == 30


# --- юнит: apply_boosts -----------------------------------------------------


def _row(
    query: str = "docker compose",
    title: str | None = None,
    updated_at: str | None = None,
    has_links: bool = False,
) -> dict:
    return {
        "query": query,
        "title": title,
        "updated_at": updated_at,
        "has_links": has_links,
    }


def test_title_exact_lifts_note_over_identical_rrf() -> None:
    scores = {1: 0.01, 2: 0.01}
    rows = {
        1: _row(title="Docker Compose"),
        2: _row(title="Заметка про миграцию"),
    }
    boosted = apply_boosts(scores, rows, now=NOW, enabled=True)
    assert boosted[1] == pytest.approx(scores[1] * (1.0 + TITLE_EXACT_BOOST))
    assert boosted[2] == pytest.approx(scores[2])
    assert boosted[1] > boosted[2]


def test_title_prefix_lifts_note_and_exact_beats_prefix() -> None:
    scores = {1: 0.01, 2: 0.01, 3: 0.01}
    rows = {
        1: _row(title="Docker Compose"),  # точное
        2: _row(title="Docker Compose в бою"),  # префикс
        3: _row(title="Миграция кластера"),  # мимо
    }
    boosted = apply_boosts(scores, rows, now=NOW, enabled=True)
    assert boosted[1] == pytest.approx(scores[1] * (1.0 + TITLE_EXACT_BOOST))
    assert boosted[2] == pytest.approx(scores[2] * (1.0 + TITLE_PREFIX_BOOST))
    assert boosted[3] == pytest.approx(scores[3])
    assert boosted[1] > boosted[2] > boosted[3]


def test_freshness_lifts_fresh_note_over_stale() -> None:
    scores = {1: 0.01, 2: 0.01}
    rows = {
        1: _row(updated_at=NOW),  # сегодня
        2: _row(updated_at="2026-08-01T09:00:00Z"),  # за окном
    }
    boosted = apply_boosts(scores, rows, now=NOW, enabled=True)
    assert boosted[1] == pytest.approx(scores[1] * (1.0 + FRESHNESS_BOOST_MAX))
    assert boosted[2] == pytest.approx(scores[2])
    assert boosted[1] > boosted[2]


def test_linked_boost_applies() -> None:
    scores = {1: 0.01, 2: 0.01}
    rows = {1: _row(has_links=True), 2: _row()}
    boosted = apply_boosts(scores, rows, now=NOW, enabled=True)
    assert boosted[1] == pytest.approx(scores[1] * (1.0 + LINKED_BOOST))
    assert boosted[1] > boosted[2]


def test_all_boosts_sum_into_one_multiplier() -> None:
    """Σ бустов: точный title + свежесть + связь → один множитель (1 + Σ)."""
    scores = {1: 0.01}
    rows = {
        1: _row(title="Docker Compose", updated_at=NOW, has_links=True)
    }
    boosted = apply_boosts(scores, rows, now=NOW, enabled=True)
    assert boosted[1] == pytest.approx(scores[1] * 1.40)


def test_missing_row_gets_no_boost() -> None:
    assert apply_boosts({1: 0.01}, {}, now=NOW, enabled=True) == pytest.approx(
        {1: 0.01}
    )


def test_disabled_keeps_golden_order_and_values() -> None:
    """RANKING_BOOSTS_ENABLED=false — значения И порядок ровно как до бустов."""
    scores = {7: 0.02, 3: 0.01}
    rows = {7: _row(), 3: _row(title="Docker Compose", has_links=True)}
    boosted = apply_boosts(scores, rows, now=NOW, enabled=False)
    assert boosted == scores
    assert list(boosted) == list(scores)  # «золотой» порядок ключей сохранён


def test_equal_profile_keeps_relative_order() -> None:
    """Одинаковый профиль бустов — одинаковый множитель, порядок не меняется."""
    scores = {5: 0.01, 9: 0.005, 2: 0.02}
    profile = _row(title="Docker Compose", updated_at="2026-10-20T00:00:00Z")
    rows = {5: dict(profile), 9: dict(profile), 2: dict(profile)}
    boosted = apply_boosts(scores, rows, now=NOW, enabled=True)
    assert list(boosted) == list(scores)
    factors = [boosted[i] / scores[i] for i in scores]
    assert factors == pytest.approx([factors[0]] * 3)


def test_utc_now_iso_matches_note_stamp_format() -> None:
    stamp = utc_now_iso()
    assert len(stamp) == 20 and stamp.endswith("Z")
    assert stamp[10] == "T"


# --- интеграция: SearchService.search ---------------------------------------


def _backdate(settings: Settings, note_id: int, stamp: str) -> None:
    """Прямая правка updated_at (триггер FTS — UPDATE OF text,title — не
    срабатывает; приёмка поиска свежести не задета)."""
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE notes SET updated_at = ? WHERE id = ?", (stamp, note_id)
        )


def test_freshness_boost_flips_rrf_order_and_keeps_raw_score(dim8) -> None:
    """Свежая заметка поднимается над более релевантной по BM25, но старой
    (за окном свежести); rrf_score остаётся сырым RRF — контракт выдачи
    (§4.2.6, шапки F-13) не меняется."""
    notes = NoteService(dim8)
    notes.save("TaskFlow упомянут один раз")  # id=1 — свежая, BM25 ниже
    notes.save("TaskFlow помянут. TaskFlow снова. TaskFlow в конце")  # id=2
    _backdate(dim8, 2, "2026-09-11T00:00:00Z")  # 45 дней назад — за окном
    searcher = SearchService(dim8, FailingEmbedder())
    hits = searcher.search("TaskFlow")["results"]
    assert [hit["id"] for hit in hits] == [1, 2]
    # поле rrf_score — сырой RRF одного FTS-источника (1/(K+rank)), не буст
    k = dim8.rrf_k
    assert sorted(hit["rrf_score"] for hit in hits) == pytest.approx(
        [1 / (k + 2), 1 / (k + 1)]
    )


def test_linked_boost_lifts_note_over_equal_rrf(dim8) -> None:
    """LINKED_BOOST: строка в links (любая сторона) поднимает кандидата."""
    notes = NoteService(dim8)
    notes.save("Kubernetes деплой обновлён")  # id=1
    notes.save("Kubernetes Kubernetes деплой")  # id=2 — BM25 выше
    searcher = SearchService(dim8, FailingEmbedder())
    assert [hit["id"] for hit in searcher.search("Kubernetes")["results"]] == [2, 1]
    with session(dim8) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO links (note_a, note_b, kind, score) VALUES (1, 3, 'cosine', 0.9)"
        )
    assert [hit["id"] for hit in searcher.search("Kubernetes")["results"]] == [1, 2]


def test_disabled_keeps_golden_search_order(dim8_no_boosts) -> None:
    """Выключатель: порядок выдачи ровно как до бустов (BM25: больше
    вхождений выше — золотой порядок до бустов)."""
    notes = NoteService(dim8_no_boosts)
    notes.save("TaskFlow упомянут один раз")  # id=1
    notes.save("TaskFlow помянут. TaskFlow снова. TaskFlow в конце")  # id=2
    searcher = SearchService(dim8_no_boosts, FailingEmbedder())
    assert [
        hit["id"] for hit in searcher.search("TaskFlow")["results"]
    ] == [2, 1]
