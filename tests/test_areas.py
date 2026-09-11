"""Обвязка областей (субстрат 3.0.0): нормализация, триграммное сходство, AreaSearch.

Постановка 00, пул 2: юниты на `normalize_key`/`trigram_similarity` и
гибридный поиск области (vec0-KNN + FTS5 BM25 → RRF, фильтр `deleted_at IS
NULL`, дедупликация id, FTS-only деградация + warning). Изоляция проверяется
в обе стороны: поиск области не отдаёт заметки, поиск заметок — области.
"""

from __future__ import annotations

import pytest
from fakes import FailingEmbedder, HashEmbedder

from app.config import get_settings
from app.services.areas import (
    HINT_NO_RESULTS,
    SKILLS_AREA,
    TERMS_AREA,
    USER_FACTS_AREA,
    AreaSearch,
    AreaSearchValidationError,
    normalize_key,
    trigram_similarity,
)
from app.services.notes import NoteService
from app.services.search import HINT_SHORT_QUERY, WARNING_FTS_ONLY, SearchService
from app.storage import area_vectors
from app.storage.db import init_db, session, transaction

DIM = 8


@pytest.fixture
def dim8(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """БД размерности 8 + настройки (вектора — литеральные, без сети)."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def _skill(conn, name: str, description: str = "описание", steps: str = "шаги",
           text: str = "текст", status: str = "ok") -> int:
    cursor = conn.execute(
        "INSERT INTO skills (name, description, steps, text, vector_status) "
        "VALUES (?, ?, ?, ?, ?)",
        (name, description, steps, text, status),
    )
    return int(cursor.lastrowid)


def _term(conn, term: str, context: str, definition: str = "определение",
          status: str = "ok") -> int:
    cursor = conn.execute(
        "INSERT INTO terms (term, term_norm, context, context_norm, definition, "
        "vector_status) VALUES (?, ?, ?, ?, ?, ?)",
        (term, normalize_key(term), context, normalize_key(context), definition, status),
    )
    return int(cursor.lastrowid)


def _fact(conn, name: str, body: str, status: str = "ok") -> int:
    cursor = conn.execute(
        "INSERT INTO user_facts (name, body, vector_status) VALUES (?, ?, ?)",
        (name, body, status),
    )
    return int(cursor.lastrowid)


def _vectorize(conn, spec, row_id: int, text: str) -> None:
    """Вектор записи области — как его пишет петля areas воркера."""
    area_vectors.upsert(
        conn, spec.vec_table, spec.vec_id_column, row_id, HashEmbedder(DIM).embed(text)
    )


# --- normalize_key ----------------------------------------------------------


class TestNormalizeKey:
    """lower → trim → схлопывание пробелов → ё→е (канон lsb-0008/0009)."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  ГЗ   МГУ ", "гз мгу"),
            ("Ёлка", "елка"),
            ("ЕЛКА", "елка"),
            ("уже готово", "уже готово"),  # «уже» не трогаем — там «ж», не «ё»
            ("a\tb\n  c", "a b c"),
            ("Хлеб\u00a0и\u00a0соль", "хлеб и соль"),  # NBSP — пробел для \s
            ("", ""),
        ],
    )
    def test_normalization(self, raw: str, expected: str) -> None:
        assert normalize_key(raw) == expected

    def test_idempotent(self) -> None:
        once = normalize_key("  Ёжик   в  тумане ")
        assert normalize_key(once) == once


# --- trigram_similarity -----------------------------------------------------


class TestTrigramSimilarity:
    def test_bounds_and_symmetry(self) -> None:
        """Значение всегда 0..1 и симметрично по аргументам."""
        samples = [
            "МГУ",
            "студенты МГУ",
            "бухгалтерия",
            "Ёлка",
            "елка",
            "очень длинный контекст про деплой сервиса",
            "аб",
            "",
            "гз",
        ]
        for left in samples:
            for right in samples:
                value = trigram_similarity(left, right)
                assert 0.0 <= value <= 1.0, (left, right, value)
                assert value == pytest.approx(trigram_similarity(right, left))

    def test_normalization_applied(self) -> None:
        """Регистр и ё/е не создают различий; пробелы схлопываются."""
        assert trigram_similarity("МГУ", "мгу") == 1.0
        assert trigram_similarity("Ёлка", "елка") == 1.0
        assert trigram_similarity("ГЗ   МГУ", "гз мгу") == 1.0

    def test_close_contexts_above_term_threshold(self) -> None:
        """Канонный кейс lsb-0008 §3.5: «МГУ» — близкий контекст «студенты МГУ»."""
        assert trigram_similarity("МГУ", "студенты МГУ") >= 0.75

    def test_unrelated_and_degenerate(self) -> None:
        assert trigram_similarity("МГУ", "бухгалтерия") == 0.0
        assert trigram_similarity("", "МГУ") == 0.0
        assert trigram_similarity("", "") == 0.0
        # короче триграммы и не равны — триграмм нет, сходства нет
        assert trigram_similarity("аб", "вг") == 0.0
        # равные короткие строки — тождество
        assert trigram_similarity("аб", "аб") == 1.0

    def test_containment_is_high(self) -> None:
        """Тригаммы одной строки целиком содержатся в другой — близко."""
        assert trigram_similarity("деплой", "деплой сервиса") == 1.0


# --- AreaSearch -------------------------------------------------------------


class TestAreaSearchHybrid:
    def test_both_sources_give_full_rrf_score(self, dim8) -> None:
        """Запись найдена и вектором, и FTS → score = 2/(RRF_K+1)."""
        with session(dim8) as conn, transaction(conn):
            row_id = _skill(conn, "Деплой релиза", "как катить релиз")
            _vectorize(conn, SKILLS_AREA, row_id, "Деплой релиза\nкак катить релиз")
        result = AreaSearch(dim8, SKILLS_AREA, HashEmbedder(DIM)).search("деплой релиза")
        assert [hit["id"] for hit in result["results"]] == [row_id]
        assert result["results"][0]["score"] == pytest.approx(2 / (dim8.rrf_k + 1))
        assert result["warning"] is None

    def test_projection_is_area_specific(self, dim8) -> None:
        """Выдача — проекция области + id/score/cosine (общий помощник)."""
        with session(dim8) as conn, transaction(conn):
            row_id = _term(conn, "ГЗ", "студенты МГУ", "госэкзамен")
            _vectorize(conn, TERMS_AREA, row_id, "ГЗ\nстуденты МГУ\nгосэкзамен")
        hit = AreaSearch(dim8, TERMS_AREA, HashEmbedder(DIM)).search("госэкзамен")[
            "results"
        ][0]
        assert {"id", "term", "context", "definition", "score"} == set(hit)
        assert hit["term"] == "ГЗ"

    def test_soft_deleted_records_hidden(self, dim8) -> None:
        """Мягкое чтение: удалённая запись исчезает, строка и индексы живы."""
        with session(dim8) as conn, transaction(conn):
            row_id = _fact(conn, "Часовой пояс", "Москва")
        assert AreaSearch(dim8, USER_FACTS_AREA, HashEmbedder(DIM)).search("часовой пояс")[
            "results"
        ]
        with session(dim8) as conn, transaction(conn):
            conn.execute(
                "UPDATE user_facts SET deleted_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ?",
                (row_id,),
            )
        result = AreaSearch(dim8, USER_FACTS_AREA, HashEmbedder(DIM)).search("часовой пояс")
        assert result["results"] == []
        assert result["hint"] == HINT_NO_RESULTS
        with session(dim8) as conn:
            # строка и FTS-индекс физически остаются (trash)
            assert conn.execute(
                "SELECT COUNT(*) FROM user_facts_fts WHERE rowid = ?", (row_id,)
            ).fetchone()[0] == 1

    def test_empty_result_is_soft_answer_with_hint(self, dim8) -> None:
        result = AreaSearch(dim8, SKILLS_AREA, HashEmbedder(DIM)).search("нетакогонавыка")
        assert result["results"] == []
        assert result["hint"] == HINT_NO_RESULTS
        assert result["warning"] is None

    def test_vector_gate_score_threshold_keeps_probe_real(self, dim8) -> None:
        """Порог косинуса: vec0-KNN отдаёт k ближайших — нерелевантное режем.

        Без гейта порога (прецедент заметок, SCORE_THRESHOLD) любая непустая
        область отвечала бы на ЛЮБОЙ запрос, и «пусто = такой записи нет»
        (мягкий hint пробы, субстрат §3.5) стало бы недостижимым.
        """
        with session(dim8) as conn, transaction(conn):
            row_id = _skill(conn, "Деплой релиза", "как катить релиз")
            _vectorize(conn, SKILLS_AREA, row_id, "Деплой релиза\nкак катить релиз")
        searcher = AreaSearch(dim8, SKILLS_AREA, HashEmbedder(DIM))
        # связанный запрос: вектор выше порога (+ FTS-совпадение) — хит есть
        assert [hit["id"] for hit in searcher.search("деплой релиза")["results"]] == [row_id]
        # нерелевантный запрос: FTS не совпал, вектор ниже порога — пусто + hint
        result = searcher.search("метеостанция на крыше общежития")
        assert result["results"] == []
        assert result["hint"] == HINT_NO_RESULTS
        assert result["warning"] is None

    def test_short_query_hint_without_expression(self, dim8) -> None:
        """Все слова короче 3 символов: trigram не ищет — отдельный hint.

        Ветка возможна только без векторной стороны (все слова <3 символов),
        поэтому эмбеддер здесь отказывает (деградация к FTS-only).
        """
        result = AreaSearch(dim8, SKILLS_AREA, FailingEmbedder()).search("аб вг")
        assert result["results"] == []
        assert result["hint"] == HINT_SHORT_QUERY
        assert result["warning"] == WARNING_FTS_ONLY

    def test_top_k_limits(self, dim8) -> None:
        with session(dim8) as conn, transaction(conn):
            for index in range(3):
                _skill(conn, f"Навык номер {index}", "про деплой релиза")
        result = AreaSearch(dim8, SKILLS_AREA, HashEmbedder(DIM)).search(
            "навык деплой", top_k=2
        )
        assert len(result["results"]) == 2

    @pytest.mark.parametrize("top_k", [0, 21, -1])
    def test_top_k_validation(self, dim8, top_k: int) -> None:
        with pytest.raises(AreaSearchValidationError, match="top_k"):
            AreaSearch(dim8, SKILLS_AREA, HashEmbedder(DIM)).search("запрос", top_k=top_k)

    def test_query_validation(self, dim8) -> None:
        searcher = AreaSearch(dim8, SKILLS_AREA, HashEmbedder(DIM))
        with pytest.raises(AreaSearchValidationError, match="query"):
            searcher.search("")
        with pytest.raises(AreaSearchValidationError, match="query"):
            searcher.search("x" * (dim8.max_query_chars + 1))


class TestAreaSearchDegradation:
    def test_fts_only_when_embedder_fails(self, dim8) -> None:
        """Отказ эмбеддера — результаты есть, warning про FTS-only (NFR-3)."""
        with session(dim8) as conn, transaction(conn):
            row_id = _skill(conn, "Деплой релиза", "как катить", status="pending")
        result = AreaSearch(dim8, SKILLS_AREA, FailingEmbedder()).search("деплой")
        assert [hit["id"] for hit in result["results"]] == [row_id]
        assert result["warning"] == WARNING_FTS_ONLY
        assert result["results"][0]["score"] == pytest.approx(1 / (dim8.rrf_k + 1))

    def test_fts_only_empty_still_soft(self, dim8) -> None:
        result = AreaSearch(dim8, TERMS_AREA, FailingEmbedder()).search("нетакого")
        assert result["results"] == []
        assert result["warning"] == WARNING_FTS_ONLY
        assert result["hint"] == HINT_NO_RESULTS


class TestAreaIsolation:
    """Изоляция в обе стороны: заметки ↔ области (архитектура субстрата §3.2)."""

    def test_area_search_ignores_notes(self, dim8) -> None:
        with session(dim8) as conn, transaction(conn):
            skill_id = _skill(conn, "Деплой сервиса", "как катить релиз")
            _vectorize(conn, SKILLS_AREA, skill_id, "Деплой сервиса\nкак катить релиз")
        # заметка с теми же словами — в индексах заметок, но не в области
        NoteService(dim8, FailingEmbedder()).save(
            "деплой сервиса релиза", title="Деплой заметки"
        )
        result = AreaSearch(dim8, SKILLS_AREA, HashEmbedder(DIM)).search("деплой сервиса")
        assert [hit["id"] for hit in result["results"]] == [skill_id]
        assert all("description" in hit for hit in result["results"])

    def test_notes_search_ignores_areas(self, dim8) -> None:
        with session(dim8) as conn, transaction(conn):
            skill_id = _skill(conn, "Деплой сервиса", "как катить релиз")
            _vectorize(conn, SKILLS_AREA, skill_id, "Деплой сервиса\nкак катить релиз")
        note_id = NoteService(dim8, FailingEmbedder()).save(
            "деплой сервиса релиза", title="Деплой заметки"
        )["id"]
        results = SearchService(dim8, HashEmbedder(DIM)).search("деплой сервиса")["results"]
        assert [hit["id"] for hit in results] == [note_id]

    def test_areas_do_not_leak_into_each_other(self, dim8) -> None:
        with session(dim8) as conn, transaction(conn):
            skill_id = _skill(conn, "Деплой сервиса", "как катить релиз")
            term_id = _term(conn, "деплой", "сервиса", "катить релиз")
            fact_id = _fact(conn, "Деплой сервиса", "как катить релиз")
            _vectorize(conn, SKILLS_AREA, skill_id, "Деплой сервиса\nкак катить релиз")
        skills_hits = AreaSearch(dim8, SKILLS_AREA, HashEmbedder(DIM)).search(
            "деплой сервиса"
        )["results"]
        assert [hit["id"] for hit in skills_hits] == [skill_id]
        terms_hits = AreaSearch(dim8, TERMS_AREA, HashEmbedder(DIM)).search(
            "деплой сервиса"
        )["results"]
        assert [hit["id"] for hit in terms_hits] == [term_id]
        facts_hits = AreaSearch(dim8, USER_FACTS_AREA, HashEmbedder(DIM)).search(
            "деплой сервиса"
        )["results"]
        assert [hit["id"] for hit in facts_hits] == [fact_id]
