"""Тесты SearchService (Фаза 3, шаг 3.3): FTS-сторона, BM25, snippet, hint.

База гибрида (векторная сторона — vec0 ортогональные критерии, деградация,
RRF-слияние — tests/test_search_hybrid.py). Здесь trigram-контракт FTS:
подстроки/словоформы, OR-совпадение любого слова (BUG-001), кавычки, выдача
без полного текста.
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder

from app.config import get_settings
from app.services.notes import NoteService
from app.services.search import MAX_TOP_K, SearchService, SearchValidationError
from app.storage.db import init_db


def _searcher(settings=None) -> SearchService:
    """SearchService с детерминированным HashEmbedder (без сети, ARCH §7)."""
    settings = settings or get_settings()
    return SearchService(settings, HashEmbedder(settings.embedding_dim))


@pytest.fixture
def service() -> tuple[SearchService, NoteService]:
    settings = get_settings()
    init_db(settings)
    return _searcher(settings), NoteService(settings)


def unique(text: str) -> str:
    """Уникальный текст: HashEmbedder не примет нумерованные siblings за дубли.

    Дедуп (Фаза 3) отсекает близкие тексты — тестам счётчиков/пагинации нужны
    гарантированно «разные» заметки; вводим uuid-хвост в текст.
    """
    import uuid

    return f"{text} [{uuid.uuid4().hex[:8]}]"


def long_text(n_chars: int) -> str:
    word = "слово "
    return (word * (n_chars // len(word) + 1))[:n_chars]


class TestMatching:
    def test_substring_and_wordforms(self, service) -> None:
        """Trigram-контракт (REQUIREMENTS §5.4): подстроки ≥3 симв.,
        включая русские словоформы и IP/токены."""
        searcher, notes_service = service
        notes_service.save("Развёртывание TaskFlow выполнено на 192.168.1.50")
        assert searcher.search("развёртыван")["results"][0]["id"] == 1
        assert searcher.search("192.168.1.50")["results"][0]["id"] == 1
        # кейс-независимость токенизатора:
        assert searcher.search("TASKFLOW")["results"][0]["id"] == 1

    def test_multi_word_is_or_of_substrings(self, service) -> None:
        """BUG-001: слова ≥3 симв. — OR подстрок, не AND. Заметка со всеми
        словами выше по BM25, но заметка с частью слов тоже находится:
        AND терял релевантные заметки, если одно слово запроса нигде
        не встречалось («openwebui chat_id» мимо заметки про chat_id)."""
        searcher, notes_service = service
        notes_service.save("Сервис TaskFlow развёрнут на сервере")  # оба слова
        notes_service.save("TaskFlow — это сервис задач")  # только TaskFlow
        notes_service.save("Заметка про котиков")  # ни одного
        notes_service.save("Рецепт борща на вечер")
        notes_service.save("Прогулка по парку")
        hits = searcher.search("TaskFlow сервер")["results"]
        assert [r["id"] for r in hits] == [1, 2]  # оба слова — выше по BM25
        # «deploy» не встречается нигде — релевантные всё равно найдены:
        deployed = searcher.search("TaskFlow deploy")["results"]
        assert {r["id"] for r in deployed} == {1, 2}

    def test_compound_token_parts_searched(self, service) -> None:
        """BUG-001: составной токен запроса ищется и по ≥3-символьным
        частям: «open-webui» находит «Open WebUI» (написание в тексте
        может отличаться от написания в запросе)."""
        searcher, notes_service = service
        notes_service.save("Контейнер Open WebUI слушает порт 3000")
        result = searcher.search("open-webui контейнер")["results"]
        assert [r["id"] for r in result] == [1]

    def test_short_words_dropped_from_expression(self, service) -> None:
        """Слова 1–2 символов отбрасываются (trigram их не видит вообще);
        трёхсимвольные уже ищутся как обычные слова."""
        searcher, notes_service = service
        notes_service.save("Долгосрочная память работает")
        result = searcher.search("ре память")  # "ре" отброшено, ищем "память"
        assert [r["id"] for r in result["results"]] == [1]
        notes_service.save("Или память, или диск")  # теперь "или" можно искать
        assert searcher.search("или память")["results"][0]["id"] == 2

    def test_punctuation_query_is_safe(self, service) -> None:
        """`AND`/кавычки/скобки в запросе — не синтаксис FTS5, а подстроки:
        запрос не роняет MATCH и ищется буквально."""
        searcher, notes_service = service
        notes_service.save('Вызов fts("x") AND (y) * готов')
        result = searcher.search('fts AND (y)')
        assert [r["id"] for r in result["results"]] == [1]  # оба слова есть
        # FTS-операторы в кавычках — обычные символы:
        assert [r["id"] for r in searcher.search('"пусто" OR "хлам"')["results"]] == []

    def test_bm25_ranking_more_hits_first(self, service) -> None:
        """Больше вхождений термина — выше в выдаче."""
        _, notes_service = service
        notes_service.save("TaskFlow в начале и не упоминается больше")  # id=1
        notes_service.save(
            "TaskFlow помянут. TaskFlow снова. TaskFlow в конце"
        )  # id=2, три вхождения
        results = _searcher().search("TaskFlow")
        assert [r["id"] for r in results["results"]] == [2, 1]

    def test_deleted_notes_not_searchable(self, service) -> None:
        """Soft delete прячет заметку и из поиска (FTS-индекс в trash)."""
        searcher, notes_service = service
        notes_service.save("Секрет про TaskFlow")
        notes_service.save("Ещё TaskFlow вариант")
        notes_service.delete(1)
        hits = searcher.search("TaskFlow")["results"]
        assert [r["id"] for r in hits] == [2]


class TestOutput:
    def _first(self, searcher: SearchService, note_id: int) -> dict:
        hits = searcher.search("TaskFlow")["results"]
        return next(r for r in hits if r["id"] == note_id)

    def test_element_contract(self, service) -> None:
        """Ровно ключи FR-1; полного текста в выдаче нет (memory_get адресно)."""
        searcher, notes_service = service
        notes_service.save("Сервис TaskFlow общается", author="gpt")
        hit = self._first(searcher, 1)
        assert set(hit) == {
            "id", "summary", "snippet", "summary_status", "rrf_score",
            "cosine", "created_at", "updated_at", "author",
        }
        assert hit["author"] == "gpt"
        assert hit["cosine"] is None  # Фаза 2 без векторов
        assert hit["summary_status"] == "pending"

    def test_snippet_head_of_text(self, service) -> None:
        """Snippet — первые SNIPPET_CHARS=120 символов текста (ARCH §4.2)."""
        searcher, notes_service = service
        notes_service.save("Сервис TaskFlow " + long_text(300))
        hit = self._first(searcher, 1)
        assert len(hit["snippet"]) == 120
        assert hit["snippet"] == ("Сервис TaskFlow " + long_text(300))[:120]
        notes_service.save("TaskFlow")  # короче лимита — snippet = весь текст
        assert searcher.search("TaskFlow")["results"][0]["snippet"] == "TaskFlow"

    def test_summary_fallback_truncated(self, service) -> None:
        searcher, notes_service = service
        notes_service.save("Сервис TaskFlow " + long_text(250))
        hit = self._first(searcher, 1)
        expected = ("Сервис TaskFlow " + long_text(250))[:200]
        assert hit["summary"] == expected
        assert hit["summary_status"] == "pending"

    def test_rrf_score_single_source_formula(self, service) -> None:
        """rrf_score = 1/(RRF_K + rank), rank с 1; косинуса нет."""
        searcher, notes_service = service
        notes_service.save("TaskFlow один")
        notes_service.save("TaskFlow TaskFlow TaskFlow")  # выше по BM25
        result = searcher.search("TaskFlow")
        scores = [r["rrf_score"] for r in result["results"]]
        assert scores[0] == 1 / (get_settings().rrf_k + 1)
        assert scores[1] == 1 / (get_settings().rrf_k + 2)
        assert all(score > 0 and score <= 1 / 61 for score in scores)

    def test_no_warning_when_semantics_alive(self, service) -> None:
        """Фаза 3: warning только при деградации (отказ кодирования запроса);
        при живой векторизации warning отсутствует."""
        _, notes_service = service
        notes_service.save("TaskFlow заметка")
        result = _searcher().search("TaskFlow")
        assert result["warning"] is None


class TestEmptyAndValidation:
    def test_no_results_hint(self, service) -> None:
        """§5.3: модели не боятся искать — hint вместо ошибки."""
        searcher, _notes_service = service
        result = searcher.search("нетакогослова")
        assert result["results"] == []
        assert "переформулируй" in result["hint"]

    def test_all_words_too_short_hint(self, service) -> None:
        """Отдельный hint: каждое слово <3 символов."""
        searcher, _notes_service = service
        result = searcher.search("и а б")
        assert result["results"] == []
        assert "3 символов" in result["hint"]

    def test_query_length_validated(self, service) -> None:
        searcher, _ = service
        with pytest.raises(SearchValidationError):
            searcher.search("")
        with pytest.raises(SearchValidationError):
            searcher.search("х" * 513)  # > MAX_QUERY_CHARS

    def test_top_k_validation(self, service) -> None:
        searcher, _ = service
        with pytest.raises(SearchValidationError):
            searcher.search("слово", top_k=0)
        with pytest.raises(SearchValidationError):
            searcher.search("слово", top_k=MAX_TOP_K + 1)

    def test_top_k_limits_output(self, service) -> None:
        _, notes_service = service
        for i in range(1, 8):  # 7 подходящих заметок
            notes_service.save(unique(f"TaskFlow догма номер {i}"))
        searcher = _searcher()
        assert len(searcher.search("TaskFlow")["results"]) == 5  # DEFAULT_TOP_K
        assert len(searcher.search("TaskFlow", top_k=3)["results"]) == 3

    def test_default_top_k_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEFAULT_TOP_K", "2")
        get_settings.cache_clear()
        init_db(get_settings())
        from app.services.notes import NoteService as NS

        notes_service = NS(get_settings())
        for i in range(1, 5):
            notes_service.save(unique(f"TaskFlow число {i}"))
        result = _searcher().search("TaskFlow")
        assert len(result["results"]) == 2