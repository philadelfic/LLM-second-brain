"""Тесты SearchService (Фаза 2, Шаг 2.3): FTS-only, BM25, snippet, hint.

REQUIREMENTS FR-1 в ограничении Фазы 2 (векторов нет): rrf_score по одному
источнику, cosine=null, warning «без семантики»; выдача без полного текста.
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.services.notes import NoteService
from app.services.search import MAX_TOP_K, SearchService, SearchValidationError
from app.storage.db import init_db, session


@pytest.fixture
def service() -> tuple[SearchService, NoteService]:
    init_db(get_settings())
    settings = get_settings()
    return SearchService(settings), NoteService(settings)


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

    def test_multi_word_is_and_of_substrings(self, service) -> None:
        """Каждое слово ≥3 симв. должно присутствовать (AND)."""
        searcher, notes_service = service
        notes_service.save("Сервис TaskFlow развёрнут на сервере")
        notes_service.save("TaskFlow — это сервис задач")  # без сервера
        hits = searcher.search("TaskFlow сервер")["results"]
        assert [r["id"] for r in hits] == [1]

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
        results = SearchService(get_settings()).search("TaskFlow")
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

    def test_warning_marks_no_semantics(self, service) -> None:
        """Фаза 2: warning в каждом ответе — семантики пока нет (ARCH §4.2)."""
        _, notes_service = service
        notes_service.save("TaskFlow заметка")
        result = SearchService(get_settings()).search("TaskFlow")
        assert result["warning"]
        assert "семантик" in result["warning"]


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
            notes_service.save(f"TaskFlow догма номер {i}")
        searcher = SearchService(get_settings())
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
            notes_service.save(f"TaskFlow число {i}")
        result = SearchService(get_settings()).search("TaskFlow")
        assert len(result["results"]) == 2