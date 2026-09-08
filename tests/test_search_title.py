"""Тесты title-режима поиска (lsb-0001-01, Шаг 3): SearchService.search_title.

Контракт (FR-2.1/FR-2.2 + решения D2/D3 владельца): строго по названиям
(подстрока в text не матчится — D3), компактная выдача ровно из пяти полей,
ступеньки score (1.0 — название совпало с запросом целиком, 0.5 — подстрока;
D2), порядок (точные выше; среди частичных — вхождение ближе к началу выше;
tie — updated_at DESC → id DESC), регистронезависимость для кириллицы И
латиницы; trash и title=NULL исключены; ns-фильтр — как в search(); трigram-
правило «слова ≥3 симв.» не применяется (это не FTS). Веса bm25 (FR-1.2):
совпадение подстроки в title выше того же совпадения в text.
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder

from app.config import get_settings
from app.services.namespaces import NamespaceError, NamespaceService
from app.services.notes import NoteService
from app.services.search import (
    HINT_NO_RESULTS_TITLE,
    MAX_TOP_K,
    SearchService,
    SearchValidationError,
)
from app.storage.db import init_db, session, transaction


def _searcher(settings=None) -> SearchService:
    """SearchService с детерминированным HashEmbedder (без сети, ARCH §7)."""
    settings = settings or get_settings()
    return SearchService(settings, HashEmbedder(settings.embedding_dim))


@pytest.fixture
def service() -> tuple[SearchService, NoteService]:
    settings = get_settings()
    init_db(settings)
    return _searcher(settings), NoteService(settings)


def set_meta(
    note_id: int,
    *,
    updated_at: str | None = None,
    summary: str | None = None,
) -> None:
    """Прямая правка notes: детерминированный updated_at / готовое суммари.

    updated_at правится только в notes (триггер FTS — UPDATE OF text, title —
    не срабатывает, индекс не трогается); summary + summary_status='ok' —
    имитация отработавшего суммаризатора.
    """
    fields: list[str] = []
    params: list[object] = []
    if updated_at is not None:
        fields.append("updated_at = ?")
        params.append(updated_at)
    if summary is not None:
        fields.append("summary = ?, summary_status = 'ok'")
        params.append(summary)
    with session(get_settings()) as conn, transaction(conn):
        conn.execute(
            f"UPDATE notes SET {', '.join(fields)} WHERE id = ?",
            (*params, note_id),
        )


class TestMatching:
    def test_substring_in_title_found(self, service) -> None:
        """Подстрока названия находится (кириллица и латиница)."""
        searcher, notes_service = service
        notes_service.save("План работ по кластеру", title="Развёртывание TaskFlow")
        hits = searcher.search_title("развёртыван")["results"]
        assert [hit["id"] for hit in hits] == [1]

    def test_case_insensitive_both_directions(self, service) -> None:
        """Регистронезависимость в обе стороны: ЗАГЛАВНЫЙ запрос против
        строчного названия и наоборот (SQLite lower() кириллицу не понижает —
        потому и сопоставление на Python)."""
        searcher, notes_service = service
        notes_service.save("Текст раз", title="Обновление сервера")
        notes_service.save("Текст два", title="open webui настройка")
        hit = searcher.search_title("ОБНОВЛЕНИЕ СЕРВЕРА")["results"][0]
        assert hit["id"] == 1 and hit["score"] == 1.0
        # title целиком в другом регистре — тоже точное совпадение:
        hit = searcher.search_title("OPEN WEBUI НАСТРОЙКА")["results"][0]
        assert hit["id"] == 2 and hit["score"] == 1.0
        # ЗАГЛАВНЫЙ запрос как подстрока строчного названия — 0.5:
        hit = searcher.search_title("OPEN WEBUI")["results"][0]
        assert hit["id"] == 2 and hit["score"] == 0.5
        # частичное совпадение тоже регистронезависимо (0.5, не точное)
        assert searcher.search_title("СЕРВЕРА")["results"][0]["score"] == 0.5

    def test_no_trigram_rule(self, service) -> None:
        """Trigram-правило «слова ≥3 симв.» здесь НЕ применяется — это не FTS:
        двухсимвольная подстрока названия ищется."""
        searcher, notes_service = service
        notes_service.save("Текст про реактор", title="Реакторный зал")
        assert [hit["id"] for hit in searcher.search_title("ре")["results"]] == [1]

    def test_text_only_substring_not_matched_d3(self, service) -> None:
        """Решение D3: подстрока, встречающаяся ТОЛЬКО в text, не матчится
        title-режимом; обычный гибридный search() её находит."""
        searcher, notes_service = service
        notes_service.save(
            "Заметка про TaskFlow и его настройки", title="Список покупок"
        )
        empty = searcher.search_title("taskflow")
        assert empty["results"] == []
        assert empty["hint"] == HINT_NO_RESULTS_TITLE
        assert [hit["id"] for hit in searcher.search("TaskFlow")["results"]] == [1]

    def test_title_null_excluded(self, service) -> None:
        """Заметки без названия (легаси-путь save) в title-режиме не участвуют."""
        searcher, notes_service = service
        notes_service.save("TaskFlow упоминается только в тексте легаси")
        assert searcher.search_title("taskflow")["results"] == []
        notes_service.save("Свежая заметка с названием", title="TaskFlow")
        hits = searcher.search_title("taskflow")["results"]
        assert [hit["id"] for hit in hits] == [2]
        assert hits[0]["score"] == 1.0  # название совпало с запросом целиком

    def test_trash_excluded(self, service) -> None:
        """Soft delete (trash) прячет заметку и из title-режима."""
        searcher, notes_service = service
        notes_service.save("Текст один", title="Дежурство Халпи")
        notes_service.save("Текст два", title="Дежурство Олега")
        notes_service.delete(1)
        assert [hit["id"] for hit in searcher.search_title("дежурство")["results"]] == [2]


class TestScoringAndOrder:
    def test_exact_beats_substring(self, service) -> None:
        """Ступеньки D2: точное совпадение названия (1.0) выше подстроки (0.5)."""
        searcher, notes_service = service
        notes_service.save("Текст про планировщик", title="Планировщик задач TaskFlow")
        notes_service.save("Ещё про планировщик", title="TaskFlow")
        hits = searcher.search_title("taskflow")["results"]
        assert [hit["id"] for hit in hits] == [2, 1]
        assert [hit["score"] for hit in hits] == [1.0, 0.5]

    def test_earlier_occurrence_first(self, service) -> None:
        """Среди частичных (равный score 0.5): вхождение ближе к началу
        названия выше (updated_at выровнен — решает только позиция)."""
        searcher, notes_service = service
        notes_service.save("Текст раз", title="Планировщик задач TaskFlow")
        notes_service.save("Текст два", title="TaskFlow — планировщик задач")
        set_meta(1, updated_at="2026-01-01T00:00:00Z")
        set_meta(2, updated_at="2026-01-01T00:00:00Z")
        hits = searcher.search_title("taskflow")["results"]
        assert [hit["id"] for hit in hits] == [2, 1]  # позиция 0 против 17
        assert all(hit["score"] == 0.5 for hit in hits)

    def test_tie_updated_at_desc_then_id_desc(self, service) -> None:
        """Tie (равные score и позиция): updated_at DESC; при равном
        updated_at — id DESC."""
        searcher, notes_service = service
        notes_service.save("Текст один", title="Дежурство Халпи")
        notes_service.save("Текст два", title="Дежурство Халпи")
        set_meta(1, updated_at="2026-01-01T00:00:00Z")
        set_meta(2, updated_at="2026-02-01T00:00:00Z")
        hits = searcher.search_title("дежурство")["results"]
        assert [hit["id"] for hit in hits] == [2, 1]
        # равный updated_at → id DESC (заметка id=3 вставлена позже)
        notes_service.save("Текст три", title="Дежурство Халпи")
        set_meta(3, updated_at="2026-02-01T00:00:00Z")
        hits = searcher.search_title("дежурство")["results"]
        assert [hit["id"] for hit in hits] == [3, 2, 1]

    def test_top_k_slice(self, service) -> None:
        """top_k срез; порядок при равных score/позиции — updated_at DESC."""
        searcher, notes_service = service
        for i in range(1, 4):
            notes_service.save(f"Текст дежурства номер {i}", title="Дежурство Халпи")
            set_meta(i, updated_at=f"2026-01-0{i}T00:00:00Z")
        assert len(searcher.search_title("дежурство")["results"]) == 3
        hits = searcher.search_title("дежурство", top_k=2)["results"]
        assert [hit["id"] for hit in hits] == [3, 2]


class TestValidation:
    def test_empty_query_rejected(self, service) -> None:
        searcher, _ = service
        with pytest.raises(SearchValidationError):
            searcher.search_title("")

    def test_too_long_query_rejected(self, service) -> None:
        """Тот же потолок, что у semantic: max_query_chars=512."""
        searcher, _ = service
        with pytest.raises(SearchValidationError):
            searcher.search_title("х" * 513)

    def test_top_k_validation(self, service) -> None:
        """Диапазон top_k 1..MAX_TOP_K — SearchValidationError, как в search()."""
        searcher, _ = service
        with pytest.raises(SearchValidationError):
            searcher.search_title("слово", top_k=0)
        with pytest.raises(SearchValidationError):
            searcher.search_title("слово", top_k=MAX_TOP_K + 1)
        assert searcher.search_title("слово", top_k=1)["results"] == []


class TestOutput:
    def test_element_contract_exact_keys(self, service) -> None:
        """Компактный контракт FR-2.2: ровно пять полей — без snippet,
        полного текста, rrf_score/cosine/author/дат."""
        searcher, notes_service = service
        notes_service.save("Сервис TaskFlow общается", title="Обновление TaskFlow")
        hit = searcher.search_title("обновление")["results"][0]
        assert set(hit.keys()) == {"id", "title", "namespace", "summary", "score"}
        assert hit["title"] == "Обновление TaskFlow"
        assert hit["namespace"] == "default"
        assert hit["score"] == 0.5  # «обновление» — подстрока, не всё название

    def test_summary_pending_fallback_like_semantic(self, service) -> None:
        """summary_status='pending' → summary_of отдаёт первые
        MAX_SUMMARY_CHARS текста — как в semantic-выдаче (§5.5)."""
        searcher, notes_service = service
        text = "Сервис TaskFlow общается через Ollama " + "слово " * 40
        notes_service.save(text, title="Обновление TaskFlow")
        hit = searcher.search_title("обновление")["results"][0]
        assert hit["summary"] == text[: get_settings().max_summary_chars]

    def test_summary_ready_returned_as_is(self, service) -> None:
        """Готовое суммари (summary_status='ok') отдаётся как есть."""
        searcher, notes_service = service
        notes_service.save("Текст заметки про обновление", title="Обновление TaskFlow")
        set_meta(1, summary="Готовое суммари от суммаризатора")
        hit = searcher.search_title("обновление")["results"][0]
        assert hit["summary"] == "Готовое суммари от суммаризатора"

    def test_envelope_no_warning_and_hint_on_empty(self, service) -> None:
        """Обёртка: warning в title-режиме не бывает (эмбеддинг не
        используется); пустой результат — {"results": [], "hint": ...}."""
        searcher, notes_service = service
        notes_service.save("Текст", title="Обновление TaskFlow")
        result = searcher.search_title("обновление")
        assert set(result.keys()) == {"results"}
        empty = searcher.search_title("несуществующее название")
        assert set(empty.keys()) == {"results", "hint"}
        assert empty["results"] == []
        assert empty["hint"] == HINT_NO_RESULTS_TITLE


class TestNamespaceFilter:
    """ns-фильтр — как в search() (§5.7): поддерево / точный узел / ошибка."""

    @pytest.fixture
    def seeded_ns(self) -> SearchService:
        settings = get_settings()
        init_db(settings)
        ns = NamespaceService(settings)
        ns.create("work", "Рабочие заметки. Подпроекты — в листьях.")
        ns.create("work/sbos2020", "СУБО 2020: сервисы HR.")
        ns.create("projects", "Личные проекты.")
        notes = NoteService(settings)
        notes.save(
            "Реестр зарплат на сервере",
            title="Реестр зарплат СУБО",
            namespace="work/sbos2020",
        )
        notes.save(
            "Общие процессы работы",
            title="Общие процессы work",
            namespace="work",
        )
        notes.save(
            "Резюме и портфолио",
            title="Портфолио resume",
            namespace="projects",
        )
        return _searcher(settings)

    def test_subtree_filter(self, seeded_ns: SearchService) -> None:
        """work → work + work/sbos2020; чужой узел не выдаётся."""
        hits = seeded_ns.search_title("реестр", namespace="work")["results"]
        assert [hit["id"] for hit in hits] == [1]
        assert hits[0]["namespace"] == "work/sbos2020"

    def test_exact_filter(self, seeded_ns: SearchService) -> None:
        """namespace_exact=True — только сам узел; лист исключён."""
        empty = seeded_ns.search_title("реестр", namespace="work", namespace_exact=True)
        assert empty["results"] == []
        hits = seeded_ns.search_title(
            "процессы", namespace="work", namespace_exact=True
        )["results"]
        assert [hit["id"] for hit in hits] == [2]

    def test_other_branch_excluded(self, seeded_ns: SearchService) -> None:
        """Заметка в чужом узле (поддереве) не выдаётся ни в одну сторону."""
        assert seeded_ns.search_title("реестр", namespace="projects")["results"] == []
        assert seeded_ns.search_title("портфолио", namespace="work")["results"] == []

    def test_unknown_namespace_is_error(self, seeded_ns: SearchService) -> None:
        with pytest.raises(NamespaceError):
            seeded_ns.search_title("реестр", namespace="unknown")


class TestBm25TitleWeight:
    """FR-1.2: совпадение в title ранжируется выше того же совпадения в text."""

    @pytest.fixture
    def titled_pair(self) -> SearchService:
        """Пара заметок с ОДИНАКОВОЙ подстрокой в разных колонках notes_fts:
        A (id=1) — в title, B (id=2) — в text. Векторной стороны нет
        (save без довекторизации): ранжирование задаёт только FTS."""
        settings = get_settings()
        init_db(settings)
        notes = NoteService(settings)
        notes.save("Планировка встречи в переговорке", title="TaskFlow деплой")
        notes.save("Обсуждение TaskFlow в переговорке офиса", title="Общая заметка")
        return _searcher(settings)

    def test_fts_candidates_title_above_text(self, titled_pair: SearchService) -> None:
        """Прямой тест _fts_candidates (внутренний, отсортирован по badness):
        заметка с подстрокой в title раньше заметки с подстрокой в text."""
        expression = titled_pair._match_expression("taskflow")
        with session(get_settings()) as conn:
            rows = titled_pair._fts_candidates(conn, expression, None)
        assert [row["id"] for row in rows] == [1, 2]

    def test_search_ranks_title_match_above_text_match(
        self, titled_pair: SearchService
    ) -> None:
        """Через гибрид search(): векторная сторона пуста (векторов нет),
        ранжирование задают bm25-веса — A выше B."""
        hits = titled_pair.search("taskflow")["results"]
        assert [hit["id"] for hit in hits] == [1, 2]