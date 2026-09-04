"""Фаза 10 (Шаг 2): namespace-фильтры поиска/обзора и дедуп в пределах неймспейса.

Проверяются (REQUIREMENTS §5.7): запрос по узлу = его поддерево (лист — только
себя), namespace_exact — только сам узел, глобальный поиск без namespace не
меняется; поле namespace в выдачах search/list/get (белые списки §5.6
пополняются одним ключом); FTS-ветка фильтруется JOIN'ом; дедуп — в пределах
неймспейса (меж-узловые дубли легитимны).
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder

from app.config import get_settings
from app.services.dedup import DeduplicationService
from app.services.namespaces import NamespaceError, NamespaceService
from app.services.notes import NoteService
from app.services.search import SearchService
from app.storage import chunks
from app.storage.db import init_db, session

DIM = 8

NOTES = [
    (1, "СУБО 2020: реестр зарплат на сервере appsrv payroll", "work/sbos2020"),
    (2, "Общие рабочие процессы: деплой контейнеров docker", "work"),
    (3, "Проект resume: резюме и сайт portfolio", "projects"),
    (4, "Общий конспект: методика конспектирования заметок", "default"),
]

# Одинаковый текст в двух узлах — дословный «меж-узловой» дубль для дедупа.
CROSS_NOTE = "СУБО 2020: реестр зарплат на сервере appsrv payroll"


@pytest.fixture
def seeded(monkeypatch: pytest.MonkeyPatch) -> tuple[SearchService, HashEmbedder]:
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    init_db(get_settings())
    settings = get_settings()
    NamespaceService(settings).create("work", "Рабочие заметки. Подпроекты — в листьях.")
    NamespaceService(settings).create("work/sbos2020", "СУБО 2020: сервисы HR.")
    NamespaceService(settings).create("projects", "Личные проекты. Сайт-резюме.")
    embedder = HashEmbedder(dim=DIM)
    with session(settings) as conn:
        for note_id, text, namespace in NOTES:
            conn.execute(
                "INSERT INTO notes (id, text, namespace, vector_status) "
                "VALUES (?, ?, ?, 'pending')",
                (note_id, text, namespace),
            )
            chunks.replace_note_chunks(conn, note_id, [(text, len(text))])
        # вектора чанков — после вставки чанков (по одному на заметку)
        for note_id, text, namespace in NOTES:
            chunk_ids = [
                row[0]
                for row in conn.execute(
                    "SELECT id FROM notes_chunks WHERE note_id = ? ORDER BY idx",
                    (note_id,),
                )
            ]
            for chunk_id in chunk_ids:
                chunks.upsert_vector(conn, chunk_id, embedder.embed(text), ns=namespace)
    return SearchService(settings, embedding=embedder), embedder


class TestSearchNamespace:
    def test_global_search_unaffected_and_exposes_namespace(
        self, seeded: tuple[SearchService, HashEmbedder]
    ) -> None:
        """Без namespace — глобальный поиск (как в Фазе 9), каждый результат
        несёт свой namespace."""
        search, _ = seeded
        result = search.search("реестр зарплат", top_k=10)
        assert [r["id"] for r in result["results"]][:1] == [1]
        namespaces = {r["id"]: r["namespace"] for r in result["results"]}
        assert namespaces[1] == "work/sbos2020"

    def test_search_domain_covers_subtree(
        self, seeded: tuple[SearchService, HashEmbedder]
    ) -> None:
        """work → work + work/sbos2020; projects в выдачу не попадает."""
        search, _ = seeded
        result = search.search("реестр зарплат docker", namespace="work", top_k=10)
        ids = [r["id"] for r in result["results"]]
        assert 1 in ids and 2 in ids
        assert 3 not in ids and 4 not in ids

    def test_search_leaf_covers_only_itself(
        self, seeded: tuple[SearchService, HashEmbedder]
    ) -> None:
        search, _ = seeded
        result = search.search("реестр зарплат docker", namespace="work/sbos2020", top_k=10)
        ids = [r["id"] for r in result["results"]]
        assert 1 in ids
        assert 2 not in ids

    def test_search_exact_skips_subtree(
        self, seeded: tuple[SearchService, HashEmbedder]
    ) -> None:
        search, _ = seeded
        result = search.search(
            "реестр зарплат docker", namespace="work", namespace_exact=True, top_k=10
        )
        ids = [r["id"] for r in result["results"]]
        assert 2 in ids
        assert 1 not in ids  # лист work/sbos2020 исключён точным фильтром

    def test_search_unknown_namespace_is_error(
        self, seeded: tuple[SearchService, HashEmbedder]
    ) -> None:
        search, _ = seeded
        with pytest.raises(NamespaceError):
            search.search("реестр зарплат", namespace="unknown")

    def test_fts_branch_respects_namespace(
        self, seeded: tuple[SearchService, HashEmbedder]
    ) -> None:
        """Слово «portfolio» есть только в projects-заметке: поиск по work
        через FTS-ветку не должен его вернуть (фильтр — JOIN'ом)."""
        search, _ = seeded
        result = search.search("portfolio", namespace="work", top_k=10)
        assert result["results"] == []
        global_result = search.search("portfolio", top_k=10)
        assert [r["id"] for r in global_result["results"]] == [3]


class TestListNamespace:
    def test_list_filters_subtree_and_total(
        self, seeded: tuple[SearchService, HashEmbedder]
    ) -> None:
        init_db(get_settings())
        notes = NoteService(get_settings())
        result = notes.list(limit=50, namespace="work")
        assert result["total"] == 2
        assert {item["namespace"] for item in result["items"]} <= {"work", "work/sbos2020"}
        exact = notes.list(limit=50, namespace="work", namespace_exact=True)
        assert exact["total"] == 1
        assert exact["items"][0]["namespace"] == "work"

    def test_list_global_unchanged(
        self, seeded: tuple[SearchService, HashEmbedder]
    ) -> None:
        notes = NoteService(get_settings())
        result = notes.list(limit=50)
        assert result["total"] == 4

    def test_list_unknown_namespace_is_error(
        self, seeded: tuple[SearchService, HashEmbedder]
    ) -> None:
        notes = NoteService(get_settings())
        with pytest.raises(NamespaceError):
            notes.list(namespace="unknown")


class TestGetNamespace:
    def test_get_exposes_namespace(self, seeded: tuple[SearchService, HashEmbedder]) -> None:
        notes = NoteService(get_settings())
        saved = notes.save("Свежая заметка в default без неймспейса")
        fetched = notes.get([saved["id"]])["notes"][0]
        assert fetched["namespace"] == "default"


class TestDedupWithinNamespace:
    """Дедуп — в пределах неймспейса (§5.7): меж-узловые дубли легитимны."""

    def test_find_by_text_scoped(self, seeded: tuple[SearchService, HashEmbedder]) -> None:
        init_db(get_settings())
        dedup = DeduplicationService(get_settings())
        text = "СУБО 2020: реестр зарплат на сервере appsrv payroll"
        # Дословный дубль в СВОЕМ узле — найден; в ЧУЖОМ — нет.
        assert dedup.find_by_text(text, namespace="work/sbos2020") is not None
        assert dedup.find_by_text(text, namespace="projects") is None

    def test_find_candidates_scoped(self, seeded: tuple[SearchService, HashEmbedder]) -> None:
        init_db(get_settings())
        dedup = DeduplicationService(get_settings())
        text = "Дублирующийся текст для проверки меж-узловой изоляции"
        # Одинаковый текст в двух узлах (раскладка руками: Шаг 3 добавит
        # namespace-параметр в save): вектора идентичны — cosine 1.0.
        with session(get_settings()) as conn:
            conn.execute(
                "INSERT INTO notes (id, text, namespace) VALUES (21, ?, 'work')",
                (text,),
            )
            conn.execute(
                "INSERT INTO notes (id, text, namespace) VALUES (22, ?, 'projects')",
                (text,),
            )
            from app.storage import vectors

            vector = HashEmbedder(dim=DIM).embed(text)
            vectors.upsert(conn, 21, vector, ns="work")
            vectors.upsert(conn, 22, vector, ns="projects")
        # В своём узле — кандидат найден; в чужом — нет (партиция изолирует).
        found_work = dedup.find_candidates(vector, exclude_id=22, namespace="work")
        assert [candidate_id for candidate_id, _ in found_work] == [21]
        found_projects = dedup.find_candidates(vector, exclude_id=22, namespace="projects")
        assert found_projects == []