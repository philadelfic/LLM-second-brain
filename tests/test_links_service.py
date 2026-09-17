"""lsb-0010-01: связи заметок, уровень 0 («ленивый граф»), релиз 3.1.0.

Проверяется механика уровня 0 без транспорта (выдача в memory_get/REST —
следующая постановка): KNN по ПОЛНОМУ вектору заметки (`SearchService.similar_notes`,
без фильтра неймспейса) и отсечения/форма выдачи `LinksService.related`
(FR-1.1…FR-1.6):

* связи приходят только из ДРУГИХ неймспейсов (свой узел исключён);
* потолок LINK_TOP (3) и сортировка по убыванию близости;
* порог LINK_LAZY_THRESHOLD — граница включительная; ниже порога кандидат не идёт;
* сама заметка и soft-deleted исключены; заметка удалённого узла исключена;
* заметка без вектора (`vector_status='pending'`) даёт пустой список без ошибки
  (отсутствие связей — не ошибка и не повод для hint);
* заметка без связей → пустой список;
* элемент — ровно {id, title, namespace, chars}, `chars` = len(полного текста);
* KNN идёт БЕЗ фильтра неймспейса: далёкая по узлу, но близкая по вектору
  заметка находится.

Векторы задаются явно (`app.storage.vectors`) — косинусы детерминированы, без
сети и без LLM (в этом коде моделей нет вовсе).
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.services import build_services
from app.services.links import LinksService
from app.services.namespaces import NamespaceService
from app.services.search import SearchService
from app.storage import vectors
from app.storage.db import init_db, session

DIM = 8  # маленькая размерность: косинусы задаются вектором явно
SOURCE = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _vec(cosine: float) -> list[float]:
    """Единичный вектор с заданным косинусом к SOURCE (второе измерение)."""
    return [cosine, (1.0 - cosine * cosine) ** 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# (id, title, namespace, text, vector | None, deleted_at)
NOTES = [
    (1, "Source", "default", "исходная заметка про реестр", SOURCE, None),
    (2, "Closest", "work", "ближайшая заметка", _vec(1.0), None),
    (3, "Mid", "projects", "средняя заметка", _vec(0.8), None),
    (4, "Low", "work", "низкая близость", _vec(0.6), None),
    (5, "Zero", "projects", "нулевая близость", _vec(0.0), None),
    (6, "SameNode", "default", "заметка того же узла", _vec(1.0), None),
    (7, "GhostNode", "ghost", "заметка удалённого узла", _vec(1.0), None),
    (8, "Trashed", "work", "заметка в корзине", _vec(1.0), "2026-01-01T00:00:00Z"),
    (9, "Pending", "default", "заметка без вектора", None, None),
    (10, "Second", "work", "вторая по близости", _vec(0.9), None),
    (11, "Third", "projects", "третья по близости", _vec(0.7), None),
]


@pytest.fixture
def seeded(monkeypatch: pytest.MonkeyPatch):
    """БД с узлами work/projects (default — системный) и заметками с векторами."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    init_db(get_settings())
    settings = get_settings()
    namespaces = NamespaceService(settings)
    namespaces.create("work", "Рабочие заметки.")
    namespaces.create("projects", "Личные проекты.")
    with session(settings) as conn:
        for note_id, title, namespace, text, vector, deleted in NOTES:
            conn.execute(
                "INSERT INTO notes "
                "(id, title, text, namespace, vector_status, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    note_id,
                    title,
                    text,
                    namespace,
                    "ok" if vector is not None else "pending",
                    deleted,
                ),
            )
            if vector is not None:
                vectors.upsert(conn, note_id, vector, namespace)
    return settings


def _links(monkeypatch: pytest.MonkeyPatch, **env: str) -> LinksService:
    """LinksService на переопределённом окружении (порог/пул/потолок)."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    return LinksService(settings, search=SearchService(settings))


class TestRelatedLevel0:
    """`LinksService.related` — уровень 0 (KNN + отсечения + форма выдачи)."""

    def test_other_namespaces_only_and_ceiling(
        self, seeded, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Свой узел исключён; потолок 3; порядок — по убыванию близости."""
        related = _links(monkeypatch).related(1)
        # Кандидаты ≥ 0.50: 2/10/3/11/4; 6 (тот же узел), 7 (удалённый узел),
        # 8 (корзина) отсечены; потолок оставляет три ближайших по порядку.
        assert [item["id"] for item in related] == [2, 10, 3]
        assert all(item["namespace"] != "default" for item in related)

    def test_far_by_node_close_by_vector_is_found(
        self, seeded, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Исходная — в default, близкая — в projects: KNN без фильтра неймспейса
        (фильтр по поддереву default не вернул бы ничего)."""
        ids = [item["id"] for item in _links(monkeypatch).related(1, limit=10)]
        assert 3 in ids  # namespace=projects, косинус 0.8

    def test_threshold_boundary_is_inclusive(
        self, seeded, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Кандидат ровно на пороге (косинус 1.0 при пороге 1.0) остаётся."""
        related = _links(monkeypatch, LINK_LAZY_THRESHOLD="1.0").related(1)
        assert [item["id"] for item in related] == [2]

    def test_candidate_below_threshold_is_dropped(
        self, seeded, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Порог 0.85: 3 (0.8) и ниже не проходят, 10 (0.9) проходит."""
        related = _links(monkeypatch, LINK_LAZY_THRESHOLD="0.85").related(1)
        assert [item["id"] for item in related] == [2, 10]

    def test_item_shape_and_chars(
        self, seeded, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Элемент — ровно {id, title, namespace, chars}; chars = len(text)."""
        related = _links(monkeypatch).related(1)
        closest = related[0]
        assert set(closest) == {"id", "title", "namespace", "chars"}
        assert closest["id"] == 2
        assert closest["title"] == "Closest"
        assert closest["namespace"] == "work"
        assert closest["chars"] == len("ближайшая заметка")

    def test_pending_source_gives_empty_without_error(
        self, seeded, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Заметка без готового вектора — пустой список, без ошибки."""
        assert _links(monkeypatch).related(9) == []

    def test_no_links_gives_empty(
        self, seeded, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Нет кандидатов выше порога (порог 1.0, косинуса 1.0 ни у кого нет)."""
        assert _links(monkeypatch, LINK_LAZY_THRESHOLD="1.0").related(5) == []

    def test_limit_lowers_but_never_raises_ceiling(
        self, seeded, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`limit` понижает потолок, но выше LINK_TOP не поднимает (FR-1.1)."""
        assert len(_links(monkeypatch).related(1, limit=2)) == 2
        assert len(_links(monkeypatch).related(1, limit=10)) == 3


class TestSimilarNotes:
    """`SearchService.similar_notes` — KNN по полному вектору заметки."""

    def test_returns_close_active_candidates_without_namespace_filter(
        self, seeded, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Без фильтра неймспейса, по убыванию; сама заметка и trash отсечены."""
        search = _links(monkeypatch)._search
        results = search.similar_notes(1, 20, 0.50)
        ids = [note_id for note_id, _ in results]
        # 6/7 — из других узлов (фильтра неймспейса нет); 8 — trash, 5 — ниже порога.
        assert set(ids) == {2, 3, 4, 6, 7, 10, 11}
        assert 1 not in ids  # сама заметка
        assert 8 not in ids  # soft-deleted
        assert 5 not in ids  # ниже порога (косинус 0.0)
        cosines = [cosine for _, cosine in results]
        assert cosines == sorted(cosines, reverse=True)
        assert {note_id for note_id, cosine in results if cosine == 1.0} == {2, 6, 7}

    def test_threshold_is_inclusive(
        self, seeded, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Кандидат с косинусом ровно 1.0 при пороге 1.0 остаётся."""
        results = _links(monkeypatch)._search.similar_notes(1, 20, 1.0)
        assert {note_id for note_id, _ in results} == {2, 6, 7}

    def test_pool_caps_results(self, seeded, monkeypatch: pytest.MonkeyPatch) -> None:
        results = _links(monkeypatch)._search.similar_notes(1, 2, 0.50)
        assert len(results) == 2

    def test_pending_and_unknown_note_give_empty(
        self, seeded, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Нет вектора / нет заметки — пустой список, без ошибки (FR-1.5)."""
        search = _links(monkeypatch)._search
        assert search.similar_notes(9, 20, 0.50) == []
        assert search.similar_notes(999, 20, 0.50) == []


def test_build_services_wires_shared_search(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`build_services` кладёт LinksService поверх общего экземпляра search."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    services = build_services(get_settings())
    assert isinstance(services.links, LinksService)
    assert services.links._search is services.search
