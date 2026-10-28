"""LSBN-F-16: карта окружения заметки — `memory_get(map=true)` (релиз 3.4.0).

Проверяется `LinksService.neighborhood` (источник `auto`/`1`/`0`, страница
`limit`/`offset` + `total`, направление `mention`) и её выдача транспортом:
флаг `map` одиночного чтения `memory_get` меняет перечень `links` на карту:

* строка карты — ровно `{id, title, namespace, kind}`; `direction` — только
  у `mention` («out»: название соседа в text заметки, «in» — наоборот),
  у `entities`/`cosine` направления нет;
* `map_level`: `auto` — уровень 1 с фолбэком на 0 (как `related`), `1` —
  только хранимые связи (без фолбэка; решение гейта 4A — при маркере
  `links_at`), `0` — только уровень 0 (`kind="cosine"`); источник виден в
  `map_level_used`;
* пагинация `map_limit`/`map_offset` + `total` (тот же фильтр без страницы);
* пустая карта (нет связей) — `neighbors: []` без ошибки и без `hint`;
* batch + map — карты нет (мягкий отказ: карта живёт при одиночном чтении,
  там же, где `links`);
* бюджет: строки карты влезают в `MAP_BUDGET_CHARS_DEFAULT` (1500 символов),
  не влезающие деградируют до шапки `"truncated": true` + hint «read
  point-wise» (как F-13);
* перечень `links` без `map` не меняется (tests/test_links_output.py —
  зелёный без правок).

Моделей никто не зовёт: векторы задаются явно (`app.storage.vectors`), строки
`links` в юнитах кладутся прямым SQL — вид/score/порядок детерминированы.
"""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings
from app.services import Services, build_services
from app.services.links import LinksService
from app.services.namespaces import NamespaceService
from app.services.output_budget import (
    MAP_BUDGET_CHARS_DEFAULT,
    item_chars,
)
from app.services.search import SearchService
from app.storage import vectors
from app.storage.db import init_db, session, transaction
from app.transport.mcp import build_mcp

DIM = 8  # маленькая размерность: косинусы задаются вектором явно
SOURCE = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _vec(cosine: float) -> list[float]:
    """Единичный вектор с заданным косинусом к SOURCE (второе измерение)."""
    return [cosine, (1.0 - cosine * cosine) ** 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# (id, title, namespace, text, vector | None, deleted_at) — как в
# test_links_output.py, но text заметки 1 несёт «Closest» (mention «out»),
# text заметки 3 — «Source» (mention «in»), а заметка 1 НЕ упоминает «Mid».
NOTES = [
    (
        1,
        "Source",
        "default",
        "план релиза: сначала спросить Closest, потом остальное",
        SOURCE,
        None,
    ),
    (2, "Closest", "work", "ближайшая заметка", _vec(1.0), None),
    (3, "Mid", "projects", "встреча по Source и смежным темам", _vec(0.8), None),
    (4, "Low", "work", "низкая близость", _vec(0.6), None),
    (5, "SameNode", "default", "заметка того же узла", _vec(1.0), None),
    (6, "GhostNode", "ghost", "заметка удалённого узла", _vec(1.0), None),
    (7, "Trashed", "work", "заметка в корзине", _vec(1.0), "2026-01-01T00:00:00Z"),
    (8, "Pending", "default", "заметка без вектора", None, None),
    (9, "Fourth", "work", "четвёртая заметка", _vec(0.5), None),
    (10, "Fifth", "projects", "пятая заметка", _vec(0.4), None),
]

# Полный пул уровня 0 для заметки 1 (KNN ≥ LINK_LAZY_THRESHOLD=0.5, после
# отсечений): 2 (1.0) → 3 (0.8) → 4 (0.6) → 9 (0.5). Карту LINK_TOP=3 НЕ режет
# (в `related` заметка 9 «за потолком») — у карты свой потолок `map_limit`;
# 5 — свой узел, 6 — удалённый узел, 7 — корзина, 8 — без вектора, 10 — ниже
# порога.
MAP_LEVEL0_IDS = [2, 3, 4, 9]


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """БД с узлами work/projects и заметками с явными векторами."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
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


def _links_service(settings: Settings) -> LinksService:
    """LinksService поверх явных векторов (KNN детерминирован)."""
    return LinksService(settings, search=SearchService(settings))


def _add_link(
    settings: Settings, note_a: int, note_b: int, kind: str, score: float | None = None
) -> None:
    """Строка `links` прямым SQL + маркер `links_at` (как в test_links_output)."""
    first, second = sorted((note_a, note_b))  # канонический порядок пары
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO links (note_a, note_b, kind, score) VALUES (?, ?, ?, ?)",
            (first, second, kind, score),
        )
        conn.execute(
            "UPDATE notes SET links_at = '2026-01-01T00:00:00Z' "
            "WHERE id IN (?, ?)",
            (first, second),
        )


def _reset_links_at(settings: Settings, note_id: int) -> None:
    """Сбросить маркер заметки — как правка `text`/`title` (связи устарели)."""
    with session(settings) as conn, transaction(conn):
        conn.execute("UPDATE notes SET links_at = NULL WHERE id = ?", (note_id,))


def _touch(settings: Settings, note_id: int, updated_at: str) -> None:
    """Свежесть заметки — tie-break порядка связей (`n.updated_at` в SQL)."""
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE notes SET updated_at = ? WHERE id = ?", (updated_at, note_id)
        )


def _neighbor_ids(data: dict) -> list[int]:
    return [item["id"] for item in data["neighbors"]]


class TestNeighborhoodService:
    """`LinksService.neighborhood` — источник, страница, направление."""

    def test_shape_and_directions(self, settings: Settings) -> None:
        """Строка — {id, title, namespace, kind}; direction — только mention."""
        _add_link(settings, 1, 2, "mention")  # «Closest» в text заметки 1 → out
        _add_link(settings, 1, 3, "mention")  # «Mid» НЕ в text заметки 1 → in
        _add_link(settings, 1, 9, "entities", 0.5)
        _add_link(settings, 1, 4, "cosine", 0.5)
        _touch(settings, 3, "2026-01-01T00:00:00Z")  # 2 (свежее) выше 3
        data = _links_service(settings).neighborhood(1, "auto", 8, 0)
        assert data["level_used"] == 1
        assert data["total"] == 4
        assert data["neighbors"] == [
            {
                "id": 2,
                "title": "Closest",
                "namespace": "work",
                "kind": "mention",
                "direction": "out",
            },
            {
                "id": 3,
                "title": "Mid",
                "namespace": "projects",
                "kind": "mention",
                "direction": "in",
            },
            {"id": 9, "title": "Fourth", "namespace": "work", "kind": "entities"},
            {"id": 4, "title": "Low", "namespace": "work", "kind": "cosine"},
        ]

    def test_level_zero_forced_gives_cosine_kind(self, settings: Settings) -> None:
        """`level="0"` — KNN-пул: kind="cosine", направлений нет."""
        _add_link(settings, 1, 2, "mention")  # уровень 1 игнорируется
        data = _links_service(settings).neighborhood(1, "0", 8, 0)
        assert data["level_used"] == 0
        assert data["total"] == len(MAP_LEVEL0_IDS)
        assert _neighbor_ids(data) == MAP_LEVEL0_IDS
        assert all(
            item.keys() == {"id", "title", "namespace", "kind"}
            and item["kind"] == "cosine"
            for item in data["neighbors"]
        )

    def test_level_one_strict_never_falls_back(self, settings: Settings) -> None:
        """`level="1"` — только хранимые связи: пусто остаётся пустым."""
        _add_link(settings, 1, 9, "mention")  # 9 в уровень 0 не входит
        assert _neighbor_ids(
            _links_service(settings).neighborhood(1, "1", 8, 0)
        ) == [9]
        _reset_links_at(settings, 1)  # маркер сброшен — строки устарели (4A)
        data = _links_service(settings).neighborhood(1, "1", 8, 0)
        assert data == {"neighbors": [], "level_used": 1, "total": 0}

    def test_auto_falls_back_to_level_zero(self, settings: Settings) -> None:
        """`auto`: связи уровня 1 отсечены (свой узел) → уровень 0."""
        _add_link(settings, 1, 5, "cosine", 0.99)  # тот же узел default
        data = _links_service(settings).neighborhood(1, "auto", 8, 0)
        assert data["level_used"] == 0
        assert _neighbor_ids(data) == MAP_LEVEL0_IDS
        assert data["total"] == len(MAP_LEVEL0_IDS)  # total — фильтр уровня 0

    def test_pagination_limit_offset_total(self, settings: Settings) -> None:
        """Страница `limit`/`offset` по тому же порядку; `total` не зависит."""
        _add_link(settings, 1, 2, "mention")
        _add_link(settings, 1, 3, "entities", 0.5)
        _add_link(settings, 1, 9, "entities", 0.2)
        _add_link(settings, 1, 4, "cosine", 0.5)
        _touch(settings, 3, "2026-01-01T00:00:00Z")  # порядок: 2, 3, 9, 4
        service = _links_service(settings)
        page_one = service.neighborhood(1, "auto", 2, 0)
        page_two = service.neighborhood(1, "auto", 2, 2)
        whole = service.neighborhood(1, "auto", 20, 0)
        assert _neighbor_ids(page_one) == [2, 3]
        assert _neighbor_ids(page_two) == [9, 4]
        assert _neighbor_ids(whole) == [2, 3, 9, 4]
        assert page_one["total"] == page_two["total"] == whole["total"] == 4

    def test_missing_note_gives_empty_map(self, settings: Settings) -> None:
        """Нет активной заметки — пустая карта, level_used 0, без ошибки."""
        data = _links_service(settings).neighborhood(999, "auto", 8, 0)
        assert data == {"neighbors": [], "level_used": 0, "total": 0}

    def test_unknown_level_raises(self, settings: Settings) -> None:
        """Неизвестный уровень — громко (диапазон проверяет транспорт)."""
        with pytest.raises(ValueError):
            _links_service(settings).neighborhood(1, "2", 8, 0)


@pytest.fixture
def services(settings: Settings) -> Services:
    """Полная сборка сервисов поверх тестовой БД."""
    return build_services(settings)


@pytest.fixture
def mcp_server(settings: Settings, services: Services):
    """In-process MCP поверх тех же сервисов (один код с REST)."""
    return build_mcp(settings, services)


class TestMCPMap:
    """Флаг `map` одиночного чтения `memory_get` (LSBN-F-16)."""

    @pytest.fixture
    def linked(self, settings: Settings) -> Settings:
        """Четыре связи у заметки 1: mention out, mention in, entities, cosine."""
        _add_link(settings, 1, 2, "mention")
        _add_link(settings, 1, 3, "mention")
        _add_link(settings, 1, 9, "entities", 0.5)
        _add_link(settings, 1, 4, "cosine", 0.5)
        _touch(settings, 3, "2026-01-01T00:00:00Z")  # 2 (свежее) выше 3
        return settings

    @pytest.mark.asyncio
    async def test_map_flag_replaces_links_with_rows(
        self, linked: Settings, mcp_server
    ) -> None:
        """`map=true`: строки {id, title, namespace, kind(, direction)}; без links."""
        got = (
            await mcp_server.call_tool("memory_get", {"id": 1, "map": True})
        ).structured_content
        assert got["neighbors"] == [
            {
                "id": 2,
                "title": "Closest",
                "namespace": "work",
                "kind": "mention",
                "direction": "out",
            },
            {
                "id": 3,
                "title": "Mid",
                "namespace": "projects",
                "kind": "mention",
                "direction": "in",
            },
            {"id": 9, "title": "Fourth", "namespace": "work", "kind": "entities"},
            {"id": 4, "title": "Low", "namespace": "work", "kind": "cosine"},
        ]
        assert got["map_level_used"] == 1
        assert got["total"] == 4
        assert "links" not in got  # карта ЗАМЕНЯЕТ перечень
        assert "hint" not in got

    @pytest.mark.asyncio
    async def test_map_level_zero_over_mcp(
        self, linked: Settings, mcp_server
    ) -> None:
        """`map_level="0"`: kind="cosine" у всех, map_level_used=0."""
        got = (
            await mcp_server.call_tool(
                "memory_get", {"id": 1, "map": True, "map_level": "0"}
            )
        ).structured_content
        assert _neighbor_ids(got) == MAP_LEVEL0_IDS
        assert all(item["kind"] == "cosine" for item in got["neighbors"])
        assert all("direction" not in item for item in got["neighbors"])
        assert got["map_level_used"] == 0
        assert got["total"] == len(MAP_LEVEL0_IDS)

    @pytest.mark.asyncio
    async def test_map_pagination_over_mcp(
        self, linked: Settings, mcp_server
    ) -> None:
        """`map_limit`/`map_offset` странируют карту, `total` константен."""
        first = (
            await mcp_server.call_tool(
                "memory_get", {"id": 1, "map": True, "map_limit": 2}
            )
        ).structured_content
        second = (
            await mcp_server.call_tool(
                "memory_get", {"id": 1, "map": True, "map_limit": 2, "map_offset": 2}
            )
        ).structured_content
        assert _neighbor_ids(first) == [2, 3]
        assert _neighbor_ids(second) == [9, 4]
        assert first["total"] == second["total"] == 4

    @pytest.mark.asyncio
    async def test_empty_map_is_normal_answer(
        self, settings: Settings, mcp_server
    ) -> None:
        """Нет связей (заметка 8 без вектора) — пустая карта без ошибки/hint."""
        got = (
            await mcp_server.call_tool("memory_get", {"id": 8, "map": True})
        ).structured_content
        assert got["neighbors"] == []
        assert got["map_level_used"] == 0
        assert got["total"] == 0
        assert "hint" not in got

    @pytest.mark.asyncio
    async def test_batch_with_map_is_soft_rejected(
        self, linked: Settings, mcp_server
    ) -> None:
        """batch + map — карты нет: мягкий отказ (карта живёт при одном id)."""
        from app.transport.mcp import HINT_MAP_SINGLE_ID

        got = (
            await mcp_server.call_tool("memory_get", {"ids": [1, 2], "map": True})
        ).structured_content
        assert got == {"notes": [], "hint": HINT_MAP_SINGLE_ID}
        assert "neighbors" not in got

    @pytest.mark.asyncio
    async def test_map_parameter_ranges_are_soft_rejected(
        self, linked: Settings, mcp_server
    ) -> None:
        """Диапазоны map* — мягкие отказы (как budget_chars/mode/detail)."""
        from app.transport.mcp import (
            HINT_INVALID_MAP_LEVEL,
            HINT_INVALID_MAP_LIMIT,
            HINT_INVALID_MAP_OFFSET,
        )

        bad_level = (
            await mcp_server.call_tool(
                "memory_get", {"id": 1, "map": True, "map_level": "2"}
            )
        ).structured_content
        assert bad_level == {"notes": [], "hint": HINT_INVALID_MAP_LEVEL}
        bad_limit = (
            await mcp_server.call_tool(
                "memory_get", {"id": 1, "map": True, "map_limit": 0}
            )
        ).structured_content
        assert bad_limit == {"notes": [], "hint": HINT_INVALID_MAP_LIMIT}
        bad_offset = (
            await mcp_server.call_tool(
                "memory_get", {"id": 1, "map": True, "map_offset": -1}
            )
        ).structured_content
        assert bad_offset == {"notes": [], "hint": HINT_INVALID_MAP_OFFSET}

    @pytest.mark.asyncio
    async def test_map_rows_fit_char_budget_with_truncation_hint(
        self, settings: Settings, services: Services, mcp_server
    ) -> None:
        """Бюджет: 40 соседей с длинными названиями — выдача в потолке 1500,
        не влезающие деградируют с hint «read point-wise» (как F-13)."""
        with session(settings) as conn, transaction(conn):
            for index in range(40):
                note_id = 100 + index
                title = f"Регламент пересборки поисковых индексов памяти {index:02d}"
                conn.execute(
                    "INSERT INTO notes (id, title, text, namespace, vector_status) "
                    "VALUES (?, ?, ?, 'work', 'ok')",
                    (note_id, title, f"сосед {index} для бюджета карты"),
                )
            conn.execute(
                "UPDATE notes SET links_at = '2026-01-01T00:00:00Z' WHERE id = 1"
            )
            for index in range(40):
                conn.execute(
                    "INSERT INTO links (note_a, note_b, kind, score) "
                    "VALUES (1, ?, 'cosine', 0.5)",
                    (100 + index,),
                )

        got = (
            await mcp_server.call_tool(
                "memory_get", {"id": 1, "map": True, "map_limit": 20}
            )
        ).structured_content
        assert got["map_level_used"] == 1
        assert got["total"] == 40
        assert 0 < len(got["neighbors"]) < 20  # потолок символов сработал раньше
        assert (
            sum(item_chars(item) for item in got["neighbors"])
            <= MAP_BUDGET_CHARS_DEFAULT
        )
        assert "hint" in got
        assert "read point-wise" in got["hint"]

    @pytest.mark.asyncio
    async def test_links_without_map_are_unchanged(
        self, linked: Settings, mcp_server
    ) -> None:
        """Без `map` перечень `links` прежний (test_links_output не менялся):
        LINK_TOP=3 режет четвёртую связь (потолок перечня, не карты)."""
        got = (
            await mcp_server.call_tool("memory_get", {"id": 1})
        ).structured_content
        assert [item["id"] for item in got["links"]] == [2, 3, 9]
        assert all(set(item) == {"id", "title", "namespace", "chars"} for item in got["links"])
        assert "neighbors" not in got
