"""lsb-0010-05: выдача связей заметок в MCP и REST (релиз 3.1.0).

Проверяется полная форма `LinksService.related` (уровень 1 с фолбэком на
уровень 0, arch §3.5) и её выдача транспортами: `memory_get` при одиночном
чтении (в том числе в chunk-режиме) и REST `GET /notes/{id}` (FR-3.1…FR-3.5):

* уровень 1 имеет приоритет над уровнем 0; фолбэк — ТОЛЬКО если после
  отсечений уровня 1 не осталось ни одной связи;
* отсечения при выдаче: свой неймспейс, soft-deleted, заметки удалённых узлов;
* порядок: приоритет вида (mention → entities → cosine) → `score` DESC →
  свежесть (`updated_at`) → id DESC; потолок `LINK_TOP` (3);
* элемент связи — ровно `{id, title, namespace, chars}`, `chars` = len(полного
  текста связанной заметки);
* `memory_get`: `links` есть при одиночном `id` и при `ids` из одного элемента,
  при `ids` длиной > 1 поля `links` нет вовсе; в chunk-режиме верхнеуровневый
  `chars` (сумма отданных чанков) не переопределяется;
* пустые связи — нормальный ответ: без ошибки и без `hint`;
* бюджет: overhead связей в `memory_get` (3 связи, названия ≤ 60 символов) ≤
  0.5 КБ; поверхность MCP — те же 21 инструмент, новых REST-ручек нет.

Моделей никто не зовёт: векторы задаются явно (`app.storage.vectors`), строки
`links` в юнитах кладутся прямым SQL — вид/score/порядок детерминированы.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.services import Services, build_services
from app.services.links import LinksService
from app.services.namespaces import NamespaceService
from app.services.search import SearchService
from app.storage import vectors
from app.storage.db import init_db, session, transaction
from app.transport.mcp import TOOL_NAMES, build_mcp

DIM = 8  # маленькая размерность: косинусы задаются вектором явно
SOURCE = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Бюджет §3.7: overhead связей в memory_get (3 связи, названия ≤ 60 символов).
LINK_OVERHEAD_BUDGET_BYTES = 500


def _vec(cosine: float) -> list[float]:
    """Единичный вектор с заданным косинусом к SOURCE (второе измерение)."""
    return [cosine, (1.0 - cosine * cosine) ** 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# (id, title, namespace, text, vector | None, deleted_at)
NOTES = [
    (1, "Source", "default", "исходная заметка про реестр", SOURCE, None),
    (2, "Closest", "work", "ближайшая заметка", _vec(1.0), None),
    (3, "Mid", "projects", "средняя заметка", _vec(0.8), None),
    (4, "Low", "work", "низкая близость", _vec(0.6), None),
    (5, "SameNode", "default", "заметка того же узла", _vec(1.0), None),
    (6, "GhostNode", "ghost", "заметка удалённого узла", _vec(1.0), None),
    (7, "Trashed", "work", "заметка в корзине", _vec(1.0), "2026-01-01T00:00:00Z"),
    (8, "Pending", "default", "заметка без вектора", None, None),
    (9, "Fourth", "work", "четвёртая заметка", _vec(0.5), None),
    (10, "Fifth", "projects", "пятая заметка", _vec(0.4), None),
]

# Уровень 0 для заметки 1 (KNN ≥ LINK_LAZY_THRESHOLD=0.5, без фильтра узла):
# 2 (1.0) → 3 (0.8) → 4 (0.6) → 9 (0.5, за потолком); 5 — свой узел, 6 —
# удалённый узел, 7 — корзина, 8 — без вектора, 10 — ниже порога.
LEVEL0_IDS = [2, 3, 4]

TITLE_MAX_CHARS = 60


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
    """Строка `links` прямым SQL: вид/score задаются тестом (моделей нет)."""
    first, second = sorted((note_a, note_b))  # канонический порядок пары
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO links (note_a, note_b, kind, score) VALUES (?, ?, ?, ?)",
            (first, second, kind, score),
        )


def _touch(settings: Settings, note_id: int, updated_at: str) -> None:
    """Свежесть заметки — tie-break порядка связей (`n.updated_at` в SQL)."""
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE notes SET updated_at = ? WHERE id = ?", (updated_at, note_id)
        )


def _ids(settings: Settings, note_id: int, limit: int | None = None) -> list[int]:
    """Id связей заметки в порядке выдачи."""
    return [item["id"] for item in _links_service(settings).related(note_id, limit)]


class TestRelatedLevel1:
    """`LinksService.related` — уровень 1 с фолбэком на уровень 0 (arch §3.5)."""

    def test_kind_priority_orders_mention_entities_cosine(self, settings: Settings) -> None:
        """Порядок: приоритет вида — mention → entities → cosine (score ниже)."""
        _add_link(settings, 1, 2, "cosine", 0.99)  # самый близкий, но низший вид
        _add_link(settings, 1, 3, "entities", 0.5)
        _add_link(settings, 1, 4, "mention", None)
        assert _ids(settings, 1) == [4, 3, 2]

    def test_level1_wins_over_closer_level0(self, settings: Settings) -> None:
        """Уровень 1 имеет приоритет: связи уровня 0 не примешиваются."""
        _add_link(settings, 1, 4, "mention")  # 4 — далёкая по вектору
        assert _ids(settings, 1, limit=10) == [4]  # 2 (косинус 1.0) не появилась

    def test_ceiling_is_three(self, settings: Settings) -> None:
        """Потолок LINK_TOP (3) режет обе ветки; лишние связи не отдаются."""
        _add_link(settings, 1, 2, "cosine", 0.9)
        _add_link(settings, 1, 3, "entities", 0.5)
        _add_link(settings, 1, 9, "entities", 0.2)
        _add_link(settings, 1, 4, "mention")
        assert _ids(settings, 1, limit=10) == [4, 3, 9]

    def test_score_desc_within_one_kind(self, settings: Settings) -> None:
        """Внутри вида — `score` DESC (свежесть/ id у всех равны)."""
        _add_link(settings, 1, 2, "cosine", 0.4)
        _add_link(settings, 1, 3, "cosine", 0.8)
        _add_link(settings, 1, 4, "cosine", 0.6)
        assert _ids(settings, 1) == [3, 4, 2]

    def test_freshness_then_id_desc(self, settings: Settings) -> None:
        """Равный вид и score: свежесть (`updated_at` DESC), затем id DESC."""
        _add_link(settings, 1, 2, "entities", 0.5)
        _add_link(settings, 1, 3, "entities", 0.5)
        _touch(settings, 2, "2026-03-01T00:00:00Z")
        _touch(settings, 3, "2026-01-01T00:00:00Z")
        assert _ids(settings, 1) == [2, 3]  # свежее — выше
        _touch(settings, 2, "2026-01-01T00:00:00Z")  # свежесть уравняли
        assert _ids(settings, 1) == [3, 2]  # tie-break — id DESC

    def test_own_namespace_trash_and_deleted_node_are_cut(
        self, settings: Settings
    ) -> None:
        """Отсечения при выдаче: свой неймспейс, soft-deleted, удалённый узел."""
        _add_link(settings, 1, 5, "cosine", 0.99)  # тот же узел default
        _add_link(settings, 1, 7, "mention")  # soft-deleted (trash)
        _add_link(settings, 1, 6, "mention")  # узел ghost не в реестре
        _add_link(settings, 1, 2, "cosine", 0.5)  # единственная валидная
        assert _ids(settings, 1, limit=10) == [2]

    def test_fallback_only_when_level1_fully_cut(self, settings: Settings) -> None:
        """Все строки уровня 1 отсечены → приходит уровень 0."""
        _add_link(settings, 1, 5, "cosine", 0.99)
        _add_link(settings, 1, 7, "mention")
        _add_link(settings, 1, 6, "mention")
        assert _ids(settings, 1) == LEVEL0_IDS

    def test_empty_table_falls_back_to_level0(self, settings: Settings) -> None:
        """Пустая таблица связей — уровень 0 (прежнее поведение, FR-1)."""
        assert _ids(settings, 1) == LEVEL0_IDS

    def test_level1_item_shape_and_chars(self, settings: Settings) -> None:
        """Элемент уровня 1 — ровно {id, title, namespace, chars} (FR-3.1)."""
        _add_link(settings, 1, 2, "mention")
        related = _links_service(settings).related(1)
        assert related == [
            {
                "id": 2,
                "title": "Closest",
                "namespace": "work",
                "chars": len("ближайшая заметка"),
            }
        ]

    def test_missing_or_trashed_note_gives_empty(self, settings: Settings) -> None:
        """Нет активной заметки (неизвестный id / trash) — `[]` без ошибки."""
        _add_link(settings, 1, 2, "mention")
        assert _ids(settings, 999) == []
        assert _ids(settings, 7) == []

    def test_limit_lowers_but_never_raises_ceiling(self, settings: Settings) -> None:
        """`limit` понижает потолок обеих веток, выше LINK_TOP не поднимает."""
        _add_link(settings, 1, 2, "mention")
        _add_link(settings, 1, 3, "entities", 0.5)
        _add_link(settings, 1, 4, "cosine", 0.5)
        assert _ids(settings, 1, limit=2) == [2, 3]
        assert len(_ids(settings, 1, limit=10)) == 3


@pytest.fixture
def services(settings: Settings) -> Services:
    """Полная сборка сервисов поверх тестовой БД."""
    return build_services(settings)


@pytest.fixture
def mcp_server(settings: Settings, services: Services):
    """In-process MCP поверх тех же сервисов (один код с REST)."""
    return build_mcp(settings, services)


def payload_size(out: dict) -> int:
    """Размер выдачи в байтах — как она уходит по проводу (UTF-8, «КБ» = 1000)."""
    return len(
        json.dumps(out, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


class TestMCPLinks:
    """`links` в компактной MCP-выдаче `memory_get` (FR-3.1…FR-3.5)."""

    @pytest.fixture
    def linked(self, settings: Settings) -> Settings:
        """Три связи уровня 1 у заметки 1 — по одному виду на связь."""
        _add_link(settings, 1, 2, "mention")
        _add_link(settings, 1, 3, "entities", 0.5)
        _add_link(settings, 1, 4, "cosine", 0.9)
        return settings

    @staticmethod
    def _expected() -> list[dict]:
        """Ожидаемый компактный перечень: порядок по приоритету вида."""
        return [
            {
                "id": 2,
                "title": "Closest",
                "namespace": "work",
                "chars": len("ближайшая заметка"),
            },
            {
                "id": 3,
                "title": "Mid",
                "namespace": "projects",
                "chars": len("средняя заметка"),
            },
            {
                "id": 4,
                "title": "Low",
                "namespace": "work",
                "chars": len("низкая близость"),
            },
        ]

    @pytest.mark.asyncio
    async def test_single_id_and_one_item_list_carry_links(
        self, linked: Settings, mcp_server
    ) -> None:
        """Одиночный `id` и `ids` из одного элемента — `links` есть (FR-3.1)."""
        by_alias = (
            await mcp_server.call_tool("memory_get", {"id": 1})
        ).structured_content
        by_list = (
            await mcp_server.call_tool("memory_get", {"ids": [1]})
        ).structured_content
        assert by_alias["links"] == self._expected()
        assert by_list["links"] == self._expected()
        # Верхний уровень выдачи не подменился: заметка — по-прежнему одна.
        assert len(by_alias["notes"]) == 1 and by_alias["notes"][0]["id"] == 1

    @pytest.mark.asyncio
    async def test_batch_read_has_no_links_field(
        self, linked: Settings, mcp_server
    ) -> None:
        """`ids` длиной > 1 — поля `links` нет ВООБЩЕ (экономия контекста)."""
        got = (
            await mcp_server.call_tool("memory_get", {"ids": [1, 2]})
        ).structured_content
        assert len(got["notes"]) == 2
        assert "links" not in got

    @pytest.mark.asyncio
    async def test_chunk_mode_adds_links_without_touching_chars(
        self, settings: Settings, services: Services, mcp_server
    ) -> None:
        """Chunk-режим: `links` рядом с чанками, верхний `chars` не подменён."""
        text = "\n".join(
            f"строка {i} длинной заметки про чанки и вектора" for i in range(400)
        )
        note_id = services.notes.save(text, title="Длинная заметка")["id"]
        _add_link(settings, note_id, 2, "cosine", 0.5)

        got = (
            await mcp_server.call_tool(
                "memory_get", {"id": note_id, "chunk": 0, "limit": 2}
            )
        ).structured_content
        assert got["total_chunks"] >= 2
        # lsb-0003 не переопределён: верхний `chars` — сумма ОТДАННЫХ чанков.
        assert got["chars"] == sum(len(chunk["text"]) for chunk in got["chunks"])
        assert got["chars"] < len(text)  # отдана лишь часть заметки
        assert got["links"] == [
            {
                "id": 2,
                "title": "Closest",
                "namespace": "work",
                "chars": len("ближайшая заметка"),
            }
        ]

    @pytest.mark.asyncio
    async def test_empty_links_is_normal_answer(
        self, settings: Settings, mcp_server
    ) -> None:
        """Пустые связи (в т.ч. заметка без вектора) — пустое поле, без `hint`."""
        got = (
            await mcp_server.call_tool("memory_get", {"ids": [8]})
        ).structured_content
        assert got["notes"][0]["id"] == 8
        assert got["links"] == []
        assert "hint" not in got

    @pytest.mark.asyncio
    async def test_links_overhead_within_budget(
        self, settings: Settings, services: Services, mcp_server
    ) -> None:
        """Бюджет §3.7: overhead 3 связей с названиями ≤ 60 символов ≤ 0.5 КБ."""
        titles = [
            "Регламент ретроспективного анализа инцидентов биллинга",
            "Порядок пересборки поисковых индексов памяти",
            "Схема выдачи связей заметок разделов",
        ]
        for title in titles:
            note_id = services.notes.save(
                f"заметка метрики про {title}", title=title, namespace="work"
            )["id"]
            _add_link(settings, 1, note_id, "cosine", 0.5)

        got = (
            await mcp_server.call_tool("memory_get", {"id": 1})
        ).structured_content
        assert len(got["links"]) == 3
        assert all(len(item["title"]) <= TITLE_MAX_CHARS for item in got["links"])
        without = {key: value for key, value in got.items() if key != "links"}
        overhead = payload_size(got) - payload_size(without)
        assert overhead <= LINK_OVERHEAD_BUDGET_BYTES, f"overhead связей: {overhead} B"

    def test_mcp_surface_is_21_tools(self) -> None:
        """Поверхность не расширялась: те же 21 инструмент (arch §3.8)."""
        assert len(TOOL_NAMES) == 21


class TestRESTLinks:
    """REST `GET /notes/{id}`: полный контракт + связи (FR-3.2, arch §3.5)."""

    @staticmethod
    def _headers(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def test_note_read_carries_links_and_chars(
        self, client: TestClient, token: str
    ) -> None:
        """`GET /notes/{id}` несёт `chars` и `links` (элемент того же вида)."""
        services = client.app.state.services
        services.namespaces.create("work", "Рабочие заметки.")
        text_default = "релиз 3.1.0: деплой billing прошёл 2026-08-29"
        text_work = "инцидент billing разобран 2026-09-01"
        # POST /notes узла не принимает (NoteCreate без namespace) — вторую
        # заметку кладём через сервисный слой (как и записи MCP в узел).
        first = services.notes.save(text_default, title="Деплой billing")["id"]
        second = services.notes.save(
            text_work, title="Инцидент billing", namespace="work"
        )["id"]
        _add_link(get_settings(), first, second, "mention")

        got = client.get(
            f"/notes/{first}", headers=self._headers(token)
        ).json()
        assert got["chars"] == len(text_default)  # полный контракт чтения
        assert got["links"] == [
            {
                "id": second,
                "title": "Инцидент billing",
                "namespace": "work",
                "chars": len(text_work),
            }
        ]

    def test_no_links_endpoint_appeared(self, client: TestClient) -> None:
        """Отдельной ручки просмотра связей НЕТ (решение О. 2026-09-17)."""
        paths = {getattr(route, "path", "") for route in client.app.routes}
        assert not any("link" in path for path in paths)
