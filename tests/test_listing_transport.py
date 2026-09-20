"""Транспорт листингов (lsb-0013-02, релиз 3.1.0): MCP и REST.

Arch lsb-0013 §3.3–3.4 + постановка 02: у обоих MCP-листингов (`memory_list`,
`skills_list`) дефолт и жёсткий потолок 20 записей, `limit=50` — мягкий отказ
с единым текстом сервиса (не schema-error транспорта); выдачи несут поля
страницы (`total`/`has_more`/`next_offset`/`next_cursor`) и ровно одну
подсказку «есть ещё» — `+N more — offset=K` при `has_more`. Прежние подсказки
`memory is empty` / `page beyond the memory` сохраняются дословно и «+N more»
не дополняются. REST оператора (`GET /notes`, `GET /skills`) идёт с потолком
50 и теми же полями страницы, но без текстовой подсказки листания.
`memory_search` и состав поверхности (21 инструмент) не меняются.

Поверхность — in-process MCP и TestClient приложения: внешние LLM недоступны
(штатная деградация, NFR-3), поэтому выдачи детерминированы. Полный контракт
страницы на сервисном слое — tests/test_listing_service.py (постановка 01);
здесь проверяется транспорт.
"""

from __future__ import annotations

import uuid

import pytest
from fakes import clear_seeded_skills

from app.config import Settings, get_settings
from app.services import Services, build_services
from app.storage.db import init_db
from app.transport.mcp import TOOL_NAMES, build_mcp

DIM = 64

NOT_FOUND_QUERY = "неттакогословафффф"  # запрос без совпадений (мягкий ответ)
HINT_BEYOND = "page beyond the memory: offset ≥ total; reduce offset"


@pytest.fixture
def settings_and_services(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> tuple[Settings, Services]:
    """Свежая БД + полная сборка сервисов (пустой реестр навыков)."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)  # пустой реестр: листинги считаем точно
    return settings, build_services(settings)


@pytest.fixture
def mcp_server(settings_and_services: tuple[Settings, Services]):
    """In-process MCP поверх тех же сервисов (один код с REST)."""
    settings, services = settings_and_services
    return build_mcp(settings, services)


def unique(text: str) -> str:
    """Уникальный текст: дословный дедуп не должен схлопнуть записи."""
    return f"{text} [{uuid.uuid4().hex[:8]}]"


def seed_notes(services: Services, count: int) -> None:
    """Активные заметки через сервис — страницы листинга, а не путь save."""
    for i in range(count):
        services.notes.save(unique(f"страница {i}"), title=f"Заметка {i}")


def seed_skills(services: Services, count: int) -> None:
    """Активные навыки через сервис (антисинонимия деградирует — эмбеддер off)."""
    for i in range(count):
        services.skills.save(
            name=f"Листинг навык {i}",
            description="How to page a listing",
            steps="1) call the listing; 2) read the page fields",
            text="Call the listing, then pass next_offset to reach the next page.",
        )


class TestMCPNotesListing:
    """`memory_list`: дефолт/потолок 20, поля страницы, подсказка «есть ещё»."""

    @pytest.mark.asyncio
    async def test_default_page_is_capped_with_more_hint(
        self, settings_and_services: tuple[Settings, Services], mcp_server
    ) -> None:
        """Дефолт — 20 записей (потолок MCP), остаток виден по подсказке."""
        settings, services = settings_and_services
        seed_notes(services, 21)
        got = (
            await mcp_server.call_tool("memory_list", {})
        ).structured_content
        assert len(got["items"]) == settings.list_max_limit_mcp == 20
        assert got["total"] == 21
        assert got["has_more"] is True
        assert got["next_offset"] == settings.list_max_limit_mcp
        assert got["next_cursor"] is None  # залог под keyset (FR-3.2)
        assert got["hint"] == "+1 more — offset=20"

    @pytest.mark.asyncio
    async def test_last_page_has_no_hint(
        self, settings_and_services: tuple[Settings, Services], mcp_server
    ) -> None:
        """Последняя страница: полей страницы хватает, подсказки нет."""
        settings, services = settings_and_services
        seed_notes(services, 21)
        got = (
            await mcp_server.call_tool("memory_list", {"offset": 20})
        ).structured_content
        assert len(got["items"]) == 1
        assert got["has_more"] is False
        assert got["next_offset"] is None and got["next_cursor"] is None
        assert "hint" not in got

    @pytest.mark.asyncio
    async def test_limit_50_is_soft_refusal_not_schema_error(
        self, settings_and_services: tuple[Settings, Services], mcp_server
    ) -> None:
        """limit=50 — мягкий отказ с единым текстом сервиса (требование приёмки)."""
        settings, _ = settings_and_services
        call = await mcp_server.call_tool("memory_list", {"limit": 50})
        assert call.is_error is False  # не ошибка транспорта
        got = call.structured_content
        assert got["items"] == [] and got["total"] == 0
        assert got["hint"] == (
            f"limit: expected 1..{settings.list_max_limit_mcp}, got 50"
        )

    @pytest.mark.asyncio
    async def test_titles_detail_carries_page_fields(
        self, settings_and_services: tuple[Settings, Services], mcp_server
    ) -> None:
        """Деталь `titles`: те же поля страницы, а `chars` — не её (lsb-0010)."""
        _, services = settings_and_services
        seed_notes(services, 3)
        got = (
            await mcp_server.call_tool(
                "memory_list", {"limit": 2, "detail": "titles"}
            )
        ).structured_content
        assert set(got["items"][0]) == {"id", "title", "namespace"}
        assert got["total"] == 3
        assert got["has_more"] is True and got["next_offset"] == 2
        assert got["next_cursor"] is None
        assert got["hint"] == "+1 more — offset=2"
        assert all("chars" not in item for item in got["items"])

    @pytest.mark.asyncio
    async def test_empty_memory_hint_kept_without_more(
        self, mcp_server
    ) -> None:
        """Пустая память: прежний hint дословно, поля страницы нулевые."""
        got = (
            await mcp_server.call_tool("memory_list", {})
        ).structured_content
        assert got["items"] == [] and got["total"] == 0
        assert got["has_more"] is False and got["next_offset"] is None
        assert got["hint"] == "memory is empty"  # без «+N more»

    @pytest.mark.asyncio
    async def test_page_beyond_hint_kept_without_more(
        self, settings_and_services: tuple[Settings, Services], mcp_server
    ) -> None:
        """offset ≥ total: прежний hint дословно и без «+N more»."""
        _, services = settings_and_services
        seed_notes(services, 3)
        got = (
            await mcp_server.call_tool("memory_list", {"limit": 1, "offset": 10})
        ).structured_content
        assert got["items"] == [] and got["has_more"] is False
        assert got["hint"] == HINT_BEYOND  # дословно, без «+N more»

    @pytest.mark.asyncio
    async def test_exactly_one_hint_per_page(
        self, settings_and_services: tuple[Settings, Services], mcp_server
    ) -> None:
        """Подсказка «есть ещё» — одна на выдачу, не на запись."""
        _, services = settings_and_services
        seed_notes(services, 5)
        got = (
            await mcp_server.call_tool("memory_list", {"limit": 2})
        ).structured_content
        assert got["hint"] == "+3 more — offset=2"
        assert all("hint" not in item for item in got["items"])


class TestMCPSkillsListing:
    """`skills_list(limit, offset)`: листает так же, как `memory_list`."""

    @pytest.mark.asyncio
    async def test_default_page_is_capped_with_more_hint(
        self, settings_and_services: tuple[Settings, Services], mcp_server
    ) -> None:
        """Дефолт/потолок те же, что у заметок; выдача — компактный срез."""
        settings, services = settings_and_services
        seed_skills(services, 21)
        got = (
            await mcp_server.call_tool("skills_list", {})
        ).structured_content
        assert len(got["items"]) == settings.list_max_limit_mcp == 20
        assert set(got["items"][0]) == {"id", "name", "description"}
        assert got["total"] == 21 and got["has_more"] is True
        assert got["next_offset"] == settings.list_max_limit_mcp
        assert got["next_cursor"] is None
        assert got["hint"] == "+1 more — offset=20"

    @pytest.mark.asyncio
    async def test_pages_like_memory_list(
        self, settings_and_services: tuple[Settings, Services], mcp_server
    ) -> None:
        """Смещение листает реестр навыков ровно как листинг заметок."""
        _, services = settings_and_services
        seed_skills(services, 21)
        got = (
            await mcp_server.call_tool("skills_list", {"offset": 20})
        ).structured_content
        assert len(got["items"]) == 1
        assert got["has_more"] is False and got["next_offset"] is None
        assert "hint" not in got

    @pytest.mark.asyncio
    async def test_limit_50_is_soft_refusal(
        self, settings_and_services: tuple[Settings, Services], mcp_server
    ) -> None:
        """limit=50 на навыках — тот же мягкий отказ с единым текстом."""
        settings, _ = settings_and_services
        call = await mcp_server.call_tool("skills_list", {"limit": 50})
        assert call.is_error is False
        got = call.structured_content
        assert got["items"] == [] and got["total"] == 0
        assert got["hint"] == (
            f"limit: expected 1..{settings.list_max_limit_mcp}, got 50"
        )

    @pytest.mark.asyncio
    async def test_schema_has_page_params(
        self, settings_and_services: tuple[Settings, Services], mcp_server
    ) -> None:
        """Параметры страницы появились; верхнюю границу схема не объявляет."""
        settings, _ = settings_and_services
        schemas = {
            tool.name: tool.input_schema for tool in await mcp_server.list_tools()
        }
        props = schemas["skills_list"]["properties"]
        assert set(props) == {"limit", "offset"}
        assert props["limit"]["default"] == settings.default_list_limit == 20
        assert props["limit"]["minimum"] == 1
        assert "maximum" not in props["limit"]  # потолок проверяет сервис
        assert props["limit"]["description"] == "Page size (1..20)"
        assert props["offset"]["default"] == 0
        assert props["offset"]["minimum"] == 0
        assert props["offset"]["description"] == "Page offset"


class TestMCPUnchanged:
    """Постановка не трогает поиск, реестр узлов и состав поверхности."""

    @pytest.mark.asyncio
    async def test_surface_is_still_21_tools(self, mcp_server) -> None:
        """Регресс поверхности: 21 инструмент, состав не изменился."""
        names = {tool.name for tool in await mcp_server.list_tools()}
        assert len(names) == 21
        assert names == TOOL_NAMES

    @pytest.mark.asyncio
    async def test_memory_search_has_no_page_fields(
        self, settings_and_services: tuple[Settings, Services], mcp_server
    ) -> None:
        """У поиска своя семантика: полей листания в выдаче нет (FR-5.1)."""
        _, services = settings_and_services
        seed_notes(services, 3)
        got = (
            await mcp_server.call_tool(
                "memory_search", {"query": NOT_FOUND_QUERY}
            )
        ).structured_content
        assert set(got) <= {"results", "hint"}  # warning срезан, полей нет
        assert not ({"has_more", "next_offset", "next_cursor"} & set(got))

    @pytest.mark.asyncio
    async def test_memory_search_schema_unchanged(self, mcp_server) -> None:
        """Входы поиска не менялись: те же параметры и потолок top_k."""
        schemas = {
            tool.name: tool.input_schema for tool in await mcp_server.list_tools()
        }
        props = schemas["memory_search"]["properties"]
        assert set(props) == {
            "query", "top_k", "namespace", "namespace_exact", "mode"
        }
        assert props["top_k"]["maximum"] == 20


class TestRESTListing:
    """`GET /notes` и `GET /skills`: потолок 50 и те же поля страницы."""

    @staticmethod
    def _headers(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def test_notes_page_fields(self, client, token: str) -> None:
        """Одна страница: те же поля пагинации, что у MCP."""
        for i in range(2):
            client.post(
                "/notes",
                json={"text": unique(f"рест-страница {i}"), "title": f"Рест {i}"},
                headers=self._headers(token),
            )
        body = client.get("/notes", headers=self._headers(token)).json()
        assert body["total"] == 2 and len(body["items"]) == 2
        assert body["has_more"] is False
        assert body["next_offset"] is None and body["next_cursor"] is None

    def test_notes_rest_ceiling_is_50_without_more_hint(
        self, client, token: str
    ) -> None:
        """До 50 записей за страницу, дефолт 20; подсказки «+N more» нет."""
        settings = get_settings()
        notes = client.app.state.services.notes  # type: ignore[attr-defined]
        for i in range(settings.list_max_limit_rest + 1):
            notes.save(unique(f"рест-потолок {i}"), title=f"Потолок {i}")
        default_page = client.get("/notes", headers=self._headers(token)).json()
        assert len(default_page["items"]) == settings.default_list_limit == 20
        assert default_page["has_more"] is True
        assert default_page["next_offset"] == settings.default_list_limit
        assert "hint" not in default_page  # REST — без текстовых подсказок
        full = client.get(
            "/notes",
            params={"limit": settings.list_max_limit_rest},
            headers=self._headers(token),
        ).json()
        assert len(full["items"]) == settings.list_max_limit_rest == 50
        assert full["total"] == settings.list_max_limit_rest + 1
        assert full["has_more"] is True
        assert full["next_offset"] == settings.list_max_limit_rest
        assert full["next_cursor"] is None
        over = client.get(
            "/notes",
            params={"limit": settings.list_max_limit_rest + 1},
            headers=self._headers(token),
        )
        assert over.status_code == 422  # выше потолка поверхности — отказ

    def test_skills_page_fields(self, client, token: str) -> None:
        """Реестр навыков: полные записи + те же поля страницы, без подсказки."""
        settings = get_settings()
        skills = client.app.state.services.skills  # type: ignore[attr-defined]
        for i in range(settings.list_max_limit_rest + 1):
            skills.save(
                name=f"Рест навык {i}",
                description="How to page a listing",
                steps="1) call the listing; 2) read the page fields",
                text="Call the listing, then pass next_offset to reach the next page.",
            )
        body = client.get(
            "/skills",
            params={"limit": settings.list_max_limit_rest},
            headers=self._headers(token),
        ).json()
        assert len(body["items"]) == settings.list_max_limit_rest == 50
        assert body["total"] == settings.list_max_limit_rest + 1
        assert body["has_more"] is True
        assert body["next_offset"] == settings.list_max_limit_rest
        assert body["next_cursor"] is None
        assert "hint" not in body  # подсказка листания — только в MCP
        item = body["items"][0]  # полная запись, а не MCP-срез
        assert {
            "id", "name", "description", "steps", "text", "instruction_template"
        } <= set(item)
