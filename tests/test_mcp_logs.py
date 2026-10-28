"""Тесты MCP-поверхности области «Логи ассистента» (LSBN-F-10, f10-02/f10-03).

Arch assistant-logs §7.1: шесть ручек `logs_save`/`logs_get`/`logs_update`/
`logs_delete` + `logs_search`/`logs_list` появляются ТОЛЬКО при
`AREA_LOGS_ENABLED=true` (выключенная область не даёт агенту ни одной ручки),
выдачи компактны (белые списки, тело записи целиком отдаёт только `logs_get`,
`action` в поиске/листинге срезан до `logs_search_excerpt_chars`), `hint`
приходит только при мягком отказе. Короткий EN-раздел области появляется в
инструкциях вместе с флагом (f10-04, §10), а логи в карту узлов не входят (§13.13).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.config import Settings, get_settings
from app.services import Services
from app.services.areas import HINT_SHORT_QUERY
from app.services.logs import (
    HINT_ACTION_REQUIRED,
    HINT_DATE_ABSOLUTE,
    HINT_NOT_FOUND,
    HINT_UPDATE_EMPTY,
    LogsService,
)
from app.services.namespaces import NamespaceService
from app.services.worker import BackgroundWorker
from app.transport.mcp import (
    tool_descriptions,
    TOOL_NAMES,
    _NOTE_LIFETIME_SECTION,
    _namespace_map,
    build_instructions,
    build_mcp,
)


from app.storage.db import init_db
from fakes import HashEmbedder


# 3.4.1 (LSBN-T-26): тексты с числом лимита собираются функциями от
# настроек — здесь сборка от значений тестового окружения (роль прежних
# констант; число приходит из Settings, не литерал).
TOOL_DESCRIPTIONS = tool_descriptions(get_settings().max_note_chars)

DIM = 8

LOGS_TOOL_NAMES = frozenset(
    {
        "logs_save",
        "logs_get",
        "logs_update",
        "logs_delete",
        "logs_search",
        "logs_list",
    }
)

ACTION = "записал замер эмбеддингов"
REASON = "нужно было решить про слот"


def _services(settings: Settings, embedding) -> Services:
    """Сервисы in-process MCP: только область логов (DI-эмбеддер, прочее None)."""
    return Services(
        notes=None,
        search=None,
        embedding=embedding,
        dedup=None,
        summary=None,
        judge=None,
        backup=None,
        namespaces=NamespaceService(settings),
        classifier=None,
        promotion=None,
        user_facts=None,
        terms=None,
        links=None,
        logs=LogsService(settings, embedding=embedding),
    )


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def mcp_logs(settings: Settings):
    """In-process MCP с областью логов (детерминированный эмбеддер)."""
    return build_mcp(settings, _services(settings, HashEmbedder(DIM)))


def _tool_names(mcp) -> set[str]:
    async def _list() -> set[str]:
        return {tool.name for tool in await mcp.list_tools()}

    return asyncio.run(_list())


async def _call(mcp, name: str, arguments: dict) -> dict[str, Any]:
    """Вызвать инструмент и вернуть структурированную (компактную) выдачу."""
    result = await mcp.call_tool(name, arguments)
    return result.structured_content


async def _save(mcp, **extra) -> dict:
    arguments = {"action": ACTION, "reason": REASON, **extra}
    return await _call(mcp, "logs_save", arguments)


class TestToolRegistry:
    def test_logs_tools_registered(self, settings: Settings, mcp_logs) -> None:
        """При включённом флаге в реестре ровно шесть ручек области §7.1."""
        names = _tool_names(mcp_logs)
        assert LOGS_TOOL_NAMES <= names
        assert set(TOOL_NAMES) == names
        for name in LOGS_TOOL_NAMES:
            assert TOOL_DESCRIPTIONS[name]

    def test_logs_write_tools_absent_when_disabled(
        self,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`AREA_LOGS_ENABLED=false` убирает все ручки области из реестра."""
        monkeypatch.setenv("AREA_LOGS_ENABLED", "false")
        get_settings.cache_clear()
        disabled = get_settings()
        names = _tool_names(
            build_mcp(disabled, _services(disabled, HashEmbedder(DIM)))
        )
        assert not (LOGS_TOOL_NAMES & names)
        # Ядро поверхности гейту области не подлежит; прочие области на месте.
        assert {"memory_search", "memory_get", "history_list"} <= names
        assert "skills_save" in names

    def test_logs_section_present_when_enabled(self, settings: Settings) -> None:
        """`build_instructions` несёт короткий EN-раздел области (f10-04, §10)."""
        text = build_instructions(_services(settings, HashEmbedder(DIM)))
        assert "Assistant logs are a separate area" in text
        assert "logs_save" in text and "logs_search" in text and "logs_get" in text
        # Раздел — обезличенный и без реестра записей: ни имён, ни «оператора»,
        # ни перечня конкретных логов.
        assert "operator" not in text.lower()
        assert "assistant-logs" not in text

    def test_logs_section_absent_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`AREA_LOGS_ENABLED=false` — раздела области в инструкциях нет."""
        monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
        monkeypatch.setenv("AREA_LOGS_ENABLED", "false")
        get_settings.cache_clear()
        off = get_settings()
        init_db(off)
        text = build_instructions(_services(off, HashEmbedder(DIM)))
        assert "Assistant logs" not in text
        assert "logs_save" not in text
        # no46-02: строка про lifetime от области логов НЕ зависит — остаётся.
        assert _NOTE_LIFETIME_SECTION in text

    def test_logs_not_in_namespace_map(self, settings: Settings) -> None:
        """Карта узлов — про неймспейсы заметок: логи в неё не входят (§13.13)."""
        NamespaceService(settings).create("work", "Рабочие заметки. Подпроекты — в листьях.")
        text = _namespace_map(_services(settings, HashEmbedder(DIM)))
        assert "- work: Рабочие заметки. Подпроекты — в листьях." in text
        assert "logs" not in text


class TestCompactAnswers:
    @pytest.mark.asyncio
    async def test_logs_save_compact(self, settings: Settings, mcp_logs) -> None:
        """`logs_save`: ровно {id, created, version}; отказ — {created, hint}."""
        saved = await _save(mcp_logs)
        assert set(saved) == {"id", "created", "version"}
        assert saved["created"] is True and saved["version"] == 1
        assert "hint" not in saved
        # Лимиты валидирует сервис: нарушение → мягкий отказ с hint §7.3.
        refused = await _call(
            mcp_logs, "logs_save", {"action": " ", "reason": REASON}
        )
        assert refused == {"created": False, "hint": HINT_ACTION_REQUIRED}

    @pytest.mark.asyncio
    async def test_logs_get_compact_and_hint(
        self, settings: Settings, mcp_logs
    ) -> None:
        """`logs_get`: белый список полей записи; нет/удалена → id + hint."""
        saved = await _save(
            mcp_logs, entity="note", entity_id=41, entity_version=3
        )
        record = await _call(mcp_logs, "logs_get", {"id": saved["id"]})
        assert set(record) == {
            "id",
            "action",
            "reason",
            "entity",
            "entity_id",
            "entity_version",
            "entity_ref",
            "outcome",
            "verified",
            "source",
            "session",
            "created_at",
            "updated_at",
            # f10-04 (§11): состояние объекта ссылки (заметки id=41 нет).
            "entity_state",
        }
        assert record["action"] == ACTION and record["reason"] == REASON
        assert record["entity_id"] == 41 and record["entity_version"] == 3
        assert "hint" not in record
        missing = await _call(mcp_logs, "logs_get", {"id": 404})
        assert missing == {"id": 404, "hint": HINT_NOT_FOUND}

    @pytest.mark.asyncio
    async def test_logs_update_and_delete_compact(
        self, settings: Settings, mcp_logs
    ) -> None:
        """`logs_update`/`logs_delete`: компактные выдачи и мягкие отказы."""
        saved = await _save(mcp_logs)
        updated = await _call(
            mcp_logs, "logs_update", {"id": saved["id"], "outcome": "готово"}
        )
        assert updated == {"id": saved["id"], "updated": True}
        # «Ничего не передано» — мягкий отказ, а не исключение.
        empty = await _call(mcp_logs, "logs_update", {"id": saved["id"]})
        assert empty == {
            "id": saved["id"],
            "updated": False,
            "hint": HINT_UPDATE_EMPTY,
        }
        deleted = await _call(mcp_logs, "logs_delete", {"id": saved["id"]})
        assert deleted == {"id": saved["id"], "deleted": True}
        again = await _call(mcp_logs, "logs_delete", {"id": saved["id"]})
        assert again == {
            "id": saved["id"],
            "deleted": False,
            "hint": HINT_NOT_FOUND,
        }
        gone = await _call(mcp_logs, "logs_get", {"id": saved["id"]})
        assert gone["hint"] == HINT_NOT_FOUND

    @pytest.mark.asyncio
    async def test_logs_search_compact(self, settings: Settings, mcp_logs) -> None:
        """`logs_search`: белый список хита, срез `action`, hint только при отказе."""
        long_action = "обрезать " + "х" * 400
        saved = await _save(mcp_logs, action=long_action, entity="note", entity_id=5)
        BackgroundWorker(settings, HashEmbedder(DIM)).process_pending_areas()

        found = await _call(mcp_logs, "logs_search", {"query": "обрезать"})
        assert set(found) == {"results"}, found  # без warning/hint — не отказ
        hit = found["results"][0]
        assert set(hit) == {
            "id",
            "action",
            "created_at",
            "entity",
            "entity_id",
            "entity_state",
            "score",
        }
        assert hit["id"] == saved["id"]
        assert hit["action"] == long_action[: settings.logs_search_excerpt_chars]
        assert len(hit["action"]) == settings.logs_search_excerpt_chars

        # Пустой запрос и запрос вне лимита top_k — мягкие отказы, не исключения.
        empty = await _call(mcp_logs, "logs_search", {"query": "   "})
        assert empty == {"results": [], "hint": HINT_SHORT_QUERY}
        # Срез по относительной дате — мягкий отказ (только абсолютные дни, §13.12).
        relative = await _call(
            mcp_logs, "logs_search", {"query": "обрезать", "date_from": "вчера"}
        )
        assert relative == {"results": [], "hint": HINT_DATE_ABSOLUTE}

    @pytest.mark.asyncio
    async def test_logs_list_compact(self, settings: Settings, mcp_logs) -> None:
        """`logs_list`: страница без `next_cursor`, срез `action`, срезы работают."""
        long_action = "обрезать " + "х" * 400
        first = await _save(mcp_logs, action=long_action, entity="note", entity_id=5)
        second = await _save(mcp_logs)

        page = await _call(mcp_logs, "logs_list", {})
        assert set(page) == {"items", "total", "has_more", "next_offset"}
        assert [item["id"] for item in page["items"]] == [second["id"], first["id"]]
        assert page["total"] == 2 and page["has_more"] is False
        assert page["next_offset"] is None
        # Обратный запрос «почему изменился объект» — срез по объекту и его id.
        sliced = await _call(
            mcp_logs, "logs_list", {"entity": "note", "entity_id": 5}
        )
        assert [item["id"] for item in sliced["items"]] == [first["id"]]
        assert sliced["total"] == 1
        item = sliced["items"][0]
        assert set(item) == {
            "id",
            "action",
            "created_at",
            "entity",
            "entity_id",
            "entity_state",
        }
        assert item["action"] == long_action[: settings.logs_search_excerpt_chars]
        # Потолок MCP — свой, меньший, чем у оператора (list_max_limit_rest).
        over = await _call(
            mcp_logs, "logs_list", {"limit": settings.list_max_limit_mcp + 1}
        )
        assert over == {
            "items": [],
            "hint": f"limit: expected 1..{settings.list_max_limit_mcp}, "
            f"got {settings.list_max_limit_mcp + 1}",
        }
