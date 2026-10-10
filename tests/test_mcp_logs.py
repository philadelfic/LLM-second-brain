"""Тесты MCP-поверхности области «Логи ассистента» (LSBN-F-10, f10-02).

Arch assistant-logs §7.1: четыре ручки `logs_save`/`logs_get`/`logs_update`/
`logs_delete` появляются ТОЛЬКО при `AREA_LOGS_ENABLED=true` (выключенная
область не даёт агенту ни одной ручки), выдачи компактны (белые списки, тело
записи целиком отдаёт только `logs_get`), `hint` приходит только при мягком
отказе. Поиск/листинг (`logs_search`/`logs_list`) — f10-03, раздела инструкции
нет (f10-04): инъекции при initialize не появляется.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.config import Settings, get_settings
from app.services import Services
from app.services.logs import (
    HINT_ACTION_REQUIRED,
    HINT_NOT_FOUND,
    HINT_UPDATE_EMPTY,
    LogsService,
)
from app.services.namespaces import NamespaceService
from app.transport.mcp import (
    TOOL_DESCRIPTIONS,
    TOOL_NAMES,
    build_instructions,
    build_mcp,
)
from app.storage.db import init_db
from fakes import HashEmbedder

DIM = 8

LOGS_TOOL_NAMES = frozenset(
    {"logs_save", "logs_get", "logs_update", "logs_delete"}
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
        """При включённом флаге в реестре ровно четыре ручки области §7.1."""
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

    def test_no_logs_section_in_instructions(self, settings: Settings) -> None:
        """Раздела логов в инструкции НЕТ (f10-04): инъекции при initialize нет."""
        text = build_instructions(_services(settings, HashEmbedder(DIM)))
        assert "logs_save" not in text
        assert "assistant-logs" not in text


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
