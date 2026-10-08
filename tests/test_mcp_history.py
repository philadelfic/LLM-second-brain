"""Чтение истории из MCP (№41, постановка no41-04, §7.1/§13.14).

История отдаётся агенту РОВНО ДВУМЯ общими инструментами —
`history_list` (список версий без тел, FR-5.1) и `history_get` (содержимое
одной версии, FR-5.2); ручек истории по областям нет (поверхность 21 → 23,
бюджет F-11). История не подмешивается в обычные выдачи
(`memory_get`/`skills_get`/`terms_get`/`user_get`, FR-5.5), `session` версии в
MCP не выводится (§13.4), откат модели не даётся (§13.6). Неизвестная
сущность/версия — мягкий `{…, hint}`, без исключения.

Дополнение постановки: при `HISTORY_ENABLED=false` (§8) версии не пишутся, а
оба инструмента отдают пусто с hint «history is off».
"""

from __future__ import annotations

import uuid

import pytest
from fakes import HashEmbedder, clear_seeded_skills

from app.config import get_settings
from app.services import Services
from app.services.history import HINT_HISTORY_OFF
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.services.skills import SkillsService
from app.services.terms import TermsService
from app.services.user_facts import UserFactsService
from app.storage.db import init_db, session
from app.transport.mcp import TOOL_DESCRIPTIONS, TOOL_NAMES, build_mcp

DIM = 64

SKILL_FORM = {
    "name": "Deploy the service",
    "description": "How to deploy this service",
    "steps": "1) build; 2) ship; 3) verify",
    "text": "Run make deploy, then check /health.",
}


def unique(text: str) -> str:
    """Уникальный текст: дословный дедуп не должен блокировать запись."""
    return f"{text} [{uuid.uuid4().hex[:8]}]"


def _services(settings, embedding) -> Services:
    """In-process MCP с детерминированным эмбеддером (без внешних LLM)."""
    return Services(
        notes=NoteService(settings, embedding),
        search=None,
        embedding=embedding,
        dedup=None,
        summary=None,
        judge=None,
        backup=None,
        namespaces=NamespaceService(settings),
        classifier=None,
        promotion=None,
        user_facts=UserFactsService(settings, embedding=embedding),
        terms=TermsService(settings, embedding=embedding),
        links=None,
        skills=SkillsService(settings, embedding),
    )


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """БД теста: реестр навыков пуст (сид skill-создателя снят)."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)
    return settings


@pytest.fixture
def mcp_history(settings):
    """In-process MCP с включённым слоем истории (дефолт)."""
    return build_mcp(settings, _services(settings, HashEmbedder(DIM)))


@pytest.fixture
def mcp_history_off(monkeypatch: pytest.MonkeyPatch):
    """In-process MCP при `HISTORY_ENABLED=false` (§8)."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    monkeypatch.setenv("HISTORY_ENABLED", "false")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)
    return build_mcp(settings, _services(settings, HashEmbedder(DIM)))


async def _call(mcp, name: str, arguments: dict) -> dict:
    """Вызвать инструмент и вернуть структурированную (компактную) выдачу."""
    return (await mcp.call_tool(name, arguments)).structured_content


async def _save_note(mcp, text: str, title: str = "History note") -> int:
    body = await _call(mcp, "memory_save", {"text": text, "title": title})
    assert body["stored"] is True, body
    return int(body["id"])


class TestHistoryTools:
    @pytest.mark.asyncio
    async def test_history_pair_registered(self, mcp_history) -> None:
        """Пара общих инструментов истории; поверхность 21 → 23 (FR-5.3)."""
        names = {tool.name for tool in await mcp_history.list_tools()}
        assert {"history_list", "history_get"} <= names
        assert set(TOOL_NAMES) == names
        assert len(names) == 24
        assert TOOL_DESCRIPTIONS["history_list"]
        assert TOOL_DESCRIPTIONS["history_get"]

    @pytest.mark.asyncio
    async def test_history_list_tool(self, mcp_history) -> None:
        """`history_list` компактен (тел нет); `has_more`/`total` верны."""
        note_id = await _save_note(mcp_history, unique("first text"))
        await _call(
            mcp_history, "memory_update", {"id": note_id, "text": unique("second text")}
        )
        body = await _call(mcp_history, "history_list", {"entity": "note", "id": note_id})
        assert body["entity"] == "note" and body["id"] == note_id
        assert body["total"] == 2 and body["has_more"] is False
        assert [item["version"] for item in body["versions"]] == [2, 1]
        for item in body["versions"]:
            assert set(item) == {"version", "created_at", "event", "source", "changed"}
            # In-process вызов не проходит HTTP-middleware источника (§13.4):
            # версия несёт источник правки живого сервера (`mcp`) либо дефолт.
            assert item["source"] in {"mcp", "unknown"}
            assert "text" not in item and "session" not in item  # тел и сессии нет
        assert body["versions"][0]["changed"] == ["text"]
        assert body["versions"][1]["event"] == "baseline"
        page = await _call(
            mcp_history,
            "history_list",
            {"entity": "note", "id": note_id, "limit": 1},
        )
        assert len(page["versions"]) == 1 and page["has_more"] is True
        unknown = await _call(mcp_history, "history_list", {"entity": "nope", "id": 1})
        assert unknown["versions"] == [] and "hint" in unknown

    @pytest.mark.asyncio
    async def test_history_get_tool_content(self, mcp_history) -> None:
        """`history_get` отдаёт поля версии; нет сущности/версии — мягкий hint."""
        first = unique("first text")
        note_id = await _save_note(mcp_history, first)
        await _call(
            mcp_history, "memory_update", {"id": note_id, "text": unique("second text")}
        )
        body = await _call(
            mcp_history, "history_get", {"entity": "note", "id": note_id, "version": 1}
        )
        assert body["version"] == 1 and body["event"] == "baseline"
        assert body["text"] == first
        assert body["title"] == "History note"
        assert body["namespace"] == "default"
        assert body["changed"] == []
        missing = await _call(
            mcp_history, "history_get", {"entity": "note", "id": note_id, "version": 99}
        )
        assert "fields" not in missing and "hint" in missing
        unknown = await _call(
            mcp_history, "history_get", {"entity": "nope", "id": 1, "version": 1}
        )
        assert "hint" in unknown and "fields" not in unknown

    @pytest.mark.asyncio
    async def test_history_not_in_normal_reads(self, mcp_history) -> None:
        """Обычные выдачи историю не несут (FR-5.5/FR-9.2)."""
        note_id = await _save_note(mcp_history, unique("read text"))
        skill_id = (await _call(mcp_history, "skills_save", SKILL_FORM))["id"]
        term_id = (
            await _call(
                mcp_history,
                "terms_save",
                {"term": "LSB", "context": "проект", "definition": "Память."},
            )
        )["id"]
        fact_id = (
            await _call(
                mcp_history,
                "user_save",
                {"name": "Moscow timezone", "body": "Oleg is in Moscow."},
            )
        )["id"]
        reads = [
            await _call(mcp_history, "memory_get", {"id": note_id}),
            await _call(mcp_history, "skills_get", {"id": skill_id}),
            await _call(mcp_history, "terms_get", {"id": term_id}),
            await _call(mcp_history, "user_get", {"id": fact_id}),
        ]
        for out in reads:
            for forbidden in ("versions", "history", "changed", "event", "restore_from"):
                assert forbidden not in out


class TestHistoryGate:
    @pytest.mark.asyncio
    async def test_history_off_writes_nothing(self, mcp_history_off) -> None:
        """При выключенном гейте MCP-правка версий не создаёт (§8)."""
        note_id = await _save_note(mcp_history_off, unique("gate text"))
        await _call(
            mcp_history_off,
            "memory_update",
            {"id": note_id, "text": unique("gate text 2")},
        )
        with session(get_settings()) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM record_versions WHERE entity = 'note' "
                "AND entity_id = ?",
                (note_id,),
            ).fetchone()[0]
        assert count == 0

    @pytest.mark.asyncio
    async def test_history_off_reads_empty_with_hint(self, mcp_history_off) -> None:
        """При выключенном гейте оба инструмента отдают пусто + hint (§8)."""
        note_id = await _save_note(mcp_history_off, unique("gate text"))
        listing = await _call(
            mcp_history_off, "history_list", {"entity": "note", "id": note_id}
        )
        assert listing["versions"] == [] and listing["total"] == 0
        assert listing["hint"] == HINT_HISTORY_OFF
        single = await _call(
            mcp_history_off,
            "history_get",
            {"entity": "note", "id": note_id, "version": 1},
        )
        assert "fields" not in single and single["hint"] == HINT_HISTORY_OFF


def _mcp_with_areas(monkeypatch: pytest.MonkeyPatch, **flags: str):
    """In-process MCP при заданных флагах областей (env → настройки)."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    for name, value in flags.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)
    return build_mcp(settings, _services(settings, HashEmbedder(DIM)))


@pytest.mark.asyncio
async def test_history_tool_entity_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    """№40 (§7.1/§13.10): сущность выключенной области — «неизвестная».

    Допустимый словарь `entity` выводится из флагов: `note` — всегда,
    `skill`/`term`/`user_fact` — по флагу своей области. Запрос по сущности
    выключенной области неотличим от опечатки: мягкий `{…, hint}`, и подсказка
    НЕ раскрывает выключенную область. Ядровые `memory_*` гейту не подлежат.
    """
    mcp = _mcp_with_areas(monkeypatch, AREA_TERMS_ENABLED="false")
    # Ядровая сущность работает при любых флагах.
    note_id = await _save_note(mcp, unique("gated note"))
    listing = await _call(mcp, "history_list", {"entity": "note", "id": note_id})
    assert listing["total"] == 1 and "hint" not in listing

    # Сущность выключенной области — как неизвестная: hint без раскрытия.
    for name, args in (
        ("history_list", {"entity": "term", "id": 1}),
        ("history_get", {"entity": "term", "id": 1, "version": 1}),
    ):
        out = await _call(mcp, name, args)
        assert "hint" in out, out
        assert out.get("versions", []) == [] and "fields" not in out
        assert "term" not in out["hint"]

    # Включённые области не задеты: их сущности «известны» (пустая выдача без
    # подсказки про неизвестную сущность; записи для id=1 нет).
    for entity in ("skill", "user_fact"):
        out = await _call(mcp, "history_list", {"entity": entity, "id": 1})
        assert out["versions"] == [] and "hint" not in out, out

    # Схема `entity` тоже собирается от флагов — выключенной области в ней нет.
    tools = {tool.name: tool for tool in await mcp.list_tools()}
    props = tools["history_list"].input_schema["properties"]
    description = props["entity"]["description"]
    assert "note" in description and "skill" in description
    assert "term" not in description
