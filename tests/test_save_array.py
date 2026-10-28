"""memory_save массивом (LSBN-F-23, п. 3): несколько заметок за операцию.

Ручка принимает массив `notes` ПОВЕРХ скалярного контракта: агент, чей
текст не влезает в мягкий лимит (LSBN-S-1), сохраняет 2–3 самодостаточные
заметки одним действием; слияние (п. 5) пишет свои части тем же путём.
Каждая заметка валидируется ПОЭЛЕМЕНТНО — ошибка одной не топит остальные;
ответ несёт id и title каждой.

LLM не вызывается: save синхронен и без моделей (векторизация — фон).
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder, clear_seeded_skills

from app.config import get_settings
from app.services import Services
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.storage.db import init_db, session
from app.transport.mcp import SAVE_NOTES_MAX, build_mcp, tool_descriptions


# 3.4.1 (LSBN-T-26): тексты с числом лимита собираются функциями от
# настроек — здесь сборка от значений тестового окружения (роль прежних
# констант; число приходит из Settings, не литерал).
TOOL_DESCRIPTIONS = tool_descriptions(get_settings().max_note_chars)

# Граница мягкого отказа в batch-save — из настроек окружения (LSBN-T-26).
LONG_TEXT = "x" * (get_settings().max_note_chars + 1)

DIM = 64


def _services(settings) -> Services:
    return Services(
        notes=NoteService(settings, HashEmbedder(DIM)),
        search=None,
        embedding=HashEmbedder(DIM),
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
        skills=None,
    )


@pytest.fixture
def mcp(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)
    return build_mcp(settings, _services(settings))


async def _call(mcp, name: str, arguments: dict) -> dict:
    return (await mcp.call_tool(name, arguments)).structured_content




class TestSaveArray:
    @pytest.mark.asyncio
    async def test_array_saves_each_and_returns_ids_titles(self, mcp) -> None:
        """Штатный массив: каждая заметка сохранена, ответ — id+title по каждой."""
        body = await _call(
            mcp,
            "memory_save",
            {
                "notes": [
                    {"text": "Встреча 12 сентября в 14:00, переговорная Браво.", "title": "Встреча сентября"},
                    {"text": "Роадстран обновлён до осени 2026.", "title": "Роадстран"},
                ]
            },
        )
        assert body["stored_count"] == 2
        assert [n["title"] for n in body["notes"]] == ["Встреча сентября", "Роадстран"]
        assert all(n["stored"] is True for n in body["notes"])
        with session(get_settings()) as conn:
            rows = conn.execute(
                "SELECT id, title, text FROM notes ORDER BY id"
            ).fetchall()
        assert [r["title"] for r in rows] == ["Встреча сентября", "Роадстран"]
        assert [int(n["id"]) for n in body["notes"]] == [r["id"] for r in rows]

    @pytest.mark.asyncio
    async def test_array_validates_each_note_separately(self, mcp) -> None:
        """Ошибка одного элемента (нет title, текст длиннее лимита) не топит
        остальные: ответ поэлементный — stored/hint у проблемных."""
        body = await _call(
            mcp,
            "memory_save",
            {
                "notes": [
                    {"text": "Заметка без названия не сохранится."},
                    {"text": LONG_TEXT, "title": "Слишком длинно"},
                    {"text": "Эта сохранится.", "title": "Гуд"},
                ]
            },
        )
        assert body["stored_count"] == 1
        assert body["notes"][0]["stored"] is False
        assert "title" in body["notes"][0]["hint"]
        assert body["notes"][1]["stored"] is False
        assert str(get_settings().max_note_chars) in body["notes"][1]["hint"]
        assert body["notes"][2]["stored"] is True

    @pytest.mark.asyncio
    async def test_array_and_text_are_exclusive(self, mcp) -> None:
        """Смешивать `notes` и `text` нельзя — мягкий отказ с hint."""
        body = await _call(
            mcp,
            "memory_save",
            {
                "text": "скалярный текст",
                "notes": [{"text": "массивный текст", "title": "Массив"}],
            },
        )
        assert body["stored"] is False
        assert "either" in body["hint"]
        with session(get_settings()) as conn:
            assert conn.execute("SELECT COUNT(*) c FROM notes").fetchone()["c"] == 0

    @pytest.mark.asyncio
    async def test_array_item_must_be_object_with_text(self, mcp) -> None:
        """Не-объект внутри массива бракует СХЕМА вызова (ToolError); объект
        без текста доходит до поэлементной валидации — отказ с hint."""
        with pytest.raises(Exception, match="valid dictionary"):
            await _call(
                mcp,
                "memory_save",
                {"notes": ["строка вместо объекта"]},
            )
        body = await _call(
            mcp,
            "memory_save",
            {"notes": [{"title": "Без текста"}]},
        )
        assert body["stored_count"] == 0
        assert body["notes"][0]["stored"] is False
        assert "text" in body["notes"][0]["hint"]

    @pytest.mark.asyncio
    async def test_array_schema_is_capped(self, mcp) -> None:
        """Потолок массива — SAVE_NOTES_MAX (порядок ids-потолка memory_get):
        SAVE_NOTES_MAX+1 элементов бракуются схемой вызова."""
        assert SAVE_NOTES_MAX == 20
        with pytest.raises(Exception, match="notes"):
            await _call(
                mcp,
                "memory_save",
                {
                    "notes": [
                        {"text": f"заметка {i}", "title": f"Заметка {i}"}
                        for i in range(SAVE_NOTES_MAX + 1)
                    ]
                },
            )

    def test_description_carries_array_rule(self) -> None:
        """П. 8: описание ручки несёт правило «не влезает в лимит → несколько
        самодостаточных заметок одним вызовом через notes»."""
        text = TOOL_DESCRIPTIONS["memory_save"]
        assert "`notes`" in text and "SEVERAL" in text
        assert "self-contained" in text
