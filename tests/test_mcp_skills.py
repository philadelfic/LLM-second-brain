"""Тесты MCP-поверхности области навыков (lsb-0007-03): 5 инструментов,
компактные выдачи, hint'ы мягких отказов, антисинонимия создания.

ARCH lsb-0007 §3.3–3.4, §3.8: описания инструментов и hint'ы — дословный канон
(сверяем литералами), выдачи компактны (тело навыка — только `skills_get`),
антисинонимия создания живёт в сервисном слое (её же обязан звать REST
POST /skills) и проверяется здесь сквозь MCP-поверхность. Поверхность —
in-process MCP (живой uvicorn — tests/test_mcp.py): caplog видит событие
деградации, эмбеддер подменён детерминированным фейком.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid

import pytest
from fakes import FailingEmbedder, HashEmbedder

from app.config import get_settings
from app.services import Services, build_services
from app.services.namespaces import NamespaceService
from app.services.skills import SkillsService
from app.storage.db import init_db
from app.transport.mcp import (
    SERVER_INSTRUCTIONS,
    TOOL_DESCRIPTIONS,
    TOOL_NAMES,
    build_instructions,
    build_mcp,
)

DIM = 64

# --- Дословные литералы канона arch lsb-0007 §3.8 (править только в арх-доке) -

CANON_TOOL_DESCRIPTIONS = {
    "skills_search": (
        "Search the skills area BEFORE doing a routine or repeatable task: "
        "skills are stored procedures (how-to), kept separately from notes. "
        "Empty result = there is no such skill — do not browse skills without "
        "reason. Returns short hits (id, name, description); the full "
        "procedure — only via skills_get."
    ),
    "skills_list": (
        "List all available skills (id, name, description) — compact, without "
        "bodies. Use it to see which routines this memory already has; many "
        "conversations need no skills at all."
    ),
    "skills_get": (
        "Read a full skill by id: name + description + example (if any) + "
        "steps (the order) + text (what exactly each step does), composed over "
        "the global instruction_template (how to execute steps). Follow it as "
        "a procedure when the task matches its description; if a step cannot "
        "be executed, stop and report what is missing instead of skipping it."
    ),
    "skills_save": (
        "Create a new skill or update an existing one by id. A skill is a "
        "stored procedure: name ≤65 characters (≤5 words recommended), "
        "description ≤250 (what it does), steps ≤500 (the order: what after "
        "what), text ≤4000 (what exactly each step does); optional example "
        "≤1000 and optional class fields (trigger, mode, preconditions, "
        "fallbacks, invariant, exceptions, guardrails, references, "
        "output_contract, behavior_contract) — only when this skill class "
        "needs them. Run skills_search first: if a similar skill exists, "
        "update it instead of creating a duplicate (a too-similar creation is "
        "refused with a hint). Read the skill via skills_get before editing; "
        "every update keeps the previous version as a copy automatically."
    ),
    "skills_delete": (
        "Delete a skill by id (soft delete: it disappears from search, list "
        "and the skills announce; restoring is the operator's job). Delete "
        "only a skill that is factually wrong, fully duplicates another one "
        "or was created by mistake."
    ),
}

# Field-описания параметров (EN; канон §3.8 задаёт тексты инструментов и
# hint'ы, список полей формы — в §3.1/§3.4 описания `skills_save`).
CANON_FIELD_DESCRIPTIONS = {
    "skills_search": {
        "query": "Task wording: what you are about to do",
        "top_k": "Number of results",
    },
    "skills_list": {},
    "skills_get": {"id": "Skill id"},
    "skills_save": {
        "name": "Skill name: ≤65 characters (≤5 words recommended)",
        "description": "What it does: ≤250 characters",
        "steps": "The order: what after what: ≤500 characters",
        "text": "What exactly each step does: ≤4000 characters",
        "id": "Skill id to update; omitted — create a new skill",
        "example": "Optional example: ≤1000 characters",
        "extra": (
            "Optional class fields (trigger, mode, preconditions, fallbacks, "
            "invariant, exceptions, guardrails, references, output_contract, "
            "behavior_contract; each ≤500 characters, together ≤2000) — only "
            "when this skill class needs them"
        ),
    },
    "skills_delete": {"id": "Skill id"},
}

# Hint'ы мягких отказов канона §3.8 (дословно).
CANON_HINT_NAME_LIMIT = (
    "skill not saved: name limit is 65 characters (≤5 words recommended)"
)
CANON_HINT_DESCRIPTION_LIMIT = (
    "skill not saved: description limit is 250 characters — shorten it"
)
CANON_HINT_STEPS_LIMIT = "skill not saved: steps limit is 500 characters — shorten it"
CANON_HINT_TEXT_LIMIT = "skill not saved: text limit is 4000 characters — shorten it"
CANON_HINT_EXAMPLE_LIMIT = (
    "skill not saved: example limit is 1000 characters — shorten it"
)
CANON_HINT_NOT_FOUND = (
    "skill not found (possibly deleted); the actual list — skills_list"
)
CANON_HINT_SEARCH_EMPTY = (
    "no skill found for this task — do the task as usual; if you worked out a "
    "repeatable procedure, save it via skills_save"
)
CANON_HINT_SIMILAR_SKILL = (
    "there is a similar skill: {id} — {name}; reuse or update it "
    "(skills_save with id=…) or make this skill clearly different"
)

# 8 ручек v2.2.1 (lsb-0006): имена и выдачи в этом пуле не меняются.
MEMORY_TOOL_NAMES = frozenset(
    {
        "memory_search",
        "memory_list",
        "memory_get",
        "memory_save",
        "memory_update",
        "memory_delete",
        "memory_namespaces",
        "memory_namespace_create",
    }
)


def form(**overrides: object) -> dict[str, object]:
    """Валидная форма навыка; overrides правят отдельные поля."""
    payload: dict[str, object] = {
        "name": "Deploy the service",
        "description": "How to deploy this service",
        "steps": "1) build; 2) ship; 3) verify",
        "text": "Run make deploy, then check /health.",
    }
    payload.update(overrides)
    return payload


def _services(settings, embedding) -> Services:
    """Сервисы in-process MCP: область навыков + DI-эмбеддер (прочее — None)."""
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
        skills=SkillsService(settings, embedding=embedding),
    )


@pytest.fixture
def mcp_skills(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """In-process MCP с детерминированным эмбеддером (префильтр работает)."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return build_mcp(settings, _services(settings, HashEmbedder(DIM)))


@pytest.fixture
def mcp_fail(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """In-process MCP с отказавшим эмбеддером (деградация префильтра, NFR-3)."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return build_mcp(settings, _services(settings, FailingEmbedder()))


@pytest.fixture
def mcp_notes(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """In-process MCP на полной сборке: регресс выдач заметок."""
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return build_mcp(settings, build_services(settings))


async def _tool_schemas(mcp) -> dict[str, dict]:
    """Схемы аргументов всех инструментов поверхности (in-process MCP)."""
    return {tool.name: tool.input_schema for tool in await mcp.list_tools()}


async def _save(mcp, **overrides: object) -> dict:
    """Вызвать skills_save и вернуть структурированную выдачу."""
    result = await mcp.call_tool("skills_save", form(**overrides))
    return result.structured_content


class TestToolRegistry:
    """Регистрация 5 ручек области навыков: описания и Field-описания — канон."""

    @pytest.mark.asyncio
    async def test_five_skill_tools_added_without_touching_memory_tools(
        self, mcp_skills
    ) -> None:
        names = {tool.name for tool in await mcp_skills.list_tools()}
        assert set(CANON_TOOL_DESCRIPTIONS) <= names
        assert names - set(CANON_TOOL_DESCRIPTIONS) == MEMORY_TOOL_NAMES
        assert set(TOOL_NAMES) == names
        assert len(names) == 13

    @pytest.mark.asyncio
    async def test_tool_descriptions_are_canon_verbatim(self, mcp_skills) -> None:
        tools = {tool.name: tool for tool in await mcp_skills.list_tools()}
        for name, canon in CANON_TOOL_DESCRIPTIONS.items():
            assert TOOL_DESCRIPTIONS[name] == canon
            assert tools[name].description == canon

    @pytest.mark.asyncio
    async def test_field_descriptions_are_canon_verbatim(self, mcp_skills) -> None:
        schemas = await _tool_schemas(mcp_skills)
        for tool_name, expected in CANON_FIELD_DESCRIPTIONS.items():
            props = schemas[tool_name]["properties"]
            assert set(props) == set(expected), tool_name
            for param, description in expected.items():
                assert props[param]["description"] == description, (tool_name, param)

    @pytest.mark.asyncio
    async def test_required_params_and_bounds(self, mcp_skills) -> None:
        """Контракт формы: обязательные поля, лимиты top_k — как у поиска."""
        schemas = await _tool_schemas(mcp_skills)
        assert schemas["skills_save"]["required"] == [
            "name",
            "description",
            "steps",
            "text",
        ]
        for tool_name in ("skills_get", "skills_delete"):
            assert schemas[tool_name]["required"] == ["id"]
        assert schemas["skills_search"]["required"] == ["query"]
        top_k = schemas["skills_search"]["properties"]["top_k"]
        assert top_k["default"] == get_settings().default_top_k
        assert top_k["minimum"] == 1 and top_k["maximum"] == 20
        # Лимиты формы валидирует сервис (не схема) — иначе не будет hint'а.
        for param in ("name", "description", "steps", "text", "example"):
            assert "maxLength" not in schemas["skills_save"]["properties"][param]

    def test_safety_rule_present_in_descriptions(self) -> None:
        """Страховка к анонсу: перед рутиной — search/list, тело — только get."""
        search = CANON_TOOL_DESCRIPTIONS["skills_search"]
        assert "BEFORE doing a routine or repeatable task" in search
        assert "the full procedure — only via skills_get" in search
        assert "do not browse skills without reason" in search
        # save требует поиска перед созданием и чтения перед правкой.
        save = CANON_TOOL_DESCRIPTIONS["skills_save"]
        assert "Run skills_search first" in save
        assert "Read the skill via skills_get before editing" in save
        assert "skills_search" in TOOL_DESCRIPTIONS["skills_save"]

    def test_instructions_not_changed_by_this_pool(self, test_env: dict[str, str]) -> None:
        """Анонс скиллов — отдельный пул 7-04: инструкции не трогаем."""
        settings = get_settings()
        init_db(settings)
        services = _services(settings, HashEmbedder(DIM))
        instructions = build_instructions(services)
        assert instructions.startswith(SERVER_INSTRUCTIONS)
        assert "Node map (path: description)" in instructions
        # В этом пуле блока анонса скиллов в instructions нет.
        assert "skill" not in instructions.lower()
        assert "skill" not in SERVER_INSTRUCTIONS.lower()


class TestCompactOutputs:
    """Белые списки выдач: только разрешённые поля, тела — лишь в skills_get."""

    @pytest.mark.asyncio
    async def test_save_returns_id_version_and_marker(self, mcp_skills) -> None:
        assert await _save(mcp_skills) == {"id": 1, "version": 1, "created": True}

    @pytest.mark.asyncio
    async def test_update_returns_id_version_and_marker(self, mcp_skills) -> None:
        skill_id = (await _save(mcp_skills))["id"]
        result = await _save(mcp_skills, id=skill_id, name="Deploy v2")
        assert result == {"id": skill_id, "version": 2, "updated": True}

    @pytest.mark.asyncio
    async def test_get_is_composite_without_service_fields(self, mcp_skills) -> None:
        """Композит §3.1: без id/extra (полная запись — REST), +шаблон."""
        skill_id = (await _save(mcp_skills))["id"]
        got = (
            await mcp_skills.call_tool("skills_get", {"id": skill_id})
        ).structured_content
        assert set(got) == {
            "name",
            "description",
            "steps",
            "text",
            "instruction_template",
        }
        assert got["name"] == "Deploy the service"
        assert got["steps"] == "1) build; 2) ship; 3) verify"
        assert got["text"] == "Run make deploy, then check /health."
        assert got["instruction_template"]  # сид шаблона «как исполнять шаги»
        assert "id" not in got and "extra" not in got

    @pytest.mark.asyncio
    async def test_get_shows_example_only_when_set(self, mcp_skills) -> None:
        skill_id = (await _save(mcp_skills, example="make deploy"))["id"]
        got = (
            await mcp_skills.call_tool("skills_get", {"id": skill_id})
        ).structured_content
        assert got["example"] == "make deploy"
        assert set(got) == {
            "name",
            "description",
            "example",
            "steps",
            "text",
            "instruction_template",
        }

    @pytest.mark.asyncio
    async def test_list_is_compact(self, mcp_skills) -> None:
        skill_id = (await _save(mcp_skills))["id"]
        got = (await mcp_skills.call_tool("skills_list", {})).structured_content
        assert got["total"] == 1
        assert set(got) == {"items", "total"}
        assert got["items"] == [
            {
                "id": skill_id,
                "name": "Deploy the service",
                "description": "How to deploy this service",
            }
        ]

    @pytest.mark.asyncio
    async def test_search_hits_are_compact(self, mcp_skills) -> None:
        skill_id = (await _save(mcp_skills))["id"]
        got = (
            await mcp_skills.call_tool("skills_search", {"query": "deploy the service"})
        ).structured_content
        assert set(got) == {"results"}  # warning сервиса срезан
        hit = got["results"][0]
        assert hit["id"] == skill_id
        assert set(hit) == {"id", "name", "description", "score"}
        assert "text" not in hit and "steps" not in hit  # тела в выдаче нет

    @pytest.mark.asyncio
    async def test_delete_is_compact(self, mcp_skills) -> None:
        skill_id = (await _save(mcp_skills))["id"]
        got = (
            await mcp_skills.call_tool("skills_delete", {"id": skill_id})
        ).structured_content
        assert got == {"id": skill_id, "deleted": True}

    @pytest.mark.asyncio
    async def test_note_outputs_unchanged(self, mcp_notes) -> None:
        """Регресс: компактные выдачи заметок те же, что до пула (Фаза 9)."""
        saved = (
            await mcp_notes.call_tool(
                "memory_save", {"text": "регресс выдач заметок", "title": "Регресс"}
            )
        ).structured_content
        assert set(saved) == {"id", "stored", "summary_pending"}
        note_id = saved["id"]
        got = (
            await mcp_notes.call_tool("memory_get", {"ids": [note_id]})
        ).structured_content
        assert set(got["notes"][0]) == {
            "id",
            "text",
            "created_at",
            "updated_at",
            "namespace",
            "expires_at",
        }
        listed = (await mcp_notes.call_tool("memory_list", {})).structured_content
        assert set(listed["items"][0]) == {
            "id",
            "title",
            "summary",
            "created_at",
            "updated_at",
            "namespace",
            "expires_at",
        }


class TestFormLimits:
    """Лимиты формы (§3.8): мягкий отказ + дословный hint по каждому полю."""

    @pytest.mark.parametrize(
        ("field", "setting", "hint"),
        [
            ("name", "skill_name_max_chars", CANON_HINT_NAME_LIMIT),
            ("description", "skill_description_max_chars", CANON_HINT_DESCRIPTION_LIMIT),
            ("steps", "skill_steps_max_chars", CANON_HINT_STEPS_LIMIT),
            ("text", "skill_text_max_chars", CANON_HINT_TEXT_LIMIT),
            ("example", "skill_example_max_chars", CANON_HINT_EXAMPLE_LIMIT),
        ],
    )
    @pytest.mark.asyncio
    async def test_creation_over_limit_fails_with_canon_hint(
        self, mcp_skills, field: str, setting: str, hint: str
    ) -> None:
        limit = getattr(get_settings(), setting)
        result = await _save(mcp_skills, **{field: "x" * (limit + 1)})
        assert result == {"created": False, "hint": hint}
        # Отказ ничего не записал.
        listed = (await mcp_skills.call_tool("skills_list", {})).structured_content
        assert listed["total"] == 0

    @pytest.mark.asyncio
    async def test_update_over_limit_fails_with_canon_hint(self, mcp_skills) -> None:
        skill_id = (await _save(mcp_skills))["id"]
        limit = get_settings().skill_name_max_chars
        result = await _save(
            mcp_skills, id=skill_id, name="x" * (limit + 1)
        )
        assert result == {
            "id": skill_id,
            "updated": False,
            "hint": CANON_HINT_NAME_LIMIT,
        }

    @pytest.mark.asyncio
    async def test_update_of_missing_id_hint(self, mcp_skills) -> None:
        result = await _save(mcp_skills, id=999)
        assert result == {"id": 999, "updated": False, "hint": CANON_HINT_NOT_FOUND}


class TestAntiseonymy:
    """Антисинонимия создания (FR-5.2) сквозь MCP: сервисный слой, не транспорт."""

    marker = f"mcpanti-{uuid.uuid4().hex[:8]}"

    @pytest.mark.asyncio
    async def test_similar_creation_refused_with_hint(self, mcp_skills) -> None:
        first = await _save(mcp_skills, name=f"{self.marker} deploy")
        assert first["created"] is True
        # Кандидат — дословная копия текста `name + description` первого.
        second = await _save(mcp_skills, name=f"{self.marker} deploy")
        assert second["created"] is False
        assert second["hint"] == CANON_HINT_SIMILAR_SKILL.format(
            id=first["id"], name=f"{self.marker} deploy"
        )
        # Дубль не записан: в листинге один навык.
        listed = (await mcp_skills.call_tool("skills_list", {})).structured_content
        assert listed["total"] == 1

    @pytest.mark.asyncio
    async def test_distinct_creation_succeeds(self, mcp_skills) -> None:
        assert (await _save(mcp_skills, name=f"{self.marker} deploy"))["created"] is True
        other = await _save(
            mcp_skills,
            name=f"{self.marker} rotate keys",
            description="Rotate the API key and restart nothing",
        )
        assert other["created"] is True
        listed = (await mcp_skills.call_tool("skills_list", {})).structured_content
        assert listed["total"] == 2

    @pytest.mark.asyncio
    async def test_update_by_id_skips_prefilter(self, mcp_skills) -> None:
        """Правка по id префильтр не гоняет (иначе нельзя переименовать навык)."""
        first = await _save(mcp_skills, name=f"{self.marker} deploy")
        second = await _save(
            mcp_skills,
            name=f"{self.marker} rotate keys",
            description="Rotate the API key and restart nothing",
        )
        # Правка приводит текст второго к тексту первого — отказа быть не должно.
        updated = await _save(
            mcp_skills,
            id=second["id"],
            name=f"{self.marker} deploy",
            description="How to deploy this service",
        )
        assert updated == {
            "id": second["id"],
            "version": 2,
            "updated": True,
        }
        assert updated["id"] != first["id"]

    @pytest.mark.asyncio
    async def test_embedding_failure_still_creates_and_logs(
        self, mcp_fail, test_env: dict[str, str], caplog
    ) -> None:
        """Отказ эмбеддера → префильтр пропущен: создание + событие в логе."""
        settings = get_settings()
        # Активный навык в БД: иначе префильтр вышел бы до вызова кодировщика.
        with sqlite3.connect(settings.db_path) as conn:
            conn.execute(
                "INSERT INTO skills (name, description, steps, text, version, "
                "vector_status) VALUES (?, ?, ?, ?, 1, 'pending')",
                (
                    f"{self.marker} deploy",
                    "How to deploy this service",
                    "1) build",
                    "Run make deploy.",
                ),
            )
        with caplog.at_level(logging.WARNING, logger="app"):
            result = await _save(
                mcp_fail,
                name=f"{self.marker} deploy",
                description="How to deploy this service",
            )
        assert result["created"] is True  # деградация: создание проходит
        events = [
            record
            for record in caplog.records
            if getattr(record, "event", None) == "skills_antiseonymy_skipped"
        ]
        assert events


class TestSearchEmptyAndDelete:
    """Пустой поиск — hint канона; delete — soft delete, повтор → «не найден»."""

    @pytest.mark.asyncio
    async def test_empty_search_returns_canon_hint(self, mcp_skills) -> None:
        got = (
            await mcp_skills.call_tool("skills_search", {"query": "нет такого навыка"})
        ).structured_content
        assert got == {"results": [], "hint": CANON_HINT_SEARCH_EMPTY}

    @pytest.mark.asyncio
    async def test_delete_then_repeat_hint(self, mcp_skills) -> None:
        skill_id = (await _save(mcp_skills))["id"]
        first = (
            await mcp_skills.call_tool("skills_delete", {"id": skill_id})
        ).structured_content
        assert first == {"id": skill_id, "deleted": True}
        # Навык исчез из выдач (soft delete).
        assert (
            await mcp_skills.call_tool("skills_list", {})
        ).structured_content["total"] == 0
        got = (
            await mcp_skills.call_tool("skills_get", {"id": skill_id})
        ).structured_content
        assert got == {"hint": CANON_HINT_NOT_FOUND}
        # Повторный delete — мягкий отказ с hint «не найден».
        again = (
            await mcp_skills.call_tool("skills_delete", {"id": skill_id})
        ).structured_content
        assert again == {"id": skill_id, "deleted": False, "hint": CANON_HINT_NOT_FOUND}
