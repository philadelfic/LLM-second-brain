"""Тесты MCP-поверхности области «user» (lsb-0009-02): 5 инструментов,
компактные выдачи и все четыре канала хинтов атомарности; инъекции нет.

ARCH lsb-0009 §3.3, §3.5, §3.7: описания инструментов и hint'ы — дословный
канон (сверяем литералами), выдачи компактны (`excerpt` вместо тела; полное
тело — только `user_get`), успешный `user_save` ВСЕГДА несёт постоянный hint
атомарности (FR-7.2), дедуп/hint средней зоны/лимит `body`/«не найден»/пустой
поиск — мягкие ответы с hint. Инъекции при initialize нет (arch §2): блока
«user» в instructions не появляется, текст относительно предыдущего состояния
не меняется (анонс навыков существовал и раньше, lsb-0007-04). Поверхность —
in-process MCP, эмбеддер подменён детерминированным фейком.
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder, clear_seeded_skills

from app.config import get_settings
from app.services import Services, build_services
from app.services.namespaces import NamespaceService
from app.services.terms import TermsService
from app.services.user_facts import (
    HINT_NOT_FOUND,
    HINT_RELATED_FACTS,
    HINT_REQUIRED_UNSET,
    UserFactsService,
)
from app.storage.db import init_db, session
from app.transport.mcp import (
    SERVER_INSTRUCTIONS,
    TOOL_DESCRIPTIONS,
    TOOL_NAMES,
    _NS_RULES,
    _namespace_map,
    _skills_announce,
    build_instructions,
    build_mcp,
)

DIM = 64

# --- Дословные литералы канона arch lsb-0009 §3.7 (править только в арх-доке) -

CANON_TOOL_DESCRIPTIONS = {
    "user_search": (
        "Search the user area: atomic facts about the user (steady preferences, "
        "working habits, agreements — ONE FACT PER RECORD, never a list). Returns "
        "{id, name, excerpt}; the full body — user_get. Search here when the "
        "answer depends on the user's preferences or arrangements; do not invent "
        "what might be stored — search first."
    ),
    "user_save": (
        "Save ONE atomic durable fact about the user: name ≤5 words (like a note "
        "title) + body ≤1200 characters. ONE FACT = ONE RECORD: never pack a list "
        "of facts into one record and never repeat a fact that is already stored "
        "— several facts mean several separate calls. A strong overlap with an "
        "existing fact is refused with a hint pointing to it: the same fact — "
        "refine it via user_update(id=…); a new fact — save it as a separate "
        "record. Never store secrets (passwords, tokens, keys)."
    ),
    "user_update": (
        "Update a fact by id: name and/or body; a value that is not passed = keep "
        "the current one. Run user_get first so you don't lose details. This is "
        "the right tool when user_save hinted that a similar fact already exists."
    ),
    "user_delete": (
        "Delete a fact by id (soft delete: it disappears from all outputs; "
        "restoring is the operator's job). Delete only a fact that is wrong or "
        "fully duplicates another one."
    ),
    "user_get": (
        "Read one fact by id: name + body."
    ),
}

# Field-описания параметров (EN): канон §3.7 задаёт тексты инструментов и
# hint'ы; описания полей собраны из той же канонической формулировки §3.1/§3.3
# («name ≤5 words (like a note title)», «body ≤1200 characters», «not passed —
# keep the current one») — как в пуле области навыков.
CANON_FIELD_DESCRIPTIONS = {
    "user_search": {
        "query": (
            "Topic wording: the user's preferences or arrangements the answer "
            "depends on"
        ),
        "top_k": "Number of results",
    },
    "user_save": {
        "name": "Fact name: ≤5 words (like a note title)",
        "body": "Fact body: ≤1200 characters; one fact per record",
    },
    "user_update": {
        "id": "Fact id",
        "name": (
            "New name: ≤5 words (like a note title); not passed — the current "
            "one stays"
        ),
        "body": "New body: ≤1200 characters; not passed — the current one stays",
    },
    "user_delete": {"id": "Fact id"},
    "user_get": {"id": "Fact id"},
}

# Hint'ы канона §3.7 (дословно) — четыре канала FR-7.
CANON_HINT_ATOMIC = "one fact = one record — several facts mean several separate calls"
CANON_HINT_SIMILAR = (
    "similar fact already exists: {id} — {name}; the same fact? update it via "
    "user_update(id={id}); a new fact — save it as a separate record"
)
CANON_HINT_BODY_LIMIT = (
    "not saved: looks like several facts in one record — split them and save "
    "one fact per call (body limit is 1200 characters)"
)
CANON_HINT_SEARCH_EMPTY = (
    "nothing found in the user area — no fact matching this request is stored"
)

# 8 ручек заметок/узлов и 5 ручек навыков (lsb-0006/lsb-0007-03): в этом пуле
# их имена и выдачи не меняются.
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
SKILL_TOOL_NAMES = frozenset(
    {"skills_search", "skills_list", "skills_get", "skills_save", "skills_delete"}
)
# 3 ручки области terms (lsb-0008-02) — поверхность релиза 3.0.0 растёт.
TERMS_TOOL_NAMES = frozenset({"terms_search", "terms_save", "terms_get"})

FACT_NAME = "Moscow timezone"
FACT_BODY = "Oleg is in Moscow, Europe/Moscow is the default for weather and time"
# Триграммное сходство с FACT ≈ 0.71 — средняя зона (≥ weak, < strong).
RELATED_NAME = "Working timezone"
RELATED_BODY = "Oleg works in Europe/Moscow timezone"


def _services(settings, embedding) -> Services:
    """Сервисы in-process MCP: область «user» + DI-эмбеддер (прочее — None)."""
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
        user_facts=UserFactsService(settings, embedding=embedding),
        terms=TermsService(settings, embedding=embedding),
    )


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """БД теста: сид skill-создателя снят (область user его не читает)."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)
    return settings


@pytest.fixture
def mcp_user(settings) -> object:
    """In-process MCP области «user» с детерминированным эмбеддером."""
    return build_mcp(settings, _services(settings, HashEmbedder(DIM)))


@pytest.fixture
def mcp_full(settings) -> object:
    """In-process MCP на полной сборке: регресс выдач заметок и навыков."""
    return build_mcp(settings, build_services(settings))


def _active_count(settings) -> int:
    """Число активных фактов области (записи «отказа» проверять нечего)."""
    with session(settings) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM user_facts WHERE deleted_at IS NULL"
        ).fetchone()
    return int(row["n"])


async def _schemas(mcp) -> dict[str, dict]:
    """Схемы аргументов всех инструментов поверхности (in-process MCP)."""
    return {tool.name: tool.input_schema for tool in await mcp.list_tools()}


async def _call(mcp, name: str, arguments: dict) -> dict:
    """Вызвать инструмент и вернуть структурированную (компактную) выдачу."""
    result = await mcp.call_tool(name, arguments)
    return result.structured_content


class TestToolRegistry:
    """Регистрация 5 ручек области «user»: описания и Field-описания — канон."""

    @pytest.mark.asyncio
    async def test_five_user_tools_added_without_touching_others(
        self, mcp_user
    ) -> None:
        names = {tool.name for tool in await mcp_user.list_tools()}
        assert set(CANON_TOOL_DESCRIPTIONS) <= names
        assert names - set(CANON_TOOL_DESCRIPTIONS) == (
            MEMORY_TOOL_NAMES | SKILL_TOOL_NAMES | TERMS_TOOL_NAMES
        )
        assert set(TOOL_NAMES) == names
        assert len(names) == 21

    @pytest.mark.asyncio
    async def test_tool_descriptions_are_canon_verbatim(self, mcp_user) -> None:
        tools = {tool.name: tool for tool in await mcp_user.list_tools()}
        for name, canon in CANON_TOOL_DESCRIPTIONS.items():
            assert TOOL_DESCRIPTIONS[name] == canon
            assert tools[name].description == canon

    @pytest.mark.asyncio
    async def test_field_descriptions_are_canon_verbatim(self, mcp_user) -> None:
        schemas = await _schemas(mcp_user)
        for tool_name, expected in CANON_FIELD_DESCRIPTIONS.items():
            props = schemas[tool_name]["properties"]
            assert set(props) == set(expected), tool_name
            for param, description in expected.items():
                assert props[param]["description"] == description, (tool_name, param)

    @pytest.mark.asyncio
    async def test_required_params_bounds_and_no_schema_limits(
        self, mcp_user
    ) -> None:
        """Контракт формы: обязательные поля, границы top_k — как у поиска.

        Лимиты `name`/`body` валидирует СЕРВИС (не схема) — иначе не будет
        дословного hint'а канона (FR-7.4); «не передано» у правки — сентинел,
        в схеме показан как null (прецедент memory_update).
        """
        schemas = await _schemas(mcp_user)
        assert schemas["user_save"]["required"] == ["name", "body"]
        assert schemas["user_search"]["required"] == ["query"]
        for tool_name in ("user_get", "user_delete"):
            assert schemas[tool_name]["required"] == ["id"]
        assert schemas["user_update"]["required"] == ["id"]
        top_k = schemas["user_search"]["properties"]["top_k"]
        assert top_k["default"] == get_settings().default_top_k
        assert top_k["minimum"] == 1 and top_k["maximum"] == 20
        for param in ("name", "body"):
            assert "maxLength" not in schemas["user_save"]["properties"][param]
            assert schemas["user_update"]["properties"][param]["default"] is None

    def test_instructions_have_no_user_block_and_unchanged(self, settings) -> None:
        """Инъекции нет (arch §2): блока «user» нет и текст не изменился.

        Хвост анонса навыков — существующий (lsb-0007-04); всё остальное —
        база + правило неймспейсов + карта, как до этого пула.
        """
        services = _services(settings, HashEmbedder(DIM))
        expected = (
            SERVER_INSTRUCTIONS
            + _NS_RULES
            + _namespace_map(services)
            + _skills_announce(services)
        )
        text = build_instructions(services)
        assert text == expected
        for needle in (
            "user_search",
            "user_save",
            "user area",
            "atomic facts about the user",
        ):
            assert needle not in text
        # Инструкции handshake (initialize) собираются тем же текстом.
        assert build_mcp(settings, services).instructions == text


class TestSaveHints:
    """Каналы FR-7: постоянный hint, дедуп, лимит `body`, средняя зона."""

    @pytest.mark.asyncio
    async def test_successful_save_always_carries_atomic_hint(
        self, mcp_user, settings
    ) -> None:
        first = await _call(
            mcp_user, "user_save", {"name": FACT_NAME, "body": FACT_BODY}
        )
        assert first == {"id": first["id"], "stored": True, "hint": CANON_HINT_ATOMIC}
        second = await _call(
            mcp_user,
            "user_save",
            {"name": "Coffee preference", "body": "prefers espresso in the morning"},
        )
        assert second["stored"] is True
        assert second["hint"] == CANON_HINT_ATOMIC  # в КАЖДОМ успешном save
        assert _active_count(settings) == 2

    @pytest.mark.asyncio
    async def test_strong_overlap_refused_with_dedup_hint(
        self, mcp_user, settings
    ) -> None:
        first = await _call(
            mcp_user, "user_save", {"name": FACT_NAME, "body": FACT_BODY}
        )
        again = await _call(
            mcp_user,
            "user_save",
            {"name": FACT_NAME, "body": FACT_BODY.replace("and time", "and tasks")},
        )
        assert again == {
            "stored": False,
            "hint": CANON_HINT_SIMILAR.format(id=first["id"], name=FACT_NAME),
        }
        assert _active_count(settings) == 1  # отказ записи не создаёт

    @pytest.mark.asyncio
    async def test_body_over_limit_soft_refusal_with_split_hint(
        self, mcp_user, settings
    ) -> None:
        result = await _call(
            mcp_user, "user_save", {"name": "Too long fact", "body": "x" * 1201}
        )
        assert result == {"stored": False, "hint": CANON_HINT_BODY_LIMIT}
        assert _active_count(settings) == 0

    @pytest.mark.asyncio
    async def test_middle_zone_writes_with_related_and_hint(
        self, mcp_user, settings
    ) -> None:
        related = await _call(
            mcp_user, "user_save", {"name": RELATED_NAME, "body": RELATED_BODY}
        )
        result = await _call(
            mcp_user, "user_save", {"name": FACT_NAME, "body": FACT_BODY}
        )
        assert result["stored"] is True
        assert result["related"] == [{"id": related["id"], "name": RELATED_NAME}]
        assert result["hint"] == (
            f"{CANON_HINT_ATOMIC}; "
            + HINT_RELATED_FACTS.format(
                related=f"{related['id']} — {RELATED_NAME}"
            )
        )
        assert _active_count(settings) == 2  # запись состоялась


class TestUpdateSemantics:
    """«не передано» = оставить (arch §3.3); null — мягкий отказ."""

    @pytest.mark.asyncio
    async def test_not_passed_keeps_values_on_two_params(
        self, mcp_user, settings
    ) -> None:
        saved = await _call(
            mcp_user, "user_save", {"name": FACT_NAME, "body": FACT_BODY}
        )
        renamed = await _call(
            mcp_user, "user_update", {"id": saved["id"], "name": "Timezone fact"}
        )
        assert renamed == {"id": saved["id"], "changed": True}
        after_name = await _call(mcp_user, "user_get", {"id": saved["id"]})
        assert after_name == {
            "id": saved["id"],
            "name": "Timezone fact",
            "body": FACT_BODY,  # body не передан — остался прежним
        }
        rebodied = await _call(
            mcp_user, "user_update", {"id": saved["id"], "body": "Moscow, MSK."}
        )
        assert rebodied == {"id": saved["id"], "changed": True}
        after_body = await _call(mcp_user, "user_get", {"id": saved["id"]})
        assert after_body == {
            "id": saved["id"],
            "name": "Timezone fact",  # name не передан — остался прежним
            "body": "Moscow, MSK.",
        }

    @pytest.mark.asyncio
    async def test_null_in_mandatory_field_soft_refusal(self, mcp_user) -> None:
        saved = await _call(
            mcp_user, "user_save", {"name": FACT_NAME, "body": FACT_BODY}
        )
        result = await _call(
            mcp_user, "user_update", {"id": saved["id"], "name": None}
        )
        assert result == {
            "id": saved["id"],
            "changed": False,
            "hint": HINT_REQUIRED_UNSET.format(field="name"),
        }
        assert (await _call(mcp_user, "user_get", {"id": saved["id"]}))[
            "name"
        ] == FACT_NAME

    @pytest.mark.asyncio
    async def test_update_missing_id_hint(self, mcp_user) -> None:
        assert await _call(
            mcp_user, "user_update", {"id": 999, "body": "new body text"}
        ) == {"id": 999, "changed": False, "hint": HINT_NOT_FOUND}


class TestSearchGetDelete:
    """Поиск отдаёт excerpt; пусто/не найден — мягкие ответы с hint."""

    @pytest.mark.asyncio
    async def test_search_returns_excerpt_and_body_only_via_get(
        self, mcp_user, settings
    ) -> None:
        body = "Oleg prefers short answers in work. " * 11 + "Short."
        saved = await _call(mcp_user, "user_save", {"name": "Answer style", "body": body})
        found = await _call(mcp_user, "user_search", {"query": "prefers short answers"})
        hit = found["results"][0]
        assert hit == {
            "id": saved["id"],
            "name": "Answer style",
            "excerpt": body[: settings.user_search_excerpt_chars],
        }
        assert hit["excerpt"] != body  # тело срезано (не excerpt целиком)
        assert await _call(mcp_user, "user_get", {"id": saved["id"]}) == {
            "id": saved["id"],
            "name": "Answer style",
            "body": body,  # полное тело — только user_get
        }

    @pytest.mark.asyncio
    async def test_empty_search_soft_answer_with_hint(self, mcp_user) -> None:
        assert await _call(
            mcp_user, "user_search", {"query": "quantum gardening habits"}
        ) == {"results": [], "hint": CANON_HINT_SEARCH_EMPTY}

    @pytest.mark.asyncio
    async def test_delete_repeat_and_get_missing_hints(self, mcp_user) -> None:
        saved = await _call(
            mcp_user, "user_save", {"name": FACT_NAME, "body": FACT_BODY}
        )
        assert await _call(mcp_user, "user_delete", {"id": saved["id"]}) == {
            "id": saved["id"],
            "deleted": True,
        }
        assert await _call(mcp_user, "user_delete", {"id": saved["id"]}) == {
            "id": saved["id"],
            "deleted": False,
            "hint": HINT_NOT_FOUND,
        }
        assert await _call(mcp_user, "user_get", {"id": saved["id"]}) == {
            "id": saved["id"],
            "hint": HINT_NOT_FOUND,
        }
        assert await _call(mcp_user, "user_search", {"query": FACT_NAME}) == {
            "results": [],
            "hint": CANON_HINT_SEARCH_EMPTY,
        }


class TestOtherOutputsRegression:
    """Выдачи заметок и области навыков этим пулом не изменились."""

    @pytest.mark.asyncio
    async def test_notes_and_skills_outputs_unchanged(self, mcp_full) -> None:
        note = await _call(
            mcp_full,
            "memory_save",
            {"text": "Regression note for the user-area pool.", "title": "Regression note"},
        )
        assert set(note) == {"id", "stored", "summary_pending"}
        found = await _call(mcp_full, "memory_search", {"query": "regression"})
        assert set(found["results"][0]) == {
            "id",
            "summary",
            "created_at",
            "updated_at",
            "namespace",
            "title",
        }
        skill = await _call(
            mcp_full,
            "skills_save",
            {
                "name": "Deploy the service",
                "description": "How to deploy this service",
                "steps": "1) build; 2) ship; 3) verify",
                "text": "Run make deploy, then check /health.",
            },
        )
        assert set(skill) == {"id", "version", "created"}
        # Область «user» в этих выдачах не появляется (изоляция каналов).
        assert "user_facts" not in {tool.name for tool in await mcp_full.list_tools()}
