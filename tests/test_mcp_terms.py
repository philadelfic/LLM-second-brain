"""Тесты MCP-поверхности области terms (lsb-0008-02): 3 инструмента, компактные
выдачи, все каналы hint'ов; листинга нет, инъекции в инструкции нет.

ARCH lsb-0008 §3.4–3.5, §3.7: описания инструментов и hint'ы — дословный канон
(сверяем литералами), выдачи компактны (белые списки; внутренние нормализованные
колонки не просачиваются), `terms_search` отдаёт ВСЕ смыслы термина (`exact:
true`), неточная ветка — ближайшие по смыслу `+term/score` и hint «не точное
совпадение», `terms_save` по ключу (term + context) обновляет запись, новый
контекст создаёт смысл без затирания старого, близкий контекст — мягкий отказ ДО
записи с hint'ом на существующий контекст, лимиты/пустой контекст — дословные
hint'ы. Инъекции при initialize нет (arch §2): блока «terms» в instructions не
появляется; текст инструкций прежний (анонс навыков существовал и раньше).
Поверхность — in-process MCP, эмбеддер подменён детерминированным фейком.
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.services import Services, build_services
from app.services.namespaces import NamespaceService
from app.services.terms import (
    HINT_CONTEXT_CLOSE,
    HINT_CONTEXT_LIMIT,
    HINT_CONTEXT_REQUIRED,
    HINT_DEFINITION_LIMIT,
    HINT_NO_EXACT,
    HINT_NOT_FOUND,
    HINT_TERM_LIMIT,
    TermsService,
)
from app.services.user_facts import UserFactsService
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
from fakes import HashEmbedder

DIM = 64

# --- Дословные литералы канона arch lsb-0008 §3.7 (править только в арх-доке) -

CANON_TOOL_DESCRIPTIONS = {
    "terms_search": (
        "Look up a term or an abbreviation in the terms area. Returns ALL "
        "senses of the term together with their contexts: one term means "
        "different things in different contexts — never pick a single sense "
        "silently, choose by the context of the conversation, and if it is "
        "unclear, ask the user instead of guessing. If there is no exact term, "
        "the closest senses by meaning are returned (not an exact match). There "
        "is no listing — search is the only way to a definition."
    ),
    "terms_save": (
        "Save a term with a MANDATORY context: term ≤100 characters, context "
        "≤40 (always filled in — even when the term has a single sense), "
        "definition ≤350. The key is (term + context): the same key updates the "
        "record; the same term with a new context creates a NEW sense and never "
        "overwrites the old one. Reuse one of the contexts already used in this "
        "memory (the response lists them) instead of inventing a near-duplicate "
        "wording — a too-close context is refused with a hint that points to the "
        "existing context. The response also lists the senses this term already "
        "has."
    ),
    "terms_get": "Read one term record by id: term, context, definition.",
}

# Field-описания параметров (EN): канон §3.7 задаёт тексты инструментов и
# hint'ы; описания полей собраны из той же канонической формулировки §3.1
# (лимиты 100/40/350, «always filled in — even when the term has a single
# sense») — как в пулах областей навыков и «user».
CANON_FIELD_DESCRIPTIONS = {
    "terms_search": {
        "query": "Term or abbreviation to look up; all senses are returned",
        "top_k": "Number of results",
    },
    "terms_save": {
        "term": "Term: ≤100 characters",
        "context": (
            "Context: ≤40 characters; always required — even for a single sense"
        ),
        "definition": "Definition: ≤350 characters",
    },
    "terms_get": {"id": "Term record id"},
}

# Hint'ы мягких отказов канона §3.7 (дословно).
CANON_HINT_NO_EXACT = (
    "no exact term — the senses above are the closest by meaning, not an "
    "exact match: check them against the context of the conversation"
)
CANON_HINT_NOT_FOUND = (
    "term not found — there is no such term in this memory; save it via "
    "terms_save(term, context, definition) if it is worth keeping"
)
CANON_HINT_CONTEXT_CLOSE = (
    "not saved: context '{given}' is too close to the existing one "
    "'{existing}' (id={id}) — reuse that context wording, or make the new "
    "context clearly different"
)
CANON_HINT_TERM_LIMIT = "not saved: term limit is 100 characters — shorten it"
CANON_HINT_CONTEXT_LIMIT = "not saved: context limit is 40 characters — shorten it"
CANON_HINT_DEFINITION_LIMIT = (
    "not saved: definition limit is 350 characters — shorten it"
)
CANON_HINT_CONTEXT_REQUIRED = (
    "not saved: context is required — specify the context in which the term "
    "is used (≤40 characters)"
)

# 8 ручек заметок/узлов, 5 навыков и 5 «user» (lsb-0006/lsb-0007-03/lsb-0009-02):
# в этом пуле их имена и выдачи не меняются.
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
USER_TOOL_NAMES = frozenset(
    {"user_search", "user_save", "user_update", "user_delete", "user_get"}
)

TERM = "ГЗ"
CONTEXT_A = "студенты МГУ"
CONTEXT_B = "бухгалтерия"


def _services(settings, embedding) -> Services:
    """Сервисы in-process MCP: область terms + DI-эмбеддер (прочее — None)."""
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
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def mcp_terms(settings) -> object:
    """In-process MCP области terms с детерминированным эмбеддером."""
    return build_mcp(settings, _services(settings, HashEmbedder(DIM)))


@pytest.fixture
def mcp_full(settings) -> object:
    """In-process MCP на полной сборке: регресс выдач заметок/навыков/user."""
    return build_mcp(settings, build_services(settings))


def _active_count(settings) -> int:
    """Число активных записей области (отказ записи проверять нечего)."""
    with session(settings) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM terms WHERE deleted_at IS NULL"
        ).fetchone()
    return int(row["n"])


async def _schemas(mcp) -> dict[str, dict]:
    """Схемы аргументов всех инструментов поверхности (in-process MCP)."""
    return {tool.name: tool.input_schema for tool in await mcp.list_tools()}


async def _call(mcp, name: str, arguments: dict) -> dict:
    """Вызвать инструмент и вернуть структурированную (компактную) выдачу."""
    result = await mcp.call_tool(name, arguments)
    return result.structured_content


async def _save(mcp, term=TERM, context=CONTEXT_A, definition="госэкзамен") -> dict:
    return await _call(
        mcp, "terms_save", {"term": term, "context": context, "definition": definition}
    )


class TestToolRegistry:
    """Регистрация 3 ручек области terms: описания и Field-описания — канон."""

    @pytest.mark.asyncio
    async def test_three_terms_tools_added_without_touching_others(
        self, mcp_terms
    ) -> None:
        names = {tool.name for tool in await mcp_terms.list_tools()}
        assert set(CANON_TOOL_DESCRIPTIONS) <= names
        assert names - set(CANON_TOOL_DESCRIPTIONS) == (
            MEMORY_TOOL_NAMES | SKILL_TOOL_NAMES | USER_TOOL_NAMES
        )
        assert set(TOOL_NAMES) == names
        assert len(names) == 21  # листинга terms нет: только search/save/get

    @pytest.mark.asyncio
    async def test_tool_descriptions_are_canon_verbatim(self, mcp_terms) -> None:
        tools = {tool.name: tool for tool in await mcp_terms.list_tools()}
        for name, canon in CANON_TOOL_DESCRIPTIONS.items():
            assert TOOL_DESCRIPTIONS[name] == canon
            assert tools[name].description == canon

    @pytest.mark.asyncio
    async def test_field_descriptions_are_canon_verbatim(self, mcp_terms) -> None:
        schemas = await _schemas(mcp_terms)
        for tool_name, expected in CANON_FIELD_DESCRIPTIONS.items():
            props = schemas[tool_name]["properties"]
            assert set(props) == set(expected), tool_name
            for param, description in expected.items():
                assert props[param]["description"] == description, (tool_name, param)

    @pytest.mark.asyncio
    async def test_required_params_bounds_and_no_schema_limits(
        self, mcp_terms
    ) -> None:
        """Контракт формы: обязательные поля, границы top_k — как у поиска.

        Лимиты term/context/definition валидирует СЕРВИС (не схема) — иначе не
        будет дословного hint'а канона §3.7.
        """
        schemas = await _schemas(mcp_terms)
        assert schemas["terms_save"]["required"] == ["term", "context", "definition"]
        assert schemas["terms_search"]["required"] == ["query"]
        assert schemas["terms_get"]["required"] == ["id"]
        assert set(schemas["terms_search"]["properties"]) == {"query", "top_k"}
        top_k = schemas["terms_search"]["properties"]["top_k"]
        assert top_k["default"] == get_settings().default_top_k
        assert top_k["minimum"] == 1 and top_k["maximum"] == 20
        for param in ("term", "context", "definition"):
            assert "maxLength" not in schemas["terms_save"]["properties"][param]

    def test_hint_constants_are_canon_verbatim(self) -> None:
        assert HINT_NO_EXACT == CANON_HINT_NO_EXACT
        assert HINT_NOT_FOUND == CANON_HINT_NOT_FOUND
        assert HINT_TERM_LIMIT == CANON_HINT_TERM_LIMIT
        assert HINT_CONTEXT_LIMIT == CANON_HINT_CONTEXT_LIMIT
        assert HINT_DEFINITION_LIMIT == CANON_HINT_DEFINITION_LIMIT
        assert HINT_CONTEXT_REQUIRED == CANON_HINT_CONTEXT_REQUIRED
        filled = {"given": "g", "existing": "e", "id": 1}
        assert HINT_CONTEXT_CLOSE.format(**filled) == (
            CANON_HINT_CONTEXT_CLOSE.format(**filled)
        )

    def test_instructions_have_no_terms_block_and_unchanged(self, settings) -> None:
        """Инъекции нет (arch §2): блока «terms» нет и текст не изменился.

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
        for needle in ("terms_search", "terms_save", "terms_get", "terms area"):
            assert needle not in text
        assert build_mcp(settings, services).instructions == text


class TestSearch:
    """Поиск: все смыслы (exact), ближайшие (не exact), hint'ы §3.7."""

    @pytest.mark.asyncio
    async def test_exact_search_returns_all_senses_without_internal_fields(
        self, mcp_terms
    ) -> None:
        first = await _save(mcp_terms)
        second = await _save(mcp_terms, context=CONTEXT_B, definition="годовой отчёт")
        found = await _call(mcp_terms, "terms_search", {"query": TERM})
        assert found == {
            "senses": [
                {"id": first["id"], "context": CONTEXT_A, "definition": "госэкзамен"},
                {
                    "id": second["id"],
                    "context": CONTEXT_B,
                    "definition": "годовой отчёт",
                },
            ],
            "exact": True,
        }
        for sense in found["senses"]:
            # Внутренние нормализованные колонки и term/score в точной ветке
            # не выводятся: смысл опознан по термину.
            assert set(sense) == {"id", "context", "definition"}

    @pytest.mark.asyncio
    async def test_near_search_returns_closest_with_term_score_and_hint(
        self, mcp_terms
    ) -> None:
        saved = await _save(mcp_terms)
        found = await _call(mcp_terms, "terms_search", {"query": "госэкзамен"})
        assert found["exact"] is False
        assert found["hint"] == CANON_HINT_NO_EXACT
        assert "warning" not in found  # деградация остаётся в REST/логе
        sense = found["senses"][0]
        assert sense["id"] == saved["id"]
        assert sense["term"] == TERM
        assert sense["context"] == CONTEXT_A
        assert sense["definition"] == "госэкзамен"
        assert sense["score"] > 0  # RRF-слияние области
        assert set(sense) == {"id", "term", "context", "definition", "score"}

    @pytest.mark.asyncio
    async def test_no_match_soft_answer_with_not_found_hint(self, mcp_terms) -> None:
        assert await _call(
            mcp_terms, "terms_search", {"query": "quantum gardening"}
        ) == {"senses": [], "exact": False, "hint": CANON_HINT_NOT_FOUND}


class TestSave:
    """Запись по ключу (term + context): создание смысла, правка, отказы §3.5."""

    @pytest.mark.asyncio
    async def test_new_context_creates_sense_and_lists_senses_and_contexts(
        self, mcp_terms, settings
    ) -> None:
        first = await _save(mcp_terms)
        assert first == {
            "created": True,
            "id": first["id"],
            "senses": [{"id": first["id"], "context": CONTEXT_A}],
            "contexts": [CONTEXT_A],
        }
        second = await _save(mcp_terms, context=CONTEXT_B, definition="годовой отчёт")
        assert second["created"] is True
        assert second["id"] != first["id"]  # новый смысл, не перезапись
        assert second["senses"] == [
            {"id": first["id"], "context": CONTEXT_A},
            {"id": second["id"], "context": CONTEXT_B},
        ]
        assert set(second["contexts"]) == {CONTEXT_A, CONTEXT_B}
        assert all(isinstance(item, str) for item in second["contexts"])
        assert _active_count(settings) == 2

    @pytest.mark.asyncio
    async def test_same_key_updates_record_without_new_sense(
        self, mcp_terms, settings
    ) -> None:
        first = await _save(mcp_terms)
        again = await _save(mcp_terms, definition="государственный экзамен")
        assert again == {
            "updated": True,
            "id": first["id"],
            "senses": [{"id": first["id"], "context": CONTEXT_A}],
            "contexts": [CONTEXT_A],
        }
        assert _active_count(settings) == 1  # дубля ключа нет
        assert await _call(mcp_terms, "terms_get", {"id": first["id"]}) == {
            "id": first["id"],
            "term": TERM,
            "context": CONTEXT_A,
            "definition": "государственный экзамен",
        }

    @pytest.mark.asyncio
    async def test_close_context_soft_refusal_with_existing_context_hint(
        self, mcp_terms, settings
    ) -> None:
        first = await _save(mcp_terms)
        # Триграммное сходство с CONTEXT_A = 1.0 (>= порога 0.75), ключи разные.
        given = "студенты МГУ в вузе"
        result = await _save(mcp_terms, context=given)
        assert result == {
            "created": False,
            "hint": CANON_HINT_CONTEXT_CLOSE.format(
                given=given, existing=CONTEXT_A, id=first["id"]
            ),
        }
        assert _active_count(settings) == 1  # записи нет: подсказка ДО записи

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("payload", "hint"),
        [
            (
                {"term": "x" * 101, "context": "c", "definition": "d"},
                CANON_HINT_TERM_LIMIT,
            ),
            (
                {"term": "t", "context": "c" * 41, "definition": "d"},
                CANON_HINT_CONTEXT_LIMIT,
            ),
            (
                {"term": "t", "context": "c", "definition": "d" * 351},
                CANON_HINT_DEFINITION_LIMIT,
            ),
            (
                {"term": "t", "context": "", "definition": "d"},
                CANON_HINT_CONTEXT_REQUIRED,
            ),
            (
                {"term": "t", "context": "   ", "definition": "d"},
                CANON_HINT_CONTEXT_REQUIRED,
            ),
        ],
    )
    async def test_limits_and_empty_context_soft_refusals(
        self, mcp_terms, settings, payload: dict, hint: str
    ) -> None:
        assert await _call(mcp_terms, "terms_save", payload) == {
            "created": False,
            "hint": hint,
        }
        assert _active_count(settings) == 0


class TestGet:
    """Чтение записи по id; не найдена — мягкий ответ с hint канона §3.7."""

    @pytest.mark.asyncio
    async def test_get_found_returns_record_and_missing_gives_hint(
        self, mcp_terms
    ) -> None:
        saved = await _save(mcp_terms)
        assert await _call(mcp_terms, "terms_get", {"id": saved["id"]}) == {
            "id": saved["id"],
            "term": TERM,
            "context": CONTEXT_A,
            "definition": "госэкзамен",
        }
        assert await _call(mcp_terms, "terms_get", {"id": 999}) == {
            "hint": CANON_HINT_NOT_FOUND
        }


class TestOtherOutputsRegression:
    """Выдачи заметок, навыков и области «user» этим пулом не изменились."""

    @pytest.mark.asyncio
    async def test_notes_skills_user_outputs_unchanged_and_areas_isolated(
        self, mcp_full
    ) -> None:
        note = await _call(
            mcp_full,
            "memory_save",
            {"text": "Regression note for the terms-area pool.", "title": "Regression note"},
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
        fact = await _call(
            mcp_full, "user_save", {"name": "Answer style", "body": "prefers brevity"}
        )
        assert set(fact) == {"id", "stored", "hint"}
        # Изоляция в обе стороны: термин не виден в заметках, заметка — в terms.
        await _save(mcp_full)
        assert (await _call(mcp_full, "memory_search", {"query": TERM}))[
            "results"
        ] == []
        assert await _call(mcp_full, "terms_search", {"query": "regression"}) == {
            "senses": [],
            "exact": False,
            "hint": CANON_HINT_NOT_FOUND,
        }
