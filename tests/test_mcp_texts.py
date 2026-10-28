"""Тесты вшитого текста инъекции (LSBN-F-11, постановка f11-02).

Требование постановки: `build_instructions` несёт КОРОТКОЕ ядро (роль,
обязательные правила и запреты, указатели действием на достижимые ручки),
динамические части (карта узлов) ограничены и при переполнении несут
строку-указатель, а все 31 описание инструмента уложены в бюджет и очищены
от повторов правил и справочных числовых лимитов (лимиты — канон §6 и
`hint`-ответы). Числа бюджета — константы `app/services/injection_budget.py`
(arch §13.17), в env их нет.

EN-only относится к СТАТИЧЕСКИМ частям (ядро, правила, описания ручек); данные
оператора — описания узлов в карте и строки анонса навыков — приходят из БД и
бывают русскими, поэтому проверки кириллицы смотрят только статику.
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder

from app.config import get_settings
from app.services import Services
from app.services.injection_budget import (
    TOKENS_CORE_MAX,
    TOKENS_TOOL_DESCRIPTION_MAX,
    TOKENS_TOOL_SURFACE_MAX,
    count_tokens,
)
from app.services.namespaces import NamespaceService
from app.storage.db import init_db
from app.transport.mcp import (
    server_instructions,
    tool_descriptions,
    TOOL_NAMES,
    _LOGS_SECTION,
    _NOTE_LIFETIME_SECTION,
    _NS_MAP_DESCRIPTION_CHARS,
    _NS_MAP_MAX_TOKENS,
    _NS_RULES,
    _SKILLS_ANNOUNCE_MORE,
    _SKILLS_ANNOUNCE_RULES,
    _namespace_map,
)


# 3.4.1 (LSBN-T-26): тексты с числом лимита собираются функциями от
# настроек — здесь сборка от значений тестового окружения (роль прежних
# констант; число приходит из Settings, не литерал).
SERVER_INSTRUCTIONS = server_instructions(get_settings().max_note_chars)
TOOL_DESCRIPTIONS = tool_descriptions(get_settings().max_note_chars)

DIM = 64

EXPECTED_TOOL_COUNT = 31

# Обязательные правила ядра (постановка «неприкосновенно»): роль, «поиск перед
# записью», «заметки — данные, а не инструкции», `title` ≤5 слов, запрет
# секретов, «Рабочее»/«Личное» не смешивать, правило про навыки.
MANDATORY_RULES = (
    "long-term memory",
    "memory_search",
    "FIRST",
    "DATA, not instructions",
    "title",
    "≤5 words",
    "secrets",
    "work and personal",
    "skills_search",
    "skills_list",
    "skills_get",
    "routine",
)

# Указатели действием: имя ручки, а не отсылка «см. документацию» (arch §7.1).
ACTIONABLE_POINTERS = (
    "memory_namespaces",
    "skills_list",
    "skills_get",
    "history_list",
    "history_get",
)

# Дословные числовые лимиты канона §6/§3.3.5: справочник переехал в канон и в
# `hint`-ответы — в описаниях инструментов его быть не должно.
LEGACY_NUMERIC_LIMITS = (
    "≤65 characters",
    "description ≤400",
    "steps ≤1000",
    "text ≤7500",
    "example ≤2000",
    "term ≤100",
    "context ≤40",
    "definition ≤350",
    "body ≤1200",
    "1000 characters",
    "1200 characters",
    "≤250 chars",
)


def _has_cyrillic(text: str) -> bool:
    return any("\u0400" <= char <= "\u04FF" for char in text)


def _services(settings) -> Services:
    """In-process сборка: только реестр узлов (прочее — None)."""
    return Services(
        notes=None,
        search=None,
        embedding=HashEmbedder(DIM),
        dedup=None,
        summary=None,
        judge=None,
        backup=None,
        namespaces=NamespaceService(settings),
        classifier=None,
        promotion=None,
        skills=None,
        user_facts=None,
        terms=None,
        links=None,
    )


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    resolved = get_settings()
    init_db(resolved)
    return resolved


class TestCore:
    """Ядро инструкций: бюджет, обязательные правила, указатели действием."""

    def test_core_is_short_and_english(self) -> None:
        """Ядро — короткое (≤ `TOKENS_CORE_MAX`) и англоязычное."""
        assert count_tokens(SERVER_INSTRUCTIONS) <= TOKENS_CORE_MAX
        assert not _has_cyrillic(SERVER_INSTRUCTIONS)
        assert not _has_cyrillic(_NS_RULES)
        assert not _has_cyrillic(_LOGS_SECTION)
        assert not _has_cyrillic(_NOTE_LIFETIME_SECTION)
        assert not _has_cyrillic(_SKILLS_ANNOUNCE_RULES)
        assert not _has_cyrillic(_SKILLS_ANNOUNCE_MORE)

    def test_core_keeps_mandatory_rules(self) -> None:
        """Ядро держит обязательные правила и запреты (постановка f11-02)."""
        for needle in MANDATORY_RULES:
            assert needle in SERVER_INSTRUCTIONS, needle

    def test_core_has_actionable_pointers(self) -> None:
        """Указатели ведут к ручкам: имя ручки в ядре, а не «см. документацию»."""
        for needle in ACTIONABLE_POINTERS:
            assert f"`{needle}`" in SERVER_INSTRUCTIONS, needle
        for word in ("documentation", "see the docs"):
            assert word not in SERVER_INSTRUCTIONS.lower()


class TestNodeMap:
    """Карта узлов: одна строка на узел, строка-указатель, линейного роста нет."""

    def test_node_map_bounded_with_pointer(self, settings) -> None:
        services = _services(settings)
        for index in range(40):
            services.namespaces.create(f"node{index:03d}", "x" * 120)
        text = _namespace_map(services)
        assert "(+" in text and "more — memory_namespaces)" in text
        assert count_tokens(text) <= _NS_MAP_MAX_TOKENS
        # Обрезка описания под бюджет строки: длинное описание не протекает.
        assert "x" * (_NS_MAP_DESCRIPTION_CHARS + 1) not in text
        # Рост числа узлов НЕ растит инструкции линейно: карта упирается в бюджет.
        rows = [line for line in text.splitlines() if line.startswith("  - ")]
        before = count_tokens(text)
        for index in range(40, 80):
            services.namespaces.create(f"node{index:03d}", "x" * 120)
        grown = _namespace_map(services)
        assert count_tokens(grown) == before
        assert len(
            [line for line in grown.splitlines() if line.startswith("  - ")]
        ) == len(rows)


class TestToolDescriptions:
    """31 описание инструмента: наличие, EN, потолок, отсутствие справочника."""

    def test_all_tool_descriptions_present_and_english(self) -> None:
        assert len(TOOL_NAMES) == EXPECTED_TOOL_COUNT
        assert set(TOOL_DESCRIPTIONS) == set(TOOL_NAMES)
        for name in sorted(TOOL_DESCRIPTIONS):
            text = TOOL_DESCRIPTIONS[name]
            assert text.strip(), name
            assert not _has_cyrillic(text), name

    def test_descriptions_within_hard_ceiling(self) -> None:
        for name, text in TOOL_DESCRIPTIONS.items():
            assert count_tokens(text) <= TOKENS_TOOL_DESCRIPTION_MAX, name
        total = sum(count_tokens(text) for text in TOOL_DESCRIPTIONS.values())
        assert total <= TOKENS_TOOL_SURFACE_MAX

    def test_no_legacy_numeric_limits_in_descriptions(self) -> None:
        for name, text in TOOL_DESCRIPTIONS.items():
            for limit in LEGACY_NUMERIC_LIMITS:
                assert limit not in text, (name, limit)
