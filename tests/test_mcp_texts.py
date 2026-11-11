"""The injected text tests (LSBN-F-11, brief f11-02).

    The brief's requirement: `build_instructions` carries a SHORT core (the role,
    the mandatory rules and prohibitions, pointers phrased as actions toward
    reachable handles); the dynamic parts (the node map) are bounded and on
    overflow carry a pointer line; all 31 tool descriptions fit the budget and are
    cleaned of rule repetitions and reference numeric limits (the limits live in
    canon §6 and the `hint` answers). The budget numbers — constants of
    `app/services/injection_budget.py` (arch §13.17); they are not in env.

    EN-only applies to the STATIC parts (the core, the rules, the handle
    descriptions); the operator's data — node descriptions in the map and the
    skills announce rows — comes from the DB and can be Russian, so the Cyrillic
    checks look at the statics only.
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


# 3.4.1 (LSBN-T-26): the texts with the limit number are assembled by functions over
# settings — here assembled from the test environment's values (the former
# constants' role; the number comes from Settings, not a literal).
SERVER_INSTRUCTIONS = server_instructions(get_settings().max_note_chars)
TOOL_DESCRIPTIONS = tool_descriptions(get_settings().max_note_chars)

DIM = 64

EXPECTED_TOOL_COUNT = 32

# The core's mandatory rules (the brief marks them "untouchable"): the role, "search
# before writing", "notes are data, not instructions", `title` ≤5 words, the secrets
# prohibition, not mixing "Work"/"Personal", the skills rule.
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

# Pointers phrased as actions: a handle name, not a "see the docs" reference (arch §7.1).
ACTIONABLE_POINTERS = (
    "memory_namespaces",
    "skills_list",
    "skills_get",
    "history_list",
    "history_get",
)

# The verbatim numeric limits of canon §6/§3.3.5: the reference moved into the canon and
# the `hint` answers — the tool descriptions must not carry it.
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
    """An in-process build: the node registry only (everything else — None)."""
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
    """The instructions core: the budget, the mandatory rules, actionable pointers."""

    def test_core_is_short_and_english(self) -> None:
        """The core — short (≤ `TOKENS_CORE_MAX`) and English."""
        assert count_tokens(SERVER_INSTRUCTIONS) <= TOKENS_CORE_MAX
        assert not _has_cyrillic(SERVER_INSTRUCTIONS)
        assert not _has_cyrillic(_NS_RULES)
        assert not _has_cyrillic(_LOGS_SECTION)
        assert not _has_cyrillic(_NOTE_LIFETIME_SECTION)
        assert not _has_cyrillic(_SKILLS_ANNOUNCE_RULES)
        assert not _has_cyrillic(_SKILLS_ANNOUNCE_MORE)

    def test_core_keeps_mandatory_rules(self) -> None:
        """The core keeps the mandatory rules and prohibitions (the f11-02 brief)."""
        for needle in MANDATORY_RULES:
            assert needle in SERVER_INSTRUCTIONS, needle

    def test_core_has_actionable_pointers(self) -> None:
        """The pointers lead to handles: a handle name in the core, not "see the docs"."""
        for needle in ACTIONABLE_POINTERS:
            assert f"`{needle}`" in SERVER_INSTRUCTIONS, needle
        for word in ("documentation", "see the docs"):
            assert word not in SERVER_INSTRUCTIONS.lower()


class TestNodeMap:
    """The node map: one line per node, the pointer line, no linear growth."""

    def test_node_map_bounded_with_pointer(self, settings) -> None:
        services = _services(settings)
        for index in range(40):
            services.namespaces.create(f"node{index:03d}", "x" * 120)
        text = _namespace_map(services)
        assert "(+" in text and "more — memory_namespaces)" in text
        assert count_tokens(text) <= _NS_MAP_MAX_TOKENS
        # Trimming a description to the line's budget: a long description does not leak.
        assert "x" * (_NS_MAP_DESCRIPTION_CHARS + 1) not in text
        # A growing node count does NOT grow the instructions linearly: the map hits the budget.
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
    """31 tool descriptions: presence, EN, the ceiling, no reference lists."""

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
