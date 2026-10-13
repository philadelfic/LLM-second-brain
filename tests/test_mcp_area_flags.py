"""Гейт MCP-поверхности по флагам областей (№40, FR-2.1/FR-2.4, FR-4.1).

Выключенная область в поверхность не попадает вовсе: её блок ручек не
регистрируется, анонс навыков не вшивается по флагу области; включённые
области и ядровые инструменты (память, история) на месте. Состав поверхности
детерминирован одним набором env и совпадает с реестром `enabled_areas`.
"""

from __future__ import annotations

import pytest
from fakes import clear_seeded_skills

from app.config import get_settings
from app.services import build_services
from app.services.areas import enabled_areas
from app.storage.db import init_db, session, transaction
from app.transport.mcp import build_instructions, build_mcp

DIM = 64

# Ручек на область ровно столько, сколько объявлено (no40-01).
AREA_TOOLS = {
    "skills": frozenset(
        {
            "skills_search",
            "skills_list",
            "skills_get",
            "skills_save",
            "skills_delete",
        }
    ),
    "terms": frozenset({"terms_search", "terms_save", "terms_get"}),
    "user": frozenset(
        {"user_search", "user_save", "user_update", "user_delete", "user_get"}
    ),
}

FLAG_NAMES = (
    "AREA_SKILLS_ENABLED",
    "AREA_TERMS_ENABLED",
    "AREA_USER_FACTS_ENABLED",
)

_ANNOUNCE_MARKER = "Skills are stored procedures"


def _settings(monkeypatch: pytest.MonkeyPatch, **flags: str):
    """Настройки с заданным набором флагов (остальные — сняты, дефолт true)."""
    for name in FLAG_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in flags.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)
    return settings


def _seed_skill(settings) -> None:
    """Активный навык — чтобы анонс был непустым при включённой области."""
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO skills (name, description, steps, text) "
            "VALUES ('Деплой релиза', 'как катить релиз', 'шаги', 'текст')"
        )


@pytest.mark.asyncio
async def test_disabled_area_tools_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """AREA_SKILLS_ENABLED=false: ручек skills_* нет, прочие области на месте."""
    settings = _settings(monkeypatch, AREA_SKILLS_ENABLED="false")
    names = {
        tool.name
        for tool in await build_mcp(settings, build_services(settings)).list_tools()
    }
    assert not (AREA_TOOLS["skills"] & names)
    assert AREA_TOOLS["terms"] <= names
    assert AREA_TOOLS["user"] <= names
    # Ядровые инструменты не трогаются гейтом области.
    assert {"memory_search", "memory_list", "memory_get"} <= names
    assert {"history_list", "history_get"} <= names


@pytest.mark.asyncio
async def test_disabled_area_announce_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Анонс навыков гасится ПО ФЛАГУ области (а не по None-сервису)."""
    settings = _settings(monkeypatch, AREA_SKILLS_ENABLED="false")
    _seed_skill(settings)
    assert _ANNOUNCE_MARKER not in build_instructions(build_services(settings))

    enabled = _settings(monkeypatch)  # флаг снят → дефолт true
    _seed_skill(enabled)
    assert _ANNOUNCE_MARKER in build_instructions(build_services(enabled))


@pytest.mark.asyncio
async def test_surface_matches_services(monkeypatch: pytest.MonkeyPatch) -> None:
    """Для каждого набора флагов поверхность = реестр включённых областей."""
    combos = (
        {},
        {"AREA_SKILLS_ENABLED": "false"},
        {"AREA_TERMS_ENABLED": "false"},
        {"AREA_USER_FACTS_ENABLED": "false"},
        {
            "AREA_SKILLS_ENABLED": "false",
            "AREA_TERMS_ENABLED": "false",
            "AREA_USER_FACTS_ENABLED": "false",
        },
    )
    for flags in combos:
        settings = _settings(monkeypatch, **flags)
        services = build_services(settings)
        # Сервисы собраны всегда (решение 2026-10-08) — флаг на сборку не влияет.
        assert services.skills is not None
        assert services.terms is not None
        assert services.user_facts is not None

        names = {
            tool.name
            for tool in await build_mcp(settings, services).list_tools()
        }
        enabled = {spec.name for spec in enabled_areas(settings)}
        for area, tools in AREA_TOOLS.items():
            if area in enabled:
                assert tools <= names, f"{area} enabled but tools missing: {flags}"
            else:
                assert not (tools & names), f"{area} disabled but tools present: {flags}"
