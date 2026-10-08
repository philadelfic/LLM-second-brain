"""Флаги областей (№40, релиз 3.2.0): конфигурация, реестр включённых, REST.

FR-1.1…FR-1.4: env-флаги `AREA_*_ENABLED` (дефолт «включено») и реестр
`enabled_areas(settings)` фильтруют `ALL_AREAS`; невалидное значение —
фатальный старт с перечислением ВСЕХ нарушений (ConfigError, NFR-6), а не
молчаливое выключение. Сервисы областей при выключенном флаге по-прежнему
собираются (решение владельца 2026-10-08): операторский REST и слой истории
должны иметь доступ к данным выключенной области — флаг гейтит поверхность
MCP, раздел/анонс в инструкции и фоновую петлю `areas`, но не сборку.
"""

from __future__ import annotations

import asyncio
import itertools

import pytest
from fastapi.testclient import TestClient

from app.config import ConfigError, get_settings, load_settings
from app.main import create_app
from app.services import build_services
from app.services.areas import ALL_AREAS, enabled_areas
from app.transport.mcp import build_mcp


def _settings(monkeypatch: pytest.MonkeyPatch, **flags: str):
    """Пересобрать настройки с флагами областей (кэш снимаем — env меняется)."""
    for name, value in flags.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    return get_settings()


def _tool_names(mcp) -> set[str]:
    """Имена инструментов собранной MCP-поверхности (in-process)."""
    async def _list() -> set[str]:
        return {tool.name for tool in await mcp.list_tools()}

    return asyncio.run(_list())


class TestRegistry:
    def test_defaults_all_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Дефолт без правок compose: все три области включены и собраны."""
        settings = _settings(monkeypatch)
        assert [spec.name for spec in enabled_areas(settings)] == [
            "skills",
            "terms",
            "user",
        ]
        services = build_services(settings)
        assert services.skills is not None
        assert services.terms is not None
        assert services.user_facts is not None

    def test_every_area_has_settings_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """У каждой области реестра есть имя поля-флага (`settings_flag`)."""
        settings = _settings(monkeypatch)
        for spec in ALL_AREAS:
            assert spec.settings_flag.startswith("area_")
            assert isinstance(getattr(settings, spec.settings_flag), bool)


class TestDisabledSurface:
    def test_disabled_area_surface_gated(
        self, monkeypatch: pytest.MonkeyPatch, token: str
    ) -> None:
        """AREA_TERMS_ENABLED=false: нет в реестре/на MCP, но REST работает."""
        settings = _settings(monkeypatch, AREA_TERMS_ENABLED="false")
        assert [spec.name for spec in enabled_areas(settings)] == ["skills", "user"]

        services = build_services(settings)
        # Решение 2026-10-08: сервис собран всегда — данные доступны REST/истории.
        assert services.terms is not None
        names = _tool_names(build_mcp(settings, services))
        assert not {name for name in names if name.startswith("terms_")}
        assert {"skills_search", "user_search"} <= names

        # REST оператора остаётся полным: запись термина работает.
        app = create_app(settings)
        with TestClient(app) as client:
            response = client.post(
                "/terms",
                json={
                    "term": "ГЗ",
                    "context": "студенты МГУ",
                    "definition": "государственный экзамен",
                },
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 201, response.text
            assert response.json()["created"] is True


class TestInvalidFlag:
    def test_invalid_flag_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Невалидный флаг — фатальный старт; в сообщении ВСЕ нарушения."""
        get_settings.cache_clear()
        monkeypatch.setenv("AREA_USER_FACTS_ENABLED", "maybe")
        monkeypatch.setenv("AREA_TERMS_ENABLED", "nope")
        with pytest.raises(ConfigError) as exc:
            load_settings()
        message = str(exc.value)
        assert "area_user_facts_enabled" in message
        assert "area_terms_enabled" in message


# Флаги областей (порядок — как в реестре) и ядро агентской поверхности:
# 8 ручек памяти + 2 общие ручки истории (№41) + 1 предупреждения (№12).
# Ядро гейту области не подлежит.
_FLAG_NAMES = (
    "AREA_SKILLS_ENABLED",
    "AREA_TERMS_ENABLED",
    "AREA_USER_FACTS_ENABLED",
)
_CORE_TOOLS = frozenset(
    {
        "memory_search",
        "memory_list",
        "memory_get",
        "memory_save",
        "memory_update",
        "memory_delete",
        "memory_namespaces",
        "memory_namespace_create",
        "history_list",
        "history_get",
        "warnings_list",
    }
)


def _areas_in_surface(names: set[str]) -> set[str]:
    """Области, чьи ручки фактически зарегистрированы (определяем по префиксу)."""
    return {
        spec.name
        for spec in ALL_AREAS
        if any(name.startswith(f"{spec.name}_") for name in names)
    }


def test_surface_consistency(monkeypatch: pytest.MonkeyPatch) -> None:
    """FR-4.1/FR-4.2: состав агентской поверхности детерминированно выводится из флагов.

    Для ВСЕХ комбинаций флагов: области, чьи ручки зарегистрированы, совпадают
    с реестром `enabled_areas` (ни лишних, ни пропавших), ядро неизменно,
    сервисы областей собраны всегда (решение 2026-10-08 — сборка от флага не
    зависит), а та же конфигурация даёт тот же состав.
    """
    for combo in itertools.product(("true", "false"), repeat=len(_FLAG_NAMES)):
        flags = dict(zip(_FLAG_NAMES, combo))
        settings = _settings(monkeypatch, **flags)
        services = build_services(settings)
        # Сервис собран всегда: операторский REST и история данных не теряют.
        assert services.skills is not None
        assert services.terms is not None
        assert services.user_facts is not None

        names = _tool_names(build_mcp(settings, services))
        enabled = {spec.name for spec in enabled_areas(settings)}
        assert _areas_in_surface(names) == enabled, flags
        assert _CORE_TOOLS <= names, flags
        # Лишних ручек нет: всё вне ядра принадлежит ВКЛЮЧЁННОЙ области.
        prefixes = tuple(f"{area}_" for area in enabled)
        assert all(name.startswith(prefixes) for name in names - _CORE_TOOLS), flags
        # Детерминизм: та же конфигурация → тот же состав поверхности.
        rebuilt = _tool_names(build_mcp(settings, build_services(settings)))
        assert rebuilt == names, flags
