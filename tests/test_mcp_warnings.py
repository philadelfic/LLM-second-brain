"""№12 (no12-02, 3.2.0): MCP `warnings_list` — единая поверхность предупреждений.

Arch `jobs-and-terms-architecture.md` §7.1/§13.4: один инструмент на систему
(предупреждения памяти и терминов пишут в общую таблицу `warnings`); выдача —
белый список (`id`, `kind`, `subject{type,id}`, `message`, `payload`,
`created_at`) без тел заметок/терминов и без служебных `status`/`resolved_at`;
поля страницы — из общего `page_fields`; сверх потолка `WARNINGS_MAX_LIMIT_MCP`
— мягкий отказ с hint (не schema-error). Чтение — только SQL, без модели.

Поверхность — in-process MCP (один код с REST); полный контракт страницы и
`/health.warnings` — tests/test_rest_warnings.py и test_health_warnings.py.
"""

from __future__ import annotations

import pytest
from fakes import clear_seeded_skills

from app.config import Settings, get_settings
from app.services import Services, build_services
from app.services.warnings import (
    KIND_NOTE_MISPLACED,
    KIND_NOTE_NO_NODE,
    KIND_TERM_CONTEXTS_EXCESS,
)
from app.storage.db import init_db
from app.transport.mcp import TOOL_NAMES, build_mcp

DIM = 64


@pytest.fixture
def settings_and_services(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> tuple[Settings, Services]:
    """Свежая БД + полная сборка сервисов (пустой реестр навыков)."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)  # пустой реестр: счётчики предупреждений точны
    return settings, build_services(settings)


@pytest.fixture
def mcp_server(settings_and_services: tuple[Settings, Services]):
    """In-process MCP поверх тех же сервисов (один код с REST)."""
    settings, services = settings_and_services
    return build_mcp(settings, services)


def seed_warning(
    services: Services,
    *,
    kind: str = KIND_NOTE_MISPLACED,
    subject_type: str = "note",
    subject_id: int = 1,
    message: str = "note 1 is in 'default' but the classifier places it in 'work'",
) -> int:
    """Открыть предупреждение сервисом — SQL, без модели (seed для чтения)."""
    return services.warnings.open(  # type: ignore[union-attr]
        kind, subject_type, subject_id, message, {"hint_path": "work"}
    )


@pytest.mark.asyncio
async def test_list_compact_shape(
    settings_and_services: tuple[Settings, Services], mcp_server
) -> None:
    """Выдача — белый список: subject{type,id}; тел и служебных полей нет."""
    _, services = settings_and_services
    warning_id = seed_warning(services)
    got = (await mcp_server.call_tool("warnings_list", {})).structured_content
    assert got["total"] == 1 and got["has_more"] is False
    assert got["next_offset"] is None
    item = got["warnings"][0]
    assert set(item) == {
        "id", "kind", "subject", "message", "payload", "created_at"
    }
    assert item["id"] == warning_id
    assert item["kind"] == KIND_NOTE_MISPLACED
    assert item["subject"] == {"type": "note", "id": 1}
    assert item["payload"] == {"hint_path": "work"}
    # Служебные поля и тексты тел в MCP не просачиваются (§7.1, бюджет F-11).
    assert set(got) <= {"warnings", "total", "has_more", "next_offset", "hint"}
    assert "status" not in item and "resolved_at" not in item
    assert "text" not in item and "summary" not in item


@pytest.mark.asyncio
async def test_list_filters(
    settings_and_services: tuple[Settings, Services], mcp_server
) -> None:
    """Фильтры kind/subject_type/subject_id/status отбирают правильно."""
    _, services = settings_and_services
    seed_warning(services, subject_id=1)
    seed_warning(services, kind=KIND_NOTE_NO_NODE, subject_id=2)
    dismissed = seed_warning(
        services,
        kind=KIND_TERM_CONTEXTS_EXCESS,
        subject_type="term",
        subject_id=7,
    )
    services.warnings.dismiss(dismissed)  # type: ignore[union-attr]

    by_kind = (
        await mcp_server.call_tool(
            "warnings_list", {"kind": KIND_NOTE_NO_NODE}
        )
    ).structured_content
    assert by_kind["total"] == 1
    assert by_kind["warnings"][0]["subject"] == {"type": "note", "id": 2}

    by_subject = (
        await mcp_server.call_tool(
            "warnings_list", {"subject_type": "term", "status": "dismissed"}
        )
    ).structured_content
    assert by_subject["total"] == 1
    assert by_subject["warnings"][0]["kind"] == KIND_TERM_CONTEXTS_EXCESS

    by_id = (
        await mcp_server.call_tool("warnings_list", {"subject_id": 1})
    ).structured_content
    assert by_id["total"] == 1
    assert by_id["warnings"][0]["id"] != dismissed


@pytest.mark.asyncio
async def test_pagination(
    settings_and_services: tuple[Settings, Services], mcp_server
) -> None:
    """limit/offset дают has_more/next_offset, total не зависит от страницы."""
    _, services = settings_and_services
    for i in range(5):
        seed_warning(services, subject_id=i + 1)

    first = (
        await mcp_server.call_tool("warnings_list", {"limit": 2})
    ).structured_content
    assert len(first["warnings"]) == 2
    assert first["total"] == 5
    assert first["has_more"] is True
    assert first["next_offset"] == 2

    last = (
        await mcp_server.call_tool(
            "warnings_list", {"limit": 2, "offset": 4}
        )
    ).structured_content
    assert len(last["warnings"]) == 1
    assert last["total"] == 5  # total — по всему фильтру, не по странице
    assert last["has_more"] is False
    assert last["next_offset"] is None


@pytest.mark.asyncio
async def test_limit_over_max_is_soft(
    settings_and_services: tuple[Settings, Services], mcp_server
) -> None:
    """limit выше потолка MCP — мягкий ответ с hint, не schema-error."""
    settings, _ = settings_and_services
    over = settings.warnings_max_limit_mcp + 1
    call = await mcp_server.call_tool("warnings_list", {"limit": over})
    assert call.is_error is False
    got = call.structured_content
    assert got["warnings"] == [] and got["total"] == 0
    assert got["hint"] == (
        f"limit: expected 1..{settings.warnings_max_limit_mcp}, got {over}"
    )


@pytest.mark.asyncio
async def test_surface_grew_by_exactly_one(mcp_server) -> None:
    """Регресс поверхности: набор вырос ровно на один — `warnings_list`."""
    names = {tool.name for tool in await mcp_server.list_tools()}
    assert len(names) == 25
    assert names == TOOL_NAMES
    assert "warnings_list" in names
