"""Хинт «слот не работает — предупреди своего человека» (№39 FR-4.4/FR-4.5).

Юнит без MCP-клиента: middleware дописывает в ответ инструмента короткий
обезличенный хинт при паузе слота, с троттлингом (не чаще
`SLOT_HINT_EVERY_SEC` на слот). Имён людей и слова «оператор» в тексте нет.
"""

from __future__ import annotations

import pytest
from mcp.types import CallToolResult, TextContent

from app.config import get_settings
from app.transport.mcp import SlotHintMiddleware


class _Ctx:
    """Минимальный контекст MCP-запроса: middleware читает только метод."""

    method = "tools/call"


class _Worker:
    """Снимок слотов: то, что middleware берёт у воркера."""

    def __init__(self, states: dict) -> None:
        self._states = states

    def slot_states(self) -> dict:
        return self._states


def _result() -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text="{}")],
        structured_content={"items": []},
        is_error=False,
    )


async def _call(mw: SlotHintMiddleware, result: CallToolResult) -> CallToolResult:
    async def call_next(ctx) -> CallToolResult:
        return result

    return await mw(_Ctx(), call_next)


@pytest.mark.asyncio
async def test_hint_on_paused_slot() -> None:
    """При разрыве ответ инструмента несёт hint с именем слота и исходом."""
    settings = get_settings()
    worker = _Worker(
        {
            "summary": {
                "slot": "summary",
                "state": "paused",
                "last_outcome": "auth",
                "paused_until": "2026-10-08T12:15:00Z",
                "failures": 1,
            }
        }
    )
    out = await _call(SlotHintMiddleware(settings, worker), _result())
    hint = out.structured_content["hint"]
    assert "summary" in hint
    assert "auth" in hint
    assert "tell your human" in hint
    assert "operator" not in hint.lower()  # обезличенность (FR-4.7)


@pytest.mark.asyncio
async def test_hint_throttled() -> None:
    """Второй вызов подряд хинта не несёт до SLOT_HINT_EVERY_SEC."""
    settings = get_settings()
    worker = _Worker(
        {
            "judge": {
                "slot": "judge",
                "state": "paused",
                "last_outcome": "auth",
                "paused_until": "2026-10-08T12:15:00Z",
                "failures": 1,
            }
        }
    )
    mw = SlotHintMiddleware(settings, worker)
    first = await _call(mw, _result())
    assert "hint" in first.structured_content
    second = await _call(mw, _result())
    assert "hint" not in (second.structured_content or {})


@pytest.mark.asyncio
async def test_no_hint_when_slots_closed() -> None:
    """Все слоты закрыты — ответ не трогаем."""
    settings = get_settings()
    worker = _Worker(
        {
            "embedding": {
                "slot": "embedding",
                "state": "closed",
                "last_outcome": None,
                "paused_until": None,
                "failures": 0,
            }
        }
    )
    out = await _call(SlotHintMiddleware(settings, worker), _result())
    assert "hint" not in (out.structured_content or {})
    assert len(out.content) == 1
