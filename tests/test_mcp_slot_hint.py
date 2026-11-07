"""The "the slot is down — warn your human" hint (#39 FR-4.4/FR-4.5).

    A unit without an MCP client: the middleware appends a short impersonal hint
    to a tool's answer at a slot pause, throttled (no more than
    `SLOT_HINT_EVERY_SEC` per slot). No people's names and no word "operator" in
    the text.
"""

from __future__ import annotations

import pytest
from mcp.types import CallToolResult, TextContent

from app.config import get_settings
from app.transport.mcp import SlotHintMiddleware


class _Ctx:
    """A minimal MCP request context: the middleware reads only the method."""

    method = "tools/call"


class _Worker:
    """The slots snapshot: what the middleware takes from the worker."""

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
    """On a breaker the tool's answer carries a hint with the slot name and the outcome."""
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
    assert "operator" not in hint.lower()  # impersonality (FR-4.7)


@pytest.mark.asyncio
async def test_hint_throttled() -> None:
    """A second call in a row carries no hint until SLOT_HINT_EVERY_SEC."""
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
    """All slots closed — the answer untouched."""
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
