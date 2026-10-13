"""№12 (no12-02, 3.2.0): чтение предупреждений НЕ зовёт модель (§13.4).

Сторож: `warnings_list`, `GET /warnings` и `/health.warnings` — только `SELECT`
по таблице `warnings`; ни классификатора, ни суммаризатора, ни эмбеддера, ни
судьи. Ловим на счётчиках клиентов слотов (`LLMClient.chat`/`embed`).
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.services.llm_client import LLMClient
from app.services.warnings import KIND_NOTE_MISPLACED
from app.transport.mcp import build_mcp

AUTH = "Bearer test-secret-token"


def test_read_never_calls_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ни один путь чтения не увеличивает счётчики вызовов клиентов слотов."""
    calls = {"chat": 0, "embed": 0}

    def chat(self: LLMClient, *args: object, **kwargs: object) -> str:
        calls["chat"] += 1
        raise AssertionError("the read path called the LLM slot (chat)")

    def embed(self: LLMClient, *args: object, **kwargs: object) -> object:
        calls["embed"] += 1
        raise AssertionError("the read path called the embedding slot (embed)")

    monkeypatch.setattr(LLMClient, "chat", chat)
    monkeypatch.setattr(LLMClient, "embed", embed)

    settings = get_settings()
    services = client.app.state.services  # type: ignore[attr-defined]
    services.warnings.open(KIND_NOTE_MISPLACED, "note", 1, "m", {})
    mcp = build_mcp(settings, services)

    # Три пути чтения одной поверхности: MCP, REST-листинг, агрегаты /health.
    asyncio.run(mcp.call_tool("warnings_list", {}))
    client.get("/warnings", headers={"Authorization": AUTH})
    client.get("/health")
    assert calls == {"chat": 0, "embed": 0}
