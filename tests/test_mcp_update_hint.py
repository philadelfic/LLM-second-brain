"""Хинт о свежем релизе в ответах MCP-инструментов (№45, 3.2.0; arch §7.1).

Юнит без MCP-клиента: middleware `UpdateHintMiddleware` дописывает в ответ
инструмента поле верхнего уровня `update_hint` (обезличенный текст, EN), если
`UpdateChecker.hint()` вернул текст. В критическом пути ручки — только чтение
кеша (никаких файловых/сетевых обращений). По умолчанию проверка выключена:
выдача ручки не меняется ни на байт.
"""

from __future__ import annotations

import copy

import pytest
from mcp.types import CallToolResult, TextContent

from app.config import Settings
from app.services.update_check import UpdateChecker
from app.transport.mcp import UpdateHintMiddleware

REQUIRED_ENV: dict[str, str] = {
    "EMBEDDING_BASE_URL": "http://localhost:11434",
    "SUMMARY_BASE_URL": "http://localhost:11434",
    "SUMMARY_MODEL": "lfm2.5:latest",
    "JUDGE_BASE_URL": "http://localhost:11434",
    "JUDGE_MODEL": "lfm2.5:latest",
    "MCP_AUTH_TOKEN": "test-secret-token",
}


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    env = {name.lower(): value for name, value in REQUIRED_ENV.items()}
    env.update({name.lower(): value for name, value in overrides.items()})
    for name, value in env.items():
        monkeypatch.setenv(name.upper(), value)
    return Settings()


class _Ctx:
    """Минимальный контекст MCP-запроса: middleware читает только метод."""

    method = "tools/call"


def _result(payload: dict | None = None) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text="{...}")],
        structured_content=payload if payload is not None else {"items": []},
        is_error=False,
    )


async def _call(mw: UpdateHintMiddleware, result: CallToolResult) -> CallToolResult:
    async def call_next(ctx) -> CallToolResult:
        return result

    return await mw(_Ctx(), call_next)


def _enabled_checker(monkeypatch: pytest.MonkeyPatch, *, version: str = "9.9.9") -> UpdateChecker:
    """Чекер с заполненным кешем: источник подменён фетчером (без файла/сети)."""
    settings = _settings(
        monkeypatch,
        UPDATE_CHECK_ENABLED="true",
        UPDATE_SOURCE_FILE="/tmp/irrelevant.json",
    )
    checker = UpdateChecker(
        settings, fetcher=lambda: {"version": version, "notes": "shiny new release"}
    )
    checker.refresh()
    return checker


@pytest.mark.asyncio
async def test_tool_response_carries_update_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """При готовом хинте ответ инструмента несёт поле update_hint."""
    checker = _enabled_checker(monkeypatch)
    out = await _call(UpdateHintMiddleware(checker), _result())
    hint = out.structured_content["update_hint"]
    assert "v9.9.9" in hint
    assert "shiny new release" in hint
    assert "operator" not in hint.lower()  # обезличенность


@pytest.mark.asyncio
async def test_update_hint_does_not_replace_hint_or_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """update_hint — отдельное поле; hint/fail не подменяются."""
    checker = _enabled_checker(monkeypatch)
    payload = {"hint": "set a title", "fail": "soft", "items": []}
    out = await _call(UpdateHintMiddleware(checker), _result(dict(payload)))
    assert out.structured_content["hint"] == "set a title"
    assert out.structured_content["fail"] == "soft"
    assert "update_hint" in out.structured_content


@pytest.mark.asyncio
async def test_call_path_reads_cache_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Путь ручки не читает источник: фетчер зовётся только фоном (один раз)."""
    calls: list[int] = []

    def fetcher() -> dict | None:
        calls.append(1)
        if len(calls) > 1:
            raise AssertionError("источник прочитан в пути вызова ручки")
        return {"version": "9.9.9", "notes": "n"}

    settings = _settings(
        monkeypatch, UPDATE_CHECK_ENABLED="true", UPDATE_SOURCE_FILE="/tmp/x.json"
    )
    checker = UpdateChecker(settings, fetcher=fetcher)
    checker.refresh()  # фоновая петля в проде; здесь — подготовка кеша
    out = await _call(UpdateHintMiddleware(checker), _result())
    assert out.structured_content["update_hint"]
    assert len(calls) == 1  # вызов ручки источник не читал


@pytest.mark.asyncio
async def test_default_disabled_output_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Дефолт (проверка выключена) — выдача ручки не меняется ни на байт."""
    settings = _settings(monkeypatch)  # UPDATE_CHECK_ENABLED по умолчанию false
    checker = UpdateChecker(
        settings,
        fetcher=lambda: (_ for _ in ()).throw(AssertionError("источник не читаем")),
    )
    result = _result({"hint": "keep", "items": [1, 2, 3]})
    before = copy.deepcopy(result.structured_content)
    content_before = list(result.content)
    out = await _call(UpdateHintMiddleware(checker), result)
    assert out.structured_content == before
    assert out.content == content_before
    assert "update_hint" not in out.structured_content
