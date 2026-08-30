"""Тесты MCP-поверхности (Фаза 1, Шаг 3) — против живого uvicorn-сервера.

ARCHITECTURE §7 «MCP-поверхность»: handshake (включая поле instructions),
tools/list → все 6, вызовы реальным MCP-клиентом, негативы токена, кастомный
MCP_PATH. Сервер поднимается в subprocess на отдельных портах.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import (
    create_mcp_http_client,
    streamable_http_client,
)

from app.services.dedup import DEDUP_HINT
from app.transport.mcp import (
    SERVER_INSTRUCTIONS,
    SERVER_NAME,
    TOOL_DESCRIPTIONS,
    TOOL_NAMES,
)
from tests.conftest import TEST_ENV

TOKEN = TEST_ENV["MCP_AUTH_TOKEN"]
SERVER_PORT = 18765
CUSTOM_PORT = 18766
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _start_server(port: int, extra_env: dict[str, str]) -> Iterator[str]:
    """Поднять uvicorn в subprocess и дождаться готовности /health."""
    env = {**os.environ, **TEST_ENV, "PORT": str(port), **extra_env}
    log = tempfile.NamedTemporaryFile(  # noqa: SIM115 — закрывается ниже
        mode="w+", prefix=f"second-brain-test-{port}-", suffix=".log", delete=False
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
         "--port", str(port)],
        cwd=REPO_ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 30.0
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                log.flush()
                raise RuntimeError(
                    f"тестовый сервер (порт {port}) упал при старте:\n"
                    f"{open(log.name).read()}"
                )
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        else:
            raise RuntimeError(f"тестовый сервер (порт {port}) не поднялся за 30с")
        yield f"http://127.0.0.1:{port}"
    finally:
        process.terminate()
        process.wait(timeout=10)


@pytest.fixture(scope="session")
def server_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Сервер с настройками по умолчанию (MCP_PATH=/mcp, своя БД)."""
    db = tmp_path_factory.mktemp("mcp-server") / "notes.db"
    yield from _start_server(SERVER_PORT, {"DB_PATH": str(db)})


@pytest.fixture(scope="session")
def custom_path_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Сервер с переопределённым MCP_PATH=/memory (и своей БД)."""
    db = tmp_path_factory.mktemp("mcp-custom") / "notes.db"
    yield from _start_server(CUSTOM_PORT, {"MCP_PATH": "/memory", "DB_PATH": str(db)})


@asynccontextmanager
async def connect(base_url: str, path: str = "/mcp", token: str | None = TOKEN) -> AsyncIterator[ClientSession]:
    """Подключиться к MCP-эндпоинту и выполнить initialize."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    http_client = create_mcp_http_client(headers=headers)
    async with streamable_http_client(f"{base_url}{path}", http_client=http_client) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            yield session


class TestHandshake:
    @pytest.mark.asyncio
    async def test_initialize_delivers_server_info_and_instructions(
        self, server_url: str
    ) -> None:
        """Handshake: имя сервера и полный текст instructions (ARCH §5.1)."""
        headers = {"Authorization": f"Bearer {TOKEN}"}
        async with streamable_http_client(
            f"{server_url}/mcp", http_client=create_mcp_http_client(headers=headers)
        ) as streams, ClientSession(*streams) as session:
            result = await session.initialize()
        assert result.server_info.name == SERVER_NAME
        assert result.instructions == SERVER_INSTRUCTIONS
        assert result.protocol_version  # версия согласована при handshake

    @pytest.mark.asyncio
    async def test_health_open_on_live_server(self, server_url: str) -> None:
        """Интеграция каркаса: /health живого сервера доступен без токена."""
        response = httpx.get(f"{server_url}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestToolsList:
    @pytest.mark.asyncio
    async def test_exactly_six_memory_tools(self, server_url: str) -> None:
        async with connect(server_url) as session:
            tools = await session.list_tools()
        assert {tool.name for tool in tools.tools} == TOOL_NAMES

    @pytest.mark.asyncio
    async def test_descriptions_are_canonical_teaching_texts(
        self, server_url: str
    ) -> None:
        """description каждого инструмента — дословно текст ARCHITECTURE §5.2."""
        async with connect(server_url) as session:
            tools = {t.name: t for t in (await session.list_tools()).tools}
        for name, description in TOOL_DESCRIPTIONS.items():
            assert tools[name].description == description

    @pytest.mark.asyncio
    async def test_search_schema(self, server_url: str) -> None:
        """Контракт FR-1: query обязателен (1..512), top_k 1..20, дефолт из env."""
        async with connect(server_url) as session:
            tools = {t.name: t for t in (await session.list_tools()).tools}
        schema = tools["memory_search"].input_schema
        assert schema["required"] == ["query"]
        props = schema["properties"]
        assert props["query"]["maxLength"] == 512  # MAX_QUERY_CHARS
        assert props["query"]["minLength"] == 1
        assert props["top_k"]["default"] == 5  # DEFAULT_TOP_K
        assert props["top_k"]["minimum"] == 1
        assert props["top_k"]["maximum"] == 20

    @pytest.mark.asyncio
    async def test_list_schema(self, server_url: str) -> None:
        """Контракт FR-2: limit 1..50 (дефолт 20), offset ≥ 0."""
        async with connect(server_url) as session:
            tools = {t.name: t for t in (await session.list_tools()).tools}
        props = tools["memory_list"].input_schema["properties"]
        assert props["limit"]["default"] == 20  # DEFAULT_LIST_LIMIT
        assert props["limit"]["maximum"] == 50
        assert props["offset"]["default"] == 0
        assert props["offset"]["minimum"] == 0

    @pytest.mark.asyncio
    async def test_get_schema_batch(self, server_url: str) -> None:
        """Контракт FR-3: ids — список 1..20 (MAX_GET_BATCH), id — алиас."""
        async with connect(server_url) as session:
            tools = {t.name: t for t in (await session.list_tools()).tools}
        schema = tools["memory_get"].input_schema
        assert set(schema["properties"]) == {"ids", "id"}
        assert schema.get("required") is None
        ids = schema["properties"]["ids"]["anyOf"][0]
        assert ids["maxItems"] == 20  # MAX_GET_BATCH
        assert ids["minItems"] == 1

    @pytest.mark.asyncio
    async def test_save_update_schema(self, server_url: str) -> None:
        """Контракты FR-4/FR-5: text 1..MAX_NOTE_CHARS (35000), id обязателен."""
        async with connect(server_url) as session:
            tools = {t.name: t for t in (await session.list_tools()).tools}
        save_text = tools["memory_save"].input_schema["properties"]["text"]
        assert save_text["maxLength"] == 35000
        assert tools["memory_save"].input_schema["required"] == ["text"]
        assert tools["memory_update"].input_schema["required"] == ["id", "text"]


class TestToolCalls:
    @pytest.mark.asyncio
    async def test_schema_rejects_empty_query(self, server_url: str) -> None:
        """Ограничения схемы реально отклоняют мусорные аргументы."""
        async with connect(server_url) as session:
            call = await session.call_tool("memory_search", {"query": ""})
        assert call.is_error is True
        assert "string_too_short" in call.content[0].text

    @pytest.mark.asyncio
    async def test_schema_rejects_too_long_note(self, server_url: str) -> None:
        async with connect(server_url) as session:
            call = await session.call_tool("memory_save", {"text": "x" * 36000})
        assert call.is_error is True
        assert "string_too_long" in call.content[0].text

    @pytest.mark.asyncio
    async def test_schema_rejects_too_many_ids(self, server_url: str) -> None:
        async with connect(server_url) as session:
            call = await session.call_tool("memory_get", {"ids": list(range(1, 30))})
        assert call.is_error is True
        assert "too_long" in call.content[0].text or "too_many" in call.content[0].text


class TestTokenNegatives:
    @pytest.mark.asyncio
    async def test_initialize_without_token_fails(self, server_url: str) -> None:
        """MCP-клиент без токена не проходит даже initialize (HTTP 401)."""
        with pytest.raises(BaseException) as exc_info:
            async with connect(server_url, token=None) as session:
                await session.list_tools()
        assert _has_mcp_error(exc_info.value)

    def test_raw_initialize_without_token_401(self, server_url: str) -> None:
        """Сырой JSON-RPC initialize без токена → 401 (не JSON-RPC-ответ)."""
        response = httpx.post(
            f"{server_url}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "raw-test", "version": "0"},
                },
            },
            headers={"Accept": "application/json, text/event-stream"},
        )
        assert response.status_code == 401
        assert response.json() == {"detail": "Unauthorized"}

    def test_raw_initialize_with_wrong_token_401(self, server_url: str) -> None:
        response = httpx.post(
            f"{server_url}/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={
                "Authorization": "Bearer wrong",
                "Accept": "application/json, text/event-stream",
            },
        )
        assert response.status_code == 401


class TestCustomMcpPath:
    @pytest.mark.asyncio
    async def test_handshake_on_custom_path(self, custom_path_url: str) -> None:
        """MCP_PATH из env реально меняет путь эндпоинта (§8)."""
        async with connect(custom_path_url, path="/memory") as session:
            tools = await session.list_tools()
        assert {tool.name for tool in tools.tools} == TOOL_NAMES

    def test_default_path_absent_on_custom_server(self, custom_path_url: str) -> None:
        """/mcp на сервере с MCP_PATH=/memory — с верным токеном 404."""
        response = httpx.post(
            f"{custom_path_url}/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Accept": "application/json, text/event-stream",
            },
        )
        assert response.status_code == 404


def _has_mcp_error(exc: BaseException) -> bool:
    """В дереве исключений есть ошибка транспорта MCP (HTTP 401)."""
    if type(exc).__name__ == "MCPError":
        return True
    subs = getattr(exc, "exceptions", None)
    if subs:
        return any(_has_mcp_error(sub) for sub in subs)
    return False

class TestMemoryFlow:
    """E2E по живому серверу: полный CRUD + поиск через MCP-клиента."""

    marker = f"genmarker-{uuid.uuid4().hex[:8]}"

    async def _saved_id(self, session: ClientSession, text: str) -> int:
        call = await session.call_tool("memory_save", {"text": text})
        assert call.is_error is False, call.content
        content = call.structured_content
        assert content["stored"] is True
        assert content["summary_pending"] is True  # режим суммаризации
        return content["id"]

    @pytest.mark.asyncio
    async def test_save_get_roundtrip(self, server_url: str) -> None:
        text = f"{self.marker}: деплой TaskFlow прошёл 2026-08-29"
        async with connect(server_url) as session:
            note_id = await self._saved_id(session, text)
            got = (await session.call_tool("memory_get", {"ids": [note_id]})).structured_content
        note = got["notes"][0]
        assert note["id"] == note_id
        assert note["text"] == text
        assert note["created_at"].endswith("Z") and note["updated_at"].endswith("Z")
        # Компактный контракт Фазы 9: get — белый список из четырёх полей.
        assert set(note) == {"id", "text", "created_at", "updated_at"}
        assert "summary" not in note
        assert "summary_status" not in note
        assert "author" not in note

    @pytest.mark.asyncio
    async def test_single_id_alias(self, server_url: str) -> None:
        text = f"{self.marker}: алиас одиночного id"
        async with connect(server_url) as session:
            note_id = await self._saved_id(session, text)
            got = (await session.call_tool("memory_get", {"id": note_id})).structured_content
        assert got["notes"][0]["id"] == note_id

    @pytest.mark.asyncio
    async def test_search_returns_no_full_text(self, server_url: str) -> None:
        """FR-1 (Фаза 9): компактные хиты без snippet/оценок; warning срезан."""
        text = f"{self.marker}: квантовый кулер в стойке 192.168.7.7"
        async with connect(server_url) as session:
            await self._saved_id(session, text)
            found = (await session.call_tool(
                "memory_search", {"query": "квантовый кулер"}
            )).structured_content
        # Тестовая среда без семантики: fallback-усечение summary начинается
        # с маркера — по нему и выбираем хит (ранее выбор шёл по snippet).
        hit = next(
            r for r in found["results"] if r["summary"].startswith(f"{self.marker}")
        )
        assert set(hit) == {"id", "summary", "created_at", "updated_at"}
        assert "snippet" not in hit
        assert "cosine" not in hit
        assert "rrf_score" not in hit
        assert "summary_status" not in hit
        assert "author" not in hit
        assert "text" not in hit
        assert "warning" not in found  # warning срезан даже при деградации

    @pytest.mark.asyncio
    async def test_list_shows_summaries_only(self, server_url: str) -> None:
        async with connect(server_url) as session:
            await self._saved_id(session, f"{self.marker}: для списка")
            listed = (await session.call_tool(
                "memory_list", {"limit": 5}
            )).structured_content
        assert listed["total"] >= 1
        assert listed["items"]  # среди первой страницы есть наша
        for item in listed["items"]:
            assert set(item) == {"id", "summary", "created_at", "updated_at"}
            assert "summary_status" not in item
            assert "author" not in item

    @pytest.mark.asyncio
    async def test_update_full_rewrite(self, server_url: str) -> None:
        async with connect(server_url) as session:
            note_id = await self._saved_id(
                session, f"{self.marker}: старый текст для апдейта"
            )
            updated = (await session.call_tool("memory_update", {
                "id": note_id, "text": f"{self.marker}: новый полный текст",
            })).structured_content
            assert updated == {"id": note_id, "updated": True}
            assert "summary_pending" not in updated
            got = (await session.call_tool("memory_get", {"ids": [note_id]})).structured_content
        assert got["notes"][0]["text"] == f"{self.marker}: новый полный текст"

    @pytest.mark.asyncio
    async def test_delete_is_soft(self, server_url: str) -> None:
        async with connect(server_url) as session:
            note_id = await self._saved_id(session, f"{self.marker}: на удаление")
            deleted = (await session.call_tool("memory_delete", {"id": note_id})).structured_content
            assert deleted == {"id": note_id, "deleted": True}
            got = (await session.call_tool("memory_get", {"ids": [note_id]})).structured_content
            listed = (await session.call_tool("memory_list", {})).structured_content
        # Мягкий ответ: ни в get, ни в list удалённая не видна (rowcount-wise trash).
        assert got["notes"] == []
        assert all(item["id"] != note_id for item in listed["items"])

    @pytest.mark.asyncio
    async def test_soft_answers_for_missing_ids(self, server_url: str) -> None:
        """FR-3/FR-5/FR-6: неизвестные id — мягкие ответы, не ошибки."""
        async with connect(server_url) as session:
            got = (await session.call_tool("memory_get", {"ids": [10**9]})).structured_content
            assert got["notes"] == [] and got["hint"]
            upd = (await session.call_tool(
                "memory_update", {"id": 10**9, "text": "_body_"}
            )).structured_content
            assert upd["updated"] is False and upd["hint"]
            dele = (await session.call_tool(
                "memory_delete", {"id": 10**9}
            )).structured_content
            assert dele["deleted"] is False and dele["hint"]

    @pytest.mark.asyncio
    async def test_empty_search_gives_hint(self, server_url: str) -> None:
        async with connect(server_url) as session:
            found = (await session.call_tool(
                "memory_search", {"query": "неттакогословафффф"}
            )).structured_content
        assert found["results"] == []
        assert "переформулируй" in found["hint"]
        assert "warning" not in found

    @pytest.mark.asyncio
    async def test_save_duplicate_compact_answer(self, server_url: str) -> None:
        """Дубль через MCP: id существующей, stored=false, hint; text/duplicated срезаны."""
        text = f"{self.marker}: дубль через повторный save"
        async with connect(server_url) as session:
            first = (await session.call_tool(
                "memory_save", {"text": text}
            )).structured_content
            assert first["stored"] is True
            second = (await session.call_tool(
                "memory_save", {"text": text}
            )).structured_content
        assert second["id"] == first["id"]
        assert second["stored"] is False
        assert second["hint"] == DEDUP_HINT  # дословно — константа сервиса
        assert "text" not in second
        assert "duplicated" not in second

    @pytest.mark.asyncio
    async def test_list_beyond_total_gives_page_hint(self, server_url: str) -> None:
        """offset >= total: пустая страница + каноничный hint сервиса."""
        async with connect(server_url) as session:
            listed = (await session.call_tool(
                "memory_list", {"limit": 1, "offset": 10**9}
            )).structured_content
        assert set(listed) == {"items", "total", "hint"}
        assert listed["items"] == []
        assert listed["hint"] == (
            "страница за пределом памяти: offset ≥ total; уменьши offset"
        )

    @pytest.mark.asyncio
    async def test_get_partial_batch_keeps_request_order(self, server_url: str) -> None:
        """Частичный batch: недостающий id молча выпадает, порядок — как в запросе."""
        first_text = f"{self.marker}: batch первая заметка"
        second_text = f"{self.marker}: batch вторая заметка"
        async with connect(server_url) as session:
            id1 = await self._saved_id(session, first_text)
            id2 = await self._saved_id(session, second_text)
            got = (await session.call_tool(
                "memory_get", {"ids": [id1, 10**9, id2]}
            )).structured_content
        notes = got["notes"]
        assert [note["id"] for note in notes] == [id1, id2]  # порядок запроса
        assert [note["text"] for note in notes] == [first_text, second_text]
        assert "hint" not in got  # успешный ответ — без hint (Фаза 9)
