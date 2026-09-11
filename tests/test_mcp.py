"""Тесты MCP-поверхности (Фаза 1, Шаг 3) — против живого uvicorn-сервера.

ARCHITECTURE §7 «MCP-поверхность»: handshake (включая поле instructions),
tools/list → все 13 (8 ручек заметок/узлов + 5 области навыков, lsb-0007-03),
вызовы реальным MCP-клиентом, негативы токена, кастомный MCP_PATH. Сервер
поднимается в subprocess на отдельных портах.
"""

from __future__ import annotations

import logging
import os
import sqlite3
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
NS_PORT = 18767
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


@pytest.fixture(scope="session")
def ns_db(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Отдельная БД для namespace-тестов (изоляция от общей памяти)."""
    return str(tmp_path_factory.mktemp("mcp-ns") / "notes.db")


@pytest.fixture(scope="session")
def ns_url(ns_db: str) -> Iterator[str]:
    """Сервер с дефолтным MCP_PATH=/mcp и своей (namespace-изолированной) БД."""
    yield from _start_server(NS_PORT, {"DB_PATH": ns_db})


def _register_namespace(db_path: str, path: str, description: str) -> None:
    """Зарегистрировать узел прямо в БД namespace-сервера (операторская ручка
    REST появится в Шаге 6; для MCP-теста реестр наполняем напрямую)."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO namespaces (path, description) VALUES (?, ?)",
            (path, description),
        )


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
        # Фаза 10: инструкции = база + правило неймспейсов + карта реестра
        # (§5.7, слой 1 ориентирования). База — префикс; карта зависит от
        # реестра на момент сборки.
        assert result.instructions.startswith(SERVER_INSTRUCTIONS)
        assert "Node map (path: description)" in result.instructions
        assert result.protocol_version  # версия согласована при handshake

    @pytest.mark.asyncio
    async def test_health_open_on_live_server(self, server_url: str) -> None:
        """Интеграция каркаса: /health живого сервера доступен без токена."""
        response = httpx.get(f"{server_url}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestToolsList:
    @pytest.mark.asyncio
    async def test_exactly_thirteen_tools(self, server_url: str) -> None:
        """lsb-0007-03: поверхность = 8 ручек заметок/узлов + 5 области навыков."""
        async with connect(server_url) as session:
            tools = await session.list_tools()
        assert len(tools.tools) == 13
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
        """Контракт FR-3 (lsb-0003): ids — список 1..20 (MAX_GET_BATCH),
        id — алиас; query/chunk/limit — чтение чанка, все опциональны."""
        async with connect(server_url) as session:
            tools = {t.name: t for t in (await session.list_tools()).tools}
        schema = tools["memory_get"].input_schema
        assert set(schema["properties"]) == {"ids", "id", "query", "chunk", "limit"}
        assert schema.get("required") is None
        ids = schema["properties"]["ids"]["anyOf"][0]
        assert ids["maxItems"] == 20  # MAX_GET_BATCH
        assert ids["minItems"] == 1

    @pytest.mark.asyncio
    async def test_save_update_schema(self, server_url: str) -> None:
        """Контракты FR-4/FR-5: text 1..MAX_NOTE_CHARS (35000), id обязателен;
        title (Фаза 11, решение №9) — опционален в схеме: отказ за сервисом
        (fail+hint «set a title ≤5 words»), не схемой. lsb-0004-01: text в
        memory_update стал опционален (можно править title/summary/namespace
        без перезаписи текста) — обязателен только id.
        """
        async with connect(server_url) as session:
            tools = {t.name: t for t in (await session.list_tools()).tools}
        save_text = tools["memory_save"].input_schema["properties"]["text"]
        assert save_text["maxLength"] == 35000
        assert tools["memory_save"].input_schema["required"] == ["text"]
        assert tools["memory_update"].input_schema["required"] == ["id"]
        assert tools["memory_save"].input_schema["properties"]["title"]["default"] is None
        assert tools["memory_update"].input_schema["properties"]["title"]["default"] is None

    @pytest.mark.asyncio
    async def test_namespace_params_in_schemas(self, server_url: str) -> None:
        """§5.7: namespace в save/update (save default), search/list +
        namespace_exact; delete без namespace (по id)."""
        async with connect(server_url) as session:
            tools = {t.name: t for t in (await session.list_tools()).tools}
        assert tools["memory_save"].input_schema["properties"]["namespace"]["default"] == "default"
        assert tools["memory_update"].input_schema["properties"]["namespace"]["default"] is None
        srch = tools["memory_search"].input_schema["properties"]
        assert srch["namespace"]["default"] is None
        assert srch["namespace_exact"]["default"] is False
        lst = tools["memory_list"].input_schema["properties"]
        assert lst["namespace"]["default"] is None
        assert lst["namespace_exact"]["default"] is False
        assert "namespace" not in tools["memory_delete"].input_schema["properties"]
        assert "namespace" not in tools["memory_get"].input_schema["properties"]

    @pytest.mark.asyncio
    async def test_search_schema_mode(self, server_url: str) -> None:
        """FR-3.1: mode в схеме memory_search, дефолт 'semantic', опционален."""
        async with connect(server_url) as session:
            tools = {t.name: t for t in (await session.list_tools()).tools}
        schema = tools["memory_search"].input_schema
        assert schema["required"] == ["query"]  # mode не обязателен
        assert schema["properties"]["mode"]["default"] == "semantic"

    @pytest.mark.asyncio
    async def test_list_schema_detail(self, server_url: str) -> None:
        """FR-4.1: detail в схеме memory_list, дефолт 'summaries'."""
        async with connect(server_url) as session:
            tools = {t.name: t for t in (await session.list_tools()).tools}
        props = tools["memory_list"].input_schema["properties"]
        assert props["detail"]["default"] == "summaries"


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

async def _saved_id(
    session: ClientSession, text: str, title: str = "Тестовая заметка"
) -> int:
    """helper E2E: сохранить заметку с валидным title (решение №9) и вернуть id."""
    call = await session.call_tool("memory_save", {"text": text, "title": title})
    assert call.is_error is False, call.content
    content = call.structured_content
    assert content["stored"] is True
    assert content["summary_pending"] is True  # режим суммаризации
    return content["id"]


class TestMemoryFlow:
    """E2E по живому серверу: полный CRUD + поиск через MCP-клиента."""

    marker = f"genmarker-{uuid.uuid4().hex[:8]}"

    @pytest.mark.asyncio
    async def test_save_get_roundtrip(self, server_url: str) -> None:
        text = f"{self.marker}: деплой TaskFlow прошёл 2026-08-29"
        async with connect(server_url) as session:
            note_id = await _saved_id(session, text)
            got = (await session.call_tool("memory_get", {"ids": [note_id]})).structured_content
        note = got["notes"][0]
        assert note["id"] == note_id
        assert note["text"] == text
        assert note["created_at"].endswith("Z") and note["updated_at"].endswith("Z")
        # Компактный контракт Фазы 9: get — белый список из пяти полей (+namespace Фаза 10).
        assert set(note) == {"id", "text", "created_at", "updated_at", "namespace", "expires_at"}
        assert note["namespace"] == "default"  # save без узла → default (§5.7)
        assert "title" not in note  # Фаза 11: get без названия (там полный текст)

    @pytest.mark.asyncio
    async def test_single_id_alias(self, server_url: str) -> None:
        text = f"{self.marker}: алиас одиночного id"
        async with connect(server_url) as session:
            note_id = await _saved_id(session, text)
            got = (await session.call_tool("memory_get", {"id": note_id})).structured_content
        assert got["notes"][0]["id"] == note_id

    @pytest.mark.asyncio
    async def test_chunk_read_by_index(self, server_url: str) -> None:
        """lsb-0003: memory_get с chunk=N возвращает чанки заметки (списком)."""
        text = f"{self.marker}: " + ("длинный текст заметки для нескольких чанков " * 60)
        async with connect(server_url) as session:
            note_id = await _saved_id(session, text, title="Чанк чтение")
            got = (
                await session.call_tool("memory_get", {"id": note_id, "chunk": 0})
            ).structured_content
        assert got["chunks"]
        assert got["chunks"][0]["chunk_index"] == 0
        assert got["total_chunks"] >= 1
        assert "chars" in got

    @pytest.mark.asyncio
    async def test_get_without_chunk_stays_full_note(self, server_url: str) -> None:
        """Обратная совместимость (FR-1): без query/chunk — заметка целиком."""
        text = f"{self.marker}: по-прежнему целиком"
        async with connect(server_url) as session:
            note_id = await _saved_id(session, text)
            got = (
                await session.call_tool("memory_get", {"ids": [note_id]})
            ).structured_content
        assert got["notes"][0]["id"] == note_id

    @pytest.mark.asyncio
    async def test_chunk_read_multi_id_refused(self, server_url: str) -> None:
        """Чтение чанком по нескольким id — мягкий отказ (ids+чанки)."""
        async with connect(server_url) as session:
            got = (
                await session.call_tool("memory_get", {"ids": [1, 2], "chunk": 0})
            ).structured_content
        assert got["chunks"] == []
        assert "single id" in got["hint"]

    @pytest.mark.asyncio
    async def test_limit_without_query_refused(self, server_url: str) -> None:
        """limit без query/chunk — мягкий отказ."""
        async with connect(server_url) as session:
            got = (
                await session.call_tool("memory_get", {"id": 1, "limit": 3})
            ).structured_content
        assert got["chunks"] == []
        assert "limit" in got["hint"]

    @pytest.mark.asyncio
    async def test_query_and_chunk_refused(self, server_url: str) -> None:
        """query + chunk вместе — мягкий отказ (hint)."""
        async with connect(server_url) as session:
            got = (
                await session.call_tool(
                    "memory_get", {"id": 1, "query": "x", "chunk": 0}
                )
            ).structured_content
        assert got["chunks"] == []
        assert "not both" in got["hint"]

    @pytest.mark.asyncio
    async def test_search_returns_no_full_text(self, server_url: str) -> None:
        """FR-1 (Фаза 9): компактные хиты без snippet/оценок; warning срезан."""
        text = f"{self.marker}: квантовый кулер в стойке 192.168.7.7"
        async with connect(server_url) as session:
            await _saved_id(session, text)
            found = (await session.call_tool(
                "memory_search", {"query": "квантовый кулер"}
            )).structured_content
        # Тестовая среда без семантики: fallback-усечение summary начинается
        # с маркера — по нему и выбираем хит (ранее выбор шёл по snippet).
        hit = next(
            r for r in found["results"] if r["summary"].startswith(f"{self.marker}")
        )
        assert set(hit) == {
            "id", "summary", "created_at", "updated_at", "namespace", "title",
        }  # Фаза 11 (решение №9): +title (ключ резервируется — search.py вне пула 5)
        assert hit["namespace"] == "default"
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
            await _saved_id(session, f"{self.marker}: для списка")
            listed = (await session.call_tool(
                "memory_list", {"limit": 5}
            )).structured_content
        assert listed["total"] >= 1
        assert listed["items"]  # среди первой страницы есть наша
        for item in listed["items"]:
            assert set(item) == {
                "id", "summary", "created_at", "updated_at", "namespace", "title",
                "expires_at",  # lsb-0004-02
            }  # Фаза 11 (решение №9): +title
            assert "summary_status" not in item
            assert "author" not in item

    @pytest.mark.asyncio
    async def test_search_title_mode_finds_by_title(self, server_url: str) -> None:
        """FR-3.1/FR-2.1: mode='title' находит по подстроке названия (компактно)."""
        title = f"Квантовый кулер {self.marker}"
        async with connect(server_url) as session:
            note_id = await _saved_id(session, f"{self.marker}: текст для title-поиска", title=title)
            found = (await session.call_tool(
                "memory_search", {"query": "квантовый", "mode": "title"}
            )).structured_content
        hit = next(r for r in found["results"] if r["id"] == note_id)
        # title-контракт FR-2.2: score вместо created_at/updated_at.
        assert set(hit) == {"id", "title", "namespace", "summary", "score"}
        assert hit["title"] == title
        assert hit["namespace"] == "default"
        assert "created_at" not in hit

    @pytest.mark.asyncio
    async def test_search_semantic_mode_unchanged(self, server_url: str) -> None:
        """FR-3.2: mode='semantic' — как раньше (компактный контракт semantic)."""
        text = f"{self.marker}: семантический режим явно"
        async with connect(server_url) as session:
            await _saved_id(session, text)
            found = (await session.call_tool(
                "memory_search", {"query": "семантический режим", "mode": "semantic"}
            )).structured_content
        hit = next(
            r for r in found["results"] if r["summary"].startswith(f"{self.marker}")
        )
        assert set(hit) == {
            "id", "summary", "created_at", "updated_at", "namespace", "title",
        }
        assert "score" not in hit

    @pytest.mark.asyncio
    async def test_search_invalid_mode_soft_fail(self, server_url: str) -> None:
        """FR-3.4: неизвестный mode — мягкий отказ с хинтом доступных режимов."""
        async with connect(server_url) as session:
            found = (await session.call_tool(
                "memory_search", {"query": "что-то", "mode": "hybrid"}
            )).structured_content
        assert found["results"] == []
        assert "semantic" in found["hint"] and "title" in found["hint"]

    @pytest.mark.asyncio
    async def test_list_titles_detail_compact(self, server_url: str) -> None:
        """FR-4.1: detail='titles' — компактно (id, title, namespace), без summary."""
        async with connect(server_url) as session:
            await _saved_id(session, f"{self.marker}: для titles-листинга")
            listed = (await session.call_tool(
                "memory_list", {"limit": 5, "detail": "titles"}
            )).structured_content
        assert listed["total"] >= 1
        assert listed["items"]
        for item in listed["items"]:
            assert set(item) == {"id", "title", "namespace"}
            assert "summary" not in item
            assert "created_at" not in item

    @pytest.mark.asyncio
    async def test_list_summaries_detail_unchanged(self, server_url: str) -> None:
        """FR-4.2: detail='summaries' — как сейчас (полный компактный контракт)."""
        async with connect(server_url) as session:
            await _saved_id(session, f"{self.marker}: для summaries-листинга")
            listed = (await session.call_tool(
                "memory_list", {"limit": 5, "detail": "summaries"}
            )).structured_content
        assert listed["total"] >= 1
        for item in listed["items"]:
            assert set(item) == {
                "id", "summary", "created_at", "updated_at", "namespace", "title",
                "expires_at",  # lsb-0004-02
            }

    @pytest.mark.asyncio
    async def test_list_invalid_detail_soft_fail(self, server_url: str) -> None:
        """FR-4.1: неизвестный detail — мягкий отказ с хинтом доступных форм."""
        async with connect(server_url) as session:
            listed = (await session.call_tool(
                "memory_list", {"limit": 5, "detail": "full"}
            )).structured_content
        assert listed["items"] == [] and listed["total"] == 0
        assert "summaries" in listed["hint"] and "titles" in listed["hint"]

    @pytest.mark.asyncio
    async def test_update_full_rewrite(self, server_url: str) -> None:
        async with connect(server_url) as session:
            note_id = await _saved_id(
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
            note_id = await _saved_id(session, f"{self.marker}: на удаление")
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
        assert "rephrase" in found["hint"]
        assert "warning" not in found

    @pytest.mark.asyncio
    async def test_save_duplicate_compact_answer(self, server_url: str) -> None:
        """Дубль через MCP: id существующей, stored=false, hint; text/duplicated срезаны."""
        text = f"{self.marker}: дубль через повторный save"
        async with connect(server_url) as session:
            first = (await session.call_tool(
                "memory_save", {"text": text, "title": "Дубль тест"}
            )).structured_content
            assert first["stored"] is True
            second = (await session.call_tool(
                "memory_save", {"text": text, "title": "Дубль тест"}
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
            "page beyond the memory: offset ≥ total; reduce offset"
        )

    @pytest.mark.asyncio
    async def test_get_partial_batch_keeps_request_order(self, server_url: str) -> None:
        """Частичный batch: недостающий id молча выпадает, порядок — как в запросе."""
        first_text = f"{self.marker}: batch первая заметка"
        second_text = f"{self.marker}: batch вторая заметка"
        async with connect(server_url) as session:
            id1 = await _saved_id(session, first_text)
            id2 = await _saved_id(session, second_text)
            got = (await session.call_tool(
                "memory_get", {"ids": [id1, 10**9, id2]}
            )).structured_content
        notes = got["notes"]
        assert [note["id"] for note in notes] == [id1, id2]  # порядок запроса
        assert [note["text"] for note in notes] == [first_text, second_text]
        assert "hint" not in got  # успешный ответ — без hint (Фаза 9)


class TestTitleMCP:
    """Фаза 11 (решение №9): title в save/update и выдачах — по живому серверу.

    Отказ save без title / с длиннее 5 слов — fail+hint «set a title ≤5 words»,
    заметка не создаётся; title в search/list; в get — НЕТ; update —
    перезапись валидного, сохранение при отсутствии.
    """

    marker = f"titlemarker-{uuid.uuid4().hex[:8]}"

    @pytest.mark.asyncio
    async def test_save_without_title_fails_with_hint(self, server_url: str) -> None:
        """save без title → fail+hint, заметка НЕ создаётся."""
        async with connect(server_url) as session:
            before = (await session.call_tool(
                "memory_list", {"limit": 1}
            )).structured_content
            saved = (await session.call_tool(
                "memory_save", {"text": f"{self.marker}: заметка без названия"}
            )).structured_content
            after = (await session.call_tool(
                "memory_list", {"limit": 1}
            )).structured_content
        assert saved == {"stored": False, "hint": "set a title ≤5 words"}
        assert after["total"] == before["total"]  # заметка не создана

    @pytest.mark.asyncio
    async def test_save_too_long_title_fails_with_hint(self, server_url: str) -> None:
        """6 слов (> TITLE_MAX_WORDS) → fail+hint, заметка НЕ создаётся."""
        async with connect(server_url) as session:
            saved = (await session.call_tool(
                "memory_save", {"text": f"{self.marker}: длинное название",
                                "title": "раз два три четыре пять шесть"}
            )).structured_content
        assert saved == {"stored": False, "hint": "set a title ≤5 words"}

    @pytest.mark.asyncio
    async def test_save_five_word_title_stored_and_visible(self, server_url: str) -> None:
        """5 слов — граница валидности: сохранено, title виден в list."""
        text = f"{self.marker}: пять слов в названии"
        async with connect(server_url) as session:
            saved = (await session.call_tool(
                "memory_save", {"text": text, "title": "раз два три четыре пять"}
            )).structured_content
            assert saved["stored"] is True
            listed = (await session.call_tool(
                "memory_list", {"limit": 20}
            )).structured_content
        item = next(i for i in listed["items"] if i["id"] == saved["id"])
        assert item["title"] == "раз два три четыре пять"

    @pytest.mark.asyncio
    async def test_update_title_overwrite_and_keep(self, server_url: str) -> None:
        """update: title перезаписывает; без title — прежний остаётся."""
        async with connect(server_url) as session:
            note_id = await _saved_id(
                session, f"{self.marker}: апдейт названия", title="Первое название"
            )
            upd = (await session.call_tool(
                "memory_update", {"id": note_id,
                                  "text": f"{self.marker}: апдейт с названием",
                                  "title": "Второе название"}
            )).structured_content
            assert upd == {"id": note_id, "updated": True}
            listed = (await session.call_tool(
                "memory_list", {"limit": 20}
            )).structured_content
            upd2 = (await session.call_tool(
                "memory_update", {"id": note_id,
                                  "text": f"{self.marker}: апдейт без названия"}
            )).structured_content
            assert upd2 == {"id": note_id, "updated": True}
            listed2 = (await session.call_tool(
                "memory_list", {"limit": 20}
            )).structured_content
        item = next(i for i in listed["items"] if i["id"] == note_id)
        assert item["title"] == "Второе название"  # перезапись
        item2 = next(i for i in listed2["items"] if i["id"] == note_id)
        assert item2["title"] == "Второе название"  # не передан — прежний

    @pytest.mark.asyncio
    async def test_update_too_long_title_fails(self, server_url: str) -> None:
        """update с невалидным title → мягкий отказ + hint; заметка не тронута."""
        async with connect(server_url) as session:
            note_id = await _saved_id(
                session, f"{self.marker}: апдейт плохого названия"
            )
            upd = (await session.call_tool(
                "memory_update", {"id": note_id,
                                  "text": f"{self.marker}: не должно записаться",
                                  "title": "раз два три четыре пять шесть"}
            )).structured_content
            listed = (await session.call_tool(
                "memory_list", {"limit": 20}
            )).structured_content
        assert upd["updated"] is False
        assert upd["hint"] == "set a title ≤5 words"
        item = next(i for i in listed["items"] if i["id"] == note_id)
        assert item["summary"].startswith(f"{self.marker}: апдейт плохого названия")

    @pytest.mark.asyncio
    async def test_get_note_has_no_title(self, server_url: str) -> None:
        """memory_get — БЕЗ title (экономия контекста: там полный текст)."""
        async with connect(server_url) as session:
            note_id = await _saved_id(
                session, f"{self.marker}: get без названия"
            )
            got = (await session.call_tool(
                "memory_get", {"ids": [note_id]}
            )).structured_content
        assert "title" not in got["notes"][0]


class TestNamespaceMCP:
    """Фаза 10 (Шаг 3): namespace-параметры и memory_namespaces по живому серверу.
    Изолированная БД (ns_url/ns_db) — общая память других тестов не трогается."""

    marker = f"nsmarker-{uuid.uuid4().hex[:8]}"

    @pytest.mark.asyncio
    async def test_save_into_registered_namespace(self, ns_url: str, ns_db: str) -> None:
        """save с зарегистрированным узлом: stored=True, метка в выдаче get (§5.7)."""
        _register_namespace(ns_db, "work", "Рабочие заметки. Подпроекты — в листьях.")
        text = f"{self.marker}: деплой в work-узел"
        async with connect(ns_url) as session:
            saved = (await session.call_tool(
                "memory_save", {"text": text, "title": "Деплой в work", "namespace": "work"}
            )).structured_content
            assert saved["stored"] is True
            got = (await session.call_tool(
                "memory_get", {"ids": [saved["id"]]}
            )).structured_content
        assert got["notes"][0]["namespace"] == "work"

    @pytest.mark.asyncio
    async def test_save_unregistered_namespace_fails_with_hint(
        self, ns_url: str, ns_db: str
    ) -> None:
        """save в незарегистрированный узел → fail + hint (клиент не создаёт узлы)."""
        async with connect(ns_url) as session:
            saved = (await session.call_tool(
                "memory_save", {"text": f"{self.marker}: в неизвестный узел",
                                "title": "В неизвестный узел",
                                "namespace": "nope"}
            )).structured_content
        assert saved == {"stored": False, "hint": "namespace «nope» is not registered; create the missing domains via memory_namespace_create with a description from the purpose; up-to-date map — memory_namespaces"}

    @pytest.mark.asyncio
    async def test_search_unregistered_namespace_gives_hint(
        self, ns_url: str, ns_db: str
    ) -> None:
        async with connect(ns_url) as session:
            found = (await session.call_tool(
                "memory_search", {"query": "что угодно", "namespace": "nope"}
            )).structured_content
        assert found["results"] == []
        assert "nope" in found["hint"]

    @pytest.mark.asyncio
    async def test_list_namespace_filter(self, ns_url: str, ns_db: str) -> None:
        _register_namespace(ns_db, "work", "Рабочие заметки. Подпроекты — в листьях.")
        _register_namespace(ns_db, "projects", "Личные проекты. Сайт-резюме.")
        async with connect(ns_url) as session:
            await session.call_tool(
                "memory_save", {"text": f"{self.marker}: в work",
                                "title": "В work", "namespace": "work"}
            )
            await session.call_tool(
                "memory_save", {"text": f"{self.marker}: в projects",
                                "title": "В projects", "namespace": "projects"}
            )
            work = (await session.call_tool(
                "memory_list", {"namespace": "work", "limit": 5}
            )).structured_content
        assert all(item["namespace"] == "work" for item in work["items"])

    @pytest.mark.asyncio
    async def test_update_unregistered_namespace_fails(self, ns_url: str, ns_db: str) -> None:
        _register_namespace(ns_db, "work", "Рабочие заметки.")
        async with connect(ns_url) as session:
            saved = (await session.call_tool(
                "memory_save", {"text": f"{self.marker}: к перемещению",
                                "title": "К перемещению"}
            )).structured_content
            upd = (await session.call_tool(
                "memory_update", {"id": saved["id"],
                                  "text": f"{self.marker}: новая", "namespace": "nope"}
            )).structured_content
        assert upd["updated"] is False
        assert "nope" in upd["hint"]

    @pytest.mark.asyncio
    async def test_update_moves_namespace(self, ns_url: str, ns_db: str) -> None:
        _register_namespace(ns_db, "work", "Рабочие заметки.")
        async with connect(ns_url) as session:
            saved = (await session.call_tool(
                "memory_save", {"text": f"{self.marker}: в default",
                                "title": "В default"}
            )).structured_content
            upd = (await session.call_tool(
                "memory_update", {"id": saved["id"],
                                  "text": f"{self.marker}: переехала", "namespace": "work"}
            )).structured_content
            got = (await session.call_tool(
                "memory_get", {"ids": [saved["id"]]}
            )).structured_content
        assert upd["updated"] is True
        assert got["notes"][0]["namespace"] == "work"

    @pytest.mark.asyncio
    async def test_save_foreign_duplicate_gives_hint(self, ns_url: str, ns_db: str) -> None:
        """Дословный дубль в чужом узле: запись не блокирует, hint (US-8)."""
        _register_namespace(ns_db, "work", "Рабочие заметки. Подпроекты — в листьях.")
        text = f"{self.marker}: дубль между узлами"
        async with connect(ns_url) as session:
            first = (await session.call_tool(
                "memory_save", {"text": text, "title": "Межузловой дубль",
                                "namespace": "work"}
            )).structured_content
            assert first["stored"] is True
            second = (await session.call_tool(
                "memory_save", {"text": text, "title": "Межузловой дубль"}
            )).structured_content
        assert second["stored"] is True
        assert "work" in second["hint"]

    @pytest.mark.asyncio
    async def test_memory_namespaces_registry(self, ns_url: str, ns_db: str) -> None:
        """memory_namespaces: компактный контракт (§5.7) + promotion_candidates."""
        _register_namespace(ns_db, "work", "Рабочие заметки. Подпроекты — в листьях.")
        async with connect(ns_url) as session:
            registry = (await session.call_tool("memory_namespaces", {})).structured_content
        paths = {node["path"] for node in registry["namespaces"]}
        nsmap = {node["path"]: node for node in registry["namespaces"]}
        assert "default" in paths and "work" in paths
        assert set(nsmap["work"]) == {
            "path", "description", "status", "notes_count",
            "subtree_count", "updated_at",
        }
        assert nsmap["work"]["status"] == "confirmed"
        assert nsmap["work"]["description"] == "Рабочие заметки. Подпроекты — в листьях."
        # promotion_candidates: растущие группы default-заметок с общим hint
        # (Шаг 5): агрегация по живой БД, ещё не прогнанные через судью.
        _seed_default_group(ns_db, "work", "subo", 15)
        async with connect(ns_url) as session:
            registry = (await session.call_tool("memory_namespaces", {})).structured_content
        assert registry["promotion_candidates"] == [
            {"domain": "work", "subdomain": "subo", "count": 15}
        ]


class TestNamespaceCreateMCP:
    """lsb-0005-05 (FR-6/FR-8): memory_namespace_create — модель создаёт узел
    любого уровня (1..3, включая корни) с обязательным описанием, confirmed,
    прямое создание (судья не вызывается). Отказы (дубль / несуществующий
    родитель / невалидный путь / пустое описание) — мягкий fail + hint.
    Отдельный процесс от save: save узлы не создаёт."""

    marker = f"nscreate-{uuid.uuid4().hex[:8]}"

    @pytest.mark.asyncio
    async def test_create_root_depth1(self, ns_url: str) -> None:
        """Корень глубины 1 создаётся confirmed с обязательным описанием."""
        path = f"{self.marker}"
        async with connect(ns_url) as session:
            res = (await session.call_tool(
                "memory_namespace_create",
                {"path": path, "description": "Созданный моделью корень. Домен теста."},
            )).structured_content
        assert res["created"] is True
        assert res["path"] == path
        assert res["status"] == "confirmed"
        assert res["description"] == "Созданный моделью корень. Домен теста."

    @pytest.mark.asyncio
    async def test_create_depth2_and_depth3(self, ns_url: str) -> None:
        """Поддомен глубины 2 и раздел глубины 3 (родители создаются первыми)."""
        root = f"{self.marker}2"
        async with connect(ns_url) as session:
            root_res = (await session.call_tool(
                "memory_namespace_create",
                {"path": root, "description": "Корень для глубины."},
            )).structured_content
            assert root_res["created"] is True
            sub = (await session.call_tool(
                "memory_namespace_create",
                {"path": f"{root}/sub", "description": "Поддомен глубины 2."},
            )).structured_content
            assert sub["created"] is True
            assert sub["path"] == f"{root}/sub"
            leaf = (await session.call_tool(
                "memory_namespace_create",
                {"path": f"{root}/sub/leaf", "description": "Раздел глубины 3."},
            )).structured_content
        assert leaf["created"] is True
        assert leaf["path"] == f"{root}/sub/leaf"

    @pytest.mark.asyncio
    async def test_create_rejects_default_nesting(self, ns_url: str) -> None:
        """default/x — невалидный путь: мягкий отказ с hint."""
        async with connect(ns_url) as session:
            res = (await session.call_tool(
                "memory_namespace_create",
                {"path": "default/x", "description": "Вложенность под default."},
            )).structured_content
        assert res["created"] is False
        assert "default" in res["hint"]

    @pytest.mark.asyncio
    async def test_create_rejects_missing_parent(self, ns_url: str) -> None:
        """Несуществующий родитель глубины 2 → мягкий отказ с hint."""
        async with connect(ns_url) as session:
            res = (await session.call_tool(
                "memory_namespace_create",
                {"path": f"{self.marker}-nope/sub",
                 "description": "Лист без корня."},
            )).structured_content
        assert res["created"] is False
        assert "parent" in res["hint"]

    @pytest.mark.asyncio
    async def test_create_duplicate_fails_with_hint(self, ns_url: str) -> None:
        """Дубль узла → мягкий отказ с hint («уже зарегистрирован»)."""
        path = f"{self.marker}-dup"
        async with connect(ns_url) as session:
            first = (await session.call_tool(
                "memory_namespace_create",
                {"path": path, "description": "Первый раз."},
            )).structured_content
            assert first["created"] is True
            second = (await session.call_tool(
                "memory_namespace_create",
                {"path": path, "description": "Другое описание."},
            )).structured_content
        assert second["created"] is False
        assert "already" in second["hint"]

    @pytest.mark.asyncio
    async def test_create_rejects_empty_description(self, ns_url: str) -> None:
        """Пустое/пробельное описание → мягкий отказ с hint (обязательность)."""
        async with connect(ns_url) as session:
            res = (await session.call_tool(
                "memory_namespace_create",
                {"path": f"{self.marker}-empty", "description": ""},
            )).structured_content
        assert res["created"] is False
        assert "empty" in res["hint"]

    @pytest.mark.asyncio
    async def test_create_rejects_too_long_description(self, ns_url: str) -> None:
        """lsb-0005-08: >2 предложений (контракт описаний) → мягкий отказ."""
        async with connect(ns_url) as session:
            res = (await session.call_tool(
                "memory_namespace_create",
                {"path": f"{self.marker}-long",
                 "description": "Первое. Второе. Третье."},
            )).structured_content
        assert res["created"] is False
        assert "sentences" in res["hint"]

    @pytest.mark.asyncio
    async def test_create_rejects_depth4(self, ns_url: str) -> None:
        """Глубина 4 (> MAX_DEPTH) → мягкий отказ с hint."""
        async with connect(ns_url) as session:
            res = (await session.call_tool(
                "memory_namespace_create",
                {"path": "a/b/c/d", "description": "Слишком глубоко."},
            )).structured_content
        assert res["created"] is False
        assert "levels" in res["hint"]

class TestNamespaceCreateAntiseonymy:
    """lsb-0005-06 (FR-6): антисинонимия при создании — косинус-предфильтр
    описания нового узла против тематических узлов реестра. In-process MCP
    с детерминированным HashEmbedder (живые эмбеддинги в юнитах недоступны).
    Близкое описание (>0.90) → мягкий отказ «есть похожий: <ближайший>»
    (1 ближайший), создание не происходит; непохожее → создание; корень
    сверяется против корней; отказ эмбеддинга → создание (деградация)."""

    marker = f"nsanti-{uuid.uuid4().hex[:8]}"

    @pytest.fixture
    def mcp_fake(self, test_env: dict[str, str]):
        """In-process MCP над тестовой БД с детерминированным фейк-эмбеддингом."""
        from app.config import get_settings
        from app.services import Services
        from app.services.namespaces import NamespaceService
        from app.storage.db import init_db
        from app.transport.mcp import build_mcp
        from fakes import HashEmbedder

        settings = get_settings()
        init_db(settings)
        services = Services(
            notes=None,
            search=None,
            embedding=HashEmbedder(64),
            dedup=None,
            summary=None,
            judge=None,
            backup=None,
            namespaces=NamespaceService(settings),
            classifier=None,
            promotion=None,
        )
        return build_mcp(settings, services)

    @pytest.fixture
    def mcp_fail(self, test_env: dict[str, str]):
        """In-process MCP с отказавшим эмбеддингом (проверка деградации NFR-3)."""
        from app.config import get_settings
        from app.services import Services
        from app.services.namespaces import NamespaceService
        from app.storage.db import init_db
        from app.transport.mcp import build_mcp
        from fakes import FailingEmbedder

        settings = get_settings()
        init_db(settings)
        services = Services(
            notes=None,
            search=None,
            embedding=FailingEmbedder(),
            dedup=None,
            summary=None,
            judge=None,
            backup=None,
            namespaces=NamespaceService(settings),
            classifier=None,
            promotion=None,
        )
        return build_mcp(settings, services)

    @pytest.mark.asyncio
    async def test_similar_description_rejected_with_hint(self, mcp_fake) -> None:
        """Близкое описание (>0.90) → мягкий отказ: created False, hint
        «есть похожий: <ближайший>» (1 ближайший), создание не происходит."""
        await mcp_fake.call_tool(
            "memory_namespace_create",
            {"path": "work",
             "description": "Рабочие заметки. Подпроекты в листьях."},
        )
        res = (await mcp_fake.call_tool(
            "memory_namespace_create",
            {"path": "work2",
             "description": "Рабочие заметки. Подпроекты в листьях."},
        )).structured_content
        assert res == {"created": False, "hint": "there is a similar one: work"}

    @pytest.mark.asyncio
    async def test_unrelated_description_creates(self, mcp_fake) -> None:
        """Непохожее описание → узел создаётся."""
        await mcp_fake.call_tool(
            "memory_namespace_create",
            {"path": "work",
             "description": "Рабочие заметки. Подпроекты."},
        )
        res = (await mcp_fake.call_tool(
            "memory_namespace_create",
            {"path": "finance",
             "description": "Личные финансы и бюджетирование."},
        )).structured_content
        assert res["created"] is True
        assert res["path"] == "finance"

    @pytest.mark.asyncio
    async def test_root_checked_against_roots_only(self, mcp_fake) -> None:
        """Корень сверяется против корней: похожий ЛИСТ корень не блокирует."""
        await mcp_fake.call_tool(
            "memory_namespace_create",
            {"path": "work",
             "description": "Рабочие заметки. Подпроекты."},
        )
        await mcp_fake.call_tool(
            "memory_namespace_create",
            {"path": "work/sub",
             "description": "Подпроекты финансового планирования."},
        )
        # Новый корень с описанием, идентичным листу work/sub: корень
        # сверяется только против корней (work), лист не учитывается → создаётся.
        res = (await mcp_fake.call_tool(
            "memory_namespace_create",
            {"path": "finance",
             "description": "Подпроекты финансового планирования."},
        )).structured_content
        assert res["created"] is True
        assert res["path"] == "finance"

    @pytest.mark.asyncio
    async def test_leaf_compared_against_all_thematic(self, mcp_fake) -> None:
        """Лист сверяется против всех тематических узлов: близкий корень → отказ."""
        await mcp_fake.call_tool(
            "memory_namespace_create",
            {"path": "work",
             "description": "Сервисы HR и зарплаты."},
        )
        res = (await mcp_fake.call_tool(
            "memory_namespace_create",
            {"path": "work/payroll",
             "description": "Сервисы HR и зарплаты."},
        )).structured_content
        assert res["created"] is False
        assert res["hint"] == "there is a similar one: work"

    @pytest.mark.asyncio
    async def test_embedding_failure_still_creates(self, mcp_fail) -> None:
        """Отказ эмбеддинга → предфильтр пропущен: создание происходит (NFR-3)."""
        res = (await mcp_fail.call_tool(
            "memory_namespace_create",
            {"path": "work", "description": "Рабочие заметки."},
        )).structured_content
        assert res["created"] is True
        assert res["path"] == "work"


def _seed_default_group(
    db_path: str, domain: str, slug: str, count: int
) -> None:
    """default-заметки с готовой разметкой (вход триггера — SQL-агрегация)."""
    with sqlite3.connect(db_path) as conn:
        for i in range(count):
            conn.execute(
                "INSERT INTO notes (text, summary, summary_status, namespace, "
                "hint_path, confidence, classified_at) "
                "VALUES (?, ?, 'ok', 'default', ?, 0.7, ?)",
                (f"mcp seed {i} про {slug}", f"суммари {i}", f"{domain}/{slug}",
                 "2026-09-03T00:00:00Z"),
            )


class TestFailLogging:
    """Пул 11: отказы инструментов наблюдаемы в логах (NFR-4) + строгий memory_get.

    Юнит-слой против in-process MCP-сервера: caplog ловит app-логи в том же
    процессе (e2e-серверы — subprocess, их stdout в caplog не попадает).
    """

    @pytest.fixture
    def mcp_inproc(self, test_env: dict[str, str]):
        """In-process MCP-сервер над тестовой БД (caplog видит его логи)."""
        from app.config import get_settings
        from app.services import build_services
        from app.storage.db import init_db
        from app.transport.mcp import build_mcp

        settings = get_settings()
        init_db(settings)
        services = build_services(settings)
        return build_mcp(settings, services)

    @pytest.mark.asyncio
    async def test_memory_get_rejects_ambiguous_input(self, mcp_inproc) -> None:
        """Оба параметра (ids + id) → громкая ошибка вызова, не молчаливый приоритет."""
        with pytest.raises(Exception) as exc_info:
            await mcp_inproc.call_tool("memory_get", {"ids": [1], "id": 1})
        # SDK оборачивает ValueError в UnexpectedToolError; причина — наш ValueError.
        assert isinstance(exc_info.value.__cause__, ValueError)

    @pytest.mark.asyncio
    async def test_save_invalid_title_fail_logged(self, mcp_inproc, caplog) -> None:
        """save с невалидным title: fail-лог (failed=True, tool, latency), ответ не меняется."""
        with caplog.at_level(logging.INFO, logger="app"):
            result = await mcp_inproc.call_tool(
                "memory_save", {"text": "x", "title": "раз два три четыре пять шесть"}
            )
        assert result.structured_content == {
            "stored": False, "hint": "set a title ≤5 words",
        }
        fail = _tool_fail_records(caplog, "memory_save")
        assert fail
        assert fail[-1].reason == "set a title ≤5 words"
        assert fail[-1].latency_ms >= 0

    @pytest.mark.asyncio
    async def test_save_unregistered_namespace_fail_logged(self, mcp_inproc, caplog) -> None:
        """save в незарегистрированный узел: fail-лог с именем узла, ответ не меняется."""
        with caplog.at_level(logging.INFO, logger="app"):
            result = await mcp_inproc.call_tool(
                "memory_save", {"text": "x", "title": "В неизвестный узел",
                                "namespace": "nope"}
            )
        assert result.structured_content["stored"] is False
        assert "nope" in result.structured_content["hint"]
        fail = _tool_fail_records(caplog, "memory_save")
        assert fail
        assert "nope" in fail[-1].reason

    @pytest.mark.asyncio
    async def test_search_fail_logged(self, mcp_inproc, caplog) -> None:
        """search в незарегистрированный узел: fail-лог (failed=True, tool, latency)."""
        with caplog.at_level(logging.INFO, logger="app"):
            result = await mcp_inproc.call_tool(
                "memory_search", {"query": "что угодно", "namespace": "nope"}
            )
        assert result.structured_content["results"] == []
        assert "nope" in result.structured_content["hint"]
        fail = _tool_fail_records(caplog, "memory_search")
        assert fail
        assert "nope" in fail[-1].reason
        assert fail[-1].latency_ms >= 0

    @pytest.mark.asyncio
    async def test_list_fail_logged(self, mcp_inproc, caplog) -> None:
        """list в незарегистрированный узел: fail-лог (failed=True, tool, latency)."""
        with caplog.at_level(logging.INFO, logger="app"):
            result = await mcp_inproc.call_tool("memory_list", {"namespace": "nope"})
        assert result.structured_content["items"] == []
        assert "nope" in result.structured_content["hint"]
        fail = _tool_fail_records(caplog, "memory_list")
        assert fail
        assert "nope" in fail[-1].reason
        assert fail[-1].latency_ms >= 0

    @pytest.mark.asyncio
    async def test_namespace_create_fail_logged(self, mcp_inproc, caplog) -> None:
        """memory_namespace_create: дубль узла и пустое описание — fail-лог,
        ответ {created: False, hint}, узел не создан дважды."""
        with caplog.at_level(logging.INFO, logger="app"):
            first = await mcp_inproc.call_tool(
                "memory_namespace_create",
                {"path": "failns", "description": "Создан для проверки."},
            )
            duplicate = await mcp_inproc.call_tool(
                "memory_namespace_create",
                {"path": "failns", "description": "Дубль."},
            )
        assert first.structured_content["created"] is True
        assert first.structured_content["path"] == "failns"
        assert duplicate.structured_content == {
            "created": False, "hint": "node «failns» is already registered",
        }
        fail = _tool_fail_records(caplog, "memory_namespace_create")
        assert fail
        assert "already registered" in fail[-1].reason
        assert fail[-1].latency_ms >= 0


def _tool_fail_records(caplog, tool: str) -> list:
    """Записи caplog: tool_call для данного инструмента с failed=True."""
    return [
        r for r in caplog.records
        if getattr(r, "event", None) == "tool_call"
        and getattr(r, "tool", None) == tool
        and getattr(r, "failed", None) is True
    ]


class TestInstructionsBudget:
    """Замер бюджета §2: карта неймспейсов в инструкциях ≤ ~1300 токенов.
    Строим полные инструкции на реальном реестре (юнит, без сети)."""

    def test_map_in_instructions_within_budget(self, test_env: dict[str, str]) -> None:
        from app.config import get_settings
        from app.services import build_services
        from app.services.namespaces import NamespaceService
        from app.storage.db import init_db
        from app.transport.mcp import build_instructions

        settings = get_settings()
        init_db(settings)
        ns = NamespaceService(settings)
        ns.create("work", "Рабочие заметки. Подпроекты — в листьях.")
        ns.create("projects", "Личные проекты. Сайт-резюме.")
        ns.create("work/sbos2020", "СУБО 2020: сервисы HR.")
        instructions = build_instructions(build_services(settings))
        assert SERVER_INSTRUCTIONS in instructions
        # Фаза 11 (решение №9): правило названий вшито в базу инструкций.
        assert "title" in SERVER_INSTRUCTIONS
        assert "≤5 words" in SERVER_INSTRUCTIONS
        assert "- work: Рабочие заметки. Подпроекты — в листьях." in instructions
        assert "- projects: Личные проекты. Сайт-резюме." in instructions
        assert "- work/sbos2020: СУБО 2020: сервисы HR." in instructions
        # бюджет §2: грубая оценка токенов ≈ длина/4; ~10 узлов — далеко
        # до ~1300 (база сама по себе ~300 токенов).
        assert len(instructions.encode("utf-8")) / 4 <= 1300
