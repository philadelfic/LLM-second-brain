"""Логирование (Фаза 5, NFR-4): JSON в stdout, tool_call-события, приватность.

Два уровня проверки:
- **Юнит** — JsonFormatter (валидный JSON, extra-поля, ts), `preview` (80
  символов), `log_tool_call` (event/tool/latency_ms), идемпотентность
  `setup_logging`.
- **e2e против прод-точки входа** (`python -m app`, subprocess): stdout
  целиком — JSON-строки (app-логи И uvicorn access/error — см. log_config);
  вызов инструмента попадает в лог с латентностью и числом результатов;
  содержимое заметки в лог НЕ попадает (NFR-4), остаётся только длина.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import (
    create_mcp_http_client,
    streamable_http_client,
)

from app.observability import (
    QUERY_PREVIEW_CHARS,
    JsonFormatter,
    log_tool_call,
    preview,
    setup_logging,
)
from tests.conftest import TEST_ENV

LOG_SERVER_PORT = 18767
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Уникальные подстроки, которые НИ В КОЕМ случае не должны появиться в логах.
SECRET_ARCHIVE = "Шифр-сейф-7кв-2026-хранится-в-ячейке-B12"


# --- юнит-слой -----------------------------------------------------------


def make_record(message: str, **extra: object) -> logging.LogRecord:
    """Собрать LogRecord вручную (минус зависимость от живого логгера)."""
    record = logging.LogRecord(
        name="app.mcp", level=logging.INFO, pathname=__file__, lineno=1,
        msg=message, args=(), exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


class TestJsonFormatter:
    def test_record_is_json_line(self) -> None:
        text = JsonFormatter().format(
            make_record("tool call", event="tool_call", tool="memory_search", results=3)
        )
        payload = json.loads(text)
        assert payload["level"] == "INFO"
        assert payload["logger"] == "app.mcp"
        assert payload["message"] == "tool call"
        assert payload["event"] == "tool_call"
        assert payload["tool"] == "memory_search"
        assert payload["results"] == 3

    def test_timestamp_is_utc_iso_with_z(self) -> None:
        text = JsonFormatter().format(make_record("x"))
        ts = json.loads(text)["ts"]
        assert ts.endswith("Z")
        datetime_ok = time.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
        assert datetime_ok.tm_year >= 2026

    def test_unicode_survives(self) -> None:
        text = JsonFormatter().format(make_record("заметка «с хвостом»"))
        assert "заметка «с хвостом»" in text

    def test_exception_field(self) -> None:
        try:
            raise ValueError("бум")
        except ValueError:
            import sys as _sys

            record = make_record("fail")
            record.exc_info = _sys.exc_info()
        payload = json.loads(JsonFormatter().format(record))
        assert "ValueError" in payload["exception"]


class TestPreview:
    def test_long_query_truncated_to_80(self) -> None:
        long_query = "д" * 200
        assert len(preview(long_query)) == QUERY_PREVIEW_CHARS == 80

    def test_short_query_untouched(self) -> None:
        assert preview("короткий запрос") == "короткий запрос"


class TestLogToolCall:
    def test_event_fields_reach_record(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app"):
            log_tool_call("memory_list", time.perf_counter(), results=7, limit=20)
        record = caplog.records[-1]
        assert record.event == "tool_call"  # type: ignore[attr-defined]
        assert record.tool == "memory_list"  # type: ignore[attr-defined]
        assert record.results == 7  # type: ignore[attr-defined]
        assert record.latency_ms >= 0  # type: ignore[attr-defined]

    def test_serialized_extra_is_json(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app"):
            log_tool_call("memory_search", time.perf_counter(), results=2)
        payload = json.loads(JsonFormatter().format(caplog.records[-1]))
        assert payload["tool"] == "memory_search"
        assert payload["results"] == 2
        assert 0 <= payload["latency_ms"] < 10_000


class TestSetupLogging:
    def test_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        root = logging.getLogger()
        # Маркер уже стоит → второй вызов не добавляет handler-пара.
        monkeypatch.setattr(root, "_second_brain_json_handler", True, raising=False)
        before = list(root.handlers)
        setup_logging("INFO")
        assert list(root.handlers) == before

    def test_sets_level(self) -> None:
        setup_logging("WARNING")
        assert logging.getLogger().level == logging.WARNING
        setup_logging("INFO")  # вернуть(INFO — по умолчанию тестового env)


# --- e2e: прод-точка входа, stdout = JSON, приватность ---------------------


@pytest.fixture(scope="module")
def logged_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Сервер через `python -m app` — тот же путь логирования, что в проде."""
    db = tmp_path_factory.mktemp("logging") / "notes.db"
    log = tempfile.NamedTemporaryFile(  # noqa: SIM115 — закрывается в finally
        mode="r", prefix="second-brain-log-", suffix=".log", delete=False
    )
    env = {**os.environ, **TEST_ENV, "PORT": str(LOG_SERVER_PORT), "DB_PATH": str(db)}
    process = subprocess.Popen(
        [sys.executable, "-m", "app"],
        cwd=REPO_ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                log.flush()
                raise RuntimeError(
                    f"сервер упал при старте:\n{Path(log.name).read_text()}"
                )
            try:
                response = httpx.get(
                    f"http://127.0.0.1:{LOG_SERVER_PORT}/health", timeout=1
                )
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        else:
            raise RuntimeError("сервер не поднялся за 30с")
        yield log.name
    finally:
        process.terminate()
        process.wait(timeout=10)
        log.close()


@asynccontextmanager
async def mcp_session(base_url: str) -> AsyncIterator[ClientSession]:
    headers = {"Authorization": f"Bearer {TEST_ENV['MCP_AUTH_TOKEN']}"}
    http_client = create_mcp_http_client(headers=headers)
    async with streamable_http_client(
        f"{base_url}/mcp", http_client=http_client
    ) as streams, ClientSession(*streams) as session:
        await session.initialize()
        yield session


def log_lines(log_path: str) -> list[dict]:
    """Прочитать лог сервера; вернуть список распарсенных JSON-объектов."""
    raw = Path(log_path).read_text(encoding="utf-8", errors="replace")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


class TestStdoutIsJson:
    def test_every_line_parses_as_json(self, logged_server: str) -> None:
        with httpx.Client() as http:
            http.get(f"http://127.0.0.1:{LOG_SERVER_PORT}/health")
        records = log_lines(logged_server)
        assert records  # строки есть
        for record in records:
            assert "ts" in record and "level" in record and "message" in record


class TestToolCallLogs:
    @pytest.mark.asyncio
    async def test_list_tool_call_with_latency_and_results(
        self, logged_server: str
    ) -> None:
        base_url = f"http://127.0.0.1:{LOG_SERVER_PORT}"
        async with mcp_session(base_url) as session:
            await session.call_tool("memory_list", {"limit": 5})
        calls = [
            record
            for record in log_lines(logged_server)
            if record.get("event") == "tool_call"
        ]
        assert calls
        last = calls[-1]
        assert last["tool"] == "memory_list"
        assert last["results"] == 0  # новая БД — пусто
        assert isinstance(last["latency_ms"], (int, float))
        assert last["level"] == "INFO"

    @pytest.mark.asyncio
    async def test_query_preview_is_80_chars_max(
        self, logged_server: str
    ) -> None:
        base_url = f"http://127.0.0.1:{LOG_SERVER_PORT}"
        query = "ЗАПРОС-" + "ф" * 200  # длинный поисковый запрос
        async with mcp_session(base_url) as session:
            await session.call_tool("memory_search", {"query": query, "top_k": 3})
        searches = [
            record
            for record in log_lines(logged_server)
            if record.get("tool") == "memory_search"
            and record.get("event") == "tool_call"
        ]
        assert searches
        logged_query = searches[-1]["query"]
        assert len(logged_query) == QUERY_PREVIEW_CHARS
        assert not logged_query.startswith("ЗАПРОС-") or (
            logged_query == query[:QUERY_PREVIEW_CHARS]
        )

    @pytest.mark.asyncio
    async def test_note_content_never_reaches_logs(self, logged_server: str) -> None:
        base_url = f"http://127.0.0.1:{LOG_SERVER_PORT}"
        secret = "ШКФ-2026-хранилище-ключа-в-сейфе-B12-пароль-Капибара42"
        note_text = (
            f"Для теста приватности логов: {secret}. Заметка содержит секрет."
        )
        async with mcp_session(base_url) as session:
            # Фаза 11 (решение №9): новые заметки создаются только с title
            # (≤5 слов) — fail+hint иначе; title не содержит секрет из text.
            saved = await session.call_tool(
                "memory_save", {"text": note_text, "title": "Тест приватности логов"}
            )
            assert not saved.is_error
            await session.call_tool(
                "memory_update", {"id": saved.structured_content["id"], "text": note_text}
            )
        records = log_lines(logged_server)
        raw_text = json.dumps(records, ensure_ascii=False)
        assert secret not in raw_text
        assert note_text not in raw_text
        saves = [
            record
            for record in records
            if record.get("tool") == "memory_save"
            and record.get("event") == "tool_call"
        ]
        assert saves and saves[-1]["note_chars"] == len(note_text)
        # Зато сам вызов присутствует — NFR-4 не противоречит приватности.
        updates = [
            record
            for record in records
            if record.get("tool") == "memory_update"
        ]
        assert updates and updates[-1]["note_chars"] == len(note_text)