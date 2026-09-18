"""Объём текста `chars` в выдачах заметок (lsb-0010-04, задача №44, релиз 3.1.0).

FR-4.1…FR-4.4 + arch lsb-0010 §3.6–3.7: `memory_search`, `memory_list` и
`memory_get` (чтение без `query`/`chunk`) несут `chars` — объём ПОЛНОГО текста
заметки в СИМВОЛАХ (не байты, не токены, не snippet и не summary). Для одной и
той же заметки значение совпадает во всех трёх ручках — по нему модель решает,
хватит ли `memory_get` или нужен чанк. В chunk-режиме верхнеуровневый `chars`
СОХРАНЯЕТ прежний смысл (сумма символов ОТДАННЫХ чанков — контракт lsb-0003,
не переопределяем); полный объём заметки там виден по `total_chunks`. Деталь
`titles` остаётся `{id, title, namespace}` — объём в режиме сканирования реестра
не нужен (обратная совместимость формы). REST отдаёт полные сервисные
контракты, поэтому `chars` появляется в `GET /notes`, `GET /notes/{id}` и
`GET /search` без отдельного слоя.

Бюджеты §3.7 фиксируются этим релизом: `memory_search` при `top_k=5` — ≤ 1.2 КБ
(существующая канонная метрика, новое поле её не увеличивает), `memory_list` —
≤ 1.5 КБ (новая метрика). Замер — сериализованная компактная MCP-выдача (то,
что уходит по проводу): UTF-8 байты, «КБ» = 1000 байт, данные — пять атомарных
заметок ниже (~50 символов текста каждая). Фактический замер на этих данных:
search ≈ 1.12 КБ, list ≈ 1.27 КБ; сводка усечена не была (тексты короче
MAX_SUMMARY_CHARS) — на длинных заметках бюджет держит само усечение сводки.

Поверхность — in-process MCP (белые списки `mcp.py`) и TestClient приложения
(REST). Внешние LLM недоступны (штатная деградация, NFR-3) — выдачи
детерминированы.
"""

from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.services import Services, build_services
from app.storage.db import init_db
from app.transport.mcp import TOOL_NAMES, build_mcp

DIM = 64

# Бюджеты выдач (§3.7, lsb-0010-04): байты сериализованной компактной выдачи.
SEARCH_BUDGET_BYTES = 1_200  # memory_search, top_k=5
LIST_BUDGET_BYTES = 1_500  # memory_list

# Данные замера метрики: 5 атомарных заметок (текст ≤ ~50 символов — одна
# мысль на запись, название ≤5 слов) — сводка в выдаче идёт fallback-
# усечением (MAX_SUMMARY_CHARS=200), как у свежей заметки до работы воркера.
# Общее слово «релиз» у всех — запрос метрики находит всю пятёрку (поиск
# идёт FTS-only, внешние LLM недоступны).
METRIC_TEXTS = [
    "Релиз 3.1.0: деплой billing прошёл 2026-08-29.",
    "Релиз 3.1.0: индексы пересобраны 2026-09-01.",
    "Релиз 3.1.0: очередь векторов разобрана 2026-09-02.",
    "Релиз 3.1.0: бэкапы переехали в /data 2026-09-03.",
    "Релиз 3.1.0: стенд lsb-test обновлён 2026-09-05.",
]
METRIC_TITLES = [
    "Деплой billing",
    "Индексы",
    "Очередь",
    "Бэкапы",
    "Стенд",
]


@pytest.fixture
def settings_and_services(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> tuple[Settings, Services]:
    """Свежая БД + полная сборка сервисов (детерминированная размерность)."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings, build_services(settings)


@pytest.fixture
def mcp_server(settings_and_services: tuple[Settings, Services]):
    """In-process MCP поверх тех же сервисов (один код с REST)."""
    settings, services = settings_and_services
    return build_mcp(settings, services)


def marker() -> str:
    """Уникальный латинский токен: FTS находит его и дословный дедуп не мешает."""
    return f"charmark{uuid.uuid4().hex[:8]}"


def payload_size(out: dict) -> int:
    """Размер компактной выдачи в байтах — как она уходит по проводу."""
    return len(
        json.dumps(out, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def multiline_text(mark: str) -> str:
    """Многострочный текст длиннее snippet: chars — объём всего текста."""
    return "\n".join(
        [
            f"{mark}: первая строка заметки про деплой",
            "вторая строка: окно 14:20-14:26, ошибок нет",
            "третья строка: откат не потребовался, стенд зелёный",
        ]
    )


def long_text() -> str:
    """Текст на несколько чанков (chunk_size=1024 токена по умолчанию)."""
    return "\n".join(f"строка {i} длинной заметки про чанки и вектора" for i in range(400))


class TestMCPChars:
    """`chars` в компактных MCP-выдачах: search / list / get (FR-4.1…FR-4.3)."""

    @pytest.mark.asyncio
    async def test_search_list_get_agree_on_chars(
        self, settings_and_services: tuple[Settings, Services], mcp_server
    ) -> None:
        """Одна заметка — одно значение `chars` во всех трёх ручках (FR-4.2)."""
        _, services = settings_and_services
        mark = marker()
        text = multiline_text(mark)
        note_id = services.notes.save(text, title="Объём текста")["id"]

        found = (
            await mcp_server.call_tool("memory_search", {"query": mark})
        ).structured_content
        hit = next(r for r in found["results"] if r["id"] == note_id)
        listed = (
            await mcp_server.call_tool("memory_list", {"limit": 5})
        ).structured_content
        item = next(i for i in listed["items"] if i["id"] == note_id)
        got = (
            await mcp_server.call_tool("memory_get", {"ids": [note_id]})
        ).structured_content
        note = got["notes"][0]

        assert hit["chars"] == item["chars"] == note["chars"] == len(text)
        # Объём — именно ПОЛНОГО текста: это не snippet (120 симв.) и не
        # fallback-summary (200 симв.).
        assert len(text) > 120

    @pytest.mark.asyncio
    async def test_chars_is_len_of_full_text_for_short_notes(
        self, settings_and_services: tuple[Settings, Services], mcp_server
    ) -> None:
        """Однострочная и минимальная (1 символ) заметки — тоже len(text)."""
        _, services = settings_and_services
        short_id = services.notes.save("однострочная заметка", title="Строка")["id"]
        tiny_id = services.notes.save("X", title="Символ")["id"]

        got = (
            await mcp_server.call_tool("memory_get", {"ids": [short_id, tiny_id]})
        ).structured_content
        by_id = {n["id"]: n for n in got["notes"]}
        assert by_id[short_id]["chars"] == len("однострочная заметка")
        assert by_id[tiny_id]["chars"] == 1

    @pytest.mark.asyncio
    async def test_chunk_mode_keeps_chunk_sum_and_total_chunks(
        self, settings_and_services: tuple[Settings, Services], mcp_server
    ) -> None:
        """Chunk-режим: `chars` = сумма символов ОТДАННЫХ чанков (lsb-0003),
        `total_chunks` на месте — полный объём не подменяет прежний смысл."""
        _, services = settings_and_services
        text = long_text()
        note_id = services.notes.save(text, title="Длинная заметка")["id"]

        chunked = (
            await mcp_server.call_tool(
                "memory_get", {"id": note_id, "chunk": 0, "limit": 2}
            )
        ).structured_content
        assert chunked["total_chunks"] >= 2
        assert [c["chunk_index"] for c in chunked["chunks"]] == [0, 1]
        assert chunked["chars"] == sum(len(c["text"]) for c in chunked["chunks"])
        assert chunked["chars"] < len(text)  # отдана лишь часть заметки

        full = (
            await mcp_server.call_tool("memory_get", {"ids": [note_id]})
        ).structured_content
        assert full["notes"][0]["chars"] == len(text)  # без чанков — полный текст

    @pytest.mark.asyncio
    async def test_titles_detail_has_no_chars(
        self, settings_and_services: tuple[Settings, Services], mcp_server
    ) -> None:
        """Деталь `titles` — прежняя форма {id, title, namespace} без `chars`."""
        _, services = settings_and_services
        services.notes.save("заметка для titles-детали", title="Реестр")
        listed = (
            await mcp_server.call_tool(
                "memory_list", {"limit": 5, "detail": "titles"}
            )
        ).structured_content
        assert listed["items"]
        for item in listed["items"]:
            assert set(item) == {"id", "title", "namespace"}

    @pytest.mark.asyncio
    async def test_mcp_surface_is_21_tools(self) -> None:
        """Поверхность не расширялась: те же 21 инструмент (arch §3.8)."""
        assert len(TOOL_NAMES) == 21


class TestRESTChars:
    """REST отдаёт полные сервисные контракты — `chars` в /notes и /search."""

    @staticmethod
    def _headers(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def test_notes_and_search_carry_chars(self, client: TestClient, token: str) -> None:
        """`GET /notes`, `GET /notes/{id}`, `GET /search` несут одинаковый `chars`."""
        mark = marker()
        text = multiline_text(mark)
        created = client.post(
            "/notes",
            json={"text": text, "title": "Рестовый объём"},
            headers=self._headers(token),
        )
        note_id = created.json()["id"]

        single = client.get(
            f"/notes/{note_id}", headers=self._headers(token)
        ).json()
        assert single["chars"] == len(text)

        page = client.get("/notes", headers=self._headers(token)).json()
        item = next(i for i in page["items"] if i["id"] == note_id)
        assert item["chars"] == len(text)

        found = client.get(
            "/search", params={"q": mark}, headers=self._headers(token)
        ).json()
        hit = next(r for r in found["results"] if r["id"] == note_id)
        assert hit["chars"] == len(text)
        assert hit["chars"] == single["chars"] == item["chars"]


class TestCompactnessBudgets:
    """Бюджеты §3.7 на данных замера: search top_k=5 ≤ 1.2 КБ, list ≤ 1.5 КБ."""

    @pytest.fixture
    def seeded(
        self, settings_and_services: tuple[Settings, Services]
    ) -> tuple[Settings, Services]:
        """Пять атомарных заметок метрики (свежие — сводка fallback-усечением)."""
        _, services = settings_and_services
        for text, title in zip(METRIC_TEXTS, METRIC_TITLES):
            services.notes.save(text, title=title)
        return settings_and_services

    @pytest.mark.asyncio
    async def test_search_top_k_5_within_budget(self, seeded, mcp_server) -> None:
        """memory_search при top_k=5 — в бюджете 1.2 КБ (метрика не выросла)."""
        got = (
            await mcp_server.call_tool(
                "memory_search", {"query": "релиз", "top_k": 5}
            )
        ).structured_content
        assert len(got["results"]) == 5
        assert all(r["chars"] > 0 for r in got["results"])  # поле на месте
        size = payload_size(got)
        assert size <= SEARCH_BUDGET_BYTES, f"memory_search top_k=5: {size} B"

    @pytest.mark.asyncio
    async def test_list_within_budget(self, seeded, mcp_server) -> None:
        """memory_list — в бюджете 1.5 КБ (новая метрика §3.7)."""
        got = (
            await mcp_server.call_tool("memory_list", {"limit": 20})
        ).structured_content
        assert got["items"] and all(i["chars"] > 0 for i in got["items"])
        size = payload_size(got)
        assert size <= LIST_BUDGET_BYTES, f"memory_list: {size} B"
