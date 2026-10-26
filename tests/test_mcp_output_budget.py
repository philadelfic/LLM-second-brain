"""LSBN-F-13: бюджет выдачи и двухуровневое чтение (релиз 3.4.0).

Дефолт 3.4.0 (решение О. 21.10.2026) — сокращение выдачи ПО УМОЛЧАНИЮ,
а не опция: агент сначала видит шапки/карту, полный текст читает точечно
по id. Проверяется транспортная сборка (`build_mcp` + `call_tool`,
один код с REST):

* `memory_search` (semantic): шапки ровно `{id, title, namespace, chars,
  score}`, score = `rrf_score` с 4 знаками; саммари и дат в выдаче нет;
* `budget_chars` малым — не влезающий элемент деградирует до шапки
  `{id, title, namespace}` + `"truncated": true`, hint усечения один
  на ответ и называет id («read point-wise by id»); пустой результат —
  hint сервиса, усечение к нему не примешивается;
* `budget_chars` вне 100..100000 — мягкий отказ `HINT_INVALID_BUDGET`
  у всех трёх ручек (search/list/get);
* `memory_list`: дефолтная деталь — `titles`; явная `detail=summaries`
  тоже в бюджете; без усечения сохраняется «+N more — offset=K»;
* `memory_get`: одиночный `id` и `ids` из одного элемента — полный текст
  без бюджета + `links`; batch (ids > 1) — шапки `{id, title, namespace,
  chars}` без `text`;
* тест-сторож объёма: сумма элементов ответа по умолчанию не превышает
  дефолт бюджета поверхности (сумма `len(json.dumps(item))` элементов,
  не весь конверт).

Моделей никто не зовёт: заметки сохраняются через `NoteService` с
детерминированным `HashEmbedder`, поиск живёт на FTS-ветке.
"""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings
from app.services import Services, build_services
from app.services.notes import NoteService
from app.services.output_budget import (
    GET_BUDGET_CHARS_DEFAULT,
    HINT_INVALID_BUDGET,
    LIST_BUDGET_CHARS_DEFAULT,
    SEARCH_BUDGET_CHARS_DEFAULT,
    item_chars,
)
from app.storage.db import init_db
from app.transport.mcp import build_mcp
from fakes import HashEmbedder

DIM = 8
# Уникальный токен поиска: trigram-FTS находит только заметки теста.
MARKER = "budgetmarker"


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """БД во временной папке, маленькая размерность эмбеддинга."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def services(settings: Settings) -> Services:
    """Полная сборка сервисов поверх тестовой БД."""
    return build_services(settings)


@pytest.fixture
def mcp_server(settings: Settings, services: Services):
    """In-process MCP поверх тех же сервисов (один код с REST)."""
    return build_mcp(settings, services)


def _note(settings: Settings, text: str, title: str) -> int:
    """Сохранить заметку детерминированным эмбеддером (моделей нет)."""
    return NoteService(settings, HashEmbedder(DIM)).save(text, title=title)["id"]


def _sum_chars(items: list[dict]) -> int:
    """Сумма элементов в символах — как уйдёт в JSON выдачи (сторож объёма)."""
    return sum(item_chars(item) for item in items)


def _budget_for_degradation(items: list[dict], full_count: int) -> int:
    """Бюджет, при котором ровно `full_count` элементов влезают целиком.

    Хвостовой элемент деградирует до шапки: его шапка влезает в остаток,
    а следующий полный элемент — уже нет (окно деградации). Размеры берём
    фактические (item_chars), поэтому тест не зависит от длины полей.
    """
    used = sum(item_chars(item) for item in items[:full_count])
    header = {key: items[full_count][key] for key in ("id", "title", "namespace")}
    header["truncated"] = True
    return used + item_chars(header) + 2


class TestSearchHeaders:
    """Дефолт semantic-поиска: шапки `{id, title, namespace, chars, score}`."""

    @pytest.fixture
    def seeded(self, settings: Settings) -> list[int]:
        """Четыре заметки под общий маркер поиска (тексты уникальны)."""
        return [
            _note(settings, f"{MARKER}: текст находки номер {index}", f"Находка{index}")
            for index in range(1, 5)
        ]

    @pytest.mark.asyncio
    async def test_default_hits_are_scored_headers(
        self, seeded: list[int], services: Services, mcp_server
    ) -> None:
        """Элемент — ровно шапка с score = rrf_score (4 знака), без саммари/дат."""
        got = (
            await mcp_server.call_tool("memory_search", {"query": MARKER})
        ).structured_content
        assert got["results"]
        reference = {
            hit["id"]: round(float(hit["rrf_score"]), 4)
            for hit in services.search.search(MARKER, top_k=20)["results"]
        }
        for hit in got["results"]:
            assert set(hit) == {"id", "title", "namespace", "chars", "score"}
            assert hit["id"] in reference
            assert hit["score"] == reference[hit["id"]]
        assert set(hit["id"] for hit in got["results"]) <= set(seeded)
        assert "hint" not in got

    @pytest.mark.asyncio
    async def test_small_budget_degrades_to_headers(
        self, seeded: list[int], services: Services, mcp_server
    ) -> None:
        """Малый бюджет: хвост деградирует до шапок с `truncated: true`.

        Бюджет подбирается по фактическим размерам шапок: две влезают
        целиком, третья — только шапкой, id не влезших — в hint.
        """
        full = (
            await mcp_server.call_tool(
                "memory_search", {"query": MARKER, "budget_chars": 100_000}
            )
        ).structured_content["results"]
        budget = _budget_for_degradation(full, 2)

        got = (
            await mcp_server.call_tool(
                "memory_search", {"query": MARKER, "budget_chars": budget}
            )
        ).structured_content
        assert [set(hit) for hit in got["results"][:2]] == [
            {"id", "title", "namespace", "chars", "score"},
        ] * 2
        degraded = got["results"][2]
        assert set(degraded) == {"id", "title", "namespace", "truncated"}
        assert degraded["truncated"] is True
        hint = got["hint"]
        assert "read point-wise" in hint
        assert str(degraded["id"]) in hint  # id деградировавшей — в hint
        assert len(got["results"]) < len(full)  # остальное за бюджетом

    @pytest.mark.asyncio
    async def test_empty_result_keeps_service_hint(
        self, settings: Settings, mcp_server
    ) -> None:
        """Пустой результат — hint сервиса (не усечения), резать нечего."""
        got = (
            await mcp_server.call_tool("memory_search", {"query": "неттакогослова"})
        ).structured_content
        assert got["results"] == []
        assert "hint" in got
        assert "read point-wise" not in got["hint"]


@pytest.mark.parametrize("budget_chars", [99, 100_001])
class TestInvalidBudget:
    """`budget_chars` вне 100..100000 — мягкий отказ у всех трёх ручек."""

    @pytest.mark.asyncio
    async def test_search_soft_refusal(self, settings: Settings, mcp_server, budget_chars: int) -> None:
        got = (
            await mcp_server.call_tool(
                "memory_search",
                {"query": MARKER, "budget_chars": budget_chars},
            )
        ).structured_content
        assert got == {"results": [], "hint": HINT_INVALID_BUDGET}

    @pytest.mark.asyncio
    async def test_list_soft_refusal(self, settings: Settings, mcp_server, budget_chars: int) -> None:
        got = (
            await mcp_server.call_tool(
                "memory_list", {"limit": 5, "budget_chars": budget_chars}
            )
        ).structured_content
        assert got == {"items": [], "total": 0, "hint": HINT_INVALID_BUDGET}

    @pytest.mark.asyncio
    async def test_get_soft_refusal(self, settings: Settings, mcp_server, budget_chars: int) -> None:
        got = (
            await mcp_server.call_tool(
                "memory_get", {"ids": [1], "budget_chars": budget_chars}
            )
        ).structured_content
        assert got == {"notes": [], "hint": HINT_INVALID_BUDGET}


class TestListForms:
    """Дефолт листинга — titles; summaries — явная деталь в бюджете."""

    @pytest.fixture
    def seeded(self, settings: Settings) -> list[int]:
        """Три заметки: страница limit=2 оставляет «+1 more»."""
        return [
            _note(
                settings,
                f"{MARKER}-list: заметка номер {index} с уникальным текстом",
                f"Список{index}",
            )
            for index in range(1, 4)
        ]

    @pytest.mark.asyncio
    async def test_default_detail_is_titles(
        self, seeded: list[int], mcp_server
    ) -> None:
        """Без detail — форма titles: ровно {id, title, namespace} + страница."""
        got = (
            await mcp_server.call_tool("memory_list", {"limit": 2})
        ).structured_content
        assert len(got["items"]) == 2
        for item in got["items"]:
            assert set(item) == {"id", "title", "namespace"}
        assert got["total"] == 3
        assert got["has_more"] is True
        assert got["hint"] == "+1 more — offset=2"  # без усечения — подсказка
        assert "summary" not in got["items"][0]

    @pytest.mark.asyncio
    async def test_summaries_within_budget_keep_more_hint(
        self, seeded: list[int], mcp_server
    ) -> None:
        """detail=summaries в дефолтном бюджете: элементы целиком, hint «+N more»."""
        got = (
            await mcp_server.call_tool(
                "memory_list", {"limit": 2, "detail": "summaries"}
            )
        ).structured_content
        assert len(got["items"]) == 2
        assert all("summary" in item for item in got["items"])
        assert all("truncated" not in item for item in got["items"])
        assert got["hint"] == "+1 more — offset=2"

    @pytest.mark.asyncio
    async def test_summaries_small_budget_degrades_to_headers(
        self, seeded: list[int], services: Services, mcp_server
    ) -> None:
        """detail=summaries с малым бюджетом: не влезающий — шапка + truncated."""
        full = (
            await mcp_server.call_tool(
                "memory_list",
                {"limit": 10, "detail": "summaries", "budget_chars": 100_000},
            )
        ).structured_content["items"]
        budget = _budget_for_degradation(full, 1)

        got = (
            await mcp_server.call_tool(
                "memory_list", {"limit": 10, "detail": "summaries", "budget_chars": budget}
            )
        ).structured_content
        first = got["items"][0]
        assert "summary" in first and "truncated" not in first  # влез целиком
        degraded = got["items"][1]
        assert set(degraded) == {"id", "title", "namespace", "truncated"}
        assert degraded["truncated"] is True
        assert "read point-wise" in got["hint"]
        assert str(degraded["id"]) in got["hint"]
        # Страница сохранена: усечение не съело поля пагинации.
        assert got["total"] == 3


class TestGetLevels:
    """Одиночное чтение — уровень 2 (полный текст), batch — шапки."""

    @pytest.fixture
    def seeded(self, settings: Settings) -> list[int]:
        return [
            _note(
                settings,
                f"{MARKER}-get: полный текст заметки номер {index} "
                "с деталями и датой 2026-10-26",
                f"Чтение{index}",
            )
            for index in range(1, 4)
        ]

    @pytest.mark.asyncio
    async def test_single_id_returns_full_text_and_links(
        self, seeded: list[int], settings: Settings, mcp_server
    ) -> None:
        """Одиночный `id` — полный текст без искажений + `links`."""
        text = f"{MARKER}-single: вычитанный целиком текст заметки"
        note_id = _note(settings, text, "Одиночка")
        got = (
            await mcp_server.call_tool("memory_get", {"id": note_id})
        ).structured_content
        assert got["notes"][0]["text"] == text  # текст не искажён
        assert got["notes"][0]["chars"] == len(text)
        assert got["links"] == []

    @pytest.mark.asyncio
    async def test_one_item_list_is_full_text_alias(
        self, seeded: list[int], settings: Settings, mcp_server
    ) -> None:
        """`ids` из одного элемента — алиас одиночного: полный текст + links."""
        text = f"{MARKER}-alias: текст вторым способом"
        note_id = _note(settings, text, "Алиас")
        got = (
            await mcp_server.call_tool("memory_get", {"ids": [note_id]})
        ).structured_content
        assert got["notes"][0]["text"] == text
        assert "links" in got

    @pytest.mark.asyncio
    async def test_batch_returns_headers_without_text(
        self, seeded: list[int], mcp_server
    ) -> None:
        """Batch (ids > 1) — шапки {id, title, namespace, chars}, без text/links."""
        got = (
            await mcp_server.call_tool("memory_get", {"ids": seeded})
        ).structured_content
        assert len(got["notes"]) == 3
        for item in got["notes"]:
            assert set(item) == {"id", "title", "namespace", "chars"}
            assert "text" not in item
        assert "links" not in got
        assert "hint" not in got  # шапки влезают в дефолт 2000

    @pytest.mark.asyncio
    async def test_batch_small_budget_hints_cut_ids(
        self, seeded: list[int], mcp_server
    ) -> None:
        """Batch с малым бюджетом: hint усечения называет id точечного чтения."""
        full = (
            await mcp_server.call_tool(
                "memory_get", {"ids": seeded, "budget_chars": 100_000}
            )
        ).structured_content["notes"]
        budget = _budget_for_degradation(full, 1)

        got = (
            await mcp_server.call_tool(
                "memory_get", {"ids": seeded, "budget_chars": budget}
            )
        ).structured_content
        first = got["notes"][0]
        assert set(first) == {"id", "title", "namespace", "chars"}  # влез целиком
        assert len(got["notes"]) < len(full)
        assert "read point-wise" in got["hint"]
        for note_id in seeded[1:]:
            assert str(note_id) in got["hint"]


class TestVolumeGuard:
    """Сторож объёма: сумма элементов дефолтного ответа ≤ дефолта бюджета."""

    @pytest.fixture
    def many(self, settings: Settings) -> list[int]:
        """Двадцать заметок: верхности поиска/листинга и batch-get."""
        return [
            _note(
                settings,
                f"{MARKER}-guard: объёмная заметка номер {index} про бюджет выдачи "
                "и двухуровневое чтение с уникальным хвостом текста",
                f"Объём{index}",
            )
            for index in range(1, 21)
        ]

    @pytest.mark.asyncio
    async def test_search_within_default_budget(self, many: list[int], mcp_server) -> None:
        got = (
            await mcp_server.call_tool("memory_search", {"query": MARKER})
        ).structured_content
        assert got["results"]
        assert _sum_chars(got["results"]) <= SEARCH_BUDGET_CHARS_DEFAULT

    @pytest.mark.asyncio
    async def test_list_titles_within_default_budget(self, many: list[int], mcp_server) -> None:
        got = (
            await mcp_server.call_tool("memory_list", {"limit": 20})
        ).structured_content
        assert len(got["items"]) == 20
        assert _sum_chars(got["items"]) <= LIST_BUDGET_CHARS_DEFAULT

    @pytest.mark.asyncio
    async def test_list_summaries_within_default_budget(
        self, many: list[int], mcp_server
    ) -> None:
        """Summaries-деталь на странице 20 не влезает — деградация держит бюджет."""
        got = (
            await mcp_server.call_tool(
                "memory_list", {"limit": 20, "detail": "summaries"}
            )
        ).structured_content
        assert _sum_chars(got["items"]) <= LIST_BUDGET_CHARS_DEFAULT
        assert "hint" in got  # часть элементов деградировала

    @pytest.mark.asyncio
    async def test_get_batch_within_default_budget(self, many: list[int], mcp_server) -> None:
        got = (
            await mcp_server.call_tool("memory_get", {"ids": many})
        ).structured_content
        assert len(got["notes"]) == 20
        assert _sum_chars(got["notes"]) <= GET_BUDGET_CHARS_DEFAULT
