"""Тесты замера вшитого текста (LSBN-F-11, постановка f11-01).

Методика — единственная на релиз: `scripts/measure_injected_text.py` меряет
три места (A — статика подключения, B — описания инструментов со схемами,
C — вес вызова по образцам ответов) в символах и токенах `tiktoken
cl100k_base`; счётчик токенов один на проект — `app.services.injection_budget`.

Тесты проверяют СТРУКТУРУ и воспроизводимость замера, а не конкретные числа:
числа зависят от состояния текстов (`f11-02` их сокращает) и живут в
`release/3.2.0/acceptance/injected-text-before.*`. Тестовое окружение —
общее (`tests/conftest.py`): обязательные env выставлены, БД — во временной
папке, боевая БД не открывается.
"""

from __future__ import annotations

import asyncio
import json

import tiktoken

from app.config import get_settings
from app.services import build_services
from app.services.injection_budget import count_tokens
from app.transport.mcp import build_mcp
from scripts.measure_injected_text import collect_measurement, render_markdown

# Поверхность блоков 1–5 (arch §7.1): 31 инструмент; блок 6 новых ручек не
# добавляет, поэтому число ниже — нижняя граница замера, а не догма: тест
# требует, чтобы замер не терял ни одного зарегистрированного инструмента.
EXPECTED_TOOL_FLOOR = 31


def _registered_tool_names() -> list[str]:
    """Инструменты, реально зарегистрированные в MCP-сборке на этих настройках."""
    settings = get_settings()
    mcp = build_mcp(settings, build_services(settings))
    return sorted(tool.name for tool in asyncio.run(mcp.list_tools()))


def test_count_tokens_matches_tiktoken() -> None:
    """`count_tokens` — ровно `len(cl100k_base.encode(text))`, пустая строка → 0."""
    encoding = tiktoken.get_encoding("cl100k_base")
    for text in (
        "",
        "a",
        "Привет, мир!",
        "memory_search — единственная точка поиска по смыслу.\n",
        "LSB 3.1.0: слот эмбеддингов переключён на mainsrv .109 (в 2 раза быстрее).",
    ):
        assert count_tokens(text) == len(encoding.encode(text))
    assert count_tokens("") == 0


def test_measure_covers_three_places() -> None:
    """Замер содержит непустые места A/B/C с символами и токенами."""
    data = collect_measurement(phase="before")

    assert data["phase"] == "before"
    assert data["metric"]["encoder"] == "cl100k_base"

    place_a = data["place_a"]
    assert place_a["total"]["chars"] > 0
    assert place_a["total"]["tokens"] > 0
    # Разбивка по слагаемым инструкций: база, правила неймспейсов, карта,
    # анонс навыков, раздел логов, строка про lifetime.
    assert len(place_a["components"]) >= 5
    assert place_a["components"][0]["block"] == "SERVER_INSTRUCTIONS"

    place_b = data["place_b"]
    assert place_b["tool_count"] == len(place_b["tools"]) > 0
    assert place_b["total_block"]["tokens"] > 0
    assert place_b["total_description"]["tokens"] > 0

    place_c = data["place_c"]
    assert place_c["samples"], "место C пусто: образцы ответов не собраны"
    assert place_c["total"]["tokens"] > 0
    assert all(sample["name"] for sample in place_c["samples"])


def test_measure_deterministic() -> None:
    """Два прогона на одном входе дают одинаковый JSON (в метриках нет времени)."""
    first = collect_measurement(phase="before")
    second = collect_measurement(phase="before")
    def dump(data: dict) -> str:
        return json.dumps(data, ensure_ascii=False, sort_keys=True)

    assert dump(first) == dump(second)


def test_measure_reports_all_tools() -> None:
    """Место B покрывает все зарегистрированные инструменты поверхности."""
    registered = _registered_tool_names()
    assert len(registered) >= EXPECTED_TOOL_FLOOR

    measured = [tool["name"] for tool in collect_measurement(phase="before")["place_b"]["tools"]]
    assert sorted(measured) == registered


def test_markdown_table_has_required_columns() -> None:
    """Markdown-отчёт — таблица «место, блок, символы, токены, комментарий»."""
    markdown = render_markdown(collect_measurement(phase="before"))
    assert "| Место | Блок | Символы | Токены | Комментарий |" in markdown
    assert "A" in markdown and "B" in markdown and "C" in markdown
