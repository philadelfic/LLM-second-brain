"""Тест-«сторож» бюджета вшитого текста (LSBN-F-11, постановка f11-03).

Требование постановки: бюджет вшитого текста стоит ПОД СТОРОЖЕМ — обычным
тестом общего прогона, который падает при превышении (FR-4.2). Числа бюджета —
константы `app/services/injection_budget.py` (arch §13.17, канон §6), в env их
нет: поднятие бюджета — осознанное изменение кода с записью причины (FR-4.4),
а не тихая правка окружения. Этот файл ничего не сокращает и не «подгоняет»:
он лишь фиксирует, что объём вшитого текста не вырос незаметно.

Метрика — одна на релиз: `injection_budget.count_tokens` (`cl100k_base`), ею же
меряет `scripts/measure_injected_text.py` (f11-01); образцы веса вызова берутся
из ТОГО ЖЕ набора приёмки (`default_samples`), поэтому сторож и отчёт меряют
одно и то же.

Проверка «сторож действительно ловит превышение» — `test_guard_fails_on_oversized`:
хелпер `assert_within` падает на искусственно завышенном измерении, так что
«тихое» разрастание текста невозможно.

Место C (вес вызова) в наборе приёмки не содержит перечня `links` (f11-01 его не
меряет), поэтому образец связей строится здесь — по потолку `settings.link_top`
и по белому списку полей `_LINK_ITEM` (`{id, title, namespace, chars}`).
"""

from __future__ import annotations

import asyncio
import json
import math

import pytest

from app.config import get_settings
from app.services import Services, build_services
from app.services.injection_budget import (
    BYTES_LINKS_MAX,
    BYTES_LIST_ITEM_MAX,
    BYTES_LIST_TITLE_ITEM_MAX,
    BYTES_SAVE_RESPONSE_MAX,
    BYTES_SEARCH_ITEM_MAX,
    CHARS_SKILLS_HINT_MAX,
    CHARS_UPDATE_HINT_MAX,
    TOKENS_CORE_MAX,
    TOKENS_INSTRUCTIONS_MAX,
    TOKENS_TOOL_DESCRIPTION_MAX,
    TOKENS_TOOL_SURFACE_MAX,
    count_tokens,
)
from app.services.skills import HINT_SKILLS_BODY
from app.services.update_check import UPDATE_HINT_TEMPLATE, UpdateChecker
from app.storage.db import init_db
from app.transport.mcp import (
    server_instructions,
    tool_descriptions,
    TOOL_NAMES,
    build_instructions,
    build_mcp,
)


from scripts.measure_injected_text import default_samples


# 3.4.1 (LSBN-T-26): тексты с числом лимита собираются функциями от
# настроек — здесь сборка от значений тестового окружения (роль прежних
# констант; число приходит из Settings, не литерал).
SERVER_INSTRUCTIONS = server_instructions(get_settings().max_note_chars)
TOOL_DESCRIPTIONS = tool_descriptions(get_settings().max_note_chars)

EXPECTED_TOOL_COUNT = 31

#: Репрезентативная карта узлов — состояние реестра, а не «пустая» БД: карта и
#: анонс навыков входят в `build_instructions` (нижняя граница замера — одна
#: дефолтная строка). Пути/описания похожи на боевые по длине и форме.
REPRESENTATIVE_NODES = (
    ("projects", "Проекты: репозитории, релизы, деплой, приёмка."),
    ("projects/llmsecondbrain", "LSB: заметки, память, слоты, навыки."),
    ("work", "Рабочее: сервисы, конфиги, доступы, стенд."),
    ("personal", "Личное: привычки, быт, планы на неделю."),
    ("tools", "Инструменты: docker, серверы, замеры, документация."),
)

#: Образец элемента `links` (`_LINK_ITEM`: id, title, namespace, chars) — берётся
#: по потолку `link_top`; длиной похож на боевые связи (см. место C в замере «до»).
LINK_SAMPLE = (
    ("LSB 3.2.0: injected-text budget under the guard", "tools", 1536),
    ("Docker backup and restore drill of the memory", "work", 1024),
    ("Embedding slot switched to the second server", "projects/llmsecondbrain", 768),
)


def assert_within(measured: int, budget: int, label: str) -> None:
    """Сторож: превышение бюджета — падение с понятным сообщением (FR-4.2).

    Сообщение фиксированного вида `label: measured N > budget M`: по нему сразу
    видно, ЧТО выросло и на сколько, и какой бюджет пересматривать (FR-4.4).
    Ровно на бюджете проверка проходит: бюджет — это потолок, а не «меньше».
    """
    assert measured <= budget, f"{label}: measured {measured} > budget {budget}"


def _bytes(text: str) -> int:
    """Байты UTF-8 — метрика веса вызова (3.1.0)."""
    return len(text.encode("utf-8"))


def _sample(name: str) -> dict[str, str]:
    """Образец приёмки по имени — тот же набор, что у замера f11-01."""
    for sample in default_samples():
        if sample["name"] == name:
            return sample
    raise AssertionError(f"acceptance sample {name!r} is missing from the f11-01 set")


@pytest.fixture
def settings(test_env: dict[str, str]):
    """Тестовые настройки + инициализированная БД (реестр узлов и навыков)."""
    get_settings.cache_clear()
    resolved = get_settings()
    init_db(resolved)
    return resolved


@pytest.fixture
def services(settings) -> Services:
    """Полная in-process сборка сервисов с репрезентативным реестром узлов."""
    built = build_services(settings)
    for path, description in REPRESENTATIVE_NODES:
        built.namespaces.create(path, description)
    return built


def test_instructions_within_budget(services: Services, settings) -> None:
    """Место A: `build_instructions` с включёнными областями, картой и анонсом."""
    text = build_instructions(services, settings)
    # Карта и анонс навыков реально попали в текст — иначе замер был бы вырожден.
    assert "projects/llmsecondbrain" in text
    assert_within(count_tokens(text), TOKENS_INSTRUCTIONS_MAX, "build_instructions")


def test_core_within_budget() -> None:
    """Место A: короткое ядро `SERVER_INSTRUCTIONS`."""
    assert_within(count_tokens(SERVER_INSTRUCTIONS), TOKENS_CORE_MAX, "SERVER_INSTRUCTIONS")


def test_each_tool_description_within_budget() -> None:
    """Место B: КАЖДОЕ описание инструмента — в своём потолке."""
    for name in sorted(TOOL_DESCRIPTIONS):
        assert_within(
            count_tokens(TOOL_DESCRIPTIONS[name]),
            TOKENS_TOOL_DESCRIPTION_MAX,
            f"TOOL_DESCRIPTIONS[{name!r}]",
        )


def test_tool_surface_within_budget() -> None:
    """Место B: сумма всех описаний поверхности — в бюджет."""
    total = sum(count_tokens(text) for text in TOOL_DESCRIPTIONS.values())
    assert_within(total, TOKENS_TOOL_SURFACE_MAX, "tool surface (sum of descriptions)")


def test_call_weight_within_budget(settings) -> None:
    """Место C: образцы веса вызова — в байтовых бюджетах.

    Элемент выдачи: `memory_search` (top_k=5) и `memory_list` (summaries/titles)
    — на элемент (сумма образца / число элементов, округление вверх: обёртка
    ответа отнесена к элементу, то есть оценка сверху); `links` — весь перечень
    одиночного чтения по потолку `link_top`; ответ `memory_save` — целиком.
    """
    search = _sample("memory_search top_k=5")
    assert_within(
        math.ceil(_bytes(search["payload"]) / 5),
        BYTES_SEARCH_ITEM_MAX,
        "memory_search top_k=5 (per result item)",
    )

    summaries = _sample("memory_list detail=summaries (limit=20)")
    assert_within(
        math.ceil(_bytes(summaries["payload"]) / 20),
        BYTES_LIST_ITEM_MAX,
        "memory_list detail=summaries (per item)",
    )

    titles = _sample("memory_list detail=titles (limit=20)")
    assert_within(
        math.ceil(_bytes(titles["payload"]) / 20),
        BYTES_LIST_TITLE_ITEM_MAX,
        "memory_list detail=titles (per item)",
    )

    links = [
        {
            "id": 400 + index,
            "title": LINK_SAMPLE[index % len(LINK_SAMPLE)][0],
            "namespace": LINK_SAMPLE[index % len(LINK_SAMPLE)][1],
            "chars": LINK_SAMPLE[index % len(LINK_SAMPLE)][2],
        }
        for index in range(settings.link_top)
    ]
    links_payload = json.dumps(
        links, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert_within(
        _bytes(links_payload), BYTES_LINKS_MAX, f"memory_get links (link_top={settings.link_top})"
    )

    saved = _sample("memory_save")
    assert_within(_bytes(saved["payload"]), BYTES_SAVE_RESPONSE_MAX, "memory_save response")


def test_hints_within_char_budget(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Хинты, подкладываемые к ответам ручек, — в символьных бюджетах."""
    assert_within(len(HINT_SKILLS_BODY), CHARS_SKILLS_HINT_MAX, "HINT_SKILLS_BODY")

    # Хинт обновления идёт по публичному пути `UpdateChecker.hint()`, а не по
    # шаблону: бюджет держит именно выдача ручки (в том числе на длинных notes).
    monkeypatch.setenv("UPDATE_CHECK_ENABLED", "true")
    monkeypatch.setenv("UPDATE_CHECK_URL", "http://127.0.0.1:1/update.json")
    get_settings.cache_clear()
    settings = get_settings()
    checker = UpdateChecker(
        settings,
        fetcher=lambda: {"version": "99.0.0", "notes": "n" * 1000},
    )
    checker.refresh()
    hint = checker.hint()
    assert hint is not None, "the update hint was not produced — the guard measured nothing"
    assert_within(len(hint), CHARS_UPDATE_HINT_MAX, "update_hint text (long notes)")
    short = UPDATE_HINT_TEMPLATE.format(version="3.2.0", notes="short notes")
    assert_within(len(short), CHARS_UPDATE_HINT_MAX, "update_hint text (short notes)")


def test_guard_fails_on_oversized() -> None:
    """Сторож ЛОВИТ превышение: искусственно завышенное измерение падает."""
    # Ровно на бюджете — проходит (бюджет это потолок, а не «строго меньше»).
    assert_within(TOKENS_CORE_MAX, TOKENS_CORE_MAX, "boundary")
    with pytest.raises(AssertionError) as excinfo:
        assert_within(TOKENS_CORE_MAX + 1, TOKENS_CORE_MAX, "SERVER_INSTRUCTIONS")
    assert (
        f"SERVER_INSTRUCTIONS: measured {TOKENS_CORE_MAX + 1} > budget {TOKENS_CORE_MAX}"
        in str(excinfo.value)
    ), f"unexpected guard message: {excinfo.value}"


def test_surface_tool_count_is_31() -> None:
    """Число инструментов равно бюджетному (31): рост поверхности — пересмотр.

    Расхождение означает, что поверхность выросла без пересмотра бюджета
    `TOKENS_TOOL_SURFACE_MAX` — это сигнал, а не повод молча поднять число.
    """
    assert len(TOOL_NAMES) == EXPECTED_TOOL_COUNT, (
        f"TOOL_DESCRIPTIONS declares {len(TOOL_NAMES)} tools, "
        f"the budget was revised for {EXPECTED_TOOL_COUNT}"
    )
    settings = get_settings()
    mcp = build_mcp(settings, build_services(settings))
    registered = sorted(tool.name for tool in asyncio.run(mcp.list_tools()))
    assert len(registered) == EXPECTED_TOOL_COUNT, (
        f"the registered MCP surface lists {len(registered)} tools "
        f"(expected {EXPECTED_TOOL_COUNT}) — revise TOKENS_TOOL_SURFACE_MAX "
        f"deliberately (FR-4.4): {registered}"
    )
