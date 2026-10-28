"""Бюджет выдачи MCP-ручек в символах (LSBN-F-13, релиз 3.4.0).

Решение О. 21.10.2026: сокращение выдачи — ПОВЕДЕНИЕ ПО УМОЛЧАНИЮ, а не
опция: рука отдаёт шапки в пределах бюджета, полный текст — точечным чтением
по id (`memory_get` одного id). Единица — СИМВОЛЫ (решение О. 17.10.2026),
не токены и не байты: параметр зовётся `budget_chars`, мера элемента —
длина его JSON-сериализации (`len(json.dumps(...))`).

Числа-дефолты живут В КОДЕ, не в env (по прецеденту `injection_budget`,
FR-4.4): поднятие бюджета — осознанное изменение кода с записью причины.

Модуль — чистые функции без домена и I/O: транспорт (`mcp.py`) сериализует
элементы сам и зовёт `fit_within_budget`. Диапазон параметра проверяет
транспорт (мягкий отказ `HINT_INVALID_BUDGET`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

#: Диапазон параметра `budget_chars` (символов на ответ).
BUDGET_CHARS_MIN = 100
BUDGET_CHARS_MAX = 100_000

#: Дефолты поверхностей (символов на ответ). Search: шапка ~60–100 симв.,
#: top_k=5 укладывается с запасом, top_k=20 — режется честно. List: страница
#: titles (20 × ~60) влезает целиком, summaries режется до ~6 элементов.
#: Get: batch-шапки (20 × ~70) — с запасом.
SEARCH_BUDGET_CHARS_DEFAULT = 1200
LIST_BUDGET_CHARS_DEFAULT = 2000
GET_BUDGET_CHARS_DEFAULT = 2000

#: Карта окружения заметки (LSBN-F-16): свой потолок символов на строки
#: `neighbors` при `memory_get(map=true)`. Карта живёт при одиночном чтении
#: (там `budget_chars` не применяется), поэтому потолок свой — константа
#: кода, не env и не параметр ручки; не влезающая строка деградирует до
#: шапки с признаком усечения, как в F-13.
MAP_BUDGET_CHARS_DEFAULT = 1500

#: Шапка-проекция элемента при деградации (п.4 брифа: не влезающий целиком
#: элемент отдаётся шапкой + признак усечения).
HEADER_FIELDS: tuple[str, ...] = ("id", "title", "namespace")

#: Мягкий отказ параметра (fail + hint, как mode/detail).
HINT_INVALID_BUDGET = (
    f"budget_chars out of range; expected {BUDGET_CHARS_MIN}.."
    f"{BUDGET_CHARS_MAX} characters"
)

#: Признак усечения — ЕДИНСТВЕННАЯ hint успешного ответа (расширение
#: исключения листингов канона §4.2.6, черновик 3.4.0): сколько элементов
#: не влезло, их id (до 8) и как дочитать.
HINT_TRUNCATED = (
    "truncated: +{n} did not fit the {chars}-char budget ({ids}) — "
    "read point-wise by id"
)

#: Сколько id не влезших элементов показываем в hint (остальное — «+k»).
HINT_IDS_MAX = 8


def item_chars(item: dict[str, Any]) -> int:
    """Размер элемента в символах — как он уйдёт в JSON выдачи."""
    return len(json.dumps(item, ensure_ascii=False, separators=(",", ":")))


def header_of(item: dict[str, Any]) -> dict[str, Any]:
    """Шапка-проекция элемента: только идентичность (id, title, namespace)."""
    return {name: item[name] for name in HEADER_FIELDS if name in item}


def fit_within_budget(
    items: Sequence[dict[str, Any]], budget_chars: int
) -> tuple[list[dict[str, Any]], list[Any], bool]:
    """Разложить элементы под бюджет символов (двухуровневое чтение, п.2/п.4).

    Элемент влезает целиком — отдаётся как есть. Не влезает — деградирует до
    шапки с признаком `"truncated": true` (шапка влезает — отдаём шапку,
    агент знает id; не влезает даже шапка — пропускаем). Возвращает
    `(fitted, cut_ids, any_truncated)`: отданные элементы, id всех, кто не
    получен целиком, и был ли случай деградации.
    """
    fitted: list[dict[str, Any]] = []
    cut_ids: list[Any] = []
    used = 0
    any_truncated = False
    for item in items:
        cost = item_chars(item)
        if used + cost <= budget_chars:
            fitted.append(dict(item))
            used += cost
            continue
        cut_ids.append(item.get("id"))
        header = header_of(item) | {"truncated": True}
        header_cost = item_chars(header)
        if used + header_cost <= budget_chars:
            fitted.append(header)
            used += header_cost
            any_truncated = True
    return fitted, cut_ids, any_truncated


def truncation_hint(cut_ids: Sequence[Any], budget_chars: int) -> str:
    """Hint усечения: сколько не влезло и какие id (до `HINT_IDS_MAX`)."""
    shown = [str(note_id) for note_id in cut_ids[:HINT_IDS_MAX]]
    hidden = len(cut_ids) - len(shown)
    if hidden > 0:
        shown.append(f"+{hidden}")
    return HINT_TRUNCATED.format(
        n=len(cut_ids), chars=budget_chars, ids=", ".join(shown)
    )
