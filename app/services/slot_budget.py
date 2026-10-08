"""Политика бюджета вывода слота «сколько влезет» (№49, релиз 3.2.0; arch §6.1, §13.2).

Чистые функции без импорта домена, БД и сети: токены промпта уже посчитаны
вызывающим (клиент слотов — существующий сплиттер `splitter.count_tokens`),
здесь остаётся только арифметика бюджета и его источник. Один источник истины
на все слоты: явный операторский бюджет не зажимается, при заявленном контексте
бюджет = «сколько влезет» минус промпт и запас, зажатый защитным потолком; при
неизвестном контексте — прежние потолки v3.1.0 без падения (деградация).

Источник бюджета (`explicit`/`context`/`legacy`) уходит в `ChatMeta` и в журнал
(`slot_call`) — оператор видит, почему у вызова именно такой бюджет (§13.2).
"""

from __future__ import annotations

# Прежние потолки v3.1.0: остаются ОПЕРАТОРСКИМИ значениями env, а в коде живут
# как деградация при неизвестном контексте (== поведение v3.1.0). "title" —
# title-догенерация разделяет потолок выжимки; "classify" — бюджет JSON-разметки.
LEGACY_OUTPUT_BUDGET: dict[str, int] = {
    "summary": 35000,
    "merge": 35000,
    "title": 35000,
    "judge": 256,
    "classify": 768,
}

# Неизвестная операция (имя слота без операции, новая операция без потолка):
# деградация к общему прежнему потолку — вызов не роняется.
LEGACY_DEFAULT_OUTPUT_BUDGET = 35000

# Источники бюджета (arch §6.1): воля оператора / «сколько влезет» / деградация.
SOURCE_EXPLICIT = "explicit"
SOURCE_CONTEXT = "context"
SOURCE_LEGACY = "legacy"


def resolve_output_budget(
    *,
    operation: str,
    prompt_tokens: int,
    explicit: int | None,
    context_tokens: int,
    reserve_tokens: int,
    defensive_cap: int,
) -> tuple[int, str]:
    """Бюджет вывода и его источник: (`explicit` | `context` | `legacy`).

    Порядок решений обязателен: явный операторский бюджет (`explicit > 0`) —
    как есть и не зажимается; иначе при заявленном контексте — «сколько влезет»
    (`context − prompt − reserve`), зажатое защитным потолком и никогда не ниже
    `1` (промпт длиннее контекста не даёт отрицательный бюджет); иначе —
    прежний потолок `LEGACY_OUTPUT_BUDGET[operation]` (деградация к v3.1.0).
    """
    if explicit is not None and explicit > 0:
        return explicit, SOURCE_EXPLICIT
    if context_tokens > 0:
        budget = min(context_tokens - prompt_tokens - reserve_tokens, defensive_cap)
        return max(1, budget), SOURCE_CONTEXT
    return LEGACY_OUTPUT_BUDGET.get(operation, LEGACY_DEFAULT_OUTPUT_BUDGET), SOURCE_LEGACY
