"""Политика бюджета вывода слота «сколько влезет» (№49, релиз 3.2.0; arch §6.1, §13.2).

Чистые функции без импорта домена, БД и сети: токены промпта уже посчитаны
вызывающим (клиент слотов — существующий сплиттер `splitter.count_tokens`),
здесь остаётся только арифметика бюджета и его источник. Один источник истины
на все слоты: явный операторский бюджет не зажимается, при заявленном контексте
бюджет = «сколько влезет» минус промпт и запас, зажатый защитным потолком; при
неизвестном контексте — прежние потолки v3.1.0 без падения (деградация).

Источник бюджета (`explicit`/`context`/`legacy`) уходит в `ChatMeta` и в журнал
(`slot_call`) — оператор видит, почему у вызова именно такой бюджет (§13.2).

Тут же живёт **отпечаток конфигурации слота** (`slot_config_fingerprint`,
lsbdef0008-02, arch §5/§13.8): состав конфигурации, смена которого на старте
снимает «требует вмешательства» у заданий (механизм внимания — блок 2, здесь
только состав отпечатка).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # только для аннотации — модуль остаётся чистым (без домена)
    from app.config import Settings

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


# Состав отпечатка конфигурации слота (lsbdef0008-02, arch §5/§13.8): порядок
# слагаемых — часть контракта, значения приводятся к строкам. Провайдер, модель,
# тип модели, бюджет вывода, заявленный контекст, запас и защитный потолок, плюс
# параметры decision-слота: поднятый бюджет и включённый decision — тоже «причина
# ушла» (FR-2.3), поэтому входят в отпечаток наравне с моделью.
FINGERPRINT_FIELDS: tuple[str, ...] = (
    "summary_model",
    "summary_provider",
    "summary_model_type",
    "summary_num_predict",
    "summary_context_tokens",
    "slot_max_output_tokens",
    "slot_prompt_reserve_tokens",
    "judge_model",
    "judge_provider",
    "judge_model_type",
    "judge_num_predict",
    "judge_context_tokens",
    "decision_enabled",
    "decision_model",
    "decision_threshold",
    "decision_uncertainty_zone",
)

# Дефолты полей decision-слота (arch §8). Сам слот приезжает блоком F-12
# (`DECISION_ENABLED|DECISION_MODEL|DECISION_THRESHOLD|DECISION_UNCERTAINTY_ZONE`),
# поэтому до него поля читаются `getattr`-ом с этими значениями: отпечаток уже
# включает decision, а как только поля появятся в `Settings`, подхватит их без
# правки. Секретов в составе нет (`*_API_KEY` не входят — отпечаток живёт в
# `meta` и попадает в события журнала).
FINGERPRINT_DEFAULTS: dict[str, object] = {
    "decision_enabled": False,
    "decision_model": "",
    "decision_threshold": 0.5,
    "decision_uncertainty_zone": 0.05,
}


def slot_config_fingerprint(settings: Settings) -> str:
    """Детерминированный отпечаток конфигурации слотов (lsbdef0008-02).

    Строка из значений `FINGERPRINT_FIELDS` в фиксированном порядке — сравнимая
    и записываемая в `meta` (ключ `slot_config_fingerprint`, блок 2). Отпечаток
    отсутствует или отличается от текущего на старте → внимание у заданий всех
    слотов снимается (`attention_at=NULL, attempts=0`), то есть правка
    конфигурации слота сама возвращает «безнадёжные» задания в работу.
    """
    return "|".join(
        str(getattr(settings, name, FINGERPRINT_DEFAULTS.get(name, "")))
        for name in FINGERPRINT_FIELDS
    )
