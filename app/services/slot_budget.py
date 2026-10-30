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

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  # только для аннотации — модуль остаётся чистым (без домена)
    from app.config import Settings

# Прежние потолки v3.1.0: остаются ОПЕРАТОРСКИМИ значениями env, а в коде живут
# как деградация при неизвестном контексте (== поведение v3.1.0). "title" —
# title-догенерация разделяет потолок выжимки; "classify" — бюджет JSON-разметки.
LEGACY_OUTPUT_BUDGET: dict[str, int] = {
    "summary": 35000,
    # LSBN-F-34: реестровые имена операций (цепочка env операции → слота →
    # дефолт); деградация при неизвестном контексте — прежние потолки.
    "summarize": 35000,
    "merge": 35000,
    "title": 35000,
    "judge": 256,
    "structure_judge": 256,
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


class Budgets(NamedTuple):
    """Раздельные бюджеты одного вызова (LSBN-T-33, arch §6.1).

    `answer_budget` — гарантированная заявка на полезный ответ; `reasoning_budget`
    — потолок рассуждения (0 — не резервируется); `sent_budget` — что реально
    уходит в `num_predict`/`max_tokens` (у транспорта лимит СУММАРНЫЙ:
    thinking + content, поэтому разделение обеспечивает сторона LSB, а не провод).
    Транспорт ограничивает сумму, поэтому «ответ не меньше `answer_budget`»
    на стороне запроса не гарантируется — это заявка; фактическая проверка —
    по ответу (пустой content при непустом рассуждении — бюджет съеден).
    """

    answer_budget: int
    reasoning_budget: int
    sent_budget: int
    answer_source: str
    reasoning_source: str


def resolve_budgets(
    *,
    operation: str,
    prompt_tokens: int,
    explicit_answer: int | None,
    explicit_reasoning: int | None,
    context_tokens: int,
    reserve_tokens: int,
    defensive_cap: int,
    think_enabled: bool,
) -> Budgets:
    """Раздельные бюджеты ответа и рассуждения (LSBN-T-33, arch §6.1).

    Порядок решений обязателен, деградация при неизвестном контексте — как в
    `resolve_output_budget`:
    - `answer_budget` — политика ответа прежняя (`resolve_output_budget`):
      явный бюджет → «сколько влезет» из контекста → прежний потолок;
    - `reasoning_budget` — явный операторский (`explicit_reasoning > 0`); иначе
      при известном контексте — остаток окна после ответа (зажат защитным
      потолком, не ниже 0); иначе 0 (деградация: рассуждение не резервируется,
      поведение 3.4.1);
    - `sent_budget` — при выключенном размышлении (`think_enabled=False`) ровно
      ответ; иначе сумма ответ+рассуждение, зажатая окном контекста и защитным
      потолком; при неизвестном контексте — только ответ (деградация).

    Важно: рассуждение не уменьшает заявку на ответ — `answer_budget` входит в
    `sent_budget` первым слагаемым, а рассуждение занимает остаток окна.
    """
    answer_budget, answer_source = resolve_output_budget(
        operation=operation,
        prompt_tokens=prompt_tokens,
        explicit=explicit_answer,
        context_tokens=context_tokens,
        reserve_tokens=reserve_tokens,
        defensive_cap=defensive_cap,
    )
    window = context_tokens - prompt_tokens - reserve_tokens
    if explicit_reasoning is not None and explicit_reasoning > 0:
        reasoning_budget, reasoning_source = explicit_reasoning, SOURCE_EXPLICIT
    elif context_tokens > 0:
        reasoning_budget = max(0, min(window - answer_budget, defensive_cap))
        reasoning_source = SOURCE_CONTEXT
    else:
        reasoning_budget, reasoning_source = 0, SOURCE_LEGACY
    if not think_enabled or context_tokens <= 0:
        sent_budget = answer_budget
    else:
        sent_budget = max(
            1, min(answer_budget + reasoning_budget, window, defensive_cap)
        )
    return Budgets(
        answer_budget=answer_budget,
        reasoning_budget=reasoning_budget,
        sent_budget=sent_budget,
        answer_source=answer_source,
        reasoning_source=reasoning_source,
    )


# Состав отпечатка конфигурации слота (lsbdef0008-02/-03, arch §5/§8/§13.8):
# порядок слагаемых — часть контракта, значения приводятся к строкам. Провайдер,
# модель, тип модели, «думать / не думать» (`*_THINK`), бюджет вывода, заявленный
# контекст, клиентский таймаут (`*_TIMEOUT_SEC`), запас и защитный потолок, плюс
# параметры decision-слота: поднятый бюджет, включённый decision, правка
# размышления или таймаута — тоже «причина ушла» (FR-2.3), поэтому входят в
# отпечаток наравне с моделью. THINK и таймаут здесь не для полноты картинки:
# они меняют исход вызова (`think: false` — другой ответ; таймаут — `timeout`
# вместо ответа), а боевой случай 30.09–01.10.2026 показал цену их отсутствия —
# слоты на обходе `JUDGE_THINK=false` + малый бюджет не размораживали пары после
# правки размышления.
FINGERPRINT_FIELDS: tuple[str, ...] = (
    "summary_model",
    "summary_provider",
    "summary_model_type",
    "summary_think",
    "summary_num_predict",
    "summary_context_tokens",
    "summary_timeout_sec",
    "slot_max_output_tokens",
    "slot_prompt_reserve_tokens",
    # LSBN-T-33: бюджет рассуждения (глобальный слот-уровень) — правка
    # меняет исход вызова (sent_budget) так же, как `*_num_predict`.
    "slot_reasoning_num_predict",
    "judge_model",
    "judge_provider",
    "judge_model_type",
    "judge_think",
    "judge_num_predict",
    "judge_context_tokens",
    "judge_timeout_sec",
    "judge_reasoning_num_predict",
    # LSBN-F-34: плоские поля операций (цепочка env операции → слота →
    # дефолт) — правка MERGE_MODEL/TITLE_TIMEOUT_SEC и т.п. тоже «причина
    # ушла»: без них правка конфигурации операции не снимала бы «требует
    # вмешательства» у заданий. API_KEY и BASE_URL не входят: секреты —
    # никогда; адрес слота и раньше не входил (симметрично слотам).
    "summarize_model",
    "summarize_provider",
    "summarize_model_type",
    "summarize_think",
    "summarize_num_predict",
    "summarize_context_tokens",
    "summarize_timeout_sec",
    "summarize_reasoning_num_predict",
    "title_model",
    "title_provider",
    "title_model_type",
    "title_think",
    "title_num_predict",
    "title_context_tokens",
    "title_timeout_sec",
    "title_reasoning_num_predict",
    "merge_model",
    "merge_provider",
    "merge_model_type",
    "merge_think",
    "merge_num_predict",
    "merge_context_tokens",
    "merge_timeout_sec",
    "merge_reasoning_num_predict",
    "split_model",
    "split_provider",
    "split_model_type",
    "split_think",
    "split_num_predict",
    "split_context_tokens",
    "split_timeout_sec",
    "split_reasoning_num_predict",
    "classify_model",
    "classify_provider",
    "classify_model_type",
    "classify_think",
    "classify_num_predict",
    "classify_context_tokens",
    "classify_timeout_sec",
    "classify_reasoning_num_predict",
    "describe_model",
    "describe_provider",
    "describe_model_type",
    "describe_think",
    "describe_num_predict",
    "describe_context_tokens",
    "describe_timeout_sec",
    "describe_reasoning_num_predict",
    "structure_judge_model",
    "structure_judge_provider",
    "structure_judge_model_type",
    "structure_judge_think",
    "structure_judge_num_predict",
    "structure_judge_context_tokens",
    "structure_judge_timeout_sec",
    "structure_judge_reasoning_num_predict",
    "embed_model",
    "embed_provider",
    "embed_timeout_sec",
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
