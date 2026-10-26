"""Реестр операций LLM (LSBN-F-34, релиз 3.4.0): модель под каждый процесс.

Исторически модель выбиралась слотом-группой (канон 4.2.8): слот `summary`
обслуживал `summarize`, `title` и `merge`, слот `judge` — вердикт дедупа и
судью структуры. Поменять модель только для одного процесса (например,
только для сшивания дублей — живой пример T-23) было нельзя. Реестр режет
слоты-группы на **операции**: плоское env-переопределение `<ОПЕРАЦИЯ>_<ПОЛЕ>`
(без префикса `OP_`, решение О. 21.10.2026) действует поверх env слота.

Цепочка разрешения по каждому полю: **env операции → env слота → дефолт**
(существующие поля Settings). `None` у поля операции = «не задано» —
наследуется слот, поэтому частичное переопределение (только `MERGE_MODEL`)
— норма. Операции без переопределения ведут себя ровно как раньше.

Границы (решения брифа 21.10.2026):
- слот `SPLIT_*` как отдельная слот-группа отменён — реестр не восстанавливает
  его; `split` остаётся ОПЕРАЦИЕЙ слота summary (зарегистрирована впрок, как
  и `describe` — сегодня не вызывается);
- переопределение операции действует только на основном (генеративном) пути;
  резервный слот (decision, механика F-12) главнее: при переключении на резерв
  берутся настройки `DECISION_*` вместе с его моделью;
- секреты (`*_API_KEY`) переопределяются env операции, но в отпечаток
  конфигурации и наружные поверхности (/health) не попадают.

Модуль чистый: знает только `app.config.Settings`, без клиентов, БД и сети —
разрешение конфигурации тестируется без поднятия приложения.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

from pydantic_core import PydanticUndefined

if TYPE_CHECKING:  # только для аннотаций — модуль остаётся чистым
    from app.config import Settings

# Реестр «имя операции → слот-группа» (LSBN-F-34, решение №1 брифа). Порядок
# — часть контракта (/health.operations перечисляет операции в этом порядке).
OPERATIONS: dict[str, str] = {
    "summarize": "summary",
    "title": "summary",
    "merge": "summary",
    "split": "summary",
    "classify": "summary",
    "describe": "summary",
    "judge": "judge",
    "structure_judge": "judge",
    "embed": "embedding",
}

# Поля переопределения (решение №1 брифа): плоский env `<ОПЕРАЦИЯ>_<ПОЛЕ>`.
# Порядок — часть контракта (источники в /health и тестах перечисляют так).
OPERATION_FIELDS: tuple[str, ...] = (
    "model",
    "provider",
    "base_url",
    "api_key",
    "think",
    "num_predict",
    "timeout_sec",
    "context_tokens",
    "model_type",
)

# Поле операции → атрибут SlotSpec (llm_client.SlotSpec): таймаут операции
# ложится в клиентский read-таймаут, остальные — одноимённо.
SPEC_FIELDS: dict[str, str] = {
    "model": "model",
    "provider": "provider",
    "base_url": "base_url",
    "api_key": "api_key",
    "think": "think",
    "num_predict": "num_predict",
    "timeout_sec": "read_timeout",
    "context_tokens": "context_tokens",
    "model_type": "model_type",
}

# Кодовые дефолты полей операций (низший слой цепочки — ДО дефолта SlotSpec):
# описание узла исторически шло с явным бюджетом 128 и без рассуждений
# (promotion.py, DESCRIBE_NUM_PREDICT + think=False) — дефолты переезжают
# сюда, чтобы DESCRIBE_NUM_PREDICT/DESCRIBE_THINK и наследование слота
# работали по общей цепочке.
OPERATION_FIELD_DEFAULTS: dict[tuple[str, str], object] = {
    ("describe", "num_predict"): 512,
    ("describe", "think"): False,
}

# Известные env-префиксы верхнего уровня (для WARN о неизвестном имени
# операции): операции реестра + слот-группы + общие слот-параметры.
KNOWN_ENV_PREFIXES: frozenset[str] = frozenset(OPERATIONS) | {
    "embedding",
    "summary",
    "judge",
    "decision",
}

# Маска «плоского операционного» env: `<ПРЕФИКС>_<ПОЛЕ>` по полям реестра.
_OPERATION_ENV_RE = re.compile(
    r"^([A-Z][A-Z0-9_]*)_(" + "|".join(field.upper() for field in OPERATION_FIELDS) + r")$"
)


def env_name(operation: str, field: str) -> str:
    """Плоское имя env операции: `<ОПЕРАЦИЯ>_<ПОЛЕ>` (решение №1 брифа)."""
    return f"{operation.upper()}_{field.upper()}"


def _settings_field(operation: str, field: str) -> str:
    """Имя Settings-поля плоского env операции (pydantic читает env сам)."""
    return f"{operation}_{field}"


def _slot_settings_field(slot: str, field: str) -> str:
    """Имя Settings-поля слот-группы (provider/model/... per-slot канона 4.2.8)."""
    return f"{slot}_{field}"


def _source_of(settings: "Settings", settings_field: str) -> str:
    """Источник значения слот-слоя: `slot` (задано оператором) | `default`.

    Обязательные поля (`SUMMARY_MODEL` и т.п., без умолчания) — всегда
    `slot`; опциональные сравниваются с объявленным дефолтом Settings:
    значение равно дефолту → `default`, отличается → `slot`. Поля, которых
    у слота нет (например, `embedding_think`), — `default`.
    """
    field_info = type(settings).model_fields.get(settings_field)
    if field_info is None:
        return "default"
    value = getattr(settings, settings_field)
    if field_info.default is PydanticUndefined or field_info.default != value:
        return "slot"
    return "default"


def resolve_operation(
    operation: str, settings: "Settings"
) -> tuple[dict[str, object], dict[str, str]]:
    """Эффективные значения полей операции и источник каждого: (values, sources).

    values — ТОЛЬКО поля, заданные плоским env операции (слой `op`); sources —
    полное покрытие OPERATION_FIELDS: `op` | `slot` | `default`. Операция,
    чьё имя совпадает со слот-группой (`judge`), отдельного слоя не имеет —
    её env и есть env слота (`JUDGE_*`), источники честно слот-слоя.
    `merge_num_predict` (историческое поле, config.py) — уже операционный
    env: значение 0 («авто») считается ЗАДАННЫМ операцией — поведение
    прежнее (наследование слота для merge не включается).
    """
    slot = OPERATIONS.get(operation)
    if slot is None:
        raise ValueError(f"неизвестная операция: {operation}")
    sources: dict[str, str] = {}
    values: dict[str, object] = {}
    for field in OPERATION_FIELDS:
        if operation != slot:
            value = getattr(settings, _settings_field(operation, field), None)
            if value is not None:
                values[field] = value
                sources[field] = "op"
                continue
        slot_field = _slot_settings_field(slot, field)
        if type(settings).model_fields.get(slot_field) is not None:
            sources[field] = _source_of(settings, slot_field)
        else:
            sources[field] = "default"
    return values, sources


def operation_overrides_active(settings: "Settings") -> bool:
    """Хотя бы одна операция переопределена плоским env (слой `op`)."""
    return any(resolve_operation(operation, settings)[1].get("op") for operation in OPERATIONS)  # noqa: E501


def operations_health(settings: "Settings") -> dict[str, dict[str, str]]:
    """Снимок для /health.operations: {операция: {model, provider, source}}.

    `source` — источник МОДЕЛИ операции (`op` | `slot` | `default`). Адресов
    и ключей здесь нет сознательно (решение №4 брифа): /health открыт наружу
    без Bearer, адрес живёт в журнале (`slot_call`) и в событии джобы при
    ошибке.
    """
    block: dict[str, dict[str, str]] = {}
    for operation in OPERATIONS:
        _values, sources = resolve_operation(operation, settings)
        block[operation] = {
            "model": str(
                getattr(settings, _settings_field(operation, "model"), None)
                or getattr(settings, _slot_settings_field(OPERATIONS[operation], "model"))
            ),
            "provider": str(
                getattr(settings, _settings_field(operation, "provider"), None)
                or getattr(
                    settings, _slot_settings_field(OPERATIONS[operation], "provider")
                )
            ),
            "source": sources["model"],
        }
    return block


def unknown_operation_warnings() -> list[tuple[str, str]]:
    """WARN об env с неизвестным именем операции (решение №4 брифа).

    Сканирует окружение по маске `<ПРЕФИКС>_<ПОЛЕ>` (поля — реестра);
    префикс вне известных операций/слот-групп — кандидат на опечатку
    (`MERG_MODEL` вместо `MERGE_MODEL`). Возвращает `(code, EN-текст)` —
    как `LLMClient.slot_config_warnings`, логирует вызывающий (main.py);
    сервис стартует, предупреждение не фатально.
    """
    warnings: list[tuple[str, str]] = []
    for name in sorted(os.environ):
        match = _OPERATION_ENV_RE.match(name)
        if match is None:
            continue
        prefix = match.group(1).lower()
        if prefix in KNOWN_ENV_PREFIXES:
            continue
        warnings.append(
            (
                "unknown_operation",
                f"unknown operation env {name} — operation names are: "
                + ", ".join(sorted(OPERATIONS))
                + "; the variable is ignored (typo? check the registry in "
                "docs/CONFIG.md).",
            )
        )
    return warnings
