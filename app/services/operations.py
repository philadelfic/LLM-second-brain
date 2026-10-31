"""The LLM operations registry (LSBN-F-34, 3.4.0): a model per process.

    Historically the model was chosen by the slot group (canon 4.2.8): the
    `summary` slot served `summarize`, `title` and `merge`; the `judge` slot - the
    dedup verdict and the structure judge. Changing the model for one process
    only (for example, only for merging duplicates - the live T-23 case) was
    impossible. The registry cuts slot groups into OPERATIONS: a flat env
    override `<OP>_<FIELD>` (no `OP_` prefix) acts over the slot env.

    The resolution chain per field: operation env -> slot env -> default (the
    existing Settings fields). `None` at an operation field = "not set" - the slot
    is inherited, so a partial override (only `MERGE_MODEL`) is the norm.
    Operations without overrides behave exactly as before.

    The boundaries (the brief decisions):
    - the `SPLIT_*` slot as a separate slot group stays retired - the registry
      does not restore it; `split` remains an OPERATION of the summary slot
      (registered ahead of use, like `describe` - not called today);
    - an operation override acts on the primary (generative) path only; the
      reserve slot (decision, the F-12 mechanics) is senior: switching to the
      reserve takes the `DECISION_*` settings with its model;
    - secrets (`*_API_KEY`) are overridden by the operation env but never reach
      the config fingerprint or the outer surfaces (/health).

    The module is pure: it knows only `app.config.Settings`, no clients, DB or
    network - the configuration resolution is testable without booting the app.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

from pydantic_core import PydanticUndefined

from app.services.slot_budget import resolve_budgets

if TYPE_CHECKING:  # annotations only - the module stays pure
    from app.config import Settings

# The "operation name -> slot group" registry (LSBN-F-34, brief decision No1). The order
# is part of the contract (/health.operations lists operations in this order).
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

# The override fields (the brief's decision No1): the flat env `<OP>_<FIELD>`.
# The order is part of the contract (/health and the tests list the sources this way).
# LSBN-T-33: `reasoning_num_predict` - the operation's REASONING budget (apart from
# the answer budget `num_predict`); its slot layer is one for all slots - the global
# `slot_reasoning_num_predict` (see resolve_operation).
OPERATION_FIELDS: tuple[str, ...] = (
    "model",
    "provider",
    "base_url",
    "api_key",
    "think",
    "num_predict",
    "reasoning_num_predict",
    "timeout_sec",
    "context_tokens",
    "model_type",
)

# An operation field -> a SlotSpec attribute (llm_client.SlotSpec): the operation timeout
# lands in the client read timeout; the rest map by name.
SPEC_FIELDS: dict[str, str] = {
    "model": "model",
    "provider": "provider",
    "base_url": "base_url",
    "api_key": "api_key",
    "think": "think",
    "num_predict": "num_predict",
    "reasoning_num_predict": "reasoning_num_predict",
    "timeout_sec": "read_timeout",
    "context_tokens": "context_tokens",
    "model_type": "model_type",
}

# The code defaults of the operation fields (the lowest chain layer - BEFORE the SlotSpec default):
# the node description historically ran with an explicit budget of 128 and no
# reasoning (promotion.py, DESCRIBE_NUM_PREDICT + think=False) - the defaults
# move here so DESCRIBE_NUM_PREDICT/DESCRIBE_THINK and slot inheritance
# work through the shared chain.
OPERATION_FIELD_DEFAULTS: dict[tuple[str, str], object] = {
    ("describe", "num_predict"): 512,
    ("describe", "think"): False,
}

# The known top-level env prefixes (for the WARN about an unknown operation
# name): the registry operations + the slot groups + the shared slot parameters.
KNOWN_ENV_PREFIXES: frozenset[str] = frozenset(OPERATIONS) | {
    "embedding",
    "summary",
    "judge",
    "decision",
}

# The "flat operator" env mask: `<PREFIX>_<FIELD>` over the registry fields.
_OPERATION_ENV_RE = re.compile(
    r"^([A-Z][A-Z0-9_]*)_(" + "|".join(field.upper() for field in OPERATION_FIELDS) + r")$"
)


def env_name(operation: str, field: str) -> str:
    """The operation's flat env name: `<OP>_<FIELD>` (the brief's decision No1)."""
    return f"{operation.upper()}_{field.upper()}"


def _settings_field(operation: str, field: str) -> str:
    """The Settings field name of the operation's flat env (pydantic reads env itself)."""
    return f"{operation}_{field}"


def _slot_settings_field(slot: str, field: str) -> str:
    """The Settings field name of a slot group (provider/model/... per the 4.2.8 canon)."""
    return f"{slot}_{field}"


def _source_of(settings: "Settings", settings_field: str) -> str:
    """The value source of the slot layer: `slot` (set by the operator) | `default`.

            Required fields (`SUMMARY_MODEL` etc., no default) are always `slot`;
            optional ones compare against the declared Settings default: equal to the
            default -> `default`, different -> `slot`. Fields a slot does not have
            (for example, `embedding_think`) - `default`.

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
    """The operation fields' effective values and each one's source: (values, sources).

            values - ONLY the fields set by the flat operation env (the `op` layer);
            sources - the full coverage of OPERATION_FIELDS: `op` | `slot` |
            `default`. An operation whose name equals the slot group (`judge`) has no
            separate layer - its env IS the slot env (`JUDGE_*`), the sources are
            honestly the slot layer's. `merge_num_predict` (a historical field,
            config.py) is already an operation env: a value of 0 ("auto") counts as
            SET by the operation - the behavior unchanged (slot inheritance for merge
            stays off).
            The exception - `reasoning_num_predict` (LSBN-T-33): it has no slot-group
            fields, so the operation layer is read for `judge` too
            (`JUDGE_REASONING_NUM_PREDICT`), and the slot layer is the global
            `slot_reasoning_num_predict`.

    """
    slot = OPERATIONS.get(operation)
    if slot is None:
        raise ValueError(f"неизвестная операция: {operation}")
    sources: dict[str, str] = {}
    values: dict[str, object] = {}
    for field in OPERATION_FIELDS:
        if operation != slot or field == "reasoning_num_predict":
            value = getattr(settings, _settings_field(operation, field), None)
            if value is not None:
                values[field] = value
                sources[field] = "op"
                continue
        slot_field = (
            "slot_reasoning_num_predict"
            if field == "reasoning_num_predict"
            else _slot_settings_field(slot, field)
        )
        if type(settings).model_fields.get(slot_field) is not None:
            sources[field] = _source_of(settings, slot_field)
        else:
            sources[field] = "default"
    return values, sources


def operation_overrides_active(settings: "Settings") -> bool:
    """At least one operation is overridden by flat env (the `op` layer)."""
    return any(resolve_operation(operation, settings)[1].get("op") for operation in OPERATIONS)  # noqa: E501


def _operation_budgets(operation: str, settings: "Settings") -> dict[str, object]:
    """The operation budgets for /health.operations (LSBN-T-33, the brief 3.7).

            The effective field values are assembled by the same chain as
            `for_operation` (operation env -> slot layer -> default), then the split
            budget is computed (`slot_budget.resolve_budgets`, a pure function - no
            import cycle). `prompt_tokens=0` - this is the operation's CONFIGURED
            ceiling (the maximum), not a specific call's budget: the actual budgets
            live in `ChatMeta` and the journal (`slot_call`). `think=None`/True -
            thinking allowed (the reserve counts); False - not (sent = the answer).
            An unset operation's effective think - the slot group's flag.

    """
    slot = OPERATIONS[operation]
    values, _sources = resolve_operation(operation, settings)

    def effective(field: str, fallback: object) -> object:
        if field in values:
            return values[field]
        slot_field = (
            "slot_reasoning_num_predict"
            if field == "reasoning_num_predict"
            else _slot_settings_field(slot, field)
        )
        if type(settings).model_fields.get(slot_field) is not None:
            return getattr(settings, slot_field)
        return fallback

    budgets = resolve_budgets(
        operation=operation,
        prompt_tokens=0,
        explicit_answer=int(effective("num_predict", 0) or 0),  # type: ignore[arg-type]
        explicit_reasoning=effective("reasoning_num_predict", None),  # type: ignore[arg-type]
        context_tokens=int(effective("context_tokens", 0) or 0),  # type: ignore[arg-type]
        reserve_tokens=settings.slot_prompt_reserve_tokens,
        defensive_cap=settings.slot_max_output_tokens,
        think_enabled=effective("think", None) is not False,
    )
    return {
        "answer_budget": budgets.answer_budget,
        "reasoning_budget": budgets.reasoning_budget,
        "sent_budget": budgets.sent_budget,
        "answer_source": budgets.answer_source,
        "reasoning_source": budgets.reasoning_source,
    }


def operations_health(settings: "Settings") -> dict[str, dict[str, object]]:
    """The /health.operations snapshot: {operation: {model, provider, source, budgets}}.

            `source` - the source of the operation's MODEL (`op` | `slot` |
            `default`); with LSBN-T-33 the budgets sit next to it
            (`answer_budget`/`reasoning_budget`/`sent_budget` + their sources) - the
            operation's configured ceiling with an empty prompt. Addresses and keys
            are deliberately absent (the brief's decision No4): /health is open
            without Bearer; the address lives in the journal (`slot_call`) and the job
            error event.

    """
    block: dict[str, dict[str, object]] = {}
    for operation in OPERATIONS:
        _values, sources = resolve_operation(operation, settings)
        entry: dict[str, object] = {
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
        entry.update(_operation_budgets(operation, settings))
        block[operation] = entry
    return block


def unknown_operation_warnings() -> list[tuple[str, str]]:
    """The WARN about env with an unknown operation name (the brief's decision No4).

            Scans the environment against the `<PREFIX>_<FIELD>` mask (the registry's
            fields); a prefix outside the known operations/slot groups is a typo
            candidate (`MERG_MODEL` instead of `MERGE_MODEL`). Returns
            `(code, EN text)` - like `LLMClient.slot_config_warnings`; the caller
            (main.py) logs; the service starts, the warning is not fatal.

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
