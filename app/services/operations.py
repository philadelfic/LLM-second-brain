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

    Release 3.9.0 (LSBN-S-5): the OPERATION REGISTRY - `OperationSpec` +
    `OPERATION_REGISTRY` (ten operations with their contract) and the mode flag
    `SLOT_MODEL` (`legacy` | `operation`). In the `operation` mode the slot
    layer does not take part in the chain ("operation env -> default") and the
    breaker of an operation is its own. The `legacy` mode keeps the historical
    behaviour in every respect - the flag is the ONE switch (the design, B2).

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
from dataclasses import dataclass
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

# --- the operation registry (release 3.9.0, LSBN-S-5) ----------------------
# The contracts of the operations (the design, B7): the CLOSED set of the
# answer shapes. A contract is a fact about the code (the shape of the answer
# the operation expects from the model), not an operator switch - there is no
# env for it.
CONTRACTS: frozenset[str] = frozenset(
    {"text", "marker", "label", "probability", "parts-by-delimiter"}
)

# The per-operation CONTROL fields (release 3.9.0): the thresholds of the
# operation and its enable switch. They are operation env names too, so the
# typo scanner must know them (`MERG_PAUSE_AFTER` is a typo like `MERG_MODEL`).
OPERATION_CONTROL_FIELDS: tuple[str, ...] = ("pause_after", "pause_sec", "enabled")

# The default provider of an operation (the code default of the slot layer:
# the native ollama API).
DEFAULT_PROVIDER = "ollama"

# The modes of the slot layer (the design, B2): `legacy` - the slot groups own
# the breakers and the env chain is "operation -> slot -> default"; `operation`
# - every operation owns its breaker and the chain is "operation -> default".
SLOT_MODEL_LEGACY = "legacy"
SLOT_MODEL_OPERATION = "operation"


@dataclass(frozen=True)
class OperationSpec:
    """One operation of the registry (release 3.9.0, LSBN-S-5).

            name - the operation name: the env prefix `<OP>` and the breaker
            key of the `operation` mode;
            contract - the shape of the answer the operation expects (B7): a
            fact about the code, not an operator switch;
            enabled - the code state of the operation (the operator's switch
            is the per-operation env `<OP>_ENABLED`);
            legacy_slot - the slot group the operation lives in today (the
            `legacy` mode);
            fields - the flat env fields of the operation (`<OP>_<FIELD>`);
            pause_after / pause_sec - the code defaults of the operation's
            thresholds: None = "not set" -> the global base applies (the
            owner's decision, 15.11.2026: base + override - a numeric default
            would always beat the base).

    """

    name: str
    contract: str
    enabled: bool
    legacy_slot: str | None
    fields: tuple[str, ...]
    pause_after: int | None = None
    pause_sec: int | None = None


# The registry of the ten operations (release 3.9.0, LSBN-S-5). The order is
# part of the contract (the health block lists the operations this way).
# `OPERATIONS` above stays the LEGACY layer (operation -> slot group) and is
# not rewritten: the `legacy` mode keeps reading it.
OPERATION_REGISTRY: dict[str, OperationSpec] = {
    "summarize": OperationSpec(
        name="summarize",
        contract="text",
        enabled=True,
        legacy_slot="summary",
        fields=OPERATION_FIELDS,
    ),
    "title": OperationSpec(
        name="title",
        contract="text",
        enabled=True,
        legacy_slot="summary",
        fields=OPERATION_FIELDS,
    ),
    "merge": OperationSpec(
        name="merge",
        contract="parts-by-delimiter",
        enabled=True,
        legacy_slot="summary",
        fields=OPERATION_FIELDS,
    ),
    "split": OperationSpec(
        name="split",
        contract="parts-by-delimiter",
        enabled=True,
        legacy_slot="summary",
        fields=OPERATION_FIELDS,
    ),
    "classify": OperationSpec(
        name="classify",
        contract="label",
        enabled=True,
        legacy_slot="summary",
        fields=OPERATION_FIELDS,
    ),
    "describe": OperationSpec(
        name="describe",
        contract="text",
        enabled=True,
        legacy_slot="summary",
        fields=OPERATION_FIELDS,
    ),
    "judge": OperationSpec(
        name="judge",
        contract="marker",
        enabled=True,
        legacy_slot="judge",
        fields=OPERATION_FIELDS,
    ),
    "structure_judge": OperationSpec(
        name="structure_judge",
        contract="marker",
        enabled=True,
        legacy_slot="judge",
        fields=OPERATION_FIELDS,
    ),
    "embed": OperationSpec(
        name="embed",
        # The embedding call returns a vector, not a text answer: there is no
        # JSON/marker/label/probability contract. `text` is the closest value
        # of the closed set - "no structured answer" (the contract of the
        # embedding operation is settled in a later stage of the release).
        contract="text",
        enabled=True,
        legacy_slot="embedding",
        fields=OPERATION_FIELDS,
    ),
    "decision": OperationSpec(
        name="decision",
        contract="probability",
        enabled=True,
        # The reserve decision slot (F-12) is its own slot group; the
        # operation has NO job queue (it answers questions).
        legacy_slot="decision",
        fields=OPERATION_FIELDS,
    ),
}

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

# The "flat operator" env mask: `<PREFIX>_<FIELD>` over the registry fields
# (the configuration fields of an operation + its control fields).
_OPERATION_ENV_RE = re.compile(
    r"^([A-Z][A-Z0-9_]*)_("
    + "|".join(
        field.upper() for field in OPERATION_FIELDS + OPERATION_CONTROL_FIELDS
    )
    + r")$"
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


def slot_model(settings: "Settings") -> str:
    """The active model of the slot layer: `legacy` | `operation`.

            THE switch of the release 3.9.0: every point that depends on the
            mode reads this one value (the design, B2) - the flag lives in
            `Settings` (`SLOT_MODEL`) and is read once (the settings are
            frozen).

    """
    return getattr(settings, "slot_model", SLOT_MODEL_LEGACY)


def operation_mode(settings: "Settings") -> bool:
    """Whether the operation model is active (`SLOT_MODEL=operation`)."""
    return slot_model(settings) == SLOT_MODEL_OPERATION


def breaker_key(operation: str, settings: "Settings") -> str:
    """The breaker dictionary key of an operation.

            `operation` mode - the operation name (one breaker per operation);
            `legacy` mode - the slot group of the operation (as before).

    """
    if operation_mode(settings):
        return operation
    slot = OPERATIONS.get(operation)
    if slot is not None:
        return slot
    spec = OPERATION_REGISTRY.get(operation)
    if spec is None:
        raise ValueError(f"неизвестная операция: {operation}")
    return spec.legacy_slot or operation


def pause_after(operation: str, settings: "Settings") -> int:
    """The breaker threshold of an operation: the per-operation value, else the
    global base `SLOT_PAUSE_AFTER` (base + override, the owner's decision of
    15.11.2026).

            The resolution is mode-agnostic: it answers "the threshold of THIS
            operation". The `legacy` mode applies the global base to the whole
            slot group (one breaker per group) - the per-operation value is
            read by the `operation` mode only.

    """
    spec = OPERATION_REGISTRY.get(operation)
    if spec is None:
        raise ValueError(f"неизвестная операция: {operation}")
    override = getattr(settings, f"{operation}_pause_after", None)
    if override is not None:
        return int(override)
    return int(settings.slot_pause_after)


def pause_sec(operation: str, settings: "Settings") -> int:
    """The pause of an operation: the per-operation value, else the global base
    `SLOT_PAUSE_SEC` (base + override, 15.11.2026). See `pause_after`."""
    spec = OPERATION_REGISTRY.get(operation)
    if spec is None:
        raise ValueError(f"неизвестная операция: {operation}")
    override = getattr(settings, f"{operation}_pause_sec", None)
    if override is not None:
        return int(override)
    return int(settings.slot_pause_sec)


def operation_enabled(operation: str, settings: "Settings") -> bool:
    """Whether an operation is enabled: the code state (`OperationSpec.enabled`)
    AND the operator's switch (`<OP>_ENABLED`).

            Unset switch - the operation is on. The `decision` operation reuses
            the F-12 field `DECISION_ENABLED` (the reserve decision slot, off
            by default) - there is no second env with that name.

    """
    spec = OPERATION_REGISTRY.get(operation)
    if spec is None:
        raise ValueError(f"неизвестная операция: {operation}")
    if operation == "decision":
        override: bool | None = bool(settings.decision_enabled)
    else:
        override = getattr(settings, f"{operation}_enabled", None)
    if override is None:
        return spec.enabled
    return bool(override) and spec.enabled


def resolve_operation(
    operation: str, settings: "Settings"
) -> tuple[dict[str, object], dict[str, str]]:
    """The operation fields' effective values and each one's source: (values, sources).

            The chain depends on the slot model (release 3.9.0, LSBN-S-5):
            `legacy` - operation env -> slot env -> default (as before);
            `operation` - operation env -> default (the slot layer does not
            take part).

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
    if operation_mode(settings):
        return _resolve_operation_mode(operation, settings)
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


def _resolve_operation_mode(
    operation: str, settings: "Settings"
) -> tuple[dict[str, object], dict[str, str]]:
    """The `operation` mode chain: the operation env over the code defaults.

            The slot group does NOT take part (the design, B1): a field the
            operation does not set has the source `default`. The global
            `SLOT_REASONING_NUM_PREDICT` stays the default layer of the
            reasoning budget (a global safeguard, the owner's decision Q4).

    """
    if operation not in OPERATION_REGISTRY:
        raise ValueError(f"неизвестная операция: {operation}")
    sources: dict[str, str] = {}
    values: dict[str, object] = {}
    for field in OPERATION_FIELDS:
        value = getattr(settings, _settings_field(operation, field), None)
        if value is not None:
            values[field] = value
            sources[field] = "op"
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
    slot = OPERATIONS.get(operation)
    values, _sources = resolve_operation(operation, settings)

    def effective(field: str, fallback: object) -> object:
        if field in values:
            return values[field]
        if operation_mode(settings):
            # The slot layer does not take part; only the global safeguard of
            # the reasoning budget stays as the default layer.
            if field == "reasoning_num_predict":
                return settings.slot_reasoning_num_predict
            return fallback
        slot_field = (
            "slot_reasoning_num_predict"
            if field == "reasoning_num_predict"
            else _slot_settings_field(slot or "", field)
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


def _settings_class() -> type["Settings"]:
    """The Settings class, imported lazily: the module stays config-free
    at import time (the settings type is an annotation above only)."""
    from app.config import Settings

    return Settings


def settings_env_names(settings: "Settings | None" = None) -> frozenset[str]:
    """The env names pydantic reads INTO Settings (upper-cased field names).

            3.8.1: the scanner must not mistake a *setting* for an operation name.
            The global `*_TIMEOUT_SEC` knobs (`BACKUP_SHIP_TIMEOUT_SEC`,
            `SHUTDOWN_TIMEOUT_SEC`, `SLOT_MIN_TIMEOUT_SEC`, ...) are Settings
            fields, but the flat `<PREFIX>_<FIELD>` mask reads them as
            `BACKUP_SHIP` + `TIMEOUT_SEC` - a legitimate setting looked like an
            unknown operation. Field names ARE the env names (see
            `Settings.model_config`: no `env_prefix`, no aliases), so the Settings
            field set is the authoritative list.

    """
    target = type(settings) if settings is not None else _settings_class()
    return frozenset(name.upper() for name in target.model_fields)


def unknown_operation_warnings(
    settings: "Settings | None" = None,
) -> list[tuple[str, str]]:
    """The WARN about env with an unknown operation name (the brief's decision No4).

            Scans the environment against the `<PREFIX>_<FIELD>` mask (the registry's
            fields); a prefix outside the known operations/slot groups is a typo
            candidate (`MERG_MODEL` instead of `MERGE_MODEL`). Returns
            `(code, EN text)` - like `LLMClient.slot_config_warnings`; the caller
            (main.py) logs; the service starts, the warning is not fatal.

            A name that is a Settings field is a SETTING, not an operation
            (3.8.1): it never warns, whatever the mask splits off it. Only
            a name unknown to both the settings and the registry stays a
            typo candidate.

    """
    known_settings = settings_env_names(settings)
    warnings: list[tuple[str, str]] = []
    for name in sorted(os.environ):
        match = _OPERATION_ENV_RE.match(name)
        if match is None:
            continue
        if name in known_settings:
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
