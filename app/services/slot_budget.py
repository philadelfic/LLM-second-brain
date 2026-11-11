"""The slot output budget policy "as much as fits" (No49, 3.2.0; arch §6.1, §13.2).

    Pure functions importing neither the domain, the DB nor the network: the prompt
    tokens are already counted by the caller (the slot client - the existing
    `splitter.count_tokens` splitter); only the budget arithmetic and its source
    remain here. One source of truth for all slots: the operator's explicit budget
    is never clamped; with a declared context the budget = "as much as fits" minus
    the prompt and the reserve, clamped by the defensive cap; with an unknown
    context - the previous v3.1.0 caps without failing (degradation).

    The budget source (`explicit`/`context`/`legacy`) goes into `ChatMeta` and the
    journal (`slot_call`) - the operator sees why a call got exactly this budget
    (§13.2).

    The **slot configuration fingerprint** also lives here
    (`slot_config_fingerprint`, lsbdef0008-02, arch §5/§13.8): the configuration
    composition whose change at startup clears "requires attention" from jobs
    (the attention mechanism - block 2; only the fingerprint composition here).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  # annotation only - the module stays pure (no domain)
    from app.config import Settings

# The former v3.1.0 caps: they remain OPERATOR env values, and in code they live
# as the degradation for an unknown context (== the v3.1.0 behavior). "title" -
# the title backfill shares the summary cap; "classify" - the JSON placement budget.
LEGACY_OUTPUT_BUDGET: dict[str, int] = {
    "summary": 35000,
    # LSBN-F-34: the registry operation names (the chain operation env -> slot ->
    # default); the unknown-context degradation - the previous caps.
    "summarize": 35000,
    "merge": 35000,
    "title": 35000,
    "judge": 256,
    "structure_judge": 256,
    "classify": 768,
}

# An unknown operation (a slot name without an operation, a new operation without a cap):
# the degradation to the shared previous cap - the call is not failed.
LEGACY_DEFAULT_OUTPUT_BUDGET = 35000

# The budget sources (arch §6.1): the operator's will / "as much as fits" / degradation.
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
    """The output budget and its source: (`explicit` | `context` | `legacy`).

            The decision order is mandatory: the operator's explicit budget
            (`explicit > 0`) - as is, never clamped; else with a declared context -
            "as much as fits" (`context - prompt - reserve`) clamped by the defensive
            cap and never below `1` (a prompt longer than the context yields no
            negative budget); else - the previous cap
            `LEGACY_OUTPUT_BUDGET[operation]` (the degradation to v3.1.0).

    """
    if explicit is not None and explicit > 0:
        return explicit, SOURCE_EXPLICIT
    if context_tokens > 0:
        budget = min(context_tokens - prompt_tokens - reserve_tokens, defensive_cap)
        return max(1, budget), SOURCE_CONTEXT
    return LEGACY_OUTPUT_BUDGET.get(operation, LEGACY_DEFAULT_OUTPUT_BUDGET), SOURCE_LEGACY


class Budgets(NamedTuple):
    """The split budgets of one call (LSBN-T-33, arch §6.1).

            `answer_budget` - the guaranteed claim on the useful answer;
            `reasoning_budget` - the reasoning ceiling (0 - not reserved);
            `sent_budget` - what actually goes into `num_predict`/`max_tokens` (the
            transport's limit is the SUM: thinking + content, so LSB enforces the
            split, not the wire). The transport caps the sum, so "the answer is at
            least `answer_budget`" cannot be guaranteed on the request side - it is a
            claim; the actual check is over the answer (an empty content with
            non-empty reasoning - the budget was eaten).

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
    """The split answer and reasoning budgets (LSBN-T-33, arch §6.1).

            The decision order is mandatory; the unknown-context degradation - as in
            `resolve_output_budget`:
            - `answer_budget` - the answer policy unchanged (`resolve_output_budget`):
              an explicit budget -> "as much as fits" from the context -> the previous
              cap;
            - `reasoning_budget` - the operator's explicit one (`explicit_reasoning >
              0`); else with a known context - the window remainder after the answer
              (clamped by the defensive cap, not below 0); else 0 (degradation: the
              reasoning is not reserved - the 3.4.1 behavior);
            - `sent_budget` - with thinking off (`think_enabled=False`) exactly the
              answer; else the answer+reasoning sum clamped by the context window and
              the defensive cap; with an unknown context - the answer only
              (degradation).

            Important: reasoning never reduces the answer claim - `answer_budget`
            enters `sent_budget` as the first term, and the reasoning occupies the
            window remainder.

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


# The slot config fingerprint composition (lsbdef0008-02/-03, arch §5/§8/§13.8):
# the term order is part of the contract; values are stringified. Provider,
# model, model type, "think / do not think" (`*_THINK`), the output budget, the declared
# context, the client timeout (`*_TIMEOUT_SEC`), the reserve and the defensive cap, plus
# the decision-slot parameters: the raised budget, decision enabled, a reasoning
# or timeout change - also "the cause went away" (FR-2.3), so they enter the
# fingerprint alongside the model. THINK and the timeout are not here for the
# picture's completeness: they change the call's outcome (`think: false` - a
# different answer; a timeout - `timeout` instead of an answer), and the live
# case showed the price of their absence - slots on the
# `JUDGE_THINK=false` + small budget bypass never unfroze pairs after a reasoning change.
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
    # LSBN-T-33: the reasoning budget (the global slot level) - changing it
    # changes the call outcome (sent_budget) just like `*_num_predict`.
    "slot_reasoning_num_predict",
    "judge_model",
    "judge_provider",
    "judge_model_type",
    "judge_think",
    "judge_num_predict",
    "judge_context_tokens",
    "judge_timeout_sec",
    "judge_reasoning_num_predict",
    # LSBN-F-34: the flat operation fields (the chain operation env -> slot ->
    # default) - a MERGE_MODEL/TITLE_TIMEOUT_SEC change etc. is also "the cause
    # went away": without them an operation config change would never clear the
    #"requires attention" of jobs. API_KEY and BASE_URL are excluded: secrets -
    # never; the slot address was never included either (symmetric to slots).
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

# The decision-slot field defaults (arch §8). The slot itself arrives with the
# (`DECISION_ENABLED|DECISION_MODEL|DECISION_THRESHOLD|DECISION_UNCERTAINTY_ZONE`),
# F-12 block, so before it the fields are read via `getattr` with these values:
# the fingerprint already includes decision, and once the fields appear in
# `Settings` it picks them up without edits. No secrets in the composition
# (`*_API_KEY` excluded - the fingerprint lives in `meta` and reaches the journal events).
FINGERPRINT_DEFAULTS: dict[str, object] = {
    "decision_enabled": False,
    "decision_model": "",
    "decision_threshold": 0.5,
    "decision_uncertainty_zone": 0.05,
}


def slot_config_fingerprint(settings: Settings) -> str:
    """The deterministic fingerprint of the slot configuration (lsbdef0008-02).

            A string of the `FINGERPRINT_FIELDS` values in a fixed order - comparable
            and storable in `meta` (the `slot_config_fingerprint` key, block 2). A
            missing or different fingerprint at startup -> attention clears from every
            slot's jobs (`attention_at=NULL, attempts=0`): a slot configuration change
            itself returns "hopeless" jobs to work.

    """
    return "|".join(
        str(getattr(settings, name, FINGERPRINT_DEFAULTS.get(name, "")))
        for name in FINGERPRINT_FIELDS
    )
