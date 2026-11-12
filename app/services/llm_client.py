"""LLMClient - the single HTTP client of the external LLM slots (Phase 11).

    v2.0 kept one client per operation (EmbeddingService, SummaryService,
    JudgeService - three copies of the httpx wiring) and could only speak native
    Ollama. v2.1: each of the three slots (embedding, summary, judge) gets its own
    address/model/key AND provider - `EMBEDDING_PROVIDER` / `SUMMARY_PROVIDER` /
    `JUDGE_PROVIDER` in {ollama, openai}. Here one `LLMClient` class with a
    four-operation API: `chat`, `embed`, `check`, `close`; the slot configuration
    is a frozen `SlotSpec` with the `SlotSpec.for_embedding/for_summary/for_judge`
    factories (built from Settings fields).

    Who consumes it (pools 4 and 6):
    - `embed` - EmbeddingService (the single encoding point). The response-contract
      checks (the vector count = the input count, the dimension = EMBEDDING_DIM)
      and the only retry of transient failures (ARCH SS3.2) live on the caller;
      the retry signal is `LLMError.transient`.
    - `chat` - SummaryService (+ the classifier/describer) and JudgeService
      (+ the structure judge). Neither the client nor chat retries - the repeat is
      the worker's decision via back-off (NFR-3).
    - `check` - main.py at startup: returns the outcome, does not raise; the
      caller decides fatality.
    - `close` - a clean shutdown (the application lifespan).

    Key decisions:
    - The F1 queue (ollama_gate): `ollama_slot` is taken ONLY by an ollama provider
      of the slot for the duration of the HTTP call - one request to the server at
      a time; the read timeout ticks from the actual send. The openai provider is
      an external API, the slot is not taken at all; `check()` does not take the
      slot either (not generation).
    - keep_alive is NOT sent to anyone: the server manages the model through
      OLLAMA_KEEP_ALIVE; in v2.0 the client sent "15m" - removed.
    - Bearer - only to the openai provider and only with a key set. 401/403 at
      openai - `LLMAuthError` without a retry, with a hint: an empty key - "set
      {SLOT}_API_KEY in docker-compose and restart"; a set (wrong) key - "the key
      was rejected by the API". The ollama provider never sends Bearer, so its
      401/403 is a plain status error WITHOUT the key hint but with the `auth`
      flag: the call outcome is `AUTH` in both cases (arch SS6.1: "401/403 -
      waiting is pointless"; FR-4.3 - pause immediately, not after a series).
    - 429/5xx/transport/timeout - `LLMError.transient = True`: the embedder
      retries (429 is transient), chat does not. The client itself never retries -
      the repeat decision belongs to the caller reading the flag.
    - Timeouts: connect - ollama 2 s (LAN), openai 10 s (cloud); read - per-slot
      from Settings: summary_timeout_sec / judge_timeout_sec; the embedder has a
      fixed 720 s (SS8 does not set the embedding timeout; it was sized for CPU
      inference of long texts).
    - check(): one attempt, about 5 s; ollama - GET /api/tags with the slot model
      checked against the list; openai - GET /v1/models (+Bearer). Outcomes:
      ok | auth_failed | unreachable | model_missing. 401/403 -> auth_failed;
      network/timeout/5xx/404/garbage -> unreachable (WARN + degradation, not
      fatal); the model not in the list -> model_missing.
    - The reasoning fields of the response (`message.thinking` at ollama;
      `reasoning_content`/`reasoning` at openai) are read into the call metadata
      only (No49 SS13.3): content stays the only response payload; an empty
      content - the call fails (SS5.5) with the `empty` flag (No39 SS6.3).
    - No39 SS6.3: `LLMError` carries the `timeout`/`empty`/`truncated` flags -
      timeout and transport are distinguishable, an empty answer is not mixed with
      unavailability; the call outcome (SlotOutcome) is read by the service layer.
      `finish_reason`/`usage` and filling `truncated` - block No49.
    - DI without the network: `transport` takes an httpx.MockTransport or a
      handler - unit tests (tests/test_llm_client.py); the service fakes
      (tests/fakes.py) are unchanged - consumers depend on protocols, not this
      class.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

import httpx

from app.config import Settings
from app.services.operations import (
    OPERATIONS,
    OPERATION_FIELD_DEFAULTS,
    SPEC_FIELDS,
    resolve_operation,
)
from app.services.ollama_gate import ollama_slot
from app.services.slot_budget import Budgets, resolve_budgets, resolve_output_budget
from app.services.splitter import count_tokens

# The LLM slot providers (Phase 11).
PROVIDER_OLLAMA = "ollama"
PROVIDER_OPENAI = "openai"

# Timeouts (s): connect per provider; read - per slot from Settings,
# except the embedder - SS8 sets no timeout for it, a fixed 720 s (sized
# for CPU inference of long texts: qwen3-embedding:8b on a CPU host, long
# texts must finish, not tear).
CONNECT_TIMEOUT_SEC_OLLAMA = 2.0
CONNECT_TIMEOUT_SEC_OPENAI = 10.0
EMBEDDING_READ_TIMEOUT_SEC = 720.0

# The startup check: one attempt, ~5 s - a liveness probe, not generation.
#
CHECK_TIMEOUT_SEC = 5.0

# The shared temperature of generative calls (as in v2.0): stable
# summary and verdict wordings matter more than variety.
TEMPERATURE = 0.1

# 401/403 - the key is rejected: retrying is pointless.
STATUS_UNAUTHORIZED = frozenset({401, 403})

# Transient statuses: 429 (overload) + 5xx (as in v2.0).
# Only the embedder retries on this signal; chat does not (NFR-3).
STATUS_TRANSIENT = frozenset({429, 500, 502, 503, 504})

# httpx network/temporary failures - retry candidates.
_RETRIABLE = (httpx.TimeoutException, httpx.TransportError)

# A chunk of the response body in the error text (keep the logs lean).
_ERROR_BODY_CHARS = 120


def _as_int(value: Any) -> int | None:
    """A response counter as int; absent/wrong type/bool -> None (a guard)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_float(value: Any) -> float | None:
    """A response number as float; absent/wrong type/bool -> None (a guard)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_probabilities(value: Any) -> dict[str, float]:
    """The response probability distribution: {label: number}; garbage skipped."""
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            number = _as_float(item)
            if number is not None:
                result[str(key)] = number
    return result

# The startup check outcomes: ok - the slot is ready; auth_failed -
# 401/403 (fatal, main.py decides); unreachable - network/timeout/5xx/404/
# non-JSON (WARN + degradation); model_missing - the slot model is not listed.
CheckResult = Literal["ok", "auth_failed", "unreachable", "model_missing"]
CHECK_OK: CheckResult = "ok"
CHECK_AUTH_FAILED: CheckResult = "auth_failed"
CHECK_UNREACHABLE: CheckResult = "unreachable"
CHECK_MODEL_MISSING: CheckResult = "model_missing"

# LSBN-F-12 (arch §6.2/§13.6): the decision slot's model type (its own DECISION_* config).
SLOT_KIND_DECISION = "decision"

# The decision model's native route (ollama >= 0.35.1): NOT OpenAI-compatible.
DECISION_PATH = "/v1/systemone"

# The statuses for "the server does not know the route / rejects the question kind" (arch §6.2).
_DECISION_UNSUPPORTED_STATUSES = frozenset({400, 404, 405, 422})

# The startup check outcomes of "the model can do decision" (FR-1.3/FR-6.2). Its own type:
# the CheckResult of the generative slots (block 2) is untouched.
DecisionCheck = Literal["ok", "unsupported", "unreachable", "auth_failed"]
DECISION_CHECK_OK: DecisionCheck = "ok"
DECISION_CHECK_UNSUPPORTED: DecisionCheck = "unsupported"
DECISION_CHECK_UNREACHABLE: DecisionCheck = "unreachable"
DECISION_CHECK_AUTH_FAILED: DecisionCheck = "auth_failed"


class LLMError(RuntimeError):
    """An LLM slot call failed (status/transport/response contract).

            `status` - the HTTP status when the error came from the server.
            `transient` - a temporary failure (429/5xx/transport/timeout): the signal
            for the embedder's single retry (ARCH SS3.2); chat does not retry by the
            flag (NFR-3). `auth` - 401/403 from ANY provider (openai also carries
            `LLMAuthError` with the key hint): the outcome is `AUTH`
            (`slot_outcome.classify_llm_error`). `timeout`/`empty`/`truncated` - the
            cause flags (No39, SS6.3): the outcome is distinguished at the client
            boundary (`slot_outcome.classify_llm_error`), not by the message text.
            `truncated` is filled by block No49 (finish_reason). `meta` - the metadata
            of the call that happened (No49): a failure after the model answered
            carries them for the service/worker journal diagnostics.

    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        transient: bool = False,
        auth: bool = False,
        timeout: bool = False,
        empty: bool = False,
        truncated: bool = False,
        meta: ChatMeta | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.transient = transient
        self.auth = auth
        self.timeout = timeout
        self.empty = empty
        self.truncated = truncated
        self.meta = meta


class LLMAuthError(LLMError):
    """401/403 from the openai provider: the key is unset or rejected.

            Never retried: the error text carries the hint - "set {SLOT}_API_KEY in
            docker-compose and restart" for an empty key, or "the key was rejected by
            the API" for a set wrong one.

    """

    def __init__(self, message: str, *, status: int) -> None:
        super().__init__(message, status=status, transient=False, auth=True)


@dataclass(frozen=True)
class SlotSpec:
    """The configuration of one LLM slot: address, model, provider, key.

            The provider is "ollama" (the native API) or "openai" (OpenAI-compatible
            with Bearer). `think` - the thinking flag of a generative slot (None - do
            not send the field; False - send an explicit ban; True - do not send,
            thinking allowed): the ollama provider only.

    """

    name: str  # embedding | summary | judge - for the error texts/hints
    provider: str
    base_url: str
    model: str
    api_key: str
    read_timeout: float
    think: bool | None = None
    # No49 (arch §13.1): the slot model type - reasoning | generation; it affects
    # the budget, the think treatment and the startup warnings
    # (`slot_config_warnings`), but NOT the request body (the wire stays the same).
    model_type: str = "generation"
    # 12.11.2026: the slot's provider accepts no generation limits at all (the
    # ollama.com cloud models): the request carries neither options.num_predict /
    # options.num_ctx (ollama) nor max_tokens (openai).
    no_output_limits: bool = False
    # No49 (arch §6.2): the declared model context (0 - unknown; then the budget
    # degrades to the previous cap), the auto-budget defensive cap and the
    # reserve for the chat template/service fields/tokenizer drift.
    context_tokens: int = 0
    max_output_tokens: int = 32768
    prompt_reserve_tokens: int = 512
    # No49 (arch §8): the WARN threshold "no room for thinking" and the operator's
    # explicit slot budget (0 - auto "as much as fits"); the WARN threshold "the
    # timeout is small for long thinking" - from Settings (for a clean `slot_config_warnings`).
    reasoning_min_output_tokens: int = 4096
    num_predict: int = 0
    # LSBN-T-33: the operator's REASONING budget (0 - unset: "as much as fits"
    # within the context). Filled by the factories from the global
    # SLOT_REASONING_NUM_PREDICT and overridden by the flat operation env.
    reasoning_num_predict: int = 0
    min_timeout_sec: int = 300
    # LSBN-F-34: the source of this spec's MODEL (`op` | `slot` | `default`) -
    # the flat operation env, the slot env, or the default. Filled by for_operation;
    # the slot factories leave "slot" (their model is the slot env). Goes into
    # ChatMeta (the journal) and /health.operations; the address/key are not duplicated here.
    model_source: str = "slot"

    @classmethod
    def for_embedding(cls, settings: Settings) -> SlotSpec:
        """The vectorization slot: a fixed 720 s timeout, no think."""
        return cls(
            name="embedding",
            provider=settings.embedding_provider,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            read_timeout=EMBEDDING_READ_TIMEOUT_SEC,
        )

    @classmethod
    def for_summary(cls, settings: Settings) -> SlotSpec:
        """The summarization slot (and classification/descriptions - the same model)."""
        return cls(
            name="summary",
            provider=settings.summary_provider,
            base_url=settings.summary_base_url,
            model=settings.summary_model,
            api_key=settings.summary_api_key,
            read_timeout=float(settings.summary_timeout_sec),
            think=settings.summary_think,
            model_type=settings.summary_model_type,
            no_output_limits=settings.summary_no_output_limits,
            context_tokens=settings.summary_context_tokens,
            max_output_tokens=settings.slot_max_output_tokens,
            prompt_reserve_tokens=settings.slot_prompt_reserve_tokens,
            reasoning_min_output_tokens=settings.slot_reasoning_min_output_tokens,
            num_predict=settings.summary_num_predict,
            reasoning_num_predict=settings.slot_reasoning_num_predict,
            min_timeout_sec=settings.slot_min_timeout_sec,
        )

    @classmethod
    def for_judge(cls, settings: Settings) -> SlotSpec:
        """The judge slot (dedup + the structure judge - the same model)."""
        return cls(
            name="judge",
            provider=settings.judge_provider,
            base_url=settings.judge_base_url,
            model=settings.judge_model,
            api_key=settings.judge_api_key,
            read_timeout=float(settings.judge_timeout_sec),
            think=settings.judge_think,
            model_type=settings.judge_model_type,
            no_output_limits=settings.judge_no_output_limits,
            context_tokens=settings.judge_context_tokens,
            max_output_tokens=settings.slot_max_output_tokens,
            prompt_reserve_tokens=settings.slot_prompt_reserve_tokens,
            reasoning_min_output_tokens=settings.slot_reasoning_min_output_tokens,
            num_predict=settings.judge_num_predict,
            reasoning_num_predict=settings.slot_reasoning_num_predict,
            min_timeout_sec=settings.slot_min_timeout_sec,
        )

    @classmethod
    def for_operation(
        cls, operation: str, settings: Settings
    ) -> tuple[SlotSpec, dict[str, str]]:
        """The effective spec of an OPERATION (LSBN-F-34) + the source of every field.

                The resolution chain per field: the flat operation env
                (`<OP>_<FIELD>`) -> the slot-group env -> the default (the slot-factory
                values above). Returns `(SlotSpec, sources)` where `sources` fully covers
                `OPERATION_FIELDS` with `op` | `slot` | `default`; the `op`-layer fields in
                the values dict contain ONLY the overridden ones. An operation whose name
                equals the slot group (`judge`) has no separate layer - its env IS the
                slot env. The code defaults of operations (`OPERATION_FIELD_DEFAULTS`,
                e.g. the 128 describe budget) apply when neither the operation nor the
                slot sets anything. The reserve slot (decision, the F-12 mechanics) takes
                no part here: an operation override acts on the primary path only.

        """
        slot = OPERATIONS.get(operation)
        if slot is None:
            raise ValueError(
                f"неизвестная операция: {operation} (реестр: "
                + ", ".join(sorted(OPERATIONS))
                + ")"
            )
        factory = {
            "embedding": cls.for_embedding,
            "summary": cls.for_summary,
            "judge": cls.for_judge,
        }[slot]
        base = factory(settings)
        values, sources = resolve_operation(operation, settings)
        updates: dict[str, object] = {}
        for field, value in values.items():
            updates[SPEC_FIELDS[field]] = (
                float(value) if field == "timeout_sec" else value
            )
        for field, default in OPERATION_FIELD_DEFAULTS.items():
            op_name, op_field = field
            if op_name == operation and op_field not in values and sources.get(op_field) == "default":  # noqa: E501
                updates[SPEC_FIELDS[op_field]] = default
        spec = replace(base, **updates, model_source=sources["model"])
        return spec, sources

    @classmethod
    def for_decision(cls, settings: Settings) -> SlotSpec:
        """The decision slot (LSBN-F-12): the native ollama route `/v1/systemone`.

                The provider is always `ollama` (the route is native, NOT
                OpenAI-compatible); the generation budget does not apply (`num_predict` is
                not set), decision has no thinking flag at all (think=None, arch SS13.1).
                Bearer goes only when `DECISION_API_KEY` is set (see `_headers`).

        """
        return cls(
            name=SLOT_KIND_DECISION,
            provider=PROVIDER_OLLAMA,
            base_url=settings.decision_base_url,
            model=settings.decision_model,
            api_key=settings.decision_api_key,
            read_timeout=float(settings.decision_timeout_sec),
            think=None,
            model_type=SLOT_KIND_DECISION,
        )


@dataclass(frozen=True)
class ChatMeta:
    """The metadata of one slot chat call (No49, arch SS6.2/SS13.3).

            Filled ONLY by the client: `budget`/`budget_source` - the output budget
            policy (`slot_budget.resolve_output_budget`), the counters and the finish
            reason - from the provider's response, `truncated` - cut by the output
            limit (`finish_reason`/`done_reason == "length"`). `model_type` comes from
            `spec.model_type` (the slot model type, arch SS13.1). No secrets and no
            prompt/response bodies here (FR-3.4).

    """

    slot: str
    provider: str
    model: str
    operation: str | None
    model_type: str
    budget: int
    budget_source: str  # explicit | context | legacy
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    had_thinking: bool
    thinking_chars: int
    finish_reason: str | None
    truncated: bool
    duration_ms: int
    # LSBN-F-34: the slot address and the operation config source (`op`/`slot`/
    # `default`) - the brief's decision No4: the address is not exposed (/health),
    # but it goes to the journal (`slot_call`) and the job error event. The address
    # is not a secret (unlike the key, which never reaches here).
    base_url: str = ""
    config_source: str = "slot"
    # LSBN-T-33: the split call budgets. `budget` above is the total
    # (legacy-compatible), equal to `sent_budget`; `budget_source` equals
    # `answer_source`. `escalated_from` - the first attempt's sent_budget when the
    # call was retried after "the budget eaten by reasoning" (None - no retry).
    answer_budget: int = 0
    reasoning_budget: int = 0
    sent_budget: int = 0
    answer_source: str = ""
    reasoning_source: str = ""
    escalated_from: int | None = None


@dataclass(frozen=True)
class ChatResult:
    """The chat call result: content together with the call metadata (SS6.2)."""

    content: str
    meta: ChatMeta


class LLMReasoningBudgetError(LLMError):
    """The answer budget eaten by reasoning (LSBN-T-33; the lsbdef-0008 class).

            Content is empty while thinking happened (ollama - a non-empty
            `message.thinking`; openai - `reasoning_tokens > 0` or a non-empty
            `reasoning_content`). This is NOT a generic "empty answer" and not a "slot
            failure": the distinct type lets the caller recognize the outcome and
            repeat the call with a bigger budget (in the client - one auto-retry up to
            the context window; beyond that - up the stack). Flags: `truncated=True`
            (the budget is exhausted), `empty=False`, `transient=False` - the outcome
            is TRUNCATED (the job waits for attention), but the cause in the error
            text is its own, readable in the journal.

    """

    def __init__(self, message: str) -> None:
        super().__init__(message, truncated=True)


class LLMClient:
    """The HTTP client of one LLM slot: chat / embed / check / close.

            The provider is chosen by `spec.provider`: ollama - POST /api/chat,
            /api/embed, GET /api/tags (through the ollama_gate queue for chat/embed);
            openai - POST /v1/chat/completions, /v1/embeddings, GET /v1/models
            (+Bearer with a key set; the queue slot is not taken).

    """

    def __init__(
        self,
        spec: SlotSpec,
        transport: httpx.BaseTransport
        | Callable[[httpx.Request], httpx.Response]
        | None = None,
    ) -> None:
        self.spec = spec
        # An httpx.MockTransport handler in the constructor - convenient for unit tests.
        if transport is not None and not isinstance(transport, httpx.BaseTransport):
            transport = httpx.MockTransport(transport)
        connect = (
            CONNECT_TIMEOUT_SEC_OLLAMA
            if spec.provider == PROVIDER_OLLAMA
            else CONNECT_TIMEOUT_SEC_OPENAI
        )
        self._client = httpx.Client(
            base_url=spec.base_url,
            timeout=httpx.Timeout(spec.read_timeout, connect=connect),
            transport=transport,
        )

    #--- public API -------------------------------------------------------

    def chat_result(
        self,
        system_prompt: str,
        user_text: str,
        *,
        num_predict: int | None = None,
        think: bool | None = None,
        operation: str | None = None,
    ) -> ChatResult:
        """One generation: system+user, non-stream; returns content + metadata.

                num_predict - the operator's explicit ANSWER budget (None - the "as much
                as fits" policy: the declared slot context minus the prompt and the
                reserve, clamped by the defensive cap; with an unknown context - the
                operation's previous cap). The budgets are split (LSBN-T-33): the call
                carries `sent_budget = answer_budget + reasoning_budget` (clamped by the
                context window) because the transport caps the SUM of thinking+content.
                `operation` - the operation name for degradation
                (`summary`/`title`/`merge`/`judge`/`classify`). think=None - the flag from
                the spec (the slot); a bool - an override (the structure judge:
                NAMESPACE_JUDGE_THINK). An empty content - a refusal (SS5.5); the
                reasoning fields go to the metadata only. The exception - "the budget
                eaten by reasoning" (`LLMReasoningBudgetError`): the content is empty
                while thinking happened - ONE auto-retry with a doubled budget up to the
                context window (not a slot retry: back-off and the failure counters are
                untouched); when a retry is impossible (the window exhausted, the context
                unknown) - the refusal goes up. Any failure - LLMError/LLMAuthError.

        """
        think_flag = self.spec.think if think is None else think
        # The default operation is the slot name: the client is also called without an
        # explicit operation (units, simple calls), and the budget degradation needs a key.
        operation = operation or self.spec.name
        prompt_tokens = count_tokens(system_prompt) + count_tokens(user_text)
        # The caller's explicit budget wins; when unset, the operation ceiling of the
        # spec applies (LSBN-F-34: the flat operation env -> the slot env -> 0/auto).
        if num_predict is None:
            num_predict = self.spec.num_predict or None
        budgets = resolve_budgets(
            operation=operation,
            prompt_tokens=prompt_tokens,
            explicit_answer=num_predict,
            explicit_reasoning=self.spec.reasoning_num_predict or None,
            context_tokens=self.spec.context_tokens,
            reserve_tokens=self.spec.prompt_reserve_tokens,
            defensive_cap=self.spec.max_output_tokens,
            think_enabled=think_flag is not False,
        )
        try:
            return self._chat_once(
                system_prompt,
                user_text,
                operation=operation,
                budgets=budgets,
                sent_budget=budgets.sent_budget,
                think_flag=think_flag,
                prompt_tokens=prompt_tokens,
                escalated_from=None,
            )
        except LLMReasoningBudgetError:
            escalation = self._escalation_budget(budgets, prompt_tokens)
            if escalation is None:
                raise
            # One auto-retry with a bigger sent_budget: this is request-shaping,
            # not a slot-failure repeat - the back-off and the slot break
            # do not engage, the failure counters do not grow.
            return self._chat_once(
                system_prompt,
                user_text,
                operation=operation,
                budgets=budgets,
                sent_budget=escalation,
                think_flag=think_flag,
                prompt_tokens=prompt_tokens,
                escalated_from=budgets.sent_budget,
            )

    def _escalation_budget(self, budgets: Budgets, prompt_tokens: int) -> int | None:
        """The bigger sent_budget for the retry after "eaten by reasoning".

                x2 of what was sent, capped by the context window
                (`context - prompt - reserve`); no retry (None) when the context is
                unknown, the reasoning was not reserved (think=false/degradation), or the
                window is already exhausted by the first call - the operator decides next
                (raise the budget/context or change the model).

        """
        if budgets.reasoning_budget <= 0:
            return None  # the degradation (unknown context) or think=false
        window = self.spec.context_tokens - prompt_tokens - self.spec.prompt_reserve_tokens
        escalated = min(budgets.sent_budget * 2, window)
        return escalated if escalated > budgets.sent_budget else None

    def _chat_once(
        self,
        system_prompt: str,
        user_text: str,
        *,
        operation: str,
        budgets: Budgets,
        sent_budget: int,
        think_flag: bool | None,
        prompt_tokens: int,
        escalated_from: int | None,
    ) -> ChatResult:
        """One HTTP call with the given sent_budget + the response parse.

                Split out of `chat_result` (LSBN-T-33): the first call goes with
                `budgets.sent_budget`, the auto-retry after "the budget eaten by
                reasoning" - with a bigger one. The metadata always describes the ACTUAL
                call (`sent_budget`, and `escalated_from` on a retry).

        """
        budget = sent_budget
        if self.spec.provider == PROVIDER_OPENAI:
            path = "/v1/chat/completions"
            payload = {
                "model": self.spec.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                "temperature": TEMPERATURE,
            }
            if not self.spec.no_output_limits:
                payload["max_tokens"] = budget
        else:
            path = "/api/chat"
            payload = self._ollama_chat_payload(
                system_prompt, user_text, budget, think_flag
            )
        started = time.monotonic()
        data = self._request_json("POST", path, payload)
        duration_ms = int((time.monotonic() - started) * 1000)
        # No49/No50 (§13.8): the finish reason and the counters are read BEFORE
        # parsing the content - with an empty cut answer the call metadata already
        # exists and leaves with the refusal (worker diagnostics without a second write).
        finish_reason, truncated = self._read_finish_reason(data, self.spec.provider)
        (
            prompt_count,
            completion_count,
            reasoning_count,
            had_thinking,
            thinking_chars,
        ) = self._read_usage(data, self.spec.provider)
        meta = ChatMeta(
            slot=self.spec.name,
            provider=self.spec.provider,
            model=self.spec.model,
            operation=operation,
            model_type=self.spec.model_type,
            budget=budget,
            budget_source=budgets.answer_source,
            prompt_tokens=prompt_count,
            completion_tokens=completion_count,
            reasoning_tokens=reasoning_count,
            had_thinking=had_thinking,
            thinking_chars=thinking_chars,
            finish_reason=finish_reason,
            truncated=truncated,
            duration_ms=duration_ms,
            base_url=self.spec.base_url,
            config_source=self.spec.model_source,
            answer_budget=budgets.answer_budget,
            reasoning_budget=budgets.reasoning_budget,
            sent_budget=sent_budget,
            answer_source=budgets.answer_source,
            reasoning_source=budgets.reasoning_source,
            escalated_from=escalated_from,
        )
        try:
            content = self._parse_chat_content(
                data, path, truncated=truncated, had_thinking=had_thinking
            )
        except LLMError as exc:
            exc.meta = meta  # a failure after the model answered - the call metadata
            raise
        return ChatResult(content=content, meta=meta)

    def chat(
        self,
        system_prompt: str,
        user_text: str,
        *,
        num_predict: int | None = None,
        think: bool | None = None,
        operation: str | None = None,
    ) -> str:
        """A wrapper over `chat_result(...)` - compatibility: returns content.

                Unit tests and fakes keep working with the plain string result;
                `num_predict` is optional now (None - the "as much as fits" policy).

        """
        return self.chat_result(
            system_prompt,
            user_text,
            num_predict=num_predict,
            think=think,
            operation=operation,
        ).content

    @staticmethod
    def slot_config_warnings(spec: SlotSpec) -> list[tuple[str, str]]:
        """The slot configuration warnings (No49, arch SS13.5) - a pure function.

                Returns a list of `(code, EN text)` and logs nothing (main.py logs at
                startup, the `slot_config_warning` event); needs neither the network nor
                Settings. The effective budget comes from the same source of truth as in
                the call (`slot_budget.resolve_output_budget`): the operator's explicit
                slot budget, else "as much as fits" from the declared context, else the
                operation's previous cap. Five codes: `reasoning_tight_budget`,
                `reasoning_think_off`, `budget_gt_context`, `context_unknown`,
                `timeout_short`.

        """
        warnings: list[tuple[str, str]] = []
        prefix = f"slot {spec.name}"
        upper = spec.name.upper()
        budget, _source = resolve_output_budget(
            operation=spec.name,
            prompt_tokens=0,
            explicit=spec.num_predict or None,
            context_tokens=spec.context_tokens,
            reserve_tokens=spec.prompt_reserve_tokens,
            defensive_cap=spec.max_output_tokens,
        )
        if (
            spec.model_type == "reasoning"
            and budget < spec.reasoning_min_output_tokens
        ):
            warnings.append(
                (
                    "reasoning_tight_budget",
                    f"{prefix}: model declared as reasoning but the output budget "
                    f"({budget}) leaves little room for thinking — raise the "
                    f"budget, add {upper}_CONTEXT_TOKENS, or set "
                    f"{upper}_MODEL_TYPE intentionally.",
                )
            )
        if spec.model_type == "reasoning" and spec.think is False:
            warnings.append(
                (
                    "reasoning_think_off",
                    f"{prefix}: reasoning model with THINK=false — thinking is "
                    "disabled on purpose? A reasoning model without thinking may "
                    f"answer poorly or empty; set {upper}_THINK=true unless this "
                    "is deliberate.",
                )
            )
        if spec.context_tokens > 0 and budget > spec.context_tokens:
            warnings.append(
                (
                    "budget_gt_context",
                    f"{prefix}: output budget ({budget}) is above the model "
                    f"context ({spec.context_tokens}) — the server will truncate "
                    f"or reject; lower the budget or fix {upper}_CONTEXT_TOKENS.",
                )
            )
        if spec.context_tokens == 0:
            warnings.append(
                (
                    "context_unknown",
                    f"{prefix}: model context is unknown — the output budget falls "
                    "back to the previous per-operation caps; set "
                    f"{upper}_CONTEXT_TOKENS to get the \"as much as fits\" "
                    "policy.",
                )
            )
        if (
            spec.model_type == "reasoning"
            and spec.read_timeout < spec.min_timeout_sec
        ):
            warnings.append(
                (
                    "timeout_short",
                    f"{prefix}: timeout {spec.read_timeout:g}s is short for a "
                    "reasoning model — a long-thinking call will look like a "
                    f"timeout; raise {upper}_TIMEOUT_SEC.",
                )
            )
        return warnings

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts; the result order = the input order.

                ollama - POST /api/embed (input: [...]); openai - POST /v1/embeddings
                (+Bearer; the response items are ordered by `index`). The response-contract
                checks (the vector count, the dimension = EMBEDDING_DIM) belong to the
                caller (EmbeddingService, pool 4): the client returns exactly what the
                server sent.

        """
        if not texts:
            raise ValueError("embed: пустой список текстов")
        path = "/v1/embeddings" if self.spec.provider == PROVIDER_OPENAI else "/api/embed"
        payload = {"model": self.spec.model, "input": list(texts)}
        data = self._request_json("POST", path, payload)
        return self._parse_embeddings(data, path)

    def check(self) -> CheckResult:
        """The startup slot check: an outcome, not an exception.

                One attempt, about 5 s, no queue slot (not generation). ollama - GET
                /api/tags + the slot model checked against the list; openai - GET
                /v1/models (+Bearer with a key set). The fatality of the outcome is the
                caller's (main.py, pool 4): auth_failed is fatal; unreachable/
                model_missing - WARN + degradation.

        """
        try:
            response = self._client.get(
                "/v1/models" if self.spec.provider == PROVIDER_OPENAI else "/api/tags",
                headers=self._headers(),
                timeout=CHECK_TIMEOUT_SEC,
            )
        except _RETRIABLE:
            # Network/timeout: WARN + degradation, not fatal.
            return CHECK_UNREACHABLE
        if response.status_code in STATUS_UNAUTHORIZED:
            return CHECK_AUTH_FAILED
        if response.status_code != 200:
            return CHECK_UNREACHABLE
        try:
            data = response.json()
        except ValueError:
            return CHECK_UNREACHABLE
        if self.spec.model not in self._model_names(data):
            return CHECK_MODEL_MISSING
        return CHECK_OK

    def close(self) -> None:
        """Close the HTTP pool (a clean process shutdown)."""
        self._client.close()

    def decide(self, state: str, questions: list[dict]) -> dict:
        """One `POST /v1/systemone`: state + ALL questions in one body (FR-2.3).

                Returns the parsed `answers` block (the server's actual shape, verified
                live); `questions` - the canonical dicts (`name`/`kind`/`text`/`labels`/
                `criteria`); only `_systemone_payload` knows the server's field names. A
                failure (network/non-200/garbage) - `LLMError` with the usual flags
                (`transient`/`status`); the `empty` flag is never set - there is no
                generation, so there can be no "empty answer".

        """
        payload = {
            "model": self.spec.model,
            **self._systemone_payload(state, questions),
        }
        data = self._request_json("POST", DECISION_PATH, payload)
        return self._parse_systemone(data)

    def check_decision(self) -> DecisionCheck:
        """The startup "can the model do decision" check (FR-1.3/FR-6.2).

                One attempt, the slot timeout, no queue slot (not generation): a minimal
                `/v1/systemone` with one trivial `noul` question. Outcomes: `ok` - a
                correct `answers` block; `unsupported` - the server does not know the
                route or rejects the question kind (400-404/200 with a foreign shape);
                `auth_failed` - 401/403; `unreachable` - network/timeout/5xx/non-JSON.
                The fatality of the outcome is the caller's (main.py).

        """
        questions = [
            {
                "name": "probe",
                "kind": "noul",
                "text": "Is this a routine availability probe of the decision route?",
                "labels": (),
                "criteria": None,
            }
        ]
        payload = {
            "model": self.spec.model,
            **self._systemone_payload("probe", questions),
        }
        try:
            response = self._client.post(
                DECISION_PATH,
                json=payload,
                headers=self._headers(),
                timeout=self.spec.read_timeout,
            )
        except _RETRIABLE:
            return DECISION_CHECK_UNREACHABLE
        if response.status_code in STATUS_UNAUTHORIZED:
            return DECISION_CHECK_AUTH_FAILED
        if response.status_code in _DECISION_UNSUPPORTED_STATUSES:
            return DECISION_CHECK_UNSUPPORTED
        if response.status_code != 200:
            return DECISION_CHECK_UNREACHABLE
        try:
            data = response.json()
        except ValueError:
            return DECISION_CHECK_UNREACHABLE
        try:
            self._parse_systemone(data)
        except LLMError:
            # The server answered but does not know the contract - the "cannot do decision" outcome.
            return DECISION_CHECK_UNSUPPORTED
        return DECISION_CHECK_OK

    @staticmethod
    def _systemone_payload(state: str, questions: list[dict]) -> dict:
        """The `/v1/systemone` body - THE single adaptation point of the format (arch SS6.2/SS13.6).

                The server's actual contract (ollama >= 0.35.1; verified by a live
                request): `questions` is an OBJECT "question name -> question" (1..64); a
                question carries `type` (choice|noul|score) and `instructions` (the
                wording, NOT `name`/`text`); `criteria` - the `choice` options (the key is
                the label, the value - a description or null for a bare label; the server
                requires at least two), an array of descriptions for `score` (from the
                lowest score up), or `{"true": ..., "false": ...}` for `noul` (optional).
                `state` - a non-empty string (an object/array is accepted too). `model` is
                added by the caller (`decide`/`check_decision`): the function is pure and
                testable without a client. If the server diverges from this format - only
                this pair (`_systemone_payload`/`_parse_systemone`) and its contract test
                change; consumers do not.

        """
        items: dict[str, dict[str, Any]] = {}
        for question in questions:
            kind = str(question["kind"])
            item: dict[str, Any] = {
                "type": kind,
                "instructions": question["text"],
            }
            criteria = question.get("criteria")
            if kind == "choice":
                # The choice options come from the criteria object itself: the labels are its keys.
                # Labels without a description arrive as null (a bare label).
                options: dict[str, str | None] = {
                    str(label): None for label in (question.get("labels") or ())
                }
                if criteria:
                    for label, description in criteria.items():
                        options[str(label)] = (
                            None if description is None else str(description)
                        )
                item["criteria"] = options
            elif kind == "score":
                # score: the level descriptions from the lowest score up (0..N-1).
                levels = (
                    criteria.values() if isinstance(criteria, Mapping) else criteria
                )
                item["criteria"] = [str(level) for level in (levels or ())]
            elif criteria:
                # noul: the optional descriptions of the two outcomes; the server knows
                # only the 'true'/'false' keys (it rejects the rest).
                item["criteria"] = {
                    str(key): str(value)
                    for key, value in criteria.items()
                    if str(key) in ("true", "false")
                }
            items[str(question["name"])] = item
        return {"state": state, "questions": items}

    @staticmethod
    def _parse_systemone(data: Any) -> dict:
        """Parse the `/v1/systemone` response - THE single adaptation point (arch SS6.2).

                Returns `{question name: canonical answer}`; the three server shapes
                (`noul`/`choice`/`score`) normalize into the canonical keys
                `kind`/`label`/`probability`/`confidence`/`probabilities`/`score`/`raw`.
                A response without the `answers` block or with an unknown question shape -
                `LLMError` (this is NOT an "empty answer": there is no generation).

        """
        if not isinstance(data, dict):
            raise LLMError(f"{DECISION_PATH}: ответ не объект JSON")
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise LLMError(f"{DECISION_PATH}: в ответе нет блока answers")
        return {
            str(name): LLMClient._normalize_decision_answer(str(name), value)
            for name, value in answers.items()
        }

    @staticmethod
    def _normalize_decision_answer(name: str, value: Any) -> dict:
        """One question's answer -> a canonical dict (the server shapes)."""
        if not isinstance(value, dict):
            raise LLMError(
                f"{DECISION_PATH}: ответ вопроса «{name}» — не объект JSON"
            )
        raw = dict(value)
        if "noul" in value:
            noul = _as_float(value.get("noul"))
            if noul is None:
                raise LLMError(
                    f"{DECISION_PATH}: noul-ответ вопроса «{name}» без числа"
                )
            return {
                "kind": "noul",
                "label": "true" if noul >= 0.5 else "false",
                "probability": noul,
                "confidence": _as_float(value.get("confidence")),
                "probabilities": {},
                "score": None,
                "raw": raw,
            }
        if "choice" in value:
            choice = value.get("choice")
            if not isinstance(choice, str) or not choice.strip():
                raise LLMError(
                    f"{DECISION_PATH}: choice-ответ вопроса «{name}» без метки"
                )
            probabilities = _as_probabilities(value.get("probabilities"))
            return {
                "kind": "choice",
                "label": choice,
                "probability": probabilities.get(choice),
                "confidence": _as_float(value.get("confidence")),
                "probabilities": probabilities,
                "score": None,
                "raw": raw,
            }
        if "score" in value:
            score = _as_float(value.get("score"))
            if score is None:
                raise LLMError(
                    f"{DECISION_PATH}: score-ответ вопроса «{name}» без числа"
                )
            return {
                "kind": "score",
                "label": None,
                "probability": score,
                "confidence": _as_float(value.get("confidence")),
                "probabilities": _as_probabilities(value.get("probabilities")),
                "score": score,
                "raw": raw,
            }
        raise LLMError(
            f"{DECISION_PATH}: незнакомая форма ответа вопроса «{name}»"
        )

    #--- internal ----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Bearer - with a key set: the openai provider and the decision slot.

                The native ollama route `/v1/systemone` (decision) goes with Bearer when
                the operator set `DECISION_API_KEY` (arch SS8); generative ollama slots
                never send a key.

        """
        if not self.spec.api_key.strip():
            return {}
        if (
            self.spec.provider == PROVIDER_OPENAI
            or self.spec.model_type == SLOT_KIND_DECISION
        ):
            return {"Authorization": f"Bearer {self.spec.api_key.strip()}"}
        return {}

    def _ollama_chat_payload(
        self,
        system_prompt: str,
        user_text: str,
        num_predict: int,
        think_flag: bool | None,
    ) -> dict[str, Any]:
        """The /api/chat body: model, messages system+user, stream:false,
                    num_predict, temperature; think per the flag (False - an explicit ban).

                    keep_alive is NOT sent: the server manages the model
                    (OLLAMA_KEEP_ALIVE); in v2.0 the client sent "15m" - removed.

        """
        # The generation parameters belong INSIDE `options` (LSBN-D-14, live
        # 2026-11-11): ollama ignores unknown top-level fields, so a top-level
        # `num_predict` was silently dropped - the model generated with its own
        # default (unlimited) and the whole answer/reasoning budget arithmetic was
        # never applied. Proven on the stand: top-level num_predict=4 returned 45
        # tokens, options.num_predict=4 returned 4 (done_reason=length).
        if self.spec.no_output_limits:
            # No answer cap at all (12.11.2026, О.: «лимиты — только в промптах»):
            # `num_predict` is not sent, the model stops when it is done. The WINDOW
            # stays: without num_ctx the server allocates its own default (262 144
            # on the AI server), the model no longer fits the GPU and generation
            # degrades several times over.
            body: dict[str, Any] = {
                "model": self.spec.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                "stream": False,
            }
            if self.spec.context_tokens and self.spec.context_tokens > 0:
                body["options"] = {"num_ctx": int(self.spec.context_tokens)}
            if think_flag is False:
                body["think"] = False
            return body
        options: dict[str, Any] = {
            "num_predict": int(num_predict),
            "temperature": TEMPERATURE,
        }
        # The declared context is also the request's num_ctx: without it the server
        # allocates its own default (262 144 on the AI server), the model no longer
        # fits the GPU and generation degrades several times (the slow "thinking"
        # of 2026-11-11).
        if self.spec.context_tokens and self.spec.context_tokens > 0:
            options["num_ctx"] = int(self.spec.context_tokens)
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "stream": False,
            "options": options,
        }
        if think_flag is False:
            # An explicit thinking ban (as in v2.0: think:false saves seconds on
            # thoughtless tasks).
            payload["think"] = False
        return payload

    def _request_json(self, method: str, path: str, payload: dict[str, Any]) -> Any:
        """One HTTP call (ollama - through the queue slot) + a JSON parse.

                No retries: transience is marked by `LLMError.transient`; the repeat
                decision belongs to the caller (the embedder retries, chat does not).

        """
        try:
            if self.spec.provider == PROVIDER_OLLAMA:
                with ollama_slot(self.spec.base_url):
                    response = self._client.request(
                        method, path, json=payload, headers=self._headers()
                    )
            else:
                response = self._client.request(
                    method, path, json=payload, headers=self._headers()
                )
        except _RETRIABLE as exc:
            # No39 §6.3: timeout and transport are no longer indistinguishable - the timeout
            # carries its own `timeout` flag (the TIMEOUT outcome, not UNAVAILABLE).
            raise LLMError(
                f"сервер слота {self.spec.name} недоступен "
                f"({self.spec.base_url}): {exc}",
                transient=True,
                timeout=isinstance(exc, httpx.TimeoutException),
            ) from exc
        if response.status_code != 200:
            self._raise_for_status(response, path)
        try:
            return response.json()
        except ValueError as exc:  # json.JSONDecodeError is a subclass of ValueError
            raise LLMError(f"не-JSON ответ от {path}: {exc}") from exc

    def _raise_for_status(self, response: httpx.Response, path: str) -> None:
        """Non-200 -> a typed error.

                401/403 - an access refusal from ANY provider (arch SS6.1: `AUTH` =
                "401/403 - waiting is pointless"): at an openai slot it is `LLMAuthError`
                with the `{SLOT}_API_KEY` hint; at an ollama slot (no Bearer) - a plain
                `LLMError` with the `auth` flag. The outcome is `AUTH` in both cases
                (FR-4.3) - only the text differs: the key hint stays where a key actually
                exists, and does not mislead.

        """
        status = response.status_code
        body = " ".join(response.text[:_ERROR_BODY_CHARS].split())
        if status in STATUS_UNAUTHORIZED:
            if self.spec.provider == PROVIDER_OPENAI:
                raise LLMAuthError(self._auth_message(status), status=status)
            raise LLMError(
                f"HTTP {status} от {path}: {body}",
                status=status,
                transient=False,
                auth=True,
            )
        raise LLMError(
            f"HTTP {status} от {path}: {body}",
            status=status,
            transient=status in STATUS_TRANSIENT,
        )

    def _auth_message(self, status: int) -> str:
        """The 401/403 hint: an empty key - set one; a set one - wrong."""
        slot_env = f"{self.spec.name.upper()}_API_KEY"
        prefix = f"провайдер openai слота {self.spec.name}: API отклонил запрос ({status}) — "
        if not self.spec.api_key.strip():
            return prefix + f"задай {slot_env} в docker-compose и перезапусти"
        return prefix + "ключ отклонён API"

    @staticmethod
    def _read_finish_reason(data: Any, provider: str) -> tuple[str | None, bool]:
        """The finish reason and the truncation flag (SS13.3).

                ollama - `done_reason` (there is NO `finish_reason` field at ollama);
                openai - `choices[0].finish_reason`. `"length"` - cut by the output limit
                (`truncated=True`); otherwise/an absent field - `(None, False)` - the
                truncation unknown, an old/unfamiliar server does not fail the call.

        """
        reason: str | None = None
        if isinstance(data, dict):
            if provider == PROVIDER_OPENAI:
                choices = data.get("choices")
                if (
                    isinstance(choices, list)
                    and choices
                    and isinstance(choices[0], dict)
                ):
                    value = choices[0].get("finish_reason")
                    reason = value if isinstance(value, str) else None
            else:
                value = data.get("done_reason")
                reason = value if isinstance(value, str) else None
        return reason, reason == "length"

    @staticmethod
    def _read_usage(
        data: Any, provider: str
    ) -> tuple[int | None, int | None, int | None, bool, int]:
        """The counters and the thinking fact of the response (SS13.3).

                -> `(prompt_tokens, completion_tokens, reasoning_tokens, had_thinking,
                thinking_chars)`. ollama: `prompt_eval_count`, `eval_count`,
                `reasoning_tokens=None` (ollama has NO reasoning-token counter),
                `message.thinking` -> the fact/length of the text. openai:
                `usage.prompt_tokens`, `usage.completion_tokens`,
                `usage.completion_tokens_details.reasoning_tokens` (when the server sends
                it); the thinking text - `message.reasoning_content` and
                `message.reasoning` (at Selectel - `reasoning`). Every read is guarded: an
                absent field does not fail the call and stays `None`.

        """
        prompt: int | None = None
        completion: int | None = None
        reasoning: int | None = None
        thinking: str | None = None
        if isinstance(data, dict):
            if provider == PROVIDER_OPENAI:
                usage = data.get("usage")
                if isinstance(usage, dict):
                    prompt = _as_int(usage.get("prompt_tokens"))
                    completion = _as_int(usage.get("completion_tokens"))
                    details = usage.get("completion_tokens_details")
                    if isinstance(details, dict):
                        reasoning = _as_int(details.get("reasoning_tokens"))
                choices = data.get("choices")
                message = (
                    choices[0].get("message")
                    if isinstance(choices, list)
                    and choices
                    and isinstance(choices[0], dict)
                    else None
                )
                if isinstance(message, dict):
                    for key in ("reasoning_content", "reasoning"):
                        value = message.get(key)
                        if isinstance(value, str) and value:
                            thinking = value
                            break
            else:
                prompt = _as_int(data.get("prompt_eval_count"))
                completion = _as_int(data.get("eval_count"))
                message = data.get("message")
                if isinstance(message, dict):
                    value = message.get("thinking")
                    if isinstance(value, str) and value:
                        thinking = value
        had_thinking = bool(thinking)
        return prompt, completion, reasoning, had_thinking, len(thinking or "")

    def _parse_chat_content(
        self, data: Any, path: str, *, truncated: bool, had_thinking: bool = False
    ) -> str:
        """Extract the chat response content; malformation/emptiness - a refusal (SS5.5).

                ollama - `message.content` (the `thinking` field is discarded in
                principle); openai - `choices[0].message.content` (the
                `reasoning_content`/`reasoning` fields are ignored: an answer made of
                pure reasoning without content is not an answer).
                The precedence rule (No50 SS13.8): when `truncated` (the finish reason
                `length`), an empty content is a BUDGET cut, the `truncated` flag (the
                `empty` flag is not set); otherwise - the old `empty`.
                LSBN-T-33: an empty content with NON-EMPTY reasoning - the separate
                "budget eaten by reasoning" outcome (`LLMReasoningBudgetError`),
                recognizable by the type, not by parsing the error text.

        """
        content = None
        if isinstance(data, dict):
            if self.spec.provider == PROVIDER_OPENAI:
                choices = data.get("choices")
                message = (
                    choices[0].get("message")
                    if isinstance(choices, list)
                    and choices
                    and isinstance(choices[0], dict)
                    else None
                )
                content = message.get("content") if isinstance(message, dict) else None
            else:
                message = data.get("message")
                content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            if had_thinking:
                # No answer while thinking happened: the reasoning ate the output budget
                # (lsbdef-0008 in its explicit form). The distinct type lets the caller
                # tell "eaten by reasoning" from "the slot refused".
                detail = (
                    "done_reason='length' — обрезано по суммарному лимиту вывода"
                    if truncated
                    else "модель завершилась без content"
                )
                raise LLMReasoningBudgetError(
                    f"бюджет ответа съеден рассуждением: content пуст, "
                    f"thinking был ({detail}; провайдер {self.spec.provider}, "
                    f"слот {self.spec.name}, {path})"
                )
            if truncated:
                # A reasoning model may have eaten the whole output budget -
                # the call is cut, not "empty" (No50 §13.8, the lsbdef-0008 defect).
                raise LLMError(
                    f"пустой content при причине завершения 'length' "
                    f"(обрезано по лимиту вывода; провайдер {self.spec.provider}, "
                    f"слот {self.spec.name}, {path})",
                    truncated=True,
                )
            # No39 §6.3: an empty answer gets its own `empty` flag -
            # the EMPTY outcome is not mixed with unavailability (FR-3.3). The text unchanged.
            raise LLMError(
                f"пустой content в ответе модели "
                f"(провайдер {self.spec.provider}, слот {self.spec.name}, {path})",
                empty=True,
            )
        return content.strip()

    @staticmethod
    def _parse_embeddings(data: Any, path: str) -> list[list[float]]:
        """The embed response vectors: ollama - `embeddings`; openai - `data`.

                The openai items are sorted by `index` (the result order = the input
                order). The response shape (the vector count, the dimension) is not
                checked - the response contract stays with EmbeddingService.

        """
        if isinstance(data, dict) and isinstance(data.get("embeddings"), list):
            return list(data["embeddings"])  # ollama: already in the input order
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            items = sorted(
                (item for item in data["data"] if isinstance(item, dict)),
                key=lambda item: item.get("index", 0)
                if isinstance(item.get("index"), int)
                else 0,
            )
            embeddings: list[list[float]] = []
            for item in items:
                if not isinstance(item.get("embedding"), list):
                    raise LLMError(f"{path}: элемент ответа без embedding")
                embeddings.append(item["embedding"])
            return embeddings
        raise LLMError(f"{path}: неожиданный формат ответа (нет списка векторов)")

    @staticmethod
    def _model_names(data: Any) -> set[str]:
        """The model names of a check() response: ollama `models[].name`,
                    openai `data[].id` - the slot model is matched by the exact name.
        """
        names: set[str] = set()
        if not isinstance(data, dict):
            return names
        for item in data.get("models", []) or []:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                names.add(item["name"])
        for item in data.get("data", []) or []:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                names.add(item["id"])
        return names
