"""Decision slot (LSBN-F-12, release 3.2.0; arch §6.4/§13.6).

    The third slot type next to the generative and embedding ones: the client branch
    of `POST /v1/systemone` (state + typed questions → probabilities). This module
    is the decision domain layer: types, the threshold/uncertainty zone and ITS OWN
    slot breaker (`SlotBreaker` of block 2 — reused, not duplicated).

    Three switch reasons to the generative slot (exactly these, §13.6):

    - `unavailable` — the slot is NOT THERE NOW: network/5xx/timeout/the decision
      slot breaker, a pause, as well as the `unreachable`/`auth_failed` outcome
      suppressed by the startup check and the `DECISION_ENABLED=false` flag-off path
      (but a flag-off is not a switch at all — it moves no counters, FR-4.4);
    - `unsupported` — the contract does not add up: the server does not know
      `/v1/systemone`, does not accept the question kind, there are more labels
      than `DECISION_MAX_LABELS`, or the answer has no solution for this question;
      the same name marks a slot suppressed by a check with the `unsupported`
      outcome ("the model cannot do decision");
    - `low_confidence` — the probability/confidence is in the uncertainty zone
      around the threshold (`|p − threshold| < zone`), or `choice` is below
      `threshold + zone`.

    The reason dictionary is closed: a foreign name in `_fallback` is a development
    error (`ValueError`), and "the decision is not applicable" (a `choice` with
    fewer than two candidates) does not count as a switch (`decision_skipped`,
    `f12-02`).

    `DecisionService.decide` does not throw foreign exceptions: any switch reason is
    a `DecisionUnavailable` with the `reason` field (and `answer` when the solution
    was parsed but rejected by confidence). Probabilities are not stored in the DB
    (FR-3.3) — only the journal (`decision_answer`/`decision_fallback`, written by
    the service itself: it is the only one that knows the question, the probability
    and the reason, §7.2/§13.6) and in-process counters (`snapshot()`).

    The decision breaker lives HERE, not in the worker (arch §13.8): the decision
    call happens inside the service, the worker does not see it; "one breaker point
    per slot" is preserved — the `SlotBreaker` class is the same.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from app.config import Settings
from app.services.llm_client import (
    DECISION_CHECK_AUTH_FAILED,
    DECISION_CHECK_OK,
    DECISION_CHECK_UNSUPPORTED,
    ChatMeta,
    DecisionCheck,
    LLMClient,
    LLMError,
)
from app.services.operations import operation_mode, pause_after, pause_sec
from app.services.slot_breaker import SlotBreaker
from app.services.slot_outcome import SlotOutcome, classify_llm_error

if TYPE_CHECKING:  # pragma: no cover — the registry is injected, no import cycle
    from app.services.prompts import PromptRegistry

# Decision-model question kinds (arch §6.4): there are no others.
DECISION_KINDS = ("choice", "noul", "score")

# Exactly three switch reasons (arch §13.6, FR-4.2): there are and can be no others —
# the reason passes this whitelist in `_fallback`; a foreign name is an immediate ValueError.
DECISION_REASONS = ("unavailable", "unsupported", "low_confidence")

# LSBN-F-37: the scale a threshold is compared on. The journal names it next to the
# threshold (`applied_threshold`/`applied_scale` in `decision_answer`), so a number in
# the log is never read against a threshold of the OTHER scale (review defect of
# 13.11.2026): `confidence` — the excess over chance (a `choice` answer carries it),
# `probability` — the bare label probability (`noul`, `score`, and `choice` whose
# answer came without a confidence).
SCALE_CONFIDENCE = "confidence"
SCALE_PROBABILITY = "probability"
DECISION_SCALES = (SCALE_CONFIDENCE, SCALE_PROBABILITY)

# Failure statuses meaning "waiting is pointless": the breaker opens immediately
# (like AUTH of generative slots, block 2).
_AUTH_STATUSES = frozenset({401, 403})

# Statuses meaning "the server does not know the route/question kind".
_UNSUPPORTED_STATUSES = frozenset({400, 404, 405, 422})

# The decision slot: the name in the journal and in snapshot (§6.4).
SLOT_DECISION = "decision"


@dataclass(frozen=True)
class DecisionQuestion:
    """A decision-model question (arch §6.4): name, wording, kind, criteria.

            `name` — the question key (also the key in `answers`); `text` — the wording
            (EN, impersonal); `kind` — choice | noul | score; `labels` — the options
            of `choice`; `criteria` — a criterion per label (`choice`) or an array of
            criteria (`score`); `threshold` — the question threshold from the registry,
            the STRONGEST of all (LSBN-F-37): it wins over the knob and lives on the
            scale of the question's kind (for `choice` — confidence, for `noul`/`score`
            — probability), so the author of a question writes it in that scale.
            `threshold is None` → the knob decides, and not one number for everything:
            `DECISION_CHOICE_THRESHOLD` for `choice` judged by confidence, otherwise
            `DECISION_THRESHOLD` — that includes `choice` whose answer came WITHOUT
            `confidence`: there is no confidence scale then, the bare probability is
            compared and the general threshold applies (the two scales never meet in
            one comparison).
    """

    name: str
    text: str
    kind: str
    labels: tuple[str, ...] = ()
    # LSBN-F-37: a `choice` criterion may be null — a bare label without a
    # description (the live server contract accepts it); the classifier sends
    # node paths with their descriptions here.
    criteria: Mapping[str, str | None] | None = None
    threshold: float | None = None


@dataclass(frozen=True)
class DecisionAnswer:
    """The decision model's answer to one question (arch §6.4).

            `label` — the chosen label (`choice`), 'true'|'false' (`noul`), None
            (`score`); `probability` — the probability of the chosen/true one;
            `confidence` — the answer confidence (if provided); `probabilities` —
            the distribution; `score` — a `score`-kind score; `raw` — the question's
            raw answer (for the journal, not written to the DB).
    """

    question: str
    kind: str
    label: str | None
    probability: float | None
    confidence: float | None
    probabilities: Mapping[str, float]
    score: float | None
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class DecisionResult:
    """The accepted decision: answers per question + call metadata (§6.4)."""

    answers: dict[str, DecisionAnswer]
    meta: ChatMeta


class DecisionUnavailable(RuntimeError):
    """No decision — switch to the generative slot (§13.6).

            `reason` ∈ {'unavailable','unsupported','low_confidence'}; `answer` —
            the parsed question answer when the decision was rejected by confidence.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        answer: DecisionAnswer | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.answer = answer


def answer_from_parsed(question: str, parsed: Mapping[str, Any]) -> DecisionAnswer:
    """Canonical `_parse_systemone` answer → `DecisionAnswer` (§6.2/§6.4).

            The canonical keys (`kind`/`label`/`probability`/`confidence`/
            `probabilities`/`score`/`raw`) are picked by EXACTLY the client's pair of
            functions — here we only carry them into the domain type; the server's
            field names are unknown here.
    """
    return DecisionAnswer(
        question=question,
        kind=parsed["kind"],
        label=parsed.get("label"),
        probability=parsed.get("probability"),
        confidence=parsed.get("confidence"),
        probabilities=dict(parsed.get("probabilities") or {}),
        score=parsed.get("score"),
        raw=dict(parsed.get("raw") or {}),
    )


def _outcome_of_check(result: DecisionCheck) -> SlotOutcome | None:
    """The probe outcome of `check_decision()` → the breaker outcome (None for ok)."""
    if result == DECISION_CHECK_OK:
        return None
    if result == DECISION_CHECK_AUTH_FAILED:
        return SlotOutcome.AUTH
    if result == DECISION_CHECK_UNSUPPORTED:
        return SlotOutcome.OFF_CONTRACT
    return SlotOutcome.UNAVAILABLE  # unreachable


class DecisionService:
    """The decision service: threshold/zone, its own breaker, counters (arch §6.4).

            Built ONLY when `settings.decision_enabled` (the assembly point is
            `build_services`); with the flag off `can_decide()` is false and no
            transport call is made. Consumers (the judge, the classifier) connect via
            a separate brief `f12-02`.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        llm: LLMClient | None = None,
        registry: PromptRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._registry = registry
        self._enabled = bool(settings.decision_enabled)
        # The startup check may suppress the slot (disable()); before it the slot is alive.
        self._ok = self._enabled
        # The suppressed slot's switch reason: by default "the slot is not
        # there now"; `disable(outcome=unsupported)` raises it to "cannot do it".
        self._disabled_reason = "unavailable"
        # The decision breaker lives HERE, not in the worker (arch §13.8): the
        # decision call happens inside the service, the worker does not see it;
        # "one breaker point per slot" is preserved — the `SlotBreaker` class is
        # the same.
        #
        # Release 3.9.0 (LSBN-S-13): the `decision` operation is a full operation of
        # the registry — its own breaker with its own thresholds. In the `operation`
        # mode the thresholds are the per-operation ones (`DECISION_PAUSE_AFTER`/
        # `DECISION_PAUSE_SEC` — the env prefix stays `DECISION_*`), else the global
        # base; in the `legacy` mode the slot's own base is used (as before).
        if operation_mode(settings):
            self._breaker = SlotBreaker(
                SLOT_DECISION,
                pause_sec("decision", settings),
                open_after=pause_after("decision", settings),
            )
        else:
            self._breaker = SlotBreaker(
                SLOT_DECISION,
                settings.slot_pause_sec,
                open_after=settings.slot_pause_after,
            )
        self._answers = 0
        self._fallbacks = 0
        self._by_reason: dict[str, int] = {reason: 0 for reason in DECISION_REASONS}
        # LSBN-F-37 (review): a `choice` answer without `confidence` is judged by the
        # bare probability — legal, but reported ONCE per service (see
        # `_warn_bare_probability`), so the fallback is visible in the log.
        self._bare_probability_warned = False

    # --- state ----------------------------------------------------------------

    def can_decide(self) -> bool:
        """May we ask for a decision (enabled and not suppressed by the startup check)."""
        return self._enabled and self._ok and self._llm is not None

    @property
    def breaker(self) -> SlotBreaker:
        """The operation's own breaker (the state of the decision slot).

                    Release 3.9.0 (LSBN-S-13): the `decision` operation's
                    breaker — its own pause and its own thresholds (the
                    per-operation values in the `operation` mode, the global
                    base in the `legacy` mode). The worker reads the snapshot
                    of it for `/health.slots`.

        """
        return self._breaker

    def disable(self, *, outcome: str | None = None) -> None:
        """Suppress the slot: the startup check did not confirm "can do decision".

                    `outcome` — the `check_decision()` outcome (`DECISION_CHECK_*` constants
                    known to `main.py`). It defines the reason of the future switch:
                    `unsupported` — "the model cannot do decision" (the server does not
                    know the route / does not accept the question kind), any other outcome
                    (`unreachable`/`auth_failed`) and a call without an outcome — "the
                    slot is not there now" (`unavailable`). The reason is memorized here:
                    `decide()` returns it on the suppressed-slot branch so the operator
                    sees an honest picture (§13.6).
        """
        self._ok = False
        self._disabled_reason = (
            "unsupported"
            if outcome == DECISION_CHECK_UNSUPPORTED
            else "unavailable"
        )

    def close(self) -> None:
        """Close the client's HTTP pool (clean process shutdown)."""
        if self._llm is not None:
            self._llm.close()

    def snapshot(self) -> dict:
        """A slot state snapshot for /health (arch §6.4/§7.1): memory only.

                    `by_reason` always carries exactly three keys (zeros included), so
                    `fallbacks == sum(by_reason.values())` holds for an empty slot too
                    (§7.3/FR-6.3): the operator sees an honest picture of reasons.
        """
        return {
            "enabled": self._enabled,
            "ok": self._ok,
            "breaker": self._breaker.snapshot(),
            "answers": self._answers,
            "fallbacks": self._fallbacks,
            "by_reason": {
                reason: self._by_reason.get(reason, 0) for reason in DECISION_REASONS
            },
        }

    # --- question registry ----------------------------------------------------

    def questions_for(self, group: str) -> list[DecisionQuestion]:
        """Questions of a group ('judge'/'classify') from the registry; none — an empty list."""
        if self._registry is None:
            return []
        return list(self._registry.decision_questions.get(group, ()))

    def probe(self, operation: str | None = None) -> DecisionCheck:
        """A probe knock on the slot: `check_decision()`, success lifts the breaker (§6.4).

                    Writes the block-2 breaker events (`slot_probe`/`slot_resumed`/
                    `slot_paused`, §7.2): the decision breaker lives HERE, the worker does
                    not see it (§13.8), so the service journals it itself — the event
                    format is the same. `operation` — the job name for the `job` field
                    (none — the slot itself).
        """
        if self._llm is None:
            return DECISION_CHECK_UNSUPPORTED
        job = operation or SLOT_DECISION
        result = self._llm.check_decision()
        self._log_slot_event("slot_probe", job, result=result)
        outcome = _outcome_of_check(result)
        if outcome is None:
            self._breaker.note_success()
            self._log_slot_event(
                "slot_resumed", job, after_sec=self._breaker.pause_sec
            )
        elif self._breaker.note_failure(outcome):
            self._log_slot_event(
                "slot_paused",
                job,
                level=logging.WARNING,
                outcome=outcome.value,
                paused_until=self._breaker.snapshot()["paused_until"],
            )
        return result

    # --- decision --------------------------------------------------------------

    def decide(
        self,
        state: str,
        questions: list[DecisionQuestion],
        *,
        operation: str,
        note_id: int | None = None,
        older_id: int | None = None,
        candidate_id: int | None = None,
    ) -> DecisionResult:
        """Ask for a decision in one call; otherwise — `DecisionUnavailable` (§6.4).

                    Order: a disabled slot → the label ceiling → the slot breaker (no call);
                    then one `POST /v1/systemone` with all questions, response parsing,
                    threshold/zone check per question. Both paths are visible in the journal:
                    `decision_answer` (probabilities and duration) and `decision_fallback`
                    (the switch reason) — probabilities live ONLY there (FR-3.3).
        """
        ids: dict[str, object] = {
            "note_id": note_id,
            "older_id": older_id,
            "candidate_id": candidate_id,
        }
        if not self._settings.decision_enabled:
            # The flag-off path is NOT a switch (FR-4.4): no counters,
            # no event, no breaker. A defensive branch: normally the service is
            # not built at all with the flag off (`Services.decision is
            # None`), but the decision on this branch must be honest.
            detail = "decision-слот выключен флагом DECISION_ENABLED"
            raise DecisionUnavailable(detail, reason="unavailable")
        if not self.can_decide():
            # The slot is ENABLED but not there now: suppressed by the startup check
            # (`check_decision() != ok`) or `disable()`. The reason is the one
            # memorized by `disable(outcome)`: `unsupported` ("the model cannot do it")
            # on a contract check outcome, otherwise `unavailable` ("the slot
            # is not there now", an acceptance defect of 2026-10-12).
            reason = self._disabled_reason
            detail = (
                "decision-слот недоступен (погашен проверкой) — решения идут "
                "генеративным путём"
            )
            self._fallback(
                reason, operation=operation, detail=detail, ids=ids
            )
            raise DecisionUnavailable(detail, reason=reason)
        assert self._llm is not None  # can_decide() guarantees the client
        if any(len(q.labels) > self._settings.decision_max_labels for q in questions):
            # The options ceiling (arch §8): the node list exceeds the model's limit.
            detail = "вариантов в choice-вопросе больше DECISION_MAX_LABELS"
            self._fallback(
                "unsupported", operation=operation, detail=detail, ids=ids
            )
            raise DecisionUnavailable(detail, reason="unsupported")
        if self._breaker.is_paused():
            # The breaker is open and no knock happened — save the call (§6.4).
            detail = "разрыв decision-слота открыт — вызов пропущен"
            self._fallback(
                "unavailable", operation=operation, detail=detail, ids=ids
            )
            raise DecisionUnavailable(detail, reason="unavailable")
        if self._breaker.probe_due() and self.probe(operation) != DECISION_CHECK_OK:
            detail = "пробный стук decision-слота не прошёл — слот в паузе"
            self._fallback(
                "unavailable", operation=operation, detail=detail, ids=ids
            )
            raise DecisionUnavailable(detail, reason="unavailable")
        payload = [self._question_payload(q) for q in questions]
        started = time.monotonic()
        try:
            parsed = self._llm.decide(state, payload)
        except LLMError as exc:
            outcome = self._outcome_of_error(exc)
            if self._breaker.note_failure(outcome):
                self._log_slot_event(
                    "slot_paused",
                    operation,
                    level=logging.WARNING,
                    outcome=outcome.value,
                    paused_until=self._breaker.snapshot()["paused_until"],
                )
            reason = (
                "unsupported"
                if outcome is SlotOutcome.OFF_CONTRACT
                else "unavailable"
            )
            self._fallback(
                reason, operation=operation, detail=str(exc), ids=ids
            )
            raise DecisionUnavailable(str(exc), reason=reason) from exc
        duration_ms = int((time.monotonic() - started) * 1000)
        answers = {
            name: answer_from_parsed(name, item) for name, item in parsed.items()
        }
        for question in questions:
            answer = answers.get(question.name)
            if answer is None:
                self._breaker.note_failure(SlotOutcome.OFF_CONTRACT)
                detail = f"в ответе нет решения вопроса «{question.name}»"
                self._fallback(
                    "unsupported", operation=operation, detail=detail, ids=ids
                )
                raise DecisionUnavailable(detail, reason="unsupported")
            if not self._accepted(answer, question):
                detail = (
                    f"уверенность вопроса «{question.name}» в зоне "
                    "неопределённости"
                )
                self._fallback(
                    "low_confidence",
                    operation=operation,
                    detail=detail,
                    answer=answer,
                    ids=ids,
                )
                raise DecisionUnavailable(
                    detail, reason="low_confidence", answer=answer
                )
        self._answers += 1
        meta = self._meta(operation, duration_ms)
        self._log_answer(operation, questions, answers, meta, ids=ids)
        return DecisionResult(answers=answers, meta=meta)

    # --- internal --------------------------------------------------------------

    @staticmethod
    def _question_payload(question: DecisionQuestion) -> dict:
        """A question → the client's canonical dict (the server's field names live there)."""
        return {
            "name": question.name,
            "kind": question.kind,
            "text": question.text,
            "labels": tuple(question.labels),
            "criteria": dict(question.criteria) if question.criteria else None,
        }

    def _accepted(self, answer: DecisionAnswer, question: DecisionQuestion) -> bool:
        """Whether the chosen option's value passed the threshold and zone (§6.4).

                    `|value − threshold| < zone` → low_confidence; `value ≥ threshold +
                    zone` → accepted (the answer label); for `noul` the "opposite"
                    decision (`value ≤ threshold − zone`) is accepted too;
                    `choice`/`score` below the threshold — low_confidence (a
                    multi-option question has no "opposite").

                    LSBN-F-37: for `choice` the compared quantity is the answer's
                    CONFIDENCE, not the label probability — the live contract of
                    13.11.2026 defines it as the excess over chance,
                    `(p1 − 1/N) / (1 − 1/N)`, N — the number of labels. On a 20-label
                    question a bare probability of 0.5 is "just above a coin toss", so
                    choice has its own threshold; `noul`/`score` keep the probability
                    and `DECISION_THRESHOLD`. The zone is shared.

                    Which threshold it is and on which scale it lives — the single
                    answer is `_applied_threshold` (the same pair goes to the journal).
        """
        zone = self._settings.decision_uncertainty_zone
        threshold, scale = self._applied_threshold(answer, question)
        value = answer.confidence if scale == SCALE_CONFIDENCE else answer.probability
        if value is None:
            return False
        if abs(value - threshold) < zone:
            return False
        if value >= threshold + zone:
            return True
        return answer.kind == "noul" and value <= threshold - zone

    def _applied_threshold(
        self, answer: DecisionAnswer, question: DecisionQuestion
    ) -> tuple[float, str]:
        """The threshold ACTUALLY applied to the answer and the scale it lives on.

                    LSBN-F-37: the priority is question → knob. A question threshold
                    from the registry (`decision_questions.json`) is the strongest —
                    its author writes it on the scale of the question's kind; with no
                    question threshold the knob answers: `DECISION_CHOICE_THRESHOLD`
                    for `choice` judged by confidence, `DECISION_THRESHOLD` for all the
                    rest (`noul`, `score`, and `choice` without confidence).

                    Review defect of 13.11.2026: a `choice` answer WITHOUT `confidence`
                    has no confidence scale at all — the bare label probability is
                    compared (as it was before LSBN-F-37) and the default here is
                    `DECISION_THRESHOLD`, NEVER the choice one: the two scales must not
                    meet in one comparison. Such an answer is reported once per service
                    (`_warn_bare_probability`): it means the server did not send
                    `confidence` at all.

                    The returned pair is what the journal writes next to the answer
                    (`applied_threshold`/`applied_scale` in `decision_answer`): a reader
                    of the log must never guess which number the threshold belongs to.
        """
        by_confidence = answer.kind == "choice" and answer.confidence is not None
        if answer.kind == "choice" and answer.confidence is None:
            self._warn_bare_probability()
        scale = SCALE_CONFIDENCE if by_confidence else SCALE_PROBABILITY
        default_threshold = (
            self._settings.decision_choice_threshold
            if by_confidence
            else self._settings.decision_threshold
        )
        threshold = (
            question.threshold
            if question.threshold is not None
            else default_threshold
        )
        return threshold, scale

    def _warn_bare_probability(self) -> None:
        """Warn ONCE per service: a `choice` answer came without `confidence`.

                    LSBN-F-37 (review defect): the fallback to the bare probability is
                    legal but must be visible — otherwise a `choice` judged by 0.5 looks
                    exactly like a `choice` judged by the confidence threshold 0.6. One
                    line per process is enough: the case belongs to the server's
                    contract, not to a single answer.
        """
        if self._bare_probability_warned:
            return
        self._bare_probability_warned = True
        logging.getLogger("app").warning(
            "decision: choice-ответ без confidence — сравнение по голой probability "
            "и DECISION_THRESHOLD (шкалы не смешиваются)",
            extra={
                "event": "decision_scale_fallback",
                "slot": SLOT_DECISION,
                "scale": SCALE_PROBABILITY,
                "threshold": self._settings.decision_threshold,
            },
        )

    def _outcome_of_error(self, exc: LLMError) -> SlotOutcome:
        """A client error → the breaker outcome; 401/403 (native ollama) — AUTH."""
        outcome = classify_llm_error(exc)
        if outcome is not SlotOutcome.AUTH and getattr(exc, "status", None) in _AUTH_STATUSES:
            return SlotOutcome.AUTH
        return outcome

    def _meta(self, operation: str, duration_ms: int) -> ChatMeta:
        """Decision call metadata (§6.4): no generation — no output budget."""
        spec = self._llm.spec  # type: ignore[union-attr]
        return ChatMeta(
            slot=SLOT_DECISION,
            provider=spec.provider,
            model=spec.model,
            operation=operation,
            model_type=SLOT_DECISION,
            budget=0,
            budget_source="legacy",
            prompt_tokens=None,
            completion_tokens=None,
            reasoning_tokens=None,
            had_thinking=False,
            thinking_chars=0,
            finish_reason=None,
            truncated=False,
            duration_ms=duration_ms,
            base_url=spec.base_url,
            config_source="slot",
        )

    def _fallback(
        self,
        reason: str,
        *,
        operation: str,
        detail: str | None = None,
        answer: DecisionAnswer | None = None,
        ids: Mapping[str, object] | None = None,
    ) -> None:
        """Account for the switch: counters (§7.3/FR-6.3) + a journal event.

                    The journal is written by `DecisionService` itself (arch §7.2/§13.6):
                    it is the only one that knows the question, the probability and the
                    switch reason. Only numbers and a `detail` fragment go into the event —
                    no response bodies there (§7.2).

                    `reason` is the SINGLE point of counter increments and event writes;
                    the reason name passes the `DECISION_REASONS` whitelist (§13.6): a
                    foreign name is a development error, not a fourth counter.
        """
        if reason not in DECISION_REASONS:
            raise ValueError(
                f"причина переключения decision вне словаря: {reason!r}"
            )
        self._fallbacks += 1
        self._by_reason[reason] += 1
        fields: dict[str, object] = {
            "operation": operation,
            "reason": reason,
            "probability": answer.probability if answer is not None else None,
            "confidence": answer.confidence if answer is not None else None,
            "detail": detail,
        }
        if ids:
            fields.update(ids)
        self._log("decision_fallback", fields)

    def _log_answer(
        self,
        operation: str,
        questions: list[DecisionQuestion],
        answers: dict[str, DecisionAnswer],
        meta: ChatMeta,
        *,
        ids: Mapping[str, object] | None = None,
    ) -> None:
        """The `decision_answer` event (§7.2): probabilities — ONLY in the journal.

                    Fields: `slot`, `operation`, question names, answers (`kind`/`label`/
                    `probability`/`confidence`), per answer — `applied_threshold` and
                    `applied_scale` (`confidence`|`probability`), the model and
                    `duration_ms`. Probabilities are not written to the DB at all
                    (FR-3.3).

                    LSBN-F-37 (review defect): the event used to carry ONE `threshold` —
                    always `DECISION_THRESHOLD`, whatever was really compared. Now the
                    applied pair travels WITH the answer: one call may ask several
                    questions of different kinds and each is judged on its own scale, so
                    a single number at the top of the event would lie again.
        """
        by_name = {question.name: question for question in questions}
        logged: dict[str, dict[str, object]] = {}
        for name, answer in answers.items():
            entry: dict[str, object] = {
                "kind": answer.kind,
                "label": answer.label,
                "probability": answer.probability,
                "confidence": answer.confidence,
            }
            question = by_name.get(name)
            if question is not None:
                threshold, scale = self._applied_threshold(answer, question)
                entry["applied_threshold"] = threshold
                entry["applied_scale"] = scale
            logged[name] = entry
        fields: dict[str, object] = {
            "operation": operation,
            "questions": [question.name for question in questions],
            "answers": logged,
            "model": meta.model,
            "duration_ms": meta.duration_ms,
        }
        if ids:
            fields.update(ids)
        self._log("decision_answer", fields)

    @staticmethod
    def _log(event: str, fields: Mapping[str, object]) -> None:
        """A journal JSON event: `event` + `slot` + fields as available (None — skipped)."""
        payload: dict[str, object] = {"event": event, "slot": SLOT_DECISION}
        for key, value in fields.items():
            if value is not None:
                payload[key] = value
        logging.getLogger("app").info(event, extra=payload)

    @staticmethod
    def _log_slot_event(
        event: str, job: str, *, level: int = logging.INFO, **fields: object
    ) -> None:
        """A decision-slot breaker/probe event (block 2, §7.2): `slot`=decision.

                    The format is the same as the worker's (`event`/`slot`/`job` + fields
                    as available), so the decision breaker reads on par with other slots.
        """
        payload: dict[str, object] = {
            "event": event,
            "slot": SLOT_DECISION,
            "job": job,
        }
        for key, value in fields.items():
            if value is not None:
                payload[key] = value
        logging.getLogger("app").log(level, "decision slot event", extra=payload)
