"""ClassificationService — background grooming (Phase 10, Step 4; REQUIREMENTS §5.7).

    The default-note classifier: the summary slot — the same model as
    summarization (SUMMARY_MODEL / SUMMARY_BASE_URL), a separate prompt,
    JSON output, a small `num_predict`. The output is two tagging fields:
    `hint_path` (a full path of 1..3 slugs or null = generic), `confidence` (0–1).
    Tagging parameters are internal data,
    NOT part of the MCP contracts (not tags-2.0): the client model sees only
    registry nodes.

    Phase 11: the HTTP transport is factored out into `LLMClient`
    (app/services/llm_client.py) — the classifier shares the summary slot client
    (provider `ollama` | `openai`, decision #1: classification lives in one slot
    with summarization). The service no longer keeps httpx or builds payloads
    itself: system+user, non-stream, a small `num_predict` and an explicit
    `think: false` go through `LLMClient.chat`; the F1 queue (ollama_slot —
    ollama provider only), Bearer when a key is set, and response parsing
    (`message.content`, reasoning fields discarded) live inside the client.
    keep_alive is not sent to anyone (decision #6: the server governs the model,
    OLLAMA_KEEP_ALIVE). The read timeout — the summary slot timeout
    (SUMMARY_TIMEOUT_SEC): since v2.1 the client is one per slot, a separate
    CLASSIFIER_TIMEOUT_SEC is no longer set.

    Invoked only by the background worker after summarizing a default note
    (sequentially, not in parallel — ollama_slot serializes calls to one
    base_url). Failure — `ClassificationError`: the note stays in default,
    `classified_at` is not set, a retry comes after `memory_update` (anti-looping
    §5.7). `last_attempt_ok` — like the summarizer (NFR-4, /health).

    The single hint path (`hint_path`) is validated via the registry's
    `validate_path` (slugs of latin-digits-hyphen, 1..3 segments, nesting under
    `default/...` forbidden); a violation — invalid tagging →
    `ClassificationError`, the note stays in default (a retry after update).

    Prompt: the user message carries the known nodes (path: description) —
    "if an existing one fits, use it; if specific and nothing fits — a new slug
    (latin-digits-hyphen); generic — null". Slug consistency grows together with
    the list of known nodes. The text comes from the prompt registry (Phase 11,
    decision #7): `classifier_system` is a baked-in constant, never created by
    files.

    Unit tests — transport without the network: `transport` accepts
    httpx.MockTransport or a handler (passed into the slot's LLMClient).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol

import httpx

from app.config import Settings
from app.services.decision import SLOT_DECISION, DecisionService, DecisionUnavailable
from app.services.llm_client import ChatMeta, LLMClient, LLMError, SlotSpec
from app.services.namespaces import NamespaceService, NamespaceValidationError
from app.services.prompts import PromptRegistry
from app.services.slot_outcome import SlotOutcome, classify_llm_error

# Call parameters not configurable via env (ARCH §4.7): a small JSON-tagging
# budget; think:false — an explicit no-reasoning switch (faster). #49: this is
# the classifier's legacy ceiling ("classify" in slot_budget.LEGACY_OUTPUT_BUDGET),
# applied when the context is unknown; with a declared one — the auto-budget.
CLASSIFIER_NUM_PREDICT = 768

# The fallback label of the tagging decision question (registry, arch §6.4): a
# note not bound to a node → `hint_path is None`.
GENERIC_LABEL = "generic"

# The minimum candidates of the decision model's `choice` question (live contract
# of `/v1/systemone`, verified 2026-10-10): the server requires 2–26 options in
# `criteria`, otherwise 400. One candidate — nothing to choose from: the decision
# is NOT applicable, no call (arch §13.6).
DECISION_MIN_CANDIDATES = 2


class ClassificationError(RuntimeError):
    """Classification failed: the server is unavailable or the response is invalid.

            `outcome` — the single call-outcome type of a slot (#39, arch §6.4): a
            client failure — `classify_llm_error`; "marker/structure/JSON not found" —
            explicitly `OFF_CONTRACT` (the default). The worker writes the outcome
            to the journal.
    """

    def __init__(
        self,
        message: str,
        *,
        outcome: SlotOutcome = SlotOutcome.OFF_CONTRACT,
        meta: ChatMeta | None = None,
    ) -> None:
        super().__init__(message)
        self.outcome = outcome
        # Metadata of a completed call (#49): a failure after the model's response
        # (tagging not parsed) carries them to the worker.
        self.meta = meta


@dataclass(frozen=True)
class Classification:
    """Tagging of a default note (internal data, §5.7)."""

    hint_path: str | None  # full path of 1..3 slugs or null (generic)
    confidence: float  # 0..1
    meta: ChatMeta | None = None  # #49: call metadata (filled by the service)


class Classifier(Protocol):
    """The classifier contract: it knows nothing about the implementation nature.

            Implementations: ClassificationService (the live summary slot) and a test
            fake (tests/fakes.py) — the worker depends on the protocol, not the class.
    """

    def classify(self, text: str, known_nodes: list[dict[str, Any]]) -> Classification:
        """Tag a default note against the known registry nodes."""
        ...

    def close(self) -> None:
        """Close resources (interface-compatible with EmbeddingService)."""
        ...


class ClassificationService:
    """Tag a default note via the summary slot's chat (non-stream)."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport
        | Callable[[httpx.Request], httpx.Response]
        | None = None,
        *,
        llm: LLMClient | None = None,
        registry: PromptRegistry | None = None,
        decider: DecisionService | None = None,
    ) -> None:
        self._settings = settings
        self._prompts = registry if registry is not None else PromptRegistry()
        # LSBN-F-12: the decision slot (None — disabled, v3.1.0 behavior).
        self._decider = decider
        # DI builds (the shared summary slot client) or a dedicated client with
        # a transport injection for unit tests (MockTransport). The `classify`
        # operation (LSBN-F-34): flat env CLASSIFY_* overrides the tagging
        # configuration on top of the summary slot; the decision fallback
        # (LSBN-F-12) stays on its own DECISION_*.
        self._llm = (
            llm
            if llm is not None
            else LLMClient(
                SlotSpec.for_operation("classify", settings)[0], transport=transport
            )
        )
        # None — no attempts yet (health does not lie before the first data).
        self.last_attempt_ok: bool | None = None
        # Metadata of the last call (#49/#50): diagnostics in /health.slots
        # without reading the journal (None — no call yet). `Classification`
        # already carries the metadata (#49) — the service additionally stores it here.
        self.last_meta: ChatMeta | None = None

    def classify(
        self, text: str, known_nodes: list[dict[str, Any]]
    ) -> Classification:
        """Tag a default note against the known registry nodes.
        """
        if not text or not text.strip():
            raise ValueError("classify: пустой текст заметки")
        if self._decider is not None:
            labels = self._decision_labels(known_nodes)
            if len(labels) < DECISION_MIN_CANDIDATES:
                # Nothing to choose from: the decision is NOT applicable — this is not one of
                # the three switch reasons (§13.6). No call, the switch
                # counters and the decision-slot breaker are untouched; go straight
                # down the old generative path.
                self._log_decision_skipped(len(labels))
            else:
                try:
                    return self._classify_decision(text, known_nodes, labels)
                except DecisionUnavailable as exc:
                    if not self._settings.decision_fallback:
                        self.last_attempt_ok = False
                        raise ClassificationError(
                            str(exc), outcome=SlotOutcome.UNAVAILABLE
                        ) from exc
        return self._classify_generative(text, known_nodes)

    @staticmethod
    def _decision_labels(known_nodes: list[dict[str, Any]]) -> list[str]:
        """Labels of the tagging `choice` question: node paths of the map + the fallback `generic`."""
        return [node["path"] for node in known_nodes] + [GENERIC_LABEL]

    def _classify_decision(
        self, text: str, known_nodes: list[dict[str, Any]], labels: list[str]
    ) -> Classification:
        """The decision slot's verdict: label → `hint_path`, probability → `confidence`.

                    The question comes from the registry (`decider.questions_for("classify")`);
                    labels come from `classify` (paths of known nodes + the fallback
                    `generic` → `hint_path is None`), state — the note + nodes
                    (`path: description`, as today). The call happens only at
                    `len(labels) >= DECISION_MIN_CANDIDATES`: otherwise the decision is
                    not applicable and this code is not reached. The same question serves
                    the `warnings` job ("a note is not in its place"): it walks through
                    the same `classify` (arch §13.6, FR-8.4).
        """
        assert self._decider is not None
        questions = self._decider.questions_for("classify")
        if not questions:
            raise DecisionUnavailable(
                "в реестре нет вопроса разметки узла", reason="unsupported"
            )
        question = replace(questions[0], labels=tuple(labels))
        result = self._decider.decide(
            self._user_message(text, known_nodes), [question], operation="classify"
        )
        answer = result.answers.get(question.name)
        if answer is None or answer.label is None:
            raise DecisionUnavailable(
                f"в ответе нет метки вопроса «{question.name}»",
                reason="unsupported",
            )
        hint_path: str | None = None
        if answer.label != GENERIC_LABEL:
            # The path is validated by the same `validate_path` as in `_parse`
            # (slugs of latin-digits-hyphen, 1..3 segments, not under default/...).
            try:
                hint_path = NamespaceService(self._settings).validate_path(
                    answer.label
                )
            except NamespaceValidationError as exc:
                raise DecisionUnavailable(
                    f"hint_path: {exc}", reason="unsupported"
                ) from exc
        confidence = (
            answer.probability
            if answer.probability is not None
            else float(answer.confidence or 0.0)
        )
        self.last_attempt_ok = True
        self.last_meta = result.meta
        return Classification(
            hint_path=hint_path, confidence=confidence, meta=result.meta
        )

    def _classify_generative(
        self, text: str, known_nodes: list[dict[str, Any]]
    ) -> Classification:
        """The old generative path (JSON tagging of the summary slot)."""
        try:
            chat = self._llm.chat_result(
                self._prompts.classifier_system,
                self._user_message(text, known_nodes),
                # #49: num_predict=None — the auto-budget "as much as fits"; with an
                # unknown context the legacy ceiling applies
                # CLASSIFIER_NUM_PREDICT ("classify" = 768).
                num_predict=None,
                think=False,  # JSON tagging without reasoning (faster)
                operation="classify",
            )
        except LLMError as exc:
            # The client text is preserved (HTTP status/key hint, decision #4).
            # The call outcome (#39, arch §6.4) — a single type at the service boundary.
            self.last_attempt_ok = False  # the degradation becomes visible in /health
            self.last_meta = exc.meta  # call metadata for diagnostics
            raise ClassificationError(
                str(exc), outcome=classify_llm_error(exc), meta=exc.meta
            ) from exc
        try:
            result = self._parse(chat.content)
        except ClassificationError as exc:
            self.last_attempt_ok = False  # invalid tagging is also a failure
            self.last_meta = chat.meta
            # The call happened — the metadata goes to the worker along with the failure.
            raise ClassificationError(
                str(exc), outcome=exc.outcome, meta=chat.meta
            ) from exc
        self.last_attempt_ok = True
        self.last_meta = chat.meta
        return Classification(
            hint_path=result.hint_path,
            confidence=result.confidence,
            meta=chat.meta,
        )

    def close(self) -> None:
        """Close the HTTP pool (clean process shutdown)."""
        self._llm.close()

    @staticmethod
    def _log_decision_skipped(candidates: int) -> None:
        """The `decision_skipped` event: the decision is not applicable, NO call happened (§13.6).

                    Written when the `choice` question has fewer candidates than
                    `DECISION_MIN_CANDIDATES` (nothing to choose from). This is not a
                    switch: the `/health.decision` counters (`fallbacks`/`by_reason`) and
                    the slot breaker do not move, and the event carries no probabilities
                    (FR-3.3).
        """
        logging.getLogger("app").info(
            "decision_skipped",
            extra={
                "event": "decision_skipped",
                "slot": SLOT_DECISION,
                "operation": "classify",
                "reason": "single_candidate",
                "candidates": candidates,
            },
        )

    # --- internal -------------------------------------------------------------

    @staticmethod
    def _user_message(text: str, known_nodes: list[dict[str, Any]]) -> str:
        """The note + known nodes (path: description) — the choice context."""
        nodes = "\n".join(
            f"- {node['path']}: {node['description']}" for node in known_nodes
        )
        if not nodes:
            nodes = "(нет известных узлов)"
        return f"Заметка:\n{text}\n\nИзвестные узлы:\n{nodes}"

    def _parse(self, content: str) -> Classification:
        """Extract and validate the JSON tagging; violations — ClassificationError."""
        data = self._extract_json(content)
        if not isinstance(data, dict):
            raise ClassificationError("не-JSON объект в ответе классификатора")
        hint_path = data.get("hint_path")
        confidence = data.get("confidence")
        if hint_path is not None and not isinstance(hint_path, str):
            raise ClassificationError("hint_path: ожидается строка или null")
        if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            raise ClassificationError("confidence: ожидается число 0..1")
        if hint_path is not None:
            # The full path is validated as a registry node path: slugs
            # (latin-digits-hyphen), 1..3 segments, nesting under
            # `default/...` is forbidden (default is a system node).
            try:
                hint_path = NamespaceService(self._settings).validate_path(hint_path)
            except NamespaceValidationError as exc:
                raise ClassificationError(f"hint_path: {exc}") from exc
        return Classification(hint_path=hint_path, confidence=float(confidence))

    @staticmethod
    def _extract_json(content: str) -> Any:
        """Extract the JSON object from the response (robust to code fences and wrappers).

                    The thinking/response wrapper is cut off (the ornith:35b model wraps
                    JSON in thinking/response tags, where reasoning may contain curly
                    braces) — before the json.loads attempt.
        """
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        text = ClassificationService._strip_reasoning_wrapper(text)
        try:
            return json.loads(text)
        except ValueError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except ValueError:
                pass
        raise ClassificationError("не удалось извлечь JSON из ответа классификатора")

    @staticmethod
    def _strip_reasoning_wrapper(text: str) -> str:
        """Cut the thinking/response wrapper if the model added one.

                    Walk from the end: if there is a `response` marker line (the last of
                    {response, <response>, </response>}) — return everything after it.
                    Otherwise remove the marker lines {thinking, <thinking>, </thinking>,
                    response, <response>, </response>}. If nothing was cut — return the
                    original text.
        """
        lines = text.splitlines()
        response_markers = {"response", "<response>", "</response>"}
        all_markers = response_markers | {"thinking", "<thinking>", "</thinking>"}
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().lower() in response_markers:
                return "\n".join(lines[i + 1 :]).strip()
        stripped = [ln for ln in lines if ln.strip().lower() not in all_markers]
        result = "\n".join(stripped).strip()
        return result if result != text.strip() else text