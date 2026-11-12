"""JudgeService — the LLM dedup judge (Phase 8, Stage 3.1).

    The system's third external call: the judge slot's chat, the JUDGE_MODEL model
    (`ornith-1.5:35b`). It decides on a pair of texts "duplicate / not duplicate" —
    the same thought retold in other words. Background dedup (Phase 8, Stage 2)
    stays the prefilter: the judge is asked only about cosine candidates (top-N,
    DEDUP_CANDIDATE_*), not about all notes (the Phase 8 brief, the "cosine as a
    prefilter" decision).

    Phase 11: the HTTP transport is factored out into `LLMClient`
    (app/services/llm_client.py) — one client per judge slot (provider `ollama` |
    `openai`, per-slot address/model/key from Settings, decision #1; the env block
    renamed DEDUP_JUDGE_* → JUDGE_*). The service no longer keeps httpx or builds
    payloads itself: system+user, non-stream, `judge_num_predict` and think go
    through `LLMClient.chat`; the F1 queue (ollama_slot — ollama provider only),
    Bearer when a key is set, and response parsing (`message.content`, reasoning
    fields discarded) live inside the client. keep_alive is not sent to anyone
    (decision #6: the server governs the model, OLLAMA_KEEP_ALIVE).

    Model verification (2026-08-30, the Phase 8 brief): the verdict in
    `message.content` is marked bold — `**DUPLICATE**` / `**NOT DUPLICATE**`
    (markdown `**` is stripped, then a substring search; "NOT DUPLICATE" first —
    otherwise it would match "DUPLICATE"). Answers without a verdict — a failure
    (safer "not to merge"): the worker keeps both notes and retries on back-off
    (NFR-3).

    `think` in the request — a history of decisions: at Stage 3.1 the judge did not
    think (fast); after the Phase 8 end-to-end test (2026-08-30) reasoning was
    enabled — a thoughtless verdict produced false DUPLICATES on long single-project
    notes (the thinking judge answered correctly on the same pair, 46 s). With
    JUDGE_THINK=false (the code default) the body carries an explicit
    `"think": false`; with true (prod) the field is not sent, and an incoming
    `message.thinking` is discarded anyway (only content is read — as in the
    summarizer); the thinking+verdict budget is JUDGE_NUM_PREDICT (prod 1024:
    a measurement of ~1200 tokens, done_reason=stop).

    Key decisions (modeled on SummaryService, Stage 2.2):
    - A synchronous httpx transport inside the client — the worker runs it in
      `asyncio.to_thread`; connect 2 s (LAN), read — JUDGE_TIMEOUT_SEC. The code
      default (30 s from the brief) is outdated: the Phase 8 E2E showed 46 s of a
      thinking judge on a long pair (a prompt of ~2.9k tokens), the worst live call
      — ~121 s — prod sets 300 s.
      The ollama_slot queue (F1) guarantees: the read timeout ticks only from the
      actual request send — waiting behind a merge generation does not eat the
      budget.
    - No retries — the worker decides on repeats (queue back-off, NFR-3).
    - Failure — `JudgeError` and only it: non-200 HTTP, non-JSON, empty content,
      an unrecognized verdict, transport unavailable. The client's error text is
      preserved (including the {SLOT}_API_KEY hint on an auth failure, decision #4).
    - `last_attempt_ok` (NFR-4, like the summarizer) — the source of the /health
      status; the slot startup check (main.py, decision #5) initializes it before
      the first real call.
    - Prompts — from the registry (Phase 11, decision #7): the system `judge_system`
      is editable (the "DUPLICATE"/"NOT DUPLICATE" markers are validated at startup —
      the verdicts are parsed by exactly them), the user template `judge_user` is a
      baked-in constant; `registry` is injected by the build (build_services),
      without DI — the registry's built-in texts.
    - DI: unit tests — httpx.MockTransport via `transport`; fake judges
      (tests/fakes.py) — for integration tests in dedup (Task 3.2).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import Settings
from app.services.decision import DecisionService, DecisionUnavailable
from app.services.llm_client import ChatMeta, LLMClient, LLMError, SlotSpec
from app.services.prompts import PromptRegistry
from app.services.slot_outcome import SlotOutcome, classify_llm_error


class JudgeError(RuntimeError):
    """The judge gave no verdict: the server is unavailable or the response is invalid.

            `outcome` — the single call-outcome type of a slot (#39, arch §6.4): a
            client failure is classified by `classify_llm_error`, and "the verdict is
            not recognized" — explicitly `OFF_CONTRACT` (the default). The worker
            writes the outcome to the journal.
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
        # (the verdict not recognized) carries them to the worker — the failure is visible in the journal
        # with the same call fields as success (§7.2).
        self.meta = meta


@dataclass(frozen=True)
class JudgeResult:
    """The judge's result together with the call metadata (#49, §6.3)."""

    verdict: bool
    meta: ChatMeta


class Judge(Protocol):
    """The dedup judge contract: it knows nothing about the implementation nature.

            Implementations: JudgeService (the live judge slot) and test fakes
            (tests/fakes.py) — the worker depends on the protocol, not the client
            class.
    """

    def judge(self, text_new: str, text_candidate: str) -> JudgeResult:
        """A verdict on a pair of texts: True — duplicate, False — not duplicate.

                    Inability to give a verdict — a failure (safer "not to merge"): the
                    worker keeps both notes and retries on back-off (NFR-3).
        """
        ...

    def probe(self) -> str:
        """A cheap probe knock on the slot (#39, arch §6.4): `check()` without generation;
                    the outcome — a CheckResult from llm_client.
        """
        ...

    def close(self) -> None:
        """Close resources (interface-compatible with the other clients)."""
        ...


#: The split verdict (Oleg's decision of 2026-11-11; batched 2026-11-12): the
#: source note and the fact-notes it was split into go to the judge with one
#: question - was anything from the source lost. Not a prompt file: a
#: hardwired constant like the other judge strings. The wording covers BOTH
#: verification shapes: the whole source (a short note — one batch) and a
#: SEGMENT of a long note (the source is cut into overlapping segments, each
#: verified with its parts separately — the whole source plus all parts in one
#: prompt exploded on real notes: 75 fragments of one source, a giant prompt,
#: timeouts instead of a verdict).
SPLIT_VERDICT_SYSTEM = (
    "You check how a long note was decomposed into notes. You are given a part "
    "of the SOURCE (the whole source if it is short) and the NOTES made from "
    "it. Check ONLY one thing: whether every fact of this source part - names, "
    "numbers, dates, statuses, configs, paths, decisions - is present in the "
    "notes, and nothing is lost or distorted. Rewording, splitting into pieces "
    "of different sizes and changing the shape of the text are NOT losses; a "
    "missing or changed fact IS. Answer with exactly one line, no explanations: "
    "COMPLETE - if nothing is lost; MISSING: <what exactly is lost> - otherwise."
)


def plan_split_verification(
    source: str,
    parts: list[str],
    *,
    segment_chars: int,
    overlap_chars: int,
) -> list[tuple[str, list[str]]]:
    """The (source segment, its parts) batches for the split verdict.

            A source not longer than `segment_chars` — ONE batch (the former
            behavior verbatim: the whole source + all parts). A longer source
            is cut into segments of `segment_chars` with a border cushion (at
            most a quarter of the segment, see the code — a border fact is
            judged with its full sentence on one of the sides), and every part
            is assigned to the segment holding its estimated position — the
            cumulative share of the parts' length.

            A segment may get NO parts (the model dropped that stretch of the
            source): the batch still goes to the judge with an empty note list
            — a MISSING verdict is how the loss surfaces.

            Raises:
                ValueError: an empty source/parts list or a non-positive
                    `segment_chars` (the caller validates, the pure function
                    stays honest).
    """
    if not source or not source.strip():
        raise ValueError("plan_split_verification: пустой источник")
    if not parts:
        raise ValueError("plan_split_verification: пустой список частей")
    if segment_chars < 1:
        raise ValueError(
            f"segment_chars должен быть ≥ 1, получено {segment_chars}"
        )
    if len(source) <= segment_chars:
        return [(source, list(parts))]

    # The border cushion: at most a quarter of the segment (and never the
    # whole `overlap_chars` — a note limit close to the segment size would
    # otherwise make the windows crawl: overlap → segment-1, no progress).
    overlap = max(0, min(overlap_chars, segment_chars // 4))
    segments: list[tuple[int, int]] = []
    start = 0
    total = len(source)
    while True:
        end = min(start + segment_chars, total)
        segments.append((start, end))
        if end >= total:
            break
        start = end - overlap

    parts_chars = sum(len(part) for part in parts)
    total_chars = max(parts_chars, 1)
    batches: list[list[str]] = [[] for _ in segments]
    seen = 0.0
    for part in parts:
        # The part's estimated midpoint in the source: the parts' cumulative
        # length share mapped onto the source length.
        midpoint = (seen + len(part) / 2.0) / total_chars * total
        seen += len(part)
        index = next(
            (j for j, (_, end) in enumerate(segments) if midpoint < end),
            len(segments) - 1,
        )
        batches[index].append(part)
    return [(source[s:e], batches[k]) for k, (s, e) in enumerate(segments)]


class JudgeService:
    """The "duplicate / not duplicate" verdict via the judge slot's chat (non-stream)."""

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
        # DI builds (the shared judge slot client) or a dedicated client with
        # a transport injection for unit tests (MockTransport). The `judge`
        # operation (LSBN-F-34): its env is the slot's env — the chain adds no
        # layer, the spec equals for_judge; the operation override acts
        # only on this generative path (the fallback decision slot has
        # its own DECISION_*).
        self._llm = (
            llm
            if llm is not None
            else LLMClient(
                SlotSpec.for_operation("judge", settings)[0], transport=transport
            )
        )
        # None — no attempts yet (health does not lie before the first data).
        self.last_attempt_ok: bool | None = None
        # Metadata of the last call (#49/#50): diagnostics in /health.slots
        # without reading the journal (None — no call yet).
        self.last_meta: ChatMeta | None = None

    def judge(self, text_new: str, text_candidate: str) -> JudgeResult:
        """A verdict on a pair of texts: True — duplicate, False — not duplicate.

                    Inability to give a verdict — a failure (safer "not to merge"): the
                    worker keeps both notes and retries on back-off (NFR-3).
        """
        if not text_new or not text_new.strip():
            raise ValueError("judge: пустая заметка недопустима")
        if not text_candidate or not text_candidate.strip():
            raise ValueError("judge: пустая заметка недопустима")
        if self._decider is not None:
            try:
                return self._judge_decision(text_new, text_candidate)
            except DecisionUnavailable as exc:
                if not self._settings.decision_fallback:
                    # The #39 contract: no generative verdict is taken — the task waits.
                    self.last_attempt_ok = False
                    raise JudgeError(
                        str(exc), outcome=SlotOutcome.UNAVAILABLE
                    ) from exc
        return self._judge_generative(text_new, text_candidate)

    def _judge_decision(
        self, text_new: str, text_candidate: str
    ) -> JudgeResult:
        """The decision slot's verdict: `label == "true"` → the DUPLICATE verdict.

                    The journal (`decision_answer`/`decision_fallback`) lives inside
                    `DecisionService` (arch §7.2/§13.6): the consumer does not know by which
                    slot type the decision was obtained (FR-5.3).
        """
        assert self._decider is not None
        questions = self._decider.questions_for("judge")
        if not questions:
            raise DecisionUnavailable(
                "в реестре нет вопроса судьи дедупа", reason="unsupported"
            )
        state = self._prompts.judge_user.format(
            text_new=text_new, text_candidate=text_candidate
        )
        result = self._decider.decide(state, questions, operation="judge")
        answer = result.answers.get(questions[0].name)
        if answer is None or answer.label is None:
            raise DecisionUnavailable(
                f"в ответе нет вердикта вопроса «{questions[0].name}»",
                reason="unsupported",
            )
        self.last_attempt_ok = True
        self.last_meta = result.meta
        return JudgeResult(verdict=answer.label == "true", meta=result.meta)

    def _judge_generative(
        self, text_new: str, text_candidate: str
    ) -> JudgeResult:
        """The old generative path (the `**DUPLICATE**`/`**NOT DUPLICATE**` markers)."""
        try:
            result = self._llm.chat_result(
                self._prompts.judge_system,
                self._prompts.judge_user.format(
                    text_new=text_new, text_candidate=text_candidate
                ),
                # The budget — the operation chain (JUDGE_NUM_PREDICT; 0 = auto, #49).
                operation="judge",
            )
        except LLMError as exc:
            # The client text is preserved: HTTP status, the slot address, the
            # {SLOT}_API_KEY hint on an auth failure (decision #4). The call outcome
            # (#39, arch §6.4) — a single outcome type at the service boundary.
            self.last_attempt_ok = False  # the degradation becomes visible in /health
            self.last_meta = exc.meta  # call metadata for diagnostics
            raise JudgeError(
                str(exc), outcome=classify_llm_error(exc), meta=exc.meta
            ) from exc
        try:
            verdict = self._verdict(result.content)
        except JudgeError as exc:
            self.last_attempt_ok = False  # an unrecognized verdict is also a failure
            self.last_meta = result.meta
            # The call happened — the metadata goes to the worker along with the failure.
            raise JudgeError(str(exc), outcome=exc.outcome, meta=result.meta) from exc
        self.last_attempt_ok = True
        self.last_meta = result.meta
        return JudgeResult(verdict=verdict, meta=result.meta)

    def verify_split(self, source: str, parts: list[str]) -> str | None:
        """The split verdict: None — nothing is lost, else the judge's reason.

            Oleg's decision of 2026-11-11: the split is accepted only after the
            judge looks at the source and the new notes and confirms that
            everything was decomposed and nothing was missed. Always the
            generative path: the decision slot is schema-bounded (8k context,
            64 KiB input) and cannot take a long note plus its parts.

            Batched (2026-11-12): the source plus ALL parts in ONE call exploded
            on real notes — the "one part per fact" era fragmented one note into
            75 scraps, the judge's prompt became a wall of text, the verdict did
            not arrive. Now the source is verified segment by segment
            (`plan_split_verification`, the segment size —
            `settings.split_verify_segment_chars`): every batch answers
            COMPLETE/MISSING, the FIRST MISS is the verdict. The batches are
            sequential — one slot call at a time, the slot delimiter of the
            migration job paces the load.
        """
        if not source or not source.strip():
            raise ValueError("verify_split: пустой источник")
        if not parts:
            raise ValueError("verify_split: пустой список частей")
        batches = plan_split_verification(
            source,
            parts,
            segment_chars=int(self._settings.split_verify_segment_chars),
            overlap_chars=int(self._settings.max_note_chars),
        )
        for segment, batch_parts in batches:
            verdict = self._verify_split_batch(segment, batch_parts)
            if verdict is not None:
                return verdict
        return None

    def _verify_split_batch(self, segment: str, parts: list[str]) -> str | None:
        """One verification batch: a source segment + the parts made from it."""
        listing = "\n\n".join(
            f"NOTE {number}:\n{text}" for number, text in enumerate(parts, start=1)
        )
        if not parts:
            listing = "(no notes were provided for this part of the source)"
        user_text = f"SOURCE:\n{segment}\n\nNOTES:\n{listing}"
        try:
            result = self._llm.chat_result(
                SPLIT_VERDICT_SYSTEM, user_text, operation="judge"
            )
        except LLMError as exc:
            self.last_attempt_ok = False
            self.last_meta = exc.meta
            raise JudgeError(
                str(exc), outcome=classify_llm_error(exc), meta=exc.meta
            ) from exc
        self.last_meta = result.meta
        text = (result.content or "").strip()
        upper = text.upper()
        if "MISSING" in upper:
            self.last_attempt_ok = True
            return text[:500] or "MISSING"
        if "COMPLETE" in upper:
            self.last_attempt_ok = True
            return None
        self.last_attempt_ok = False
        raise JudgeError(
            "verify_split: нераспознанный вердикт судьи", outcome=SlotOutcome.OFF_CONTRACT
        )

    def close(self) -> None:
        """Close resources (interface-compatible with the other clients)."""
        self._llm.close()

    def probe(self) -> str:
        """A cheap probe knock on the slot (#39, arch §6.4): `check()` without generation;
                    the outcome — a CheckResult from llm_client.
        """
        return self._llm.check()

    # --- internal -------------------------------------------------------------

    @staticmethod
    def _verdict(content: str) -> bool:
        """Parse the verdict from content (the brief's format: `**DUPLICATE**/…`).

                    Markdown-bold is stripped, case does not matter; search "NOT DUPLICATE"
                    first (the substring contains "DUPLICATE"), then "DUPLICATE". An answer
                    without a verdict — a failure: uncertainty is never turned into
                    "not duplicate".
        """
        normalized = " ".join(content.replace("*", " ").upper().split())
        if "NOT DUPLICATE" in normalized:
            return False
        if "DUPLICATE" in normalized:
            return True
        raise JudgeError(
            f"judge gave no verdict DUPLICATE/NOT DUPLICATE: {content[:120]}",
            outcome=SlotOutcome.OFF_CONTRACT,
        )