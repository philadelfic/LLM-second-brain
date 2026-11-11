"""SummaryService — the summary slot's summarization client (ARCH §3.2, §4.7; REQ §5.5).

    The system's second external call: generating a note's brief summary —
    1–2 capacious sentences in the note's language. The summary length limit is 150
    CHARACTERS (the 3.1.0 gate decision, 2026-09-19): the prompt asks for it ("no
    more than 150 characters in total"), but code enforces it hard — `cap_summary`
    is applied at the point where a model summary is saved
    (worker.process_summary_pending, the settings.max_summary_chars limit). The
    slot call itself does not trim the result:
    the transport returns content as is (the prompt — ARCH §4.7, the text lives in
    the prompt registry app/services/prompts.py — Phase 11, decision #7).

    Phase 11: the HTTP transport is factored out into `LLMClient`
    (app/services/llm_client.py) — one client per summary slot (provider `ollama` |
    `openai`, per-slot address/model/key from Settings, decision #1). The service
    no longer keeps httpx or builds payloads itself: system+user, non-stream,
    num_predict and think go through `LLMClient.chat`; the F1 queue (ollama_slot —
    ollama provider only), Bearer when a key is set, and response parsing
    (`message.content` / `choices[0].message.content`, reasoning fields discarded)
    live inside the client. keep_alive is not sent to anyone (decision #6: the
    server governs the model, OLLAMA_KEEP_ALIVE).

    Mode "B": the call is made **only by the background worker** (the
    pending_summary flow, and since Phase 8 Stage 2.2 — the duplicate-merge loop
    too), never from the synchronous write path — save/update are not blocked
    (REQUIREMENTS §5.5, NFR-5: a warm call with reasoning takes seconds).

    `merge(text_a, text_b)` (Phase 8, Stage 2.2 — the dedup variant B; the
    LSBN-F-23 contract): a duplicate pair is not thrown away but **merged** — both
    texts go in one call with the merge prompt, the answer is a **list of
    self-sufficient parts** (JSON `{"parts": [{"title", "text"}, ...]}`), each part
    with its own title (≤5 words) and text ≤ settings.max_note_chars (3.4.1: the
    note's soft limit lives in the config, LSBN-S-1/LSBN-T-26). Completeness
    (the parts' sum covers the sources over the normalized text) and limits are
    checked by code: a part longer than the limit or an incomplete answer — ONE
    follow-up call in the same context, 2 rounds at most; no convergence —
    `MergeIncompleteError`, the pair is deferred by the worker (both notes stay
    alive). Only the worker calls it (the dedup
    merge) — the synchronous write path is not blocked.

    Key decisions:
    - The F1 queue (after the E2E): summarize and merge go through the ollama_gate
      slot inside the client — one request to the server at a moment; the judge on
      the same base_url shares this slot (an E2E finding: without the queue the
      verdict burned out on a timeout while idling behind a merge generation).
    - Thinking is not limited (REQUIREMENTS §5.5, the 2026-08-29 decision): with
      `SUMMARY_THINK=true` (the default) the `think` field is NOT sent in the
      request; with false — `"think": false`. Reasoning comes in a separate field
      (`message.thinking` at ollama; `reasoning_content` at openai) —
      **discarded**; only content reaches the DB. The num_predict budget is shared
      over thinking+content and split per operation:
      the digest — SUMMARY_NUM_PREDICT (35000), the duplicate merge — its own
      MERGE_NUM_PREDICT (35000: the merged text can be long). Time control — the
      client read timeout of the SUMMARY_TIMEOUT_SEC slot (default
      60 s; prod — 750 s: for merging 35k-char notes).
    - No retries — the only repeat is the worker's back-off
      (PENDING_RETRY_SEC → ×2 → 15 min); there is nothing here to manage a retry
      with (the client does not retry a chat either — NFR-3).
    - Contract checks are strict: empty content — a failure (§5.5), `LLMClient.chat`
      itself guarantees that; the summary's character limit is held by two: the
      prompt (≤ 150 characters) and code — `cap_summary` at the save point (the
      worker) truncates the model's answer at a word boundary with an ellipsis;
      the merge safety net stays its own: parts must pass the limit and
      completeness checks (`_incomplete_reason`) before the result is returned to
      the worker; the hard CHECK (MAX_NOTE_CHARS=35000) does not change meanwhile.
    - Failure — `SummaryError`, and only it: the client's error text (including
      the HTTP status and the {SLOT}_API_KEY hint on an auth failure, decision #4)
      is preserved; the worker decides the degradation itself (the fallback
      truncation is already in the outputs, the status stays pending). No failure
      breaks a user operation (NFR-3).
    - `last_attempt_ok` (NFR-4) — the source of `/health.summarizer_ok`: None —
      no attempts yet, True/False — the last one's outcome. The slot startup check
      (main.py, decision #5) initializes it before the first real call.
    - Prompts — from the registry (decision #7): system ones (`summary_system`,
      `summary_merge_system`) — editable; the user template (`merge_user`) — a
      baked-in constant; `registry` is injected by the build (build_services),
      without DI — the registry's built-in texts.
    - Unit tests — transport without the network: `transport` accepts
      httpx.MockTransport or a handler; for the worker's tests — fake
      summarizers (tests/fakes.py).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from app.config import TITLE_MAX_WORDS, Settings
from app.services.llm_client import ChatMeta, ChatResult, LLMClient, LLMError, SlotSpec
from app.services.prompts import PromptRegistry
from app.services.slot_outcome import SlotOutcome, classify_llm_error

# --- the merge contract (LSBN-F-23) -----------------------------------------
#
#: The merge rounds maximum: the main call + ONE follow-up call with history.
MERGE_MAX_ROUNDS = 2
#: The required share of the sources' significant tokens in the normalized text
#: of the parts (the completeness check, brief p. 4): "the parts' sum covers
#: the merged text; if not — we do not save and go into a follow-up call" — the contract
#: is STRICT, full coverage after normalization (the pilot's decision of 2026-10-26;
#: the 0.9 threshold of an orphaned pass — a bug, not canon). Normalization removes
#: shape noise: case, punctuation and compound-word splitting — so
#: no tolerance is needed; a missing significant token = a lost fact/name/
#: number (silent truncation is forbidden).
MERGE_COVERAGE_MIN = 1.0

#: The split's own mechanical floor (LSBN-D-13). The strict word coverage is for
#: merging; a fact-split legitimately rephrases the prose, so the mechanical check
#: only catches gross loss. The authoritative verdict is the judge's
#: (`JudgeService.verify_split`): the source and ALL parts go to it with the
#: "nothing missed" question. Live 2026-11-11: two live splits stopped at 0.72 and
#: 0.90 and never reached the judge — the gate was stricter than the reviewer.
SPLIT_FACTS_MIN = 0.90


@dataclass(frozen=True)
class MergePart:
    """One self-sufficient merge part: its own title (≤5 words) and text,
            fitting the note's soft limit settings.max_note_chars
            (LSBN-S-1; 3.4.1 — the number from the config).
    """

    title: str
    text: str


@dataclass(frozen=True)
class MergeResult:
    """The merge result as a list of parts + call metadata (for the journal).

            `parts` — the model answer's parts; `meta` — the call of the LAST round
            (`/health.last_meta` carries that one); `metas` — all completed calls by
            rounds (the jobs journal writes a `slot_call` per call); `rounds` — how
            many rounds there actually were (1 — no follow-up was needed).
    """

    parts: list[MergePart]
    meta: ChatMeta
    metas: list[ChatMeta] = field(default_factory=list)
    rounds: int = 1


class SummaryError(RuntimeError):
    """Summarization failed: the server is unavailable or the response is invalid.

            `outcome` — the single call-outcome type of a slot (#39, arch §6.4): the
            service sets it on a client failure (`classify_llm_error`), and on "the
            marker/structure not found" — explicitly `OFF_CONTRACT` (the default). The
            worker reads the outcome via `slot_outcome.outcome_of` and writes it to
            the journal.
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
        # Metadata of a completed call (#49/#50): a failure after the model's response
        # carries them to the worker — the truncation reason is visible in the journal (§7.2).
        self.meta = meta


class MergeIncompleteError(SummaryError):
    """The merge did not converge within MERGE_MAX_ROUNDS rounds (LSBN-F-23, p. 6).

            It differs from the transient `SummaryError` (that one retries on
            back-off): the sources did not change — a retry would give the same two
            rounds, so the pair is DEFERRED: the worker releases the merge work, keeps
            both notes alive, writes an event to the jobs journal and a record into
            `warnings`. "Silent truncation" is forbidden — not saving beats losing
            text.
    """

    def __init__(
        self,
        message: str,
        *,
        metas: list[ChatMeta] | None = None,
        rounds: int = MERGE_MAX_ROUNDS,
    ) -> None:
        last = metas[-1] if metas else None
        super().__init__(message, outcome=SlotOutcome.OFF_CONTRACT, meta=last)
        self.metas = list(metas or [])
        self.rounds = rounds


# The truncated summary's ellipsis: one character — the limit's budget is spent on text
# (U+2026; "..." would eat three characters of the same 150).
_ELLIPSIS = "…"


#: The hard-token classes of the fact check: digits and ASCII identifiers
#: (versions, paths, commands, configs) - the parts of a fact that must survive
#: verbatim even when the connecting prose is tightened.
_HARD_TOKEN_RE = re.compile(r"[0-9A-Za-z_./:=-]{2,}")

#: Separators INSIDE a digit group ("5 000", "1_000", "3-5"): folded away
#: before the fact comparison so formatting is not mistaken for a lost fact.
_DIGIT_SEPARATOR_RE = re.compile(r"(?<=\d)[\s_.\u00a0]+(?=\d)")


def facts_coverage(source: str, parts: list[str]) -> float:
    """The share of the source's HARD tokens (numbers, latin identifiers,
            paths, versions) found in the parts' text (LSBN-D-13, split only).

            Oleg's decision of 2026-11-11: split = one note per fact, so the model
            legitimately rephrases the prose. Checking it with the strict word
            coverage (merge_coverage) would quarantine every note (the live case:
            0.86 on a clean split). What must not be lost is the skeleton of facts:
            numbers, dates, versions, paths and commands. Cyrillic words are allowed
            to change; a missing number or path is still a lost fact.
    """
    def hard(text: str) -> set[str]:
        # Number shapes drift in the model's answer: "5 000" / "5_000" / "5-000"
        # instead of "5000", or "1 000 000". The separators inside digit groups are
        # folded before the tokens are compared (LSBN-D-13, the live 0.72 coverage
        # case: facts that were present but written with a space).
        folded = _DIGIT_SEPARATOR_RE.sub("", text)
        return {
            token.casefold()
            for token in _HARD_TOKEN_RE.findall(folded)
            if any(char.isdigit() for char in token)
        }

    source_tokens = hard(source)
    if not source_tokens:
        return 1.0
    parts_tokens = hard(" ".join(parts))
    return len(source_tokens & parts_tokens) / len(source_tokens)


def merge_coverage(source: str, parts: list[str]) -> float:
    """The share of the sources' significant tokens found in the parts' text (F-23, p. 4).

            The normalization of BOTH sides is the same: casefold + a split into
            alphanumeric runs of ≥3 characters (the same character class as the search's
            canonical significant tokens: "open-webui" → "open", "webui";
            punctuation and case are not shape noise — "home." and "home;" are one
            token). The check — EVERY source token's occurrence in the parts'
            combined text: a coherent retelling legitimately changes the order and
            service words; losing facts/names/numbers — no. A source without
            significant tokens (a degenerate case) counts as covered. The threshold —
            MERGE_COVERAGE_MIN (1.0, strict coverage: a missing token = a lost fact;
            a follow-up call, then deferring the pair — pp. 2/4/6 of the brief).
    """
    source_tokens = _significant_normalized(source)
    if not source_tokens:
        return 1.0
    parts_tokens = _significant_normalized(" ".join(parts))
    return len(source_tokens & parts_tokens) / len(source_tokens)


#: The significant token's character class — like ranking._TOKEN_SPLIT_RE (words and
#: compound-word parts over Cyrillic/latin/digits).
_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


def _significant_normalized(text: str) -> set[str]:
    """Significant tokens of a normalized text: alphanumeric runs of ≥3 characters
            of the casefolded text, no duplicates (the merge_coverage normalization).
    """
    return {
        token
        for token in _TOKEN_RE.findall(text.casefold())
        if len(token) >= 3
    }


def _escape_raw_quotes(text: str) -> str:
    """Escape "raw" quotes inside JSON string values (T-33/T-23).

            The flint probe of 2026-10-30 (the 354+413 pair): a reasoning model writes
            the value ``"text": "the .113 "primary" slot and .109"`` — quotes inside
            the value are not escaped, ``json.loads`` fails, and the whole answer goes
            to OFF_CONTRACT (973 failures of lsbdef-0008 — a neighboring class). A pass
            over the text distinguishes STRUCTURAL quotes (opening/closing a value)
            from RAW ones (inside a value): a structural quote stands before a
            structural character (`:`, `,`, `}`, `]`, the end of a string value); a raw
            one is escaped. Conservatively: if the repair does not lead to valid JSON,
            the caller returns the old failure — a model without this defect passes
            the repairer by (the first loads succeeds).
    """
    fixed: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if char != '"' or not (in_string and not escaped):
            # Outside a value any quote is structural (it opens a value);
            # inside a value an escaped (\") one — content.
            if char == '"' and not in_string:
                in_string = True
                escaped = False
            elif in_string and escaped:
                escaped = False
            elif in_string and char == "\\":
                escaped = True
            elif in_string:
                escaped = False
            fixed.append(char)
            continue
        # We are inside a value, the quote is not escaped: it is structural when it is
        # the value's end — `:`, `,`, `}` or `]` follow it (after spaces).
        rest = text[index + 1 :].lstrip()
        if rest[:1] in (":", ",", "}", "]"):
            in_string = False
            escaped = False
            fixed.append(char)
            continue
        # A raw quote inside a value — escape it.
        fixed.append('\\"')
    return "".join(fixed)


def _parse_merge_parts(content: str, label: str = "merge") -> list[MergePart]:
    """Parse the JSON answer (merge/split) into a list of parts (F-23, p. 1).

            The contract: exactly one JSON object ``{"parts": [{"title", "text"}, ...]}``.
            Violations (non-JSON, "parts" not a list/empty, a part without a title or
            text, EMPTY content — lsbdef-0008) — `SummaryError` OFF_CONTRACT: the
            answer is off-contract as a whole; the retry is the worker's back-off's
            business, not a follow-up call. A title longer than TITLE_MAX_WORDS words
            does NOT reject the part — it is trimmed to the limit (the title-gen
            precedent: the trimming is code's job, the worker validates at write).
    """
    data = SummaryService._extract_json(content)
    raw_parts = data.get("parts") if isinstance(data, dict) else None
    if not isinstance(raw_parts, list) or not raw_parts:
        raise SummaryError(f"{label}: в ответе модели нет списка частей")
    parts: list[MergePart] = []
    for item in raw_parts:
        if not isinstance(item, dict):
            raise SummaryError(f"{label}: часть ответа не объект JSON")
        title = str(item.get("title") or "").strip()
        text = str(item.get("text") or "").strip()
        if not title or not text:
            raise SummaryError(f"{label}: у части нет title или текста")
        words = title.split()
        if len(words) > TITLE_MAX_WORDS:
            title = " ".join(words[:TITLE_MAX_WORDS])
        parts.append(MergePart(title=title, text=text))
    return parts


def cap_summary(text: str, limit: int) -> str:
    """Hard-limit the summary by a CHARACTER limit at a word boundary.

            A safety net against an undisciplined model (the 3.1.0 gate decision,
            2026-09-19): the length is held by both the prompt (≤ 150 characters) and
            code — this function is applied wherever a model summary is saved (the
            single write point — `worker.process_summary_pending`), so the limit does
            not depend on a concrete model's discipline. Truncation by STRING
            characters (not bytes): Cyrillic is measured honestly, as in the fallback
            output (app/services/emit.py).

            The word boundary: when the budget breaks inside a word — the stub is
            dropped entirely (the ellipsis is placed instead of it, not on top), so the
            result is always ≤ `limit`. A word longer than the budget itself (no
            delimiters) is cut by characters — no word boundary exists there. A text
            shorter than the limit is returned as is, without an ellipsis.
    """
    if limit < 1:
        raise ValueError("cap_summary: limit должен быть ≥ 1")
    text = text.strip()
    if len(text) <= limit:
        return text
    # One budget character is reserved for the ellipsis.
    head = text[: limit - 1]
    if not text[limit - 1].isspace():
        # Broken inside a word — roll back to its start (the stub is not kept).
        head = re.sub(r"\S+$", "", head)
    head = head.rstrip()
    if not head:
        # A word longer than the budget: no word boundary — cut by characters.
        return text[:limit]
    return f"{head}{_ELLIPSIS}"


# The title-generation prompt (Phase 11, decision #9): baked in — needed only for
# migration notes (title IS NULL), never created by a file. A thinking call
# of the summary slot (like summarization); the trimming to TITLE_MAX_WORDS is the worker's job.
TITLE_PROMPT = (
    "Сформулируй короткое название заметки: 1–5 слов, по сути текста, "
    "без кавычек и точки в конце."
)


@dataclass(frozen=True)
class SummaryResult:
    """The summarization result together with the call metadata (#49, §6.3).

            `text` — the answer's content (what the service returned before); `meta` —
            the budget, the counters, the done reason and the call duration (for the
            worker's `slot_call` journal entry).
    """

    text: str
    meta: ChatMeta


class Summarizer(Protocol):
    """The summarizer contract: it knows nothing about the implementation nature.

            Implementations: SummaryService (the live summary slot) and test fakes
            (tests/fakes.py) — the worker depends on the protocol, not the client
            class.
    """

    def summarize(self, text: str) -> SummaryResult:
        """Generate the brief summary of one text."""
        ...

    def title(self, text: str) -> SummaryResult:
        """A note title of ≤5 words (Phase 11, decision #9: the title generation
                    of migration notes; a thinking call of the summary slot; the trimming
                    to TITLE_MAX_WORDS is the worker's job).
        """
        ...

    def merge(self, text_a: str, text_b: str) -> MergeResult:
        """Merge a pair of duplicate texts into a list of self-sufficient parts
                    (dedup, Stage 2.2; LSBN-F-23).
        """
        ...

    def split(self, text: str) -> MergeResult:
        """Split a long note into self-sufficient parts ≤ the limit
                    (self-migration, LSBN-F-22 M2; the `split` operation — SPLIT_* env).
        """
        ...

    def probe(self) -> str:
        """A cheap probe knock on the slot (#39, arch §6.4): `check()` without
                    generation; the outcome — a CheckResult from llm_client
                    (`ok`/`auth_failed`/`unreachable`/`model_missing`).
        """
        ...

    def close(self) -> None:
        """Close resources (interface-compatible with EmbeddingService)."""
        ...


class SummaryService:
    """Generate `summary` via the summary slot's chat (non-stream)."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport
        | Callable[[httpx.Request], httpx.Response]
        | None = None,
        *,
        llm: LLMClient | Mapping[str, LLMClient] | None = None,
        registry: PromptRegistry | None = None,
    ) -> None:
        self._settings = settings
        # 3.4.1 (LSBN-T-26): the number in the merge/split prompts — from the build settings.
        self._prompts = (
            registry
            if registry is not None
            else PromptRegistry(max_note_chars=settings.max_note_chars)
        )
        # Clients PER OPERATION (LSBN-F-34): the flat env SUMMARIZE_/TITLE_/
        # MERGE_/SPLIT_* overrides the model/provider/address/key/timeout/
         # budget of one process on top of the summary slot; without overrides
        # the specs are identical (the build shares one client). DI: one client (the other
        # interfaces) — for all operations; the mapping — for point substitutions
        # by tests. "split" — migration splitting (LSBN-F-22 M2).
        if llm is None:
            self._llms: dict[str, LLMClient] = {
                operation: LLMClient(
                    SlotSpec.for_operation(operation, settings)[0],
                    transport=transport,
                )
                for operation in ("summarize", "title", "merge", "split")
            }
        elif isinstance(llm, Mapping):
            self._llms = dict(llm)
        else:
            self._llms = {
                "summarize": llm,
                "title": llm,
                "merge": llm,
                "split": llm,
            }
        # The main client (summarize) — the former field of DI builds and tests.
        self._llm = self._llms["summarize"]
        # None — no attempts yet (health does not lie before the first data).
        self.last_attempt_ok: bool | None = None
        # Metadata of the last call (#49/#50): diagnostics in /health.slots
        # without reading the journal (None — no call yet).
        self.last_meta: ChatMeta | None = None

    def summarize(self, text: str) -> SummaryResult:
        """Generate the brief summary of one text."""
        if not text or not text.strip():
            raise ValueError("summarize: пустой текст заметки")
        try:
            # max_chars = None: the transport returns the model's content as is (without
            # trimming) — the hard summary limit is applied at the save point
            # (the worker, cap_summary); merge trims itself — max_note_chars.
            # The budget — the operation chain (SUMMARIZE_NUM_PREDICT → slot → auto).
            result = self._chat(self._prompts.summary_system, text, operation="summarize")
        except SummaryError:
            self.last_attempt_ok = False  # the degradation becomes visible in /health
            raise
        self.last_attempt_ok = True
        return SummaryResult(text=result.content, meta=result.meta)

    def title(self, text: str) -> SummaryResult:
        """A note title of ≤5 words (Phase 11, decision #9: the title generation
                    of migration notes; a thinking call of the summary slot; the trimming
                    to TITLE_MAX_WORDS is the worker's job).
        """
        if not text or not text.strip():
            raise ValueError("title: пустой текст заметки")
        try:
            result = self._chat(TITLE_PROMPT, text, operation="title")
        except SummaryError:
            self.last_attempt_ok = False  # the degradation becomes visible in /health
            raise
        self.last_attempt_ok = True
        return SummaryResult(text=result.content, meta=result.meta)

    def merge(self, text_a: str, text_b: str) -> MergeResult:
        """Merge a pair of duplicate texts into a list of self-sufficient parts
                    (dedup, Stage 2.2; LSBN-F-23).
        """
        if not text_a or not text_a.strip() or not text_b or not text_b.strip():
            raise ValueError("merge: пустая заметка недопустима")
        try:
            metas: list[ChatMeta] = []
            system = self._prompts.summary_merge_system
            base_user = self._prompts.merge_user.format(text_a=text_a, text_b=text_b)
            base_user += self._size_hint(len(text_a) + len(text_b))
            parts, raw, meta = self._merge_round(system, base_user)
            metas.append(meta)
            rounds = 1
            issue = self._incomplete_reason(parts, text_a, text_b)
            if issue is not None:
                # The follow-up call in the same context (p. 2): the sources are already in the prompt,
                # so it is cheaper than a new call. One retry, no more rounds.
                followup_user = (
                    f"{base_user}\n\nYOUR ANSWER:\n{raw}\n\n{issue}\n"
                    "Return the FULL corrected set of parts as one JSON object."
                )
                parts, _raw, meta = self._merge_round(system, followup_user)
                metas.append(meta)
                rounds = 2
                issue = self._incomplete_reason(parts, text_a, text_b)
                if issue is not None:
                    raise MergeIncompleteError(
                        f"merge incomplete after {MERGE_MAX_ROUNDS} rounds: {issue}",
                        metas=metas,
                        rounds=rounds,
                    )
        except SummaryError:
            self.last_attempt_ok = False  # the degradation becomes visible in /health
            raise
        self.last_attempt_ok = True
        return MergeResult(parts=parts, meta=meta, metas=metas, rounds=rounds)

    def split(self, text: str) -> MergeResult:
        """Split a long note into self-sufficient parts ≤ the limit
                    (self-migration, LSBN-F-22 M2; the `split` operation — SPLIT_* env).
        """
        if not text or not text.strip():
            raise ValueError("split: пустая заметка недопустима")
        try:
            metas: list[ChatMeta] = []
            system = self._prompts.summary_split_system
            base_user = self._prompts.split_user.format(text=text)
            base_user += self._facts_hint(len(text))
            parts, raw, meta = self._merge_round(system, base_user, operation="split")
            metas.append(meta)
            rounds = 1
            issue = self._incomplete_reason(parts, text, operation="split")
            if issue is not None:
                followup_user = (
                    f"{base_user}\n\nYOUR ANSWER:\n{raw}\n\n{issue}\n"
                    "Return the FULL corrected set of parts as one JSON object."
                )
                parts, _raw, meta = self._merge_round(
                    system, followup_user, operation="split"
                )
                metas.append(meta)
                rounds = 2
                issue = self._incomplete_reason(parts, text, operation="split")
                if issue is not None:
                    raise MergeIncompleteError(
                        f"split incomplete after {MERGE_MAX_ROUNDS} rounds: {issue}",
                        metas=metas,
                        rounds=rounds,
                    )
        except SummaryError:
            self.last_attempt_ok = False  # the degradation becomes visible in /health
            raise
        self.last_attempt_ok = True
        return MergeResult(parts=parts, meta=meta, metas=metas, rounds=rounds)

    # --- internal -------------------------------------------------------------

    def _merge_round(
        self,
        system_prompt: str,
        user_text: str,
        *,
        operation: str = "merge",
    ) -> tuple[list[MergePart], str, ChatMeta]:
        """One merge/split round: chat → parse the JSON parts contract.

                    Returns (parts, raw content, call metadata). A structure violation
                    (non-JSON, no parts list, a part without a title/text) —
                    `SummaryError` OFF_CONTRACT: this is an answer failure, not "a part is
                    long" — cured by a back-off retry, not a follow-up call (see
                    merge/split).
        """
        result = self._chat(system_prompt, user_text, operation=operation)
        return (
            _parse_merge_parts(result.content, label=operation),
            result.content,
            result.meta,
        )

    def _incomplete_reason(
        self, parts: list[MergePart], *sources: str, operation: str = "merge"
    ) -> str | None:
        """The reason of an incomplete merge/split or None (the F-23/F-22 briefs).

                    "Part N is longer than the limit" takes priority over "the answer is
                    incomplete": both go into ONE follow-up call; the wording names the main
                    problem. The limit — the build snapshot's settings.max_note_chars
                    (3.4.1, LSBN-T-26).
        """
        limit = self._settings.max_note_chars
        long_parts = [
            number
            for number, part in enumerate(parts, start=1)
            if len(part.text) > limit
        ]
        if long_parts:
            listed = ", ".join(map(str, long_parts))
            return (
                f"part {listed} exceeds the {limit}-character note "
                "limit — split it into several self-contained parts"
            )
        merged_source = "\n".join(sources)
        texts = [p.text for p in parts]
        if operation == "split":
            # one note = one fact: prose may be tightened, the fact skeleton may not
            coverage = facts_coverage(merged_source, texts)
            if coverage < SPLIT_FACTS_MIN:
                return (
                    "the parts lost hard facts of the source — numbers, dates, "
                    f"versions or paths are missing (fact coverage {coverage:.2f}); "
                    "keep every number, date, version and path verbatim"
                )
            return None
        coverage = merge_coverage(merged_source, texts)
        if coverage < MERGE_COVERAGE_MIN:
            return (
                "the parts do not cover the source texts — key content "
                f"(facts, names, numbers, dates) is missing (coverage "
                f"{coverage:.2f}); rewrite the parts so that together they "
                "preserve every fact"
            )
        return None

    def _size_hint(self, chars: int) -> str:
        """The part-count hint for merge/split (LSBN-D-13).

            The model is bad at arithmetic under a limit: without the number it
            returns one oversized part or silently shortens the text. The hint gives
            the exact size of the source and the MINIMUM number of parts, so "two
            notes instead of one" is asked for in numbers, not in prose.
        """
        limit = max(1, int(self._settings.max_note_chars))
        parts = max(1, -(-chars // limit))
        return (
            f"\n\nSIZE: the source is {chars} characters; the note limit is {limit}. "
            f"Return at least {parts} part(s), each within {limit} characters. "
            "Dropping, shortening or summarizing the content is a failure: every fact, "
            "name, number and date must be present in the parts."
        )

    def _facts_hint(self, chars: int) -> str:
        """The "one note = one fact" hint for splitting (Oleg's decision 11.11.2026)."""
        limit = max(1, int(self._settings.max_note_chars))
        return (
            f"\n\nFACTS: the source is {chars} characters. Return ONE PART PER FACT — "
            "as many parts as there are distinct facts in the source, each well "
            f"under {limit} characters. Keep every number, date, version, path and "
            "command verbatim; only the connecting prose may be tightened. "
            "Losing a fact is a failure."
        )

    @staticmethod
    def _balanced_objects(text: str) -> list[str]:
        """Every balanced JSON object/array in the answer, in order of appearance.

            A greedy `first '{' .. last '}'` slice breaks on answers where the model
            adds prose with braces after the object or writes two objects in a row
            (the live case of 2026-11-11: "не-JSON объект" on a real duplicate pair).
            The scanner is string-aware: braces inside string values and escaped
            quotes do not close the structure.
        """
        found: list[str] = []
        start: int | None = None
        depth = 0
        in_string = False
        escaped = False
        for index, char in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char in "{[":
                if depth == 0:
                    start = index
                depth += 1
            elif char in "}]":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        found.append(text[start : index + 1])
                        start = None
        return found

    @staticmethod
    def _extract_json(content: str) -> Any:
        """Extract the JSON object from the answer (robust to code fences and wrappers).

                    The same technique as the classifier's (`ClassifierService._extract_json`):
                    the thinking/response wrapper is cut off before json.loads — the model
                    may wrap JSON in reasoning containing curly braces. The "raw quotes"
                    repair (LSBN-T-33/T-23, the flint probe of 2026-10-30): reasoning models
                    return an object valid in meaning but with unescaped quotes inside
                    string values — json.loads fails, and whole parts were lost
                    (OFF_CONTRACT → back-off). The repair pass escapes the quotes INSIDE
                    values and repeats the parse; if it did not help — the old failure (the
                    answer is off-contract as a whole, the retry is the back-off's business;
                    other models' behavior does not change).
        """
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        def _as_parts(value: Any) -> Any:
            # A bare array is the same answer in a different shape: {"parts": [...]}.
            if isinstance(value, list):
                return {"parts": value}
            return value

        for candidate in [text, *SummaryService._balanced_objects(text)]:
            for prepared in (candidate, _escape_raw_quotes(candidate)):
                try:
                    return _as_parts(json.loads(prepared))
                except ValueError:
                    continue
        raise SummaryError("merge: не-JSON объект в ответе модели")

    def close(self) -> None:
        """Close resources (interface-compatible with EmbeddingService)."""
        closed: list[int] = []
        for client in self._llms.values():
            if id(client) not in closed:
                client.close()
                closed.append(id(client))

    def probe(self) -> str:
        """A cheap probe knock on the slot (#39, arch §6.4): `check()` without
                    generation; the outcome — a CheckResult from llm_client
                    (`ok`/`auth_failed`/`unreachable`/`model_missing`).
        """
        return self._llm.check()

    def _chat(
        self,
        system_prompt: str,
        user_text: str,
        *,
        operation: str,
    ) -> ChatResult:
        """One chat call BY THE OPERATION CLIENT (no retries — the worker decides on repeats).

                    system_prompt/user_text — the summarize/merge differences; the client
                    is chosen by the operations registry (LSBN-F-34): its spec already
                    carries the effective chain operation env → slot env → default,
                    including the output ceiling
                    (`SUMMARIZE_/TITLE_/MERGE_NUM_PREDICT`; None — the auto-budget "as much
                    as fits", #49). think — the spec flag; keep_alive is not sent
                    (decision #6). Content is returned as is: the character limits are held
                    by the save points (cap_summary — the worker; the merge/split parts
                    limit — settings.max_note_chars).
        """
        try:
            result = self._llms[operation].chat_result(
                system_prompt,
                user_text,
                operation=operation,
            )
        except LLMError as exc:
            # The client text is preserved: HTTP status, the slot address, the
            # {SLOT}_API_KEY hint on an auth failure (decision #4). The call outcome
            # (#39, arch §6.4) — a single outcome type at the service boundary.
            self.last_meta = exc.meta  # call metadata for diagnostics
            raise SummaryError(
                str(exc), outcome=classify_llm_error(exc), meta=exc.meta
            ) from exc
        self.last_meta = result.meta
        # The thinking/reasoning_content field goes only into the call metadata:
        # we read only content. The character safety net is no longer needed here:
        # merge holds the parts limit itself (the completeness/limit checks — pp. 1/4
        # of F-23); the summary limit is held by the caller (the worker) — cap_summary at
        # the save point (the 3.1.0 gate decision, 2026-09-19).
        return result