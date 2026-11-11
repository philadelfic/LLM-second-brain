"""PromotionService — the domain trigger and automatic leaf creation (Phase 10, Step 5).

    "A model for models" (§5.7): the structure grows by the system. The pipeline of
    one `run()`:

    1. **The trigger — SQL aggregation, not an LLM**: `GROUP BY hint_path` (the
       single full path of the lsb-0005-02 markup) among default notes; a group
       with a counter ≥ NAMESPACE_PROMOTION_THRESHOLD
       (15) at each note's confidence ≥ NAMESPACE_PROMOTION_MIN_CONFIDENCE
       (0.60) — a candidate for automatic leaf creation. `candidates()` — the same
       aggregation for `memory_namespaces.promotion_candidates` (an up-to-date map
       for models).
    2. **Description generation**: an LLM over 2–3 of the group's summaries (the
       Describer — the summarization model, a small num_predict, think:false — the
       Step-4 classifier pattern); the ≤2-sentences descriptions contract is held
       by a truncation to 2 sentences.
    3. **The cosine anti-synonymy prefilter**: the candidate description's
       embedding against the descriptions of all registry nodes; cosine ≥
       NAMESPACE_SYNONYM_SIMILARITY (0.85) → a merge WITHOUT an LLM (the Phase 8
       pattern: cosine is the prefilter, the LLM — the borderline zone).
    4. **The structure judge — an LLM gate before creation** (a StructureJudge —
       the dedup judge's model, a separate prompt): (1) borderline anti-synonymy —
       the "MERGE <path>" verdict; (2) the slug/description meaningfulness —
       "REJECT"; otherwise "CREATE". The model verdicts, not a human.
    5. **The action**: create → a provisional leaf (only leaves inside existing
       roots — `NamespaceService.create` never creates roots);
       merge → a retro-move into the canonical node by one UPDATE with hint
       canonization; reject → the verdict is recorded, the notes stay in default
       (honestly generic). The retro-move is one UPDATE (+vector_status='pending'
       — re-encoding the vectors into the new node's partition by the regular
       queue).

    **Cooldown** (the brief "limits/cooldown"): three mechanisms — (a) recording
    the verdict in `promotions` (merged/rejected): the group no longer pokes the
    describer/judge (otherwise a rejected candidate would loop LLM calls on every
    run); (b) the node's existence: a group with an already-registered path is not
    a candidate; (c) storm-protection limits: NAMESPACE_AUTO_MAX_PER_DAY
    (provisional nodes created today by UTC) and NAMESPACE_MAX_LEAVES_PER_DOMAIN
    (the leaves ceiling in a root) — candidates over the limit are skipped with a
    log before any LLM call. A created node needs no `promotions` record: the
    retro-move takes the group out of default, and new high-confidence default
    notes with the same hint move into the node by the regular Step-4 grooming
    (the node already exists).

    Autonomy bounds (§5.7): the hint's domain must be a registered root (new roots
    — only by the operator; an unregistered hint — a log signal "default
    accumulates content outside known roots"); automatic — leaves only. A failure
    of any step (DescriberError/StructureJudgeError/EmbeddingError) does not spoil
    the data: the candidate stays, a retry — at the next run (NFR-3); the next
    default-note classification in the worker wakes the group.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from typing import Any, Protocol

import httpx

from app.config import Settings
from app.services.llm_client import LLMClient, LLMError, SlotSpec
from app.services.namespaces import (
    NamespaceService,
    count_sentences,
)
from app.services.prompts import PromptRegistry
from app.storage.db import DEFAULT_NAMESPACE, session, transaction

# The candidate's descriptions are built from summaries: top by confidence, deterministically.
SUMMARIES_PER_CANDIDATE = 3

# The description call parameters (budget 512, think:false) with LSBN-F-34 live in the
# operations registry (app/services/operations.OPERATION_FIELD_DEFAULTS —
# the `describe` operation's code defaults) and are overridden by the flat env
# DESCRIBE_NUM_PREDICT / DESCRIBE_THINK. Timeouts/temperature — the level
# of the slot client (LLMClient): the read timeout — the operation chain; keep_alive
# is not sent to anyone (decision #6 — the server governs the model).


def _l2_norm(vec: list[float]) -> float:
    """The vector's Euclidean norm (L2)."""
    return math.sqrt(sum(v * v for v in vec))


class DescriberError(RuntimeError):
    """The description generator gave no text: the server is unavailable or the answer is empty."""


class StructureJudgeError(RuntimeError):
    """The structure judge gave no verdict: the server is unavailable or the answer is invalid."""


class Verdict:
    """The structure judge's verdict: the action create|merge|reject + the merge target."""

    __slots__ = ("action", "target")

    def __init__(self, action: str, target: str | None = None) -> None:
        self.action = action
        self.target = target


class Describer(Protocol):
    """The node description generator contract: it knows nothing about the implementation nature."""

    def describe(
        self, summaries: list[str], slug: str, domain: str
    ) -> str:
        """The new node's description from the group's summaries (≤2 sentences)."""
        ...

    def close(self) -> None:
        """Close resources (interface-compatible with the other clients)."""
        ...


class StructureJudge(Protocol):
    """The structure judge contract (the gate before creation, §5.7)."""

    def review(
        self,
        description: str,
        slug: str,
        domain: str,
        existing: list[dict[str, Any]],
        nearest_path: str | None,
        nearest_cosine: float | None,
    ) -> Verdict:
        """A verdict on a candidate: CREATE / MERGE <path> / REJECT."""
        ...

    def close(self) -> None:
        """Close resources (interface-compatible with the other clients)."""
        ...


    # --- description generator (the summarization model, the classifier pattern) ---

class DescriptionService:
    """A node description via the summary slot's chat (the summarization model).

            A separate small call (think:false, num_predict 128) — the Step-4
            classifier pattern: the same model as summarization, one slot client
            (Phase 11: the summary slot's LLMClient; ollama calls are serialized by
            ollama_slot). An overly long answer is trimmed to 2 sentences — the
            descriptions contract is held by mechanics, not by hoping in the model.
    """

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport
        | Callable[[httpx.Request], httpx.Response]
        | None = None,
        *,
        llm: LLMClient | None = None,
        registry: PromptRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._prompts = registry if registry is not None else PromptRegistry()
        # DI builds or a dedicated client with a transport injection for unit tests.
        # The `describe` operation (LSBN-F-34): the flat env DESCRIBE_* on top of the summary
        # slot; the operation's code defaults — the budget 128 and think=False
        # (operations.OPERATION_FIELD_DEFAULTS, the former DESCRIBE_NUM_PREDICT).
        self._llm = (
            llm
            if llm is not None
            else LLMClient(
                SlotSpec.for_operation("describe", settings)[0], transport=transport
            )
        )
        # None — no attempts yet (health does not lie before the first data).
        self.last_attempt_ok: bool | None = None

    def describe(self, summaries: list[str], slug: str, domain: str) -> str:
        """The new node's description from the group's summaries (≤2 sentences)."""
        if not summaries:
            raise ValueError("describe: ожидается непустой список суммари")
        try:
            content = self._llm.chat(
                self._prompts.describe_system,
                self._prompts.describe_user.format(
                    domain=domain,
                    slug=slug,
                    summaries="\n".join(f"- {s}" for s in summaries),
                ),
                operation="describe",  # LSBN-F-34: the budget/think — the operation chain
            )
        except LLMError as exc:
            # The client text is preserved: HTTP status, the slot address, the
            # {SLOT}_API_KEY hint on an auth failure (decision #4).
            self.last_attempt_ok = False
            raise DescriberError(str(exc)) from exc
        self.last_attempt_ok = True
        return self._trim(content)

    def close(self) -> None:
        """Close resources (interface-compatible with the other clients)."""
        self._llm.close()

    # --- internal -------------------------------------------------------------

    @staticmethod
    def _trim(content: str) -> str:
        """Normalize and trim to 2 sentences (the descriptions contract).

                    The model may ignore "1–2 sentences" — we hold the contract by
                    trimming: the first two sentences, the rest is dropped. An empty
                    result — DescriberError (a node without a description never exists).
        """
        text = " ".join(content.split())
        if not text:
            raise DescriberError("пустое описание узла")
        sentences = [part.strip() for part in re.split(r"[.!?]+(?:\s|$)", text)]
        sentences = [part for part in sentences if part]
        if not sentences:
            raise DescriberError(f"описание без предложений: {content[:120]}")
        if len(sentences) > 2:
            sentences = sentences[:2]
        trimmed = ". ".join(sentences)
        if not trimmed.endswith((".", "!", "?")):
            trimmed += "."
        if count_sentences(trimmed) > 2:  # a safety net: the ≤2-sentences contract
            raise DescriberError(f"описание не обрезается до 2 предложений: {trimmed[:120]}")
        return trimmed


# --- structure judge (the dedup judge's model, the Phase 8 pattern) --------

# The MERGE verdict's target path: slugs of latin/digits/hyphen, a path of 1..3 levels
# (lsb-0005-03: node nesting up to depth 3).
VERDICT_PATH_RE = re.compile(
    r"[a-z0-9]+(?:-[a-z0-9]+)*"
    r"(?:/[a-z0-9]+(?:-[a-z0-9]+)*){0,2}"
)


class StructureJudgeService:
    """The structure judge's verdict via the judge slot's chat (non-stream).

            The dedup judge's model (JUDGE_MODEL, REQUIREMENTS §5.7 "the same dedup
            judge model, a separate prompt"); Phase 11: the transport — the shared
            judge slot client (LLMClient), the call parameters — like JudgeService
            (Phase 8): think from NAMESPACE_JUDGE_THINK (None — inherits JUDGE_THINK),
            the JUDGE_NUM_PREDICT budget; keep_alive is not sent (decision #6). A judge
            failure (transport) — StructureJudgeError: the candidate stays without a
            verdict and retries at the next run (NFR-3). A bad verdict (MERGE without
            a node) — also StructureJudgeError: an underdefined decision is never
            turned into a creation.
    """

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport
        | Callable[[httpx.Request], httpx.Response]
        | None = None,
        *,
        llm: LLMClient | None = None,
        registry: PromptRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._prompts = registry if registry is not None else PromptRegistry()
        # DI builds (the shared judge slot client) or a dedicated client with
        # a transport injection for unit tests (MockTransport). The
        # `structure_judge` operation (LSBN-F-34): the flat env STRUCTURE_JUDGE_* on top of
        # the judge slot; NAMESPACE_JUDGE_THINK stays the senior think channel
        # (None — the operation chain: STRUCTURE_JUDGE_THINK → JUDGE_THINK).
        self._llm = (
            llm
            if llm is not None
            else LLMClient(
                SlotSpec.for_operation("structure_judge", settings)[0],
                transport=transport,
            )
        )
        self.last_attempt_ok: bool | None = None

    def review(
        self,
        description: str,
        slug: str,
        domain: str,
        existing: list[dict[str, Any]],
        nearest_path: str | None,
        nearest_cosine: float | None,
    ) -> Verdict:
        """A verdict on a candidate: CREATE / MERGE <path> / REJECT."""
        try:
            content = self._llm.chat(
                self._prompts.structure_judge_system,
                self._prompts.structure_judge_user.format(
                    domain=domain,
                    slug=slug,
                    description=description,
                    nodes="\n".join(
                        f"- {node['path']}: {node['description']}"
                        for node in existing
                    )
                    or "(нет)",
                    nearest=(
                        f"{nearest_path} (косинус {nearest_cosine:.2f})"
                        if nearest_path is not None
                        else "нет близких узлов"
                    ),
                ),
                # The budget/think — the operation chain (LSBN-F-34):
                # STRUCTURE_JUDGE_NUM_PREDICT → JUDGE_NUM_PREDICT → auto;
                # think: NAMESPACE_JUDGE_THINK → STRUCTURE_JUDGE_THINK →
                # JUDGE_THINK (None — the client's spec flag).
                num_predict=None,
                think=self._think(),
                operation="structure_judge",
            )
        except LLMError as exc:
            # The client text is preserved: HTTP status, the slot address, the
            # {SLOT}_API_KEY hint on an auth failure (decision #4).
            self.last_attempt_ok = False
            raise StructureJudgeError(str(exc)) from exc
        try:
            verdict = self._parse(content)
        except StructureJudgeError:
            self.last_attempt_ok = False  # an unrecognized verdict is also a failure
            raise
        self.last_attempt_ok = True
        return verdict

    def close(self) -> None:
        """Close resources (interface-compatible with the other clients)."""
        self._llm.close()

    # --- internal -------------------------------------------------------------

    def _think(self) -> bool | None:
        """The structure judge's think flag: NAMESPACE_JUDGE_THINK — the senior channel
                    (None — the LSBN-F-34 operation chain: STRUCTURE_JUDGE_THINK →
                    JUDGE_THINK through the client's spec). The flag is separated from the
                    dedup judge (the Step-7 E2E — a thinking structure judge "stuck" on
                    twin pairs at 20 tok/s, starving summarization on the shared slot;
                    a thoughtless verdict — 10–50 tokens).
        """
        return self._settings.namespace_judge_think

    @staticmethod
    def _parse(content: str) -> Verdict:
        """Parse the CREATE / MERGE <path> / REJECT mark.

                    Markdown-bold is stripped, case does not matter. "MERGE" requires a
                    target node in the answer: the path is extracted from the original
                    content (not from the upper-cased) by the slug regexp; without a path —
                    a failure (an underdefined verdict is never turned into a creation).
                    An answer without a mark — StructureJudgeError.
        """
        upper = " ".join(content.replace("*", " ").upper().split())
        if "REJECT" in upper:
            return Verdict("reject")
        if "MERGE" in upper:
            match = VERDICT_PATH_RE.search(content.replace("*", ""))
            if match is None:
                raise StructureJudgeError(
                    f"MERGE verdict without target node: {content[:120]}"
                )
            return Verdict("merge", match.group(0))
        if "CREATE" in upper:
            return Verdict("create")
        raise StructureJudgeError(
            f"structure judge gave no verdict CREATE/MERGE/REJECT: {content[:120]}"
        )


# --- the domain trigger -----------------------------------------------------


class PromotionService:
    """The trigger + automatic leaf creation: hint-group aggregation → gate → action.

            Dependencies are injected (DI): embedding — the cosine prefilter, the
            describer — description generation, the judge — the gate before creation,
            namespaces — the registry (the shared instance with the worker/
            NoteService). describer/judge None — the trigger does not start (a test
            mode, like the worker's classifier=None).
    """

    def __init__(
        self,
        settings: Settings,
        embedding: Any,
        describer: Describer | None = None,
        judge: StructureJudge | None = None,
        namespaces: NamespaceService | None = None,
    ) -> None:
        self._settings = settings
        self._embedding = embedding
        self._describer = describer
        self._judge = judge
        self._namespaces = namespaces if namespaces is not None else NamespaceService(settings)

    # --- reads: candidates (SQL aggregation, §5.7 "the trigger is not an LLM") ---

    def candidates(self) -> list[dict[str, Any]]:
        """Default-note groups that reached the threshold and have no verdict.

                    SQL aggregation: a counter ≥ NAMESPACE_PROMOTION_THRESHOLD at every
                    note's confidence ≥ NAMESPACE_PROMOTION_MIN_CONFIDENCE; then the
                    cooldown filters: the hint's domain is registered (roots — the
                    operator), the node is not created yet, there is no merged/rejected
                    verdict. Grouping — by the single full hint_path of depth 2 or 3
                    (lsb-0005-03): depth-1 (domain-generic) is not promoted; the parent must
                    exist (a hierarchy without holes — depth-3 waits for its depth-2 parent
                    to be created). Sorted by the counter descending — bigger groups first,
                    deterministically.
        """
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT n.hint_path AS hint_path, COUNT(*) AS cnt, "
                "ROUND(AVG(n.confidence), 2) AS avg_confidence "
                "FROM notes n "
                "WHERE n.namespace = 'default' AND n.deleted_at IS NULL "
                "AND n.hint_path IS NOT NULL AND n.hint_path != '' "
                "AND n.confidence >= ? "
                "GROUP BY n.hint_path "
                "HAVING COUNT(*) >= ? "
                "ORDER BY cnt DESC, n.hint_path",
                (
                    self._settings.namespace_promotion_min_confidence,
                    self._settings.namespace_promotion_threshold,
                ),
            ).fetchall()
        decided = self._decided_hints()
        result: list[dict[str, Any]] = []
        for row in rows:
            hint = row["hint_path"]
            depth = len(hint.split("/"))
            if depth not in (2, 3):
                continue  # depth-1 (domain-generic) is not promoted; >3 never happens
            domain = hint.split("/", 1)[0]
            slug = "/".join(hint.split("/")[1:])
            parent = "/".join(hint.split("/")[:-1])
            if not self._namespaces.exists(parent):
                continue  # the parent must exist (a hierarchy without holes)
            if hint in decided:
                continue  # cooldown: the judge's verdict is already issued
            if self._namespaces.exists(hint):
                continue  # the node already exists — grooming will sort the group out
            result.append(
                {
                    "domain": domain,
                    "subdomain": slug,
                    "count": int(row["cnt"]),
                    "avg_confidence": float(row["avg_confidence"]),
                }
            )
        return result

    # --- the pipeline run ----------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run all ready candidates; a summary {created, merged, rejected}.

                    Called by the worker after classifying a default note (the markup could
                    have pushed a group over the threshold). A failure on one candidate
                    does not cancel the others (NFR-3): the candidate is skipped until the
                    next run (it does not enter the summary). The day/leaves limits are
                    checked before the LLM calls.
        """
        report: dict[str, Any] = {"created": [], "merged": [], "rejected": []}
        if self._describer is None or self._judge is None:
            return report  # the trigger is off (a test mode)
        logger = logging.getLogger("app")
        self._signal_root_orphans()
        day_limit = self._day_limit_reached()
        for candidate in self.candidates():
            domain, slug = candidate["domain"], candidate["subdomain"]
            path = f"{domain}/{slug}"
            if day_limit:
                logger.warning(
                    "promotion: daily limit reached — candidate skipped",
                    extra={"event": "promotion_skipped", "path": path,
                           "reason": "daily_limit"},
                )
                continue
            if self._leaves_limit_reached(domain):
                logger.warning(
                    "promotion: leaves limit reached — candidate skipped",
                    extra={"event": "promotion_skipped", "path": path,
                           "reason": "leaves_limit"},
                )
                continue
            action = self._promote_one(domain, slug)
            if action in ("created", "merged", "rejected"):
                report[action].append(path)
            if action == "created":
                day_limit = self._day_limit_reached()
        return report

    def _signal_root_orphans(self) -> None:
        """A signal to the operator: default accumulates content outside known roots (§5.7).

                    The system does not create new roots itself (the autonomy bounds): if
                    the grooming markup consistently points at an unregistered domain and
                    the group reached the threshold — the operator decides whether such a
                    root should exist (the Step-6 REST). The signal — by the trigger
                    threshold (noise is cut off), in the logs (event=root_orphans); it is
                    not carried into memory_namespaces — structural signaling stays with
                    the operator.
        """
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT n.hint_path AS hint_path, COUNT(*) AS cnt "
                "FROM notes n WHERE n.namespace = 'default' AND n.deleted_at IS NULL "
                "AND n.hint_path IS NOT NULL AND n.hint_path != '' "
                "AND n.confidence >= ? "
                "GROUP BY n.hint_path HAVING COUNT(*) >= ?",
                (
                    self._settings.namespace_promotion_min_confidence,
                    self._settings.namespace_promotion_threshold,
                ),
            ).fetchall()
        orphans = []
        for row in rows:
            root = row["hint_path"].split("/")[0]
            if not self._namespaces.exists(root):
                orphans.append({"domain": root, "count": int(row["cnt"])})
        if orphans:
            logging.getLogger("app").warning(
                "promotion: default accumulates content outside known roots",
                extra={"event": "root_orphans", "orphans": orphans},
            )

    def _promote_one(self, domain: str, slug: str) -> str | None:
        """The full cycle of one candidate: created|merged|rejected|None.

                    None — a describer/judge failure (transport): the candidate stays
                    WITHOUT a verdict, a retry at the next run (NFR-3); it does not count
                    in the summary. A returned verdict is always recorded (cooldown):
                    merged/rejected — into promotions, created — by a registry node.
        """
        logger = logging.getLogger("app")
        path = f"{domain}/{slug}"
        summaries = self._group_summaries(path)
        try:
            description = self._describer.describe(summaries, slug, domain)  # type: ignore[union-attr]
        except DescriberError:
            logger.warning(
                "promotion: description generation failed — candidate kept",
                extra={"event": "promotion_failed", "domain": domain, "slug": slug,
                       "reason": "describer"},
            )
            return None  # no candidate description: a retry — the next run
        nearest_path, nearest_cosine = self._nearest_node(description)
        if nearest_cosine is not None and (
            nearest_cosine >= self._settings.namespace_synonym_similarity
        ):
            # The cosine prefilter: a merge without an LLM (the Phase 8 pattern).
            self._merge(path, nearest_path)  # type: ignore[arg-type]
            return "merged"
        try:
            verdict = self._judge.review(
                description,
                slug,
                domain,
                self._thematic_nodes(),
                nearest_path,
                nearest_cosine,
            )
        except StructureJudgeError:
            logger.warning(
                "promotion: structure judge failed — candidate kept",
                extra={"event": "promotion_failed", "domain": domain, "slug": slug,
                       "reason": "judge"},
            )
            return None
        if verdict.action == "merge":
            target = self._namespaces.validate_path(verdict.target or "")
            if target == path:
                # The judge "merged" the candidate with itself — the verdict is incorrect:
                # we neither create the candidate nor forbid it forever (no record),
                # a retry — the next run; consistently garbage answers are visible
                # in the logs (promotion_failed, reason=judge_self_merge).
                logger.warning(
                    "promotion: judge merged candidate into itself — bad verdict",
                    extra={"event": "promotion_failed", "domain": domain,
                           "slug": slug, "reason": "judge_self_merge"},
                )
                return None
            if target == DEFAULT_NAMESPACE:
                # A merge with default is meaningless: the candidate lies in it already.
                # We do not record the verdict (a confusion, not a decision); a retry —
                # the next run; the garbage is visible in the logs.
                logger.warning(
                    "promotion: judge merged candidate into default — bad verdict",
                    extra={"event": "promotion_failed", "domain": domain,
                           "slug": slug, "reason": "judge_merge_default"},
                )
                return None
            if not self._namespaces.exists(target):
                # The judge named an unregistered node: the verdict is unreliable,
                # but it is its DECISION (not a transport failure) — we record it as
                # a reject so the group does not poke the judge again.
                logger.warning(
                    "promotion: judge merge target unknown — recorded as rejected",
                    extra={"event": "promotion_rejected", "domain": domain,
                           "slug": slug, "target": target},
                )
                self._record(path, "rejected")
                return "rejected"
            self._merge(path, target)
            return "merged"
        if verdict.action == "reject":
            self._record(path, "rejected")
            logger.info(
                "promotion: structure judge rejected candidate",
                extra={"event": "promotion_rejected", "domain": domain, "slug": slug,
                       "description": description},
            )
            return "rejected"
        self._create(path, description)
        return "created"

    def _create(self, path: str, description: str) -> None:
        """Create a provisional leaf and move the group (one UPDATE)."""
        self._namespaces.create(path, description, status="provisional")
        moved = self._retro_move(path, path)
        logging.getLogger("app").info(
            "promotion: provisional leaf created",
            extra={"event": "node_created", "path": path, "moved": moved,
                   "status": "provisional"},
        )

    def _merge(self, path: str, canonical: str) -> None:
        """Merge the candidate into the canonical node (one UPDATE + the verdict).

                    The group's notes have moved — the hint is canonized
                    (hint_path = the node's canonical path).
        """
        moved = self._retro_move(path, canonical)
        self._record(path, "merged", canonical)
        logging.getLogger("app").info(
            "promotion: candidate merged into existing node",
            extra={"event": "node_merged", "canonical": canonical,
                   "hint": path, "moved": moved},
        )

    # --- SQL mechanics ---------------------------------------------------------

    def _thematic_nodes(self) -> list[dict[str, Any]]:
        """The registry's thematic nodes (without the system default swap).

                    The cosine prefilter and the judge compare the candidate only with
                    them: a merge with default is meaningless (the candidate lies in it
                    already), and the swap in the merge-candidate list would only confuse
                    the verdict.
        """
        return [
            node
            for node in self._namespaces.list_all()["namespaces"]
            if node["path"] != DEFAULT_NAMESPACE
        ]

    def _retro_move(self, path: str, canonical: str) -> int:
        """A retro-move of the group into the canonical node by ONE UPDATE (§5.7).

                    hint_path canonization: hint_path = the node's canonical path — into a
                    leaf, the leaf's full path; into a root, the root ("generic" for the
                    domain). vector_status='pending' — the regular re-encoding of vectors
                    into the new node's batch (the worker). Returns the moved note count.
        """
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE notes SET namespace = ?, hint_path = ?, "
                "vector_status = 'pending' "
                "WHERE namespace = 'default' AND deleted_at IS NULL "
                "AND hint_path = ?",
                (canonical, canonical, path),
            )
            return cursor.rowcount

    def _record(
        self, path: str, status: str, canonical: str | None = None
    ) -> None:
        """Record the judge's verdict (cooldown: the group is no longer a candidate)."""
        with session(self._settings) as conn, transaction(conn):
            conn.execute(
                "INSERT INTO promotions (hint_path, status, canonical_path) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(hint_path) DO UPDATE SET "
                "status = excluded.status, canonical_path = excluded.canonical_path, "
                "decided_at = excluded.decided_at",
                (path, status, canonical),
            )

    def _decided_hints(self) -> set[str]:
        """Groups with an issued verdict (cooldown: we do not poke the judge)."""
        with session(self._settings) as conn:
            rows = conn.execute("SELECT hint_path FROM promotions").fetchall()
        return {row["hint_path"] for row in rows}

    def _group_summaries(self, path: str) -> list[str]:
        """The group's top-3 summaries (by confidence, then id) — the description input."""
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT summary FROM notes "
                "WHERE namespace = 'default' AND deleted_at IS NULL "
                "AND hint_path = ? "
                "AND summary != '' "
                "ORDER BY confidence DESC, id LIMIT ?",
                (path, SUMMARIES_PER_CANDIDATE),
            ).fetchall()
        return [row["summary"] for row in rows]

    def _nearest_node(self, description: str) -> tuple[str | None, float | None]:
        """The cosine prefilter: the nearest node by description (embeddings).

                    The candidate's description + the thematic nodes' descriptions of the
                    registry — in one embed_texts batch (few descriptions — 3–7 nodes).
                    default is excluded: merging the candidate into the swap is
                    meaningless, it already lies there. Returns (path, cosine) of the
                    nearest or (None, None) when the registry is empty / the embedding
                    failed. An encoding failure (EmbeddingError) — the prefilter simply
                    finds no nearest: the gate stays with the judge (a degradation, not a
                    pipeline failure).

                    The cosine is computed over L2-normalized vectors: a dot product over
                    normalized ones = an honest cosine, insensitive to the provider's
                    scale (symmetric to `vectors.py`, where the cosine does not depend on
                    the norm). A zero vector norm — the pair gives cosine 0.0 (no merge —
                    the safe direction).
        """
        nodes = self._thematic_nodes()
        if not nodes:
            return None, None
        try:
            vectors = self._embedding.embed_texts(
                [description] + [node["description"] for node in nodes]
            )
        except Exception:
            logging.getLogger("app").warning(
                "promotion: embedding failed — cosine prefilter skipped",
                extra={"event": "promotion_prefilter_skipped"},
            )
            return None, None
        candidate_vec, node_vecs = vectors[0], vectors[1:]
        # L2 normalization: a dot product over normalized ones = an honest cosine,
        # independent of the provider's scale.
        candidate_norm = _l2_norm(candidate_vec)
        if candidate_norm == 0.0:
            return None, None
        candidate_normed = [v / candidate_norm for v in candidate_vec]
        best_index = 0
        best_cosine = -1.0
        for i, node_vec in enumerate(node_vecs):
            node_norm = _l2_norm(node_vec)
            if node_norm == 0.0:
                continue
            node_normed = [v / node_norm for v in node_vec]
            cos = sum(
                a * b for a, b in zip(candidate_normed, node_normed)
            )
            if cos > best_cosine:
                best_cosine = cos
                best_index = i
        if best_cosine == -1.0:
            # All nodes are zero — the gate stays with the judge.
            return None, None
        return nodes[best_index]["path"], best_cosine

    def _day_limit_reached(self) -> bool:
        """NAMESPACE_AUTO_MAX_PER_DAY: provisional nodes created today (UTC)."""
        with session(self._settings) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM namespaces "
                "WHERE status = 'provisional' "
                "AND created_at >= strftime('%Y-%m-%dT00:00:00Z','now')"
            ).fetchone()[0]
        return int(count) >= self._settings.namespace_auto_max_per_day

    def _leaves_limit_reached(self, domain: str) -> bool:
        """NAMESPACE_MAX_LEAVES_PER_DOMAIN: the leaves ceiling in a root."""
        with session(self._settings) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM namespaces WHERE path LIKE ? || '/%'",
                (domain,),
            ).fetchone()[0]
        return int(count) >= self._settings.namespace_max_leaves_per_domain