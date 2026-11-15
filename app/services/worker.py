"""Background worker (ARCHITECTURE §3.4): pre-vectorization + pre-summarization.

    The only worker per process, **four independent loops per SLOTS**
    (Phase 11, decision #10; release 3.0.0 — the areas loop) — the states are
    pending statuses in the DB (they survive a restart and are caught up at service
    start):

    - **the embedding job** (`build_embedding_job`): note vectors (`pending_vector` →
      an `embed_texts` batch → notes_vec, vector_status='ok'; the full vector — over
      the title+text concatenation, `_embed_input`, lsb-0001 FR-1.1) + the chunk
      queue (Phase 7, the anti-join "no row in notes_chunks_vec"). Merged into one
      loop; the EMBEDDING_CONCURRENT_REQUESTS Semaphore stays. After each note's
      vector is ready a judge job (dedup) is created — the dependency dispatcher
      (decision #10): the judge is asked only about a vectorized note. The same
      moment wakes the `links` job (`notify_links_pending`,
      the gate-1c decision): link level 1 is computed right after vectorization,
      not after an interval/back-off. The embedding job's shape is "by interval +
      the `embedding` event + a queue recheck" (gate 2026-09-19; the recheck of the
      lost wakeup — LSBN-T-30): writing/editing a note
      (`memory_save`/`memory_update`), merging duplicates and title generation
      return the note to `vector_status='pending'` and wake the loop with the
      `notify_embedding_pending` signal — a fresh note is vectorized at once, not
      after the grown back-off (up to 15 min); the `_embedding_queue_empty` recheck
      insures the signal erased by `clear()` (pool 6). The event only accelerates:
      the task lives in the note's status; with unavailable models
      (`EmbeddingError`) there is no signal and the loop honestly goes into a wait
      with back-off (a failure is not progress, arch §3.2: the recheck gives no
      idle storm).
    - **the summary job** (`build_summary_job`): title generation (migration, title
      IS NULL) → summarize → merge (duplicate merging) → classification → node
      description. The notify wakes this loop (save/update). A note's ready summary
      wakes the `nodes` loop too (`notify_nodes_pending`, lsb-0012, modeled on
      `notify_links_pending` after vectorization): its `reclass` task is placed at
      the moment of duplicate stitching, while the merged note's summary is still
      being recomputed — and it waits exactly for that moment. The signal — only
      on actual readiness (`summary_status='ok'`): on a summarization failure there
      is none, and the task honestly waits on back-off.
    - **the judge job** (`build_judge_job`): the dedup judge (over judge jobs
      created by the embedding loop) + the structure judge (inside
      PromotionService, the trigger after classification).
    - **the areas job** (`build_areas_job`, substrate 3.0.0): area records' vectors
      (skills/terms/user; `vector_status='pending'` → an `embed_texts` batch → the
      area's vec0, 'ok'). No LLM at write time (substrate architecture §3.3):
      area texts — `AreaSpec.embed_text` (skills `name + description`,
      terms `term + context + definition`, user `name + body`); a failure — an
      `area_embed_failed` event, the records stay pending, a retry on its own
      back-off. A model/dimension change drops area-vecs together with notes_vec
      (`db._sync_embedding_meta`) — all area records return to pending.

    The loops are described by `JobSpec`s and served by the job framework
    (lsb-0014-02, `app/services/jobs.py`): loop bodies, back-off and the iteration
    supervisor are shared; the worker keeps the work (`process_*`), the queue
    recheck, the signals and hygiene. Own job builders — `build_*_job` at the end
    of the module (registration — a line in the framework registry). Work events
    carry the mandatory `job` field (FR-1.4).
    Queue snapshots for `/health.queues` are assembled by `queues_health` over the
    registry: each job describes its own queue itself (`queue_stat`); `/health` is
    not edited when a job is added (lsb-0014-03, FR-2.2).

    The `nodes` job (lsb-0011) parses the accumulated `default` and reclassifies
    merged notes: (1) promotion, (2) `reclass` tasks after stitching
    (lsb-0012 — the priority source), (3) the fast sweep pool — a move by the ready
    markup (`hint_path` + `confidence`) without model calls,
    (4) the classifier sweep pool — a classifier call within the hard budget
    (`JOB_NODES_CLASSIFIER_BUDGET`) over `title` + a ready summary. The move
    mechanics is one for all sources and for the grooming after summarization
    (`_apply_node_order`). The `node_order_at` marker — anti-looping (set on any
    parse outcome).

    DB job queues per slots (`worker_jobs`): a judge job (kind='dedup') is created
    ONLY after the note's vector is ready; a merge job (kind='merge') is created
    after the judge's verdict and walks into the summary slot. The note order in
    the queue — by id. Back-off is separate per loop (30 s → ×2 → 15 min, as now);
    notify wakes the summary loop; save/update do not vectorize synchronously —
    vectorization is background (the embedding loop).

    Dedup and ONE source (3.6.2, LSBN-D-14): the parts of a split are not
    duplicates of each other - the own origin group is dropped by the candidate
    prefilter (`find_candidates`), the merge path carries the same barrier, and a
    frozen pair of one source is closed, not released (see process_merge_pending
    and _review_attention_jobs).

    Title generation (decision #9): notes with `title IS NULL` (only migration
    ones — new notes always have a title) → a thinking call of the summary slot
    (think per the flag, like summarization); the result is trimmed to
    TITLE_MAX_WORDS words mechanically; the title is written; the queue is filled
    only by migration and empties after processing. The generation prompt is baked
    into the worker (TITLE_PROMPT), never created by a file.

    Guarantees:
    - a failure of any external server does not spoil the data: the status stays
      pending (NFR-3); the polling interval grows by back-off: PENDING_RETRY_SEC
      (30 s) → ×2 → max 15 minutes (REQUIREMENTS §5.3) — **independently per
      loop** (ARCH §3.4): an unavailable vectorizer does not stop summarization and
      vice versa; success in a loop resets only its interval and keeps draining it
      immediately.
    - concurrent access — via busy_timeout/WAL (§3.3); each batch — short
      transactions, parallelism with queries is safe.
    - `process_*` are synchronous (run in `asyncio.to_thread` — the event loop is
      not occupied); the Phase-7 chunk queue has its own async handling: encoding
      sub-batches is multiplied by a Semaphore right in the loop, write transactions
      — short ones in to_thread. `run` — an asyncio task (all registry jobs under
      the framework `run_loop`), start/stop — in lifespan.

    The summary write is protected from the memory_update race: between reading
    the text and writing, the worker could receive an updated text — the UPDATE is
    constrained by `AND text = ?` (the same text; otherwise the summary of a stale
    text would overwrite the fresh one). Vectorization is protected the same way
    (pool 1): the notes loop writes the vector only when unchanged since the read
    (id, text, title, namespace, vector_status) — a guard UPDATE in the same
    transaction; a stale batch is not written, the retry — the next batch (the
    analogue of `AND text = ?` at summarization and `upsert_vector_if_exists` at
    chunks). The title is part of the vector (lsb-0001 FR-1.1): writing the title
    (title generation) in the same transaction returns vectorization to pending
    and resets notes_vec — the old vector over the bare text feeds neither search
    nor dedup until re-encoding.

    The chunk queue (Phase 7) closes both problems:
    - the single-chunk reuse (brief §6): for a note with exactly one chunk
      ≤ CHUNK_SIZE the chunk's vector = the full text's vector from notes_vec,
      without calling the encoder. This is the case where the step-3 reuse did not
      make it at save — at write time the full vector did not exist yet (an Ollama
      failure); encoding the same text a second time is pointless.
    - the update race (ARCH §4.5, the analogue of `AND text = ?`): a chunk's vector
      is written only if (id, text, tokens) did not change since the read — update
      could have replaced the chunks (DELETE+INSERT; ids are reused on re-insert,
      which would give a foreign text's vector on a new id).

    External server statuses (for `/health.*_ok`, NFR-4) are kept by the services
    themselves — the worker does not aggregate: all encodings go through
    EmbeddingService (`embedding_ok` is updated in `embed_texts` — the single
    encoding point), all generations — through the Summarizer.

    The summarizer is injected by DI (build_services): None — the loop does not
    start (the Phase-3 test mode); in prod a SummaryService is always passed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from typing import TYPE_CHECKING
import threading
import time

from app.config import TITLE_MAX_WORDS, Settings
from app.services import history
from app.services import jobs
from app.services.areas import AreaSpec, ALL_AREAS, enabled_areas
from app.services.classifier import Classification, ClassificationError, Classifier
from app.services.decision import SLOT_DECISION, DecisionService
from app.services.dedup import DeduplicationService
from app.services.embedding import Embedder, EmbeddingError
# The back-off and its ceiling moved into the job framework (lsb-0014, arch §3.3):
# the framework owns the reliability contract; the worker's loops reuse the names.
# `MAX_INTERVAL_SEC as MAX_INTERVAL_SEC` — an explicit re-export: existing tests
# still reference it (tests/test_worker.py).
from app.services.jobs import MAX_INTERVAL_SEC as MAX_INTERVAL_SEC
from app.services.jobs import (
    BackoffState,
    JobSpec,
    build_job_specs,
    gated_queue_stat,
    next_interval,
    queue_snapshot,
    run_loop,
)
from app.services.judge import Judge, JudgeError
from app.services.llm_client import CHECK_AUTH_FAILED, CHECK_OK, ChatMeta
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.services.operations import (
    OPERATION_REGISTRY,
    SLOT_MODEL_OPERATION,
    breaker_key,
    pause_after,
    pause_sec,
    slot_model,
)
from app.services.promotion import PromotionService
from app.services.slot_breaker import SlotBreaker
from app.services.slot_budget import slot_config_fingerprint
from app.services.slot_outcome import SlotOutcome, outcome_of

if TYPE_CHECKING:  # a DI annotation only: warnings → worker, no cycle
    from app.services.warnings import WarningsService
from app.services.summary import (
    MERGE_MAX_ROUNDS,
    MergeIncompleteError,
    Summarizer,
    SummaryError,
    cap_summary,
)
from app.services.terms import TermsService
from app.storage import area_vectors, chunks, vectors
from app.storage.db import delete_note_physical, session, transaction

# How long to keep done jobs in worker_jobs (retention). Not env —
# following the TITLE_MAX_WORDS pattern: after this term the jobs are cleaned by the idle branch
# of the embedding loop (_purge_done_jobs); the queue does not grow without bound.
WORKER_JOBS_RETENTION_DAYS = 7

# The expired-notes cleanup job interval (lsb-0004-02, stage 4):
# a fixed 5 minutes, without a compose setting.
# A job with no queue and no event: its back-off state is marked
# `fixed=True` — the framework holds the 300 s pause, the back-off does not grow (the
# behavior preserved verbatim, FR-1.5).
EXPIRATION_CLEANUP_INTERVAL_SEC = 5 * 60

# The worker's job names (the lsb-0014 framework): `JobSpec.name`, the back-off state key
# and the mandatory `job` field in the journal — one line per job.
EMBEDDING_JOB = "embedding"
SUMMARY_JOB = "summary"
JUDGE_JOB = "judge"
AREAS_JOB = "areas"
EXPIRATION_JOB = "expiration"
# The "node order" job (lsb-0011-01): sweeping the accumulated `default`.
NODES_JOB = "nodes"

# The split jobs of the summary cycle (release 3.9.0, LSBN-S-6): in the
# `operation` mode the single `summary` job (title-gen → summarize → merge)
# is cut into three jobs — one per operation. Each job has its own queue,
# its own signal and its own breaker; in the `legacy` mode (the default)
# the whole cycle stays in the `summary` job, as before.
TITLE_JOB = "title"
SUMMARIZE_JOB = "summarize"
MERGE_JOB = "merge"

# The jobs' LLM slots (#39, arch §6.2): the breaker lives on the SLOT, not the job —
# the summary slot is shared by summarization/merge/classification/descriptions and the
# `nodes` sweep; the judge slot — the dedup judge and the structure judge.
SLOT_EMBEDDING = "embedding"
SLOT_SUMMARY = "summary"
SLOT_JUDGE = "judge"
SLOT_NAMES = (SLOT_EMBEDDING, SLOT_SUMMARY, SLOT_JUDGE)

# The `nodes` slot's job name after stitching (lsb-0012, arch §3.5): the merged
# note may live in ANY node — the `default` sweep will not find it, so
# reclassification is placed as a task.
NODES_RECLASS_KIND = "reclass"

#: The deferred merge pair's warning kind (LSBN-F-23, p. 6): within 2 rounds
#: the model gave no full coverage — both notes are alive; the pair awaits attention.
MERGE_DEFERRED_WARNING = "merge_deferred"

# The pair attempt counter and the "needs attention" state (#39, arch §6.8):
# non-transient outcomes freeze the pair — the task stops being taken
# (no model calls burn) but stays `pending`. `unavailable`/`timeout`/
# `auth` do NOT freeze the counter: back-off and the slot breaker handle them.
_NON_TRANSIENT_OUTCOMES = frozenset(
    {SlotOutcome.EMPTY, SlotOutcome.TRUNCATED, SlotOutcome.OFF_CONTRACT}
)

# The wait-class outcomes that make continuing the batch pointless (LSBN-T-28):
# `unavailable`/`timeout` — the provider is down or hanging; each next position
# of the batch is one more potential read timeout (900 s: config.py
# `summary_timeout_sec`/`judge_timeout_sec`), so a batch of up to
# `embedding_batch_size` positions could hold the loop and a thread for up to 8 hours,
# neither in back-off nor writing `queue_waiting` — the degradation invisible in
# the journal. `auth` — waiting is pointless in principle: the slot breaker opens
# immediately (`slot_breaker.note_failure`), and until it is lifted the framework gate (LSBN-T-29)
# will not let the batch through. EMPTY/TRUNCATED/OFF_CONTRACT stay per-item (`continue`):
# they do not mean "the provider is down" (they return fast), and their coverage — the
# slot breaker after three failures of one outcome (`slot_breaker.py`). The
# batch-release precedent — the migration job (`if outcome == "retry-later": break`).
# The "no work is lost" requirement: releasing the batch does not
# contradict it — the failed position stays pending (as does the batch tail);
# it is picked up by the loop's back-off, the breaker's probe knock after the pause and the
# notifier's wake on a new save; the return — the count processed BEFORE the failure
# (done work counts, as with migration).
_BATCH_ABORT_OUTCOMES = frozenset(
    {SlotOutcome.UNAVAILABLE, SlotOutcome.TIMEOUT, SlotOutcome.AUTH}
)

# Slot → the owning job's queue name (for attention events and /jobs/attention).
_SLOT_JOB_NAMES = {
    SLOT_EMBEDDING: EMBEDDING_JOB,
    SLOT_SUMMARY: SUMMARY_JOB,
    SLOT_JUDGE: JUDGE_JOB,
    NODES_JOB: NODES_JOB,
}

# An OPERATION → the job that works it (release 3.9.0, LSBN-S-9): the journal
# field `job` stays the FACTUAL job (the operation is visible by the breaker
# key — the `slot` field of the event), so the owner of a task row (an
# operation in the `operation` mode) needs the name of its job: the reclass
# tasks belong to the `nodes` job, the tasks of the summary family — to the
# split jobs of the cycle (`title`/`summarize`/`merge`), the dedup verdict — to
# the `judge` job. An operation without a job of its own (`describe`,
# `structure_judge`, `embed`, `decision`) has no task rows — the name of the
# operation is the fallback.
_OPERATION_JOB_NAMES = {
    "summarize": SUMMARIZE_JOB,
    "title": TITLE_JOB,
    "merge": MERGE_JOB,
    "judge": JUDGE_JOB,
    "classify": NODES_JOB,
}

# The operations whose path goes through the DECISION slot (the design, B4):
# with the decision unavailable and the fallback off, they wait — the dedup
# judge and the classifier walk into the decision slot. The rest
# (summarize/title/merge/split/describe/embed) go.
# The decision rule of the former `summary` slot (no39-05) does NOT hold for
# the summary family (title/summarize/merge) any more: the family goes.
_DECISION_BLOCKED_OPERATIONS = ("judge", "classify")

# A task row → the operation that owns it (release 3.9.0, LSBN-S-7): the
# attention state is counted and reset per OPERATION in the `operation` mode
# (a title failure must not clear the attention of a summarize pair). The
# rows keep their legacy slot values until the migration of `worker_jobs.slot`
# (LSBN-S-14) — this map is the reader of those values. An unknown pair keeps
# its slot value: the task is not lost (it stays visible in attention).
_OPERATION_BY_TASK: dict[tuple[str, str], str] = {
    (SLOT_SUMMARY, "merge"): MERGE_JOB,
    (SLOT_SUMMARY, TITLE_JOB): TITLE_JOB,
    (SLOT_SUMMARY, SUMMARIZE_JOB): SUMMARIZE_JOB,
    (SLOT_JUDGE, "dedup"): SLOT_JUDGE,
    (NODES_JOB, NODES_RECLASS_KIND): "classify",
}


def _call_meta_fields(meta: ChatMeta) -> dict[str, object]:
    """Call fields for a failure event (§7.2): numbers/flags, no secrets.

            LSBN-F-34 (the brief's decision #4): the slot address (`base_url`) and the
            operation configuration source (`config_source`: op|slot|default) go into
            the journal — into the successful `slot_call` (via `_call_log_fields`) and
            into the job's event on error. The outward surfaces (/health) do not show
            addresses.
    """
    return {
        "operation": meta.operation,
        "model_type": meta.model_type,
        "budget": meta.budget,
        "budget_source": meta.budget_source,
        # LSBN-T-33: separate call budgets — the journal shows how much
        # was allotted to the answer and how much to reasoning (and whether there was an auto-retry).
        "answer_budget": meta.answer_budget,
        "reasoning_budget": meta.reasoning_budget,
        "sent_budget": meta.sent_budget,
        "reasoning_source": meta.reasoning_source,
        "escalated_from": meta.escalated_from,
        "base_url": meta.base_url,
        "config_source": meta.config_source,
        "prompt_tokens": meta.prompt_tokens,
        "completion_tokens": meta.completion_tokens,
        "reasoning_tokens": meta.reasoning_tokens,
        "had_thinking": meta.had_thinking,
        "finish_reason": meta.finish_reason,
        "duration_ms": meta.duration_ms,
    }


def _call_log_fields(meta: ChatMeta) -> dict[str, object]:
    """The full set of fields of a successful `slot_call` (#49, §7.2)."""
    fields = _call_meta_fields(meta)
    fields.update(
        {
            "provider": meta.provider,
            "model": meta.model,
            "thinking_chars": meta.thinking_chars,
            "truncated": meta.truncated,
        }
    )
    return fields

# The meta key of the slot configuration fingerprint (#39, arch §6.8/§13.3): a
# slot configuration change is part of the attention "reason", so a start with a changed
# fingerprint clears `attention_at` of all tasks. The fingerprint's composition lives in
# `slot_budget.slot_config_fingerprint` (lsbdef0008-02): the model, kind, budget,
# context, reserve, ceiling and decision parameters — one implementation system-wide.
SLOT_CONFIG_META_KEY = "slot_config_fingerprint"

# The single selection rule of the fast `default` sweep pool (lsb-0011-01, arch §3.2):
# active default notes with ready markup (`hint_path` + `confidence` not
# below the auto-move threshold), not yet parsed (`node_order_at IS NULL`),
# newest first — a move without a model call.
_NODES_SWEEP_SELECT = (
    "SELECT id, hint_path, confidence FROM notes "
    "WHERE namespace = 'default' AND deleted_at IS NULL "
    "AND node_order_at IS NULL AND hint_path IS NOT NULL AND confidence >= ? "
    "ORDER BY updated_at DESC, id DESC LIMIT ?"
)

# The classifier sweep pool of `default` (lsb-0011-02, arch §3.2): active
# default notes WITHOUT markup (`classified_at IS NULL`) but with a ready non-empty
# summary, not yet parsed, newest first — a classifier call over
# `title` + `summary` (arch §3.4). A not-ready summary does not enter the pool: the note
# waits and gets no marker (FR-2.4). The call count is limited by the budget
# `JOB_NODES_CLASSIFIER_BUDGET`.
_NODES_CLASSIFY_SELECT = (
    "SELECT id, title, summary FROM notes "
    "WHERE namespace = 'default' AND deleted_at IS NULL "
    "AND node_order_at IS NULL AND classified_at IS NULL "
    "AND summary_status = 'ok' AND summary IS NOT NULL AND summary <> '' "
    "ORDER BY updated_at DESC, id DESC LIMIT ?"
)

# The predicate "a `reclass` task is takeable in THIS run" (lsb-0012, arch §3.5):
# `_process_reclass_job` releases the task when the note is gone or in trash
# (nothing to decide), and decides it over the READY non-empty summary; a task waiting
# for the summary stays pending and does not count as progress. The queue snapshot counts
# only takeable tasks: one waiting for the summary is not the run's work (the note itself
# is visible in the `summary` queue), otherwise the counter would show work the job
# does not have (a 3.1.0 acceptance FAIL, the 2026-09-19 gate decision: `pending=1` held for 7
# minutes — how long the merged note waited for its summary).
_NODES_RECLASS_TAKEABLE = (
    "(n.id IS NULL OR n.deleted_at IS NOT NULL "
    "OR (n.summary_status = 'ok' AND trim(coalesce(n.summary, '')) <> ''))"
)


def _nodes_queue_stat_sql() -> str:
    """The SQL of the `nodes` queue snapshot under the run's ACTUAL pools (FR-2.2).

            `pending` — exactly the work `process_nodes` can take in this run, and
            nothing more:
            (1) `reclass` tasks after stitching (lsb-0012) that are solvable now
            (`_NODES_RECLASS_TAKEABLE`);
            (2) candidates of the fast `default` sweep pool — the same predicate as
            `_NODES_SWEEP_SELECT` (a move by the ready markup, no model);
            (3) candidates of the classifier pool — the same predicate as
            `_NODES_CLASSIFY_SELECT` (a ready non-empty summary, no markup yet).
            `oldest_pending_sec` — the age of the oldest AMONG THE COUNTED sources
            (tasks — `created_at`, candidates — `updated_at`); `null` — the queue is
            empty. SQL only, no model calls; the snapshot works for a disabled job
            too. The `:threshold` parameter — the auto-move threshold
            (`NAMESPACE_AUTO_MOVE_MIN_CONFIDENCE`).
    """
    reclass_jobs = (
        "SELECT COUNT(*) FROM worker_jobs j "
        "LEFT JOIN notes n ON n.id = j.note_id "
        "WHERE j.slot = 'nodes' AND j.kind = 'reclass' "
        f"AND j.status = 'pending' AND {_NODES_RECLASS_TAKEABLE}"
    )
    reclass_age = (
        "SELECT MAX(CAST(strftime('%s','now') AS INTEGER) "
        "- CAST(strftime('%s', j.created_at) AS INTEGER)) "
        "FROM worker_jobs j LEFT JOIN notes n ON n.id = j.note_id "
        "WHERE j.slot = 'nodes' AND j.kind = 'reclass' "
        f"AND j.status = 'pending' AND {_NODES_RECLASS_TAKEABLE}"
    )
    fast_pool = (
        "SELECT COUNT(*) FROM notes WHERE namespace = 'default' "
        "AND deleted_at IS NULL AND node_order_at IS NULL "
        "AND hint_path IS NOT NULL AND confidence >= :threshold"
    )
    fast_age = (
        "SELECT MAX(CAST(strftime('%s','now') AS INTEGER) "
        "- CAST(strftime('%s', updated_at) AS INTEGER)) FROM notes "
        "WHERE namespace = 'default' AND deleted_at IS NULL "
        "AND node_order_at IS NULL AND hint_path IS NOT NULL "
        "AND confidence >= :threshold"
    )
    classifier_pool = (
        "SELECT COUNT(*) FROM notes WHERE namespace = 'default' "
        "AND deleted_at IS NULL AND node_order_at IS NULL "
        "AND classified_at IS NULL AND summary_status = 'ok' "
        "AND summary IS NOT NULL AND summary <> ''"
    )
    classifier_age = (
        "SELECT MAX(CAST(strftime('%s','now') AS INTEGER) "
        "- CAST(strftime('%s', updated_at) AS INTEGER)) FROM notes "
        "WHERE namespace = 'default' AND deleted_at IS NULL "
        "AND node_order_at IS NULL AND classified_at IS NULL "
        "AND summary_status = 'ok' AND summary IS NOT NULL "
        "AND summary <> ''"
    )
    return (
        "SELECT "
        f"({reclass_jobs}) + ({fast_pool}) + ({classifier_pool}) AS pending, "
        "MAX("
        f"COALESCE(({reclass_age}), 0), "
        f"COALESCE(({fast_age}), 0), "
        f"COALESCE(({classifier_age}), 0)"
        ") AS oldest_pending_sec"
    )


# The title-generation prompt (decision #9): BAKED INTO SummaryService.title
# (follow-up 6b — the Summarizer protocol gained the title method; there used to be a
# dead duplicate of the constant here; generation ran with the summarization prompt).
# A thinking call of the summary slot; trimming to TITLE_MAX_WORDS — the worker's mechanics.


def _embed_input(title: str | None, text: str) -> str:
    """The input for encoding a note's full vector (lsb-0001 FR-1.1).

            The note vector is built over the title and text concatenation:
            f"{title}
        {text}". title is NULL (a migration note before title generation) —
            the bare text is encoded without a prefix. Chunk vectors are unaffected
            (decision D1): there the chunk's bare text is encoded without the title.
    """
    return text if title is None else f"{title}\n{text}"


class BackgroundWorker:
    """The only background worker; the queues — pending statuses in the DB + worker_jobs.

            The loop, back-off and the iteration supervisor are framework-side
            (app/services/jobs.py, lsb-0014-02): the worker describes its jobs with
            `JobSpec`s (the `build_*_job` builders at the end of the module) and
            registers them in the registry. Work events carry the mandatory `job`
            field (the name of the job owning the work).
    """

    def __init__(
        self,
        settings: Settings,
        embedding: Embedder,
        summarizer: Summarizer | None = None,
        dedup: DeduplicationService | None = None,
        judge: Judge | None = None,
        classifier: Classifier | None = None,
        promoter: PromotionService | None = None,
        terms: TermsService | None = None,
        decision: DecisionService | None = None,
        warnings: "WarningsService | None" = None,
    ) -> None:
        self._settings = settings
        self._embedding = embedding
        self._summarizer = summarizer
        # The LLM dedup judge (Phase 8, Stage 3.1, DI from build_services — one
        # instance per process): the "duplicate / not duplicate" verdict per each
        # cosine candidate (the judge loop, Stage 3.2). None — a test mode:
        # the worker merges by the Stage-2.2 cosine fallback.
        self._judge = judge
        # LSBN-F-12: the decision slot (None — disabled, v3.1.0 behavior). The decision
        # breaker lives INSIDE the service (§13.8); the worker does not keep it — we only take
        # its snapshot for `/health.slots.decision`.
        self._decision = decision
        # Background dedup (Phase 8, Stage 2): searching cosine candidates against
        # earlier notes. DI for tests.
        self._dedup = (
            dedup if dedup is not None else DeduplicationService(settings)
        )
        # The warnings surface (#12): F-23 p. 6 writes "the merge
        # is deferred" into warnings; None — test DI builds (journal only).
        self._warnings = warnings
        # Duplicate merging (Stage 2.2 → F-23) goes through the regular NOTE logic:
        # part-notes are created and both originals go to trash
        # (NoteService.merge_into_parts, one transaction). The service is assembled
        # from the same deps as the worker (embedding — a DI fake in unit tests);
        # it needs no summarization notifier — after the merge the worker wakes
        # its own loops (notify_summary_pending/notify_embedding_pending).
        self._notes = NoteService(settings, embedding=embedding)
        # Grooming (Phase 10, Step 4): the default-note classifier after
        # summarization; None — a test mode without classification.
        self._classifier = classifier
        # The namespace registry: known nodes for the classifier and the check
        # of the auto-move target node.
        self._namespaces = NamespaceService(settings)
        # The terms area (3.0.0, lsb-0008; 3.2.0 — the `terms-normalize` job):
        # the same instance as the transport/areas loop (DI from build_services).
        # None — a DI build without the area: the job builder is not registered (§6.6).
        self._terms = terms
        # The domain trigger (Phase 10, Step 5): automatic leaf creation from hint groups
        # after classification; None — a test mode without the trigger (in prod
        # DI from build_services: describer + structure judge).
        self._promoter = promoter
        # The per-job back-off state (the lsb-0014 framework): one object per job,
        # kept in the worker — the interval survives loop iterations and is visible in
        # diagnostics (the interval/summary_interval/judge_interval/
        # areas_interval properties). The starting intervals — as before: PENDING_RETRY_SEC for
        # queues, a fixed 300 s for expired-notes cleanup.
        # Cleanup — the only job with a FIXED schedule
        # (`fixed=True`): the pause is always 300 s, the back-off does not grow (FR-1.5).
        self._backoff: dict[str, BackoffState] = {
            EMBEDDING_JOB: BackoffState(settings.pending_retry_sec),
            SUMMARY_JOB: BackoffState(settings.pending_retry_sec),
            JUDGE_JOB: BackoffState(settings.pending_retry_sec),
            AREAS_JOB: BackoffState(settings.pending_retry_sec),
            EXPIRATION_JOB: BackoffState(
                EXPIRATION_CLEANUP_INTERVAL_SEC, fixed=True
            ),
        }
        # Slot breakers (#39, arch §6.2): the SLOT's state, not the job's —
        # different loops share a slot (summary: summarization/merge/
        # classification/descriptions and the `nodes` sweep; judge: dedup and structure),
        # so the breaker lives in one point per slot. The breaker before a run is
        # checked by the framework gate (`JobSpec.gate` → `_slot_gate`, 3.4.1,
        # LSBN-T-29: a single point, no manual `if`s in job bodies),
        # `_log_slot_failure` records the failure outcome.
        # The pause — SLOT_PAUSE_SEC (the back-off ceiling); the opening threshold —
        # SLOT_PAUSE_AFTER (auth opens immediately). The state is in process
        # memory: no restart survival needed (the slot configuration
        # is re-evaluated by the startup check, arch §13.2).
        # Release 3.9.0 (LSBN-S-5): the KEY of the breaker is the operation in
        # the `operation` mode (`SLOT_MODEL=operation`) — one breaker per
        # operation; in the `legacy` mode the key stays the slot group (the
        # historical behaviour). The mode is read through the ONE switch
        # (`_mode`), never inlined.
        self._breakers: dict[str, SlotBreaker] = self._build_breakers(settings)
        self._stopping = False
        # The "a note with a pending vector appeared" signal — wakes the embedding loop
        # right after the write (save/update/merge/title-gen), bypassing the grown
        # back-off: a fresh note is encoded immediately, not in 15 min
        # (gate 2026-09-19; the summary-loop precedent). It is set by NoteService
        # (a DI notifier from main.py) and by the worker itself — on the paths where it writes
        # a note (merging, title generation).
        self._embedding_event = asyncio.Event()
        # The "a note with a pending summary appeared" signal — wakes the
        # summarization loop immediately (save/update), bypassing the grown back-off.
        self._summary_event = asyncio.Event()
        # The "a note with a pending title appeared" signal (release 3.9.0,
        # LSBN-S-6): wakes the title loop of the `operation` mode. Its OWN
        # event: a shared `asyncio.Event` would be cleared by a neighbour loop
        # (`jobs.py` clears the event before the queue recheck) — the neighbour
        # would lose the wakeup.
        self._title_event = asyncio.Event()
        # The "a merge job appeared" signal (release 3.9.0, LSBN-S-6): wakes
        # the merge loop of the `operation` mode. The signal is set at the
        # merge job's creation (the judge loop) — its own event, for the same
        # reason as the title one.
        self._merge_event = asyncio.Event()
        # The "a judge job appeared" signal — wakes the judge loop (the embedding loop
        # created a dedup job after vectorization).
        self._judge_event = asyncio.Event()
        # The "a pending area record appeared" signal — wakes the areas loop
        # (the areas' save/update services call notify_areas_pending): the text
        # is written instantly, the vector is caught up in the background (substrate
        # architecture §3.3 — no LLM at write time).
        self._areas_event = asyncio.Event()
        # The "an after-stitching task appeared" signal (lsb-0012): wakes the
        # `nodes` loop right after the merge — no hourly wait. The signal is set by
        # the same summary loop (`process_merge_pending`); no DI notifier needed:
        # the job lives inside the worker (arch §3.5).
        self._nodes_event = asyncio.Event()
        # The "a note got a ready vector" signal (the gate-1c decision): wakes the
        # `links` loop right after vectorization — link level 1
        # appears immediately instead of waiting for the interval/back-off. It is set by
        # the embedding loop (`process_pending`); the `links` job is described in its own
        # module (`links.py`) but takes the event from the worker (the shared owner
        # of the embedding queue; no DI notifier needed).
        self._links_event = asyncio.Event()
        # The worker's own note service wakes the worker's embedding loop on
        # the paths where the worker writes a note itself (`merge_pair` returns the earlier
        # note to the vectorization queue) — the API layer's DI notifier is not
        # available here (the service is assembled before the events).
        self._notes.set_vector_notifier(self.notify_embedding_pending)
        # Memoizing the job-queue table creation (pool 5): the DDL runs
        # once per worker instance, not on every queue access
        # (_create_job/_pending_jobs/_mark_job_done called _ensure_job_table
        # on every call — 3 times per job).
        self._jobs_table_ready = False
        # The lazy migration lock (lsbdef-0013): a check-then-act on a bare flag
        # above let two threads into the ALTER at once — the job loops run
        # synchronous SQL through `asyncio.to_thread`, and the startup attention
        # release (`run`) runs in parallel with them; the second ALTER failed on
        # `duplicate column name` and crashed the loop iteration. The fast exit by
        # the flag stays lock-free; the DDL itself — under the lock.
        self._jobs_table_lock = threading.Lock()
        # A test hook (no39-04): how many times the slot released (tried to release)
        # work from attention, by reason (`probe`/`edit`/`operator`/`startup`).
        # The invariant "the pause is per slot, the count per pair" is checked by the "a knock
        # does not reset" test: a successful knock must NOT appear here with the reason
        # `probe` at all (`_reset_attention_for_slot(..., "probe")` was removed from
        # `_slot_gate`, no39-04).
        self._attention_resets: dict[str, int] = {}
        # Job queues per slots (Phase 11, decision #10): the table is created
        # by the worker lazily at the first queue access (the schema is the
        # worker's zone, not db.py). Not created in __init__: the worker is assembled in
        # create_app() at import time, when the DB is not initialized yet.

    @property
    def interval(self) -> float:
        """The embedding job's current interval (diagnostics, tests)."""
        return self._backoff[EMBEDDING_JOB].interval

    @property
    def summary_interval(self) -> float:
        """The summary job's current interval (diagnostics, tests)."""
        return self._backoff[SUMMARY_JOB].interval

    @property
    def chunk_interval(self) -> float:
        """The chunk queue's interval (diagnostics, tests, Phase 7).

                    The chunk queue is merged with the vector queue into one job
                    (decision #10) — the interval is shared with `interval`.
        """
        return self._backoff[EMBEDDING_JOB].interval

    @property
    def judge_interval(self) -> float:
        """The judge job's current interval (diagnostics, tests, Phase 11)."""
        return self._backoff[JUDGE_JOB].interval

    @property
    def areas_interval(self) -> float:
        """The areas job's current interval (diagnostics, tests, substrate 3.0.0)."""
        return self._backoff[AREAS_JOB].interval

    def _mode(self) -> str:
        """The active model of the slot layer: `legacy` | `operation`.

                The ONE switch of the release 3.9.0 (the design, B2): every
                point that depends on the mode reads this value — the flag
                lives in `Settings` (`SLOT_MODEL`) and is read once (the
                settings are frozen).

        """
        return slot_model(self._settings)

    def _operation_slot(self, operation: str) -> str:
        """The breaker key of an operation: the operation itself in the
        `operation` mode, the slot group in the `legacy` mode.

                The ONE point where the slot of an operation is resolved (the
                design, B2): the `legacy` mode keeps the slot group (one
                breaker per group — the historical behaviour), the `operation`
                mode gives every operation its own breaker. The registry's
                `breaker_key` is the mapping (S-5).

        """
        return breaker_key(operation, self._settings)

    def _operation_service(self, operation: str) -> object | None:
        """The service behind an operation — the probe map (LSBN-S-7).

                The breaker key of the `operation` mode is the OPERATION name,
                so the knock (the probe) must find the service by the
                operation and not by the slot group: the summarizer serves
                `summarize`/`title`/`merge`/`split`, the dedup judge serves
                `judge`, the structural judge serves `structure_judge`, the
                classifier serves `classify`, the promoter serves `describe`,
                the embedder serves `embed` and the decision service serves
                `decision`. An unknown operation — no service (`None`): the
                caller answers `CHECK_OK` (the precedent of an absent
                service).

                The `legacy` mode keeps the slot map (`embedding`/`summary`/
                `judge`) — the probe of a slot group is the service of that
                group, as before.

        """
        if self._mode() == SLOT_MODEL_OPERATION:
            if operation in (SUMMARIZE_JOB, TITLE_JOB, MERGE_JOB, "split"):
                return self._summarizer
            if operation == SLOT_JUDGE:
                return self._judge
            if operation == "structure_judge":
                # Release 3.9.0 (LSBN-S-8): the structural judge is a service of
                # its own (it lives inside the promotion trigger) — the dedup
                # judge is a different operation with its own breaker. No
                # trigger (a test mode) — the dedup judge, as before.
                structure_judge = self._structure_judge()
                return structure_judge if structure_judge is not None else self._judge
            if operation == "classify":
                return self._classifier
            if operation == "describe":
                return self._promoter
            if operation == "embed":
                return self._embedding
            if operation == SLOT_DECISION:
                return self._decision
            return None
        return {
            SLOT_EMBEDDING: self._embedding,
            SLOT_SUMMARY: self._summarizer,
            SLOT_JUDGE: self._judge,
        }.get(operation)

    def _structure_judge(self) -> object | None:
        """The structural judge service (the `structure_judge` operation).

                The service lives inside the promotion trigger (the trigger
                owns the describer and the structural judge); None — a test
                mode without the trigger.
        """
        return getattr(self._promoter, "structure_judge", None)

    def _task_operation(self, slot: str, kind: str | None = None) -> str:
        """The owner of a task row: the operation in the `operation` mode, the
                slot group in the `legacy` mode.

                The attention state is per PAIR (a `worker_jobs` row), but the
                wholesale reset ("the reason is gone") is per breaker key: in
                the `operation` mode that is the operation — a title failure
                must not clear the attention of a summarize pair (LSBN-S-7).

        """
        if self._mode() != SLOT_MODEL_OPERATION:
            return slot
        return _OPERATION_BY_TASK.get((slot, kind or ""), slot)

    def _job_name(self, operation: str) -> str:
        """The job name of an operation: the operation's own job in the
        `operation` mode (`title`/`summarize`/`merge`), the legacy `summary`
        job in the `legacy` mode (one job for the whole cycle).

                The journal field `job` and the breaker key travel together:
                in the `legacy` mode the whole summary family belongs to the
                `summary` job and to the `summary` slot group; in the
                `operation` mode every operation is a job of its own.

        """
        if self._mode() == SLOT_MODEL_OPERATION:
            return operation
        return SUMMARY_JOB

    def _breaker_paused(self, key: str) -> bool:
        """Is the breaker under the key paused right now? No breaker under the
        key — not paused.

                The `operation` mode keys the breakers by operation (the
                design, B2): a slot name that is not an operation (`summary`)
                has no breaker there. The lookup is defensive for the same
                reason as `_slot_blocked`: observability and the queue
                recheck must not crash the loop (LSBN-S-6, the S-5 finding
                M1 — a `KeyError` on the slot name).

        """
        breaker = self._breakers.get(key)
        return breaker is not None and breaker.is_paused()

    def _operation_blocked(self, operation: str) -> bool:
        """Is the operation's work blocked for its loop?

                The breaker of the operation (its own key in the `operation`
                mode) plus the decision rule (no39-05): with the decision slot
                unavailable and the fallback off, the operations whose path
                goes through the decision slot wait — the classifier walks
                into the decision slot. The `legacy` mode keeps the whole
                summary cycle under one rule (as before the split).

        """
        return self._slot_blocked(self._operation_slot(operation))

    def _classify_blocked(self) -> bool:
        """Is the classifier NOT to be called (the `classify` operation is blocked)?

                Release 3.9.0 (LSBN-S-9): in the `operation` mode the pause of
                the `classify` operation stops the CLASSIFIER calls of the paths
                that call it — the grooming (inside the `summarize` job), the
                reclass tasks and the sweep (inside the `nodes` job). The gate
                of a job stays the gate of its job (the `summarize` job is
                gated by the `summarize` operation — the check is INSIDE the
                path, the S-8 precedent: the structural judge is not called on
                the pause of its operation). The work is not lost: the note
                stays unclassified (the sweep of the `nodes` job takes it
                later), the reclass task stays `pending`.

                The `legacy` mode: False — the shared breaker of the `summary`
                slot group is read by the gate of the job (as before): the
                pause stopped the `summary` job as a whole, and it still does
                (the classifier is not called, but the job is not the one
                that is gated by the `classify` operation there).

        """
        if self._mode() != SLOT_MODEL_OPERATION:
            return False
        return self._operation_blocked("classify")

    def _build_breakers(self, settings: Settings) -> dict[str, SlotBreaker]:
        """The breaker dictionary: one breaker per SLOT GROUP (`legacy` mode)
        or one per OPERATION (`operation` mode).

                The thresholds of the `operation` mode are the per-operation
                ones (base + override: `<OP>_PAUSE_AFTER`/`<OP>_PAUSE_SEC`,
                else the global base); the `legacy` mode keeps the global
                values of the slot group.

        """
        if self._mode() == SLOT_MODEL_OPERATION:
            return {
                operation: SlotBreaker(
                    operation,
                    pause_sec(operation, settings),
                    open_after=pause_after(operation, settings),
                )
                for operation in OPERATION_REGISTRY
            }
        return {
            name: SlotBreaker(
                name,
                settings.slot_pause_sec,
                open_after=settings.slot_pause_after,
            )
            for name in SLOT_NAMES
        }

    def stop(self) -> None:
        """Soft stop: the loops will finish after unwinding the current iteration."""
        self._stopping = True

    def notify_embedding_pending(self) -> None:
        """Wake the embedding loop: a note returned to the vectorization queue.

                    Called from save/update/merge (NoteService, an `asyncio.to_thread`
                    thread) and from the worker itself (duplicate merging, title
                    generation) — `asyncio.Event.set()` is thread-safe. The loop
                    immediately leaves the wait and encodes pending notes without waiting
                    out the grown back-off (up to 15 min). The event only accelerates: the
                    task lives in the note's status itself (`vector_status='pending'`) and
                    will be picked by the next run. The signal's safety net — the queue
                    recheck (`queue_empty` → `_embedding_queue_empty`, LSBN-T-30): a
                    signal fired between an empty run and the event's `clear()` would be
                    erased — the queue select restores it with an immediate run (lost
                    wakeup closed). The recheck gives no idle storm with unavailable models
                    (`EmbeddingError`): a failure is not loop progress (arch §3.2,
                    no39-05); after one immediate run the loop honestly goes into a
                    back-off wait (see `build_embedding_job`).
        """
        self._embedding_event.set()

    def notify_summary_pending(self) -> None:
        """Wake the summarization loop: a note with a pending summary appeared.

                    Called from save/update (an `asyncio.to_thread` thread) —
                    `asyncio.Event.set()` is thread-safe. The loop immediately leaves the
                    wait and catches up the queue without waiting out the grown back-off.

                    The `operation` mode (release 3.9.0, LSBN-S-6): the summary
                    cycle is split into three loops — the signal wakes the
                    summarize loop AND the title loop (a note without a title
                    waits for the title operation). The merge loop has its own
                    signal — at the merge job's creation.
        """
        self._summary_event.set()
        if self._mode() == SLOT_MODEL_OPERATION:
            # The split summary cycle (release 3.9.0, LSBN-S-6): the signal
            # wakes the title loop too — a saved note without a title (the
            # migration path) waits for the title operation, and the note
            # service knows only one "summary pending" signal. The merge loop
            # has its own signal — at the merge job's creation.
            self._title_event.set()

    def notify_merge_pending(self) -> None:
        """Wake the merge loop: a merge job appeared.

                    Called at the merge job's creation (the judge loop) —
                    `asyncio.Event.set()` is thread-safe. The `operation`
                    mode's merge loop has its own event (LSBN-S-6): the job is
                    taken at once, not after the grown back-off.

        """
        self._merge_event.set()

    def notify_judge_pending(self) -> None:
        """Wake the judge loop: a judge job (dedup) appeared.

                    Called from the embedding loop after creating a dedup job —
                    `asyncio.Event.set()` is thread-safe.
        """
        self._judge_event.set()

    def notify_areas_pending(self) -> None:
        """Wake the areas loop: a pending area record appeared.

                    Called by the area services from save/update (an `asyncio.to_thread`
                    thread) — `asyncio.Event.set()` is thread-safe. The loop immediately
                    leaves the wait and catches up the vector queue without waiting out the
                    grown back-off.
        """
        self._areas_event.set()

    def notify_nodes_pending(self) -> None:
        """Wake the `nodes` loop: work appeared (lsb-0012, two occasions).

                    Called from the summary loop at two moments, both synchronous code in
                    `asyncio.to_thread`, so `asyncio.Event.set()` is thread-safe:

                    * after a successful stitch (`process_merge_pending`) — the merged
                      note's `reclass` task is placed, but its summary is still being
                      recomputed: the task legitimately waits;
                    * after the summary is actually ready (`process_summary_pending`,
                      `summary_status='ok'`) — the task waiting for that summary became
                      takeable; on a summarization failure there is no signal.

                    The loop immediately leaves the wait and parses `reclass` without
                    waiting for the hourly `JOB_NODES_INTERVAL_SEC`/grown back-off. The
                    task lives in a `worker_jobs` row — the event only accelerates (the
                    "a task is not lost" contract does not change). No DI notifier needed:
                    the job lives inside the worker (arch §3.5).
        """
        self._nodes_event.set()

    def notify_links_pending(self) -> None:
        """Wake the `links` loop: a note got a ready vector (gate 1c).

                    Called from the embedding loop after actual vectorization
                    (`process_pending` — synchronous SQL/HTTP in `asyncio.to_thread`);
                    `asyncio.Event.set()` is thread-safe. The loop immediately leaves the
                    wait and parses the links-computation queue (`links_at IS NULL`)
                    without waiting for `JOB_LINKS_INTERVAL_SEC`/the grown back-off. The
                    task lives in the `links_at` marker in the DB — the event only
                    accelerates (the reliability contract does not change); with
                    unavailable models (`EmbeddingError`) there is no signal, the `links`
                    queue is empty and the loop waits on back-off.
        """
        self._links_event.set()

    def _is_stopping(self) -> bool:
        """The soft-stop predicate for the framework loop (`run_loop`)."""
        return self._stopping

    def _job_backoff(self, spec: JobSpec) -> BackoffState:
        """A job's back-off state (for an unregistered one — its interval)."""
        return self._backoff.setdefault(
            spec.name, BackoffState(spec.interval_sec)
        )

    async def run(self) -> None:
        """All registry jobs — one framework loop (an asyncio task).

                    The jobs are assembled by `build_job_specs` (the framework's registry
                    of `JobSpec`s; the worker registers its own jobs at the end of the
                    module) and served by `run_loop`. Processed batches go one after
                    another (we drain the queue at once); an empty run — waiting for a
                    `notify_*` signal (the "on demand" form) or a back-off pause. Jobs are
                    independent: the interval and draining are separate (a `BackoffState`
                    per job).
        """
        specs = build_job_specs(self, self._settings)
        await asyncio.gather(
            *(
                run_loop(spec, self._is_stopping, self._job_backoff(spec))
                for spec in specs
            ),
            # A start with a changed slot configuration — "the reason is gone": attention
            # is cleared, the fingerprint rewritten (#39 §6.8/§13.3). The startup
            # check runs IN PARALLEL with the loops (it does not delay the first run).
            self._startup_attention_reset_async(),
        )

    # --- slot breakers (#39, arch §6.2/§6.7) ----------------------------------

    def slot_states(self) -> dict[str, dict]:
        """A snapshot of slot breakers (for the journal and a human hint; #39).

                    Reading process memory only — no model calls. The consumer — the MCP
                    middleware of the "warn your human" hint. The full `/health.slots`
                    surface and queue observability — the no39-03 brief.
        """
        return {name: breaker.snapshot() for name, breaker in self._breakers.items()}

    def slots_health(self) -> dict[str, dict]:
        """A slots snapshot for `/health.slots` (#39, arch §7.2): SQL/memory only.

                    Per slot (`embedding`/`summary`/`judge`) — the breaker snapshot
                    (`SlotBreaker.snapshot`: `last_outcome`, `paused_until`, etc.), the
                    last attempt's outcome (`ok` — the `embedding_ok`/`summarizer_ok`/
                    `judge_ok` precedent) and the slot's tasks in attention (`attention`,
                    no39-02). #50 (lsbdef-0008, §7.3): plus the last call's diagnostics from
                    `services.*.last_meta` — `last_finish_reason`/`last_budget`/
                    `last_budget_source`/`last_had_thinking` (no metadata — None):
                    the operator reads the reason (trimmed by the budget / reasoning ate
                    the output) without the journal.
                    Models are NOT called: only in-memory state and SQL over `worker_jobs`
                    (assembling `/health` must not wake slots).

                    The `operation` mode (release 3.9.0, LSBN-S-7): the entries are
                    the OPERATIONS (the breakers are keyed by operation) — the block
                    must not fall over and must not lose an operation. The full
                    form of the operations block — the step S-15; the legacy slot
                    keys are NOT added here.
        """
        if self._mode() == SLOT_MODEL_OPERATION:
            return self._operations_health()
        attention = self._attention_by_slot()
        services = {
            SLOT_EMBEDDING: self._embedding,
            SLOT_SUMMARY: self._summarizer,
            SLOT_JUDGE: self._judge,
        }
        health: dict[str, dict] = {}
        for name in SLOT_NAMES:
            snapshot = dict(self._breakers[name].snapshot())
            service = services.get(name)
            snapshot["ok"] = getattr(service, "last_attempt_ok", None)
            snapshot["attention"] = attention.get(name, 0)
            meta = getattr(service, "last_meta", None)
            snapshot["last_finish_reason"] = meta.finish_reason if meta else None
            snapshot["last_budget"] = meta.budget if meta else None
            snapshot["last_budget_source"] = meta.budget_source if meta else None
            snapshot["last_had_thinking"] = meta.had_thinking if meta else None
            # LSBN-T-33: separate budgets of the last call — "eaten by
            # reasoning" is visible via the pair `last_outcome=truncated` +
            # `last_had_thinking=true` + the budget split.
            snapshot["last_answer_budget"] = meta.answer_budget if meta else None
            snapshot["last_reasoning_budget"] = meta.reasoning_budget if meta else None
            snapshot["last_sent_budget"] = meta.sent_budget if meta else None
            health[name] = snapshot
        health[SLOT_DECISION] = self._decision_slot_health(attention)
        return health

    def _operations_health(self) -> dict[str, dict]:
        """`/health.slots` in the `operation` mode: one entry per OPERATION.

                    The breakers are keyed by operation there, so the entries are
                    the operations: every breaker of the mode gets an entry (no
                    operation is lost) and the block never falls over (the
                    legacy slot keys are not looked up — LSBN-S-7). The full
                    form of the block (the deprecated slot keys as the
                    aggregates) — the step S-15.

        """
        attention = self._attention_by_operation()
        health: dict[str, dict] = {}
        for name, breaker in self._breakers.items():
            snapshot = dict(breaker.snapshot())
            service = self._operation_service(name)
            snapshot["ok"] = getattr(service, "last_attempt_ok", None)
            snapshot["attention"] = attention.get(name, 0)
            meta = getattr(service, "last_meta", None)
            snapshot["last_finish_reason"] = meta.finish_reason if meta else None
            snapshot["last_budget"] = meta.budget if meta else None
            snapshot["last_budget_source"] = meta.budget_source if meta else None
            snapshot["last_had_thinking"] = meta.had_thinking if meta else None
            snapshot["last_answer_budget"] = meta.answer_budget if meta else None
            snapshot["last_reasoning_budget"] = meta.reasoning_budget if meta else None
            snapshot["last_sent_budget"] = meta.sent_budget if meta else None
            health[name] = snapshot
        if SLOT_DECISION in health:
            # The decision operation's breaker lives in the SERVICE (§13.8) —
            # its own state is the truthful one.
            health[SLOT_DECISION] = {
                **health[SLOT_DECISION],
                **self._decision_slot_health(attention),
            }
        return health

    def _attention_by_slot(self) -> dict[str, int]:
        """The tasks in attention per SLOT group (the `legacy` form)."""
        attention: dict[str, int] = {}
        try:
            self._ensure_job_table()
            with session(self._settings) as conn:
                rows = conn.execute(
                    "SELECT slot, COUNT(*) AS c FROM worker_jobs "
                    "WHERE attention_at IS NOT NULL GROUP BY slot"
                ).fetchall()
            attention = {rows[i]["slot"]: int(rows[i]["c"]) for i in range(len(rows))}
        except Exception:  # noqa: BLE001 — observability does not crash /health
            attention = {}
        return attention

    def _attention_by_operation(self) -> dict[str, int]:
        """The tasks in attention per OPERATION (the `operation` mode).

                    The rows keep their legacy slot values until the migration
                    of `worker_jobs.slot` (LSBN-S-14) — the owner of a row is
                    read through `_task_operation`.

                    Only the KNOWN operations are counted (release 3.9.0,
                    LSBN-S-9): an unknown pair (an unknown slot or kind — the
                    rows keep their legacy values) has no operation, so it does
                    not land in a counter of `/health` under a non-operation key
                    (e.g. `nodes`). The task is not lost — it stays visible in
                    `/jobs/attention` (the listing shows the row's own slot).

        """
        counts: dict[str, int] = {}
        try:
            self._ensure_job_table()
            with session(self._settings) as conn:
                rows = conn.execute(
                    "SELECT slot, kind, COUNT(*) AS c FROM worker_jobs "
                    "WHERE attention_at IS NOT NULL GROUP BY slot, kind"
                ).fetchall()
        except Exception:  # noqa: BLE001 — observability does not crash /health
            return {}
        for row in rows:
            operation = self._task_operation(row["slot"], row["kind"])
            if operation not in OPERATION_REGISTRY:
                continue  # an unknown pair — not an operation's counter
            counts[operation] = counts.get(operation, 0) + int(row["c"])
        return counts

    def _decision_slot_health(self, attention: dict[str, int]) -> dict:
        """The `decision` slot for `/health.slots` (LSBN-F-12, §7.1): memory only.

                    The state comes from `services.decision.snapshot()` (the breaker lives
                    in the service, §13.8): `ok`, `last_outcome`, `paused_until`;
                    `attention` — over the task table (decision has no own queue — 0); the
                    four last-call diagnostic fields — `None` (decision has no
                    `last_meta`). The slot off/not built — the "disabled" shape (§7.1).
        """
        service = self._decision
        if service is None:
            return self._decision_disabled_stat()
        snapshot = service.snapshot()
        if not snapshot["enabled"]:
            return self._decision_disabled_stat()
        breaker = snapshot["breaker"]
        return {
            "ok": snapshot["ok"],
            "last_outcome": breaker["last_outcome"],
            "paused_until": breaker["paused_until"],
            "attention": attention.get(SLOT_DECISION, 0),
            "enabled": True,
            "last_finish_reason": None,
            "last_budget": None,
            "last_budget_source": None,
            "last_had_thinking": None,
        }

    @staticmethod
    def _decision_disabled_stat() -> dict:
        """The shape of a disabled decision slot in `/health.slots` (§7.1)."""
        return {
            "ok": None,
            "last_outcome": None,
            "paused_until": None,
            "attention": 0,
            "enabled": False,
        }

    def _probe_slot(self, slot: str) -> str:
        """One cheap probe call of a slot (no generation, #39 §6.2).

                    By breaker key: the SLOT name in the `legacy` mode, the
                    OPERATION name in the `operation` mode (release 3.9.0,
                    LSBN-S-7) — the service behind the key is resolved by
                    `_operation_service`. A missing service or a fake without
                    `probe` (a test mode) — the `CHECK_OK` stub (the
                    startup-check precedent).
        """
        service = self._operation_service(slot)
        probe = getattr(service, "probe", None)
        if probe is None:
            return CHECK_OK
        return probe()

    def _slot_gate(self, slot: str | None, job: str | None = None) -> bool:
        """The slot gate: may the batch be taken (#39, arch §6.2/§6.7).

                    `None`/unknown slot — True (no breaker). The breaker closed — True.
                    The breaker open and no knock yet — False (the batch is not taken, no
                    model calls burn; the run returns 0 → the loop's back-off). The knock
                    is due — one `probe()`: `CHECK_OK` lifts the breaker (the batch goes in
                    the same run), `auth_failed`/other — the pause is rescheduled, no
                    batch.
        """
        if slot is None:
            return True
        breaker = self._breakers.get(slot)
        if breaker is None:
            return True
        effective_job = job or slot
        now = time.time()
        if breaker.probe_due(now):
            result = self._probe_slot(slot)
            self._log_slot_event("slot_probe", slot, effective_job, result=result)
            if result == CHECK_OK:
                # A knock lifts the SLOT's breaker but does NOT reset the PAIR's count (no39-04,
                # the 2026-10-12 decision): "the slot revived" ≠ "the pair's reason is gone".
                # There is no `_reset_attention_for_slot(slot, "probe")` here — otherwise
                # a successful knock would reset the pair's `attempts` and `attention_at`,
                # the `SLOT_MAX_ATTEMPTS` threshold would become unreachable, and
                # "needs attention" would never trigger. Work is removed from
                # attention only by the pair-edit protocols, the operator's, and the
                # configuration-fingerprint change (arch §6.8).
                breaker.note_success()
                self._log_slot_event(
                    "slot_resumed",
                    slot,
                    effective_job,
                    after_sec=self._settings.slot_pause_sec,
                )
                return True
            outcome = (
                SlotOutcome.AUTH
                if result == CHECK_AUTH_FAILED
                else SlotOutcome.UNAVAILABLE
            )
            self._note_slot_outcome(effective_job, slot, outcome)
            return False
        if breaker.is_paused(now):
            return False
        return True

    def _note_slot_outcome(self, job: str, slot: str, outcome: SlotOutcome) -> bool:
        """Record the outcome at the slot breaker; True — the breaker opened now."""
        breaker = self._breakers.get(slot)
        if breaker is None:
            return False
        opened = breaker.note_failure(outcome)
        if opened:
            self._log_slot_event(
                "slot_paused",
                slot,
                job,
                level=logging.WARNING,
                outcome=outcome.value
                if isinstance(outcome, SlotOutcome)
                else str(outcome),
                paused_until=breaker.snapshot()["paused_until"],
            )
        return opened

    def _note_classify_failure(self, job: str, outcome: SlotOutcome) -> None:
        """A classifier failure of a job → the breaker of the `classify` operation.

                Release 3.9.0 (LSBN-S-9): in the `operation` mode the failure
                of the classifier of the `warnings` job is recorded at the
                breaker of the `classify` operation — the operation owns the
                failure, its own breaker (the design, B4). The `legacy` mode:
                nothing — the `warnings` job never touched the breaker (the
                failure stayed in the journal, the `warnings_classifier_failed`
                event).

                The job name stays the factual one (`warnings`); the operation
                is visible by the breaker key.
        """
        if self._mode() != SLOT_MODEL_OPERATION:
            return
        self._note_slot_outcome(job, self._operation_slot("classify"), outcome)

    def _note_split_failure(self, job: str, outcome: SlotOutcome) -> None:
        """A splitting failure of a job → the breaker of the `split` operation.

                Release 3.9.0 (LSBN-S-11): in the `operation` mode the failure
                of the splitting of the `migration` job is recorded at the
                breaker of the `split` operation — the operation owns the
                failure, its own breaker (the design, B4). The `legacy` mode:
                nothing — the job never touched the breaker (the failure stayed
                in the journal, the `migration_split_failed` event).

                The failure of the title generation INSIDE the split is a
                failure of the split too (the unit of work is the split): it
                comes here as well, and the breaker of the `title` operation is
                NOT touched (one failure - one count).

                The job name stays the factual one (`migration`); the operation
                is visible by the breaker key.
        """
        if self._mode() != SLOT_MODEL_OPERATION:
            return
        self._note_slot_outcome(job, self._operation_slot("split"), outcome)

    def _log_slot_failure(
        self,
        job: str,
        slot: str,
        outcome: SlotOutcome,
        *,
        event: str,
        message: str = "slot call failed",
        note_id: int | None = None,
        older_id: int | None = None,
        candidate_id: int | None = None,
        detail: str | None = None,
        meta: ChatMeta | None = None,
        **extra: object,
    ) -> None:
        """The single point of a slot failure (#39, arch §6.7): an event + the outcome into the journal.

                    Event fields: `event`, `job`, `slot`, `outcome` (the string value),
                    `detail` (a fragment ≤160 chars; secrets masked to 4 chars + `***`),
                    `note_id`/`older_id`/`candidate_id` as available. #49: when the model
                    call happened (a failure after the answer — the error carries `meta`),
                    the event is extended with the call fields (§7.2) — no second record.
                    The failure is also recorded at the slot breaker: if it opened —
                    `slot_paused` is written. The caller keeps the old behavior for
                    per-item outcomes (the work stays pending, `continue`); a wait-class
                    outcome (unavailable/timeout/auth, `_BATCH_ABORT_OUTCOMES` — LSBN-T-28,
                    the summary/title/judge batches) makes the caller release the batch
                    (`break`, the migration retry-later precedent): the work stays pending
                    — the back-off picks it up after the slot returns.
        """
        fields: dict[str, object] = {
            "event": event,
            "job": job,
            "slot": slot,
            "outcome": outcome.value
            if isinstance(outcome, SlotOutcome)
            else str(outcome),
        }
        for key, value in (
            ("note_id", note_id),
            ("older_id", older_id),
            ("candidate_id", candidate_id),
            ("detail", self._mask_detail(detail)),
        ):
            if value is not None:
                fields[key] = value
        if meta is not None:
            fields.update(_call_meta_fields(meta))
        fields.update(extra)
        logging.getLogger("app").warning(message, extra=fields)
        self._note_slot_outcome(job, slot, outcome)

    def _log_slot_event(
        self,
        event: str,
        slot: str,
        job: str,
        *,
        level: int = logging.INFO,
        **fields: object,
    ) -> None:
        """A breaker/probe event (#39 §7.4): `event`/`slot`/`job` + fields as available."""
        payload: dict[str, object] = {"event": event, "slot": slot, "job": job}
        for key, value in fields.items():
            if value is not None:
                payload[key] = value
        logging.getLogger("app").log(level, event, extra=payload)

    def _log_slot_call(
        self,
        job: str,
        slot: str,
        meta: ChatMeta | None,
        *,
        note_id: int | None = None,
        older_id: int | None = None,
        candidate_id: int | None = None,
    ) -> None:
        """A successful slot chat call — the `slot_call` event (#49, §7.2).

                    One event per call: the budget and its source, the counters, the fact
                    and volume of reasoning, the finish reason, the truncation, the
                    duration. Without metadata (a test double / a slot without
                    `chat_result`) the event is not written — a live service always
                    provides them (#49 FR-3.1).
        """
        if meta is None:
            return
        fields = _call_log_fields(meta)
        for key, value in (
            ("note_id", note_id),
            ("older_id", older_id),
            ("candidate_id", candidate_id),
        ):
            if value is not None:
                fields[key] = value
        self._log_slot_event("slot_call", slot, job, level=logging.INFO, **fields)

    def _mask_detail(self, detail: str | None) -> str | None:
        """A reason fragment ≤160 chars with secrets masked (4 chars + ***).

                    Secrets (slot API keys, the MCP token) can end up in a failure text
                    (URL/echo) — they never reach the journal in any form.
        """
        if detail is None:
            return None
        fragment = " ".join(str(detail).split())[:160]
        secrets = (
            self._settings.embedding_api_key,
            self._settings.summary_api_key,
            self._settings.judge_api_key,
            self._settings.mcp_auth_token,
        )
        for secret in secrets:
            if secret and secret.strip() and secret.strip() in fragment:
                fragment = fragment.replace(secret.strip(), secret.strip()[:4] + "***")
        return fragment

    # --- the pair attempt counter and "needs attention" (#39, arch §6.8) ------

    @staticmethod
    def _load_payload(raw: str | None) -> dict:
        """Parse a job's payload (tolerant to empty/broken JSON)."""
        try:
            loaded = json.loads(raw or "{}")
        except (TypeError, ValueError):
            loaded = {}
        return loaded if isinstance(loaded, dict) else {}

    def _set_job_payload(self, job_id: int, payload: dict) -> None:
        """Overwrite a job's payload (the pair identity and its stamp)."""
        self._ensure_job_table()
        with session(self._settings) as conn, transaction(conn):
            conn.execute(
                "UPDATE worker_jobs SET payload = ? WHERE id = ?",
                (json.dumps(payload, ensure_ascii=False), job_id),
            )

    def _job_for_slot(self, slot: str) -> str:
        """The owning job's name of a slot's queue (for events and /jobs/attention).

                The `operation` mode: the owner of a task row is an OPERATION,
                not a slot group — the job of the operation comes from
                `_OPERATION_JOB_NAMES` (the factual job: the reclass tasks
                belong to the `nodes` job, the merge tasks — to the `merge`
                job; the journal never names a job that does not exist).
                The `legacy` mode: the slot group → its job, as before.
        """
        if self._mode() == SLOT_MODEL_OPERATION:
            job = _OPERATION_JOB_NAMES.get(slot)
            if job is not None:
                return job
        return _SLOT_JOB_NAMES.get(slot, slot)

    def _log_attention_event(self, event: str, **fields: object) -> None:
        """A task attention event (arch §7.4): `event` + fields as available."""
        payload: dict[str, object] = {"event": event}
        for key, value in fields.items():
            if value is not None:
                payload[key] = value
        logging.getLogger("app").info(event, extra=payload)

    def _note_attempt(
        self,
        job_id: int,
        outcome: SlotOutcome,
        *,
        detail: str | None = None,
        pair_stamp: str | None = None,
    ) -> dict:
        """Record a pair attempt's outcome (#39, arch §6.8; no39-04).

                    The INVARIANT "the pause is per slot, the count per pair": the slot
                    breaker's state is not read here at all. The pause is the SLOT's
                    unavailability/access failure (back-off and the probe knock); the
                    `attempts` counter is the PAIR's own hopelessness (note/candidate).
                    Therefore non-transient failures count regardless of whether the slot
                    breaker is open now: the pair reaches `SLOT_MAX_ATTEMPTS` through
                    cycles of "pause → knock → failure again" instead of freezing at
                    `SLOT_PAUSE_AFTER`.

                    A transient outcome (`unavailable`/`timeout`/`auth`) updates only
                    `last_outcome`/`last_error` — the counter does not grow (waiting is a
                    normal state; back-off and the breaker handle it). A non-transient one
                    (`empty`/`truncated`/`off_contract`) increments `attempts`, and on
                    reaching `SLOT_MAX_ATTEMPTS` sets `attention_at` — the work freezes
                    while staying `pending`. All — in one transaction
                    (`UPDATE … WHERE id = ?`). `pair_stamp` — the pair's fingerprint
                    (note/candidate), updated in the payload.
        """
        self._ensure_job_table()
        masked = self._mask_detail(detail)
        value = outcome.value if isinstance(outcome, SlotOutcome) else str(outcome)
        nontransient = outcome in _NON_TRANSIENT_OUTCOMES
        with session(self._settings) as conn, transaction(conn):
            row = conn.execute(
                "SELECT id, slot, kind, note_id, payload, attempts, attention_at "
                "FROM worker_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return {"job_id": job_id, "attempts": 0, "attention": False}
            attempts = int(row["attempts"] or 0)
            attention = row["attention_at"] is not None
            pair = self._load_payload(row["payload"])
            if pair_stamp is not None:
                pair["updated_at"] = pair_stamp
                conn.execute(
                    "UPDATE worker_jobs SET payload = ? WHERE id = ?",
                    (json.dumps(pair, ensure_ascii=False), job_id),
                )
            if nontransient:
                attempts += 1
                if attempts >= self._settings.slot_max_attempts:
                    attention = True
                    conn.execute(
                        "UPDATE worker_jobs SET attempts = ?, last_outcome = ?, "
                        "last_error = ?, attention_at = COALESCE(attention_at, "
                        "strftime('%Y-%m-%dT%H:%M:%SZ','now')), updated_at = "
                        "strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ?",
                        (attempts, value, masked, job_id),
                    )
                else:
                    conn.execute(
                        "UPDATE worker_jobs SET attempts = ?, last_outcome = ?, "
                        "last_error = ?, updated_at = "
                        "strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ?",
                        (attempts, value, masked, job_id),
                    )
            else:
                conn.execute(
                    "UPDATE worker_jobs SET last_outcome = ?, last_error = ?, "
                    "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ?",
                    (value, masked, job_id),
                )
            slot = row["slot"]
            note_id = row["note_id"]
            older_id = pair.get("older_id")
        if nontransient and attention:
            # The owner of the pair: the operation in the `operation` mode, the
            # slot group in the `legacy` mode (LSBN-S-7).
            owner = self._task_operation(slot, row["kind"])
            self._log_attention_event(
                "job_attention",
                job=self._job_for_slot(owner),
                slot=owner,
                note_id=note_id,
                older_id=older_id,
                outcome=value,
                attempts=attempts,
            )
        return {"job_id": job_id, "attempts": attempts, "attention": attention}

    def _release_attention(self, job_id: int, reason: str) -> dict:
        """Reset the pair counter and clear attention (#39, arch §6.8/§13.3).

                    `attempts = 0, attention_at = NULL, last_error = NULL` — "the reason
                    is gone": the work goes as new. `released` — was the task in attention
                    (the operator handle returns 404 for a task outside attention);
                    `found` — whether the row exists at all. A `job_attention_released`
                    event with the reason (`edit`/`probe`/`operator`/`startup`) is written
                    for a found row.
        """
        self._ensure_job_table()
        with session(self._settings) as conn, transaction(conn):
            row = conn.execute(
                "SELECT id, slot, kind, note_id, payload, attention_at "
                "FROM worker_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return {"job_id": job_id, "released": False, "found": False}
            was_attention = row["attention_at"] is not None
            conn.execute(
                "UPDATE worker_jobs SET attempts = 0, attention_at = NULL, "
                "last_error = NULL, updated_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ?",
                (job_id,),
            )
            slot = row["slot"]
            note_id = row["note_id"]
            older_id = self._load_payload(row["payload"]).get("older_id")
        # The owner of the pair: the operation in the `operation` mode, the slot
        # group in the `legacy` mode (LSBN-S-7).
        owner = self._task_operation(slot, row["kind"])
        self._log_attention_event(
            "job_attention_released",
            job=self._job_for_slot(owner),
            slot=owner,
            note_id=note_id,
            older_id=older_id,
            reason=reason,
        )
        return {
            "job_id": job_id,
            "released": was_attention,
            "found": True,
            "attempts": 0,
        }

    def release_attention(self, job_id: int) -> dict:
        """The operator's reopening of a task from attention (arch §6.7/§7.4).

                    Only for a task in attention: outside attention (or without a row) —
                    `released: false` without edits (the REST handle returns 404).
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT attention_at FROM worker_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None or row["attention_at"] is None:
            return {"job_id": job_id, "released": False}
        self._release_attention(job_id, "operator")
        return {"job_id": job_id, "released": True, "attempts": 0}

    def _reset_attention_for_slot(self, slot: str, reason: str) -> int:
        """Clear attention of a slot's tasks (`*` — all); the number cleared.

                    The only "wholesale" "the reason is gone" reset — a start with a
                    changed configuration fingerprint (`reason="startup"`, arch
                    §6.8/§13.3). A successful slot knock (`reason="probe"`) leads here NO
                    MORE (no39-04): "the slot revived" is not "the pair's reason is
                    gone".

                    The key is the breaker key: the SLOT group in the `legacy`
                    mode, the OPERATION in the `operation` mode (release 3.9.0,
                    LSBN-S-7) — the neighbour operation keeps its attention.

        """
        self._ensure_job_table()
        # an observability hook (tests): release attempts per reason
        self._attention_resets[reason] = self._attention_resets.get(reason, 0) + 1
        with session(self._settings) as conn:
            if slot == "*":
                rows = conn.execute(
                    "SELECT id FROM worker_jobs WHERE attention_at IS NOT NULL"
                ).fetchall()
            elif self._mode() == SLOT_MODEL_OPERATION:
                # The `operation` mode: the reset is per OPERATION (the breaker
                # key) — the tasks of a neighbour operation keep their
                # attention (a title failure must not clear the attention of a
                # summarize pair). The rows keep their legacy slot values
                # until the migration (LSBN-S-14) — the owner of a row is read
                # through `_task_operation`.
                rows = conn.execute(
                    "SELECT id, slot, kind FROM worker_jobs "
                    "WHERE attention_at IS NOT NULL"
                ).fetchall()
                rows = [
                    row
                    for row in rows
                    if self._task_operation(row["slot"], row["kind"]) == slot
                ]
            else:
                rows = conn.execute(
                    "SELECT id FROM worker_jobs WHERE attention_at IS NOT NULL "
                    "AND slot = ?",
                    (slot,),
                ).fetchall()
        released = 0
        for row in rows:
            if self._release_attention(int(row["id"]), reason)["released"]:
                released += 1
        return released

    def attention_jobs(self) -> list[dict]:
        """Tasks waiting for a human (arch §6.7/§7.4): SQL only.

                    The record shape: `{job_id, job, slot, note_id, older_id, outcome,
                    attempts, since}` — `since` is the `attention_at` moment. Consumers —
                    `/health` and the operator handle `GET /jobs/attention` (no models
                    called).
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, slot, note_id, payload, attempts, last_outcome, "
                "attention_at FROM worker_jobs WHERE attention_at IS NOT NULL "
                "ORDER BY id"
            ).fetchall()
        jobs: list[dict] = []
        for row in rows:
            jobs.append(
                {
                    "job_id": int(row["id"]),
                    "job": self._job_for_slot(row["slot"]),
                    "slot": row["slot"],
                    "note_id": row["note_id"],
                    "older_id": self._load_payload(row["payload"]).get("older_id"),
                    "outcome": row["last_outcome"],
                    "attempts": int(row["attempts"] or 0),
                    "since": row["attention_at"],
                }
            )
        return jobs

    def attention_count(self) -> int:
        """How many tasks wait for a human (an aggregate over all slots, arch §6.7)."""
        self._ensure_job_table()
        with session(self._settings) as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM worker_jobs "
                    "WHERE attention_at IS NOT NULL"
                ).fetchone()[0]
            )

    def _review_attention_jobs(self, slot: str, kind: str) -> int:
        """Re-check a pair of frozen jobs without a model call (arch §6.8).

                    The "pair edit" protocol without manual edits: an edit of the
                    note/candidate or a new best candidate returns the job to the queue
                    (`attempts=0`); the verdict/merge decision is made by the next run.
                    LSBN-D-14 (3.6.2): a frozen merge pair of ONE source is CLOSED here
                    (`_mark_job_done` + `dedup_skipped`), not released - the reason is
                    gone for good, and a release would freeze the pair again.
                    SQL and vector reads only — no models called. Returns the unfrozen
                    count.
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, note_id, payload FROM worker_jobs "
                "WHERE slot = ? AND kind = ? AND status = 'pending' "
                "AND attention_at IS NOT NULL ORDER BY id",
                (slot, kind),
            ).fetchall()
        released = 0
        for row in rows:
            payload = self._load_payload(row["payload"])
            note_id = int(row["note_id"])
            with session(self._settings) as conn:
                note = conn.execute(
                    "SELECT id, namespace, vector_status, deleted_at, updated_at, origin_note_id "
                    "FROM notes WHERE id = ?",
                    (note_id,),
                ).fetchone()
                vector = (
                    vectors.get_vector(conn, note_id) if note is not None else None
                )
            if note is None or note["deleted_at"] is not None:
                continue
            # a pair side's edit (the note/the earlier one) — the reason is gone
            if (
                payload.get("note_updated_at") is not None
                and payload["note_updated_at"] != note["updated_at"]
            ) or (
                payload.get("newer_updated_at") is not None
                and payload["newer_updated_at"] != note["updated_at"]
            ):
                if self._release_attention(int(row["id"]), "edit")["released"]:
                    released += 1
                continue
            older_id = payload.get("older_id")
            if older_id is not None:
                with session(self._settings) as conn:
                    other = conn.execute(
                        "SELECT deleted_at, updated_at, origin_note_id FROM notes WHERE id = ?",
                        (int(older_id),),
                    ).fetchone()
                if other is None or other["deleted_at"] is not None:
                    continue
                if (
                    note["origin_note_id"] is not None
                    and note["origin_note_id"] == other["origin_note_id"]
                ):
                    # 3.6.2 (LSBN-D-14): the reason is gone for good - the pair is two
                    # parts of ONE source (the verdict and the merge rule read the same
                    # way). CLOSED, not released: a released pair returns to the queue,
                    # is skipped there and freezes again - the frozen counter of
                    # /health.queues would count it forever.
                    logging.getLogger("app").info(
                        "dedup: frozen pair closed — one source",
                        extra={
                            "event": "dedup_skipped",
                            "job": self._job_for_slot(slot),
                            "note_id": note_id,
                            "older_id": int(older_id),
                            "reason": "same origin",
                            "detail": "the reason is gone: the pair is one source",
                        },
                    )
                    self._mark_job_done(int(row["id"]))
                    continue
                if (
                    payload.get("older_updated_at") is not None
                    and payload["older_updated_at"] != other["updated_at"]
                ):
                    if self._release_attention(int(row["id"]), "edit")["released"]:
                        released += 1
                continue
            # dedup: the best candidate is recomputed over the vector; without a vector
            # the pair cannot be checked — we wait for re-vectorization
            if note["vector_status"] != "ok" or vector is None:
                continue
            candidates = self._find_dedup_candidates(
                note_id, vector, note["namespace"]
            )
            alive = [pair for pair in candidates if pair[0] < note_id]
            best_id = int(alive[0][0]) if alive else None
            stored_id = payload.get("candidate_id")
            if (
                stored_id is not None
                and best_id is not None
                and stored_id != best_id
            ):
                if self._release_attention(int(row["id"]), "edit")["released"]:
                    released += 1
                continue
            if best_id is not None:
                with session(self._settings) as conn:
                    candidate = conn.execute(
                        "SELECT updated_at FROM notes WHERE id = ?", (best_id,)
                    ).fetchone()
                if (
                    candidate is not None
                    and payload.get("candidate_updated_at") is not None
                    and payload["candidate_updated_at"] != candidate["updated_at"]
                ):
                    if self._release_attention(int(row["id"]), "edit")["released"]:
                        released += 1
        return released

    def _startup_attention_reset(self) -> int:
        """The startup check of the slot configuration fingerprint (arch §6.8/§13.3).

                    We compare the current fingerprint
                    (`slot_budget.slot_config_fingerprint`, lsbdef0008-02: model/kind/
                    budget/context/reserve/ceiling + decision parameters) with the one
                    recorded in `meta` (`SLOT_CONFIG_META_KEY`). No fingerprint or a
                    different one — "the reason is gone": attention is cleared for the
                    tasks of all slots (the block-2 `job_attention_released` event,
                    `reason="startup"`), the fingerprint is rewritten. The same
                    fingerprint — nothing is cleared.

                    `SLOT_ATTENTION_RESET_ON_START=false` disables the step entirely: the
                    operator forbade the reset; reading and rewriting the fingerprint is
                    unnecessary.
        """
        if not self._settings.slot_attention_reset_on_start:
            return 0
        fingerprint = slot_config_fingerprint(self._settings)
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = ?",
                (SLOT_CONFIG_META_KEY,),
            ).fetchone()
        stored = row["value"] if row is not None else None
        released = 0
        if stored != fingerprint:
            released = self._reset_attention_for_slot("*", "startup")
        with session(self._settings) as conn, transaction(conn):
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (SLOT_CONFIG_META_KEY, fingerprint),
            )
        return released

    async def _startup_attention_reset_async(self) -> None:
        """The startup attention check in a worker thread (does not block the loops).

                    A check failure does not crash the worker: attention clears itself by
                    a successful slot knock or a pair edit, and the check itself is the
                    "the reason is gone on a configuration change" optimization
                    (arch §6.8).
        """
        try:
            await asyncio.to_thread(self._startup_attention_reset)
        except Exception:  # noqa: BLE001 — a background optimization does not crash the start
            logging.getLogger("app").warning(
                "slot attention startup reset failed",
                extra={"event": "job_attention_reset_failed"},
            )

    # --- job work (the loop, back-off and supervisor — in the framework) ------

    async def _process_embedding(self) -> int:
        """The embedding job's run: the notes queue (in a thread) + the chunk queue.

                    Both queues — one job (decision #10): full note vectors
                    (`process_pending` — synchronous SQL/HTTP, goes into `to_thread`) and
                    the chunk queue (`process_pending_chunks` — its own async handling
                    with a Semaphore inside). Returns the number of actually processed
                    tasks: 0 sends the framework loop into a back-off wait (arch §3.2).
                    A slot breaker never reaches the run body: the gate is in the
                    framework (`JobSpec.gate`, 3.4.1, LSBN-T-29) — the manual `if` is
                    gone, one check for all jobs.
        """
        with history.bind_origin(f"job:{EMBEDDING_JOB}"):
            processed = await asyncio.to_thread(self.process_pending)
            processed += await self.process_pending_chunks()
        return processed

    async def _run_embedding_single(self) -> None:
        """Serve the embedding job with the framework loop (a single-job run).

                    See `_run_summary`: one job's isolation — for the pool-6 tests
                    (lost wakeup) over the vectorization queue (LSBN-T-30: the
                    `_embedding_queue_empty` recheck against an erased `clear()` signal).
        """
        spec = build_embedding_job(self, self._settings)
        await run_loop(spec, self._is_stopping, self._backoff[EMBEDDING_JOB])

    async def _process_summary(self) -> int:
        """The summary job's run: title-gen → summarize → merge (three steps).

                    Each step — synchronous SQL/LLM, goes into `to_thread`; the step order
                    is preserved (as in the original loop). Returns the number of
                    processed tasks: 0 sends the framework loop to waiting for the
                    `notify_summary_pending` signal (the "on demand" form, pool 6) or a
                    timeout. The summary slot breaker is checked by the framework gate
                    (`JobSpec.gate`, 3.4.1, LSBN-T-29) — the manual `if` is gone.

                    The `legacy` mode only (release 3.9.0, LSBN-S-6): in the
                    `operation` mode the cycle is cut into three jobs — the
                    title, the summarize and the merge loops (`_process_title`,
                    `_process_summarize`, `_process_merge`), each with its own
                    breaker.
        """
        with history.bind_origin(f"job:{SUMMARY_JOB}"):
            processed = await asyncio.to_thread(self.process_title_pending)
            processed += await asyncio.to_thread(self.process_summary_pending)
            processed += await asyncio.to_thread(self.process_merge_pending)
        return processed

    async def _process_title(self) -> int:
        """The `title` job's run: the title generation (the `title IS NULL` work).

                    One of the three jobs of the split summary cycle (release
                    3.9.0, LSBN-S-6): its own loop, its own signal and its own
                    breaker (the `title` operation). A title failure does not
                    stop the summarize loop — the failure belongs to the
                    operation that produced it.

        """
        with history.bind_origin(f"job:{TITLE_JOB}"):
            return await asyncio.to_thread(self.process_title_pending)

    async def _process_summarize(self) -> int:
        """The `summarize` job's run: the pending summarizations.

                    One of the three jobs of the split summary cycle (release
                    3.9.0, LSBN-S-6): its own loop, its own signal and its own
                    breaker (the `summarize` operation).

        """
        with history.bind_origin(f"job:{SUMMARIZE_JOB}"):
            return await asyncio.to_thread(self.process_summary_pending)

    async def _process_merge(self) -> int:
        """The `merge` job's run: the pending merge jobs.

                    One of the three jobs of the split summary cycle (release
                    3.9.0, LSBN-S-6): its own loop, its own signal (set at the
                    merge job's creation) and its own breaker (the `merge`
                    operation).

        """
        with history.bind_origin(f"job:{MERGE_JOB}"):
            return await asyncio.to_thread(self.process_merge_pending)

    async def _run_summary(self) -> None:
        """Serve the summary job with the framework loop (a single-job run).

                    The loop body lives in the framework; the method is kept for an
                    isolated run of one job — this is how the pool-6 tests (lost wakeup)
                    check the queue select and the signal's `clear()` without the other
                    loops.
        """
        spec = build_summary_job(self, self._settings)
        if spec is None:
            return  # without a summarizer the job is not applicable (a test mode)
        await run_loop(spec, self._is_stopping, self._backoff[SUMMARY_JOB])

    async def _run_title(self) -> None:
        """Serve the `title` job with the framework loop (a single-job run).

                    See `_run_summary`: an isolated run of one job of the split
                    summary cycle (the `operation` mode).
        """
        spec = build_title_job(self, self._settings)
        if spec is None:
            return  # the `legacy` mode (or a test mode without a summarizer)
        await run_loop(spec, self._is_stopping, self._job_backoff(spec))

    async def _run_summarize(self) -> None:
        """Serve the `summarize` job with the framework loop (a single-job run).

                    See `_run_title`.
        """
        spec = build_summarize_job(self, self._settings)
        if spec is None:
            return  # the `legacy` mode (or a test mode without a summarizer)
        await run_loop(spec, self._is_stopping, self._job_backoff(spec))

    async def _run_merge(self) -> None:
        """Serve the `merge` job with the framework loop (a single-job run).

                    See `_run_title`.
        """
        spec = build_merge_job(self, self._settings)
        if spec is None:
            return  # the `legacy` mode (or a test mode without a summarizer)
        await run_loop(spec, self._is_stopping, self._job_backoff(spec))

    async def _process_judge(self) -> int:
        """The judge job's run: a batch of judge jobs (dedup) in `to_thread`.

                    0 sends the framework loop to waiting for the `notify_judge_pending`
                    signal (the "on demand" form, pool 6) or an interval timeout. The
                    judge slot breaker is checked by the framework gate (`JobSpec.gate`,
                    3.4.1, LSBN-T-29) — the manual `if` is gone.
        """
        with history.bind_origin(f"job:{JUDGE_JOB}"):
            return await asyncio.to_thread(self.process_judge_pending)

    async def _run_judge(self) -> None:
        """Serve the judge job with the framework loop (a single-job run).

                    See `_run_summary`: one job's isolation — for the pool-6 tests
                    (lost wakeup) over the judge queue.
        """
        spec = build_judge_job(self, self._settings)
        await run_loop(spec, self._is_stopping, self._backoff[JUDGE_JOB])

    # --- the areas job: area records' vectors (substrate 3.0.0) ---------------

    async def _process_areas(self) -> int:
        """The areas job's run: a batch of pending area records in `to_thread`.

                    Its own queue, its own back-off and its own signal: an area
                    vectorization failure does not hinder notes and vice versa
                    (ARCH §3.4). 0 sends the framework loop to waiting for
                    `notify_areas_pending` (pool 6) or an interval timeout.
        """
        with history.bind_origin(f"job:{AREAS_JOB}"):
            return await asyncio.to_thread(self.process_pending_areas)

    # --- the expired-notes cleanup job (lsb-0004-02, stage 4) -----------------

    async def _process_expiration(self) -> int:
        """The expiration job's run: expired-notes cleanup.

                    It drains note_expirations: notes with `expires_at <= now()` are
                    deleted FULLY (notes+chunks+vectors+fts) together with their
                    note_expirations row (`process_expired_notes` in `to_thread`);
                    deleted > 0 — an `expiration_cleanup` event with the mandatory `job`
                    field. The job runs on a FIXED schedule (its back-off state is
                    `fixed=True`): the pause is always EXPIRATION_CLEANUP_INTERVAL_SEC
                    (a fixed 5 minutes), the back-off does not grow — cleanup does not
                    depend on external services and the schedule neither speeds up nor
                    slows down (FR-1.5). Returns the deleted note count (the schedule is
                    led by the interval, not progress — see `run_loop`).
        """
        with history.bind_origin(f"job:{EXPIRATION_JOB}"):
            deleted = await asyncio.to_thread(self.process_expired_notes)
            if deleted:
                logging.getLogger("app").info(
                    "expiration cleanup: purged expired notes",
                    extra={
                        "event": "expiration_cleanup",
                        "job": EXPIRATION_JOB,
                        "count": deleted,
                    },
                )
        return deleted

    def process_expired_notes(self) -> int:
        """Delete expired notes; returns the deleted count.

                    SELECT note_id FROM note_expirations WHERE expires_at <= now() (now —
                    in the same ISO-8601 UTC format as expires_at: strftime
                    '%Y-%m-%dT%H:%M:%SZ','now'); per id — full physical deletion
                    (notes+chunks+vectors+fts) + removing the row from note_expirations
                    (delete_note_physical).

                    Idempotent and safe: if the note is already deleted (e.g. by the
                    operator) — delete_note_physical simply matches nothing, no crash;
                    the note_expirations row is removed anyway. Each note — a short
                    transaction: one failure does not roll back the others.
        """
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT note_id FROM note_expirations "
                "WHERE expires_at <= strftime('%Y-%m-%dT%H:%M:%SZ','now')"
            ).fetchall()
        if not rows:
            return 0
        deleted = 0
        for row in rows:
            note_id = int(row["note_id"])
            with session(self._settings) as conn, transaction(conn):
                delete_note_physical(conn, note_id)
                deleted += 1
        return deleted

    # --- job queues per slots (Phase 11, decision #10) ------------------------

    def _ensure_job_table(self) -> None:
        """Create the job-queues table + index (idempotently, once).

                    The schema is the worker's zone (not db.py): per-slot job queues are
                    the background pipeline's internal mechanics, the loops' dependency
                    dispatcher.

                    Pool 5, one-time creation: the full DDL runs only at the first access
                    per worker instance (memoized on the `_jobs_table_ready` flag);
                    repeated calls (3 per job: `_create_job`, `_pending_jobs`,
                    `_mark_job_done`) return immediately. The `idx_worker_jobs_queue`
                    index on (slot, kind, status, id): a slot's pending queue is selected
                    by the index, not a full table scan. The attention columns (#39, arch
                    §5.2/§6.8) — `attempts`/`last_outcome`/`last_error`/`attention_at` —
                    are in the fresh DDL and added by an idempotent ALTER over
                    `PRAGMA table_info(worker_jobs)` for a live database of the previous
                    version; the `idx_worker_jobs_attention` index serves the "needs
                    attention" selection.

                    The DDL — under a lock (lsbdef-0013): two threads (a loop via
                    `asyncio.to_thread` + the startup attention release in `run`)
                    simultaneously saw the flag `False` and the column missing — the
                    second ALTER failed on `duplicate column name`. A double flag check
                    inside the lock; the ALTER also tolerates `duplicate column name`:
                    the lock only covers threads of one process — a competing worker
                    process is not one of them.
        """
        if self._jobs_table_ready:
            return
        with self._jobs_table_lock:
            if self._jobs_table_ready:  # another thread already migrated it
                return
            with session(self._settings) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS worker_jobs ("
                    "  id           INTEGER PRIMARY KEY,"
                    "  slot         TEXT NOT NULL,"
                    "  kind         TEXT NOT NULL,"
                    "  note_id      INTEGER NOT NULL,"
                    "  payload      TEXT,"
                    "  status       TEXT NOT NULL DEFAULT 'pending',"
                    "  created_at   TEXT NOT NULL DEFAULT "
                    "    (strftime('%Y-%m-%dT%H:%M:%SZ','now')),"
                    "  updated_at   TEXT NOT NULL DEFAULT "
                    "    (strftime('%Y-%m-%dT%H:%M:%SZ','now')),"
                    "  attempts     INTEGER NOT NULL DEFAULT 0,"
                    "  last_outcome TEXT,"
                    "  last_error   TEXT,"
                    "  attention_at TEXT"
                    ")"
                )
                existing = {
                    row["name"]
                    for row in conn.execute(
                        "PRAGMA table_info(worker_jobs)"
                    ).fetchall()
                }
                for name, ddl in (
                    ("attempts", "INTEGER NOT NULL DEFAULT 0"),
                    ("last_outcome", "TEXT"),
                    ("last_error", "TEXT"),
                    ("attention_at", "TEXT"),
                ):
                    if name not in existing:
                        try:
                            conn.execute(
                                f"ALTER TABLE worker_jobs ADD COLUMN {name} {ddl}"
                            )
                        except sqlite3.OperationalError as exc:
                            # A competing process managed to add the column
                            # (lsbdef-0013): we do not crash; the migration continues.
                            if "duplicate column name" not in str(exc):
                                raise
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_worker_jobs_queue "
                    "ON worker_jobs(slot, kind, status, id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_worker_jobs_attention "
                    "ON worker_jobs(status, attention_at)"
                )
            self._jobs_table_ready = True

    def _purge_done_jobs(self) -> None:
        """Purge done jobs older than the retention window (pool 5, hygiene).

                    worker_jobs accumulates ≥1 row per note (more on re-vectorizations
                    after update); without cleanup the table and the queue's full scans
                    would grow without bound. The DELETE is constrained by
                    `status = 'done'` and `updated_at` older than
                    WORKER_JOBS_RETENTION_DAYS — pending jobs are untouched. Called in the
                    embedding loop's idle branch (no jobs, before the sleep). Deleted > 0
                    → an info `jobs_purged` event.
        """
        self._ensure_job_table()
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "DELETE FROM worker_jobs WHERE status = 'done' AND updated_at < "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now',"
                f"'-{WORKER_JOBS_RETENTION_DAYS} days')"
            )
        if cursor.rowcount > 0:
            logging.getLogger("app").info(
                "worker_jobs: purged done jobs older than retention window",
                extra={
                    "event": "jobs_purged",
                    "job": EMBEDDING_JOB,
                    "count": cursor.rowcount,
                    "retention_days": WORKER_JOBS_RETENTION_DAYS,
                },
            )

    def _create_job(
        self, slot: str, kind: str, note_id: int, payload: str | None = None
    ) -> None:
        """Place a job into a slot's queue (the dependency dispatcher)."""
        self._ensure_job_table()
        with session(self._settings) as conn, transaction(conn):
            conn.execute(
                "INSERT INTO worker_jobs (slot, kind, note_id, payload) "
                "VALUES (?, ?, ?, ?)",
                (slot, kind, note_id, payload),
            )
        if kind == "merge" and slot in (SLOT_SUMMARY, self._operation_slot("merge")):
            # A merge job appeared (release 3.9.0, LSBN-S-6): the merge loop of
            # the `operation` mode has its own signal — the job is taken at
            # once, not after the grown back-off. The signal is set HERE — at
            # the creation of the merge task — so every path that creates one
            # wakes the loop. Release 3.9.0 (LSBN-S-7): the slot value of the
            # task is still the legacy one (the migration of `worker_jobs.slot`
            # is LSBN-S-14) — the check also answers the operation key.
            self.notify_merge_pending()

    def _ensure_job(
        self, slot: str, kind: str, note_id: int, payload: str | None = None
    ) -> None:
        """Idempotently place a job into a slot's queue (arch §3.5, FR-2.5).

                    `INSERT ... WHERE NOT EXISTS (a pending with the same
                    slot/kind/note_id)`: frequent edits of one note do not multiply task
                    duplicates (needed by the jobs that place work on an event). The
                    existing `_create_job` (judge/merge) is untouched: no dedup is needed
                    there — the "later ↔ earlier" pair and the judge's verdict give their
                    own invariants.
        """
        self._ensure_job_table()
        with session(self._settings) as conn, transaction(conn):
            conn.execute(
                "INSERT INTO worker_jobs (slot, kind, note_id, payload) "
                "SELECT ?, ?, ?, ? WHERE NOT EXISTS ("
                "SELECT 1 FROM worker_jobs WHERE slot = ? AND kind = ? "
                "AND note_id = ? AND status = 'pending')",
                (slot, kind, note_id, payload, slot, kind, note_id),
            )

    def _pending_jobs(
        self, slot: str, kind: str, limit: int
    ) -> list:
        """Read a slot's pending jobs (the order by id).

                    Frozen jobs (`attention_at IS NOT NULL`, #39 §6.8) are not taken: a
                    repeating non-transient failure stops burning model calls until the
                    reason is gone (the job stays `pending`).
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            return conn.execute(
                "SELECT id, note_id, payload FROM worker_jobs "
                "WHERE slot = ? AND kind = ? AND status = 'pending' "
                "AND attention_at IS NULL "
                "ORDER BY id LIMIT ?",
                (slot, kind, limit),
            ).fetchall()

    def _mark_job_done(self, job_id: int) -> None:
        """Mark a job done (removed from the queue).

                    A closed task is not waiting for a human (3.6.2, LSBN-D-14): the
                    attention state is cleared together with the status - a frozen pair
                    closed for good (`_review_attention_jobs`) must leave both
                    `/health.queues.*.attention` and the operator handle
                    `GET /jobs/attention` (both read `attention_at`).
        """
        self._ensure_job_table()
        with session(self._settings) as conn, transaction(conn):
            conn.execute(
                "UPDATE worker_jobs SET status = 'done', attention_at = NULL, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ?",
                (job_id,),
            )

    # --- lost wakeup: a cheap queue recheck before the sleep (pool 6)

    def _decision_blocks_work(self) -> bool:
        """Can the current work be taken: is decision available (#39, §6.9).

                    The dedup judge and the `nodes` classifier walk into the decision
                    slot: if the slot is unavailable (`can_decide()` false — suppressed by
                    the startup check or `disable()`) and the fallback is off
                    (`DECISION_FALLBACK=false`), no pair/candidate will get a decision —
                    such work is not taken by the loop (it is visible in
                    `/health.queues[].waiting`). The check lives in process memory: no SQL
                    beyond the existing.
        """
        service = self._decision
        if service is None or self._settings.decision_fallback:
            return False
        return not service.can_decide()

    def _summary_queue_empty(self) -> bool:
        """Is the summary queue empty (a cheap SELECT, pool 6, lost wakeup).

                    Before `clear()+wait` the loop rechecks the queue itself, not only the
                    event: notify could arrive while the batch was being drained, and the
                    `clear()` after it would erase the signal (a pause up to the interval
                    on a non-empty queue). The queue is empty only when there is no
                    title-gen (title IS NULL), no pending summarization and no pending
                    merge jobs (all in the summary slot).

                    #39 (arch §6.9/§13.9): we count only what is TAKEABLE now — merge jobs
                    without attention (`attention_at IS NULL`). A frozen task
                    (`attention_at`, no39-02) cannot be taken by the loop, and if the
                    queue counted it non-empty the loop would spin idly (busy-loop).
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT (SELECT COUNT(*) FROM notes WHERE deleted_at IS NULL "
                "  AND (title IS NULL OR summary_status = 'pending')) "
                "+ (SELECT COUNT(*) FROM worker_jobs "
                "  WHERE slot = 'summary' AND status = 'pending' "
                "  AND attention_at IS NULL) AS c"
            ).fetchone()
        # #39 (arch §6.9): with the slot breaker open the loop takes no batch —
        # the queue is empty for it (otherwise a deferred task drives a busy-loop).
        # The same — with decision unavailable and fallback off: there is
        # nowhere to get a decision; the work waits (no39-05, `_decision_blocks_work`).
        # Release 3.9.0 (LSBN-S-6): the breaker is read through the operation's
        # key — the operation itself in the `operation` mode, the slot group in
        # the `legacy` mode (the slot name `summary` has no breaker there —
        # the S-5 finding M1: a direct lookup raised `KeyError`).
        # Release 3.9.0 (LSBN-S-7): the decision rule is per operation — in the
        # `operation` mode the summary family goes (only `judge`/`classify` wait).
        return (
            int(row["c"]) == 0
            or self._breaker_paused(self._operation_slot("summarize"))
            or self._decision_blocks_key(self._operation_slot("summarize"))
        )

    def _title_queue_empty(self) -> bool:
        """Is the title queue empty (the title work: `title IS NULL`)?

                    The title queue of the split summary cycle (release 3.9.0,
                    LSBN-S-6): the title generation is a job of its own. The
                    predicate is the title work ONLY — the summarization and
                    the merge have their own predicates (each loop takes its
                    own work; a failure of one operation does not stop the
                    others).

        """
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM notes WHERE title IS NULL "
                "AND deleted_at IS NULL"
            ).fetchone()
        return int(row[0]) == 0 or self._operation_blocked("title")

    def _summarize_queue_empty(self) -> bool:
        """Is the summarize queue empty (the summarization work:
        `summary_status='pending'`)?

                    The summarize queue of the split summary cycle (release
                    3.9.0, LSBN-S-6): its own predicate — the pending
                    summarizations only.

        """
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM notes WHERE summary_status = 'pending' "
                "AND deleted_at IS NULL"
            ).fetchone()
        return int(row[0]) == 0 or self._operation_blocked("summarize")

    def _merge_queue_empty(self) -> bool:
        """Is the merge queue empty (the pending merge jobs)?

                    The merge queue of the split summary cycle (release 3.9.0,
                    LSBN-S-6): the merge jobs live in `worker_jobs` under the
                    slot value `summary` (the legacy slot group) — the
                    migration of the slot values is a later step of the
                    release (LSBN-S-14), so the jobs are read under their
                    current value: no work is orphaned by the split.

                    #39 (arch §6.9/§13.9): only TAKEABLE work is counted — a
                    frozen job (`attention_at`) cannot be taken by the loop.

        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM worker_jobs "
                "WHERE slot = 'summary' AND kind = 'merge' "
                "AND status = 'pending' AND attention_at IS NULL"
            ).fetchone()
        return int(row[0]) == 0 or self._operation_blocked("merge")

    def _judge_queue_empty(self) -> bool:
        """Is the judge queue empty (a cheap SELECT, pool 6, lost wakeup).

                    It counts pending dedup jobs of the judge slot — the same queue
                    process_judge_pending drains in this loop. #39 (arch §6.9/§13.9): a
                    frozen task (`attention_at`, no39-02) is not counted — the loop does
                    not take it, and the queue is empty for it (no busy-loop).
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM worker_jobs "
                "WHERE slot = 'judge' AND kind = 'dedup' AND status = 'pending' "
                "AND attention_at IS NULL"
            ).fetchone()
        # #39 (arch §6.9): with the judge breaker open no batch is taken —
        # the queue is empty for the loop (no busy-loop); the same with decision
        # unavailable and fallback off (no39-05). Release 3.9.0 (LSBN-S-7): the
        # breaker is read through the operation's key (the operation itself in
        # the `operation` mode, the slot group in the `legacy` mode) and the
        # decision rule is per operation there.
        return (
            int(row[0]) == 0
            or self._breaker_paused(self._operation_slot("judge"))
            or self._decision_blocks_key(self._operation_slot("judge"))
        )

    def _embedding_queue_empty(self) -> bool:
        """Is the embedding queue empty (a cheap recheck, pool 6, LSBN-T-30).

                    Before `clear()+wait` the loop rechecks the queue itself, not only the
                    event: a `notify_embedding_pending` signal fired BETWEEN an empty
                    `process()` and the `clear()` (the note committed, the notifier
                    struck, `clear()` erased the signal) would otherwise erase the alarm —
                    the loop would sleep the current interval, and on an idle system the
                    back-off grows to 15 min, so a fresh note would wait for its vector
                    just as long with live models (the direct priority "a fresh note is
                    encoded immediately"; the gate 2026-09-19 precedent). The recheck
                    closes that window: takeable work is visible to the select and gets
                    ONE immediate run.

                    The embedding job's takeable work (one job — two queues, decision
                    #10): pending vectorization notes (`vector_status='pending'`; the
                    source — the same `NoteService.vector_queue_stat()` as the
                    `/health.queues` snapshot's — reuse, not a SQL copy) plus pending
                    chunks (the same predicate as the `_read_chunk_batch` read: a chunk
                    without a row in `notes_chunks_vec`, `chunks.count_pending`). SQL/
                    memory only, no models (FR-2.2: observability and schedules burn no
                    calls).

                    A busy-loop with unavailable models is impossible: in the framework,
                    work the loop cannot take is not progress (arch §3.2, no39-05) — after
                    the recheck one run on a failure returns 0, and the loop goes into
                    `queue_waiting` + a back-off sleep. An embedding-slot breaker
                    (`_slot_blocked`, the single point of the gate condition, 3.4.1,
                    LSBN-T-29; the `_areas_queue_empty`/`_nodes_queue_empty` precedent)
                    makes the queue empty for the loop: the gate takes no batch — nothing
                    to spin (no39-05).
        """
        # #39 (arch §6.9): with the embedding slot breaker open the loop takes no
        # batch — the queue is empty for it (otherwise a deferred task drives a
        # busy-loop). The memory check — before SQL: the breaker closes both of the
        # job's queues at once (notes and chunks).
        if self._slot_blocked(self._operation_slot("embed")):
            return True
        pending = int(self._notes.vector_queue_stat()["pending"] or 0)
        with session(self._settings) as conn:
            pending += chunks.count_pending(conn)
        return pending == 0

    # --- queue observability (lsb-0014-03, FR-2.2) ----------------------------

    def _slot_blocked(self, slot: str) -> bool:
        """Is the slot blocked right now? (#39, arch §6.9; LSBN-T-29).

                    The single point of the gate condition for queue SNAPSHOTS (it used to
                    live in `_gated_queue_stat`): the slot breaker is open
                    (`SlotBreaker.is_paused`) OR the slot's work cannot be taken (decision
                    unavailable, `DECISION_FALLBACK=false` — the `summary`/`judge` slots,
                    no39-05). WITHOUT side effects — memory only, never calls models and
                    never moves a pause: the jobs' `is_gated` read it from `/health.queues`
                    and `queue_waiting` events (observability burns no calls, the
                    `/health.queues` contract; the probe knock after the pause expires is
                    `gate`'s concern, i.e. `worker._slot_gate`, in the framework
                    `run_loop`).

                    The key is the breaker key: the SLOT group in the `legacy`
                    mode, the OPERATION in the `operation` mode (release 3.9.0,
                    LSBN-S-7) — the decision rule is per operation there (the
                    design, B4): only `judge` and `classify` wait for the
                    decision slot.

        """
        breaker = self._breakers.get(slot)
        return (
            breaker is not None and breaker.is_paused()
        ) or self._decision_blocks_key(slot)

    def _decision_blocks_key(self, key: str) -> bool:
        """Does the unavailable decision slot block this breaker key?

                    The decision rule (no39-05) is per operation in the
                    `operation` mode (release 3.9.0, LSBN-S-7; the design, B4):
                    only the operations whose path goes through the decision
                    slot wait — `judge` and `classify` (the classifier walks
                    into the decision slot). The `summarize`/`title`/`merge`/
                    `describe`/`embed` operations go. The `legacy` mode keeps
                    the historical rule: the whole `summary` slot family and
                    the `judge` slot wait.

        """
        if not self._decision_blocks_work():
            return False
        if self._mode() == SLOT_MODEL_OPERATION:
            return key in _DECISION_BLOCKED_OPERATIONS
        return key in (SLOT_SUMMARY, SLOT_JUDGE)

    def queues_health(self) -> dict[str, dict]:
        """Assemble `/health.queues` over the job registry (FR-2.2): SQL only.

                    For every registered job with a queue its own snapshot is taken
                    (`JobSpec.queue_stat`) — `/health` is not edited when a job is added.
                    A job without a queue (`expiration`) does not enter the object.
                    Slots uniformly (3.4.1, LSBN-T-29): with a blocked slot
                    (`spec.is_gated` → `_slot_blocked`) the raw snapshot goes through the
                    framework `gated_queue_stat` — work moves from `pending` to `waiting`
                    for ALL jobs (before, the external migration/warnings and areas did
                    not obey this). No model calls (only reading the pending state;
                    `is_gated` — memory/SQL, see `_slot_blocked`).
        """
        queues: dict[str, dict] = {}
        for spec in build_job_specs(self, self._settings):
            if spec.queue is None or spec.queue_stat is None:
                continue
            stat = spec.queue_stat()
            if spec.is_gated is not None and spec.is_gated():
                stat = gated_queue_stat(stat)
            queues[spec.queue] = stat
        return queues

    def _vector_queue_stat(self) -> dict:
        """The raw snapshot of the note-vectorization queue: pending and age.

                    The source — `notes.vector_status='pending'` (the same predicate as
                    the legacy `pending_vector` counter); the SQL lives next to
                    `health_counts` in NoteService. The pending → waiting move on an
                    embedding breaker is done by the framework (`queues_health`/
                    `queue_waiting` via `is_gated`, 3.4.1, LSBN-T-29) — the method returns
                    the snapshot as is.
        """
        return self._notes.vector_queue_stat()

    def _summary_queue_stat(self) -> dict:
        """The raw snapshot of the summarization queue (#39, arch §6.9).

                    `pending` — work the loop can take NOW: notes (title-gen + pending
                    summarization) plus the summary slot's pending merge jobs WITHOUT
                    attention. `attention` — pending merge jobs with attention (no39-02,
                    waiting for a human); `waiting = attention`: what the summary slot has
                    deferred is exactly the frozen ones. `oldest_pending_sec` — the maximum
                    age over both sources. The pending → waiting move on a slot breaker is
                    done by the framework (`is_gated` → `gated_queue_stat`, 3.4.1,
                    LSBN-T-29) — the method's own waiting/attention computation is
                    untouched.
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT "
                "(SELECT COUNT(*) FROM notes WHERE deleted_at IS NULL "
                "  AND (title IS NULL OR summary_status = 'pending')) "
                "+ (SELECT COUNT(*) FROM worker_jobs WHERE slot = 'summary' "
                "  AND status = 'pending' AND attention_at IS NULL) AS pending, "
                "(SELECT COUNT(*) FROM worker_jobs WHERE slot = 'summary' "
                "  AND status = 'pending' AND attention_at IS NOT NULL) "
                "AS attention, "
                "MAX("
                "COALESCE((SELECT MAX(CAST(strftime('%s','now') AS INTEGER) - "
                "  CAST(strftime('%s', updated_at) AS INTEGER)) FROM notes "
                "  WHERE deleted_at IS NULL AND (title IS NULL OR "
                "  summary_status = 'pending')), 0), "
                "COALESCE((SELECT MAX(CAST(strftime('%s','now') AS INTEGER) - "
                "  CAST(strftime('%s', created_at) AS INTEGER)) FROM worker_jobs "
                "  WHERE slot = 'summary' AND status = 'pending'), 0)"
                ") AS oldest_pending_sec"
            ).fetchone()
        attention = int(row["attention"])
        return queue_snapshot(
            row["pending"],
            row["oldest_pending_sec"],
            waiting=attention,
            attention=attention,
        )

    def _title_queue_stat(self) -> dict:
        """The raw snapshot of the title queue: the notes without a title.

                    The title queue of the split summary cycle (release 3.9.0,
                    LSBN-S-6): `pending` — the notes with `title IS NULL` (the
                    title work), `oldest_pending_sec` — their age. The pending
                    → waiting move on a breaker is done by the framework
                    (`is_gated` → `gated_queue_stat`, 3.4.1, LSBN-T-29).

        """
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS pending, "
                "MAX(CAST(strftime('%s','now') AS INTEGER) - "
                "CAST(strftime('%s', updated_at) AS INTEGER)) "
                "AS oldest_pending_sec FROM notes "
                "WHERE title IS NULL AND deleted_at IS NULL"
            ).fetchone()
        return queue_snapshot(row["pending"], row["oldest_pending_sec"])

    def _summarize_queue_stat(self) -> dict:
        """The raw snapshot of the summarize queue: the pending summarizations.

                    The summarize queue of the split summary cycle (release
                    3.9.0, LSBN-S-6): `pending` — the notes with
                    `summary_status='pending'`, `oldest_pending_sec` — their
                    age.

        """
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS pending, "
                "MAX(CAST(strftime('%s','now') AS INTEGER) - "
                "CAST(strftime('%s', updated_at) AS INTEGER)) "
                "AS oldest_pending_sec FROM notes "
                "WHERE summary_status = 'pending' AND deleted_at IS NULL"
            ).fetchone()
        return queue_snapshot(row["pending"], row["oldest_pending_sec"])

    def _merge_queue_stat(self) -> dict:
        """The raw snapshot of the merge queue: the pending merge jobs.

                    The merge queue of the split summary cycle (release 3.9.0,
                    LSBN-S-6): `pending` — the merge jobs WITHOUT attention
                    (takeable now); `attention` — with attention (no39-02,
                    waiting for a human); `waiting = attention`. The age
                    (3.3.1, LSBN-S-3) — over ALL of the pending jobs (an
                    attention-only state must not give `null`). The jobs are
                    read under their current slot value (`summary`) — the
                    migration of the values is a later step (LSBN-S-14).

        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(attention_at IS NULL), 0) AS pending, "
                "COALESCE(SUM(attention_at IS NOT NULL), 0) AS attention, "
                "MAX(CAST(strftime('%s','now') AS INTEGER) - "
                "CAST(strftime('%s', created_at) AS INTEGER)) "
                "AS oldest_pending_sec FROM worker_jobs "
                "WHERE slot = 'summary' AND kind = 'merge' "
                "AND status = 'pending'"
            ).fetchone()
        attention = int(row["attention"])
        return queue_snapshot(
            row["pending"],
            row["oldest_pending_sec"],
            waiting=attention,
            attention=attention,
        )

    def _judge_queue_stat(self) -> dict:
        """The raw snapshot of the judge queue (#39, arch §6.9).

                    `pending` — the judge slot's dedup jobs WITHOUT attention (takeable
                    now; the `_judge_queue_empty` driver); `attention` — with attention
                    (no39-02); `waiting = attention`: what the judge has deferred is
                    exactly the frozen ones. The age (3.3.1, LSBN-S-3) —
                    `now - MIN(created_at)` over ALL of the job's pending (attention work
                    also waits to be done; otherwise an attention-only state
                    `pending=0, waiting>0` would give `null` contrary to the definition).
                    The pending → waiting move on a judge breaker is done by the framework
                    (`is_gated` → `gated_queue_stat`, 3.4.1, LSBN-T-29). The table is
                    created lazily (`_ensure_job_table`); no models called.
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT "
                "COALESCE(SUM(status = 'pending' AND attention_at IS NULL), 0) "
                "AS pending, "
                "COALESCE(SUM(status = 'pending' AND attention_at IS NOT NULL), 0) "
                "AS attention, "
                "MAX(CASE WHEN status = 'pending' "
                "  THEN CAST(strftime('%s','now') AS INTEGER) - "
                "  CAST(strftime('%s', created_at) AS INTEGER) END) "
                "AS oldest_pending_sec "
                "FROM worker_jobs WHERE slot = 'judge'"
            ).fetchone()
        attention = int(row["attention"])
        return queue_snapshot(
            row["pending"],
            row["oldest_pending_sec"],
            waiting=attention,
            attention=attention,
        )

    def _areas_queue_stat(self) -> dict:
        """The raw snapshot of the areas queue: the pending sum over ENABLED ones (FR-2.5).

                    The pending source — the same as `_areas_queue_empty`: pending records
                    of active rows of the enabled areas (`enabled_areas`, #40) — a
                    disabled area is not drained in the background and does not enter the
                    shared counter. The sums add up; the oldest's age — the maximum
                    `now - updated_at` over the areas; empty — `null`. A disabled area's
                    accumulation is visible separately in `/health.areas` (`areas_health`).
                    The pending → waiting move on an embedding breaker (areas are encoded
                    by the same slot, #39, arch §6.9) is done by the framework (`is_gated`
                    → `gated_queue_stat`, 3.4.1, LSBN-T-29).
        """
        pending = 0
        oldest: int | None = None
        with session(self._settings) as conn:
            for spec in enabled_areas(self._settings):
                row = conn.execute(
                    "SELECT COUNT(*) AS pending, "
                    "MAX(CAST(strftime('%s','now') AS INTEGER) - "
                    "CAST(strftime('%s', updated_at) AS INTEGER)) "
                    "AS oldest_pending_sec FROM "
                    f"{spec.table} WHERE vector_status = 'pending' "
                    "AND deleted_at IS NULL"
                ).fetchone()
                pending += int(row["pending"])
                value = row["oldest_pending_sec"]
                if value is not None:
                    oldest = int(value) if oldest is None else max(oldest, int(value))
        return queue_snapshot(pending, oldest)

    def areas_health(self) -> dict[str, dict]:
        """`/health.areas` (FR-4.3, #40): per ALL areas `{enabled, pending}`.

                    Served over `ALL_AREAS`, not `enabled_areas`: a disabled area's
                    accumulation must be visible (the `JOB_*_ENABLED` precedent: "the
                    queue stays visible in /health"). `pending` — the same SQL as
                    `_areas_queue_stat` but WITHOUT summing (per area); no model calls —
                    only reading the pending state.
        """
        enabled = {spec.name for spec in enabled_areas(self._settings)}
        health: dict[str, dict] = {}
        with session(self._settings) as conn:
            for spec in ALL_AREAS:
                row = conn.execute(
                    f"SELECT COUNT(*) FROM {spec.table} "
                    "WHERE vector_status = 'pending' "
                    "AND deleted_at IS NULL"
                ).fetchone()
                health[spec.name] = {
                    "enabled": spec.name in enabled,
                    "pending": int(row[0]),
                }
        return health

    # --- synchronous work (runs in to_thread) ---------------------------------

    def process_pending(self, limit: int | None = None) -> int:
        """Vectorize one pending batch; returns the processed count.

                    The default batch size — settings.embedding_batch_size (pool 5: the
                    notes loop is cut by EMBEDDING_BATCH_SIZE, like the chunk
                    process_pending_chunks); an explicit `limit` overrides.

                    The full vector is built over title+text (_embed_input, lsb-0001
                    FR-1.1); a titleless note (title IS NULL — the legacy path before
                    title generation) is encoded over the bare text.

                    An encoding failure — 0: statuses untouched, the worker will wait out
                    the back-off. The guard (pool 1): the vector is written only when
                    unchanged since the read (id, text, title, namespace,
                    vector_status='pending') — otherwise a memory_update/a move/title
                    generation in flight would leave a stale vector with the 'ok' status.
                    The missed ones stay pending → the next batch re-encodes them (a new
                    text or title).
                    After each actual vectorization a judge job (dedup) is created in the
                    judge slot's queue (decision #10): the judge is asked only over a
                    ready vector — the dependency dispatcher. The merging itself is in the
                    judge loop (process_judge_pending) and the summary loop
                    (process_merge_pending).
                    If at least one batch note actually got a vector, the `links` loop is
                    woken too (`notify_links_pending`, the gate-1c decision): link level 1
                    is computed at once, not after an interval — but by the same `links_at`
                    marker, so the task is not lost by construction. One event per batch
                    suffices. An encoding failure (`EmbeddingError`) exits earlier — no
                    event; the tasks wait for a ready vector (back-off).
        """
        batch = (
            limit if limit is not None else self._settings.embedding_batch_size
        )
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, text, title, namespace FROM notes "
                "WHERE vector_status = 'pending' AND deleted_at IS NULL "
                "ORDER BY id LIMIT ?",
                (batch,),
            ).fetchall()
        if not rows:
            return 0
        try:
            embeddings = self._embedding.embed_texts(
                [_embed_input(row["title"], row["text"]) for row in rows]
            )
        except EmbeddingError:
            return 0
        processed = 0
        for row, vector in zip(rows, embeddings):
            with session(self._settings) as conn, transaction(conn):
                # The guard (pool 1): the vector — only if the note did not change since
                # the read (id, text, title, namespace, vector_status='pending')
                # — otherwise a memory_update/a move/title generation in flight would leave
                # a stale vector with the 'ok' status. `title IS ?`
                # also compares NULL (SQLite): a legacy titleless note
                # passes the guard only with the former NULL. The missed ones
                # stay pending → the next batch re-encodes them
                # (a new text or title).
                cursor = conn.execute(
                    "UPDATE notes SET vector_status = 'ok' "
                    "WHERE id = ? AND text = ? AND title IS ? AND namespace = ? "
                    "AND vector_status = 'pending'",
                    (row["id"], row["text"], row["title"], row["namespace"]),
                )
                if not cursor.rowcount:
                    continue
                # Phase 10: the vector is written into the note namespace's partition.
                vectors.upsert(conn, row["id"], vector, row["namespace"])
                processed += 1
            # Phase 11 (decision #10): the vector is ready — a judge job (dedup) goes
            # into the judge slot's queue; the dependency dispatcher.
            self._create_job("judge", "dedup", int(row["id"]))
        if processed:
            # A batch note actually got a vector — wake `links` (gate 1c).
            # One event per batch suffices; an encoding failure does not
            # reach here (EmbeddingError above) — "models unavailable → tasks
            # wait" holds.
            self.notify_links_pending()
        self.notify_judge_pending()
        return processed

    # --- the areas loop: area records' vectors (substrate 3.0.0) --------------

    def process_pending_areas(self, limit: int | None = None) -> int:
        """Vectorize a batch of pending area records; the processed count.

                    Each area (skills/terms/user) — by its own batch: pending records
                    (`vector_status='pending'`, active) → `embed_texts` over the area's
                    text (`AreaSpec.embed_text`) → an upsert into its vec0 → `'ok'`. The
                    batch size — `embedding_batch_size` (as the notes queue's). Guard: the
                    vector is written only if the record did not change since the read
                    (the vectorization fields) — otherwise an update in flight would leave
                    a stale vector with the 'ok' status. An encoder failure — the records
                    stay pending, an `area_embed_failed` event (NFR-3), a retry on the
                    areas loop's back-off.
        """
        batch = (
            limit if limit is not None else self._settings.embedding_batch_size
        )
        processed = 0
        for spec in enabled_areas(self._settings):
            processed += self._process_area_batch(spec, batch)
        return processed

    def _process_area_batch(self, spec: AreaSpec, batch: int) -> int:
        """One area's batch: read pending → encode → upsert + 'ok'."""
        columns = ", ".join(("id", *spec.embed_fields))
        with session(self._settings) as conn:
            rows = conn.execute(
                f"SELECT {columns} FROM {spec.table} "
                "WHERE vector_status = 'pending' AND deleted_at IS NULL "
                "ORDER BY id LIMIT ?",
                (batch,),
            ).fetchall()
        if not rows:
            return 0
        try:
            embeddings = self._embedding.embed_texts(
                [spec.embed_text(row) for row in rows]
            )
        except EmbeddingError as exc:
            # #39: an embedding slot failure carries the outcome type (the records — pending).
            # Release 3.9.0 (LSBN-S-7): the breaker key is the operation — the
            # `embed` operation in the `operation` mode, the `embedding` slot in
            # the `legacy` mode.
            self._log_slot_failure(
                AREAS_JOB,
                self._operation_slot("embed"),
                outcome_of(exc),
                event="area_embed_failed",
                message="area vectorization failed — records stay pending",
                detail=str(exc),
                area=spec.name,
                count=len(rows),
            )
            return 0
        processed = 0
        guard = " AND ".join(f"{column} IS ?" for column in spec.embed_fields)
        for row, vector in zip(rows, embeddings):
            with session(self._settings) as conn, transaction(conn):
                # The guard over vectorization fields: `IS ?` also compares NULL.
                cursor = conn.execute(
                    f"UPDATE {spec.table} SET vector_status = 'ok' "
                    "WHERE id = ? AND vector_status = 'pending' "
                    f"AND deleted_at IS NULL AND {guard}",
                    (row["id"], *(row[column] for column in spec.embed_fields)),
                )
                if not cursor.rowcount:
                    continue
                area_vectors.upsert(
                    conn,
                    spec.vec_table,
                    spec.vec_id_column,
                    int(row["id"]),
                    vector,
                )
                processed += 1
        return processed

    def _areas_queue_empty(self) -> bool:
        """Is the areas queue empty (a cheap SELECT, pool 6, lost wakeup).

                    Pending records are counted over the registry's enabled areas
                    (`enabled_areas`, #40) — the same queue process_pending_areas
                    drains.
        """
        pending = 0
        with session(self._settings) as conn:
            for spec in enabled_areas(self._settings):
                pending += int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM {spec.table} "
                        "WHERE vector_status = 'pending' "
                        "AND deleted_at IS NULL"
                    ).fetchone()[0]
                )
        # #39 (arch §6.9): areas are encoded by the embedding slot — on its
        # breaker no batch is taken; the queue is empty for the loop (no busy-loop).
        # Release 3.9.0 (LSBN-S-6): the breaker is read through the operation's
        # key — the `embed` operation in the `operation` mode (the slot name
        # `embedding` has no breaker there — the S-5 finding M1: a direct
        # lookup raised `KeyError`).
        return pending == 0 or self._breaker_paused(self._operation_slot("embed"))

    # --- the judge loop: the dedup judge (Phase 8, Stage 3.2; Phase 11, #10) --

    def process_judge_pending(self, limit: int | None = None) -> int:
        """Process a batch of judge jobs (dedup); the processed count.

                    Per judge job (created after a note's vectorization): cosine
                    candidates against earlier notes (_find_dedup_candidates,
                    DEDUP_CANDIDATE_* — a prefilter only) and a "duplicate" verdict — the
                    LLM judge (Stage 3.2, JudgeService; without a judge — the Stage-2.2
                    cosine fallback). A recognized duplicate → a merge job into the
                    summary slot's queue (decision #10: merge walks into the summary
                    slot). A judge failure (JudgeError) — the work stays pending, a retry
                    on the judge loop's back-off (NFR-3: both notes intact); a wait-class
                    outcome (unavailable/timeout/auth, `_BATCH_ABORT_OUTCOMES`, LSBN-T-28)
                    also releases the batch — the other positions stay pending and are
                    picked up on back-off after the slot returns (the migration
                    retry-later precedent); otherwise a "hanging" provider would burn up
                    to 32 × 900 s per batch. A stale note (deleted/the vector not ready) —
                    the job is released.
                    A note with a timer (TTL, no46-03) — also released, but BEFORE the
                    judge and without model calls (a `dedup_skipped` event,
                    reason="note has a lifetime").
                    A part of a split (3.6.2, LSBN-D-14): the candidates of the OWN
                    origin group are dropped by the prefilter (find_candidates) - the
                    parts of one source never reach the judge and never create a merge
                    job.
        """
        # The "pair edit" protocol: frozen jobs are rechecked without
        # a model call — the gone reason unfreezes them (arch §6.8).
        self._review_attention_jobs("judge", "dedup")
        jobs = self._pending_jobs(
            "judge",
            "dedup",
            limit if limit is not None else self._settings.embedding_batch_size,
        )
        if not jobs:
            return 0
        done = 0
        for job in jobs:
            note_id = int(job["note_id"])
            with session(self._settings) as conn:
                row = conn.execute(
                    "SELECT id, text, namespace, vector_status, deleted_at, "
                    "expires_at "
                    "FROM notes WHERE id = ?",
                    (note_id,),
                ).fetchone()
                vector = vectors.get_vector(conn, note_id) if row is not None else None
            if (
                row is None
                or row["deleted_at"] is not None
                or row["vector_status"] != "ok"
                or vector is None
            ):
                self._mark_job_done(job["id"])
                done += 1
                continue
            if row["expires_at"] is not None:
                # no46-03 (3.2.0): a note with a timer (TTL, `expires_at IS NOT NULL`) is
                # NEVER merged — an early exit BEFORE the judge and before
                # candidate search: no models poked, the job is released;
                # an observable dedup_skipped event with the reason.
                # Links of such a note are allowed (links is not here).
                logging.getLogger("app").info(
                    "dedup: skipped — note has a lifetime",
                    extra={
                        "event": "dedup_skipped",
                        "job": JUDGE_JOB,
                        "note_id": note_id,
                        "reason": "note has a lifetime",
                    },
                )
                self._mark_job_done(job["id"])
                done += 1
                continue
            older = self._find_dedup_candidates(
                note_id, vector, row["namespace"]
            )
            if older:
                # Re-read the fresh note's and all candidates' texts (the race with
                # memory_update/delete: stale candidates are cut off in advance —
                # neither the judge nor the summarizer is asked about them).
                ids = [note_id] + [candidate_id for candidate_id, _ in older]
                placeholders = ",".join("?" * len(ids))
                with session(self._settings) as conn:
                    rows = conn.execute(
                        f"SELECT id, text, deleted_at, updated_at FROM notes "
                        f"WHERE id IN ({placeholders})",
                        ids,
                    ).fetchall()
                by_id = {r["id"]: r for r in rows}
                newer = by_id.get(note_id)
                if newer is None or newer["deleted_at"] is not None:
                    self._mark_job_done(job["id"])
                    done += 1
                    continue
                alive = [
                    (candidate_id, cosine_value)
                    for candidate_id, cosine_value in older
                    if (r := by_id.get(candidate_id)) is not None
                    and r["deleted_at"] is None
                ]
                # The pair identity (#39 §6.8/§13.3): the best candidate and the sides'
                # stamps ("note", "candidate"). A new best candidate or
                # an edit of either side — a DIFFERENT pair: the counter and attention
                # are reset, the work goes as new; the stamps are kept for the
                # next comparison (including for a frozen job,
                # `_review_attention_jobs`).
                best_id = int(alive[0][0]) if alive else None
                stored = self._load_payload(job["payload"])
                if best_id is not None:
                    candidate_updated = by_id[best_id]["updated_at"]
                    pair_changed = (
                        stored.get("candidate_id") is not None
                        and stored["candidate_id"] != best_id
                    ) or (
                        stored.get("note_updated_at") is not None
                        and stored["note_updated_at"] != newer["updated_at"]
                    ) or (
                        stored.get("candidate_updated_at") is not None
                        and stored["candidate_updated_at"] != candidate_updated
                    )
                    if pair_changed:
                        self._release_attention(job["id"], "edit")
                    stored["candidate_id"] = best_id
                    stored["note_updated_at"] = newer["updated_at"]
                    stored["candidate_updated_at"] = candidate_updated
                    self._set_job_payload(job["id"], stored)
                try:
                    best = self._pick_duplicate(note_id, alive, by_id)
                except JudgeError as exc:
                    # #39 (§6.7): a slot failure carries the outcome type and a reason fragment.
                    # Release 3.9.0 (LSBN-S-7): the breaker key is the operation.
                    self._log_slot_failure(
                        JUDGE_JOB,
                        self._operation_slot("judge"),
                        outcome_of(exc),
                        event="dedup_judge_failed",
                        message="dedup: judge undecidable — both notes kept, retry queued",
                        note_id=note_id,
                        detail=str(exc),
                        candidates=[
                            candidate_id for candidate_id, _ in alive
                        ],
                        meta=getattr(exc, "meta", None),
                    )
                    self._note_attempt(
                        job["id"],
                        outcome_of(exc),
                        detail=str(exc),
                    )
                    if outcome_of(exc) in _BATCH_ABORT_OUTCOMES:
                        # transient/auth: release the batch; the rest — a retry on
                        # back-off (the migration retry-later precedent).
                        break
                    continue  # the work stays pending — a retry on back-off
                if best is not None:
                    older_id, cosine_value = best
                    # A merge job into the summary slot's queue (decision #10).
                    self._create_job(
                        "summary",
                        "merge",
                        note_id,
                        payload=json.dumps(
                            {"older_id": older_id, "cosine": cosine_value}
                        ),
                    )
            self._mark_job_done(job["id"])
            done += 1
        return done

    # --- the summary loop: merge jobs (duplicate merging, Stage 2.2) ----------

    def process_merge_pending(self, limit: int | None = None) -> int:
        """Process a batch of merge jobs (duplicate merging); the processed count.

                    A merge job is created by the judge loop after the judge's verdict
                    (decision #10: merge walks into the summary slot). The procedure
                    (variant B): 1) re-read the texts of the earlier and later notes
                    (the race with memory_update/delete: a stale pair is cut off);
                    2) summarizer.merge(earlier_text, later_text) — merge;
                    3) NoteService.merge_pair — update the earlier note in ONE transaction
                    (text = merged; re-vectorization and re-summarization — regular, by
                    their queues; the earlier note's title is untouched) and soft-delete
                    the later one (trash). Pool 6: a single transaction excludes the
                    half-state "the earlier updated, the later alive" on a failure between
                    the operations.

                    no46-03 (3.2.0), the second barrier: if any note of the pair has a
                    timer (`expires_at IS NOT NULL`) — no merge, the job is released, a
                    `dedup_skipped` event (reason="note has a lifetime"). This is defense
                    in depth: a TTL note is not merged from the outside either (the normal
                    path never reaches here — the judge loop does not process such a
                    note).

                    LSBN-D-14 (3.6.2), the third barrier: a pair of ONE source (both
                    notes carry the same non-NULL `origin_note_id` - the parts of one
                    split) is never merged: the parts are legitimate neighbours by
                    construction, merging them back loses content. The job is
                    released, a `dedup_skipped` event (reason="same origin").
                    A merge failure (SummaryError, NFR-3) does not spoil the data: both
                    notes stay, the work stays pending — a retry on the summary loop's
                    back-off.

                    Release 3.9.0 (LSBN-S-12): in the `operation` mode the merge
                    answers with the free text (the contract `parts-by-delimiter`)
                    and the code names the parts: the FIRST part keeps the title of
                    the earlier note (the decision O. Q1 of 15.11.2026 — a merge
                    does not create a new entity, no model call is made), the other
                    parts are written with a NULL title and go to the `title` queue.
                    The `legacy` mode is unchanged.
        """
        if self._summarizer is None:
            return 0  # a test mode without a summarizer: merging is impossible
        self._review_attention_jobs("summary", "merge")
        jobs = self._pending_jobs(
            "summary",
            "merge",
            limit if limit is not None else self._settings.embedding_batch_size,
        )
        if not jobs:
            return 0
        done = 0
        for job in jobs:
            note_id = int(job["note_id"])
            try:
                payload = json.loads(job["payload"] or "{}")
            except (TypeError, ValueError):
                payload = {}
            older_id = payload.get("older_id")
            if older_id is None:
                self._mark_job_done(job["id"])
                done += 1
                continue
            with session(self._settings) as conn:
                rows = conn.execute(
                    "SELECT id, text, title, deleted_at, updated_at, expires_at, "
                    "origin_note_id "
                    "FROM notes "
                    "WHERE id IN (?, ?)",
                    (older_id, note_id),
                ).fetchall()
            by_id = {r["id"]: r for r in rows}
            older = by_id.get(older_id)
            newer = by_id.get(note_id)
            if (
                older is None
                or newer is None
                or older["deleted_at"] is not None
                or newer["deleted_at"] is not None
            ):
                self._mark_job_done(job["id"])
                done += 1
                continue
            if older["expires_at"] is not None or newer["expires_at"] is not None:
                # no46-03 (3.2.0), the second barrier (defense in depth): if ANY note of the
                # pair has a timer — no merge (the judge/summarizer are
                # not called), the job is released, an event of the same kind.
                logging.getLogger("app").info(
                    "dedup: merge skipped — a note has a lifetime",
                    extra={
                        "event": "dedup_skipped",
                        "job": self._job_name("merge"),
                        "older_id": older_id,
                        "note_id": note_id,
                        "reason": "note has a lifetime",
                    },
                )
                self._mark_job_done(job["id"])
                done += 1
                continue
            if (
                older["origin_note_id"] is not None
                and older["origin_note_id"] == newer["origin_note_id"]
            ):
                # 3.6.2 (LSBN-D-14), the barrier of the merge path: a pair of ONE source
                # (both notes carry the same non-NULL `origin_note_id` - the parts of one
                # split) is never merged - the parts are legitimate neighbours by
                # construction, merging them back loses content. The job is released and
                # an observable dedup_skipped follows, as for all non-candidate classes.
                logging.getLogger("app").info(
                    "dedup: merge skipped — one source",
                    extra={
                        "event": "dedup_skipped",
                        "job": self._job_name("merge"),
                        "older_id": older_id,
                        "note_id": note_id,
                        "reason": "same origin",
                    },
                )
                self._mark_job_done(job["id"])
                done += 1
                continue
            # The merge job's pair — the note (`older_id`); the sides' stamps for
            # the pair-edit comparison (#39 §6.8/§13.3): changed — the counter and
            # attention reset, the work goes as new.
            pair_changed = (
                payload.get("older_updated_at") is not None
                and payload["older_updated_at"] != older["updated_at"]
            ) or (
                payload.get("newer_updated_at") is not None
                and payload["newer_updated_at"] != newer["updated_at"]
            )
            if pair_changed:
                self._release_attention(job["id"], "edit")
            if (
                payload.get("older_updated_at") != older["updated_at"]
                or payload.get("newer_updated_at") != newer["updated_at"]
            ):
                payload["older_updated_at"] = older["updated_at"]
                payload["newer_updated_at"] = newer["updated_at"]
                self._set_job_payload(job["id"], payload)
            try:
                merge_result = self._summarizer.merge(
                    older["text"], newer["text"], title=older["title"]
                )
                # Observability (F-23 p. 7): a slot_call for EVERY call —
                # the follow-up call (the 2nd round) is visible in the journal as a separate line.
                for meta in merge_result.metas or [merge_result.meta]:
                    self._log_slot_call(
                        self._job_name("merge"),
                        self._operation_slot("merge"),
                        meta,
                        older_id=older_id,
                        note_id=note_id,
                    )
                outcome = self._notes.merge_into_parts(
                    older_id,
                    note_id,
                    # An empty title — the part of the `parts-by-delimiter`
                    # contract: it is written with a NULL title and goes to the
                    # `title` queue (release 3.9.0, LSBN-S-12).
                    [(part.title, part.text) for part in merge_result.parts],
                )
                if not outcome["merged"]:
                    # The earlier note vanished (the operator deleted it) between the read and
                    # the merge — no merge happened, the job is released;
                    # the later one is not touched (merge_into_parts writes
                    # nothing when there is no active earlier note).
                    self._mark_job_done(job["id"])
                    done += 1
                    continue
            except MergeIncompleteError as exc:
                # F-23 p. 6: the merge did not converge within MERGE_MAX_ROUNDS (the model
                # loses content). A retry would give the same two rounds — the pair is
                # DEFERRED: the job is released, BOTH notes are alive, an event
                # into the jobs journal and a warnings record (the subject — the pair's earlier note,
                # the payload carries the later one and the reason).
                self._mark_job_done(job["id"])
                done += 1
                self._log_slot_failure(
                    self._job_name("merge"),
                    self._operation_slot("merge"),
                    outcome_of(exc),
                    event="dedup_merge_deferred",
                    message=(
                        "dedup: merge incomplete after "
                        f"{getattr(exc, 'rounds', MERGE_MAX_ROUNDS)} rounds — "
                        "pair deferred, both notes kept"
                    ),
                    older_id=older_id,
                    note_id=note_id,
                    detail=str(exc),
                    meta=getattr(exc, "meta", None),
                )
                self._defer_merge_pair(
                    older_id,
                    note_id,
                    reason=str(exc),
                    rounds=int(getattr(exc, "rounds", MERGE_MAX_ROUNDS)),
                )
                continue
            except SummaryError as exc:
                self._log_slot_failure(
                    self._job_name("merge"),
                    self._operation_slot("merge"),
                    outcome_of(exc),
                    event="dedup_merge_failed",
                    message="dedup: summarizer merge failed — both notes kept, retry queued",
                    older_id=older_id,
                    note_id=note_id,
                    detail=str(exc),
                    meta=getattr(exc, "meta", None),
                )
                self._note_attempt(
                    job["id"],
                    outcome_of(exc),
                    detail=str(exc),
                )
                continue  # the work stays pending — a retry on back-off
            logging.getLogger("app").info(
                "dedup: duplicate merged into self-contained parts",
                extra={
                    "event": "dedup_merged",
                    "job": self._job_name("merge"),
                    "older_id": older_id,
                    "note_id": note_id,
                    "cosine": payload.get("cosine"),
                    # F-23 p. 7: the rounds, the model, the parts count — in the jobs journal.
                    "rounds": merge_result.rounds,
                    "model": merge_result.meta.model,
                    "parts": len(merge_result.parts),
                    "part_ids": [part["id"] for part in outcome["parts"]],
                },
            )
            self._mark_job_done(job["id"])
            done += 1
            # Placing the parts (lsb-0012, arch §3.5): each part — a new note — is a node-sweep
            # candidate (classified_at NULL); we place one task per part into the
            # `nodes` slot and wake its loop at once, without waiting
            # for the hourly interval. A failed placement/parse
            # does not spoil the data: the part stays in the earlier note's node.
            for part in outcome["parts"]:
                self._ensure_job(NODES_JOB, NODES_RECLASS_KIND, int(part["id"]))
            self.notify_nodes_pending()
            # The parts wait for summarization and vectorization — wake their own loops,
            # without waiting for the back-off.
            self.notify_summary_pending()
            self.notify_embedding_pending()
        return done

    def _defer_merge_pair(
        self, older_id: int, note_id: int, *, reason: str, rounds: int
    ) -> None:
        """Defer a merge pair (F-23 p. 6): a warnings record.

                    An open kind=`merge_deferred` warning on the pair's earlier note
                    (upsert: re-deferring the same pair moves updated_at, does not
                    multiply rows); the payload carries the later note, the rounds count
                    and the reason. With the warnings service unavailable (test DI builds)
                    the deferring remains a journal event only — it does not spoil the
                    data.
        """
        if self._warnings is None:
            return
        try:
            self._warnings.open(
                MERGE_DEFERRED_WARNING,
                "note",
                older_id,
                (
                    f"merge deferred after {rounds} rounds: parts do not fully "
                    "cover the sources — both notes kept"
                ),
                {"newer_id": note_id, "rounds": rounds, "reason": reason[:500]},
            )
        except Exception:  # noqa: BLE001 — a warning does not break the loop
            logging.getLogger("app").warning(
                "dedup: failed to record merge_deferred warning",
                extra={
                    "event": "warning_open_failed",
                    "job": self._job_name("merge"),
                    "older_id": older_id,
                    "note_id": note_id,
                },
            )

    # --- background dedup (Phase 8, Stages 2–3) -------------------------------

    def _find_dedup_candidates(
        self, note_id: int, vector: list[float], namespace: str = "default"
    ) -> list[tuple[int, float]]:
        """Dedup cosine candidates after a note's vectorization.

                    Candidates are searched only against **earlier** notes (a smaller id):
                    a "later ↔ earlier" pair is processed once, from the later side — the
                    merge (Stage 2.2) updates the earlier duplicate and soft-deletes the
                    later one, and the reverse run of the same pair from the earlier
                    note's side would loop the processing. Phase 10 (§5.7): only within
                    the note's namespace. Candidates are logged (observability); the
                    "duplicate" verdict is made by process_judge_pending: each candidate
                    is asked of the judge (Stage 3.2, JudgeService — cosine is only a
                    prefilter); without a judge — the DEDUP_SIMILARITY cosine fallback
                    (Stage 2.2).

                    Returns a list [(candidate_id, cosine)] — the merge's input.
        """
        found = self._dedup.find_candidates(
            vector, exclude_id=note_id, namespace=namespace
        )
        older = [pair for pair in found if pair[0] < note_id]
        if older:
            logging.getLogger("app").info(
                "dedup: cosine candidates found for vectorized note",
                extra={
                    "event": "dedup_candidates",
                    "job": JUDGE_JOB,
                    "note_id": note_id,
                    "candidates": older,
                },
            )
        return older

    def _pick_duplicate(
        self,
        note_id: int,
        candidates: list[tuple[int, float]],
        by_id: dict,
    ) -> tuple[int, float] | None:
        """Pick a candidate for merging: the judge (Stage 3.2) or the fallback.

                    The judge (Judge): asked per each live candidate in closeness
                    descending order (the find_candidates output order); the first
                    "DUPLICATE" verdict — the sentence (candidate_id, cosine). "NOT
                    DUPLICATE" for all — None: no merge, the note counts as processed
                    (repeating the question over an unchanged text is pointless: dedup
                    waits for the next vectorization, and that happens only after a text
                    change). Each pair is logged (event=dedup_judge) — the prefilter
                    verdicts' observability.

                    The fallback without a judge (DI None, the Stage-2.2 test mode): the
                    first candidate with cosine ≥ DEDUP_SIMILARITY (the output is
                    sorted).

                    JudgeError is propagated — process_judge_pending treats a judge
                    failure as a merge failure (the work stays pending, a retry on
                    back-off, NFR-3): uncertainty is never turned into "not duplicate".

                    by_id — freshly read notes rows (id → row): the judge compares the
                    pair's texts (the fresh one, the candidate) — the argument order is
                    fixed by JUDGE_USER_TEMPLATE (TEXT 1 — the new one, TEXT 2 — the
                    candidate).
        """
        if self._judge is None:
            # The Stage-2.2 fallback (a test mode): the output is sorted.
            return next(
                (
                    pair
                    for pair in candidates
                    if pair[1] >= self._settings.dedup_similarity
                ),
                None,
            )
        for candidate_id, cosine_value in candidates:
            result = self._judge.judge(
                by_id[note_id]["text"], by_id[candidate_id]["text"]
            )
            self._log_slot_call(
                JUDGE_JOB,
                self._operation_slot("judge"),
                result.meta,
                note_id=note_id,
                candidate_id=candidate_id,
            )
            logging.getLogger("app").info(
                "dedup: judge verdict for cosine candidate",
                extra={
                    "event": "dedup_judge",
                    "job": JUDGE_JOB,
                    "note_id": note_id,
                    "candidate_id": candidate_id,
                    "cosine": cosine_value,
                    "verdict": result.verdict,
                },
            )
            if result.verdict:
                return (candidate_id, cosine_value)
        return None

    # --- the summary loop: title generation (decision #9) ---------------------

    def process_title_pending(self, limit: int | None = None) -> int:
        """Generate titles for migration notes (title IS NULL).

                    Only migration notes (new ones always have a title — the decision-#9
                    contract); the queue is filled only by migration and empties after
                    processing. A thinking call of the summary slot (SummaryService.title
                    — the prompt baked in as TITLE_PROMPT, Phase 11); the result is
                    trimmed to TITLE_MAX_WORDS words mechanically; the title is written.
                    A generation failure (SummaryError) — the note stays titleless, a
                    retry on back-off (NFR-3); a wait-class outcome
                    (unavailable/timeout/auth, `_BATCH_ABORT_OUTCOMES`, LSBN-T-28) also
                    releases the batch — the other positions stay titleless and are picked
                    up on back-off after the slot returns (the migration retry-later
                    precedent). Returns the number of written titles (up to the failure,
                    if the batch was released).

                    Writing the title invalidates the full vector (lsb-0001 FR-1.1: the
                    title takes part in encoding) — in the same transaction vectorization
                    returns to 'pending' and notes_vec is reset; the notes-queue worker
                    re-encodes with the title (_embed_input). Chunk vectors are untouched
                    (decision D1: the title does not affect them).
        """
        if self._summarizer is None:
            return 0
        batch = (
            limit if limit is not None else self._settings.embedding_batch_size
        )
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, text FROM notes "
                "WHERE title IS NULL AND deleted_at IS NULL "
                "ORDER BY id LIMIT ?",
                (batch,),
            ).fetchall()
        done = 0
        for row in rows:
            try:
                generated = self._summarizer.title(row["text"])
            except SummaryError as exc:
                self._log_slot_failure(
                    self._job_name("title"),
                    self._operation_slot("title"),
                    outcome_of(exc),
                    event="title_failed",
                    message="title: generation failed — kept null, retry by back-off",
                    note_id=row["id"],
                    detail=str(exc),
                    meta=getattr(exc, "meta", None),
                )
                if outcome_of(exc) in _BATCH_ABORT_OUTCOMES:
                    # transient/auth: release the batch; the rest — a retry on
                    # back-off (the migration retry-later precedent).
                    break
                continue  # failure: the title stays NULL, a retry on back-off
            self._log_slot_call(
                self._job_name("title"),
                self._operation_slot("title"),
                generated.meta,
                note_id=row["id"],
            )
            title = self._truncate_title(generated.text)
            if not title:
                continue
            with session(self._settings) as conn, transaction(conn):
                cursor = conn.execute(
                    "UPDATE notes SET title = ? "
                    "WHERE id = ? AND title IS NULL AND deleted_at IS NULL",
                    (title, row["id"]),
                )
                if cursor.rowcount:
                    # lsb-0001 FR-1.1: the title is part of the full vector —
                    # writing the title makes the old vector (over the bare text)
                    # stale: the status to 'pending', notes_vec reset (the
                    # same meaning as vectors.drop in notes.update(): in the
                    # pending window the note takes part in FTS search only; the old
                    # vector without the title feeds neither search nor dedup).
                    # Chunk vectors are untouched (decision D1: they are encoded over the
                    # chunk's bare text).
                    conn.execute(
                        "UPDATE notes SET vector_status = 'pending' WHERE id = ?",
                        (row["id"],),
                    )
                    stale = vectors.get_vector(conn, row["id"]) is not None
                    vectors.drop(conn, row["id"])
            if cursor.rowcount:
                done += 1
                # The note returned to the vectorization queue (the title is part of the
                # full vector) — wake the embedding loop at once.
                self.notify_embedding_pending()
                logging.getLogger("app").info(
                    "title: generated for migration note",
                    extra={
                        "event": "title_generated",
                        "job": self._job_name("title"),
                        "note_id": row["id"],
                        "vector_invalidated": stale,
                    },
                )
        return done

    @staticmethod
    def _truncate_title(text: str) -> str:
        """Trim a generated title to TITLE_MAX_WORDS words (decision #9).

                    Words = len(title.split()) — as in the validation contract
                    (notes.py).
        """
        words = text.split()
        return " ".join(words[:TITLE_MAX_WORDS])

    def process_summary_pending(self, limit: int | None = None) -> int:
        """Summarize one pending batch; the count brought to 'ok'.

                    Mode "B" (§5.5): generation — only here, over the queue's notes. One
                    note's generation failure does not cancel the others (NFR-3): the
                    status stays pending, the note is caught up by the next batch. The
                    exception — wait-class outcomes (unavailable/timeout/auth,
                    `_BATCH_ABORT_OUTCOMES`, LSBN-T-28): the provider is down or hanging
                    (a read timeout up to 900 s per position) — the batch is released (the
                    migration retry-later precedent), the positions processed before the
                    failure count, the tail stays pending and is picked up on back-off
                    after the slot returns. Trash (deleted_at IS NOT NULL) is not served —
                    as in vectorization.

                    The memory_update race (ARCH §4.5): the summary is written only if the
                    text did not change since the read (`AND text = ?`) — a stale summary
                    does not overwrite a fresh note.

                    The length limit (the 3.1.0 gate decision, 2026-09-19): the model's
                    answer is saved only after `cap_summary` — a hard trim to
                    MAX_SUMMARY_CHARS at a word boundary with an ellipsis. This is the
                    only point where a model summary is written, so the limit does not
                    depend on the model's discipline.

                    The short-note short-circuit (LSBN-T-37, 3.6.3): a note not longer
                    than SUMMARY_SHORT_TEXT_CHARS (0 — the rule is off) needs no model —
                    its summary is its own text, cut by the same `cap_summary` and saved
                    by the same guarded UPDATE; the journal gets one `summary_shortcircuit`
                    event per note (note_id, text_chars, summary_chars).

                    The summary's actual readiness — a signal to the neighboring loop
                    (modeled on `notify_links_pending` after vectorization, the
                    2026-09-20 gate decision): if at least one batch note was actually
                    brought to `summary_status='ok'`, the `nodes` loop is woken
                    (`notify_nodes_pending`) — its `reclass` task after duplicate
                    stitching may have waited exactly for that summary and "slept" on
                    back-off (up to 15 min in the acceptance contour). A per-item
                    generation failure (`continue` above — the status stays pending), the
                    position that released the batch (a wait class, `break` — its status
                    also stays pending), and an empty run give no signal: "models
                    unavailable → the task waits" holds.
        """
        if self._summarizer is None:
            return 0
        batch = (
            limit if limit is not None else self._settings.embedding_batch_size
        )
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, text, namespace, classified_at FROM notes "
                "WHERE summary_status = 'pending' AND deleted_at IS NULL "
                "ORDER BY id LIMIT ?",
                (batch,),
            ).fetchall()
        done = 0
        short_chars = self._settings.summary_short_text_chars
        for row in rows:
            text = row["text"]
            if short_chars > 0 and len(text) <= short_chars:
                # LSBN-T-37 (3.6.3): the summary of a short note is its own text —
                # a model has nothing to add to 100 characters "in other words",
                # it only burns the slot. The write path below is shared, so the
                # cap, the race guard and the readiness signal stay one.
                summary = cap_summary(text, self._settings.max_summary_chars)
                logging.getLogger("app").info(
                    "summary: short note — summarized without a model call",
                    extra={
                        "event": "summary_shortcircuit",
                        "job": self._job_name("summarize"),
                        "note_id": row["id"],
                        "text_chars": len(text),
                        "summary_chars": len(summary),
                    },
                )
            else:
                try:
                    summary_result = self._summarizer.summarize(text)
                except SummaryError as exc:
                    self._log_slot_failure(
                        self._job_name("summarize"),
                        self._operation_slot("summarize"),
                        outcome_of(exc),
                        event="summary_failed",
                        message=(
                            "summary: generation failed — kept pending, "
                            "retry by back-off"
                        ),
                        note_id=row["id"],
                        detail=str(exc),
                        meta=getattr(exc, "meta", None),
                    )
                    if outcome_of(exc) in _BATCH_ABORT_OUTCOMES:
                        # transient/auth: release the batch; the rest — a retry on
                        # back-off (the migration retry-later precedent).
                        break
                    continue  # failure: the pending status stays, a retry on back-off
                self._log_slot_call(
                    self._job_name("summarize"),
                    self._operation_slot("summarize"),
                    summary_result.meta,
                    note_id=row["id"],
                )
                # The length safety net (the 3.1.0 gate decision, 2026-09-19): the
                # model may ignore the prompt ("no more than 150 characters") — what
                # reaches the DB is a summary truncated at a word boundary, always
                # ≤ MAX_SUMMARY_CHARS. The short-note branch above goes through the
                # same cap: one rule for both sources.
                summary = cap_summary(
                    summary_result.text, self._settings.max_summary_chars
                )
            with session(self._settings) as conn, transaction(conn):
                cursor = conn.execute(
                    "UPDATE notes SET summary = ?, summary_status = 'ok' "
                    "WHERE id = ? AND summary_status = 'pending' AND text = ?",
                    (summary, row["id"], row["text"]),
                )
            if cursor.rowcount:
                done += 1
                # Grooming (Phase 10, Step 4): after summarizing a default note
                # (not yet classified) — the markup and the auto-move.
                if row["namespace"] == "default" and row["classified_at"] is None:
                    try:
                        self._classify_default_note(int(row["id"]), row["text"])
                    except asyncio.CancelledError:
                        raise  # loop cancellation — do not swallow
                    except Exception:
                        # An unexpected grooming failure (including a bug) does not crash the
                        # summary loop: the batch's other notes keep being
                        # processed; this one's classification — after the next update.
                        logging.getLogger("app").warning(
                            "classify: internal error — enrichment deferred",
                            extra={
                                "event": "classify_crashed",
                                "job": SUMMARY_JOB,
                                "note_id": row["id"],
                            },
                            exc_info=True,
                        )
        if done:
            # At least one batch note actually got a ready summary — wake the
            # `nodes` loop (lsb-0012, modeled on `notify_links_pending`).
            # One event per batch suffices; a failed position does not reach here with
            # a ready summary (a per-item `continue`, or `break` on the
            # wait class — the status stays pending), but the positions counted before the failure
            # released the batch honestly — the signal for them is justified.
            self.notify_nodes_pending()
        return done

    # --- grooming (Phase 10, Step 4) ------------------------------------------

    def _apply_node_order(
        self,
        note_id: int,
        expected_namespace: str,
        result: Classification,
        *,
        current_namespace: str | None = None,
        no_hint_reason: str = "low_confidence",
    ) -> tuple[str, str, str | None]:
        """The single point of a markup-driven note move (lsb-0011-01, arch §3.3).

                    One UPDATE: the markup (`hint_path`, `confidence`, `classified_at`) +
                    the parse marker `node_order_at` + (on a move) `namespace` and
                    `vector_status='pending'` (re-encoding into the new node's partition —
                    the existing mechanism). The `namespace = :expected_namespace` guard
                    and `deleted_at IS NULL`: an operator/client move in flight is not
                    overridden by the background — `rowcount = 0` gives
                    `kept`/`node_changed`.

                    The target node is computed BEFORE the transaction: a markup
                    validation failure (`NamespaceValidationError` from `_auto_move_target`,
                    including a garbage `hint_path`) writes nothing to the DB at all — the
                    strict "a failure = not marked" semantics. Returns
                    (outcome, reason, target): `moved` — a move; `kept` — kept
                    (`hint_unknown` — the node is not in the registry, `no_hint_reason` —
                    the model proposed no node, `low_confidence` — below the threshold,
                    `same_node` — the same node, `node_changed` — the node changed in
                    flight). All reasons — from the arch §3.7 dictionary.

                    One mechanic for all sources (arch §3.3): the grooming after
                    summarization (input by the full text), the `default` sweep (the fast
                    path and the classifier pool) and the after-stitching task all reuse
                    it.

                    `current_namespace` — the note's node read AT DECISION TIME (the
                    after-stitching task only, lsb-0012): then a no-node result gives
                    `no_hint_reason` (`result_default` — the note is NOT downgraded,
                    FR-2.4), and a confident choice of the current node —
                    `kept`/`same_node` without needless re-vectorization. For the sweep
                    and grooming (`None`) the behavior is as before: `expected_namespace`
                    — `'default'`.
        """
        target = self._auto_move_target(result)
        confident = (
            result.confidence >= self._settings.namespace_auto_move_min_confidence
        )
        move = target is not None and confident
        if current_namespace is not None and not result.hint_path:
            # A "generic" result: a "node → dump" move is forbidden (FR-2.4) —
            # the note stays in the earlier note's node.
            target = None
            move = False
        elif (
            current_namespace is not None
            and confident
            and target == current_namespace
        ):
            move = False  # the same node: no move (and no re-vectorization)
        # The parse marker is set in THE SAME transaction as the markup
        # (lsb-0011-01, §3.6): atomically; the anti-looping never lags the decision.
        columns = [
            "hint_path = ?",
            "confidence = ?",
            "classified_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')",
            "node_order_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')",
        ]
        params: list[object] = [result.hint_path, result.confidence]
        if move:
            columns.append("namespace = ?")
            columns.append("vector_status = 'pending'")
            params.append(target)
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE notes SET "
                + ", ".join(columns)
                + " WHERE id = ? AND namespace = ? AND deleted_at IS NULL",
                (*params, note_id, expected_namespace),
            )
            if cursor.rowcount and move:
                # #41 (FR-3.1/FR-4.4, §13.5): an actual `namespace` move —
                # a content change; the version is written by THE SAME transaction.
                # The source — from the job body's `bind_origin` (`job:<name>`);
                # the service `hint_path`/`confidence` are not in the projection.
                row = conn.execute(
                    "SELECT text, title, namespace, expires_at, comment FROM notes "
                    "WHERE id = ?",
                    (note_id,),
                ).fetchone()
                history.record_change(
                    conn, history.ENTITY_NOTE, note_id, history.note_projection(row)
                )
        if not cursor.rowcount:
            return "kept", "node_changed", target
        if move:
            return "moved", "hint_exists", target
        if target is None:
            return (
                "kept",
                "hint_unknown" if result.hint_path else no_hint_reason,
                None,
            )
        if current_namespace is not None and confident and target == current_namespace:
            return "kept", "same_node", target
        return "kept", "low_confidence", target

    def _classify_default_note(self, note_id: int, text: str) -> None:
        """Mark a default note after summarization; the auto-move.

                    Only default notes (placed ones are not shaken up, §5.7) and only one
                    pass (classified_at — anti-looping; a retry — after memory_update,
                    which resets the summary to pending). A classifier failure
                    (ClassificationError) does not spoil the data: the note stays in
                    default, classified_at is not set — a retry at the next update.

                    The auto-move into an existing node — at confidence ≥
                    NAMESPACE_AUTO_MOVE_MIN_CONFIDENCE: into an existing leaf (if the
                    subdomain_hint matched a registered one), else into the domain's root;
                    a new leaf (the subdomain_hint not registered) stays in default — the
                    trigger (Step 5) will create it and move the notes by a retro-move.
                    The move sets vector_status='pending' — the worker re-encodes the
                    vector into the new namespace's partition (the old vector goes
                    DELETE+INSERT).

                    Pool 6: the markup and the move — ONE transaction (one UPDATE, the
                    `namespace='default'` guard): the race with a client memory_update no
                    longer leaves the half-state "the markup written, the move by the old
                    estimate"; an operator move in flight is not overridden by the
                    background (rowcount). The auto-move target is computed before the
                    transaction — a `_auto_move_target` failure (including
                    NamespaceValidationError on a garbage hint) writes nothing to the DB
                    (the strict "a classification failure = not marked" semantics).

                    lsb-0011-01: the move mechanics itself is factored into the shared
                    `_apply_node_order` (the `default` sweep reuses it too) — only the
                    full-text input and the promotion trigger remain here; the
                    classification-after-summarization behavior does not change.
        """
        if self._classifier is None:
            return  # a test mode without a classifier
        if self._classify_blocked():
            # Release 3.9.0 (LSBN-S-9): the `classify` operation is paused —
            # the model is NOT called. The note stays unclassified (the work is
            # not lost: the sweep of the `nodes` job takes it after the pause).
            return
        known = self._namespaces.list_all()["namespaces"]
        try:
            result = self._classifier.classify(text, known)
        except ClassificationError as exc:
            # Release 3.9.0 (LSBN-S-7): the breaker key is the operation — the
            # `classify` operation in the `operation` mode, the `summary` slot
            # in the `legacy` mode.
            self._log_slot_failure(
                SUMMARY_JOB,
                self._operation_slot("classify"),
                outcome_of(exc),
                event="classify_failed",
                message="classify: failed — note stays in default, retry on next update",
                note_id=note_id,
                detail=str(exc),
                meta=getattr(exc, "meta", None),
            )
            return
        self._log_slot_call(
            SUMMARY_JOB, self._operation_slot("classify"), result.meta, note_id=note_id
        )
        # Pool 6: the move — ONE shared point `_apply_node_order` (the markup +
        # namespace/vector_status in one UPDATE, the guard inside); a failure
        # of computing the target (including NamespaceValidationError on a garbage hint)
        # writes nothing to the DB.
        outcome, _reason, target = self._apply_node_order(note_id, "default", result)
        # The log — only on an ACTUAL move (rowcount+condition).
        if outcome == "moved":
            logging.getLogger("app").info(
                "classify: default note auto-moved into existing node",
                extra={
                    "event": "classified_moved",
                    "job": SUMMARY_JOB,
                    "note_id": note_id,
                    "namespace": target,
                    "confidence": result.confidence,
                },
            )
        # The domain trigger (Phase 10, Step 5): the markup could push a hint group
        # over the threshold — run the promotion pipeline (auto-create/merge).
        self._run_promotion()

    def _run_promotion(self, job: str = SUMMARY_JOB) -> None:
        """The domain trigger (Step 5) after a default note's classification.

                    Called by the grooming (the summary slot, `job=summary`) and the
                    `nodes` sweep before sweeping `default` (`job=nodes`, lsb-0011-01):
                    promotion goes first so a note does not "move" into a node that will
                    appear in the same run. The job name in the journal — a parameter (the
                    events are not renamed; only the value of the mandatory `job` field
                    changes).

                    Trigger failures do not crash the worker: this is an enrichment stage,
                    not the data pipeline — summarization/vectorization matter more than
                    structural automation. The expected describer/judge failures are
                    handled inside PromotionService (the candidate stays without a
                    verdict, NFR-3); here EVERYTHING else is caught (including bugs) — a
                    warning with traceback into the logs, the queue loops live. A retry —
                    the next default-note classification or the next sweep run: groups are
                    not lost, they just reach the threshold later.

                    Release 3.9.0 (LSBN-S-8): in the `operation` mode the structural
                    judge has its OWN breaker — while it is closed the judge is not
                    called (the candidate waits), and its failure is recorded at that
                    breaker (the dedup judge's breaker is not touched). The `legacy`
                    mode is unchanged: the failures stay inside the trigger and the
                    whole slot group shares one breaker.

                    Release 3.9.0 (LSBN-S-10): the same for the `describe`
                    operation — its own breaker: while it is closed the
                    description model is not called (the candidate waits), and
                    its failure is recorded at that breaker (the neighbours —
                    `classify`, `summarize`, `judge` — are not touched). The
                    `legacy` mode is unchanged: the description is attempted as
                    before (the pause of the slot group stops the job).
        """
        if self._promoter is None:
            return  # a test mode without the trigger
        is_operation_mode = self._mode() == SLOT_MODEL_OPERATION
        try:
            if is_operation_mode:
                report = self._promoter.run(
                    judge_enabled=not self._operation_blocked("structure_judge"),
                    describe_enabled=not self._operation_blocked("describe"),
                )
            else:
                report = self._promoter.run()
        except Exception:
            logging.getLogger("app").warning(
                "promotion: run failed — trigger deferred to next classification",
                extra={
                    "event": "promotion_failed",
                    "job": job,
                    "reason": "run",
                },
                exc_info=True,
            )
            return
        if is_operation_mode:
            self._note_structure_judge_failures(job)
            self._note_describe_failures(job)
        if any(report.values()):
            # The summary — under one key: 'created' clashes with the LogRecord
            # attribute (the creation time) — extra does not accept it.
            logging.getLogger("app").info(
                "promotion: trigger run finished",
                extra={
                    "event": "promotion_run",
                    "job": job,
                    "report": report,
                },
            )

    def _note_structure_judge_failures(self, job: str) -> None:
        """The structural judge's failures — at the breaker of its operation.

                    Release 3.9.0 (LSBN-S-8): the failure of the structural judge
                    is recorded at the breaker of the `structure_judge` operation —
                    the dedup judge's breaker is NOT touched (the two operations
                    are independent: a dead model of one does not silence the
                    other). The candidate itself stays without a verdict (the
                    trigger keeps it) — the work is not lost, a retry is at the
                    next run.
        """
        for failure in getattr(self._promoter, "judge_failures", []):
            error = failure.get("error")
            self._log_slot_failure(
                job,
                self._operation_slot("structure_judge"),
                outcome_of(error) if isinstance(error, BaseException)
                else SlotOutcome.OFF_CONTRACT,
                event="promotion_judge_failed",
                message="promotion: structure judge failed — candidate kept, retry",
                detail=str(error) if error is not None else None,
                domain=failure.get("domain"),
                slug=failure.get("slug"),
            )

    def _note_describe_failures(self, job: str) -> None:
        """The description's failures — at the breaker of its operation.

                    Release 3.9.0 (LSBN-S-10): the failure of the description
                    generator is recorded at the breaker of the `describe`
                    operation — the neighbours (`classify`, `summarize`, `judge`,
                    `structure_judge`) are NOT touched (the operations are
                    independent: a dead model of one does not silence the
                    others). The candidate itself stays without a description
                    (the trigger keeps it) — the work is not lost, a retry is at
                    the next run.
        """
        for failure in getattr(self._promoter, "describer_failures", []):
            error = failure.get("error")
            self._log_slot_failure(
                job,
                self._operation_slot("describe"),
                outcome_of(error)
                if isinstance(error, BaseException)
                else SlotOutcome.OFF_CONTRACT,
                event="promotion_describe_failed",
                message="promotion: description generation failed — candidate kept, retry",
                detail=str(error) if error is not None else None,
                domain=failure.get("domain"),
                slug=failure.get("slug"),
            )

    def _auto_move_target(self, result) -> str | None:
        """The auto-move target node (existing nodes only, §5.7).

                    hint_path — the markup's full path (1..3 slugs); if it is not
                    registered (the model proposed a new node — the Step-5 trigger will
                    create it/move notes into it) — do not move. Null (generic) — do not
                    move. A non-path (classifier garbage) is validated inside by `exists`
                    — NamespaceValidationError, nothing is written to the DB (§5.7).
        """
        hint = result.hint_path
        if not hint:
            return None
        if not self._namespaces.exists(hint):
            return None
        return hint

    # --- the "node order" job (lsb-0011) --------------------------------------

    def process_nodes(self, budget: int | None = None) -> int:
        """The `nodes` job's run: promotion → tasks → the `default` sweep.

                    The full run order (FR-1.2/FR-2.3/FR-3.3, arch §3.2):
                    (1) promotion — `PromotionService.run()` (leaf creation/merging, the
                    retro-move); a failure does not crash the job;
                    (2) `reclass` tasks after stitching (lsb-0012) — the priority source:
                    the merged note may live in any node; the `default` sweep will not
                    find it;
                    (3) the fast sweep pool — with the budget remainder, a move WITHOUT a
                    model call;
                    (4) the classifier sweep pool — with the budget remainder, but no more
                    than `JOB_NODES_CLASSIFIER_BUDGET` classifier calls per run.
                    The `JOB_NODES_BATCH` budget (default 20) — SHARED over both sources
                    per run: no more parses than that (`moved` + `kept`); repeated runs
                    continue the backlog. A note waiting for a model or a summary is not
                    parsed and gets no marker — it lands in the next run.

                    The fast pool (arch §3.2): default notes with ready markup
                    (`hint_path` + `confidence` not below the auto-move threshold), not
                    yet parsed (`node_order_at IS NULL`), newest first. The node is
                    registered → a move (`moved`/`hint_exists`, the markup unchanged); the
                    node is not in the registry → `kept`/`hint_unknown` (promotion will
                    create the leaf — we do not create nodes here).

                    Each parsed note — a `node_order` event (`job='nodes'`,
                    `source='sweep'`/`'after_merge'`, `outcome`, `reason`, `target`,
                    `note_id`). Returns the parsed note count (0 sends the framework loop
                    to waiting for the interval or the `notify_nodes_pending` signal).
                    The summary slot breaker (the classifier walks into it, #39, §6.2) is
                    checked by the framework gate (`JobSpec.gate`, 3.4.1, LSBN-T-29) — the
                    manual `if` is gone: on a breaker no batch is taken at all, the loop
                    goes into back-off; the task and queue are not lost (a signal/timeout
                    will wake them).
        """
        limit = self._settings.job_nodes_batch if budget is None else budget
        # (1) Promotion first: a note does not "move" into a node that
        # will appear in the same run (FR-3.1). A failure does not crash the job (FR-3.2).
        self._run_promotion(NODES_JOB)
        # (2) After-stitching tasks — the priority source (FR-3.3).
        done = self._process_reclass_jobs(limit)
        # (3) The fast sweep pool; (4) the classifier one — with the budget remainder.
        done += self._sweep_fast_marks(limit - done)
        done += self._sweep_classifier(limit - done)
        return done

    def _process_reclass_jobs(self, limit: int) -> int:
        """Process a batch of `reclass` tasks (after stitching); their count.

                    The order — by the queue's `id` (fresh events do not wait behind old
                    ones). A task waiting for a ready summary stays pending and does not
                    count as progress: the loop goes to sleep on back-off — and when the
                    summary becomes ready, `notify_nodes_pending` from
                    `process_summary_pending` will wake the loop.
        """
        if limit <= 0:
            return 0
        done = 0
        for job in self._pending_jobs(NODES_JOB, NODES_RECLASS_KIND, limit):
            if self._process_reclass_job(int(job["note_id"])):
                self._mark_job_done(job["id"])
                done += 1
        return done

    def _process_reclass_job(self, note_id: int) -> bool:
        """Parse one after-stitching task; True — the task left the queue.

                    We read the note and decide (FR-2.2/FR-2.5 of lsb-0012, arch §3.5):

                    * no note (the operator deleted it) — release the task: nothing to
                      decide;
                    * the summary is not ready yet (`summary_status != 'ok'` or empty) —
                      the task STAYS pending and does not count as progress (it waits on
                      back-off: the summary loop will finish the summary and wake this
                      loop with `notify_nodes_pending`); no decision over the fallback
                      truncation, no parse marker — a `node_order`/`summary_pending`
                      event;
                    * over the ready summary — a classifier call on `title` + `summary`
                      (no two-stage with the full `text` — arch §3.4) and the shared move
                      point `_apply_node_order` with the guard over the node READ at
                      decision time: an operator/client move in flight is not overridden
                      by the background (`node_changed`). A `default` result does not
                      downgrade the note (`result_default`); the same node —
                      `same_node`.

                    Without a classifier (a test mode) no decision is made: the task stays
                    pending, the data is not spoiled.
        """
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT title, summary, summary_status, namespace FROM notes "
                "WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            ).fetchone()
        if row is None:
            return True  # no note — release the task
        summary = row["summary"] or ""
        if row["summary_status"] != "ok" or not summary.strip():
            self._log_node_order(
                note_id, "kept", "summary_pending", None, source="after_merge"
            )
            return False  # waiting for the summary: the task stays pending
        if self._classifier is None:
            return False  # a test mode without a classifier
        if self._classify_blocked():
            # Release 3.9.0 (LSBN-S-9): the `classify` operation is paused —
            # the model is NOT called, the task stays `pending` (the work is
            # not lost; no attempt is counted).
            return False
        known = self._namespaces.list_all()["namespaces"]
        text = f"{row['title']}\n{summary}" if row["title"] else summary
        try:
            result = self._classifier.classify(text, known)
        except ClassificationError as exc:
            # A classifier failure does not spoil the data (FR-3.1): the note stays
            # in its current node, no markup is written, the task — pending. #39:
            # the event carries the outcome type of the summary slot's failure.
            # Release 3.9.0 (LSBN-S-7): the breaker key is the operation.
            self._log_slot_failure(
                NODES_JOB,
                self._operation_slot("classify"),
                outcome_of(exc),
                event="node_order_failed",
                message="nodes: reclass failed — note stays in its node, retry by back-off",
                note_id=note_id,
                detail=str(exc),
                source="after_merge",
                meta=getattr(exc, "meta", None),
            )
            return False
        self._log_slot_call(
            NODES_JOB, self._operation_slot("classify"), result.meta, note_id=note_id
        )
        outcome, reason, target = self._apply_node_order(
            note_id,
            row["namespace"],
            result,
            current_namespace=row["namespace"],
            no_hint_reason="result_default",
        )
        self._log_node_order(note_id, outcome, reason, target, source="after_merge")
        return True

    def _sweep_fast_marks(self, budget: int) -> int:
        """The fast `default` sweep pool: a move by the ready markup, no model.

                    Notes with `hint_path` + `confidence` not below the auto-move
                    threshold, not yet parsed (`node_order_at IS NULL`), newest first. The
                    `node_order_at` marker is set on any outcome (anti-looping, §3.6): a
                    kept note does not eat the budget again until its text/title changes
                    (the marker reset — NoteService.update/merge_pair).
        """
        if budget <= 0:
            return 0
        threshold = self._settings.namespace_auto_move_min_confidence
        with session(self._settings) as conn:
            rows = conn.execute(_NODES_SWEEP_SELECT, (threshold, budget)).fetchall()
        done = 0
        for row in rows:
            note_id = int(row["id"])
            result = Classification(row["hint_path"], row["confidence"])
            outcome, reason, target = self._apply_node_order(
                note_id, "default", result
            )
            done += 1
            self._log_node_order(note_id, outcome, reason, target)
        return done

    def _sweep_classifier(self, budget: int) -> int:
        """The classifier sweep pool of `default`: markup through the model.

                    Notes without markup (`classified_at IS NULL`) with a READY non-empty
                    summary, newest first; the classifier's input — `title` + `summary`
                    (arch §3.4; the full `text` is not given). The classifier call count
                    per run is limited by the `JOB_NODES_CLASSIFIER_BUDGET` budget and the
                    shared budget's remainder. A classifier failure does not mark the note
                    — it stays a candidate of the next run (FR-3.2).
        """
        limit = min(budget, self._settings.job_nodes_classifier_budget)
        if limit <= 0 or self._classifier is None:
            return 0
        if self._classify_blocked():
            # Release 3.9.0 (LSBN-S-9): the `classify` operation is paused —
            # the model is NOT called, the notes stay candidates of the next
            # run (the work is not lost).
            return 0
        with session(self._settings) as conn:
            rows = conn.execute(_NODES_CLASSIFY_SELECT, (limit,)).fetchall()
        known = self._namespaces.list_all()["namespaces"]
        done = 0
        for row in rows:
            note_id = int(row["id"])
            summary = row["summary"] or ""
            text = f"{row['title']}\n{summary}" if row["title"] else summary
            try:
                result = self._classifier.classify(text, known)
            except ClassificationError as exc:
                # Release 3.9.0 (LSBN-S-7): the breaker key is the operation.
                self._log_slot_failure(
                    NODES_JOB,
                    self._operation_slot("classify"),
                    outcome_of(exc),
                    event="classify_failed",
                    message="nodes: classify failed — note stays in default, retry later",
                    note_id=note_id,
                    detail=str(exc),
                    meta=getattr(exc, "meta", None),
                )
                continue  # no marker: the note is a sweep candidate again
            self._log_slot_call(
                NODES_JOB, self._operation_slot("classify"), result.meta, note_id=note_id
            )
            outcome, reason, target = self._apply_node_order(
                note_id,
                "default",
                result,
                no_hint_reason="no_hint_classified",
            )
            done += 1
            self._log_node_order(note_id, outcome, reason, target)
        return done

    def _log_node_order(
        self,
        note_id: int,
        outcome: str,
        reason: str,
        target: str | None,
        source: str = "sweep",
    ) -> None:
        """The `nodes` job's `node_order` event (FR-4.2): a note parse outcome.

                    `reason` — from the arch §3.7 dictionary (`hint_exists`,
                    `hint_unknown`, `low_confidence`, `no_hint_classified`,
                    `summary_pending`, `same_node`, `result_default`, `node_changed`);
                    `source` — the decision source (`sweep` — the `default` sweep,
                    `after_merge` — the after-stitching task). The event and field names
                    are the existing ones — the job's observability does not change.
        """
        logging.getLogger("app").info(
            "nodes: decided note order",
            extra={
                "event": "node_order",
                "job": NODES_JOB,
                "source": source,
                "note_id": note_id,
                "outcome": outcome,
                "reason": reason,
                "target": target,
            },
        )

    def _nodes_queue_stat(self) -> dict:
        """The `nodes` queue snapshot for `/health.queues` (FR-2.2).

                    It counts EXACTLY the work the `process_nodes` run can take in this
                    run: (1) solvable `reclass` tasks after stitching — a note with a ready
                    summary, or the note is gone from the living (such a task is released
                    at once); a task waiting for the summary is NOT counted — the job does
                    not take it, and the note itself is visible in the `summary` queue;
                    (2) candidates of the fast `default` sweep pool; (3) candidates of the
                    classifier pool. Otherwise the counter would show work the job does not
                    have (a 3.1.0 acceptance FAIL, the 2026-09-19 gate decision).

                    `oldest_pending_sec` — the age of the oldest AMONG THE COUNTED sources
                    (tasks — `created_at`, candidates — `updated_at`); `null` — the queue
                    is empty. SQL only, no model calls; the snapshot works for a disabled
                    job too (`SQL` is assembled in `_nodes_queue_stat_sql`).
                    The pending → waiting move on a summary slot breaker (the classifier
                    walks into it, #39, arch §6.9) is done by the framework (`is_gated` →
                    `gated_queue_stat`, 3.4.1, LSBN-T-29).
        """
        self._ensure_job_table()
        with session(self._settings) as conn:
            row = conn.execute(
                _nodes_queue_stat_sql(),
                {
                    "threshold":
                        self._settings.namespace_auto_move_min_confidence
                },
            ).fetchone()
        return queue_snapshot(row["pending"], row["oldest_pending_sec"])

    def _nodes_queue_empty(self) -> bool:
        """Is the `nodes` queue empty (a cheap recheck, pool 6, lost wakeup).

                    Before `clear()+wait` the loop rechecks the queue itself, not only the
                    event: a `notify_nodes_pending` signal (an after-stitching task) that
                    arrived between an empty run and `clear()` would otherwise be erased —
                    a takeable task would wait out the interval/back-off (up to 15 min).
                    The source — exactly the same as `queue_stat` (arch §3.7): takeable
                    `reclass` tasks + candidates of both `default` sweep pools. A
                    note/task waiting for a summary does NOT make the queue non-empty: the
                    job will not take it, and the note itself is visible in the `summary`
                    queue — otherwise the loop would spin idly (busy-loop) instead of
                    back-off. The slot — via the single point of the gate condition
                    (`_slot_blocked`, 3.4.1, LSBN-T-29; the gate used to be smeared over
                    the snapshot): a blocked slot is an empty queue for the loop
                    (no39-05). SQL only, no model calls (FR-2.2).
        """
        return (
            self._slot_blocked(self._operation_slot("classify"))
            or int(self._nodes_queue_stat()["pending"] or 0) == 0
        )

    # --- the chunk queue (Phase 7) ---------------------------------------------

    async def process_pending_chunks(self, limit: int | None = None) -> int:
        """Vectorize one read batch of pending chunks (Phase 7).

                    A read batch — EMBEDDING_BATCH_SIZE × EMBEDDING_CONCURRENT_REQUESTS
                    chunks (an anti-join, oldest first); it is cut into sub-batches of
                    EMBEDDING_BATCH_SIZE, and the sub-batches are encoded in parallel —
                    no more than EMBEDDING_CONCURRENT_REQUESTS at a time (Semaphore +
                    asyncio.to_thread: encoding is blocking HTTP; the event loop is not
                    occupied).

                    A sub-batch failure (EmbeddingError) does not spoil the others
                    (NFR-3): successful sub-batches' vectors are written in short
                    transactions; failed chunks stay pending — caught up by the next run.
                    A full failure — 0, the worker waits back-off; `embedding_ok` is kept
                    by EmbeddingService itself (the single encoding point — as in the
                    other loops).

                    Reuse (brief §6): for a note with exactly one chunk ≤ CHUNK_SIZE the
                    chunk's vector = the full text's vector from notes_vec, without
                    calling the encoder. The memory_update race (ARCH §4.5) — like the
                    summary's: a write only when the chunk's (id, text, tokens) are
                    unchanged; chunks replaced in flight are skipped.

                    Returns the number of actually written vectors (reuse + successful
                    sub-batches; chunks replaced in flight are not counted).
        """
        batch = self._settings.embedding_batch_size
        drain = (
            limit
            if limit is not None
            else batch * self._settings.embedding_concurrent_requests
        )
        rows = await asyncio.to_thread(self._read_chunk_batch, drain)
        if not rows:
            return 0
        reused = await asyncio.to_thread(self._reuse_single_chunk_vectors, rows)
        remaining = [row for row in rows if row["id"] not in reused]
        batches = [remaining[i : i + batch] for i in range(0, len(remaining), batch)]
        results = await self._encode_batches(batches)
        written = await asyncio.to_thread(self._store_chunk_vectors, batches, results)
        return len(reused) + written

    def _read_chunk_batch(self, limit: int) -> list:
        """Read a batch of pending chunks (oldest first), a short read."""
        with session(self._settings) as conn:
            return chunks.pending_chunk_rows(conn, limit)

    def _reuse_single_chunk_vectors(self, rows: list) -> set[int]:
        """Copy the note's full vector into its single chunk ≤ CHUNK_SIZE.

                    Step 3 applies the reuse only with notes_vec ready at save time; if
                    the full vector was completed later (a failure at save — then the
                    notes-queue worker), the chunk stays pending. Re-encoding the same
                    text is pointless: with one chunk the text = the note's full text, the
                    vector is identical — copy without calling Ollama. The copy stays
                    consistent with notes_vec, which now includes the title (title+text —
                    lsb-0001 FR-1.1); chunk encoding over the bare text (decision D1)
                    does not change that.
        """
        by_note: dict[int, list] = {}
        for row in rows:
            by_note.setdefault(row["note_id"], []).append(row)
        reused: set[int] = set()
        with session(self._settings) as conn:
            for note_id, note_rows in by_note.items():
                if len(note_rows) != 1:
                    continue  # the batch note has several pending chunks
                count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM notes_chunks WHERE note_id = ?",
                        (note_id,),
                    ).fetchone()[0]
                )
                if count != 1 or note_rows[0]["tokens"] > self._settings.chunk_size:
                    continue
                full = vectors.get_vector(conn, note_id)
                if full is None:
                    continue  # no full vector — the chunk is encoded on par
                with transaction(conn):
                    if chunks.upsert_vector_if_exists(
                        conn,
                        note_rows[0]["id"],
                        full,
                        note_rows[0]["text"],
                        note_rows[0]["tokens"],
                        ns=note_rows[0]["namespace"],
                    ):
                        reused.add(note_rows[0]["id"])
        return reused

    async def _encode_batches(self, batches: list[list]) -> list:
        """Encode sub-batches in parallel; each call — an embed_texts batch.

                    No more than EMBEDDING_CONCURRENT_REQUESTS sub-batches at a time.
                    The return is aligned by sub-batches: a list of vectors or an
                    exception (EmbeddingError — a regular failure; others — an unexpected
                    crash).
        """
        semaphore = asyncio.Semaphore(self._settings.embedding_concurrent_requests)

        async def encode(batch: list) -> list[list[float]]:
            async with semaphore:
                return await asyncio.to_thread(
                    self._embedding.embed_texts, [row["text"] for row in batch]
                )

        return list(await asyncio.gather(*map(encode, batches), return_exceptions=True))

    def _store_chunk_vectors(self, batches: list[list], results: list) -> int:
        """Write the successful sub-batches' vectors (a batch — a short transaction)."""
        written = 0
        for batch, result in zip(batches, results):
            if isinstance(result, EmbeddingError):
                continue  # an encoding failure: the chunks stay pending (NFR-3)
            if isinstance(result, BaseException):
                raise result  # an unexpected failure — loud, as in the other loops
            with session(self._settings) as conn, transaction(conn):
                for row, vector in zip(batch, result):
                    if chunks.upsert_vector_if_exists(
                        conn,
                        row["id"],
                        vector,
                        row["text"],
                        row["tokens"],
                        ns=row["namespace"],
                    ):
                        written += 1
        return written


# --- the worker's job builders (the lsb-0014-02 framework) -----------------
#
# Registration instead of a loop copy (FR-1.1): each job is described by a `JobSpec`
# and added to the framework registry; the loop, back-off and iteration supervisor are shared
# (`app/services/jobs.py`). The shapes: "by interval" (`expiration`) and "on
# demand" with an event and a queue recheck (summary, judge, areas,
# links, `nodes` — the after-stitching task; the `nodes` recheck since 2026-09-20
# and `embedding` since LSBN-T-30 — a fresh note's signal is not lost); the
# "interval + event" shape without a recheck is gone from the registry. The intervals
# of the existing loops —
# as before (FR-1.5):
# PENDING_RETRY_SEC for the queues and a fixed 300 s for cleanup (this brief introduces no
# new env). `queue_stat` — the snapshot of one's own queue for
# `/health.queues` (lsb-0014-03, FR-2.2): `expiration` has no queue (`None`).
# The slot gate (3.4.1, LSBN-T-29): `gate` (for `run_loop`, with a probe knock)
# and `is_gated` (for snapshots, no models) pass `worker._slot_gate`/
# `worker._slot_blocked` — the breaker check lives in the framework; no manual `if`s in
# job bodies anymore.


def build_embedding_job(worker: BackgroundWorker, settings: Settings) -> JobSpec:
    """The `embedding` job: the notes queue + the chunk queue (one job).

                The shape "by interval + the `embedding` event + a queue recheck"
                (gate 2026-09-19; the recheck — LSBN-T-30): the `PENDING_RETRY_SEC`
                interval with back-off stays the safety net, and the
                `notify_embedding_pending` signal (a note written/updated/merged — it
                returned to `vector_status='pending'`) wakes the loop at once: a fresh
                note is encoded immediately, not after the grown back-off (up to
                15 min). An empty run — `worker_jobs` hygiene (the `idle_hook`,
                pool 5) and a PENDING_RETRY_SEC pause with back-off.

                The queue recheck (`queue_empty` → `_embedding_queue_empty`) closes
                the lost-wakeup race (pool 6): a signal fired BETWEEN an empty
                `process()` and the event's `clear()` would be erased — the loop would
                sleep the current interval, and on an idle system the back-off grows
                to 15 min, so a fresh note would wait up to 15 minutes for its vector
                with live models (the direct priority "a fresh note is encoded
                immediately"; the gate 2026-09-19 precedent). Why the recheck is safe
                now (that is the design): in the framework, work the loop cannot take
                is NOT progress (arch §3.2, no39-05; the live case of 2026-10-12,
                ≈11 calls/s, is closed) — after the recheck one run on a down slot
                returns 0, and the loop goes into `queue_waiting` + a back-off sleep;
                no idle storm. Since LSBN-T-28 a batch is released at once on
                wait-class failures — the run does not hang. "Takeable work" — SQL/
                memory only (pending vectorization notes + pending chunks, no models);
                on an embedding-slot breaker (`_slot_blocked`) the recheck counts the
                queue empty — the loop sleeps, no models called (no39-05). The task
                was never lost even before: it lives in the note's status, and
                `queue_waiting` events — going into a wait with a non-empty queue —
                are visible in the journal (FR-2.3).

                The slot gate (3.4.1, LSBN-T-29): `gate` — `_slot_gate` (with a probe
                knock after the pause expires), `is_gated` — `_slot_blocked`
                (snapshots without models).
    """
    return JobSpec(
        name=EMBEDDING_JOB,
        queue="vector",
        interval_sec=settings.pending_retry_sec,
        batch=None,
        enabled=True,
        process=worker._process_embedding,
        queue_empty=worker._embedding_queue_empty,
        wait_event=worker._embedding_event,
        idle_hook=worker._purge_done_jobs,
        queue_stat=worker._vector_queue_stat,
        slot=worker._operation_slot("embed"),
        gate=lambda: worker._slot_gate(worker._operation_slot("embed"), EMBEDDING_JOB),
        is_gated=lambda: worker._slot_blocked(worker._operation_slot("embed")),
    )


def build_summary_job(
    worker: BackgroundWorker, settings: Settings
) -> JobSpec | None:
    """The `summary` job: title-gen → summarize → merge, the `summary` signal.

                The slot gate (3.4.1, LSBN-T-29): `gate` — `_slot_gate` (with a probe
                knock), `is_gated` — `_slot_blocked` (snapshots without models).
                Without a summarizer (the Phase-3 test mode) the job is not
                applicable — the builder returns None and it does not enter the
                registry (the loop did not exist before either).

                The `legacy` mode only (release 3.9.0, LSBN-S-6): in the
                `operation` mode the cycle is cut into three jobs — the
                `title`, `summarize` and `merge` ones (their own builders
                below); the combined job does not exist there.
    """
    if worker._mode() == SLOT_MODEL_OPERATION:
        return None  # the split jobs take its place (LSBN-S-6)
    if worker._summarizer is None:
        return None
    return JobSpec(
        name=SUMMARY_JOB,
        queue="summary",
        interval_sec=settings.pending_retry_sec,
        batch=None,
        enabled=True,
        process=worker._process_summary,
        queue_empty=worker._summary_queue_empty,
        wait_event=worker._summary_event,
        idle_hook=None,
        queue_stat=worker._summary_queue_stat,
        slot=SLOT_SUMMARY,
        gate=lambda: worker._slot_gate(SLOT_SUMMARY, SUMMARY_JOB),
        is_gated=lambda: worker._slot_blocked(SLOT_SUMMARY),
    )


def build_title_job(
    worker: BackgroundWorker, settings: Settings
) -> JobSpec | None:
    """The `title` job: the title generation, the `title` signal.

                One of the three jobs of the split summary cycle (release 3.9.0,
                LSBN-S-6): its own queue (the notes without a title), its own
                signal, its own breaker (the `title` operation). The gate —
                `_slot_gate` with the operation's breaker (with a probe knock),
                `is_gated` — the same condition without models.

                The `operation` mode only: in the `legacy` mode the whole cycle
                is the `summary` job (as before) — the builder returns None and
                the job does not enter the registry.
    """
    if worker._mode() != SLOT_MODEL_OPERATION:
        return None
    if worker._summarizer is None:
        return None
    return JobSpec(
        name=TITLE_JOB,
        queue=TITLE_JOB,
        interval_sec=settings.pending_retry_sec,
        batch=None,
        enabled=True,
        process=worker._process_title,
        queue_empty=worker._title_queue_empty,
        wait_event=worker._title_event,
        idle_hook=None,
        queue_stat=worker._title_queue_stat,
        slot=worker._operation_slot("title"),
        gate=lambda: worker._slot_gate(worker._operation_slot("title"), TITLE_JOB),
        is_gated=lambda: worker._operation_blocked("title"),
    )


def build_summarize_job(
    worker: BackgroundWorker, settings: Settings
) -> JobSpec | None:
    """The `summarize` job: the pending summarizations, the `summary` signal.

                One of the three jobs of the split summary cycle (release 3.9.0,
                LSBN-S-6): its own queue (`summary_status='pending'`), its own
                signal, its own breaker (the `summarize` operation).

                The `operation` mode only — see `build_title_job`.
    """
    if worker._mode() != SLOT_MODEL_OPERATION:
        return None
    if worker._summarizer is None:
        return None
    return JobSpec(
        name=SUMMARIZE_JOB,
        queue=SUMMARIZE_JOB,
        interval_sec=settings.pending_retry_sec,
        batch=None,
        enabled=True,
        process=worker._process_summarize,
        queue_empty=worker._summarize_queue_empty,
        wait_event=worker._summary_event,
        idle_hook=None,
        queue_stat=worker._summarize_queue_stat,
        slot=worker._operation_slot("summarize"),
        gate=lambda: worker._slot_gate(
            worker._operation_slot("summarize"), SUMMARIZE_JOB
        ),
        is_gated=lambda: worker._operation_blocked("summarize"),
    )


def build_merge_job(worker: BackgroundWorker, settings: Settings) -> JobSpec | None:
    """The `merge` job: the pending merge jobs, the `merge` signal.

                One of the three jobs of the split summary cycle (release 3.9.0,
                LSBN-S-6): its own queue (the merge jobs of `worker_jobs`), its
                own signal (set at the merge job's creation), its own breaker
                (the `merge` operation).

                The `operation` mode only — see `build_title_job`.
    """
    if worker._mode() != SLOT_MODEL_OPERATION:
        return None
    if worker._summarizer is None:
        return None
    return JobSpec(
        name=MERGE_JOB,
        queue=MERGE_JOB,
        interval_sec=settings.pending_retry_sec,
        batch=None,
        enabled=True,
        process=worker._process_merge,
        queue_empty=worker._merge_queue_empty,
        wait_event=worker._merge_event,
        idle_hook=None,
        queue_stat=worker._merge_queue_stat,
        slot=worker._operation_slot("merge"),
        gate=lambda: worker._slot_gate(worker._operation_slot("merge"), MERGE_JOB),
        is_gated=lambda: worker._operation_blocked("merge"),
    )


def build_judge_job(worker: BackgroundWorker, settings: Settings) -> JobSpec:
    """The `judge` job: a batch of judge jobs (dedup), the `judge` signal.

                The operator brake (LSBN-D-14, 3.6.2): `DEDUP_ENABLED=false` does
                not start the job — the registry keeps its queue, so the judge
                backlog stays visible in `/health.queues` and is worked off after
                the flag is back on (the `JOB_NODES_ENABLED` canon). The judge tasks
                keep being created on vectorization — the backlog is honest, not
                hidden.

                The slot gate (3.4.1, LSBN-T-29): `gate` — `_slot_gate` (with a probe
                knock), `is_gated` — `_slot_blocked` (snapshots without models).

                Release 3.9.0 (LSBN-S-8): the judge operation goes through the
                operation API (`_operation_slot`/`_operation_blocked`): the gate
                and the blocked check are read from the breaker of the
                `judge` operation — in the `operation` mode its own breaker
                (the dedup judge and the structural judge do not share one),
                in the `legacy` mode the `judge` slot group (as before).
    """
    return JobSpec(
        name=JUDGE_JOB,
        queue="judge",
        interval_sec=settings.pending_retry_sec,
        batch=None,
        # LSBN-D-14 (3.6.2): the operator brake — false does not start the job, its
        # queue stays visible in /health (the nodes canon). The judge tasks keep
        # being created on vectorization: the backlog is honest and drains after
        # the flag is turned back on.
        enabled=settings.dedup_enabled,
        process=worker._process_judge,
        queue_empty=worker._judge_queue_empty,
        wait_event=worker._judge_event,
        idle_hook=None,
        queue_stat=worker._judge_queue_stat,
        slot=worker._operation_slot("judge"),
        gate=lambda: worker._slot_gate(worker._operation_slot("judge"), JUDGE_JOB),
        is_gated=lambda: worker._operation_blocked("judge"),
    )


def build_areas_job(worker: BackgroundWorker, settings: Settings) -> JobSpec:
    """The `areas` job: area records' vectors (substrate 3.0.0), the `areas` signal.

                The slot — `embedding` (#39, arch §6.9; LSBN-T-29): areas are encoded
                by the same slot as notes (the `_areas_queue_empty` precedent) — now
                this is declared both in `JobSpec.slot` and in the gate: `gate` — the
                embedding slot's `_slot_gate` (with a probe knock), `is_gated` —
                `_slot_blocked` (snapshots without models). Before, no slot was
                declared (None) — on a breaker the job kept poking the dead slot.
    """
    return JobSpec(
        name=AREAS_JOB,
        queue="areas",
        interval_sec=settings.pending_retry_sec,
        batch=None,
        enabled=True,
        process=worker._process_areas,
        queue_empty=worker._areas_queue_empty,
        wait_event=worker._areas_event,
        idle_hook=None,
        queue_stat=worker._areas_queue_stat,
        slot=worker._operation_slot("embed"),
        gate=lambda: worker._slot_gate(worker._operation_slot("embed"), AREAS_JOB),
        is_gated=lambda: worker._slot_blocked(worker._operation_slot("embed")),
    )


def build_expiration_job(worker: BackgroundWorker, settings: Settings) -> JobSpec:
    """The `expiration` job: expired-notes cleanup (lsb-0004-02, stage 4).

                No queue (cleanup is not a pending state) and no signal: the job is
                described by the "by interval" shape, and its back-off state is marked
                `fixed=True` (created in `__init__`) — the pause is always 300 s,
                without back-off (the behavior preserved verbatim, FR-1.5).
    """
    return JobSpec(
        name=EXPIRATION_JOB,
        queue=None,
        interval_sec=EXPIRATION_CLEANUP_INTERVAL_SEC,
        batch=None,
        enabled=True,
        process=worker._process_expiration,
        queue_empty=None,
        wait_event=None,
        idle_hook=None,
        queue_stat=None,
        slot=None,
    )


def build_nodes_job(worker: BackgroundWorker, settings: Settings) -> JobSpec:
    """The `nodes` job: the `default` sweep and reclassification after stitching (lsb-0011).

                The shape "by interval + the `nodes` event + a queue recheck"
                (arch §3.1): an empty run — a sleep on `JOB_NODES_INTERVAL_SEC`; the
                `notify_nodes_pending` signal (an after-stitching task and the merged
                note's ready summary, lsb-0012) wakes the loop at once, and
                `queue_empty` closes the lost-wakeup race (pool 6): before
                `clear()+wait` the queue is rechecked (see `_nodes_queue_empty`). The
                run — promotion, `reclass` tasks, then the `default` sweep (the fast
                pool without a model + the classifier pool within
                `JOB_NODES_CLASSIFIER_BUDGET`). `JOB_NODES_ENABLED=false` does not
                start the job, but the queue stays visible in `/health` (the registry
                keeps it). Synchronous SQL/mechanics go into a thread — the event loop
                is not occupied.

                The recheck (`queue_empty`) counts as work exactly what the job can
                take now (a takeable `reclass` task or a pool candidate) and makes the
                loop continue the run without sleep; a note/task waiting for a model
                or a summary is not work, and the loop goes to sleep on back-off (no
                busy-loop, arch §3.2, "a deferred task is not progress"); a blocked
                summary slot (`_slot_blocked`) also makes the queue empty for the loop
                (no39-05).

                The slot gate (3.4.1, LSBN-T-29): `gate` — the summary slot's
                `_slot_gate` (with a probe knock), `is_gated` — `_slot_blocked`
                (snapshots without models).
    """
    batch = settings.job_nodes_batch

    async def process() -> int:
        # #41 (§13.4): the job body carries the `job:nodes` source — a `namespace` move
        # writes a version with that mark (FR-4.4).
        with history.bind_origin(f"job:{NODES_JOB}"):
            return await asyncio.to_thread(worker.process_nodes, batch)

    return JobSpec(
        name=NODES_JOB,
        queue=NODES_JOB,
        interval_sec=settings.job_nodes_interval_sec,
        batch=batch,
        enabled=settings.job_nodes_enabled,
        process=process,
        queue_empty=worker._nodes_queue_empty,
        wait_event=worker._nodes_event,
        idle_hook=None,
        queue_stat=worker._nodes_queue_stat,
        slot=worker._operation_slot("classify"),
        gate=lambda: worker._slot_gate(worker._operation_slot("classify"), NODES_JOB),
        is_gated=lambda: worker._slot_blocked(worker._operation_slot("classify")),
    )


# Registering the worker's loops in the framework registry (FR-1.1): the framework assembles jobs
# from `JOB_BUILDERS`; one's own module appends its builders after their definitions
# (the import is one-way — worker → jobs, no cycle). The order — as the loops had:
# embedding, summary, judge, areas, expiration, nodes. Release 3.9.0 (LSBN-S-6): the
# three split jobs of the summary cycle stand where the `summary` job stands — in the
# `legacy` mode they do not enter the registry (the builders return None), so the
# registry of the default mode is unchanged.
jobs.JOB_BUILDERS += (
    build_embedding_job,
    build_summary_job,
    build_title_job,
    build_summarize_job,
    build_merge_job,
    build_judge_job,
    build_areas_job,
    build_expiration_job,
    build_nodes_job,
)
