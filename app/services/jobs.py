"""Background job framework (lsb-0014, release 3.1.0): registry, single loop, journal.

    Task #48: a job is described by a `JobSpec` structure and served by one
    `run_loop` — a new job is added **by registration, not by copying the loop**.
    The existing `BackgroundWorker` loops moved onto this framework (brief 4):
    the worker does not duplicate loop bodies — it describes its jobs with
    `build_*_job` builders and registers them in `JOB_BUILDERS` (the same extension
    point assembles `build_job_specs(worker, settings)`).

    The reliability contract (REQUIREMENTS §5.2, job-framework arch §3.3) is not
    reinvented here but carried over as is: back-off `PENDING_RETRY_SEC` (30 s)
    → ×2 → `MAX_INTERVAL_SEC` (15 min) with a reset on progress
    (`next_interval`/`MAX_INTERVAL_SEC` live in this module — the framework owns
    the contract, the worker reuses the same names); a loop-iteration exception
    does not kill the loop (a `loop_iteration_failed` event with the mandatory
    `job` field). The schedule shape is read from `JobSpec` and comes in three
    forms: "by interval" (`wait_event is None`), "by interval + event" (`wait_event`
    without `queue_empty`: the interval remains the safety net, the event wakes the
    loop earlier) and "on demand" (`wait_event` + `queue_empty`: after `clear()`
    the queue is rechecked — pool 6, lost wakeup; this is how `summary`, `judge`,
    `areas`, `links` and `nodes` (rechecked since 2026-09-20: a signal after
    stitching is not lost) work, and since 3.4.1 LSBN-T-30 also `embedding` (a
    fresh pending note, gate 2026-09-19): a signal fired between an empty run and
    `clear()` would otherwise be erased — takeable work would wait out the grown
    back-off up to 15 min; the recheck does not cause storms with unavailable
    models — a failure is not progress, arch §3.2).

    A job with no queue and no event (a periodic sweep, `expiration`) is served by
    a **fixed** interval: `backoff_state.fixed=True` — the pause always equals
    `interval_sec`, the back-off does not grow (FR-1.5 — the behavior is preserved
    verbatim).

    The key rule of deferred tasks (arch §3.2): `process()` returns the number of
    **actually processed** tasks. A task waiting for a model or a ready summary
    stays pending and does not count as progress — otherwise the loop would spin
    idly without back-off.

    Queue observability (lsb-0014-03, FR-2.2/FR-2.3): the queue is described by the
    job itself (`JobSpec.queue_stat` — a `{"pending", "oldest_pending_sec"}`
    snapshot; the age — over pending+waiting, 3.3.1);
    the framework only assembles the object over the registry and writes a
    `queue_waiting` event when the loop goes to sleep with a non-empty queue —
    "there is work, but it is not being done" is visible in the journal, not only
    in `/health`.

    The slot gate (3.4.1, LSBN-T-29; arch §6.2/§6.7): the slot-breaker check is a
    SINGLE POINT in the framework, not manual `if`s in job bodies. A job declares
    two predicates: `gate` ("may the batch be taken", for `run_loop`, WITH a side
    effect — a probe knock after the pause expires) and `is_gated` ("the slot is
    blocked right now", for queue snapshots — memory/SQL only, no model calls).
    Gate-0 ("the batch was not taken") is not loop progress (no39-05): the regular
    `run_loop` paths produce `queue_waiting` + a back-off sleep.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # annotations only: the framework has no runtime dependencies
    from app.config import Settings
    from app.services.worker import BackgroundWorker

# The back-off ceiling (REQUIREMENTS §5.3 "max 15 min"), not configurable via env.
MAX_INTERVAL_SEC = 15 * 60


def next_interval(current: float, start: int) -> float:
    """Back-off step: the interval doubles, the ceiling is 15 minutes (§3.4)."""
    return min(max(current * 2.0, float(start)), float(MAX_INTERVAL_SEC))


def queue_snapshot(
    pending: int | None,
    oldest_pending_sec: int | None,
    *,
    waiting: int = 0,
    attention: int = 0,
) -> dict[str, int | None]:
    """A queue snapshot for `/health.queues` (FR-2.2, #39 arch §6.9/§7.2).

            One shape for all jobs: `{"pending": int, "oldest_pending_sec":
            int | null, "waiting": int, "attention": int}`. The age is seconds computed
            in SQL (`now - MIN(ts)`), so the caller already provides it as a number;
            3.3.1 (LSBN-S-3): the age is over pending+waiting (all work waiting to be
            done), `null` — both categories empty (or the age was not passed by the
            caller). Protection against clock skew and an unexpected NULL on a
            non-empty queue — `max(0, ...)`/`null`.

            `waiting`/`attention` — the backward-compatible #39 extension: `pending` —
            work the loop can take **now**; `waiting` — deferred by the slot or by a
            dependent slot's readiness (model waiting is visible without polling the
            journal); `attention` — a subset of `waiting` waiting for a human
            (no39-02). The old `queue_snapshot(n, age)` call remains valid: the
            counters are simply zero.
    """
    count = int(pending or 0)
    # 3.3.1 (LSBN-S-3): on a slot breaker the work moves pending → waiting
    # (same age) — zero it only when the work waiting to be done
    # is gone entirely (pending == 0 AND waiting == 0) or was not passed.
    if (count == 0 and int(waiting or 0) == 0) or oldest_pending_sec is None:
        oldest = None
    else:
        oldest = max(0, int(oldest_pending_sec))
    return {
        "pending": count,
        "oldest_pending_sec": oldest,
        "waiting": int(waiting or 0),
        "attention": int(attention or 0),
    }


def gated_queue_stat(stat: dict) -> dict:
    """A queue snapshot with the slot unavailable: pending moves to waiting (#39).

            The body of the former `worker._gated_queue_stat`, the gated branch (3.4.1,
            LSBN-T-29): while the slot is blocked the loop takes no batch, so all
            current work moves from `pending` to `waiting` (`pending=0`), while
            `oldest_pending_sec`/`attention` are preserved (3.3.1: the age is zeroed
            only when pending+waiting are both empty — the preservation is factual):
            the waiting is visible in `/health.queues` and in the `queue_waiting`
            event, but the loop does not spin idly. Applied by the caller ONLY when
            `spec.is_gated() is True` (see the worker's `queues_health` and
            `_log_queue_waiting`) — with a healthy slot the job's snapshot is returned
            as is (the snapshot shape is the job's own concern, e.g. the two-key one at
            `migration`).
    """
    pending = int(stat.get("pending", 0) or 0)
    waiting = int(stat.get("waiting", 0) or 0)
    attention = int(stat.get("attention", 0) or 0)
    return queue_snapshot(
        0,
        stat.get("oldest_pending_sec"),
        waiting=waiting + pending,
        attention=attention,
    )


@dataclass(frozen=True)
class JobSpec:
    """A background job description (FR-1.1): registration instead of a loop copy.

            `name` — the job name in the journal (`job=...`); `queue` — the queue name
            for `/health.queues` (None — the queue is not observable); `interval_sec` —
            the starting wait pause (= the back-off start); `batch` — the batch size
            per run (if applicable); `enabled=False` — the job is not started at all
            (but does not vanish from the registry: a disabled job's queue must be
            visible); `process` — a run, returns the number of PROCESSED tasks;
            `queue_empty` — a cheap queue recheck (pool 6); `wait_event` — wakeup by
            event ("by interval + event"/"on demand": without `queue_empty` the
            interval remains the safety net; with it — a queue recheck after `clear()`);
            `idle_hook` — hygiene in the idle branch (e.g. `worker_jobs` retention);
            `queue_stat` — the snapshot of one's own queue
            `{"pending": int, "oldest_pending_sec": int | null}` for `/health.queues`
            (the age — over pending+waiting, 3.3.1);
            `slot` — the job's LLM slot (#39, arch §6.2): `embedding`/`summary`/`judge`
            or None (no breaker; `nodes` declares `summary` — its classifier walks into
            the summary slot);
            `pause_after_run` — the slot delimiter (3.4.1, LSBN-T-27): after a
            PRODUCTIVE run the loop sleeps `interval_sec` before the next batch, even
            if the queue is not empty (if empty — the pause is skipped, the job simply
            falls asleep on an empty run). For migration splitting: a batch is done —
            the generation slot is yielded; without the flag the loop would drain the
            queue batch after batch without a pause;
            `gate` — the slot gate for `run_loop` (3.4.1, LSBN-T-29; arch §6.2/§6.7):
            "may the batch be taken" (True — yes). The check comes WITH a side effect:
            when the breaker pause has expired, one probe knock (`probe`) is made and
            the breaker may be lifted. The single point of the slot-breaker check: the
            manual `if not gate: return 0` in job bodies are gone — the framework wraps
            both `process()` calls (the main loop and the "on demand" recheck), and
            gate-0 is not loop progress (no39-05) → `queue_waiting` + a back-off sleep;
            `is_gated` — "the slot is blocked right now" (True — blocked) for queue
            snapshots (`queues_health`, `queue_waiting`): WITHOUT side effects —
            memory/SQL only, never calls models (the `/health.queues` contract:
            observability does not burn calls and does not move pauses).
    """

    name: str
    queue: str | None
    interval_sec: int
    batch: int | None
    enabled: bool
    process: Callable[[], Awaitable[int]]
    queue_empty: Callable[[], bool] | None
    wait_event: asyncio.Event | None
    idle_hook: Callable[[], None] | None
    queue_stat: Callable[[], dict] | None
    # #39 (3.2.0): the job slot — the default value keeps builders
    # without a slot (links/areas/expiration) valid, backward compatibility.
    slot: str | None = None
    # 3.4.1 (LSBN-T-27): a pause AFTER a productive run (the slot
    # delimiter of splitting) — the default False keeps the former pauseless draining.
    pause_after_run: bool = False
    # 3.4.1 (LSBN-T-29): the slot gate — the single point of the breaker check:
    # `gate` in `run_loop` (with a probe knock), `is_gated` in queue snapshots
    # (no models). The None defaults keep former builders valid
    # (slotless jobs — links/expiration/terms_normalize — have no gate),
    # backward compatibility across positional/keyword constructions.
    gate: Callable[[], bool] | None = None
    is_gated: Callable[[], bool] | None = None


class BackoffState:
    """Mutable back-off state of the loop: the current wait interval.

            Kept by the caller (the worker), not inside `run_loop`: the interval
            survives iterations and resets to `spec.interval_sec` on progress, so it is
            visible in diagnostics (`worker.interval` in tests).

            `fixed=True` — a schedule WITHOUT back-off: interval growth is off, the
            pause always equals the job's starting interval (`expiration`, arch §3.6:
            300 s "fixed, as now") — a periodic sweep does not depend on external
            services, and an iteration failure neither speeds up nor slows down the
            schedule (FR-1.5).
    """

    __slots__ = ("interval", "fixed")

    def __init__(self, start_sec: int, *, fixed: bool = False) -> None:
        self.interval = float(max(start_sec, 0))
        self.fixed = fixed

    def grow(self, start_sec: int) -> None:
        """The next wait interval: back-off up to the ceiling, or fixed.

                    The single point of interval growth for `run_loop`: for regular jobs —
                    `next_interval` (doubling up to `MAX_INTERVAL_SEC`); for fixed-schedule
                    jobs — the starting interval unchanged.
        """
        self.interval = (
            float(max(start_sec, 0))
            if self.fixed
            else next_interval(self.interval, start_sec)
        )


def log_job(
    job: JobSpec,
    event: str,
    *,
    note_id: int | None = None,
    outcome: str | None = None,
    reason: str | None = None,
    target: str | None = None,
    **extra: Any,
) -> None:
    """Write a job event to the shared journal (FR-1.4): the mandatory `job` field.

            The format is one for all jobs: the event name + `job` and common fields as
            available — `note_id`, `outcome`, `reason`, `target` (None is not written:
            the field appears only where applicable) plus service fields from `extra`.
            The level is INFO; an unexpected iteration failure is logged by the loop
            itself (a warning with traceback, the `loop_iteration_failed` event — the
            same `event`/`job` fields).
    """
    fields: dict[str, Any] = {"event": event, "job": job.name}
    for key, value in (
        ("note_id", note_id),
        ("outcome", outcome),
        ("reason", reason),
        ("target", target),
    ):
        if value is not None:
            fields[key] = value
    fields.update(extra)
    logging.getLogger("app").info(event, extra=fields)


async def _call_hook(hook: Callable[[], Any]) -> Any:
    """Call `queue_empty`/`idle_hook`: a coroutine — directly, sync — in to_thread.

            The existing loops call queue checks and hygiene through
            `asyncio.to_thread` (synchronous SQL inside): the framework preserves that
            behavior — a sync hook goes to a thread and does not occupy the event loop;
            a coroutine is awaited as is.
    """
    if inspect.iscoroutinefunction(hook):
        return await hook()
    return await asyncio.to_thread(hook)


async def _log_queue_waiting(spec: JobSpec) -> None:
    """The `queue_waiting` event before the sleep with a non-empty own queue (FR-2.3).

            "There is work, but it is not being done" (a model unavailable, a task
            waiting) — the sticking is visible in the journal without polling
            `/health`. The snapshot is taken from the job itself (`queue_stat`, SQL
            only through a thread — like the other hooks); a job without a queue or
            with an empty queue stays silent. With a blocked slot (`spec.is_gated`,
            LSBN-T-29) the snapshot goes through `gated_queue_stat`: the event sees the
            factual state "work waits for the slot" (pending=0, waiting>0), not a raw
            snapshot as if the slot were healthy.
    """
    if spec.queue is None or spec.queue_stat is None:
        return
    stat = await _call_hook(spec.queue_stat)
    if spec.is_gated is not None and spec.is_gated():
        stat = gated_queue_stat(stat)
    pending = int(stat.get("pending", 0) or 0)
    waiting = int(stat.get("waiting", 0) or 0)
    # #39 (arch §6.9): waiting for a deferred task is visible work too, although
    # it does not drive the loop into a busy-loop (the `queue_empty` driver counts takeable work).
    if pending + waiting <= 0:
        return
    log_job(
        spec,
        "queue_waiting",
        queue=spec.queue,
        pending=pending,
        waiting=waiting,
        oldest_pending_sec=stat.get("oldest_pending_sec"),
    )


async def _gated_process(spec: JobSpec) -> int:
    """A batch run through the slot gate (3.4.1, LSBN-T-29; arch §6.2/§6.7).

            The single point of the slot-breaker check: if the job has a `gate` and it
            is closed (the breaker open, the probe knock failed) — the batch is NOT
            taken, `process()` is not called, no models burn, return 0. Then the
            framework's regular paths apply: 0 — "a job failure is not progress"
            (no39-05, the live case of 2026-10-12: ≈11 calls/s) → `queue_waiting` +
            a back-off sleep, no busy-loop. BOTH `process()` calls of the loop are
            wrapped: the main cycle and the "on demand" recheck after `clear()`
            (the `pause_after_run` branch makes no second run — nothing to wrap
            there). A job without a `gate` (no slot — links/expiration/terms_normalize)
            goes into the run directly.
    """
    if spec.gate is not None and not spec.gate():
        return 0
    return await spec.process()


async def run_loop(
    spec: JobSpec,
    stopping: Callable[[], bool],
    backoff_state: BackoffState,
) -> None:
    """The single job loop (arch §3.2): run → idle → wait with back-off.

            `stopping` — the soft-stop predicate (checked at the start of every
            iteration); `backoff_state` — the current wait interval, which the loop
            resets to `spec.interval_sec` on progress and grows on a timeout.

            A run returned > 0 processed tasks — the interval resets and the next batch
            goes immediately (the queue is drained); for a job with `pause_after_run`
            (the slot delimiter, 3.4.1) there is an `interval_sec` pause between
            batches until the queue empties. An empty run — `idle_hook`, then the
            wait: in the form with `wait_event` the queue (if `queue_empty` is set) is
            rechecked **after** the event's `clear()` (pool 6: work before clear() is
            visible to the select, after — it wakes the event), and the loop waits in
            `wait_for(event.wait(), timeout=interval)` — the event wakes earlier (the
            "by interval + event" form at `embedding`), the timeout grows the back-off;
            in the form without an event — `sleep(interval)`. The wait timeout is
            `next_interval()`. `CancelledError` is propagated (graceful stop); any
            other exception does not kill the loop — a warning with traceback, a pause,
            back-off.

            The schedule may be fixed (`backoff_state.fixed` — a periodic sweep without
            a queue, `expiration`, arch §3.6): the pause always equals the job's
            starting interval, the back-off does not grow, a run does not shift the
            schedule (FR-1.5 — the behavior is preserved verbatim).

            A disabled job (`enabled=False`) is not started at all: the registry keeps
            it (its queue is visible in `/health`), but there is nothing to serve.

            Before every wait a `queue_waiting` event is written if the own queue is
            not empty (FR-2.3): "there is work, but it is not being done" — in the
            journal.

            The slot gate (3.4.1, LSBN-T-29): both run calls — the main cycle and the
            "on demand" recheck — go through `_gated_process` (arch §6.2/§6.7). Closed
            gates give 0 without calling `process()`: a slot breaker is not loop
            progress (no39-05), the loop goes into `queue_waiting` + back-off; when the
            pause has expired the `gate` itself makes one probe knock and may lift the
            breaker (the batch goes in the same run).
    """
    if not spec.enabled:
        return
    while not stopping():
        try:
            processed = await _gated_process(spec)
            if backoff_state.fixed:
                # A periodic sweep with a fixed schedule: the pause is the
                # job interval, not the presence of work (nothing to drain).
                await _log_queue_waiting(spec)
                await asyncio.sleep(backoff_state.interval)
                continue
            if processed > 0:
                backoff_state.interval = float(spec.interval_sec)
                # The slot delimiter (3.4.1, LSBN-T-27): a batch is done —
                # a pause AFTER the run before the next one, even if the queue is
                # not empty yet (splitting does not monopolize the generation slot). An empty
                # queue after the run needs no pause: then the regular
                # empty run with the usual wait applies.
                if spec.pause_after_run and not (
                    spec.queue_empty is not None
                    and await _call_hook(spec.queue_empty)
                ):
                    await asyncio.sleep(spec.interval_sec)
                continue
            # An empty run: idle-branch hygiene (if the job has one).
            if spec.idle_hook is not None:
                await _call_hook(spec.idle_hook)
            if spec.wait_event is None:
                await _log_queue_waiting(spec)
                await asyncio.sleep(backoff_state.interval)
                backoff_state.grow(spec.interval_sec)
                continue
            # The "on demand" form: the queue recheck AFTER clear() (pool 6,
            # lost wakeup) — work that appeared before clear() is visible to the select,
            # so it gets ONE immediate run; work that appeared after —
            # wakes the already-cleared event, the signal is not erased.
            spec.wait_event.clear()
            if spec.queue_empty is not None and not await _call_hook(spec.queue_empty):
                processed = await _gated_process(spec)
                if processed > 0:
                    backoff_state.interval = float(spec.interval_sec)
                    continue
                # The run was empty: there is work in the queue but it could not be taken
                # (the slot refuses the task — it stays pending). A job
                # failure does not count as loop progress (#39, no39-05): otherwise
                # a sleepless retry would become an idle storm (the live
                # case of 2026-10-12: ≈11 calls/s).
                await _log_queue_waiting(spec)
                await asyncio.sleep(backoff_state.interval)
                backoff_state.grow(spec.interval_sec)
                continue
            await _log_queue_waiting(spec)
            try:
                await asyncio.wait_for(
                    spec.wait_event.wait(), timeout=backoff_state.interval
                )
            except asyncio.TimeoutError:
                backoff_state.grow(spec.interval_sec)
        except asyncio.CancelledError:
            raise  # loop cancellation (graceful stop) — do not swallow
        except Exception:
            # The iteration supervisor (pool 4): an unexpected failure does not kill
            # the coroutine — a warning with traceback, a pause and a retry on back-off.
            logging.getLogger("app").warning(
                "job loop iteration failed — loop continues",
                extra={"event": "loop_iteration_failed", "job": spec.name},
                exc_info=True,
            )
            await asyncio.sleep(backoff_state.interval)
            backoff_state.grow(spec.interval_sec)


# A job builder: describes its job and returns None when the job is not applicable
# in the current build (e.g. the summarization job without a summarizer).
JobBuilder = Callable[["BackgroundWorker", "Settings"], JobSpec | None]

# The builder registry is the framework's extension point (FR-1.6). A new job = a
# `build_job(worker, settings)` function in its own module + a line in this tuple.
# Disabled jobs do not vanish from the registry: their queue must be visible in
# `/health` (the degradation is visible, not silent) — the loop does not start such jobs.
JOB_BUILDERS: tuple[JobBuilder, ...] = ()


def build_job_specs(
    worker: BackgroundWorker, settings: Settings
) -> list[JobSpec]:
    """Assemble the job registry over the registered builders (FR-1.1).

            A builder that returned None (the job is not applicable in this build) does
            not enter the registry. Own builders are appended to the registry by the
            module owning the jobs: the existing worker loops register in `worker.py`
            (brief 4), new jobs — each in its own module (FR-1.6).
    """
    specs: list[JobSpec] = []
    for build in JOB_BUILDERS:
        spec = build(worker, settings)
        if spec is not None:
            specs.append(spec)
    return specs
