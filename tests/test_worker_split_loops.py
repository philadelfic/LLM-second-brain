"""The split of the combined summary cycle into three jobs (release 3.9.0, LSBN-S-6).

The worker's combined cycle `title → summarize → merge` lived in ONE job
(`summary`) with one event and one slot breaker. In the `operation` mode
(`SLOT_MODEL=operation`) the cycle is split into three jobs — `title`,
`summarize`, `merge` — each with its own queue, its own event, its own counter
and its own breaker (the design, B3, stage 2).

The checked contracts (the criteria §4 of the step):
1. in the `operation` mode the job registry holds three jobs with different
   events and different queues;
2. in the `legacy` mode the registry did NOT change (the snapshot was taken
   from the `dev` base before the edits);
3. a failure of the title does not block summarize (the pause belongs to its
   own operation only);
4. the merge signal reaches the merge loop (it is set on the job creation);
5. no work is lost: both loops take the note, the merge job does not vanish;
6. in `/health.queues` in the `operation` mode three records are visible;
7. in the `operation` mode the worker does not crash on the references to the
   breaker of the `summary` slot (the finding of the S-5 check, M1).

A unit test without the network: the models are fakes (`fakes.FixedSummarizer`
and its descendant with a title failure), the embedder is `HashEmbedder`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time

import pytest
from fakes import FixedSummarizer, HashEmbedder, clear_seeded_skills

from app.config import get_settings
from app.services.jobs import build_job_specs
from app.services.notes import NoteService
from app.services.summary import SummaryError
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session, transaction

DIM = 8

#: A snapshot of the job registry in the `legacy` mode, taken from the `dev`
#: base (LSBN-S-5, commit 4950d09) BEFORE this edit: (job name, its queue,
#: its slot). Criterion 2: the registry of the legacy mode did not change.
LEGACY_REGISTRY: tuple[tuple[str, str | None, str | None], ...] = (
    ("links", "links", None),
    ("migration", "migration", "summary"),
    ("embedding", "vector", "embedding"),
    ("summary", "summary", "summary"),
    ("judge", "judge", "judge"),
    ("areas", "areas", "embedding"),
    ("expiration", None, None),
    ("nodes", "nodes", "summary"),
)

#: The three jobs of the split cycle.
SPLIT_JOBS = ("title", "summarize", "merge")


class TitleFailingSummarizer(FixedSummarizer):
    """A summarizer: the title always fails, the summary works.

    The title failure is a `SummaryError` without an outcome: the outcome is
    read as `off_contract` (not transient) — the pause of the operation opens
    on the first such failure (with `SLOT_PAUSE_AFTER=1`).
    """

    def title(self, text: str):  # noqa: ANN201 — the signature of the fake
        self.calls.append(text)
        self.last_attempt_ok = False
        raise SummaryError("модель вернула не-заголовок")


def _make_settings(tmp_path, monkeypatch: pytest.MonkeyPatch, *, mode: str):
    """Test settings: own DB, the slot mode and the pause threshold in one failure."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    # LSBN-T-37 (3.6.3): the short-text rule is off — the test checks the
    # model path of the summary.
    monkeypatch.setenv("SUMMARY_SHORT_TEXT_CHARS", "0")
    # The pause threshold of one failure: a non-transient failure opens the
    # pause of the operation at once — otherwise only the third failure would
    # open it.
    monkeypatch.setenv("SLOT_PAUSE_AFTER", "1")
    if mode == "operation":
        monkeypatch.setenv("SLOT_MODEL", "operation")
    else:
        monkeypatch.delenv("SLOT_MODEL", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)
    return settings


@pytest.fixture
def legacy_settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """The default mode (`legacy`): the job registry — as before the split."""
    return _make_settings(tmp_path, monkeypatch, mode="legacy")


@pytest.fixture
def operation_settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """The `operation` mode: the cycle is split into three jobs."""
    return _make_settings(tmp_path, monkeypatch, mode="operation")


def make_worker(settings, summarizer=None) -> BackgroundWorker:
    """A worker with a summarizer (otherwise the jobs of the combined cycle do
    not apply)."""
    return BackgroundWorker(
        settings,
        HashEmbedder(DIM),
        summarizer if summarizer is not None else FixedSummarizer("Фикс-суммари."),
    )


def _registry(settings) -> list[tuple[str, str | None, str | None]]:
    """The job registry: (name, queue, slot) — in the registration order."""
    worker = make_worker(settings)
    specs = build_job_specs(worker, settings)
    return [(spec.name, spec.queue, spec.slot) for spec in specs]


def _specs(settings) -> dict:
    """The job registry by name (for the gate/event/queue checks)."""
    worker = make_worker(settings)
    return {spec.name: spec for spec in build_job_specs(worker, settings)}


def _pending_summaries(settings) -> int:
    """How many notes wait for a summary (`summary_status='pending'`)."""
    with session(settings) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM notes WHERE summary_status = 'pending' "
            "AND deleted_at IS NULL"
        ).fetchone()
    return int(row[0])


def _pending_merge_jobs(settings) -> int:
    """How many merge jobs wait in the queue.

    Release 3.9.0 (LSBN-S-14): in the `operation` mode the job is written with
    the operation name (`merge`), in `legacy` — with the slot group name
    (`summary`).
    """
    slot = "merge" if settings.slot_model == "operation" else "summary"
    with session(settings) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM worker_jobs WHERE slot = ? "
            "AND kind = 'merge' AND status = 'pending'",
            (slot,),
        ).fetchone()
    return int(row[0])


async def wait_until(predicate, timeout: float = 3.0) -> None:
    """Wait for a condition of the live loop (polling with yields)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("условие не наступило за отведённое время")


# --- criterion 1: in the operation mode — three jobs with their own event and queue


def test_operation_registry_has_three_split_jobs(operation_settings) -> None:
    """The `operation` registry: the three jobs of the split cycle, no combined one."""
    names = [name for name, _queue, _slot in _registry(operation_settings)]
    for name in SPLIT_JOBS:
        assert name in names
    assert "summary" not in names  # there is no combined job in this mode


def test_split_jobs_have_own_event_and_queue(operation_settings) -> None:
    """Each job has its OWN event and its own queue (a shared Event is not allowed).

    `jobs.py` does `spec.wait_event.clear()` before the queue check — a shared
    `asyncio.Event` would erase the wake-up of the neighbouring loop.
    """
    specs = _specs(operation_settings)
    events = [specs[name].wait_event for name in SPLIT_JOBS]
    queues = [specs[name].queue for name in SPLIT_JOBS]
    slots = [specs[name].slot for name in SPLIT_JOBS]

    assert all(event is not None for event in events)
    assert len({id(event) for event in events}) == 3  # different objects
    assert len(set(queues)) == 3
    assert len(set(slots)) == 3


def test_split_jobs_have_own_process_and_gate(operation_settings) -> None:
    """Each job has its own handler and its own gate (the breaker of the operation)."""
    specs = _specs(operation_settings)
    processes = [specs[name].process for name in SPLIT_JOBS]
    gates = [specs[name].gate for name in SPLIT_JOBS]

    assert len({id(process) for process in processes}) == 3
    assert len({id(gate) for gate in gates}) == 3
    assert specs["title"].slot == "title"
    assert specs["summarize"].slot == "summarize"
    assert specs["merge"].slot == "merge"


# --- criterion 2: in the legacy mode the registry did not change


def test_legacy_registry_unchanged(legacy_settings) -> None:
    """The legacy registry — exactly the one taken from the base before the edits."""
    assert tuple(_registry(legacy_settings)) == LEGACY_REGISTRY


def test_legacy_registry_has_no_split_jobs(legacy_settings) -> None:
    """In the legacy mode the registry holds no split jobs."""
    names = {name for name, _queue, _slot in _registry(legacy_settings)}
    for name in SPLIT_JOBS:
        assert name not in names
    assert "summary" in names


# --- criterion 3: a failure of the title does not block summarize


def test_title_failure_does_not_block_summarize(operation_settings) -> None:
    """A title failure: the pause belongs to the title only, the summary is processed.

    The title fails (`off_contract`), the summary works: the summarize queue
    is drained, and the pause (`paused_until`) belongs to the `title` operation
    only.
    """
    worker = make_worker(operation_settings, TitleFailingSummarizer())
    NoteService(operation_settings, HashEmbedder(DIM)).save(
        "заметка без заголовка: ждёт и заголовок, и выжимку"
    )
    assert _pending_summaries(operation_settings) == 1

    assert worker.process_title_pending() == 0  # the title was not written
    assert worker._breakers["title"].snapshot()["paused_until"] is not None
    assert worker._breakers["summarize"].snapshot()["paused_until"] is None

    # The gate of the summarize job is open, the gate of the title is closed
    # (the pause of the operation). The registry — of the same worker: the
    # pause lives in its memory.
    specs = {spec.name: spec for spec in build_job_specs(worker, operation_settings)}
    assert specs["summarize"].gate() is True
    assert specs["title"].gate() is False

    # The summary is processed: the summarize queue is drained.
    assert worker.process_summary_pending() == 1
    assert _pending_summaries(operation_settings) == 0


def test_title_failure_leaves_title_work_pending(operation_settings) -> None:
    """A title failure does not lose the work: the note waits for the next run."""
    worker = make_worker(operation_settings, TitleFailingSummarizer())
    NoteService(operation_settings, HashEmbedder(DIM)).save("заметка без заголовка")
    worker.process_title_pending()
    with session(operation_settings) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM notes WHERE title IS NULL "
            "AND deleted_at IS NULL"
        ).fetchone()
    assert int(row[0]) == 1


# --- criterion 4: the merge signal reaches the merge loop


def test_merge_job_creation_wakes_merge_loop(operation_settings) -> None:
    """The creation of a merge job wakes the merge loop (the signal is set on
    the creation)."""
    worker = make_worker(operation_settings)
    assert worker._merge_event.is_set() is False
    worker._create_job("summary", "merge", 1, payload=json.dumps({"older_id": 1}))
    assert worker._merge_event.is_set() is True


@pytest.mark.asyncio
async def test_merge_loop_takes_the_job(operation_settings) -> None:
    """The merge loop takes the job by its signal (without a common back-off)."""
    worker = make_worker(operation_settings)
    worker._create_job("summary", "merge", 1, payload=json.dumps({"older_id": 1}))
    assert _pending_merge_jobs(operation_settings) == 1

    task = asyncio.create_task(worker._run_merge())
    try:
        await wait_until(lambda: _pending_merge_jobs(operation_settings) == 0)
    finally:
        worker.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


# --- criterion 5: no work is lost


def test_note_is_taken_by_both_loops(operation_settings) -> None:
    """A note without a title and without a summary is taken by both loops (the
    predicates are independent: `title IS NULL` and `summary_status='pending'`)."""
    worker = make_worker(operation_settings)
    NoteService(operation_settings, HashEmbedder(DIM)).save(
        "заметка без заголовка и без выжимки"
    )

    assert worker.process_title_pending() == 1
    assert worker.process_summary_pending() == 1
    with session(operation_settings) as conn:
        row = conn.execute(
            "SELECT title, summary, summary_status FROM notes WHERE id = 1"
        ).fetchone()
    assert row["title"] is not None
    assert row["summary_status"] == "ok"


def test_merge_job_is_not_lost_by_the_split(operation_settings) -> None:
    """The merge job does not vanish at the split: the merge loop takes it."""
    worker = make_worker(operation_settings)
    notes = NoteService(operation_settings, HashEmbedder(DIM))
    notes.save("первая заметка про ежедневный бэкап", title="Бэкап")
    notes.save("вторая заметка про ежедневный бэкап", title="Бэкап вечером")
    worker._create_job(
        "summary", "merge", 2, payload=json.dumps({"older_id": 1, "cosine": 0.9})
    )

    assert worker._merge_queue_empty() is False
    assert worker.process_merge_pending() == 1
    assert worker._merge_queue_empty() is True
    assert _pending_merge_jobs(operation_settings) == 0


# --- criterion 6: /health.queues in the operation mode


def test_health_queues_show_three_split_queues(operation_settings) -> None:
    """The queue snapshot holds the three records of the split cycle.

    Release 3.9.0 (LSBN-S-15): the legacy key `summary` is BACK in this mode —
    the old consumers must not break (the key is the historical aggregate of
    the cycle, not the counter of an operation).
    """
    worker = make_worker(operation_settings)
    queues = worker.queues_health()
    for name in SPLIT_JOBS:
        assert name in queues
        assert queues[name]["pending"] == 0
    assert "summary" in queues  # LSBN-S-15: the old key is back


def test_health_queues_split_queues_count_their_own_work(operation_settings) -> None:
    """Each queue has its own work: the title and the summary are counted
    separately."""
    worker = make_worker(operation_settings)
    NoteService(operation_settings, HashEmbedder(DIM)).save(
        "заметка без заголовка: ждёт и заголовок, и выжимку"
    )
    queues = worker.queues_health()
    assert queues["title"]["pending"] == 1
    assert queues["summarize"]["pending"] == 1
    assert queues["merge"]["pending"] == 0


def test_legacy_health_queues_unchanged(legacy_settings) -> None:
    """In the legacy mode the queue of the combined cycle is one — as before
    the edits."""
    worker = make_worker(legacy_settings)
    queues = worker.queues_health()
    assert "summary" in queues
    for name in SPLIT_JOBS:
        assert name not in queues


# --- criterion 7: in the operation mode no KeyError on the summary slot breaker


def test_operation_mode_breaker_lookups_do_not_raise(operation_settings) -> None:
    """The references to the breaker of the `summary`/`embedding` slot do not
    crash the worker.

    The finding of the S-5 check (M1): in the `operation` mode the dictionary
    `_breakers` is keyed by the operations — a direct lookup by a slot name
    raised a `KeyError`.
    """
    worker = make_worker(operation_settings)
    NoteService(operation_settings, HashEmbedder(DIM)).save(
        "заметка без заголовка: очередь не пуста"
    )
    assert worker._summary_queue_empty() is False  # the combined predicate (legacy)
    assert worker._title_queue_empty() is False
    assert worker._summarize_queue_empty() is False
    assert worker._areas_queue_empty() is True


@pytest.mark.asyncio
async def test_operation_mode_worker_serves_queues(operation_settings, caplog) -> None:
    """In the `operation` mode the worker starts and drains the queue.

    A failure is checked too: an exception in an iteration of the loop is
    written as the `loop_iteration_failed` event (the iteration does not kill
    the loop) — it must not be in the log.
    """
    worker = make_worker(operation_settings)
    NoteService(operation_settings, HashEmbedder(DIM)).save(
        "заметка без заголовка: ждёт и заголовок, и выжимку"
    )

    with caplog.at_level(logging.WARNING, logger="app"):
        task = asyncio.create_task(worker.run())
        try:
            await wait_until(lambda: _pending_summaries(operation_settings) == 0)
        finally:
            worker.stop()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    failures = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "loop_iteration_failed"
    ]
    assert failures == []
    with session(operation_settings) as conn:
        row = conn.execute("SELECT title FROM notes WHERE id = 1").fetchone()
    assert row["title"] is not None
